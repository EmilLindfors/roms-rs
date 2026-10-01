//! Hydrostatic 3D Physics Module.
//!
//! Handles the full 3D primitive equations with hydrostatic approximation.
//! Contains the 2D barotropic physics module as a sub-component.
//!
//! # Barotropic coupling
//!
//! Under mode splitting ([`crate::time::ModeSplitIntegrator`]) the 2D module
//! owns the depth-mean flow: the barotropic pressure gradient (with the DG face
//! coupling of `η`), advection of `ū`, Coriolis on `ū`, and any bottom friction
//! on `ū`. Configure it with the same Coriolis parameter as the 3D model
//! (`with_source(CoriolisSource2D::…)`), and without wind or friction sources
//! that duplicate [`Forcing`] or [`Hydrostatic3D::with_bottom_drag`]. The
//! slow forcing it receives ([`ModeSplitPhysics::slow_forcing_into`]) is
//!
//! ```text
//!     G = D·⟨R_PGF+Cor(u)⟩ + Σ_l A_l(u) − A(ū) − D·R_Cor(ū) + (τ_s − τ_b)/ρ₀ − r·(u_b − ū)
//! ```
//!
//! `⟨R_PGF+Cor(u)⟩` is the depth mean of the 3D momentum tendency that does
//! not move with the layer fluxes (baroclinic PGF, Coriolis, and the
//! horizontal viscosity of the shear, [`Hydrostatic3D::with_horizontal_viscosity`],
//! whose column integral is zero for a constant ν: the depth mean's viscosity
//! is the 2D module's; with [`Hydrostatic3D::with_smagorinsky_viscosity`] it
//! is the stress of the shear on the mean flow). `A_l(u)` is the momentum
//! advection of layer `l` in inventory form, `−∇·(Q_l u_l) − δ(Ω u)_l`
//! ([`crate::solver::rhs::apply_momentum_transport_3d`]), with the state's own
//! layer transports `Q_l = H_z u_l` (the barotropic transport of the step
//! does not exist yet when `G` is built). `A(ū)` is the same operator on one
//! layer carrying `ū` with the transport `Dū`: with `R_Cor(ū)`, the advection
//! and Coriolis of the mean flow, which the 2D module already has. What
//! remains is the depth-mean baroclinic PGF and the momentum dispersion of
//! the vertical shear, `−∇·(Σ_l Q_l u_l − Dūū)`. Without shear
//! `Q_l = Δσ_l Dū`, so `Σ_l A_l(u) = A(ū)` exactly, Coriolis cancels (it is
//! pointwise and linear), and `G` reduces to the stresses. The vertical
//! fluxes sum to zero over the column.

//! The last term is the vertical-shear part of the quadratic bottom drag
//! ([`Hydrostatic3D::with_bottom_drag`], rate `r = C_d|u_b|`); the splitter
//! applies its depth-mean part `−r·ū` implicitly in the barotropic pass (see
//! [`crate::physics::bottom_drag`]). `τ_b` is the prescribed stress of
//! [`Forcing`], if any. Thin columns ([`Hydrostatic3D::with_min_column_depth`])
//! get no `G`: their depth mean is the 2D module's alone.
//!
//! The 3D PGF is baroclinic-only (`ρ − ρ₀`), so the barotropic pressure
//! gradient comes from the 2D module. (Since P4.3 the 3D PGF lifts pressure
//! jumps at element faces and could carry `−g∇η` too; the division of labour
//! is kept, see TODO P4.1.)

use std::sync::{Arc, Mutex, Once};

use crate::boundary::{Nesting3D, SWEBoundaryCondition2D};
use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::physics::SWEPhysics2D;
use crate::physics::bottom_drag::BottomDrag3D;
use crate::physics::eos::EquationOfState;
use crate::physics::traits::PhysicsModule; // For SWEPhysics2D
use crate::physics::vertical_diffusion::apply_vertical_diffusion;
use crate::physics::vertical_mixing::{Forcing, VerticalMixing};
use crate::solver::SWESolution2D;
use crate::solver::core::blocks::{for_each_block, reduce_blocks};
use crate::solver::rhs::{
    BarotropicFlux, Boundaries3D, Exterior3D, ExtrapolationTracerBC3D, HorizontalViscosity3D,
    LayerTransport, Rhs3DConfig, TracerBoundaryCondition3D, VerticalAdvection, ViscosityScratch3D,
    apply_coriolis_3d, apply_horizontal_viscosity_3d, apply_momentum_transport_3d,
    compute_momentum_rhs_3d, compute_transport_rhs_3d, element_dt_viscous_swe_2d,
    largest_horizontal_viscosity_3d,
};
use crate::solver::state::SWE_VAR_H;
use crate::solver::state::Solution3D;
use crate::solver::{TracerLimiter3DConfig, TracerLimiter3DStats, apply_tracer_limiters_3d};
use crate::source::CoriolisSource2D;
use crate::time::{Integrable, ModeSplitPhysics};
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// Hydrostatic 3D Physics Module.
///
/// Bundles all configuration and static data for the 3D solver.
pub struct Hydrostatic3D<EOS, MIX, BC>
where
    EOS: EquationOfState,
    MIX: VerticalMixing,
    BC: SWEBoundaryCondition2D,
{
    pub mesh: Arc<Mesh2D>,
    pub ops: Arc<DGOperators2D>,
    pub geom: Arc<GeometricFactors2D>,
    pub sigma: Arc<SigmaGrid>,
    pub bathymetry: Arc<Bathymetry2D>,
    pub coriolis: Arc<CoriolisSource2D>,
    pub eos: EOS,
    pub mixing: MIX,
    pub swe_physics: SWEPhysics2D<BC>, // 2D sub-model
    pub forcing: Forcing,
    pub g: f64,
    pub rho0: f64,
    /// Temperature of water flowing in through a physical boundary (walls
    /// carry none; see `TracerBoundaryCondition3D`).
    pub temp_bc: Arc<dyn TracerBoundaryCondition3D>,
    /// Salinity of water flowing in through a physical boundary (see `temp_bc`).
    pub salt_bc: Arc<dyn TracerBoundaryCondition3D>,
    /// Walls and open faces of the 3D kernels (see
    /// [`Self::with_wall_tags`]).
    pub boundaries: Boundaries3D,
    pub tracer_limiter: TracerLimiter3DConfig,
    /// Columns shallower than this (m) are thin (3D wetting and drying; see
    /// [`Self::with_min_column_depth`]).
    pub min_column_depth: f64,
    /// Reconstruction of the tracers at the σ-surfaces (see
    /// [`Self::with_vertical_advection`]).
    pub vertical_advection: VerticalAdvection,
    /// Reconstruction of the velocity at the σ-surfaces (see
    /// [`Self::with_momentum_vertical_advection`]).
    pub momentum_vertical_advection: VerticalAdvection,
    /// Quadratic drag of the bottom-layer velocity, if any (see
    /// [`Self::with_bottom_drag`]).
    pub bottom_drag: Option<BottomDrag3D>,
    /// Horizontal eddy viscosity of the vertical shear (see
    /// [`Self::with_horizontal_viscosity`] and
    /// [`Self::with_smagorinsky_viscosity`]).
    pub horizontal_viscosity: HorizontalViscosity3D,
    /// A parent model's profiles at open boundaries and in a relaxation
    /// band, if nested (see [`Self::with_nesting`]).
    pub nesting: Option<Nesting3D>,
    /// Warns once about stratified open boundaries without nesting.
    open_boundary_check: Once,
    /// Layer transports (and their Ω) of the last 3D stage.
    transport_scratch: Mutex<LayerTransport>,
    /// Buffers of the slow forcing (allocated on the first step).
    slow_forcing_scratch: Mutex<Option<SlowForcingScratch>>,
    /// `state` with the velocity of thin columns zeroed, for the momentum
    /// advection (allocated on the first step with a thin column).
    masked_scratch: Mutex<Option<Solution3D>>,
    /// Buffers of the horizontal viscosity (allocated on first use).
    viscosity_scratch: Mutex<Option<ViscosityScratch3D>>,
}

impl<EOS, MIX, BC> Hydrostatic3D<EOS, MIX, BC>
where
    EOS: EquationOfState,
    MIX: VerticalMixing,
    SWEPhysics2D<BC>: PhysicsModule<crate::solver::SWESolution2D>,
    BC: Clone + Send + Sync + SWEBoundaryCondition2D,
{
    pub fn new(
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
        sigma: Arc<SigmaGrid>,
        bathymetry: Arc<Bathymetry2D>,
        coriolis: Arc<CoriolisSource2D>,
        eos: EOS,
        mixing: MIX,
        swe_physics: SWEPhysics2D<BC>,
        forcing: Forcing,
        g: f64,
        rho0: f64,
    ) -> Self {
        geom.assert_affine("Hydrostatic3D (the 3D horizontal kernels)");
        let transport_scratch =
            Mutex::new(LayerTransport::new(mesh.n_elements, &ops, sigma.n_levels()));
        // The 3D walls are the 2D boundary condition's
        let boundaries = Boundaries3D::matching(&mesh, &swe_physics.bc);
        Self {
            mesh,
            ops,
            geom,
            sigma,
            bathymetry,
            coriolis,
            eos,
            mixing,
            swe_physics,
            forcing,
            g,
            rho0,
            temp_bc: Arc::new(ExtrapolationTracerBC3D),
            salt_bc: Arc::new(ExtrapolationTracerBC3D),
            boundaries,
            tracer_limiter: TracerLimiter3DConfig::none(),
            min_column_depth: Self::DEFAULT_MIN_COLUMN_DEPTH,
            vertical_advection: VerticalAdvection::default(),
            momentum_vertical_advection: VerticalAdvection::Centred,
            bottom_drag: None,
            horizontal_viscosity: HorizontalViscosity3D::default(),
            nesting: None,
            open_boundary_check: Once::new(),
            transport_scratch,
            slow_forcing_scratch: Mutex::new(None),
            masked_scratch: Mutex::new(None),
            viscosity_scratch: Mutex::new(None),
        }
    }

    /// Default of [`Self::with_min_column_depth`] (m), ROMS's usual `Dcrit`.
    pub const DEFAULT_MIN_COLUMN_DEPTH: f64 = 0.1;

    /// 3D wetting and drying: columns shallower than `depth` (m), down to dry
    /// nodes, are thin (Warner et al. 2013, ROMS's `Dcrit` masks):
    /// - they carry no vertical shear: their 3D velocity is the depth mean ū,
    ///   their momentum tendency is zero, and they get no vertical diffusion;
    /// - the surface and bottom stresses do not reach the depth mean through
    ///   them (masked in G);
    /// - they exert and feel no baroclinic pressure difference;
    ///
    /// (The tracers need no threshold: the mode splitter carries them as
    /// element means per level wherever the 2D pass balanced an element only
    /// as a whole, see [`crate::solver::rhs::inventory_to_concentration`].)
    ///
    /// Everything else stays 3D, including the wet nodes of shoreline
    /// elements. The 2D module's own wetting and drying (`WetDry`) is
    /// unaffected.
    pub fn with_min_column_depth(mut self, depth: f64) -> Self {
        assert!(
            depth > 0.0,
            "minimum column depth must be positive, got {depth}"
        );
        self.min_column_depth = depth;
        self
    }

    /// The vertical advection of the tracers: fourth-order Akima under a TVD
    /// limiter by default ([`VerticalAdvection::LimitedAkima`]; see
    /// [`VerticalAdvection`]).
    pub fn with_vertical_advection(mut self, scheme: VerticalAdvection) -> Self {
        self.vertical_advection = scheme;
        self
    }

    /// The vertical advection of the velocity, in the 3D stages and in the
    /// slow forcing `G`: second-order centred by default (see
    /// [`VerticalAdvection`]).
    pub fn with_momentum_vertical_advection(mut self, scheme: VerticalAdvection) -> Self {
        self.momentum_vertical_advection = scheme;
        self
    }

    /// Quadratic bottom drag `τ_b/ρ₀ = C_d|u_b|u_b` of the bottom-layer
    /// velocity, with a constant or log-layer `C_d` (see
    /// [`crate::physics::bottom_drag`] for the time discretisation). It adds
    /// to any prescribed `Forcing::bottom_stress`.
    ///
    /// Thin columns ([`Self::with_min_column_depth`]) have no shear, so their
    /// drag is `C_d|ū|ū` (with the `C_d` of their thin bottom layer, which a
    /// log layer puts at its upper bound). The 2D module should then carry no
    /// bottom friction of its own: it would count the drag twice.
    pub fn with_bottom_drag(mut self, drag: BottomDrag3D) -> Self {
        self.bottom_drag = Some(drag);
        self
    }

    /// Horizontal eddy viscosity `nu` (m²/s, constant) of the 3D momentum,
    /// along σ-surfaces, on the vertical shear `u − ū` only (BR1; see
    /// [`crate::solver::rhs::viscosity_3d`]). The depth mean is the 2D
    /// module's: give it its own `HorizontalViscosity2D` for a viscous mean
    /// flow. With [`Self::with_smagorinsky_viscosity`] it is the background
    /// added to Smagorinsky's.
    ///
    /// Higher orders need it in sheared, stratified flow: without it the
    /// shear instability of an interface grows at the grid scale (a P2 lock
    /// exchange blows up).
    pub fn with_horizontal_viscosity(mut self, nu: f64) -> Self {
        assert!(
            nu >= 0.0 && nu.is_finite(),
            "horizontal viscosity must be finite and non-negative, got {nu}"
        );
        self.horizontal_viscosity.background = nu;
        self
    }

    /// Smagorinsky's (1963) horizontal viscosity `(C_s Δ)²|S|` of the 3D
    /// shear, with coefficient `cs`, the strain rate `|S|` of each layer's
    /// velocity and the node spacing `Δ` (see
    /// [`crate::solver::rhs::viscosity_3d`]); added to the constant
    /// background of [`Self::with_horizontal_viscosity`], if any. Its column
    /// integral, the stress of the shear on the mean flow, reaches the depth
    /// mean through `G`.
    pub fn with_smagorinsky_viscosity(mut self, cs: f64) -> Self {
        assert!(
            cs >= 0.0 && cs.is_finite(),
            "Smagorinsky coefficient must be finite and non-negative, got {cs}"
        );
        self.horizontal_viscosity.smagorinsky = cs;
        self
    }

    /// Nest the 3D fields in a parent model: its velocity and tracer profiles
    /// at the open faces of the nesting's tags, and relaxation within its
    /// band (see [`Nesting3D`]). The depth mean stays the
    /// 2D module's: nest it there too (its open-boundary condition and
    /// [`crate::boundary::NestingRelaxation2D`]).
    ///
    /// Stratified runs with open boundaries need it, with a relaxation band:
    /// without, the extrapolated open faces let the boundary columns'
    /// stratification run away (see [`crate::boundary::Nesting3D`]'s module
    /// docs; [`crate::boundary::ReferenceColumns`] relaxes to a fixed state
    /// where there is no parent model).
    ///
    /// # Panics
    /// If a nested tag is a wall for the 3D kernels.
    pub fn with_nesting(mut self, nesting: Nesting3D) -> Self {
        for tag in nesting.tags() {
            assert!(
                !self.boundaries.wall_tags.contains(tag),
                "nested boundary tag {tag:?} is a wall for the 3D kernels"
            );
        }
        self.nesting = Some(nesting);
        self
    }

    /// Drag rate `r = C_d(z_b)·|u_b|` (m/s) of the column at node `idx`
    /// (`[element][node]`), with `z_b` the height of the bottom-layer centre
    /// above the bed; zero where the column is dry.
    #[inline]
    fn drag_rate(&self, drag: &BottomDrag3D, state: &Solution3D, idx: usize) -> f64 {
        let depth = state.eta.data[idx] - self.bathymetry.data[idx];
        if depth <= 0.0 {
            return 0.0;
        }
        let bottom = idx * state.n_levels;
        let (u, v) = if depth < self.min_column_depth {
            (state.ubar.data[idx], state.vbar.data[idx])
        } else {
            (state.u[bottom], state.v[bottom])
        };
        let z_b = (1.0 + self.sigma.sigma_rho()[0]) * depth;
        drag.rate(z_b, (u * u + v * v).sqrt())
    }

    /// Whether the column at node `idx` (`[element][node]`) is thin.
    #[inline]
    fn is_thin(&self, state: &Solution3D, idx: usize) -> bool {
        state.eta.data[idx] - self.bathymetry.data[idx] < self.min_column_depth
    }

    /// Call `f` with `state`, or with a copy of it whose thin columns have no
    /// velocity: the momentum advection sees them at rest (their momentum is
    /// the 2D module's; a film at the 2D velocity cap would otherwise carry
    /// its speed into the 3D shear), consistently in the stages and in `G`.
    fn with_thin_columns_at_rest<R>(
        &self,
        state: &Solution3D,
        f: impl FnOnce(&Solution3D) -> R,
    ) -> R {
        let nl = state.n_levels;
        let n_columns = state.eta.data.len();
        if !(0..n_columns).any(|idx| self.is_thin(state, idx)) {
            return f(state);
        }
        let mut guard = self
            .masked_scratch
            .lock()
            .expect("Failed to lock masked_scratch");
        let masked = guard.get_or_insert_with(|| {
            Solution3D::new(state.n_elements, state.n_nodes, state.n_levels)
        });
        masked.copy_from(state);
        for idx in 0..n_columns {
            if self.is_thin(state, idx) {
                masked.u[idx * nl..(idx + 1) * nl].fill(0.0);
                masked.v[idx * nl..(idx + 1) * nl].fill(0.0);
                masked.ubar.data[idx] = 0.0;
                masked.vbar.data[idx] = 0.0;
            }
        }
        f(masked)
    }

    /// Zero the momentum tendency of thin columns: they carry no shear, and
    /// the splitter adds the depth-mean rate.
    fn zero_thin_momentum(&self, state: &Solution3D, rhs: &mut Solution3D) {
        let nl = state.n_levels;
        for idx in 0..state.eta.data.len() {
            if self.is_thin(state, idx) {
                rhs.u[idx * nl..(idx + 1) * nl].fill(0.0);
                rhs.v[idx * nl..(idx + 1) * nl].fill(0.0);
            }
        }
    }

    /// Override the boundary tags that are walls for the 3D kernels (untagged
    /// faces are then walls too). Every other boundary is open: its layers
    /// carry the 2D open-boundary flux in the interior's vertical profile,
    /// and the 3D velocity is extrapolated there (see
    /// [`crate::solver::rhs::boundary_3d`]).
    ///
    /// [`Self::new`] already derives the walls from the 2D module's boundary
    /// condition ([`Boundaries3D::matching`]); override only for a condition
    /// that cannot tell (`SWEBoundaryCondition2D::is_wall` is `None`). A tag
    /// the 2D condition treats as a wall must be a wall here.
    pub fn with_wall_tags(
        mut self,
        tags: impl IntoIterator<Item = crate::mesh::BoundaryTag>,
    ) -> Self {
        self.boundaries = Boundaries3D::with_walls(tags);
        self
    }

    /// Override scalar tracer boundary conditions used by the high-level RHS path.
    pub fn with_tracer_boundary_conditions(
        mut self,
        temp_bc: Arc<dyn TracerBoundaryCondition3D>,
        salt_bc: Arc<dyn TracerBoundaryCondition3D>,
    ) -> Self {
        self.temp_bc = temp_bc;
        self.salt_bc = salt_bc;
        self
    }

    /// Set scalar tracer boundary conditions used by the high-level RHS path.
    pub fn set_tracer_boundary_conditions(
        &mut self,
        temp_bc: Arc<dyn TracerBoundaryCondition3D>,
        salt_bc: Arc<dyn TracerBoundaryCondition3D>,
    ) {
        self.temp_bc = temp_bc;
        self.salt_bc = salt_bc;
    }

    /// Override 3D tracer limiter configuration.
    pub fn with_tracer_limiter(mut self, tracer_limiter: TracerLimiter3DConfig) -> Self {
        self.tracer_limiter = tracer_limiter;
        self
    }

    /// Set 3D tracer limiter configuration.
    pub fn set_tracer_limiter(&mut self, tracer_limiter: TracerLimiter3DConfig) {
        self.tracer_limiter = tracer_limiter;
    }

    /// Apply configured 3D tracer limiters and refresh density if tracers changed.
    pub fn apply_tracer_limiters(&self, state: &mut Solution3D) -> TracerLimiter3DStats {
        let stats = apply_tracer_limiters_3d(
            state,
            &self.mesh,
            &self.ops,
            &self.geom,
            &self.bathymetry,
            &self.sigma,
            &self.tracer_limiter,
        );

        if stats.changed() {
            self.eos.update_density(state);
        }

        stats
    }

    fn rhs_config(&self) -> Rhs3DConfig<'_> {
        Rhs3DConfig {
            mesh: &self.mesh,
            ops: &self.ops,
            geom: &self.geom,
            bathymetry: &self.bathymetry,
            sigma: &self.sigma,
            coriolis: &self.coriolis,
            temp_bc: &*self.temp_bc,
            salt_bc: &*self.salt_bc,
            boundaries: &self.boundaries,
            exterior: Exterior3D::default(),
            g: self.g,
            rho0: self.rho0,
            min_column_depth: self.min_column_depth,
            vertical_advection: self.vertical_advection,
            momentum_vertical_advection: self.momentum_vertical_advection,
        }
    }

    /// Overwrite `rhs.u` and `rhs.v` with the velocity tendency of the
    /// momentum terms of `state` at time `t` that do not move with the layer
    /// fluxes (baroclinic PGF, Coriolis; see [`compute_momentum_rhs_3d`]; the
    /// horizontal viscosity of the shear; and the nesting's relaxation of
    /// the shear, which has no depth mean), zero in thin columns.
    /// `state.rho` must be current.
    pub fn compute_momentum_rhs_into(&self, state: &Solution3D, t: f64, rhs: &mut Solution3D) {
        compute_momentum_rhs_3d(rhs, state, &self.rhs_config());
        if !self.horizontal_viscosity.is_zero() {
            let mut guard = self
                .viscosity_scratch
                .lock()
                .expect("Failed to lock viscosity_scratch");
            let scratch = guard
                .get_or_insert_with(|| ViscosityScratch3D::new(self.mesh.n_elements, &self.ops));
            apply_horizontal_viscosity_3d(
                &mut rhs.u,
                &mut rhs.v,
                state,
                self.horizontal_viscosity,
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.bathymetry,
                &self.sigma,
                &self.boundaries,
                self.min_column_depth,
                scratch,
            );
        }
        if let Some(nesting) = &self.nesting {
            nesting
                .columns(state, &self.bathymetry, &self.sigma, t)
                .relax_shear(state, &self.sigma, rhs, |idx| self.is_thin(state, idx));
        }
        self.zero_thin_momentum(state, rhs);
    }

    /// The layer transports of `state` corrected to `barotropic`, then the
    /// inventory tendency of the momentum advection added to `rhs.u`, `rhs.v`
    /// (zero in thin columns, which the advection sees at rest) and the tracer
    /// inventory tendencies written to `rhs.temp`, `rhs.salt` (see
    /// [`compute_transport_rhs_3d`]), with the nesting parent's values at
    /// time `t` outside its open faces and its relaxation of the tracers.
    pub fn compute_transport_rhs_into(
        &self,
        state: &Solution3D,
        t: f64,
        barotropic: BarotropicFlux,
        rhs: &mut Solution3D,
    ) {
        let mut guard = self
            .transport_scratch
            .lock()
            .expect("Failed to lock transport_scratch");
        let transport = &mut *guard;
        self.open_boundary_check
            .call_once(|| self.warn_if_stratified_open_boundaries(state));
        let columns = self
            .nesting
            .as_ref()
            .map(|nesting| nesting.columns(state, &self.bathymetry, &self.sigma, t));
        let exterior = columns
            .as_ref()
            .map_or_else(Exterior3D::default, |c| c.exterior());
        // Thin columns at rest change no layer transport: theirs are uniform
        // in σ, which the correction to the barotropic transport replaces
        self.with_thin_columns_at_rest(state, |state| {
            transport.compute(
                state,
                Some(barotropic),
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.sigma,
                &self.bathymetry,
                &self.boundaries,
                &exterior,
            );
            let config = Rhs3DConfig {
                exterior,
                ..self.rhs_config()
            };
            compute_transport_rhs_3d(rhs, state, transport, &config);
        });
        if let Some(columns) = &columns {
            let thin = |idx| self.is_thin(state, idx);
            columns.relax_tracers(state, &self.bathymetry, &self.sigma, rhs, thin);
        }
        self.zero_thin_momentum(state, rhs);
    }

    /// Warn if `state` is stratified and has open 3D faces that no nesting
    /// relaxes (see [`Self::with_nesting`]).
    fn warn_if_stratified_open_boundaries(&self, state: &Solution3D) {
        let nested = |tag| {
            self.nesting
                .as_ref()
                .is_some_and(|n| n.tags().contains(&tag))
        };
        let open_tag = (0..self.mesh.n_elements).find_map(|k| {
            (0..4).find_map(|f| {
                match self
                    .boundaries
                    .exterior(&self.mesh, ElementIndex::new(k), f)
                {
                    crate::solver::rhs::FaceExterior::Open(tag) if !nested(tag) => Some(tag),
                    _ => None,
                }
            })
        });
        let Some(tag) = open_tag else {
            return;
        };
        let nl = state.n_levels;
        let stratified = state.rho.chunks_exact(nl).any(|column| {
            let (lo, hi) = column
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &r| {
                    (lo.min(r), hi.max(r))
                });
            hi - lo > 1e-6 * self.rho0
        });
        if stratified {
            eprintln!(
                "warning: Hydrostatic3D: the open boundary {tag:?} is not nested, and the \
                 water is stratified. Extrapolated open faces let the boundary columns' \
                 stratification run away; relax it with `with_nesting` (a parent model, or \
                 `ReferenceColumns` of the state at rest) and a relaxation band."
            );
        }
    }

    /// Largest surface residual of `Ω` in the last 3D stage before it was
    /// spread over the column (m/s): round-off where the barotropic pass keeps
    /// its nodal identity (see [`LayerTransport::surface_residual`]).
    pub fn last_surface_residual(&self) -> f64 {
        self.transport_scratch
            .lock()
            .expect("Failed to lock transport_scratch")
            .surface_residual
    }

    /// Update density field based on current temperature and salinity.
    pub fn update_density(&self, state: &mut Solution3D) {
        self.eos.update_density(state);
    }

    /// Compute permissible time step based on 3D CFL condition.
    ///
    /// Limited by horizontal advection speed + gravity wave speed (if explicit)
    /// or just advection (if split).
    /// Since we use mode splitting, the 3D step is limited by:
    /// 1. Internal wave speed (baroclinic modes)
    /// 2. 3D Advection velocity
    pub fn compute_dt(&self, state: &Solution3D, cfl: f64) -> f64 {
        // Simplified estimate: use 2D CFL but scaled for internal waves?
        // Or just use advection speed.
        // For mode splitting, dt_3d can be much larger than dt_2d (barotropic).
        // Typically dt_3d is limited by internal gravity waves c_n ~ sqrt(g' H).

        // For now, let's delegate to SWEPhysics2D compute_dt and multiply by a factor?
        // No, better to compute explicit advection limit.

        // Placeholder: Return a conservative estimate
        // Min(dx / (|u| + c_internal))

        // Let's assume c_internal << c_external
        // So we can take a larger step.
        // For this prototype, return 0.1s or something safe.
        // Or better: use the 2D dt computation but with a larger CFL factor.

        // We really should iterate over elements and find max(|u| + c_bc) / dx.
        // c_bc approx NH * N * H / pi?

        // Let's use the 2D physics dt as a baseline.
        // But 2D physics uses sqrt(gH), which is fast.
        // We want to skip that.

        // Let's iterate elements.
        let mut min_dt = reduce_blocks::<f64, _, _, 0>(
            self.mesh.n_elements,
            [],
            || (),
            |_, k, []| {
                let j_inv = self.geom.affine_metric(k).det_j_inv;
                // length scale h ~ 1/sqrt(J_inv) ?
                // For parallelogram: Area = J. h ~ sqrt(Area).
                let h_len = 1.0 / j_inv.sqrt(); // Approx element size

                // Max velocity in column
                let mut max_vel = 0.0;
                let el = crate::types::ElementIndex::new(k);
                for i in 0..state.n_nodes {
                    // Thin films are the 2D module's (and its own CFL's) business
                    if self.is_thin(state, k * state.n_nodes + i) {
                        continue;
                    }
                    for l in 0..state.n_levels {
                        let u = state.u_column(el, i)[l];
                        let v = state.v_column(el, i)[l];
                        let vel = (u * u + v * v).sqrt();
                        if vel > max_vel {
                            max_vel = vel;
                        }
                    }
                }

                // Internal wave speed approximation: c ~ 2.0 m/s (typical)
                let c_internal = 2.0;
                let wave_speed = max_vel + c_internal;

                if wave_speed > 1e-6 {
                    cfl * h_len / wave_speed / (self.ops.order as f64 + 1.0).powi(2)
                } else {
                    f64::INFINITY
                }
            },
            || f64::INFINITY,
            f64::min,
        );

        // The horizontal viscosity, explicit in the SSP-RK3 stages: BR1
        // couples an element to its face neighbours' gradients, so each
        // element is bounded by the largest ν of its own and theirs
        if !self.horizontal_viscosity.is_zero() {
            let mut largest = vec![0.0; self.mesh.n_elements];
            let mut guard = self
                .viscosity_scratch
                .lock()
                .expect("Failed to lock viscosity_scratch");
            let scratch = guard
                .get_or_insert_with(|| ViscosityScratch3D::new(self.mesh.n_elements, &self.ops));
            largest_horizontal_viscosity_3d(
                &mut largest,
                state,
                self.horizontal_viscosity,
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.bathymetry,
                &self.sigma,
                &self.boundaries,
                self.min_column_depth,
                scratch,
            );
            for k in 0..self.mesh.n_elements {
                let nu = (0..4)
                    .filter_map(|face| self.mesh.neighbor(ElementIndex::new(k), face))
                    .map(|nb| largest[nb.element])
                    .fold(largest[k], f64::max);
                min_dt = min_dt.min(element_dt_viscous_swe_2d(
                    &self.mesh,
                    &self.ops,
                    &self.geom,
                    nu,
                    self.ops.order,
                    cfl,
                    k,
                ));
            }
        }

        if min_dt == f64::INFINITY { 1.0 } else { min_dt }
    }

    pub fn post_process(&self, state: &mut Solution3D) {
        self.apply_tracer_limiters(state);

        self.update_vertical_velocity(state);
    }

    /// Write `Ω` (m/s) at the layer centres to `state.w`, for output: the
    /// mean of the two bounding σ-surfaces, from the state's own layer
    /// transports ([`LayerTransport::compute`] without a barotropic flux, thin
    /// columns at rest as in the stages). Their column sum is `Dū`, which
    /// after a mode-split step is the filtered barotropic transport, and `Ω`
    /// closes with the free-surface rate its divergence implies.
    pub fn update_vertical_velocity(&self, state: &mut Solution3D) {
        let mut guard = self
            .transport_scratch
            .lock()
            .expect("Failed to lock transport_scratch");
        let transport = &mut *guard;
        self.with_thin_columns_at_rest(state, |state| {
            transport.compute(
                state,
                None,
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.sigma,
                &self.bathymetry,
                &self.boundaries,
                &Exterior3D::default(),
            );
        });
        let nl = state.n_levels;
        for (w_col, omega_col) in state
            .w
            .chunks_exact_mut(nl)
            .zip(transport.omega.chunks_exact(nl + 1))
        {
            for (l, w) in w_col.iter_mut().enumerate() {
                *w = 0.5 * (omega_col[l] + omega_col[l + 1]);
            }
        }
    }
}

impl<EOS, MIX, BC> ModeSplitPhysics for Hydrostatic3D<EOS, MIX, BC>
where
    EOS: EquationOfState,
    MIX: VerticalMixing,
    SWEPhysics2D<BC>: PhysicsModule<SWESolution2D>,
    BC: Clone + Send + Sync + SWEBoundaryCondition2D,
{
    type Barotropic = SWEPhysics2D<BC>;

    fn barotropic(&self) -> &SWEPhysics2D<BC> {
        &self.swe_physics
    }

    fn sigma(&self) -> &SigmaGrid {
        &self.sigma
    }

    fn bathymetry(&self) -> &Bathymetry2D {
        &self.bathymetry
    }

    fn momentum_rhs_into(&self, state: &Solution3D, t: f64, out: &mut Solution3D) {
        self.compute_momentum_rhs_into(state, t, out);
    }

    fn transport_rhs_into(
        &self,
        state: &Solution3D,
        t: f64,
        barotropic: BarotropicFlux,
        out: &mut Solution3D,
    ) {
        self.compute_transport_rhs_into(state, t, barotropic, out);
    }

    /// `G = D·⟨R_PGF+Cor(u)⟩ + Σ_l A_l(u) − A(ū) − D·R_Cor(ū) + (τ_s − τ_b)/ρ₀
    /// − r·(u_b − ū)`, zero in thin columns (see the module docs).
    fn slow_forcing_into(
        &self,
        state: &Solution3D,
        rhs: &Solution3D,
        _t: f64,
        g: &mut SWESolution2D,
    ) {
        let (ne, nn, nl) = (state.n_elements, state.n_nodes, state.n_levels);
        let mut guard = self
            .slow_forcing_scratch
            .lock()
            .expect("Failed to lock slow_forcing_scratch");
        let scratch = guard.get_or_insert_with(|| SlowForcingScratch::new(ne, &self.ops, nl));
        let mut transport_guard = self
            .transport_scratch
            .lock()
            .expect("Failed to lock transport_scratch");
        let transport = &mut *transport_guard;
        let SlowForcingScratch {
            advection_u,
            advection_v,
            bar,
            bar_rhs,
            bar_transport,
            bar_sigma,
        } = scratch;

        self.with_thin_columns_at_rest(state, |state| {
            // Σ_l A_l(u): the columns advected with their own layer transports
            transport.compute(
                state,
                None,
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.sigma,
                &self.bathymetry,
                &self.boundaries,
                &Exterior3D::default(),
            );
            advection_u.fill(0.0);
            advection_v.fill(0.0);
            apply_momentum_transport_3d(
                advection_u,
                advection_v,
                &state.u,
                &state.v,
                transport,
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.boundaries,
                None,
                self.momentum_vertical_advection,
            );

            // A(ū) + D·R_Cor(ū): the mean flow as one layer
            bar.eta.copy_from(&state.eta);
            bar.u.copy_from_slice(&state.ubar.data);
            bar.v.copy_from_slice(&state.vbar.data);
            bar_transport.compute(
                bar,
                None,
                &self.mesh,
                &self.ops,
                &self.geom,
                bar_sigma,
                &self.bathymetry,
                &self.boundaries,
                &Exterior3D::default(),
            );
            bar_rhs.u.fill(0.0);
            bar_rhs.v.fill(0.0);
            apply_coriolis_3d(bar_rhs, bar, &self.mesh, &self.ops, &self.coriolis);
            for ((u, v), (&eta, &b)) in bar_rhs
                .u
                .iter_mut()
                .zip(&mut bar_rhs.v)
                .zip(state.eta.data.iter().zip(&self.bathymetry.data))
            {
                *u *= eta - b;
                *v *= eta - b;
            }
            apply_momentum_transport_3d(
                &mut bar_rhs.u,
                &mut bar_rhs.v,
                &bar.u,
                &bar.v,
                bar_transport,
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.boundaries,
                None,
                self.momentum_vertical_advection,
            );
        });

        let [tau_sx, tau_sy] = self.forcing.surface_stress;
        let [tau_bx, tau_by] = self.forcing.bottom_stress;
        let stress_x = (tau_sx - tau_bx) / self.rho0;
        let stress_y = (tau_sy - tau_by) / self.rho0;

        g.data[SWE_VAR_H].fill(0.0);
        let [_, g_hu, g_hv] = &mut g.data;
        let (advection_u, advection_v, bar_rhs): (&[f64], &[f64], &Solution3D) =
            (advection_u, advection_v, bar_rhs);
        for_each_block(
            ne,
            [&mut g_hu[..ne * nn], &mut g_hv[..ne * nn]],
            || (),
            |_, k, [g_hu, g_hv]| {
                let bed = self.bathymetry.element(ElementIndex::new(k));
                for (i, &b) in bed.iter().enumerate() {
                    let idx = k * nn + i;
                    let depth = state.eta.data[idx] - b;
                    // Thin columns: their depth mean is the 2D module's alone
                    if depth < self.min_column_depth {
                        g_hu[i] = 0.0;
                        g_hv[i] = 0.0;
                        continue;
                    }
                    let columns = idx * nl..(idx + 1) * nl;
                    let mean_u = self.sigma.depth_average(&rhs.u[columns.clone()]);
                    let mean_v = self.sigma.depth_average(&rhs.v[columns.clone()]);
                    let advection_x: f64 = advection_u[columns.clone()].iter().sum();
                    let advection_y: f64 = advection_v[columns].iter().sum();
                    // The shear part of the bottom drag; the pass applies −r·ū
                    let (drag_x, drag_y) = match &self.bottom_drag {
                        Some(drag) => {
                            let r = self.drag_rate(drag, state, idx);
                            (
                                r * (state.u[idx * nl] - state.ubar.data[idx]),
                                r * (state.v[idx * nl] - state.vbar.data[idx]),
                            )
                        }
                        None => (0.0, 0.0),
                    };
                    g_hu[i] = depth * mean_u + advection_x - bar_rhs.u[idx] + stress_x - drag_x;
                    g_hv[i] = depth * mean_v + advection_y - bar_rhs.v[idx] + stress_y - drag_y;
                }
            },
        );
    }

    /// `r = C_d|u_b|` of every column from the bottom-layer velocity (the
    /// depth mean in thin columns), if a drag is set.
    fn bottom_drag_into(&self, state: &Solution3D, _t: f64, rate: &mut [f64]) -> bool {
        let Some(drag) = &self.bottom_drag else {
            return false;
        };
        for (idx, r) in rate.iter_mut().enumerate() {
            *r = self.drag_rate(drag, state, idx);
        }
        true
    }

    fn vertical_implicit(&self, state: &mut Solution3D, dt: f64, bottom_drag: Option<&[f64]>) {
        apply_vertical_diffusion(
            state,
            &self.sigma,
            &self.bathymetry,
            dt,
            &self.mixing,
            &self.forcing,
            self.rho0,
            self.min_column_depth,
            bottom_drag,
        );
        // Thin columns carry the depth mean only
        let nl = state.n_levels;
        for idx in 0..state.eta.data.len() {
            if self.is_thin(state, idx) {
                state.u[idx * nl..(idx + 1) * nl].fill(state.ubar.data[idx]);
                state.v[idx * nl..(idx + 1) * nl].fill(state.vbar.data[idx]);
            }
        }
    }

    /// Tracer limiters, then the density of the limited tracers.
    fn post_stage(&self, state: &mut Solution3D) {
        if !self.apply_tracer_limiters(state).changed() {
            self.update_density(state);
        }
    }
}

/// Buffers of [`Hydrostatic3D`]'s slow forcing.
struct SlowForcingScratch {
    /// `A_l(u)` of every layer, `[element][node][level]`.
    advection_u: Vec<f64>,
    advection_v: Vec<f64>,
    /// One layer carrying `ū`, and its advection + Coriolis tendency
    /// (inventory form).
    bar: Solution3D,
    bar_rhs: Solution3D,
    bar_transport: LayerTransport,
    bar_sigma: SigmaGrid,
}

impl SlowForcingScratch {
    fn new(n_elements: usize, ops: &DGOperators2D, n_levels: usize) -> Self {
        let nn = ops.n_nodes;
        Self {
            advection_u: vec![0.0; n_elements * nn * n_levels],
            advection_v: vec![0.0; n_elements * nn * n_levels],
            bar: Solution3D::new(n_elements, nn, 1),
            bar_rhs: Solution3D::new(n_elements, nn, 1),
            bar_transport: LayerTransport::new(n_elements, ops, 1),
            bar_sigma: SigmaGrid::uniform(1),
        }
    }
}
