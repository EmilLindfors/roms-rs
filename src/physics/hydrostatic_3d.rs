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
//! that duplicate [`Forcing`]. The slow forcing it receives
//! ([`ModeSplitPhysics::slow_forcing_into`]) is
//!
//! ```text
//!     G = D·(⟨R₃D(u)⟩ − R_adv+Cor(ū)) + (τ_s − τ_b)/ρ₀
//! ```
//!
//! `⟨R₃D(u)⟩` is the depth mean of the 3D momentum tendency (baroclinic PGF,
//! advection, Coriolis). `R_adv+Cor(ū)` is the same horizontal advection and
//! Coriolis operator applied to columns of uniform `ū`, the part the 2D module
//! already has. What remains is the depth-mean baroclinic PGF and the momentum
//! dispersion of the vertical shear, `−∇·⟨u′u′⟩`. For flow without shear, `G`
//! reduces to the stresses exactly. Coriolis is pointwise and linear, so its
//! share cancels exactly.
//!
//! The 3D PGF is baroclinic-only (`ρ − ρ₀`), so the barotropic pressure
//! gradient comes from the 2D module. (Since P4.3 the 3D PGF lifts pressure
//! jumps at element faces and could carry `−g∇η` too; the division of labour
//! is kept, see TODO P4.1.)

use std::sync::{Arc, Mutex};

use crate::boundary::SWEBoundaryCondition2D;
use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::physics::SWEPhysics2D;
use crate::physics::eos::EquationOfState;
use crate::physics::traits::PhysicsModule; // For SWEPhysics2D
use crate::physics::vertical_diffusion::apply_vertical_diffusion;
use crate::physics::vertical_mixing::{Forcing, VerticalMixing};
use crate::physics::vertical_velocity::compute_vertical_velocity;
use crate::solver::SWESolution2D;
use crate::solver::rhs::{
    BarotropicFlux, ExtrapolationTracerBC3D, LayerTransport, Rhs3DConfig,
    TracerBoundaryCondition3D, TracerTransportScratch, apply_coriolis_3d,
    apply_horizontal_advection_3d, compute_momentum_rhs_3d, compute_transport_rhs_3d,
};
use crate::solver::state::Solution3D;
use crate::solver::state::{SWE_VAR_H, SWE_VAR_HU, SWE_VAR_HV};
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
    pub tracer_limiter: TracerLimiter3DConfig,
    /// Columns shallower than this (m) are thin (3D wetting and drying; see
    /// [`Self::with_min_column_depth`]).
    pub min_column_depth: f64,
    /// Ω of the 3D velocities alone, for the `w` output of [`Self::post_process`].
    pub w_scratch: Mutex<Vec<f64>>,
    /// Layer transports (and their Ω) of the last 3D stage, and the tracer
    /// kernel's buffers.
    transport_scratch: Mutex<(LayerTransport, TracerTransportScratch)>,
    /// One-level states for the mean-flow part of the slow forcing:
    /// `(uniform ū columns, their advection + Coriolis tendency)`.
    mean_flow_scratch: Mutex<Option<(Solution3D, Solution3D)>>,
    /// `state` with the velocity of thin columns zeroed, for the momentum
    /// advection (allocated on the first step with a thin column).
    masked_scratch: Mutex<Option<Solution3D>>,
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
        // Omega lives at the w-points: n_levels + 1 interfaces per column.
        let n_w = mesh.n_elements * ops.n_nodes * (sigma.n_levels() + 1);
        let transport_scratch = Mutex::new((
            LayerTransport::new(mesh.n_elements, &ops, sigma.n_levels()),
            TracerTransportScratch::new(&ops, sigma.n_levels()),
        ));
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
            tracer_limiter: TracerLimiter3DConfig::none(),
            min_column_depth: Self::DEFAULT_MIN_COLUMN_DEPTH,
            w_scratch: Mutex::new(vec![0.0; n_w]),
            transport_scratch,
            mean_flow_scratch: Mutex::new(None),
            masked_scratch: Mutex::new(None),
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
        assert!(depth > 0.0, "minimum column depth must be positive, got {depth}");
        self.min_column_depth = depth;
        self
    }

    /// Whether the column at node `idx` (`[element][node]`) is thin.
    #[inline]
    fn is_thin(&self, state: &Solution3D, idx: usize) -> bool {
        state.eta.data[idx] - self.bathymetry.data[idx] < self.min_column_depth
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
            g: self.g,
            rho0: self.rho0,
            min_column_depth: self.min_column_depth,
        }
    }

    /// Overwrite `rhs.u` and `rhs.v` with the horizontal momentum tendency of
    /// `state` (baroclinic PGF, horizontal advection, Coriolis; see
    /// [`compute_momentum_rhs_3d`]). `state.rho` must be current.
    ///
    /// Thin columns enter with zero velocity (their momentum is the 2D
    /// module's; a film at the 2D velocity cap would set the 3D advection's
    /// dissipation speed), consistently with the mean-flow part of G.
    pub fn compute_momentum_rhs_into(&self, state: &Solution3D, rhs: &mut Solution3D) {
        let nl = state.n_levels;
        let n_columns = state.eta.data.len();
        if (0..n_columns).any(|idx| self.is_thin(state, idx)) {
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
                }
            }
            compute_momentum_rhs_3d(rhs, masked, &self.rhs_config());
        } else {
            compute_momentum_rhs_3d(rhs, state, &self.rhs_config());
        }
        self.zero_thin_momentum(state, rhs);
    }

    /// The layer transports of `state` corrected to `barotropic`, then the
    /// vertical momentum advection added to `rhs.u`, `rhs.v` and the tracer
    /// inventory tendencies written to `rhs.temp`, `rhs.salt` (see
    /// [`compute_transport_rhs_3d`]).
    pub fn compute_transport_rhs_into(
        &self,
        state: &Solution3D,
        barotropic: BarotropicFlux,
        rhs: &mut Solution3D,
    ) {
        let mut guard = self
            .transport_scratch
            .lock()
            .expect("Failed to lock transport_scratch");
        let (transport, scratch) = &mut *guard;
        transport.compute(
            state,
            barotropic,
            &self.mesh,
            &self.ops,
            &self.geom,
            &self.sigma,
            &self.bathymetry,
        );
        compute_transport_rhs_3d(rhs, state, transport, &self.rhs_config(), scratch);
        self.zero_thin_momentum(state, rhs);
    }

    /// Largest surface residual of `Ω` in the last 3D stage before it was
    /// spread over the column (m/s): round-off where the barotropic pass keeps
    /// its nodal identity (see [`LayerTransport::surface_residual`]).
    pub fn last_surface_residual(&self) -> f64 {
        self.transport_scratch
            .lock()
            .expect("Failed to lock transport_scratch")
            .0
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
        let mut min_dt = f64::INFINITY;

        for k in 0..self.mesh.n_elements {
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
                let dt_loc = cfl * h_len / wave_speed / (self.ops.order as f64 + 1.0).powi(2);
                if dt_loc < min_dt {
                    min_dt = dt_loc;
                }
            }
        }

        if min_dt == f64::INFINITY { 1.0 } else { min_dt }
    }

    pub fn post_process(&self, state: &mut Solution3D) {
        self.apply_tracer_limiters(state);

        // Vertical velocity for output. Between steps there is no barotropic
        // transport, so this is Ω of the 3D velocities alone, with ∂η/∂t from
        // their divergence (`compute_vertical_velocity`); the stages use the Ω
        // of the layer transports.
        let mut w_vel = self.w_scratch.lock().expect("Failed to lock w_scratch");
        compute_vertical_velocity(
            &mut *w_vel,
            state,
            &self.mesh,
            &self.ops,
            &self.sigma,
            &self.bathymetry,
            &self.geom,
            self.g,
        );

        // state.w is an output field at the layer centres: average the two
        // bounding interface values of each layer.
        let n_levels = state.n_levels;
        for (w_col, omega_col) in state
            .w
            .chunks_exact_mut(n_levels)
            .zip(w_vel.chunks_exact(n_levels + 1))
        {
            for (l, w) in w_col.iter_mut().enumerate() {
                *w = 0.5 * (omega_col[l] + omega_col[l + 1]);
            }
        }

        // Could also apply equation of state update here to ensure rho is fresh
        // self.eos.update_density(state);
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

    fn momentum_rhs_into(&self, state: &Solution3D, _t: f64, out: &mut Solution3D) {
        self.compute_momentum_rhs_into(state, out);
    }

    fn transport_rhs_into(
        &self,
        state: &Solution3D,
        _t: f64,
        barotropic: BarotropicFlux,
        out: &mut Solution3D,
    ) {
        self.compute_transport_rhs_into(state, barotropic, out);
    }

    /// `G = D·(⟨R₃D(u)⟩ − R_adv+Cor(ū)) + (τ_s − τ_b)/ρ₀` (see the module docs).
    fn slow_forcing_into(
        &self,
        state: &Solution3D,
        rhs: &Solution3D,
        _t: f64,
        g: &mut SWESolution2D,
    ) {
        let (ne, nn, nl) = (state.n_elements, state.n_nodes, state.n_levels);

        // Advection + Coriolis of uniform ū columns, on one level
        let mut scratch = self
            .mean_flow_scratch
            .lock()
            .expect("Failed to lock mean_flow_scratch");
        let (bar, bar_rhs) =
            scratch.get_or_insert_with(|| (Solution3D::new(ne, nn, 1), Solution3D::new(ne, nn, 1)));
        bar.u.copy_from_slice(&state.ubar.data);
        bar.v.copy_from_slice(&state.vbar.data);
        // Thin columns masked as in `compute_momentum_rhs_into`
        for idx in 0..state.eta.data.len() {
            if self.is_thin(state, idx) {
                bar.u[idx] = 0.0;
                bar.v[idx] = 0.0;
            }
        }
        bar_rhs.u.fill(0.0);
        bar_rhs.v.fill(0.0);
        apply_horizontal_advection_3d(bar_rhs, bar, &self.mesh, &self.ops, &self.geom);
        apply_coriolis_3d(bar_rhs, bar, &self.mesh, &self.ops, &self.coriolis);

        let [tau_sx, tau_sy] = self.forcing.surface_stress;
        let [tau_bx, tau_by] = self.forcing.bottom_stress;
        let stress_x = (tau_sx - tau_bx) / self.rho0;
        let stress_y = (tau_sy - tau_by) / self.rho0;

        g.data[SWE_VAR_H].fill(0.0);
        for k in 0..ne {
            let bed = self.bathymetry.element(ElementIndex::new(k));
            for (i, &b) in bed.iter().enumerate() {
                let idx = k * nn + i;
                let columns = idx * nl..(idx + 1) * nl;
                let depth = state.eta.data[idx] - b;
                let mean_u = self.sigma.depth_average(&rhs.u[columns.clone()]);
                let mean_v = self.sigma.depth_average(&rhs.v[columns]);
                // Thin columns: no 3D stress (their depth mean is the 2D
                // module's alone)
                let wet = if depth < self.min_column_depth { 0.0 } else { 1.0 };
                g.data[SWE_VAR_HU][idx] = depth * (mean_u - bar_rhs.u[idx]) + wet * stress_x;
                g.data[SWE_VAR_HV][idx] = depth * (mean_v - bar_rhs.v[idx]) + wet * stress_y;
            }
        }
    }

    fn vertical_implicit(&self, state: &mut Solution3D, dt: f64) {
        apply_vertical_diffusion(
            state,
            &self.sigma,
            &self.bathymetry,
            dt,
            &self.mixing,
            &self.forcing,
            self.rho0,
            self.min_column_depth,
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
