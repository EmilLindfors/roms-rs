//! Right-hand side computation for 2D DG shallow water equations.
//!
//! For the 2D SWE:
//!   ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0
//!   ∂(hu)/∂t + ∂(hu² + gh²/2)/∂x + ∂(huv)/∂y = fhv + S_x
//!   ∂(hv)/∂t + ∂(huv)/∂x + ∂(hv² + gh²/2)/∂y = -fhu + S_y
//!
//! The DG semi-discrete form uses the weak formulation:
//!   dq/dt = L(q) = -(volume terms) + (surface terms) + (source terms)
//!
//! Two spatial formulations are available ([`SWEFormulation2D`]): the standard
//! collocated form below, and the entropy-stable, well-balanced split form of
//! Wintermeyer et al. (2017) in `swe_2d_split_form.rs`.

use crate::boundary::{BCContext2D, BoundaryState, SWEBoundaryCondition2D};
use crate::equations::ShallowWater2D;
use crate::flux::{SWEFluxType2D, compute_flux_swe_2d};
use crate::mesh::{Bathymetry2D, Mesh2D};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::state::{SWE_VAR_HU, SWE_VAR_HV};
use crate::solver::{SWESolution2D, SWEState2D};
use crate::source::swe_2d::viscosity::HorizontalViscosity2D;
use crate::source::{ElementSources, HydrostaticReconstruction2D, SourceTerm2D};
use crate::types::ElementIndex;

use super::diffusion_2d::{compute_br1_diffusion_rhs_2d, compute_br1_gradient_2d};
use super::swe_2d_split_form::{SplitFormSWE2D, SplitFormWorkspace};
#[cfg(feature = "simd")]
use crate::solver::simd::{apply_diff_matrix, apply_lift, combine_derivatives, coriolis_source};
#[cfg(not(feature = "simd"))]
use crate::solver::simd::{
    apply_diff_matrix_scalar as apply_diff_matrix, apply_lift_scalar as apply_lift,
    combine_derivatives_scalar as combine_derivatives, coriolis_source_scalar as coriolis_source,
};
use std::cell::RefCell;

/// Spatial discretization of the 2D SWE flux divergence and bathymetry terms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SWEFormulation2D {
    /// Collocated nodal DG: the nodal flux is differentiated directly, faces use
    /// the `flux_type` Riemann solver (optionally on hydrostatically reconstructed
    /// states, `well_balanced`), and the bed slope comes from `BathymetrySource2D`
    /// in `source_terms`. Lake-at-rest is exact only for deg B ≤ p/2 per element.
    #[default]
    Standard,
    /// Wintermeyer et al. (2017) split-form DGSEM with entropy-conservative
    /// fluxes in the volume *and* at faces (no interface dissipation).
    ///
    /// Conserves mass exactly and the total energy up to round-off on periodic
    /// domains; meant for verification. Use [`Self::EntropyStable`] for runs.
    EntropyConservative,
    /// Wintermeyer et al. (2017) split-form DGSEM with an entropy-stable
    /// interface flux (entropy-conservative flux plus Lax–Friedrichs dissipation
    /// in entropy variables).
    ///
    /// Well-balanced for *any* nodal bathymetry, including bathymetry that jumps
    /// across faces, with exact mass conservation and a discrete entropy
    /// (energy) inequality. The bed slope is part of the operator: `flux_type`
    /// and `well_balanced` are ignored, and `source_terms` must not contain
    /// `BathymetrySource2D` (the RHS panics if it does). Assumes wet nodes; there
    /// is no wetting/drying treatment (use [`Self::WetDry`]). See
    /// `solver/rhs/swe_2d_split_form.rs`.
    EntropyStable,
    /// Split form for wetting and drying:
    /// - fully wet elements use the flux-differencing volume term of
    ///   [`Self::EntropyStable`];
    /// - elements with a node shallower than `SWE2DRhsConfig::h_dry` use a
    ///   second-order (limited linear reconstruction) finite-volume update on
    ///   their GLL subcells;
    /// - every interface, faces and subcell faces alike, uses HLL on
    ///   Audusse et al. (2004) hydrostatically reconstructed states.
    ///
    /// Well-balanced at lake at rest also across wet/dry shorelines,
    /// mass-conservative, and positivity preserving for the element means under
    /// `positivity_cfl_swe_2d` (with the positivity limiter). Like the other
    /// split forms the bed slope is part of the operator: no
    /// `BathymetrySource2D`; `flux_type` and `well_balanced` are ignored.
    ///
    /// Compared with [`Self::Standard`] with hydrostatic reconstruction, which
    /// leaves a stationary spurious circulation at a lake-at-rest shoreline
    /// (≈ 5 cm/s where h > 1 cm on a 2 % beach), it is also more accurate on a
    /// moving shoreline (Thacker's paraboloid at 40², L1 depth error: P1
    /// 3.2 % vs 10 %, P2 1.2 % vs 1.3 %, P3 0.9 % vs 1.8 %, P4 0.7 % vs 2.3 %;
    /// `tests/wet_dry_2d_test.rs`). `SWEPhysics2DBuilder` uses it by default
    /// for runs with wetting/drying.
    WetDry,
}

/// Configuration for 2D SWE RHS computation.
pub struct SWE2DRhsConfig<'a, BC: SWEBoundaryCondition2D> {
    /// The shallow water equation parameters
    pub equation: &'a ShallowWater2D,
    /// Spatial formulation (collocated or split-form entropy-stable)
    pub formulation: SWEFormulation2D,
    /// Numerical flux type (`SWEFormulation2D::Standard` only)
    pub flux_type: SWEFluxType2D,
    /// Boundary condition handler
    pub bc: &'a BC,
    /// Whether to include Coriolis source term (legacy, use source_terms instead)
    pub include_coriolis: bool,
    /// Optional trait-based source terms (preferred over include_coriolis)
    ///
    /// When set, these source terms are evaluated at each node and added to the RHS.
    /// Multiple sources can be combined using `CombinedSource2D`.
    pub source_terms: Option<&'a dyn SourceTerm2D>,
    /// Optional bathymetry data for well-balanced schemes and source terms.
    ///
    /// When set, bathymetry values and gradients are passed to boundary conditions
    /// and source terms for proper handling of variable bottom topography.
    pub bathymetry: Option<&'a Bathymetry2D>,
    /// Enable hydrostatic reconstruction for well-balanced treatment of bathymetry.
    ///
    /// When enabled, the numerical flux is evaluated on Audusse et al. (2004)
    /// hydrostatically reconstructed interface states and the interior side gets
    /// the momentum correction ½g(h*² − h²)·n. This balances the bathymetry
    /// *jumps* across faces; the interior flux stays that of the nodal state, so
    /// mass is conserved exactly.
    ///
    /// Requires `bathymetry` to be set. The in-element slope ∂B still has to be
    /// balanced by `BathymetrySource2D` in `source_terms`:
    /// - cell-constant B (`Bathymetry2D::to_cell_average`): the source is zero and
    ///   may be omitted; lake-at-rest is preserved to round-off;
    /// - nodal (non-constant) B: **include** `BathymetrySource2D`, otherwise the
    ///   volume term leaves an unbalanced `g h ∂B/∂x` (O(1) m/s² accelerations).
    ///   Even with the source, lake-at-rest is exact only if `½gh²` is resolved by
    ///   the element polynomials (deg B ≤ p/2).
    pub well_balanced: bool,
    /// Optional horizontal viscosity for momentum diffusion.
    ///
    /// When set, adds ∇·(ν h ∇u) and ∇·(ν h ∇v) to the momentum equations.
    /// This is computed via a double-derivative Laplacian (differentiate velocity
    /// gradients, then differentiate viscous fluxes), following the same pattern
    /// as tracer diffusion.
    pub viscosity: Option<&'a HorizontalViscosity2D>,
    /// Depth (m) below which a node counts as dry for
    /// [`SWEFormulation2D::WetDry`]: elements with such a node use the subcell
    /// finite-volume update. Default 1 mm (`WetDryConfig::DEFAULT_H_DRY`).
    pub h_dry: f64,
}

impl<'a, BC: SWEBoundaryCondition2D> SWE2DRhsConfig<'a, BC> {
    /// Create a new RHS configuration.
    pub fn new(equation: &'a ShallowWater2D, bc: &'a BC) -> Self {
        Self {
            equation,
            formulation: SWEFormulation2D::Standard,
            flux_type: SWEFluxType2D::Roe,
            bc,
            include_coriolis: true,
            source_terms: None,
            bathymetry: None,
            well_balanced: false,
            viscosity: None,
            h_dry: crate::solver::WetDryConfig::DEFAULT_H_DRY,
        }
    }

    /// Set the spatial formulation.
    ///
    /// # Example
    /// ```ignore
    /// // Well-balanced for arbitrary (also face-discontinuous) nodal bathymetry.
    /// // The bed slope is part of the operator: no BathymetrySource2D here.
    /// let config = SWE2DRhsConfig::new(&equation, &bc)
    ///     .with_formulation(SWEFormulation2D::EntropyStable)
    ///     .with_bathymetry(&bathymetry)
    ///     .with_source_terms(&coriolis);
    /// ```
    pub fn with_formulation(mut self, formulation: SWEFormulation2D) -> Self {
        self.formulation = formulation;
        self
    }

    /// Set the numerical flux type.
    pub fn with_flux_type(mut self, flux_type: SWEFluxType2D) -> Self {
        self.flux_type = flux_type;
        self
    }

    /// Set whether to include Coriolis (legacy method).
    ///
    /// For new code, prefer using `with_source_terms` with `CoriolisSource2D`.
    pub fn with_coriolis(mut self, include: bool) -> Self {
        self.include_coriolis = include;
        self
    }

    /// Set trait-based source terms.
    ///
    /// Source terms are evaluated at each quadrature node and added to the RHS.
    /// Use `CombinedSource2D` to compose multiple source terms.
    ///
    /// # Example
    /// ```ignore
    /// let coriolis = CoriolisSource2D::norwegian_coast();
    /// let config = SWE2DRhsConfig::new(&equation, &bc)
    ///     .with_coriolis(false)  // Disable legacy Coriolis
    ///     .with_source_terms(&coriolis);
    /// ```
    pub fn with_source_terms(mut self, sources: &'a dyn SourceTerm2D) -> Self {
        self.source_terms = Some(sources);
        self
    }

    /// Set bathymetry data for well-balanced schemes.
    ///
    /// When bathymetry is provided, the values and gradients are passed to:
    /// - Boundary condition contexts (for correct ghost state computation)
    /// - Source term contexts (for bathymetry-dependent physics)
    ///
    /// # Example
    /// ```ignore
    /// let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
    ///     // Sill profile
    ///     0.5 * (-((x - 5.0).powi(2)) / 2.0).exp()
    /// });
    /// let config = SWE2DRhsConfig::new(&equation, &bc)
    ///     .with_bathymetry(&bathymetry);
    /// ```
    pub fn with_bathymetry(mut self, bathymetry: &'a Bathymetry2D) -> Self {
        self.bathymetry = Some(bathymetry);
        self
    }

    /// Enable hydrostatic reconstruction for well-balanced treatment of bathymetry.
    ///
    /// Uses the Audusse et al. (2004) hydrostatic reconstruction at element faces,
    /// which balances bathymetry *jumps* across faces while keeping discrete mass
    /// conservation exact. Lake-at-rest (η = h + B = const, u = v = 0) is preserved
    /// to machine precision when the in-element bathymetry is also balanced, see
    /// the requirements below.
    ///
    /// # Requirements
    /// - `bathymetry` must be set via `with_bathymetry()`
    /// - For nodal (non-constant within elements) bathymetry, **include**
    ///   `BathymetrySource2D` in `source_terms`; it balances the volume pressure
    ///   term. Only for cell-constant bathymetry (`Bathymetry2D::to_cell_average`),
    ///   whose gradient is zero, is the source term a no-op that may be omitted.
    /// - Even with the source term, nodal bathymetry is balanced exactly only when
    ///   deg B ≤ p/2 in each element.
    ///
    /// # Example
    /// ```ignore
    /// // Well-balanced scheme for steep Norwegian bathymetry
    /// let bathy_source = BathymetrySource2D::new(g);
    /// let config = SWE2DRhsConfig::new(&equation, &bc)
    ///     .with_bathymetry(&bathymetry)
    ///     .with_source_terms(&bathy_source)
    ///     .with_well_balanced(true);
    /// ```
    pub fn with_well_balanced(mut self, enable: bool) -> Self {
        self.well_balanced = enable;
        self
    }

    /// Set horizontal viscosity for momentum diffusion.
    ///
    /// When enabled, adds ∇·(ν h ∇u) and ∇·(ν h ∇v) to the momentum RHS.
    /// Use `compute_dt_viscosity` to compute the diffusive time step restriction,
    /// and take `dt = dt_advective.min(dt_viscous)`.
    pub fn with_viscosity(mut self, visc: &'a HorizontalViscosity2D) -> Self {
        self.viscosity = Some(visc);
        self
    }

    /// Set the dry-node depth of [`SWEFormulation2D::WetDry`].
    pub fn with_dry_threshold(mut self, h_dry: f64) -> Self {
        assert!(h_dry >= 0.0, "dry threshold must be non-negative");
        self.h_dry = h_dry;
        self
    }
}

/// Boundary state from the configured boundary condition at `node` of
/// boundary face `face` of element `k`: a ghost state for the Riemann solver,
/// or the state on the boundary whose physical flux is the face flux. Shared
/// by the standard and split-form kernels and the viscous boundary terms.
#[allow(clippy::too_many_arguments)]
pub(super) fn boundary_state<BC: SWEBoundaryCondition2D>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    k: ElementIndex,
    face: usize,
    node: usize,
    normal: (f64, f64),
) -> BoundaryState {
    let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[node], ops.nodes_s[node]);
    let state = q.get_state(k, node);
    let bathy_value = config.bathymetry.map_or(0.0, |b| b.get(k, node));
    let g = config.equation.g;
    let h_min = config.equation.h_min.meters();

    let ctx = match mesh.boundary_tag(k, face) {
        Some(tag) => BCContext2D::with_tag(time, (x, y), state, bathy_value, normal, g, h_min, tag),
        None => BCContext2D::new(time, (x, y), state, bathy_value, normal, g, h_min),
    }
    .with_node_index(k.as_usize() * ops.n_nodes + node);
    config.bc.boundary_state(&ctx)
}

#[allow(clippy::too_many_arguments)]
fn boundary_velocity_component<BC: SWEBoundaryCondition2D>(
    component: usize,
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    visc_h_min: f64,
    k: ElementIndex,
    face: usize,
    node: usize,
) -> f64 {
    let normal = geom.normals[k.as_usize()][face];
    let ghost = boundary_state(q, mesh, ops, config, time, k, face, node, normal).state();
    if ghost.h <= visc_h_min {
        return 0.0;
    }

    let h_safe = ghost.h.max(visc_h_min);
    match component {
        0 => ghost.hu / h_safe,
        1 => ghost.hv / h_safe,
        _ => unreachable!("invalid velocity component"),
    }
}

fn add_br1_viscosity<BC: SWEBoundaryCondition2D>(
    rhs: &mut SWESolution2D,
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
) {
    let Some(visc) = config.viscosity else {
        return;
    };

    let n_nodes = ops.n_nodes;
    let total_nodes = mesh.n_elements * n_nodes;
    let mut u = vec![0.0; total_nodes];
    let mut v = vec![0.0; total_nodes];

    for k in ElementIndex::iter(mesh.n_elements) {
        for i in 0..n_nodes {
            let flat = k.as_usize() * n_nodes + i;
            let state = q.get_state(k, i);
            if state.h > visc.h_min {
                let h_safe = state.h.max(visc.h_min);
                u[flat] = state.hu / h_safe;
                v[flat] = state.hv / h_safe;
            }
        }
    }

    let boundary_u_for_gradient = |k, face, _fi, node, _interior| {
        boundary_velocity_component(
            0, q, mesh, ops, geom, config, time, visc.h_min, k, face, node,
        )
    };
    let boundary_v_for_gradient = |k, face, _fi, node, _interior| {
        boundary_velocity_component(
            1, q, mesh, ops, geom, config, time, visc.h_min, k, face, node,
        )
    };

    let grad_u = compute_br1_gradient_2d(&u, mesh, ops, geom, boundary_u_for_gradient);
    let grad_v = compute_br1_gradient_2d(&v, mesh, ops, geom, boundary_v_for_gradient);

    let mut coeff = vec![0.0; total_nodes];
    for k in ElementIndex::iter(mesh.n_elements) {
        let delta = geom.det_j[k.as_usize()].sqrt();
        for i in 0..n_nodes {
            let flat = k.as_usize() * n_nodes + i;
            let state = q.get_state(k, i);
            if state.h > visc.h_min {
                let nu = visc.compute_viscosity(
                    grad_u[flat].dx,
                    grad_u[flat].dy,
                    grad_v[flat].dx,
                    grad_v[flat].dy,
                    delta,
                );
                coeff[flat] = nu * state.h;
            }
        }
    }

    let diff_u = compute_br1_diffusion_rhs_2d(&u, &coeff, &grad_u, mesh, ops, geom);
    let diff_v = compute_br1_diffusion_rhs_2d(&v, &coeff, &grad_v, mesh, ops, geom);

    for k in ElementIndex::iter(mesh.n_elements) {
        for i in 0..n_nodes {
            let flat = k.as_usize() * n_nodes + i;
            rhs.data[SWE_VAR_HU][flat] += diff_u[flat];
            rhs.data[SWE_VAR_HV][flat] += diff_v[flat];
        }
    }
}

/// Scratch space for one element of the RHS kernel (SoA, one `Vec` per
/// variable).
struct ElementWorkspace {
    n_nodes: usize,
    n_face_nodes: usize,
    flux_x: [Vec<f64>; 3],
    flux_y: [Vec<f64>; 3],
    dfx_dr: [Vec<f64>; 3],
    dfx_ds: [Vec<f64>; 3],
    dfy_dr: [Vec<f64>; 3],
    dfy_ds: [Vec<f64>; 3],
    /// hu, hv of the element (f-plane Coriolis kernel)
    momentum: [Vec<f64>; 2],
    /// F(q⁻)·n − F* at the face nodes
    flux_diff: [Vec<f64>; 3],
    /// Exterior (neighbour or ghost) states at the face nodes
    ext: [Vec<f64>; 3],
    /// Face nodes whose exterior state is a boundary state `q_b` with flux
    /// `F(q_b)·n` ([`BoundaryState::Exact`])
    ext_exact: Vec<bool>,
    int_bathy: Vec<f64>,
    ext_bathy: Vec<f64>,
    split_form: SplitFormWorkspace,
}

/// `n` zeros with two cache lines of unused capacity after them.
///
/// Workspaces of different threads are small and allocated at about the same
/// time, so without the slack their buffers can share cache lines, and every
/// write then invalidates another core's copy (false sharing): the cached
/// workspaces made the parallel RHS ~25 % slower than fresh ones.
fn padded(n: usize) -> Vec<f64> {
    const SLACK: usize = 16; // 128 bytes
    let mut v = Vec::with_capacity(n + SLACK);
    v.resize(n, 0.0);
    v
}

impl ElementWorkspace {
    fn new(n_nodes: usize, n_face_nodes: usize) -> Self {
        let nodes = || [padded(n_nodes), padded(n_nodes), padded(n_nodes)];
        let face = || {
            [
                padded(n_face_nodes),
                padded(n_face_nodes),
                padded(n_face_nodes),
            ]
        };
        Self {
            n_nodes,
            n_face_nodes,
            flux_x: nodes(),
            flux_y: nodes(),
            dfx_dr: nodes(),
            dfx_ds: nodes(),
            dfy_dr: nodes(),
            dfy_ds: nodes(),
            momentum: [padded(n_nodes), padded(n_nodes)],
            flux_diff: face(),
            ext: face(),
            ext_exact: vec![false; n_face_nodes],
            int_bathy: padded(n_face_nodes),
            ext_bathy: padded(n_face_nodes),
            split_form: SplitFormWorkspace::new(n_nodes),
        }
    }
}

thread_local! {
    /// Cached element workspace of this thread (rayon workers persist, so after
    /// the first RHS evaluation nothing is allocated).
    static ELEMENT_WORKSPACE: RefCell<Option<ElementWorkspace>> = const { RefCell::new(None) };
}

/// An element workspace taken from this thread's cache for one serial loop or
/// one rayon job, and returned to the cache on drop.
///
/// Taking it once per job instead of looking it up per element matters: a
/// per-element lookup (lock or thread-local) made the parallel RHS ~25 %
/// slower. A new workspace is allocated only if the cache is empty (first
/// use, or a nested job on the same thread) or sized for other operators.
struct WorkspaceGuard(Option<ElementWorkspace>);

impl WorkspaceGuard {
    fn take(ops: &DGOperators2D) -> Self {
        let (n_nodes, n_face_nodes) = (ops.n_nodes, ops.n_face_nodes);
        let cached = ELEMENT_WORKSPACE
            .with(|cell| cell.try_borrow_mut().ok().and_then(|mut slot| slot.take()))
            .filter(|ws| ws.n_nodes == n_nodes && ws.n_face_nodes == n_face_nodes);
        Self(Some(cached.unwrap_or_else(|| {
            ElementWorkspace::new(n_nodes, n_face_nodes)
        })))
    }
}

impl std::ops::Deref for WorkspaceGuard {
    type Target = ElementWorkspace;
    fn deref(&self) -> &ElementWorkspace {
        self.0.as_ref().expect("workspace present until drop")
    }
}

impl std::ops::DerefMut for WorkspaceGuard {
    fn deref_mut(&mut self) -> &mut ElementWorkspace {
        self.0.as_mut().expect("workspace present until drop")
    }
}

impl Drop for WorkspaceGuard {
    fn drop(&mut self) {
        let ws = self.0.take();
        // `try_with`: the thread-local may already be gone at thread exit
        let _ = ELEMENT_WORKSPACE.try_with(|cell| {
            if let Ok(mut slot) = cell.try_borrow_mut()
                && slot.is_none()
            {
                *slot = ws;
            }
        });
    }
}

thread_local! {
    /// Interior-face flux buffer of the split forms, cached on the thread
    /// that drives the RHS (reused by every evaluation after the first).
    static FACE_FLUXES: RefCell<Vec<SWEState2D>> = const { RefCell::new(Vec::new()) };
}

/// The face-flux buffer, taken from this thread's cache for one RHS
/// evaluation and returned on drop (a nested evaluation on the same thread
/// gets a fresh one).
struct FaceFluxGuard(Vec<SWEState2D>);

impl FaceFluxGuard {
    fn take() -> Self {
        Self(
            FACE_FLUXES
                .with(|cell| cell.try_borrow_mut().map(|mut v| std::mem::take(&mut *v)))
                .unwrap_or_default(),
        )
    }
}

impl std::ops::Deref for FaceFluxGuard {
    type Target = Vec<SWEState2D>;
    fn deref(&self) -> &Vec<SWEState2D> {
        &self.0
    }
}

impl std::ops::DerefMut for FaceFluxGuard {
    fn deref_mut(&mut self) -> &mut Vec<SWEState2D> {
        &mut self.0
    }
}

impl Drop for FaceFluxGuard {
    fn drop(&mut self) {
        let faces = std::mem::take(&mut self.0);
        // Keep the larger buffer; `try_with`: the thread-local may be gone at
        // thread exit
        let _ = FACE_FLUXES.try_with(|cell| {
            if let Ok(mut slot) = cell.try_borrow_mut()
                && slot.capacity() < faces.capacity()
            {
                *slot = faces;
            }
        });
    }
}

/// One 2D SWE RHS evaluation. `element` is the only place the per-element
/// terms are computed; the serial and parallel drivers just iterate over it.
struct SWE2DRhsKernel<'a, 'c, BC: SWEBoundaryCondition2D> {
    q: &'a SWESolution2D,
    mesh: &'a Mesh2D,
    ops: &'a DGOperators2D,
    geom: &'a GeometricFactors2D,
    config: &'a SWE2DRhsConfig<'c, BC>,
    time: f64,
    /// Split-form (Wintermeyer et al. 2017) operator, if selected
    split_form: Option<SplitFormSWE2D<'a, 'c, BC>>,
}

impl<'a, 'c, BC: SWEBoundaryCondition2D> SWE2DRhsKernel<'a, 'c, BC> {
    fn new(
        q: &'a SWESolution2D,
        mesh: &'a Mesh2D,
        ops: &'a DGOperators2D,
        geom: &'a GeometricFactors2D,
        config: &'a SWE2DRhsConfig<'c, BC>,
        time: f64,
    ) -> Self {
        Self {
            q,
            mesh,
            ops,
            geom,
            config,
            time,
            split_form: SplitFormSWE2D::new(q, mesh, ops, geom, config, time),
        }
    }

    /// Interior-face fluxes of the split forms, once per face, into `faces`
    /// (resized as needed; left empty for the collocated formulation).
    fn face_fluxes(&self, faces: &mut Vec<SWEState2D>) {
        let Some(split_form) = &self.split_form else {
            faces.clear();
            return;
        };
        faces.resize(split_form.face_buffer_len(), SWEState2D::zero());
        let per_edge = 2 * self.ops.n_face_nodes;
        for (e, slots) in faces.chunks_exact_mut(per_edge).enumerate() {
            split_form.edge_fluxes(e, slots);
        }
    }

    /// [`Self::face_fluxes`] over the edges in parallel (identical result).
    #[cfg(feature = "parallel")]
    fn face_fluxes_parallel(&self, faces: &mut Vec<SWEState2D>) {
        use rayon::prelude::*;

        let Some(split_form) = &self.split_form else {
            faces.clear();
            return;
        };
        faces.resize(split_form.face_buffer_len(), SWEState2D::zero());
        let per_edge = 2 * self.ops.n_face_nodes;
        faces
            .par_chunks_exact_mut(per_edge)
            .enumerate()
            .for_each(|(e, slots)| split_form.edge_fluxes(e, slots));
    }

    /// Volume, surface and source terms of element `k`, written (not added)
    /// to `out = [h, hu, hv]`; `faces` from [`Self::face_fluxes`].
    ///
    /// `face_mass`, if given (`4 · n_face_nodes` values), receives the mass
    /// component of `F*` at every face node of the element, along its outward
    /// normal (see [`compute_rhs_swe_2d_face_mass_into`]).
    fn element(
        &self,
        k: usize,
        ws: &mut ElementWorkspace,
        faces: &[SWEState2D],
        out: [&mut [f64]; 3],
        face_mass: Option<&mut [f64]>,
    ) {
        let k_idx = ElementIndex::new(k);
        let [out_h, out_hu, out_hv] = out;

        // 1–2. Volume and surface terms
        match &self.split_form {
            Some(split_form) => split_form.element_rhs(
                k_idx,
                &mut ws.split_form,
                faces,
                [out_h, out_hu, out_hv],
                face_mass,
            ),
            None => self.collocated_terms(k_idx, ws, [out_h, out_hu, out_hv], face_mass),
        }

        // 3–4. Source terms
        self.source_terms(k_idx, ws, [out_h, out_hu, out_hv]);
    }

    /// Collocated nodal DG volume and surface terms:
    ///   −J⁻¹[Dr·Fr + Ds·Fs] + J⁻¹ Σ_f LIFT_f sJ_f (F(q⁻)·n − F*)
    /// with Fr = F·∇r, Fs = F·∇s.
    fn collocated_terms(
        &self,
        k: ElementIndex,
        ws: &mut ElementWorkspace,
        out: [&mut [f64]; 3],
        mut face_mass: Option<&mut [f64]>,
    ) {
        let (q, mesh, ops, geom, config) = (self.q, self.mesh, self.ops, self.geom, self.config);
        let ki = k.as_usize();
        let n_nodes = ops.n_nodes;
        let n_face_nodes = ops.n_face_nodes;
        let g = config.equation.g;
        let h_min = config.equation.h_min.meters();
        let [out_h, out_hu, out_hv] = out;

        // 1. Volume term: −(∇·F), from the reference derivatives of the nodal flux
        {
            let [fx_h, fx_hu, fx_hv] = &mut ws.flux_x;
            let [fy_h, fy_hu, fy_hv] = &mut ws.flux_y;
            for i in 0..n_nodes {
                let state = q.get_state(k, i);
                let fx = config.equation.flux_x(&state);
                let fy = config.equation.flux_y(&state);
                (fx_h[i], fx_hu[i], fx_hv[i]) = (fx.h, fx.hu, fx.hv);
                (fy_h[i], fy_hu[i], fy_hv[i]) = (fy.h, fy.hu, fy.hv);
            }
        }
        for (d, flux, out) in [
            (&ops.dr_row_major, &ws.flux_x, &mut ws.dfx_dr),
            (&ops.ds_row_major, &ws.flux_x, &mut ws.dfx_ds),
            (&ops.dr_row_major, &ws.flux_y, &mut ws.dfy_dr),
            (&ops.ds_row_major, &ws.flux_y, &mut ws.dfy_ds),
        ] {
            let [o_h, o_hu, o_hv] = out;
            apply_diff_matrix(d, &flux[0], &flux[1], &flux[2], o_h, o_hu, o_hv, n_nodes);
        }
        combine_derivatives(
            &ws.dfx_dr[0],
            &ws.dfx_dr[1],
            &ws.dfx_dr[2],
            &ws.dfx_ds[0],
            &ws.dfx_ds[1],
            &ws.dfx_ds[2],
            &ws.dfy_dr[0],
            &ws.dfy_dr[1],
            &ws.dfy_dr[2],
            &ws.dfy_ds[0],
            &ws.dfy_ds[1],
            &ws.dfy_ds[2],
            out_h,
            out_hu,
            out_hv,
            geom.rx[ki],
            geom.sx[ki],
            geom.ry[ki],
            geom.sy[ki],
            n_nodes,
        );

        // 2. Surface terms
        let well_balanced = config.bathymetry.filter(|_| config.well_balanced);
        let hr = well_balanced.map(|_| HydrostaticReconstruction2D::new(g, h_min));
        let j_inv = geom.det_j_inv[ki];

        for face in 0..4 {
            let normal = geom.normals[ki][face];
            let face_nodes = &ops.face_nodes[face];
            let [ext_h, ext_hu, ext_hv] = &mut ws.ext;

            // Interior bathymetry at the face (well-balanced reconstruction only)
            match well_balanced {
                Some(b) => {
                    for (fi, &node) in face_nodes.iter().enumerate() {
                        ws.int_bathy[fi] = b.get(k, node);
                    }
                }
                None => ws.int_bathy.fill(0.0),
            }

            // Exterior states and bathymetry: neighbour (face nodes reversed) or ghost
            ws.ext_exact.fill(false);
            if let Some(neighbor) = mesh.neighbor(k, face) {
                let nb = ElementIndex::new(neighbor.element);
                let nb_face_nodes = &ops.face_nodes[neighbor.face];
                for fi in 0..n_face_nodes {
                    let nb_node = nb_face_nodes[n_face_nodes - 1 - fi];
                    let state = q.get_state(nb, nb_node);
                    (ext_h[fi], ext_hu[fi], ext_hv[fi]) = (state.h, state.hu, state.hv);
                    ws.ext_bathy[fi] = well_balanced.map_or(0.0, |b| b.get(nb, nb_node));
                }
            } else {
                for (fi, &node) in face_nodes.iter().enumerate() {
                    let ghost = match boundary_state(
                        q, mesh, ops, config, self.time, k, face, node, normal,
                    ) {
                        BoundaryState::Ghost(ghost) => ghost,
                        BoundaryState::Exact(q_b) => {
                            ws.ext_exact[fi] = true;
                            q_b
                        }
                    };
                    (ext_h[fi], ext_hu[fi], ext_hv[fi]) = (ghost.h, ghost.hu, ghost.hv);
                }
                // Boundary faces mirror the interior bathymetry
                ws.ext_bathy.copy_from_slice(&ws.int_bathy);
            }

            // Numerical flux and flux difference at the face nodes
            let [diff_h, diff_hu, diff_hv] = &mut ws.flux_diff;
            for (fi, &node) in face_nodes.iter().enumerate() {
                let q_int = q.get_state(k, node);
                let q_ext = SWEState2D::new(ext_h[fi], ext_hu[fi], ext_hv[fi]);

                // Boundary state q_b: the face flux is its physical flux F(q_b)·n
                if ws.ext_exact[fi] {
                    let f_b = config.equation.normal_flux(&q_ext, normal);
                    if let Some(face_mass) = face_mass.as_deref_mut() {
                        face_mass[face * n_face_nodes + fi] = f_b.h;
                    }
                    let diff = config.equation.normal_flux(&q_int, normal) - f_b;
                    (diff_h[fi], diff_hu[fi], diff_hv[fi]) = (diff.h, diff.hu, diff.hv);
                    continue;
                }

                // Hydrostatically reconstructed states for F* (if enabled)
                let (q_int_flux, q_ext_flux) = match &hr {
                    Some(r) => r.reconstruct(&q_int, &q_ext, ws.int_bathy[fi], ws.ext_bathy[fi]),
                    None => (q_int, q_ext),
                };
                let f_star = compute_flux_swe_2d(
                    &q_int_flux,
                    &q_ext_flux,
                    normal,
                    g,
                    h_min,
                    config.flux_type,
                );
                if let Some(face_mass) = face_mass.as_deref_mut() {
                    face_mass[face * n_face_nodes + fi] = f_star.h;
                }

                // Interior flux F(q⁻)·n of the actual nodal state. It must match the
                // volume term for the surface/volume pair to telescope (SBP); using the
                // reconstructed state here adds −∮(h⁻ − h*⁻)u⁻·n to the element mass.
                let mut diff = config.equation.normal_flux(&q_int, normal) - f_star;

                // Well-balancing enters as the Audusse momentum correction ½g(h*⁻² − h⁻²)·n
                if let Some(r) = &hr {
                    diff = diff + r.pressure_correction(q_int.h, q_int_flux.h, normal);
                }
                (diff_h[fi], diff_hu[fi], diff_hv[fi]) = (diff.h, diff.hu, diff.hv);
            }

            // out += J⁻¹ sJ LIFT_f (F(q⁻)·n − F*)
            apply_lift(
                &ops.lift_row_major[face],
                diff_h,
                diff_hu,
                diff_hv,
                out_h,
                out_hu,
                out_hv,
                n_nodes,
                n_face_nodes,
                j_inv * geom.surface_j[ki][face],
            );
        }
    }

    /// Legacy built-in Coriolis and the trait-based source terms, added to `out`.
    fn source_terms(&self, k: ElementIndex, ws: &mut ElementWorkspace, out: [&mut [f64]; 3]) {
        let (q, mesh, ops, config) = (self.q, self.mesh, self.ops, self.config);
        let n_nodes = ops.n_nodes;
        let g = config.equation.g;
        let h_min = config.equation.h_min.meters();
        let [out_h, out_hu, out_hv] = out;

        // 3. Legacy Coriolis (only without trait-based source terms)
        if config.include_coriolis
            && config.source_terms.is_none()
            && (config.equation.f0.abs() > 1e-14 || config.equation.beta.abs() > 1e-14)
        {
            if config.equation.beta.abs() < 1e-14 {
                let [hu, hv] = &mut ws.momentum;
                for i in 0..n_nodes {
                    let state = q.get_state(k, i);
                    (hu[i], hv[i]) = (state.hu, state.hv);
                }
                coriolis_source(hu, hv, out_hu, out_hv, config.equation.f0, n_nodes);
            } else {
                for i in 0..n_nodes {
                    let [_x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                    let source = config.equation.coriolis_source(&q.get_state(k, i), y);
                    out_hu[i] += source.hu;
                    out_hv[i] += source.hv;
                }
            }
        }

        // 4. Trait-based source terms (preferred), one call per element: each
        //    source reads only the node data it needs
        if let Some(sources) = config.source_terms {
            let element = ElementSources {
                element: k,
                time: self.time,
                solution: q,
                mesh,
                ops,
                bathymetry: config.bathymetry,
                g,
                h_min,
            };
            sources.add_element(&element, out_h, out_hu, out_hv);
        }
    }
}

/// Compute the right-hand side for 2D SWE.
///
/// Implements the DG weak form for a system:
///   dq/dt = -1/J * [Dr * (Fr) + Ds * (Fs)] + 1/J * Σ_f LIFT_f * sJ_f * (F(q⁻)·n - F*) + S
///
/// where:
///   Fr = F · ∇r = F_x * rx + F_y * ry
///   Fs = F · ∇s = F_x * sx + F_y * sy
///
/// Allocates the result; time steppers should reuse an output buffer with
/// [`compute_rhs_swe_2d_into`] instead.
pub fn compute_rhs_swe_2d<BC: SWEBoundaryCondition2D>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
) -> SWESolution2D {
    let mut rhs = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    compute_rhs_swe_2d_into(q, mesh, ops, geom, config, time, &mut rhs);
    rhs
}

/// [`compute_rhs_swe_2d`] into `out`, which is overwritten.
///
/// Allocation-free after the first call on a thread (the element workspace
/// is cached per thread), unless horizontal viscosity is enabled. Gives the same result, bit for bit, as
/// [`compute_rhs_swe_2d_parallel_into`].
pub fn compute_rhs_swe_2d_into<BC: SWEBoundaryCondition2D>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    out: &mut SWESolution2D,
) {
    rhs_serial(q, mesh, ops, geom, config, time, out, None);
}

/// [`compute_rhs_swe_2d_into`] that also writes the numerical mass flux of
/// every element face into `face_mass`.
///
/// Layout: `face_mass[(k · 4 + face) · n_face_nodes + fi]` is the mass
/// component of `F*` at face node `fi` of face `face` of element `k`, along
/// the element's outward normal (m²/s; [`face_mass_len`] values). With the
/// nodal `(hu, hv)` it is the transport whose DG divergence is the mass
/// tendency: `dh/dt = −(∇·(hu, hv) − J⁻¹ Σ_f LIFT_f sJ_f ((hu, hv)·n − F*_h))`
/// for the collocated and flux-differencing volume terms (the mass part of
/// the two-point flux is the mean of `(hu, hv)`). In `WetDry` elements with a
/// dry node the volume term is a subcell finite-volume update; there only the
/// element balance `∫ dh/dt = −∮ F*_h` holds.
#[allow(clippy::too_many_arguments)]
pub fn compute_rhs_swe_2d_face_mass_into<BC: SWEBoundaryCondition2D>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    out: &mut SWESolution2D,
    face_mass: &mut [f64],
) {
    rhs_serial(q, mesh, ops, geom, config, time, out, Some(face_mass));
}

/// Length of the face mass flux buffer of
/// [`compute_rhs_swe_2d_face_mass_into`]: four faces of `n_face_nodes` per
/// element.
pub fn face_mass_len(mesh: &Mesh2D, ops: &DGOperators2D) -> usize {
    mesh.n_elements * 4 * ops.n_face_nodes
}

#[allow(clippy::too_many_arguments)]
fn rhs_serial<BC: SWEBoundaryCondition2D>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    out: &mut SWESolution2D,
    face_mass: Option<&mut [f64]>,
) {
    check_rhs_output(out, mesh, ops);
    let kernel = SWE2DRhsKernel::new(q, mesh, ops, geom, config, time);
    let mut faces = FaceFluxGuard::take();
    kernel.face_fluxes(&mut faces);
    let n = ops.n_nodes;
    let per_element = 4 * ops.n_face_nodes;
    let mut face_mass = face_mass.map(|fm| {
        assert_eq!(
            fm.len(),
            face_mass_len(mesh, ops),
            "face mass flux buffer length"
        );
        fm.chunks_exact_mut(per_element)
    });
    let [out_h, out_hu, out_hv] = &mut out.data;
    let mut ws = WorkspaceGuard::take(ops);
    for (k, ((h, hu), hv)) in out_h
        .chunks_exact_mut(n)
        .zip(out_hu.chunks_exact_mut(n))
        .zip(out_hv.chunks_exact_mut(n))
        .enumerate()
    {
        let fm = face_mass.as_mut().and_then(Iterator::next);
        kernel.element(k, &mut ws, &faces, [h, hu, hv], fm);
    }
    add_br1_viscosity(out, q, mesh, ops, geom, config, time);
}

fn check_rhs_output(out: &SWESolution2D, mesh: &Mesh2D, ops: &DGOperators2D) {
    assert!(
        out.n_elements == mesh.n_elements && out.n_nodes == ops.n_nodes,
        "RHS output is {}×{}, mesh and operators need {}×{}",
        out.n_elements,
        out.n_nodes,
        mesh.n_elements,
        ops.n_nodes
    );
}

/// Compute the stable time step for 2D SWE.
///
/// Each node pairs its own wave speeds with its element's metric:
///
///   Δt = CFL/(2N+1) · min over elements k and nodes i of 4 / (λ_r + λ_s),
///   λ_r = |u·∇r| + c|∇r|,  λ_s = |u·∇s| + c|∇s|,
///
/// the spectral radii of the flux Jacobian along the reference directions
/// (reference length 2). On a square element of side Δx at rest this is
/// CFL·Δx/((2N+1)c), the classic form, so CFL values keep their meaning. On an
/// element of width w ≪ length L the limit follows w, not √(wL).
///
/// Summing the two directions is the 2D tensor-product bound (as in Trixi.jl's
/// `max_dt`); with it the SSP-RK3 + Zhang–Shu positivity bound is a single
/// CFL per order for every element shape, see [`positivity_cfl_swe_2d`].
///
/// Returns `f64::INFINITY` when every node is dry or at rest in zero depth.
pub fn compute_dt_swe_2d(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    geom: &GeometricFactors2D,
    equation: &ShallowWater2D,
    order: usize,
    cfl: f64,
) -> f64 {
    let max_rate = ElementIndex::iter(mesh.n_elements)
        .map(|k| element_max_reference_rate(q, geom, equation, k))
        .fold(0.0_f64, f64::max);
    dt_from_reference_rate(max_rate, order, cfl)
}

/// Parallel version of [`compute_dt_swe_2d`] (identical result).
#[cfg(feature = "parallel")]
pub fn compute_dt_swe_2d_parallel(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    geom: &GeometricFactors2D,
    equation: &ShallowWater2D,
    order: usize,
    cfl: f64,
) -> f64 {
    use rayon::prelude::*;

    let max_rate = (0..mesh.n_elements)
        .into_par_iter()
        .map(|k| element_max_reference_rate(q, geom, equation, ElementIndex::new(k)))
        .reduce(|| 0.0_f64, f64::max);
    dt_from_reference_rate(max_rate, order, cfl)
}

/// Largest CFL for which [`compute_dt_swe_2d`] satisfies the DGSEM positivity
/// bound of Zhang & Shu (2010) for SSP-RK3 with a positivity-preserving flux
/// (HLL, Rusanov/Lax–Friedrichs; not Roe).
///
/// The cell mean stays non-negative under forward Euler when
/// Δt·(λ_x/Δx + λ_y/Δy) ≤ ŵ₀ = 1/(N(N+1)), the first GLL weight on [0, 1].
/// In this module's units that is CFL ≤ (2N+1)/(2N(N+1)): 0.75, 0.42, 0.29,
/// 0.23 for N = 1–4, independent of element shape. SSP-RK3 (SSP coefficient 1)
/// inherits it. Wet/dry runs should use `cfl.min(positivity_cfl_swe_2d(N))`.
pub const fn positivity_cfl_swe_2d(order: usize) -> f64 {
    assert!(order >= 1, "positivity bound needs N ≥ 1");
    let n = order as f64;
    (2.0 * n + 1.0) / (2.0 * n * (n + 1.0))
}

/// max over the nodes of element `k` of (λ_r + λ_s)/4 (1/s).
#[inline]
fn element_max_reference_rate(
    q: &SWESolution2D,
    geom: &GeometricFactors2D,
    equation: &ShallowWater2D,
    k: ElementIndex,
) -> f64 {
    let ki = k.as_usize();
    let grad_r = (geom.rx[ki], geom.ry[ki]);
    let grad_s = (geom.sx[ki], geom.sy[ki]);
    let norm_r = (grad_r.0 * grad_r.0 + grad_r.1 * grad_r.1).sqrt();
    let norm_s = (grad_s.0 * grad_s.0 + grad_s.1 * grad_s.1).sqrt();
    let dir_r = (grad_r.0 / norm_r, grad_r.1 / norm_r);
    let dir_s = (grad_s.0 / norm_s, grad_s.1 / norm_s);

    (0..q.n_nodes)
        .map(|i| {
            let state = q.get_state(k, i);
            let lambda_r = norm_r * equation.max_wave_speed_normal(&state, dir_r);
            let lambda_s = norm_s * equation.max_wave_speed_normal(&state, dir_s);
            0.25 * (lambda_r + lambda_s)
        })
        .fold(0.0_f64, f64::max)
}

#[inline]
fn dt_from_reference_rate(max_rate: f64, order: usize, cfl: f64) -> f64 {
    if max_rate < 1e-14 {
        return f64::INFINITY;
    }
    cfl / ((2.0 * order as f64 + 1.0) * max_rate)
}

/// Compute the diffusive time step restriction for horizontal viscosity.
///
/// The diffusive CFL condition is:
///   Δt ≤ CFL × Δx² / (ν × (2N+1)²)
///
/// where N is the polynomial order and (2N+1)² accounts for the DG
/// spectral radius of the diffusion operator.
///
/// Users should combine with the advective time step:
///   `dt = compute_dt_swe_2d(...).min(compute_dt_viscosity(...))`
///
/// # Arguments
/// * `nu_max` - Maximum viscosity coefficient [m²/s]
/// * `min_h_elem` - Minimum element size (e.g. from `geom.det_j[k].sqrt() * 2.0`)
/// * `order` - Polynomial order of the DG scheme
/// * `cfl` - CFL number (typically 0.1–0.5 for diffusion)
pub fn compute_dt_viscosity(nu_max: f64, min_h_elem: f64, order: usize, cfl: f64) -> f64 {
    if nu_max < 1e-14 {
        return f64::INFINITY;
    }
    let dg_factor = (2.0 * order as f64 + 1.0).powi(2);
    cfl * min_h_elem * min_h_elem / (nu_max * dg_factor)
}

/// Parallel version of [`compute_rhs_swe_2d`] (identical result).
#[cfg(feature = "parallel")]
pub fn compute_rhs_swe_2d_parallel<BC: SWEBoundaryCondition2D + Sync>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
) -> SWESolution2D {
    let mut rhs = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    compute_rhs_swe_2d_parallel_into(q, mesh, ops, geom, config, time, &mut rhs);
    rhs
}

/// Parallel version of [`compute_rhs_swe_2d_into`]: the same element kernel
/// over `par_chunks_mut`, so the result is identical bit for bit.
#[cfg(feature = "parallel")]
pub fn compute_rhs_swe_2d_parallel_into<BC: SWEBoundaryCondition2D + Sync>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    out: &mut SWESolution2D,
) {
    rhs_parallel(q, mesh, ops, geom, config, time, out, None);
}

/// Parallel version of [`compute_rhs_swe_2d_face_mass_into`] (identical
/// result).
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
pub fn compute_rhs_swe_2d_parallel_face_mass_into<BC: SWEBoundaryCondition2D + Sync>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    out: &mut SWESolution2D,
    face_mass: &mut [f64],
) {
    rhs_parallel(q, mesh, ops, geom, config, time, out, Some(face_mass));
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn rhs_parallel<BC: SWEBoundaryCondition2D + Sync>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    out: &mut SWESolution2D,
    face_mass: Option<&mut [f64]>,
) {
    use rayon::prelude::*;

    check_rhs_output(out, mesh, ops);
    let kernel = SWE2DRhsKernel::new(q, mesh, ops, geom, config, time);
    let mut faces = FaceFluxGuard::take();
    kernel.face_fluxes_parallel(&mut faces);
    let faces: &[SWEState2D] = &faces;
    let n = ops.n_nodes;
    let [out_h, out_hu, out_hv] = &mut out.data;
    let elements = out_h
        .par_chunks_exact_mut(n)
        .zip(out_hu.par_chunks_exact_mut(n))
        .zip(out_hv.par_chunks_exact_mut(n))
        .enumerate();
    match face_mass {
        Some(face_mass) => {
            assert_eq!(
                face_mass.len(),
                face_mass_len(mesh, ops),
                "face mass flux buffer length"
            );
            elements
                .zip(face_mass.par_chunks_exact_mut(4 * ops.n_face_nodes))
                .for_each_init(
                    || WorkspaceGuard::take(ops),
                    |ws, ((k, ((h, hu), hv)), fm)| {
                        kernel.element(k, ws, faces, [h, hu, hv], Some(fm))
                    },
                );
        }
        None => elements.for_each_init(
            || WorkspaceGuard::take(ops),
            |ws, (k, ((h, hu), hv))| kernel.element(k, ws, faces, [h, hu, hv], None),
        ),
    }
    add_br1_viscosity(out, q, mesh, ops, geom, config, time);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::Reflective2D;
    use crate::source::CoriolisSource2D;

    const G: f64 = 10.0;

    /// The face mass flux output (mode splitting, TODO P4.1) leaves the RHS
    /// unchanged bit for bit, is the same serial and parallel, and balances
    /// every element, `∫_K dh/dt = −∮_K F*_h`, for every formulation, with
    /// a dry region (the `WetDry` subcells included).
    #[test]
    fn face_mass_flux_balances_each_element() {
        let (mesh, ops, geom) = create_test_setup(2);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _| -1.0 + 2.0 * x);
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let eta = 0.2 + 0.05 * (2.0 * std::f64::consts::PI * x).sin();
                let h = (eta - bathymetry.get(k, i)).max(0.0);
                let (u, v) = (
                    0.2 * (std::f64::consts::PI * y).cos(),
                    0.1 * (std::f64::consts::PI * x).sin(),
                );
                q.set_state(k, i, SWEState2D::new(h, h * u, h * v));
            }
        }
        let n_face = ops.n_face_nodes;

        for formulation in [
            SWEFormulation2D::Standard,
            SWEFormulation2D::EntropyStable,
            SWEFormulation2D::WetDry,
        ] {
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_coriolis(false)
                .with_formulation(formulation)
                .with_bathymetry(&bathymetry)
                .with_flux_type(SWEFluxType2D::HLL);
            let plain = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
            let mut rhs = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
            let mut face_mass = vec![f64::NAN; face_mass_len(&mesh, &ops)];
            compute_rhs_swe_2d_face_mass_into(
                &q,
                &mesh,
                &ops,
                &geom,
                &config,
                0.0,
                &mut rhs,
                &mut face_mass,
            );
            assert_eq!(
                rhs.data, plain.data,
                "{formulation:?}: the face output changed the RHS"
            );
            assert!(
                face_mass.iter().all(|f| f.is_finite()),
                "{formulation:?}: face slots not all written"
            );

            #[cfg(feature = "parallel")]
            {
                let mut rhs_par = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
                let mut face_par = vec![f64::NAN; face_mass.len()];
                compute_rhs_swe_2d_parallel_face_mass_into(
                    &q,
                    &mesh,
                    &ops,
                    &geom,
                    &config,
                    0.0,
                    &mut rhs_par,
                    &mut face_par,
                );
                assert_eq!(rhs_par.data, rhs.data, "{formulation:?}: parallel RHS");
                assert_eq!(face_par, face_mass, "{formulation:?}: parallel face fluxes");
            }

            let mut max_inflow: f64 = 0.0;
            for k in 0..mesh.n_elements {
                let dh = &rhs.data[0][k * ops.n_nodes..(k + 1) * ops.n_nodes];
                let volume: f64 =
                    geom.det_j[k] * dh.iter().zip(&ops.weights).map(|(d, w)| d * w).sum::<f64>();
                let outflow: f64 = (0..4)
                    .map(|face| {
                        let f = &face_mass[(k * 4 + face) * n_face..][..n_face];
                        geom.surface_j[k][face]
                            * f.iter()
                                .zip(&ops.weights_1d)
                                .map(|(f, w)| f * w)
                                .sum::<f64>()
                    })
                    .sum();
                max_inflow = max_inflow.max(outflow.abs());
                assert!(
                    (volume + outflow).abs() < 1e-12 * max_inflow.max(1e-3),
                    "{formulation:?}, element {k}: ∫dh/dt = {volume:.3e}, ∮F* = {outflow:.3e}"
                );
            }
        }
    }

    fn create_test_setup(order: usize) -> (Mesh2D, DGOperators2D, GeometricFactors2D) {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh);
        (mesh, ops, geom)
    }

    #[test]
    fn test_rhs_still_water() {
        // Lake at rest: h = const, u = v = 0, bathymetry = 0
        // RHS should be zero (well-balanced test)
        let (mesh, ops, geom) = create_test_setup(2);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        // Set h = 2.0, u = v = 0 everywhere
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(2.0, 0.0, 0.0));
            }
        }

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        // Check RHS is zero (lake at rest)
        let mut max_rhs: f64 = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let state = rhs.get_state(k, i);
                max_rhs = max_rhs.max(state.h.abs());
                max_rhs = max_rhs.max(state.hu.abs());
                max_rhs = max_rhs.max(state.hv.abs());
            }
        }

        assert!(
            max_rhs < 1e-10,
            "Lake at rest RHS should be zero, got {}",
            max_rhs
        );
    }

    #[test]
    fn test_rhs_uniform_flow() {
        // Uniform flow: h = const, u = const, v = 0
        // RHS should be zero (steady state)
        let (_mesh, ops, _geom) = create_test_setup(2);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

        // Use periodic mesh to avoid boundary effects
        let mesh_periodic = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let geom_periodic = GeometricFactors2D::compute(&mesh_periodic);

        let mut q = SWESolution2D::new(mesh_periodic.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh_periodic.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::from_primitives(2.0, 1.0, 0.0));
            }
        }

        let rhs = compute_rhs_swe_2d(&q, &mesh_periodic, &ops, &geom_periodic, &config, 0.0);

        // Uniform flow should give zero RHS (for periodic domain)
        let mut max_rhs: f64 = 0.0;
        for k in ElementIndex::iter(mesh_periodic.n_elements) {
            for i in 0..ops.n_nodes {
                let state = rhs.get_state(k, i);
                max_rhs = max_rhs.max(state.h.abs());
                max_rhs = max_rhs.max(state.hu.abs());
                max_rhs = max_rhs.max(state.hv.abs());
            }
        }

        assert!(
            max_rhs < 1e-10,
            "Uniform flow RHS should be zero, got {}",
            max_rhs
        );
    }

    #[test]
    fn test_mass_conservation() {
        // Test that total mass is conserved (RHS of h integrates to zero)
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

        // Non-uniform initial condition
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        q.set_from_functions(
            &mesh,
            &ops,
            |x, y| {
                1.0 + 0.5
                    * (2.0 * std::f64::consts::PI * x).sin()
                    * (2.0 * std::f64::consts::PI * y).sin()
            },
            |_, _| 0.0,
            |_, _| 0.0,
        );

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        // Integrate RHS of h
        let mut integral = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            let ki = k.as_usize();
            let j = geom.det_j[ki];
            for (i, &w) in ops.weights.iter().enumerate() {
                integral += w * rhs.get_var(k, i, 0) * j;
            }
        }

        assert!(
            integral.abs() < 1e-10,
            "Mass should be conserved: d(mass)/dt = {:.2e}",
            integral
        );
    }

    #[test]
    fn test_dt_computation() {
        let (mesh, ops, geom) = create_test_setup(2);
        let equation = ShallowWater2D::new(G);

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        // h = 1, u = 2 -> wave speed ≈ 2 + sqrt(10) ≈ 5.16
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::from_primitives(1.0, 2.0, 0.0));
            }
        }

        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, 0.5);

        assert!(dt > 0.0);
        assert!(dt < f64::INFINITY);
        assert!(dt < 0.1); // Should be small for this test case
    }

    /// Depth `depth(k)` in element k, velocity (u, 0) everywhere.
    fn element_wise_state(
        mesh: &Mesh2D,
        n_nodes: usize,
        depth: impl Fn(usize) -> f64,
        u: f64,
    ) -> SWESolution2D {
        let mut q = SWESolution2D::new(mesh.n_elements, n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..n_nodes {
                q.set_state(
                    k,
                    i,
                    SWEState2D::from_primitives(depth(k.as_usize()), u, 0.0),
                );
            }
        }
        q
    }

    #[test]
    fn test_dt_square_elements_match_classic_formula() {
        // At rest on squares the per-direction bound reduces to CFL·Δx/((2N+1)c),
        // so CFL numbers keep the meaning they had before P0.21.
        let order = 2;
        let (cfl, dx, depth) = (0.4, 250.0, 100.0);
        let mesh = Mesh2D::uniform_rectangle(0.0, 4.0 * dx, 0.0, 3.0 * dx, 4, 3);
        let geom = GeometricFactors2D::compute(&mesh);
        let n_nodes = (order + 1) * (order + 1);
        let equation = ShallowWater2D::new(G);
        let c = (G * depth).sqrt();
        let dg = 2.0 * order as f64 + 1.0;

        let q = element_wise_state(&mesh, n_nodes, |_| depth, 0.0);
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, order, cfl);
        let expected = cfl * dx / (dg * c);
        assert!(
            (dt / expected - 1.0).abs() < 1e-12,
            "dt = {dt}, expected {expected}"
        );

        // Flow along x: λ_r = 2(|u| + c)/Δx, λ_s = 2c/Δx.
        let u = 3.0;
        let q = element_wise_state(&mesh, n_nodes, |_| depth, u);
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, order, cfl);
        let expected = 2.0 * cfl * dx / (dg * (u + 2.0 * c));
        assert!(
            (dt / expected - 1.0).abs() < 1e-12,
            "dt = {dt}, expected {expected}"
        );
    }

    /// Two elements side by side, 50 m × 1000 m and 1000 m × 1000 m.
    fn graded_anisotropic_mesh() -> Mesh2D {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 1);
        for v in mesh.vertices.iter_mut() {
            v[0] = if v[0] < 0.75 {
                100.0 * v[0]
            } else {
                50.0 + 1000.0 * (v[0] - 0.5) * 2.0
            };
            v[1] *= 1000.0;
        }
        mesh
    }

    #[test]
    fn test_dt_pairs_local_wave_speed_with_local_anisotropic_size() {
        // P0.21 regression. The old bound used min_k 2√detJ_k with max over all
        // nodes of |u| + c: here √(50·1000) = 224 m paired with the deep-water
        // celerity, ≈ 3.8× too small, while for the thin element alone √detJ
        // overestimates the 50 m width.
        let order = 3;
        let cfl = 0.25;
        let mesh = graded_anisotropic_mesh();
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let n_nodes = (order + 1) * (order + 1);
        let depths = [5.0, 400.0];
        let q = element_wise_state(&mesh, n_nodes, |k| depths[k], 0.0);

        let dg = 2.0 * order as f64 + 1.0;
        let sizes = [(50.0, 1000.0), (1000.0, 1000.0)];
        let expected = depths
            .iter()
            .zip(sizes)
            .map(|(&h, (dx, dy))| 2.0 * cfl / (dg * (G * h).sqrt() * (1.0 / dx + 1.0 / dy)))
            .fold(f64::INFINITY, f64::min);

        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, order, cfl);
        assert!(
            (dt / expected - 1.0).abs() < 1e-12,
            "dt = {dt}, expected {expected}"
        );

        let old = cfl * (4.0 * geom.det_j[0]).sqrt() / (dg * (G * 400.0).sqrt());
        assert!(
            dt > 3.0 * old,
            "dt = {dt} should exceed the old global pairing {old}"
        );

        // The thin element alone: limited by its 50 m width, below the √detJ size.
        let q = element_wise_state(&mesh, n_nodes, |_| 5.0, 0.0);
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, order, cfl);
        let sqrt_det_j = cfl * (4.0 * geom.det_j[0]).sqrt() / (dg * (G * 5.0).sqrt());
        assert!(
            dt < 0.5 * sqrt_det_j,
            "dt = {dt} vs √detJ-based {sqrt_det_j}"
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_dt_parallel_matches_serial() {
        let order = 2;
        let mesh = graded_anisotropic_mesh();
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let q = element_wise_state(&mesh, (order + 1) * (order + 1), |k| [3.0, 250.0][k], 1.5);
        let serial = compute_dt_swe_2d(&q, &mesh, &geom, &equation, order, 0.3);
        let parallel = compute_dt_swe_2d_parallel(&q, &mesh, &geom, &equation, order, 0.3);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn test_positivity_cfl_values() {
        // (2N+1)/(2N(N+1)): REVIEW.md §1.7 quotes 0.75 / 0.42 / 0.29 / 0.23.
        let expected = [0.75, 5.0 / 12.0, 7.0 / 24.0, 9.0 / 40.0];
        for (n, &e) in (1..=4).zip(&expected) {
            assert!((positivity_cfl_swe_2d(n) - e).abs() < 1e-15);
        }
    }

    #[test]
    fn test_dt_dry_domain_is_unbounded() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let q = SWESolution2D::new(mesh.n_elements, 9);
        assert_eq!(
            compute_dt_swe_2d(&q, &mesh, &geom, &equation, 2, 0.5),
            f64::INFINITY
        );
    }

    #[test]
    fn test_rhs_into_overwrites_and_reuses_workspace() {
        // P1.1: `_into` must overwrite a reused (stale) output buffer and give the
        // allocating result, for both formulations and repeated evaluations (the
        // thread-local workspace is reused across them and across orders).
        let (mesh, ops, geom, bathymetry) = sloped_periodic_setup(2, false);
        let q = sloped_state(&mesh, &ops, &bathymetry, 0.5);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let bathy_source = crate::source::BathymetrySource2D::new(G);
        let standard = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(true);
        let split = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_formulation(SWEFormulation2D::EntropyStable)
            .with_bathymetry(&bathymetry);

        let mut out = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for config in [&standard, &split, &standard] {
            let expected = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, config, 0.0);
            for v in out.data.iter_mut().flatten() {
                *v = 1.0e30;
            }
            compute_rhs_swe_2d_into(&q, &mesh, &ops, &geom, config, 0.0, &mut out);
            assert_eq!(out.data, expected.data);
            #[cfg(feature = "parallel")]
            {
                for v in out.data.iter_mut().flatten() {
                    *v = -1.0e30;
                }
                compute_rhs_swe_2d_parallel_into(&q, &mesh, &ops, &geom, config, 0.0, &mut out);
                assert_eq!(out.data, expected.data);
            }
        }
    }

    #[test]
    #[should_panic(expected = "RHS output is")]
    fn test_rhs_into_rejects_wrong_output_size() {
        let (mesh, ops, geom) = create_test_setup(2);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc);
        let q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        let mut out = SWESolution2D::new(mesh.n_elements + 1, ops.n_nodes);
        compute_rhs_swe_2d_into(&q, &mesh, &ops, &geom, &config, 0.0, &mut out);
    }

    #[test]
    fn test_coriolis_source() {
        // Test that Coriolis source term is included correctly
        let ops = DGOperators2D::new(2);
        let equation = ShallowWater2D::with_coriolis(G, 1.0e-4);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(true);

        // Use periodic mesh to eliminate surface terms
        let mesh_periodic = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let geom_periodic = GeometricFactors2D::compute(&mesh_periodic);

        let mut q = SWESolution2D::new(mesh_periodic.n_elements, ops.n_nodes);
        // h = 10, hu = 100 (so v-momentum source = f*hu = 1e-4 * 100 = 0.01)
        for k in ElementIndex::iter(mesh_periodic.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(10.0, 100.0, 0.0));
            }
        }

        let rhs_with = compute_rhs_swe_2d(&q, &mesh_periodic, &ops, &geom_periodic, &config, 0.0);

        let config_no_coriolis = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);
        let rhs_without = compute_rhs_swe_2d(
            &q,
            &mesh_periodic,
            &ops,
            &geom_periodic,
            &config_no_coriolis,
            0.0,
        );

        // The difference should be the Coriolis source
        // Source: d(hu)/dt += f*hv = 0, d(hv)/dt -= f*hu = -0.01
        let k0 = ElementIndex::new(0);
        let diff_hv = rhs_with.get_var(k0, 0, 2) - rhs_without.get_var(k0, 0, 2);
        let expected = -1.0e-4 * 100.0; // -f*hu

        assert!(
            (diff_hv - expected).abs() < 1e-10,
            "Coriolis source difference: got {}, expected {}",
            diff_hv,
            expected
        );
    }

    #[test]
    fn test_stability_single_step() {
        // Test that a single time step doesn't blow up
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

        // Smooth initial condition
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        q.set_from_functions(
            &mesh,
            &ops,
            |x, y| {
                1.0 + 0.1
                    * (2.0 * std::f64::consts::PI * x).sin()
                    * (2.0 * std::f64::consts::PI * y).sin()
            },
            |_, _| 0.0,
            |_, _| 0.0,
        );

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        // RHS should be bounded
        let max_rhs = rhs.max_abs();
        assert!(max_rhs < 100.0, "RHS should be bounded, got {}", max_rhs);

        // Do one forward Euler step
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, 0.1);
        q.axpy(dt, &rhs);

        // Solution should still be bounded
        assert!(!q.has_negative_depth(), "Depth should remain positive");
        assert!(
            q.max_abs() < 100.0,
            "Solution should be bounded after one step"
        );
    }

    #[test]
    fn test_flux_types() {
        // Test that different flux types work
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::from_primitives(2.0, 0.5, 0.2));
            }
        }

        for flux_type in [
            SWEFluxType2D::Roe,
            SWEFluxType2D::HLL,
            SWEFluxType2D::Rusanov,
        ] {
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_flux_type(flux_type)
                .with_coriolis(false);

            let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

            // For uniform state, all fluxes should give zero RHS
            let max_rhs = rhs.max_abs();
            assert!(
                max_rhs < 1e-10,
                "Uniform state should give zero RHS for {:?}, got {}",
                flux_type,
                max_rhs
            );
        }
    }

    #[test]
    fn test_trait_based_coriolis() {
        // Test that trait-based CoriolisSource2D gives same result as legacy Coriolis
        let ops = DGOperators2D::new(2);
        let equation = ShallowWater2D::with_coriolis(G, 1.0e-4);
        let bc = Reflective2D::new();

        // Use periodic mesh to eliminate surface terms
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let geom = GeometricFactors2D::compute(&mesh);

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(10.0, 100.0, 50.0));
            }
        }

        // Legacy Coriolis
        let config_legacy = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(true);
        let rhs_legacy = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config_legacy, 0.0);

        // Trait-based Coriolis
        let coriolis_source = CoriolisSource2D::f_plane(1.0e-4);
        let config_trait = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false) // Disable legacy
            .with_source_terms(&coriolis_source);
        let rhs_trait = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config_trait, 0.0);

        // Compare results
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let legacy = rhs_legacy.get_state(k, i);
                let trait_based = rhs_trait.get_state(k, i);

                assert!(
                    (legacy.h - trait_based.h).abs() < 1e-12,
                    "h mismatch at ({}, {}): {} vs {}",
                    k,
                    i,
                    legacy.h,
                    trait_based.h
                );
                assert!(
                    (legacy.hu - trait_based.hu).abs() < 1e-12,
                    "hu mismatch at ({}, {}): {} vs {}",
                    k,
                    i,
                    legacy.hu,
                    trait_based.hu
                );
                assert!(
                    (legacy.hv - trait_based.hv).abs() < 1e-12,
                    "hv mismatch at ({}, {}): {} vs {}",
                    k,
                    i,
                    legacy.hv,
                    trait_based.hv
                );
            }
        }
    }

    #[test]
    fn test_well_balanced_steep_slope() {
        // Test lake-at-rest preservation with steep bathymetry gradient
        // This tests the hydrostatic reconstruction implementation
        // Note: We need BOTH reconstruction (for surface terms) AND source term (for volume balance)
        use crate::mesh::Bathymetry2D;
        use crate::source::BathymetrySource2D;

        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        // Steep bathymetry: B = 0.5 * x (Norwegian-like gradient)
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| 0.5 * x);

        // Lake-at-rest: η = h + B = 10.0
        let eta_ref = 10.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let b = bathymetry.get(k, i);
                let h = (eta_ref - b).max(0.0);
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }

        // Config WITH hydrostatic reconstruction AND bathymetry source term
        // Reconstruction: handles surface flux well-balancing
        // Source term: balances volume term pressure gradient
        let bathy_source = BathymetrySource2D::new(G);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(true);

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        // RHS should be zero to machine precision
        let max_rhs = rhs.max_abs();
        assert!(
            max_rhs < 1e-10,
            "Lake at rest with hydrostatic reconstruction: max RHS = {:.2e}, expected < 1e-10",
            max_rhs
        );
    }

    #[test]
    fn test_well_balanced_bilinear_slope() {
        // Test with bilinear bathymetry (x + y direction slope)
        // This is still linear within elements, so well-balanced should work perfectly
        use crate::mesh::Bathymetry2D;
        use crate::source::BathymetrySource2D;

        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 6, 6);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        // Bilinear bathymetry: B = 0.3*x + 0.2*y
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| 0.3 * x + 0.2 * y);

        // Lake-at-rest: η = 10.0
        let eta_ref = 10.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let b = bathymetry.get(k, i);
                let h = (eta_ref - b).max(0.0);
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }

        let bathy_source = BathymetrySource2D::new(G);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(true);

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        let max_rhs = rhs.max_abs();
        assert!(
            max_rhs < 1e-10,
            "Lake at rest (bilinear slope): max RHS = {:.2e}",
            max_rhs
        );
    }

    #[test]
    fn test_well_balanced_all_flux_types() {
        // Verify all flux types work with hydrostatic reconstruction
        use crate::flux::SWEFluxType2D;
        use crate::mesh::Bathymetry2D;
        use crate::source::BathymetrySource2D;

        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| 0.3 * x + 0.4 * y);

        let eta_ref = 10.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let b = bathymetry.get(k, i);
                q.set_state(k, i, SWEState2D::new((eta_ref - b).max(0.0), 0.0, 0.0));
            }
        }

        let bathy_source = BathymetrySource2D::new(G);
        for flux_type in [
            SWEFluxType2D::Roe,
            SWEFluxType2D::HLL,
            SWEFluxType2D::Rusanov,
        ] {
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_flux_type(flux_type)
                .with_coriolis(false)
                .with_bathymetry(&bathymetry)
                .with_source_terms(&bathy_source)
                .with_well_balanced(true);

            let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
            let max_rhs = rhs.max_abs();

            assert!(
                max_rhs < 1e-10,
                "{:?} flux with well-balanced: max RHS = {:.2e}",
                flux_type,
                max_rhs
            );
        }
    }

    #[test]
    fn test_source_term_alone_well_balanced_linear() {
        // For LINEAR bathymetry, the standard DG scheme with bathymetry source term
        // is already well-balanced because the volume gradient and source term
        // are discretized consistently using the same DG differentiation operators.
        // This test verifies this property.
        use crate::mesh::Bathymetry2D;
        use crate::source::BathymetrySource2D;

        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        // Linear bathymetry
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| 0.5 * x);

        let eta_ref = 10.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let b = bathymetry.get(k, i);
                q.set_state(k, i, SWEState2D::new((eta_ref - b).max(0.0), 0.0, 0.0));
            }
        }

        // Config with source term only (no reconstruction)
        let bathy_source = BathymetrySource2D::new(G);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(false); // No reconstruction

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        let max_rhs = rhs.max_abs();

        // For linear bathymetry, source term alone should be well-balanced
        // (DG computes gradients exactly for linear functions)
        assert!(
            max_rhs < 1e-10,
            "Source term alone should be well-balanced for linear bathymetry: {:.2e}",
            max_rhs
        );
    }

    #[test]
    fn test_reconstruction_improves_accuracy() {
        // Verify that hydrostatic reconstruction is correctly applied
        // by checking that reconstructed states match expected values.
        use crate::mesh::Bathymetry2D;
        use crate::source::BathymetrySource2D;

        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        // Linear bathymetry
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| 0.5 * x);

        let eta_ref = 10.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let b = bathymetry.get(k, i);
                q.set_state(k, i, SWEState2D::new((eta_ref - b).max(0.0), 0.0, 0.0));
            }
        }

        // Config WITH reconstruction AND source term
        let bathy_source = BathymetrySource2D::new(G);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(true);

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        let max_rhs = rhs.max_abs();

        // With reconstruction, should also be well-balanced
        assert!(
            max_rhs < 1e-10,
            "With reconstruction should be well-balanced: {:.2e}",
            max_rhs
        );
    }

    /// Periodic 20 km domain with smooth bathymetry of large amplitude.
    ///
    /// B = −200 + 150·sin(2πx/L)·cos(2πy/L), optionally cell-averaged so that B
    /// jumps across every element face (as `examples/froya_real_data.rs` does).
    fn sloped_periodic_setup(
        order: usize,
        cell_average: bool,
    ) -> (Mesh2D, DGOperators2D, GeometricFactors2D, Bathymetry2D) {
        const L: f64 = 20_000.0;
        let mesh = Mesh2D::uniform_periodic(0.0, L, 0.0, L, 12, 12);
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh);
        let tau = 2.0 * std::f64::consts::PI / L;
        let mut bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
            -200.0 + 150.0 * (tau * x).sin() * (tau * y).cos()
        });
        if cell_average {
            bathymetry.to_cell_average();
        }
        (mesh, ops, geom, bathymetry)
    }

    /// State with free surface η = 0.3 m over `bathymetry` and momentum
    /// hu = velocity_scale·h·cos(2πx/L)·cos(2πy/L), hv = 0.
    ///
    /// The velocity is correlated with the bathymetry slope on purpose: a
    /// uniform velocity hides reconstruction-induced mass errors by symmetry.
    fn sloped_state(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        bathymetry: &Bathymetry2D,
        velocity_scale: f64,
    ) -> SWESolution2D {
        let tau = 2.0 * std::f64::consts::PI / 20_000.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let h = 0.3 - bathymetry.get(k, i);
                let hu = velocity_scale * h * (tau * x).cos() * (tau * y).cos();
                q.set_state(k, i, SWEState2D::new(h, hu, 0.0));
            }
        }
        q
    }

    /// ∫ q_var dA using the GLL quadrature.
    fn integrate_var(
        q: &SWESolution2D,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        var: usize,
    ) -> f64 {
        let mut integral = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            let j = geom.det_j[k.as_usize()];
            for (i, &w) in ops.weights.iter().enumerate() {
                integral += w * j * q.get_var(k, i, var);
            }
        }
        integral
    }

    #[test]
    fn test_mass_conservation_hydrostatic_reconstruction_discontinuous_bathymetry() {
        // Regression (REVIEW.md §1.3): with cell-averaged B the reconstruction is
        // active at every face. Evaluating the interior flux on the reconstructed
        // state broke the SBP telescoping and leaked ≈5e-5 of the total mass per
        // second on this configuration.
        use crate::source::BathymetrySource2D;

        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let bathy_source = BathymetrySource2D::new(G);

        for order in 1..=3 {
            for cell_average in [false, true] {
                let (mesh, ops, geom, bathymetry) = sloped_periodic_setup(order, cell_average);
                let q = sloped_state(&mesh, &ops, &bathymetry, 1.0);
                let config = SWE2DRhsConfig::new(&equation, &bc)
                    .with_coriolis(false)
                    .with_bathymetry(&bathymetry)
                    .with_source_terms(&bathy_source)
                    .with_well_balanced(true);

                let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
                let mass = integrate_var(&q, &mesh, &ops, &geom, 0);
                let mass_rate = integrate_var(&rhs, &mesh, &ops, &geom, 0);

                assert!(
                    mass_rate.abs() / mass < 1e-12,
                    "p={order}, cell_average={cell_average}: \
                     d(mass)/dt / mass = {:.3e} (d(mass)/dt = {:.3e} m³/s)",
                    mass_rate / mass,
                    mass_rate
                );
            }
        }
    }

    #[test]
    fn test_lake_at_rest_cell_averaged_bathymetry() {
        // Cell-constant B: zero volume gradient, every bathymetry effect comes from
        // the hydrostatic interface correction. Must stay balanced to round-off.
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        for order in 1..=3 {
            let (mesh, ops, geom, bathymetry) = sloped_periodic_setup(order, true);
            let q = sloped_state(&mesh, &ops, &bathymetry, 0.0);
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_coriolis(false)
                .with_bathymetry(&bathymetry)
                .with_well_balanced(true);

            let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
            let max_rhs = rhs.max_abs();
            assert!(
                max_rhs < 1e-10,
                "p={order}: lake at rest over cell-averaged B, max RHS = {max_rhs:.3e}"
            );

            #[cfg(all(feature = "parallel", feature = "simd"))]
            {
                let rhs_par = compute_rhs_swe_2d_parallel(&q, &mesh, &ops, &geom, &config, 0.0);
                let max_rhs_par = rhs_par.max_abs();
                assert!(
                    max_rhs_par < 1e-10,
                    "p={order}: parallel lake at rest over cell-averaged B, \
                     max RHS = {max_rhs_par:.3e}"
                );
            }
        }
    }

    #[test]
    fn test_lake_at_rest_linearized_bathymetry() {
        // P0.14: `Bathymetry2D::linearize` used to claim planar B is well-balanced.
        // With the collocated scheme that holds only for p ≥ 2 (½gh² resolved);
        // at p = 1 the interpolation of ½gh² leaves g(h̄ − hᵢ)∂B. The split form
        // is balanced at every order.
        use crate::source::BathymetrySource2D;

        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let bathy_source = BathymetrySource2D::new(G);

        for order in 1..=3 {
            let (mesh, ops, geom, mut bathymetry) = sloped_periodic_setup(order, false);
            bathymetry.linearize(&mesh, &ops, &geom);
            let q = sloped_state(&mesh, &ops, &bathymetry, 0.0);

            let standard = SWE2DRhsConfig::new(&equation, &bc)
                .with_coriolis(false)
                .with_bathymetry(&bathymetry)
                .with_source_terms(&bathy_source)
                .with_well_balanced(true);
            let residual = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &standard, 0.0).max_abs();
            if order == 1 {
                assert!(
                    residual > 1e-3,
                    "p=1: expected an aliasing residual for planar B, got {residual:.3e}"
                );
            } else {
                assert!(
                    residual < 1e-10,
                    "p={order}: lake at rest over planar B, max RHS = {residual:.3e}"
                );
            }

            let split = SWE2DRhsConfig::new(&equation, &bc)
                .with_coriolis(false)
                .with_formulation(SWEFormulation2D::EntropyStable)
                .with_bathymetry(&bathymetry);
            let residual = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &split, 0.0).max_abs();
            assert!(
                residual < 1e-9,
                "p={order}: split form over planar B, max RHS = {residual:.3e}"
            );
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_matches_serial() {
        use crate::source::BathymetrySource2D;

        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        // Case 1: flat bottom, non-uniform flow.
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh);
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [x, y] = mesh.reference_to_physical(k, r, s);
                let h =
                    2.0 + 0.1 * (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).cos();
                let u = 0.5 * (std::f64::consts::PI * y).sin();
                let v = 0.3 * (std::f64::consts::PI * x).cos();
                q.set_state(k, i, SWEState2D::from_primitives(h, u, v));
            }
        }
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);
        assert_parallel_matches_serial(&q, &mesh, &ops, &geom, &config, "flat bottom");

        // Case 2: cell-averaged (face-discontinuous) bathymetry with hydrostatic
        // reconstruction active at every face and a slope-correlated flow.
        let (mesh, ops, geom, bathymetry) = sloped_periodic_setup(3, true);
        let q = sloped_state(&mesh, &ops, &bathymetry, 1.0);
        let bathy_source = BathymetrySource2D::new(G);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(true);
        assert_parallel_matches_serial(
            &q,
            &mesh,
            &ops,
            &geom,
            &config,
            "cell-averaged bathymetry + reconstruction",
        );

        // Case 3: reflective walls (ghost states), nodal bathymetry, trait sources
        // including Coriolis, both formulations.
        let mesh = Mesh2D::uniform_rectangle(0.0, 20_000.0, 0.0, 10_000.0, 6, 3);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
            -150.0 + 0.004 * x - 50.0 * (y / 10_000.0).powi(2)
        });
        let q = sloped_state(&mesh, &ops, &bathymetry, 0.8);
        let coriolis = CoriolisSource2D::f_plane(1.2e-4);
        let sources = crate::source::CombinedSource2D::new(vec![&coriolis, &bathy_source]);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&sources)
            .with_well_balanced(true);
        assert_parallel_matches_serial(&q, &mesh, &ops, &geom, &config, "walls + sources");
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_formulation(SWEFormulation2D::EntropyStable)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&coriolis);
        assert_parallel_matches_serial(&q, &mesh, &ops, &geom, &config, "walls, split form");
    }

    #[cfg(feature = "parallel")]
    fn assert_parallel_matches_serial(
        q: &SWESolution2D,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        config: &SWE2DRhsConfig<Reflective2D>,
        case: &str,
    ) {
        // P1.1: one element kernel for both drivers, so the results are identical.
        let rhs_serial = compute_rhs_swe_2d(q, mesh, ops, geom, config, 0.0);
        let rhs_parallel = compute_rhs_swe_2d_parallel(q, mesh, ops, geom, config, 0.0);
        assert!(rhs_serial.max_abs() > 0.0, "{case}: trivial RHS");
        assert_eq!(
            rhs_serial.data, rhs_parallel.data,
            "{case}: serial != parallel"
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_with_coriolis() {
        use super::compute_rhs_swe_2d_parallel;

        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::with_coriolis(G, 1.0e-4); // f-plane Coriolis
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(true);

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(10.0, 100.0, 50.0));
            }
        }

        let rhs_serial = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
        let rhs_parallel = compute_rhs_swe_2d_parallel(&q, &mesh, &ops, &geom, &config, 0.0);

        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let serial = rhs_serial.get_state(k, i);
                let parallel = rhs_parallel.get_state(k, i);

                assert!(
                    (serial.h - parallel.h).abs() < 1e-10,
                    "h mismatch with Coriolis at ({}, {})",
                    k,
                    i
                );
                assert!(
                    (serial.hu - parallel.hu).abs() < 1e-10,
                    "hu mismatch with Coriolis at ({}, {})",
                    k,
                    i
                );
                assert!(
                    (serial.hv - parallel.hv).abs() < 1e-10,
                    "hv mismatch with Coriolis at ({}, {})",
                    k,
                    i
                );
            }
        }
    }
}
