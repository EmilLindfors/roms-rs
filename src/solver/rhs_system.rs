//! Right-hand side computation for shallow water equations.
//!
//! Computes dq/dt for the DG discretization of the 1D shallow water equations:
//!
//! ∂h/∂t + ∂(hu)/∂x = 0
//! ∂(hu)/∂t + ∂(hu² + gh²/2)/∂x = -gh ∂B/∂x - τ_f
//!
//! The DG semi-discrete form is:
//! dq/dt = -Dr * F(q) / J + LIFT * (F* - F⁻) / J + S(q)

use crate::boundary::{BCContext, SWEBoundaryCondition};
use crate::equations::{ConservationLaw, ShallowWater1D};
use crate::flux::{hll_flux_swe, roe_flux_swe};
use crate::mesh::{Bathymetry1D, Mesh1D};
use crate::operators::DGOperators1D;
use crate::solver::{SWESolution, SWEState};
use crate::source::{HydrostaticReconstruction, SourceTerm};

/// Numerical flux type for SWE.
#[derive(Clone, Copy, Debug, Default)]
pub enum SWEFluxType {
    /// Roe approximate Riemann solver
    #[default]
    Roe,
    /// HLL two-wave solver
    Hll,
    /// Lax-Friedrichs (Rusanov)
    LaxFriedrichs,
}

/// Configuration for SWE RHS computation.
#[derive(Clone)]
pub struct SWERhsConfig<'a> {
    /// Shallow water equation parameters
    pub equation: &'a ShallowWater1D,
    /// Numerical flux type
    pub flux_type: SWEFluxType,
    /// Left boundary condition
    pub bc_left: &'a dyn SWEBoundaryCondition,
    /// Right boundary condition
    pub bc_right: &'a dyn SWEBoundaryCondition,
    /// Bathymetry data (optional, defaults to flat)
    pub bathymetry: Option<&'a Bathymetry1D>,
    /// Additional source terms (friction, etc.)
    pub source: Option<&'a dyn SourceTerm>,
    /// Use well-balanced scheme for bathymetry
    pub well_balanced: bool,
}

impl<'a> SWERhsConfig<'a> {
    /// Create a minimal config with flat bathymetry.
    pub fn new(
        equation: &'a ShallowWater1D,
        bc_left: &'a dyn SWEBoundaryCondition,
        bc_right: &'a dyn SWEBoundaryCondition,
    ) -> Self {
        Self {
            equation,
            flux_type: SWEFluxType::Roe,
            bc_left,
            bc_right,
            bathymetry: None,
            source: None,
            well_balanced: true,
        }
    }

    /// Set bathymetry.
    pub fn with_bathymetry(mut self, bathymetry: &'a Bathymetry1D) -> Self {
        self.bathymetry = Some(bathymetry);
        self
    }

    /// Set additional source term.
    pub fn with_source(mut self, source: &'a dyn SourceTerm) -> Self {
        self.source = Some(source);
        self
    }

    /// Set numerical flux type.
    pub fn with_flux_type(mut self, flux_type: SWEFluxType) -> Self {
        self.flux_type = flux_type;
        self
    }

    /// Disable well-balancing.
    pub fn without_well_balancing(mut self) -> Self {
        self.well_balanced = false;
        self
    }
}

/// Compute the right-hand side of the SWE DG discretization.
///
/// For dq/dt + dF/dx = S, the DG semi-discrete form is:
/// dq/dt = -Dr * F(q) / J + LIFT * (F* - F⁻) / J + S
///
/// # Arguments
/// * `q` - Current solution
/// * `mesh` - 1D mesh
/// * `ops` - DG operators
/// * `config` - RHS configuration
/// * `time` - Current simulation time
#[allow(clippy::needless_range_loop)]
pub fn compute_rhs_swe(
    q: &SWESolution,
    mesh: &Mesh1D,
    ops: &DGOperators1D,
    config: &SWERhsConfig,
    time: f64,
) -> SWESolution {
    let n = ops.n_nodes;
    let g = config.equation.g;
    let h_min = config.equation.h_min;

    let mut rhs = SWESolution::new(mesh.n_elements, n);

    // Hydrostatic reconstruction for well-balancing
    let hr = HydrostaticReconstruction::new(g, h_min);

    for k in 0..mesh.n_elements {
        let j_inv = mesh.jacobian_inv(k);

        // Get bathymetry for this element
        let get_bath = |i: usize| config.bathymetry.map(|b| b.get(k, i)).unwrap_or(0.0);

        let get_bath_grad = |i: usize| {
            config
                .bathymetry
                .map(|b| b.get_gradient(k, i))
                .unwrap_or(0.0)
        };

        // ============================================
        // 1. VOLUME TERM: -Dr * F(q) / J
        // ============================================

        // Compute flux at each node
        let mut flux_h = vec![0.0; n];
        let mut flux_hu = vec![0.0; n];

        for i in 0..n {
            let state = q.get_state(k, i);
            let f = compute_physical_flux(&state, g, h_min);
            flux_h[i] = f.h;
            flux_hu[i] = f.hu;
        }

        // Apply differentiation matrix: dF/dr = Dr * F
        let mut dflux_h = vec![0.0; n];
        let mut dflux_hu = vec![0.0; n];

        for i in 0..n {
            for j in 0..n {
                dflux_h[i] += ops.dr[(i, j)] * flux_h[j];
                dflux_hu[i] += ops.dr[(i, j)] * flux_hu[j];
            }
        }

        // Scale by -j_inv (volume contribution to RHS)
        for i in 0..n {
            let state = q.get_state(k, i);
            let mut rhs_h = -j_inv * dflux_h[i];
            let mut rhs_hu = -j_inv * dflux_hu[i];

            // ============================================
            // 2. SOURCE TERMS (volume integral)
            // ============================================

            // Bathymetry source: S_hu = -g h dB/dx
            let db_dx = get_bath_grad(i);
            rhs_hu -= g * state.h * db_dx;

            // Additional source terms (friction, etc.)
            if let Some(source) = config.source {
                let x = mesh.reference_to_physical(k, ops.nodes[i]);
                let s = source.evaluate(&state, db_dx, x, time);
                rhs_h += s.h;
                rhs_hu += s.hu;
            }

            // Store volume contribution
            rhs.set(k, i, [rhs_h, rhs_hu]);
        }

        // ============================================
        // 3. SURFACE TERMS: LIFT * (F* - F⁻) / J
        // ============================================

        // Left face (local node 0, normal = -1)
        let (flux_jump_left_h, flux_jump_left_hu) = {
            let q_int = q.get_state(k, 0);
            let b_int = get_bath(0);
            let q_ext = get_exterior_state(q, mesh, config, k, 0, time);
            let b_ext = get_exterior_bathymetry(config.bathymetry, mesh, k, 0);

            compute_flux_jump(&q_int, &q_ext, b_int, b_ext, -1.0, config, &hr)
        };

        // Right face (local node n-1, normal = +1)
        let (flux_jump_right_h, flux_jump_right_hu) = {
            let q_int = q.get_state(k, n - 1);
            let b_int = get_bath(n - 1);
            let q_ext = get_exterior_state(q, mesh, config, k, 1, time);
            let b_ext = get_exterior_bathymetry(config.bathymetry, mesh, k, 1);

            compute_flux_jump(&q_int, &q_ext, b_int, b_ext, 1.0, config, &hr)
        };

        // Apply LIFT operator
        for i in 0..n {
            let lift_h = ops.lift[(i, 0)] * flux_jump_left_h + ops.lift[(i, 1)] * flux_jump_right_h;
            let lift_hu =
                ops.lift[(i, 0)] * flux_jump_left_hu + ops.lift[(i, 1)] * flux_jump_right_hu;

            let [rhs_h, rhs_hu] = rhs.get(k, i);
            rhs.set(k, i, [rhs_h + j_inv * lift_h, rhs_hu + j_inv * lift_hu]);
        }
    }

    rhs
}

/// Compute physical flux F(q) = [hu, hu²/h + gh²/2].
fn compute_physical_flux(state: &SWEState, g: f64, h_min: f64) -> SWEState {
    if state.h <= h_min {
        return SWEState::zero();
    }

    let u = state.hu / state.h;
    SWEState::new(state.hu, state.hu * u + 0.5 * g * state.h * state.h)
}

/// Get exterior state at a face (from neighbor or boundary condition).
fn get_exterior_state(
    q: &SWESolution,
    mesh: &Mesh1D,
    config: &SWERhsConfig,
    element: usize,
    face: usize, // 0 = left, 1 = right
    time: f64,
) -> SWEState {
    let n = q.n_nodes;

    // Check if this is a boundary
    if let Some(boundary) = mesh.is_boundary(element, face) {
        // Get interior state and bathymetry for BC context
        let (interior_state, bathymetry, position, normal) = match boundary {
            crate::mesh::BoundaryFace::Left => {
                let state = q.get_state(element, 0);
                let bath = config.bathymetry.map(|b| b.get(element, 0)).unwrap_or(0.0);
                let pos = mesh.reference_to_physical(element, -1.0);
                (state, bath, pos, -1.0)
            }
            crate::mesh::BoundaryFace::Right => {
                let state = q.get_state(element, n - 1);
                let bath = config
                    .bathymetry
                    .map(|b| b.get(element, n - 1))
                    .unwrap_or(0.0);
                let pos = mesh.reference_to_physical(element, 1.0);
                (state, bath, pos, 1.0)
            }
        };

        let ctx = BCContext::new(time, position, interior_state, bathymetry, normal);

        // Apply appropriate boundary condition
        match boundary {
            crate::mesh::BoundaryFace::Left => config.bc_left.ghost_state(&ctx),
            crate::mesh::BoundaryFace::Right => config.bc_right.ghost_state(&ctx),
        }
    } else {
        // Interior face - get from neighbor
        match face {
            0 => {
                // Left face: neighbor's right node
                let neighbor = mesh.neighbors[element].0.unwrap();
                q.get_state(neighbor, n - 1)
            }
            1 => {
                // Right face: neighbor's left node
                let neighbor = mesh.neighbors[element].1.unwrap();
                q.get_state(neighbor, 0)
            }
            _ => panic!("Invalid face index"),
        }
    }
}

/// Get exterior bathymetry at a face.
fn get_exterior_bathymetry(
    bathymetry: Option<&Bathymetry1D>,
    mesh: &Mesh1D,
    element: usize,
    face: usize,
) -> f64 {
    let Some(bathy) = bathymetry else {
        return 0.0;
    };

    let n = bathy.n_nodes;

    // Check if boundary
    if mesh.is_boundary(element, face).is_some() {
        // Use interior bathymetry at boundary
        match face {
            0 => bathy.get(element, 0),
            1 => bathy.get(element, n - 1),
            _ => 0.0,
        }
    } else {
        // Interior face - get from neighbor
        match face {
            0 => {
                let neighbor = mesh.neighbors[element].0.unwrap();
                bathy.get(neighbor, n - 1)
            }
            1 => {
                let neighbor = mesh.neighbors[element].1.unwrap();
                bathy.get(neighbor, 0)
            }
            _ => 0.0,
        }
    }
}

/// Compute flux jump (F* - F⁻) · n at a face.
fn compute_flux_jump(
    q_int: &SWEState,
    q_ext: &SWEState,
    b_int: f64,
    b_ext: f64,
    normal: f64,
    config: &SWERhsConfig,
    hr: &HydrostaticReconstruction,
) -> (f64, f64) {
    let g = config.equation.g;
    let h_min = config.equation.h_min;

    // Apply hydrostatic reconstruction for well-balancing
    let (q_int_star, q_ext_star) = if config.well_balanced {
        hr.reconstruct(q_int, q_ext, b_int, b_ext)
    } else {
        (*q_int, *q_ext)
    };

    // Compute numerical flux
    let f_star = match config.flux_type {
        SWEFluxType::Roe => {
            if normal > 0.0 {
                roe_flux_swe(&q_int_star, &q_ext_star, g, h_min)
            } else {
                let f = roe_flux_swe(&q_ext_star, &q_int_star, g, h_min);
                SWEState::new(-f.h, -f.hu)
            }
        }
        SWEFluxType::Hll => {
            if normal > 0.0 {
                hll_flux_swe(&q_int_star, &q_ext_star, g, h_min)
            } else {
                let f = hll_flux_swe(&q_ext_star, &q_int_star, g, h_min);
                SWEState::new(-f.h, -f.hu)
            }
        }
        SWEFluxType::LaxFriedrichs => {
            compute_lax_friedrichs_flux(&q_int_star, &q_ext_star, normal, g, h_min)
        }
    };

    // Physical flux at interior (using reconstructed state for well-balancing)
    let f_int = compute_physical_flux(&q_int_star, g, h_min);
    let f_int_n = SWEState::new(f_int.h * normal, f_int.hu * normal);

    // Flux jump
    (f_star.h - f_int_n.h, f_star.hu - f_int_n.hu)
}

/// Lax-Friedrichs flux for SWE.
fn compute_lax_friedrichs_flux(
    q_l: &SWEState,
    q_r: &SWEState,
    normal: f64,
    g: f64,
    h_min: f64,
) -> SWEState {
    let f_l = compute_physical_flux(q_l, g, h_min);
    let f_r = compute_physical_flux(q_r, g, h_min);

    // Maximum wave speed
    let lambda_l = if q_l.h > h_min {
        (q_l.hu / q_l.h).abs() + (g * q_l.h).sqrt()
    } else {
        0.0
    };
    let lambda_r = if q_r.h > h_min {
        (q_r.hu / q_r.h).abs() + (g * q_r.h).sqrt()
    } else {
        0.0
    };
    let lambda = lambda_l.max(lambda_r);

    // F* = 0.5 * (F_l + F_r) * n - 0.5 * λ * (q_r - q_l)
    if normal > 0.0 {
        SWEState::new(
            0.5 * (f_l.h + f_r.h) - 0.5 * lambda * (q_r.h - q_l.h),
            0.5 * (f_l.hu + f_r.hu) - 0.5 * lambda * (q_r.hu - q_l.hu),
        )
    } else {
        SWEState::new(
            -0.5 * (f_l.h + f_r.h) - 0.5 * lambda * (q_r.h - q_l.h),
            -0.5 * (f_l.hu + f_r.hu) - 0.5 * lambda * (q_r.hu - q_l.hu),
        )
    }
}

/// Compute maximum wave speed across the domain for CFL calculation.
pub fn compute_max_wave_speed(q: &SWESolution, equation: &ShallowWater1D) -> f64 {
    let mut max_speed: f64 = 0.0;

    for k in 0..q.n_elements {
        for i in 0..q.n_nodes {
            let state = q.get_state(k, i);
            let speed = equation.max_wave_speed(&state.to_array());
            max_speed = max_speed.max(speed);
        }
    }

    max_speed
}

/// Compute stable time step for SWE.
pub fn compute_dt_swe(
    q: &SWESolution,
    mesh: &Mesh1D,
    equation: &ShallowWater1D,
    order: usize,
    cfl: f64,
) -> f64 {
    let max_speed = compute_max_wave_speed(q, equation);

    if max_speed < 1e-10 {
        // Still water - use a default based on gravity wave speed
        let h_max = q.max_abs_var(0).max(1.0);
        let c = (equation.g * h_max).sqrt();
        cfl * mesh.h_min() / (c * (2 * order + 1) as f64)
    } else {
        cfl * mesh.h_min() / (max_speed * (2 * order + 1) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::ReflectiveBC;

    const G: f64 = 10.0;
    const TOL: f64 = 1e-10;

    fn setup() -> (Mesh1D, DGOperators1D, ShallowWater1D) {
        let mesh = Mesh1D::uniform(0.0, 10.0, 10);
        let ops = DGOperators1D::new(3);
        let eq = ShallowWater1D::new(G);
        (mesh, ops, eq)
    }

    #[test]
    fn test_rhs_still_water_flat() {
        let (mesh, ops, eq) = setup();
        let bc = ReflectiveBC::new();

        // Still water on flat bottom
        let mut q = SWESolution::new(mesh.n_elements, ops.n_nodes);
        q.set_from_functions(&mesh, &ops, |_| 2.0, |_| 0.0);

        let config = SWERhsConfig::new(&eq, &bc, &bc);
        let rhs = compute_rhs_swe(&q, &mesh, &ops, &config, 0.0);

        // RHS should be zero for still water
        for k in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let [rhs_h, rhs_hu] = rhs.get(k, i);
                assert!(
                    rhs_h.abs() < 1e-10,
                    "RHS_h should be zero, got {} at elem {}, node {}",
                    rhs_h,
                    k,
                    i
                );
                assert!(
                    rhs_hu.abs() < 1e-10,
                    "RHS_hu should be zero, got {} at elem {}, node {}",
                    rhs_hu,
                    k,
                    i
                );
            }
        }
    }

    #[test]
    fn test_rhs_lake_at_rest_with_bathymetry() {
        let (mesh, ops, eq) = setup();
        let bc = ReflectiveBC::new();

        // Lake at rest: η = h + B = const = 3.0
        let eta = 3.0;
        let bathy = Bathymetry1D::from_function(&mesh, &ops, |x| 0.5 * (x / 10.0)); // Slope

        let mut q = SWESolution::new(mesh.n_elements, ops.n_nodes);
        for k in 0..mesh.n_elements {
            for (i, &r) in ops.nodes.iter().enumerate() {
                let x = mesh.reference_to_physical(k, r);
                let b = 0.5 * (x / 10.0);
                let h = (eta - b).max(0.0);
                q.set_state(k, i, SWEState::new(h, 0.0));
            }
        }

        let config = SWERhsConfig::new(&eq, &bc, &bc).with_bathymetry(&bathy);
        let rhs = compute_rhs_swe(&q, &mesh, &ops, &config, 0.0);

        // RHS should be zero for lake at rest (well-balanced property)
        for k in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let [rhs_h, rhs_hu] = rhs.get(k, i);
                assert!(
                    rhs_h.abs() < 1e-8,
                    "Lake at rest: RHS_h should be ~0, got {} at elem {}, node {}",
                    rhs_h,
                    k,
                    i
                );
                assert!(
                    rhs_hu.abs() < 1e-8,
                    "Lake at rest: RHS_hu should be ~0, got {} at elem {}, node {}",
                    rhs_hu,
                    k,
                    i
                );
            }
        }
    }

    #[test]
    fn test_rhs_uniform_flow() {
        let (mesh, ops, eq) = setup();
        let bc = ReflectiveBC::new();

        // Uniform flow: h = 2, u = 1
        let mut q = SWESolution::new(mesh.n_elements, ops.n_nodes);
        q.set_from_functions(&mesh, &ops, |_| 2.0, |_| 1.0);

        let config = SWERhsConfig::new(&eq, &bc, &bc);
        let rhs = compute_rhs_swe(&q, &mesh, &ops, &config, 0.0);

        // For uniform flow on flat bottom with periodic-like interior,
        // RHS should be small in the interior (boundaries may differ)
        for k in 1..mesh.n_elements - 1 {
            for i in 1..ops.n_nodes - 1 {
                let [rhs_h, _] = rhs.get(k, i);
                assert!(
                    rhs_h.abs() < 1e-8,
                    "Uniform flow interior: RHS_h should be ~0, got {}",
                    rhs_h
                );
                // Momentum RHS may be non-zero due to BC effects
            }
        }
    }

    #[test]
    fn test_rhs_dam_break_nonzero() {
        let (mesh, ops, eq) = setup();
        let bc = ReflectiveBC::new();

        // Dam break: h = 2 left, h = 1 right
        let mut q = SWESolution::new(mesh.n_elements, ops.n_nodes);
        for k in 0..mesh.n_elements {
            for (i, &r) in ops.nodes.iter().enumerate() {
                let x = mesh.reference_to_physical(k, r);
                let h = if x < 5.0 { 2.0 } else { 1.0 };
                q.set_state(k, i, SWEState::new(h, 0.0));
            }
        }

        let config = SWERhsConfig::new(&eq, &bc, &bc);
        let rhs = compute_rhs_swe(&q, &mesh, &ops, &config, 0.0);

        // RHS should be non-zero (especially near the dam)
        let max_rhs = rhs.max_abs();
        assert!(max_rhs > 0.1, "Dam break should produce non-zero RHS");
    }

    #[test]
    fn test_flux_types_consistency() {
        let (mesh, ops, eq) = setup();
        let bc = ReflectiveBC::new();

        let mut q = SWESolution::new(mesh.n_elements, ops.n_nodes);
        q.set_from_functions(&mesh, &ops, |_| 2.0, |_| 0.0);

        // All flux types should give same result for still water
        let config_roe = SWERhsConfig::new(&eq, &bc, &bc).with_flux_type(SWEFluxType::Roe);
        let config_hll = SWERhsConfig::new(&eq, &bc, &bc).with_flux_type(SWEFluxType::Hll);
        let config_lf = SWERhsConfig::new(&eq, &bc, &bc).with_flux_type(SWEFluxType::LaxFriedrichs);

        let rhs_roe = compute_rhs_swe(&q, &mesh, &ops, &config_roe, 0.0);
        let rhs_hll = compute_rhs_swe(&q, &mesh, &ops, &config_hll, 0.0);
        let rhs_lf = compute_rhs_swe(&q, &mesh, &ops, &config_lf, 0.0);

        // All should be approximately zero for still water
        assert!(rhs_roe.max_abs() < 1e-10);
        assert!(rhs_hll.max_abs() < 1e-10);
        assert!(rhs_lf.max_abs() < 1e-10);
    }

    #[test]
    fn test_compute_max_wave_speed() {
        let (mesh, ops, eq) = setup();

        let mut q = SWESolution::new(mesh.n_elements, ops.n_nodes);
        // h = 2, u = 1, c = sqrt(20) ≈ 4.47, max_speed = |u| + c ≈ 5.47
        q.set_from_functions(&mesh, &ops, |_| 2.0, |_| 1.0);

        let speed = compute_max_wave_speed(&q, &eq);
        let expected = 1.0 + (G * 2.0_f64).sqrt();

        assert!((speed - expected).abs() < TOL);
    }

    #[test]
    fn test_compute_dt_swe() {
        let (mesh, ops, eq) = setup();

        let mut q = SWESolution::new(mesh.n_elements, ops.n_nodes);
        q.set_from_functions(&mesh, &ops, |_| 2.0, |_| 1.0);

        let dt = compute_dt_swe(&q, &mesh, &eq, ops.order, 0.5);

        assert!(dt > 0.0);
        assert!(dt < 1.0); // Should be reasonably small
    }
}
