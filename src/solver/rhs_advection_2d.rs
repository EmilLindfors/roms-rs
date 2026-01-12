//! Right-hand side computation for 2D DG advection.
//!
//! For the 2D advection equation du/dt + a_x ∂u/∂x + a_y ∂u/∂y = 0,
//! the DG semi-discrete form uses the weak formulation:
//!
//! du/dt = L(u) = -(volume terms) + (surface terms)
//!
//! Volume term:
//!   -1/J * [Dr * (a_x * rx + a_y * ry) + Ds * (a_x * sx + a_y * sy)] * u
//!
//! Surface term:
//!   1/J * LIFT_f * sJ_f * (F^* - F^-) for each face f

use crate::equations::Advection2D;
use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::DGSolution2D;

/// Numerical flux type for advection.
#[derive(Clone, Copy, Debug, Default)]
pub enum AdvectionFluxType {
    /// Upwind flux (exact for linear advection)
    #[default]
    Upwind,
    /// Lax-Friedrichs flux (more dissipative)
    LaxFriedrichs,
}

/// Boundary condition for 2D advection.
///
/// For each boundary face, provides the exterior (ghost) value.
pub trait AdvectionBoundaryCondition2D: Send + Sync {
    /// Get the ghost state for a boundary face.
    ///
    /// # Arguments
    /// * `u_interior` - Interior solution value at the face
    /// * `x`, `y` - Physical coordinates of the face node
    /// * `normal` - Outward unit normal (nx, ny)
    /// * `time` - Current simulation time
    /// * `equation` - The advection equation (for velocity info)
    fn ghost_value(
        &self,
        u_interior: f64,
        x: f64,
        y: f64,
        normal: (f64, f64),
        time: f64,
        equation: &Advection2D,
    ) -> f64;
}

/// Periodic boundary condition (uses neighbor values).
#[derive(Clone, Copy, Debug, Default)]
pub struct PeriodicBC2D;

impl AdvectionBoundaryCondition2D for PeriodicBC2D {
    fn ghost_value(
        &self,
        u_interior: f64,
        _x: f64,
        _y: f64,
        _normal: (f64, f64),
        _time: f64,
        _equation: &Advection2D,
    ) -> f64 {
        // For periodic meshes, this should never be called
        // If called, just return interior (no-flux)
        u_interior
    }
}

/// Dirichlet boundary condition with a prescribed function.
pub struct DirichletBC2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// Function g(x, y, t) specifying the boundary value
    pub value_fn: F,
}

impl<F> DirichletBC2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    pub fn new(value_fn: F) -> Self {
        Self { value_fn }
    }
}

impl<F> AdvectionBoundaryCondition2D for DirichletBC2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn ghost_value(
        &self,
        u_interior: f64,
        x: f64,
        y: f64,
        normal: (f64, f64),
        time: f64,
        equation: &Advection2D,
    ) -> f64 {
        // Check if this is inflow or outflow
        let a_dot_n = equation.normal_wave_speed(normal);

        if a_dot_n < 0.0 {
            // Inflow: use prescribed value
            (self.value_fn)(x, y, time)
        } else {
            // Outflow: extrapolate (use interior value)
            u_interior
        }
    }
}

/// Constant Dirichlet boundary condition.
#[derive(Clone, Copy, Debug)]
pub struct ConstantBC2D {
    pub value: f64,
}

impl ConstantBC2D {
    pub fn new(value: f64) -> Self {
        Self { value }
    }

    pub fn zero() -> Self {
        Self { value: 0.0 }
    }
}

impl AdvectionBoundaryCondition2D for ConstantBC2D {
    fn ghost_value(
        &self,
        u_interior: f64,
        _x: f64,
        _y: f64,
        normal: (f64, f64),
        _time: f64,
        equation: &Advection2D,
    ) -> f64 {
        let a_dot_n = equation.normal_wave_speed(normal);

        if a_dot_n < 0.0 {
            self.value
        } else {
            u_interior
        }
    }
}

/// Compute the right-hand side for 2D advection.
///
/// Implements the DG weak form:
/// du/dt = -1/J * [Dr * (ar) + Ds * (as)] * u + 1/J * Σ_f LIFT_f * sJ_f * (F* - F-)
///
/// where ar = a_x * rx + a_y * ry and as = a_x * sx + a_y * sy.
pub fn compute_rhs_advection_2d<BC: AdvectionBoundaryCondition2D>(
    u: &DGSolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    equation: &Advection2D,
    bc: &BC,
    flux_type: AdvectionFluxType,
    time: f64,
) -> DGSolution2D {
    let n_nodes = ops.n_nodes;
    let n_face_nodes = ops.n_face_nodes;
    let mut rhs = DGSolution2D::new(mesh.n_elements, n_nodes);

    for k in 0..mesh.n_elements {
        let u_k = u.element(k);
        let j_inv = geom.det_j_inv[k];

        // Compute velocity components in reference space
        // ar = a · ∇r = a_x * rx + a_y * ry (transforms a to reference coords)
        let (a_x, a_y) = equation.velocity();
        let ar = a_x * geom.rx[k] + a_y * geom.ry[k];
        let as_ = a_x * geom.sx[k] + a_y * geom.sy[k];

        // 1. Volume term: -(a · ∇u) = -(ar * ∂u/∂r + as * ∂u/∂s)
        // No j_inv here because collocation mass matrix M = J * diag(w),
        // and M^{-1} * M cancels the Jacobian in the volume integral.
        let rhs_k = rhs.element_mut(k);

        for i in 0..n_nodes {
            let mut du_dr = 0.0;
            let mut du_ds = 0.0;
            for j in 0..n_nodes {
                du_dr += ops.dr[(i, j)] * u_k[j];
                du_ds += ops.ds[(i, j)] * u_k[j];
            }
            rhs_k[i] = -(ar * du_dr + as_ * du_ds);
        }

        // 2. Surface terms: 1/J * LIFT_f * sJ_f * (F* - F-)
        for face in 0..4 {
            let normal = geom.normals[k][face];
            let s_jac = geom.surface_j[k][face];

            // Get interior face values
            let face_nodes = &ops.face_nodes[face];

            // Get exterior values (neighbor or boundary)
            let (u_ext, _use_bc) = if let Some(neighbor) = mesh.neighbor(k, face) {
                // Interior face: get values from neighbor
                let u_neighbor = u.element(neighbor.element);
                let neighbor_face_nodes = &ops.face_nodes[neighbor.face];

                // Note: face nodes are ordered consistently, but neighbor's face
                // runs in opposite direction, so we need to reverse
                let mut ext = vec![0.0; n_face_nodes];
                for i in 0..n_face_nodes {
                    ext[i] = u_neighbor[neighbor_face_nodes[n_face_nodes - 1 - i]];
                }
                (ext, false)
            } else {
                // Boundary face: compute ghost values
                let mut ext = vec![0.0; n_face_nodes];
                for i in 0..n_face_nodes {
                    let node_idx = face_nodes[i];
                    let u_int = u_k[node_idx];
                    let (r, s) = (ops.nodes_r[node_idx], ops.nodes_s[node_idx]);
                    let (x, y) = mesh.reference_to_physical(k, r, s);
                    ext[i] = bc.ghost_value(u_int, x, y, normal, time, equation);
                }
                (ext, true)
            };

            // Compute numerical flux at face nodes
            let mut flux_jump = vec![0.0; n_face_nodes];
            for i in 0..n_face_nodes {
                let node_idx = face_nodes[i];
                let u_int = u_k[node_idx];
                let u_exterior = u_ext[i];

                // Numerical flux
                let f_star = match flux_type {
                    AdvectionFluxType::Upwind => equation.upwind_flux(u_int, u_exterior, normal),
                    AdvectionFluxType::LaxFriedrichs => {
                        equation.lax_friedrichs_flux(u_int, u_exterior, normal)
                    }
                };

                // Interior flux: F- · n = (a · n) * u_int
                let f_int = equation.normal_flux(u_int, normal);

                // Flux difference: (F^- - F*) for upwind dissipation
                // Note: This sign convention matches the 1D implementation
                // and ensures dissipation (energy decay) rather than growth
                flux_jump[i] = f_int - f_star;
            }

            // Apply LIFT: rhs += j_inv * LIFT_f * (sJ * flux_jump)
            // LIFT has shape (n_nodes, n_face_nodes)
            for i in 0..n_nodes {
                for fi in 0..n_face_nodes {
                    rhs_k[i] += j_inv * ops.lift[face][(i, fi)] * s_jac * flux_jump[fi];
                }
            }
        }
    }

    rhs
}

/// Compute the stable time step for 2D advection.
///
/// Uses CFL condition: dt ≤ CFL * h_min / (|a| * (2*p + 1))
pub fn compute_dt_advection_2d(
    mesh: &Mesh2D,
    geom: &GeometricFactors2D,
    equation: &Advection2D,
    order: usize,
    cfl: f64,
) -> f64 {
    let wave_speed = equation.max_wave_speed();

    if wave_speed < 1e-14 {
        return f64::INFINITY;
    }

    // Find minimum element size
    let mut h_min = f64::INFINITY;
    for k in 0..mesh.n_elements {
        // Use Jacobian to estimate element size
        // For uniform mesh: h ≈ sqrt(det_J) * 2
        let h_k = geom.det_j[k].sqrt() * 2.0;
        h_min = h_min.min(h_k);
    }

    // DG CFL: account for polynomial order
    // Factor (2p+1) comes from characteristic length of highest mode
    let dg_factor = 2.0 * order as f64 + 1.0;

    cfl * h_min / (wave_speed * dg_factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_setup(order: usize) -> (Mesh2D, DGOperators2D, GeometricFactors2D) {
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh);
        (mesh, ops, geom)
    }

    #[test]
    fn test_rhs_constant_solution() {
        let (mesh, ops, geom) = create_test_setup(2);
        let equation = Advection2D::new(1.0, 0.5);
        let bc = PeriodicBC2D;

        let mut u = DGSolution2D::new(mesh.n_elements, ops.n_nodes);
        u.fill(1.0); // Constant solution

        let rhs = compute_rhs_advection_2d(
            &u,
            &mesh,
            &ops,
            &geom,
            &equation,
            &bc,
            AdvectionFluxType::Upwind,
            0.0,
        );

        // RHS of constant should be zero
        for k in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                assert!(
                    rhs.element(k)[i].abs() < 1e-12,
                    "RHS of constant should be 0, got {} at elem {}, node {}",
                    rhs.element(k)[i],
                    k,
                    i
                );
            }
        }
    }

    #[test]
    fn test_volume_term_only() {
        // Test the volume term in isolation by using a smooth solution
        // For a smooth continuous solution, surface terms should be small
        use std::f64::consts::PI;

        // Use a fine mesh and high order
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 8, 8);
        let ops = DGOperators2D::new(4);
        let geom = GeometricFactors2D::compute(&mesh);

        let a_x = 1.0;
        let a_y = 0.0;
        let equation = Advection2D::new(a_x, a_y);
        let bc = PeriodicBC2D;

        // u = sin(2πx) -> du/dx = 2π cos(2πx)
        // RHS = -a_x * du/dx = -2π cos(2πx)
        let mut u = DGSolution2D::new(mesh.n_elements, ops.n_nodes);
        u.set_from_function(&mesh, &ops, |x, _y| (2.0 * PI * x).sin());

        let rhs = compute_rhs_advection_2d(
            &u,
            &mesh,
            &ops,
            &geom,
            &equation,
            &bc,
            AdvectionFluxType::Upwind,
            0.0,
        );

        // Check the middle element
        let k = 27; // 8x8 mesh, element (3,3)
        let mut max_error: f64 = 0.0;
        let mut max_expected: f64 = 0.0;
        let mut max_actual: f64 = 0.0;
        for i in 0..ops.n_nodes {
            let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
            let (x, _y) = mesh.reference_to_physical(k, r, s);
            let expected = -a_x * 2.0 * PI * (2.0 * PI * x).cos();
            let actual = rhs.element(k)[i];
            let error = (actual - expected).abs();
            max_error = max_error.max(error);
            max_expected = max_expected.max(expected.abs());
            max_actual = max_actual.max(actual.abs());
        }

        let relative_error = max_error / max_expected.max(1.0);
        assert!(
            relative_error < 0.5,
            "Relative RHS error should be small, got {} (max_error={}, max_expected={}, max_actual={})",
            relative_error,
            max_error,
            max_expected,
            max_actual
        );
    }

    #[test]
    fn test_single_step_stability() {
        // Test that a single time step doesn't blow up the solution
        use std::f64::consts::PI;

        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);

        let equation = Advection2D::new(1.0, 0.0);
        let bc = PeriodicBC2D;

        let mut u = DGSolution2D::new(mesh.n_elements, ops.n_nodes);
        u.set_from_function(&mesh, &ops, |x, _y| (2.0 * PI * x).sin());

        let initial_max = u.max_abs();

        // Compute RHS
        let rhs = compute_rhs_advection_2d(
            &u,
            &mesh,
            &ops,
            &geom,
            &equation,
            &bc,
            AdvectionFluxType::Upwind,
            0.0,
        );

        let rhs_max = rhs.max_abs();

        // Do one forward Euler step with small dt
        let dt = 0.001;
        u.axpy(dt, &rhs);

        let final_max = u.max_abs();

        // Solution should not have grown significantly
        println!(
            "Single step: initial_max={:.4e}, rhs_max={:.4e}, final_max={:.4e}",
            initial_max, rhs_max, final_max
        );

        // RHS should be O(1) since du/dx is O(2π)
        assert!(rhs_max < 100.0, "RHS should be O(2π) ≈ 6, got {}", rhs_max);

        // Solution should stay bounded after one step
        assert!(
            final_max < 2.0 * initial_max,
            "Solution grew too much: {} -> {}",
            initial_max,
            final_max
        );
    }

    #[test]
    fn test_multiple_steps_stability() {
        // Test stability over multiple time steps
        use crate::time::ssp_rk3_step_2d_timed;
        use std::f64::consts::PI;

        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);

        let equation = Advection2D::new(1.0, 0.0);
        let bc = PeriodicBC2D;

        let mut u = DGSolution2D::new(mesh.n_elements, ops.n_nodes);
        u.set_from_function(&mesh, &ops, |x, _y| (2.0 * PI * x).sin());

        let initial_max = u.max_abs();

        // Small CFL
        let dt = compute_dt_advection_2d(&mesh, &geom, &equation, ops.order, 0.1);
        println!("dt = {:.6e}", dt);

        // Run 100 steps
        let n_steps = 100;
        for step in 0..n_steps {
            let t = step as f64 * dt;
            ssp_rk3_step_2d_timed(
                &mut u,
                |u_, _time| {
                    compute_rhs_advection_2d(
                        u_,
                        &mesh,
                        &ops,
                        &geom,
                        &equation,
                        &bc,
                        AdvectionFluxType::Upwind,
                        0.0,
                    )
                },
                t,
                dt,
            );

            if step % 10 == 0 {
                println!("Step {}: max = {:.4e}", step, u.max_abs());
            }
        }

        let final_max = u.max_abs();
        println!("Final max: {:.4e}", final_max);

        // Solution should stay bounded
        assert!(
            final_max < 10.0 * initial_max,
            "Solution blew up: {} -> {}",
            initial_max,
            final_max
        );
    }

    #[test]
    fn test_dt_computation() {
        let (mesh, ops, geom) = create_test_setup(2);
        let equation = Advection2D::new(1.0, 1.0);

        let dt = compute_dt_advection_2d(&mesh, &geom, &equation, ops.order, 0.5);

        // dt should be positive and finite
        assert!(dt > 0.0);
        assert!(dt < f64::INFINITY);

        // Higher order should give smaller dt
        let (mesh2, ops2, geom2) = create_test_setup(4);
        let dt_high = compute_dt_advection_2d(&mesh2, &geom2, &equation, ops2.order, 0.5);
        assert!(
            dt_high < dt,
            "Higher order should have smaller dt: {} vs {}",
            dt_high,
            dt
        );
    }

    #[test]
    fn test_dt_zero_velocity() {
        let (mesh, _ops, geom) = create_test_setup(2);
        let equation = Advection2D::new(0.0, 0.0);

        let dt = compute_dt_advection_2d(&mesh, &geom, &equation, 2, 0.5);
        assert!(dt == f64::INFINITY);
    }

    #[test]
    fn test_flux_types_give_same_for_continuous() {
        let (mesh, ops, geom) = create_test_setup(2);
        let equation = Advection2D::new(1.0, 0.5);
        let bc = PeriodicBC2D;

        // Smooth solution
        let mut u = DGSolution2D::new(mesh.n_elements, ops.n_nodes);
        u.fill(2.5);

        let rhs_upwind = compute_rhs_advection_2d(
            &u,
            &mesh,
            &ops,
            &geom,
            &equation,
            &bc,
            AdvectionFluxType::Upwind,
            0.0,
        );

        let rhs_lf = compute_rhs_advection_2d(
            &u,
            &mesh,
            &ops,
            &geom,
            &equation,
            &bc,
            AdvectionFluxType::LaxFriedrichs,
            0.0,
        );

        // For continuous solution, both should give same result
        for k in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let diff = (rhs_upwind.element(k)[i] - rhs_lf.element(k)[i]).abs();
                assert!(
                    diff < 1e-12,
                    "Upwind and LF should match for continuous: diff {} at elem {}, node {}",
                    diff,
                    k,
                    i
                );
            }
        }
    }

    #[test]
    fn test_boundary_condition_inflow() {
        // Non-periodic mesh with Dirichlet BC
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);

        // Velocity pointing into domain from left
        let equation = Advection2D::new(1.0, 0.0);

        // BC: u = 1 at inflow
        let bc = ConstantBC2D::new(1.0);

        // Interior solution = 0
        let u = DGSolution2D::new(mesh.n_elements, ops.n_nodes);

        let rhs = compute_rhs_advection_2d(
            &u,
            &mesh,
            &ops,
            &geom,
            &equation,
            &bc,
            AdvectionFluxType::Upwind,
            0.0,
        );

        // Check that boundary effect is present (RHS is non-zero near boundary)
        let mut has_nonzero = false;
        for k in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                if rhs.element(k)[i].abs() > 1e-10 {
                    has_nonzero = true;
                }
            }
        }
        assert!(has_nonzero, "Inflow BC should create non-zero RHS");
    }
}
