//! Strong Stability Preserving Runge-Kutta time integration.
//!
//! SSP-RK3 (Shu-Osher form) is optimal for hyperbolic conservation laws.
//! It maintains the TVD property of the spatial discretization.

use crate::mesh::Mesh1D;
use crate::solver::DGSolution1D;

/// Perform one step of SSP-RK3 time integration.
///
/// The Shu-Osher form:
/// u1 = u + dt * L(u)
/// u2 = 3/4 * u + 1/4 * u1 + 1/4 * dt * L(u1)
/// u_new = 1/3 * u + 2/3 * u2 + 2/3 * dt * L(u2)
///
/// # Arguments
/// * `u` - Solution to update (modified in place)
/// * `rhs_fn` - Function that computes the RHS given the solution
/// * `dt` - Time step
///
/// Note: For time-dependent boundary conditions, use `ssp_rk3_step_timed` instead.
pub fn ssp_rk3_step<F>(u: &mut DGSolution1D, rhs_fn: F, dt: f64)
where
    F: Fn(&DGSolution1D) -> DGSolution1D,
{
    // Stage 1: u1 = u + dt * L(u)
    let l_u = rhs_fn(u);
    let mut u1 = u.clone();
    u1.axpy(dt, &l_u);

    // Stage 2: u2 = 3/4 * u + 1/4 * u1 + 1/4 * dt * L(u1)
    let l_u1 = rhs_fn(&u1);
    let mut u2 = u.clone();
    u2.scale(0.75);
    u2.axpy(0.25, &u1);
    u2.axpy(0.25 * dt, &l_u1);

    // Stage 3: u_new = 1/3 * u + 2/3 * u2 + 2/3 * dt * L(u2)
    let l_u2 = rhs_fn(&u2);
    u.scale(1.0 / 3.0);
    u.axpy(2.0 / 3.0, &u2);
    u.axpy(2.0 / 3.0 * dt, &l_u2);
}

/// Perform one step of SSP-RK3 with time-dependent RHS.
///
/// This version passes the correct time to each RK stage, which is essential
/// for achieving high-order accuracy with time-dependent boundary conditions.
///
/// Stage times for SSP-RK3:
/// - Stage 1: t
/// - Stage 2: t + dt
/// - Stage 3: t + dt/2
///
/// # Arguments
/// * `u` - Solution to update (modified in place)
/// * `rhs_fn` - Function that computes RHS given solution and time
/// * `t` - Current time
/// * `dt` - Time step
pub fn ssp_rk3_step_timed<F>(u: &mut DGSolution1D, rhs_fn: F, t: f64, dt: f64)
where
    F: Fn(&DGSolution1D, f64) -> DGSolution1D,
{
    // Stage 1: u1 = u + dt * L(u, t)
    let l_u = rhs_fn(u, t);
    let mut u1 = u.clone();
    u1.axpy(dt, &l_u);

    // Stage 2: u2 = 3/4 * u + 1/4 * u1 + 1/4 * dt * L(u1, t + dt)
    let t1 = t + dt;
    let l_u1 = rhs_fn(&u1, t1);
    let mut u2 = u.clone();
    u2.scale(0.75);
    u2.axpy(0.25, &u1);
    u2.axpy(0.25 * dt, &l_u1);

    // Stage 3: u_new = 1/3 * u + 2/3 * u2 + 2/3 * dt * L(u2, t + dt/2)
    // Note: The effective time for stage 3 is t + dt/2 because
    // u2 = 3/4*u(t) + 1/4*u1(t+dt) which is centered at t + dt/4,
    // but after applying L(u2), we get solution at t + dt.
    // The standard SSP-RK3 stage times are: t, t+dt, t+dt/2
    let t2 = t + 0.5 * dt;
    let l_u2 = rhs_fn(&u2, t2);
    u.scale(1.0 / 3.0);
    u.axpy(2.0 / 3.0, &u2);
    u.axpy(2.0 / 3.0 * dt, &l_u2);
}

/// Compute the CFL-limited time step for advection.
///
/// For DG with polynomial order N:
/// dt <= CFL * h_min / (|a| * (2*N + 1))
///
/// The factor (2*N + 1) accounts for the eigenvalue scaling of DG.
///
/// # Arguments
/// * `mesh` - The mesh (for h_min)
/// * `a` - Advection velocity
/// * `order` - Polynomial order
/// * `cfl` - CFL number (typically 0.1 - 0.5)
pub fn compute_dt(mesh: &Mesh1D, a: f64, order: usize, cfl: f64) -> f64 {
    let h_min = mesh.h_min();
    let a_abs = a.abs();

    if a_abs < 1e-14 {
        // No advection, any timestep is fine
        return f64::INFINITY;
    }

    cfl * h_min / (a_abs * (2 * order + 1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dt_scaling() {
        let mesh = Mesh1D::uniform(0.0, 1.0, 10);
        let a = 1.0;
        let cfl = 0.5;

        // dt should scale inversely with (2*N + 1)
        let dt_p1 = compute_dt(&mesh, a, 1, cfl);
        let dt_p3 = compute_dt(&mesh, a, 3, cfl);

        let expected_ratio = 3.0 / 7.0; // (2*1+1) / (2*3+1)
        let actual_ratio = dt_p3 / dt_p1;

        assert!(
            (actual_ratio - expected_ratio).abs() < 1e-14,
            "dt scaling: expected {}, got {}",
            expected_ratio,
            actual_ratio
        );
    }

    #[test]
    fn test_dt_velocity_scaling() {
        let mesh = Mesh1D::uniform(0.0, 1.0, 10);
        let order = 3;
        let cfl = 0.5;

        // dt should scale inversely with |a|
        let dt1 = compute_dt(&mesh, 1.0, order, cfl);
        let dt2 = compute_dt(&mesh, 2.0, order, cfl);

        assert!(
            (dt2 / dt1 - 0.5).abs() < 1e-14,
            "dt should halve when velocity doubles"
        );

        // Negative velocity should give same dt
        let dt_neg = compute_dt(&mesh, -1.0, order, cfl);
        assert!(
            (dt_neg - dt1).abs() < 1e-14,
            "dt should depend on |a|, not a"
        );
    }

    #[test]
    fn test_ssp_rk3_linear_rhs() {
        // Test with L(u) = c * u (exponential growth/decay)
        // Exact: u(t) = u_0 * exp(c * t)
        let n_elem = 2;
        let n_nodes = 3;
        let mut u = DGSolution1D::new(n_elem, n_nodes);

        // Initial condition
        for v in &mut u.data {
            *v = 1.0;
        }

        let c = 1.0;
        let dt = 0.01;
        let n_steps = 10;

        for _ in 0..n_steps {
            ssp_rk3_step(
                &mut u,
                |u_| {
                    let mut rhs = u_.clone();
                    rhs.scale(c);
                    rhs
                },
                dt,
            );
        }

        let t = dt * n_steps as f64;
        let expected = (c * t).exp();

        for &v in &u.data {
            // RK3 is 3rd order accurate, so error should be O(dt^3)
            let error = (v - expected).abs();
            assert!(
                error < 1e-4,
                "Expected {}, got {} (error {})",
                expected,
                v,
                error
            );
        }
    }
}
