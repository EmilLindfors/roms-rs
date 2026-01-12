//! SSP-RK3 time integration for 2D problems.
//!
//! Same algorithm as the 1D version, but works with DGSolution2D.

use crate::solver::DGSolution2D;

/// Perform one step of SSP-RK3 time integration for 2D problems.
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
pub fn ssp_rk3_step_2d<F>(u: &mut DGSolution2D, rhs_fn: F, dt: f64)
where
    F: Fn(&DGSolution2D) -> DGSolution2D,
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

/// Perform one step of SSP-RK3 with time-dependent RHS for 2D problems.
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
pub fn ssp_rk3_step_2d_timed<F>(u: &mut DGSolution2D, rhs_fn: F, t: f64, dt: f64)
where
    F: Fn(&DGSolution2D, f64) -> DGSolution2D,
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
    let t2 = t + 0.5 * dt;
    let l_u2 = rhs_fn(&u2, t2);
    u.scale(1.0 / 3.0);
    u.axpy(2.0 / 3.0, &u2);
    u.axpy(2.0 / 3.0 * dt, &l_u2);
}

/// Run a complete 2D advection simulation and return final solution.
///
/// # Arguments
/// * `u` - Initial solution (modified in place during simulation)
/// * `t_final` - Final simulation time
/// * `dt` - Time step
/// * `rhs_fn` - Function computing RHS at given time
///
/// # Returns
/// The number of time steps taken.
pub fn run_advection_2d<F>(u: &mut DGSolution2D, t_final: f64, dt: f64, rhs_fn: F) -> usize
where
    F: Fn(&DGSolution2D, f64) -> DGSolution2D,
{
    let mut t = 0.0;
    let mut n_steps = 0;

    while t < t_final {
        let dt_actual = dt.min(t_final - t);
        ssp_rk3_step_2d_timed(u, &rhs_fn, t, dt_actual);
        t += dt_actual;
        n_steps += 1;
    }

    n_steps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssp_rk3_2d_linear_growth() {
        // Test with L(u) = c * u (exponential growth)
        // Exact: u(t) = u_0 * exp(c * t)
        let n_elem = 4;
        let n_nodes = 9; // Order 2
        let mut u = DGSolution2D::new(n_elem, n_nodes);

        // Initial condition
        u.fill(1.0);

        let c = 1.0;
        let dt = 0.01;
        let n_steps = 10;

        for _ in 0..n_steps {
            ssp_rk3_step_2d(
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

    #[test]
    fn test_ssp_rk3_2d_timed_time_dependency() {
        // Test that time is correctly passed to stages
        // Use RHS = cos(t), so u(t) = u_0 + sin(t)
        let n_elem = 1;
        let n_nodes = 4;
        let mut u = DGSolution2D::new(n_elem, n_nodes);
        u.fill(0.0);

        let dt = 0.1;
        let n_steps = 31; // Integrate from 0 to ~π

        for i in 0..n_steps {
            let t = i as f64 * dt;
            ssp_rk3_step_2d_timed(
                &mut u,
                |_, time| {
                    let mut rhs = DGSolution2D::new(n_elem, n_nodes);
                    rhs.fill(time.cos());
                    rhs
                },
                t,
                dt,
            );
        }

        let t_final = n_steps as f64 * dt;
        let expected = t_final.sin();

        for &v in &u.data {
            let error = (v - expected).abs();
            // RK3 error for this ODE
            assert!(
                error < 1e-3,
                "Time-dependent RHS: expected {}, got {} (error {})",
                expected,
                v,
                error
            );
        }
    }

    #[test]
    fn test_run_advection_2d_basic() {
        let mut u = DGSolution2D::new(1, 4);
        u.fill(1.0);

        let n_steps = run_advection_2d(&mut u, 0.1, 0.01, |u_, _t| {
            // Zero RHS: solution should be unchanged
            let mut rhs = u_.clone();
            rhs.fill(0.0);
            rhs
        });

        // Allow for floating point in step counting (10 or 11 steps)
        assert!(
            n_steps >= 10 && n_steps <= 11,
            "Expected ~10 steps, got {}",
            n_steps
        );

        // Solution should be unchanged
        for &v in &u.data {
            assert!((v - 1.0).abs() < 1e-14, "Zero RHS should preserve solution");
        }
    }
}
