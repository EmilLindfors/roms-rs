//! SSP-RK3 time integration for shallow water equations.
//!
//! Extends the basic SSP-RK3 to support:
//! - SystemSolution (SWESolution)
//! - Optional limiters applied after each stage
//! - Proper stage times for time-dependent BCs

use crate::mesh::Mesh1D;
use crate::operators::DGOperators1D;
use crate::solver::{SWESolution, TVBParameter, apply_swe_limiters};

/// Configuration for SWE time integration.
#[derive(Clone, Debug)]
pub struct SWETimeConfig {
    /// CFL number (typically 0.1-0.5 for DG)
    pub cfl: f64,
    /// Whether to apply limiters after each stage
    pub apply_limiters: bool,
    /// TVB parameter for slope limiting
    pub tvb: TVBParameter,
    /// Minimum depth for positivity
    pub h_min: f64,
}

impl SWETimeConfig {
    /// Create a new configuration.
    pub fn new(cfl: f64, tvb_m: f64, dx: f64, h_min: f64) -> Self {
        Self {
            cfl,
            apply_limiters: true,
            tvb: TVBParameter::new(tvb_m, dx),
            h_min,
        }
    }

    /// Create with limiters disabled.
    pub fn without_limiters(cfl: f64) -> Self {
        Self {
            cfl,
            apply_limiters: false,
            tvb: TVBParameter::new(0.0, 1.0),
            h_min: 1e-10,
        }
    }

    /// Set TVB parameter.
    pub fn with_tvb(mut self, m: f64, dx: f64) -> Self {
        self.tvb = TVBParameter::new(m, dx);
        self
    }
}

/// Perform one step of SSP-RK3 for SWE.
///
/// # Arguments
/// * `q` - Solution to update (modified in place)
/// * `rhs_fn` - Function that computes RHS given solution
/// * `dt` - Time step
/// * `mesh` - Mesh (for limiters)
/// * `ops` - DG operators (for limiters)
/// * `config` - Time integration configuration
pub fn ssp_rk3_swe_step<F>(
    q: &mut SWESolution,
    rhs_fn: F,
    dt: f64,
    mesh: &Mesh1D,
    ops: &DGOperators1D,
    config: &SWETimeConfig,
) where
    F: Fn(&SWESolution) -> SWESolution,
{
    // Stage 1: q1 = q + dt * L(q)
    let l_q = rhs_fn(q);
    let mut q1 = q.clone();
    q1.axpy(dt, &l_q);

    if config.apply_limiters {
        apply_swe_limiters(&mut q1, mesh, ops, &config.tvb, config.h_min);
    }

    // Stage 2: q2 = 3/4 * q + 1/4 * q1 + 1/4 * dt * L(q1)
    let l_q1 = rhs_fn(&q1);
    let mut q2 = q.clone();
    q2.scale(0.75);
    q2.axpy(0.25, &q1);
    q2.axpy(0.25 * dt, &l_q1);

    if config.apply_limiters {
        apply_swe_limiters(&mut q2, mesh, ops, &config.tvb, config.h_min);
    }

    // Stage 3: q_new = 1/3 * q + 2/3 * q2 + 2/3 * dt * L(q2)
    let l_q2 = rhs_fn(&q2);
    q.scale(1.0 / 3.0);
    q.axpy(2.0 / 3.0, &q2);
    q.axpy(2.0 / 3.0 * dt, &l_q2);

    if config.apply_limiters {
        apply_swe_limiters(q, mesh, ops, &config.tvb, config.h_min);
    }
}

/// Perform one step of SSP-RK3 for SWE with time-dependent RHS.
///
/// Stage times for SSP-RK3:
/// - Stage 1: t
/// - Stage 2: t + dt
/// - Stage 3: t + dt/2
///
/// # Arguments
/// * `q` - Solution to update (modified in place)
/// * `rhs_fn` - Function that computes RHS given solution and time
/// * `t` - Current time
/// * `dt` - Time step
/// * `mesh` - Mesh (for limiters)
/// * `ops` - DG operators (for limiters)
/// * `config` - Time integration configuration
pub fn ssp_rk3_swe_step_timed<F>(
    q: &mut SWESolution,
    rhs_fn: F,
    t: f64,
    dt: f64,
    mesh: &Mesh1D,
    ops: &DGOperators1D,
    config: &SWETimeConfig,
) where
    F: Fn(&SWESolution, f64) -> SWESolution,
{
    // Stage 1: q1 = q + dt * L(q, t)
    let l_q = rhs_fn(q, t);
    let mut q1 = q.clone();
    q1.axpy(dt, &l_q);

    if config.apply_limiters {
        apply_swe_limiters(&mut q1, mesh, ops, &config.tvb, config.h_min);
    }

    // Stage 2: q2 = 3/4 * q + 1/4 * q1 + 1/4 * dt * L(q1, t + dt)
    let t1 = t + dt;
    let l_q1 = rhs_fn(&q1, t1);
    let mut q2 = q.clone();
    q2.scale(0.75);
    q2.axpy(0.25, &q1);
    q2.axpy(0.25 * dt, &l_q1);

    if config.apply_limiters {
        apply_swe_limiters(&mut q2, mesh, ops, &config.tvb, config.h_min);
    }

    // Stage 3: q_new = 1/3 * q + 2/3 * q2 + 2/3 * dt * L(q2, t + dt/2)
    let t2 = t + 0.5 * dt;
    let l_q2 = rhs_fn(&q2, t2);
    q.scale(1.0 / 3.0);
    q.axpy(2.0 / 3.0, &q2);
    q.axpy(2.0 / 3.0 * dt, &l_q2);

    if config.apply_limiters {
        apply_swe_limiters(q, mesh, ops, &config.tvb, config.h_min);
    }
}

/// Run a complete SWE simulation.
///
/// # Arguments
/// * `q` - Initial condition (modified in place)
/// * `t_end` - End time
/// * `rhs_fn` - RHS function (solution, time) -> RHS
/// * `dt_fn` - Function to compute dt given current solution
/// * `mesh` - Mesh
/// * `ops` - DG operators
/// * `config` - Time integration config
/// * `callback` - Optional callback(t, q) called after each step
///
/// # Returns
/// Final time reached and number of steps taken.
#[allow(clippy::too_many_arguments)]
pub fn run_swe_simulation<F, D, C>(
    q: &mut SWESolution,
    t_end: f64,
    rhs_fn: F,
    dt_fn: D,
    mesh: &Mesh1D,
    ops: &DGOperators1D,
    config: &SWETimeConfig,
    mut callback: Option<C>,
) -> (f64, usize)
where
    F: Fn(&SWESolution, f64) -> SWESolution,
    D: Fn(&SWESolution) -> f64,
    C: FnMut(f64, &SWESolution),
{
    let mut t = 0.0;
    let mut n_steps = 0;

    while t < t_end {
        // Compute adaptive time step
        let mut dt = dt_fn(q);

        // Don't overshoot end time
        if t + dt > t_end {
            dt = t_end - t;
        }

        // Take one step
        ssp_rk3_swe_step_timed(q, &rhs_fn, t, dt, mesh, ops, config);

        t += dt;
        n_steps += 1;

        // Call callback if provided
        if let Some(ref mut cb) = callback {
            cb(t, q);
        }
    }

    (t, n_steps)
}

/// Compute total mass (integral of h) in the solution.
pub fn total_mass(q: &SWESolution, mesh: &Mesh1D, ops: &DGOperators1D) -> f64 {
    let mut mass = 0.0;

    for k in 0..q.n_elements {
        let jac = mesh.jacobian(k);

        for i in 0..ops.n_nodes {
            let h = q.get(k, i)[0];
            let w = ops.weights[i];
            mass += w * h * jac;
        }
    }

    mass
}

/// Compute total momentum (integral of hu) in the solution.
pub fn total_momentum(q: &SWESolution, mesh: &Mesh1D, ops: &DGOperators1D) -> f64 {
    let mut momentum = 0.0;

    for k in 0..q.n_elements {
        let jac = mesh.jacobian(k);

        for i in 0..ops.n_nodes {
            let hu = q.get(k, i)[1];
            let w = ops.weights[i];
            momentum += w * hu * jac;
        }
    }

    momentum
}

/// Compute total energy (integral of h*u²/2 + g*h²/2) in the solution.
pub fn total_energy(q: &SWESolution, mesh: &Mesh1D, ops: &DGOperators1D, g: f64) -> f64 {
    let h_min = 1e-10;
    let mut energy = 0.0;

    for k in 0..q.n_elements {
        let jac = mesh.jacobian(k);

        for i in 0..ops.n_nodes {
            let [h, hu] = q.get(k, i);
            let w = ops.weights[i];

            if h > h_min {
                let u = hu / h;
                // Kinetic + potential energy
                let e = 0.5 * h * u * u + 0.5 * g * h * h;
                energy += w * e * jac;
            }
        }
    }

    energy
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Mesh1D, DGOperators1D) {
        let mesh = Mesh1D::uniform(0.0, 1.0, 10);
        let ops = DGOperators1D::new(3);
        (mesh, ops)
    }

    #[test]
    fn test_swe_time_config() {
        let config = SWETimeConfig::new(0.5, 10.0, 0.1, 1e-6);
        assert!((config.cfl - 0.5).abs() < 1e-14);
        assert!(config.apply_limiters);
        assert!((config.tvb.m - 10.0).abs() < 1e-14);
    }

    #[test]
    fn test_ssp_rk3_swe_still_water() {
        let (mesh, ops) = setup();
        let config = SWETimeConfig::without_limiters(0.5);

        // Still water initial condition
        let mut q = SWESolution::new(10, 4);
        for k in 0..10 {
            for i in 0..4 {
                q.set(k, i, [2.0, 0.0]); // h=2, hu=0
            }
        }

        let initial_mass = total_mass(&q, &mesh, &ops);

        // RHS = 0 for still water
        let rhs_fn = |_q: &SWESolution| SWESolution::zeros(10, 4);

        ssp_rk3_swe_step(&mut q, rhs_fn, 0.01, &mesh, &ops, &config);

        // Should remain unchanged
        for k in 0..10 {
            for i in 0..4 {
                let [h, hu] = q.get(k, i);
                assert!((h - 2.0).abs() < 1e-12);
                assert!(hu.abs() < 1e-12);
            }
        }

        // Mass should be conserved
        let final_mass = total_mass(&q, &mesh, &ops);
        assert!((initial_mass - final_mass).abs() < 1e-12);
    }

    #[test]
    fn test_ssp_rk3_swe_with_limiters() {
        let (mesh, ops) = setup();
        let dx = 0.1;
        let config = SWETimeConfig::new(0.5, 10.0, dx, 1e-6);

        // Create solution with some oscillations
        let mut q = SWESolution::new(10, 4);
        for k in 0..10 {
            for i in 0..4 {
                let h = 1.0 + 0.5 * ops.nodes[i]; // Varies within element
                q.set(k, i, [h, 0.0]);
            }
        }

        // RHS = 0 for this test
        let rhs_fn = |_q: &SWESolution| SWESolution::zeros(10, 4);

        ssp_rk3_swe_step(&mut q, rhs_fn, 0.01, &mesh, &ops, &config);

        // All depths should be positive
        for k in 0..10 {
            for i in 0..4 {
                let h = q.get(k, i)[0];
                assert!(h >= 0.0, "Depth should be non-negative");
            }
        }
    }

    #[test]
    fn test_total_mass_conservation() {
        let (mesh, ops) = setup();

        // Create varying solution
        let mut q = SWESolution::new(10, 4);
        for k in 0..10 {
            for i in 0..4 {
                let x = mesh.reference_to_physical(k, ops.nodes[i]);
                let h = 1.0 + 0.3 * (2.0 * std::f64::consts::PI * x).sin();
                q.set(k, i, [h, h * 0.5]);
            }
        }

        let mass = total_mass(&q, &mesh, &ops);

        // For domain [0,1], average h ≈ 1.0, so mass ≈ 1.0
        assert!(
            (mass - 1.0).abs() < 0.1,
            "Expected mass near 1.0, got {}",
            mass
        );
    }

    #[test]
    fn test_total_energy() {
        let (mesh, ops) = setup();
        let g = 10.0;

        // Still water: only potential energy
        let mut q = SWESolution::new(10, 4);
        for k in 0..10 {
            for i in 0..4 {
                q.set(k, i, [2.0, 0.0]); // h=2, u=0
            }
        }

        let energy = total_energy(&q, &mesh, &ops, g);

        // E = 0.5 * g * h² per unit length
        // Total = 0.5 * 10 * 4 * 1 = 20
        assert!(
            (energy - 20.0).abs() < 0.1,
            "Expected energy ~20, got {}",
            energy
        );
    }

    #[test]
    fn test_run_simulation() {
        let (mesh, ops) = setup();
        let config = SWETimeConfig::without_limiters(0.5);

        // Still water
        let mut q = SWESolution::new(10, 4);
        for k in 0..10 {
            for i in 0..4 {
                q.set(k, i, [2.0, 0.0]);
            }
        }

        let rhs_fn = |_q: &SWESolution, _t: f64| SWESolution::zeros(10, 4);
        let dt_fn = |_q: &SWESolution| 0.01; // Fixed dt for test

        let (t_final, n_steps) = run_swe_simulation(
            &mut q,
            0.1,
            rhs_fn,
            dt_fn,
            &mesh,
            &ops,
            &config,
            None::<fn(f64, &SWESolution)>,
        );

        assert!((t_final - 0.1).abs() < 1e-10);
        // Allow 10 or 11 steps due to floating point
        assert!(
            n_steps >= 10 && n_steps <= 11,
            "Expected 10-11 steps, got {}",
            n_steps
        );

        // Solution should be unchanged
        for k in 0..10 {
            for i in 0..4 {
                let [h, hu] = q.get(k, i);
                assert!((h - 2.0).abs() < 1e-12);
                assert!(hu.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_timed_step_stage_times() {
        let (mesh, ops) = setup();
        let config = SWETimeConfig::without_limiters(0.5);

        let mut q = SWESolution::new(10, 4);
        for k in 0..10 {
            for i in 0..4 {
                q.set(k, i, [1.0, 0.0]);
            }
        }

        // Track times passed to RHS
        use std::cell::RefCell;
        let times: RefCell<Vec<f64>> = RefCell::new(Vec::new());

        let rhs_fn = |_q: &SWESolution, t: f64| {
            times.borrow_mut().push(t);
            SWESolution::zeros(10, 4)
        };

        let t0 = 1.0;
        let dt = 0.1;
        ssp_rk3_swe_step_timed(&mut q, rhs_fn, t0, dt, &mesh, &ops, &config);

        let recorded = times.borrow();
        assert_eq!(recorded.len(), 3, "Should have 3 RHS evaluations");

        // Stage times: t, t+dt, t+dt/2
        assert!((recorded[0] - 1.0).abs() < 1e-14, "Stage 1 time");
        assert!((recorded[1] - 1.1).abs() < 1e-14, "Stage 2 time");
        assert!((recorded[2] - 1.05).abs() < 1e-14, "Stage 3 time");
    }
}
