//! GPU SSP-RK3 time integration using Burn.
//!
//! Provides Strong Stability Preserving Runge-Kutta 3rd order time stepping
//! for the 2D SWE on GPU. The entire RK3 update stays on GPU, minimizing
//! data transfers.

#[cfg(feature = "burn")]
use burn::prelude::*;

#[cfg(feature = "burn")]
use crate::solver::burn::{
    BurnConnectivity, BurnOperators2D, BurnSWESolution2D,
    rhs::{BurnGeometricFactors2D, BurnRhsConfig, compute_rhs_swe_2d_burn},
};

/// Perform one SSP-RK3 time step on GPU.
///
/// The SSP-RK3 scheme is:
/// ```text
/// u^(1) = u^n + dt * L(u^n)
/// u^(2) = 3/4 * u^n + 1/4 * (u^(1) + dt * L(u^(1)))
/// u^(n+1) = 1/3 * u^n + 2/3 * (u^(2) + dt * L(u^(2)))
/// ```
///
/// All stages are computed on GPU without CPU round-trips.
///
/// # Arguments
/// * `state` - Current solution state (modified in place)
/// * `dt` - Time step size
/// * `ops` - GPU operators
/// * `geom` - GPU geometric factors
/// * `connectivity` - Pre-computed connectivity
/// * `config` - Physical parameters
///
/// # Returns
/// The time at the end of the step (t + dt)
#[cfg(feature = "burn")]
pub fn ssp_rk3_step_burn<B: Backend>(
    state: &mut BurnSWESolution2D<B>,
    dt: f64,
    ops: &BurnOperators2D<B>,
    geom: &BurnGeometricFactors2D<B>,
    connectivity: &BurnConnectivity<B>,
    config: &BurnRhsConfig,
) where
    B::FloatElem: From<f64>,
    B::IntElem: From<i64>,
    f64: From<B::FloatElem>,
    i64: From<B::IntElem>,
{
    // Stage 1: u1 = u + dt * L(u)
    let rhs0 = compute_rhs_swe_2d_burn(state, ops, geom, connectivity, config);
    let u0 = state.clone_solution();

    state.h = state.h.clone().add(rhs0.h.mul_scalar(dt));
    state.hu = state.hu.clone().add(rhs0.hu.mul_scalar(dt));
    state.hv = state.hv.clone().add(rhs0.hv.mul_scalar(dt));

    // Stage 2: u2 = 3/4 * u0 + 1/4 * (u1 + dt * L(u1))
    let rhs1 = compute_rhs_swe_2d_burn(state, ops, geom, connectivity, config);

    // First compute u1 + dt * L(u1)
    state.h = state.h.clone().add(rhs1.h.mul_scalar(dt));
    state.hu = state.hu.clone().add(rhs1.hu.mul_scalar(dt));
    state.hv = state.hv.clone().add(rhs1.hv.mul_scalar(dt));

    // Then blend: u2 = 3/4 * u0 + 1/4 * (u1 + dt*L(u1))
    state.h = u0.h.clone().mul_scalar(0.75).add(state.h.clone().mul_scalar(0.25));
    state.hu = u0.hu.clone().mul_scalar(0.75).add(state.hu.clone().mul_scalar(0.25));
    state.hv = u0.hv.clone().mul_scalar(0.75).add(state.hv.clone().mul_scalar(0.25));

    // Stage 3: u_new = 1/3 * u0 + 2/3 * (u2 + dt * L(u2))
    let rhs2 = compute_rhs_swe_2d_burn(state, ops, geom, connectivity, config);

    // First compute u2 + dt * L(u2)
    state.h = state.h.clone().add(rhs2.h.mul_scalar(dt));
    state.hu = state.hu.clone().add(rhs2.hu.mul_scalar(dt));
    state.hv = state.hv.clone().add(rhs2.hv.mul_scalar(dt));

    // Then blend: u_new = 1/3 * u0 + 2/3 * (u2 + dt*L(u2))
    let one_third = 1.0 / 3.0;
    let two_thirds = 2.0 / 3.0;
    state.h = u0.h.clone().mul_scalar(one_third).add(state.h.clone().mul_scalar(two_thirds));
    state.hu = u0.hu.clone().mul_scalar(one_third).add(state.hu.clone().mul_scalar(two_thirds));
    state.hv = u0.hv.mul_scalar(one_third).add(state.hv.clone().mul_scalar(two_thirds));
}

/// Compute CFL-stable time step on GPU.
///
/// The CFL condition for shallow water equations is:
/// dt <= CFL * min_k(dx_k / (|u_k| + c_k))
///
/// where c = sqrt(g*h) is the wave celerity.
///
/// # Arguments
/// * `solution` - Current state on GPU
/// * `geom` - GPU geometric factors
/// * `g` - Gravitational acceleration
/// * `h_min` - Minimum depth for dry cell protection
/// * `cfl` - CFL number (typically 0.5 for RK3)
///
/// # Returns
/// Maximum stable time step
#[cfg(feature = "burn")]
pub fn compute_dt_burn<B: Backend>(
    solution: &BurnSWESolution2D<B>,
    geom: &BurnGeometricFactors2D<B>,
    g: f64,
    h_min: f64,
    cfl: f64,
) -> f64
where
    B::FloatElem: From<f64>,
    f64: From<B::FloatElem>,
{
    // Compute wave speed at each node: |u| + c
    let h_reg = solution.h.clone().clamp_min(h_min);
    let h_inv = h_reg.clone().recip();

    let u = solution.hu.clone().mul(h_inv.clone());
    let v = solution.hv.clone().mul(h_inv);

    // Velocity magnitude
    let u_mag = u.clone().powf_scalar(2.0).add(v.powf_scalar(2.0)).sqrt();

    // Wave celerity
    let c = h_reg.mul_scalar(g).sqrt();

    // Maximum wave speed
    let wave_speed = u_mag.add(c);
    let max_speed_tensor = wave_speed.max();
    let max_speed = f64::from(max_speed_tensor.to_data().to_vec::<B::FloatElem>().unwrap()[0]);

    // Minimum element size (approximate using Jacobian)
    let min_j_tensor = geom.det_j.clone().min();
    let min_j = f64::from(min_j_tensor.to_data().to_vec::<B::FloatElem>().unwrap()[0]);

    // For quadrilateral elements, dx ~ sqrt(J)
    let dx_min = min_j.sqrt();

    // CFL condition
    if max_speed > 1e-10 {
        cfl * dx_min / max_speed
    } else {
        cfl * dx_min / 1e-10 // Avoid division by zero
    }
}

/// GPU time configuration.
#[cfg(feature = "burn")]
#[derive(Clone, Debug)]
pub struct BurnTimeConfig {
    /// Final simulation time
    pub t_final: f64,
    /// CFL number (typically 0.3-0.5 for RK3)
    pub cfl: f64,
    /// Maximum allowed time step
    pub dt_max: f64,
    /// Output interval (None = no intermediate output)
    pub output_interval: Option<f64>,
}

#[cfg(feature = "burn")]
impl Default for BurnTimeConfig {
    fn default() -> Self {
        Self {
            t_final: 1.0,
            cfl: 0.5,
            dt_max: 1.0,
            output_interval: None,
        }
    }
}

/// Run a GPU simulation using SSP-RK3.
///
/// This function runs the complete time evolution on GPU, only
/// transferring data back to CPU at output intervals.
///
/// # Arguments
/// * `state` - Initial condition (modified in place)
/// * `ops` - GPU operators
/// * `geom` - GPU geometric factors
/// * `connectivity` - Pre-computed connectivity
/// * `rhs_config` - Physical parameters
/// * `time_config` - Time stepping parameters
///
/// # Returns
/// Final time and number of steps taken
#[cfg(feature = "burn")]
pub fn run_swe_2d_burn<B: Backend>(
    state: &mut BurnSWESolution2D<B>,
    ops: &BurnOperators2D<B>,
    geom: &BurnGeometricFactors2D<B>,
    connectivity: &BurnConnectivity<B>,
    rhs_config: &BurnRhsConfig,
    time_config: &BurnTimeConfig,
) -> (f64, usize)
where
    B::FloatElem: From<f64>,
    B::IntElem: From<i64>,
    f64: From<B::FloatElem>,
    i64: From<B::IntElem>,
{
    let mut t = 0.0;
    let mut n_steps = 0;

    while t < time_config.t_final {
        // Compute adaptive time step
        let dt_cfl = compute_dt_burn(state, geom, rhs_config.g, rhs_config.h_min, time_config.cfl);
        let dt = dt_cfl.min(time_config.dt_max).min(time_config.t_final - t);

        // Take one RK3 step
        ssp_rk3_step_burn(state, dt, ops, geom, connectivity, rhs_config);

        t += dt;
        n_steps += 1;

        // Check for NaN
        if !state.is_valid() {
            eprintln!("Warning: NaN detected at t={}, step {}", t, n_steps);
            break;
        }
    }

    (t, n_steps)
}

#[cfg(test)]
#[cfg(all(test, feature = "burn-ndarray"))]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use crate::mesh::Mesh2DBuilder;
    use crate::operators::{DGOperators2D, GeometricFactors2D};
    use crate::solver::burn::BurnConnectivity;
    use crate::types::ElementIndex;

    #[test]
    fn test_compute_dt_burn() {
        let mesh = Mesh2DBuilder::unit_square()
            .with_resolution(2, 2)
            .build();

        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let device = burn_ndarray::NdArrayDevice::Cpu;

        let burn_geom = BurnGeometricFactors2D::<NdArray<f64>>::from_cpu(&geom, &device);

        // Create a simple state
        let mut cpu_sol = crate::solver::state::SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                cpu_sol.set_state(k, i, crate::solver::state::SWEState2D::new(1.0, 0.0, 0.0));
            }
        }

        let burn_sol = BurnSWESolution2D::<NdArray<f64>>::from_cpu(&cpu_sol, &device);

        let dt = compute_dt_burn(&burn_sol, &burn_geom, 9.81, 1e-4, 0.5);

        // Should get a positive, finite time step
        assert!(dt > 0.0);
        assert!(dt.is_finite());
        assert!(dt < 1.0); // Should be reasonably small for this mesh
    }

    #[test]
    fn test_ssp_rk3_step_lake_at_rest() {
        let mesh = Mesh2DBuilder::unit_square()
            .with_resolution(2, 2)
            .build();

        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let device = burn_ndarray::NdArrayDevice::Cpu;

        let burn_ops = BurnOperators2D::<NdArray<f64>>::from_cpu(&ops, &device);
        let burn_geom = BurnGeometricFactors2D::<NdArray<f64>>::from_cpu(&geom, &device);
        let connectivity = BurnConnectivity::<NdArray<f64>>::from_mesh(
            &mesh, &geom, &ops.face_nodes, ops.n_face_nodes, &device,
        );

        // Lake at rest: h=1, u=v=0
        let mut cpu_sol = crate::solver::state::SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                cpu_sol.set_state(k, i, crate::solver::state::SWEState2D::new(1.0, 0.0, 0.0));
            }
        }

        let mut burn_sol = BurnSWESolution2D::<NdArray<f64>>::from_cpu(&cpu_sol, &device);
        let config = BurnRhsConfig::default();

        // Take a small time step
        let dt = 0.001;
        ssp_rk3_step_burn(&mut burn_sol, dt, &burn_ops, &burn_geom, &connectivity, &config);

        // Lake at rest should stay at rest (approximately)
        let result = burn_sol.to_cpu();
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let state = result.get_state(k, i);
                // Allow some numerical error, but state should be close to initial
                assert!((state.h - 1.0).abs() < 0.1, "h changed too much: {}", state.h);
                assert!(state.hu.abs() < 0.1, "hu should be ~0: {}", state.hu);
                assert!(state.hv.abs() < 0.1, "hv should be ~0: {}", state.hv);
            }
        }
    }
}
