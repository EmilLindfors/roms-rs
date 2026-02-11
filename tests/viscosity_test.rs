//! Integration tests for horizontal viscosity in 2D SWE.
//!
//! Tests:
//! 1. Exponential decay: sinusoidal velocity perturbation decays as exp(-ν·k²·t)
//! 2. Smagorinsky smoke test: short simulation doesn't blow up
//! 3. Parallel consistency: parallel RHS matches sequential

use dg_rs::{
    DGOperators2D, GeometricFactors2D, HorizontalViscosity2D, Mesh2D, Reflective2D,
    SWE2DRhsConfig, SWEFluxType2D, SWESolution2D, ShallowWater2D,
    compute_dt_swe_2d, compute_dt_viscosity, compute_rhs_swe_2d,
};
use std::f64::consts::PI;

/// SSP-RK3 step for SWE with viscosity.
fn ssp_rk3_step<BC: dg_rs::SWEBoundaryCondition2D>(
    q: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    dt: f64,
    time: f64,
) {
    let n = q.data[0].len(); // total nodes per variable (SoA layout)

    // Stage 1
    let rhs1 = compute_rhs_swe_2d(q, mesh, ops, geom, config, time);
    let mut u1 = SWESolution2D::new(q.n_elements, q.n_nodes);
    u1.copy_from(q);
    u1.axpy(dt, &rhs1);

    // Stage 2
    let rhs2 = compute_rhs_swe_2d(&u1, mesh, ops, geom, config, time + dt);
    u1.axpy(dt, &rhs2);
    let mut u2 = SWESolution2D::new(q.n_elements, q.n_nodes);
    for var in 0..3 {
        for i in 0..n {
            u2.data[var][i] = 0.75 * q.data[var][i] + 0.25 * u1.data[var][i];
        }
    }

    // Stage 3
    let rhs3 = compute_rhs_swe_2d(&u2, mesh, ops, geom, config, time + 0.5 * dt);
    u2.axpy(dt, &rhs3);
    for var in 0..3 {
        for i in 0..n {
            q.data[var][i] = (1.0 / 3.0) * q.data[var][i] + (2.0 / 3.0) * u2.data[var][i];
        }
    }
}

/// Exponential decay test: a sinusoidal velocity perturbation on constant depth
/// with constant viscosity should decay as exp(-ν·k²·t).
///
/// Setup: periodic domain [0, 2π]², h = 1 (uniform), hu = ε·sin(x)·sin(y), hv = 0.
/// With ν constant and no other physics, the x-momentum perturbation decays as
/// ε·exp(-ν·(kx² + ky²)·t) where kx = ky = 1, so k² = 2.
#[test]
fn test_viscosity_exponential_decay() {
    let l = 2.0 * PI;
    let nx = 8;
    let ny = 8;
    let order = 3;
    let nu = 0.1;
    let eps = 0.01; // small perturbation to keep linearized regime

    let mesh = Mesh2D::uniform_periodic(0.0, l, 0.0, l, nx, ny);
    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh);

    // Use low gravity to minimize wave effects relative to diffusion
    let equation = ShallowWater2D::new(0.1);
    let bc = Reflective2D::new(); // periodic mesh, BC won't be hit

    let visc = HorizontalViscosity2D::constant(nu);
    let config = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_flux_type(SWEFluxType2D::Roe)
        .with_viscosity(&visc);

    // Initialize: h = 1, hu = ε·sin(x)·sin(y), hv = 0
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |_x, _y| 1.0,
        |x, y| eps * x.sin() * y.sin(), // u = hu/h = ε·sin(x)·sin(y)
        |_x, _y| 0.0,
    );

    // Measure initial amplitude (SoA: data[1] = all hu values)
    let initial_max_hu = q
        .data[1]
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);

    // Time integration
    let t_final = 1.0;
    let mut time = 0.0;
    let cfl_adv = 0.3;
    let cfl_visc = 0.2;

    let min_h_elem = (0..mesh.n_elements)
        .map(|k_elem| geom.det_j[k_elem].sqrt() * 2.0)
        .fold(f64::INFINITY, f64::min);

    while time < t_final {
        let dt_adv = compute_dt_swe_2d(&q, &mesh, &geom, &equation, order, cfl_adv);
        let dt_visc = compute_dt_viscosity(nu, min_h_elem, order, cfl_visc);
        let dt = dt_adv.min(dt_visc).min(t_final - time);
        ssp_rk3_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;
    }

    // Measure final amplitude (SoA: data[1] = all hu values)
    let final_max_hu = q
        .data[1]
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);

    // Theoretical decay factor: exp(-ν·k²·T) = exp(-0.1·2·1.0) = exp(-0.2)
    let theoretical_decay = (-nu * 2.0 * t_final).exp();
    let measured_decay = final_max_hu / initial_max_hu;

    println!("Initial max |hu| = {initial_max_hu:.6e}");
    println!("Final max |hu|   = {final_max_hu:.6e}");
    println!("Measured decay    = {measured_decay:.6}");
    println!("Theoretical decay = {theoretical_decay:.6}");

    // The amplitude should have decreased
    assert!(
        final_max_hu < initial_max_hu,
        "Viscosity should damp the perturbation: initial={initial_max_hu:.6e}, final={final_max_hu:.6e}"
    );

    // Allow generous tolerance since we have wave coupling and numerical diffusion
    // The measured decay should be in the right ballpark (within factor of 2)
    let relative_error = ((measured_decay - theoretical_decay) / theoretical_decay).abs();
    println!("Relative error    = {relative_error:.4}");
    assert!(
        relative_error < 0.5,
        "Decay rate should approximate exp(-ν·k²·t): measured={measured_decay:.6}, theory={theoretical_decay:.6}, rel_err={relative_error:.4}"
    );
}

/// Smoke test for Smagorinsky viscosity: verify no blow-up.
#[test]
fn test_smagorinsky_smoke() {
    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 6, 6);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh);
    let equation = ShallowWater2D::new(9.81);
    let bc = Reflective2D::new();

    let visc = HorizontalViscosity2D::smagorinsky(0.15);
    let config = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_viscosity(&visc);

    // Initialize with a dam-break-like perturbation
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| {
            if (x - 5.0).powi(2) + (y - 5.0).powi(2) < 4.0 {
                2.0
            } else {
                1.0
            }
        },
        |_, _| 0.0,
        |_, _| 0.0,
    );

    let initial_mass = q.integrate_depth(&ops, &geom);

    // Take 20 time steps
    let cfl = 0.2;
    let min_h_elem = (0..mesh.n_elements)
        .map(|k_elem| geom.det_j[k_elem].sqrt() * 2.0)
        .fold(f64::INFINITY, f64::min);

    let mut time = 0.0;
    for _ in 0..20 {
        let dt_adv = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        // Use a generous viscosity estimate for dt
        let dt_visc = compute_dt_viscosity(10.0, min_h_elem, ops.order, 0.1);
        let dt = dt_adv.min(dt_visc);
        ssp_rk3_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;
    }

    // Check no blow-up
    let max_val = q.max_abs();
    assert!(
        max_val < 100.0,
        "Smagorinsky should not blow up: max_abs = {max_val:.6e}"
    );

    // Mass should be roughly conserved (viscosity only acts on momentum)
    let final_mass = q.integrate_depth(&ops, &geom);
    let mass_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        mass_error < 1e-8,
        "Mass should be conserved: initial={initial_mass:.10}, final={final_mass:.10}, error={mass_error:.2e}"
    );
}

/// Verify that viscosity doesn't affect continuity (mass conservation).
/// With viscosity on, the h equation should get zero contribution from viscous terms.
#[test]
fn test_viscosity_mass_conservation() {
    let l = 2.0 * PI;
    let mesh = Mesh2D::uniform_periodic(0.0, l, 0.0, l, 4, 4);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh);
    let equation = ShallowWater2D::new(9.81);
    let bc = Reflective2D::new();

    let visc = HorizontalViscosity2D::constant(1.0);
    let config = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_viscosity(&visc);

    // Initialize with non-trivial velocity
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |_x, _y| 1.0,
        |x, _y| 0.1 * x.sin(),
        |_x, y| 0.1 * y.cos(),
    );

    let initial_mass = q.integrate_depth(&ops, &geom);

    // Take a few steps
    let min_h_elem = (0..mesh.n_elements)
        .map(|k_elem| geom.det_j[k_elem].sqrt() * 2.0)
        .fold(f64::INFINITY, f64::min);

    let mut time = 0.0;
    for _ in 0..5 {
        let dt_adv = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, 0.2);
        let dt_visc = compute_dt_viscosity(1.0, min_h_elem, ops.order, 0.1);
        let dt = dt_adv.min(dt_visc);
        ssp_rk3_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;
    }

    let final_mass = q.integrate_depth(&ops, &geom);
    let mass_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        mass_error < 1e-10,
        "Viscosity must not affect continuity: mass error = {mass_error:.2e}"
    );
}

/// Test the diffusive CFL helper.
#[test]
fn test_compute_dt_viscosity() {
    // Basic sanity
    let dt = compute_dt_viscosity(1.0, 1.0, 2, 0.5);
    // (2*2+1)^2 = 25, dt = 0.5 * 1.0 / (1.0 * 25) = 0.02
    assert!((dt - 0.02).abs() < 1e-14, "dt = {dt}");

    // Zero viscosity → infinite dt
    let dt_zero = compute_dt_viscosity(0.0, 1.0, 2, 0.5);
    assert!(dt_zero.is_infinite());

    // Smaller element → smaller dt (quadratic)
    let dt_small = compute_dt_viscosity(1.0, 0.5, 2, 0.5);
    assert!((dt_small / dt - 0.25).abs() < 1e-14, "dt_small = {dt_small}");
}

/// Test parallel RHS matches sequential when viscosity is enabled.
#[cfg(feature = "parallel")]
#[test]
fn test_viscosity_parallel_consistency() {
    use dg_rs::compute_rhs_swe_2d_parallel;

    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 4, 4);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh);
    let equation = ShallowWater2D::new(9.81);
    let bc = Reflective2D::new();

    let visc = HorizontalViscosity2D::constant(0.5);
    let config = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_viscosity(&visc);

    // Non-trivial initial condition
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| 1.0 + 0.1 * (2.0 * PI * x / 10.0).sin() * (2.0 * PI * y / 10.0).cos(),
        |x, _y| 0.05 * (2.0 * PI * x / 10.0).cos(),
        |_x, y| 0.05 * (2.0 * PI * y / 10.0).sin(),
    );

    let rhs_serial = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
    let rhs_parallel = compute_rhs_swe_2d_parallel(&q, &mesh, &ops, &geom, &config, 0.0);

    let max_diff = (0..3)
        .flat_map(|var| {
            rhs_serial.data[var]
                .iter()
                .zip(rhs_parallel.data[var].iter())
                .map(|(a, b)| (a - b).abs())
        })
        .fold(0.0_f64, f64::max);

    assert!(
        max_diff < 1e-10,
        "Parallel RHS with viscosity should match serial: max diff = {max_diff:.2e}"
    );
}
