//! Integration tests for 2D Shallow Water Equations solver.
//!
//! These tests verify:
//! - Lake-at-rest (well-balanced property)
//! - Mass conservation
//! - Dam break evolution
//! - Coriolis effects

use dg_rs::{
    DGOperators2D, GeometricFactors2D, Mesh2D, Reflective2D, SWE2DRhsConfig, SWEFluxType2D,
    SWESolution2D, SWEState2D, ShallowWater2D, compute_dt_swe_2d, compute_rhs_swe_2d,
};

const G: f64 = 10.0;

/// Run SSP-RK3 step for 2D SWE.
fn ssp_rk3_swe_step<BC: dg_rs::SWEBoundaryCondition2D>(
    q: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    dt: f64,
    time: f64,
) {
    let n = q.data.len();

    // Stage 1: u1 = u + dt * L(u)
    let rhs1 = compute_rhs_swe_2d(q, mesh, ops, geom, config, time);
    let mut u1 = SWESolution2D::new(q.n_elements, q.n_nodes);
    u1.copy_from(q);
    u1.axpy(dt, &rhs1);

    // Stage 2: u2 = 3/4 * u + 1/4 * (u1 + dt * L(u1))
    let rhs2 = compute_rhs_swe_2d(&u1, mesh, ops, geom, config, time + dt);
    u1.axpy(dt, &rhs2);
    let mut u2 = SWESolution2D::new(q.n_elements, q.n_nodes);
    for i in 0..n {
        u2.data[i] = 0.75 * q.data[i] + 0.25 * u1.data[i];
    }

    // Stage 3: u_new = 1/3 * u + 2/3 * (u2 + dt * L(u2))
    let rhs3 = compute_rhs_swe_2d(&u2, mesh, ops, geom, config, time + 0.5 * dt);
    u2.axpy(dt, &rhs3);
    for i in 0..n {
        q.data[i] = (1.0 / 3.0) * q.data[i] + (2.0 / 3.0) * u2.data[i];
    }
}

/// Test lake-at-rest: uniform depth, zero velocity.
///
/// For a well-balanced scheme, the RHS should be zero regardless of bathymetry.
#[test]
fn test_lake_at_rest() {
    let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
    let ops = DGOperators2D::new(3);
    let geom = GeometricFactors2D::compute(&mesh);
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Initialize: h = 5.0, u = v = 0
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    for k in 0..mesh.n_elements {
        for i in 0..ops.n_nodes {
            q.set_state(k, i, SWEState2D::new(5.0, 0.0, 0.0));
        }
    }

    // Compute RHS
    let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

    // RHS should be zero (lake at rest)
    let max_rhs = rhs.max_abs();
    assert!(
        max_rhs < 1e-10,
        "Lake at rest should have zero RHS, got {}",
        max_rhs
    );
}

/// Test lake-at-rest with perturbation.
///
/// Verifies that small perturbations evolve correctly.
#[test]
fn test_lake_at_rest_perturbation() {
    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 8, 8);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh);
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Initialize: h = 2.0 + perturbation
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| {
            2.0 + 0.1
                * (2.0 * std::f64::consts::PI * x / 10.0).sin()
                * (2.0 * std::f64::consts::PI * y / 10.0).sin()
        },
        |_, _| 0.0,
        |_, _| 0.0,
    );

    let initial_mass = q.integrate_depth(&ops, &geom);

    // Take a few time steps
    let cfl = 0.3;
    let mut time = 0.0;
    for _ in 0..10 {
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        ssp_rk3_swe_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;
    }

    // Check mass conservation
    let final_mass = q.integrate_depth(&ops, &geom);
    let mass_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        mass_error < 1e-12,
        "Mass should be conserved: error = {:.2e}",
        mass_error
    );

    // Check no negative depths
    assert!(!q.has_negative_depth(), "Depth should remain positive");
}

/// Test mass conservation for smooth initial condition.
///
/// Uses a smooth initial condition to test conservation properties
/// without the complications of discontinuities.
#[test]
fn test_mass_conservation_smooth() {
    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 8, 8);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh);
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Smooth Gaussian bump initial condition
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| {
            let cx = 5.0;
            let cy = 5.0;
            let dist_sq = (x - cx).powi(2) + (y - cy).powi(2);
            2.0 + 0.5 * (-dist_sq / 2.0).exp()
        },
        |_, _| 0.0,
        |_, _| 0.0,
    );

    let initial_mass = q.integrate_depth(&ops, &geom);
    let initial_x_momentum = q.integrate_x_momentum(&ops, &geom);
    let initial_y_momentum = q.integrate_y_momentum(&ops, &geom);

    // Run simulation for a short time
    let cfl = 0.2;
    let mut time = 0.0;
    let end_time = 0.5;
    while time < end_time {
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);
        ssp_rk3_swe_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;

        // Early termination if solution has issues
        if q.max_abs() > 100.0 || q.has_negative_depth() {
            panic!(
                "Solution unstable at t = {}: max = {}, min_h = {}",
                time,
                q.max_abs(),
                q.min_depth()
            );
        }
    }

    // Check mass conservation (should be exact for periodic)
    let final_mass = q.integrate_depth(&ops, &geom);
    let mass_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        mass_error < 1e-11,
        "Mass should be conserved: error = {:.2e}",
        mass_error
    );

    // Check x-momentum conservation (should be conserved for periodic without Coriolis)
    let final_x_momentum = q.integrate_x_momentum(&ops, &geom);
    let x_mom_error = (final_x_momentum - initial_x_momentum).abs();
    assert!(
        x_mom_error < 1e-10,
        "X-momentum should be conserved: error = {:.2e}",
        x_mom_error
    );

    // Check y-momentum conservation
    let final_y_momentum = q.integrate_y_momentum(&ops, &geom);
    let y_mom_error = (final_y_momentum - initial_y_momentum).abs();
    assert!(
        y_mom_error < 1e-10,
        "Y-momentum should be conserved: error = {:.2e}",
        y_mom_error
    );
}

/// Test circular dam break.
///
/// A circular dam break tests radial symmetry and 2D behavior.
#[test]
fn test_circular_dam_break() {
    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 8, 8);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh);
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Circular dam: higher water in center
    let cx = 5.0;
    let cy = 5.0;
    let r = 2.0;
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| {
            let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            if dist < r { 3.0 } else { 1.0 }
        },
        |_, _| 0.0,
        |_, _| 0.0,
    );

    let initial_mass = q.integrate_depth(&ops, &geom);

    // Run simulation
    let cfl = 0.2;
    let mut time = 0.0;
    let end_time = 0.3;
    while time < end_time {
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);
        ssp_rk3_swe_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;

        // Early termination if solution blows up
        if q.max_abs() > 100.0 {
            panic!("Solution blew up at t = {}", time);
        }
    }

    // Check mass conservation
    let final_mass = q.integrate_depth(&ops, &geom);
    let mass_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        mass_error < 1e-11,
        "Mass should be conserved: error = {:.2e}",
        mass_error
    );

    // Check no negative depths
    assert!(
        !q.has_negative_depth(),
        "Depth should remain positive, min = {}",
        q.min_depth()
    );
}

/// Test Coriolis effect on uniform flow.
///
/// Verifies that Coriolis source term is computed correctly
/// by checking that it deflects flow as expected.
#[test]
fn test_coriolis_effect() {
    // Small domain for simplicity
    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 4, 4);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh);

    // Norwegian coast Coriolis
    let f = 1.2e-4;
    let equation = ShallowWater2D::with_coriolis(G, f);
    let bc = Reflective2D::new();

    // Uniform flow in x-direction: should be deflected to the right (positive y)
    let h = 10.0;
    let u = 1.0;
    let v = 0.0;

    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(&mesh, &ops, |_, _| h, |_, _| u, |_, _| v);

    // Compute RHS with Coriolis
    let config_with = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(true);
    let rhs_with = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config_with, 0.0);

    // Compute RHS without Coriolis
    let config_without = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);
    let rhs_without = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config_without, 0.0);

    // The difference should be the Coriolis source term
    // d(hu)/dt += f * hv = f * h * v = 0 (since v = 0)
    // d(hv)/dt -= f * hu = -f * h * u = -1.2e-4 * 10 * 1 = -1.2e-3
    let expected_hv_source = -f * h * u;

    // Check that the hv difference is approximately the expected source
    let diff_hv = rhs_with.get_var(0, 0, 2) - rhs_without.get_var(0, 0, 2);
    assert!(
        (diff_hv - expected_hv_source).abs() < 1e-8,
        "Coriolis hv source: got {:.2e}, expected {:.2e}",
        diff_hv,
        expected_hv_source
    );

    // Run a few time steps and verify momentum changes direction
    let mut q_with = q.clone();
    let cfl = 0.3;
    let mut time = 0.0;
    let end_time = 100.0; // Need long enough for Coriolis to have visible effect
    while time < end_time {
        let dt = compute_dt_swe_2d(&q_with, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);
        ssp_rk3_swe_step(&mut q_with, &mesh, &ops, &geom, &config_with, dt, time);
        time += dt;
    }

    // After evolving, should have developed v-momentum (rightward deflection)
    let final_y_momentum = q_with.integrate_y_momentum(&ops, &geom);
    // Coriolis should have deflected momentum to negative v (in Northern Hemisphere)
    assert!(
        final_y_momentum < -1e-6,
        "Coriolis should deflect x-momentum to negative v: got y-momentum = {:.2e}",
        final_y_momentum
    );
}

/// Test different flux types give similar results for smooth solutions.
#[test]
fn test_flux_type_comparison() {
    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 4, 4);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh);
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();

    // Smooth initial condition
    let create_ic = || {
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        q.set_from_functions(
            &mesh,
            &ops,
            |x, y| {
                2.0 + 0.2
                    * (2.0 * std::f64::consts::PI * x / 10.0).sin()
                    * (2.0 * std::f64::consts::PI * y / 10.0).sin()
            },
            |_, _| 0.0,
            |_, _| 0.0,
        );
        q
    };

    let flux_types = [
        SWEFluxType2D::Roe,
        SWEFluxType2D::HLL,
        SWEFluxType2D::Rusanov,
    ];

    // Run a few steps with each flux type
    let end_time = 0.1;
    let cfl = 0.2;
    let mut final_masses = Vec::new();
    let mut final_max_h = Vec::new();

    for flux_type in &flux_types {
        let mut q = create_ic();
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_flux_type(*flux_type)
            .with_coriolis(false);

        let mut time = 0.0;
        while time < end_time {
            let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
            let dt = dt.min(end_time - time);
            ssp_rk3_swe_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
            time += dt;
        }

        final_masses.push(q.integrate_depth(&ops, &geom));
        final_max_h.push(q.max_depth());
    }

    // All flux types should conserve mass
    let initial_mass = create_ic().integrate_depth(&ops, &geom);
    for (i, &mass) in final_masses.iter().enumerate() {
        let error = ((mass - initial_mass) / initial_mass).abs();
        assert!(
            error < 1e-12,
            "{:?} flux: mass error = {:.2e}",
            flux_types[i],
            error
        );
    }

    // Results should be similar for smooth solution
    let max_diff = final_max_h.iter().fold(0.0_f64, |acc, &h| {
        final_max_h
            .iter()
            .fold(acc, |inner, &h2| inner.max((h - h2).abs()))
    });
    assert!(
        max_diff < 0.1,
        "Different flux types should give similar results for smooth solutions: diff = {}",
        max_diff
    );
}

/// Test reflective boundary conditions.
#[test]
fn test_reflective_boundary() {
    let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh);
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Wave hitting wall
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, _y| {
            // Gaussian bump near left boundary
            let x0 = 2.0;
            2.0 + 0.5 * (-(x - x0).powi(2) / 1.0).exp()
        },
        |x, _y| {
            // Moving right
            let x0 = 2.0;
            0.5 * (-(x - x0).powi(2) / 1.0).exp()
        },
        |_, _| 0.0,
    );

    let initial_mass = q.integrate_depth(&ops, &geom);

    // Run simulation
    let cfl = 0.2;
    let mut time = 0.0;
    let end_time = 2.0;
    while time < end_time {
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);
        ssp_rk3_swe_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;

        // Check stability
        if q.max_abs() > 100.0 || q.has_negative_depth() {
            panic!(
                "Solution unstable at t = {}: max = {}, min_h = {}",
                time,
                q.max_abs(),
                q.min_depth()
            );
        }
    }

    // Mass should be approximately conserved (some error due to wall reflection)
    let final_mass = q.integrate_depth(&ops, &geom);
    let mass_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        mass_error < 1e-6,
        "Mass should be approximately conserved: error = {:.2e}",
        mass_error
    );
}

/// Test stability for long simulation.
#[test]
fn test_long_term_stability() {
    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 4, 4);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh);
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Smooth initial perturbation
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| {
            1.0 + 0.1
                * (2.0 * std::f64::consts::PI * x / 10.0).sin()
                * (2.0 * std::f64::consts::PI * y / 10.0).sin()
        },
        |_, _| 0.0,
        |_, _| 0.0,
    );

    let initial_mass = q.integrate_depth(&ops, &geom);

    // Run for many time steps
    let cfl = 0.3;
    let mut time = 0.0;
    let end_time = 10.0;
    let mut step = 0;
    while time < end_time {
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);
        ssp_rk3_swe_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;
        step += 1;

        // Check stability every 100 steps
        if step % 100 == 0 {
            assert!(
                !q.has_negative_depth(),
                "Negative depth at t = {}, step {}",
                time,
                step
            );
            let current_mass = q.integrate_depth(&ops, &geom);
            let mass_error = ((current_mass - initial_mass) / initial_mass).abs();
            assert!(
                mass_error < 1e-10,
                "Mass drift at t = {}: {:.2e}",
                time,
                mass_error
            );
        }
    }

    // Final checks
    assert!(
        !q.has_negative_depth(),
        "Should remain positive for {} steps",
        step
    );
    let final_mass = q.integrate_depth(&ops, &geom);
    let final_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        final_error < 1e-10,
        "Final mass error after {} steps: {:.2e}",
        step,
        final_error
    );
}
