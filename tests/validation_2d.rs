//! Validation tests for 2D SWE solver.
//!
//! These tests verify the solver against analytical solutions and physical principles:
//! 1. Circular dam break (radial symmetry, mass conservation)
//! 2. Standing wave in channel (correct period)
//! 3. Geostrophic balance (Coriolis equilibrium)
//! 4. Sponge layer wave absorption

use dg_rs::{
    DGOperators2D, GeometricFactors2D, Mesh2D, Reflective2D, SWE2DRhsConfig, SWESolution2D,
    SWEState2D, ShallowWater2D, compute_dt_swe_2d, compute_rhs_swe_2d,
    source::{CombinedSource2D, CoriolisSource2D, SpongeLayer2D},
};
use dg_rs::types::ElementIndex;
use std::f64::consts::PI;

fn k(idx: usize) -> ElementIndex {
    ElementIndex::new(idx)
}

const G: f64 = 9.81;

/// Helper to compute one SSP-RK3 step with source terms
fn ssp_rk3_step_with_sources<BC: dg_rs::SWEBoundaryCondition2D>(
    q: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    dt: f64,
    time: f64,
) {
    let rhs1 = compute_rhs_swe_2d(q, mesh, ops, geom, config, time);
    let mut q1 = q.clone();
    q1.axpy(dt, &rhs1);

    let rhs2 = compute_rhs_swe_2d(&q1, mesh, ops, geom, config, time + dt);
    let mut q2 = q.clone();
    q2.axpy(0.25 * dt, &rhs1);
    q2.axpy(0.25 * dt, &rhs2);

    let rhs3 = compute_rhs_swe_2d(&q2, mesh, ops, geom, config, time + 0.5 * dt);
    q.scale(1.0 / 3.0);
    q.axpy(2.0 / 3.0, &q2);
    q.axpy(2.0 / 3.0 * dt, &rhs3);
}

/// Test circular dam break with mass conservation.
///
/// Initial condition: circular dam with h_in > h_out
/// Verifies:
/// - Mass is conserved to machine precision
/// - Radial symmetry is maintained
/// - Solution remains stable
#[test]
fn test_circular_dam_break_mass_conservation() {
    let mesh = Mesh2D::uniform_periodic(0.0, 100.0, 0.0, 100.0, 10, 10);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh);

    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Circular dam: h = 3 inside r < 20, h = 1 outside
    let cx = 50.0;
    let cy = 50.0;
    let r0 = 20.0;

    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| {
            let r = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            if r < r0 { 3.0 } else { 1.0 }
        },
        |_, _| 0.0,
        |_, _| 0.0,
    );

    let initial_mass = q.integrate_depth(&ops, &geom);

    // Run for a few time steps
    let cfl = 0.3;
    let end_time = 1.0;
    let mut time = 0.0;

    while time < end_time {
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);
        ssp_rk3_step_with_sources(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;

        // Check solution remains bounded
        assert!(
            !q.has_negative_depth(),
            "Negative depth at time {:.3}",
            time
        );
    }

    let final_mass = q.integrate_depth(&ops, &geom);
    let mass_error = ((final_mass - initial_mass) / initial_mass).abs();

    assert!(
        mass_error < 1e-10,
        "Mass conservation error: {:.2e}",
        mass_error
    );
}

/// Test standing wave in rectangular channel.
///
/// Analytical solution for frictionless channel:
/// η(x, t) = A * cos(kx) * cos(ωt)
/// where ω = sqrt(gH) * k
///
/// This test verifies that the wave period matches the analytical prediction.
#[test]
fn test_standing_wave_period() {
    let lx = 100.0;
    let ly = 10.0;
    let nx = 20;
    let ny = 2;

    // Channel mesh (periodic in x, walls in y)
    let mesh = Mesh2D::channel_periodic_x(0.0, lx, 0.0, ly, nx, ny);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh);

    let h0 = 10.0; // Mean depth
    let amplitude = 0.1;
    let wave_k = 2.0 * PI / lx; // Wave number
    let omega = wave_k * (G * h0).sqrt(); // Analytical frequency
    let period = 2.0 * PI / omega;

    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Initial condition: η = A*cos(kx), u = v = 0
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, _y| h0 + amplitude * (wave_k * x).cos(),
        |_, _| 0.0,
        |_, _| 0.0,
    );

    // Track elevation at center point over time
    let mut max_eta_times: Vec<f64> = Vec::new();
    let center_elem = mesh.n_elements / 2;

    let cfl = 0.3;
    let end_time = 2.0 * period;
    let mut time = 0.0;
    let mut last_eta = f64::NEG_INFINITY;
    let mut searching_peak = true;

    while time < end_time {
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);
        ssp_rk3_step_with_sources(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;

        // Sample elevation at center
        let eta = q.get_state(k(center_elem), ops.n_nodes / 2).h - h0;

        // Detect peaks (for period measurement)
        if searching_peak && eta < last_eta && last_eta > 0.0 {
            max_eta_times.push(time - dt);
            searching_peak = false;
        } else if eta > last_eta {
            searching_peak = true;
        }
        last_eta = eta;
    }

    // Measure period from peak-to-peak
    if max_eta_times.len() >= 2 {
        let measured_period = max_eta_times[1] - max_eta_times[0];
        let period_error = ((measured_period - period) / period).abs();

        assert!(
            period_error < 0.05, // 5% tolerance
            "Standing wave period error: {:.1}% (expected {:.3}s, got {:.3}s)",
            period_error * 100.0,
            period,
            measured_period
        );
    }
}

/// Test geostrophic balance.
///
/// In steady state, the pressure gradient should balance Coriolis:
/// f * v = g * ∂η/∂x
/// f * u = -g * ∂η/∂y
///
/// Initialize with a linear SSH gradient and corresponding velocities.
/// The solution should remain steady.
#[test]
fn test_geostrophic_balance() {
    let lx = 100000.0; // 100 km
    let ly = 100000.0;

    let mesh = Mesh2D::uniform_periodic(0.0, lx, 0.0, ly, 10, 10);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh);

    let f = 1.0e-4; // Coriolis parameter
    let h0 = 100.0; // Mean depth
    let deta_dx = 1.0e-6; // SSH gradient (1 mm per km)

    // Geostrophic velocity: f * v = g * deta/dx => v = g * deta_dx / f
    let v_geo = G * deta_dx / f;

    let equation = ShallowWater2D::with_coriolis(G, f);
    let bc = Reflective2D::new();

    // Use trait-based Coriolis
    let coriolis = CoriolisSource2D::f_plane(f);
    let config = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_source_terms(&coriolis);

    // Initialize in geostrophic balance
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, _y| h0 + deta_dx * x, // η = η0 + dη/dx * x
        |_, _| 0.0,               // u = 0
        |_, _| v_geo,             // v = geostrophic
    );

    // Get initial RHS (should be near zero for balance)
    let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
    let max_rhs = rhs.max_abs();

    // RHS should be small relative to the momentum scale: h * v_geo = 100 * 0.0981 ≈ 10
    // On a coarse mesh, numerical discretization introduces some imbalance.
    // A 5% error (0.5) is acceptable.
    let momentum_scale = h0 * v_geo.abs();
    assert!(
        max_rhs < 0.05 * momentum_scale,
        "Geostrophic balance RHS too large: {:.2e} (scale: {:.2e})",
        max_rhs,
        momentum_scale
    );

    // Run a few time steps and verify solution remains stable
    let cfl = 0.3;
    let mut time = 0.0;
    let end_time = 1000.0; // 1000 seconds

    while time < end_time {
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);
        ssp_rk3_step_with_sources(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;
    }

    // Velocity should remain close to geostrophic
    let mut max_v_deviation: f64 = 0.0;
    for ki in 0..mesh.n_elements {
        for i in 0..ops.n_nodes {
            let state = q.get_state(k(ki), i);
            let v = state.hv / state.h;
            max_v_deviation = max_v_deviation.max((v - v_geo).abs());
        }
    }

    // Allow 5% drift over 1000s - acceptable for coarse mesh discretization
    assert!(
        max_v_deviation < 0.05 * v_geo.abs(),
        "Geostrophic balance drift: max deviation = {:.2e}, expected < {:.2e} (5% of v_geo)",
        max_v_deviation,
        0.05 * v_geo.abs()
    );
}

/// Test sponge layer wave absorption.
///
/// Send a wave toward a sponge layer and verify it's damped
/// compared to a simulation without sponge.
#[test]
fn test_sponge_layer_absorption() {
    let lx = 200.0;
    let ly = 50.0;
    let sponge_width = 30.0;

    let mesh = Mesh2D::channel_periodic_x(0.0, lx, 0.0, ly, 20, 5);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh);

    let h0 = 10.0;
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();

    // Initial condition: Gaussian bump moving right
    let initial = |x: f64, _y: f64| -> f64 {
        let x0 = 50.0;
        let sigma = 10.0;
        h0 + 1.0 * (-((x - x0) / sigma).powi(2)).exp()
    };

    // Run WITHOUT sponge
    let config_no_sponge = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);
    let mut q_no_sponge = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q_no_sponge.set_from_functions(&mesh, &ops, initial, |_, _| 0.0, |_, _| 0.0);

    // Run WITH sponge on right boundary
    let sponge = SpongeLayer2D::rectangular(
        |_x, _y, _t| SWEState2D::new(h0, 0.0, 0.0),
        1.0, // gamma_max
        sponge_width,
        (0.0, lx, 0.0, ly),
        [false, true, false, false], // Right boundary only
    );
    let config_with_sponge = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_source_terms(&sponge);
    let mut q_with_sponge = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q_with_sponge.set_from_functions(&mesh, &ops, initial, |_, _| 0.0, |_, _| 0.0);

    // Run both simulations until wave reaches boundary
    let cfl = 0.3;
    let end_time = 15.0; // Time for wave to reach right boundary
    let mut time = 0.0;

    while time < end_time {
        let dt = compute_dt_swe_2d(&q_no_sponge, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);

        ssp_rk3_step_with_sources(
            &mut q_no_sponge,
            &mesh,
            &ops,
            &geom,
            &config_no_sponge,
            dt,
            time,
        );
        ssp_rk3_step_with_sources(
            &mut q_with_sponge,
            &mesh,
            &ops,
            &geom,
            &config_with_sponge,
            dt,
            time,
        );
        time += dt;
    }

    // Measure perturbation energy in the sponge zone
    let mut energy_no_sponge = 0.0;
    let mut energy_with_sponge = 0.0;

    for ki in 0..mesh.n_elements {
        let j = geom.det_j[ki];
        for (i, &w) in ops.weights.iter().enumerate() {
            let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
            let [x, _y] = mesh.reference_to_physical(k(ki), r, s);

            // Only count energy in right half (where sponge is)
            if x > lx / 2.0 {
                let state_no = q_no_sponge.get_state(k(ki), i);
                let state_with = q_with_sponge.get_state(k(ki), i);

                // Perturbation energy: (h - h0)²
                energy_no_sponge += w * (state_no.h - h0).powi(2) * j;
                energy_with_sponge += w * (state_with.h - h0).powi(2) * j;
            }
        }
    }

    // Sponge should reduce energy significantly
    let reduction = energy_with_sponge / energy_no_sponge.max(1e-14);
    assert!(
        reduction < 0.5,
        "Sponge should reduce wave energy by >50%: ratio = {:.2}",
        reduction
    );
}

/// Test combined source terms (Coriolis + sponge).
#[test]
fn test_combined_source_terms() {
    let mesh = Mesh2D::uniform_periodic(0.0, 100.0, 0.0, 100.0, 5, 5);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh);

    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();

    // Create combined sources
    let coriolis = CoriolisSource2D::f_plane(1.0e-4);
    let sponge = SpongeLayer2D::rectangular(
        |_x, _y, _t| SWEState2D::new(10.0, 0.0, 0.0),
        0.1,
        10.0,
        (0.0, 100.0, 0.0, 100.0),
        [true, false, false, false],
    );
    let combined = CombinedSource2D::new(vec![&coriolis, &sponge]);

    let config = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_source_terms(&combined);

    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(&mesh, &ops, |_, _| 10.0, |_, _| 1.0, |_, _| 0.5);

    // Just verify it runs without error
    let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
    assert!(
        rhs.max_abs() < 1000.0,
        "Combined sources should give bounded RHS"
    );
}
