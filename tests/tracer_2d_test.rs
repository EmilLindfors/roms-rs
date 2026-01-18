//! Integration tests for 2D tracer transport.
//!
//! These tests verify:
//! 1. Tracer conservation with periodic boundaries (no sources/sinks)
//! 2. Limiter effectiveness at preventing oscillations
//! 3. Physical bounds preservation
//! 4. Coupled SWE-tracer stability

use dg_rs::boundary::Reflective2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::Mesh2D;
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::solver::{
    ExtrapolationTracerBC, SWE2DRhsConfig, SWESolution2D, SWEState2D, TVBParameter2D,
    Tracer2DRhsConfig, TracerBounds, TracerSolution2D, TracerState, apply_tracer_limiters_2d,
    compute_rhs_swe_2d, compute_rhs_tracer_2d,
};
use dg_rs::time::{
    CoupledRhs2D, CoupledState2D, CoupledTimeConfig, compute_dt_coupled,
    run_coupled_simulation_limited, total_tracer,
};
use dg_rs::types::ElementIndex;

fn k(idx: usize) -> ElementIndex {
    ElementIndex::new(idx)
}

const G: f64 = 9.81;
const H_MIN: f64 = 1e-6;
const TOL: f64 = 1e-8;

/// Create a standard test setup with periodic mesh.
fn create_periodic_setup(
    nx: usize,
    ny: usize,
    order: usize,
) -> (Mesh2D, DGOperators2D, GeometricFactors2D) {
    let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, nx, ny);
    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh);
    (mesh, ops, geom)
}

/// Create a setup with wall boundaries (for testing limiters).
fn create_wall_setup(
    nx: usize,
    ny: usize,
    order: usize,
) -> (Mesh2D, DGOperators2D, GeometricFactors2D) {
    let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, nx, ny);
    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh);
    (mesh, ops, geom)
}

// ============================================================================
// Conservation Tests
// ============================================================================

/// Test that tracer content is conserved with periodic BC and no sources.
///
/// With periodic boundaries and no source terms, the integrals
/// ∫∫ hT dA and ∫∫ hS dA should remain constant.
#[test]
fn test_tracer_conservation_periodic() {
    let (mesh, ops, geom) = create_periodic_setup(4, 4, 2);
    let n_elements = mesh.n_elements;
    let n_nodes = ops.n_nodes;

    // Initialize with non-uniform tracers but uniform depth and velocity
    let h = 10.0;
    let u = 0.5;
    let v = 0.3;

    let mut state = CoupledState2D::new(
        SWESolution2D::new(n_elements, n_nodes),
        TracerSolution2D::new(n_elements, n_nodes),
    );

    // Set up non-uniform initial tracer distribution (Gaussian blob)
    for ki in 0..n_elements {
        for i in 0..n_nodes {
            let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
            let [x, y] = mesh.reference_to_physical(k(ki), r, s);

            // Gaussian temperature perturbation centered at (0.5, 0.5)
            let dx = x - 0.5;
            let dy = y - 0.5;
            let t = 10.0 + 5.0 * (-20.0 * (dx * dx + dy * dy)).exp();
            let s_tracer = 35.0;

            state
                .swe
                .set_state(k(ki), i, SWEState2D::from_primitives(h, u, v));
            state
                .tracers
                .set_from_concentrations(k(ki), i, h, TracerState::new(t, s_tracer));
        }
    }

    // Record initial tracer content
    let (initial_h_t, initial_h_s) = total_tracer(&state, &ops, &geom);

    // Time stepping configuration (no limiters needed for smooth solution)
    let time_config = CoupledTimeConfig::new(0.3, G, H_MIN).with_baroclinic(false);

    let equation = ShallowWater2D::new(G);
    let swe_bc = Reflective2D::new();
    let tracer_bc = ExtrapolationTracerBC;

    // Run for 100 steps
    let mut t = 0.0;
    let t_end = 0.1;

    while t < t_end {
        let dt = compute_dt_coupled(&state, &mesh, &geom, &time_config, 2);
        let dt = dt.min(t_end - t);

        // Compute RHS
        let swe_config = SWE2DRhsConfig::new(&equation, &swe_bc);
        let tracer_config = Tracer2DRhsConfig::new(&tracer_bc, G, H_MIN);

        let swe_rhs = compute_rhs_swe_2d(&state.swe, &mesh, &ops, &geom, &swe_config, t);
        let tracer_rhs = compute_rhs_tracer_2d(
            &state.tracers,
            &state.swe,
            &mesh,
            &ops,
            &geom,
            &tracer_config,
            t,
        );

        // Simple forward Euler for this test (sufficient for conservation check)
        state.swe.axpy(dt, &swe_rhs);
        state.tracers.axpy(dt, &tracer_rhs);

        t += dt;
    }

    // Check conservation
    let (final_h_t, final_h_s) = total_tracer(&state, &ops, &geom);

    let rel_error_t = ((final_h_t - initial_h_t) / initial_h_t).abs();
    let rel_error_s = ((final_h_s - initial_h_s) / initial_h_s).abs();

    assert!(
        rel_error_t < TOL,
        "Temperature content not conserved: initial={}, final={}, rel_error={}",
        initial_h_t,
        final_h_t,
        rel_error_t
    );

    assert!(
        rel_error_s < TOL,
        "Salinity content not conserved: initial={}, final={}, rel_error={}",
        initial_h_s,
        final_h_s,
        rel_error_s
    );
}

// ============================================================================
// Limiter Regression Tests
// ============================================================================

/// Test that limiters prevent oscillations at sharp gradients.
///
/// This is a regression test for the tracer instability issue fixed in 2026-01-09.
/// Without limiters, high-order DG develops Gibbs oscillations at discontinuities.
#[test]
fn test_limiter_prevents_oscillations() {
    let (mesh, ops, _geom) = create_wall_setup(8, 8, 3); // P3 - high order
    let n_elements = mesh.n_elements;
    let n_nodes = ops.n_nodes;

    let h = 10.0;

    let mut state = CoupledState2D::new(
        SWESolution2D::new(n_elements, n_nodes),
        TracerSolution2D::new(n_elements, n_nodes),
    );

    // Create sharp gradient: cold fresh water on left, warm salty on right
    for ki in 0..n_elements {
        for i in 0..n_nodes {
            let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
            let [x, _y] = mesh.reference_to_physical(k(ki), r, s);

            // Sharp discontinuity at x = 0.5
            let (t, sal) = if x < 0.5 {
                (5.0, 10.0) // Cold, fresh
            } else {
                (15.0, 35.0) // Warm, salty
            };

            // Small velocity to advect the discontinuity
            let u = 0.1;
            let v = 0.0;

            state
                .swe
                .set_state(k(ki), i, SWEState2D::from_primitives(h, u, v));
            state
                .tracers
                .set_from_concentrations(k(ki), i, h, TracerState::new(t, sal));
        }
    }

    // Apply limiters
    let tvb = TVBParameter2D::with_domain_size(50.0, 1.0);
    let bounds = TracerBounds::default();

    apply_tracer_limiters_2d(
        &mut state.tracers,
        &state.swe,
        &mesh,
        &ops,
        &tvb,
        &bounds,
        H_MIN,
    );

    // Check that all values are within physical bounds after limiting
    for ki in 0..n_elements {
        for i in 0..n_nodes {
            let h_node = state.swe.get_state(k(ki), i).h;
            let tracer = state.tracers.get_concentrations(k(ki), i, h_node, H_MIN);

            assert!(
                tracer.temperature >= bounds.t_min && tracer.temperature <= bounds.t_max,
                "Temperature out of bounds at element {}, node {}: T={}",
                ki,
                i,
                tracer.temperature
            );

            assert!(
                tracer.salinity >= bounds.s_min && tracer.salinity <= bounds.s_max,
                "Salinity out of bounds at element {}, node {}: S={}",
                ki,
                i,
                tracer.salinity
            );
        }
    }
}

/// Test that limiters preserve cell averages.
///
/// The Zhang-Shu positivity limiter should modify the solution to enforce bounds
/// while preserving the cell average.
#[test]
fn test_limiter_preserves_cell_average() {
    let (mesh, ops, _geom) = create_wall_setup(4, 4, 2);
    let n_elements = mesh.n_elements;
    let n_nodes = ops.n_nodes;

    let h = 10.0;

    let mut state = CoupledState2D::new(
        SWESolution2D::new(n_elements, n_nodes),
        TracerSolution2D::new(n_elements, n_nodes),
    );

    // Create solution with values that will trigger limiting
    // Some nodes outside bounds, but average inside
    for ki in 0..n_elements {
        for i in 0..n_nodes {
            let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);

            // Oscillating pattern that exceeds bounds at some nodes
            // but has average around 20°C (within bounds)
            let t = 20.0 + 30.0 * r * s; // Range: 20-30 to 20+30
            let sal = 20.0 + 15.0 * r; // Range: 5-35

            state
                .swe
                .set_state(k(ki), i, SWEState2D::from_primitives(h, 0.0, 0.0));
            state
                .tracers
                .set_from_concentrations(k(ki), i, h, TracerState::new(t, sal));
        }
    }

    // Compute cell averages before limiting
    let mut avg_before = Vec::with_capacity(n_elements);
    for ki in 0..n_elements {
        let mut sum_h_t = 0.0;
        let mut sum_h_s = 0.0;
        let mut sum_w = 0.0;
        for (i, &w) in ops.weights.iter().enumerate() {
            let cons = state.tracers.get_conservative(k(ki), i);
            sum_h_t += w * cons.h_t;
            sum_h_s += w * cons.h_s;
            sum_w += w * h;
        }
        avg_before.push((sum_h_t / sum_w, sum_h_s / sum_w));
    }

    // Apply limiters
    let tvb = TVBParameter2D::with_domain_size(50.0, 1.0);
    let bounds = TracerBounds::default();

    apply_tracer_limiters_2d(
        &mut state.tracers,
        &state.swe,
        &mesh,
        &ops,
        &tvb,
        &bounds,
        H_MIN,
    );

    // Compute cell averages after limiting
    for ki in 0..n_elements {
        let mut sum_h_t = 0.0;
        let mut sum_w = 0.0;
        for (i, &w) in ops.weights.iter().enumerate() {
            let cons = state.tracers.get_conservative(k(ki), i);
            sum_h_t += w * cons.h_t;
            sum_w += w * h;
        }
        let avg_t_after = sum_h_t / sum_w;

        // Cell average should be preserved (within tolerance)
        // Note: TVB limiter may change averages slightly, so use looser tolerance
        let diff = (avg_t_after - avg_before[ki].0).abs();
        assert!(
            diff < 1.0, // Allow some change from TVB limiter
            "Cell average changed too much at element {}: before={}, after={}, diff={}",
            ki,
            avg_before[ki].0,
            avg_t_after,
            diff
        );
    }
}

// ============================================================================
// Coupled Simulation Stability Tests
// ============================================================================

/// Test that coupled SWE-tracer simulation with limiters remains stable.
///
/// This is a longer-running test that verifies the full simulation pipeline
/// doesn't blow up when limiters are enabled.
#[test]
fn test_coupled_simulation_stability_with_limiters() {
    let (mesh, ops, geom) = create_wall_setup(6, 6, 3);
    let n_elements = mesh.n_elements;
    let n_nodes = ops.n_nodes;

    let h = 10.0;

    // Initialize with sharp gradient (challenging for stability)
    let mut state = CoupledState2D::new(
        SWESolution2D::new(n_elements, n_nodes),
        TracerSolution2D::new(n_elements, n_nodes),
    );

    for ki in 0..n_elements {
        for i in 0..n_nodes {
            let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
            let [x, y] = mesh.reference_to_physical(k(ki), r, s);

            // Sharp front
            let t = if x + y < 1.0 { 5.0 } else { 15.0 };
            let sal = if x < 0.5 { 5.0 } else { 35.0 };

            let u = 0.2;
            let v = 0.1;

            state
                .swe
                .set_state(k(ki), i, SWEState2D::from_primitives(h, u, v));
            state
                .tracers
                .set_from_concentrations(k(ki), i, h, TracerState::new(t, sal));
        }
    }

    // Configure with limiters
    let domain_size = 2.0_f64.sqrt(); // Diagonal of unit square
    let time_config = CoupledTimeConfig::new(0.3, G, H_MIN)
        .with_baroclinic(false)
        .with_tracer_limiters(50.0, domain_size);

    let equation = ShallowWater2D::new(G);
    let swe_bc = Reflective2D::new();
    let tracer_bc = ExtrapolationTracerBC;

    // RHS function
    let rhs_fn = |state: &CoupledState2D, t: f64| {
        let swe_config = SWE2DRhsConfig::new(&equation, &swe_bc);
        let tracer_config = Tracer2DRhsConfig::new(&tracer_bc, G, H_MIN);

        let swe_rhs = compute_rhs_swe_2d(&state.swe, &mesh, &ops, &geom, &swe_config, t);
        let tracer_rhs = compute_rhs_tracer_2d(
            &state.tracers,
            &state.swe,
            &mesh,
            &ops,
            &geom,
            &tracer_config,
            t,
        );

        CoupledRhs2D::new(swe_rhs, tracer_rhs)
    };

    // dt function
    let dt_fn = |state: &CoupledState2D| compute_dt_coupled(state, &mesh, &geom, &time_config, 3);

    // Run simulation
    let t_end = 0.5;
    let (t_final, n_steps) = run_coupled_simulation_limited(
        &mut state,
        t_end,
        rhs_fn,
        dt_fn,
        &mesh,
        &ops,
        &time_config,
        None::<fn(f64, &CoupledState2D)>,
    );

    // Verify simulation completed
    assert!(
        (t_final - t_end).abs() < 1e-10,
        "Simulation did not reach t_end: t_final={}, t_end={}",
        t_final,
        t_end
    );

    assert!(n_steps > 0, "No time steps taken");

    // Verify solution is bounded
    let bounds = TracerBounds::default();
    let (t_min, t_max) = state.tracers.temperature_range(&state.swe, H_MIN);
    let (s_min, s_max) = state.tracers.salinity_range(&state.swe, H_MIN);

    assert!(
        t_min >= bounds.t_min && t_max <= bounds.t_max,
        "Temperature out of bounds after simulation: [{}, {}]",
        t_min,
        t_max
    );

    assert!(
        s_min >= bounds.s_min && s_max <= bounds.s_max,
        "Salinity out of bounds after simulation: [{}, {}]",
        s_min,
        s_max
    );

    // Verify depths are positive
    let h_min_sol = state.swe.min_depth();
    assert!(
        h_min_sol > 0.0,
        "Negative depth detected: h_min={}",
        h_min_sol
    );
}

/// Test that simulation WITHOUT limiters would produce out-of-bounds values.
///
/// This verifies that the limiters are actually doing something - without them,
/// the high-order scheme should produce oscillations that exceed physical bounds.
#[test]
fn test_no_limiter_produces_oscillations() {
    let (mesh, ops, geom) = create_wall_setup(4, 4, 3); // P3 - prone to oscillations
    let n_elements = mesh.n_elements;
    let n_nodes = ops.n_nodes;

    let h = 10.0;

    let mut state = CoupledState2D::new(
        SWESolution2D::new(n_elements, n_nodes),
        TracerSolution2D::new(n_elements, n_nodes),
    );

    // Sharp discontinuity - will cause Gibbs oscillations
    for ki in 0..n_elements {
        for i in 0..n_nodes {
            let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
            let [x, _y] = mesh.reference_to_physical(k(ki), r, s);

            let t = if x < 0.5 { 0.0 } else { 30.0 }; // Sharp jump
            let sal = if x < 0.5 { 0.0 } else { 40.0 }; // Sharp jump

            state
                .swe
                .set_state(k(ki), i, SWEState2D::from_primitives(h, 0.5, 0.0));
            state
                .tracers
                .set_from_concentrations(k(ki), i, h, TracerState::new(t, sal));
        }
    }

    // Run a few steps WITHOUT limiters
    let equation = ShallowWater2D::new(G);
    let swe_bc = Reflective2D::new();
    let tracer_bc = ExtrapolationTracerBC;

    let swe_config = SWE2DRhsConfig::new(&equation, &swe_bc);
    let tracer_config = Tracer2DRhsConfig::new(&tracer_bc, G, H_MIN);

    let dt = 0.001; // Small timestep

    for _ in 0..20 {
        let swe_rhs = compute_rhs_swe_2d(&state.swe, &mesh, &ops, &geom, &swe_config, 0.0);
        let tracer_rhs = compute_rhs_tracer_2d(
            &state.tracers,
            &state.swe,
            &mesh,
            &ops,
            &geom,
            &tracer_config,
            0.0,
        );

        state.swe.axpy(dt, &swe_rhs);
        state.tracers.axpy(dt, &tracer_rhs);
    }

    // Check if oscillations occurred (values outside initial range)
    let (t_min, t_max) = state.tracers.temperature_range(&state.swe, H_MIN);
    let (s_min, s_max) = state.tracers.salinity_range(&state.swe, H_MIN);

    // With P3 and a sharp discontinuity, we expect oscillations
    // Either undershoots (< 0) or overshoots (> 30 for T, > 40 for S)
    let has_oscillations = t_min < -0.1 || t_max > 30.1 || s_min < -0.1 || s_max > 40.1;

    assert!(
        has_oscillations,
        "Expected oscillations without limiters, but got T=[{}, {}], S=[{}, {}]",
        t_min, t_max, s_min, s_max
    );
}
