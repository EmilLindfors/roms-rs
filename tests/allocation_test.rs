//! Allocation regression test for the production 2D stepping path (TODO P1.1).
//!
//! `Simulation` + `SSPRK3` + `SWEPhysics2D` used to allocate two stage copies
//! and three RHS arrays per step (plus the kernel scratch). With the stage
//! workspace and the `_into` RHS a warm step should allocate (next to)
//! nothing. Counted with a wrapping global allocator, so this binary cannot
//! run with the `mimalloc` feature, which installs its own.
#![cfg(not(feature = "mimalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dg_rs::boundary::Reflective2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::{Bathymetry2D, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::PhysicsBuilder;
use dg_rs::simulation::Simulation;
use dg_rs::solver::{SWESolution2D, SWEState2D, WetDryConfig};
use dg_rs::source::{BathymetrySource2D, ManningFriction2D};
use dg_rs::time::SSPRK3;
use dg_rs::types::ElementIndex;

struct CountingAllocator;

static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATED_BYTES.fetch_add(new_size.saturating_sub(layout.size()), Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

#[test]
fn simulation_steps_do_not_allocate_state_sized_buffers() {
    const G: f64 = 9.81;
    let mesh = Arc::new(Mesh2D::uniform_rectangle(
        0.0, 20_000.0, 0.0, 20_000.0, 24, 24,
    ));
    let ops = Arc::new(DGOperators2D::new(3));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh));
    let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
        -100.0 + 40.0 * (x / 20_000.0) - 20.0 * (y / 20_000.0).powi(2)
    }));

    let mut state = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    for k in ElementIndex::iter(mesh.n_elements) {
        for i in 0..ops.n_nodes {
            let [x, _] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
            let h = 0.2 * (x / 5_000.0).sin() - bathymetry.get(k, i);
            state.set_state(k, i, SWEState2D::from_primitives(h, 0.1, -0.05));
        }
    }
    let state_bytes = 3 * mesh.n_elements * ops.n_nodes * std::mem::size_of::<f64>();

    let physics = PhysicsBuilder::swe_2d(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        ShallowWater2D::new(G),
        Reflective2D::new(),
    )
    .with_bathymetry(bathymetry)
    .with_well_balanced(true)
    .with_source(BathymetrySource2D::new(G))
    // Wet/dry correction and implicit friction run every stage (P1.2); the
    // positivity limiter is left out: it still allocates cell averages (P2.3)
    .with_wet_dry(WetDryConfig::default())
    .with_implicit_friction(ManningFriction2D::new(G, 0.025))
    .build();

    // Fixed dt (well below the CFL limit) so both runs take a known step count
    let dt = 0.5;
    let sim = Simulation::new(physics, SSPRK3)
        .with_cfl(0.5)
        .with_dt_max(dt);
    let bytes_for = |state: &mut SWESolution2D, n_steps: usize| {
        let before = ALLOCATED_BYTES.load(Ordering::Relaxed);
        let result = sim.run(state, 0.0, n_steps as f64 * dt);
        assert!(result.success && result.n_steps == n_steps, "{result:?}");
        ALLOCATED_BYTES.load(Ordering::Relaxed) - before
    };

    // Each run warms its own stage workspace, so the difference between a
    // 20- and a 10-step run is what 10 warm steps allocate.
    let short = bytes_for(&mut state, 10);
    let long = bytes_for(&mut state, 20);
    let per_step = long.saturating_sub(short) as f64 / 10.0;
    println!("state {state_bytes} B; per warm step {per_step:.0} B (runs: {short} B, {long} B)");
    // Before: ≥ 5 state-sized arrays per step (≈ 5 × 110 kB here)
    assert!(
        per_step < 0.01 * state_bytes as f64,
        "a warm step allocated {per_step:.0} B (state is {state_bytes} B)"
    );
}
