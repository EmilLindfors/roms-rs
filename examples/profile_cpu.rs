//! CPU profiling benchmark - for comparison with GPU
//!
//! Run with:
//! ```bash
//! cargo run --release --features parallel --example profile_cpu
//! ```

use std::time::Instant;

use dg_rs::boundary::{MultiBoundaryCondition2D, Reflective2D};
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::{BoundaryTag, Mesh2D, Bathymetry2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::solver::{SWESolution2D, SWE2DRhsConfig};
use dg_rs::time::{ssp_rk3_swe_2d_step_limited, SWE2DTimeConfig};
use dg_rs::types::ElementIndex;
use dg_rs::SWEState2D;
use dg_rs::SWEFluxType2D;

// Use the best available RHS function based on enabled features
#[cfg(all(feature = "parallel", feature = "simd"))]
use dg_rs::solver::compute_rhs_swe_2d_parallel_simd;
#[cfg(all(feature = "parallel", not(feature = "simd")))]
use dg_rs::solver::compute_rhs_swe_2d_parallel;
#[cfg(all(feature = "simd", not(feature = "parallel")))]
use dg_rs::solver::compute_rhs_swe_2d_simd;
#[cfg(not(any(feature = "parallel", feature = "simd")))]
use dg_rs::solver::compute_rhs_swe_2d;

#[cfg(feature = "parallel")]
use dg_rs::solver::compute_dt_swe_2d_parallel;
#[cfg(not(feature = "parallel"))]
use dg_rs::solver::compute_dt_swe_2d;

fn k(idx: usize) -> ElementIndex {
    ElementIndex::new(idx)
}

const G: f64 = 9.81;
const H_MIN: f64 = 1.0;

fn main() {
    // Same mesh as GPU profiling
    let order = 3;
    let nx = 100;
    let ny = 50;

    #[cfg(all(feature = "parallel", feature = "simd"))]
    let backend = "CPU (parallel+simd)";
    #[cfg(all(feature = "parallel", not(feature = "simd")))]
    let backend = "CPU (parallel only)";
    #[cfg(all(feature = "simd", not(feature = "parallel")))]
    let backend = "CPU (simd only)";
    #[cfg(not(any(feature = "parallel", feature = "simd")))]
    let backend = "CPU (serial)";

    println!("=== CPU Profiling ({}) ===", backend);
    println!("Mesh: {}x{} = {} elements, P{}", nx, ny, nx * ny, order);

    let mesh = Mesh2D::uniform_rectangle_with_bc(
        0.0, 10000.0, 0.0, 5000.0,
        nx, ny, BoundaryTag::Wall,
    );

    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh);
    let n_elements = mesh.n_elements;
    let n_nodes = ops.n_nodes;

    // Initialize state
    let mut state = SWESolution2D::new(n_elements, n_nodes);
    for ki in 0..n_elements {
        for i in 0..n_nodes {
            state.set_state(k(ki), i, SWEState2D::from_primitives(20.0, 0.0, 0.0));
        }
    }

    let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |_, _| -20.0);
    let wall_bc = Reflective2D::new();
    let bc = MultiBoundaryCondition2D::new(&wall_bc);
    let equation = ShallowWater2D::new(G);
    let time_config = SWE2DTimeConfig::new(0.2, G, H_MIN)
        .with_kuzmin_limiters(1.0)
        .with_wet_dry_treatment();

    println!("\n--- compute_dt timing ---");
    for i in 0..3 {
        let t0 = Instant::now();
        #[cfg(feature = "parallel")]
        let dt = compute_dt_swe_2d_parallel(&state, &mesh, &geom, &equation, ops.order, 0.2);
        #[cfg(not(feature = "parallel"))]
        let dt = compute_dt_swe_2d(&state, &mesh, &geom, &equation, ops.order, 0.2);
        println!("  dt computation {}: {:?} (dt={:.6})", i, t0.elapsed(), dt);
    }

    println!("\n--- RHS computation timing ---");
    for i in 0..5 {
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_flux_type(SWEFluxType2D::Rusanov)
            .with_bathymetry(&bathymetry)
            .with_well_balanced(true);

        let t0 = Instant::now();
        #[cfg(all(feature = "parallel", feature = "simd"))]
        let _rhs = compute_rhs_swe_2d_parallel_simd(&state, &mesh, &ops, &geom, &config, 0.0);
        #[cfg(all(feature = "parallel", not(feature = "simd")))]
        let _rhs = compute_rhs_swe_2d_parallel(&state, &mesh, &ops, &geom, &config, 0.0);
        #[cfg(all(feature = "simd", not(feature = "parallel")))]
        let _rhs = compute_rhs_swe_2d_simd(&state, &mesh, &ops, &geom, &config, 0.0);
        #[cfg(not(any(feature = "parallel", feature = "simd")))]
        let _rhs = compute_rhs_swe_2d(&state, &mesh, &ops, &geom, &config, 0.0);
        println!("  RHS {}: {:?}", i, t0.elapsed());
    }

    println!("\n--- Full RK3 step timing ---");
    let mut t = 0.0;
    for i in 0..5 {
        #[cfg(feature = "parallel")]
        let dt = compute_dt_swe_2d_parallel(&state, &mesh, &geom, &equation, ops.order, 0.2);
        #[cfg(not(feature = "parallel"))]
        let dt = compute_dt_swe_2d(&state, &mesh, &geom, &equation, ops.order, 0.2);

        let rhs_fn = |s: &SWESolution2D, time: f64| {
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_flux_type(SWEFluxType2D::Rusanov)
                .with_bathymetry(&bathymetry)
                .with_well_balanced(true);

            #[cfg(all(feature = "parallel", feature = "simd"))]
            { compute_rhs_swe_2d_parallel_simd(s, &mesh, &ops, &geom, &config, time) }
            #[cfg(all(feature = "parallel", not(feature = "simd")))]
            { compute_rhs_swe_2d_parallel(s, &mesh, &ops, &geom, &config, time) }
            #[cfg(all(feature = "simd", not(feature = "parallel")))]
            { compute_rhs_swe_2d_simd(s, &mesh, &ops, &geom, &config, time) }
            #[cfg(not(any(feature = "parallel", feature = "simd")))]
            { compute_rhs_swe_2d(s, &mesh, &ops, &geom, &config, time) }
        };

        let t0 = Instant::now();
        ssp_rk3_swe_2d_step_limited(&mut state, dt, t, &mesh, &ops, rhs_fn, &time_config);
        println!("  RK3 step {}: {:?} (dt={:.6})", i, t0.elapsed(), dt);
        t += dt;
    }

    // Verify result
    let h = state.get_state(k(0), 0).h;
    println!("\nResult check: h[0,0] = {:.6}", h);
}
