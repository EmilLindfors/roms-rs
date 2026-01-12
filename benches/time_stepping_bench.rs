//! Benchmarks for time stepping and diagnostics.
//!
//! Run with: `cargo bench --bench time_stepping_bench`
//!
//! Benchmarks SSP-RK3 time integration and diagnostic computations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dg_rs::boundary::Reflective2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::Mesh2D;
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::solver::{
    compute_dt_swe_2d, compute_rhs_swe_2d, DiagnosticsTracker, SWE2DRhsConfig, SWEDiagnostics2D,
    SWEState2D, SWESolution2D,
};
use dg_rs::time::{ssp_rk3_swe_2d_step_limited, SWE2DTimeConfig};

/// Setup a test problem.
fn setup_problem(
    nx: usize,
    ny: usize,
    order: usize,
) -> (
    Mesh2D,
    DGOperators2D,
    GeometricFactors2D,
    SWESolution2D,
    ShallowWater2D,
) {
    let mesh = Mesh2D::uniform_rectangle(0.0, 1000.0, 0.0, 1000.0, nx, ny);
    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh);
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);

    // Initialize with uniform flow
    let h0 = 10.0;
    let u0 = 0.5;
    let v0 = 0.3;
    for k in 0..q.n_elements {
        for i in 0..ops.n_nodes {
            q.set_state(k, i, SWEState2D::new(h0, h0 * u0, h0 * v0));
        }
    }

    let equation = ShallowWater2D::new(9.81);

    (mesh, ops, geom, q, equation)
}

/// Benchmark CFL-based time step computation.
fn bench_compute_dt(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_dt");

    for (nx, ny) in [(8, 8), (16, 16), (32, 32)] {
        let n_elements = nx * ny;
        let (mesh, ops, geom, q, equation) = setup_problem(nx, ny, 3);

        group.bench_with_input(
            BenchmarkId::new("swe_2d", format!("{}_elements", n_elements)),
            &n_elements,
            |b, _| {
                b.iter(|| {
                    compute_dt_swe_2d(
                        black_box(&q),
                        black_box(&mesh),
                        black_box(&geom),
                        black_box(&equation),
                        black_box(ops.order),
                        black_box(0.5),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark single SSP-RK3 step with limiters.
fn bench_ssp_rk3_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssp_rk3_step");
    group.sample_size(30);

    for (nx, ny) in [(8, 8), (16, 16)] {
        let n_elements = nx * ny;
        let (mesh, ops, geom, q, equation) = setup_problem(nx, ny, 3);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc);
        let time_config = SWE2DTimeConfig::new(0.5, 9.81, 1e-6);
        let dt = 0.1;
        let q_backup = q.clone();

        group.bench_with_input(
            BenchmarkId::new("step", format!("{}_elements", n_elements)),
            &n_elements,
            |b, _| {
                let mut q_work = q_backup.clone();
                b.iter(|| {
                    q_work = q_backup.clone();
                    let rhs_fn = |s: &SWESolution2D, time: f64| {
                        compute_rhs_swe_2d(s, &mesh, &ops, &geom, &config, time)
                    };
                    ssp_rk3_swe_2d_step_limited(
                        black_box(&mut q_work),
                        black_box(dt),
                        black_box(0.0),
                        black_box(&mesh),
                        black_box(&ops),
                        rhs_fn,
                        black_box(&time_config),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark multiple time steps (short simulation).
fn bench_multiple_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiple_steps");
    group.sample_size(20);

    let (nx, ny) = (10, 10);
    let (mesh, ops, geom, q, equation) = setup_problem(nx, ny, 3);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc);
    let time_config = SWE2DTimeConfig::new(0.5, 9.81, 1e-6);
    let dt = 0.1;
    let q_backup = q.clone();

    for n_steps in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("steps", n_steps.to_string()),
            &n_steps,
            |b, &n_steps| {
                let mut q_work = q_backup.clone();
                b.iter(|| {
                    q_work = q_backup.clone();
                    let mut t = 0.0;
                    for _ in 0..n_steps {
                        let rhs_fn = |s: &SWESolution2D, time: f64| {
                            compute_rhs_swe_2d(s, &mesh, &ops, &geom, &config, time)
                        };
                        ssp_rk3_swe_2d_step_limited(
                            black_box(&mut q_work),
                            black_box(dt),
                            black_box(t),
                            black_box(&mesh),
                            black_box(&ops),
                            rhs_fn,
                            black_box(&time_config),
                        );
                        t += dt;
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark diagnostics computation.
fn bench_diagnostics(c: &mut Criterion) {
    let mut group = c.benchmark_group("diagnostics");

    for (nx, ny) in [(8, 8), (16, 16), (32, 32)] {
        let n_elements = nx * ny;
        let (mesh, ops, geom, q, _) = setup_problem(nx, ny, 3);
        let g = 9.81;
        let dt = 0.1;

        group.bench_with_input(
            BenchmarkId::new("compute", format!("{}_elements", n_elements)),
            &n_elements,
            |b, _| {
                b.iter(|| {
                    SWEDiagnostics2D::compute(
                        black_box(&q),
                        black_box(&mesh),
                        black_box(&ops),
                        black_box(&geom),
                        black_box(g),
                        black_box(dt),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark diagnostics tracker update.
fn bench_diagnostics_tracker(c: &mut Criterion) {
    let mut group = c.benchmark_group("diagnostics_tracker");

    let (mesh, ops, geom, q, _) = setup_problem(16, 16, 3);
    let g = 9.81;
    let dt = 0.1;

    let initial = SWEDiagnostics2D::compute(&q, &mesh, &ops, &geom, g, dt);
    let mut tracker = DiagnosticsTracker::new(initial.clone());

    // Pre-compute some diagnostics to update with
    let diags: Vec<_> = (0..100)
        .map(|i| {
            // Slightly modify the diagnostics
            let mut d = initial.clone();
            d.total_mass *= 1.0 + (i as f64) * 1e-6;
            d.total_energy *= 1.0 - (i as f64) * 1e-7;
            d
        })
        .collect();

    group.bench_function("update", |b| {
        let mut idx = 0;
        b.iter(|| {
            let diag = &diags[idx % diags.len()];
            tracker.update(black_box(idx as f64), black_box(diag.clone()));
            idx += 1;
        });
    });

    group.bench_function("is_stable", |b| {
        b.iter(|| tracker.is_stable());
    });

    group.bench_function("mass_error", |b| {
        b.iter(|| tracker.mass_error());
    });

    group.finish();
}

/// Benchmark RHS computation as baseline for time stepping.
fn bench_rhs_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("rhs_baseline");
    group.sample_size(50);

    let (nx, ny) = (16, 16);
    let (mesh, ops, geom, q, equation) = setup_problem(nx, ny, 3);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc);

    group.bench_function("single_rhs", |b| {
        b.iter(|| {
            compute_rhs_swe_2d(
                black_box(&q),
                black_box(&mesh),
                black_box(&ops),
                black_box(&geom),
                black_box(&config),
                black_box(0.0),
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_compute_dt,
    bench_ssp_rk3_step,
    bench_multiple_steps,
    bench_diagnostics,
    bench_diagnostics_tracker,
    bench_rhs_baseline
);
criterion_main!(benches);
