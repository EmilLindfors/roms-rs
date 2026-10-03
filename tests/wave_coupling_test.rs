//! Gates of the coupling from the spectral wave model to the circulation (TODO
//! F.4): the radiation-stress force of `WaveForce2D` and the wave-enhanced bed
//! friction of `WaveCurrentFriction2D` on the 2D shallow-water model, through
//! the production path (`Simulation` + `SSPRK3` + `SWEPhysics2D`).

use std::sync::Arc;

use dg_rs::boundary::Reflective2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::PhysicsBuilder;
use dg_rs::simulation::Simulation;
use dg_rs::solver::{SWESolution2D, SWEState2D};
use dg_rs::source::{
    ChezyFriction2D, SourceContext2D, SourceTerm2D, WaveCurrentFriction2D, WaveForce2D,
};
use dg_rs::time::SSPRK3;
use dg_rs::types::ElementIndex;
use dg_rs::waves::{SpectralGrid, WaveModel2D, WaveWorkspace, wavenumber};

const G: f64 = 9.81;

/// Linear damping of the momentum, `−r (hu, hv)`, to settle the basin.
struct LinearDamping(f64);

impl SourceTerm2D for LinearDamping {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        SWEState2D::new(0.0, -self.0 * ctx.state.hu, -self.0 * ctx.state.hv)
    }

    fn name(&self) -> &'static str {
        "linear_damping"
    }
}

/// Swell of 10 s and 1 m normal to the shore over a closed basin shoaling from
/// 20 to 3 m (no breaking): seaward of the breakers the mean level sets down as
/// `η = −E k / sinh 2kh` (Longuet-Higgins & Stewart 1962; E = H²/8, the local
/// variance), the balance of the radiation-stress gradient and the pressure
/// gradient, `g h ∂η/∂x = −g ∂S_xx/∂x`. The wave model shoals the swell, its
/// force drives the basin to rest, and the set-down follows the formula to
/// 0.27 % of its 3.3 cm range across the basin. A current of 2e-5 m/s remains,
/// held by the damping against the part of the element-local force that no
/// level can balance (the jumps of S between elements are left out).
#[test]
fn shoaling_waves_set_the_water_down() {
    const L: f64 = 2000.0;
    let depth = |x: f64| 20.0 - 17.0 * x / L;
    let order = 2;
    let mesh = Mesh2D::uniform_rectangle_with_sides(
        0.0,
        L,
        0.0,
        100.0,
        40,
        1,
        [
            BoundaryTag::Wall,
            BoundaryTag::Open,
            BoundaryTag::Wall,
            BoundaryTag::Open,
        ],
    );
    let ops = Arc::new(DGOperators2D::new(order));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _| {
        -depth(x)
    }));
    let mesh = Arc::new(mesh);

    // The steady swell: one component, 10 s along +x, H = 1 m offshore
    let grid = SpectralGrid::new(0.1, 0.12, 2, 36);
    let mut e = vec![0.0; grid.n_components()];
    let c = grid.component(0, 0);
    e[c] = 1.0 / 8.0 / (grid.d_sigma[0] * grid.d_theta);
    let waves = WaveModel2D::new(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        &bathymetry,
        grid,
        G,
    )
    .with_boundary_spectrum(&e);
    let mut n = waves.zero_state();
    let mut ws = WaveWorkspace::default();
    let dt = waves.compute_dt(0.5);
    let steps = (3.0 * L / (3.0 * dt)).ceil() as usize; // c_g ≥ 3 m/s
    for s in 0..steps {
        waves.step(&mut n, s as f64 * dt, dt, &mut ws);
    }

    // The basin at rest under the waves' force
    let force = WaveForce2D::new(&waves, &n).with_ramp(1200.0);
    let physics = PhysicsBuilder::swe_2d(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        ShallowWater2D::new(G),
        Reflective2D::new(),
    )
    .with_bathymetry(bathymetry.clone())
    .with_source(force)
    .with_source(LinearDamping(2e-3))
    .build();
    let nn = ops.n_nodes;
    let mut q = SWESolution2D::new(mesh.n_elements, nn);
    for k in ElementIndex::iter(mesh.n_elements) {
        for i in 0..nn {
            q.set_state(k, i, SWEState2D::new(-bathymetry.get(k, i), 0.0, 0.0));
        }
    }
    let result = Simulation::new(physics, SSPRK3).run(&mut q, 0.0, 8000.0);
    assert!(result.success, "{result:?}");

    // η against the formula, both relative to the offshore end
    let params = waves.parameters(&n);
    let mut samples: Vec<(f64, f64, f64)> = Vec::new(); // (x, η, formula)
    for k in ElementIndex::iter(mesh.n_elements) {
        for i in 0..nn {
            let p = k.as_usize() * nn + i;
            let [x, _] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
            let h = depth(x);
            let eta = q.h_data()[p] + bathymetry.get(k, i);
            let kw = wavenumber(waves.grid.sigma[0], h, G);
            let formula = -params[p].m0 * kw / (2.0 * kw * h).sinh();
            samples.push((x, eta, formula));
        }
    }
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (_, eta0, formula0) = samples[0];
    let range = samples
        .iter()
        .map(|s| s.2 - formula0)
        .fold(0.0f64, |a, b| a.max(b.abs()));
    let worst = samples
        .iter()
        .map(|s| ((s.1 - eta0) - (s.2 - formula0)).abs())
        .fold(0.0f64, f64::max);
    let speed = q
        .hu_data()
        .iter()
        .zip(q.h_data())
        .map(|(hu, h)| (hu / h).abs())
        .fold(0.0f64, f64::max);
    println!(
        "set-down across the basin {:.2} cm; largest departure {:.2e} m ({:.2} %); |u| ≤ {speed:.1e} m/s",
        100.0 * range,
        worst,
        100.0 * worst / range
    );
    assert!(range > 0.01, "a set-down of {range} m");
    assert!(speed < 5e-5, "not at rest: {speed} m/s");
    assert!(worst < 0.01 * range, "η off the formula by {worst:e} m");
}

/// A uniform body force `G` per unit mass along x.
struct BodyForce(f64);

impl SourceTerm2D for BodyForce {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        SWEState2D::new(0.0, ctx.state.h * self.0, 0.0)
    }

    fn name(&self) -> &'static str {
        "body_force"
    }
}

/// A current driven along a periodic channel 10 m deep by a body force, under
/// a uniform sea whose bed stress is that of a 12.5 s wave of 1 m (from the
/// wave model, on a 2 mm roughness). Steady at
/// `G h = C_d u² [1 + 1.2 (τ_w/(C_d u² + τ_w))^3.2]`: the waves slow the
/// current by enhancing its friction, and the model holds the root of that
/// balance to 1e-7 (the point-implicit friction, applied through
/// `with_implicit_friction`): 0.63 m/s without the waves, 0.46 m/s under
/// their 6 Pa.
#[test]
fn waves_slow_a_current_by_soulsbys_enhanced_friction() {
    let (h, forcing, cd, z0) = (10.0, 1e-4, 2.5e-3, 0.002);
    let mesh = Mesh2D::uniform_periodic(0.0, 2000.0, 0.0, 2000.0, 2, 2);
    let ops = Arc::new(DGOperators2D::new(2));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let bathymetry = Arc::new(Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -h));
    let mesh = Arc::new(mesh);
    // The waves' bed stress: one component, 1 m, along y
    let grid = SpectralGrid::new(0.05, 0.3, 20, 12);
    let (i, j) = (5, 3);
    let mut e = vec![0.0; grid.n_components()];
    e[grid.component(i, j)] = 1.0 / 8.0 / (grid.d_sigma[i] * grid.d_theta);
    let waves = WaveModel2D::new(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        &bathymetry,
        grid,
        G,
    );
    let tau_w = waves.bed_wave_stress(&waves.uniform_state(&e), z0);
    let tw = tau_w[0];
    assert!(tau_w.iter().all(|&t| (t - tw).abs() < 1e-15 * tw));

    let speed = |tau_w: f64| {
        let law =
            WaveCurrentFriction2D::new(ChezyFriction2D::new(cd), vec![tau_w; waves.n_points()]);
        let physics = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::new(),
        )
        .with_bathymetry(bathymetry.clone())
        .with_source(BodyForce(forcing))
        .with_implicit_friction(law)
        .build();
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }
        let result = Simulation::new(physics, SSPRK3).run(&mut q, 0.0, 60_000.0);
        assert!(result.success, "{result:?}");
        q.hu_data()[0] / q.h_data()[0]
    };
    // The root of G h = C_d u² E(C_d u², τ_w), by bisection
    let exact = |tau_w: f64| {
        let residual = |u: f64| {
            let tau_c = cd * u * u;
            tau_c * WaveCurrentFriction2D::<ChezyFriction2D>::enhancement(tau_c, tau_w)
                - forcing * h
        };
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if residual(mid) > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        0.5 * (lo + hi)
    };
    let (calm, rough) = (speed(0.0), speed(tw));
    println!(
        "τ_w {:.2} Pa: u {calm:.4} m/s without waves, {rough:.4} m/s under them (exact {:.4}, {:.4})",
        1025.0 * tw,
        exact(0.0),
        exact(tw)
    );
    assert!((calm / exact(0.0) - 1.0).abs() < 1e-7, "calm: {calm}");
    assert!(
        (rough / exact(tw) - 1.0).abs() < 1e-9,
        "under waves: {rough}"
    );
    assert!(rough < 0.75 * calm, "the waves barely slowed the current");
}
