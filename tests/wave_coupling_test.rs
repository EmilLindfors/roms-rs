//! Gates of the coupling from the spectral wave model to the circulation (TODO
//! F.4): the radiation-stress force of `WaveForce2D` on the 2D shallow-water
//! model, through the production path (`Simulation` + `SSPRK3` +
//! `SWEPhysics2D`).

use std::sync::Arc;

use dg_rs::boundary::Reflective2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::PhysicsBuilder;
use dg_rs::simulation::Simulation;
use dg_rs::solver::{SWESolution2D, SWEState2D};
use dg_rs::source::{SourceContext2D, SourceTerm2D, WaveForce2D};
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
