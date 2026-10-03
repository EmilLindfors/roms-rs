//! Gates of the spectral wave model (`dg_rs::waves`, TODO F.4): the DG
//! propagation converges at its order, the propagation with refraction keeps the
//! total action, and the steady states of shoaling, refraction on a sloping shelf
//! (Snell's law) and a following current (conservation of the absolute frequency)
//! are the analytic ones (stage 1). With the full source terms (stage 2: the DIA,
//! the diagnostic tail and the limiter), a wind sea grows with fetch as Kahma &
//! Calkoen (1992) observed, and with duration towards Pierson–Moskowitz.

use std::f64::consts::{PI, TAU};
use std::sync::Arc;

use dg_rs::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::types::ElementIndex;
use dg_rs::waves::{
    SourceTerms, SpectralGrid, WaveModel2D, WaveSolution, WaveWorkspace, Wind, group_velocity,
    wavenumber,
};

const G: f64 = 9.81;

/// A model on `mesh` at order `order` over the bed `bed(x, y)` (negative under water).
fn model(
    mesh: Mesh2D,
    order: usize,
    bed: impl Fn(f64, f64) -> f64,
    grid: SpectralGrid,
) -> WaveModel2D {
    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, bed);
    WaveModel2D::new(
        Arc::new(mesh),
        Arc::new(ops),
        Arc::new(geom),
        &bathymetry,
        grid,
        G,
    )
}

/// Mesh coordinates of every node.
fn nodes(model: &WaveModel2D) -> Vec<[f64; 2]> {
    let (mesh, ops) = (&model.mesh, &model.ops);
    ElementIndex::iter(mesh.n_elements)
        .flat_map(|k| {
            (0..ops.n_nodes)
                .map(move |i| mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]))
        })
        .collect()
}

/// Run from `t = 0` to `t_end` at Courant number `cfl`, landing on `t_end`.
fn run(model: &WaveModel2D, n: &mut WaveSolution, t_end: f64, cfl: f64) {
    let mut ws = WaveWorkspace::default();
    let dt_max = model.compute_dt(cfl);
    let steps = (t_end / dt_max).ceil() as usize;
    let dt = t_end / steps as f64;
    for s in 0..steps {
        model.step(n, s as f64 * dt, dt, &mut ws);
    }
}

/// L2 error of component `c` against `exact(x, y)`, by the GLL quadrature.
fn l2_error(
    model: &WaveModel2D,
    n: &WaveSolution,
    c: usize,
    exact: impl Fn(f64, f64) -> f64,
) -> f64 {
    let (ops, geom) = (&model.ops, &model.geom);
    let xy = nodes(model);
    let field = n.component(c);
    let mut sum = 0.0;
    for k in 0..model.mesh.n_elements {
        for a in 0..ops.n_nodes {
            let p = k * ops.n_nodes + a;
            let e = field[p] - exact(xy[p][0], xy[p][1]);
            sum += ops.weights[a] * geom.jacobian(k, a) * e * e;
        }
    }
    sum.sqrt()
}

/// One component of a uniform deep sea (1000 m) carries a smooth periodic field
/// along x and along the diagonal at its group velocity; the L2 error after a
/// quarter period falls at the DG order + 1.
#[test]
fn propagation_converges_at_the_dg_order() {
    const L: f64 = 1000.0;
    for (j, label) in [(0, "along x"), (1, "diagonal")] {
        let mut errors = Vec::new();
        for n_el in [4, 8, 16] {
            let grid = SpectralGrid::new(0.08, 0.2, 2, 8);
            let m = model(
                Mesh2D::uniform_periodic(0.0, L, 0.0, L, n_el, n_el),
                2,
                |_, _| -1000.0,
                grid,
            );
            let sigma = m.grid.sigma[0];
            let cg = group_velocity(sigma, wavenumber(sigma, 1000.0, G), 1000.0);
            let theta = m.grid.theta[j];
            // A field periodic on the square, constant across the direction of travel
            let (a, b) = if j == 0 { (1.0, 0.0) } else { (1.0, 1.0) };
            let speed = cg * (a * theta.cos() + b * theta.sin());
            let f =
                |x: f64, y: f64, t: f64| 1.0 + 0.5 * (TAU * (a * x + b * y - speed * t) / L).sin();
            let c = m.grid.component(0, j);
            let mut n = m.zero_state();
            let xy = nodes(&m);
            for (p, x) in n.component_mut(c).iter_mut().enumerate() {
                *x = f(xy[p][0], xy[p][1], 0.0);
            }
            let t_end = 0.25 * L / cg;
            run(&m, &mut n, t_end, 0.3);
            errors.push(l2_error(&m, &n, c, |x, y| f(x, y, t_end)));
        }
        let rates: Vec<f64> = errors.windows(2).map(|e| (e[0] / e[1]).log2()).collect();
        println!("{label}: errors {errors:?}, rates {rates:?}");
        assert!(rates[1] > 2.7, "{label}: P2 converges at {rates:?}");
    }
}

/// Refraction over a periodic shoal moves energy between directions and places,
/// but the total action `Σ Δσ Δθ ∫ N dA` stays to round-off.
#[test]
fn refraction_keeps_the_total_action() {
    const L: f64 = 2000.0;
    let grid = SpectralGrid::new(0.06, 0.4, 10, 24);
    let m = model(
        Mesh2D::uniform_periodic(0.0, L, 0.0, L, 6, 6),
        2,
        |x, y| -(10.0 + 5.0 * (TAU * x / L).sin() * (TAU * y / L).cos()),
        grid,
    );
    let e = m.grid.jonswap(1.0, 6.0, 3.3, PI / 6.0, 2.0);
    let mut n = m.uniform_state(&e);
    let initial = n.clone();
    let total = m.total_action(&n);
    run(&m, &mut n, 300.0, 0.5);
    let change = (m.total_action(&n) / total - 1.0).abs();
    let moved = n
        .data
        .iter()
        .zip(&initial.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("relative change of the total action {change:.2e}, largest change of N {moved:.3e}");
    assert!(moved > 1e-3 * initial.data.iter().fold(0.0f64, |a, &b| a.max(b)));
    assert!(change < 1e-12, "total action changed by {change:e}");
}

/// Normal incidence on a slope from 20 to 3 m: in the steady state the action
/// flux `c_g N` is the same everywhere (Green's law for a spectrum without
/// refraction or sources).
#[test]
fn shoaling_keeps_the_action_flux() {
    const L: f64 = 2000.0;
    let depth = |x: f64| 20.0 - 17.0 * x / L;
    let grid = SpectralGrid::new(0.1, 0.15, 3, 36);
    let i = 1;
    let mut e = vec![0.0; grid.n_components()];
    e[grid.component(i, 0)] = 1.0;
    let m = model(
        Mesh2D::uniform_rectangle_with_sides(
            0.0,
            L,
            0.0,
            200.0,
            40,
            2,
            [
                BoundaryTag::Wall,
                BoundaryTag::Open,
                BoundaryTag::Wall,
                BoundaryTag::Open,
            ],
        ),
        2,
        |x, _| -depth(x),
        grid,
    )
    .with_boundary_spectrum(&e);
    let sigma = m.grid.sigma[i];
    let cg_at = |d: f64| group_velocity(sigma, wavenumber(sigma, d, G), d);
    let mut n = m.zero_state();
    run(&m, &mut n, 4.0 * L / cg_at(3.0), 0.5);
    let (c, xy) = (m.grid.component(i, 0), nodes(&m));
    let flux0 = cg_at(depth(0.0)) * e[c] / sigma;
    let worst = (0..m.n_points())
        .map(|p| (cg_at(depth(xy[p][0])) * n.component(c)[p] / flux0 - 1.0).abs())
        .fold(0.0, f64::max);
    println!("largest relative departure of c_g N from the boundary's: {worst:.2e}");
    assert!(worst < 2e-3, "c_g N departs by {worst:e}");
}

/// Oblique incidence (30° from the shore normal) on a shelf from 20 to 4 m deep:
/// the steady mean direction follows Snell's law, `sin α / c` constant, to the
/// first order in the direction bins (the refraction is first-order upwind).
#[test]
fn refraction_follows_snells_law() {
    const LX: f64 = 400.0;
    const LY: f64 = 2000.0;
    let depth = |y: f64| 20.0 - 16.0 * y / LY;
    let mut errors = Vec::new();
    for n_dir in [36, 72, 144] {
        let mut mesh = Mesh2D::channel_periodic_x(0.0, LX, 0.0, LY, 4, 40);
        for edge in mesh.edges.iter_mut().filter(|e| e.right.is_none()) {
            edge.boundary_tag = Some(BoundaryTag::Open);
        }
        let grid = SpectralGrid::new(0.12, 0.15, 2, n_dir);
        // θ = 60°: 30° off +y, the shore normal
        let (i, j0) = (0, n_dir / 6);
        let mut e = vec![0.0; grid.n_components()];
        e[grid.component(i, j0)] = 1.0;
        let m = model(mesh, 2, |_, y| -depth(y), grid).with_boundary_spectrum(&e);
        let sigma = m.grid.sigma[i];
        let celerity = |d: f64| sigma / wavenumber(sigma, d, G);
        let alpha0 = 30.0f64.to_radians();
        let mut n = m.zero_state();
        let cg_min = group_velocity(sigma, wavenumber(sigma, 4.0, G), 4.0);
        run(&m, &mut n, 3.0 * LY / (cg_min * alpha0.cos()), 0.5);
        let (xy, params) = (nodes(&m), m.parameters(&n));
        let mut worst: f64 = 0.0;
        for p in (0..m.n_points()).filter(|&p| (200.0..1800.0).contains(&xy[p][1])) {
            let d = depth(xy[p][1]);
            let alpha = (alpha0.sin() * celerity(d) / celerity(depth(0.0))).asin();
            let expected = 90.0 - alpha.to_degrees();
            worst = worst.max((params[p].direction.to_degrees() - expected).abs());
        }
        errors.push(worst);
    }
    let rates: Vec<f64> = errors.windows(2).map(|e| (e[0] / e[1]).log2()).collect();
    println!(
        "largest departure of the mean direction from Snell's law: {errors:?}°, rates {rates:?}"
    );
    assert!(
        rates.iter().all(|&r| r > 0.8),
        "first order in Δθ: {rates:?}"
    );
    assert!(errors[1] < 1.5, "{}° off Snell with 5° bins", errors[1]);
}

/// A following current accelerating from 0 to 0.6 m/s over deep water stretches
/// the waves: the absolute frequency `σ + k U` is conserved, so the intrinsic
/// frequency falls as `σ + σ² U/g = σ_0`.
#[test]
fn a_following_current_shifts_the_frequency_doppler() {
    const L: f64 = 2000.0;
    let grid = SpectralGrid::new(0.15, 0.35, 41, 8);
    let i0 = (0..grid.n_freq())
        .min_by(|&a, &b| {
            let d = |i: usize| (grid.sigma[i] / TAU - 0.3).abs();
            d(a).total_cmp(&d(b))
        })
        .unwrap();
    let mut e = vec![0.0; grid.n_components()];
    e[grid.component(i0, 0)] = 1.0;
    let mut m = model(
        Mesh2D::uniform_rectangle_with_sides(
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
        ),
        2,
        |_, _| -1000.0,
        grid,
    )
    .with_boundary_spectrum(&e);
    let xy = nodes(&m);
    let current = |x: f64| 0.6 * x / L;
    let u: Vec<f64> = xy.iter().map(|p| current(p[0])).collect();
    m.set_currents(&u, &vec![0.0; u.len()]);
    let sigma0 = m.grid.sigma[i0];
    let mut n = m.zero_state();
    run(&m, &mut n, 4.0 * L / (G / (2.0 * sigma0)), 0.5);
    let (nd, mut worst) = (m.grid.n_dir(), 0.0f64);
    for p in (0..m.n_points()).filter(|&p| xy[p][0] > 200.0) {
        let (mut num, mut den) = (0.0, 0.0);
        for ii in 0..m.grid.n_freq() {
            let w = n.component(ii * nd)[p] * m.grid.d_sigma[ii];
            num += m.grid.sigma[ii] * w;
            den += w;
        }
        let uu = current(xy[p][0]);
        let expected = (-1.0 + (1.0 + 4.0 * uu / G * sigma0).sqrt()) / (2.0 * uu / G);
        worst = worst.max((num / den / expected - 1.0).abs());
    }
    println!("largest relative departure of the mean σ from σ + kU = σ₀: {worst:.2e}");
    assert!(worst < 0.01, "mean frequency off by {worst:e}");
}

/// `(E*, f_p*) = (g² m_0/U₁₀⁴, f_p U₁₀/g)` of the variance density `e`.
fn dimensionless(grid: &SpectralGrid, e: &[f64], u10: f64) -> (f64, f64) {
    let p = grid.parameters(e);
    (G * G * p.m0 / u10.powi(4), u10 / (p.tp * G))
}

/// The deep-water sources of SWAN's GEN3 KOMEN set (wind, whitecapping, the DIA,
/// the f⁻⁴ tail and the limiter; no bed or breaking in 1000 m).
fn deep_water_sources() -> SourceTerms {
    SourceTerms::swan_defaults(G)
        .with_bottom_friction(None)
        .with_breaking(None)
}

/// Fetch-limited growth: a steady 10 m/s wind blowing off a straight coast over
/// deep water (a strip periodic along the coast, 120 km offshore, P1). In the
/// steady state the peak frequency follows Kahma & Calkoen's (1992) composite
/// `f_p* = 2.18 X*^−0.27` to a few per cent from X* = 10³ to 1.2·10⁴, and the
/// energy `E* = 5.2e-7 X*^0.9` within the known deficit of Komen whitecapping at
/// long fetch: measured 1.3× at X* = 1200 falling to 0.60× at 1.2·10⁴ (E* ∝
/// X*^0.56; f_p within 0.90–1.05×). The curve is converged: P2, twice the
/// elements, 36 directions, a 2 Hz grid or no tail move it by ≤ 4.5 %.
#[test]
fn fetch_limited_growth_follows_kahma_and_calkoen() {
    const LENGTH: f64 = 120e3;
    let u10 = 10.0;
    let mut mesh = Mesh2D::channel_periodic_x(0.0, 1000.0, 0.0, LENGTH, 1, 30);
    // The coast at y = 0 lets nothing in; the open sea at y = L lets waves out
    for edge in mesh.edges.iter_mut().filter(|e| e.right.is_none()) {
        if mesh.vertices[edge.vertices.0][1] > 0.5 * LENGTH {
            edge.boundary_tag = Some(BoundaryTag::Open);
        }
    }
    let wind = Wind {
        u10,
        direction: PI / 2.0,
    };
    let m = model(
        mesh,
        1,
        |_, _| -1000.0,
        SpectralGrid::new(0.06, 1.0, 25, 24),
    )
    .with_sources(deep_water_sources())
    .with_wind(wind);
    let mut n = m.zero_state();
    run(&m, &mut n, 10.0 * 3600.0, 0.5);
    let xy = nodes(&m);
    let mut e = vec![0.0; m.grid.n_components()];
    let mut curve: Vec<(f64, f64, f64)> = Vec::new();
    for (p, [_, y]) in xy.into_iter().enumerate() {
        let x = G * y / (u10 * u10);
        if !(1000.0..12500.0).contains(&x) || curve.iter().any(|c| (c.0 - x).abs() < 1.0) {
            continue;
        }
        m.energy_spectrum_into(&n, p, &mut e);
        let (es, fs) = dimensionless(&m.grid, &e, u10);
        curve.push((x, es, fs));
    }
    curve.sort_by(|a, b| a.0.total_cmp(&b.0));
    for &(x, es, fs) in &curve {
        let (e_kc, f_kc) = (5.2e-7 * x.powf(0.9), 2.18 * x.powf(-0.27));
        println!(
            "X* {x:7.0}: E* {es:.3e} ({:.2}× KC92), f_p* {fs:.4} ({:.2}× KC92)",
            es / e_kc,
            fs / f_kc
        );
        assert!(
            (0.85..1.12).contains(&(fs / f_kc)),
            "X* {x:.0}: f_p* {fs:.4} against {f_kc:.4}"
        );
        assert!(
            (0.55..1.4).contains(&(es / e_kc)),
            "X* {x:.0}: E* {es:.3e} against {e_kc:.3e}"
        );
    }
    // Growing with fetch, slower than KC92's 0.9 but not stalled
    let (first, last) = (curve[0], curve[curve.len() - 1]);
    let exponent = (last.1 / first.1).ln() / (last.0 / first.0).ln();
    println!("E* ∝ X*^{exponent:.2}");
    assert!(curve.windows(2).all(|w| w[1].1 > w[0].1));
    assert!((0.5..1.0).contains(&exponent), "E* ∝ X*^{exponent}");
}

/// Duration-limited growth at one node: a 10 m/s wind over a calm deep sea, the
/// sources alone. The energy grows and the peak moves to lower frequencies
/// monotonically, towards Pierson–Moskowitz (E* = 3.64e-3, f_p* = 0.13): 0.85×
/// and 0.95× of it after 96 h (t* = 3.4·10⁵). The step does not matter: 10, 60
/// and 300 s agree to 0.3 % at 96 h. (Run on, the sea passes Pierson–Moskowitz
/// after ≈ 180 h and reaches 1.5× at 1000 h: Komen whitecapping with δ = 1 and
/// the δ = 0 constant C_ds, as SWAN warns.)
#[test]
fn duration_limited_growth_approaches_pierson_moskowitz() {
    let grid = SpectralGrid::new(0.06, 1.0, 30, 24);
    let (u10, depth, dt) = (10.0, 1000.0, 300.0);
    let wind = Wind {
        u10,
        direction: 0.0,
    };
    let sources = deep_water_sources();
    let k: Vec<f64> = grid
        .sigma
        .iter()
        .map(|&s| wavenumber(s, depth, G))
        .collect();
    let (nc, nd) = (grid.n_components(), grid.n_dir());
    let mut n = vec![0.0; nc];
    let (mut e, mut a, mut b) = (vec![0.0; nc], vec![0.0; nc], vec![0.0; nc]);
    let mut history = Vec::new();
    for step in 1..=(96 * 3600) / 300 {
        sources.integrate(&grid, &mut n, &k, depth, wind, dt, &mut e, &mut a, &mut b);
        if step % 12 == 0 {
            let e: Vec<f64> = (0..nc).map(|c| n[c] * grid.sigma[c / nd]).collect();
            history.push(dimensionless(&grid, &e, u10));
        }
    }
    let (es, fs) = *history.last().unwrap();
    println!(
        "after 96 h: E* {es:.3e} ({:.2}× PM), f_p* {fs:.4}",
        es / 3.64e-3
    );
    assert!(history.windows(2).all(|w| w[1].0 > w[0].0));
    // The parabolic peak may sit still between hourly samples
    assert!(history.windows(2).all(|w| w[1].1 <= w[0].1 * (1.0 + 1e-9)));
    assert!((0.75..1.0).contains(&(es / 3.64e-3)), "E* {es:e}");
    assert!((fs / 0.13 - 1.0).abs() < 0.1, "f_p* {fs}");
}
