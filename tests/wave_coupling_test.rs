//! Gates of the coupling from the spectral wave model to the circulation (TODO
//! F.4): the radiation-stress force of `WaveForce2D` and the wave-enhanced bed
//! friction of `WaveCurrentFriction2D` on the 2D shallow-water model, through
//! the production path (`Simulation` + `SSPRK3` + `SWEPhysics2D`).

use std::sync::Arc;

use dg_rs::boundary::Reflective2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::{PhysicsBuilder, SWEPhysics2DBuilder};
use dg_rs::simulation::Simulation;
use dg_rs::solver::{SWESolution2D, SWEState2D};
use dg_rs::source::{
    ChezyFriction2D, SourceContext2D, SourceTerm2D, WaveCurrentFriction2D, WaveForce2D,
};
use dg_rs::time::SSPRK3;
use dg_rs::types::ElementIndex;
use dg_rs::waves::{
    SourceTerms, SpectralGrid, WaveModel2D, WaveSolution, WaveWorkspace, wavenumber,
};

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

/// Linear damping of the cross-shore (y) momentum only, `−r hv`, to settle the
/// beach's seiches without touching the alongshore balance.
struct CrossShoreDamping(f64);

impl SourceTerm2D for CrossShoreDamping {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        SWEState2D::new(0.0, 0.0, -self.0 * ctx.state.hv)
    }

    fn name(&self) -> &'static str {
        "cross_shore_damping"
    }
}

/// A plane beach, uniform alongshore (x, periodic, one element wide): the depth
/// falls linearly from 4 m at the open sea (y = 0) to 0.5 m at a wall at
/// y = 350 m, a little offshore of where the shoreline would be. P2, 25 m
/// elements.
struct Beach {
    mesh: Arc<Mesh2D>,
    ops: Arc<DGOperators2D>,
    geom: Arc<GeometricFactors2D>,
    bathymetry: Arc<Bathymetry2D>,
}

impl Beach {
    const L: f64 = 350.0;
    const H0: f64 = 4.0;
    const H1: f64 = 0.5;
    const NY: usize = 14;

    fn depth(y: f64) -> f64 {
        Self::H0 + (Self::H1 - Self::H0) * y / Self::L
    }

    fn new() -> Self {
        let width = Self::L / Self::NY as f64;
        let mesh = Mesh2D::channel_periodic_x_with_sides(
            0.0,
            width,
            0.0,
            Self::L,
            1,
            Self::NY,
            [BoundaryTag::Open, BoundaryTag::Wall],
        );
        let ops = Arc::new(DGOperators2D::new(2));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |_, y| {
            -Self::depth(y)
        }));
        Self {
            mesh: Arc::new(mesh),
            ops,
            geom,
            bathymetry,
        }
    }

    /// Swell of 8 s and H_rms = 0.6 m offshore travelling to `direction` (rad
    /// from +x; π/2 is straight at the beach) over `n_dir` directions, all in
    /// one bin, or cos^`m` spread. Battjes–Janssen breaking (α = 1, γ = 0.73)
    /// is the only source.
    fn waves(&self, direction: f64, m: Option<i32>, n_dir: usize) -> WaveModel2D {
        let f = 1.0 / 8.0;
        let grid = SpectralGrid::new(f, 1.2 * f, 2, n_dir);
        let mut weights: Vec<f64> = grid
            .theta
            .iter()
            .map(|&t| {
                let c = (t - direction).cos();
                match m {
                    Some(m) => c.max(0.0).powi(m),
                    None if c > 1.0 - 1e-9 => 1.0,
                    None => 0.0,
                }
            })
            .collect();
        let total: f64 = weights.iter().sum::<f64>() * grid.d_theta;
        weights.iter_mut().for_each(|w| *w /= total);
        let mut e = vec![0.0; grid.n_components()];
        for (j, w) in weights.iter().enumerate() {
            e[grid.component(0, j)] = 0.6f64.powi(2) / 8.0 / grid.d_sigma[0] * w;
        }
        WaveModel2D::new(
            self.mesh.clone(),
            self.ops.clone(),
            self.geom.clone(),
            &self.bathymetry,
            grid,
            G,
        )
        .with_sources(SourceTerms::none(G).with_breaking(Some((1.0, 0.73))))
        .with_boundary_spectrum(&e)
    }

    /// Run `waves` to steady from `n` (four crossings at 2 m/s).
    fn settle(waves: &WaveModel2D, n: &mut WaveSolution) {
        let mut ws = WaveWorkspace::default();
        let dt = waves.compute_dt(0.5);
        let steps = (4.0 * Self::L / (2.0 * dt)).ceil() as usize;
        for s in 0..steps {
            waves.step(n, s as f64 * dt, dt, &mut ws);
        }
    }

    fn physics(&self) -> SWEPhysics2DBuilder<Reflective2D> {
        PhysicsBuilder::swe_2d(
            self.mesh.clone(),
            self.ops.clone(),
            self.geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::new(),
        )
        .with_bathymetry(self.bathymetry.clone())
        .with_source(CrossShoreDamping(1e-2))
    }

    fn still_water(&self) -> SWESolution2D {
        let nn = self.ops.n_nodes;
        let mut q = SWESolution2D::new(self.mesh.n_elements, nn);
        for k in ElementIndex::iter(self.mesh.n_elements) {
            for i in 0..nn {
                q.set_state(k, i, SWEState2D::new(-self.bathymetry.get(k, i), 0.0, 0.0));
            }
        }
        q
    }

    fn eta(&self, q: &SWESolution2D) -> Vec<f64> {
        q.h_data()
            .iter()
            .zip(&self.bathymetry.data)
            .map(|(h, b)| h + b)
            .collect()
    }

    /// y of every node.
    fn ys(&self) -> Vec<f64> {
        ElementIndex::iter(self.mesh.n_elements)
            .flat_map(|k| {
                (0..self.ops.n_nodes).map(move |i| {
                    self.mesh
                        .reference_to_physical(k, self.ops.nodes_r[i], self.ops.nodes_s[i])[1]
                })
            })
            .collect()
    }

    /// The nodes of the column r = −1 from the sea to the wall, and at each
    /// one `∫₀^y F_y/(g D) dy`: the level the momentum balance
    /// `g D ∂η/∂y = F_y` gives for the force `force` on the depths of `q`
    /// (integrated exactly for the quadratic through each element's three
    /// nodes).
    fn balance(&self, force: &[[f64; 2]], q: &SWESolution2D) -> Vec<(usize, f64)> {
        let (ops, nn) = (&self.ops, self.ops.n_nodes);
        let mut column: Vec<usize> = (0..nn)
            .filter(|&i| (ops.nodes_r[i] + 1.0).abs() < 1e-12)
            .collect();
        column.sort_by(|&a, &b| ops.nodes_s[a].total_cmp(&ops.nodes_s[b]));
        let ys = self.ys();
        let mut out = vec![(column[0], 0.0)];
        let mut integral = 0.0;
        for k in 0..self.mesh.n_elements {
            let p: Vec<usize> = column.iter().map(|&i| k * nn + i).collect();
            let f: Vec<f64> = p
                .iter()
                .map(|&p| force[p][1] / (G * q.h_data()[p]))
                .collect();
            let dy = ys[p[2]] - ys[p[0]];
            let mid = integral + dy / 24.0 * (5.0 * f[0] + 8.0 * f[1] - f[2]);
            integral += dy / 6.0 * (f[0] + 4.0 * f[1] + f[2]);
            out.push((p[1], mid));
            out.push((p[2], integral));
        }
        out
    }
}

/// Swell of 8 s and H_rms = 0.6 m runs straight at a plane beach (1:100, 4 m
/// to a wall at 0.5 m) and breaks (Battjes–Janssen). The radiation stress
/// sets the water down to the breaker line (the largest H_rms) and up across
/// the surf zone: 0.69 cm down, 5.6 cm up at the wall. Two-way coupled: the
/// waves see the circulation's level (`set_water_level`), which deepens the
/// surf zone and lowers the setup at the wall by 3.5 % against waves on still
/// water; the third pass changes it by 0.15 %.
///
/// The level follows the cross-shore momentum balance `g D ∂η/∂y = F_y` of the
/// force on the circulation's own depths to 0.04 % of its range. The setup
/// slope over the inner surf zone is 0.039 of the bed's, below Bowen et al.'s
/// (1968) `3κ²/8/(1 + 3κ²/8)` = 0.062 for its mean κ = H_rms/D = 0.42:
/// Bowen's waves are saturated (κ constant), while Battjes–Janssen's κ rises
/// from 0.34 to 0.48 towards the wall, so S falls more slowly than `(3/16)κ²D²`
/// with κ fixed.
#[test]
fn breaking_waves_set_the_water_up_in_the_surf_zone() {
    let beach = Beach::new();
    let mut waves = beach.waves(std::f64::consts::FRAC_PI_2, None, 36);
    let mut n = waves.zero_state();
    let mut q = beach.still_water();
    let ys = beach.ys();
    let mut walls = Vec::new();
    for pass in 0..3 {
        Beach::settle(&waves, &mut n);
        let force = WaveForce2D::new(&waves, &n);
        let f = force.force().to_vec();
        // Ramp the first pass up from rest; later ones start from the last level
        let force = if pass == 0 {
            force.with_ramp(300.0)
        } else {
            force
        };
        let physics = beach.physics().with_source(force).build();
        let result = Simulation::new(physics, SSPRK3).run(&mut q, 0.0, 3000.0);
        assert!(result.success, "{result:?}");
        let eta = beach.eta(&q);

        let balance = beach.balance(&f, &q);
        let eta0 = eta[balance[0].0];
        let rise: Vec<f64> = balance.iter().map(|&(p, _)| eta[p] - eta0).collect();
        let (low, high) = rise
            .iter()
            .fold((0.0f64, 0.0f64), |(lo, hi), &r| (lo.min(r), hi.max(r)));
        let worst = balance
            .iter()
            .zip(&rise)
            .map(|(&(_, level), r)| (r - level).abs())
            .fold(0.0f64, f64::max);
        let speed = q
            .hv_data()
            .iter()
            .zip(q.h_data())
            .map(|(hv, h)| (hv / h).abs())
            .fold(0.0f64, f64::max);

        // The breaker line (the largest H_rms) and the inner surf zone
        let params = waves.parameters(&n);
        let &(breaker, _) = balance
            .iter()
            .max_by(|a, b| params[a.0].m0.total_cmp(&params[b.0].m0))
            .unwrap();
        let &(lowest, _) = balance
            .iter()
            .min_by(|a, b| eta[a.0].total_cmp(&eta[b.0]))
            .unwrap();
        let wall = balance.last().unwrap().0;
        let kappa: Vec<f64> = balance
            .iter()
            .filter(|&&(p, _)| ys[p] > ys[breaker])
            .map(|&(p, _)| (8.0 * params[p].m0).sqrt() / q.h_data()[p])
            .collect();
        let mean_kappa = kappa.iter().sum::<f64>() / kappa.len() as f64;
        let slope =
            (eta[wall] - eta[breaker]) / (Beach::depth(ys[breaker]) - Beach::depth(ys[wall]));
        let bowen = 3.0 * mean_kappa.powi(2) / 8.0;
        println!(
            "pass {pass}: set-down {:.2} cm at y = {:.0} m (breaker line {:.0} m), setup {:.2} cm \
             at the wall; balance off by {worst:.1e} m ({:.3} %), |v| ≤ {speed:.1e} m/s; \
             surf slope {slope:.3} of the bed's, Bowen {:.3} for κ = {mean_kappa:.2} \
             ({:.2}–{:.2})",
            100.0 * low,
            ys[lowest],
            ys[breaker],
            100.0 * (eta[wall] - eta0),
            100.0 * worst / (high - low),
            bowen / (1.0 + bowen),
            kappa.iter().cloned().fold(f64::INFINITY, f64::min),
            kappa.iter().cloned().fold(0.0, f64::max),
        );
        assert!(
            worst < 2e-3 * (high - low),
            "off the balance by {worst:e} m"
        );
        assert!(speed < 1e-3, "not at rest: {speed} m/s");
        assert!(
            (ys[lowest] - ys[breaker]).abs() <= Beach::L / Beach::NY as f64,
            "the set-down is lowest at {} m, the breaker line is at {} m",
            ys[lowest],
            ys[breaker]
        );
        assert!(low < -5e-3 && eta[wall] - eta0 > 5e-2, "{low}, {high}");
        walls.push(eta[wall] - eta0);
        waves.set_water_level(&eta);
    }
    let change = (walls[2] / walls[1] - 1.0).abs();
    assert!(change < 5e-3, "the coupling has not converged: {walls:?}");
}

/// The longshore force by Longuet-Higgins (1970) per node, from the waves'
/// dissipation alone: with Snell's `k cos θ` (θ from the shore) constant along
/// each ray, the alongshore radiation stress changes only by what the waves
/// lose, `−∂S_xy/∂y = Σ (k cos θ/σ) D` (`D` the variance each component loses
/// per second, its rate `−B` times `E`), so `F_x = g Σ (k cos θ/σ) D`.
fn longuet_higgins_force(waves: &WaveModel2D, n: &WaveSolution) -> Vec<f64> {
    let grid = &waves.grid;
    let nf = grid.n_freq();
    let mut e = vec![0.0; grid.n_components()];
    let (mut a, mut b) = (e.clone(), e.clone());
    let mut k = vec![0.0; nf];
    (0..waves.n_points())
        .map(|p| {
            waves.energy_spectrum_into(n, p, &mut e);
            (0..nf).for_each(|i| k[i] = waves.wavenumber(i, p));
            let depth = waves.depth()[p];
            waves
                .sources
                .rates(grid, &e, &k, depth, Default::default(), &mut a, &mut b);
            let mut force = 0.0;
            for (i, &ki) in k.iter().enumerate() {
                for (j, cos) in grid.cos_theta.iter().enumerate() {
                    let c = grid.component(i, j);
                    let lost = -b[c] * e[c] * grid.d_sigma[i] * grid.d_theta;
                    force += ki * cos / grid.sigma[i] * lost;
                }
            }
            G * force
        })
        .collect()
}

/// The same swell, cos²⁰-spread about 20° off the normal (72 directions),
/// drives a current along the beach. With no lateral mixing each line across
/// the beach is its own balance, `C_d |u| u = F_x`, and Longuet-Higgins's
/// (1970) force from the dissipation (`longuet_higgins_force`) gives
/// `u = √(F_x/C_d)`: 0.60 m/s at most (the model's 0.615), two thirds of the way across the surf
/// zone. Where that force is over a tenth of its largest the model's current
/// follows it to 3.8 % (2.2 % at twice the resolution). The radiation-stress
/// force of the wave model is above the dissipation's, by up to 9.5 % of the
/// largest force at the ends of elements (where the derivative within each
/// element is one-sided; the jumps of S between elements are left out) and
/// ≈ 5 % inside them, converging with the mesh: 3 % at 12.5 m elements.
///
/// Seaward of the breakers there is no dissipation and the force should
/// vanish: what remains is the discrete refraction (the direction bins), which
/// leaves the flux of alongshore momentum not quite constant, and a current
/// of −0.8 cm/s (−3.8 cm/s at 36 directions, −0.15 cm/s at 144).
#[test]
fn oblique_breaking_waves_drive_a_longshore_current() {
    let beach = Beach::new();
    let direction = std::f64::consts::FRAC_PI_2 - 20f64.to_radians();
    let waves = beach.waves(direction, Some(20), 72);
    let mut n = waves.zero_state();
    Beach::settle(&waves, &mut n);
    let force = WaveForce2D::new(&waves, &n);
    let lh = longuet_higgins_force(&waves, &n);
    let cd = 2.5e-3;
    let physics = beach
        .physics()
        .with_source(force.clone().with_ramp(300.0))
        .with_implicit_friction(ChezyFriction2D::new(cd))
        .build();
    let mut q = beach.still_water();
    let result = Simulation::new(physics, SSPRK3).run(&mut q, 0.0, 10_000.0);
    assert!(result.success, "{result:?}");

    let largest = lh.iter().cloned().fold(0.0, f64::max);
    let expected = |p: usize| (lh[p].max(0.0) / cd).sqrt();
    let (mut surf, mut offshore, mut fastest) = (0.0f64, 0.0f64, 0.0f64);
    for (p, &f) in lh.iter().enumerate() {
        let u = q.hu_data()[p] / q.h_data()[p];
        fastest = fastest.max(u);
        if f > 0.1 * largest {
            surf = surf.max((u / expected(p) - 1.0).abs());
        } else if f < 1e-3 * largest {
            offshore = offshore.max(u.abs());
        }
    }
    let excess = force
        .force()
        .iter()
        .zip(&lh)
        .map(|(f, lh)| (f[0] - lh).abs())
        .fold(0.0f64, f64::max)
        / largest;
    println!(
        "longshore current {fastest:.3} m/s (Longuet-Higgins {:.3}); off by ≤ {:.2} % in the \
         surf zone; force off the dissipation's by ≤ {:.1} % of its largest; |u| ≤ {offshore:.4} m/s offshore",
        (largest / cd).sqrt(),
        100.0 * surf,
        100.0 * excess
    );
    assert!(surf < 0.05, "the current is {surf} off Longuet-Higgins's");
    assert!(excess < 0.1, "the force is {excess} off the dissipation's");
    assert!(
        offshore < 0.02 * fastest,
        "an offshore current of {offshore} m/s"
    );
    assert!((fastest / (largest / cd).sqrt() - 1.0).abs() < 0.05);
}
