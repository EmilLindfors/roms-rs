//! Wetting/drying and bottom-friction tests for the 2D SWE (TODO P1.2, P1.7).
//!
//! All runs use the production path, `Simulation` + `SSPRK3` + `SWEPhysics2D`,
//! with the wet/dry defaults: HLL flux, positivity limiting towards h ≥ 0,
//! Kurganov–Petrova desingularization, point-implicit thin-layer relaxation
//! and the positivity CFL cap.
//!
//! Two formulations handle wetting and drying: `Standard` (collocated, with
//! hydrostatic reconstruction at faces) and `WetDry` (split form, subcell
//! finite volumes in partially dry elements). `WetDry` keeps a shoreline lake
//! at rest to round-off, where `Standard` settles into a spurious circulation;
//! `Standard` is about 3× more accurate for a moving shoreline (Thacker).
//!
//! Before P1.2 (same probes, HLL flux, `main` at 521ca1a): the shoreline lake
//! at rest reached the 20 m/s velocity cap at P1 and P3 and 5.6 m/s at P2, with
//! 0.13–0.57 m surface errors; Thacker's L1 depth error was 9.2 % (P2, 20²).

use std::f64::consts::PI;
use std::sync::Arc;

use dg_rs::boundary::Reflective2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::{Bathymetry2D, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::{PhysicsBuilder, PhysicsModule, SWEPhysics2DBuilder};
use dg_rs::simulation::Simulation;
use dg_rs::solver::{
    KuzminParameter2D, SWEFormulation2D, SWESolution2D, SWEState2D, StandardLimiter2D,
    WetDryConfig, positivity_cfl_swe_2d,
};
use dg_rs::source::{BathymetrySource2D, ManningFriction2D};
use dg_rs::time::SSPRK3;
use dg_rs::types::{Depth, ElementIndex};

const G: f64 = 9.81;

struct Setup {
    mesh: Arc<Mesh2D>,
    ops: Arc<DGOperators2D>,
    geom: Arc<GeometricFactors2D>,
}

impl Setup {
    fn new(mesh: Mesh2D, order: usize) -> Self {
        let ops = Arc::new(DGOperators2D::new(order));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh));
        Self {
            mesh: Arc::new(mesh),
            ops,
            geom,
        }
    }

    fn builder(&self) -> SWEPhysics2DBuilder<Reflective2D> {
        PhysicsBuilder::swe_2d(
            self.mesh.clone(),
            self.ops.clone(),
            self.geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::new(),
        )
    }

    /// Wet/dry physics over bathymetry `bed` with the positivity limiter and
    /// the wet/dry treatment. `Standard` adds hydrostatic reconstruction and
    /// the bed-slope source; the split forms carry the bed in the operator.
    fn wet_dry_over(
        &self,
        formulation: SWEFormulation2D,
        h_dry: f64,
        bed: impl Fn(f64, f64) -> f64,
    ) -> SWEPhysics2DBuilder<Reflective2D> {
        let bathymetry = Bathymetry2D::from_function(&self.mesh, &self.ops, &self.geom, bed);
        let builder = self
            .builder()
            .with_bathymetry(Arc::new(bathymetry))
            .with_formulation(formulation)
            .with_limiter(StandardLimiter2D::Positivity(h_dry))
            .with_wet_dry(WetDryConfig::new(Depth::new(h_dry), G));
        match formulation {
            SWEFormulation2D::Standard => builder
                .with_well_balanced(true)
                .with_source(BathymetrySource2D::new(G)),
            _ => builder,
        }
    }

    fn node_xy(&self, k: ElementIndex, i: usize) -> (f64, f64) {
        let [x, y] = self
            .mesh
            .reference_to_physical(k, self.ops.nodes_r[i], self.ops.nodes_s[i]);
        (x, y)
    }

    fn nodes(&self) -> impl Iterator<Item = (ElementIndex, usize)> + '_ {
        ElementIndex::iter(self.mesh.n_elements)
            .flat_map(|k| (0..self.ops.n_nodes).map(move |i| (k, i)))
    }

    fn fill(&self, f: impl Fn(f64, f64) -> SWEState2D) -> SWESolution2D {
        let mut q = SWESolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
        for (k, i) in self.nodes() {
            let (x, y) = self.node_xy(k, i);
            q.set_state(k, i, f(x, y));
        }
        q
    }

    /// ∫ f(q, x, y) dA with the GLL quadrature (affine elements)
    fn integrate(&self, q: &SWESolution2D, f: impl Fn(SWEState2D, f64, f64) -> f64) -> f64 {
        self.nodes()
            .map(|(k, i)| {
                let (x, y) = self.node_xy(k, i);
                self.ops.weights[i] * self.geom.det_j[k.as_usize()] * f(q.get_state(k, i), x, y)
            })
            .sum()
    }

    fn mass(&self, q: &SWESolution2D) -> f64 {
        self.integrate(q, |s, _, _| s.h)
    }
}

/// Largest |u| over nodes deeper than `h_floor`.
fn max_speed(q: &SWESolution2D, h_floor: f64) -> f64 {
    let [h, hu, hv] = &q.data;
    (0..h.len())
        .filter(|&j| h[j] > h_floor)
        .map(|j| hu[j].hypot(hv[j]) / h[j])
        .fold(0.0, f64::max)
}

fn min_depth(q: &SWESolution2D) -> f64 {
    q.h_data().iter().copied().fold(f64::INFINITY, f64::min)
}

// ---------------------------------------------------------------------------
// Dam break onto a dry bed
// ---------------------------------------------------------------------------

/// Ritter (1892): depth h0 for x < x0 and a dry bed beyond, released at t = 0.
fn ritter(x: f64, t: f64, h0: f64, x0: f64) -> (f64, f64) {
    let c0 = (G * h0).sqrt();
    let xi = (x - x0) / t;
    if xi <= -c0 {
        (h0, 0.0)
    } else if xi >= 2.0 * c0 {
        (0.0, 0.0)
    } else {
        ((2.0 * c0 - xi).powi(2) / (9.0 * G), 2.0 / 3.0 * (xi + c0))
    }
}

/// Relative L1 depth error against Ritter's solution on an n × 2 mesh (P2,
/// Kuzmin + positivity limiter), checking mass, positivity and speeds.
fn dam_break_error(formulation: SWEFormulation2D, n: usize) -> f64 {
    let (h0, x0, t_end) = (1.0, 50.0, 5.0);
    let c0 = (G * h0).sqrt();
    let setup = Setup::new(Mesh2D::uniform_rectangle(0.0, 100.0, 0.0, 4.0, n, 2), 2);
    let physics = setup
        .builder()
        .with_formulation(formulation)
        .with_limiter(StandardLimiter2D::KuzminWithPositivity {
            kuzmin: KuzminParameter2D::strict(),
            h_min: WetDryConfig::DEFAULT_H_DRY,
        })
        .with_wet_dry(WetDryConfig::default())
        .build();
    let mut q = setup.fill(|x, _| SWEState2D::new(if x < x0 { h0 } else { 0.0 }, 0.0, 0.0));
    let mass0 = setup.mass(&q);

    let (mut lowest, mut fastest) = (f64::INFINITY, 0.0_f64);
    // CFL 1 is capped at the positivity bound
    let sim = Simulation::new(physics, SSPRK3).with_cfl(1.0);
    let result = sim.run_with_callback(&mut q, 0.0, t_end, |q, _| {
        lowest = lowest.min(min_depth(q));
        fastest = fastest.max(max_speed(q, 0.0));
    });
    assert!(result.success, "{result:?}");

    let tag = format!("{formulation:?}, n = {n}");
    let mass_drift = (setup.mass(&q) - mass0).abs() / mass0;
    assert!(mass_drift < 1e-13, "{tag}: mass drift {mass_drift:e}");
    assert!(lowest >= 0.0, "{tag}: negative depth {lowest:e}");
    assert_eq!(sim.physics().negative_depth_clips(), 0, "{tag}");
    // The front moves at 2c0 = 6.3 m/s; nothing may outrun it
    assert!(fastest < 2.0 * c0, "{tag}: |u| reached {fastest:.2} m/s");

    setup.integrate(&q, |s, x, _| (s.h - ritter(x, t_end, h0, x0).0).abs()) / mass0
}

#[test]
fn dam_break_onto_dry_bed_matches_ritter() {
    // First order: a strict Kuzmin limiter at the dry front and the
    // rarefaction corners (and subcell finite volumes in the front elements
    // with WetDry)
    for (formulation, bound) in [
        (SWEFormulation2D::Standard, 0.02),
        (SWEFormulation2D::WetDry, 0.02),
    ] {
        let (coarse, fine) = (
            dam_break_error(formulation, 50),
            dam_break_error(formulation, 100),
        );
        let rate = (coarse / fine).log2();
        println!("dam break {formulation:?} L1(h)/mass: n=50 {coarse:.3e}, n=100 {fine:.3e}");
        assert!(fine < bound, "{formulation:?}: L1 error {fine:.3e}");
        assert!(rate > 0.8, "{formulation:?}: rate {rate:.2}");
    }
}

// ---------------------------------------------------------------------------
// Thacker's planar surface in a paraboloid
// ---------------------------------------------------------------------------

/// Thacker (1981) planar oscillation in a paraboloid, SWASHES parameters
/// (Delestre et al. 2013): the wet disc slides around the bowl with period
/// 2π/ω, so the shoreline wets and dries everywhere.
struct Thacker {
    a: f64,
    h0: f64,
    eta: f64,
    l: f64,
}

impl Thacker {
    const SWASHES: Self = Self {
        a: 1.0,
        h0: 0.1,
        eta: 0.5,
        l: 4.0,
    };

    fn omega(&self) -> f64 {
        (2.0 * G * self.h0).sqrt() / self.a
    }

    fn bed(&self, x: f64, y: f64) -> f64 {
        let r2 = (x - 0.5 * self.l).powi(2) + (y - 0.5 * self.l).powi(2);
        -self.h0 * (1.0 - r2 / (self.a * self.a))
    }

    fn exact(&self, x: f64, y: f64, t: f64) -> SWEState2D {
        let w = self.omega();
        let (xc, yc) = (x - 0.5 * self.l, y - 0.5 * self.l);
        let surface = self.eta * self.h0 / (self.a * self.a)
            * (2.0 * xc * (w * t).cos() + 2.0 * yc * (w * t).sin() - self.eta);
        let h = surface - self.bed(x, y);
        if h > 0.0 {
            let (u, v) = (-self.eta * w * (w * t).sin(), self.eta * w * (w * t).cos());
            SWEState2D::from_primitives(h, u, v)
        } else {
            SWEState2D::zero()
        }
    }
}

/// Thacker's case over one period on an n × n P2 mesh; returns the relative
/// L1 depth error and the largest speed where h > 1 cm (a tenth of the depth).
///
/// The dry threshold is 1e-5 m = 1e-4 of the 0.1 m bowl depth: the 1 mm
/// field-scale default is 1 % of this laboratory case and dominates its
/// error (7.4 % instead of 4.5 % for Standard at 20², 19.7 % instead of
/// 17.3 % for WetDry).
fn thacker_one_period(formulation: SWEFormulation2D, n: usize) -> (f64, f64) {
    let case = Thacker::SWASHES;
    let period = 2.0 * PI / case.omega();
    let setup = Setup::new(Mesh2D::uniform_rectangle(0.0, case.l, 0.0, case.l, n, n), 2);
    let physics = setup
        .wet_dry_over(formulation, 1e-5, |x, y| case.bed(x, y))
        .build();

    let mut q = setup.fill(|x, y| case.exact(x, y, 0.0));
    let mass0 = setup.mass(&q);
    let mut lowest = f64::INFINITY;
    let sim = Simulation::new(physics, SSPRK3).with_cfl(1.0);
    let result = sim.run_with_callback(&mut q, 0.0, period, |q, _| {
        lowest = lowest.min(min_depth(q));
    });
    assert!(result.success, "{result:?}");

    let exact = |x, y| case.exact(x, y, period);
    let exact_mass = setup.integrate(&q, |_, x, y| exact(x, y).h);
    let depth_error = setup.integrate(&q, |s, x, y| (s.h - exact(x, y).h).abs()) / exact_mass;
    let speed = max_speed(&q, 1e-2);
    let mass_drift = (setup.mass(&q) - mass0).abs() / mass0;
    println!(
        "Thacker {formulation:?} P2 {n}²: L1(h) {depth_error:.3e}, max |u| {speed:.2} (exact {:.2}), mass drift {mass_drift:.1e}",
        case.eta * case.omega()
    );

    assert!(mass_drift < 1e-12, "mass drift {mass_drift:e}");
    assert!(lowest >= 0.0, "negative depth {lowest:e}");
    assert_eq!(sim.physics().negative_depth_clips(), 0);
    (depth_error, speed)
}

/// Errors at two resolutions must be below `bound` and converge faster than
/// first order; speeds where h > 1 cm stay near the exact 0.70 m/s.
fn check_thacker(formulation: SWEFormulation2D, bound: f64) {
    let (coarse, _) = thacker_one_period(formulation, 16);
    let (fine, speed) = thacker_one_period(formulation, 32);
    let rate = (coarse / fine).log2();
    assert!(fine < bound, "{formulation:?}: L1 depth error {fine:.3e}");
    assert!(rate > 1.5, "{formulation:?}: rate {rate:.2}");
    assert!(speed < 1.0, "{formulation:?}: max |u| {speed:.2} m/s");
}

#[test]
fn thacker_planar_oscillation_converges_standard() {
    // Measured L1 errors: 6.4 % / 1.9 % at 16² / 32² (rate 1.8), and 4.5 %
    // / 1.3 % / 0.45 % at 20² / 40² / 80² (9.2 % at 20² before P1.2).
    // Max |u| where h > 1 cm: 0.85 m/s at 32².
    check_thacker(SWEFormulation2D::Standard, 0.025);
}

#[test]
fn thacker_planar_oscillation_converges_wet_dry() {
    // Measured: 22.6 % / 7.3 % at 16² / 32² (rate 1.6), and 17.3 % / 4.8 %
    // / 1.3 % at 20² / 40² / 80², about 3× Standard: every element the moving
    // shoreline crosses drops to first-order subcell finite volumes.
    // Max |u| where h > 1 cm: 0.91 m/s at 32².
    check_thacker(SWEFormulation2D::WetDry, 0.09);
}

// ---------------------------------------------------------------------------
// Lake at rest with a shoreline
// ---------------------------------------------------------------------------

/// Plane beach B = −1 + x/50 on [0, 100] m, shoreline at x = 50, η = 0.
fn beach(x: f64, _y: f64) -> f64 {
    -1.0 + x / 50.0
}

fn run_beach_at_rest(
    formulation: SWEFormulation2D,
    order: usize,
    t_end: f64,
) -> (Setup, SWESolution2D) {
    let setup = Setup::new(
        Mesh2D::uniform_rectangle(0.0, 100.0, 0.0, 10.0, 23, 2),
        order,
    );
    let physics = setup
        .wet_dry_over(formulation, WetDryConfig::DEFAULT_H_DRY, beach)
        .build();
    let mut q = setup.fill(|x, y| SWEState2D::new((-beach(x, y)).max(0.0), 0.0, 0.0));
    let sim = Simulation::new(physics, SSPRK3).with_cfl(1.0);
    assert!(sim.run(&mut q, 0.0, t_end).success);
    assert_eq!(sim.physics().negative_depth_clips(), 0);
    (setup, q)
}

/// Largest surface error |η| over nodes deeper than 1 cm, and over the
/// offshore part x < 30 m.
fn beach_errors(setup: &Setup, q: &SWESolution2D) -> (f64, f64) {
    let (mut near, mut offshore) = (0.0_f64, 0.0_f64);
    for (k, i) in setup.nodes() {
        let (x, y) = setup.node_xy(k, i);
        let s = q.get_state(k, i);
        let eta = (s.h + beach(x, y)).abs();
        if s.h > 1e-2 {
            near = near.max(eta);
        }
        if x < 30.0 {
            offshore = offshore.max(eta);
        }
    }
    (near, offshore)
}

#[test]
fn lake_at_rest_with_shoreline_stays_near_rest() {
    // Standard. Regression bound, not exact balance: partially dry elements settle into
    // a small stationary circulation within ~10 s. Measured at 100 s: |u| ≤
    // 0.06 m/s where h > 1 cm, offshore |η| ≤ 2e-4 m. Before P1.2: 5.6 m/s
    // (P2) and the 20 m/s cap (P3), with 0.1–0.2 m surface errors.
    for order in [2, 3] {
        let (setup, q) = run_beach_at_rest(SWEFormulation2D::Standard, order, 100.0);
        let speed = max_speed(&q, 1e-2);
        let (near, offshore) = beach_errors(&setup, &q);
        println!("P{order}: |u| {speed:.2e} m/s, |η| {near:.2e} m (offshore {offshore:.2e} m)");
        assert!(speed < 0.1, "P{order}: |u| = {speed:.3e} m/s");
        assert!(offshore < 1e-3, "P{order}: offshore |η| = {offshore:.3e} m");
    }
}

#[test]
fn lake_at_rest_with_shoreline_is_exact() {
    // P1.7 gate. WetDry: subcell finite volumes with hydrostatic
    // reconstruction in the shoreline elements
    for order in 1..=4 {
        let (setup, q) = run_beach_at_rest(SWEFormulation2D::WetDry, order, 100.0);
        let speed = max_speed(&q, 0.0);
        let (near, _) = beach_errors(&setup, &q);
        assert!(
            speed < 1e-10 && near < 1e-10,
            "P{order}: |u| {speed:.2e}, |η| {near:.2e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Bottom friction
// ---------------------------------------------------------------------------

#[test]
fn implicit_manning_friction_is_stable_where_explicit_flips() {
    // Uniform 1 m/s flow in 1 cm of water on a periodic flat domain: the flux
    // divergence vanishes and only friction acts, d|u|/dt = −a|u|² with
    // a = g n²/h^{4/3} = 2.85 /m. At Δt = 1 s, Δt·a|u| = 2.85 is beyond
    // SSP-RK3's stability interval (2.51).
    let (h, u0, manning_n, dt) = (0.01_f64, 1.0, 0.025, 1.0);
    let a = G * manning_n * manning_n / h.powf(4.0 / 3.0);
    let setup = Setup::new(Mesh2D::uniform_periodic(0.0, 1000.0, 0.0, 1000.0, 2, 2), 2);
    let initial = setup.fill(|_, _| SWEState2D::from_primitives(h, u0, 0.0));
    let velocity = |q: &SWESolution2D| q.hu_data()[0] / q.h_data()[0];

    // Explicit source term: the momentum changes sign in the first step
    let explicit = setup
        .builder()
        .with_source(ManningFriction2D::new(G, manning_n))
        .build();
    let mut q = initial.clone();
    Simulation::new(explicit, SSPRK3)
        .with_dt_max(dt)
        .run(&mut q, 0.0, dt);
    assert!(
        velocity(&q) < 0.0,
        "explicit friction kept u = {}",
        velocity(&q)
    );

    // Point-implicit: positive, monotone, and close to the exact decay
    let implicit = setup
        .builder()
        .with_implicit_friction(ManningFriction2D::new(G, manning_n))
        .build();
    let sim = Simulation::new(implicit, SSPRK3).with_dt_max(dt);
    let mut q = initial.clone();
    let mut previous = u0;
    for step in 1..=10 {
        sim.run(&mut q, 0.0, dt);
        let u = velocity(&q);
        let exact = u0 / (1.0 + a * u0 * step as f64 * dt);
        assert!(u > 0.0 && u < previous, "step {step}: {previous} -> {u}");
        // First order in Δt·Λ ≈ 3 here: 32 % above the exact value after one
        // step, 13 % after ten
        let ratio = u / exact;
        assert!(
            (1.0..1.35).contains(&ratio),
            "step {step}: u = {u}, exact {exact}"
        );
        previous = u;
        // Still uniform, and the depth is untouched
        assert!(q.h_data().iter().all(|&hq| (hq - h).abs() < 1e-15));
        assert!(
            q.hu_data()
                .iter()
                .all(|&m| (m - q.hu_data()[0]).abs() < 1e-15)
        );
    }
}

#[test]
fn implicit_friction_converges_to_exact_decay() {
    // Same setup, t = 2 s: the relaxed friction is first-order accurate
    let (h, u0, manning_n, t_end) = (0.01_f64, 1.0, 0.025, 2.0);
    let a = G * manning_n * manning_n / h.powf(4.0 / 3.0);
    let exact = u0 / (1.0 + a * u0 * t_end);
    let setup = Setup::new(Mesh2D::uniform_periodic(0.0, 1000.0, 0.0, 1000.0, 2, 2), 1);
    let error = |dt: f64| {
        let physics = setup
            .builder()
            .with_implicit_friction(ManningFriction2D::new(G, manning_n))
            .build();
        let mut q = setup.fill(|_, _| SWEState2D::from_primitives(h, u0, 0.0));
        Simulation::new(physics, SSPRK3)
            .with_dt_max(dt)
            .run(&mut q, 0.0, t_end);
        (q.hu_data()[0] / h - exact).abs()
    };
    let (coarse, fine) = (error(0.02), error(0.01));
    let rate = (coarse / fine).log2();
    assert!(
        fine < 2e-3 && rate > 0.9,
        "errors {coarse:.2e}, {fine:.2e}, rate {rate:.2}"
    );
}

// ---------------------------------------------------------------------------
// Positivity CFL
// ---------------------------------------------------------------------------

#[test]
fn simulation_caps_cfl_at_positivity_bound() {
    let setup = Setup::new(Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4), 2);
    let q0 = setup.fill(|x, _| SWEState2D::new(1.0 + 0.1 * x, 0.0, 0.0));
    // Time of the first step, from the callback (called at t_start, then
    // after every step)
    let first_dt = |physics| {
        let mut times = Vec::new();
        let mut q = q0.clone();
        Simulation::new(physics, SSPRK3)
            .with_cfl(1.0)
            .run_with_callback(&mut q, 0.0, 1e3, |_, t| {
                if times.len() < 2 {
                    times.push(t);
                }
            });
        times[1] - times[0]
    };

    let wet = setup.builder().build();
    let expected_wet = wet.compute_dt(&q0, 1.0);
    assert_eq!(first_dt(wet), expected_wet);

    let wet_dry = setup.builder().with_wet_dry_correction(true).build();
    let expected = wet_dry.compute_dt(&q0, positivity_cfl_swe_2d(2));
    assert_eq!(first_dt(wet_dry), expected);
    assert!(expected < expected_wet);
}
