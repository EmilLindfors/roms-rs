//! Gates for 3D particle tracking (`ParticleTracker3D`, TODO F.2 Stage B).
//!
//! - Visser's (1997) well-mixed test: under a diffusivity that varies with
//!   depth, a uniform population stays uniform with the drift `K'Δt`; without
//!   it, particles collect where `K` is small.
//! - Advection through a sheared flow with a σ-velocity is followed exactly
//!   (RK4 is exact for the quadratic-in-time trajectory).
//! - Sinking particles reach the bed when they should, and settle there or
//!   reflect.

use dg_rs::mesh::Mesh2D;
use dg_rs::operators::DGOperators2D;
use dg_rs::particles::{Particle3D, ParticleStatus, ParticleTracker3D, ParticleVelocity3D};
use dg_rs::types::ElementIndex;

/// `K(z, D)` and `∂K/∂z` of a diffusivity profile.
type Profile = fn(f64, f64) -> (f64, f64);

/// One tracking step of a tracker through some field.
type StepFn<'a> = dyn Fn(&ParticleTracker3D, &mut [Particle3D], f64, f64) + 'a;

/// A horizontally uniform water column: depth `depth`, velocity
/// `u(σ) = u0 + u1 σ`, `v`, a constant σ-velocity, and a diffusivity
/// `K(z)` with its derivative.
struct Column {
    depth: f64,
    u: [f64; 2],
    v: f64,
    sigma_rate: f64,
    diffusivity: Option<Profile>,
}

impl Column {
    fn still(depth: f64) -> Self {
        Self {
            depth,
            u: [0.0, 0.0],
            v: 0.0,
            sigma_rate: 0.0,
            diffusivity: None,
        }
    }
}

impl ParticleVelocity3D for Column {
    fn velocity(&self, _: ElementIndex, _: &[f64], sigma: f64, _: f64) -> [f64; 3] {
        [self.u[0] + self.u[1] * sigma, self.v, self.sigma_rate]
    }

    fn depth(&self, _: ElementIndex, _: &[f64], _: f64) -> f64 {
        self.depth
    }

    fn diffusivity(&self, _: ElementIndex, _: &[f64], sigma: f64, _: f64) -> Option<(f64, f64)> {
        self.diffusivity.map(|k| k(sigma * self.depth, self.depth))
    }
}

/// Visser-type profile over depth `H` (ζ = −z the depth below the surface):
/// `K = 10⁻³ + 0.02 (ζ/H)(1 − ζ/H)²`, smallest at the surface and the bed,
/// largest at H/3, with `K' ≠ 0` at the surface.
fn visser_profile(z: f64, depth: f64) -> (f64, f64) {
    let x = -z / depth;
    let k = 1e-3 + 0.02 * x * (1.0 - x) * (1.0 - x);
    // dK/dz = −dK/dζ
    let dk_dz = -(0.02 / depth) * (1.0 - x) * (1.0 - 3.0 * x);
    (k, dk_dz)
}

/// The same `K` with its derivative dropped: the naive walk.
fn visser_profile_without_drift(z: f64, depth: f64) -> (f64, f64) {
    (visser_profile(z, depth).0, 0.0)
}

fn mesh() -> (Mesh2D, DGOperators2D) {
    (
        Mesh2D::uniform_periodic(0.0, 10_000.0, 0.0, 10_000.0, 4, 4),
        DGOperators2D::new(1),
    )
}

/// χ² of the particles' σ-levels against a uniform distribution, `bins`
/// bins.
fn chi_squared(particles: &[Particle3D], bins: usize) -> f64 {
    let mut counts = vec![0.0; bins];
    for p in particles {
        let b = ((-p.sigma() * bins as f64) as usize).min(bins - 1);
        counts[b] += 1.0;
    }
    let expected = particles.len() as f64 / bins as f64;
    counts
        .iter()
        .map(|c| (c - expected) * (c - expected) / expected)
        .sum()
}

/// Gate: Visser's well-mixed condition. 2000 particles spread evenly over
/// a 10 m column, 12 h (≈ 1.2 vertical mixing times H²/K̄) at 20 s steps
/// (`Δt·max|K''|` ≈ 0.02). With the drift the population stays uniform:
/// χ² over 20 bins below the 0.1 % critical value of 43.8 (19 degrees of
/// freedom). Without the drift it is far from uniform.
#[test]
fn a_well_mixed_population_stays_well_mixed() {
    let (mesh, ops) = mesh();
    let run = |profile: Profile| {
        let tracker = ParticleTracker3D::new(&mesh, &ops)
            .with_vertical_random_walk()
            .with_seed(7);
        let field = Column {
            diffusivity: Some(profile),
            ..Column::still(10.0)
        };
        let n = 2000;
        let mut particles: Vec<Particle3D> = (0..n)
            .map(|i| {
                let sigma = -(i as f64 + 0.5) / n as f64;
                tracker
                    .release(i as u64, [5_000.0, 5_000.0], sigma, 0.0)
                    .unwrap()
            })
            .collect();
        let dt = 20.0;
        for step in 0..(12 * 3600 / 20) {
            tracker.step(&mut particles, &field, step as f64 * dt, dt);
        }
        for p in &particles {
            assert_eq!(p.status(), ParticleStatus::Active);
            assert!((-1.0..=0.0).contains(&p.sigma()), "σ {}", p.sigma());
            // No horizontal motion in still water
            assert_eq!(p.position(), [5_000.0, 5_000.0]);
        }
        chi_squared(&particles, 20)
    };
    let with_drift = run(visser_profile);
    let without = run(visser_profile_without_drift);
    // Measured χ² 7.3 with the drift, 417 without
    assert!(
        with_drift < 43.8,
        "the well-mixed population lost uniformity: χ² = {with_drift:.1}"
    );
    assert!(
        without > 5.0 * 43.8,
        "the naive walk should collect particles where K is small: χ² = {without:.1}"
    );
}

/// Gate: a sheared flow `u = u₀ + u₁σ` with a constant σ-velocity `c`
/// carries a particle to `σ(t) = σ₀ + ct`,
/// `x(t) = x₀ + u₀t + u₁(σ₀t + ct²/2)`, `y(t) = y₀ + vt`, which RK4
/// follows exactly (across element faces and the periodic boundary).
#[test]
fn a_sheared_flow_with_a_sigma_velocity_is_followed_exactly() {
    let (mesh, ops) = mesh();
    let tracker = ParticleTracker3D::new(&mesh, &ops);
    let field = Column {
        depth: 30.0,
        u: [0.4, 0.3],
        v: -0.15,
        sigma_rate: 1e-4,
        diffusivity: None,
    };
    let (x0, y0, s0) = (1_234.0, 8_765.0, -0.9);
    let mut particles = vec![tracker.release(1, [x0, y0], s0, 0.0).unwrap()];
    let (dt, steps) = (100.0, 60);
    for n in 0..steps {
        tracker.step(&mut particles, &field, n as f64 * dt, dt);
    }
    let t = dt * steps as f64;
    let sigma = s0 + 1e-4 * t;
    let x = (x0 + 0.4 * t + 0.3 * (s0 * t + 0.5 * 1e-4 * t * t)).rem_euclid(10_000.0);
    let y = (y0 - 0.15 * t).rem_euclid(10_000.0);
    let p = &particles[0];
    let [px, py] = p.position();
    assert!(
        (p.sigma() - sigma).abs() < 1e-12,
        "σ {} vs {sigma}",
        p.sigma()
    );
    assert!((px - x).abs() < 1e-8, "x {px} vs {x}");
    assert!((py - y).abs() < 1e-8, "y {py} vs {y}");
}

/// Gate: particles sinking at 1 cm/s from 1 m below the surface of a 10 m
/// column reach the bed after 900 s. With bed settling they stay there
/// ([`ParticleStatus::Settled`], at σ = −1, no further motion); without,
/// they reflect off the bed (and, still sinking, bounce along it).
#[test]
fn sinking_particles_settle_on_the_bed() {
    let (mesh, ops) = mesh();
    let field = Column {
        u: [0.1, 0.0],
        ..Column::still(10.0)
    };
    let release = |tracker: &ParticleTracker3D| {
        vec![tracker.release(3, [5_000.0, 5_000.0], -0.1, -0.01).unwrap()]
    };
    let dt = 60.0;

    let settling = ParticleTracker3D::new(&mesh, &ops).with_bed_settling();
    let mut particles = release(&settling);
    let mut landed = None;
    for n in 0..30 {
        settling.step(&mut particles, &field, n as f64 * dt, dt);
        if landed.is_none() && particles[0].status() == ParticleStatus::Settled {
            landed = Some((n + 1) as f64 * dt);
        }
    }
    let p = &particles[0];
    let landed = landed.expect("the particle never settled");
    // σ reaches −1 exactly at 900 s; the step that would pass it settles
    assert!((900.0..=960.0).contains(&landed), "settled at {landed} s");
    assert_eq!(p.sigma(), -1.0);
    assert!((p.depth_below_surface(10.0) - 10.0).abs() < 1e-12);
    // It drifted with the current until it landed, then stopped
    let x = p.position()[0];
    assert!((x - (5_000.0 + 0.1 * landed)).abs() < 1e-6, "x {x}");

    let reflecting = ParticleTracker3D::new(&mesh, &ops);
    let mut particles = release(&reflecting);
    for n in 0..16 {
        reflecting.step(&mut particles, &field, n as f64 * dt, dt);
    }
    // 960 s: 0.6 m below the bed, reflected to 0.6 m above it
    let p = &particles[0];
    assert_eq!(p.status(), ParticleStatus::Active);
    assert!((p.sigma() + 0.94).abs() < 1e-12, "σ {}", p.sigma());
    for n in 16..40 {
        reflecting.step(&mut particles, &field, n as f64 * dt, dt);
        assert!((-1.0..=-0.94 + 1e-12).contains(&particles[0].sigma()));
    }
}

/// Gate: the well-mixed condition with the model's own fields. A
/// `Solution3D` at rest on 20 σ-levels holds the parabolic diffusivity of
/// a log-layer channel, `K = κu*|z|(1 − |z|/H) + K₀`, at its w-points
/// (`Solution3D::eddy_diffusivity`), sampled through
/// `Solution3DVelocity`: piecewise linear in z, so the drift `K'` jumps at
/// every w-point. A uniform population stays uniform; with the same field
/// stripped of its drift it does not.
#[test]
fn the_model_diffusivity_keeps_a_well_mixed_population_well_mixed() {
    use dg_rs::mesh::Bathymetry2D;
    use dg_rs::particles::Solution3DVelocity;
    use dg_rs::solver::state::Solution3D;
    use dg_rs::vertical::{SigmaGrid, UniformStretching};

    let (mesh, ops) = mesh();
    let (depth, n_levels, u_star) = (10.0, 20, 0.01);
    let sigma = SigmaGrid::new(n_levels, UniformStretching);
    let bathymetry = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -depth);
    let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, n_levels);
    for column in state.eddy_diffusivity.chunks_exact_mut(n_levels + 1) {
        for (k, &s) in column.iter_mut().zip(sigma.sigma_w()) {
            let z = -s * depth;
            *k = 0.41 * u_star * z * (1.0 - z / depth) + 1e-5;
        }
    }
    let field = Solution3DVelocity::steady(&state, &sigma, &bathymetry, 0.01);

    /// `field` with the drift dropped: the naive walk
    struct WithoutDrift<'a>(Solution3DVelocity<'a>);
    impl ParticleVelocity3D for WithoutDrift<'_> {
        fn velocity(&self, k: ElementIndex, w: &[f64], s: f64, t: f64) -> [f64; 3] {
            self.0.velocity(k, w, s, t)
        }
        fn depth(&self, k: ElementIndex, w: &[f64], t: f64) -> f64 {
            self.0.depth(k, w, t)
        }
        fn diffusivity(&self, k: ElementIndex, w: &[f64], s: f64, t: f64) -> Option<(f64, f64)> {
            self.0.diffusivity(k, w, s, t).map(|(k, _)| (k, 0.0))
        }
    }

    let run = |field: &StepFn<'_>| {
        let tracker = ParticleTracker3D::new(&mesh, &ops)
            .with_vertical_random_walk()
            .with_seed(11);
        let n = 2000;
        let mut particles: Vec<Particle3D> = (0..n)
            .map(|i| {
                let s = -(i as f64 + 0.5) / n as f64;
                tracker
                    .release(i as u64, [3_000.0, 7_000.0], s, 0.0)
                    .unwrap()
            })
            .collect();
        // Mixing time H²/K̄ ≈ 1.5e4 s (K̄ = κu*H/6); 6 h at 5 s steps
        let dt = 5.0;
        for step in 0..(6 * 3600 / 5) {
            field(&tracker, &mut particles, step as f64 * dt, dt);
        }
        chi_squared(&particles, 20)
    };
    let with_drift = run(&|tracker, particles, t, dt| tracker.step(particles, &field, t, dt));
    let naive = WithoutDrift(field);
    let without = run(&|tracker, particles, t, dt| tracker.step(particles, &naive, t, dt));
    // Measured χ² 5.6 with the drift, 5435 without
    assert!(
        with_drift < 43.8,
        "the well-mixed population lost uniformity: χ² = {with_drift:.1}"
    );
    assert!(
        without > 5.0 * 43.8,
        "the naive walk should collect particles where K is small: χ² = {without:.1}"
    );
}
