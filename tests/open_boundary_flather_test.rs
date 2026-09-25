//! Regression tests for Flather-type open boundary conditions.
//!
//! In the weak (ghost-state + upwind flux) DG setting, the boundary flux is
//! computed by the Riemann solver from the interior trace and the ghost state.
//! For the linearised 1D problem with Riemann invariants
//!
//! ```text
//! w± = u_n ± sqrt(g/H) η
//! ```
//!
//! the upwind flux takes the outgoing invariant `w+` from the interior and the
//! incoming invariant `w−` from the ghost. The Flather (1976) condition
//! `u_n = u_n,ext + sqrt(g/H) (η − η_ext)` is exactly the statement
//! `w− = w−_ext`, so the correct ghost is simply the external state
//! `(η_ext, u_n,ext)`.
//!
//! The Flather-type BCs used to build the ghost normal velocity as
//! `u_n,ext + sqrt(g/H) (η_int − η_ext)`, which applies the characteristic
//! relation a second time. The incoming invariant then picks up
//! `sqrt(g/H) (η_int − η_ext)`, giving a reflection coefficient of −1/3 for an
//! outgoing wave (−0.29 at a nesting weight of 0.8).
//!
//! These tests run a quasi-1D channel (one element across, free-slip walls on
//! the sides) and check that
//!
//! 1. an outgoing pulse leaves through the open boundary with < 1% reflection,
//! 2. a progressive wave forced through the open boundary is delivered at the
//!    prescribed amplitude everywhere in the channel.

use dg_rs::boundary::{
    Chapman2D, ChapmanFlather2D, Flather2D, HarmonicFlather2D, MultiBoundaryCondition2D,
    NestingBC2D, Reflective2D, SWEBoundaryCondition2D, TSTConfig, TSTOBC2D,
};
use dg_rs::io::{BoundaryTimeSeries, TimeSeriesRecord};
use dg_rs::mesh::{Bathymetry2D, BoundaryTag};
use dg_rs::time::{SSPRK3, TimeIntegrator};
use dg_rs::types::ElementIndex;
use dg_rs::{
    DGOperators2D, GeometricFactors2D, Mesh2D, SWE2DRhsConfig, SWESolution2D, SWEState2D,
    ShallowWater2D, compute_dt_swe_2d, compute_rhs_swe_2d,
};

const G: f64 = 9.81;
/// Still-water depth (m); bed elevation is B = −H0.
const H0: f64 = 10.0;
/// Channel length (m).
const LENGTH: f64 = 500.0;
/// Number of elements along the channel.
const NX: usize = 25;
/// Element size (m); the channel is one square element wide.
const DX: f64 = LENGTH / NX as f64;
/// Polynomial order.
const ORDER: usize = 3;
const CFL: f64 = 0.5;

fn celerity() -> f64 {
    (G * H0).sqrt()
}

/// Quasi-1D channel `[0, LENGTH] × [0, DX]` with walls on the long sides.
struct Channel {
    mesh: Mesh2D,
    ops: DGOperators2D,
    geom: GeometricFactors2D,
    equation: ShallowWater2D,
    bathymetry: Bathymetry2D,
}

impl Channel {
    /// Build the channel with the given tags on the west (x = 0) and east
    /// (x = LENGTH) ends.
    fn new(west: BoundaryTag, east: BoundaryTag) -> Self {
        // Side order: [south, east, north, west]
        let mesh = Mesh2D::uniform_rectangle_with_sides(
            0.0,
            LENGTH,
            0.0,
            DX,
            NX,
            1,
            [BoundaryTag::Wall, east, BoundaryTag::Wall, west],
        );
        let ops = DGOperators2D::new(ORDER);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bathymetry = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -H0);
        Self {
            mesh,
            ops,
            geom,
            equation,
            bathymetry,
        }
    }

    fn x(&self, k: usize, i: usize) -> f64 {
        let (r, s) = (self.ops.nodes_r[i], self.ops.nodes_s[i]);
        self.mesh.reference_to_physical(ElementIndex::new(k), r, s)[0]
    }

    /// Initial state with surface elevation η(x) and velocity u(x).
    fn initial_state(&self, eta: impl Fn(f64) -> f64, u: impl Fn(f64) -> f64) -> SWESolution2D {
        let mut q = SWESolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
        for k in 0..self.mesh.n_elements {
            for i in 0..self.ops.n_nodes {
                let x = self.x(k, i);
                let h = H0 + eta(x);
                q.set_state(
                    ElementIndex::new(k),
                    i,
                    SWEState2D::from_primitives(h, u(x), 0.0),
                );
            }
        }
        q
    }

    /// Advance `q` from t = 0 to `t_end` with SSP-RK3, calling `observe`
    /// after every step.
    fn run<BC: SWEBoundaryCondition2D>(
        &self,
        bc: &BC,
        q: &mut SWESolution2D,
        t_end: f64,
        mut observe: impl FnMut(&SWESolution2D, f64),
    ) {
        let config = SWE2DRhsConfig::new(&self.equation, bc)
            .with_coriolis(false)
            .with_bathymetry(&self.bathymetry);

        // The linear-regime wave speed is ~constant, so a single dt suffices.
        let dt_cfl = compute_dt_swe_2d(q, &self.mesh, &self.geom, &self.equation, ORDER, CFL);
        let n_steps = (t_end / dt_cfl).ceil() as usize;
        let dt = t_end / n_steps as f64;

        let mut t = 0.0;
        for _ in 0..n_steps {
            SSPRK3.step(q, dt, t, |s, stage_t| {
                compute_rhs_swe_2d(s, &self.mesh, &self.ops, &self.geom, &config, stage_t)
            });
            t += dt;
            observe(q, t);
        }
    }
}

// ---------------------------------------------------------------------------
// Outgoing pulse: reflection coefficient
// ---------------------------------------------------------------------------

/// Pulse amplitude (m). a/H = 1e-3 keeps the problem in the linear regime.
const PULSE_AMP: f64 = 0.01;
const PULSE_X0: f64 = 250.0;
const PULSE_WIDTH: f64 = 30.0;

/// Launch a right-going Gaussian pulse from mid-channel and return the largest
/// |η| left in the channel (relative to the pulse amplitude) once the pulse has
/// fully left through the east boundary. Any remaining signal is reflection.
///
/// Both channel ends carry `BoundaryTag::Open`, handled by `bc`.
fn outgoing_pulse_reflection<BC: SWEBoundaryCondition2D>(bc: &BC) -> f64 {
    let ch = Channel::new(BoundaryTag::Open, BoundaryTag::Open);

    // Exact right-going simple wave: the left-going invariant
    // u − 2 sqrt(g h) is uniform, so no left-going signal is launched.
    let eta0 = |x: f64| PULSE_AMP * (-((x - PULSE_X0) / PULSE_WIDTH).powi(2)).exp();
    let u0 = |x: f64| 2.0 * ((G * (H0 + eta0(x))).sqrt() - celerity());
    let mut q = ch.initial_state(eta0, u0);

    // The pulse centre reaches the east end at ~25 s and its tail
    // (4 widths behind) has left by ~37 s. By 50 s a reflected pulse would
    // sit near mid-channel.
    let t_end = 50.0;
    ch.run(bc, &mut q, t_end, |_, _| {});

    let mut max_eta: f64 = 0.0;
    for k in 0..ch.mesh.n_elements {
        for i in 0..ch.ops.n_nodes {
            let eta = q.get_state(ElementIndex::new(k), i).h - H0;
            max_eta = max_eta.max(eta.abs());
        }
    }
    max_eta / PULSE_AMP
}

/// Maximum allowed reflected amplitude (fraction of the incident pulse).
const MAX_REFLECTION: f64 = 0.01;

fn assert_no_reflection(name: &str, reflection: f64) {
    println!("{name}: reflected amplitude = {:.4}%", 100.0 * reflection);
    assert!(
        reflection < MAX_REFLECTION,
        "{name}: outgoing pulse reflected with {:.1}% of its amplitude \
         (limit {:.1}%) — ghost state double-applies the Flather relation?",
        100.0 * reflection,
        100.0 * MAX_REFLECTION
    );
}

fn with_walls(open: &dyn SWEBoundaryCondition2D) -> MultiBoundaryCondition2D<'_> {
    static WALL: Reflective2D = Reflective2D { h_min: 1e-6 };
    MultiBoundaryCondition2D::new(&WALL).with_open(open)
}

#[test]
fn flather2d_outgoing_pulse_does_not_reflect() {
    let flather = Flather2D::new(|_, _, _| 0.0, H0);
    let bc = with_walls(&flather);
    assert_no_reflection("Flather2D", outgoing_pulse_reflection(&bc));
}

#[test]
fn harmonic_flather2d_outgoing_pulse_does_not_reflect() {
    let flather = HarmonicFlather2D::new(vec![], H0);
    let bc = with_walls(&flather);
    assert_no_reflection("HarmonicFlather2D", outgoing_pulse_reflection(&bc));
}

#[test]
fn chapman_flather2d_outgoing_pulse_does_not_reflect() {
    // Library defaults (h_ref = 10 m, no dt) with the grid spacing as dx.
    let cf = ChapmanFlather2D::new(|_, _, _| (0.0, 0.0, 0.0), DX);
    let bc = with_walls(&cf);
    assert_no_reflection("ChapmanFlather2D", outgoing_pulse_reflection(&bc));
}

#[test]
fn tst_obc_outgoing_pulse_does_not_reflect() {
    let tst = TSTOBC2D::new(TSTConfig {
        mean_elevation: 0.0,
        constituents: vec![],
        h_ref: H0,
        dx: DX,
        subtidal_weight: 1.0,
        h_min: 1e-6,
    });
    let bc = with_walls(&tst);
    assert_no_reflection("TSTOBC2D", outgoing_pulse_reflection(&bc));
}

/// Parent model at rest: h = H0 (η = 0), zero velocity.
fn parent_at_rest() -> BoundaryTimeSeries {
    BoundaryTimeSeries::from_records(vec![
        TimeSeriesRecord::from_primitives(0.0, H0, 0.0, 0.0),
        TimeSeriesRecord::from_primitives(1.0e6, H0, 0.0, 0.0),
    ])
    .unwrap()
}

#[test]
fn nesting_bc_outgoing_pulse_does_not_reflect() {
    // Default Flather mode (weight 1.0) and the 0.8 weight used by the
    // Frøya example must both be non-reflecting, as must Dirichlet mode.
    for (name, nesting) in [
        (
            "NestingBC2D (weight 1.0)",
            NestingBC2D::new(parent_at_rest()),
        ),
        (
            "NestingBC2D (weight 0.8)",
            NestingBC2D::new(parent_at_rest()).with_flather_weight(0.8),
        ),
        (
            "NestingBC2D (Dirichlet)",
            NestingBC2D::new(parent_at_rest()).without_flather(),
        ),
    ] {
        let bc = with_walls(&nesting);
        assert_no_reflection(name, outgoing_pulse_reflection(&bc));
    }
}

/// `Chapman2D` is not a Flather BC (it never touches the normal velocity),
/// but its ghost blends interior elevation into the incoming Riemann
/// invariant. With the old fallback for a missing time step (c·dt/dx = 1,
/// α = 1/2) that feedback made the discretisation blow up within ~75 s here.
/// Without a dt it must now fall back to a stable elevation clamp (α = 1),
/// which reflects the pulse (≈ −1) but keeps it bounded.
#[test]
fn chapman2d_without_dt_stays_bounded() {
    let chapman = Chapman2D::new(|_, _, _| 0.0, DX);
    let bc = with_walls(&chapman);

    let ch = Channel::new(BoundaryTag::Open, BoundaryTag::Open);
    let eta0 = |x: f64| PULSE_AMP * (-((x - PULSE_X0) / PULSE_WIDTH).powi(2)).exp();
    let u0 = |x: f64| 2.0 * ((G * (H0 + eta0(x))).sqrt() - celerity());
    let mut q = ch.initial_state(eta0, u0);

    // Three boundary encounters of the (reflected) pulse.
    let mut max_eta: f64 = 0.0;
    ch.run(&bc, &mut q, 100.0, |q, _| {
        for k in 0..ch.mesh.n_elements {
            for i in 0..ch.ops.n_nodes {
                let eta = q.get_state(ElementIndex::new(k), i).h - H0;
                // NaN must fail the bound below.
                max_eta = if eta.is_nan() {
                    f64::INFINITY
                } else {
                    max_eta.max(eta.abs())
                };
            }
        }
    });

    println!(
        "Chapman2D (no dt): max |η| = {:.3} × pulse amplitude",
        max_eta / PULSE_AMP
    );
    assert!(
        max_eta <= 1.05 * PULSE_AMP,
        "Chapman2D without dt grew to {:.3e} × the pulse amplitude",
        max_eta / PULSE_AMP
    );
}

// ---------------------------------------------------------------------------
// Forced progressive wave: delivered amplitude
// ---------------------------------------------------------------------------

/// Prescribed wave amplitude (m).
const WAVE_AMP: f64 = 0.01;
/// Wavelength (m): 6.25 elements (25 nodes) per wavelength at order 3.
const WAVELENGTH: f64 = 125.0;

fn wave_period() -> f64 {
    WAVELENGTH / celerity()
}

/// External state of an eastward progressive wave at the west end (x = 0):
/// η_ext = A sin(ωt), u_ext = sqrt(g/H) η_ext for t ≥ 0, at rest before.
fn eastward_wave_at_west(t: f64) -> (f64, f64) {
    let omega = 2.0 * std::f64::consts::PI / wave_period();
    let eta = if t > 0.0 {
        WAVE_AMP * (omega * t).sin()
    } else {
        0.0
    };
    (eta, (G / H0).sqrt() * eta)
}

/// Force an eastward progressive wave through the west boundary (tag
/// `TidalForcing`) of a channel whose east end is a radiating open boundary
/// (tag `Open`), and return the (min, max) over the channel interior of the
/// wave amplitude, relative to the prescribed amplitude, over the last period.
///
/// A correct Flather BC delivers the prescribed amplitude and lets it leave
/// through the east end, so the channel carries a clean progressive wave of
/// uniform amplitude. Reflections at either end superpose a standing
/// component whose envelope varies along the channel.
fn delivered_amplitude_range<BC: SWEBoundaryCondition2D>(bc: &BC) -> (f64, f64) {
    let ch = Channel::new(BoundaryTag::TidalForcing, BoundaryTag::Open);
    let mut q = ch.initial_state(|_| 0.0, |_| 0.0);

    // Let the wave cross the channel, reflect off the east end if the BC is
    // reflective, and come back across once more before measuring.
    let period = wave_period();
    let t_end = 2.0 * LENGTH / celerity() + 3.0 * period;
    let t_measure = t_end - period;

    // Sample the interior away from the boundaries (50 m ≤ x ≤ 450 m).
    let samples: Vec<(usize, usize)> = (0..ch.mesh.n_elements)
        .flat_map(|k| (0..ch.ops.n_nodes).map(move |i| (k, i)))
        .filter(|&(k, i)| {
            let x = ch.x(k, i);
            (50.0..=450.0).contains(&x)
        })
        .collect();
    let mut amplitude = vec![0.0_f64; samples.len()];

    ch.run(bc, &mut q, t_end, |q, t| {
        if t >= t_measure {
            for (a, &(k, i)) in amplitude.iter_mut().zip(&samples) {
                let eta = q.get_state(ElementIndex::new(k), i).h - H0;
                *a = a.max(eta.abs());
            }
        }
    });

    let min = amplitude.iter().copied().fold(f64::INFINITY, f64::min);
    let max = amplitude.iter().copied().fold(0.0, f64::max);
    (min / WAVE_AMP, max / WAVE_AMP)
}

fn assert_full_amplitude(name: &str, (min, max): (f64, f64)) {
    println!("{name}: delivered amplitude in [{min:.3}, {max:.3}] × prescribed");
    assert!(
        min >= 0.95 && max <= 1.05,
        "{name}: delivered amplitude ranges over [{min:.3}, {max:.3}] of the \
         prescribed amplitude (expected within [0.95, 1.05])"
    );
}

#[test]
fn chapman_flather2d_delivers_prescribed_progressive_wave() {
    // One BC for both ends: forcing at the west end, quiescent sea at the
    // east end. External state is (η, u_n, u_t) with u_n along the outward
    // normal, which points in −x at the west end.
    let cf = ChapmanFlather2D::new(
        |x, _y, t| {
            if x < 0.5 * LENGTH {
                let (eta, u) = eastward_wave_at_west(t);
                (eta, -u, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        },
        DX,
    )
    .with_h_ref(H0);
    static WALL: Reflective2D = Reflective2D { h_min: 1e-6 };
    let bc = MultiBoundaryCondition2D::new(&WALL)
        .with_tidal(&cf)
        .with_open(&cf);
    assert_full_amplitude("ChapmanFlather2D", delivered_amplitude_range(&bc));
}

#[test]
fn nesting_bc_delivers_prescribed_progressive_wave() {
    // Parent model output sampled every 0.1 s at the west end.
    let t_end = 2.0 * LENGTH / celerity() + 4.0 * wave_period();
    let n_records = (t_end / 0.1).ceil() as usize + 1;
    let records = (0..n_records)
        .map(|n| {
            let t = n as f64 * 0.1;
            let (eta, u) = eastward_wave_at_west(t);
            TimeSeriesRecord::from_primitives(t, H0 + eta, u, 0.0)
        })
        .collect();
    let forcing = NestingBC2D::new(BoundaryTimeSeries::from_records(records).unwrap());
    let radiating = NestingBC2D::new(parent_at_rest());

    static WALL: Reflective2D = Reflective2D { h_min: 1e-6 };
    let bc = MultiBoundaryCondition2D::new(&WALL)
        .with_tidal(&forcing)
        .with_open(&radiating);
    assert_full_amplitude("NestingBC2D", delivered_amplitude_range(&bc));
}
