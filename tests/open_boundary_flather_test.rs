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
//! `Chapman2D` fixes the same incoming invariant with zero external velocity,
//! through the ghost elevation instead of the ghost velocity. It used to blend
//! interior elevation into the ghost (`α η_ext + (1 − α) η_int`), which fed the
//! outgoing wave back into the incoming invariant: ~99% reflection for
//! α ≥ 0.7, blow-up for α ≤ 0.6. `Radiation2D` used to copy the interior
//! state into the ghost for every subcritical state, i.e. took the incoming
//! invariant from the interior; that feeds energy in through the incoming
//! characteristic and an outgoing pulse grew without bound.
//!
//! These tests run a quasi-1D channel (one element across, free-slip walls on
//! the sides) and check that
//!
//! 1. an outgoing pulse leaves through the open boundary with < 1% reflection,
//!    and (for `Chapman2D`/`Radiation2D`) the channel stays quiet long after;
//! 2. a progressive wave forced through the open boundary is delivered at the
//!    prescribed amplitude everywhere in the channel (half of it for the
//!    elevation-only `Chapman2D`, which has no external velocity).

use dg_rs::boundary::{
    Chapman2D, ChapmanFlather2D, Flather2D, HarmonicFlather2D, MultiBoundaryCondition2D,
    NestingBC2D, Radiation2D, Reflective2D, SWEBoundaryCondition2D, TSTConfig, TSTOBC2D,
};
use dg_rs::flux::SWEFluxType2D;
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

    /// Largest |η| over all nodes (∞ if any node is NaN).
    fn max_abs_eta(&self, q: &SWESolution2D) -> f64 {
        let mut max_eta: f64 = 0.0;
        for k in 0..self.mesh.n_elements {
            for i in 0..self.ops.n_nodes {
                let eta = q.get_state(ElementIndex::new(k), i).h - H0;
                if eta.is_nan() {
                    return f64::INFINITY;
                }
                max_eta = max_eta.max(eta.abs());
            }
        }
        max_eta
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

    /// Advance `q` from t = 0 to `t_end` with SSP-RK3 and the default (Roe)
    /// flux, calling `observe` after every step.
    fn run<BC: SWEBoundaryCondition2D>(
        &self,
        bc: &BC,
        q: &mut SWESolution2D,
        t_end: f64,
        observe: impl FnMut(&SWESolution2D, f64),
    ) {
        self.run_with_flux(bc, SWEFluxType2D::Roe, q, t_end, observe);
    }

    /// As [`Channel::run`], with the given numerical flux.
    fn run_with_flux<BC: SWEBoundaryCondition2D>(
        &self,
        bc: &BC,
        flux_type: SWEFluxType2D,
        q: &mut SWESolution2D,
        t_end: f64,
        mut observe: impl FnMut(&SWESolution2D, f64),
    ) {
        let config = SWE2DRhsConfig::new(&self.equation, bc)
            .with_coriolis(false)
            .with_flux_type(flux_type)
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

/// The pulse centre reaches the east end at ~25 s and its tail (4 widths
/// behind) has left by ~37 s. By 50 s a reflected pulse would sit near
/// mid-channel.
const PULSE_EXIT_TIME: f64 = 50.0;

/// |η| left in the channel by an outgoing pulse, relative to its amplitude.
struct PulseResidual {
    /// At `PULSE_EXIT_TIME`, once the pulse has left: the reflection.
    at_exit: f64,
    /// Largest over [`PULSE_EXIT_TIME`, t_end]: late reflections, or growth
    /// fed in through the open boundaries.
    after_exit: f64,
}

/// Launch a right-going Gaussian pulse of amplitude `amp` from mid-channel,
/// run to `t_end` (≥ `PULSE_EXIT_TIME`) and measure what is left in the
/// channel after the pulse has fully left through the east boundary.
///
/// Both channel ends carry `BoundaryTag::Open`, handled by `bc`.
fn pulse_residual<BC: SWEBoundaryCondition2D>(
    bc: &BC,
    amp: f64,
    flux_type: SWEFluxType2D,
    t_end: f64,
) -> PulseResidual {
    let ch = Channel::new(BoundaryTag::Open, BoundaryTag::Open);

    // Exact right-going simple wave: the left-going invariant
    // u − 2 sqrt(g h) is uniform, so no left-going signal is launched.
    let eta0 = |x: f64| amp * (-((x - PULSE_X0) / PULSE_WIDTH).powi(2)).exp();
    let u0 = |x: f64| 2.0 * ((G * (H0 + eta0(x))).sqrt() - celerity());
    let mut q = ch.initial_state(eta0, u0);

    let mut at_exit = None;
    let mut after_exit: f64 = 0.0;
    ch.run_with_flux(bc, flux_type, &mut q, t_end, |q, t| {
        if t > PULSE_EXIT_TIME - 1e-6 {
            let eta = ch.max_abs_eta(q);
            at_exit.get_or_insert(eta);
            after_exit = after_exit.max(eta);
        }
    });
    PulseResidual {
        at_exit: at_exit.expect("t_end before PULSE_EXIT_TIME") / amp,
        after_exit: after_exit / amp,
    }
}

/// Reflection of a small (linear-regime) pulse, measured once it has left.
fn outgoing_pulse_reflection<BC: SWEBoundaryCondition2D>(bc: &BC) -> f64 {
    pulse_residual(bc, PULSE_AMP, SWEFluxType2D::Roe, PULSE_EXIT_TIME).at_exit
}

/// Three channel crossings after the pulse has left: a boundary that feeds
/// energy in through the incoming characteristic shows up as growth (the old
/// `Radiation2D` reached 17× the pulse amplitude by 150 s, 42× by 190 s).
const QUIET_T_END: f64 = 200.0;

/// Maximum allowed reflected amplitude (fraction of the incident pulse).
const MAX_REFLECTION: f64 = 0.01;

fn assert_no_reflection(name: &str, reflection: f64) {
    println!("{name}: reflected amplitude = {:.4}%", 100.0 * reflection);
    assert!(
        reflection < MAX_REFLECTION,
        "{name}: outgoing pulse reflected with {:.1}% of its amplitude \
         (limit {:.1}%) — ghost feeds the interior's outgoing wave back into \
         the incoming Riemann invariant (e.g. Flather relation applied twice)?",
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

/// After the pulse has left, the channel must stay quiet until
/// `QUIET_T_END`: no late reflection and no growth fed in at the boundaries.
fn assert_stays_quiet(name: &str, after_exit: f64) {
    println!(
        "{name}: max |η| over [{PULSE_EXIT_TIME}, {QUIET_T_END}] s = {after_exit:.2e} × pulse"
    );
    assert!(
        after_exit < MAX_REFLECTION,
        "{name}: |η| reached {after_exit:.3e} × the pulse amplitude between \
         {PULSE_EXIT_TIME} s and {QUIET_T_END} s (limit {MAX_REFLECTION})"
    );
}

/// `Chapman2D` fixes the incoming invariant at `−sqrt(g/h) η_ext` through the
/// ghost elevation. Regression: its old blended ghost `α η_ext + (1 − α) η_int`
/// reflected ~99% of the pulse for α ≥ 0.7 and blew up for α ≤ 0.6.
#[test]
fn chapman2d_outgoing_pulse_does_not_reflect() {
    let sea_at_rest = |_: f64, _: f64, _: f64| 0.0;

    let chapman = Chapman2D::new(sea_at_rest, DX);
    let residual = pulse_residual(
        &with_walls(&chapman),
        PULSE_AMP,
        SWEFluxType2D::Roe,
        QUIET_T_END,
    );
    assert_no_reflection("Chapman2D", residual.at_exit);
    assert_stays_quiet("Chapman2D", residual.after_exit);

    // The time step no longer matters. c·dt/dx = 1 is the old no-dt fallback
    // (α = 1/2), which blew up within ~75 s here.
    let chapman = Chapman2D::with_dt(sea_at_rest, DX, DX / celerity());
    assert_no_reflection(
        "Chapman2D (c·dt/dx = 1)",
        outgoing_pulse_reflection(&with_walls(&chapman)),
    );
}

/// The Chapman ghost uses the nonlinear invariant `u_n − 2 sqrt(g h)`, so for
/// an outgoing simple wave it equals the interior trace and the boundary flux
/// is exact whatever the numerical flux. At a/H = 0.1 it reflects ~2e-4 with
/// Roe, HLL and Rusanov; the linearised ghost `η_ext + sqrt(h/g) u_n,int`
/// reflects 1.1%, and `Flather2D` 1.4e-3 with Rusanov.
#[test]
fn chapman2d_finite_amplitude_pulse_does_not_reflect() {
    let chapman = Chapman2D::new(|_, _, _| 0.0, DX);
    let bc = with_walls(&chapman);
    for flux_type in [
        SWEFluxType2D::Roe,
        SWEFluxType2D::HLL,
        SWEFluxType2D::Rusanov,
    ] {
        let reflection = pulse_residual(&bc, 0.1 * H0, flux_type, PULSE_EXIT_TIME).at_exit;
        println!(
            "Chapman2D a/H = 0.1, {flux_type:?}: reflected amplitude = {:.4}%",
            100.0 * reflection
        );
        assert!(
            reflection < 1e-3,
            "Chapman2D a/H = 0.1, {flux_type:?}: reflected {:.3}% of the pulse \
             (limit 0.1%) — ghost no longer matches the nonlinear invariant?",
            100.0 * reflection
        );
    }
}

/// `Radiation2D` uses the far-field state as ghost. Regression: it used to copy
/// the interior state into the ghost for every subcritical state, which left
/// 10% of the pulse behind after 50 s and grew it to 17× by 150 s.
#[test]
fn radiation2d_outgoing_pulse_does_not_reflect() {
    let radiation = Radiation2D::new(H0);
    let residual = pulse_residual(
        &with_walls(&radiation),
        PULSE_AMP,
        SWEFluxType2D::Roe,
        QUIET_T_END,
    );
    assert_no_reflection("Radiation2D", residual.at_exit);
    assert_stays_quiet("Radiation2D", residual.after_exit);

    // The nominal far-field depth does not enter the ghost; the local bed does.
    let radiation = Radiation2D::new(2.0 * H0);
    assert_no_reflection(
        "Radiation2D (h_external = 2 H0)",
        outgoing_pulse_reflection(&with_walls(&radiation)),
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

#[test]
fn chapman2d_delivers_half_amplitude_progressive_wave() {
    // Elevation-only forcing: with no external velocity the incoming invariant
    // is −sqrt(g/h) η_ext, the same as Flather2D with u_ext = 0, so a
    // progressive wave enters at η_ext / 2 (and leaves through the east end
    // without reflection, keeping the amplitude uniform).
    let forcing = Chapman2D::new(|_, _, t| eastward_wave_at_west(t).0, DX);
    let radiating = Chapman2D::new(|_, _, _| 0.0, DX);
    static WALL: Reflective2D = Reflective2D { h_min: 1e-6 };
    let bc = MultiBoundaryCondition2D::new(&WALL)
        .with_tidal(&forcing)
        .with_open(&radiating);
    let (min, max) = delivered_amplitude_range(&bc);
    println!("Chapman2D: delivered amplitude in [{min:.3}, {max:.3}] × prescribed");
    assert!(
        min >= 0.475 && max <= 0.525,
        "Chapman2D: delivered amplitude ranges over [{min:.3}, {max:.3}] of \
         η_ext (expected within [0.475, 0.525])"
    );
}
