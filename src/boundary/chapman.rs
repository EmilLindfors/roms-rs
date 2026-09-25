//! Chapman radiation boundary condition for sea surface height.
//!
//! The Chapman (1985) condition lets long gravity waves leave the domain by
//! radiating the surface elevation anomaly outward at the shallow-water wave
//! speed:
//!
//! ```text
//! ∂(η − η_ext)/∂t + c ∂η/∂n = 0,    c = sqrt(g h)
//! ```
//!
//! Finite-difference models (e.g. ROMS's Chapman free-surface option) apply it
//! as an implicit update of the boundary value,
//! `η_b^{n+1} = (η_b^n + C η_{b−1}^{n+1}) / (1 + C)` with `C = c·dt/dx`, paired
//! with a separate condition for the velocity.
//!
//! # Weak (ghost-state) DG formulation
//!
//! For the linearised shallow water equations at normal incidence, with
//! Riemann invariants `w± = u_n ± sqrt(g/h) η` travelling at `±c`, the Chapman
//! condition is equivalent to holding the incoming invariant at its external
//! value. Writing `η = (w+ − w−) / (2 sqrt(g/h))` and using `∂w−/∂t = c ∂w−/∂n`
//! (n the outward normal coordinate, η_ext uniform along n),
//!
//! ```text
//! ∂η/∂t + c ∂η/∂n = −sqrt(h/g) ∂w−/∂t = ∂η_ext/∂t   ⇔   w− = −sqrt(g/h) η_ext,
//! ```
//!
//! up to a constant fixed by the initial state (zero when the initial state
//! is in equilibrium with η_ext).
//!
//! In the weak DG setting the upwind Riemann solver at a boundary face takes
//! the outgoing invariant from the interior trace and the incoming one from
//! the ghost state, so [`Chapman2D`] imposes the condition through the ghost:
//! the velocity is extrapolated from the interior (the BC only acts on η, as in
//! ROMS) and the ghost elevation is chosen so that the ghost's incoming
//! invariant is the external one. Using the nonlinear invariant
//! `R− = u_n − 2 sqrt(g h)` and `h_ext = η_ext − B`:
//!
//! ```text
//! sqrt(h_ghost) = sqrt(h_ext) + u_n,int / (2 sqrt(g)),
//! ```
//!
//! which linearises to `η_ghost = η_ext + sqrt(h/g) u_n,int`. `dt` and `dx`
//! drop out: the Riemann solver supplies the radiation.
//!
//! ## Relation to `Flather2D`
//!
//! To first order in the wave amplitude this is the same boundary flux as
//! [`Flather2D`](crate::boundary::Flather2D) with zero external velocity
//! (ghost `(η_ext, u_n = 0)`): both fix `w− = −sqrt(g/h) η_ext`. They differ in
//! the ghost state itself. For an outgoing simple wave over a quiescent
//! exterior the Chapman ghost *equals* the interior trace, so the boundary
//! flux is exact for any consistent numerical flux, not only for an upwind
//! one. In the channel test (`tests/open_boundary_flather_test.rs`, pulse
//! amplitude a/H = 0.1) it reflects ~2e-4 with Roe, HLL and Rusanov fluxes,
//! against 4e-4 (Roe/HLL) and 1.4e-3 (Rusanov) for `Flather2D`; the linearised
//! ghost `η_ext + sqrt(h/g) u_n,int` reflects 1.1% there.
//!
//! As for `Flather2D` with zero external velocity, a *progressive* wave forced
//! through the boundary arrives at η_ext / 2; the boundary elevation equals
//! η_ext where the boundary sits at an antinode of a standing tide. Use
//! [`ChapmanFlather2D`] with `u_n,ext = −sqrt(g/h) η_ext` to force a
//! progressive wave at full amplitude.
//!
//! ## Previous formulation
//!
//! Earlier versions used Chapman's finite-difference blend directly as the
//! ghost, `η_ghost = α η_ext + (1 − α) η_int` with α = 1/(1 + c·dt/dx) and
//! extrapolated velocity. The ghost's incoming invariant then contains
//! `(1 − α) η_int`, i.e. the interior's outgoing wave: the channel test
//! reflected ~99% of an outgoing pulse for every α ≥ 0.7, and the scheme blew
//! up for α ≤ 0.6.
//!
//! # Reference
//!
//! Chapman, D.C. (1985): "Numerical treatment of cross-shelf open boundaries
//! in a barotropic coastal ocean model", Journal of Physical Oceanography.

use super::bathymetry_validation::warn_once_if_misconfigured;
use super::{BCContext2D, SWEBoundaryCondition2D};
use crate::solver::SWEState2D;

/// Chapman radiation boundary condition for 2D.
///
/// Radiates the surface elevation anomaly `η − η_ext` out of the domain.
/// The ghost keeps the interior velocity and sets the elevation so that the
/// incoming Riemann invariant `u_n − 2 sqrt(g h)` equals its external value
/// `−2 sqrt(g (η_ext − B))`. See the module docs for the derivation and its
/// relation to [`Flather2D`](crate::boundary::Flather2D).
///
/// Intended for subcritical open boundaries (|u_n| < sqrt(g h)). Like other
/// Flather-type BCs it needs the depth convention `h = η − B` (set bathymetry
/// on the RHS config; a one-time warning is printed if it looks unset).
///
/// # Time step
///
/// The condition needs neither `dt` nor `dx`: the Riemann solver provides the
/// radiation. `dx`, `dt`, [`Chapman2D::with_dt`] and [`Chapman2D::set_dt`] are
/// retained for API compatibility and do not affect the ghost state.
///
/// # Type Parameters
///
/// * `F` - External elevation function η_ext(x, y, t)
///
/// # Example
///
/// ```
/// use dg_rs::boundary::{Chapman2D, BCContext2D, SWEBoundaryCondition2D};
/// use dg_rs::SWEState2D;
///
/// // Radiate towards a sea at rest at mean sea level
/// let chapman = Chapman2D::new(|_x, _y, _t| 0.0, 100.0);
///
/// // Evaluate at boundary: 10 m of water over a bed at B = −10 m
/// let ctx = BCContext2D::new(
///     0.0, (0.0, 0.0),
///     SWEState2D::new(10.0, 5.0, 0.0), // h=10, hu=5
///     -10.0, (1.0, 0.0), // bathymetry, outward normal in x
///     9.81, 1e-6,
/// );
/// let ghost = chapman.ghost_state(&ctx);
/// ```
#[derive(Clone)]
pub struct Chapman2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// External (far-field) surface elevation function η_ext(x, y, t)
    pub external_elevation: F,
    /// Grid spacing.
    ///
    /// **Unused**: the Riemann solver provides the radiation (see the type
    /// docs). Retained for API compatibility.
    pub dx: f64,
    /// Time step.
    ///
    /// **Unused**: the Riemann solver provides the radiation (see the type
    /// docs). Retained for API compatibility.
    pub dt: Option<f64>,
    /// Minimum depth threshold
    pub h_min: f64,
}

impl<F> Chapman2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// Create Chapman BC with external elevation function.
    ///
    /// # Arguments
    /// * `external_elevation` - Function returning η_ext(x, y, t)
    /// * `dx` - Grid spacing (unused, see type docs)
    pub fn new(external_elevation: F, dx: f64) -> Self {
        Self {
            external_elevation,
            dx,
            dt: None,
            h_min: 1e-6,
        }
    }

    /// Create with a time step (unused, see type docs).
    pub fn with_dt(external_elevation: F, dx: f64, dt: f64) -> Self {
        Self {
            external_elevation,
            dx,
            dt: Some(dt),
            h_min: 1e-6,
        }
    }

    /// Set the time step (unused, see type docs).
    pub fn set_dt(&mut self, dt: f64) {
        self.dt = Some(dt);
    }

    /// Set minimum depth threshold.
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }
}

impl<F> SWEBoundaryCondition2D for Chapman2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        use std::sync::atomic::AtomicBool;
        static WARNED: AtomicBool = AtomicBool::new(false);

        let (x, y) = ctx.position;
        let (nx, ny) = ctx.normal;

        // External surface elevation and the depth it implies here
        let eta_ext = (self.external_elevation)(x, y, ctx.time);
        let h_ext = (eta_ext - ctx.bathymetry).max(0.0);

        warn_once_if_misconfigured(
            &WARNED,
            "Chapman2D",
            ctx.interior_state.h,
            ctx.bathymetry,
            eta_ext,
        );

        // Velocity extrapolated from the interior
        let (u, v) = ctx.interior_velocity();
        let un = u * nx + v * ny;

        // Ghost depth with incoming invariant u_n − 2 sqrt(g h) equal to the
        // external −2 sqrt(g h_ext). A negative root means inflow strong
        // enough to drain the ghost (u_n < −2 sqrt(g h_ext)).
        let sqrt_h = (h_ext.sqrt() + un / (2.0 * ctx.g.sqrt())).max(0.0);
        let h_ghost = (sqrt_h * sqrt_h).max(self.h_min);

        SWEState2D::from_primitives(h_ghost, u, v)
    }

    fn name(&self) -> &'static str {
        "chapman_2d"
    }
}

/// Combined Chapman (elevation) + Flather (velocity) boundary condition.
///
/// This is the recommended open boundary condition for tidal simulations
/// with known external elevation *and* normal velocity (e.g. a progressive
/// tide or parent-model output).
///
/// # Weak (ghost-state) formulation
///
/// In finite-difference models Chapman (radiation of η) and Flather
/// (characteristic relation for u_n) are two separate boundary updates. In the
/// weak DG setting the upwind Riemann solver at the boundary face performs
/// both at once: it takes the outgoing Riemann invariant
/// `w+ = u_n + sqrt(g/h) η` from the interior trace and the incoming invariant
/// `w− = u_n − sqrt(g/h) η` from the ghost. The ghost state is therefore just
/// the external state:
///
/// - h = η_ext − B (depth convention h = η − B)
/// - u_n = u_n,ext
/// - u_t from the interior
///
/// Neither the Chapman blend `α η_ext + (1 − α) η_int` nor the Flather
/// correction `sqrt(g/h)(η_int − η_ext)` is applied to the ghost: each would
/// feed the outgoing wave back into the incoming invariant and reflect it
/// (by `(1 − α)/(1 + α)` and −1/3 respectively). Consequently `dx`, `dt` and
/// `h_ref` no longer affect the ghost state; they are retained for API
/// compatibility.
///
/// To deliver a progressive wave of amplitude A travelling into the domain,
/// supply `u_n,ext = −sqrt(g/h) η_ext` (u_n is along the outward normal).
/// With `u_n,ext = 0` the boundary elevation equals η_ext only where the
/// boundary sits at an antinode of a standing tide; a progressive wave is then
/// delivered at A/2.
///
/// # Type Parameters
///
/// * `F` - External state function returning (η_ext, u_n_ext, u_t_ext) at (x, y, t)
///
/// # Example
///
/// ```
/// use dg_rs::boundary::{ChapmanFlather2D, BCContext2D, SWEBoundaryCondition2D};
/// use dg_rs::SWEState2D;
/// use std::f64::consts::PI;
///
/// // M2 tidal forcing
/// let omega = 2.0 * PI / (12.42 * 3600.0);
/// let amplitude = 1.0;
/// let cf = ChapmanFlather2D::new(
///     move |_x, _y, t| (amplitude * (omega * t).cos(), 0.0, 0.0),
///     100.0, // dx
/// );
/// ```
#[derive(Clone)]
pub struct ChapmanFlather2D<F>
where
    F: Fn(f64, f64, f64) -> (f64, f64, f64) + Send + Sync,
{
    /// External state function returning (η_ext, u_n_ext, u_t_ext) at (x, y, t)
    pub external_state: F,
    /// Grid spacing for Chapman radiation.
    ///
    /// **Unused**: the Riemann solver provides the radiation (see the type
    /// docs). Retained for API compatibility.
    pub dx: f64,
    /// Time step.
    ///
    /// **Unused**: the Riemann solver provides the radiation (see the type
    /// docs). Retained for API compatibility.
    pub dt: Option<f64>,
    /// Reference depth for Flather.
    ///
    /// **Unused**: the Riemann solver evaluates the wave speed from the
    /// boundary states (see the type docs). Retained for API compatibility.
    pub h_ref: f64,
    /// Minimum depth
    pub h_min: f64,
}

impl<F> ChapmanFlather2D<F>
where
    F: Fn(f64, f64, f64) -> (f64, f64, f64) + Send + Sync,
{
    /// Create a new ChapmanFlather BC.
    ///
    /// # Arguments
    /// * `external_state` - Function returning (η, u_n, u_t) at (x, y, t),
    ///   with u_n along the outward normal
    /// * `dx` - Grid spacing for Chapman radiation (unused, see type docs)
    pub fn new(external_state: F, dx: f64) -> Self {
        Self {
            external_state,
            dx,
            dt: None,
            h_ref: 10.0,
            h_min: 1e-6,
        }
    }

    /// Set reference depth for Flather velocity condition (unused, see type docs).
    pub fn with_h_ref(mut self, h_ref: f64) -> Self {
        self.h_ref = h_ref;
        self
    }

    /// Set time step for Chapman (unused, see type docs).
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = Some(dt);
        self
    }

    /// Set minimum depth.
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }
}

impl<F> SWEBoundaryCondition2D for ChapmanFlather2D<F>
where
    F: Fn(f64, f64, f64) -> (f64, f64, f64) + Send + Sync,
{
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        use std::sync::atomic::AtomicBool;
        static WARNED: AtomicBool = AtomicBool::new(false);

        let (x, y) = ctx.position;
        let t = ctx.time;
        let (nx, ny) = ctx.normal;

        // External state: (η_ext, u_n_ext, u_t_ext)
        let (eta_ext, un_ext, _ut_ext) = (self.external_state)(x, y, t);

        // Validate bathymetry configuration (warns once if misconfigured)
        warn_once_if_misconfigured(
            &WARNED,
            "ChapmanFlather2D",
            ctx.interior_state.h,
            ctx.bathymetry,
            eta_ext,
        );

        // Ghost = external state. The Riemann solver radiates the outgoing
        // invariant from the interior and takes the incoming one from here;
        // blending η_int into the ghost or adding a Flather correction to
        // u_n would feed the outgoing wave back in (see type docs).
        let h_ghost = (eta_ext - ctx.bathymetry).max(self.h_min);
        let un_ghost = un_ext;

        // Tangential velocity from interior (zero-gradient)
        let ut_ghost = ctx.interior_tangential_velocity();

        // Convert (un, ut) back to (u, v)
        // u = un * nx - ut * ny
        // v = un * ny + ut * nx
        let u_ghost = un_ghost * nx - ut_ghost * ny;
        let v_ghost = un_ghost * ny + ut_ghost * nx;

        SWEState2D::from_primitives(h_ghost, u_ghost, v_ghost)
    }

    fn name(&self) -> &'static str {
        "chapman_flather_2d"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 10.0;
    const TOL: f64 = 1e-10;

    fn make_context(h: f64, hu: f64, hv: f64, normal: (f64, f64)) -> BCContext2D {
        BCContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(h, hu, hv),
            0.0, // bathymetry
            normal,
            G,
            1e-6,
        )
    }

    #[test]
    fn test_chapman_steady_state() {
        // When interior matches external, ghost should match both
        let eta_ext = 5.0;
        let chapman = Chapman2D::new(|_, _, _| eta_ext, 100.0);

        let ctx = make_context(5.0, 0.0, 0.0, (1.0, 0.0)); // h=5, eta=h+B=5
        let ghost = chapman.ghost_state(&ctx);

        assert!(
            (ghost.h - 5.0).abs() < TOL,
            "Steady state should maintain h=5, got {}",
            ghost.h
        );
    }

    fn make_context_with_bed(
        h: f64,
        u: f64,
        v: f64,
        bathymetry: f64,
        normal: (f64, f64),
    ) -> BCContext2D {
        BCContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::from_primitives(h, u, v),
            bathymetry,
            normal,
            G,
            1e-6,
        )
    }

    /// Incoming nonlinear Riemann invariant u_n − 2 sqrt(g h) along `normal`.
    fn incoming_invariant(state: &SWEState2D, normal: (f64, f64)) -> f64 {
        let un = (state.hu * normal.0 + state.hv * normal.1) / state.h;
        un - 2.0 * (G * state.h).sqrt()
    }

    #[test]
    fn test_chapman_ghost_holds_external_incoming_invariant() {
        // Whatever the interior does, the ghost's incoming invariant is the
        // external one, so the Riemann solver cannot feed the interior's
        // outgoing wave back in. Regression: the old blended ghost
        // α η_ext + (1 − α) η_int carried (1 − α) η_int into it and reflected
        // outgoing waves (~99% in the channel test).
        let eta_ext = 0.3;
        let bed = -20.0;
        let chapman = Chapman2D::new(move |_, _, _| eta_ext, 100.0);
        let s = 1.0 / 2.0_f64.sqrt();
        let normals = [(1.0, 0.0), (0.0, -1.0), (s, s), (-0.6, 0.8)];
        let interiors = [
            (20.0, 0.0, 0.0),
            (21.5, 0.8, -0.4),
            (18.2, -1.1, 0.7),
            (23.0, 2.5, 1.0),
        ];

        let expected = -2.0 * (G * (eta_ext - bed)).sqrt();
        for normal in normals {
            for (h, u, v) in interiors {
                let ctx = make_context_with_bed(h, u, v, bed, normal);
                let ghost = chapman.ghost_state(&ctx);
                let w = incoming_invariant(&ghost, normal);
                assert!(
                    (w - expected).abs() < 1e-9,
                    "incoming invariant {w} != external {expected} for h={h}, \
                     u=({u}, {v}), n={normal:?}"
                );
            }
        }
    }

    #[test]
    fn test_chapman_ghost_equals_interior_for_outgoing_simple_wave() {
        // A simple wave leaving into a sea at rest at η_ext has the external
        // incoming invariant, u_n = 2 (sqrt(g h) − sqrt(g h_ext)). The ghost
        // then reproduces the interior trace, so any consistent numerical
        // flux returns the physical flux: no reflection.
        let eta_ext = 0.0;
        let bed = -10.0;
        let chapman = Chapman2D::new(move |_, _, _| eta_ext, 100.0);
        let normal = (0.6, -0.8);
        for eta in [-0.5, 0.01, 1.0] {
            let h = eta - bed;
            let un = 2.0 * ((G * h).sqrt() - (G * (eta_ext - bed)).sqrt());
            let ut = 0.3;
            let (u, v) = (un * normal.0 - ut * normal.1, un * normal.1 + ut * normal.0);
            let ctx = make_context_with_bed(h, u, v, bed, normal);
            let ghost = chapman.ghost_state(&ctx);

            assert!((ghost.h - h).abs() < 1e-9, "ghost h {} != {h}", ghost.h);
            assert!((ghost.hu - h * u).abs() < 1e-9, "ghost hu {}", ghost.hu);
            assert!((ghost.hv - h * v).abs() < 1e-9, "ghost hv {}", ghost.hv);
        }
    }

    #[test]
    fn test_chapman_linearises_to_elevation_form() {
        // For small u_n the ghost elevation is η_ext + sqrt(h/g) u_n, with an
        // O(u_n²) remainder u_n² / (4g).
        let eta_ext = 0.2;
        let bed = -15.0;
        let h_ext = eta_ext - bed;
        let chapman = Chapman2D::new(move |_, _, _| eta_ext, 100.0);
        for un in [-0.01, 0.003, 0.02] {
            let ctx = make_context_with_bed(h_ext + 0.05, un, 0.0, bed, (1.0, 0.0));
            let ghost = chapman.ghost_state(&ctx);
            let eta_ghost = ghost.h + bed;
            let linear = eta_ext + (h_ext / G).sqrt() * un;
            assert!(
                (eta_ghost - linear - un * un / (4.0 * G)).abs() < 1e-12,
                "η_ghost = {eta_ghost}, linear form {linear}"
            );
        }
    }

    #[test]
    fn test_chapman_ghost_outward_mass_flux_for_elevated_interior() {
        // Interior at rest but raised above η_ext: the Roe flux through the
        // boundary face must carry mass out of the domain.
        let chapman = Chapman2D::new(|_, _, _| 0.0, 100.0);
        let ctx = make_context_with_bed(10.5, 0.0, 0.0, -10.0, (1.0, 0.0));
        let ghost = chapman.ghost_state(&ctx);
        let flux = crate::flux::roe_flux_swe_2d(&ctx.interior_state, &ghost, ctx.normal, G, 1e-6);
        assert!(flux.h > 0.0, "expected outward mass flux, got {}", flux.h);
    }

    #[test]
    fn test_chapman_dry_ghost() {
        // Bed above the external surface: nothing flows in from outside.
        let chapman = Chapman2D::new(|_, _, _| 0.0, 100.0);
        let ctx = make_context_with_bed(0.0, 0.0, 0.0, 2.0, (1.0, 0.0));
        let ghost = chapman.ghost_state(&ctx);
        assert!(
            ghost.h <= 1e-6 + TOL,
            "ghost should be dry, h = {}",
            ghost.h
        );

        // Inflow faster than 2 sqrt(g h_ext) empties the ghost instead of
        // wrapping round to a positive depth.
        let ctx = make_context_with_bed(1.0, -10.0, 0.0, -1.0, (1.0, 0.0));
        let ghost = chapman.ghost_state(&ctx);
        assert!(
            ghost.h <= 1e-6 + TOL,
            "ghost should be dry, h = {}",
            ghost.h
        );
    }

    #[test]
    fn test_chapman_velocity_extrapolation() {
        // Velocity should be extrapolated from interior
        let chapman = Chapman2D::new(|_, _, _| 10.0, 100.0);

        let ctx = make_context(10.0, 20.0, 30.0, (1.0, 0.0)); // u=2, v=3
        let ghost = chapman.ghost_state(&ctx);

        // Velocity should be preserved from interior
        let (u, v) = (ghost.hu / ghost.h, ghost.hv / ghost.h);
        assert!((u - 2.0).abs() < TOL, "u should be extrapolated: got {}", u);
        assert!((v - 3.0).abs() < TOL, "v should be extrapolated: got {}", v);
    }

    #[test]
    fn test_chapman_flather_steady() {
        // Test combined condition in steady state
        let cf = ChapmanFlather2D::new(|_, _, _| (10.0, 0.0, 0.0), 100.0).with_h_ref(10.0);

        let ctx = make_context(10.0, 0.0, 0.0, (1.0, 0.0));
        let ghost = cf.ghost_state(&ctx);

        // In steady state matching external, ghost should match
        assert!(
            (ghost.h - 10.0).abs() < TOL,
            "Steady state h: got {}",
            ghost.h
        );
        assert!(
            ghost.hu.abs() < TOL,
            "Steady state hu should be ~0: got {}",
            ghost.hu
        );
    }

    #[test]
    fn test_chapman_flather_ghost_is_external_state() {
        // Interior η = 10 lies above the external η = 8, with inflow u_n,ext.
        // The ghost is the external state (h = η_ext − B, u_n = u_n,ext).
        // Regression: the old ghost added sqrt(g/h_ref)(η_int − η_ext) to
        // u_n, so the Riemann solver applied the Flather relation twice and
        // reflected outgoing waves.
        let cf = ChapmanFlather2D::new(|_, _, _| (8.0, -0.3, 0.0), 100.0).with_h_ref(10.0);
        let ctx = make_context(10.0, 0.0, 0.0, (1.0, 0.0));
        let ghost = cf.ghost_state(&ctx);

        assert!((ghost.h - 8.0).abs() < TOL, "ghost depth: got {}", ghost.h);
        let un_ghost = ghost.hu / ghost.h;
        assert!(
            (un_ghost - (-0.3)).abs() < TOL,
            "ghost normal velocity should be u_n,ext: got {}",
            un_ghost
        );

        // The Flather response comes from the upwind flux: the elevated
        // interior still drains outward through the boundary.
        let flux = crate::flux::roe_flux_swe_2d(&ctx.interior_state, &ghost, ctx.normal, G, 1e-6);
        assert!(flux.h > 0.0, "expected outward mass flux, got {}", flux.h);
    }

    #[test]
    fn test_chapman_ignores_dt() {
        // The Riemann solver supplies the radiation, so dt (on the BC or in
        // the context) must not change the ghost. Regression: the old blend
        // used α = 1/(1 + c·dt/dx) and was unstable for α ≲ 0.7.
        let without = Chapman2D::new(|_, _, _| 0.1, 100.0);
        let with_bc_dt = Chapman2D::with_dt(|_, _, _| 0.1, 100.0, 10.0);
        let mut ctx = make_context_with_bed(10.4, 0.3, -0.2, -10.0, (1.0, 0.0));
        let reference = without.ghost_state(&ctx);

        let ghost = with_bc_dt.ghost_state(&ctx);
        assert!((ghost.h - reference.h).abs() < TOL);
        assert!((ghost.hu - reference.hu).abs() < TOL);

        ctx.dt = Some(10.0);
        let ghost = without.ghost_state(&ctx);
        assert!((ghost.h - reference.h).abs() < TOL);
        assert!((ghost.hu - reference.hu).abs() < TOL);
    }

    #[test]
    fn test_chapman_preserves_tangential() {
        // Tangential velocity should be preserved in ChapmanFlather
        let cf = ChapmanFlather2D::new(|_, _, _| (10.0, 0.0, 0.0), 100.0).with_h_ref(10.0);

        // Flow tangential to boundary (normal is x, flow is y)
        let ctx = make_context(10.0, 0.0, 30.0, (1.0, 0.0)); // u_n=0, u_t=3
        let ghost = cf.ghost_state(&ctx);

        let v_ghost = ghost.hv / ghost.h;
        assert!(
            (v_ghost - 3.0).abs() < TOL,
            "Tangential velocity should be preserved: got {}",
            v_ghost
        );
    }

    #[test]
    fn test_chapman_different_normals() {
        // Test with different normal directions
        let chapman = Chapman2D::new(|_, _, _| 10.0, 100.0);

        for normal in [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)] {
            let ctx = make_context(10.0, 5.0, 5.0, normal);
            let ghost = chapman.ghost_state(&ctx);

            assert!(
                ghost.h > 0.0,
                "Ghost depth should be positive for normal {:?}",
                normal
            );
        }
    }
}
