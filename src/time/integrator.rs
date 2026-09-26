//! Trait-based time integrator abstraction.
//!
//! This module provides traits for time integration that enable:
//! - Generic time integrators that work with any solution type
//! - Extensible integrator implementations
//! - Both compile-time and runtime dispatch options
//!
//! # Example
//! ```
//! use dg_rs::time::{Integrable, TimeIntegrator, SSPRK3};
//! use dg_rs::solver::DGSolution1D;
//!
//! // DGSolution1D implements Integrable
//! let mut u = DGSolution1D::new(4, 3);
//! for v in &mut u.data { *v = 1.0; }
//!
//! let integrator = SSPRK3;
//! let dt = 0.01;
//! let t = 0.0;
//!
//! // Simple linear RHS: du/dt = -u (exponential decay)
//! integrator.step(&mut u, dt, t, |state, _time| {
//!     let mut rhs = state.clone();
//!     rhs.scale(-1.0);
//!     rhs
//! });
//! ```

// =============================================================================
// Integrable Trait
// =============================================================================

/// Trait for solution types that can be time-integrated.
///
/// This provides the vector space operations needed by explicit time integrators:
/// - `scale`: Multiply by scalar (x <- c * x)
/// - `axpy`: Add scaled vector (x <- x + c * y)
///
/// These operations should be implemented efficiently without allocations
/// beyond what's needed for intermediate stages.
///
/// # Example
/// ```
/// use dg_rs::time::Integrable;
/// use dg_rs::solver::DGSolution1D;
///
/// let mut u = DGSolution1D::new(4, 3);
/// for v in &mut u.data { *v = 1.0; }
///
/// let v = u.clone();
/// u.scale(2.0);      // u = 2.0 * u
/// u.axpy(0.5, &v);   // u = u + 0.5 * v
/// ```
pub trait Integrable: Clone + Send + Sized {
    /// Scale the solution by a constant: self <- c * self
    fn scale(&mut self, c: f64);

    /// Add a scaled vector: self <- self + c * other
    fn axpy(&mut self, c: f64, other: &Self);

    /// Create a zero-initialized solution with the same shape.
    ///
    /// Default implementation clones and scales by zero.
    fn zeros_like(&self) -> Self {
        let mut result = self.clone();
        result.scale(0.0);
        result
    }

    /// Overwrite `self` with `other` (same shape expected).
    ///
    /// Defaults to [`Clone::clone_from`], which reuses the allocation for the
    /// SoA solution types, so the stage buffers of
    /// [`TimeIntegrator::step_with_workspace`] are not reallocated.
    fn copy_from(&mut self, other: &Self) {
        self.clone_from(other);
    }
}

/// Reusable stage storage for [`TimeIntegrator::step_with_workspace`].
///
/// The buffers are cloned from the state on first use and reused afterwards,
/// so a step allocates nothing once the workspace is warm (given an RHS that
/// writes into its output). Keep one per simulation.
pub struct StageWorkspace<S> {
    u1: Option<S>,
    u2: Option<S>,
    k: Option<S>,
}

impl<S> Default for StageWorkspace<S> {
    fn default() -> Self {
        Self {
            u1: None,
            u2: None,
            k: None,
        }
    }
}

impl<S: Integrable> StageWorkspace<S> {
    /// Empty workspace; buffers are created on the first step.
    pub fn new() -> Self {
        Self::default()
    }

    /// Two stage buffers and one RHS buffer shaped like `like`.
    fn buffers(&mut self, like: &S) -> (&mut S, &mut S, &mut S) {
        let u1 = self.u1.get_or_insert_with(|| like.clone());
        let u2 = self.u2.get_or_insert_with(|| like.clone());
        let k = self.k.get_or_insert_with(|| like.clone());
        (u1, u2, k)
    }
}

// =============================================================================
// IntegratorInfo Trait (non-generic, dyn-compatible)
// =============================================================================

/// Non-generic information about a time integrator.
///
/// This trait is separate from [`TimeIntegrator`] to allow calling info methods
/// without specifying a solution type. It is also dyn-compatible.
pub trait IntegratorInfo: Send + Sync {
    /// Human-readable name for debugging and logging.
    fn name(&self) -> &'static str;

    /// Order of accuracy of the integrator.
    fn order(&self) -> usize;

    /// Number of stages in the integrator.
    fn n_stages(&self) -> usize;

    /// Whether the integrator is strong stability preserving (SSP).
    ///
    /// SSP integrators maintain TVD and other nonlinear stability properties.
    fn is_ssp(&self) -> bool;

    /// Times at which RHS is evaluated relative to current time.
    ///
    /// For SSP-RK3: [0, dt, dt/2] (stages evaluate at t, t+dt, t+dt/2)
    fn stage_times(&self, dt: f64) -> Vec<f64>;
}

// =============================================================================
// TimeIntegrator Trait
// =============================================================================

/// Trait for explicit time integrators.
///
/// Time integrators advance the solution from time `t` to `t + dt`
/// using one or more RHS evaluations. The RHS function receives
/// the current state and time, returning the time derivative.
///
/// # Implementation Notes
///
/// - Integrators should use the `Integrable` trait operations
/// - RHS functions should not allocate (hot path)
/// - Stage intermediate values may need to be stored in the integrator
///
/// # Example
/// ```
/// use dg_rs::time::{Integrable, TimeIntegrator, SSPRK3};
/// use dg_rs::solver::DGSolution1D;
///
/// let mut u = DGSolution1D::new(4, 3);
/// for v in &mut u.data { *v = 1.0; }
///
/// let integrator = SSPRK3;
/// integrator.step(&mut u, 0.01, 0.0, |state, _time| {
///     let mut rhs = state.clone();
///     rhs.scale(-1.0);
///     rhs
/// });
/// ```
pub trait TimeIntegrator<S: Integrable>: IntegratorInfo {
    /// Advance the solution by one time step.
    ///
    /// # Arguments
    /// * `state` - Solution to advance (modified in place)
    /// * `dt` - Time step size
    /// * `t` - Current time
    /// * `rhs` - Function computing the RHS: f(state, time) -> time_derivative
    fn step<F>(&self, state: &mut S, dt: f64, t: f64, rhs: F)
    where
        F: FnMut(&S, f64) -> S,
    {
        self.step_with_stage_hook(state, dt, t, rhs, |_| {});
    }

    /// Advance one step, applying `stage_hook` to every stage value, including
    /// the final one, before it is used.
    ///
    /// This is where nonlinear projections (positivity and slope limiters,
    /// wet/dry correction) belong: the Zhang–Shu positivity guarantee for SSP
    /// methods holds only if every RHS evaluation sees a limited state.
    ///
    /// Allocates stage storage and an RHS result per stage; long runs should
    /// use [`Self::step_with_workspace`].
    fn step_with_stage_hook<F, H>(&self, state: &mut S, dt: f64, t: f64, mut rhs: F, stage_hook: H)
    where
        F: FnMut(&S, f64) -> S,
        H: FnMut(&mut S),
    {
        let mut workspace = StageWorkspace::new();
        self.step_with_workspace(
            state,
            dt,
            t,
            |s, time, out: &mut S| *out = rhs(s, time),
            stage_hook,
            &mut workspace,
        );
    }

    /// Allocation-free version of [`Self::step_with_stage_hook`].
    ///
    /// `rhs(state, time, out)` must overwrite `out` with the time derivative;
    /// stage values live in `workspace`, reused across steps.
    fn step_with_workspace<F, H>(
        &self,
        state: &mut S,
        dt: f64,
        t: f64,
        rhs: F,
        stage_hook: H,
        workspace: &mut StageWorkspace<S>,
    ) where
        F: FnMut(&S, f64, &mut S),
        H: FnMut(&mut S),
    {
        self.step_with_relaxation(state, dt, t, rhs, |_, _, _| {}, stage_hook, workspace);
    }

    /// [`Self::step_with_workspace`] with a point-implicit relaxation of every
    /// stage. This is the one method an integrator implements; the others are
    /// built on it.
    ///
    /// In Shu–Osher form every stage is
    /// `u⁽ⁱ⁾ = Σₖ αᵢₖ u⁽ᵏ⁾ + βᵢ dt·L(u⁽ⁱ⁻¹⁾)` with `Σₖ αᵢₖ = 1`. Each is
    /// followed by `relax(u⁽ⁱ⁾, u⁽ⁱ⁻¹⁾, βᵢ dt)`, which applies a stiff damping
    /// source `−Λ(u⁽ⁱ⁻¹⁾)·q` implicitly, `u⁽ⁱ⁾ ← u⁽ⁱ⁾ / (1 + βᵢ dt·Λ)`, with the
    /// rate frozen at the state `L` was evaluated at:
    /// - the damping shrinks the stage value but never flips its sign, for any
    ///   `dt` (explicit SSP-RK3 goes unstable once `dt·Λ > 2.5`);
    /// - a steady state of `L(u) − Λ(u)u = 0` is a fixed point of every stage
    ///   (the numerator is `u*(1 + βᵢ dt Λ)`), so balances such as forcing
    ///   against friction are kept exactly, independently of `dt`;
    /// - as `dt·Λ → ∞` every stage is driven to zero (L-stable), unlike
    ///   relaxing only the Euler substeps, which leaves `u/3` per SSP-RK3 step.
    ///
    /// The damping term is integrated to first order; the rest keeps the
    /// integrator's order. `stage_hook` then runs on each stage value as in
    /// [`Self::step_with_workspace`].
    #[allow(clippy::too_many_arguments)]
    fn step_with_relaxation<F, R, H>(
        &self,
        state: &mut S,
        dt: f64,
        t: f64,
        rhs: F,
        relax: R,
        stage_hook: H,
        workspace: &mut StageWorkspace<S>,
    ) where
        F: FnMut(&S, f64, &mut S),
        R: FnMut(&mut S, &S, f64),
        H: FnMut(&mut S);
}

// =============================================================================
// SSP-RK3 Implementation
// =============================================================================

/// Strong Stability Preserving Runge-Kutta 3rd order integrator.
///
/// The SSP-RK3 method (Shu-Osher form) is optimal for hyperbolic conservation laws.
/// It maintains the TVD property of the spatial discretization.
///
/// Stages:
/// ```text
/// u1 = u + dt * L(u, t)
/// u2 = 3/4 * u + 1/4 * u1 + 1/4 * dt * L(u1, t + dt)
/// u_new = 1/3 * u + 2/3 * u2 + 2/3 * dt * L(u2, t + dt/2)
/// ```
///
/// Stage times: t, t + dt, t + dt/2
#[derive(Clone, Copy, Debug, Default)]
pub struct SSPRK3;

impl IntegratorInfo for SSPRK3 {
    fn name(&self) -> &'static str {
        "ssp-rk3"
    }

    fn order(&self) -> usize {
        3
    }

    fn n_stages(&self) -> usize {
        3
    }

    fn is_ssp(&self) -> bool {
        true
    }

    fn stage_times(&self, dt: f64) -> Vec<f64> {
        vec![0.0, dt, 0.5 * dt]
    }
}

impl<S: Integrable> TimeIntegrator<S> for SSPRK3 {
    fn step_with_relaxation<F, R, H>(
        &self,
        state: &mut S,
        dt: f64,
        t: f64,
        mut rhs: F,
        mut relax: R,
        mut stage_hook: H,
        workspace: &mut StageWorkspace<S>,
    ) where
        F: FnMut(&S, f64, &mut S),
        R: FnMut(&mut S, &S, f64),
        H: FnMut(&mut S),
    {
        let (u1, u2, k) = workspace.buffers(state);

        // Stage 1: u1 = u + dt * L(u, t)
        rhs(state, t, k);
        u1.copy_from(state);
        u1.axpy(dt, k);
        relax(u1, state, dt);
        stage_hook(u1);

        // Stage 2: u2 = 3/4 * u + 1/4 * u1 + 1/4 * dt * L(u1, t + dt)
        rhs(u1, t + dt, k);
        u2.copy_from(state);
        u2.scale(0.75);
        u2.axpy(0.25, u1);
        u2.axpy(0.25 * dt, k);
        relax(u2, u1, 0.25 * dt);
        stage_hook(u2);

        // Stage 3: u_new = 1/3 * u + 2/3 * u2 + 2/3 * dt * L(u2, t + dt/2)
        rhs(u2, t + 0.5 * dt, k);
        state.scale(1.0 / 3.0);
        state.axpy(2.0 / 3.0, u2);
        state.axpy(2.0 / 3.0 * dt, k);
        relax(state, u2, 2.0 / 3.0 * dt);
        stage_hook(state);
    }
}

// =============================================================================
// Forward Euler (for comparison/testing)
// =============================================================================

/// Forward Euler integrator (1st order).
///
/// Simple but only 1st order accurate. Useful for testing and debugging.
///
/// ```text
/// u_new = u + dt * L(u, t)
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardEuler;

impl IntegratorInfo for ForwardEuler {
    fn name(&self) -> &'static str {
        "forward-euler"
    }

    fn order(&self) -> usize {
        1
    }

    fn n_stages(&self) -> usize {
        1
    }

    fn is_ssp(&self) -> bool {
        true // Forward Euler is SSP with C_eff = 1
    }

    fn stage_times(&self, _dt: f64) -> Vec<f64> {
        vec![0.0]
    }
}

impl<S: Integrable> TimeIntegrator<S> for ForwardEuler {
    fn step_with_relaxation<F, R, H>(
        &self,
        state: &mut S,
        dt: f64,
        t: f64,
        mut rhs: F,
        mut relax: R,
        mut stage_hook: H,
        workspace: &mut StageWorkspace<S>,
    ) where
        F: FnMut(&S, f64, &mut S),
        R: FnMut(&mut S, &S, f64),
        H: FnMut(&mut S),
    {
        let (u0, _, k) = workspace.buffers(state);
        rhs(state, t, k);
        u0.copy_from(state);
        state.axpy(dt, k);
        relax(state, u0, dt);
        stage_hook(state);
    }
}

// =============================================================================
// Standard Integrator Enum (Zero-Cost Dispatch)
// =============================================================================

/// Enum wrapper for built-in integrators.
///
/// Provides zero-cost dispatch when integrator type is known at compile time,
/// while still allowing runtime selection via configuration.
#[derive(Clone, Copy, Debug, Default)]
pub enum StandardIntegrator {
    /// SSP-RK3 (default, recommended for hyperbolic problems)
    #[default]
    SSPRK3,
    /// Forward Euler (1st order, for testing)
    ForwardEuler,
}

impl IntegratorInfo for StandardIntegrator {
    fn name(&self) -> &'static str {
        match self {
            StandardIntegrator::SSPRK3 => "ssp-rk3",
            StandardIntegrator::ForwardEuler => "forward-euler",
        }
    }

    fn order(&self) -> usize {
        match self {
            StandardIntegrator::SSPRK3 => 3,
            StandardIntegrator::ForwardEuler => 1,
        }
    }

    fn n_stages(&self) -> usize {
        match self {
            StandardIntegrator::SSPRK3 => 3,
            StandardIntegrator::ForwardEuler => 1,
        }
    }

    fn is_ssp(&self) -> bool {
        true // Both are SSP
    }

    fn stage_times(&self, dt: f64) -> Vec<f64> {
        match self {
            StandardIntegrator::SSPRK3 => vec![0.0, dt, 0.5 * dt],
            StandardIntegrator::ForwardEuler => vec![0.0],
        }
    }
}

impl<S: Integrable> TimeIntegrator<S> for StandardIntegrator {
    fn step_with_relaxation<F, R, H>(
        &self,
        state: &mut S,
        dt: f64,
        t: f64,
        rhs: F,
        relax: R,
        stage_hook: H,
        workspace: &mut StageWorkspace<S>,
    ) where
        F: FnMut(&S, f64, &mut S),
        R: FnMut(&mut S, &S, f64),
        H: FnMut(&mut S),
    {
        match self {
            StandardIntegrator::SSPRK3 => {
                SSPRK3.step_with_relaxation(state, dt, t, rhs, relax, stage_hook, workspace)
            }
            StandardIntegrator::ForwardEuler => {
                ForwardEuler.step_with_relaxation(state, dt, t, rhs, relax, stage_hook, workspace)
            }
        }
    }
}

// =============================================================================
// Boxed Integrator Info (Runtime Polymorphism for Info Only)
// =============================================================================

/// Type alias for boxed integrator info (runtime polymorphism).
///
/// Note: The full `TimeIntegrator` trait is not dyn-compatible due to the
/// generic closure parameter in `step`. Use `StandardIntegrator` enum for
/// runtime selection of integrators.
pub type BoxedIntegratorInfo = Box<dyn IntegratorInfo>;

/// Create a boxed integrator info from a standard integrator type.
pub fn create_integrator_info(integrator: StandardIntegrator) -> BoxedIntegratorInfo {
    match integrator {
        StandardIntegrator::SSPRK3 => Box::new(SSPRK3),
        StandardIntegrator::ForwardEuler => Box::new(ForwardEuler),
    }
}

// =============================================================================
// Integrable Implementations for Existing Types
// =============================================================================

use crate::solver::{
    DGSolution1D, DGSolution2D, SWESolution, SWESolution2D, TracerSolution2D, state::Solution3D,
};

impl Integrable for DGSolution1D {
    fn scale(&mut self, c: f64) {
        self.scale(c);
    }

    fn axpy(&mut self, c: f64, other: &Self) {
        self.axpy(c, other);
    }
}

impl Integrable for DGSolution2D {
    fn scale(&mut self, c: f64) {
        self.scale(c);
    }

    fn axpy(&mut self, c: f64, other: &Self) {
        self.axpy(c, other);
    }
}

impl Integrable for SWESolution {
    fn scale(&mut self, c: f64) {
        self.scale(c);
    }

    fn axpy(&mut self, c: f64, other: &Self) {
        self.axpy(c, other);
    }
}

impl Integrable for SWESolution2D {
    fn scale(&mut self, c: f64) {
        self.scale(c);
    }

    fn axpy(&mut self, c: f64, other: &Self) {
        self.axpy(c, other);
    }
}

impl Integrable for TracerSolution2D {
    fn scale(&mut self, c: f64) {
        self.scale(c);
    }

    fn axpy(&mut self, c: f64, other: &Self) {
        self.axpy(c, other);
    }
}

impl Integrable for Solution3D {
    fn scale(&mut self, c: f64) {
        // Scale 2D barotropic state
        self.eta.scale(c);
        self.ubar.scale(c);
        self.vbar.scale(c);

        // Scale 3D baroclinic state
        for x in &mut self.u {
            *x *= c;
        }
        for x in &mut self.v {
            *x *= c;
        }
        for x in &mut self.w {
            *x *= c;
        }
        for x in &mut self.temp {
            *x *= c;
        }
        for x in &mut self.salt {
            *x *= c;
        }
        for x in &mut self.rho {
            *x *= c;
        }
    }

    fn axpy(&mut self, c: f64, other: &Self) {
        // AXPY 2D barotropic state
        self.eta.axpy(c, &other.eta);
        self.ubar.axpy(c, &other.ubar);
        self.vbar.axpy(c, &other.vbar);

        // AXPY 3D baroclinic state
        // Using iterators for now. Optimized SIMD/BLAS kernels should replace this later.
        for (x, y) in self.u.iter_mut().zip(other.u.iter()) {
            *x += c * y;
        }
        for (x, y) in self.v.iter_mut().zip(other.v.iter()) {
            *x += c * y;
        }
        for (x, y) in self.w.iter_mut().zip(other.w.iter()) {
            *x += c * y;
        }
        for (x, y) in self.temp.iter_mut().zip(other.temp.iter()) {
            *x += c * y;
        }
        for (x, y) in self.salt.iter_mut().zip(other.salt.iter()) {
            *x += c * y;
        }
        for (x, y) in self.rho.iter_mut().zip(other.rho.iter()) {
            *x += c * y;
        }
    }

    /// In place (the derived `Clone` reallocates every field).
    fn copy_from(&mut self, other: &Self) {
        if (self.n_elements, self.n_nodes, self.n_levels)
            != (other.n_elements, other.n_nodes, other.n_levels)
        {
            *self = other.clone();
            return;
        }
        self.eta.copy_from(&other.eta);
        self.ubar.copy_from(&other.ubar);
        self.vbar.copy_from(&other.vbar);
        self.u.copy_from_slice(&other.u);
        self.v.copy_from_slice(&other.v);
        self.w.copy_from_slice(&other.w);
        self.temp.copy_from_slice(&other.temp);
        self.salt.copy_from_slice(&other.salt);
        self.rho.copy_from_slice(&other.rho);
        self.eddy_viscosity.copy_from_slice(&other.eddy_viscosity);
        self.eddy_diffusivity
            .copy_from_slice(&other.eddy_diffusivity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssprk3_order() {
        // Test RK3 order with exponential growth: du/dt = u, u(0) = 1
        // Exact: u(t) = exp(t)
        let mut u = DGSolution1D::new(1, 3);
        for v in &mut u.data {
            *v = 1.0;
        }

        let integrator = SSPRK3;
        let dt = 0.01;
        let n_steps = 10;

        for i in 0..n_steps {
            let t = dt * i as f64;
            integrator.step(&mut u, dt, t, |state, _time| state.clone());
        }

        let t = dt * n_steps as f64;
        let expected = t.exp();

        for &v in &u.data {
            let error = (v - expected).abs();
            assert!(
                error < 1e-4,
                "Expected {}, got {} (error {})",
                expected,
                v,
                error
            );
        }
    }

    #[test]
    fn test_ssprk3_stage_hook_runs_after_each_stage() {
        let mut u = DGSolution1D::new(1, 1);
        u.data[0] = 1.0;

        let integrator = SSPRK3;
        let mut n_hooks = 0;
        integrator.step_with_stage_hook(
            &mut u,
            0.1,
            0.0,
            |state, _time| {
                let mut rhs = state.clone();
                rhs.scale(1.0);
                rhs
            },
            |stage_state| {
                n_hooks += 1;
                stage_state.data[0] = stage_state.data[0].min(1.05);
            },
        );

        assert_eq!(n_hooks, 3);
        assert!(u.data[0] <= 1.05);
    }

    /// du/dt = F − Λ|u|u (constant forcing, quadratic drag) with the drag
    /// relaxed point-implicitly: Λ|u_in| frozen at the RHS input.
    fn step_forced_drag<I: TimeIntegrator<DGSolution1D>>(
        integrator: &I,
        u: &mut DGSolution1D,
        dt: f64,
        forcing: f64,
        drag: f64,
    ) {
        integrator.step_with_relaxation(
            u,
            dt,
            0.0,
            |_, _, out: &mut DGSolution1D| out.data.fill(forcing),
            |w: &mut DGSolution1D, from: &DGSolution1D, dt| {
                for (w, &u) in w.data.iter_mut().zip(&from.data) {
                    *w /= 1.0 + dt * drag * u.abs();
                }
            },
            |_| {},
            &mut StageWorkspace::new(),
        );
    }

    #[test]
    fn test_relaxation_keeps_forced_steady_state_for_any_dt() {
        // Steady state F = Λ u*²: a fixed point of every relaxed stage,
        // however stiff Λ·dt is.
        let (forcing, drag) = (2.0_f64, 50.0);
        let u_star = (forcing / drag).sqrt();
        for integrator in [StandardIntegrator::SSPRK3, StandardIntegrator::ForwardEuler] {
            for dt in [1e-3, 1.0, 1e3] {
                let mut u = DGSolution1D::new(1, 2);
                u.data.fill(u_star);
                for _ in 0..10 {
                    step_forced_drag(&integrator, &mut u, dt, forcing, drag);
                }
                for &v in &u.data {
                    assert!(
                        (v - u_star).abs() < 1e-14,
                        "{}: dt = {dt}: {v} drifted from {u_star}",
                        integrator.name()
                    );
                }
            }
        }
    }

    #[test]
    fn test_relaxation_is_l_stable() {
        // Λ·dt → ∞ must remove the damped quantity within one step (relaxing
        // only the Euler substeps of SSP-RK3 would leave u/3)
        let mut u = DGSolution1D::new(1, 1);
        u.data[0] = 1.0;
        step_forced_drag(&SSPRK3, &mut u, 1e8, 0.0, 1.0);
        assert!(u.data[0] > 0.0 && u.data[0] < 1e-6, "{}", u.data[0]);
    }

    #[test]
    fn test_relaxation_never_flips_sign() {
        // Pure drag, Λ|u|·dt up to 1e4: explicit SSP-RK3 would oscillate and
        // blow up; the relaxed steps decay monotonically towards zero.
        for dt in [0.1, 10.0, 1e4] {
            let mut u = DGSolution1D::new(1, 1);
            u.data[0] = 1.0;
            let mut previous = 1.0;
            for _ in 0..20 {
                step_forced_drag(&SSPRK3, &mut u, dt, 0.0, 1.0);
                let v = u.data[0];
                assert!(v > 0.0 && v < previous, "dt = {dt}: {previous} -> {v}");
                previous = v;
            }
        }
    }

    #[test]
    fn test_relaxation_converges_to_exact_decay() {
        // du/dt = −Λ|u|u has u(t) = u0 / (1 + Λ u0 t); the relaxed drag is
        // first-order accurate inside SSP-RK3.
        let error_at = |dt: f64| {
            let t_end = 1.0;
            let mut u = DGSolution1D::new(1, 1);
            u.data[0] = 1.0;
            for _ in 0..(t_end / dt).round() as usize {
                step_forced_drag(&SSPRK3, &mut u, dt, 0.0, 3.0);
            }
            (u.data[0] - 1.0 / (1.0 + 3.0 * t_end)).abs()
        };
        let (coarse, fine) = (error_at(0.02), error_at(0.01));
        let rate = (coarse / fine).log2();
        assert!(fine < 1e-2, "error {fine}");
        assert!(
            rate > 0.9,
            "convergence rate {rate} (errors {coarse}, {fine})"
        );
    }

    #[test]
    fn test_forward_euler_order() {
        // Test Euler with exponential decay: du/dt = -u, u(0) = 1
        // Exact: u(t) = exp(-t)
        let mut u = DGSolution1D::new(1, 3);
        for v in &mut u.data {
            *v = 1.0;
        }

        let integrator = ForwardEuler;
        let dt = 0.001;
        let n_steps = 100;

        for i in 0..n_steps {
            let t = dt * i as f64;
            integrator.step(&mut u, dt, t, |state, _time| {
                let mut rhs = state.clone();
                rhs.scale(-1.0);
                rhs
            });
        }

        let t = dt * n_steps as f64;
        let expected = (-t).exp();

        for &v in &u.data {
            let error = (v - expected).abs();
            // Forward Euler is 1st order, so error is O(dt)
            assert!(
                error < 0.02,
                "Expected {}, got {} (error {})",
                expected,
                v,
                error
            );
        }
    }

    #[test]
    fn test_standard_integrator_dispatch() {
        let mut u = DGSolution1D::new(1, 3);
        for v in &mut u.data {
            *v = 1.0;
        }

        // Test that enum dispatch works
        let integrator = StandardIntegrator::SSPRK3;
        integrator.step(&mut u, 0.01, 0.0, |state, _time| state.clone());

        // Values should have changed
        assert!(u.data[0] > 1.0);
    }

    #[test]
    fn test_integrator_names() {
        assert_eq!(SSPRK3.name(), "ssp-rk3");
        assert_eq!(ForwardEuler.name(), "forward-euler");
        assert_eq!(StandardIntegrator::SSPRK3.name(), "ssp-rk3");
    }

    #[test]
    fn test_stage_times() {
        let dt = 0.1;
        let times = SSPRK3.stage_times(dt);
        assert_eq!(times.len(), 3);
        assert!((times[0] - 0.0).abs() < 1e-14);
        assert!((times[1] - dt).abs() < 1e-14);
        assert!((times[2] - 0.5 * dt).abs() < 1e-14);
    }

    #[test]
    fn test_ssp_flag() {
        assert!(SSPRK3.is_ssp());
        assert!(ForwardEuler.is_ssp());
    }

    #[test]
    fn test_zeros_like() {
        let u = DGSolution1D::new(2, 3);
        let zeros = u.zeros_like();
        for &v in &zeros.data {
            assert!((v - 0.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_boxed_integrator_info() {
        let info = create_integrator_info(StandardIntegrator::SSPRK3);
        assert_eq!(info.name(), "ssp-rk3");
        assert_eq!(info.order(), 3);
        assert!(info.is_ssp());
    }
}
