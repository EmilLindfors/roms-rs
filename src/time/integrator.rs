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
        F: Fn(&S, f64) -> S;
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
    fn step<F>(&self, state: &mut S, dt: f64, t: f64, rhs: F)
    where
        F: Fn(&S, f64) -> S,
    {
        // Stage 1: u1 = u + dt * L(u, t)
        let l_u = rhs(state, t);
        let mut u1 = state.clone();
        u1.axpy(dt, &l_u);

        // Stage 2: u2 = 3/4 * u + 1/4 * u1 + 1/4 * dt * L(u1, t + dt)
        let t1 = t + dt;
        let l_u1 = rhs(&u1, t1);
        let mut u2 = state.clone();
        u2.scale(0.75);
        u2.axpy(0.25, &u1);
        u2.axpy(0.25 * dt, &l_u1);

        // Stage 3: u_new = 1/3 * u + 2/3 * u2 + 2/3 * dt * L(u2, t + dt/2)
        let t2 = t + 0.5 * dt;
        let l_u2 = rhs(&u2, t2);
        state.scale(1.0 / 3.0);
        state.axpy(2.0 / 3.0, &u2);
        state.axpy(2.0 / 3.0 * dt, &l_u2);
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
    fn step<F>(&self, state: &mut S, dt: f64, t: f64, rhs: F)
    where
        F: Fn(&S, f64) -> S,
    {
        let l_u = rhs(state, t);
        state.axpy(dt, &l_u);
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
    fn step<F>(&self, state: &mut S, dt: f64, t: f64, rhs: F)
    where
        F: Fn(&S, f64) -> S,
    {
        match self {
            StandardIntegrator::SSPRK3 => SSPRK3.step(state, dt, t, rhs),
            StandardIntegrator::ForwardEuler => ForwardEuler.step(state, dt, t, rhs),
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

use crate::solver::{DGSolution1D, DGSolution2D, SWESolution, SWESolution2D, TracerSolution2D};

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
