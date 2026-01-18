//! Physics module traits.
//!
//! This module defines the core trait interfaces for physics modules.

use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::time::Integrable;

// =============================================================================
// PhysicsModuleInfo Trait (non-generic, dyn-compatible)
// =============================================================================

/// Non-generic information about a physics module.
///
/// This trait is separate from [`PhysicsModule`] to allow calling info methods
/// without specifying a solution type.
pub trait PhysicsModuleInfo: Send + Sync {
    /// Human-readable name for debugging and logging.
    fn name(&self) -> &'static str;

    /// Short description of the physics being modeled.
    fn description(&self) -> &str;

    /// Number of conserved variables in the system.
    fn n_variables(&self) -> usize;

    /// Names of the conserved variables (e.g., ["h", "hu", "hv"]).
    fn variable_names(&self) -> &[&'static str];
}

// =============================================================================
// PhysicsModule Trait
// =============================================================================

/// Core trait for physics modules in DG simulations.
///
/// A physics module encapsulates everything needed to compute the semi-discrete
/// right-hand side of a PDE system, including:
/// - Numerical flux computation
/// - Boundary condition handling
/// - Source term evaluation
/// - Post-processing (limiters, wet/dry correction)
///
/// # Type Parameters
///
/// * `S` - Solution type (must implement [`Integrable`])
///
/// # Example Implementation
///
/// ```ignore
/// impl PhysicsModule<SWESolution2D> for SWEPhysics2D {
///     fn compute_rhs(&self, state: &SWESolution2D, time: f64) -> SWESolution2D {
///         compute_rhs_swe_2d(state, &self.mesh, &self.ops, &self.geom, time, &self.config)
///     }
///
///     fn compute_dt(&self, state: &SWESolution2D, cfl: f64) -> f64 {
///         compute_dt_swe_2d(state, &self.mesh, &self.geom, &self.equation, cfl, self.order)
///     }
/// }
/// ```
pub trait PhysicsModule<S: Integrable>: PhysicsModuleInfo {
    /// Compute the semi-discrete right-hand side.
    ///
    /// Given the current state at time `t`, compute `dq/dt = L(q)` where
    /// L includes volume terms, surface terms, and source terms.
    ///
    /// # Arguments
    /// * `state` - Current solution state
    /// * `time` - Current simulation time
    ///
    /// # Returns
    /// The time derivative of the state.
    fn compute_rhs(&self, state: &S, time: f64) -> S;

    /// Compute the CFL-limited time step.
    ///
    /// # Arguments
    /// * `state` - Current solution state
    /// * `cfl` - CFL number (typically 0.1 - 0.5)
    ///
    /// # Returns
    /// Maximum stable time step.
    fn compute_dt(&self, state: &S, cfl: f64) -> f64;

    /// Post-process the solution after each time step.
    ///
    /// This is where limiters, positivity correction, and wet/dry treatment
    /// should be applied. Called after the time integrator completes a step.
    ///
    /// Default implementation does nothing.
    fn post_process(&self, _state: &mut S) {
        // Default: no post-processing
    }

    /// Get the mesh reference.
    fn mesh(&self) -> &Mesh2D;

    /// Get the DG operators reference.
    fn operators(&self) -> &DGOperators2D;

    /// Get the geometric factors reference.
    fn geometry(&self) -> &GeometricFactors2D;

    /// Polynomial order of the DG discretization.
    fn order(&self) -> usize;
}

// =============================================================================
// PhysicsConfig Trait
// =============================================================================

/// Configuration trait for physics module builders.
///
/// Types implementing this trait can be used to configure a physics module
/// before building it.
pub trait PhysicsConfig: Clone {
    /// The type of physics module produced by this configuration.
    type Module;

    /// Build the physics module from this configuration.
    fn build(self) -> Self::Module;
}
