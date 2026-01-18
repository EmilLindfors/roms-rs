//! Trait definitions for slope limiters.

use crate::mesh::Mesh2D;
use crate::operators::DGOperators2D;

use super::super::state::SWESolution2D;

/// Context provided to limiter computations.
#[derive(Clone, Copy)]
pub struct LimiterContext2D<'a> {
    /// The mesh
    pub mesh: &'a Mesh2D,
    /// DG operators
    pub ops: &'a DGOperators2D,
    /// Minimum element size
    pub h_min: f64,
}

impl<'a> std::fmt::Debug for LimiterContext2D<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LimiterContext2D")
            .field("n_elements", &self.mesh.n_elements)
            .field("h_min", &self.h_min)
            .finish()
    }
}

impl<'a> LimiterContext2D<'a> {
    /// Create a new limiter context.
    pub fn new(mesh: &'a Mesh2D, ops: &'a DGOperators2D) -> Self {
        Self {
            mesh,
            ops,
            h_min: mesh.h_min(),
        }
    }

    /// Create a limiter context with a custom h_min.
    pub fn with_h_min(mesh: &'a Mesh2D, ops: &'a DGOperators2D, h_min: f64) -> Self {
        Self { mesh, ops, h_min }
    }
}

/// Trait for slope limiters in 2D DG methods.
///
/// Limiters control oscillations near discontinuities while preserving
/// high-order accuracy in smooth regions.
///
/// # Implementation Notes
///
/// - Limiters should preserve cell averages (total mass)
/// - Limiters modify the solution in-place
/// - The `apply` method should not allocate memory in hot paths
///
/// # Extending
///
/// To add a new limiter:
/// 1. Create a struct with limiter parameters
/// 2. Implement `Limiter2D` for it
/// 3. Use `LimiterChain2D` to combine with other limiters
pub trait Limiter2D: Send + Sync {
    /// Apply the limiter to the solution in-place.
    ///
    /// # Arguments
    /// * `solution` - SWE solution to limit (modified in place)
    /// * `ctx` - Limiter context with mesh and operators
    fn apply(&self, solution: &mut SWESolution2D, ctx: &LimiterContext2D);

    /// Human-readable name for debugging and logging.
    fn name(&self) -> &'static str;

    /// Whether this limiter preserves positivity of depth.
    fn preserves_positivity(&self) -> bool {
        false
    }

    /// Whether this limiter preserves cell averages exactly.
    fn preserves_cell_average(&self) -> bool {
        true
    }
}

/// Type alias for boxed limiter (runtime polymorphism).
pub type BoxedLimiter2D = Box<dyn Limiter2D>;
