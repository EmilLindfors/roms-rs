//! Reflective (wall) boundary condition for shallow water equations.
//!
//! A reflective boundary represents a solid wall where no flow can pass.
//! The normal velocity component is reversed while the tangential component
//! (in 2D) is preserved.
//!
//! For 1D: u_ghost = -u_interior, h_ghost = h_interior
//!
//! This ensures zero normal mass flux through the boundary.

use super::{BCContext, SWEBoundaryCondition};
use crate::solver::SWEState;

/// Reflective (wall) boundary condition.
///
/// Creates a mirror state with reversed velocity, ensuring zero mass flux
/// through the boundary. This is appropriate for:
/// - Solid walls (coastlines, structures)
/// - Closed basin boundaries
/// - Symmetric domain boundaries
///
/// # Mathematical Formulation
///
/// For a boundary with outward normal n:
/// - h_ghost = h_interior
/// - (u·n)_ghost = -(u·n)_interior
///
/// In 1D, this simplifies to:
/// - h_ghost = h_interior
/// - u_ghost = -u_interior
/// - hu_ghost = -hu_interior
#[derive(Clone, Debug, Default)]
pub struct ReflectiveBC {
    /// Minimum depth for velocity computation
    pub h_min: f64,
}

impl ReflectiveBC {
    /// Create a new reflective BC.
    pub fn new() -> Self {
        Self { h_min: 1e-6 }
    }

    /// Create with custom minimum depth.
    pub fn with_h_min(h_min: f64) -> Self {
        Self { h_min }
    }
}

impl SWEBoundaryCondition for ReflectiveBC {
    fn ghost_state(&self, ctx: &BCContext) -> SWEState {
        // Mirror the velocity: u_ghost = -u_interior
        // Keep the same depth: h_ghost = h_interior
        SWEState::new(ctx.interior_state.h, -ctx.interior_state.hu)
    }

    fn name(&self) -> &'static str {
        "reflective"
    }

    fn allows_inflow(&self) -> bool {
        false
    }

    fn allows_outflow(&self) -> bool {
        false
    }
}

/// Partial slip wall boundary condition.
///
/// A generalization of the reflective BC that allows some slip at the wall.
/// The reflected velocity is scaled by a factor (1 - slip):
/// - slip = 0: perfect reflection (no slip)
/// - slip = 1: free slip (no velocity change in tangential direction)
///
/// In 1D, this affects the magnitude of the reflected velocity.
#[derive(Clone, Debug)]
pub struct PartialSlipBC {
    /// Slip coefficient (0 = no slip, 1 = free slip)
    pub slip: f64,
    /// Minimum depth
    pub h_min: f64,
}

impl PartialSlipBC {
    /// Create a new partial slip BC.
    pub fn new(slip: f64) -> Self {
        Self {
            slip: slip.clamp(0.0, 1.0),
            h_min: 1e-6,
        }
    }

    /// No-slip wall (equivalent to reflective).
    pub fn no_slip() -> Self {
        Self::new(0.0)
    }

    /// Free-slip wall.
    pub fn free_slip() -> Self {
        Self::new(1.0)
    }
}

impl SWEBoundaryCondition for PartialSlipBC {
    fn ghost_state(&self, ctx: &BCContext) -> SWEState {
        // Scale the reflected velocity by (1 - slip)
        // slip = 0: hu_ghost = -hu_interior (full reflection)
        // slip = 1: hu_ghost = 0 (no reflection)
        let factor = -(1.0 - self.slip);
        SWEState::new(ctx.interior_state.h, factor * ctx.interior_state.hu)
    }

    fn name(&self) -> &'static str {
        "partial_slip"
    }

    fn allows_inflow(&self) -> bool {
        false
    }

    fn allows_outflow(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn make_context(h: f64, hu: f64, normal: f64) -> BCContext {
        BCContext::new(0.0, 0.0, SWEState::new(h, hu), 0.0, normal)
    }

    #[test]
    fn test_reflective_still_water() {
        let bc = ReflectiveBC::new();
        let ctx = make_context(2.0, 0.0, 1.0);

        let ghost = bc.ghost_state(&ctx);

        // Still water should remain still
        assert!((ghost.h - 2.0).abs() < TOL);
        assert!(ghost.hu.abs() < TOL);
    }

    #[test]
    fn test_reflective_moving_water() {
        let bc = ReflectiveBC::new();

        // Flow towards right boundary (normal = +1)
        let ctx = make_context(2.0, 4.0, 1.0);
        let ghost = bc.ghost_state(&ctx);

        // Velocity should be reversed
        assert!((ghost.h - 2.0).abs() < TOL);
        assert!((ghost.hu - (-4.0)).abs() < TOL);
    }

    #[test]
    fn test_reflective_preserves_depth() {
        let bc = ReflectiveBC::new();

        for h in [0.1, 1.0, 10.0] {
            let ctx = make_context(h, h * 2.0, 1.0);
            let ghost = bc.ghost_state(&ctx);

            assert!((ghost.h - h).abs() < TOL, "Depth should be preserved");
        }
    }

    #[test]
    fn test_reflective_zero_flux() {
        let bc = ReflectiveBC::new();
        let ctx = make_context(2.0, 4.0, 1.0);
        let ghost = bc.ghost_state(&ctx);

        // Average velocity at interface should be zero
        let u_interior = ctx.interior_state.hu / ctx.interior_state.h;
        let u_ghost = ghost.hu / ghost.h;
        let u_avg = 0.5 * (u_interior + u_ghost);

        assert!(u_avg.abs() < TOL, "Average velocity should be zero at wall");
    }

    #[test]
    fn test_partial_slip_no_slip() {
        let bc = PartialSlipBC::no_slip();
        let reflective = ReflectiveBC::new();

        let ctx = make_context(2.0, 4.0, 1.0);

        let ghost_slip = bc.ghost_state(&ctx);
        let ghost_ref = reflective.ghost_state(&ctx);

        // No-slip should equal reflective
        assert!((ghost_slip.h - ghost_ref.h).abs() < TOL);
        assert!((ghost_slip.hu - ghost_ref.hu).abs() < TOL);
    }

    #[test]
    fn test_partial_slip_free_slip() {
        let bc = PartialSlipBC::free_slip();
        let ctx = make_context(2.0, 4.0, 1.0);

        let ghost = bc.ghost_state(&ctx);

        // Free slip: reflected momentum should be zero
        assert!((ghost.h - 2.0).abs() < TOL);
        assert!(ghost.hu.abs() < TOL);
    }

    #[test]
    fn test_partial_slip_half() {
        let bc = PartialSlipBC::new(0.5);
        let ctx = make_context(2.0, 4.0, 1.0);

        let ghost = bc.ghost_state(&ctx);

        // Half slip: reflected momentum = -0.5 * original
        assert!((ghost.h - 2.0).abs() < TOL);
        assert!((ghost.hu - (-2.0)).abs() < TOL);
    }

    #[test]
    fn test_reflective_allows_no_flow() {
        let bc = ReflectiveBC::new();
        assert!(!bc.allows_inflow());
        assert!(!bc.allows_outflow());
    }
}
