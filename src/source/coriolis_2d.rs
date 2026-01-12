//! Coriolis source term for 2D shallow water equations.
//!
//! The Coriolis effect arises from Earth's rotation and deflects moving
//! water to the right in the Northern Hemisphere. The source term is:
//!
//! S_coriolis = [0, f·hv, -f·hu]ᵀ
//!
//! where f is the Coriolis parameter:
//! - f-plane: f = f₀ (constant)
//! - β-plane: f(y) = f₀ + β(y - y₀)
//!
//! # Norwegian Coast Context
//!
//! At 60°N latitude (Norwegian coast):
//! - f ≈ 1.2×10⁻⁴ s⁻¹
//! - Coriolis effects are significant for mesoscale circulation
//! - β ≈ 1.6×10⁻¹¹ m⁻¹ s⁻¹

use super::{SourceContext2D, SourceTerm2D};
use crate::solver::SWEState2D;

/// Coriolis source term implementing SourceTerm2D trait.
///
/// Supports both f-plane (constant f) and β-plane (f varies with y).
///
/// # Example
///
/// ```
/// use dg_rs::source::{CoriolisSource2D, SourceTerm2D, SourceContext2D};
/// use dg_rs::SWEState2D;
///
/// // Norwegian coast at 60°N
/// let coriolis = CoriolisSource2D::norwegian_coast();
///
/// let ctx = SourceContext2D::new(
///     0.0,
///     (0.0, 0.0),
///     SWEState2D::new(10.0, 100.0, 50.0), // h=10, hu=100, hv=50
///     0.0,
///     (0.0, 0.0),
///     9.81,
///     1e-6,
/// );
///
/// let source = coriolis.evaluate(&ctx);
/// // source.hu = f * hv = 1.2e-4 * 50 = 0.006
/// // source.hv = -f * hu = -1.2e-4 * 100 = -0.012
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CoriolisSource2D {
    /// Coriolis parameter f₀ (s⁻¹)
    pub f0: f64,
    /// Beta-plane parameter β = ∂f/∂y (m⁻¹ s⁻¹)
    pub beta: f64,
    /// Reference y-coordinate for beta-plane (m)
    pub y0: f64,
}

impl CoriolisSource2D {
    /// Create a new Coriolis source with full parameters.
    pub fn new(f0: f64, beta: f64, y0: f64) -> Self {
        Self { f0, beta, y0 }
    }

    /// Create f-plane Coriolis (constant f).
    ///
    /// # Arguments
    /// * `f` - Coriolis parameter (s⁻¹)
    pub fn f_plane(f: f64) -> Self {
        Self {
            f0: f,
            beta: 0.0,
            y0: 0.0,
        }
    }

    /// Create beta-plane Coriolis: f(y) = f₀ + β(y - y₀).
    ///
    /// # Arguments
    /// * `f0` - Coriolis parameter at reference latitude (s⁻¹)
    /// * `beta` - Rate of change of f with latitude (m⁻¹ s⁻¹)
    /// * `y0` - Reference y-coordinate (m)
    pub fn beta_plane(f0: f64, beta: f64, y0: f64) -> Self {
        Self { f0, beta, y0 }
    }

    /// Norwegian coast at 60°N latitude.
    ///
    /// Uses f ≈ 1.2×10⁻⁴ s⁻¹ (f-plane approximation).
    pub fn norwegian_coast() -> Self {
        Self::f_plane(1.2e-4)
    }

    /// Norwegian coast with beta-plane.
    ///
    /// Uses f₀ ≈ 1.2×10⁻⁴ s⁻¹ and β ≈ 1.6×10⁻¹¹ m⁻¹ s⁻¹.
    pub fn norwegian_coast_beta() -> Self {
        Self::beta_plane(1.2e-4, 1.6e-11, 0.0)
    }

    /// Compute Coriolis parameter at given y-coordinate.
    ///
    /// f(y) = f₀ + β(y - y₀)
    #[inline]
    pub fn f_at(&self, y: f64) -> f64 {
        self.f0 + self.beta * (y - self.y0)
    }

    /// Check if this is a pure f-plane (no beta variation).
    pub fn is_f_plane(&self) -> bool {
        self.beta.abs() < 1e-20
    }
}

impl SourceTerm2D for CoriolisSource2D {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        let (_x, y) = ctx.position;
        let f = self.f_at(y);

        // S_coriolis = [0, f·hv, -f·hu]
        SWEState2D {
            h: 0.0,
            hu: f * ctx.state.hv,
            hv: -f * ctx.state.hu,
        }
    }

    fn name(&self) -> &'static str {
        "coriolis_2d"
    }
}

impl Default for CoriolisSource2D {
    /// Default is no Coriolis (f = 0).
    fn default() -> Self {
        Self::f_plane(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-14;

    fn make_context(h: f64, hu: f64, hv: f64, y: f64) -> SourceContext2D {
        SourceContext2D::new(
            0.0,
            (0.0, y),
            SWEState2D::new(h, hu, hv),
            0.0,
            (0.0, 0.0),
            9.81,
            1e-6,
        )
    }

    #[test]
    fn test_f_plane_source() {
        let coriolis = CoriolisSource2D::f_plane(1.0e-4);
        let ctx = make_context(10.0, 100.0, 50.0, 0.0);

        let source = coriolis.evaluate(&ctx);

        // S = [0, f·hv, -f·hu] = [0, 1e-4 * 50, -1e-4 * 100]
        assert!(source.h.abs() < TOL);
        assert!((source.hu - 0.005).abs() < TOL);
        assert!((source.hv - (-0.01)).abs() < TOL);
    }

    #[test]
    fn test_beta_plane_f_variation() {
        let coriolis = CoriolisSource2D::beta_plane(1.0e-4, 1.0e-11, 0.0);

        // At y = 0: f = f0
        assert!((coriolis.f_at(0.0) - 1.0e-4).abs() < TOL);

        // At y = 1e6 m (1000 km): f = f0 + beta * y = 1e-4 + 1e-11 * 1e6 = 1.1e-4
        assert!((coriolis.f_at(1.0e6) - 1.1e-4).abs() < TOL);
    }

    #[test]
    fn test_beta_plane_source() {
        let coriolis = CoriolisSource2D::beta_plane(1.0e-4, 1.0e-11, 0.0);

        // At y = 1e6 m, f = 1.1e-4
        let ctx = make_context(10.0, 100.0, 50.0, 1.0e6);
        let source = coriolis.evaluate(&ctx);

        // S = [0, f·hv, -f·hu] = [0, 1.1e-4 * 50, -1.1e-4 * 100]
        assert!(source.h.abs() < TOL);
        assert!((source.hu - 0.0055).abs() < TOL);
        assert!((source.hv - (-0.011)).abs() < TOL);
    }

    #[test]
    fn test_norwegian_coast() {
        let coriolis = CoriolisSource2D::norwegian_coast();
        assert!((coriolis.f0 - 1.2e-4).abs() < TOL);
        assert!(coriolis.is_f_plane());
    }

    #[test]
    fn test_zero_coriolis() {
        let coriolis = CoriolisSource2D::default();
        let ctx = make_context(10.0, 100.0, 50.0, 0.0);

        let source = coriolis.evaluate(&ctx);

        assert!(source.h.abs() < TOL);
        assert!(source.hu.abs() < TOL);
        assert!(source.hv.abs() < TOL);
    }

    #[test]
    fn test_coriolis_sign_convention() {
        // Northern hemisphere: Coriolis deflects to the right
        // Eastward flow (positive hu) should get deflected southward (negative hv tendency)
        let coriolis = CoriolisSource2D::f_plane(1.0e-4);
        let ctx = make_context(10.0, 100.0, 0.0, 0.0); // Pure eastward flow

        let source = coriolis.evaluate(&ctx);

        // hu tendency from hv is zero (hv = 0)
        // hv tendency from -f*hu should be negative
        assert!(
            source.hv < 0.0,
            "Eastward flow should be deflected southward"
        );
    }

    #[test]
    fn test_geostrophic_balance_tendency() {
        // In geostrophic balance: Coriolis = pressure gradient
        // Test that northward flow (positive hv) creates positive hu tendency
        let coriolis = CoriolisSource2D::f_plane(1.0e-4);
        let ctx = make_context(10.0, 0.0, 100.0, 0.0); // Pure northward flow

        let source = coriolis.evaluate(&ctx);

        // hu tendency from f*hv should be positive
        assert!(
            source.hu > 0.0,
            "Northward flow should gain eastward momentum"
        );
    }
}
