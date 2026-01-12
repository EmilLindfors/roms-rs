//! Bathymetry source term for shallow water equations.
//!
//! The momentum equation includes the source term -gh ∂B/∂x due to bottom slope.
//! This must be discretized carefully to preserve the lake-at-rest steady state
//! (h + B = const, u = 0).
//!
//! We implement the hydrostatic reconstruction method (Audusse et al. 2004),
//! which modifies the interface states to ensure well-balancing.
//!
//! Reference: Audusse, Bouchut, Bristeau, Klein, Perthame (2004)
//! "A fast and stable well-balanced scheme with hydrostatic reconstruction
//! for shallow water flows", SIAM J. Sci. Comput.

use super::SourceTerm;
use crate::solver::SWEState;

/// Bathymetry source term: S = (0, -gh ∂B/∂x).
///
/// This implements the simple pointwise source term. For well-balanced
/// schemes, use `HydrostaticReconstruction` instead.
#[derive(Clone, Debug)]
pub struct BathymetrySource {
    /// Gravitational acceleration
    pub g: f64,
}

impl BathymetrySource {
    /// Create a new bathymetry source term.
    pub fn new(g: f64) -> Self {
        Self { g }
    }

    /// Standard gravity (9.81 m/s²).
    pub fn standard() -> Self {
        Self::new(9.81)
    }
}

impl SourceTerm for BathymetrySource {
    fn evaluate(&self, state: &SWEState, db_dx: f64, _position: f64, _time: f64) -> SWEState {
        // S = (0, -g h ∂B/∂x)
        SWEState::new(0.0, -self.g * state.h * db_dx)
    }

    fn name(&self) -> &'static str {
        "bathymetry"
    }
}

/// Hydrostatic reconstruction for well-balanced schemes.
///
/// This modifies the interface states at element boundaries to ensure
/// that the lake-at-rest solution (h + B = const, u = 0) is preserved
/// exactly (to machine precision).
///
/// The key idea: at each interface, reconstruct water heights relative
/// to the maximum bed level:
///
/// B* = max(B_L, B_R)
/// h_L* = max(0, η_L - B*) = max(0, h_L + B_L - B*)
/// h_R* = max(0, η_R - B*) = max(0, h_R + B_R - B*)
///
/// where η = h + B is the water surface elevation.
#[derive(Clone, Debug)]
pub struct HydrostaticReconstruction {
    /// Gravitational acceleration
    pub g: f64,
    /// Minimum depth threshold
    pub h_min: f64,
}

impl HydrostaticReconstruction {
    /// Create a new hydrostatic reconstruction.
    pub fn new(g: f64, h_min: f64) -> Self {
        Self { g, h_min }
    }

    /// Standard gravity with default h_min.
    pub fn standard() -> Self {
        Self::new(9.81, 1e-6)
    }

    /// Reconstruct interface states for well-balanced flux computation.
    ///
    /// # Arguments
    /// * `q_l` - Left state (interior)
    /// * `q_r` - Right state (exterior)
    /// * `b_l` - Left bathymetry
    /// * `b_r` - Right bathymetry
    ///
    /// # Returns
    /// Tuple of (reconstructed left state, reconstructed right state)
    pub fn reconstruct(
        &self,
        q_l: &SWEState,
        q_r: &SWEState,
        b_l: f64,
        b_r: f64,
    ) -> (SWEState, SWEState) {
        // Water surface elevations
        let eta_l = q_l.h + b_l;
        let eta_r = q_r.h + b_r;

        // Interface bathymetry (maximum of both sides)
        let b_star = b_l.max(b_r);

        // Reconstructed water depths
        let h_l_star = (eta_l - b_star).max(0.0);
        let h_r_star = (eta_r - b_star).max(0.0);

        // Reconstruct momentum to preserve velocity
        // u = hu / h, so hu_star = h_star * u = h_star * (hu / h)
        let hu_l_star = if q_l.h > self.h_min {
            h_l_star * (q_l.hu / q_l.h)
        } else {
            0.0
        };

        let hu_r_star = if q_r.h > self.h_min {
            h_r_star * (q_r.hu / q_r.h)
        } else {
            0.0
        };

        (
            SWEState::new(h_l_star, hu_l_star),
            SWEState::new(h_r_star, hu_r_star),
        )
    }

    /// Compute the well-balanced source term contribution at an interface.
    ///
    /// This computes the correction needed to balance the flux difference
    /// with the bathymetry source term.
    ///
    /// # Arguments
    /// * `q_l` - Left state
    /// * `q_r` - Right state
    /// * `b_l` - Left bathymetry
    /// * `b_r` - Right bathymetry
    ///
    /// # Returns
    /// Source term contribution as (S_left, S_right) for the two cells
    pub fn interface_source(
        &self,
        q_l: &SWEState,
        _q_r: &SWEState,
        b_l: f64,
        b_r: f64,
    ) -> (SWEState, SWEState) {
        // The well-balanced correction accounts for the difference between
        // the flux computed with reconstructed states and the original states.
        // This is essentially the pressure force from the "step" in bathymetry.

        let b_star = b_l.max(b_r);
        let delta_b_l = b_star - b_l;
        let delta_b_r = b_star - b_r;

        // Pressure correction: 0.5 * g * h² evaluated at the step
        let h_l = q_l.h;

        // Source for left cell (from right interface)
        let s_l = SWEState::new(
            0.0,
            -0.5 * self.g * (h_l * h_l - (h_l - delta_b_l).max(0.0).powi(2)),
        );

        // Source for right cell (from left interface)
        let s_r = SWEState::new(
            0.0,
            0.5 * self.g * (h_l * h_l - (h_l - delta_b_r).max(0.0).powi(2)),
        );

        (s_l, s_r)
    }

    /// Check if a state satisfies lake-at-rest condition.
    ///
    /// Returns true if |u| < tolerance and h + B is approximately constant.
    pub fn is_lake_at_rest(
        &self,
        q: &SWEState,
        b: f64,
        reference_eta: f64,
        velocity_tol: f64,
        elevation_tol: f64,
    ) -> bool {
        let u = if q.h > self.h_min { q.hu / q.h } else { 0.0 };

        let eta = q.h + b;

        u.abs() < velocity_tol && (eta - reference_eta).abs() < elevation_tol
    }
}

impl SourceTerm for HydrostaticReconstruction {
    fn evaluate(&self, state: &SWEState, db_dx: f64, _position: f64, _time: f64) -> SWEState {
        // For the volume integral, we still use the standard source term.
        // The well-balancing magic happens at the interfaces through
        // the reconstruct() method.
        SWEState::new(0.0, -self.g * state.h * db_dx)
    }

    fn name(&self) -> &'static str {
        "hydrostatic_reconstruction"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 10.0;
    const H_MIN: f64 = 1e-6;
    const TOL: f64 = 1e-12;

    #[test]
    fn test_bathymetry_source_flat() {
        let source = BathymetrySource::new(G);
        let state = SWEState::new(2.0, 3.0);

        // Flat bottom: dB/dx = 0
        let s = source.evaluate(&state, 0.0, 0.0, 0.0);

        assert!(s.h.abs() < TOL);
        assert!(s.hu.abs() < TOL);
    }

    #[test]
    fn test_bathymetry_source_slope() {
        let source = BathymetrySource::new(G);
        let state = SWEState::new(2.0, 3.0);

        // Upward slope: dB/dx = 0.1
        let s = source.evaluate(&state, 0.1, 0.0, 0.0);

        // S_hu = -g * h * dB/dx = -10 * 2 * 0.1 = -2
        assert!(s.h.abs() < TOL);
        assert!((s.hu - (-2.0)).abs() < TOL);
    }

    #[test]
    fn test_hydrostatic_reconstruction_lake_at_rest() {
        let hr = HydrostaticReconstruction::new(G, H_MIN);

        // Lake at rest: h + B = const = 2.0, u = 0
        let eta = 2.0;
        let b_l = 0.5;
        let b_r = 1.0;
        let h_l = eta - b_l; // 1.5
        let h_r = eta - b_r; // 1.0

        let q_l = SWEState::new(h_l, 0.0);
        let q_r = SWEState::new(h_r, 0.0);

        let (q_l_star, q_r_star) = hr.reconstruct(&q_l, &q_r, b_l, b_r);

        // B* = max(0.5, 1.0) = 1.0
        // h_L* = max(0, 2.0 - 1.0) = 1.0
        // h_R* = max(0, 2.0 - 1.0) = 1.0
        // Both reconstructed states should have same h!

        assert!(
            (q_l_star.h - q_r_star.h).abs() < TOL,
            "Lake at rest: h_L* = {}, h_R* = {} should be equal",
            q_l_star.h,
            q_r_star.h
        );

        // Both should have zero velocity
        assert!(q_l_star.hu.abs() < TOL);
        assert!(q_r_star.hu.abs() < TOL);
    }

    #[test]
    fn test_hydrostatic_reconstruction_preserves_velocity() {
        let hr = HydrostaticReconstruction::new(G, H_MIN);

        // Moving water over varying bathymetry
        let h_l = 2.0;
        let u_l = 1.5;
        let b_l = 0.5;

        let h_r = 1.5;
        let u_r = 2.0;
        let b_r = 1.0;

        let q_l = SWEState::from_primitives(h_l, u_l);
        let q_r = SWEState::from_primitives(h_r, u_r);

        let (q_l_star, q_r_star) = hr.reconstruct(&q_l, &q_r, b_l, b_r);

        // Velocity should be preserved
        let u_l_star = if q_l_star.h > H_MIN {
            q_l_star.hu / q_l_star.h
        } else {
            0.0
        };
        let u_r_star = if q_r_star.h > H_MIN {
            q_r_star.hu / q_r_star.h
        } else {
            0.0
        };

        assert!(
            (u_l_star - u_l).abs() < TOL,
            "Velocity should be preserved: u_L* = {}, u_L = {}",
            u_l_star,
            u_l
        );
        assert!(
            (u_r_star - u_r).abs() < TOL,
            "Velocity should be preserved: u_R* = {}, u_R = {}",
            u_r_star,
            u_r
        );
    }

    #[test]
    fn test_hydrostatic_reconstruction_dry_bed() {
        let hr = HydrostaticReconstruction::new(G, H_MIN);

        // Wet on left, dry on right
        let q_l = SWEState::new(1.0, 0.5);
        let q_r = SWEState::new(0.0, 0.0);
        let b_l = 0.0;
        let b_r = 2.0; // Bed is above water surface on right

        let (q_l_star, q_r_star) = hr.reconstruct(&q_l, &q_r, b_l, b_r);

        // Left state should be limited by right bed
        // eta_L = 1.0 + 0.0 = 1.0
        // B* = max(0, 2) = 2
        // h_L* = max(0, 1.0 - 2.0) = 0 (water surface below bed)

        assert!(
            q_l_star.h < TOL,
            "Left should be dry at interface: h_L* = {}",
            q_l_star.h
        );
        assert!(
            q_r_star.h < TOL,
            "Right should be dry: h_R* = {}",
            q_r_star.h
        );
    }

    #[test]
    fn test_is_lake_at_rest() {
        let hr = HydrostaticReconstruction::new(G, H_MIN);

        let q = SWEState::new(1.5, 0.0);
        let b = 0.5;
        let eta = 2.0;

        assert!(hr.is_lake_at_rest(&q, b, eta, 1e-10, 1e-10));

        // Moving water is not at rest
        let q_moving = SWEState::new(1.5, 1.5); // u = 1
        assert!(!hr.is_lake_at_rest(&q_moving, b, eta, 1e-10, 1e-10));

        // Wrong surface elevation
        let q_wrong = SWEState::new(2.0, 0.0);
        assert!(!hr.is_lake_at_rest(&q_wrong, b, eta, 1e-10, 1e-10));
    }

    #[test]
    fn test_same_bathymetry_no_reconstruction() {
        let hr = HydrostaticReconstruction::new(G, H_MIN);

        // Same bathymetry on both sides
        let q_l = SWEState::new(2.0, 3.0);
        let q_r = SWEState::new(1.5, 2.0);
        let b = 0.5;

        let (q_l_star, q_r_star) = hr.reconstruct(&q_l, &q_r, b, b);

        // States should be unchanged
        assert!((q_l_star.h - q_l.h).abs() < TOL);
        assert!((q_l_star.hu - q_l.hu).abs() < TOL);
        assert!((q_r_star.h - q_r.h).abs() < TOL);
        assert!((q_r_star.hu - q_r.hu).abs() < TOL);
    }
}
