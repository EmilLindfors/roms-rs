//! HLL (Harten-Lax-van Leer) approximate Riemann solver for shallow water.
//!
//! The HLL solver uses a two-wave approximation to the Riemann problem.
//! It is simpler than the Roe solver and more robust for strong shocks,
//! but more diffusive for contact discontinuities.
//!
//! For shallow water equations in 1D, there are only two waves (no contact),
//! so HLL is quite appropriate.
//!
//! F* = (s_r * F_l - s_l * F_r + s_l * s_r * (q_r - q_l)) / (s_r - s_l)
//!
//! where s_l and s_r are the left and right wave speed estimates.
//!
//! Reference: Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics"

use crate::solver::SWEState;

/// HLL numerical flux for 1D shallow water equations.
///
/// Uses Einfeldt wave speed estimates for robustness.
///
/// # Arguments
/// * `q_l` - Left state (h, hu)
/// * `q_r` - Right state (h, hu)
/// * `g` - Gravitational acceleration
/// * `h_min` - Minimum depth for wet/dry treatment
///
/// # Returns
/// Numerical flux as SWEState (F_h, F_hu)
pub fn hll_flux_swe(q_l: &SWEState, q_r: &SWEState, g: f64, h_min: f64) -> SWEState {
    let h_l = q_l.h;
    let h_r = q_r.h;

    // Handle dry cells
    if h_l <= h_min && h_r <= h_min {
        return SWEState::zero();
    }

    // Compute velocities and celerities
    let u_l = if h_l > h_min { q_l.hu / h_l } else { 0.0 };
    let u_r = if h_r > h_min { q_r.hu / h_r } else { 0.0 };

    let c_l = (g * h_l.max(0.0)).sqrt();
    let c_r = (g * h_r.max(0.0)).sqrt();

    // Wave speed estimates (Einfeldt)
    let (s_l, s_r) = einfeldt_speeds(h_l, h_r, u_l, u_r, c_l, c_r, g, h_min);

    // Physical fluxes
    let f_l = SWEState::new(q_l.hu, q_l.hu * u_l + 0.5 * g * h_l * h_l);
    let f_r = SWEState::new(q_r.hu, q_r.hu * u_r + 0.5 * g * h_r * h_r);

    // HLL flux
    if s_l >= 0.0 {
        // All waves go right, use left flux
        f_l
    } else if s_r <= 0.0 {
        // All waves go left, use right flux
        f_r
    } else {
        // Waves go both ways, use HLL average
        let inv_ds = 1.0 / (s_r - s_l);

        let f_h = inv_ds * (s_r * f_l.h - s_l * f_r.h + s_l * s_r * (q_r.h - q_l.h));
        let f_hu = inv_ds * (s_r * f_l.hu - s_l * f_r.hu + s_l * s_r * (q_r.hu - q_l.hu));

        SWEState::new(f_h, f_hu)
    }
}

/// Einfeldt wave speed estimates.
///
/// Uses Roe averages for robust wave speed bounds.
#[allow(clippy::too_many_arguments)]
fn einfeldt_speeds(
    h_l: f64,
    h_r: f64,
    u_l: f64,
    u_r: f64,
    c_l: f64,
    c_r: f64,
    g: f64,
    h_min: f64,
) -> (f64, f64) {
    // Roe-averaged quantities
    let sqrt_h_l = h_l.max(0.0).sqrt();
    let sqrt_h_r = h_r.max(0.0).sqrt();

    let (u_roe, c_roe) = if sqrt_h_l + sqrt_h_r > 1e-10 {
        let h_roe = 0.5 * (h_l + h_r);
        let u_roe = (sqrt_h_l * u_l + sqrt_h_r * u_r) / (sqrt_h_l + sqrt_h_r);
        let c_roe = (g * h_roe).sqrt();
        (u_roe, c_roe)
    } else {
        (0.0, 0.0)
    };

    // Einfeldt estimates: min/max of left, right, and Roe characteristics
    let s_l = if h_l > h_min {
        (u_l - c_l).min(u_roe - c_roe)
    } else {
        u_r - 2.0 * c_r // Dry bed on left
    };

    let s_r = if h_r > h_min {
        (u_r + c_r).max(u_roe + c_roe)
    } else {
        u_l + 2.0 * c_l // Dry bed on right
    };

    (s_l, s_r)
}

/// HLL flux with normal direction for DG interface.
///
/// # Arguments
/// * `q_l` - Interior (minus) state
/// * `q_r` - Exterior (plus) state
/// * `g` - Gravitational acceleration
/// * `h_min` - Minimum depth
/// * `normal` - Outward normal (-1 for left face, +1 for right face)
///
/// # Returns
/// Numerical flux F* · n
pub fn hll_flux_swe_normal(
    q_l: &SWEState,
    q_r: &SWEState,
    g: f64,
    h_min: f64,
    normal: f64,
) -> SWEState {
    if normal > 0.0 {
        hll_flux_swe(q_l, q_r, g, h_min)
    } else {
        let flux = hll_flux_swe(q_r, q_l, g, h_min);
        SWEState::new(-flux.h, -flux.hu)
    }
}

/// Compute the flux jump for DG formulation.
///
/// Returns F* · n - F(q_interior) · n
pub fn hll_flux_jump_swe(
    q_interior: &SWEState,
    q_exterior: &SWEState,
    g: f64,
    h_min: f64,
    normal: f64,
) -> SWEState {
    let f_star = hll_flux_swe_normal(q_interior, q_exterior, g, h_min, normal);

    // Physical flux at interior
    let h = q_interior.h;
    let u = if h > h_min { q_interior.hu / h } else { 0.0 };
    let f_interior = SWEState::new(
        q_interior.hu * normal,
        (q_interior.hu * u + 0.5 * g * h * h) * normal,
    );

    f_star - f_interior
}

/// HLLC flux for shallow water equations.
///
/// The HLLC (HLL with Contact) solver adds a middle wave to better capture
/// contact discontinuities. For shallow water in 1D, this doesn't add much
/// since there's no contact wave, but it can improve accuracy for some problems.
///
/// Note: For 1D SWE, HLLC reduces to HLL since there's no contact discontinuity.
/// This function is provided for API completeness and future 2D extension.
pub fn hllc_flux_swe(q_l: &SWEState, q_r: &SWEState, g: f64, h_min: f64) -> SWEState {
    // For 1D shallow water, HLLC = HLL (no contact wave)
    hll_flux_swe(q_l, q_r, g, h_min)
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 10.0;
    const H_MIN: f64 = 1e-6;
    const TOL: f64 = 1e-10;

    #[test]
    fn test_hll_flux_continuous() {
        // For continuous solution, HLL flux should equal physical flux
        let q = SWEState::new(2.0, 3.0);

        let flux = hll_flux_swe(&q, &q, G, H_MIN);

        let u = 1.5;
        let expected_hu = q.hu * u + 0.5 * G * q.h * q.h;

        assert!((flux.h - q.hu).abs() < TOL);
        assert!((flux.hu - expected_hu).abs() < TOL);
    }

    #[test]
    fn test_hll_flux_still_water() {
        let q_l = SWEState::new(2.0, 0.0);
        let q_r = SWEState::new(2.0, 0.0);

        let flux = hll_flux_swe(&q_l, &q_r, G, H_MIN);

        // f = [0, gh²/2] = [0, 20]
        assert!(flux.h.abs() < TOL);
        assert!((flux.hu - 20.0).abs() < TOL);
    }

    #[test]
    fn test_hll_flux_dam_break() {
        let q_l = SWEState::new(2.0, 0.0);
        let q_r = SWEState::new(0.0, 0.0);

        let flux = hll_flux_swe(&q_l, &q_r, G, H_MIN);

        // Flux should be positive (water flowing right)
        assert!(flux.h > 0.0, "Mass flux should be positive: {}", flux.h);
        assert!(
            flux.hu > 0.0,
            "Momentum flux should be positive: {}",
            flux.hu
        );
    }

    #[test]
    fn test_hll_flux_dry_cells() {
        let q_l = SWEState::new(0.0, 0.0);
        let q_r = SWEState::new(0.0, 0.0);

        let flux = hll_flux_swe(&q_l, &q_r, G, H_MIN);

        assert!(flux.h.abs() < TOL);
        assert!(flux.hu.abs() < TOL);
    }

    #[test]
    fn test_hll_flux_normal_direction() {
        let q_l = SWEState::new(2.0, 3.0);
        let q_r = SWEState::new(1.5, 2.0);

        let flux_neg = hll_flux_swe_normal(&q_l, &q_r, G, H_MIN, -1.0);
        let flux_swapped = hll_flux_swe(&q_r, &q_l, G, H_MIN);

        assert!((flux_neg.h - (-flux_swapped.h)).abs() < TOL);
        assert!((flux_neg.hu - (-flux_swapped.hu)).abs() < TOL);
    }

    #[test]
    fn test_hll_flux_jump_continuous() {
        let q = SWEState::new(2.0, 3.0);

        let jump_left = hll_flux_jump_swe(&q, &q, G, H_MIN, -1.0);
        let jump_right = hll_flux_jump_swe(&q, &q, G, H_MIN, 1.0);

        assert!(jump_left.h.abs() < TOL);
        assert!(jump_left.hu.abs() < TOL);
        assert!(jump_right.h.abs() < TOL);
        assert!(jump_right.hu.abs() < TOL);
    }

    #[test]
    fn test_einfeldt_speeds() {
        // Subcritical flow
        let h = 2.0;
        let u = 1.0;
        let c = (G * h).sqrt();

        let (s_l, s_r) = einfeldt_speeds(h, h, u, u, c, c, G, H_MIN);

        // For uniform flow: s_l ≤ u - c, s_r ≥ u + c
        assert!(s_l <= u - c + TOL);
        assert!(s_r >= u + c - TOL);
    }

    #[test]
    fn test_hll_vs_roe_continuous() {
        use crate::flux::roe_flux_swe;

        // For continuous solution, HLL and Roe should give same result
        let q = SWEState::new(2.0, 3.0);

        let hll = hll_flux_swe(&q, &q, G, H_MIN);
        let roe = roe_flux_swe(&q, &q, G, H_MIN);

        assert!((hll.h - roe.h).abs() < TOL);
        assert!((hll.hu - roe.hu).abs() < TOL);
    }

    #[test]
    fn test_hll_supercritical() {
        // Supercritical flow: both waves go same direction
        // h = 1, u = 10 >> c = sqrt(10) ≈ 3.16
        let q_l = SWEState::new(1.0, 10.0);
        let q_r = SWEState::new(1.0, 10.0);

        let flux = hll_flux_swe(&q_l, &q_r, G, H_MIN);

        // Should equal left flux (all characteristics go right)
        let u = 10.0;
        let expected_h = q_l.hu;
        let expected_hu = q_l.hu * u + 0.5 * G * q_l.h * q_l.h;

        assert!((flux.h - expected_h).abs() < TOL);
        assert!((flux.hu - expected_hu).abs() < TOL);
    }

    #[test]
    fn test_hllc_equals_hll() {
        // For 1D SWE, HLLC should equal HLL
        let q_l = SWEState::new(2.0, 3.0);
        let q_r = SWEState::new(1.5, 1.0);

        let hll = hll_flux_swe(&q_l, &q_r, G, H_MIN);
        let hllc = hllc_flux_swe(&q_l, &q_r, G, H_MIN);

        assert!((hll.h - hllc.h).abs() < TOL);
        assert!((hll.hu - hllc.hu).abs() < TOL);
    }
}
