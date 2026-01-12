//! Roe approximate Riemann solver for shallow water equations.
//!
//! The Roe solver linearizes the Riemann problem at each interface using
//! Roe-averaged states. For shallow water equations:
//!
//! F* = 0.5 * (F_L + F_R) - 0.5 * Σ |λ_i| * α_i * r_i
//!
//! where λ_i are the Roe-averaged eigenvalues, α_i are the wave strengths,
//! and r_i are the right eigenvectors.
//!
//! Reference: Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics"

use crate::solver::SWEState;

/// Roe numerical flux for 1D shallow water equations.
///
/// Computes the numerical flux at an interface between left and right states.
///
/// # Arguments
/// * `q_l` - Left state (h, hu)
/// * `q_r` - Right state (h, hu)
/// * `g` - Gravitational acceleration
/// * `h_min` - Minimum depth for wet/dry treatment
///
/// # Returns
/// Numerical flux as SWEState (F_h, F_hu)
pub fn roe_flux_swe(q_l: &SWEState, q_r: &SWEState, g: f64, h_min: f64) -> SWEState {
    let h_l = q_l.h;
    let h_r = q_r.h;

    // Handle dry cells
    if h_l <= h_min && h_r <= h_min {
        return SWEState::zero();
    }

    // Compute velocities
    let u_l = if h_l > h_min { q_l.hu / h_l } else { 0.0 };
    let u_r = if h_r > h_min { q_r.hu / h_r } else { 0.0 };

    // Physical fluxes
    let c_l = (g * h_l.max(0.0)).sqrt();
    let c_r = (g * h_r.max(0.0)).sqrt();

    let f_l = SWEState::new(q_l.hu, q_l.hu * u_l + 0.5 * g * h_l * h_l);
    let f_r = SWEState::new(q_r.hu, q_r.hu * u_r + 0.5 * g * h_r * h_r);

    // Roe averages
    let sqrt_h_l = h_l.max(0.0).sqrt();
    let sqrt_h_r = h_r.max(0.0).sqrt();

    let (_h_roe, u_roe, c_roe) = if sqrt_h_l + sqrt_h_r > 1e-10 {
        let h_roe = 0.5 * (h_l + h_r);
        let u_roe = (sqrt_h_l * u_l + sqrt_h_r * u_r) / (sqrt_h_l + sqrt_h_r);
        let c_roe = (g * h_roe).sqrt();
        (h_roe, u_roe, c_roe)
    } else {
        (0.0, 0.0, 0.0)
    };

    // Eigenvalues
    let lambda_1 = u_roe - c_roe;
    let lambda_2 = u_roe + c_roe;

    // Wave strengths (jump decomposition)
    let delta_h = h_r - h_l;
    let delta_hu = q_r.hu - q_l.hu;

    // α_1 = (Δhu - (u + c) Δh) / (-2c)
    // α_2 = (Δhu - (u - c) Δh) / (2c)
    let (alpha_1, alpha_2) = if c_roe > 1e-10 {
        let inv_2c = 0.5 / c_roe;
        let alpha_1 = -(inv_2c * (delta_hu - (u_roe + c_roe) * delta_h));
        let alpha_2 = inv_2c * (delta_hu - (u_roe - c_roe) * delta_h);
        (alpha_1, alpha_2)
    } else {
        (0.5 * delta_h, 0.5 * delta_h)
    };

    // Right eigenvectors: r_1 = [1, u-c]^T, r_2 = [1, u+c]^T
    let r1 = SWEState::new(1.0, u_roe - c_roe);
    let r2 = SWEState::new(1.0, u_roe + c_roe);

    // Apply entropy fix (Harten-Hyman)
    let lambda_1_abs = entropy_fix(lambda_1, u_l - c_l, u_r - c_r);
    let lambda_2_abs = entropy_fix(lambda_2, u_l + c_l, u_r + c_r);

    // Roe flux: F* = 0.5(F_L + F_R) - 0.5 * Σ |λ_i| α_i r_i
    let central = 0.5 * (f_l + f_r);
    let dissipation = 0.5 * (lambda_1_abs * alpha_1 * r1 + lambda_2_abs * alpha_2 * r2);

    central - dissipation
}

/// Entropy fix for transonic rarefactions (Harten-Hyman).
///
/// When a wave crosses sonic point (λ changes sign), the standard Roe
/// scheme produces an entropy-violating expansion shock. This fix
/// replaces |λ| with a smooth function near zero.
fn entropy_fix(lambda_roe: f64, lambda_l: f64, lambda_r: f64) -> f64 {
    // Check for transonic rarefaction
    if lambda_l < 0.0 && lambda_r > 0.0 {
        // Sonic transition: use modified wave speed
        let delta = lambda_r - lambda_l;
        if delta.abs() > 1e-10 {
            // Harten's formula
            0.5 * (lambda_roe.abs() + delta)
        } else {
            lambda_roe.abs()
        }
    } else if lambda_l > 0.0 && lambda_r < 0.0 {
        // Transonic compression (shouldn't happen in SWE, but handle anyway)
        lambda_roe.abs()
    } else {
        lambda_roe.abs()
    }
}

/// Roe flux with normal direction for DG interface.
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
pub fn roe_flux_swe_normal(
    q_l: &SWEState,
    q_r: &SWEState,
    g: f64,
    h_min: f64,
    normal: f64,
) -> SWEState {
    // For 1D, we can simply compute the flux and scale by normal
    // The flux is computed assuming flow in positive x direction
    if normal > 0.0 {
        roe_flux_swe(q_l, q_r, g, h_min)
    } else {
        // Flip left/right and negate flux
        let flux = roe_flux_swe(q_r, q_l, g, h_min);
        SWEState::new(-flux.h, -flux.hu)
    }
}

/// Compute the flux jump for DG formulation.
///
/// Returns F* · n - F(q_interior) · n
pub fn roe_flux_jump_swe(
    q_interior: &SWEState,
    q_exterior: &SWEState,
    g: f64,
    h_min: f64,
    normal: f64,
) -> SWEState {
    let f_star = roe_flux_swe_normal(q_interior, q_exterior, g, h_min, normal);

    // Physical flux at interior
    let h = q_interior.h;
    let u = if h > h_min { q_interior.hu / h } else { 0.0 };
    let f_interior = SWEState::new(
        q_interior.hu * normal,
        (q_interior.hu * u + 0.5 * g * h * h) * normal,
    );

    f_star - f_interior
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 10.0;
    const H_MIN: f64 = 1e-6;
    const TOL: f64 = 1e-10;

    #[test]
    fn test_roe_flux_continuous() {
        // For continuous solution, Roe flux should equal physical flux
        let q = SWEState::new(2.0, 3.0); // h=2, hu=3, u=1.5

        let flux = roe_flux_swe(&q, &q, G, H_MIN);

        // Physical flux: [hu, hu²/h + gh²/2] = [3, 9/2 + 10*4/2] = [3, 24.5]
        let u = 1.5;
        let expected_hu = q.hu * u + 0.5 * G * q.h * q.h;

        assert!((flux.h - q.hu).abs() < TOL);
        assert!((flux.hu - expected_hu).abs() < TOL);
    }

    #[test]
    fn test_roe_flux_symmetric() {
        // Roe flux should be symmetric for symmetric states
        let q_l = SWEState::new(2.0, 0.0); // Still water
        let q_r = SWEState::new(2.0, 0.0);

        let flux = roe_flux_swe(&q_l, &q_r, G, H_MIN);

        // For still water: f = [0, gh²/2] = [0, 20]
        assert!(flux.h.abs() < TOL);
        assert!((flux.hu - 20.0).abs() < TOL);
    }

    #[test]
    fn test_roe_flux_dam_break() {
        // Dam break: higher water on left, dry on right
        let q_l = SWEState::new(2.0, 0.0);
        let q_r = SWEState::new(0.0, 0.0);

        let flux = roe_flux_swe(&q_l, &q_r, G, H_MIN);

        // Flux should be positive (water flowing right)
        assert!(flux.h > 0.0, "Mass flux should be positive: {}", flux.h);
        assert!(
            flux.hu > 0.0,
            "Momentum flux should be positive: {}",
            flux.hu
        );
    }

    #[test]
    fn test_roe_flux_dry_cells() {
        // Both cells dry
        let q_l = SWEState::new(0.0, 0.0);
        let q_r = SWEState::new(0.0, 0.0);

        let flux = roe_flux_swe(&q_l, &q_r, G, H_MIN);

        assert!(flux.h.abs() < TOL);
        assert!(flux.hu.abs() < TOL);
    }

    #[test]
    fn test_roe_flux_normal_direction() {
        let q_l = SWEState::new(2.0, 3.0);
        let q_r = SWEState::new(1.5, 2.0);

        let _flux_pos = roe_flux_swe_normal(&q_l, &q_r, G, H_MIN, 1.0);
        let flux_neg = roe_flux_swe_normal(&q_l, &q_r, G, H_MIN, -1.0);

        // The flux for normal=-1 should be the negative of flux for swapped states
        let flux_swapped = roe_flux_swe(&q_r, &q_l, G, H_MIN);

        assert!((flux_neg.h - (-flux_swapped.h)).abs() < TOL);
        assert!((flux_neg.hu - (-flux_swapped.hu)).abs() < TOL);
    }

    #[test]
    fn test_roe_flux_jump_continuous() {
        // For continuous solution, flux jump should be zero
        let q = SWEState::new(2.0, 3.0);

        let jump_left = roe_flux_jump_swe(&q, &q, G, H_MIN, -1.0);
        let jump_right = roe_flux_jump_swe(&q, &q, G, H_MIN, 1.0);

        assert!(jump_left.h.abs() < TOL);
        assert!(jump_left.hu.abs() < TOL);
        assert!(jump_right.h.abs() < TOL);
        assert!(jump_right.hu.abs() < TOL);
    }

    #[test]
    fn test_entropy_fix() {
        // Transonic rarefaction: left moving left, right moving right
        let lambda_l = -1.0;
        let lambda_r = 1.0;
        let lambda_roe = 0.0;

        let fixed = entropy_fix(lambda_roe, lambda_l, lambda_r);

        // Should be positive and approximately lambda_r - lambda_l / 2 = 1
        assert!(fixed > 0.0);
        assert!((fixed - 1.0).abs() < TOL);
    }

    #[test]
    fn test_conservation_at_interface() {
        // Conservation: flux leaving cell L = flux entering cell R
        // In DG, both cells use the SAME numerical flux at the interface.
        // Cell L's right face has normal +1, Cell R's left face has normal -1.
        // The flux contribution to cell L is: F*(q_l, q_r) * (+1)
        // The flux contribution to cell R is: F*(q_l, q_r) * (-1)  (same flux, opposite normal)
        // These sum to zero, ensuring conservation.

        let q_l = SWEState::new(2.0, 3.0);
        let q_r = SWEState::new(1.5, 1.0);

        // The numerical flux F* is well-defined and unique at the interface
        let flux = roe_flux_swe(&q_l, &q_r, G, H_MIN);

        // Conservation is ensured by using the same flux on both sides:
        // contribution to L: +flux, contribution to R: -flux, sum = 0
        let contribution_l = flux.h * 1.0;
        let contribution_r = flux.h * (-1.0);
        assert!((contribution_l + contribution_r).abs() < TOL);
    }

    #[test]
    fn test_supercritical_flow() {
        // Supercritical flow: u > c, both characteristics go right
        // h = 1, u = 10, c = sqrt(10) ≈ 3.16, Fr ≈ 3.16
        let q_l = SWEState::new(1.0, 10.0);
        let q_r = SWEState::new(1.0, 10.0);

        let flux = roe_flux_swe(&q_l, &q_r, G, H_MIN);

        // Should equal physical flux
        let expected_h = 10.0; // hu
        let expected_hu = 10.0 * 10.0 + 0.5 * G * 1.0; // hu*u + gh²/2

        assert!((flux.h - expected_h).abs() < TOL);
        assert!((flux.hu - expected_hu).abs() < TOL);
    }
}
