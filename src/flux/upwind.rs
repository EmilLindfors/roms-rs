//! Upwind numerical flux for scalar advection.
//!
//! For the advection equation du/dt + a * du/dx = 0, the upwind flux is:
//! F^* = a * u^- if a > 0 (information flows right, use left state)
//! F^* = a * u^+ if a < 0 (information flows left, use right state)
//!
//! More generally, at an interface with outward normal n:
//! F^* · n = (a · n) u^- if a · n > 0  (outflow)
//! F^* · n = (a · n) u^+ if a · n ≤ 0  (inflow)

/// Compute upwind numerical flux for scalar advection.
///
/// # Arguments
/// * `u_minus` - Solution value from interior (current element)
/// * `u_plus` - Solution value from exterior (neighbor element or boundary)
/// * `a` - Advection velocity
/// * `normal` - Outward normal direction (-1 for left face, +1 for right face)
///
/// # Returns
/// The numerical flux F^* · n
pub fn upwind_flux(u_minus: f64, u_plus: f64, a: f64, normal: f64) -> f64 {
    let a_n = a * normal; // a dot n

    if a_n > 0.0 {
        // Outflow: use interior value
        a_n * u_minus
    } else {
        // Inflow: use exterior value
        a_n * u_plus
    }
}

/// Compute the flux jump for the DG formulation.
///
/// In the strong form DG, we need:
/// df = F^* · n - F(u^-) · n = F^* · n - a * u^- * n
///
/// This is the quantity that gets lifted to the volume.
#[allow(dead_code)]
pub fn flux_jump(u_minus: f64, u_plus: f64, a: f64, normal: f64) -> f64 {
    let f_star = upwind_flux(u_minus, u_plus, a, normal);
    let f_interior = a * u_minus * normal;
    f_star - f_interior
}

/// Compute Lax-Friedrichs numerical flux for scalar advection.
///
/// The Lax-Friedrichs flux is:
/// F^* = 0.5 * (F(u^-) + F(u^+)) · n - 0.5 * λ * (u^+ - u^-)
///
/// where λ = |a| is the maximum wave speed.
///
/// This is more dissipative than the upwind flux but can be more stable.
///
/// # Arguments
/// * `u_minus` - Solution value from interior (current element)
/// * `u_plus` - Solution value from exterior (neighbor element or boundary)
/// * `a` - Advection velocity
/// * `normal` - Outward normal direction (-1 for left face, +1 for right face)
///
/// # Returns
/// The numerical flux F^* · n
pub fn lax_friedrichs_flux(u_minus: f64, u_plus: f64, a: f64, normal: f64) -> f64 {
    let f_minus = a * u_minus * normal;
    let f_plus = a * u_plus * normal;
    let lambda = a.abs();

    0.5 * (f_minus + f_plus) - 0.5 * lambda * (u_plus - u_minus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upwind_positive_velocity() {
        let a = 1.0; // Positive velocity (flow to the right)

        // Left face (normal = -1)
        // Outward normal points left, so a · n = -1 < 0 → inflow
        // Should use u_plus (value coming from the left)
        let flux = upwind_flux(2.0, 1.0, a, -1.0);
        assert!((flux - (-1.0)).abs() < 1e-14); // a * u_plus * normal = 1 * 1 * (-1) = -1

        // Right face (normal = +1)
        // Outward normal points right, so a · n = 1 > 0 → outflow
        // Should use u_minus (interior value)
        let flux = upwind_flux(2.0, 3.0, a, 1.0);
        assert!((flux - 2.0).abs() < 1e-14); // a * u_minus * normal = 1 * 2 * 1 = 2
    }

    #[test]
    fn test_upwind_negative_velocity() {
        let a = -1.0; // Negative velocity (flow to the left)

        // Left face (normal = -1)
        // a · n = -1 * -1 = 1 > 0 → outflow
        // Should use u_minus (interior value)
        let flux = upwind_flux(2.0, 1.0, a, -1.0);
        assert!((flux - 2.0).abs() < 1e-14); // a * u_minus * normal = -1 * 2 * (-1) = 2

        // Right face (normal = +1)
        // a · n = -1 * 1 = -1 < 0 → inflow
        // Should use u_plus (value coming from the right)
        let flux = upwind_flux(2.0, 3.0, a, 1.0);
        assert!((flux - (-3.0)).abs() < 1e-14); // a * u_plus * normal = -1 * 3 * 1 = -3
    }

    #[test]
    fn test_flux_jump_interior() {
        // At interior interfaces with continuous solution
        let u = 2.0;
        let a = 1.0;

        // If u_minus = u_plus, flux jump should be zero
        let jump = flux_jump(u, u, a, 1.0);
        assert!(jump.abs() < 1e-14);
    }

    #[test]
    fn test_flux_jump_discontinuity() {
        let u_minus = 2.0;
        let u_plus = 1.0;
        let a = 1.0;
        let normal = 1.0;

        // F^* = a * u_minus = 2 (outflow, use interior)
        // F_interior = a * u_minus * normal = 2
        // Jump = 2 - 2 = 0 for outflow case

        let jump = flux_jump(u_minus, u_plus, a, normal);
        assert!(jump.abs() < 1e-14);

        // For inflow case (left face, a > 0)
        let normal = -1.0;
        // F^* = a * u_plus * normal = 1 * 1 * (-1) = -1
        // F_interior = a * u_minus * normal = 1 * 2 * (-1) = -2
        // Jump = -1 - (-2) = 1
        let jump = flux_jump(u_minus, u_plus, a, normal);
        assert!((jump - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_lax_friedrichs_continuous() {
        // For continuous solution, LF flux should equal physical flux
        let u = 2.0;
        let a = 1.0;
        let normal = 1.0;

        let flux = lax_friedrichs_flux(u, u, a, normal);
        let physical = a * u * normal;
        assert!((flux - physical).abs() < 1e-14);
    }

    #[test]
    fn test_lax_friedrichs_dissipation() {
        // LF flux should be more dissipative than upwind
        let u_minus = 2.0;
        let u_plus = 1.0;
        let a = 1.0;
        let normal = 1.0;

        let lf = lax_friedrichs_flux(u_minus, u_plus, a, normal);
        let upwind = upwind_flux(u_minus, u_plus, a, normal);

        // For outflow (a*n > 0), upwind uses u_minus
        // LF: 0.5*(a*u_minus + a*u_plus)*n - 0.5*|a|*(u_plus - u_minus)
        //   = 0.5*(2 + 1)*1 - 0.5*1*(1 - 2) = 1.5 + 0.5 = 2.0
        // Upwind: a*u_minus*n = 2.0
        // They're equal in this case because LF is symmetric + dissipation
        assert!((lf - 2.0).abs() < 1e-14);
        assert!((upwind - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_lax_friedrichs_symmetry() {
        // LF flux should be symmetric in velocity sign
        let u_minus = 2.0;
        let u_plus = 1.0;
        let normal = 1.0;

        let lf_pos = lax_friedrichs_flux(u_minus, u_plus, 1.0, normal);
        let lf_neg = lax_friedrichs_flux(u_minus, u_plus, -1.0, normal);

        // F(a) = 0.5*(a*u- + a*u+)*n - 0.5*|a|*(u+ - u-)
        // F(-a) = 0.5*(-a*u- + -a*u+)*n - 0.5*|-a|*(u+ - u-)
        //       = -0.5*(a*u- + a*u+)*n - 0.5*|a|*(u+ - u-)
        // F(a) + F(-a) = -|a|*(u+ - u-) = -1*(1-2) = 1
        assert!((lf_pos + lf_neg - 1.0).abs() < 1e-14);
    }
}
