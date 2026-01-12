//! 2D scalar advection equation.
//!
//! The 2D linear advection equation:
//!
//! ∂u/∂t + ∇ · (a u) = 0
//!
//! which expands to:
//!
//! ∂u/∂t + a_x ∂u/∂x + a_y ∂u/∂y = 0
//!
//! where a = (a_x, a_y) is the constant advection velocity vector.

/// 2D linear advection equation.
///
/// du/dt + a_x * du/dx + a_y * du/dy = 0
///
/// This is the simplest 2D hyperbolic equation, useful for testing
/// 2D DG implementations before moving to systems like shallow water.
#[derive(Clone, Debug)]
pub struct Advection2D {
    /// Advection velocity in x-direction
    pub velocity_x: f64,
    /// Advection velocity in y-direction
    pub velocity_y: f64,
}

impl Advection2D {
    /// Create a new 2D advection equation with given velocity components.
    pub fn new(velocity_x: f64, velocity_y: f64) -> Self {
        Self {
            velocity_x,
            velocity_y,
        }
    }

    /// Create advection with velocity specified as (speed, angle).
    ///
    /// angle is in radians, measured counter-clockwise from the positive x-axis.
    pub fn from_polar(speed: f64, angle: f64) -> Self {
        Self {
            velocity_x: speed * angle.cos(),
            velocity_y: speed * angle.sin(),
        }
    }

    /// Get the velocity vector.
    #[inline]
    pub fn velocity(&self) -> (f64, f64) {
        (self.velocity_x, self.velocity_y)
    }

    /// Compute the x-direction flux: F_x = a_x * u
    #[inline]
    pub fn flux_x(&self, u: f64) -> f64 {
        self.velocity_x * u
    }

    /// Compute the y-direction flux: F_y = a_y * u
    #[inline]
    pub fn flux_y(&self, u: f64) -> f64 {
        self.velocity_y * u
    }

    /// Compute both flux components: (F_x, F_y) = (a_x * u, a_y * u)
    #[inline]
    pub fn flux(&self, u: f64) -> (f64, f64) {
        (self.velocity_x * u, self.velocity_y * u)
    }

    /// Compute the normal flux: F · n = (a · n) * u
    ///
    /// This is the flux through a face with outward normal (nx, ny).
    #[inline]
    pub fn normal_flux(&self, u: f64, normal: (f64, f64)) -> f64 {
        let a_dot_n = self.velocity_x * normal.0 + self.velocity_y * normal.1;
        a_dot_n * u
    }

    /// Maximum wave speed (magnitude of velocity).
    ///
    /// Used for CFL computation.
    #[inline]
    pub fn max_wave_speed(&self) -> f64 {
        (self.velocity_x * self.velocity_x + self.velocity_y * self.velocity_y).sqrt()
    }

    /// Wave speed in a given normal direction: a · n
    ///
    /// This is the eigenvalue of the 1D problem projected onto the normal direction.
    #[inline]
    pub fn normal_wave_speed(&self, normal: (f64, f64)) -> f64 {
        self.velocity_x * normal.0 + self.velocity_y * normal.1
    }

    /// Upwind numerical flux for 2D advection.
    ///
    /// F^* = (a · n) * u^upwind
    ///
    /// where u^upwind is u_left if a · n > 0 (outflow), u_right otherwise (inflow).
    #[inline]
    pub fn upwind_flux(&self, u_left: f64, u_right: f64, normal: (f64, f64)) -> f64 {
        let a_dot_n = self.velocity_x * normal.0 + self.velocity_y * normal.1;
        if a_dot_n > 0.0 {
            a_dot_n * u_left // Use interior state (outflow)
        } else {
            a_dot_n * u_right // Use exterior state (inflow)
        }
    }

    /// Lax-Friedrichs numerical flux for 2D advection.
    ///
    /// F^* = 0.5 * (F_left + F_right) · n - 0.5 * |a · n| * (u_right - u_left)
    ///
    /// More dissipative than upwind but robust for discontinuities.
    #[inline]
    pub fn lax_friedrichs_flux(&self, u_left: f64, u_right: f64, normal: (f64, f64)) -> f64 {
        let a_dot_n = self.velocity_x * normal.0 + self.velocity_y * normal.1;
        let flux_left = a_dot_n * u_left;
        let flux_right = a_dot_n * u_right;
        0.5 * (flux_left + flux_right) - 0.5 * a_dot_n.abs() * (u_right - u_left)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_advection_creation() {
        let adv = Advection2D::new(1.0, 2.0);
        assert!((adv.velocity_x - 1.0).abs() < 1e-14);
        assert!((adv.velocity_y - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_advection_from_polar() {
        // 45 degree angle, speed 1
        let adv = Advection2D::from_polar(1.0, PI / 4.0);
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((adv.velocity_x - expected).abs() < 1e-14);
        assert!((adv.velocity_y - expected).abs() < 1e-14);

        // 0 degrees (purely x-direction)
        let adv_x = Advection2D::from_polar(2.0, 0.0);
        assert!((adv_x.velocity_x - 2.0).abs() < 1e-14);
        assert!(adv_x.velocity_y.abs() < 1e-14);

        // 90 degrees (purely y-direction)
        let adv_y = Advection2D::from_polar(3.0, PI / 2.0);
        assert!(adv_y.velocity_x.abs() < 1e-14);
        assert!((adv_y.velocity_y - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_flux_components() {
        let adv = Advection2D::new(2.0, 3.0);
        let u = 1.5;

        assert!((adv.flux_x(u) - 3.0).abs() < 1e-14);
        assert!((adv.flux_y(u) - 4.5).abs() < 1e-14);

        let (fx, fy) = adv.flux(u);
        assert!((fx - 3.0).abs() < 1e-14);
        assert!((fy - 4.5).abs() < 1e-14);
    }

    #[test]
    fn test_normal_flux() {
        let adv = Advection2D::new(2.0, 1.0);
        let u = 1.5;

        // Normal in x-direction: F · (1, 0) = 2 * 1.5 = 3
        assert!((adv.normal_flux(u, (1.0, 0.0)) - 3.0).abs() < 1e-14);

        // Normal in y-direction: F · (0, 1) = 1 * 1.5 = 1.5
        assert!((adv.normal_flux(u, (0.0, 1.0)) - 1.5).abs() < 1e-14);

        // Normal in negative x-direction: F · (-1, 0) = -2 * 1.5 = -3
        assert!((adv.normal_flux(u, (-1.0, 0.0)) - (-3.0)).abs() < 1e-14);
    }

    #[test]
    fn test_max_wave_speed() {
        let adv = Advection2D::new(3.0, 4.0);
        assert!((adv.max_wave_speed() - 5.0).abs() < 1e-14);

        let adv_unit = Advection2D::new(1.0, 0.0);
        assert!((adv_unit.max_wave_speed() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_normal_wave_speed() {
        let adv = Advection2D::new(2.0, 1.0);

        // a · n = 2 * 1 + 1 * 0 = 2
        assert!((adv.normal_wave_speed((1.0, 0.0)) - 2.0).abs() < 1e-14);

        // a · n = 2 * 0 + 1 * 1 = 1
        assert!((adv.normal_wave_speed((0.0, 1.0)) - 1.0).abs() < 1e-14);

        // Diagonal normal (normalized)
        let n_diag = 1.0 / 2.0_f64.sqrt();
        let expected = (2.0 + 1.0) * n_diag;
        assert!((adv.normal_wave_speed((n_diag, n_diag)) - expected).abs() < 1e-14);
    }

    #[test]
    fn test_upwind_flux_outflow() {
        // Velocity pointing right (+x)
        let adv = Advection2D::new(2.0, 0.0);

        // Right face (normal = (1, 0)): a · n > 0, use u_left
        let flux = adv.upwind_flux(1.0, 2.0, (1.0, 0.0));
        assert!((flux - 2.0).abs() < 1e-14); // 2.0 * 1.0

        // Left face (normal = (-1, 0)): a · n < 0, use u_right
        let flux_left = adv.upwind_flux(1.0, 2.0, (-1.0, 0.0));
        assert!((flux_left - (-4.0)).abs() < 1e-14); // -2.0 * 2.0
    }

    #[test]
    fn test_upwind_flux_continuous() {
        let adv = Advection2D::new(1.0, 1.0);
        let u = 3.0;

        // For continuous solution, upwind flux = physical flux
        let normal = (1.0, 0.0);
        let flux = adv.upwind_flux(u, u, normal);
        let physical = adv.normal_flux(u, normal);
        assert!((flux - physical).abs() < 1e-14);
    }

    #[test]
    fn test_lax_friedrichs_continuous() {
        let adv = Advection2D::new(1.5, 2.5);
        let u = 2.0;

        // For continuous solution, LF flux = physical flux
        let normal = (0.6, 0.8); // Normalized
        let flux = adv.lax_friedrichs_flux(u, u, normal);
        let physical = adv.normal_flux(u, normal);
        assert!((flux - physical).abs() < 1e-14);
    }

    #[test]
    fn test_lax_friedrichs_dissipation() {
        let adv = Advection2D::new(1.0, 0.0);
        let normal = (1.0, 0.0);

        // With discontinuity, LF adds dissipation
        let u_left = 1.0;
        let u_right = 2.0;

        let flux_upwind = adv.upwind_flux(u_left, u_right, normal);
        let flux_lf = adv.lax_friedrichs_flux(u_left, u_right, normal);

        // For outflow (a · n > 0), upwind uses u_left = 1.0
        // LF uses average and adds dissipation
        assert!((flux_upwind - 1.0).abs() < 1e-14);
        // LF: 0.5 * (1.0 + 2.0) - 0.5 * 1.0 * 1.0 = 1.0
        assert!((flux_lf - 1.0).abs() < 1e-14);

        // Test with negative normal
        let flux_lf_neg = adv.lax_friedrichs_flux(u_left, u_right, (-1.0, 0.0));
        // 0.5 * (-1.0 - 2.0) - 0.5 * 1.0 * 1.0 = -1.5 - 0.5 = -2.0
        assert!((flux_lf_neg - (-2.0)).abs() < 1e-14);
    }

    #[test]
    fn test_conservation_symmetry() {
        // Test that flux across shared face is conservative
        let adv = Advection2D::new(1.0, 2.0);
        let u_left = 1.5;
        let u_right = 2.5;

        // Left element sees normal (1, 0), right sees (-1, 0)
        let normal_left = (1.0, 0.0);
        let normal_right = (-1.0, 0.0);

        let flux_from_left = adv.lax_friedrichs_flux(u_left, u_right, normal_left);
        let flux_from_right = adv.lax_friedrichs_flux(u_right, u_left, normal_right);

        // Fluxes should sum to zero (conservative)
        assert!((flux_from_left + flux_from_right).abs() < 1e-14);
    }
}
