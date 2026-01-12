//! Scalar advection equation.
//!
//! The 1D linear advection equation:
//!
//! ∂u/∂t + a ∂u/∂x = 0
//!
//! where a is the constant advection velocity.

use super::ConservationLaw;

/// 1D linear advection equation.
///
/// du/dt + a * du/dx = 0
///
/// This is the simplest hyperbolic conservation law, useful for testing
/// and as a building block for more complex systems.
#[derive(Clone, Debug)]
pub struct Advection1D {
    /// Advection velocity (positive = rightward)
    pub velocity: f64,
}

impl Advection1D {
    /// Create a new advection equation with given velocity.
    pub fn new(velocity: f64) -> Self {
        Self { velocity }
    }
}

impl ConservationLaw for Advection1D {
    const N_VARS: usize = 1;

    fn flux(&self, q: &[f64]) -> Vec<f64> {
        debug_assert_eq!(q.len(), 1);
        vec![self.velocity * q[0]]
    }

    fn max_wave_speed(&self, _q: &[f64]) -> f64 {
        self.velocity.abs()
    }

    fn eigenvalues(&self, _q: &[f64]) -> Vec<f64> {
        vec![self.velocity]
    }

    fn roe_average(&self, _q_l: &[f64], _q_r: &[f64]) -> Vec<f64> {
        // For linear advection, the Roe average is trivial
        // (any state works since the Jacobian is constant)
        vec![0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advection_flux() {
        let adv = Advection1D::new(2.0);

        // f(u) = a * u = 2 * 3 = 6
        let flux = adv.flux(&[3.0]);
        assert!((flux[0] - 6.0).abs() < 1e-14);

        // Negative velocity
        let adv_neg = Advection1D::new(-1.5);
        let flux_neg = adv_neg.flux(&[2.0]);
        assert!((flux_neg[0] - (-3.0)).abs() < 1e-14);
    }

    #[test]
    fn test_advection_wave_speed() {
        let adv_pos = Advection1D::new(2.0);
        assert!((adv_pos.max_wave_speed(&[1.0]) - 2.0).abs() < 1e-14);

        let adv_neg = Advection1D::new(-3.0);
        assert!((adv_neg.max_wave_speed(&[1.0]) - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_advection_eigenvalues() {
        let adv = Advection1D::new(2.5);
        let eigs = adv.eigenvalues(&[1.0]);
        assert_eq!(eigs.len(), 1);
        assert!((eigs[0] - 2.5).abs() < 1e-14);
    }

    #[test]
    fn test_advection_eigenvectors() {
        let adv = Advection1D::new(1.0);

        // For scalar equation, eigenvectors are trivially [1]
        let r = adv.right_eigenvectors(&[1.0]);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].len(), 1);
        assert!((r[0][0] - 1.0).abs() < 1e-14);

        let l = adv.left_eigenvectors(&[1.0]);
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].len(), 1);
        assert!((l[0][0] - 1.0).abs() < 1e-14);
    }
}
