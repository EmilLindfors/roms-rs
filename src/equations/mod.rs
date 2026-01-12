//! Conservation law abstractions.
//!
//! Provides a trait-based interface for hyperbolic conservation laws:
//!
//! ∂q/∂t + ∂f(q)/∂x = s(q, x)
//!
//! where q is the state vector, f is the flux function, and s is a source term.

mod advection;
mod advection_2d;
mod equation_of_state;
mod shallow_water;
mod shallow_water_2d;

pub use advection::Advection1D;
pub use advection_2d::Advection2D;
pub use equation_of_state::{EquationOfState, LinearEquationOfState, RHO_0};
pub use shallow_water::ShallowWater1D;
pub use shallow_water_2d::ShallowWater2D;

/// A hyperbolic conservation law in 1D.
///
/// This trait abstracts over scalar equations (like advection) and systems
/// (like shallow water equations). It provides the essential mathematical
/// operations needed for DG discretization.
///
/// # Type Parameters
///
/// Implementations define their own state representation. For scalar equations,
/// this is typically `f64`. For systems, it could be a fixed-size array or
/// a custom struct.
///
/// # Example
///
/// ```ignore
/// // Scalar advection
/// let advection = Advection1D::new(1.0);
/// let flux = advection.flux(&[1.5]);  // Returns [1.5] for a=1
///
/// // Shallow water
/// let swe = ShallowWater1D::new(9.81);
/// let flux = swe.flux(&[2.0, 3.0]);  // [h, hu] -> [hu, hu²/h + gh²/2]
/// ```
pub trait ConservationLaw: Clone + Send + Sync {
    /// Number of conserved variables.
    ///
    /// - 1 for scalar equations (advection)
    /// - 2 for 1D shallow water (h, hu)
    /// - 3 for 2D shallow water (h, hu, hv)
    const N_VARS: usize;

    /// Compute the physical flux f(q).
    ///
    /// For advection: f(u) = a * u
    /// For shallow water: f(h, hu) = (hu, hu²/h + gh²/2)
    ///
    /// # Arguments
    /// * `q` - State vector of length N_VARS
    ///
    /// # Returns
    /// Flux vector of length N_VARS
    fn flux(&self, q: &[f64]) -> Vec<f64>;

    /// Maximum absolute wave speed |λ_max| for CFL computation.
    ///
    /// For advection: |a|
    /// For shallow water: |u| + sqrt(gh)
    ///
    /// This is used to compute the stable time step.
    fn max_wave_speed(&self, q: &[f64]) -> f64;

    /// Eigenvalues of the flux Jacobian ∂f/∂q.
    ///
    /// For advection: [a]
    /// For shallow water: [u - c, u + c] where c = sqrt(gh)
    ///
    /// Used in characteristic-based Riemann solvers (Roe, HLLC).
    fn eigenvalues(&self, q: &[f64]) -> Vec<f64>;

    /// Roe-averaged state for linearized Riemann solver.
    ///
    /// Computes a state q_roe such that:
    /// f(q_r) - f(q_l) = A(q_roe) * (q_r - q_l)
    ///
    /// where A is the flux Jacobian.
    fn roe_average(&self, q_l: &[f64], q_r: &[f64]) -> Vec<f64>;

    /// Right eigenvectors of the flux Jacobian.
    ///
    /// Returns a matrix R where R[:, i] is the i-th right eigenvector.
    /// Used for characteristic decomposition in Roe solver.
    ///
    /// Default implementation returns identity (works for scalar).
    #[allow(clippy::needless_range_loop)]
    fn right_eigenvectors(&self, q: &[f64]) -> Vec<Vec<f64>> {
        let n = Self::N_VARS;
        let mut r = vec![vec![0.0; n]; n];
        for i in 0..n {
            r[i][i] = 1.0;
        }
        // Suppress unused variable warning
        let _ = q;
        r
    }

    /// Left eigenvectors of the flux Jacobian.
    ///
    /// Returns a matrix L where L[i, :] is the i-th left eigenvector.
    /// L = R^{-1}, so L * R = I.
    ///
    /// Default implementation returns identity (works for scalar).
    #[allow(clippy::needless_range_loop)]
    fn left_eigenvectors(&self, q: &[f64]) -> Vec<Vec<f64>> {
        let n = Self::N_VARS;
        let mut l = vec![vec![0.0; n]; n];
        for i in 0..n {
            l[i][i] = 1.0;
        }
        // Suppress unused variable warning
        let _ = q;
        l
    }
}

/// Numerical flux function signature for systems.
///
/// A numerical flux takes left and right states and an outward normal,
/// and returns the numerical flux F^* · n.
pub type NumericalFluxFn<L> = fn(law: &L, q_l: &[f64], q_r: &[f64], normal: f64) -> Vec<f64>;

/// Lax-Friedrichs (Rusanov) flux for any conservation law.
///
/// F^* = 0.5 * (f(q_l) + f(q_r)) · n - 0.5 * λ_max * (q_r - q_l)
///
/// where λ_max = max(|λ(q_l)|, |λ(q_r)|).
///
/// This is a simple, robust flux that works for any conservation law.
pub fn lax_friedrichs_system<L: ConservationLaw>(
    law: &L,
    q_l: &[f64],
    q_r: &[f64],
    normal: f64,
) -> Vec<f64> {
    let n = L::N_VARS;

    let f_l = law.flux(q_l);
    let f_r = law.flux(q_r);

    let lambda_l = law.max_wave_speed(q_l);
    let lambda_r = law.max_wave_speed(q_r);
    let lambda_max = lambda_l.max(lambda_r);

    let mut flux = vec![0.0; n];
    for i in 0..n {
        // Central flux
        let f_central = 0.5 * (f_l[i] + f_r[i]) * normal;
        // Dissipation term
        let dissipation = 0.5 * lambda_max * (q_r[i] - q_l[i]);
        flux[i] = f_central - dissipation;
    }

    flux
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advection_implements_conservation_law() {
        let adv = Advection1D::new(1.0);
        assert_eq!(Advection1D::N_VARS, 1);

        let flux = adv.flux(&[2.0]);
        assert!((flux[0] - 2.0).abs() < 1e-14);

        let speed = adv.max_wave_speed(&[2.0]);
        assert!((speed - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_lax_friedrichs_system_continuous() {
        let adv = Advection1D::new(1.0);

        // For continuous solution, LF flux should equal physical flux
        let q = [2.0];
        let flux = lax_friedrichs_system(&adv, &q, &q, 1.0);
        let physical = adv.flux(&q);

        assert!((flux[0] - physical[0]).abs() < 1e-14);
    }
}
