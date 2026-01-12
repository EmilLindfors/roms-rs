//! 1D Shallow Water Equations.
//!
//! The 1D shallow water equations (Saint-Venant equations):
//!
//! ∂h/∂t + ∂(hu)/∂x = 0                           (mass conservation)
//! ∂(hu)/∂t + ∂(hu² + gh²/2)/∂x = -gh ∂B/∂x      (momentum conservation)
//!
//! where:
//! - h = water depth
//! - u = velocity
//! - hu = momentum
//! - g = gravitational acceleration
//! - B = bottom topography (bathymetry)
//!
//! The source term (-gh ∂B/∂x) is handled separately in the source module.

use super::ConservationLaw;

/// 1D Shallow Water Equations.
///
/// State vector: q = [h, hu]
/// Flux: f(q) = [hu, hu²/h + gh²/2]
///
/// This implementation does not include source terms (bathymetry, friction).
/// Those are handled separately in the source module to allow well-balanced
/// discretizations.
#[derive(Clone, Debug)]
pub struct ShallowWater1D {
    /// Gravitational acceleration (default 9.81 m/s²)
    pub g: f64,
    /// Minimum depth for wet/dry treatment (default 1e-6)
    pub h_min: f64,
}

impl ShallowWater1D {
    /// Create shallow water equations with standard gravity.
    pub fn new(g: f64) -> Self {
        Self { g, h_min: 1e-6 }
    }

    /// Create with custom minimum depth threshold.
    pub fn with_h_min(g: f64, h_min: f64) -> Self {
        Self { g, h_min }
    }

    /// Standard gravity (9.81 m/s²).
    pub fn standard() -> Self {
        Self::new(9.81)
    }

    /// Compute velocity from state with desingularization.
    ///
    /// u = 2 * h * hu / (h² + max(h, h_min)²)
    pub fn velocity(&self, h: f64, hu: f64) -> f64 {
        let h_reg = h.max(self.h_min);
        2.0 * h * hu / (h * h + h_reg * h_reg)
    }

    /// Compute velocity without desingularization.
    pub fn velocity_simple(&self, h: f64, hu: f64) -> f64 {
        if h > self.h_min { hu / h } else { 0.0 }
    }

    /// Compute wave celerity c = sqrt(gh).
    pub fn celerity(&self, h: f64) -> f64 {
        (self.g * h.max(0.0)).sqrt()
    }

    /// Froude number Fr = |u| / c.
    pub fn froude(&self, h: f64, hu: f64) -> f64 {
        let u = self.velocity_simple(h, hu);
        let c = self.celerity(h);
        if c > 1e-10 { u.abs() / c } else { 0.0 }
    }

    /// Check if flow is subcritical (Fr < 1).
    pub fn is_subcritical(&self, h: f64, hu: f64) -> bool {
        self.froude(h, hu) < 1.0
    }

    /// Check if flow is supercritical (Fr > 1).
    pub fn is_supercritical(&self, h: f64, hu: f64) -> bool {
        self.froude(h, hu) > 1.0
    }
}

impl Default for ShallowWater1D {
    fn default() -> Self {
        Self::standard()
    }
}

impl ConservationLaw for ShallowWater1D {
    const N_VARS: usize = 2;

    fn flux(&self, q: &[f64]) -> Vec<f64> {
        debug_assert_eq!(q.len(), 2);

        let h = q[0];
        let hu = q[1];

        // Handle dry cells
        if h <= self.h_min {
            return vec![0.0, 0.0];
        }

        let u = hu / h;

        // f = [hu, hu² / h + g h² / 2]
        //   = [hu, h u² + g h² / 2]
        vec![hu, h * u * u + 0.5 * self.g * h * h]
    }

    fn max_wave_speed(&self, q: &[f64]) -> f64 {
        let h = q[0];
        let hu = q[1];

        if h <= self.h_min {
            return 0.0;
        }

        let u = self.velocity_simple(h, hu);
        let c = self.celerity(h);

        // λ_max = |u| + c
        u.abs() + c
    }

    fn eigenvalues(&self, q: &[f64]) -> Vec<f64> {
        let h = q[0];
        let hu = q[1];

        if h <= self.h_min {
            return vec![0.0, 0.0];
        }

        let u = hu / h;
        let c = self.celerity(h);

        // λ₁ = u - c, λ₂ = u + c
        vec![u - c, u + c]
    }

    fn roe_average(&self, q_l: &[f64], q_r: &[f64]) -> Vec<f64> {
        let h_l = q_l[0];
        let h_r = q_r[0];
        let hu_l = q_l[1];
        let hu_r = q_r[1];

        // Roe average for shallow water:
        // h_roe = (h_l + h_r) / 2
        // u_roe = (√h_l u_l + √h_r u_r) / (√h_l + √h_r)

        let sqrt_h_l = h_l.max(0.0).sqrt();
        let sqrt_h_r = h_r.max(0.0).sqrt();

        let h_roe = 0.5 * (h_l + h_r);

        let u_roe = if sqrt_h_l + sqrt_h_r > 1e-10 {
            let u_l = if h_l > self.h_min { hu_l / h_l } else { 0.0 };
            let u_r = if h_r > self.h_min { hu_r / h_r } else { 0.0 };
            (sqrt_h_l * u_l + sqrt_h_r * u_r) / (sqrt_h_l + sqrt_h_r)
        } else {
            0.0
        };

        vec![h_roe, h_roe * u_roe]
    }

    fn right_eigenvectors(&self, q: &[f64]) -> Vec<Vec<f64>> {
        let h = q[0];
        let hu = q[1];

        if h <= self.h_min {
            // Return identity for dry cells
            return vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        }

        let u = hu / h;
        let c = self.celerity(h);

        // Right eigenvectors:
        // r₁ = [1, u - c]ᵀ  (for λ₁ = u - c)
        // r₂ = [1, u + c]ᵀ  (for λ₂ = u + c)
        vec![vec![1.0, u - c], vec![1.0, u + c]]
    }

    fn left_eigenvectors(&self, q: &[f64]) -> Vec<Vec<f64>> {
        let h = q[0];
        let hu = q[1];

        if h <= self.h_min {
            // Return identity for dry cells
            return vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        }

        let u = hu / h;
        let c = self.celerity(h);

        // Left eigenvectors (rows of R⁻¹):
        // L = 1/(2c) * [ u + c, -1 ]
        //              [-u + c,  1 ]
        let inv_2c = 0.5 / c;
        vec![
            vec![inv_2c * (u + c), -inv_2c],
            vec![inv_2c * (-u + c), inv_2c],
        ]
    }
}

/// Compute the exact dam break solution (Stoker solution).
///
/// Returns (h, u) at position x and time t for a dam break at x = x_dam
/// with initial left depth h_l and right depth h_r (h_r can be 0 for dry bed).
///
/// This is useful for testing Riemann solvers.
#[allow(dead_code)]
pub fn dam_break_exact(x: f64, t: f64, x_dam: f64, h_l: f64, h_r: f64, g: f64) -> (f64, f64) {
    if t <= 0.0 {
        // Initial condition
        if x < x_dam {
            return (h_l, 0.0);
        } else {
            return (h_r, 0.0);
        }
    }

    let c_l = (g * h_l).sqrt();

    // Dry bed case (h_r = 0)
    if h_r < 1e-10 {
        let x_a = x_dam - c_l * t; // Left rarefaction head
        let x_b = x_dam + 2.0 * c_l * t; // Right rarefaction tail

        if x <= x_a {
            // Undisturbed left state
            (h_l, 0.0)
        } else if x >= x_b {
            // Dry bed
            (0.0, 0.0)
        } else {
            // Inside rarefaction fan
            let h = (2.0 * c_l - (x - x_dam) / t).powi(2) / (9.0 * g);
            let u = 2.0 / 3.0 * ((x - x_dam) / t + c_l);
            (h.max(0.0), u)
        }
    } else {
        // Wet bed case - requires iteration to find intermediate state
        // For simplicity, use the dry bed solution as approximation
        // (a proper implementation would solve the Rankine-Hugoniot conditions)
        let _c_r = (g * h_r).sqrt();
        let x_a = x_dam - c_l * t;

        if x <= x_a {
            (h_l, 0.0)
        } else {
            // Simplified: assume rarefaction on left, shock on right
            let h_m = 0.5 * (h_l + h_r); // Rough approximation
            let u_m = c_l - (g * h_m).sqrt();
            let x_m = x_dam + u_m * t;

            if x >= x_m { (h_r, 0.0) } else { (h_m, u_m) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn test_shallow_water_creation() {
        let swe = ShallowWater1D::standard();
        assert!((swe.g - 9.81).abs() < TOL);
        assert!((swe.h_min - 1e-6).abs() < TOL);
    }

    #[test]
    fn test_flux_still_water() {
        let swe = ShallowWater1D::new(10.0);

        // Still water: h = 2, u = 0, hu = 0
        let flux = swe.flux(&[2.0, 0.0]);

        // f = [hu, hu²/h + gh²/2] = [0, 0 + 10 * 4 / 2] = [0, 20]
        assert!(flux[0].abs() < TOL);
        assert!((flux[1] - 20.0).abs() < TOL);
    }

    #[test]
    fn test_flux_moving_water() {
        let swe = ShallowWater1D::new(10.0);

        // h = 2, u = 3, hu = 6
        let flux = swe.flux(&[2.0, 6.0]);

        // f = [hu, hu²/h + gh²/2]
        //   = [6, 36/2 + 10*4/2]
        //   = [6, 18 + 20]
        //   = [6, 38]
        assert!((flux[0] - 6.0).abs() < TOL);
        assert!((flux[1] - 38.0).abs() < TOL);
    }

    #[test]
    fn test_flux_dry_cell() {
        let swe = ShallowWater1D::new(10.0);

        // Dry cell
        let flux = swe.flux(&[1e-10, 1e-10]);

        assert!(flux[0].abs() < TOL);
        assert!(flux[1].abs() < TOL);
    }

    #[test]
    fn test_eigenvalues() {
        let swe = ShallowWater1D::new(10.0);

        // h = 1, u = 2, hu = 2
        // c = sqrt(10 * 1) = sqrt(10) ≈ 3.162
        let eigs = swe.eigenvalues(&[1.0, 2.0]);

        let c = (10.0_f64).sqrt();
        assert!((eigs[0] - (2.0 - c)).abs() < TOL);
        assert!((eigs[1] - (2.0 + c)).abs() < TOL);
    }

    #[test]
    fn test_max_wave_speed() {
        let swe = ShallowWater1D::new(10.0);

        // h = 1, u = 2
        // |u| + c = 2 + sqrt(10) ≈ 5.162
        let speed = swe.max_wave_speed(&[1.0, 2.0]);

        let expected = 2.0 + (10.0_f64).sqrt();
        assert!((speed - expected).abs() < TOL);
    }

    #[test]
    fn test_roe_average_symmetric() {
        let swe = ShallowWater1D::new(10.0);

        // Same state on both sides
        let q = [2.0, 3.0];
        let q_roe = swe.roe_average(&q, &q);

        // Should return the same state (approximately)
        assert!((q_roe[0] - q[0]).abs() < TOL);
        assert!((q_roe[1] - q[1]).abs() < TOL);
    }

    #[test]
    fn test_roe_average_different() {
        let swe = ShallowWater1D::new(10.0);

        // Different states
        let q_l = [1.0, 1.0]; // h=1, u=1
        let q_r = [4.0, 8.0]; // h=4, u=2

        let q_roe = swe.roe_average(&q_l, &q_r);

        // h_roe = (1 + 4) / 2 = 2.5
        assert!((q_roe[0] - 2.5).abs() < TOL);

        // u_roe = (1*1 + 2*2) / (1 + 2) = 5/3 ≈ 1.667
        let u_roe = (1.0 * 1.0 + 2.0 * 2.0) / (1.0 + 2.0);
        let hu_roe = 2.5 * u_roe;
        assert!((q_roe[1] - hu_roe).abs() < TOL);
    }

    #[test]
    fn test_eigenvector_orthogonality() {
        let swe = ShallowWater1D::new(10.0);

        let q = [2.0, 3.0];
        let r = swe.right_eigenvectors(&q);
        let l = swe.left_eigenvectors(&q);

        // L * R should be approximately identity
        for i in 0..2 {
            for j in 0..2 {
                let mut dot = 0.0;
                for k in 0..2 {
                    dot += l[i][k] * r[j][k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "L*R[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_froude_number() {
        let swe = ShallowWater1D::new(10.0);

        // Subcritical: Fr < 1
        // h = 10, u = 1, c = sqrt(100) = 10, Fr = 0.1
        assert!(swe.is_subcritical(10.0, 10.0));

        // Supercritical: Fr > 1
        // h = 0.1, u = 10, c = sqrt(1) = 1, Fr = 10
        assert!(swe.is_supercritical(0.1, 1.0));
    }

    #[test]
    fn test_celerity() {
        let swe = ShallowWater1D::new(10.0);

        let c = swe.celerity(2.5);
        let expected = (10.0 * 2.5_f64).sqrt();
        assert!((c - expected).abs() < TOL);

        // Negative depth should be treated as zero
        let c_neg = swe.celerity(-1.0);
        assert!(c_neg.abs() < TOL);
    }

    #[test]
    fn test_velocity_desingularization() {
        let swe = ShallowWater1D::with_h_min(10.0, 1e-3);

        // Normal wet cell
        let u = swe.velocity(2.0, 4.0);
        assert!((u - 2.0).abs() < 1e-10);

        // Near-dry cell - should not blow up
        let u_dry = swe.velocity(1e-6, 1e-6);
        assert!(u_dry.is_finite());
        assert!(u_dry.abs() < 10.0); // Should be bounded
    }
}
