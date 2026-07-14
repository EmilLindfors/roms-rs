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

/// Pressure function for the SWE exact Riemann solver (Toro 2009, Ch. 10).
///
/// Returns (f, f') where f is the wave function and f' its derivative,
/// evaluated at depth `h` for the wave connecting to state with depth `h_k`.
#[allow(dead_code)]
fn pressure_function_swe(h: f64, h_k: f64, g: f64) -> (f64, f64) {
    if h <= h_k {
        // Rarefaction wave
        let c = (g * h).sqrt();
        let c_k = (g * h_k).sqrt();
        let f = 2.0 * (c - c_k);
        let df = (g / h).sqrt();
        (f, df)
    } else {
        // Shock wave
        let a = g * (h + h_k) / (2.0 * h * h_k);
        let a_sqrt = a.sqrt();
        let f = (h - h_k) * a_sqrt;
        let df = a_sqrt - g * (h - h_k) / (4.0 * h * h * a_sqrt);
        (f, df)
    }
}

/// Solve the SWE Riemann problem for the star-region state (Toro 2009, Ch. 10).
///
/// Uses Newton-Raphson iteration with a two-rarefaction initial guess.
/// Returns (h_m, u_m) — the depth and velocity in the star region.
#[allow(dead_code)]
fn solve_riemann_swe(h_l: f64, h_r: f64, u_l: f64, u_r: f64, g: f64) -> (f64, f64) {
    let c_l = (g * h_l).sqrt();
    let c_r = (g * h_r).sqrt();

    // Two-rarefaction initial guess
    let c_guess = (u_l - u_r) / 4.0 + (c_l + c_r) / 2.0;
    let mut h_m = (c_guess * c_guess / g).max(1e-14);

    // Newton-Raphson iteration
    for _ in 0..50 {
        let (f_l, df_l) = pressure_function_swe(h_m, h_l, g);
        let (f_r, df_r) = pressure_function_swe(h_m, h_r, g);
        let residual = f_l + f_r + (u_r - u_l);
        let derivative = df_l + df_r;

        if derivative.abs() < 1e-30 {
            break;
        }

        let dh = residual / derivative;
        h_m -= dh;
        h_m = h_m.max(1e-14);

        if dh.abs() < 1e-12 * h_m {
            break;
        }
    }

    // Star-region velocity
    let (f_l, _) = pressure_function_swe(h_m, h_l, g);
    let (f_r, _) = pressure_function_swe(h_m, h_r, g);
    let u_m = 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l);

    (h_m, u_m)
}

/// Sample the exact SWE Riemann solution at speed s = (x - x_dam) / t.
///
/// Given the left/right states and the star-region solution (h_m, u_m),
/// returns (h, u) at the given sampling speed s.
#[allow(dead_code, clippy::too_many_arguments)]
fn sample_riemann_swe(
    s: f64,
    h_l: f64,
    h_r: f64,
    u_l: f64,
    u_r: f64,
    h_m: f64,
    u_m: f64,
    g: f64,
) -> (f64, f64) {
    let c_m = (g * h_m).sqrt();

    if s <= u_m {
        // Left of contact discontinuity
        if h_m <= h_l {
            // Left rarefaction
            let c_l = (g * h_l).sqrt();
            let s_hl = u_l - c_l; // Head speed
            let s_tl = u_m - c_m; // Tail speed

            if s <= s_hl {
                (h_l, u_l)
            } else if s <= s_tl {
                // Inside rarefaction fan
                let c = (u_l + 2.0 * c_l - s) / 3.0;
                let u = (2.0 * s + u_l + 2.0 * c_l) / 3.0;
                let h = c * c / g;
                (h.max(0.0), u)
            } else {
                (h_m, u_m)
            }
        } else {
            // Left shock
            let c_l = (g * h_l).sqrt();
            let q_l = (h_m * (h_m + h_l) / (2.0 * h_l * h_l)).sqrt();
            let s_l = u_l - c_l * q_l;

            if s <= s_l { (h_l, u_l) } else { (h_m, u_m) }
        }
    } else {
        // Right of contact discontinuity
        if h_m <= h_r {
            // Right rarefaction
            let c_r = (g * h_r).sqrt();
            let s_hr = u_r + c_r; // Head speed
            let s_tr = u_m + c_m; // Tail speed

            if s >= s_hr {
                (h_r, u_r)
            } else if s >= s_tr {
                // Inside rarefaction fan
                let c = (-u_r + 2.0 * c_r + s) / 3.0;
                let u = (2.0 * s + u_r - 2.0 * c_r) / 3.0;
                let h = c * c / g;
                (h.max(0.0), u)
            } else {
                (h_m, u_m)
            }
        } else {
            // Right shock
            let c_r = (g * h_r).sqrt();
            let q_r = (h_m * (h_m + h_r) / (2.0 * h_r * h_r)).sqrt();
            let s_r = u_r + c_r * q_r;

            if s >= s_r { (h_r, u_r) } else { (h_m, u_m) }
        }
    }
}

/// Compute the exact dam break solution (Stoker solution for dry bed,
/// exact Riemann solver for wet bed following Toro 2009, Ch. 10).
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
        // Wet bed case — exact Riemann solver
        let (h_m, u_m) = solve_riemann_swe(h_l, h_r, 0.0, 0.0, g);
        let s = (x - x_dam) / t;
        sample_riemann_swe(s, h_l, h_r, 0.0, 0.0, h_m, u_m, g)
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

    // ====================================================================
    // Dam-break wet-bed exact Riemann solver tests
    // ====================================================================

    #[test]
    fn test_dam_break_wet_symmetric() {
        // Symmetric: h_l = h_r → h_m = h_l, u_m = 0
        let g = 9.81;
        let h = 2.0;
        let t = 1.0;
        let x_dam = 5.0;

        // At x = x_dam (center), the solution should be the undisturbed state
        let (h_sol, u_sol) = dam_break_exact(x_dam, t, x_dam, h, h, g);
        assert!(
            (h_sol - h).abs() < 1e-10,
            "Symmetric dam break: h should be {}, got {}",
            h,
            h_sol
        );
        assert!(
            u_sol.abs() < 1e-10,
            "Symmetric dam break: u should be 0, got {}",
            u_sol
        );

        // Far left and far right should be undisturbed
        let (h_left, u_left) = dam_break_exact(0.0, t, x_dam, h, h, g);
        assert!((h_left - h).abs() < 1e-10);
        assert!(u_left.abs() < 1e-10);

        let (h_right, u_right) = dam_break_exact(10.0, t, x_dam, h, h, g);
        assert!((h_right - h).abs() < 1e-10);
        assert!(u_right.abs() < 1e-10);
    }

    #[test]
    fn test_dam_break_wet_classic() {
        // Classic dam break: h_l=2, h_r=1
        // Verifies the Newton solver converges and gives physically correct results
        let g = 9.81;
        let h_l = 2.0;
        let h_r = 1.0;
        let x_dam = 5.0;
        let t = 0.5;

        // Solve for the star-region state directly
        let (h_m, u_m) = solve_riemann_swe(h_l, h_r, 0.0, 0.0, g);

        // h_m must be between h_r and h_l
        assert!(
            h_m > h_r && h_m < h_l,
            "h_m={} should be between h_r={} and h_l={}",
            h_m,
            h_r,
            h_l
        );

        // u_m must be positive (flow from deep to shallow side)
        assert!(u_m > 0.0, "u_m={} should be positive", u_m);

        // Verify the residual is near zero: f_L + f_R + (u_R - u_L) = 0
        let (f_l, _) = pressure_function_swe(h_m, h_l, g);
        let (f_r, _) = pressure_function_swe(h_m, h_r, g);
        let residual = f_l + f_r; // u_l = u_r = 0
        assert!(
            residual.abs() < 1e-10,
            "Riemann solver residual should be near zero, got {}",
            residual
        );

        // Verify continuity: solution should be continuous within each region
        // Left undisturbed state
        let (h_far_left, _) = dam_break_exact(0.0, t, x_dam, h_l, h_r, g);
        assert!((h_far_left - h_l).abs() < 1e-10);

        // Right undisturbed state
        let (h_far_right, _) = dam_break_exact(10.0, t, x_dam, h_l, h_r, g);
        assert!((h_far_right - h_r).abs() < 1e-10);
    }

    #[test]
    fn test_dam_break_wet_continuity_at_boundaries() {
        // Verify smooth transitions at wave fronts for rarefaction
        let g = 9.81;
        let h_l = 2.0;
        let h_r = 1.0;
        let x_dam = 5.0;
        let t = 0.5;

        // Sample densely to check for large jumps within the rarefaction
        let n = 1000;
        let mut prev_h = h_l;
        for i in 0..=n {
            let x = i as f64 / n as f64 * 10.0;
            let (h, _) = dam_break_exact(x, t, x_dam, h_l, h_r, g);

            // No jump in h should exceed what's physically reasonable
            // (only the shock and contact can have jumps)
            assert!(
                h.is_finite() && h >= 0.0,
                "h should be finite and non-negative at x={}, got {}",
                x,
                h
            );

            // h should be monotonically non-increasing from left to right
            // (for h_l > h_r with zero initial velocity)
            assert!(
                h <= prev_h + 1e-10,
                "h should be monotonically non-increasing: h({})={} > h_prev={}",
                x,
                h,
                prev_h
            );
            prev_h = h;
        }
    }

    #[test]
    fn test_dam_break_wet_reduces_to_dry() {
        // As h_r → 0+, the wet-bed solution should approach the dry-bed solution
        let g = 9.81;
        let h_l = 2.0;
        let x_dam = 5.0;
        let t = 0.5;

        let h_r_small = 1e-6;

        // Sample at several points and compare with dry-bed solution
        for &x in &[0.0, 3.0, 5.0, 6.0, 7.0] {
            let (h_wet, u_wet) = dam_break_exact(x, t, x_dam, h_l, h_r_small, g);
            let (h_dry, u_dry) = dam_break_exact(x, t, x_dam, h_l, 0.0, g);

            // Allow larger tolerance since h_r is small but not zero
            assert!(
                (h_wet - h_dry).abs() < 0.1,
                "At x={}: wet h={} should be close to dry h={}",
                x,
                h_wet,
                h_dry
            );
            if h_dry > 0.01 {
                assert!(
                    (u_wet - u_dry).abs() < 0.5,
                    "At x={}: wet u={} should be close to dry u={}",
                    x,
                    u_wet,
                    u_dry
                );
            }
        }
    }
}
