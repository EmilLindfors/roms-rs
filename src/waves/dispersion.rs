//! Linear surface-gravity-wave kinematics: the dispersion relation and the
//! speeds a spectral wave model propagates with.
//!
//! For intrinsic radian frequency σ in water of depth d, the wavenumber k solves
//!
//! ```text
//! σ² = g k tanh(k d)
//! ```
//!
//! [`wavenumber`] starts from Guo's (2002) explicit approximation (error below
//! 0.75 % for every kd) and polishes it with Newton's method to machine precision.
//! The group velocity is `c_g = n σ/k` with `n = ½ (1 + 2kd / sinh 2kd)` (½ in deep
//! water, 1 in shallow), and the depth sensitivity of the frequency at fixed k,
//! which drives refraction and frequency shifting, is `∂σ/∂d = k σ / sinh(2kd)`
//! (Mei, "The Applied Dynamics of Ocean Surface Waves", 1989, §3.1; the SWAN
//! Scientific Documentation, §2.2).

/// Beyond this kd the water is deep to machine precision (tanh(kd) = 1 to 1e-26).
const DEEP_KD: f64 = 30.0;

/// The wavenumber k (rad/m) of radian frequency `sigma` (rad/s) in water of depth
/// `depth` (m, > 0) under gravity `g`.
pub fn wavenumber(sigma: f64, depth: f64, g: f64) -> f64 {
    debug_assert!(sigma > 0.0 && depth > 0.0, "σ = {sigma}, d = {depth}");
    let k_deep = sigma * sigma / g;
    if k_deep * depth > DEEP_KD {
        return k_deep;
    }
    // Guo (2002): kd ≈ x² (1 − exp(−x^2.4908))^(−1/2.4908), x = σ √(d/g)
    let x = sigma * (depth / g).sqrt();
    let mut k = x * x * (1.0 - (-x.powf(2.4908)).exp()).powf(-1.0 / 2.4908) / depth;
    for _ in 0..4 {
        let t = (k * depth).tanh();
        let f = g * k * t - sigma * sigma;
        let df = g * t + g * k * depth * (1.0 - t * t);
        let step = f / df;
        k -= step;
        if step.abs() <= 1e-15 * k {
            break;
        }
    }
    k
}

/// `n = c_g / c = ½ (1 + 2kd / sinh 2kd)`: ½ in deep water, 1 in shallow.
pub fn group_velocity_ratio(kd: f64) -> f64 {
    if 2.0 * kd > 2.0 * DEEP_KD {
        0.5
    } else if kd < 1e-8 {
        1.0
    } else {
        0.5 * (1.0 + 2.0 * kd / (2.0 * kd).sinh())
    }
}

/// Group velocity (m/s) of frequency `sigma` with wavenumber `k` in depth `depth`.
pub fn group_velocity(sigma: f64, k: f64, depth: f64) -> f64 {
    group_velocity_ratio(k * depth) * sigma / k
}

/// `∂σ/∂d` at fixed k: `k σ / sinh(2kd)` (1/(m·s)); 0 in deep water.
pub fn dsigma_ddepth(sigma: f64, k: f64, depth: f64) -> f64 {
    let two_kd = 2.0 * k * depth;
    if two_kd > 2.0 * DEEP_KD {
        0.0
    } else {
        k * sigma / two_kd.sinh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 9.81;

    #[test]
    fn the_wavenumber_solves_the_dispersion_relation() {
        for &depth in &[0.05, 0.5, 3.0, 20.0, 100.0, 1000.0] {
            for &period in &[1.0, 3.0, 8.0, 15.0, 25.0] {
                let sigma = 2.0 * std::f64::consts::PI / period;
                let k = wavenumber(sigma, depth, G);
                let residual = G * k * (k * depth).tanh() - sigma * sigma;
                assert!(
                    residual.abs() <= 1e-12 * sigma * sigma,
                    "d = {depth}, T = {period}: residual {residual:e}"
                );
            }
        }
    }

    #[test]
    fn the_limits_are_deep_and_shallow_water() {
        // Deep: k = σ²/g, c_g = g/(2σ)
        let sigma = 2.0;
        let k = wavenumber(sigma, 500.0, G);
        assert!((k - sigma * sigma / G).abs() < 1e-14 * k);
        assert!((group_velocity(sigma, k, 500.0) - G / (2.0 * sigma)).abs() < 1e-12);
        assert_eq!(dsigma_ddepth(sigma, k, 500.0), 0.0);
        // Shallow: c = c_g = √(gd) to O((kd)²)
        let (sigma, depth) = (0.05, 2.0);
        let k = wavenumber(sigma, depth, G);
        let kd = k * depth;
        let c0 = (G * depth).sqrt();
        assert!(kd < 0.03);
        assert!((sigma / k - c0).abs() < 0.5 * kd * kd * c0);
        assert!((group_velocity(sigma, k, depth) - c0).abs() < kd * kd * c0);
    }

    #[test]
    fn the_group_velocity_is_the_derivative_of_the_dispersion_relation() {
        // c_g = dσ/dk, by a central difference of σ(k) = √(g k tanh kd)
        let sigma_of = |k: f64, d: f64| (G * k * (k * d).tanh()).sqrt();
        for &(k, d) in &[(0.01, 5.0), (0.1, 5.0), (0.5, 3.0), (0.05, 40.0)] {
            let sigma = sigma_of(k, d);
            let h = 1e-6 * k;
            let fd = (sigma_of(k + h, d) - sigma_of(k - h, d)) / (2.0 * h);
            assert!((group_velocity(sigma, k, d) - fd).abs() < 1e-8 * fd);
            // and ∂σ/∂d at fixed k
            let hd = 1e-6 * d;
            let fd = (sigma_of(k, d + hd) - sigma_of(k, d - hd)) / (2.0 * hd);
            assert!((dsigma_ddepth(sigma, k, d) - fd).abs() < 1e-7 * fd.abs().max(1e-12));
        }
    }
}
