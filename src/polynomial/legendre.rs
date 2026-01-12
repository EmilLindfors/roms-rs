//! Legendre polynomial evaluation.
//!
//! Legendre polynomials P_n(x) are orthogonal on [-1, 1] with weight 1:
//! ∫_{-1}^{1} P_m(x) P_n(x) dx = 2/(2n+1) δ_{mn}

/// Evaluate Legendre polynomial P_n(x) using three-term recurrence.
///
/// The recurrence relation is:
/// P_0(x) = 1
/// P_1(x) = x
/// (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
pub fn legendre(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }

    let mut p_prev = 1.0; // P_{n-2}
    let mut p_curr = x; // P_{n-1}

    for k in 1..n {
        let p_next = ((2 * k + 1) as f64 * x * p_curr - k as f64 * p_prev) / (k + 1) as f64;
        p_prev = p_curr;
        p_curr = p_next;
    }

    p_curr
}

/// Evaluate derivative of Legendre polynomial P'_n(x).
///
/// Uses the relation:
/// P'_n(x) = n (x P_n(x) - P_{n-1}(x)) / (x^2 - 1)  for |x| != 1
/// P'_n(1) = n(n+1)/2
/// P'_n(-1) = (-1)^{n+1} n(n+1)/2
pub fn legendre_derivative(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }

    // Handle boundary cases where x^2 - 1 = 0
    if (x - 1.0).abs() < 1e-14 {
        return (n * (n + 1)) as f64 / 2.0;
    }
    if (x + 1.0).abs() < 1e-14 {
        let sign = if n % 2 == 0 { -1.0 } else { 1.0 };
        return sign * (n * (n + 1)) as f64 / 2.0;
    }

    let p_n = legendre(n, x);
    let p_n_minus_1 = legendre(n - 1, x);

    n as f64 * (x * p_n - p_n_minus_1) / (x * x - 1.0)
}

/// Evaluate both P_n(x) and P'_n(x) efficiently.
///
/// This is more efficient when both values are needed, as it computes
/// the recurrence only once.
pub fn legendre_and_derivative(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0);
    }

    // Compute P_n and P_{n-1} using recurrence
    let mut p_prev = 1.0; // P_0
    let mut p_curr = x; // P_1

    for k in 1..n {
        let p_next = ((2 * k + 1) as f64 * x * p_curr - k as f64 * p_prev) / (k + 1) as f64;
        p_prev = p_curr;
        p_curr = p_next;
    }

    let p_n = p_curr;
    let p_n_minus_1 = p_prev;

    // Compute derivative
    let dp_n = if (x - 1.0).abs() < 1e-14 {
        (n * (n + 1)) as f64 / 2.0
    } else if (x + 1.0).abs() < 1e-14 {
        let sign = if n % 2 == 0 { -1.0 } else { 1.0 };
        sign * (n * (n + 1)) as f64 / 2.0
    } else {
        n as f64 * (x * p_n - p_n_minus_1) / (x * x - 1.0)
    };

    (p_n, dp_n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legendre_values() {
        // P_0(x) = 1
        assert!((legendre(0, 0.5) - 1.0).abs() < 1e-14);

        // P_1(x) = x
        assert!((legendre(1, 0.5) - 0.5).abs() < 1e-14);

        // P_2(x) = (3x^2 - 1)/2
        let x = 0.5;
        let expected = (3.0 * x * x - 1.0) / 2.0;
        assert!((legendre(2, x) - expected).abs() < 1e-14);

        // P_3(x) = (5x^3 - 3x)/2
        let expected = (5.0 * x * x * x - 3.0 * x) / 2.0;
        assert!((legendre(3, x) - expected).abs() < 1e-14);
    }

    #[test]
    fn test_legendre_at_boundaries() {
        // P_n(1) = 1 for all n
        for n in 0..=5 {
            assert!((legendre(n, 1.0) - 1.0).abs() < 1e-14);
        }

        // P_n(-1) = (-1)^n
        for n in 0..=5 {
            let expected = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert!((legendre(n, -1.0) - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn test_legendre_derivative() {
        // P'_0 = 0
        assert!((legendre_derivative(0, 0.5) - 0.0).abs() < 1e-14);

        // P'_1 = 1
        assert!((legendre_derivative(1, 0.5) - 1.0).abs() < 1e-14);

        // P'_2 = 3x
        let x = 0.5;
        assert!((legendre_derivative(2, x) - 3.0 * x).abs() < 1e-14);

        // P'_3 = (15x^2 - 3)/2
        let expected = (15.0 * x * x - 3.0) / 2.0;
        assert!((legendre_derivative(3, x) - expected).abs() < 1e-14);
    }

    #[test]
    fn test_derivative_at_boundaries() {
        // P'_n(1) = n(n+1)/2
        for n in 0..=5 {
            let expected = (n * (n + 1)) as f64 / 2.0;
            assert!((legendre_derivative(n, 1.0) - expected).abs() < 1e-12);
        }

        // P'_n(-1) = (-1)^{n+1} n(n+1)/2
        for n in 0..=5 {
            let sign = if n % 2 == 0 { -1.0 } else { 1.0 };
            let expected = sign * (n * (n + 1)) as f64 / 2.0;
            assert!((legendre_derivative(n, -1.0) - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_legendre_and_derivative_consistency() {
        for n in 0..=5 {
            for &x in &[-0.9, -0.5, 0.0, 0.5, 0.9] {
                let (p, dp) = legendre_and_derivative(n, x);
                assert!((p - legendre(n, x)).abs() < 1e-14);
                assert!((dp - legendre_derivative(n, x)).abs() < 1e-14);
            }
        }
    }
}
