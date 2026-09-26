//! Gauss-Lobatto-Legendre nodes and weights.
//!
//! The N+1 Gauss-Lobatto-Legendre (GLL) nodes are the roots of (1-x²)P'_N(x),
//! which includes the endpoints x = ±1. These nodes are optimal for nodal DG
//! because:
//! 1. The mass matrix becomes diagonal
//! 2. Surface quadrature points coincide with volume nodes
//! 3. Interpolation is stable up to high orders

use super::legendre::{legendre, legendre_and_derivative};
use std::f64::consts::PI;

/// Compute Gauss-Lobatto-Legendre nodes for polynomial order N.
///
/// Returns N+1 nodes in [-1, 1], including the endpoints.
/// Uses Newton iteration starting from Chebyshev-Lobatto nodes.
pub fn gauss_lobatto_nodes(order: usize) -> Vec<f64> {
    let n = order;
    let n_nodes = n + 1;

    if n == 0 {
        return vec![0.0];
    }
    if n == 1 {
        return vec![-1.0, 1.0];
    }

    let mut nodes = Vec::with_capacity(n_nodes);

    // Initial guess: Chebyshev-Lobatto nodes
    // x_j = -cos(π j / N) for j = 0, ..., N
    for j in 0..=n {
        nodes.push(-(PI * j as f64 / n as f64).cos());
    }

    // Newton iteration to find roots of (1-x²)P'_N(x)
    // The roots are: x = ±1 (trivial) and zeros of P'_N(x) (interior)
    // For interior nodes, we solve P'_N(x) = 0

    // Endpoints are exact
    nodes[0] = -1.0;
    nodes[n] = 1.0;

    // Newton iteration for interior nodes
    for j in 1..n {
        let mut x = nodes[j];

        for _ in 0..100 {
            let (p_n, dp_n) = legendre_and_derivative(n, x);

            // We want to find roots of (1-x²)P'_N(x)
            // Taking derivative: -2x P'_N + (1-x²) P''_N
            // Using P''_N = (2x P'_N - n(n+1) P_N) / (1-x²) for |x| < 1
            // Simplifies to: the update for finding zeros of P'_N is:
            // x_new = x - P'_N(x) / P''_N(x)
            //
            // But it's easier to use: for GLL nodes, we find zeros of
            // L_N(x) = (1-x²) P'_N(x)
            // L'_N(x) = -2x P'_N(x) + (1-x²) P''_N(x)
            //
            // Using the identity: P''_N = (n(n+1) P_N - 2x P'_N) / (x² - 1)
            // We get: L'_N = -2x P'_N + (1-x²) * (n(n+1) P_N - 2x P'_N) / (x² - 1)
            //              = -2x P'_N - (n(n+1) P_N - 2x P'_N)
            //              = -n(n+1) P_N
            //
            // So the Newton update is:
            // x_new = x - (1-x²) P'_N / (-n(n+1) P_N)
            //       = x + (1-x²) P'_N / (n(n+1) P_N)

            let update = (1.0 - x * x) * dp_n / (n as f64 * (n + 1) as f64 * p_n);

            if update.abs() < 1e-15 {
                break;
            }

            x += update;
        }

        nodes[j] = x;
    }

    nodes
}

/// Compute Gauss-Lobatto-Legendre weights.
///
/// The weights are: w_j = 2 / (N(N+1) [P_N(x_j)]²)
pub fn gauss_lobatto_weights(order: usize, nodes: &[f64]) -> Vec<f64> {
    let n = order;

    if n == 0 {
        return vec![2.0];
    }

    let n_nodes = nodes.len();
    let mut weights = Vec::with_capacity(n_nodes);

    let denom = (n * (n + 1)) as f64;

    for &x in nodes {
        let p_n = legendre(n, x);
        let w = 2.0 / (denom * p_n * p_n);
        weights.push(w);
    }

    weights
}

/// Gauss–Legendre nodes and weights with `n` points (n ≥ 1).
///
/// The nodes are the roots of P_n, found by Newton iteration from the
/// Chebyshev guess; w_j = 2 / ((1 − x_j²) P_n'(x_j)²). The rule integrates
/// polynomials of degree 2n − 1 exactly on [−1, 1].
pub fn gauss_legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n >= 1, "Gauss–Legendre needs at least one point");
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for j in 0..n.div_ceil(2) {
        // Root j counted from x = +1
        let mut x = (PI * (j as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (p, dp) = legendre_and_derivative(n, x);
            let dx = p / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        let (_, dp) = legendre_and_derivative(n, x);
        let w = 2.0 / ((1.0 - x * x) * dp * dp);
        nodes[j] = -x;
        nodes[n - 1 - j] = x;
        weights[j] = w;
        weights[n - 1 - j] = w;
    }
    if n % 2 == 1 {
        nodes[n / 2] = 0.0;
    }
    (nodes, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nodes_endpoints() {
        for order in 1..=6 {
            let nodes = gauss_lobatto_nodes(order);
            assert!((nodes[0] - (-1.0)).abs() < 1e-14, "Left endpoint");
            assert!((nodes[order] - 1.0).abs() < 1e-14, "Right endpoint");
        }
    }

    #[test]
    fn test_nodes_count() {
        for order in 0..=6 {
            let nodes = gauss_lobatto_nodes(order);
            assert_eq!(nodes.len(), order + 1);
        }
    }

    #[test]
    fn test_nodes_symmetry() {
        for order in 1..=6 {
            let nodes = gauss_lobatto_nodes(order);
            let n = nodes.len();
            for i in 0..n / 2 {
                assert!(
                    (nodes[i] + nodes[n - 1 - i]).abs() < 1e-14,
                    "Nodes should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_nodes_are_gll() {
        // GLL nodes are roots of (1-x²)P'_N(x)
        // So for interior nodes, P'_N(x_j) should be zero
        for order in 2..=6 {
            let nodes = gauss_lobatto_nodes(order);
            for j in 1..order {
                let (_, dp) = legendre_and_derivative(order, nodes[j]);
                assert!(
                    dp.abs() < 1e-12,
                    "Interior node {} should be root of P'_N, got {}",
                    j,
                    dp
                );
            }
        }
    }

    #[test]
    fn test_weights_sum() {
        // Weights should sum to 2 (length of interval [-1, 1])
        for order in 0..=6 {
            let nodes = gauss_lobatto_nodes(order);
            let weights = gauss_lobatto_weights(order, &nodes);
            let sum: f64 = weights.iter().sum();
            assert!(
                (sum - 2.0).abs() < 1e-14,
                "Weights should sum to 2, got {}",
                sum
            );
        }
    }

    #[test]
    fn test_weights_symmetry() {
        for order in 1..=6 {
            let nodes = gauss_lobatto_nodes(order);
            let weights = gauss_lobatto_weights(order, &nodes);
            let n = weights.len();
            for i in 0..n / 2 {
                assert!(
                    (weights[i] - weights[n - 1 - i]).abs() < 1e-14,
                    "Weights should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_quadrature_exactness() {
        // GLL quadrature with N+1 points is exact for polynomials up to degree 2N-1
        for order in 1..=5 {
            let nodes = gauss_lobatto_nodes(order);
            let weights = gauss_lobatto_weights(order, &nodes);

            // Test integrating x^k for k = 0, 1, ..., 2N-1
            let max_degree = 2 * order - 1;
            for k in 0..=max_degree {
                // ∫_{-1}^{1} x^k dx = [x^{k+1}/(k+1)]_{-1}^{1}
                //                   = (1 - (-1)^{k+1}) / (k+1)
                //                   = 0 if k is odd
                //                   = 2/(k+1) if k is even
                let exact = if k % 2 == 0 {
                    2.0 / (k + 1) as f64
                } else {
                    0.0
                };

                let numerical: f64 = nodes
                    .iter()
                    .zip(weights.iter())
                    .map(|(&x, &w)| w * x.powi(k as i32))
                    .sum();

                assert!(
                    (numerical - exact).abs() < 1e-12,
                    "Order {}, degree {}: expected {}, got {}",
                    order,
                    k,
                    exact,
                    numerical
                );
            }
        }
    }

    #[test]
    fn test_known_nodes() {
        // Order 2: nodes at -1, 0, 1
        let nodes = gauss_lobatto_nodes(2);
        assert!((nodes[0] - (-1.0)).abs() < 1e-14);
        assert!((nodes[1] - 0.0).abs() < 1e-14);
        assert!((nodes[2] - 1.0).abs() < 1e-14);

        // Weights: 1/3, 4/3, 1/3
        let weights = gauss_lobatto_weights(2, &nodes);
        assert!((weights[0] - 1.0 / 3.0).abs() < 1e-14);
        assert!((weights[1] - 4.0 / 3.0).abs() < 1e-14);
        assert!((weights[2] - 1.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_gauss_legendre_exactness() {
        for n in 1..=12 {
            let (nodes, weights) = gauss_legendre_nodes_weights(n);
            assert_eq!(nodes.len(), n);
            assert!(nodes.windows(2).all(|w| w[0] < w[1]), "n = {n}: sorted");
            assert!(nodes.iter().all(|x| x.abs() < 1.0), "n = {n}: interior");
            for k in 0..2 * n {
                let exact = if k % 2 == 0 {
                    2.0 / (k + 1) as f64
                } else {
                    0.0
                };
                let numerical: f64 = nodes
                    .iter()
                    .zip(&weights)
                    .map(|(&x, &w)| w * x.powi(k as i32))
                    .sum();
                assert!(
                    (numerical - exact).abs() < 1e-13,
                    "n = {n}, degree {k}: {numerical} vs {exact}"
                );
            }
        }
        // Two points: ±1/√3, weights 1
        let (nodes, weights) = gauss_legendre_nodes_weights(2);
        assert!((nodes[1] - 1.0 / 3f64.sqrt()).abs() < 1e-15);
        assert!((weights[0] - 1.0).abs() < 1e-15);
    }
}
