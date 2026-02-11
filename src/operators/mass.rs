//! Mass matrix for nodal DG.
//!
//! For Gauss-Lobatto-Legendre nodes, the mass matrix is diagonal:
//! M = diag(weights)
//!
//! With (N+1) GLL points, quadrature is exact for polynomials up to degree 2N-1.
//! The product of two degree-N basis functions is degree 2N, so there is a
//! degree-2N aliasing error. However, this aliasing is harmless for the mass
//! matrix: it only affects the diagonal entries slightly without impacting
//! stability or convergence of the DG scheme.

use faer::Mat;

/// Compute the diagonal mass matrix from quadrature weights.
pub fn mass_matrix(weights: &[f64]) -> Mat<f64> {
    let n = weights.len();
    let mut m = Mat::zeros(n, n);
    for i in 0..n {
        m[(i, i)] = weights[i];
    }
    m
}

/// Compute the inverse mass matrix (diagonal).
pub fn mass_matrix_inv(weights: &[f64]) -> Mat<f64> {
    let n = weights.len();
    let mut m_inv = Mat::zeros(n, n);
    for i in 0..n {
        m_inv[(i, i)] = 1.0 / weights[i];
    }
    m_inv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial::{gauss_lobatto_nodes, gauss_lobatto_weights};

    #[test]
    fn test_mass_inverse() {
        for order in 1..=5 {
            let nodes = gauss_lobatto_nodes(order);
            let weights = gauss_lobatto_weights(order, &nodes);

            let m = mass_matrix(&weights);
            let m_inv = mass_matrix_inv(&weights);

            let n = order + 1;

            // M * M^{-1} should be identity
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += m[(i, k)] * m_inv[(k, j)];
                    }
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (sum - expected).abs() < 1e-14,
                        "M * M^{{-1}} should be identity"
                    );
                }
            }
        }
    }

    #[test]
    fn test_mass_is_diagonal() {
        for order in 1..=5 {
            let nodes = gauss_lobatto_nodes(order);
            let weights = gauss_lobatto_weights(order, &nodes);
            let m = mass_matrix(&weights);

            let n = order + 1;

            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        assert!(m[(i, j)].abs() < 1e-14, "Mass matrix should be diagonal");
                    }
                }
            }
        }
    }
}
