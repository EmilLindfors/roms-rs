//! Vandermonde matrix for nodal-modal transformations.
//!
//! The Vandermonde matrix V connects nodal and modal representations:
//! - V[i,j] = φ_j(r_i) where φ_j is the j-th basis polynomial and r_i is the i-th node
//! - nodal_values = V * modal_coeffs
//! - modal_coeffs = V^{-1} * nodal_values
//!
//! For 1D with Legendre polynomials, we use normalized polynomials so that
//! the mass matrix in modal space is the identity.

use crate::polynomial::{legendre, legendre_derivative};
use faer::{Mat, linalg::solvers::Solve};

/// Vandermonde matrix and its inverse.
#[derive(Clone)]
pub struct Vandermonde {
    /// Vandermonde matrix: V[i,j] = P_j(r_i) (normalized)
    pub v: Mat<f64>,
    /// Inverse Vandermonde matrix
    pub v_inv: Mat<f64>,
    /// Derivative Vandermonde: Vr[i,j] = P'_j(r_i) (normalized)
    pub vr: Mat<f64>,
    /// Polynomial order
    pub order: usize,
}

impl Vandermonde {
    /// Create Vandermonde matrix for given order and nodes.
    ///
    /// The basis polynomials are normalized Legendre polynomials:
    /// φ_j(x) = sqrt((2j+1)/2) * P_j(x)
    ///
    /// This normalization ensures ∫ φ_i φ_j dx = δ_{ij}
    pub fn new(order: usize, nodes: &[f64]) -> Self {
        let n = order + 1;
        assert_eq!(nodes.len(), n, "Need order+1 nodes");

        let mut v = Mat::zeros(n, n);
        let mut vr = Mat::zeros(n, n);

        for (i, &r) in nodes.iter().enumerate() {
            for j in 0..n {
                // Normalization factor: sqrt((2j+1)/2)
                let norm = ((2 * j + 1) as f64 / 2.0).sqrt();

                v[(i, j)] = norm * legendre(j, r);
                vr[(i, j)] = norm * legendre_derivative(j, r);
            }
        }

        // Compute inverse using LU decomposition
        let lu = v.as_ref().full_piv_lu();
        let mut v_inv = Mat::zeros(n, n);

        // Solve V * V_inv = I column by column
        for j in 0..n {
            let mut rhs = Mat::zeros(n, 1);
            rhs[(j, 0)] = 1.0;
            let col = lu.solve(&rhs);
            for i in 0..n {
                v_inv[(i, j)] = col[(i, 0)];
            }
        }

        Self {
            v,
            v_inv,
            vr,
            order,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial::gauss_lobatto_nodes;

    fn mat_approx_eq(a: &Mat<f64>, b: &Mat<f64>, tol: f64) -> bool {
        if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
            return false;
        }
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                if (a[(i, j)] - b[(i, j)]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_vandermonde_invertibility() {
        for order in 1..=5 {
            let nodes = gauss_lobatto_nodes(order);
            let vander = Vandermonde::new(order, &nodes);

            // V * V^{-1} should be identity
            let n = order + 1;
            let mut product = Mat::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += vander.v[(i, k)] * vander.v_inv[(k, j)];
                    }
                    product[(i, j)] = sum;
                }
            }

            let mut identity = Mat::zeros(n, n);
            for i in 0..n {
                identity[(i, i)] = 1.0;
            }

            assert!(
                mat_approx_eq(&product, &identity, 1e-12),
                "V * V^{{-1}} should be identity for order {}",
                order
            );
        }
    }

    #[test]
    fn test_nodal_modal_roundtrip() {
        let order = 4;
        let nodes = gauss_lobatto_nodes(order);
        let vander = Vandermonde::new(order, &nodes);

        // Create some nodal values
        let n = order + 1;
        let nodal: Vec<f64> = nodes.iter().map(|&x| x.powi(2) + x).collect();

        // Convert to modal: modal = V^{-1} * nodal
        let mut modal = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                modal[i] += vander.v_inv[(i, j)] * nodal[j];
            }
        }

        // Convert back to nodal: nodal' = V * modal
        let mut nodal_back = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                nodal_back[i] += vander.v[(i, j)] * modal[j];
            }
        }

        // Should match original
        for i in 0..n {
            assert!(
                (nodal[i] - nodal_back[i]).abs() < 1e-12,
                "Roundtrip failed at node {}: {} vs {}",
                i,
                nodal[i],
                nodal_back[i]
            );
        }
    }
}
