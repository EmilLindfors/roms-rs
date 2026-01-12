//! 2D Vandermonde matrix for tensor-product quadrilateral elements.
//!
//! The 2D Vandermonde matrix V connects nodal and modal representations:
//! - V[k, m] = φ_m(r_k, s_k) where φ_m is the m-th 2D basis polynomial
//!   and (r_k, s_k) is the k-th node
//! - nodal_values = V * modal_coeffs
//! - modal_coeffs = V^{-1} * nodal_values
//!
//! For tensor-product elements with normalized Legendre polynomials,
//! the mass matrix in modal space is the identity.

use crate::polynomial::{
    legendre_2d_norm, legendre_2d_with_gradient, mode_degrees, tensor_product_gll_nodes,
};
use faer::{Mat, linalg::solvers::Solve};

/// 2D Vandermonde matrix and its inverse for tensor-product elements.
#[derive(Clone)]
pub struct Vandermonde2D {
    /// Vandermonde matrix: V[k, m] = φ_m(r_k, s_k) (normalized)
    /// Shape: (n_nodes, n_modes) = ((N+1)², (N+1)²)
    pub v: Mat<f64>,

    /// Inverse Vandermonde matrix
    /// Shape: (n_modes, n_nodes)
    pub v_inv: Mat<f64>,

    /// Derivative Vandermonde w.r.t. r: Vr[k, m] = ∂φ_m/∂r(r_k, s_k)
    /// Shape: (n_nodes, n_modes)
    pub vr: Mat<f64>,

    /// Derivative Vandermonde w.r.t. s: Vs[k, m] = ∂φ_m/∂s(r_k, s_k)
    /// Shape: (n_nodes, n_modes)
    pub vs: Mat<f64>,

    /// Polynomial order
    pub order: usize,

    /// Number of nodes = (order+1)²
    pub n_nodes: usize,

    /// Number of modes = (order+1)²
    pub n_modes: usize,

    /// Number of nodes in 1D = order+1
    pub n_1d: usize,
}

impl Vandermonde2D {
    /// Create 2D Vandermonde matrix for given order.
    ///
    /// Uses tensor-product GLL nodes and normalized Legendre basis:
    /// φ_{ij}(r, s) = norm_i * norm_j * P_i(r) * P_j(s)
    ///
    /// where norm_k = √((2k+1)/2)
    ///
    /// # Arguments
    /// * `order` - Polynomial order (N), resulting in (N+1)² nodes and modes
    pub fn new(order: usize) -> Self {
        let n_1d = order + 1;
        let n_nodes = n_1d * n_1d;
        let n_modes = n_nodes;

        let nodes = tensor_product_gll_nodes(order);
        assert_eq!(nodes.len(), n_nodes);

        let mut v = Mat::zeros(n_nodes, n_modes);
        let mut vr = Mat::zeros(n_nodes, n_modes);
        let mut vs = Mat::zeros(n_nodes, n_modes);

        // Fill Vandermonde matrices
        for (k, &(r, s)) in nodes.iter().enumerate() {
            for m in 0..n_modes {
                let (i, j) = mode_degrees(m, n_1d);
                let norm = legendre_2d_norm(i, j);

                // Get polynomial value and gradients
                let (p, dp_dr, dp_ds) = legendre_2d_with_gradient(i, j, r, s);

                v[(k, m)] = norm * p;
                vr[(k, m)] = norm * dp_dr;
                vs[(k, m)] = norm * dp_ds;
            }
        }

        // Compute inverse using LU decomposition
        let lu = v.as_ref().full_piv_lu();
        let mut v_inv = Mat::zeros(n_modes, n_nodes);

        // Solve V * V_inv = I column by column
        for col in 0..n_nodes {
            let mut rhs = Mat::zeros(n_nodes, 1);
            rhs[(col, 0)] = 1.0;
            let solution = lu.solve(&rhs);
            for row in 0..n_modes {
                v_inv[(row, col)] = solution[(row, 0)];
            }
        }

        Self {
            v,
            v_inv,
            vr,
            vs,
            order,
            n_nodes,
            n_modes,
            n_1d,
        }
    }

    /// Create 2D Vandermonde matrix from custom nodes.
    ///
    /// # Arguments
    /// * `order` - Polynomial order
    /// * `nodes` - Custom node positions as (r, s) pairs
    pub fn with_nodes(order: usize, nodes: &[(f64, f64)]) -> Self {
        let n_1d = order + 1;
        let n_nodes = nodes.len();
        let n_modes = n_1d * n_1d;

        assert!(
            n_nodes >= n_modes,
            "Need at least {} nodes for order {}, got {}",
            n_modes,
            order,
            n_nodes
        );

        let mut v = Mat::zeros(n_nodes, n_modes);
        let mut vr = Mat::zeros(n_nodes, n_modes);
        let mut vs = Mat::zeros(n_nodes, n_modes);

        // Fill Vandermonde matrices
        for (k, &(r, s)) in nodes.iter().enumerate() {
            for m in 0..n_modes {
                let (i, j) = mode_degrees(m, n_1d);
                let norm = legendre_2d_norm(i, j);

                let (p, dp_dr, dp_ds) = legendre_2d_with_gradient(i, j, r, s);

                v[(k, m)] = norm * p;
                vr[(k, m)] = norm * dp_dr;
                vs[(k, m)] = norm * dp_ds;
            }
        }

        // Compute inverse (pseudo-inverse if n_nodes > n_modes)
        let lu = v.as_ref().full_piv_lu();
        let mut v_inv = Mat::zeros(n_modes, n_nodes);

        if n_nodes == n_modes {
            // Square matrix: use standard inverse
            for col in 0..n_nodes {
                let mut rhs = Mat::zeros(n_nodes, 1);
                rhs[(col, 0)] = 1.0;
                let solution = lu.solve(&rhs);
                for row in 0..n_modes {
                    v_inv[(row, col)] = solution[(row, 0)];
                }
            }
        } else {
            // Overdetermined: would need pseudo-inverse
            // For now, panic since we don't support this case yet
            panic!("Overdetermined case (n_nodes > n_modes) not yet supported");
        }

        Self {
            v,
            v_inv,
            vr,
            vs,
            order,
            n_nodes,
            n_modes,
            n_1d,
        }
    }

    /// Convert nodal values to modal coefficients.
    ///
    /// modal = V^{-1} * nodal
    pub fn nodal_to_modal(&self, nodal: &[f64]) -> Vec<f64> {
        assert_eq!(nodal.len(), self.n_nodes);
        let mut modal = vec![0.0; self.n_modes];

        for m in 0..self.n_modes {
            for k in 0..self.n_nodes {
                modal[m] += self.v_inv[(m, k)] * nodal[k];
            }
        }

        modal
    }

    /// Convert modal coefficients to nodal values.
    ///
    /// nodal = V * modal
    pub fn modal_to_nodal(&self, modal: &[f64]) -> Vec<f64> {
        assert_eq!(modal.len(), self.n_modes);
        let mut nodal = vec![0.0; self.n_nodes];

        for k in 0..self.n_nodes {
            for m in 0..self.n_modes {
                nodal[k] += self.v[(k, m)] * modal[m];
            }
        }

        nodal
    }

    /// Compute r-derivative from nodal values.
    ///
    /// Returns ∂u/∂r at each node, computed via:
    /// du_dr = Vr * V^{-1} * nodal
    pub fn differentiate_r(&self, nodal: &[f64]) -> Vec<f64> {
        let modal = self.nodal_to_modal(nodal);
        let mut du_dr = vec![0.0; self.n_nodes];

        for k in 0..self.n_nodes {
            for m in 0..self.n_modes {
                du_dr[k] += self.vr[(k, m)] * modal[m];
            }
        }

        du_dr
    }

    /// Compute s-derivative from nodal values.
    ///
    /// Returns ∂u/∂s at each node, computed via:
    /// du_ds = Vs * V^{-1} * nodal
    pub fn differentiate_s(&self, nodal: &[f64]) -> Vec<f64> {
        let modal = self.nodal_to_modal(nodal);
        let mut du_ds = vec![0.0; self.n_nodes];

        for k in 0..self.n_nodes {
            for m in 0..self.n_modes {
                du_ds[k] += self.vs[(k, m)] * modal[m];
            }
        }

        du_ds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial::{mode_degrees, tensor_product_gll_weights};

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
    fn test_vandermonde_2d_dimensions() {
        for order in 1..=4 {
            let v = Vandermonde2D::new(order);
            let n = (order + 1) * (order + 1);

            assert_eq!(v.n_nodes, n);
            assert_eq!(v.n_modes, n);
            assert_eq!(v.n_1d, order + 1);
            assert_eq!(v.v.nrows(), n);
            assert_eq!(v.v.ncols(), n);
            assert_eq!(v.v_inv.nrows(), n);
            assert_eq!(v.v_inv.ncols(), n);
            assert_eq!(v.vr.nrows(), n);
            assert_eq!(v.vr.ncols(), n);
            assert_eq!(v.vs.nrows(), n);
            assert_eq!(v.vs.ncols(), n);
        }
    }

    #[test]
    fn test_vandermonde_2d_invertibility() {
        for order in 1..=4 {
            let vander = Vandermonde2D::new(order);
            let n = vander.n_nodes;

            // V * V^{-1} should be identity
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
                mat_approx_eq(&product, &identity, 1e-11),
                "V * V^{{-1}} should be identity for order {}",
                order
            );
        }
    }

    #[test]
    fn test_nodal_modal_roundtrip() {
        let order = 3;
        let vander = Vandermonde2D::new(order);
        let nodes = tensor_product_gll_nodes(order);

        // Create nodal values from a polynomial: f(r, s) = r² + r*s + s
        let nodal: Vec<f64> = nodes.iter().map(|&(r, s)| r * r + r * s + s).collect();

        // Roundtrip: nodal -> modal -> nodal
        let modal = vander.nodal_to_modal(&nodal);
        let nodal_back = vander.modal_to_nodal(&modal);

        for i in 0..nodal.len() {
            assert!(
                (nodal[i] - nodal_back[i]).abs() < 1e-12,
                "Roundtrip failed at node {}: {} vs {}",
                i,
                nodal[i],
                nodal_back[i]
            );
        }
    }

    #[test]
    fn test_differentiation_r_polynomial_exactness() {
        // Test that differentiation is exact for polynomials up to order N
        let order = 3;
        let vander = Vandermonde2D::new(order);
        let nodes = tensor_product_gll_nodes(order);

        // f(r, s) = r³ + 2*r²*s - r + 1
        // ∂f/∂r = 3*r² + 4*r*s - 1
        let f: Vec<f64> = nodes
            .iter()
            .map(|&(r, s)| r.powi(3) + 2.0 * r * r * s - r + 1.0)
            .collect();

        let df_dr_exact: Vec<f64> = nodes
            .iter()
            .map(|&(r, s)| 3.0 * r * r + 4.0 * r * s - 1.0)
            .collect();

        let df_dr = vander.differentiate_r(&f);

        for i in 0..nodes.len() {
            assert!(
                (df_dr[i] - df_dr_exact[i]).abs() < 1e-11,
                "∂f/∂r at node {}: got {}, expected {}",
                i,
                df_dr[i],
                df_dr_exact[i]
            );
        }
    }

    #[test]
    fn test_differentiation_s_polynomial_exactness() {
        // Test that differentiation is exact for polynomials up to order N
        let order = 3;
        let vander = Vandermonde2D::new(order);
        let nodes = tensor_product_gll_nodes(order);

        // f(r, s) = s³ - r*s² + 2*s
        // ∂f/∂s = 3*s² - 2*r*s + 2
        let f: Vec<f64> = nodes
            .iter()
            .map(|&(r, s)| s.powi(3) - r * s * s + 2.0 * s)
            .collect();

        let df_ds_exact: Vec<f64> = nodes
            .iter()
            .map(|&(r, s)| 3.0 * s * s - 2.0 * r * s + 2.0)
            .collect();

        let df_ds = vander.differentiate_s(&f);

        for i in 0..nodes.len() {
            assert!(
                (df_ds[i] - df_ds_exact[i]).abs() < 1e-11,
                "∂f/∂s at node {}: got {}, expected {}",
                i,
                df_ds[i],
                df_ds_exact[i]
            );
        }
    }

    #[test]
    fn test_differentiation_constant() {
        // Derivative of constant should be zero
        let order = 2;
        let vander = Vandermonde2D::new(order);

        let constant = vec![3.14159; vander.n_nodes];
        let dr = vander.differentiate_r(&constant);
        let ds = vander.differentiate_s(&constant);

        for i in 0..vander.n_nodes {
            assert!(
                dr[i].abs() < 1e-13,
                "∂(constant)/∂r should be 0, got {}",
                dr[i]
            );
            assert!(
                ds[i].abs() < 1e-13,
                "∂(constant)/∂s should be 0, got {}",
                ds[i]
            );
        }
    }

    #[test]
    fn test_differentiation_linear() {
        // ∂(ar + bs + c)/∂r = a, ∂(ar + bs + c)/∂s = b
        let order = 2;
        let vander = Vandermonde2D::new(order);
        let nodes = tensor_product_gll_nodes(order);

        let a = 2.5;
        let b = -1.3;
        let c = 0.7;

        let linear: Vec<f64> = nodes.iter().map(|&(r, s)| a * r + b * s + c).collect();

        let dr = vander.differentiate_r(&linear);
        let ds = vander.differentiate_s(&linear);

        for i in 0..vander.n_nodes {
            assert!(
                (dr[i] - a).abs() < 1e-13,
                "∂(ar+bs+c)/∂r should be {}, got {}",
                a,
                dr[i]
            );
            assert!(
                (ds[i] - b).abs() < 1e-13,
                "∂(ar+bs+c)/∂s should be {}, got {}",
                b,
                ds[i]
            );
        }
    }

    #[test]
    fn test_orthonormality_low_order_modes() {
        // With normalized Legendre basis, V^T * diag(w) * V ≈ I
        // This is only exact when quadrature can integrate the product exactly.
        // GLL with N+1 points is exact for degree ≤ 2N-1, so mode products
        // up to total degree N-1 in each direction are exact.
        //
        // We test with order=4 (5 points, exact up to degree 7) and check
        // modes with max degree 2 (product degree ≤ 4, well within limits).
        let order = 4;
        let vander = Vandermonde2D::new(order);
        let weights = tensor_product_gll_weights(order);
        let n = vander.n_nodes;
        let n_1d = order + 1;

        // Compute diag(w) * V
        let mut wv = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                wv[(i, j)] = weights[i] * vander.v[(i, j)];
            }
        }

        // Compute V^T * (diag(w) * V)
        let mut gram = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += vander.v[(k, i)] * wv[(k, j)];
                }
                gram[(i, j)] = sum;
            }
        }

        // Only check modes with degree ≤ 2 in each direction
        // (their products have degree ≤ 4, well within GLL exactness)
        let max_check_degree = 2;
        for m1 in 0..n {
            let (i1, j1) = mode_degrees(m1, n_1d);
            if i1 > max_check_degree || j1 > max_check_degree {
                continue;
            }

            for m2 in 0..n {
                let (i2, j2) = mode_degrees(m2, n_1d);
                if i2 > max_check_degree || j2 > max_check_degree {
                    continue;
                }

                let expected = if m1 == m2 { 1.0 } else { 0.0 };
                assert!(
                    (gram[(m1, m2)] - expected).abs() < 1e-12,
                    "Gram[({},{}), ({},{})] = {}, expected {}",
                    i1,
                    j1,
                    i2,
                    j2,
                    gram[(m1, m2)],
                    expected
                );
            }
        }
    }
}
