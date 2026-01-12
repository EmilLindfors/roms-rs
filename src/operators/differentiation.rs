//! Differentiation matrix for nodal DG.
//!
//! The differentiation matrix Dr maps nodal values to derivative values:
//! (du/dr)_i = Σ_j Dr[i,j] * u_j
//!
//! Computed as Dr = Vr * V^{-1} where:
//! - V[i,j] = φ_j(r_i) is the Vandermonde matrix
//! - Vr[i,j] = φ'_j(r_i) is the derivative Vandermonde matrix
//!
//! For strong form DG with collocation, we also need:
//! D_strong = M^{-1} * Dr^T * M = diag(1/w) * Dr^T * diag(w)

use crate::basis::Vandermonde;
use faer::Mat;

/// Compute the differentiation matrix Dr = Vr * V^{-1}.
///
/// This matrix gives derivatives at nodes: (du/dr)_i = Σ_j Dr[i,j] * u_j
pub fn differentiation_matrix(vander: &Vandermonde) -> Mat<f64> {
    let n = vander.order + 1;
    let mut dr = Mat::zeros(n, n);

    // Dr = Vr * V^{-1}
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += vander.vr[(i, k)] * vander.v_inv[(k, j)];
            }
            dr[(i, j)] = sum;
        }
    }

    dr
}

/// Compute the strong form derivative operator D = M^{-1} * Dr^T * M.
///
/// For diagonal mass matrix M = diag(w), this is:
/// D[i,j] = (w_j / w_i) * Dr[j,i]
///
/// This is the correct operator for strong form DG with GLL collocation.
pub fn strong_differentiation_matrix(dr: &Mat<f64>, weights: &[f64]) -> Mat<f64> {
    let n = dr.nrows();
    let mut d = Mat::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            // D[i,j] = (w_j / w_i) * Dr^T[i,j] = (w_j / w_i) * Dr[j,i]
            d[(i, j)] = (weights[j] / weights[i]) * dr[(j, i)];
        }
    }

    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial::gauss_lobatto_nodes;

    #[test]
    fn test_differentiation_constant() {
        // Derivative of constant should be zero
        for order in 1..=5 {
            let nodes = gauss_lobatto_nodes(order);
            let vander = Vandermonde::new(order, &nodes);
            let dr = differentiation_matrix(&vander);

            let n = order + 1;
            let constant: Vec<f64> = vec![1.0; n];

            // Compute Dr * constant
            let mut deriv = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    deriv[i] += dr[(i, j)] * constant[j];
                }
            }

            for (i, &d) in deriv.iter().enumerate() {
                assert!(
                    d.abs() < 1e-12,
                    "Derivative of constant should be 0, got {} at node {}",
                    d,
                    i
                );
            }
        }
    }

    #[test]
    fn test_differentiation_linear() {
        // Derivative of x should be 1
        for order in 1..=5 {
            let nodes = gauss_lobatto_nodes(order);
            let vander = Vandermonde::new(order, &nodes);
            let dr = differentiation_matrix(&vander);

            let n = order + 1;
            let linear: Vec<f64> = nodes.clone();

            // Compute Dr * x
            let mut deriv = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    deriv[i] += dr[(i, j)] * linear[j];
                }
            }

            for (i, &d) in deriv.iter().enumerate() {
                assert!(
                    (d - 1.0).abs() < 1e-12,
                    "Derivative of x should be 1, got {} at node {}",
                    d,
                    i
                );
            }
        }
    }

    #[test]
    fn test_differentiation_quadratic() {
        // Derivative of x^2 should be 2x
        for order in 2..=5 {
            let nodes = gauss_lobatto_nodes(order);
            let vander = Vandermonde::new(order, &nodes);
            let dr = differentiation_matrix(&vander);

            let n = order + 1;
            let quadratic: Vec<f64> = nodes.iter().map(|&x| x * x).collect();
            let expected: Vec<f64> = nodes.iter().map(|&x| 2.0 * x).collect();

            // Compute Dr * x^2
            let mut deriv = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    deriv[i] += dr[(i, j)] * quadratic[j];
                }
            }

            for (i, (&d, &e)) in deriv.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (d - e).abs() < 1e-12,
                    "Derivative of x^2 should be 2x: got {} vs {} at node {}",
                    d,
                    e,
                    i
                );
            }
        }
    }

    #[test]
    fn test_differentiation_polynomial_exactness() {
        // Dr should exactly differentiate polynomials up to degree N
        let order = 4;
        let nodes = gauss_lobatto_nodes(order);
        let vander = Vandermonde::new(order, &nodes);
        let dr = differentiation_matrix(&vander);

        let n = order + 1;

        for k in 0..=order {
            // u(x) = x^k
            let u: Vec<f64> = nodes.iter().map(|&x| x.powi(k as i32)).collect();

            // du/dx = k * x^{k-1}
            let du_exact: Vec<f64> = if k == 0 {
                vec![0.0; n]
            } else {
                nodes
                    .iter()
                    .map(|&x| k as f64 * x.powi((k - 1) as i32))
                    .collect()
            };

            // Compute Dr * u
            let mut du = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    du[i] += dr[(i, j)] * u[j];
                }
            }

            for (i, (&d, &e)) in du.iter().zip(du_exact.iter()).enumerate() {
                assert!(
                    (d - e).abs() < 1e-11,
                    "Degree {}: derivative at node {} should be {}, got {}",
                    k,
                    i,
                    e,
                    d
                );
            }
        }
    }
}
