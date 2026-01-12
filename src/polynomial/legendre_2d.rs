//! 2D Legendre polynomials for tensor-product elements.
//!
//! For quadrilateral elements, we use tensor-product Legendre polynomials:
//! P_{ij}(r, s) = P_i(r) * P_j(s)
//!
//! These are orthogonal on the reference square [-1, 1]²:
//! ∫∫ P_{ij} P_{kl} dr ds = (2/(2i+1)) * (2/(2j+1)) * δ_{ik} δ_{jl}
//!
//! The normalized basis functions are:
//! φ_{ij}(r, s) = √((2i+1)/2) * √((2j+1)/2) * P_i(r) * P_j(s)
//!
//! which satisfy: ∫∫ φ_{ij} φ_{kl} dr ds = δ_{ik} δ_{jl}

use super::legendre::{legendre, legendre_and_derivative, legendre_derivative};
use super::nodes::{gauss_lobatto_nodes, gauss_lobatto_weights};

/// Evaluate tensor-product Legendre polynomial P_{ij}(r, s) = P_i(r) * P_j(s).
///
/// # Arguments
/// * `i` - Degree in r-direction
/// * `j` - Degree in s-direction
/// * `r` - First coordinate in [-1, 1]
/// * `s` - Second coordinate in [-1, 1]
#[inline]
pub fn legendre_2d(i: usize, j: usize, r: f64, s: f64) -> f64 {
    legendre(i, r) * legendre(j, s)
}

/// Evaluate normalized tensor-product Legendre polynomial.
///
/// φ_{ij}(r, s) = √((2i+1)/2) * √((2j+1)/2) * P_i(r) * P_j(s)
///
/// The normalization ensures orthonormality: ∫∫ φ_{ij} φ_{kl} dr ds = δ_{ik} δ_{jl}
#[inline]
pub fn legendre_2d_normalized(i: usize, j: usize, r: f64, s: f64) -> f64 {
    let norm_i = ((2 * i + 1) as f64 / 2.0).sqrt();
    let norm_j = ((2 * j + 1) as f64 / 2.0).sqrt();
    norm_i * norm_j * legendre(i, r) * legendre(j, s)
}

/// Compute the normalization factor for 2D Legendre polynomial of degree (i, j).
///
/// norm_{ij} = √((2i+1)/2) * √((2j+1)/2)
#[inline]
pub fn legendre_2d_norm(i: usize, j: usize) -> f64 {
    let norm_i = ((2 * i + 1) as f64 / 2.0).sqrt();
    let norm_j = ((2 * j + 1) as f64 / 2.0).sqrt();
    norm_i * norm_j
}

/// Evaluate gradient of tensor-product Legendre polynomial.
///
/// Returns (∂P_{ij}/∂r, ∂P_{ij}/∂s) where:
/// ∂P_{ij}/∂r = P'_i(r) * P_j(s)
/// ∂P_{ij}/∂s = P_i(r) * P'_j(s)
#[inline]
pub fn legendre_2d_gradient(i: usize, j: usize, r: f64, s: f64) -> (f64, f64) {
    let p_i = legendre(i, r);
    let p_j = legendre(j, s);
    let dp_i = legendre_derivative(i, r);
    let dp_j = legendre_derivative(j, s);

    (dp_i * p_j, p_i * dp_j)
}

/// Evaluate gradient of normalized tensor-product Legendre polynomial.
///
/// Returns (∂φ_{ij}/∂r, ∂φ_{ij}/∂s).
#[inline]
pub fn legendre_2d_gradient_normalized(i: usize, j: usize, r: f64, s: f64) -> (f64, f64) {
    let norm = legendre_2d_norm(i, j);
    let (dr, ds) = legendre_2d_gradient(i, j, r, s);
    (norm * dr, norm * ds)
}

/// Evaluate polynomial and its gradient together efficiently.
///
/// Returns (P_{ij}, ∂P_{ij}/∂r, ∂P_{ij}/∂s).
pub fn legendre_2d_with_gradient(i: usize, j: usize, r: f64, s: f64) -> (f64, f64, f64) {
    let (p_i, dp_i) = legendre_and_derivative(i, r);
    let (p_j, dp_j) = legendre_and_derivative(j, s);

    let value = p_i * p_j;
    let dr = dp_i * p_j;
    let ds = p_i * dp_j;

    (value, dr, ds)
}

/// Evaluate normalized polynomial and its gradient together efficiently.
///
/// Returns (φ_{ij}, ∂φ_{ij}/∂r, ∂φ_{ij}/∂s).
pub fn legendre_2d_normalized_with_gradient(i: usize, j: usize, r: f64, s: f64) -> (f64, f64, f64) {
    let norm = legendre_2d_norm(i, j);
    let (value, dr, ds) = legendre_2d_with_gradient(i, j, r, s);
    (norm * value, norm * dr, norm * ds)
}

/// Generate tensor-product GLL nodes for 2D quadrilateral elements.
///
/// Returns (order+1)² nodes as (r, s) pairs in lexicographic ordering:
/// node k = (r_i, s_j) where k = j * (order+1) + i
///
/// This ordering means r varies fastest (row-major if we think of s as rows).
///
/// # Arguments
/// * `order` - Polynomial order (N), resulting in (N+1)² nodes
pub fn tensor_product_gll_nodes(order: usize) -> Vec<(f64, f64)> {
    let nodes_1d = gauss_lobatto_nodes(order);
    let n = nodes_1d.len();
    let mut nodes_2d = Vec::with_capacity(n * n);

    // Lexicographic ordering: s varies slowest (outer), r varies fastest (inner)
    for &s in &nodes_1d {
        for &r in &nodes_1d {
            nodes_2d.push((r, s));
        }
    }

    nodes_2d
}

/// Generate tensor-product GLL weights for 2D quadrilateral elements.
///
/// Returns (order+1)² weights corresponding to tensor_product_gll_nodes.
/// Weight at node (r_i, s_j) = w_i * w_j (product of 1D weights).
///
/// # Arguments
/// * `order` - Polynomial order (N)
pub fn tensor_product_gll_weights(order: usize) -> Vec<f64> {
    let nodes_1d = gauss_lobatto_nodes(order);
    let weights_1d = gauss_lobatto_weights(order, &nodes_1d);
    let n = weights_1d.len();
    let mut weights_2d = Vec::with_capacity(n * n);

    // Same ordering as nodes: s varies slowest, r varies fastest
    for &w_s in &weights_1d {
        for &w_r in &weights_1d {
            weights_2d.push(w_r * w_s);
        }
    }

    weights_2d
}

/// Get the 1D node indices (i, j) from a 2D node index k.
///
/// Given k = j * n_1d + i, returns (i, j).
#[inline]
pub fn node_index_2d_to_1d(k: usize, n_1d: usize) -> (usize, usize) {
    let i = k % n_1d;
    let j = k / n_1d;
    (i, j)
}

/// Get the 2D node index k from 1D indices (i, j).
///
/// Returns k = j * n_1d + i.
#[inline]
pub fn node_index_1d_to_2d(i: usize, j: usize, n_1d: usize) -> usize {
    j * n_1d + i
}

/// Get the mode index from 2D mode degrees (i, j).
///
/// For tensor-product basis with modes ordered lexicographically,
/// mode (i, j) has index = j * n_1d + i.
#[inline]
pub fn mode_index(i: usize, j: usize, n_1d: usize) -> usize {
    j * n_1d + i
}

/// Get the mode degrees (i, j) from a mode index.
#[inline]
pub fn mode_degrees(mode: usize, n_1d: usize) -> (usize, usize) {
    let i = mode % n_1d;
    let j = mode / n_1d;
    (i, j)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legendre_2d_tensor_product() {
        // P_{ij}(r, s) = P_i(r) * P_j(s)
        let r = 0.5;
        let s = -0.3;

        for i in 0..=4 {
            for j in 0..=4 {
                let p_2d = legendre_2d(i, j, r, s);
                let p_expected = legendre(i, r) * legendre(j, s);
                assert!(
                    (p_2d - p_expected).abs() < 1e-14,
                    "P_{{{}{}}}({}, {}) mismatch",
                    i,
                    j,
                    r,
                    s
                );
            }
        }
    }

    #[test]
    fn test_legendre_2d_at_corners() {
        // At corners of reference element
        let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

        for (r, s) in corners {
            // P_00 = 1 everywhere
            assert!((legendre_2d(0, 0, r, s) - 1.0).abs() < 1e-14);

            // P_10 = P_1(r) * P_0(s) = r
            assert!((legendre_2d(1, 0, r, s) - r).abs() < 1e-14);

            // P_01 = P_0(r) * P_1(s) = s
            assert!((legendre_2d(0, 1, r, s) - s).abs() < 1e-14);

            // P_11 = P_1(r) * P_1(s) = r * s
            assert!((legendre_2d(1, 1, r, s) - r * s).abs() < 1e-14);
        }
    }

    #[test]
    fn test_legendre_2d_gradient() {
        let r = 0.4;
        let s = -0.6;

        // ∂P_{ij}/∂r = P'_i(r) * P_j(s)
        // ∂P_{ij}/∂s = P_i(r) * P'_j(s)
        for i in 0..=4 {
            for j in 0..=4 {
                let (dr, ds) = legendre_2d_gradient(i, j, r, s);

                let expected_dr = legendre_derivative(i, r) * legendre(j, s);
                let expected_ds = legendre(i, r) * legendre_derivative(j, s);

                assert!(
                    (dr - expected_dr).abs() < 1e-14,
                    "∂P_{{{}{}}} /∂r mismatch",
                    i,
                    j
                );
                assert!(
                    (ds - expected_ds).abs() < 1e-14,
                    "∂P_{{{}{}}} /∂s mismatch",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_legendre_2d_with_gradient_consistency() {
        let r = 0.3;
        let s = 0.7;

        for i in 0..=4 {
            for j in 0..=4 {
                let (val, dr, ds) = legendre_2d_with_gradient(i, j, r, s);

                assert!((val - legendre_2d(i, j, r, s)).abs() < 1e-14);

                let (expected_dr, expected_ds) = legendre_2d_gradient(i, j, r, s);
                assert!((dr - expected_dr).abs() < 1e-14);
                assert!((ds - expected_ds).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_normalized_orthonormality() {
        // Test that normalized basis is orthonormal under quadrature
        let order = 4;
        let nodes = tensor_product_gll_nodes(order);
        let weights = tensor_product_gll_weights(order);

        // For modes up to order 2 (to stay within quadrature exactness)
        for i1 in 0..=2 {
            for j1 in 0..=2 {
                for i2 in 0..=2 {
                    for j2 in 0..=2 {
                        // Compute ∫∫ φ_{i1,j1} φ_{i2,j2} dr ds
                        let mut integral = 0.0;
                        for ((r, s), &w) in nodes.iter().zip(weights.iter()) {
                            let phi1 = legendre_2d_normalized(i1, j1, *r, *s);
                            let phi2 = legendre_2d_normalized(i2, j2, *r, *s);
                            integral += w * phi1 * phi2;
                        }

                        let expected = if i1 == i2 && j1 == j2 { 1.0 } else { 0.0 };
                        assert!(
                            (integral - expected).abs() < 1e-12,
                            "Orthonormality failed for ({},{}) vs ({},{}): got {}, expected {}",
                            i1,
                            j1,
                            i2,
                            j2,
                            integral,
                            expected
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_tensor_product_nodes_count() {
        for order in 0..=5 {
            let nodes = tensor_product_gll_nodes(order);
            let expected = (order + 1) * (order + 1);
            assert_eq!(
                nodes.len(),
                expected,
                "Order {}: expected {} nodes, got {}",
                order,
                expected,
                nodes.len()
            );
        }
    }

    #[test]
    fn test_tensor_product_nodes_corners() {
        // Check that corners are included
        let order = 3;
        let nodes = tensor_product_gll_nodes(order);

        let has_corner = |cr: f64, cs: f64| {
            nodes
                .iter()
                .any(|(r, s)| (r - cr).abs() < 1e-14 && (s - cs).abs() < 1e-14)
        };

        assert!(has_corner(-1.0, -1.0), "Missing corner (-1, -1)");
        assert!(has_corner(1.0, -1.0), "Missing corner (1, -1)");
        assert!(has_corner(1.0, 1.0), "Missing corner (1, 1)");
        assert!(has_corner(-1.0, 1.0), "Missing corner (-1, 1)");
    }

    #[test]
    fn test_tensor_product_weights_sum() {
        // Weights should sum to 4 (area of [-1, 1]²)
        for order in 1..=5 {
            let weights = tensor_product_gll_weights(order);
            let sum: f64 = weights.iter().sum();
            assert!(
                (sum - 4.0).abs() < 1e-14,
                "Order {}: weights sum to {}, expected 4",
                order,
                sum
            );
        }
    }

    #[test]
    fn test_tensor_product_weights_are_products() {
        let order = 3;
        let nodes_1d = gauss_lobatto_nodes(order);
        let weights_1d = gauss_lobatto_weights(order, &nodes_1d);
        let weights_2d = tensor_product_gll_weights(order);
        let n = order + 1;

        for j in 0..n {
            for i in 0..n {
                let k = j * n + i;
                let expected = weights_1d[i] * weights_1d[j];
                assert!(
                    (weights_2d[k] - expected).abs() < 1e-14,
                    "Weight at ({}, {}) mismatch",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_quadrature_exactness_2d() {
        // 2D GLL quadrature with (N+1)² points is exact for polynomials
        // up to degree 2N-1 in each direction
        for order in 2..=4 {
            let nodes = tensor_product_gll_nodes(order);
            let weights = tensor_product_gll_weights(order);
            let max_degree = 2 * order - 1;

            for k_r in 0..=max_degree {
                for k_s in 0..=max_degree {
                    // ∫∫ r^{k_r} s^{k_s} dr ds
                    let exact_r = if k_r % 2 == 0 {
                        2.0 / (k_r + 1) as f64
                    } else {
                        0.0
                    };
                    let exact_s = if k_s % 2 == 0 {
                        2.0 / (k_s + 1) as f64
                    } else {
                        0.0
                    };
                    let exact = exact_r * exact_s;

                    let numerical: f64 = nodes
                        .iter()
                        .zip(weights.iter())
                        .map(|((r, s), &w)| w * r.powi(k_r as i32) * s.powi(k_s as i32))
                        .sum();

                    assert!(
                        (numerical - exact).abs() < 1e-12,
                        "Order {}, integrating r^{} s^{}: expected {}, got {}",
                        order,
                        k_r,
                        k_s,
                        exact,
                        numerical
                    );
                }
            }
        }
    }

    #[test]
    fn test_node_index_conversion() {
        let n_1d = 4;
        for j in 0..n_1d {
            for i in 0..n_1d {
                let k = node_index_1d_to_2d(i, j, n_1d);
                let (i_back, j_back) = node_index_2d_to_1d(k, n_1d);
                assert_eq!(i, i_back);
                assert_eq!(j, j_back);
            }
        }
    }

    #[test]
    fn test_mode_index_conversion() {
        let n_1d = 5;
        for j in 0..n_1d {
            for i in 0..n_1d {
                let mode = mode_index(i, j, n_1d);
                let (i_back, j_back) = mode_degrees(mode, n_1d);
                assert_eq!(i, i_back);
                assert_eq!(j, j_back);
            }
        }
    }

    #[test]
    fn test_node_ordering_is_lexicographic() {
        // Verify that the node ordering matches our indexing functions
        let order = 2;
        let nodes_1d = gauss_lobatto_nodes(order);
        let nodes_2d = tensor_product_gll_nodes(order);
        let n = order + 1;

        for k in 0..nodes_2d.len() {
            let (i, j) = node_index_2d_to_1d(k, n);
            let (r, s) = nodes_2d[k];

            assert!(
                (r - nodes_1d[i]).abs() < 1e-14,
                "r-coordinate mismatch at k={}: expected {}, got {}",
                k,
                nodes_1d[i],
                r
            );
            assert!(
                (s - nodes_1d[j]).abs() < 1e-14,
                "s-coordinate mismatch at k={}: expected {}, got {}",
                k,
                nodes_1d[j],
                s
            );
        }
    }
}
