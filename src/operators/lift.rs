//! LIFT matrix for surface integral contributions in DG.
//!
//! The LIFT matrix maps surface values to volume contributions:
//! LIFT = M^{-1} * E^T
//!
//! where E is the extraction matrix that evaluates polynomials at face nodes.
//!
//! In 1D, each element has 2 faces (left at r=-1, right at r=+1).
//! The LIFT matrix has shape (n_nodes, 2) and maps [flux_left, flux_right]
//! to volume contributions.

use faer::Mat;

/// Compute the LIFT matrix for 1D elements.
///
/// LIFT has shape (n_nodes, 2) where column 0 is for the left face (r=-1)
/// and column 1 is for the right face (r=+1).
///
/// The result is LIFT = M^{-1} * E^T where E[0, :] evaluates at r=-1
/// and E[1, :] evaluates at r=+1.
///
/// For Lagrange basis at GLL nodes, E is very simple:
/// - E[0, 0] = 1, E[0, j] = 0 for j > 0  (left face is node 0)
/// - E[1, n-1] = 1, E[1, j] = 0 for j < n-1  (right face is node n-1)
///
/// So E^T = [[1, 0], [0, 0], ..., [0, 0], [0, 1]]
/// And LIFT = M^{-1} * E^T is just:
/// - LIFT[0, 0] = M^{-1}[0, 0]
/// - LIFT[n-1, 1] = M^{-1}[n-1, n-1]
/// - LIFT[i, :] = 0 for 0 < i < n-1
pub fn lift_matrix(mass_inv: &Mat<f64>) -> Mat<f64> {
    let n = mass_inv.nrows();
    let mut lift = Mat::zeros(n, 2);

    // Left face (r = -1) is at node 0
    // LIFT[:, 0] = M^{-1} * e_0 where e_0 = [1, 0, ..., 0]^T
    // For diagonal M^{-1}, this is just the first column of M^{-1}
    for i in 0..n {
        lift[(i, 0)] = mass_inv[(i, 0)];
    }

    // Right face (r = +1) is at node n-1
    // LIFT[:, 1] = M^{-1} * e_{n-1}
    for i in 0..n {
        lift[(i, 1)] = mass_inv[(i, n - 1)];
    }

    lift
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::mass::mass_matrix_inv;
    use crate::polynomial::{gauss_lobatto_nodes, gauss_lobatto_weights};

    #[test]
    fn test_lift_shape() {
        for order in 1..=5 {
            let nodes = gauss_lobatto_nodes(order);
            let weights = gauss_lobatto_weights(order, &nodes);
            let m_inv = mass_matrix_inv(&weights);
            let lift = lift_matrix(&m_inv);

            assert_eq!(lift.nrows(), order + 1);
            assert_eq!(lift.ncols(), 2);
        }
    }

    #[test]
    fn test_lift_structure() {
        // For diagonal mass matrix, LIFT should only have non-zero entries
        // at the boundary nodes
        for order in 2..=5 {
            let nodes = gauss_lobatto_nodes(order);
            let weights = gauss_lobatto_weights(order, &nodes);
            let m_inv = mass_matrix_inv(&weights);
            let lift = lift_matrix(&m_inv);

            let n = order + 1;

            // Left face: only node 0 should have non-zero entry
            assert!(lift[(0, 0)].abs() > 1e-14, "LIFT[0,0] should be non-zero");
            for i in 1..n {
                assert!(lift[(i, 0)].abs() < 1e-14, "LIFT[{},0] should be zero", i);
            }

            // Right face: only node n-1 should have non-zero entry
            assert!(
                lift[(n - 1, 1)].abs() > 1e-14,
                "LIFT[n-1,1] should be non-zero"
            );
            for i in 0..n - 1 {
                assert!(lift[(i, 1)].abs() < 1e-14, "LIFT[{},1] should be zero", i);
            }
        }
    }

    #[test]
    fn test_lift_values() {
        // LIFT[0,0] = 1/w_0 and LIFT[n-1,1] = 1/w_{n-1}
        for order in 1..=5 {
            let nodes = gauss_lobatto_nodes(order);
            let weights = gauss_lobatto_weights(order, &nodes);
            let m_inv = mass_matrix_inv(&weights);
            let lift = lift_matrix(&m_inv);

            let n = order + 1;

            assert!(
                (lift[(0, 0)] - 1.0 / weights[0]).abs() < 1e-14,
                "LIFT[0,0] should be 1/w_0"
            );
            assert!(
                (lift[(n - 1, 1)] - 1.0 / weights[n - 1]).abs() < 1e-14,
                "LIFT[n-1,1] should be 1/w_{{n-1}}"
            );
        }
    }
}
