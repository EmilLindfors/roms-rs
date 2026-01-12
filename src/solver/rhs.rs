//! Right-hand side computation for 1D DG advection.
//!
//! For the advection equation du/dt + a * du/dx = 0, the DG semi-discrete form is:
//!
//! du/dt = L(u) where L(u) = -a * Dr * u / J + LIFT * (flux_jump) / J
//!
//! Here:
//! - Dr is the differentiation matrix in reference coordinates
//! - J = dx/dr is the Jacobian (h/2 for uniform mesh)
//! - LIFT maps surface contributions to the volume
//! - flux_jump = F^* - F(u^-) at each face

use crate::mesh::{BoundaryFace, Mesh1D};
use crate::operators::DGOperators1D;
use crate::solver::DGSolution1D;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Boundary condition specification.
#[derive(Clone, Copy)]
pub struct BoundaryCondition {
    /// Value at left boundary (used for inflow if a > 0)
    pub left: f64,
    /// Value at right boundary (used for inflow if a < 0)
    pub right: f64,
}

impl Default for BoundaryCondition {
    fn default() -> Self {
        Self {
            left: 0.0,
            right: 0.0,
        }
    }
}

/// Compute the right-hand side of the DG discretization.
///
/// For du/dt + a * du/dx = 0:
/// RHS = -a * (dr/dx) * Dr * u + (dr/dx) * LIFT * (F^* - F_interior)
///
/// # Arguments
/// * `u` - Current solution
/// * `mesh` - 1D mesh
/// * `ops` - DG operators
/// * `a` - Advection velocity
/// * `bc` - Boundary conditions
pub fn compute_rhs(
    u: &DGSolution1D,
    mesh: &Mesh1D,
    ops: &DGOperators1D,
    a: f64,
    bc: &BoundaryCondition,
) -> DGSolution1D {
    let n = ops.n_nodes;
    let mut rhs = DGSolution1D::new(mesh.n_elements, n);

    for k in 0..mesh.n_elements {
        let u_k = u.element(k);
        let j_inv = mesh.jacobian_inv(k); // dr/dx = 2/h

        // 1. Volume term: -a * (dr/dx) * Dr * u_k
        // Following Hesthaven-Warburton: use Dr directly
        let mut du_dr = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                du_dr[i] += ops.dr[(i, j)] * u_k[j];
            }
        }

        // Scale by -a * j_inv to get -a * du/dx
        let rhs_k = rhs.element_mut(k);
        for i in 0..n {
            rhs_k[i] = -a * j_inv * du_dr[i];
        }

        // 2. Surface terms

        // Left face (normal = -1)
        let u_left_interior = u_k[0]; // Value at left boundary of this element
        let u_left_exterior = get_exterior_value(u, mesh, k, 0, bc, a);

        // Right face (normal = +1)
        let u_right_interior = u_k[n - 1]; // Value at right boundary of this element
        let u_right_exterior = get_exterior_value(u, mesh, k, 1, bc, a);

        // Surface flux: upwind dissipation term
        // Following Hesthaven-Warburton: flux = (a*nx - |a*nx|)/2 * (u^- - u^+)
        // For nx = -1, a > 0: flux = -a * (u_interior - u_exterior)
        // For nx = +1, a > 0: flux = 0 (outflow)

        let a_n_left = -a;
        let flux_left = if a_n_left < 0.0 {
            // Inflow
            a_n_left * (u_left_interior - u_left_exterior)
        } else {
            // Outflow
            0.0
        };

        let a_n_right = a;
        let flux_right = if a_n_right > 0.0 {
            // Outflow
            0.0
        } else {
            // Inflow
            a_n_right * (u_right_interior - u_right_exterior)
        };

        // Apply LIFT to surface flux
        for i in 0..n {
            let lift_contribution = ops.lift[(i, 0)] * flux_left + ops.lift[(i, 1)] * flux_right;
            rhs_k[i] += j_inv * lift_contribution;
        }
    }

    rhs
}

/// Get the exterior value at a face.
fn get_exterior_value(
    u: &DGSolution1D,
    mesh: &Mesh1D,
    element: usize,
    face: usize, // 0 = left, 1 = right
    bc: &BoundaryCondition,
    a: f64,
) -> f64 {
    let n = u.n_nodes;

    match mesh.is_boundary(element, face) {
        Some(BoundaryFace::Left) => {
            // Left boundary
            if a > 0.0 {
                // Inflow: use Dirichlet BC
                bc.left
            } else {
                // Outflow: extrapolate (use interior value)
                u.element(element)[0]
            }
        }
        Some(BoundaryFace::Right) => {
            // Right boundary
            if a < 0.0 {
                // Inflow: use Dirichlet BC
                bc.right
            } else {
                // Outflow: extrapolate (use interior value)
                u.element(element)[n - 1]
            }
        }
        None => {
            // Interior face
            match face {
                0 => {
                    // Left face: neighbor is on the left
                    let neighbor = mesh.neighbors[element].0.unwrap();
                    u.element(neighbor)[n - 1] // Right node of left neighbor
                }
                1 => {
                    // Right face: neighbor is on the right
                    let neighbor = mesh.neighbors[element].1.unwrap();
                    u.element(neighbor)[0] // Left node of right neighbor
                }
                _ => panic!("Invalid face index"),
            }
        }
    }
}

/// Compute the right-hand side in parallel using rayon.
///
/// This is the parallel version of `compute_rhs` that processes elements concurrently.
/// Enable with the `parallel` feature.
#[cfg(feature = "parallel")]
pub fn compute_rhs_parallel(
    u: &DGSolution1D,
    mesh: &Mesh1D,
    ops: &DGOperators1D,
    a: f64,
    bc: &BoundaryCondition,
) -> DGSolution1D {
    let n = ops.n_nodes;

    // Compute per-element RHS in parallel
    let rhs_data: Vec<f64> = (0..mesh.n_elements)
        .into_par_iter()
        .flat_map(|k| {
            let u_k = u.element(k);
            let j_inv = mesh.jacobian_inv(k);

            // 1. Volume term: -a * (dr/dx) * Dr * u_k
            let mut du_dr = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    du_dr[i] += ops.dr[(i, j)] * u_k[j];
                }
            }

            let mut rhs_k = vec![0.0; n];
            for i in 0..n {
                rhs_k[i] = -a * j_inv * du_dr[i];
            }

            // 2. Surface terms
            let u_left_interior = u_k[0];
            let u_left_exterior = get_exterior_value(u, mesh, k, 0, bc, a);
            let u_right_interior = u_k[n - 1];
            let u_right_exterior = get_exterior_value(u, mesh, k, 1, bc, a);

            let a_n_left = -a;
            let flux_left = if a_n_left < 0.0 {
                a_n_left * (u_left_interior - u_left_exterior)
            } else {
                0.0
            };

            let a_n_right = a;
            let flux_right = if a_n_right > 0.0 {
                0.0
            } else {
                a_n_right * (u_right_interior - u_right_exterior)
            };

            for i in 0..n {
                let lift_contribution =
                    ops.lift[(i, 0)] * flux_left + ops.lift[(i, 1)] * flux_right;
                rhs_k[i] += j_inv * lift_contribution;
            }

            rhs_k
        })
        .collect();

    DGSolution1D {
        data: rhs_data,
        n_elements: mesh.n_elements,
        n_nodes: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rhs_constant_solution() {
        // RHS of constant solution should be zero
        let mesh = Mesh1D::uniform(0.0, 1.0, 4);
        let ops = DGOperators1D::new(3);
        let mut u = DGSolution1D::new(mesh.n_elements, ops.n_nodes);

        // Set constant solution
        for k in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                u.element_mut(k)[i] = 1.0;
            }
        }

        let bc = BoundaryCondition {
            left: 1.0,
            right: 1.0,
        };
        let rhs = compute_rhs(&u, &mesh, &ops, 1.0, &bc);

        // RHS should be zero everywhere
        for k in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                assert!(
                    rhs.element(k)[i].abs() < 1e-12,
                    "RHS of constant should be 0, got {} at elem {}, node {}",
                    rhs.element(k)[i],
                    k,
                    i
                );
            }
        }
    }

    #[test]
    fn test_rhs_antisymmetry() {
        // Reversing velocity should reverse the sign of RHS (approximately)
        let mesh = Mesh1D::uniform(0.0, 1.0, 4);
        let ops = DGOperators1D::new(3);
        let mut u = DGSolution1D::new(mesh.n_elements, ops.n_nodes);

        // Set a non-constant solution
        u.set_from_function(&mesh, &ops, |x| x * x);

        let bc = BoundaryCondition::default();

        let rhs_pos = compute_rhs(&u, &mesh, &ops, 1.0, &bc);
        let rhs_neg = compute_rhs(&u, &mesh, &ops, -1.0, &bc);

        // Interior points should have opposite signs
        // (boundaries may differ due to BC treatment)
        for k in 1..mesh.n_elements - 1 {
            for i in 1..ops.n_nodes - 1 {
                let sum = rhs_pos.element(k)[i] + rhs_neg.element(k)[i];
                assert!(
                    sum.abs() < 1e-10,
                    "RHS should be antisymmetric in a, got {} at elem {}, node {}",
                    sum,
                    k,
                    i
                );
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_matches_serial() {
        use super::compute_rhs_parallel;

        let mesh = Mesh1D::uniform(0.0, 1.0, 20);
        let ops = DGOperators1D::new(4);
        let mut u = DGSolution1D::new(mesh.n_elements, ops.n_nodes);

        // Set a smooth initial condition
        u.set_from_function(&mesh, &ops, |x| (std::f64::consts::PI * x).sin());

        let bc = BoundaryCondition {
            left: 0.0,
            right: 0.0,
        };

        let rhs_serial = compute_rhs(&u, &mesh, &ops, 1.0, &bc);
        let rhs_parallel = compute_rhs_parallel(&u, &mesh, &ops, 1.0, &bc);

        // Results should be identical
        for k in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let diff = (rhs_serial.element(k)[i] - rhs_parallel.element(k)[i]).abs();
                assert!(
                    diff < 1e-14,
                    "Parallel should match serial: diff {} at elem {}, node {}",
                    diff,
                    k,
                    i
                );
            }
        }
    }
}
