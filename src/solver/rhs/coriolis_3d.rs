//! 3D Coriolis force.
//!
//! Applies Coriolis force to 3D velocity field.
//!
//! du/dt = f * v
//! dv/dt = -f * u
//!
//! where f is the Coriolis parameter.

use crate::mesh::Mesh2D;
use crate::operators::DGOperators2D;
use crate::solver::state::Solution3D;
use crate::source::CoriolisSource2D;
use crate::types::ElementIndex;

/// Apply Coriolis force to 3D velocity field.
///
/// Adds the Coriolis tendency to the explicit RHS.
///
/// # Arguments
/// * `rhs` - Right-hand side state (accumulates tendency).
/// * `state` - Current state (u, v).
/// * `mesh` - 2D mesh (for y-coordinates).
/// * `ops` - 2D operators (for nodes).
/// * `coriolis` - Coriolis parameter configuration.
pub fn apply_coriolis_3d(
    rhs: &mut Solution3D,
    state: &Solution3D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    coriolis: &CoriolisSource2D,
) {
    let n_nodes = ops.n_nodes;
    let n_levels = state.n_levels;

    // Iterate over elements
    for k in ElementIndex::iter(mesh.n_elements) {
        // Iterate over nodes
        for i in 0..n_nodes {
            // Get y-coordinate
            let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
            let [_x, y] = mesh.reference_to_physical(k, r, s);

            let f = coriolis.f_at(y);

            // Get columns (read-only from state)
            let u_col = state.u_column(k, i);
            let v_col = state.v_column(k, i);

            // Get columns (mutable from rhs)
            // We access fields directly to allow split mutable borrows
            let idx = (k.as_usize() * n_nodes + i) * n_levels;
            let rhs_u_col = &mut rhs.u[idx..idx + n_levels];
            let rhs_v_col = &mut rhs.v[idx..idx + n_levels];

            // Apply tendency: du = f*v, dv = -f*u
            for l in 0..n_levels {
                rhs_u_col[l] += f * v_col[l];
                rhs_v_col[l] -= f * u_col[l];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::operators::GeometricFactors2D;

    #[test]
    fn test_coriolis_3d_f_plane() {
        let n_elem = 1;
        let n_nodes = 1;
        let n_levels = 2;

        // Mock mesh/ops
        // We need a real mesh for physical coordinates
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        let ops = DGOperators2D::new(0); // 1 node (order 0)

        let mut state = Solution3D::new(n_elem, n_nodes, n_levels);
        let mut rhs = Solution3D::new(n_elem, n_nodes, n_levels);

        // Set u=10, v=0
        state.u.fill(10.0);
        state.v.fill(0.0);

        let f0 = 1.0e-4;
        let coriolis = CoriolisSource2D::f_plane(f0);

        apply_coriolis_3d(&mut rhs, &state, &mesh, &ops, &coriolis);

        // du/dt = f*v = 0
        // dv/dt = -f*u = -1e-4 * 10 = -1e-3

        assert!((rhs.u[0] - 0.0).abs() < 1e-10);
        assert!((rhs.v[0] - (-1.0e-3)).abs() < 1e-10);
        assert!((rhs.v[1] - (-1.0e-3)).abs() < 1e-10);
    }
}
