//! Conservative DG diffusion helpers for 2D quadrilateral elements.
//!
//! The gradient is taken with the chain rule at every node; the divergence of
//! the diffusive flux in conservative form, `J⁻¹[∂_r(J∇r·F) + ∂_s(J∇s·F)]`,
//! so that the element-integrated diffusion telescopes to the face fluxes on
//! any (also non-affine) element.

use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::types::ElementIndex;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ScalarGradient2D {
    pub dx: f64,
    pub dy: f64,
}

#[inline]
fn idx(k: ElementIndex, node: usize, n_nodes: usize) -> usize {
    k.as_usize() * n_nodes + node
}

/// Compute a BR1-style gradient with central face states.
///
/// The element-local derivative is corrected with
/// `LIFT * sJ/J * n * (u* - u-)`, so discontinuous values on adjacent
/// elements contribute to the gradient reconstruction.
pub(super) fn compute_br1_gradient_2d<F>(
    values: &[f64],
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    boundary_value: F,
) -> Vec<ScalarGradient2D>
where
    F: Fn(ElementIndex, usize, usize, usize, f64) -> f64,
{
    let n_nodes = ops.n_nodes;
    let n_face_nodes = ops.n_face_nodes;
    debug_assert_eq!(values.len(), mesh.n_elements * n_nodes);

    let mut gradients = vec![ScalarGradient2D::default(); values.len()];

    for k in ElementIndex::iter(mesh.n_elements) {
        let k_usize = k.as_usize();

        for i in 0..n_nodes {
            let mut du_dr = 0.0;
            let mut du_ds = 0.0;

            for j in 0..n_nodes {
                let u_j = values[idx(k, j, n_nodes)];
                du_dr += ops.dr[(i, j)] * u_j;
                du_ds += ops.ds[(i, j)] * u_j;
            }

            let grad_idx = idx(k, i, n_nodes);
            let (dx, dy) = geom.transform_derivatives(k_usize, i, du_dr, du_ds);
            gradients[grad_idx].dx = dx;
            gradients[grad_idx].dy = dy;
        }

        for face in 0..4 {
            let face_nodes = &ops.face_nodes[face];

            for fi in 0..n_face_nodes {
                let node = face_nodes[fi];
                let normal = geom.normal(k_usize, face, fi);
                let scale = geom.lift_scale(k_usize, face, fi, node);
                let value_int = values[idx(k, node, n_nodes)];
                let value_ext = if let Some(neighbor) = mesh.neighbor(k, face) {
                    let neighbor_k = ElementIndex::new(neighbor.element);
                    let neighbor_nodes = &ops.face_nodes[neighbor.face];
                    let neighbor_node = neighbor_nodes[n_face_nodes - 1 - fi];
                    values[idx(neighbor_k, neighbor_node, n_nodes)]
                } else {
                    boundary_value(k, face, fi, node, value_int)
                };

                let correction = 0.5 * (value_ext - value_int);
                for i in 0..n_nodes {
                    let lift = ops.lift[face][(i, fi)] * scale * correction;
                    let grad_idx = idx(k, i, n_nodes);
                    gradients[grad_idx].dx += lift * normal.0;
                    gradients[grad_idx].dy += lift * normal.1;
                }
            }
        }
    }

    gradients
}

/// Compute `div(coeff * grad(value))` with conservative central face fluxes.
///
/// The returned array is a scalar RHS contribution for every element node. The
/// interface flux is single-valued and opposite on neighboring elements, so the
/// element-integrated diffusion is conservative across interior faces.
pub(super) fn compute_br1_diffusion_rhs_2d(
    values: &[f64],
    coefficients: &[f64],
    gradients: &[ScalarGradient2D],
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
) -> Vec<f64> {
    let n_nodes = ops.n_nodes;
    let n_face_nodes = ops.n_face_nodes;
    debug_assert_eq!(values.len(), mesh.n_elements * n_nodes);
    debug_assert_eq!(coefficients.len(), values.len());
    debug_assert_eq!(gradients.len(), values.len());

    let mut flux_x = vec![0.0; values.len()];
    let mut flux_y = vec![0.0; values.len()];
    // Contravariant fluxes J∇r·F and J∇s·F
    let mut flux_r = vec![0.0; values.len()];
    let mut flux_s = vec![0.0; values.len()];
    for k in 0..mesh.n_elements {
        for i in 0..n_nodes {
            let n = geom.node_index(k, i);
            let coeff = coefficients[n].max(0.0);
            flux_x[n] = coeff * gradients[n].dx;
            flux_y[n] = coeff * gradients[n].dy;
            let ((ar_x, ar_y), (as_x, as_y)) = geom.contravariant(k, i);
            flux_r[n] = ar_x * flux_x[n] + ar_y * flux_y[n];
            flux_s[n] = as_x * flux_x[n] + as_y * flux_y[n];
        }
    }

    let mut rhs = vec![0.0; values.len()];

    for k in ElementIndex::iter(mesh.n_elements) {
        let k_usize = k.as_usize();

        for i in 0..n_nodes {
            let mut dfr_dr = 0.0;
            let mut dfs_ds = 0.0;

            for j in 0..n_nodes {
                let j_idx = idx(k, j, n_nodes);
                dfr_dr += ops.dr[(i, j)] * flux_r[j_idx];
                dfs_ds += ops.ds[(i, j)] * flux_s[j_idx];
            }

            rhs[idx(k, i, n_nodes)] = geom.jacobian_inv(k_usize, i) * (dfr_dr + dfs_ds);
        }

        for face in 0..4 {
            let face_nodes = &ops.face_nodes[face];

            for fi in 0..n_face_nodes {
                let node = face_nodes[fi];
                let normal = geom.normal(k_usize, face, fi);
                let lift_scale = geom.lift_scale(k_usize, face, fi, node);
                let int_idx = idx(k, node, n_nodes);
                let flux_int_n = flux_x[int_idx] * normal.0 + flux_y[int_idx] * normal.1;

                let flux_ext_n = if let Some(neighbor) = mesh.neighbor(k, face) {
                    let neighbor_k = ElementIndex::new(neighbor.element);
                    let neighbor_nodes = &ops.face_nodes[neighbor.face];
                    let neighbor_node = neighbor_nodes[n_face_nodes - 1 - fi];
                    let ext_idx = idx(neighbor_k, neighbor_node, n_nodes);
                    flux_x[ext_idx] * normal.0 + flux_y[ext_idx] * normal.1
                } else {
                    flux_int_n
                };

                let flux_star_n = 0.5 * (flux_int_n + flux_ext_n);
                let correction = flux_star_n - flux_int_n;

                for i in 0..n_nodes {
                    rhs[idx(k, i, n_nodes)] += ops.lift[face][(i, fi)] * lift_scale * correction;
                }
            }
        }
    }

    rhs
}
