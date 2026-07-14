//! Conservative DG diffusion helpers for 2D affine quadrilateral elements.

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
        let rx = geom.rx[k_usize];
        let ry = geom.ry[k_usize];
        let sx = geom.sx[k_usize];
        let sy = geom.sy[k_usize];

        for i in 0..n_nodes {
            let mut du_dr = 0.0;
            let mut du_ds = 0.0;

            for j in 0..n_nodes {
                let u_j = values[idx(k, j, n_nodes)];
                du_dr += ops.dr[(i, j)] * u_j;
                du_ds += ops.ds[(i, j)] * u_j;
            }

            let grad_idx = idx(k, i, n_nodes);
            gradients[grad_idx].dx = du_dr * rx + du_ds * sx;
            gradients[grad_idx].dy = du_dr * ry + du_ds * sy;
        }

        for face in 0..4 {
            let normal = geom.normals[k_usize][face];
            let scale = geom.det_j_inv[k_usize] * geom.surface_j[k_usize][face];
            let face_nodes = &ops.face_nodes[face];

            for fi in 0..n_face_nodes {
                let node = face_nodes[fi];
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
    for i in 0..values.len() {
        let coeff = coefficients[i].max(0.0);
        flux_x[i] = coeff * gradients[i].dx;
        flux_y[i] = coeff * gradients[i].dy;
    }

    let mut rhs = vec![0.0; values.len()];

    for k in ElementIndex::iter(mesh.n_elements) {
        let k_usize = k.as_usize();
        let rx = geom.rx[k_usize];
        let ry = geom.ry[k_usize];
        let sx = geom.sx[k_usize];
        let sy = geom.sy[k_usize];

        for i in 0..n_nodes {
            let mut dfx_dr = 0.0;
            let mut dfx_ds = 0.0;
            let mut dfy_dr = 0.0;
            let mut dfy_ds = 0.0;

            for j in 0..n_nodes {
                let j_idx = idx(k, j, n_nodes);
                dfx_dr += ops.dr[(i, j)] * flux_x[j_idx];
                dfx_ds += ops.ds[(i, j)] * flux_x[j_idx];
                dfy_dr += ops.dr[(i, j)] * flux_y[j_idx];
                dfy_ds += ops.ds[(i, j)] * flux_y[j_idx];
            }

            rhs[idx(k, i, n_nodes)] = (rx * dfx_dr + sx * dfx_ds) + (ry * dfy_dr + sy * dfy_ds);
        }

        for face in 0..4 {
            let normal = geom.normals[k_usize][face];
            let s_jac = geom.surface_j[k_usize][face];
            let lift_scale = geom.det_j_inv[k_usize] * s_jac;
            let face_nodes = &ops.face_nodes[face];

            for fi in 0..n_face_nodes {
                let node = face_nodes[fi];
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
