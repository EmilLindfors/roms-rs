//! Conservative DG diffusion helpers for 2D quadrilateral elements.
//!
//! The gradient is taken with the chain rule at every node; the divergence of
//! the diffusive flux in conservative form, `J⁻¹[∂_r(J∇r·F) + ∂_s(J∇s·F)]`,
//! so that the element-integrated diffusion telescopes to the face fluxes on
//! any (also non-affine) element.
//!
//! Both steps are written per element ([`br1_gradient_element`],
//! [`br1_diffusion_element`]) and read the other elements through closures,
//! so kernels can run them in their own element loops (in parallel, or on a
//! subset of the elements for local time stepping) without whole-mesh
//! temporaries. [`compute_br1_gradient_2d`] and
//! [`compute_br1_diffusion_rhs_2d`] run them over the whole mesh.
//!
//! The GLL LIFT of a face is nonzero only in the rows of the face's own
//! nodes (`LIFT_f[i, fi] = δ(i, node(fi))/w_i`: diagonal mass, and the face
//! interpolation just picks the node), so a face correction touches only its
//! own nodes; the kernels apply just those entries.

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

/// BR1 gradient of element `k` into `out` (its `n_nodes` values).
///
/// The element-local derivative is corrected with
/// `LIFT * sJ/J * n * (u* - u-)`, u* the mean of the two face values, so
/// discontinuous values on adjacent elements contribute to the gradient
/// reconstruction. `value(j, node)` is the scalar at a node of element `k` or
/// of a face neighbour; `boundary_value(k, face, fi, node, interior)` the
/// exterior value on a boundary face. `own` is scratch for the element's
/// `n_nodes` values.
#[allow(clippy::too_many_arguments)]
pub(super) fn br1_gradient_element(
    k: ElementIndex,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    value: impl Fn(ElementIndex, usize) -> f64,
    boundary_value: impl Fn(ElementIndex, usize, usize, usize, f64) -> f64,
    own: &mut [f64],
    out: &mut [ScalarGradient2D],
) {
    let n_face_nodes = ops.n_face_nodes;
    let k_usize = k.as_usize();
    debug_assert_eq!(own.len(), ops.n_nodes);
    debug_assert_eq!(out.len(), ops.n_nodes);

    for (j, u) in own.iter_mut().enumerate() {
        *u = value(k, j);
    }

    let n_nodes = ops.n_nodes;
    for (i, gradient) in out.iter_mut().enumerate() {
        let mut du_dr = 0.0;
        let mut du_ds = 0.0;
        let dr = &ops.dr_row_major[i * n_nodes..][..n_nodes];
        let ds = &ops.ds_row_major[i * n_nodes..][..n_nodes];

        for ((&u_j, &dr), &ds) in own.iter().zip(dr).zip(ds) {
            du_dr += dr * u_j;
            du_ds += ds * u_j;
        }

        let (dx, dy) = geom.transform_derivatives(k_usize, i, du_dr, du_ds);
        gradient.dx = dx;
        gradient.dy = dy;
    }

    for face in 0..4 {
        let face_nodes = &ops.face_nodes[face];

        for fi in 0..n_face_nodes {
            let node = face_nodes[fi];
            let normal = geom.normal(k_usize, face, fi);
            let scale = geom.lift_scale(k_usize, face, fi, node);
            let value_int = own[node];
            let value_ext = if let Some(neighbor) = mesh.neighbor(k, face) {
                let neighbor_nodes = &ops.face_nodes[neighbor.face];
                let neighbor_node = neighbor_nodes[n_face_nodes - 1 - fi];
                value(ElementIndex::new(neighbor.element), neighbor_node)
            } else {
                boundary_value(k, face, fi, node, value_int)
            };

            let correction = 0.5 * (value_ext - value_int);
            let lift = ops.lift_row_major[face][node * n_face_nodes + fi] * scale * correction;
            out[node].dx += lift * normal.0;
            out[node].dy += lift * normal.1;
        }
    }
}

/// Scratch of [`br1_diffusion_element`]: four arrays of `n_nodes`.
pub(super) struct DiffusionScratch {
    flux_x: Vec<f64>,
    flux_y: Vec<f64>,
    flux_r: Vec<f64>,
    flux_s: Vec<f64>,
}

impl DiffusionScratch {
    pub(super) fn new(n_nodes: usize) -> Self {
        Self {
            flux_x: vec![0.0; n_nodes],
            flux_y: vec![0.0; n_nodes],
            flux_r: vec![0.0; n_nodes],
            flux_s: vec![0.0; n_nodes],
        }
    }
}

/// `div(F)` on element `k` into `out` (overwritten), for a diffusive flux
/// `F = flux(j, node)` (e.g. `coeff · grad(value)`) at the nodes of `k` and
/// of its face neighbours, with the central face flux `(F⁻ + F⁺)/2 · n`.
///
/// The interface flux is single-valued and opposite on neighbouring
/// elements, so the element-integrated diffusion is conservative across
/// interior faces. Boundary faces take the interior flux (no correction).
pub(super) fn br1_diffusion_element(
    k: ElementIndex,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    flux: impl Fn(ElementIndex, usize) -> (f64, f64),
    scratch: &mut DiffusionScratch,
    out: &mut [f64],
) {
    let n_face_nodes = ops.n_face_nodes;
    let k_usize = k.as_usize();
    debug_assert_eq!(out.len(), ops.n_nodes);
    let DiffusionScratch {
        flux_x,
        flux_y,
        flux_r,
        flux_s,
    } = scratch;

    // Contravariant fluxes J∇r·F and J∇s·F
    for i in 0..ops.n_nodes {
        let (fx, fy) = flux(k, i);
        flux_x[i] = fx;
        flux_y[i] = fy;
        let ((ar_x, ar_y), (as_x, as_y)) = geom.contravariant(k_usize, i);
        flux_r[i] = ar_x * fx + ar_y * fy;
        flux_s[i] = as_x * fx + as_y * fy;
    }

    let n_nodes = ops.n_nodes;
    for (i, rhs) in out.iter_mut().enumerate() {
        let mut dfr_dr = 0.0;
        let mut dfs_ds = 0.0;
        let dr = &ops.dr_row_major[i * n_nodes..][..n_nodes];
        let ds = &ops.ds_row_major[i * n_nodes..][..n_nodes];

        for (((&fr, &fs), &dr), &ds) in flux_r.iter().zip(flux_s.iter()).zip(dr).zip(ds) {
            dfr_dr += dr * fr;
            dfs_ds += ds * fs;
        }

        *rhs = geom.jacobian_inv(k_usize, i) * (dfr_dr + dfs_ds);
    }

    for face in 0..4 {
        let face_nodes = &ops.face_nodes[face];

        for fi in 0..n_face_nodes {
            let node = face_nodes[fi];
            let normal = geom.normal(k_usize, face, fi);
            let lift_scale = geom.lift_scale(k_usize, face, fi, node);
            let flux_int_n = flux_x[node] * normal.0 + flux_y[node] * normal.1;

            let flux_ext_n = if let Some(neighbor) = mesh.neighbor(k, face) {
                let neighbor_nodes = &ops.face_nodes[neighbor.face];
                let neighbor_node = neighbor_nodes[n_face_nodes - 1 - fi];
                let (fx, fy) = flux(ElementIndex::new(neighbor.element), neighbor_node);
                fx * normal.0 + fy * normal.1
            } else {
                flux_int_n
            };

            let flux_star_n = 0.5 * (flux_int_n + flux_ext_n);
            let correction = flux_star_n - flux_int_n;

            out[node] +=
                ops.lift_row_major[face][node * n_face_nodes + fi] * lift_scale * correction;
        }
    }
}

/// [`br1_gradient_element`] of every element, for `values` given at every
/// mesh node.
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
    debug_assert_eq!(values.len(), mesh.n_elements * n_nodes);

    let mut gradients = vec![ScalarGradient2D::default(); values.len()];
    let mut own = vec![0.0; n_nodes];
    for (k, out) in ElementIndex::iter(mesh.n_elements).zip(gradients.chunks_exact_mut(n_nodes)) {
        br1_gradient_element(
            k,
            mesh,
            ops,
            geom,
            |j, node| values[idx(j, node, n_nodes)],
            &boundary_value,
            &mut own,
            out,
        );
    }
    gradients
}

/// Compute `div(coeff * grad(value))` with conservative central face fluxes:
/// [`br1_diffusion_element`] of every element, negative coefficients taken
/// as zero.
///
/// The returned array is a scalar RHS contribution for every element node.
pub(super) fn compute_br1_diffusion_rhs_2d(
    values: &[f64],
    coefficients: &[f64],
    gradients: &[ScalarGradient2D],
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
) -> Vec<f64> {
    let n_nodes = ops.n_nodes;
    debug_assert_eq!(values.len(), mesh.n_elements * n_nodes);
    debug_assert_eq!(coefficients.len(), values.len());
    debug_assert_eq!(gradients.len(), values.len());

    let flux = |j: ElementIndex, node: usize| {
        let n = idx(j, node, n_nodes);
        let coeff = coefficients[n].max(0.0);
        (coeff * gradients[n].dx, coeff * gradients[n].dy)
    };
    let mut rhs = vec![0.0; values.len()];
    let mut scratch = DiffusionScratch::new(n_nodes);
    for (k, out) in ElementIndex::iter(mesh.n_elements).zip(rhs.chunks_exact_mut(n_nodes)) {
        br1_diffusion_element(k, mesh, ops, geom, flux, &mut scratch, out);
    }
    rhs
}

#[cfg(test)]
mod tests {
    use crate::operators::DGOperators2D;

    /// The kernels apply only the LIFT entry of each face node's own row:
    /// exact only if every other entry is zero.
    #[test]
    fn lift_touches_only_the_face_nodes() {
        for order in 1..=5 {
            let ops = DGOperators2D::new(order);
            let nf = ops.n_face_nodes;
            for face in 0..4 {
                for fi in 0..nf {
                    let node = ops.face_nodes[face][fi];
                    for i in 0..ops.n_nodes {
                        let entry = ops.lift_row_major[face][i * nf + fi];
                        assert_eq!(
                            entry == 0.0,
                            i != node,
                            "P{order}, face {face}, ({i}, {fi})"
                        );
                    }
                }
            }
        }
    }
}
