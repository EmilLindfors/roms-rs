use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::mesh::traits::MeshTopology;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::rhs::advection_3d::compute_strong_divergence;
use crate::solver::state::Solution3D;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// Computes the vertical velocity omega in sigma coordinates.
///
/// The computation is based on the vertical integration of the continuity equation:
/// $\frac{\partial \eta}{\partial t} + \nabla \cdot (H \mathbf{u}) + \frac{\partial \Omega}{\partial s} = 0$
///
/// Integrating from bottom ($s=-1$) to a level $s$:
/// $\Omega(s) = - \int_{-1}^s \nabla \cdot (H \mathbf{u}) ds' - (s+1) \frac{\partial \eta}{\partial t}$
///
/// In practice, we compute the uncorrected omega by integrating divergence from the bottom up,
/// and then apply a linear correction to enforce $\Omega(0) = 0$.
///
/// $\Omega$ is the volume flux per unit horizontal area through a sigma surface (m/s).
/// It is stored at the **w-points** (layer interfaces, `sigma.sigma_w()`), so
/// `w_out` holds `n_levels + 1` values per column in the layout
/// `[element][node][interface]` (interface 0 = bed, `n_levels` = surface; both are
/// zero by the kinematic conditions). Keeping the interface values — rather than
/// averaging to layer centres — preserves discrete layer continuity
/// $\partial_t H_z + \nabla\cdot(H_z \mathbf{u}) + \Omega_{k+1/2} - \Omega_{k-1/2} = 0$,
/// which the vertical advection in [`crate::solver::rhs::apply_vertical_advection_3d`]
/// relies on.
pub fn compute_vertical_velocity(
    w_out: &mut [f64],
    state: &Solution3D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    sigma: &SigmaGrid,
    bathymetry: &Bathymetry2D,
    geom: &GeometricFactors2D,
    g: f64,
) {
    let n_elements = mesh.n_elements();
    let n_nodes = ops.n_nodes;
    let n_levels = sigma.n_levels();
    let d_sigma = sigma.d_sigma();
    let sigma_w = sigma.sigma_w();
    let n_faces = n_levels + 1;
    assert_eq!(
        w_out.len(),
        n_elements * n_nodes * n_faces,
        "omega is stored at w-points: n_elements * n_nodes * (n_levels + 1)"
    );

    // Borrow fields
    let u = &state.u;
    let v = &state.v;
    let eta = &state.eta.data;
    // w is w_out
    let w = w_out;

    // Temporary buffers per element
    let mut div_layer = vec![0.0; n_nodes * n_levels];
    let mut flux_x = vec![0.0; n_nodes];
    let mut flux_y = vec![0.0; n_nodes];
    let mut d_f_dr = vec![0.0; n_nodes];
    let mut d_f_ds = vec![0.0; n_nodes];

    // Iterate over elements
    for e in 0..n_elements {
        let el_idx = ElementIndex::new(e);
        let _elem_nodes = &mesh.elements[e];

        let bed_elem = bathymetry.element(el_idx);
        let eta_elem = &eta[e * n_nodes..(e + 1) * n_nodes];

        // 1. Calculate Local Divergence for each layer
        for k in 0..n_levels {
            // Calculate positive water depth D = eta - B at nodes.
            // And Flux = D * u, D * v.
            for i in 0..n_nodes {
                let h_val = eta_elem[i] - bed_elem[i];
                let idx_3d = (e * n_nodes + i) * n_levels + k;
                let u_val = u[idx_3d];
                let v_val = v[idx_3d];

                flux_x[i] = h_val * u_val;
                flux_y[i] = h_val * v_val;
            }

            // Compute volume divergence
            let rx = geom.rx[e];
            let ry = geom.ry[e];
            let sx = geom.sx[e];
            let sy = geom.sy[e];

            compute_strong_divergence(
                &flux_x,
                &flux_y,
                ops,
                rx,
                ry,
                sx,
                sy,
                &mut d_f_dr,
                &mut d_f_ds,
            );

            for i in 0..n_nodes {
                div_layer[k * n_nodes + i] = d_f_dr[i] + d_f_ds[i];
            }
        }

        // 2. Add Surface Terms (Lift * Jump)
        for face in 0..4 {
            let normal = geom.normals[e][face]; // (nx, ny)
            let s_jac = geom.surface_j[e][face];
            let j_inv = geom.det_j_inv[e]; // 2D Jacobian inverse
            let lift_scale = s_jac * j_inv;
            let face_nodes = &ops.face_nodes[face];
            let n_face_nodes = face_nodes.len();

            let neighbor_info = mesh.neighbor(el_idx, face);

            for k in 0..n_levels {
                // Interior values
                let mut h_int = vec![0.0; n_face_nodes];
                let mut u_int = vec![0.0; n_face_nodes];
                let mut v_int = vec![0.0; n_face_nodes];

                for i in 0..n_face_nodes {
                    let ni = face_nodes[i];
                    h_int[i] = eta_elem[ni] - bed_elem[ni];
                    let idx_3d = (e * n_nodes + ni) * n_levels + k;
                    u_int[i] = u[idx_3d];
                    v_int[i] = v[idx_3d];
                }

                // Exterior values
                let mut h_ext = vec![0.0; n_face_nodes];
                let mut u_ext = vec![0.0; n_face_nodes];
                let mut v_ext = vec![0.0; n_face_nodes];

                if let Some(nb) = neighbor_info {
                    let nb_idx = ElementIndex::new(nb.element);
                    let nb_face_nodes = &ops.face_nodes[nb.face];
                    let nb_e = nb.element;

                    let nb_bed = bathymetry.element(nb_idx);
                    let nb_eta = &eta[nb_e * n_nodes..(nb_e + 1) * n_nodes];

                    for i in 0..n_face_nodes {
                        // Reverse order for neighbor face
                        let ni = nb_face_nodes[n_face_nodes - 1 - i];
                        h_ext[i] = nb_eta[ni] - nb_bed[ni];
                        let idx_3d = (nb_e * n_nodes + ni) * n_levels + k;
                        u_ext[i] = u[idx_3d];
                        v_ext[i] = v[idx_3d];
                    }
                } else {
                    // Physical boundary: solid (land) wall. Mirror the normal
                    // velocity and keep the tangential component, so the exterior
                    // normal velocity is the negative of the interior one and the
                    // Rusanov mass flux through the wall is zero. Setting exterior =
                    // interior (transmissive) would leak mass through the coastline.
                    for i in 0..n_face_nodes {
                        h_ext[i] = h_int[i];
                        (u_ext[i], v_ext[i]) = crate::boundary::reflect_velocity(
                            u_int[i], v_int[i], normal.0, normal.1,
                        );
                    }
                }

                // Compute Flux Jump
                let mut jump = vec![0.0; n_face_nodes];

                for i in 0..n_face_nodes {
                    let un_int = u_int[i] * normal.0 + v_int[i] * normal.1;
                    let un_ext = u_ext[i] * normal.0 + v_ext[i] * normal.1;

                    // Rusanov flux for H
                    let flux_int = h_int[i] * un_int;
                    let flux_ext = h_ext[i] * un_ext;

                    // Wave speed c = sqrt(gH)
                    let c_int = (g * h_int[i]).sqrt();
                    let c_ext = (g * h_ext[i]).sqrt();
                    let speed = (un_int.abs() + c_int).max(un_ext.abs() + c_ext);

                    let star = 0.5 * (flux_int + flux_ext) - 0.5 * speed * (h_ext[i] - h_int[i]);

                    // Jump = F_int - F_star
                    // Remember: Div_strong = Div_vol - Lift * (F_int - F_star)
                    jump[i] = flux_int - star;
                }

                // Apply Lift and subtract from divergence
                for i in 0..n_nodes {
                    let mut lift = 0.0;
                    for fi in 0..n_face_nodes {
                        lift += ops.lift[face][(i, fi)] * jump[fi];
                    }
                    // Div_strong -= Lift * Jump
                    // So we subtract from div_layer
                    div_layer[k * n_nodes + i] -= lift_scale * lift;
                }
            }
        }

        // 3. Integrate vertically to find Omega at the interfaces:
        // Omega[0] = 0 (bottom), Omega[k+1] = Omega[k] - Div_k * d_sigma_k,
        // then apply the linear correction Omega(s) -= (s + 1) * Omega(surface)
        // so that Omega[n_levels] = 0 (sigma ranges from -1 to 0).
        for i in 0..n_nodes {
            let start = (e * n_nodes + i) * n_faces;
            let w_col = &mut w[start..start + n_faces];

            w_col[0] = 0.0;
            for k in 0..n_levels {
                w_col[k + 1] = w_col[k] - div_layer[k * n_nodes + i] * d_sigma[k];
            }

            let omega_surface = w_col[n_levels];
            for (w_face, &s_face) in w_col.iter_mut().zip(sigma_w) {
                *w_face -= (s_face + 1.0) * omega_surface;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Ω is stored at the w-points and satisfies discrete layer continuity.
    ///
    /// A zero-depth-mean shear flow `u_k = U_k sin(2πx)` over a flat bed has no
    /// net column divergence, so the surface correction vanishes and each layer
    /// increment `Ω_{k+1/2} − Ω_{k−1/2} = −Δσ_k ∇·(D u_k)` must scale with `U_k`.
    /// The previous layer-centre storage (averaged back to interfaces by the
    /// advection kernels) mixed neighbouring layers and broke this.
    #[test]
    fn omega_at_w_points_satisfies_layer_continuity() {
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 1);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh);
        let n_levels = 4;
        let sigma = SigmaGrid::uniform(n_levels);
        let n_nodes = ops.n_nodes;
        let bathymetry = Bathymetry2D::constant(mesh.n_elements, n_nodes, -10.0);

        let mut state = Solution3D::new(mesh.n_elements, n_nodes, n_levels);
        state.eta.fill(0.0);
        // U_k = σ_k + 1/2 has zero depth mean on the uniform grid.
        let shear: Vec<f64> = sigma.sigma_rho().iter().map(|&s| s + 0.5).collect();
        for k in 0..mesh.n_elements {
            let el = ElementIndex::new(k);
            for i in 0..n_nodes {
                let [x, _] = mesh.reference_to_physical(el, ops.nodes_r[i], ops.nodes_s[i]);
                for (u, &shear_l) in state.u_column_mut(el, i).iter_mut().zip(&shear) {
                    *u = shear_l * (2.0 * PI * x).sin();
                }
            }
        }

        let mut w = vec![0.0; mesh.n_elements * n_nodes * (n_levels + 1)];
        compute_vertical_velocity(
            &mut w,
            &state,
            &mesh,
            &ops,
            &sigma,
            &bathymetry,
            &geom,
            9.81,
        );

        let scale = w.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        assert!(scale > 1e-3, "test flow should drive a non-trivial Ω");
        let tol = 1e-12 * scale;
        let d_sigma = sigma.d_sigma();
        for col in w.chunks_exact(n_levels + 1) {
            assert!(col[0].abs() < tol, "Ω at bed = {}", col[0]);
            assert!(
                col[n_levels].abs() < tol,
                "Ω at surface = {}",
                col[n_levels]
            );
            // Per-unit-shear increment in the bottom layer fixes the column's divergence.
            let rate = (col[1] - col[0]) / (shear[0] * d_sigma[0]);
            for l in 0..n_levels {
                let increment = col[l + 1] - col[l];
                let expected = rate * shear[l] * d_sigma[l];
                assert!(
                    (increment - expected).abs() < tol,
                    "layer {l}: Ω increment {increment}, continuity requires {expected}"
                );
            }
        }
    }
}
