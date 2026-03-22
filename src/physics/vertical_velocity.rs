use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::state::Solution3D;
use crate::vertical::SigmaGrid;
use crate::solver::rhs::advection_3d::compute_strong_divergence;
use crate::types::ElementIndex;
use crate::mesh::traits::MeshTopology;
use crate::mesh::data::Bathymetry2D;

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
        
        let h_elem = bathymetry.element(el_idx);
        let eta_elem = &eta[e * n_nodes..(e + 1) * n_nodes];

        // 1. Calculate Local Divergence for each layer
        for k in 0..n_levels {
            // Calculate H = h + eta at nodes
            // And Flux = H * u, H * v
            for i in 0..n_nodes {
                let h_val = h_elem[i] + eta_elem[i];
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
                &flux_x, &flux_y,
                ops, rx, ry, sx, sy,
                &mut d_f_dr, &mut d_f_ds
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
                    h_int[i] = h_elem[ni] + eta_elem[ni];
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
                    
                    let nb_h = bathymetry.element(nb_idx);
                    let nb_eta = &eta[nb_e * n_nodes..(nb_e + 1) * n_nodes];

                    for i in 0..n_face_nodes {
                        // Reverse order for neighbor face
                        let ni = nb_face_nodes[n_face_nodes - 1 - i];
                        h_ext[i] = nb_h[ni] + nb_eta[ni];
                        let idx_3d = (nb_e * n_nodes + ni) * n_levels + k;
                        u_ext[i] = u[idx_3d];
                        v_ext[i] = v[idx_3d];
                    }
                } else {
                    // Boundary condition
                    for i in 0..n_face_nodes {
                        let un = u_int[i] * normal.0 + v_int[i] * normal.1;
                        if un > 0.0 {
                            // Outflow
                            h_ext[i] = h_int[i];
                            u_ext[i] = u_int[i];
                            v_ext[i] = v_int[i];
                        } else {
                            // Inflow / Wall
                            h_ext[i] = h_int[i];
                            u_ext[i] = u_int[i];
                            v_ext[i] = v_int[i];
                        }
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

        // 3. Integrate vertically to find Omega
        // Omega_face has N+1 levels (interfaces)
        // Omega_face[0] = 0 (bottom)
        // Omega_face[k+1] = Omega_face[k] - Div_k * d_sigma_k
        
        let mut omega_face = vec![0.0; n_levels + 1];
        
        for i in 0..n_nodes {
            omega_face[0] = 0.0;
            
            // Integrate up
            for k in 0..n_levels {
                let div = div_layer[k * n_nodes + i];
                omega_face[k + 1] = omega_face[k] - div * d_sigma[k];
            }
            
            // Apply linear correction to enforce Omega_face[N] = 0
            // Omega_corr(s) = Omega(s) - (s + 1) * Omega(N)
            // (Assuming sigma ranges from -1 to 0)
            let omega_surface = omega_face[n_levels];
            
            for k in 0..n_levels {
                // Calculate corrected Omega at faces k and k+1
                // We need sigma_w[k]
                let s_k = sigma_w[k];
                let s_k1 = sigma_w[k + 1];
                
                let w_face_k = omega_face[k] - (s_k + 1.0) * omega_surface;
                let w_face_k1 = omega_face[k + 1] - (s_k1 + 1.0) * omega_surface;
                
                // Interpolate to center (layer k)
                let w_center = 0.5 * (w_face_k + w_face_k1);
                
                // Store in w (w_out)
                let idx_3d = (e * n_nodes + i) * n_levels + k;
                w[idx_3d] = w_center;
            }
        }
    }
}
