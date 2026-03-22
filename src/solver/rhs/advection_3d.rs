//! 3D Advection terms.
//!
//! Computes the advection terms for the 3D momentum and tracer equations.
//!
//! Uses a conservative flux formulation:
//! $\text{Adv}(\phi) = \nabla_H \cdot (\mathbf{u}_H \phi) + \frac{\partial (\Omega \phi)}{\partial s}$
//!
//! where $\nabla_H$ is the horizontal gradient along sigma surfaces.

use crate::solver::state::Solution3D;
use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// Apply horizontal advection to 3D fields.
///
/// Computes $-\nabla_H \cdot (\mathbf{u}_H \phi)$ for $\phi = u, v$.
/// (Tracers not yet connected, but same logic applies).
///
/// This implementation uses a strong-form DG kernel layer-by-layer.
pub fn apply_horizontal_advection_3d(
    rhs: &mut Solution3D,
    state: &Solution3D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
) {
    let n_levels = state.n_levels;
    let n_nodes = ops.n_nodes;
    
    // Per-element workspace
    let mut flux_x = vec![0.0; n_nodes];
    let mut flux_y = vec![0.0; n_nodes];
    let mut d_f_dr = vec![0.0; n_nodes];
    let mut d_f_ds = vec![0.0; n_nodes];
    
    // U/V values buffers
    let mut u_nodes = vec![0.0; n_nodes];
    let mut v_nodes = vec![0.0; n_nodes];
    
    for k in 0..mesh.n_elements {
        let el_idx = ElementIndex::new(k);
        
        // Geometric factors
        let rx = geom.rx[k];
        let ry = geom.ry[k];
        let sx = geom.sx[k];
        let sy = geom.sy[k];
        
        for l in 0..n_levels {
            // --- Gather level values ---
            for i in 0..n_nodes {
                 u_nodes[i] = state.u_column(el_idx, i)[l];
                 v_nodes[i] = state.v_column(el_idx, i)[l];
            }
            
            // --- 1. Momentum Advection for U: div(u*u, v*u) ---
            
            // Fluxes
            for i in 0..n_nodes {
                flux_x[i] = u_nodes[i] * u_nodes[i]; // u * u
                flux_y[i] = v_nodes[i] * u_nodes[i]; // v * u
            }
            
            // Volume Divergence
            compute_strong_divergence(
                &flux_x, &flux_y, 
                ops, rx, ry, sx, sy, 
                &mut d_f_dr, &mut d_f_ds
            );
            
            // Add volume term to RHS (sign: -div)
            for i in 0..n_nodes {
                let div = d_f_dr[i] + d_f_ds[i];
                rhs.u_column_mut(el_idx, i)[l] -= div; 
            }
            
            // --- 2. Momentum Advection for V: div(u*v, v*v) ---
            
            // Fluxes
            for i in 0..n_nodes {
                flux_x[i] = u_nodes[i] * v_nodes[i]; // u * v
                flux_y[i] = v_nodes[i] * v_nodes[i]; // v * v
            }
            
            compute_strong_divergence(
                &flux_x, &flux_y, 
                ops, rx, ry, sx, sy, 
                &mut d_f_dr, &mut d_f_ds
            );
            
            for i in 0..n_nodes {
                let div = d_f_dr[i] + d_f_ds[i];
                rhs.v_column_mut(el_idx, i)[l] -= div;
            }
        }
        
        // --- Surface Terms ---
        // We apply surface terms separately as they involve neighbor lookups
        apply_horizontal_surface_terms(rhs, state, mesh, ops, geom, k);
    }
}

/// Compute volume divergence term assuming constant geometric factors.
///
/// Output:
/// out_dr = Dr * (rx*Fx + ry*Fy)
/// out_ds = Ds * (sx*Fx + sy*Fy)
///
/// So Div = out_dr + out_ds
pub fn compute_strong_divergence(
    fx: &[f64], fy: &[f64],
    ops: &DGOperators2D,
    rx: f64, ry: f64, sx: f64, sy: f64,
    out_dr: &mut [f64], out_ds: &mut [f64] 
) {
    let n = ops.n_nodes;
    
    // Temp buffers for transformed flux components
    // Fr = rx*Fx + ry*Fy
    // Fs = sx*Fx + sy*Fy
    let mut fr = vec![0.0; n];
    let mut fs = vec![0.0; n];
    
    for i in 0..n {
        fr[i] = rx * fx[i] + ry * fy[i];
        fs[i] = sx * fx[i] + sy * fy[i];
    }
    
    // Matmul: out_dr = Dr * fr
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += ops.dr[(i, j)] * fr[j];
        }
        out_dr[i] = sum;
    }
    
    // Matmul: out_ds = Ds * fs
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += ops.ds[(i, j)] * fs[j];
        }
        out_ds[i] = sum;
    }
}

fn apply_horizontal_surface_terms(
    rhs: &mut Solution3D,
    state: &Solution3D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    k: usize,
) {
    let el_idx = ElementIndex::new(k);
    let n_nodes = ops.n_nodes;
    let n_levels = state.n_levels;
    let n_face_nodes = ops.n_face_nodes;
    
    // For each face
    for face in 0..4 {
        let normal = geom.normals[k][face];
        let s_jac = geom.surface_j[k][face];
        let j_inv = geom.det_j_inv[k];
        let lift_scale = s_jac * j_inv;
        
        let face_nodes = &ops.face_nodes[face];
        
        // Identify neighbor
        let neighbor_info = mesh.neighbor(el_idx, face);
        
        // Loop over levels
        for l in 0..n_levels {
            // Gather interior values
            let mut u_int = vec![0.0; n_face_nodes];
            let mut v_int = vec![0.0; n_face_nodes];
            
            for i in 0..n_face_nodes {
                let ni = face_nodes[i];
                u_int[i] = state.u_column(el_idx, ni)[l];
                v_int[i] = state.v_column(el_idx, ni)[l];
            }
            
            // Gather exterior values
            let mut u_ext = vec![0.0; n_face_nodes];
            let mut v_ext = vec![0.0; n_face_nodes];
            
            if let Some(nb) = neighbor_info {
                 let nb_idx = ElementIndex::new(nb.element);
                 let nb_face_nodes = &ops.face_nodes[nb.face];
                 
                 // Neighbor orientation reversal
                 for i in 0..n_face_nodes {
                     let ni = nb_face_nodes[n_face_nodes - 1 - i];
                     u_ext[i] = state.u_column(nb_idx, ni)[l];
                     v_ext[i] = state.v_column(nb_idx, ni)[l];
                 }
            } else {
                // Boundary condition: Free slip / Outflow (copy interior)
                for i in 0..n_face_nodes {
                    u_ext[i] = u_int[i];
                    v_ext[i] = v_int[i];
                }
            }
            
            // Compute Flux Jump for U and V equations
            let mut jump_u = vec![0.0; n_face_nodes];
            let mut jump_v = vec![0.0; n_face_nodes];
            
            for i in 0..n_face_nodes {
                // Rusanov Flux
                let un_int = u_int[i] * normal.0 + v_int[i] * normal.1;
                let un_ext = u_ext[i] * normal.0 + v_ext[i] * normal.1;
                let speed = un_int.abs().max(un_ext.abs()); // Approximate wave speed
                
                // Flux F(u) = u*u * nx + u*v * ny = u * (u.n)
                // Flux G(v) = v*u * nx + v*v * ny = v * (u.n)
                
                let flux_u_int = u_int[i] * un_int;
                let flux_u_ext = u_ext[i] * un_ext;
                let star_u = 0.5 * (flux_u_int + flux_u_ext) - 0.5 * speed * (u_ext[i] - u_int[i]);
                
                let flux_v_int = v_int[i] * un_int;
                let flux_v_ext = v_ext[i] * un_ext;
                let star_v = 0.5 * (flux_v_int + flux_v_ext) - 0.5 * speed * (v_ext[i] - v_int[i]);
                
                // Jump = F_int - F_star
                // Why?
                // Volume term was -Div F.
                // Integ by parts -> Surface term + F_int.
                // So Integral = - (Div F, v) = (F, grad v) - <F.n, v>
                // We replaced flux with F*.
                // So we have <(F.n - F*), v>.
                // So Lift should apply to (F.n - F*).
                
                // My F_int above is u_int * un_int = F(u_int) . n
                
                jump_u[i] = flux_u_int - star_u;
                jump_v[i] = flux_v_int - star_v;
            }
            
            // Apply Lift
            for i in 0..n_nodes {
                 let mut lift_u = 0.0;
                 let mut lift_v = 0.0;
                 for fi in 0..n_face_nodes {
                     let l = ops.lift[face][(i, fi)];
                     lift_u += l * jump_u[fi];
                     lift_v += l * jump_v[fi];
                 }
                 // Update RHS
                 rhs.u_column_mut(el_idx, i)[l] += lift_scale * lift_u;
                 rhs.v_column_mut(el_idx, i)[l] += lift_scale * lift_v;
            }
        }
    }
}

/// Apply horizontal advection to a tracer field.
///
/// Computes $-\nabla_H \cdot (\mathbf{u}_H \phi)$.
///
/// # Arguments
/// * `rhs_tracer`: Output RHS for tracer
/// * `tracer`: Input tracer field (e.g., temp or salt)
/// * `state`: Current state (for velocity)
/// * `mesh`: Mesh data
/// * `ops`: DG operators
/// * `geom`: Geometric factors
pub fn apply_tracer_advection_3d(
    rhs_tracer: &mut [f64],
    tracer: &[f64],
    state: &Solution3D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
) {
    let n_levels = state.n_levels;
    let n_nodes = ops.n_nodes;
    let n_face_nodes = ops.n_face_nodes;
    
    // Per-element workspace
    let mut flux_x = vec![0.0; n_nodes];
    let mut flux_y = vec![0.0; n_nodes];
    let mut d_f_dr = vec![0.0; n_nodes];
    let mut d_f_ds = vec![0.0; n_nodes];
    
    // Values buffers
    let mut phi_nodes = vec![0.0; n_nodes];
    let mut u_nodes = vec![0.0; n_nodes];
    let mut v_nodes = vec![0.0; n_nodes];
    
    for k in 0..mesh.n_elements {
        let el_idx = ElementIndex::new(k);
        let rx = geom.rx[k];
        let ry = geom.ry[k];
        let sx = geom.sx[k];
        let sy = geom.sy[k];
        
        for l in 0..n_levels {
            // Gather level values
            for i in 0..n_nodes {
                 phi_nodes[i] = state.get_value(tracer, el_idx, i, l);
                 u_nodes[i] = state.u_column(el_idx, i)[l];
                 v_nodes[i] = state.v_column(el_idx, i)[l];
            }
            
            // --- 1. Volume Term: div(u*phi, v*phi) ---
            for i in 0..n_nodes {
                flux_x[i] = u_nodes[i] * phi_nodes[i];
                flux_y[i] = v_nodes[i] * phi_nodes[i];
            }
            
            compute_strong_divergence(
                &flux_x, &flux_y, 
                ops, rx, ry, sx, sy, 
                &mut d_f_dr, &mut d_f_ds
            );
            
            // Add volume term to RHS
            for i in 0..n_nodes {
                let div = d_f_dr[i] + d_f_ds[i];
                let rhs_col = Solution3D::get_column_mut(rhs_tracer, n_nodes, n_levels, el_idx, i);
                rhs_col[l] -= div;
            }
        }
        
        // --- 2. Surface Terms ---
        for face in 0..4 {
            let normal = geom.normals[k][face];
            let s_jac = geom.surface_j[k][face];
            let j_inv = geom.det_j_inv[k];
            let lift_scale = s_jac * j_inv;
            let face_nodes = &ops.face_nodes[face];
            
            let neighbor_info = mesh.neighbor(el_idx, face);
            
            for l in 0..n_levels {
                // Gather interior values
                let mut phi_int = vec![0.0; n_face_nodes];
                let mut u_int = vec![0.0; n_face_nodes];
                let mut v_int = vec![0.0; n_face_nodes];
                
                for i in 0..n_face_nodes {
                    let ni = face_nodes[i];
                    phi_int[i] = state.get_value(tracer, el_idx, ni, l);
                    u_int[i] = state.u_column(el_idx, ni)[l];
                    v_int[i] = state.v_column(el_idx, ni)[l];
                }
                
                // Gather exterior values
                let mut phi_ext = vec![0.0; n_face_nodes];
                let mut u_ext = vec![0.0; n_face_nodes];
                let mut v_ext = vec![0.0; n_face_nodes];
                
                if let Some(nb) = neighbor_info {
                     let nb_idx = ElementIndex::new(nb.element);
                     let nb_face_nodes = &ops.face_nodes[nb.face];
                     
                     for i in 0..n_face_nodes {
                         let ni = nb_face_nodes[n_face_nodes - 1 - i];
                         phi_ext[i] = state.get_value(tracer, nb_idx, ni, l);
                         u_ext[i] = state.u_column(nb_idx, ni)[l];
                         v_ext[i] = state.v_column(nb_idx, ni)[l];
                     }
                } else {
                    // Boundary condition
                    for i in 0..n_face_nodes {
                        let un = u_int[i] * normal.0 + v_int[i] * normal.1;
                        if un > 0.0 {
                            // Outflow: Copy interior
                            phi_ext[i] = phi_int[i];
                        } else {
                            // Inflow: Dirichlet (e.g. 0.0 or background)
                            // TODO: Add support for boundary conditions
                            phi_ext[i] = phi_int[i]; // Simple extrapolation for now
                        }
                        u_ext[i] = u_int[i];
                        v_ext[i] = v_int[i];
                    }
                }
                
                // Compute Flux Jump
                let mut jump = vec![0.0; n_face_nodes];
                
                for i in 0..n_face_nodes {
                    let un_int = u_int[i] * normal.0 + v_int[i] * normal.1;
                    let un_ext = u_ext[i] * normal.0 + v_ext[i] * normal.1;
                    
                    // Upwind Flux
                    // F* = (un > 0) ? F_int : F_ext
                    // But we use Rusanov for robustness:
                    // F* = 0.5(F_int + F_ext) - 0.5*|un_avg|*(phi_ext - phi_int)
                    
                    let flux_int = phi_int[i] * un_int;
                    let flux_ext = phi_ext[i] * un_ext;
                    let speed = un_int.abs().max(un_ext.abs());
                    
                    let star = 0.5 * (flux_int + flux_ext) - 0.5 * speed * (phi_ext[i] - phi_int[i]);
                    
                    // Jump = F_int - F_star
                    // F_int is flux normal to face = phi_int * un_int
                    jump[i] = flux_int - star;
                }
                
                // Apply Lift
                for i in 0..n_nodes {
                     let mut lift = 0.0;
                     for fi in 0..n_face_nodes {
                         lift += ops.lift[face][(i, fi)] * jump[fi];
                     }
                     let rhs_col = Solution3D::get_column_mut(rhs_tracer, n_nodes, n_levels, el_idx, i);
                     rhs_col[l] += lift_scale * lift;
                }
            }
        }
    }
}

/// Apply vertical advection to 3D fields.
///
/// $\frac{\partial (\Omega \phi)}{\partial s}$
///
/// Uses a centered difference scheme for momentum.
/// Assumes $\Omega = 0$ at the surface and bottom (kinematic boundary condition).
///
/// # Arguments
/// * `rhs`: RHS accumulator
/// * `state`: Current state
/// * `sigma`: Sigma grid (for vertical spacing)
pub fn apply_vertical_advection_3d(
    rhs: &mut Solution3D,
    state: &Solution3D,
    w_vel: &[f64],
    sigma: &SigmaGrid,
) {
    // Momentum
    apply_vertical_advection_field(&mut rhs.u, &state.u, w_vel, state.n_elements, state.n_nodes, state.n_levels, sigma);
    apply_vertical_advection_field(&mut rhs.v, &state.v, w_vel, state.n_elements, state.n_nodes, state.n_levels, sigma);
    
    // Tracers
    apply_vertical_advection_field(&mut rhs.temp, &state.temp, w_vel, state.n_elements, state.n_nodes, state.n_levels, sigma);
    apply_vertical_advection_field(&mut rhs.salt, &state.salt, w_vel, state.n_elements, state.n_nodes, state.n_levels, sigma);
}

/// Generic vertical advection for any field.
pub fn apply_vertical_advection_field(
    rhs_field: &mut [f64],
    field: &[f64],
    w_vel: &[f64], // Omega
    n_elements: usize,
    n_nodes: usize,
    n_levels: usize,
    sigma: &SigmaGrid,
) {
    let d_sigma = sigma.d_sigma();
    let mut flux = vec![0.0; n_levels + 1];
    
    for k in 0..n_elements {
        let el_idx = ElementIndex::new(k);
        for i in 0..n_nodes {
             let w_col = Solution3D::get_column(w_vel, n_nodes, n_levels, el_idx, i);
             let phi_col = Solution3D::get_column(field, n_nodes, n_levels, el_idx, i);
             
             // Compute fluxes at interfaces
             flux[0] = 0.0;
             flux[n_levels] = 0.0;
             
             for l in 1..n_levels {
                 // Centered difference for flux
                 let w_face = 0.5 * (w_col[l-1] + w_col[l]);
                 let phi_face = 0.5 * (phi_col[l-1] + phi_col[l]);
                 flux[l] = w_face * phi_face;
             }
             
             // Update RHS
             let rhs_col = Solution3D::get_column_mut(rhs_field, n_nodes, n_levels, el_idx, i);
             for l in 0..n_levels {
                 let div = (flux[l+1] - flux[l]) / d_sigma[l];
                 rhs_col[l] -= div;
             }
        }
    }
}
