//! Slope limiters for 2D shallow water equations.
//!
//! Limiters are essential for maintaining stability in DG discretizations
//! of the shallow water equations, especially with:
//! - Steep bathymetry gradients
//! - Wetting/drying fronts
//! - Strong tidal forcing
//!
//! This module provides:
//! - Positivity-preserving limiter (Zhang-Shu) for water depth h > 0
//! - TVB (Total Variation Bounded) limiter for controlling oscillations
//! - Kuzmin vertex-based limiter for unstructured meshes
//!
//! # References
//! - Zhang & Shu (2010), "Maximum-principle-satisfying and positivity-preserving
//!   high order discontinuous Galerkin schemes..."
//! - Kuzmin (2010), "A vertex-based hierarchical slope limiter for p-adaptive DG methods"
//! - Xing & Shu (2010), "High order well-balanced finite volume WENO schemes and
//!   discontinuous Galerkin methods for a class of hyperbolic systems..."

use crate::mesh::Mesh2D;
use crate::operators::DGOperators2D;
use crate::solver::state::{SWESolution2D, SWEState2D};
use crate::types::ElementIndex;

// Re-use TVBParameter2D from tracer limiters (same algorithm)
pub use crate::solver::limiters::tracer_2d::{KuzminParameter2D, TVBParameter2D};

/// Compute cell averages for SWE variables in 2D.
///
/// Computes the mass-weighted average of h, hu, hv in each element:
/// avg_h = (∫ h * w dA) / (∫ w dA)
///
/// # Returns
/// Vector of (avg_h, avg_hu, avg_hv) for each element.
pub fn swe_cell_averages_2d(
    swe: &SWESolution2D,
    ops: &DGOperators2D,
) -> Vec<(f64, f64, f64)> {
    let n_elements = swe.n_elements;
    let n_nodes = swe.n_nodes;
    let mut averages = Vec::with_capacity(n_elements);

    // Precompute inverse total weight for faster division
    let inv_total_weight: f64 = 1.0 / ops.weights.iter().sum::<f64>();

    for k in ElementIndex::iter(n_elements) {
        let mut integral_h = 0.0;
        let mut integral_hu = 0.0;
        let mut integral_hv = 0.0;

        // Direct data access for better performance
        let elem_data = swe.element_data(k);

        for i in 0..n_nodes {
            let w = ops.weights[i];
            let base = i * 3;
            integral_h += w * elem_data[base];
            integral_hu += w * elem_data[base + 1];
            integral_hv += w * elem_data[base + 2];
        }

        // Compute averages using precomputed inverse
        averages.push((
            integral_h * inv_total_weight,
            integral_hu * inv_total_weight,
            integral_hv * inv_total_weight,
        ));
    }

    averages
}

/// Compute the Zhang-Shu theta parameter for depth positivity.
///
/// Given a cell average and minimum value, computes the maximum θ ∈ [0,1]
/// such that `θ(h - avg) + avg >= h_min`.
fn compute_theta_positivity(avg: f64, min_elem: f64, h_min: f64) -> f64 {
    if min_elem >= h_min {
        return 1.0; // No limiting needed
    }

    if (avg - min_elem).abs() < 1e-14 {
        return 1.0; // Constant, no oscillation
    }

    // Need: θ(min - avg) + avg >= h_min
    // => θ(min - avg) >= h_min - avg
    // Since min < avg (otherwise min >= h_min), we have (min - avg) < 0
    // => θ <= (avg - h_min) / (avg - min)
    let theta = (avg - h_min) / (avg - min_elem);
    theta.clamp(0.0, 1.0)
}

/// Apply Zhang-Shu positivity-preserving limiter for water depth.
///
/// Ensures h >= h_min at all nodes while preserving cell averages.
/// Uses the θ-scaling approach:
///   q_limited = θ(q - avg) + avg
///
/// where θ is chosen to enforce h >= h_min.
///
/// # Arguments
/// * `swe` - SWE solution to limit (modified in place)
/// * `ops` - DG operators (for quadrature weights)
/// * `h_min` - Minimum depth threshold
pub fn swe_positivity_limiter_2d(
    swe: &mut SWESolution2D,
    ops: &DGOperators2D,
    h_min: f64,
) {
    let n_elements = swe.n_elements;
    let n_nodes = ops.n_nodes;

    // First compute all cell averages
    let averages = swe_cell_averages_2d(swe, ops);

    for k in ElementIndex::iter(n_elements) {
        let (h_avg, hu_avg, hv_avg) = averages[k.as_usize()];

        // Skip if average is below threshold (dry cell)
        if h_avg < h_min {
            // Set entire cell to minimum state
            for i in 0..n_nodes {
                swe.set_state(k, i, SWEState2D::from_primitives(h_min, 0.0, 0.0));
            }
            continue;
        }

        // Find minimum h in element
        let mut h_min_elem = f64::INFINITY;
        for i in 0..n_nodes {
            h_min_elem = h_min_elem.min(swe.get_state(k, i).h);
        }

        // Check if limiting is needed
        if h_min_elem >= h_min {
            continue;
        }

        // Compute limiting factor
        let theta = compute_theta_positivity(h_avg, h_min_elem, h_min);

        // Apply limiting to all variables (preserves well-balancing)
        for i in 0..n_nodes {
            let state = swe.get_state(k, i);
            let h_new = theta * (state.h - h_avg) + h_avg;
            let hu_new = theta * (state.hu - hu_avg) + hu_avg;
            let hv_new = theta * (state.hv - hv_avg) + hv_avg;

            swe.set_state(k, i, SWEState2D { h: h_new, hu: hu_new, hv: hv_new });
        }
    }
}

/// Minmod function for three arguments.
///
/// Returns 0 if arguments have different signs, otherwise the smallest magnitude.
fn minmod(a: f64, b: f64, c: f64) -> f64 {
    if a > 0.0 && b > 0.0 && c > 0.0 {
        a.min(b).min(c)
    } else if a < 0.0 && b < 0.0 && c < 0.0 {
        a.max(b).max(c)
    } else {
        0.0
    }
}

/// TVB-modified minmod function.
///
/// Only activates limiting if the first argument exceeds the TVB threshold.
fn minmod_tvb(a: f64, b: f64, c: f64, threshold: f64) -> f64 {
    if a.abs() <= threshold {
        a
    } else {
        minmod(a, b, c)
    }
}

/// Apply TVB slope limiter to SWE fields in 2D.
///
/// Uses a simplified approach: computes gradients and limits based on
/// neighbor averages. For 2D, we limit each direction independently.
///
/// # Arguments
/// * `swe` - SWE solution to limit (modified in place)
/// * `mesh` - 2D mesh for neighbor connectivity
/// * `ops` - DG operators
/// * `tvb` - TVB parameter
pub fn swe_tvb_limiter_2d(
    swe: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    tvb: &TVBParameter2D,
) {
    let n_elements = swe.n_elements;
    let n_nodes = ops.n_nodes;

    // First compute all cell averages
    let averages = swe_cell_averages_2d(swe, ops);

    for k in ElementIndex::iter(n_elements) {
        let (h_avg, hu_avg, hv_avg) = averages[k.as_usize()];

        // Estimate element size from mesh
        let h_elem = mesh.element_diameter(k);
        let threshold = tvb.threshold(h_elem);

        // Get neighbor averages (use own average for boundary faces)
        // Face convention: 0=bottom, 1=right, 2=top, 3=left
        let mut neighbor_h = [h_avg; 4];
        let mut neighbor_hu = [hu_avg; 4];
        let mut neighbor_hv = [hv_avg; 4];

        for face in 0..4 {
            if let Some(neigh) = mesh.neighbor(k, face) {
                neighbor_h[face] = averages[neigh.element].0;
                neighbor_hu[face] = averages[neigh.element].1;
                neighbor_hv[face] = averages[neigh.element].2;
            }
        }

        // Compute differences to neighbors in each direction
        let delta_h_left = h_avg - neighbor_h[3];
        let delta_h_right = neighbor_h[1] - h_avg;
        let delta_hu_left = hu_avg - neighbor_hu[3];
        let delta_hu_right = neighbor_hu[1] - hu_avg;
        let delta_hv_left = hv_avg - neighbor_hv[3];
        let delta_hv_right = neighbor_hv[1] - hv_avg;

        let delta_h_bottom = h_avg - neighbor_h[0];
        let delta_h_top = neighbor_h[2] - h_avg;
        let delta_hu_bottom = hu_avg - neighbor_hu[0];
        let delta_hu_top = neighbor_hu[2] - hu_avg;
        let delta_hv_bottom = hv_avg - neighbor_hv[0];
        let delta_hv_top = neighbor_hv[2] - hv_avg;

        // Compute boundary differences in element (first vs last nodes in each direction)
        let n_1d = (n_nodes as f64).sqrt() as usize;

        // x-direction: compare edges at i=0 and i=n_1d-1
        let mut delta_h_x_elem = 0.0;
        let mut delta_hu_x_elem = 0.0;
        let mut delta_hv_x_elem = 0.0;
        for j in 0..n_1d {
            let i_left = j * n_1d;
            let i_right = j * n_1d + n_1d - 1;
            let s_left = swe.get_state(k, i_left);
            let s_right = swe.get_state(k, i_right);
            delta_h_x_elem += (s_right.h - s_left.h) / n_1d as f64;
            delta_hu_x_elem += (s_right.hu - s_left.hu) / n_1d as f64;
            delta_hv_x_elem += (s_right.hv - s_left.hv) / n_1d as f64;
        }

        // y-direction: compare edges at j=0 and j=n_1d-1
        let mut delta_h_y_elem = 0.0;
        let mut delta_hu_y_elem = 0.0;
        let mut delta_hv_y_elem = 0.0;
        for i in 0..n_1d {
            let i_bottom = i;
            let i_top = (n_1d - 1) * n_1d + i;
            let s_bottom = swe.get_state(k, i_bottom);
            let s_top = swe.get_state(k, i_top);
            delta_h_y_elem += (s_top.h - s_bottom.h) / n_1d as f64;
            delta_hu_y_elem += (s_top.hu - s_bottom.hu) / n_1d as f64;
            delta_hv_y_elem += (s_top.hv - s_bottom.hv) / n_1d as f64;
        }

        // Apply TVB minmod in x-direction
        let limited_dh_x = minmod_tvb(delta_h_x_elem, delta_h_left, delta_h_right, threshold);
        let limited_dhu_x = minmod_tvb(delta_hu_x_elem, delta_hu_left, delta_hu_right, threshold);
        let limited_dhv_x = minmod_tvb(delta_hv_x_elem, delta_hv_left, delta_hv_right, threshold);

        // Apply TVB minmod in y-direction
        let limited_dh_y = minmod_tvb(delta_h_y_elem, delta_h_bottom, delta_h_top, threshold);
        let limited_dhu_y = minmod_tvb(delta_hu_y_elem, delta_hu_bottom, delta_hu_top, threshold);
        let limited_dhv_y = minmod_tvb(delta_hv_y_elem, delta_hv_bottom, delta_hv_top, threshold);

        // Check if limiting changed the gradients significantly
        let h_changed = (limited_dh_x - delta_h_x_elem).abs() > 1e-10
            || (limited_dh_y - delta_h_y_elem).abs() > 1e-10;
        let hu_changed = (limited_dhu_x - delta_hu_x_elem).abs() > 1e-10
            || (limited_dhu_y - delta_hu_y_elem).abs() > 1e-10;
        let hv_changed = (limited_dhv_x - delta_hv_x_elem).abs() > 1e-10
            || (limited_dhv_y - delta_hv_y_elem).abs() > 1e-10;

        if !h_changed && !hu_changed && !hv_changed {
            continue; // No limiting needed
        }

        // Replace high-order content with limited linear reconstruction
        for i in 0..n_nodes {
            let r = ops.nodes_r[i];
            let s = ops.nodes_s[i];

            // Linear reconstruction from limited gradients
            let h_new = h_avg + 0.5 * limited_dh_x * r + 0.5 * limited_dh_y * s;
            let hu_new = hu_avg + 0.5 * limited_dhu_x * r + 0.5 * limited_dhu_y * s;
            let hv_new = hv_avg + 0.5 * limited_dhv_x * r + 0.5 * limited_dhv_y * s;

            swe.set_state(k, i, SWEState2D { h: h_new, hu: hu_new, hv: hv_new });
        }
    }
}

/// Map local vertex index (0-3 in CCW order) to DG node index.
#[inline]
fn vertex_to_node_index(local_vertex: usize, n_1d: usize) -> usize {
    match local_vertex {
        0 => 0,                 // (r=-1, s=-1)
        1 => n_1d - 1,          // (r=+1, s=-1)
        2 => n_1d * n_1d - 1,   // (r=+1, s=+1)
        3 => n_1d * (n_1d - 1), // (r=-1, s=+1)
        _ => panic!("Invalid local vertex index: {}", local_vertex),
    }
}

/// Compute bounds from a vertex patch.
fn compute_vertex_bounds(
    vertex: usize,
    mesh: &Mesh2D,
    averages: &[(f64, f64, f64)],
    relaxation: f64,
) -> ((f64, f64), (f64, f64), (f64, f64)) {
    let patch = mesh.elements_at_vertex(vertex);

    let mut h_min = f64::INFINITY;
    let mut h_max = f64::NEG_INFINITY;
    let mut hu_min = f64::INFINITY;
    let mut hu_max = f64::NEG_INFINITY;
    let mut hv_min = f64::INFINITY;
    let mut hv_max = f64::NEG_INFINITY;

    for &elem in patch {
        let (h_avg, hu_avg, hv_avg) = averages[elem];
        h_min = h_min.min(h_avg);
        h_max = h_max.max(h_avg);
        hu_min = hu_min.min(hu_avg);
        hu_max = hu_max.max(hu_avg);
        hv_min = hv_min.min(hv_avg);
        hv_max = hv_max.max(hv_avg);
    }

    // Apply relaxation
    if relaxation > 1.0 {
        let h_range = h_max - h_min;
        let hu_range = hu_max - hu_min;
        let hv_range = hv_max - hv_min;
        let h_expand = 0.5 * h_range * (relaxation - 1.0);
        let hu_expand = 0.5 * hu_range * (relaxation - 1.0);
        let hv_expand = 0.5 * hv_range * (relaxation - 1.0);
        h_min -= h_expand;
        h_max += h_expand;
        hu_min -= hu_expand;
        hu_max += hu_expand;
        hv_min -= hv_expand;
        hv_max += hv_expand;
    }

    ((h_min, h_max), (hu_min, hu_max), (hv_min, hv_max))
}

/// Compute the limiting factor alpha for a single value.
#[inline(always)]
fn compute_kuzmin_alpha(avg: f64, value: f64, bound_min: f64, bound_max: f64) -> f64 {
    let deviation = value - avg;

    if deviation.abs() < 1e-14 {
        return 1.0;
    }

    let mut alpha: f64 = 1.0;

    if value < bound_min && deviation < 0.0 {
        let required = (avg - bound_min) / (avg - value);
        alpha = alpha.min(required);
    }

    if value > bound_max && deviation > 0.0 {
        let required = (bound_max - avg) / (value - avg);
        alpha = alpha.min(required);
    }

    alpha.clamp(0.0, 1.0)
}

/// Apply Kuzmin vertex-based slope limiter to SWE fields in 2D.
///
/// Uses vertex-patch stencils to compute local bounds, providing tighter
/// oscillation control than face-neighbor based limiters.
///
/// # Arguments
/// * `swe` - SWE solution to limit (modified in place)
/// * `mesh` - 2D mesh with vertex_to_elements connectivity
/// * `ops` - DG operators
/// * `kuzmin` - Kuzmin limiter parameters
pub fn swe_kuzmin_limiter_2d(
    swe: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    kuzmin: &KuzminParameter2D,
) {
    let n_elements = swe.n_elements;
    let n_nodes = ops.n_nodes;
    let n_1d = ops.n_1d;

    // Compute all cell averages
    let averages = swe_cell_averages_2d(swe, ops);

    for k in ElementIndex::iter(n_elements) {
        let (h_avg, hu_avg, hv_avg) = averages[k.as_usize()];

        // Get element vertices
        let vertices = mesh.element_vertex_indices(k);

        // Compute limiting factor for each variable
        let mut alpha_h = 1.0_f64;
        let mut alpha_hu = 1.0_f64;
        let mut alpha_hv = 1.0_f64;

        for (local_v, &global_v) in vertices.iter().enumerate() {
            // Compute bounds from vertex patch
            let ((h_min, h_max), (hu_min, hu_max), (hv_min, hv_max)) =
                compute_vertex_bounds(global_v, mesh, &averages, kuzmin.relaxation);

            // Get nodal value at this vertex
            let node_idx = vertex_to_node_index(local_v, n_1d);
            let state = swe.get_state(k, node_idx);

            // Compute limiting factors
            alpha_h = alpha_h.min(compute_kuzmin_alpha(h_avg, state.h, h_min, h_max));
            alpha_hu = alpha_hu.min(compute_kuzmin_alpha(hu_avg, state.hu, hu_min, hu_max));
            alpha_hv = alpha_hv.min(compute_kuzmin_alpha(hv_avg, state.hv, hv_min, hv_max));
        }

        // Use minimum alpha for all variables (maintains consistency)
        let alpha = alpha_h.min(alpha_hu).min(alpha_hv);

        // If limiting needed, apply to all nodes
        if alpha < 1.0 - 1e-10 {
            for i in 0..n_nodes {
                let state = swe.get_state(k, i);

                let h_new = alpha * (state.h - h_avg) + h_avg;
                let hu_new = alpha * (state.hu - hu_avg) + hu_avg;
                let hv_new = alpha * (state.hv - hv_avg) + hv_avg;

                swe.set_state(k, i, SWEState2D { h: h_new, hu: hu_new, hv: hv_new });
            }
        }
    }
}

/// Apply both slope limiter and positivity limiter to SWE fields.
///
/// This applies limiting in the correct order:
/// 1. TVB limiter (controls oscillations)
/// 2. Positivity limiter (ensures h >= h_min)
///
/// # Arguments
/// * `swe` - SWE solution to limit (modified in place)
/// * `mesh` - 2D mesh
/// * `ops` - DG operators
/// * `tvb` - TVB parameter
/// * `h_min` - Minimum depth threshold
pub fn apply_swe_limiters_2d(
    swe: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    tvb: &TVBParameter2D,
    h_min: f64,
) {
    // First apply TVB limiter to control oscillations
    swe_tvb_limiter_2d(swe, mesh, ops, tvb);

    // Then apply positivity limiter to ensure h >= h_min
    swe_positivity_limiter_2d(swe, ops, h_min);
}

/// Apply Kuzmin and positivity limiters to SWE fields.
///
/// This applies limiting in the correct order:
/// 1. Kuzmin limiter (vertex-based oscillation control)
/// 2. Positivity limiter (ensures h >= h_min)
///
/// # Arguments
/// * `swe` - SWE solution to limit (modified in place)
/// * `mesh` - 2D mesh
/// * `ops` - DG operators
/// * `kuzmin` - Kuzmin limiter parameter
/// * `h_min` - Minimum depth threshold
pub fn apply_swe_limiters_kuzmin_2d(
    swe: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    kuzmin: &KuzminParameter2D,
    h_min: f64,
) {
    // First apply Kuzmin limiter to control oscillations
    swe_kuzmin_limiter_2d(swe, mesh, ops, kuzmin);

    // Then apply positivity limiter to ensure h >= h_min
    swe_positivity_limiter_2d(swe, ops, h_min);
}

// ============================================================================
// PARALLEL IMPLEMENTATIONS
// ============================================================================

/// Parallel version of cell averages computation using Rayon.
#[cfg(feature = "parallel")]
pub fn swe_cell_averages_2d_parallel(
    swe: &SWESolution2D,
    ops: &DGOperators2D,
) -> Vec<(f64, f64, f64)> {
    use rayon::prelude::*;

    let n_elements = swe.n_elements;
    let n_nodes = swe.n_nodes;
    let inv_total_weight: f64 = 1.0 / ops.weights.iter().sum::<f64>();

    (0..n_elements)
        .into_par_iter()
        .map(|k| {
            let k_idx = ElementIndex::new(k);
            let elem_data = swe.element_data(k_idx);

            let mut integral_h = 0.0;
            let mut integral_hu = 0.0;
            let mut integral_hv = 0.0;

            for i in 0..n_nodes {
                let w = ops.weights[i];
                let base = i * 3;
                integral_h += w * elem_data[base];
                integral_hu += w * elem_data[base + 1];
                integral_hv += w * elem_data[base + 2];
            }

            (
                integral_h * inv_total_weight,
                integral_hu * inv_total_weight,
                integral_hv * inv_total_weight,
            )
        })
        .collect()
}

/// Parallel positivity-preserving limiter using Rayon.
///
/// Each element's limiting is independent once cell averages are computed.
#[cfg(feature = "parallel")]
pub fn swe_positivity_limiter_2d_parallel(
    swe: &mut SWESolution2D,
    ops: &DGOperators2D,
    h_min: f64,
) {
    use rayon::prelude::*;

    let n_nodes = ops.n_nodes;

    // Step 1: Compute all cell averages in parallel
    let averages = swe_cell_averages_2d_parallel(swe, ops);

    // Step 2: Apply limiting in parallel using par_chunks_mut
    let chunk_size = n_nodes * 3;
    swe.data
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(k, elem_data)| {
            let (avg_h, avg_hu, avg_hv) = averages[k];

            // Find minimum h value in element
            let mut min_h = f64::INFINITY;
            for i in 0..n_nodes {
                min_h = min_h.min(elem_data[i * 3]);
            }

            // Compute theta for positivity
            let theta = compute_theta_positivity(avg_h, min_h, h_min);

            // Apply scaling if needed
            if theta < 1.0 - 1e-14 {
                for i in 0..n_nodes {
                    let base = i * 3;
                    elem_data[base] = theta * (elem_data[base] - avg_h) + avg_h;
                    elem_data[base + 1] = theta * (elem_data[base + 1] - avg_hu) + avg_hu;
                    elem_data[base + 2] = theta * (elem_data[base + 2] - avg_hv) + avg_hv;
                }
            }
        });
}

/// Parallel Kuzmin vertex-based slope limiter using Rayon.
///
/// Pre-computes all cell averages and vertex bounds BEFORE the parallel loop,
/// making the element-level computation truly embarrassingly parallel.
#[cfg(feature = "parallel")]
pub fn swe_kuzmin_limiter_2d_parallel(
    swe: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    kuzmin: &KuzminParameter2D,
) {
    use rayon::prelude::*;

    let n_nodes = ops.n_nodes;
    let n_1d = ops.n_1d;

    // Step 1: Compute all cell averages in parallel
    let averages = swe_cell_averages_2d_parallel(swe, ops);

    // Step 2: Pre-compute ALL vertex bounds (avoids mesh lookups in parallel loop)
    let n_vertices = mesh.vertices.len();
    let vertex_bounds: Vec<_> = (0..n_vertices)
        .into_par_iter()
        .map(|v| compute_vertex_bounds(v, mesh, &averages, kuzmin.relaxation))
        .collect();

    // Step 3: Apply limiting in parallel - now truly embarrassingly parallel
    let chunk_size = n_nodes * 3;
    swe.data
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(k, elem_data)| {
            let k_idx = ElementIndex::new(k);
            let (h_avg, hu_avg, hv_avg) = averages[k];

            // Get element vertices (just indices, no mesh lookup needed)
            let vertices = mesh.element_vertex_indices(k_idx);

            // Compute limiting factor for each variable
            let mut alpha_h = 1.0_f64;
            let mut alpha_hu = 1.0_f64;
            let mut alpha_hv = 1.0_f64;

            for (local_v, &global_v) in vertices.iter().enumerate() {
                // Use pre-computed bounds (no mesh lookup!)
                let ((h_min, h_max), (hu_min, hu_max), (hv_min, hv_max)) = vertex_bounds[global_v];

                // Get nodal value at this vertex
                let node_idx = vertex_to_node_index(local_v, n_1d);
                let base = node_idx * 3;
                let h_val = elem_data[base];
                let hu_val = elem_data[base + 1];
                let hv_val = elem_data[base + 2];

                // Compute limiting factors
                alpha_h = alpha_h.min(compute_kuzmin_alpha(h_avg, h_val, h_min, h_max));
                alpha_hu = alpha_hu.min(compute_kuzmin_alpha(hu_avg, hu_val, hu_min, hu_max));
                alpha_hv = alpha_hv.min(compute_kuzmin_alpha(hv_avg, hv_val, hv_min, hv_max));
            }

            // Use minimum alpha for all variables (maintains consistency)
            let alpha = alpha_h.min(alpha_hu).min(alpha_hv);

            // If limiting needed, apply to all nodes
            if alpha < 1.0 - 1e-10 {
                for i in 0..n_nodes {
                    let base = i * 3;
                    elem_data[base] = alpha * (elem_data[base] - h_avg) + h_avg;
                    elem_data[base + 1] = alpha * (elem_data[base + 1] - hu_avg) + hu_avg;
                    elem_data[base + 2] = alpha * (elem_data[base + 2] - hv_avg) + hv_avg;
                }
            }
        });
}

/// Parallel combined Kuzmin + positivity limiter.
///
/// Optimized to compute cell averages only once and fuse the limiting operations.
#[cfg(feature = "parallel")]
pub fn apply_swe_limiters_kuzmin_2d_parallel(
    swe: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    kuzmin: &KuzminParameter2D,
    h_min: f64,
) {
    use rayon::prelude::*;

    let n_nodes = ops.n_nodes;
    let n_1d = ops.n_1d;

    // Step 1: Compute cell averages ONCE (shared between both limiters)
    let averages = swe_cell_averages_2d_parallel(swe, ops);

    // Step 2: Pre-compute vertex bounds for Kuzmin limiter
    let n_vertices = mesh.vertices.len();
    let vertex_bounds: Vec<_> = (0..n_vertices)
        .into_par_iter()
        .map(|v| compute_vertex_bounds(v, mesh, &averages, kuzmin.relaxation))
        .collect();

    // Step 3: Apply BOTH limiters in a single parallel pass
    let chunk_size = n_nodes * 3;
    swe.data
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(k, elem_data)| {
            let k_idx = ElementIndex::new(k);
            let (h_avg, hu_avg, hv_avg) = averages[k];

            // === Kuzmin limiter ===
            let vertices = mesh.element_vertex_indices(k_idx);
            let mut alpha_h = 1.0_f64;
            let mut alpha_hu = 1.0_f64;
            let mut alpha_hv = 1.0_f64;

            for (local_v, &global_v) in vertices.iter().enumerate() {
                let ((bound_h_min, h_max), (hu_min, hu_max), (hv_min, hv_max)) = vertex_bounds[global_v];
                let node_idx = vertex_to_node_index(local_v, n_1d);
                let base = node_idx * 3;

                alpha_h = alpha_h.min(compute_kuzmin_alpha(h_avg, elem_data[base], bound_h_min, h_max));
                alpha_hu = alpha_hu.min(compute_kuzmin_alpha(hu_avg, elem_data[base + 1], hu_min, hu_max));
                alpha_hv = alpha_hv.min(compute_kuzmin_alpha(hv_avg, elem_data[base + 2], hv_min, hv_max));
            }

            let alpha_kuzmin = alpha_h.min(alpha_hu).min(alpha_hv);

            // Apply Kuzmin limiting if needed
            if alpha_kuzmin < 1.0 - 1e-10 {
                for i in 0..n_nodes {
                    let base = i * 3;
                    elem_data[base] = alpha_kuzmin * (elem_data[base] - h_avg) + h_avg;
                    elem_data[base + 1] = alpha_kuzmin * (elem_data[base + 1] - hu_avg) + hu_avg;
                    elem_data[base + 2] = alpha_kuzmin * (elem_data[base + 2] - hv_avg) + hv_avg;
                }
            }

            // === Positivity limiter (after Kuzmin) ===
            // Recompute average after Kuzmin (or use original if no limiting)
            let (pos_h_avg, pos_hu_avg, pos_hv_avg) = if alpha_kuzmin < 1.0 - 1e-10 {
                // Average unchanged by Kuzmin (preserves average)
                (h_avg, hu_avg, hv_avg)
            } else {
                (h_avg, hu_avg, hv_avg)
            };

            // Find minimum h after Kuzmin
            let mut min_h_after = f64::INFINITY;
            for i in 0..n_nodes {
                min_h_after = min_h_after.min(elem_data[i * 3]);
            }

            let theta = compute_theta_positivity(pos_h_avg, min_h_after, h_min);

            if theta < 1.0 - 1e-14 {
                for i in 0..n_nodes {
                    let base = i * 3;
                    elem_data[base] = theta * (elem_data[base] - pos_h_avg) + pos_h_avg;
                    elem_data[base + 1] = theta * (elem_data[base + 1] - pos_hu_avg) + pos_hu_avg;
                    elem_data[base + 2] = theta * (elem_data[base + 2] - pos_hv_avg) + pos_hv_avg;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_theta_positivity_no_violation() {
        // avg = 10, min = 5, h_min = 1 -> no limiting needed
        let theta = compute_theta_positivity(10.0, 5.0, 1.0);
        assert!((theta - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_theta_positivity_with_violation() {
        // avg = 10, min = -2, h_min = 1
        // Need: theta * (-2 - 10) + 10 >= 1
        // => -12*theta >= -9 => theta <= 9/12 = 0.75
        let theta = compute_theta_positivity(10.0, -2.0, 1.0);
        assert!((theta - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_minmod_same_sign_positive() {
        assert_eq!(minmod(1.0, 2.0, 3.0), 1.0);
    }

    #[test]
    fn test_minmod_same_sign_negative() {
        assert_eq!(minmod(-1.0, -2.0, -3.0), -1.0);
    }

    #[test]
    fn test_minmod_different_signs() {
        assert_eq!(minmod(1.0, -2.0, 3.0), 0.0);
    }

    #[test]
    fn test_minmod_tvb_below_threshold() {
        let result = minmod_tvb(0.5, 1.0, 2.0, 1.0);
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_minmod_tvb_above_threshold() {
        let result = minmod_tvb(1.5, 1.0, 2.0, 1.0);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vertex_to_node_index_p2() {
        assert_eq!(vertex_to_node_index(0, 3), 0);
        assert_eq!(vertex_to_node_index(1, 3), 2);
        assert_eq!(vertex_to_node_index(2, 3), 8);
        assert_eq!(vertex_to_node_index(3, 3), 6);
    }

    #[test]
    fn test_compute_kuzmin_alpha_no_violation() {
        let alpha = compute_kuzmin_alpha(10.0, 12.0, 5.0, 20.0);
        assert!((alpha - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_kuzmin_alpha_upper_violation() {
        // avg = 10, value = 25, bounds = [5, 20]
        let alpha = compute_kuzmin_alpha(10.0, 25.0, 5.0, 20.0);
        let expected = 10.0 / 15.0;
        assert!((alpha - expected).abs() < 1e-10);
    }
}
