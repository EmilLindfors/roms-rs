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
//! - Kuzmin vertex-based limiter for unstructured meshes
//!
//! # References
//! - Zhang & Shu (2010), "Maximum-principle-satisfying and positivity-preserving
//!   high order discontinuous Galerkin schemes..."
//! - Kuzmin (2010), "A vertex-based hierarchical slope limiter for p-adaptive DG methods"

use crate::mesh::Mesh2D;
use crate::operators::DGOperators2D;
use crate::solver::state::{SWESolution2D, SWEState2D};
use crate::types::ElementIndex;

// Re-use TVBParameter2D from tracer limiters (same algorithm)
pub use crate::solver::limiters::tracer_2d::KuzminParameter2D;

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

        // SoA data access - get slices for each variable
        let elem_h = swe.element_h(k);
        let elem_hu = swe.element_hu(k);
        let elem_hv = swe.element_hv(k);

        for i in 0..n_nodes {
            let w = ops.weights[i];
            integral_h += w * elem_h[i];
            integral_hu += w * elem_hu[i];
            integral_hv += w * elem_hv[i];
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

    // Get immutable slices to the SoA data
    let h_data = swe.h_data();
    let hu_data = swe.hu_data();
    let hv_data = swe.hv_data();

    (0..n_elements)
        .into_par_iter()
        .map(|k| {
            let start = k * n_nodes;
            let end = start + n_nodes;

            let elem_h = &h_data[start..end];
            let elem_hu = &hu_data[start..end];
            let elem_hv = &hv_data[start..end];

            let mut integral_h = 0.0;
            let mut integral_hu = 0.0;
            let mut integral_hv = 0.0;

            for i in 0..n_nodes {
                let w = ops.weights[i];
                integral_h += w * elem_h[i];
                integral_hu += w * elem_hu[i];
                integral_hv += w * elem_hv[i];
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
    let n_elements = swe.n_elements;

    // Step 1: Compute all cell averages in parallel
    let averages = swe_cell_averages_2d_parallel(swe, ops);

    // Step 2: Copy data for parallel processing (SoA layout)
    let h_data: Vec<f64> = swe.h_data().to_vec();
    let hu_data: Vec<f64> = swe.hu_data().to_vec();
    let hv_data: Vec<f64> = swe.hv_data().to_vec();

    // Step 3: Compute limited values in parallel
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..n_elements)
        .into_par_iter()
        .map(|k| {
            let start = k * n_nodes;
            let end = start + n_nodes;

            let mut h_out = h_data[start..end].to_vec();
            let mut hu_out = hu_data[start..end].to_vec();
            let mut hv_out = hv_data[start..end].to_vec();

            let (avg_h, avg_hu, avg_hv) = averages[k];

            // Find minimum h value in element
            let min_h = h_out.iter().cloned().fold(f64::INFINITY, f64::min);

            // Compute theta for positivity
            let theta = compute_theta_positivity(avg_h, min_h, h_min);

            // Apply scaling if needed
            if theta < 1.0 - 1e-14 {
                for i in 0..n_nodes {
                    h_out[i] = theta * (h_out[i] - avg_h) + avg_h;
                    hu_out[i] = theta * (hu_out[i] - avg_hu) + avg_hu;
                    hv_out[i] = theta * (hv_out[i] - avg_hv) + avg_hv;
                }
            }

            (h_out, hu_out, hv_out)
        })
        .collect();

    // Step 4: Write results back (borrow sequentially to avoid multiple mutable borrows)
    let (h_results, hu_results, hv_results): (Vec<_>, Vec<_>, Vec<_>) = results
        .into_iter()
        .fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut h_acc, mut hu_acc, mut hv_acc), (h, hu, hv)| {
                h_acc.push(h);
                hu_acc.push(hu);
                hv_acc.push(hv);
                (h_acc, hu_acc, hv_acc)
            },
        );

    for (k, h_elem) in h_results.into_iter().enumerate() {
        let start = k * n_nodes;
        swe.h_data_mut()[start..start + n_nodes].copy_from_slice(&h_elem);
    }
    for (k, hu_elem) in hu_results.into_iter().enumerate() {
        let start = k * n_nodes;
        swe.hu_data_mut()[start..start + n_nodes].copy_from_slice(&hu_elem);
    }
    for (k, hv_elem) in hv_results.into_iter().enumerate() {
        let start = k * n_nodes;
        swe.hv_data_mut()[start..start + n_nodes].copy_from_slice(&hv_elem);
    }
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
    let n_elements = swe.n_elements;

    // Step 1: Compute all cell averages in parallel
    let averages = swe_cell_averages_2d_parallel(swe, ops);

    // Step 2: Pre-compute ALL vertex bounds (avoids mesh lookups in parallel loop)
    let n_vertices = mesh.vertices.len();
    let vertex_bounds: Vec<_> = (0..n_vertices)
        .into_par_iter()
        .map(|v| compute_vertex_bounds(v, mesh, &averages, kuzmin.relaxation))
        .collect();

    // Step 3: Copy data for parallel processing (SoA layout)
    let h_data: Vec<f64> = swe.h_data().to_vec();
    let hu_data: Vec<f64> = swe.hu_data().to_vec();
    let hv_data: Vec<f64> = swe.hv_data().to_vec();

    // Step 4: Compute limited values in parallel
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..n_elements)
        .into_par_iter()
        .map(|k| {
            let start = k * n_nodes;
            let end = start + n_nodes;

            let mut h_out = h_data[start..end].to_vec();
            let mut hu_out = hu_data[start..end].to_vec();
            let mut hv_out = hv_data[start..end].to_vec();

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
                let h_val = h_out[node_idx];
                let hu_val = hu_out[node_idx];
                let hv_val = hv_out[node_idx];

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
                    h_out[i] = alpha * (h_out[i] - h_avg) + h_avg;
                    hu_out[i] = alpha * (hu_out[i] - hu_avg) + hu_avg;
                    hv_out[i] = alpha * (hv_out[i] - hv_avg) + hv_avg;
                }
            }

            (h_out, hu_out, hv_out)
        })
        .collect();

    // Step 5: Write results back (borrow sequentially to avoid multiple mutable borrows)
    let (h_results, hu_results, hv_results): (Vec<_>, Vec<_>, Vec<_>) = results
        .into_iter()
        .fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut h_acc, mut hu_acc, mut hv_acc), (h, hu, hv)| {
                h_acc.push(h);
                hu_acc.push(hu);
                hv_acc.push(hv);
                (h_acc, hu_acc, hv_acc)
            },
        );

    for (k, h_elem) in h_results.into_iter().enumerate() {
        let start = k * n_nodes;
        swe.h_data_mut()[start..start + n_nodes].copy_from_slice(&h_elem);
    }
    for (k, hu_elem) in hu_results.into_iter().enumerate() {
        let start = k * n_nodes;
        swe.hu_data_mut()[start..start + n_nodes].copy_from_slice(&hu_elem);
    }
    for (k, hv_elem) in hv_results.into_iter().enumerate() {
        let start = k * n_nodes;
        swe.hv_data_mut()[start..start + n_nodes].copy_from_slice(&hv_elem);
    }
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
    let n_elements = swe.n_elements;

    // Step 1: Compute cell averages ONCE (shared between both limiters)
    let averages = swe_cell_averages_2d_parallel(swe, ops);

    // Step 2: Pre-compute vertex bounds for Kuzmin limiter
    let n_vertices = mesh.vertices.len();
    let vertex_bounds: Vec<_> = (0..n_vertices)
        .into_par_iter()
        .map(|v| compute_vertex_bounds(v, mesh, &averages, kuzmin.relaxation))
        .collect();

    // Step 3: Copy data for parallel processing (SoA layout)
    let h_data: Vec<f64> = swe.h_data().to_vec();
    let hu_data: Vec<f64> = swe.hu_data().to_vec();
    let hv_data: Vec<f64> = swe.hv_data().to_vec();

    // Step 4: Apply BOTH limiters in a single parallel pass
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..n_elements)
        .into_par_iter()
        .map(|k| {
            let start = k * n_nodes;
            let end = start + n_nodes;

            let mut h_out = h_data[start..end].to_vec();
            let mut hu_out = hu_data[start..end].to_vec();
            let mut hv_out = hv_data[start..end].to_vec();

            let k_idx = ElementIndex::new(k);
            let (h_avg, hu_avg, hv_avg) = averages[k];

            // === Kuzmin limiter ===
            let vertices = mesh.element_vertex_indices(k_idx);
            let mut alpha_h = 1.0_f64;
            let mut alpha_hu = 1.0_f64;
            let mut alpha_hv = 1.0_f64;

            for (local_v, &global_v) in vertices.iter().enumerate() {
                let ((bound_h_min, bound_h_max), (hu_min, hu_max), (hv_min, hv_max)) = vertex_bounds[global_v];
                let node_idx = vertex_to_node_index(local_v, n_1d);

                alpha_h = alpha_h.min(compute_kuzmin_alpha(h_avg, h_out[node_idx], bound_h_min, bound_h_max));
                alpha_hu = alpha_hu.min(compute_kuzmin_alpha(hu_avg, hu_out[node_idx], hu_min, hu_max));
                alpha_hv = alpha_hv.min(compute_kuzmin_alpha(hv_avg, hv_out[node_idx], hv_min, hv_max));
            }

            let alpha_kuzmin = alpha_h.min(alpha_hu).min(alpha_hv);

            // Apply Kuzmin limiting if needed
            if alpha_kuzmin < 1.0 - 1e-10 {
                for i in 0..n_nodes {
                    h_out[i] = alpha_kuzmin * (h_out[i] - h_avg) + h_avg;
                    hu_out[i] = alpha_kuzmin * (hu_out[i] - hu_avg) + hu_avg;
                    hv_out[i] = alpha_kuzmin * (hv_out[i] - hv_avg) + hv_avg;
                }
            }

            // === Positivity limiter (after Kuzmin) ===
            // Average unchanged by Kuzmin (preserves average)
            let pos_h_avg = h_avg;
            let pos_hu_avg = hu_avg;
            let pos_hv_avg = hv_avg;

            // Find minimum h after Kuzmin
            let min_h_after = h_out.iter().cloned().fold(f64::INFINITY, f64::min);

            let theta = compute_theta_positivity(pos_h_avg, min_h_after, h_min);

            if theta < 1.0 - 1e-14 {
                for i in 0..n_nodes {
                    h_out[i] = theta * (h_out[i] - pos_h_avg) + pos_h_avg;
                    hu_out[i] = theta * (hu_out[i] - pos_hu_avg) + pos_hu_avg;
                    hv_out[i] = theta * (hv_out[i] - pos_hv_avg) + pos_hv_avg;
                }
            }

            (h_out, hu_out, hv_out)
        })
        .collect();

    // Step 5: Write results back (borrow sequentially to avoid multiple mutable borrows)
    let (h_results, hu_results, hv_results): (Vec<_>, Vec<_>, Vec<_>) = results
        .into_iter()
        .fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut h_acc, mut hu_acc, mut hv_acc), (h, hu, hv)| {
                h_acc.push(h);
                hu_acc.push(hu);
                hv_acc.push(hv);
                (h_acc, hu_acc, hv_acc)
            },
        );

    for (k, h_elem) in h_results.into_iter().enumerate() {
        let start = k * n_nodes;
        swe.h_data_mut()[start..start + n_nodes].copy_from_slice(&h_elem);
    }
    for (k, hu_elem) in hu_results.into_iter().enumerate() {
        let start = k * n_nodes;
        swe.hu_data_mut()[start..start + n_nodes].copy_from_slice(&hu_elem);
    }
    for (k, hv_elem) in hv_results.into_iter().enumerate() {
        let start = k * n_nodes;
        swe.hv_data_mut()[start..start + n_nodes].copy_from_slice(&hv_elem);
    }
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
