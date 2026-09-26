//! Slope limiters for 2D shallow water equations.
//!
//! Limiters are essential for maintaining stability in DG discretizations
//! of the shallow water equations, especially with:
//! - Steep bathymetry gradients
//! - Wetting/drying fronts
//! - Strong tidal forcing
//!
//! This module provides:
//! - Positivity-preserving limiter (Zhang-Shu) for water depth h ≥ 0
//! - Kuzmin vertex-based limiter for unstructured meshes
//!
//! Positivity is enforced towards h ≥ 0, not towards a minimum depth: raising
//! nodes to h_min > 0 lowers the wet nodes of a partially dry element (mass is
//! kept) and so breaks lake at rest at every shoreline (REVIEW.md §1.5).
//! With a positivity-preserving interface flux (HLL, Rusanov; not Roe) and
//! `CFL ≤ positivity_cfl_swe_2d(N)`, the cell means stay non-negative, so the
//! limiter never has to create mass. The limiters return the number of
//! elements whose mean depth was nevertheless negative (emptied, which creates
//! mass): a diagnostic that should stay zero.
//!
//! # References
//! - Zhang & Shu (2010), "Maximum-principle-satisfying and positivity-preserving
//!   high order discontinuous Galerkin schemes..."
//! - Kuzmin (2010), "A vertex-based hierarchical slope limiter for p-adaptive DG methods"

use crate::mesh::Mesh2D;
use crate::operators::DGOperators2D;
use crate::solver::state::SWESolution2D;
use crate::types::ElementIndex;

// Re-use TVBParameter2D from tracer limiters (same algorithm)
pub use crate::solver::limiters::tracer_2d::KuzminParameter2D;

/// Compute cell averages for SWE variables in 2D.
///
/// Computes the mass-weighted average of h, hu, hv in each element:
/// avg_h = (∫ h * w dA) / (∫ w dA)
///
/// # Affine element assumption
///
/// For affine elements (constant Jacobian per element), J cancels in the
/// ratio ∫q dA / ∫dA, so this function uses quadrature weights directly
/// without Jacobian weighting. This is exact for parallelogram meshes.
/// For curved/non-affine elements, per-node Jacobian weighting would be
/// required — see `GeometricFactors2D` for the affine assumption.
///
/// # Returns
/// Vector of (avg_h, avg_hu, avg_hv) for each element.
pub fn swe_cell_averages_2d(swe: &SWESolution2D, ops: &DGOperators2D) -> Vec<(f64, f64, f64)> {
    let inv_total_weight: f64 = 1.0 / ops.weights.iter().sum::<f64>();
    ElementIndex::iter(swe.n_elements)
        .map(|k| {
            element_mean(
                swe.element_h(k),
                swe.element_hu(k),
                swe.element_hv(k),
                &ops.weights,
                inv_total_weight,
            )
        })
        .collect()
}

/// The Zhang-Shu theta parameter for depth positivity: the largest θ ∈ [0, 1]
/// with `θ(min_elem − avg) + avg ≥ 0`, for `avg ≥ 0`.
fn compute_theta_positivity(avg: f64, min_elem: f64) -> f64 {
    if min_elem >= 0.0 {
        return 1.0; // No limiting needed
    }
    // min_elem < 0 ≤ avg, so avg − min_elem > 0
    (avg / (avg - min_elem)).clamp(0.0, 1.0)
}

/// Per-vertex (min, max) bounds for (h, hu, hv).
type VertexBounds = ((f64, f64), (f64, f64), (f64, f64));

/// q <- avg + factor * (q - avg) for h, hu and hv of one element.
#[inline]
fn scale_deviation(
    h: &mut [f64],
    hu: &mut [f64],
    hv: &mut [f64],
    (h_avg, hu_avg, hv_avg): (f64, f64, f64),
    factor: f64,
) {
    for ((h, hu), hv) in h.iter_mut().zip(hu.iter_mut()).zip(hv.iter_mut()) {
        *h = factor * (*h - h_avg) + h_avg;
        *hu = factor * (*hu - hu_avg) + hu_avg;
        *hv = factor * (*hv - hv_avg) + hv_avg;
    }
}

/// Mean of (h, hu, hv) over one element with GLL weights `weights` (affine
/// elements, see [`swe_cell_averages_2d`]).
#[inline]
pub(crate) fn element_mean(
    h: &[f64],
    hu: &[f64],
    hv: &[f64],
    weights: &[f64],
    inv_total_weight: f64,
) -> (f64, f64, f64) {
    let (mut sh, mut shu, mut shv) = (0.0, 0.0, 0.0);
    for (i, &w) in weights.iter().enumerate() {
        sh += w * h[i];
        shu += w * hu[i];
        shv += w * hv[i];
    }
    (
        sh * inv_total_weight,
        shu * inv_total_weight,
        shv * inv_total_weight,
    )
}

/// Zhang-Shu positivity limiting of one element (SoA slices) with mean `avg`.
///
/// All variables are scaled towards their means until h ≥ 0 at every node.
/// This keeps the element's mass, and it leaves a lake-at-rest shoreline
/// (h = max(0, η − B) ≥ 0) untouched. An element whose mean depth is below
/// `h_dry` is dry: its momentum is set to zero, its depths are kept.
///
/// Returns `true` if the mean depth was negative. The element is then emptied
/// (h = 0), which creates mass; with a positivity-preserving flux under the
/// positivity CFL this cannot happen.
#[inline]
pub(crate) fn positivity_limit_element(
    h: &mut [f64],
    hu: &mut [f64],
    hv: &mut [f64],
    avg: (f64, f64, f64),
    h_dry: f64,
) -> bool {
    let h_avg = avg.0;
    if h_avg <= 0.0 {
        h.fill(0.0);
        hu.fill(0.0);
        hv.fill(0.0);
        return h_avg < 0.0;
    }

    let h_min_elem = h.iter().copied().fold(f64::INFINITY, f64::min);
    if h_min_elem < 0.0 {
        // Limiting all variables with the same theta preserves well-balancing
        let theta = compute_theta_positivity(h_avg, h_min_elem);
        scale_deviation(h, hu, hv, avg, theta);
        // θ(min − avg) + avg may round to −ulp
        for h in h.iter_mut() {
            *h = h.max(0.0);
        }
    }

    if h_avg < h_dry {
        hu.fill(0.0);
        hv.fill(0.0);
    }
    false
}

/// Kuzmin vertex-based limiting of one element (SoA slices) with mean `avg`.
///
/// `vertices` are the element's global vertex indices (CCW) and
/// `vertex_bounds[v]` the patch bounds at global vertex v.
#[inline]
fn kuzmin_limit_element(
    h: &mut [f64],
    hu: &mut [f64],
    hv: &mut [f64],
    avg: (f64, f64, f64),
    vertices: [usize; 4],
    vertex_bounds: &[VertexBounds],
    n_1d: usize,
) {
    let (h_avg, hu_avg, hv_avg) = avg;

    // One alpha for all variables (maintains consistency)
    let mut alpha = 1.0_f64;
    for (local_v, &global_v) in vertices.iter().enumerate() {
        let ((h_lo, h_hi), (hu_lo, hu_hi), (hv_lo, hv_hi)) = vertex_bounds[global_v];
        let node = vertex_to_node_index(local_v, n_1d);
        alpha = alpha
            .min(compute_kuzmin_alpha(h_avg, h[node], h_lo, h_hi))
            .min(compute_kuzmin_alpha(hu_avg, hu[node], hu_lo, hu_hi))
            .min(compute_kuzmin_alpha(hv_avg, hv[node], hv_lo, hv_hi));
    }

    if alpha < 1.0 - 1e-10 {
        scale_deviation(h, hu, hv, avg, alpha);
    }
}

/// Run `f(k, h, hu, hv)` on the nodal slices of every element and sum its
/// results.
fn for_each_element(
    swe: &mut SWESolution2D,
    mut f: impl FnMut(usize, &mut [f64], &mut [f64], &mut [f64]) -> usize,
) -> usize {
    let n = swe.n_nodes;
    let [h, hu, hv] = &mut swe.data;
    h.chunks_exact_mut(n)
        .zip(hu.chunks_exact_mut(n))
        .zip(hv.chunks_exact_mut(n))
        .enumerate()
        .map(|(k, ((h, hu), hv))| f(k, h, hu, hv))
        .sum()
}

/// Apply Zhang-Shu positivity-preserving limiter for water depth.
///
/// Ensures h >= 0 at all nodes while preserving cell averages.
/// Uses the theta-scaling approach:
///   q_limited = theta(q - avg) + avg
///
/// where theta is chosen to enforce h >= 0. Elements whose mean depth is
/// below `h_dry` are treated as dry (see `positivity_limit_element`).
///
/// # Arguments
/// * `swe` - SWE solution to limit (modified in place)
/// * `ops` - DG operators (for quadrature weights)
/// * `h_dry` - Dry threshold: elements with a smaller mean depth lose their momentum
///
/// # Returns
/// The number of elements with a negative mean depth, emptied to h = 0
/// (creating mass). Zero under the positivity CFL with HLL or Rusanov.
pub fn swe_positivity_limiter_2d(
    swe: &mut SWESolution2D,
    ops: &DGOperators2D,
    h_dry: f64,
) -> usize {
    // Each element needs only its own mean: no separate pass, no allocation
    let inv_total_weight = 1.0 / ops.weights.iter().sum::<f64>();
    for_each_element(swe, |_, h, hu, hv| {
        let avg = element_mean(h, hu, hv, &ops.weights, inv_total_weight);
        positivity_limit_element(h, hu, hv, avg, h_dry) as usize
    })
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
) -> VertexBounds {
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

/// Patch bounds at every mesh vertex.
fn all_vertex_bounds(
    mesh: &Mesh2D,
    averages: &[(f64, f64, f64)],
    relaxation: f64,
) -> Vec<VertexBounds> {
    (0..mesh.vertices.len())
        .map(|v| compute_vertex_bounds(v, mesh, averages, relaxation))
        .collect()
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
    let averages = swe_cell_averages_2d(swe, ops);
    let vertex_bounds = all_vertex_bounds(mesh, &averages, kuzmin.relaxation);
    for_each_element(swe, |k, h, hu, hv| {
        let vertices = mesh.element_vertex_indices(ElementIndex::new(k));
        kuzmin_limit_element(h, hu, hv, averages[k], vertices, &vertex_bounds, ops.n_1d);
        0
    });
}

/// Apply Kuzmin and positivity limiters to SWE fields.
///
/// Per element, in one pass:
/// 1. Kuzmin limiter (vertex-based oscillation control)
/// 2. Positivity limiter (ensures h >= 0; dry elements lose momentum)
///
/// Kuzmin preserves the cell means, so both steps use the same averages.
/// `apply_swe_limiters_kuzmin_2d_parallel` runs the same element kernels.
///
/// # Arguments
/// * `swe` - SWE solution to limit (modified in place)
/// * `mesh` - 2D mesh
/// * `ops` - DG operators
/// * `kuzmin` - Kuzmin limiter parameter
/// * `h_dry` - Dry threshold for the element mean depth
///
/// # Returns
/// The number of negative-mean elements, as for [`swe_positivity_limiter_2d`].
pub fn apply_swe_limiters_kuzmin_2d(
    swe: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    kuzmin: &KuzminParameter2D,
    h_dry: f64,
) -> usize {
    let averages = swe_cell_averages_2d(swe, ops);
    let vertex_bounds = all_vertex_bounds(mesh, &averages, kuzmin.relaxation);
    for_each_element(swe, |k, h, hu, hv| {
        let vertices = mesh.element_vertex_indices(ElementIndex::new(k));
        kuzmin_limit_element(h, hu, hv, averages[k], vertices, &vertex_bounds, ops.n_1d);
        positivity_limit_element(h, hu, hv, averages[k], h_dry) as usize
    })
}

// ============================================================================
// PARALLEL IMPLEMENTATIONS
// ============================================================================

/// Parallel version of cell averages computation using Rayon.
///
/// See [`swe_cell_averages_2d`] for the affine element assumption.
#[cfg(feature = "parallel")]
pub fn swe_cell_averages_2d_parallel(
    swe: &SWESolution2D,
    ops: &DGOperators2D,
) -> Vec<(f64, f64, f64)> {
    use rayon::prelude::*;

    let n = swe.n_nodes;
    let inv_total_weight: f64 = 1.0 / ops.weights.iter().sum::<f64>();
    swe.h_data()
        .par_chunks_exact(n)
        .zip(swe.hu_data().par_chunks_exact(n))
        .zip(swe.hv_data().par_chunks_exact(n))
        .map(|((h, hu), hv)| element_mean(h, hu, hv, &ops.weights, inv_total_weight))
        .collect()
}

/// Parallel version of `for_each_element`, in place.
#[cfg(feature = "parallel")]
fn par_for_each_element(
    swe: &mut SWESolution2D,
    f: impl Fn(usize, &mut [f64], &mut [f64], &mut [f64]) -> usize + Sync + Send,
) -> usize {
    use rayon::prelude::*;

    let n = swe.n_nodes;
    let [h, hu, hv] = &mut swe.data;
    h.par_chunks_exact_mut(n)
        .zip(hu.par_chunks_exact_mut(n))
        .zip(hv.par_chunks_exact_mut(n))
        .enumerate()
        .map(|(k, ((h, hu), hv))| f(k, h, hu, hv))
        .sum()
}

/// Parallel patch bounds at every mesh vertex.
#[cfg(feature = "parallel")]
fn all_vertex_bounds_parallel(
    mesh: &Mesh2D,
    averages: &[(f64, f64, f64)],
    relaxation: f64,
) -> Vec<VertexBounds> {
    use rayon::prelude::*;

    (0..mesh.vertices.len())
        .into_par_iter()
        .map(|v| compute_vertex_bounds(v, mesh, averages, relaxation))
        .collect()
}

/// Parallel positivity-preserving limiter using Rayon.
///
/// Same element kernel as [`swe_positivity_limiter_2d`], applied in place.
#[cfg(feature = "parallel")]
pub fn swe_positivity_limiter_2d_parallel(
    swe: &mut SWESolution2D,
    ops: &DGOperators2D,
    h_dry: f64,
) -> usize {
    let inv_total_weight = 1.0 / ops.weights.iter().sum::<f64>();
    par_for_each_element(swe, |_, h, hu, hv| {
        let avg = element_mean(h, hu, hv, &ops.weights, inv_total_weight);
        positivity_limit_element(h, hu, hv, avg, h_dry) as usize
    })
}

/// Parallel Kuzmin vertex-based slope limiter using Rayon.
///
/// Same element kernel as [`swe_kuzmin_limiter_2d`], applied in place.
#[cfg(feature = "parallel")]
pub fn swe_kuzmin_limiter_2d_parallel(
    swe: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    kuzmin: &KuzminParameter2D,
) {
    let averages = swe_cell_averages_2d_parallel(swe, ops);
    let vertex_bounds = all_vertex_bounds_parallel(mesh, &averages, kuzmin.relaxation);
    par_for_each_element(swe, |k, h, hu, hv| {
        let vertices = mesh.element_vertex_indices(ElementIndex::new(k));
        kuzmin_limit_element(h, hu, hv, averages[k], vertices, &vertex_bounds, ops.n_1d);
        0
    });
}

/// Parallel combined Kuzmin + positivity limiter.
///
/// Same element kernels as [`apply_swe_limiters_kuzmin_2d`] (including the
/// dry-element branch), applied in place; the result is bitwise identical.
#[cfg(feature = "parallel")]
pub fn apply_swe_limiters_kuzmin_2d_parallel(
    swe: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    kuzmin: &KuzminParameter2D,
    h_dry: f64,
) -> usize {
    let averages = swe_cell_averages_2d_parallel(swe, ops);
    let vertex_bounds = all_vertex_bounds_parallel(mesh, &averages, kuzmin.relaxation);
    par_for_each_element(swe, |k, h, hu, hv| {
        let vertices = mesh.element_vertex_indices(ElementIndex::new(k));
        kuzmin_limit_element(h, hu, hv, averages[k], vertices, &vertex_bounds, ops.n_1d);
        positivity_limit_element(h, hu, hv, averages[k], h_dry) as usize
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::state::SWEState2D;

    const H_MIN: f64 = 0.01;

    /// 4×4 P2 mesh with wet, oscillating elements and three problem elements:
    /// element 5 dry with momentum (mean 0.004 < H_MIN), element 6 with a
    /// negative mean depth, element 9 wet but with a negative node.
    fn wet_dry_setup() -> (Mesh2D, DGOperators2D, SWESolution2D) {
        let mesh = Mesh2D::uniform_rectangle(0.0, 4.0, 0.0, 4.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let mut swe = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            let ki = k.as_usize();
            for i in 0..ops.n_nodes {
                let wiggle = ((7 * ki + 3 * i) % 5) as f64 - 2.0;
                let state = match ki {
                    5 => SWEState2D::new(0.004 + 0.003 * wiggle, 0.2, -0.1),
                    6 => SWEState2D::new(-0.002 + 0.001 * wiggle, -0.3, 0.05),
                    9 => SWEState2D::new(if i == 4 { -0.05 } else { 0.5 }, 0.1, 0.1),
                    _ => SWEState2D::new(1.0 + 0.2 * wiggle, 0.3 * wiggle, -0.1 * wiggle),
                };
                swe.set_state(k, i, state);
            }
        }
        (mesh, ops, swe)
    }

    fn assert_positive_and_dry_at_rest(swe: &SWESolution2D) {
        assert!(
            swe.h_data().iter().all(|&h| h >= 0.0),
            "negative depth survived"
        );
        for k in [5, 6].map(ElementIndex::new) {
            assert!(
                swe.element_hu(k)
                    .iter()
                    .chain(swe.element_hv(k))
                    .all(|&m| m == 0.0),
                "dry element {} kept momentum",
                k.as_usize()
            );
        }
    }

    #[test]
    fn test_fused_limiter_dry_elements() {
        let (mesh, ops, mut swe) = wet_dry_setup();
        let clipped = apply_swe_limiters_kuzmin_2d(
            &mut swe,
            &mesh,
            &ops,
            &KuzminParameter2D::strict(),
            H_MIN,
        );
        assert_positive_and_dry_at_rest(&swe);
        // Element 6 has a negative mean; it is emptied and reported
        assert_eq!(clipped, 1);
        assert!(
            swe.element_h(ElementIndex::new(6))
                .iter()
                .all(|&h| h == 0.0)
        );
    }

    #[test]
    fn test_positivity_keeps_lake_at_rest_shoreline() {
        // REVIEW.md §1.5: limiting towards h ≥ h_min put a film on the dry
        // nodes of a shoreline element and lowered its wet nodes, in every
        // stage. Towards h ≥ 0, a lake at rest (h = max(0, η − B)) is untouched,
        // also when the element mean is below the dry threshold.
        let ops = DGOperators2D::new(3);
        for (eta, h_dry) in [(0.3, 0.01), (0.02, 0.1)] {
            let mut swe = SWESolution2D::new(1, ops.n_nodes);
            for i in 0..ops.n_nodes {
                let bed = 0.5 * ops.nodes_r[i] + 0.1 * ops.nodes_s[i];
                swe.set_state(
                    ElementIndex::new(0),
                    i,
                    SWEState2D::new((eta - bed).max(0.0), 0.0, 0.0),
                );
            }
            let before = swe.data.clone();
            assert_eq!(swe_positivity_limiter_2d(&mut swe, &ops, h_dry), 0);
            assert_eq!(swe.data, before, "η = {eta}, h_dry = {h_dry}");
        }
    }

    #[test]
    fn test_positivity_limits_towards_zero_conserving_mass() {
        let ops = DGOperators2D::new(2);
        let mut swe = SWESolution2D::new(1, ops.n_nodes);
        let k = ElementIndex::new(0);
        for i in 0..ops.n_nodes {
            let h = if i == 0 { -0.05 } else { 0.03 }; // corner node: positive mean
            swe.set_state(k, i, SWEState2D::new(h, 0.01, -0.02));
        }
        let mass = |swe: &SWESolution2D| -> f64 {
            swe.element_h(k)
                .iter()
                .zip(&ops.weights)
                .map(|(h, w)| h * w)
                .sum()
        };
        let before = mass(&swe);

        assert_eq!(swe_positivity_limiter_2d(&mut swe, &ops, 1e-3), 0);
        let h = swe.element_h(k);
        assert!(h.iter().all(|&h| h >= 0.0));
        // The limited minimum is zero, not a positive floor
        assert!(h.iter().copied().fold(f64::INFINITY, f64::min) < 1e-15);
        assert!((mass(&swe) - before).abs() < 1e-16);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_fused_limiter_dry_elements() {
        // P0.19 regression: the parallel fused Kuzmin + positivity limiter had no
        // dry-element branch. θ clamped to 0, so dry elements kept their mean
        // momentum and a negative mean depth survived.
        let (mesh, ops, mut swe) = wet_dry_setup();
        apply_swe_limiters_kuzmin_2d_parallel(
            &mut swe,
            &mesh,
            &ops,
            &KuzminParameter2D::strict(),
            H_MIN,
        );
        assert_positive_and_dry_at_rest(&swe);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_limiters_match_serial() {
        let (mesh, ops, input) = wet_dry_setup();
        for kuzmin in [KuzminParameter2D::strict(), KuzminParameter2D::relaxed(1.5)] {
            let (mut serial, mut parallel) = (input.clone(), input.clone());
            apply_swe_limiters_kuzmin_2d(&mut serial, &mesh, &ops, &kuzmin, H_MIN);
            apply_swe_limiters_kuzmin_2d_parallel(&mut parallel, &mesh, &ops, &kuzmin, H_MIN);
            assert_eq!(serial.data, parallel.data, "fused Kuzmin + positivity");

            let (mut serial, mut parallel) = (input.clone(), input.clone());
            swe_kuzmin_limiter_2d(&mut serial, &mesh, &ops, &kuzmin);
            swe_kuzmin_limiter_2d_parallel(&mut parallel, &mesh, &ops, &kuzmin);
            assert_eq!(serial.data, parallel.data, "Kuzmin");
        }

        let (mut serial, mut parallel) = (input.clone(), input.clone());
        swe_positivity_limiter_2d(&mut serial, &ops, H_MIN);
        swe_positivity_limiter_2d_parallel(&mut parallel, &ops, H_MIN);
        assert_eq!(serial.data, parallel.data, "positivity");
    }

    #[test]
    fn test_compute_theta_positivity_no_violation() {
        // avg = 10, min = 5 -> no limiting needed
        let theta = compute_theta_positivity(10.0, 5.0);
        assert!((theta - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_theta_positivity_with_violation() {
        // avg = 6, min = -2: theta * (-2 - 6) + 6 >= 0 => theta <= 6/8
        let theta = compute_theta_positivity(6.0, -2.0);
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

    #[test]
    fn test_positivity_dry_average_does_not_inject_mass() {
        let ops = DGOperators2D::new(2);
        let mut swe = SWESolution2D::new(1, ops.n_nodes);
        let k = ElementIndex::new(0);

        for i in 0..ops.n_nodes {
            swe.set_state(k, i, SWEState2D::new(0.005, 0.1, -0.1));
        }

        swe_positivity_limiter_2d(&mut swe, &ops, 0.01);

        for i in 0..ops.n_nodes {
            let state = swe.get_state(k, i);
            assert!((state.h - 0.005).abs() < 1e-14);
            assert!(state.hu.abs() < 1e-14);
            assert!(state.hv.abs() < 1e-14);
        }
    }
}
