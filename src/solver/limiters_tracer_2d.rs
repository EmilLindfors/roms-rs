//! Slope limiters for 2D tracer transport.
//!
//! Limiters are essential for maintaining stability and preventing oscillations
//! in high-order DG discretizations of advection-dominated tracer transport.
//!
//! This module provides:
//! - Positivity-preserving limiter (Zhang-Shu) for tracer bounds
//! - TVB (Total Variation Bounded) limiter for controlling oscillations
//!
//! # References
//! - Zhang & Shu (2010), "Maximum-principle-satisfying and positivity-preserving
//!   high order discontinuous Galerkin schemes..."
//! - Kuzmin (2010), "A vertex-based hierarchical slope limiter for p-adaptive DG methods"

use crate::mesh::Mesh2D;
use crate::operators::DGOperators2D;
use crate::solver::state_2d::SWESolution2D;
use crate::solver::tracer_state::{ConservativeTracerState, TracerSolution2D};

/// Physical bounds for tracer concentrations.
///
/// Used by the positivity limiter to enforce physically realizable values.
#[derive(Clone, Copy, Debug)]
pub struct TracerBounds {
    /// Minimum temperature (°C), typically -2.0 for seawater
    pub t_min: f64,
    /// Maximum temperature (°C), typically 40.0
    pub t_max: f64,
    /// Minimum salinity (PSU), typically 0.0
    pub s_min: f64,
    /// Maximum salinity (PSU), typically 42.0
    pub s_max: f64,
}

impl Default for TracerBounds {
    fn default() -> Self {
        Self {
            t_min: -2.0,
            t_max: 40.0,
            s_min: 0.0,
            s_max: 42.0,
        }
    }
}

impl TracerBounds {
    /// Create new tracer bounds.
    pub fn new(t_min: f64, t_max: f64, s_min: f64, s_max: f64) -> Self {
        Self {
            t_min,
            t_max,
            s_min,
            s_max,
        }
    }

    /// Norwegian coastal water bounds (slightly relaxed for numerical margin).
    pub fn norwegian_coast() -> Self {
        Self {
            t_min: -2.0,
            t_max: 25.0,
            s_min: 0.0,
            s_max: 36.0,
        }
    }
}

/// TVB parameter for 2D tracer limiting.
///
/// The TVB (Total Variation Bounded) limiter allows smooth extrema to pass
/// unmodified while limiting oscillations. The threshold is computed as:
///   threshold = M * (h_elem / L_ref)²
///
/// where L_ref is a reference length scale (typically the domain size or
/// characteristic length). This normalization ensures M has consistent meaning
/// regardless of physical units.
#[derive(Clone, Copy, Debug)]
pub struct TVBParameter2D {
    /// The M parameter (larger = less limiting)
    pub m: f64,
    /// Reference length scale for normalization (m)
    pub l_ref: f64,
}

impl TVBParameter2D {
    /// Create a new TVB parameter with explicit reference length.
    ///
    /// # Arguments
    /// * `m` - TVB parameter (typically 1-100; larger = less limiting)
    /// * `l_ref` - Reference length scale for normalization (e.g., domain size)
    pub fn new(m: f64, l_ref: f64) -> Self {
        Self {
            m,
            l_ref: l_ref.max(1.0),
        }
    }

    /// Create TVB parameter with domain-relative scaling.
    ///
    /// This is the recommended constructor for coastal models where
    /// the domain size varies significantly.
    pub fn with_domain_size(m: f64, domain_size: f64) -> Self {
        Self::new(m, domain_size)
    }

    /// Compute threshold for a given element size.
    ///
    /// Returns the threshold in tracer units (°C or PSU) below which
    /// the slope is considered smooth and not limited.
    pub fn threshold(&self, h_elem: f64) -> f64 {
        let h_normalized = h_elem / self.l_ref;
        self.m * h_normalized * h_normalized
    }
}

impl Default for TVBParameter2D {
    fn default() -> Self {
        // Default to M=50 with L_ref=1 (for normalized domains)
        Self {
            m: 50.0,
            l_ref: 1.0,
        }
    }
}

/// Compute cell averages for tracer concentrations in 2D.
///
/// Computes the mass-weighted average of T and S in each element:
/// avg_C = (∫ hC dA) / (∫ h dA)
///
/// # Returns
/// Vector of (avg_T, avg_S) for each element.
pub fn tracer_cell_averages_2d(
    tracers: &TracerSolution2D,
    swe: &SWESolution2D,
    ops: &DGOperators2D,
    h_min: f64,
) -> Vec<(f64, f64)> {
    let n_elements = tracers.n_elements;
    let n_nodes = ops.n_nodes;
    let mut averages = Vec::with_capacity(n_elements);

    for k in 0..n_elements {
        let mut integral_h = 0.0;
        let mut integral_h_t = 0.0;
        let mut integral_h_s = 0.0;

        for (i, &w) in ops.weights.iter().enumerate() {
            let h = swe.get_state(k, i).h.max(h_min);
            let cons = tracers.get_conservative(k, i);
            integral_h += w * h;
            integral_h_t += w * cons.h_t;
            integral_h_s += w * cons.h_s;
        }

        // Convert to concentrations
        let avg_t = if integral_h > h_min * n_nodes as f64 * ops.weights[0] {
            integral_h_t / integral_h
        } else {
            0.0
        };
        let avg_s = if integral_h > h_min * n_nodes as f64 * ops.weights[0] {
            integral_h_s / integral_h
        } else {
            0.0
        };

        averages.push((avg_t, avg_s));
    }

    averages
}

/// Compute the Zhang-Shu theta parameter for bounds enforcement.
///
/// Given a cell average and extreme values, computes the maximum θ ∈ [0,1]
/// such that `θ(C - avg) + avg` stays within [c_min, c_max].
fn compute_theta(avg: f64, min_elem: f64, max_elem: f64, bound_min: f64, bound_max: f64) -> f64 {
    let mut theta: f64 = 1.0;

    // Lower bound violation: need θ(min - avg) + avg >= bound_min
    // => θ <= (avg - bound_min) / (avg - min_elem)
    if min_elem < bound_min && (avg - min_elem).abs() > 1e-14 {
        let t = (avg - bound_min) / (avg - min_elem);
        theta = theta.min(t);
    }

    // Upper bound violation: need θ(max - avg) + avg <= bound_max
    // => θ <= (bound_max - avg) / (max_elem - avg)
    if max_elem > bound_max && (max_elem - avg).abs() > 1e-14 {
        let t = (bound_max - avg) / (max_elem - avg);
        theta = theta.min(t);
    }

    theta.clamp(0.0, 1.0)
}

/// Apply Zhang-Shu positivity-preserving limiter for tracer bounds.
///
/// Ensures tracers stay within physical bounds while preserving cell averages.
/// Uses the θ-scaling approach:
///   C_limited = θ(C - avg) + avg
///
/// where θ is chosen to enforce the bounds.
///
/// # Arguments
/// * `tracers` - Tracer solution to limit (modified in place)
/// * `swe` - SWE solution (for depth h)
/// * `ops` - DG operators (for quadrature weights)
/// * `bounds` - Physical bounds for tracers
/// * `h_min` - Minimum depth threshold
pub fn tracer_positivity_limiter_2d(
    tracers: &mut TracerSolution2D,
    swe: &SWESolution2D,
    ops: &DGOperators2D,
    bounds: &TracerBounds,
    h_min: f64,
) {
    let n_elements = tracers.n_elements;
    let n_nodes = ops.n_nodes;

    // First compute all cell averages
    let averages = tracer_cell_averages_2d(tracers, swe, ops, h_min);

    for k in 0..n_elements {
        let (t_avg, s_avg) = averages[k];

        // Check if element is mostly dry - skip limiting
        let avg_h: f64 = (0..n_nodes)
            .map(|i| ops.weights[i] * swe.get_state(k, i).h)
            .sum::<f64>()
            / ops.weights.iter().sum::<f64>();

        if avg_h < h_min {
            continue;
        }

        // Find extreme tracer values in element
        let mut t_min_elem = f64::INFINITY;
        let mut t_max_elem = f64::NEG_INFINITY;
        let mut s_min_elem = f64::INFINITY;
        let mut s_max_elem = f64::NEG_INFINITY;

        for i in 0..n_nodes {
            let h = swe.get_state(k, i).h.max(h_min);
            let conc = tracers.get_concentrations(k, i, h, h_min);
            t_min_elem = t_min_elem.min(conc.temperature);
            t_max_elem = t_max_elem.max(conc.temperature);
            s_min_elem = s_min_elem.min(conc.salinity);
            s_max_elem = s_max_elem.max(conc.salinity);
        }

        // Check if limiting is needed
        let t_needs_limiting = t_min_elem < bounds.t_min || t_max_elem > bounds.t_max;
        let s_needs_limiting = s_min_elem < bounds.s_min || s_max_elem > bounds.s_max;

        if !t_needs_limiting && !s_needs_limiting {
            continue;
        }

        // Compute limiting factors
        let theta_t = compute_theta(t_avg, t_min_elem, t_max_elem, bounds.t_min, bounds.t_max);
        let theta_s = compute_theta(s_avg, s_min_elem, s_max_elem, bounds.s_min, bounds.s_max);

        // Apply limiting
        for i in 0..n_nodes {
            let h = swe.get_state(k, i).h.max(h_min);
            let cons = tracers.get_conservative(k, i);

            // Get current concentrations
            let t_curr = if h > h_min { cons.h_t / h } else { t_avg };
            let s_curr = if h > h_min { cons.h_s / h } else { s_avg };

            // Apply theta scaling around mean
            let t_new = theta_t * (t_curr - t_avg) + t_avg;
            let s_new = theta_s * (s_curr - s_avg) + s_avg;

            // Convert back to conservative form
            tracers.set_conservative(
                k,
                i,
                ConservativeTracerState {
                    h_t: h * t_new,
                    h_s: h * s_new,
                },
            );
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

/// Apply TVB slope limiter to tracer fields in 2D.
///
/// Uses a simplified approach: computes gradients and limits based on
/// neighbor averages. For 2D, we limit each direction independently.
///
/// # Arguments
/// * `tracers` - Tracer solution to limit (modified in place)
/// * `swe` - SWE solution (for depth h)
/// * `mesh` - 2D mesh for neighbor connectivity
/// * `ops` - DG operators
/// * `tvb` - TVB parameter
/// * `h_min` - Minimum depth threshold
pub fn tracer_tvb_limiter_2d(
    tracers: &mut TracerSolution2D,
    swe: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    tvb: &TVBParameter2D,
    h_min: f64,
) {
    let n_elements = tracers.n_elements;
    let n_nodes = ops.n_nodes;

    // First compute all cell averages
    let averages = tracer_cell_averages_2d(tracers, swe, ops, h_min);

    for k in 0..n_elements {
        let (t_avg, s_avg) = averages[k];

        // Estimate element size from mesh
        let h_elem = mesh.element_diameter(k);
        let threshold = tvb.threshold(h_elem);

        // Get neighbor averages (use own average for boundary faces)
        // Face convention: 0=bottom, 1=right, 2=top, 3=left
        let mut neighbor_t = [t_avg; 4];
        let mut neighbor_s = [s_avg; 4];

        for face in 0..4 {
            if let Some(neigh) = mesh.neighbor(k, face) {
                neighbor_t[face] = averages[neigh.element].0;
                neighbor_s[face] = averages[neigh.element].1;
            }
        }

        // Compute differences to neighbors in each direction
        // Faces 0 (bottom) and 2 (top) are in y-direction
        // Faces 1 (right) and 3 (left) are in x-direction
        let delta_t_left = t_avg - neighbor_t[3]; // Difference to left neighbor
        let delta_t_right = neighbor_t[1] - t_avg; // Difference to right neighbor
        let delta_s_left = s_avg - neighbor_s[3];
        let delta_s_right = neighbor_s[1] - s_avg;

        let delta_t_bottom = t_avg - neighbor_t[0]; // Difference to bottom neighbor
        let delta_t_top = neighbor_t[2] - t_avg; // Difference to top neighbor
        let delta_s_bottom = s_avg - neighbor_s[0];
        let delta_s_top = neighbor_s[2] - s_avg;

        // Compute boundary differences in element (first vs last nodes in each direction)
        // For tensor-product elements, nodes are ordered (i,j) with i varying fastest
        let n_1d = (n_nodes as f64).sqrt() as usize;

        // x-direction: compare edges at i=0 and i=n_1d-1
        let mut delta_t_x_elem = 0.0;
        let mut delta_s_x_elem = 0.0;
        for j in 0..n_1d {
            let i_left = j * n_1d;
            let i_right = j * n_1d + n_1d - 1;
            let h_left = swe.get_state(k, i_left).h.max(h_min);
            let h_right = swe.get_state(k, i_right).h.max(h_min);
            let t_left = tracers
                .get_concentrations(k, i_left, h_left, h_min)
                .temperature;
            let t_right = tracers
                .get_concentrations(k, i_right, h_right, h_min)
                .temperature;
            let s_left = tracers
                .get_concentrations(k, i_left, h_left, h_min)
                .salinity;
            let s_right = tracers
                .get_concentrations(k, i_right, h_right, h_min)
                .salinity;
            delta_t_x_elem += (t_right - t_left) / n_1d as f64;
            delta_s_x_elem += (s_right - s_left) / n_1d as f64;
        }

        // y-direction: compare edges at j=0 and j=n_1d-1
        let mut delta_t_y_elem = 0.0;
        let mut delta_s_y_elem = 0.0;
        for i in 0..n_1d {
            let i_bottom = i;
            let i_top = (n_1d - 1) * n_1d + i;
            let h_bottom = swe.get_state(k, i_bottom).h.max(h_min);
            let h_top = swe.get_state(k, i_top).h.max(h_min);
            let t_bottom = tracers
                .get_concentrations(k, i_bottom, h_bottom, h_min)
                .temperature;
            let t_top = tracers
                .get_concentrations(k, i_top, h_top, h_min)
                .temperature;
            let s_bottom = tracers
                .get_concentrations(k, i_bottom, h_bottom, h_min)
                .salinity;
            let s_top = tracers.get_concentrations(k, i_top, h_top, h_min).salinity;
            delta_t_y_elem += (t_top - t_bottom) / n_1d as f64;
            delta_s_y_elem += (s_top - s_bottom) / n_1d as f64;
        }

        // Apply TVB minmod in x-direction
        let limited_dt_x = minmod_tvb(delta_t_x_elem, delta_t_left, delta_t_right, threshold);
        let limited_ds_x = minmod_tvb(delta_s_x_elem, delta_s_left, delta_s_right, threshold);

        // Apply TVB minmod in y-direction
        let limited_dt_y = minmod_tvb(delta_t_y_elem, delta_t_bottom, delta_t_top, threshold);
        let limited_ds_y = minmod_tvb(delta_s_y_elem, delta_s_bottom, delta_s_top, threshold);

        // Check if limiting changed the gradients significantly
        let t_x_changed = (limited_dt_x - delta_t_x_elem).abs() > 1e-10;
        let t_y_changed = (limited_dt_y - delta_t_y_elem).abs() > 1e-10;
        let s_x_changed = (limited_ds_x - delta_s_x_elem).abs() > 1e-10;
        let s_y_changed = (limited_ds_y - delta_s_y_elem).abs() > 1e-10;

        if !t_x_changed && !t_y_changed && !s_x_changed && !s_y_changed {
            continue; // No limiting needed
        }

        // Replace high-order content with limited linear reconstruction
        // For each node at reference position (r, s):
        // T(r,s) = T_avg + (dT/dx)*x_local + (dT/dy)*y_local
        // where x_local, y_local are relative to element center
        for i in 0..n_nodes {
            let r = ops.nodes_r[i];
            let s = ops.nodes_s[i];
            // r,s in [-1,1], so normalized position
            let h = swe.get_state(k, i).h.max(h_min);

            // Linear reconstruction from limited gradients
            let t_new = t_avg + 0.5 * limited_dt_x * r + 0.5 * limited_dt_y * s;
            let s_new = s_avg + 0.5 * limited_ds_x * r + 0.5 * limited_ds_y * s;

            tracers.set_conservative(
                k,
                i,
                ConservativeTracerState {
                    h_t: h * t_new,
                    h_s: h * s_new,
                },
            );
        }
    }
}

/// Apply both TVB and positivity limiters to tracer fields.
///
/// This applies limiting in the correct order:
/// 1. TVB limiter (controls oscillations)
/// 2. Positivity limiter (ensures physical bounds)
///
/// # Arguments
/// * `tracers` - Tracer solution to limit (modified in place)
/// * `swe` - SWE solution (for depth h)
/// * `mesh` - 2D mesh
/// * `ops` - DG operators
/// * `tvb` - TVB parameter
/// * `bounds` - Physical bounds for tracers
/// * `h_min` - Minimum depth threshold
pub fn apply_tracer_limiters_2d(
    tracers: &mut TracerSolution2D,
    swe: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    tvb: &TVBParameter2D,
    bounds: &TracerBounds,
    h_min: f64,
) {
    // First apply TVB limiter to control oscillations
    tracer_tvb_limiter_2d(tracers, swe, mesh, ops, tvb, h_min);

    // Then apply positivity limiter to ensure physical bounds
    tracer_positivity_limiter_2d(tracers, swe, ops, bounds, h_min);
}

// =============================================================================
// Kuzmin Vertex-Based Slope Limiter
// =============================================================================

/// Configuration for the Kuzmin vertex-based slope limiter.
///
/// The Kuzmin limiter uses vertex-patch stencils (all elements sharing a vertex)
/// to compute local bounds, providing better oscillation control than face-neighbor
/// based limiters while preserving accuracy in smooth regions.
///
/// # Reference
/// Kuzmin (2010), "A vertex-based hierarchical slope limiter for p-adaptive DG methods"
#[derive(Clone, Copy, Debug)]
pub struct KuzminParameter2D {
    /// Relaxation factor for bounds (1.0 = strict, >1.0 = relaxed).
    /// Values > 1.0 widen the bounds slightly to prevent over-limiting.
    pub relaxation: f64,
}

impl Default for KuzminParameter2D {
    fn default() -> Self {
        Self { relaxation: 1.0 }
    }
}

impl KuzminParameter2D {
    /// Create a new Kuzmin parameter with strict bounds.
    pub fn strict() -> Self {
        Self { relaxation: 1.0 }
    }

    /// Create a Kuzmin parameter with relaxed bounds.
    ///
    /// # Arguments
    /// * `relaxation` - Relaxation factor (1.0 = strict, 1.1 = 10% wider bounds)
    pub fn relaxed(relaxation: f64) -> Self {
        Self {
            relaxation: relaxation.max(1.0),
        }
    }
}

/// Map local vertex index (0-3 in CCW order) to DG node index.
///
/// For tensor-product elements with GLL nodes:
/// ```text
/// Vertex layout (reference element):
/// 3 +-----+ 2
///   |     |
///   |     |
/// 0 +-----+ 1
///
/// Node ordering: k = j * n_1d + i (s varies slowest)
/// ```
///
/// # Arguments
/// * `local_vertex` - Local vertex index (0, 1, 2, or 3)
/// * `n_1d` - Number of nodes in 1D (order + 1)
///
/// # Returns
/// The DG node index corresponding to this vertex
#[inline]
fn vertex_to_node_index(local_vertex: usize, n_1d: usize) -> usize {
    match local_vertex {
        0 => 0,                 // (r=-1, s=-1) -> (i=0, j=0)
        1 => n_1d - 1,          // (r=+1, s=-1) -> (i=n-1, j=0)
        2 => n_1d * n_1d - 1,   // (r=+1, s=+1) -> (i=n-1, j=n-1)
        3 => n_1d * (n_1d - 1), // (r=-1, s=+1) -> (i=0, j=n-1)
        _ => panic!("Invalid local vertex index: {}", local_vertex),
    }
}

/// Compute bounds from a vertex patch.
///
/// # Arguments
/// * `vertex` - Global vertex index
/// * `mesh` - 2D mesh with vertex_to_elements connectivity
/// * `averages` - Cell averages (T, S) for all elements
/// * `relaxation` - Relaxation factor for bounds
///
/// # Returns
/// ((T_min, T_max), (S_min, S_max)) bounds from the vertex patch
fn compute_vertex_bounds(
    vertex: usize,
    mesh: &Mesh2D,
    averages: &[(f64, f64)],
    relaxation: f64,
) -> ((f64, f64), (f64, f64)) {
    let patch = mesh.elements_at_vertex(vertex);

    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    let mut s_min = f64::INFINITY;
    let mut s_max = f64::NEG_INFINITY;

    for &elem in patch {
        let (t_avg, s_avg) = averages[elem];
        t_min = t_min.min(t_avg);
        t_max = t_max.max(t_avg);
        s_min = s_min.min(s_avg);
        s_max = s_max.max(s_avg);
    }

    // Apply relaxation (widens bounds slightly to prevent over-limiting)
    if relaxation > 1.0 {
        let t_range = t_max - t_min;
        let s_range = s_max - s_min;
        let t_expand = 0.5 * t_range * (relaxation - 1.0);
        let s_expand = 0.5 * s_range * (relaxation - 1.0);
        t_min -= t_expand;
        t_max += t_expand;
        s_min -= s_expand;
        s_max += s_expand;
    }

    ((t_min, t_max), (s_min, s_max))
}

/// Compute the limiting factor alpha for a single value.
///
/// Computes the maximum alpha such that:
///   alpha * (value - avg) + avg stays within [bound_min, bound_max]
///
/// # Arguments
/// * `avg` - Cell average
/// * `value` - Nodal value at vertex
/// * `bound_min` - Lower bound from vertex patch
/// * `bound_max` - Upper bound from vertex patch
///
/// # Returns
/// Limiting factor alpha in [0, 1]
#[inline]
fn compute_kuzmin_alpha(avg: f64, value: f64, bound_min: f64, bound_max: f64) -> f64 {
    let deviation = value - avg;

    if deviation.abs() < 1e-14 {
        return 1.0; // Value equals average, no limiting needed
    }

    let mut alpha: f64 = 1.0;

    // Lower bound constraint: alpha * (value - avg) + avg >= bound_min
    if value < bound_min && deviation < 0.0 {
        let required = (avg - bound_min) / (avg - value);
        alpha = alpha.min(required);
    }

    // Upper bound constraint: alpha * (value - avg) + avg <= bound_max
    if value > bound_max && deviation > 0.0 {
        let required = (bound_max - avg) / (value - avg);
        alpha = alpha.min(required);
    }

    alpha.clamp(0.0, 1.0)
}

/// Apply Kuzmin vertex-based slope limiter to tracer fields in 2D.
///
/// Uses vertex-patch stencils (all elements sharing each vertex) to compute
/// local bounds, providing tighter oscillation control than face-neighbor
/// based limiters.
///
/// # Algorithm
/// For each element k:
/// 1. Compute cell average C_avg
/// 2. For each vertex v of element k:
///    a. Gather vertex-patch: all elements sharing vertex v
///    b. Compute vertex bounds: [C_min, C_max] from patch averages
///    c. Get nodal value at vertex
///    d. Compute limiting factor alpha_v to enforce bounds
/// 3. Take minimum alpha across all vertices
/// 4. Apply: C_limited = alpha * (C - C_avg) + C_avg
///
/// # Arguments
/// * `tracers` - Tracer solution to limit (modified in place)
/// * `swe` - SWE solution (for depth h)
/// * `mesh` - 2D mesh with vertex_to_elements connectivity
/// * `ops` - DG operators
/// * `kuzmin` - Kuzmin limiter parameters
/// * `h_min` - Minimum depth threshold
///
/// # Conservation
/// The limiter preserves cell averages exactly (mass-conservative).
pub fn tracer_kuzmin_limiter_2d(
    tracers: &mut TracerSolution2D,
    swe: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    kuzmin: &KuzminParameter2D,
    h_min: f64,
) {
    let n_elements = tracers.n_elements;
    let n_nodes = ops.n_nodes;
    let n_1d = ops.n_1d;

    // Compute all cell averages
    let averages = tracer_cell_averages_2d(tracers, swe, ops, h_min);

    for k in 0..n_elements {
        let (t_avg, s_avg) = averages[k];

        // Check if element is mostly dry - skip limiting
        let avg_h: f64 = (0..n_nodes)
            .map(|i| ops.weights[i] * swe.get_state(k, i).h)
            .sum::<f64>()
            / ops.weights.iter().sum::<f64>();

        if avg_h < h_min {
            continue;
        }

        // Get element vertices
        let vertices = mesh.element_vertex_indices(k);

        // Compute limiting factor for each tracer
        let mut alpha_t = 1.0_f64;
        let mut alpha_s = 1.0_f64;

        for (local_v, &global_v) in vertices.iter().enumerate() {
            // Compute bounds from vertex patch
            let ((t_min, t_max), (s_min, s_max)) =
                compute_vertex_bounds(global_v, mesh, &averages, kuzmin.relaxation);

            // Get nodal value at this vertex
            let node_idx = vertex_to_node_index(local_v, n_1d);
            let h = swe.get_state(k, node_idx).h.max(h_min);
            let conc = tracers.get_concentrations(k, node_idx, h, h_min);

            // Compute limiting factors
            alpha_t = alpha_t.min(compute_kuzmin_alpha(t_avg, conc.temperature, t_min, t_max));
            alpha_s = alpha_s.min(compute_kuzmin_alpha(s_avg, conc.salinity, s_min, s_max));
        }

        // If limiting needed, apply to all nodes
        if alpha_t < 1.0 - 1e-10 || alpha_s < 1.0 - 1e-10 {
            for i in 0..n_nodes {
                let h = swe.get_state(k, i).h.max(h_min);
                let cons = tracers.get_conservative(k, i);

                let t_curr = if h > h_min { cons.h_t / h } else { t_avg };
                let s_curr = if h > h_min { cons.h_s / h } else { s_avg };

                let t_new = alpha_t * (t_curr - t_avg) + t_avg;
                let s_new = alpha_s * (s_curr - s_avg) + s_avg;

                tracers.set_conservative(
                    k,
                    i,
                    ConservativeTracerState {
                        h_t: h * t_new,
                        h_s: h * s_new,
                    },
                );
            }
        }
    }
}

/// Apply Kuzmin and positivity limiters to tracer fields.
///
/// This applies limiting in the correct order:
/// 1. Kuzmin limiter (vertex-based oscillation control)
/// 2. Positivity limiter (ensures physical bounds)
///
/// # Arguments
/// * `tracers` - Tracer solution to limit (modified in place)
/// * `swe` - SWE solution (for depth h)
/// * `mesh` - 2D mesh
/// * `ops` - DG operators
/// * `kuzmin` - Kuzmin limiter parameter
/// * `bounds` - Physical bounds for tracers
/// * `h_min` - Minimum depth threshold
pub fn apply_tracer_limiters_kuzmin_2d(
    tracers: &mut TracerSolution2D,
    swe: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    kuzmin: &KuzminParameter2D,
    bounds: &TracerBounds,
    h_min: f64,
) {
    // First apply Kuzmin limiter to control oscillations
    tracer_kuzmin_limiter_2d(tracers, swe, mesh, ops, kuzmin, h_min);

    // Then apply positivity limiter to ensure physical bounds
    tracer_positivity_limiter_2d(tracers, swe, ops, bounds, h_min);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracer_bounds_default() {
        let bounds = TracerBounds::default();
        assert_eq!(bounds.t_min, -2.0);
        assert_eq!(bounds.t_max, 40.0);
        assert_eq!(bounds.s_min, 0.0);
        assert_eq!(bounds.s_max, 42.0);
    }

    #[test]
    fn test_compute_theta_no_violation() {
        // Average = 10, values in [8, 12], bounds [0, 20]
        let theta = compute_theta(10.0, 8.0, 12.0, 0.0, 20.0);
        assert!((theta - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_theta_lower_violation() {
        // Average = 5, min = -3, bounds [0, 20]
        // Need theta * (-3 - 5) + 5 >= 0
        // -8*theta >= -5 => theta <= 5/8 = 0.625
        let theta = compute_theta(5.0, -3.0, 10.0, 0.0, 20.0);
        assert!((theta - 0.625).abs() < 1e-10);
    }

    #[test]
    fn test_compute_theta_upper_violation() {
        // Average = 15, max = 25, bounds [0, 20]
        // Need theta * (25 - 15) + 15 <= 20
        // 10*theta <= 5 => theta <= 0.5
        let theta = compute_theta(15.0, 10.0, 25.0, 0.0, 20.0);
        assert!((theta - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_minmod_same_sign_positive() {
        assert_eq!(minmod(1.0, 2.0, 3.0), 1.0);
        assert_eq!(minmod(3.0, 2.0, 1.0), 1.0);
    }

    #[test]
    fn test_minmod_same_sign_negative() {
        assert_eq!(minmod(-1.0, -2.0, -3.0), -1.0);
        assert_eq!(minmod(-3.0, -2.0, -1.0), -1.0);
    }

    #[test]
    fn test_minmod_different_signs() {
        assert_eq!(minmod(1.0, -2.0, 3.0), 0.0);
        assert_eq!(minmod(-1.0, 2.0, -3.0), 0.0);
    }

    #[test]
    fn test_minmod_tvb_below_threshold() {
        // First arg below threshold, should return as-is
        let result = minmod_tvb(0.5, 1.0, 2.0, 1.0);
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_minmod_tvb_above_threshold() {
        // First arg above threshold, should apply minmod
        let result = minmod_tvb(1.5, 1.0, 2.0, 1.0);
        assert!((result - 1.0).abs() < 1e-10);
    }

    // =============================================================================
    // Kuzmin Limiter Tests
    // =============================================================================

    #[test]
    fn test_kuzmin_parameter_default() {
        let params = KuzminParameter2D::default();
        assert!((params.relaxation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kuzmin_parameter_relaxed() {
        let params = KuzminParameter2D::relaxed(1.2);
        assert!((params.relaxation - 1.2).abs() < 1e-10);

        // Check that values < 1.0 are clamped to 1.0
        let params2 = KuzminParameter2D::relaxed(0.5);
        assert!((params2.relaxation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vertex_to_node_index_p1() {
        // P1 element: n_1d = 2, nodes at corners
        // Node layout:
        // 2 -- 3
        // |    |
        // 0 -- 1
        assert_eq!(vertex_to_node_index(0, 2), 0); // bottom-left
        assert_eq!(vertex_to_node_index(1, 2), 1); // bottom-right
        assert_eq!(vertex_to_node_index(2, 2), 3); // top-right
        assert_eq!(vertex_to_node_index(3, 2), 2); // top-left
    }

    #[test]
    fn test_vertex_to_node_index_p2() {
        // P2 element: n_1d = 3, 9 nodes total
        // Node layout (column-major-ish, k = j*n_1d + i):
        // 6 -- 7 -- 8
        // |         |
        // 3    4    5
        // |         |
        // 0 -- 1 -- 2
        assert_eq!(vertex_to_node_index(0, 3), 0); // bottom-left
        assert_eq!(vertex_to_node_index(1, 3), 2); // bottom-right
        assert_eq!(vertex_to_node_index(2, 3), 8); // top-right
        assert_eq!(vertex_to_node_index(3, 3), 6); // top-left
    }

    #[test]
    fn test_compute_kuzmin_alpha_no_violation() {
        // Value within bounds, no limiting needed
        let alpha = compute_kuzmin_alpha(10.0, 12.0, 5.0, 20.0);
        assert!((alpha - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_kuzmin_alpha_lower_violation() {
        // avg = 10, value = 3, bounds = [5, 20]
        // Need: alpha * (3 - 10) + 10 >= 5
        // => -7*alpha >= -5 => alpha <= 5/7 ≈ 0.714
        let alpha = compute_kuzmin_alpha(10.0, 3.0, 5.0, 20.0);
        let expected = 5.0 / 7.0;
        assert!(
            (alpha - expected).abs() < 1e-10,
            "alpha = {}, expected = {}",
            alpha,
            expected
        );
    }

    #[test]
    fn test_compute_kuzmin_alpha_upper_violation() {
        // avg = 10, value = 25, bounds = [5, 20]
        // Need: alpha * (25 - 10) + 10 <= 20
        // => 15*alpha <= 10 => alpha <= 10/15 = 2/3
        let alpha = compute_kuzmin_alpha(10.0, 25.0, 5.0, 20.0);
        let expected = 10.0 / 15.0;
        assert!(
            (alpha - expected).abs() < 1e-10,
            "alpha = {}, expected = {}",
            alpha,
            expected
        );
    }

    #[test]
    fn test_compute_kuzmin_alpha_value_equals_average() {
        // When value equals average, no limiting needed
        let alpha = compute_kuzmin_alpha(10.0, 10.0, 5.0, 20.0);
        assert!((alpha - 1.0).abs() < 1e-10);
    }
}
