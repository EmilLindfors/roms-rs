//! Batched GEMM-based volume term computation using faer.
//!
//! This module provides highly optimized volume term computation by batching
//! all elements into single large matrix operations, rather than processing
//! each element individually.
//!
//! # Performance
//!
//! For 65,000 elements with P2 (9 nodes):
//! - Per-element approach: 780,000 small matrix-vector products
//! - Batched approach: 12 large matrix-matrix products
//!
//! The batched approach leverages faer's cache-blocking and SIMD optimizations
//! for large GEMM operations.

use crate::equations::ShallowWater2D;
use crate::operators::DGOperators2D;
use crate::operators::GeometricFactors2D;

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use std::num::NonZeroUsize;

/// Workspace for batched volume term computation.
///
/// Pre-allocates all temporary buffers to avoid allocations in the hot path.
#[derive(Clone)]
pub struct BatchedVolumeWorkspace {
    /// Physical flux in x-direction: F_h, F_hu, F_hv
    pub flux_x_h: Vec<f64>,
    pub flux_x_hu: Vec<f64>,
    pub flux_x_hv: Vec<f64>,
    /// Physical flux in y-direction: G_h, G_hu, G_hv
    pub flux_y_h: Vec<f64>,
    pub flux_y_hu: Vec<f64>,
    pub flux_y_hv: Vec<f64>,
    /// Derivatives: d(flux_x)/dr, d(flux_x)/ds, d(flux_y)/dr, d(flux_y)/ds
    pub dfx_dr_h: Vec<f64>,
    pub dfx_dr_hu: Vec<f64>,
    pub dfx_dr_hv: Vec<f64>,
    pub dfx_ds_h: Vec<f64>,
    pub dfx_ds_hu: Vec<f64>,
    pub dfx_ds_hv: Vec<f64>,
    pub dfy_dr_h: Vec<f64>,
    pub dfy_dr_hu: Vec<f64>,
    pub dfy_dr_hv: Vec<f64>,
    pub dfy_ds_h: Vec<f64>,
    pub dfy_ds_hu: Vec<f64>,
    pub dfy_ds_hv: Vec<f64>,
    /// Pre-transposed differentiation matrices (column-major for faer)
    pub dr_t: Mat<f64>,
    pub ds_t: Mat<f64>,
    /// Dimensions
    n_elements: usize,
    n_nodes: usize,
}

impl BatchedVolumeWorkspace {
    /// Create a new workspace for the given mesh size.
    pub fn new(n_elements: usize, ops: &DGOperators2D) -> Self {
        let n_nodes = ops.n_nodes;
        let total = n_elements * n_nodes;

        // Pre-transpose Dr and Ds for efficient batched GEMM
        // We want: Y = X @ D^T where X is [E, N] and D is [N, N]
        // faer's matmul expects column-major, so we store D^T
        let dr_t = ops.dr.transpose().to_owned();
        let ds_t = ops.ds.transpose().to_owned();

        Self {
            flux_x_h: vec![0.0; total],
            flux_x_hu: vec![0.0; total],
            flux_x_hv: vec![0.0; total],
            flux_y_h: vec![0.0; total],
            flux_y_hu: vec![0.0; total],
            flux_y_hv: vec![0.0; total],
            dfx_dr_h: vec![0.0; total],
            dfx_dr_hu: vec![0.0; total],
            dfx_dr_hv: vec![0.0; total],
            dfx_ds_h: vec![0.0; total],
            dfx_ds_hu: vec![0.0; total],
            dfx_ds_hv: vec![0.0; total],
            dfy_dr_h: vec![0.0; total],
            dfy_dr_hu: vec![0.0; total],
            dfy_dr_hv: vec![0.0; total],
            dfy_ds_h: vec![0.0; total],
            dfy_ds_hu: vec![0.0; total],
            dfy_ds_hv: vec![0.0; total],
            dr_t,
            ds_t,
            n_elements,
            n_nodes,
        }
    }

    /// Resize workspace for a different number of elements.
    pub fn resize(&mut self, n_elements: usize) {
        if n_elements == self.n_elements {
            return;
        }
        let total = n_elements * self.n_nodes;
        self.flux_x_h.resize(total, 0.0);
        self.flux_x_hu.resize(total, 0.0);
        self.flux_x_hv.resize(total, 0.0);
        self.flux_y_h.resize(total, 0.0);
        self.flux_y_hu.resize(total, 0.0);
        self.flux_y_hv.resize(total, 0.0);
        self.dfx_dr_h.resize(total, 0.0);
        self.dfx_dr_hu.resize(total, 0.0);
        self.dfx_dr_hv.resize(total, 0.0);
        self.dfx_ds_h.resize(total, 0.0);
        self.dfx_ds_hu.resize(total, 0.0);
        self.dfx_ds_hv.resize(total, 0.0);
        self.dfy_dr_h.resize(total, 0.0);
        self.dfy_dr_hu.resize(total, 0.0);
        self.dfy_dr_hv.resize(total, 0.0);
        self.dfy_ds_h.resize(total, 0.0);
        self.dfy_ds_hu.resize(total, 0.0);
        self.dfy_ds_hv.resize(total, 0.0);
        self.n_elements = n_elements;
    }
}

/// Compute physical fluxes for all elements (vectorized).
///
/// For SWE, the x-flux is:
///   F_h  = hu
///   F_hu = hu²/h + 0.5*g*h²
///   F_hv = hu*hv/h
///
/// And y-flux is:
///   G_h  = hv
///   G_hu = hu*hv/h
///   G_hv = hv²/h + 0.5*g*h²
#[inline]
fn compute_fluxes_vectorized(
    h: &[f64],
    hu: &[f64],
    hv: &[f64],
    g: f64,
    h_min: f64,
    flux_x_h: &mut [f64],
    flux_x_hu: &mut [f64],
    flux_x_hv: &mut [f64],
    flux_y_h: &mut [f64],
    flux_y_hu: &mut [f64],
    flux_y_hv: &mut [f64],
) {
    let n = h.len();
    debug_assert_eq!(hu.len(), n);
    debug_assert_eq!(hv.len(), n);

    let half_g = 0.5 * g;

    for i in 0..n {
        let hi = h[i].max(h_min);
        let hui = hu[i];
        let hvi = hv[i];

        // Velocity (with dry cell protection)
        let h_inv = 1.0 / hi;
        let ui = hui * h_inv;
        let vi = hvi * h_inv;

        // Pressure term
        let pressure = half_g * hi * hi;

        // X-fluxes
        flux_x_h[i] = hui;
        flux_x_hu[i] = hui * ui + pressure;
        flux_x_hv[i] = hui * vi;

        // Y-fluxes
        flux_y_h[i] = hvi;
        flux_y_hu[i] = hvi * ui;
        flux_y_hv[i] = hvi * vi + pressure;
    }
}

/// Apply batched differentiation matrix using faer GEMM.
///
/// Computes: out = flux @ D^T for all elements
/// where flux is [n_elements, n_nodes] and D^T is [n_nodes, n_nodes]
#[inline]
fn apply_diff_batched(
    flux: &[f64],
    d_t: &Mat<f64>,
    out: &mut [f64],
    n_elements: usize,
    n_nodes: usize,
) {
    // View flux as [n_elements, n_nodes] row-major matrix
    // faer expects column-major, so we interpret this as [n_nodes, n_elements] transposed
    //
    // We want: Y = X @ D^T where X is [E, N] row-major
    // In column-major terms: Y^T = D @ X^T
    // So: out^T = D @ flux^T
    //
    // Actually, let's use a simpler approach: view as column-major [n_nodes, n_elements]
    // Then: out = D^T^T @ flux = D @ flux where both are column-major

    let flux_mat = MatRef::from_column_major_slice(flux, n_nodes, n_elements);
    let mut out_mat = MatMut::from_column_major_slice_mut(out, n_nodes, n_elements);

    // out = D @ flux (both column-major, so this computes derivatives for all elements)
    // Note: d_t is D^T, so we use d_t.transpose() to get D
    matmul(
        &mut out_mat,
        Accum::Replace,
        d_t.as_ref().transpose(),
        flux_mat,
        1.0,
        Par::Rayon(NonZeroUsize::new(rayon::current_num_threads()).unwrap()),
    );
}

/// Combine derivatives with geometric factors to compute divergence.
///
/// div_F = dF_x/dx + dG_y/dy
///       = (dfx_dr * rx + dfx_ds * sx) + (dfy_dr * ry + dfy_ds * sy)
///
/// The volume term is: RHS = -div_F
#[inline]
fn combine_divergence(
    dfx_dr: &[f64],
    dfx_ds: &[f64],
    dfy_dr: &[f64],
    dfy_ds: &[f64],
    geom: &GeometricFactors2D,
    n_nodes: usize,
    rhs: &mut [f64],
) {
    let n_elements = geom.rx.len();

    for k in 0..n_elements {
        let rx = geom.rx[k];
        let ry = geom.ry[k];
        let sx = geom.sx[k];
        let sy = geom.sy[k];

        let base = k * n_nodes;
        for i in 0..n_nodes {
            let idx = base + i;
            // Divergence: dF/dx + dG/dy
            let div = dfx_dr[idx] * rx + dfx_ds[idx] * sx + dfy_dr[idx] * ry + dfy_ds[idx] * sy;
            // Volume term is negative divergence
            rhs[idx] = -div;
        }
    }
}

/// Compute volume terms for all elements using batched GEMM.
///
/// This is the main entry point for batched volume term computation.
/// It computes the volume contribution to the RHS: -∇·F
///
/// # Arguments
/// * `h`, `hu`, `hv` - Solution data slices (SoA layout)
/// * `n_elements` - Number of elements
/// * `n_nodes` - Number of nodes per element
/// * `geom` - Geometric factors for all elements
/// * `equation` - SWE parameters (g, h_min)
/// * `ws` - Pre-allocated workspace
/// * `rhs_h`, `rhs_hu`, `rhs_hv` - Output RHS slices (volume terms only, will be overwritten)
pub fn compute_volume_terms_batched(
    h: &[f64],
    hu: &[f64],
    hv: &[f64],
    n_elements: usize,
    n_nodes: usize,
    geom: &GeometricFactors2D,
    equation: &ShallowWater2D,
    ws: &mut BatchedVolumeWorkspace,
    rhs_h: &mut [f64],
    rhs_hu: &mut [f64],
    rhs_hv: &mut [f64],
) {
    let g = equation.g;
    let h_min = equation.h_min.meters();

    debug_assert_eq!(ws.n_elements, n_elements);
    debug_assert_eq!(ws.n_nodes, n_nodes);

    // Step 1: Compute physical fluxes for all nodes (vectorized)
    compute_fluxes_vectorized(
        h,
        hu,
        hv,
        g,
        h_min,
        &mut ws.flux_x_h,
        &mut ws.flux_x_hu,
        &mut ws.flux_x_hv,
        &mut ws.flux_y_h,
        &mut ws.flux_y_hu,
        &mut ws.flux_y_hv,
    );

    // Step 2: Apply differentiation matrices using batched GEMM
    // d(flux_x)/dr for h, hu, hv
    apply_diff_batched(&ws.flux_x_h, &ws.dr_t, &mut ws.dfx_dr_h, n_elements, n_nodes);
    apply_diff_batched(&ws.flux_x_hu, &ws.dr_t, &mut ws.dfx_dr_hu, n_elements, n_nodes);
    apply_diff_batched(&ws.flux_x_hv, &ws.dr_t, &mut ws.dfx_dr_hv, n_elements, n_nodes);

    // d(flux_x)/ds for h, hu, hv
    apply_diff_batched(&ws.flux_x_h, &ws.ds_t, &mut ws.dfx_ds_h, n_elements, n_nodes);
    apply_diff_batched(&ws.flux_x_hu, &ws.ds_t, &mut ws.dfx_ds_hu, n_elements, n_nodes);
    apply_diff_batched(&ws.flux_x_hv, &ws.ds_t, &mut ws.dfx_ds_hv, n_elements, n_nodes);

    // d(flux_y)/dr for h, hu, hv
    apply_diff_batched(&ws.flux_y_h, &ws.dr_t, &mut ws.dfy_dr_h, n_elements, n_nodes);
    apply_diff_batched(&ws.flux_y_hu, &ws.dr_t, &mut ws.dfy_dr_hu, n_elements, n_nodes);
    apply_diff_batched(&ws.flux_y_hv, &ws.dr_t, &mut ws.dfy_dr_hv, n_elements, n_nodes);

    // d(flux_y)/ds for h, hu, hv
    apply_diff_batched(&ws.flux_y_h, &ws.ds_t, &mut ws.dfy_ds_h, n_elements, n_nodes);
    apply_diff_batched(&ws.flux_y_hu, &ws.ds_t, &mut ws.dfy_ds_hu, n_elements, n_nodes);
    apply_diff_batched(&ws.flux_y_hv, &ws.ds_t, &mut ws.dfy_ds_hv, n_elements, n_nodes);

    // Step 3: Combine derivatives with geometric factors
    combine_divergence(
        &ws.dfx_dr_h,
        &ws.dfx_ds_h,
        &ws.dfy_dr_h,
        &ws.dfy_ds_h,
        geom,
        n_nodes,
        rhs_h,
    );
    combine_divergence(
        &ws.dfx_dr_hu,
        &ws.dfx_ds_hu,
        &ws.dfy_dr_hu,
        &ws.dfy_ds_hu,
        geom,
        n_nodes,
        rhs_hu,
    );
    combine_divergence(
        &ws.dfx_dr_hv,
        &ws.dfx_ds_hv,
        &ws.dfy_dr_hv,
        &ws.dfy_ds_hv,
        geom,
        n_nodes,
        rhs_hv,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh2D;

    #[test]
    fn test_batched_volume_terms_uniform_flow() {
        // Create a simple 2x2 mesh
        let mesh = Mesh2D::uniform_rectangle(0.0, 100.0, 0.0, 100.0, 2, 2);
        let ops = DGOperators2D::new(2); // P2 = 9 nodes
        let geom = crate::operators::GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(9.81);

        let n_elements = mesh.n_elements;
        let n_nodes = ops.n_nodes;
        let total = n_elements * n_nodes;

        // Create uniform flow state
        let h0 = 10.0;
        let u0 = 1.0;
        let v0 = 0.5;
        let h = vec![h0; total];
        let hu = vec![h0 * u0; total];
        let hv = vec![h0 * v0; total];

        // Create workspace and output
        let mut ws = BatchedVolumeWorkspace::new(n_elements, &ops);
        let mut rhs_h = vec![0.0; total];
        let mut rhs_hu = vec![0.0; total];
        let mut rhs_hv = vec![0.0; total];

        // Compute volume terms
        compute_volume_terms_batched(
            &h, &hu, &hv, n_elements, n_nodes, &geom, &equation, &mut ws,
            &mut rhs_h, &mut rhs_hu, &mut rhs_hv,
        );

        // For uniform flow, divergence should be zero (within numerical precision)
        let max_rhs_h: f64 = rhs_h.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let max_rhs_hu: f64 = rhs_hu.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let max_rhs_hv: f64 = rhs_hv.iter().map(|x| x.abs()).fold(0.0, f64::max);

        // Uniform flow should have zero divergence
        assert!(max_rhs_h < 1e-10, "RHS_h should be ~0 for uniform flow, got {}", max_rhs_h);
        assert!(max_rhs_hu < 1e-10, "RHS_hu should be ~0 for uniform flow, got {}", max_rhs_hu);
        assert!(max_rhs_hv < 1e-10, "RHS_hv should be ~0 for uniform flow, got {}", max_rhs_hv);
    }

    #[test]
    fn test_batched_matches_scalar() {
        use crate::solver::simd::kernels::apply_diff_matrix;

        // Create a mesh and operators
        let mesh = Mesh2D::uniform_rectangle(0.0, 100.0, 0.0, 100.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = crate::operators::GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(9.81);

        let n_elements = mesh.n_elements;
        let n_nodes = ops.n_nodes;
        let total = n_elements * n_nodes;

        // Create random-ish state
        let mut h = vec![0.0; total];
        let mut hu = vec![0.0; total];
        let mut hv = vec![0.0; total];
        for i in 0..total {
            h[i] = 10.0 + (i as f64 * 0.1).sin();
            hu[i] = 5.0 + (i as f64 * 0.2).cos();
            hv[i] = 3.0 + (i as f64 * 0.15).sin();
        }

        // Compute using batched approach
        let mut ws = BatchedVolumeWorkspace::new(n_elements, &ops);
        let mut rhs_batched_h = vec![0.0; total];
        let mut rhs_batched_hu = vec![0.0; total];
        let mut rhs_batched_hv = vec![0.0; total];
        compute_volume_terms_batched(
            &h, &hu, &hv, n_elements, n_nodes, &geom, &equation, &mut ws,
            &mut rhs_batched_h, &mut rhs_batched_hu, &mut rhs_batched_hv,
        );

        // Compute using per-element approach (scalar reference)
        let mut rhs_scalar_h = vec![0.0; total];
        let mut rhs_scalar_hu = vec![0.0; total];
        let mut rhs_scalar_hv = vec![0.0; total];

        // Per-element computation
        for k in 0..n_elements {
            let base = k * n_nodes;
            let g = equation.g;
            let h_min = equation.h_min.meters();

            // Compute fluxes for this element
            let mut flux_x_h = vec![0.0; n_nodes];
            let mut flux_x_hu = vec![0.0; n_nodes];
            let mut flux_x_hv = vec![0.0; n_nodes];
            let mut flux_y_h = vec![0.0; n_nodes];
            let mut flux_y_hu = vec![0.0; n_nodes];
            let mut flux_y_hv = vec![0.0; n_nodes];

            for i in 0..n_nodes {
                let idx = base + i;
                let hi = h[idx].max(h_min);
                let hui = hu[idx];
                let hvi = hv[idx];
                let h_inv = 1.0 / hi;
                let ui = hui * h_inv;
                let vi = hvi * h_inv;
                let pressure = 0.5 * g * hi * hi;

                flux_x_h[i] = hui;
                flux_x_hu[i] = hui * ui + pressure;
                flux_x_hv[i] = hui * vi;
                flux_y_h[i] = hvi;
                flux_y_hu[i] = hvi * ui;
                flux_y_hv[i] = hvi * vi + pressure;
            }

            // Apply differentiation matrices
            let mut dfx_dr_h = vec![0.0; n_nodes];
            let mut dfx_dr_hu = vec![0.0; n_nodes];
            let mut dfx_dr_hv = vec![0.0; n_nodes];
            let mut dfx_ds_h = vec![0.0; n_nodes];
            let mut dfx_ds_hu = vec![0.0; n_nodes];
            let mut dfx_ds_hv = vec![0.0; n_nodes];
            let mut dfy_dr_h = vec![0.0; n_nodes];
            let mut dfy_dr_hu = vec![0.0; n_nodes];
            let mut dfy_dr_hv = vec![0.0; n_nodes];
            let mut dfy_ds_h = vec![0.0; n_nodes];
            let mut dfy_ds_hu = vec![0.0; n_nodes];
            let mut dfy_ds_hv = vec![0.0; n_nodes];

            apply_diff_matrix(
                &ops.dr_row_major, &flux_x_h, &flux_x_hu, &flux_x_hv,
                &mut dfx_dr_h, &mut dfx_dr_hu, &mut dfx_dr_hv, n_nodes,
            );
            apply_diff_matrix(
                &ops.ds_row_major, &flux_x_h, &flux_x_hu, &flux_x_hv,
                &mut dfx_ds_h, &mut dfx_ds_hu, &mut dfx_ds_hv, n_nodes,
            );
            apply_diff_matrix(
                &ops.dr_row_major, &flux_y_h, &flux_y_hu, &flux_y_hv,
                &mut dfy_dr_h, &mut dfy_dr_hu, &mut dfy_dr_hv, n_nodes,
            );
            apply_diff_matrix(
                &ops.ds_row_major, &flux_y_h, &flux_y_hu, &flux_y_hv,
                &mut dfy_ds_h, &mut dfy_ds_hu, &mut dfy_ds_hv, n_nodes,
            );

            // Combine with geometric factors
            let rx = geom.rx[k];
            let ry = geom.ry[k];
            let sx = geom.sx[k];
            let sy = geom.sy[k];

            for i in 0..n_nodes {
                let div_h = dfx_dr_h[i] * rx + dfx_ds_h[i] * sx + dfy_dr_h[i] * ry + dfy_ds_h[i] * sy;
                let div_hu = dfx_dr_hu[i] * rx + dfx_ds_hu[i] * sx + dfy_dr_hu[i] * ry + dfy_ds_hu[i] * sy;
                let div_hv = dfx_dr_hv[i] * rx + dfx_ds_hv[i] * sx + dfy_dr_hv[i] * ry + dfy_ds_hv[i] * sy;

                rhs_scalar_h[base + i] = -div_h;
                rhs_scalar_hu[base + i] = -div_hu;
                rhs_scalar_hv[base + i] = -div_hv;
            }
        }

        // Compare results
        let tol = 1e-10;
        for i in 0..total {
            let diff_h = (rhs_batched_h[i] - rhs_scalar_h[i]).abs();
            let diff_hu = (rhs_batched_hu[i] - rhs_scalar_hu[i]).abs();
            let diff_hv = (rhs_batched_hv[i] - rhs_scalar_hv[i]).abs();

            assert!(diff_h < tol, "h mismatch at {}: batched={}, scalar={}", i, rhs_batched_h[i], rhs_scalar_h[i]);
            assert!(diff_hu < tol, "hu mismatch at {}: batched={}, scalar={}", i, rhs_batched_hu[i], rhs_scalar_hu[i]);
            assert!(diff_hv < tol, "hv mismatch at {}: batched={}, scalar={}", i, rhs_batched_hv[i], rhs_scalar_hv[i]);
        }
    }
}
