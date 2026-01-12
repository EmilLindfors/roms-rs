//! SIMD-optimized SWE solution storage using Structure-of-Arrays (SoA) layout.
//!
//! This module provides SIMD-friendly data structures for the 2D Shallow Water
//! Equations solver. The SoA layout enables efficient vectorization of the
//! primary computational kernels.
//!
//! # Data Layout
//!
//! The standard AoS (Array of Structs) layout stores variables interleaved:
//! ```text
//! [h0, hu0, hv0, h1, hu1, hv1, ..., h_n, hu_n, hv_n]
//! ```
//!
//! The SoA layout stores each variable contiguously:
//! ```text
//! h:  [h0, h1, h2, ..., h_n]
//! hu: [hu0, hu1, hu2, ..., hu_n]
//! hv: [hv0, hv1, hv2, ..., hv_n]
//! ```
//!
//! This enables SIMD operations to process 4 (AVX2) or 8 (AVX-512) nodes
//! simultaneously with packed f64 operations.

use super::state_2d::{SWESolution2D, SWEState2D};

/// SoA storage for all elements, optimized for SIMD operations.
///
/// Each variable is stored in a contiguous array, with elements stored
/// sequentially. For element k, nodes 0..n_nodes are at indices
/// `k * n_nodes .. (k+1) * n_nodes`.
#[derive(Clone, Debug)]
pub struct SWESoABuffer {
    /// Water depth h for all elements: [n_elements * n_nodes]
    pub h: Vec<f64>,
    /// X-momentum hu for all elements: [n_elements * n_nodes]
    pub hu: Vec<f64>,
    /// Y-momentum hv for all elements: [n_elements * n_nodes]
    pub hv: Vec<f64>,
    /// Number of elements
    pub n_elements: usize,
    /// Number of nodes per element
    pub n_nodes: usize,
}

impl SWESoABuffer {
    /// Create a new SoA buffer initialized to zero.
    pub fn new(n_elements: usize, n_nodes: usize) -> Self {
        let total = n_elements * n_nodes;
        Self {
            h: vec![0.0; total],
            hu: vec![0.0; total],
            hv: vec![0.0; total],
            n_elements,
            n_nodes,
        }
    }

    /// Create a new SoA buffer with given capacity (uninitialized).
    pub fn with_capacity(n_elements: usize, n_nodes: usize) -> Self {
        let total = n_elements * n_nodes;
        Self {
            h: Vec::with_capacity(total),
            hu: Vec::with_capacity(total),
            hv: Vec::with_capacity(total),
            n_elements,
            n_nodes,
        }
    }

    /// Convert from existing AoS SWESolution2D.
    ///
    /// This performs a full copy, rearranging data from interleaved to
    /// separated layout.
    pub fn from_aos(aos: &SWESolution2D) -> Self {
        let total = aos.n_elements * aos.n_nodes;
        let mut h = Vec::with_capacity(total);
        let mut hu = Vec::with_capacity(total);
        let mut hv = Vec::with_capacity(total);

        for k in 0..aos.n_elements {
            for i in 0..aos.n_nodes {
                let state = aos.get_state(k, i);
                h.push(state.h);
                hu.push(state.hu);
                hv.push(state.hv);
            }
        }

        Self {
            h,
            hu,
            hv,
            n_elements: aos.n_elements,
            n_nodes: aos.n_nodes,
        }
    }

    /// Convert back to AoS SWESolution2D.
    ///
    /// Creates a new solution with the data rearranged to interleaved layout.
    pub fn to_aos(&self) -> SWESolution2D {
        let mut aos = SWESolution2D::new(self.n_elements, self.n_nodes);
        for k in 0..self.n_elements {
            let base = k * self.n_nodes;
            for i in 0..self.n_nodes {
                let idx = base + i;
                aos.set_state(
                    k,
                    i,
                    SWEState2D {
                        h: self.h[idx],
                        hu: self.hu[idx],
                        hv: self.hv[idx],
                    },
                );
            }
        }
        aos
    }

    /// Write SoA data back into an existing AoS solution.
    ///
    /// More efficient than `to_aos()` when reusing existing storage.
    pub fn write_to_aos(&self, aos: &mut SWESolution2D) {
        debug_assert_eq!(aos.n_elements, self.n_elements);
        debug_assert_eq!(aos.n_nodes, self.n_nodes);

        for k in 0..self.n_elements {
            let base = k * self.n_nodes;
            for i in 0..self.n_nodes {
                let idx = base + i;
                aos.set_state(
                    k,
                    i,
                    SWEState2D {
                        h: self.h[idx],
                        hu: self.hu[idx],
                        hv: self.hv[idx],
                    },
                );
            }
        }
    }

    /// Get the index for element k, node i.
    #[inline(always)]
    pub fn index(&self, k: usize, i: usize) -> usize {
        k * self.n_nodes + i
    }

    /// Get state at element k, node i.
    #[inline(always)]
    pub fn get_state(&self, k: usize, i: usize) -> SWEState2D {
        let idx = self.index(k, i);
        SWEState2D {
            h: self.h[idx],
            hu: self.hu[idx],
            hv: self.hv[idx],
        }
    }

    /// Set state at element k, node i.
    #[inline(always)]
    pub fn set_state(&mut self, k: usize, i: usize, state: SWEState2D) {
        let idx = self.index(k, i);
        self.h[idx] = state.h;
        self.hu[idx] = state.hu;
        self.hv[idx] = state.hv;
    }

    /// Get slice of h values for element k.
    #[inline(always)]
    pub fn element_h(&self, k: usize) -> &[f64] {
        let start = k * self.n_nodes;
        &self.h[start..start + self.n_nodes]
    }

    /// Get slice of hu values for element k.
    #[inline(always)]
    pub fn element_hu(&self, k: usize) -> &[f64] {
        let start = k * self.n_nodes;
        &self.hu[start..start + self.n_nodes]
    }

    /// Get slice of hv values for element k.
    #[inline(always)]
    pub fn element_hv(&self, k: usize) -> &[f64] {
        let start = k * self.n_nodes;
        &self.hv[start..start + self.n_nodes]
    }

    /// Get mutable slice of h values for element k.
    #[inline(always)]
    pub fn element_h_mut(&mut self, k: usize) -> &mut [f64] {
        let start = k * self.n_nodes;
        &mut self.h[start..start + self.n_nodes]
    }

    /// Get mutable slice of hu values for element k.
    #[inline(always)]
    pub fn element_hu_mut(&mut self, k: usize) -> &mut [f64] {
        let start = k * self.n_nodes;
        &mut self.hu[start..start + self.n_nodes]
    }

    /// Get mutable slice of hv values for element k.
    #[inline(always)]
    pub fn element_hv_mut(&mut self, k: usize) -> &mut [f64] {
        let start = k * self.n_nodes;
        &mut self.hv[start..start + self.n_nodes]
    }

    /// Total number of values stored (n_elements * n_nodes).
    pub fn len(&self) -> usize {
        self.h.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.h.is_empty()
    }

    /// Fill all values with zero.
    pub fn clear(&mut self) {
        self.h.fill(0.0);
        self.hu.fill(0.0);
        self.hv.fill(0.0);
    }

    /// Scale all values by a constant.
    pub fn scale(&mut self, c: f64) {
        for v in &mut self.h {
            *v *= c;
        }
        for v in &mut self.hu {
            *v *= c;
        }
        for v in &mut self.hv {
            *v *= c;
        }
    }

    /// Add c * other to self (axpy operation).
    pub fn axpy(&mut self, c: f64, other: &Self) {
        debug_assert_eq!(self.len(), other.len());
        for (a, b) in self.h.iter_mut().zip(other.h.iter()) {
            *a += c * *b;
        }
        for (a, b) in self.hu.iter_mut().zip(other.hu.iter()) {
            *a += c * *b;
        }
        for (a, b) in self.hv.iter_mut().zip(other.hv.iter()) {
            *a += c * *b;
        }
    }

    /// Copy from another buffer.
    pub fn copy_from(&mut self, other: &Self) {
        debug_assert_eq!(self.len(), other.len());
        self.h.copy_from_slice(&other.h);
        self.hu.copy_from_slice(&other.hu);
        self.hv.copy_from_slice(&other.hv);
    }
}

/// Per-element workspace for SIMD operations.
///
/// This workspace provides aligned, contiguous storage for intermediate
/// computations within a single element. It is designed to be reused
/// across elements to avoid repeated allocations.
///
/// The workspace is cache-line aligned (64 bytes) to optimize memory
/// access patterns on modern CPUs.
#[repr(align(64))]
#[derive(Clone, Debug)]
pub struct SIMDWorkspace {
    /// Flux in x-direction: h component
    pub flux_x_h: Vec<f64>,
    /// Flux in x-direction: hu component
    pub flux_x_hu: Vec<f64>,
    /// Flux in x-direction: hv component
    pub flux_x_hv: Vec<f64>,

    /// Flux in y-direction: h component
    pub flux_y_h: Vec<f64>,
    /// Flux in y-direction: hu component
    pub flux_y_hu: Vec<f64>,
    /// Flux in y-direction: hv component
    pub flux_y_hv: Vec<f64>,

    /// Derivative of flux_x w.r.t. r: [h, hu, hv]
    pub dfx_dr_h: Vec<f64>,
    pub dfx_dr_hu: Vec<f64>,
    pub dfx_dr_hv: Vec<f64>,

    /// Derivative of flux_x w.r.t. s: [h, hu, hv]
    pub dfx_ds_h: Vec<f64>,
    pub dfx_ds_hu: Vec<f64>,
    pub dfx_ds_hv: Vec<f64>,

    /// Derivative of flux_y w.r.t. r: [h, hu, hv]
    pub dfy_dr_h: Vec<f64>,
    pub dfy_dr_hu: Vec<f64>,
    pub dfy_dr_hv: Vec<f64>,

    /// Derivative of flux_y w.r.t. s: [h, hu, hv]
    pub dfy_ds_h: Vec<f64>,
    pub dfy_ds_hu: Vec<f64>,
    pub dfy_ds_hv: Vec<f64>,

    /// RHS accumulator
    pub rhs_h: Vec<f64>,
    pub rhs_hu: Vec<f64>,
    pub rhs_hv: Vec<f64>,

    /// Number of nodes this workspace is sized for
    pub n_nodes: usize,
}

impl SIMDWorkspace {
    /// Create a new workspace for elements with `n_nodes` nodes.
    pub fn new(n_nodes: usize) -> Self {
        Self {
            flux_x_h: vec![0.0; n_nodes],
            flux_x_hu: vec![0.0; n_nodes],
            flux_x_hv: vec![0.0; n_nodes],
            flux_y_h: vec![0.0; n_nodes],
            flux_y_hu: vec![0.0; n_nodes],
            flux_y_hv: vec![0.0; n_nodes],
            dfx_dr_h: vec![0.0; n_nodes],
            dfx_dr_hu: vec![0.0; n_nodes],
            dfx_dr_hv: vec![0.0; n_nodes],
            dfx_ds_h: vec![0.0; n_nodes],
            dfx_ds_hu: vec![0.0; n_nodes],
            dfx_ds_hv: vec![0.0; n_nodes],
            dfy_dr_h: vec![0.0; n_nodes],
            dfy_dr_hu: vec![0.0; n_nodes],
            dfy_dr_hv: vec![0.0; n_nodes],
            dfy_ds_h: vec![0.0; n_nodes],
            dfy_ds_hu: vec![0.0; n_nodes],
            dfy_ds_hv: vec![0.0; n_nodes],
            rhs_h: vec![0.0; n_nodes],
            rhs_hu: vec![0.0; n_nodes],
            rhs_hv: vec![0.0; n_nodes],
            n_nodes,
        }
    }

    /// Clear all workspace arrays to zero.
    pub fn clear(&mut self) {
        self.flux_x_h.fill(0.0);
        self.flux_x_hu.fill(0.0);
        self.flux_x_hv.fill(0.0);
        self.flux_y_h.fill(0.0);
        self.flux_y_hu.fill(0.0);
        self.flux_y_hv.fill(0.0);
        self.dfx_dr_h.fill(0.0);
        self.dfx_dr_hu.fill(0.0);
        self.dfx_dr_hv.fill(0.0);
        self.dfx_ds_h.fill(0.0);
        self.dfx_ds_hu.fill(0.0);
        self.dfx_ds_hv.fill(0.0);
        self.dfy_dr_h.fill(0.0);
        self.dfy_dr_hu.fill(0.0);
        self.dfy_dr_hv.fill(0.0);
        self.dfy_ds_h.fill(0.0);
        self.dfy_ds_hu.fill(0.0);
        self.dfy_ds_hv.fill(0.0);
        self.rhs_h.fill(0.0);
        self.rhs_hu.fill(0.0);
        self.rhs_hv.fill(0.0);
    }

    /// Clear only the RHS accumulator arrays.
    pub fn clear_rhs(&mut self) {
        self.rhs_h.fill(0.0);
        self.rhs_hu.fill(0.0);
        self.rhs_hv.fill(0.0);
    }
}

/// Face-local workspace for flux computations.
///
/// Stores intermediate values for flux computation at face nodes.
#[derive(Clone, Debug)]
pub struct FaceWorkspace {
    /// Interior state h at face nodes
    pub h_int: Vec<f64>,
    /// Interior state hu at face nodes
    pub hu_int: Vec<f64>,
    /// Interior state hv at face nodes
    pub hv_int: Vec<f64>,

    /// Exterior state h at face nodes
    pub h_ext: Vec<f64>,
    /// Exterior state hu at face nodes
    pub hu_ext: Vec<f64>,
    /// Exterior state hv at face nodes
    pub hv_ext: Vec<f64>,

    /// Flux difference h at face nodes
    pub flux_diff_h: Vec<f64>,
    /// Flux difference hu at face nodes
    pub flux_diff_hu: Vec<f64>,
    /// Flux difference hv at face nodes
    pub flux_diff_hv: Vec<f64>,

    /// Number of face nodes
    pub n_face_nodes: usize,
}

impl FaceWorkspace {
    /// Create a new face workspace for `n_face_nodes` nodes per face.
    pub fn new(n_face_nodes: usize) -> Self {
        Self {
            h_int: vec![0.0; n_face_nodes],
            hu_int: vec![0.0; n_face_nodes],
            hv_int: vec![0.0; n_face_nodes],
            h_ext: vec![0.0; n_face_nodes],
            hu_ext: vec![0.0; n_face_nodes],
            hv_ext: vec![0.0; n_face_nodes],
            flux_diff_h: vec![0.0; n_face_nodes],
            flux_diff_hu: vec![0.0; n_face_nodes],
            flux_diff_hv: vec![0.0; n_face_nodes],
            n_face_nodes,
        }
    }

    /// Clear all arrays to zero.
    pub fn clear(&mut self) {
        self.h_int.fill(0.0);
        self.hu_int.fill(0.0);
        self.hv_int.fill(0.0);
        self.h_ext.fill(0.0);
        self.hu_ext.fill(0.0);
        self.hv_ext.fill(0.0);
        self.flux_diff_h.fill(0.0);
        self.flux_diff_hu.fill(0.0);
        self.flux_diff_hv.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soa_buffer_new() {
        let buf = SWESoABuffer::new(10, 16);
        assert_eq!(buf.n_elements, 10);
        assert_eq!(buf.n_nodes, 16);
        assert_eq!(buf.h.len(), 160);
        assert_eq!(buf.hu.len(), 160);
        assert_eq!(buf.hv.len(), 160);
    }

    #[test]
    fn test_soa_buffer_get_set() {
        let mut buf = SWESoABuffer::new(2, 4);

        let state = SWEState2D::new(1.5, 0.5, -0.3);
        buf.set_state(1, 2, state);

        let retrieved = buf.get_state(1, 2);
        assert_eq!(retrieved.h, 1.5);
        assert_eq!(retrieved.hu, 0.5);
        assert_eq!(retrieved.hv, -0.3);
    }

    #[test]
    fn test_soa_aos_conversion() {
        // Create AoS with known data
        let mut aos = SWESolution2D::new(2, 4);
        for k in 0..2 {
            for i in 0..4 {
                let val = (k * 4 + i) as f64;
                aos.set_state(k, i, SWEState2D::new(val, val * 2.0, val * 3.0));
            }
        }

        // Convert to SoA
        let soa = SWESoABuffer::from_aos(&aos);

        // Verify all values match
        for k in 0..2 {
            for i in 0..4 {
                let aos_state = aos.get_state(k, i);
                let soa_state = soa.get_state(k, i);
                assert_eq!(aos_state.h, soa_state.h);
                assert_eq!(aos_state.hu, soa_state.hu);
                assert_eq!(aos_state.hv, soa_state.hv);
            }
        }

        // Convert back to AoS
        let aos2 = soa.to_aos();

        // Verify round-trip
        for k in 0..2 {
            for i in 0..4 {
                let orig = aos.get_state(k, i);
                let roundtrip = aos2.get_state(k, i);
                assert_eq!(orig.h, roundtrip.h);
                assert_eq!(orig.hu, roundtrip.hu);
                assert_eq!(orig.hv, roundtrip.hv);
            }
        }
    }

    #[test]
    fn test_soa_element_slices() {
        let mut buf = SWESoABuffer::new(3, 4);

        // Set element 1 data
        for i in 0..4 {
            buf.set_state(1, i, SWEState2D::new(i as f64, 0.0, 0.0));
        }

        // Check element slice
        let h_slice = buf.element_h(1);
        assert_eq!(h_slice.len(), 4);
        assert_eq!(h_slice[0], 0.0);
        assert_eq!(h_slice[1], 1.0);
        assert_eq!(h_slice[2], 2.0);
        assert_eq!(h_slice[3], 3.0);

        // Modify via mutable slice
        let h_mut = buf.element_h_mut(1);
        h_mut[0] = 10.0;
        assert_eq!(buf.get_state(1, 0).h, 10.0);
    }

    #[test]
    fn test_soa_axpy() {
        let mut a = SWESoABuffer::new(2, 2);
        let mut b = SWESoABuffer::new(2, 2);

        // Set values
        for k in 0..2 {
            for i in 0..2 {
                a.set_state(k, i, SWEState2D::new(1.0, 2.0, 3.0));
                b.set_state(k, i, SWEState2D::new(0.5, 1.0, 1.5));
            }
        }

        // axpy: a += 2.0 * b
        a.axpy(2.0, &b);

        let state = a.get_state(0, 0);
        assert_eq!(state.h, 2.0); // 1.0 + 2.0 * 0.5
        assert_eq!(state.hu, 4.0); // 2.0 + 2.0 * 1.0
        assert_eq!(state.hv, 6.0); // 3.0 + 2.0 * 1.5
    }

    #[test]
    fn test_workspace_creation() {
        let ws = SIMDWorkspace::new(16);
        assert_eq!(ws.n_nodes, 16);
        assert_eq!(ws.flux_x_h.len(), 16);
        assert_eq!(ws.rhs_hu.len(), 16);
    }

    #[test]
    fn test_workspace_clear() {
        let mut ws = SIMDWorkspace::new(4);
        ws.flux_x_h[0] = 1.0;
        ws.rhs_hu[2] = 5.0;

        ws.clear();

        assert_eq!(ws.flux_x_h[0], 0.0);
        assert_eq!(ws.rhs_hu[2], 0.0);
    }

    #[test]
    fn test_face_workspace() {
        let fw = FaceWorkspace::new(4);
        assert_eq!(fw.n_face_nodes, 4);
        assert_eq!(fw.h_int.len(), 4);
        assert_eq!(fw.flux_diff_hv.len(), 4);
    }
}
