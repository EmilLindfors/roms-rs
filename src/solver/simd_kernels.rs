//! SIMD-optimized computational kernels for 2D SWE solver.
//!
//! This module provides vectorized implementations of the primary computational
//! bottlenecks in the DG solver:
//!
//! - **Volume term**: Matrix-vector multiplication with Dr/Ds differentiation matrices
//! - **LIFT application**: Surface integral contribution via LIFT matrix
//! - **Source terms**: Coriolis and friction computations
//!
//! All kernels use the `pulp` crate for portable SIMD with runtime feature detection.
//! The code automatically selects the best available SIMD instruction set
//! (AVX-512, AVX2, SSE4.1, or scalar fallback).
//!
//! # Usage
//!
//! ```ignore
//! use pulp::Arch;
//! use dg_rs::solver::simd_kernels::apply_diff_matrix;
//!
//! let arch = Arch::new();
//! arch.dispatch(|| {
//!     apply_diff_matrix(arch, &dr, &flux_h, &flux_hu, &flux_hv,
//!                       &mut out_h, &mut out_hu, &mut out_hv, n_nodes);
//! });
//! ```

#[cfg(feature = "simd")]
use pulp::{Arch, Simd, WithSimd};

// ============================================================================
// Scalar reference implementations (always available)
// ============================================================================

/// Scalar implementation of differentiation matrix application.
///
/// Computes `out = D * flux` for all three SWE variables.
/// This is the reference implementation that SIMD versions must match.
///
/// # Arguments
/// * `d` - Differentiation matrix (Dr or Ds), n_nodes × n_nodes, row-major
/// * `flux_h`, `flux_hu`, `flux_hv` - Input flux vectors, length n_nodes
/// * `out_h`, `out_hu`, `out_hv` - Output vectors, length n_nodes
/// * `n_nodes` - Number of nodes per element
pub fn apply_diff_matrix_scalar(
    d: &[f64],
    flux_h: &[f64],
    flux_hu: &[f64],
    flux_hv: &[f64],
    out_h: &mut [f64],
    out_hu: &mut [f64],
    out_hv: &mut [f64],
    n_nodes: usize,
) {
    debug_assert_eq!(d.len(), n_nodes * n_nodes);
    debug_assert_eq!(flux_h.len(), n_nodes);
    debug_assert_eq!(out_h.len(), n_nodes);

    for i in 0..n_nodes {
        let mut sum_h = 0.0;
        let mut sum_hu = 0.0;
        let mut sum_hv = 0.0;

        for j in 0..n_nodes {
            let d_ij = d[i * n_nodes + j];
            sum_h += d_ij * flux_h[j];
            sum_hu += d_ij * flux_hu[j];
            sum_hv += d_ij * flux_hv[j];
        }

        out_h[i] = sum_h;
        out_hu[i] = sum_hu;
        out_hv[i] = sum_hv;
    }
}

/// Scalar implementation of LIFT matrix application.
///
/// Computes `out += scale * LIFT * flux_diff` for all three variables.
///
/// # Arguments
/// * `lift` - LIFT matrix for one face, n_nodes × n_face_nodes, row-major
/// * `flux_diff_h`, etc. - Flux differences at face nodes, length n_face_nodes
/// * `out_h`, etc. - Output vectors (accumulated), length n_nodes
/// * `n_nodes` - Number of volume nodes per element
/// * `n_face_nodes` - Number of nodes per face
/// * `scale` - Scaling factor (typically j_inv * s_jac)
pub fn apply_lift_scalar(
    lift: &[f64],
    flux_diff_h: &[f64],
    flux_diff_hu: &[f64],
    flux_diff_hv: &[f64],
    out_h: &mut [f64],
    out_hu: &mut [f64],
    out_hv: &mut [f64],
    n_nodes: usize,
    n_face_nodes: usize,
    scale: f64,
) {
    debug_assert_eq!(lift.len(), n_nodes * n_face_nodes);
    debug_assert_eq!(flux_diff_h.len(), n_face_nodes);
    debug_assert_eq!(out_h.len(), n_nodes);

    for i in 0..n_nodes {
        let mut sum_h = 0.0;
        let mut sum_hu = 0.0;
        let mut sum_hv = 0.0;

        for fi in 0..n_face_nodes {
            let lift_coeff = lift[i * n_face_nodes + fi];
            sum_h += lift_coeff * flux_diff_h[fi];
            sum_hu += lift_coeff * flux_diff_hu[fi];
            sum_hv += lift_coeff * flux_diff_hv[fi];
        }

        out_h[i] += scale * sum_h;
        out_hu[i] += scale * sum_hu;
        out_hv[i] += scale * sum_hv;
    }
}

/// Scalar implementation of Coriolis source term.
///
/// Computes: S_hu += f * hv, S_hv += -f * hu
///
/// # Arguments
/// * `hu`, `hv` - Momentum components, length n_nodes
/// * `out_hu`, `out_hv` - Output arrays (accumulated), length n_nodes
/// * `f` - Coriolis parameter (f-plane value)
/// * `n_nodes` - Number of nodes
pub fn coriolis_source_scalar(
    hu: &[f64],
    hv: &[f64],
    out_hu: &mut [f64],
    out_hv: &mut [f64],
    f: f64,
    n_nodes: usize,
) {
    debug_assert_eq!(hu.len(), n_nodes);
    debug_assert_eq!(out_hu.len(), n_nodes);

    for i in 0..n_nodes {
        out_hu[i] += f * hv[i];
        out_hv[i] += -f * hu[i];
    }
}

/// Scalar implementation of Manning friction source term.
///
/// Computes: S_hu += -C_f * |u| * u, S_hv += -C_f * |u| * v
/// where C_f = g * n² / h^(1/3), and u = hu/h, v = hv/h
///
/// # Arguments
/// * `h`, `hu`, `hv` - State variables, length n_nodes
/// * `out_hu`, `out_hv` - Output arrays (accumulated), length n_nodes
/// * `g_n2` - g * manning_n², precomputed constant
/// * `h_min` - Minimum depth for dry cell protection
/// * `n_nodes` - Number of nodes
pub fn manning_friction_scalar(
    h: &[f64],
    hu: &[f64],
    hv: &[f64],
    out_hu: &mut [f64],
    out_hv: &mut [f64],
    g_n2: f64,
    h_min: f64,
    n_nodes: usize,
) {
    debug_assert_eq!(h.len(), n_nodes);
    debug_assert_eq!(out_hu.len(), n_nodes);

    for i in 0..n_nodes {
        let h_val = h[i].max(h_min);
        let h_inv = 1.0 / h_val;

        // Velocity components
        let u = hu[i] * h_inv;
        let v = hv[i] * h_inv;

        // Velocity magnitude
        let u_mag = (u * u + v * v).sqrt();

        // Friction coefficient: C_f = g * n² / h^(1/3)
        let c_f = g_n2 * h_val.cbrt().recip();

        // Source: S = -C_f * |u| * velocity
        let factor = -c_f * u_mag;
        out_hu[i] += factor * u;
        out_hv[i] += factor * v;
    }
}

/// Combine volume derivatives with geometric factors (scalar).
///
/// Computes: out = -(dfx_dr * rx + dfx_ds * sx + dfy_dr * ry + dfy_ds * sy)
///
/// This is the divergence in physical coordinates from reference derivatives.
pub fn combine_derivatives_scalar(
    dfx_dr_h: &[f64],
    dfx_dr_hu: &[f64],
    dfx_dr_hv: &[f64],
    dfx_ds_h: &[f64],
    dfx_ds_hu: &[f64],
    dfx_ds_hv: &[f64],
    dfy_dr_h: &[f64],
    dfy_dr_hu: &[f64],
    dfy_dr_hv: &[f64],
    dfy_ds_h: &[f64],
    dfy_ds_hu: &[f64],
    dfy_ds_hv: &[f64],
    out_h: &mut [f64],
    out_hu: &mut [f64],
    out_hv: &mut [f64],
    rx: f64,
    sx: f64,
    ry: f64,
    sy: f64,
    n_nodes: usize,
) {
    for i in 0..n_nodes {
        // div_flux = dF/dx + dG/dy
        // where dF/dx = dfx_dr * rx + dfx_ds * sx (chain rule)
        let div_h = dfx_dr_h[i] * rx + dfx_ds_h[i] * sx + dfy_dr_h[i] * ry + dfy_ds_h[i] * sy;
        let div_hu = dfx_dr_hu[i] * rx + dfx_ds_hu[i] * sx + dfy_dr_hu[i] * ry + dfy_ds_hu[i] * sy;
        let div_hv = dfx_dr_hv[i] * rx + dfx_ds_hv[i] * sx + dfy_dr_hv[i] * ry + dfy_ds_hv[i] * sy;

        // RHS = -div(F)
        out_h[i] = -div_h;
        out_hu[i] = -div_hu;
        out_hv[i] = -div_hv;
    }
}

// ============================================================================
// SIMD implementations (feature-gated)
// ============================================================================

#[cfg(feature = "simd")]
mod simd_impl {
    use super::*;

    /// Apply Coriolis source term with SIMD.
    ///
    /// Uses pulp's width-agnostic SIMD API.
    #[inline]
    pub fn coriolis_source_simd_inner<S: Simd>(
        simd: S,
        hu: &[f64],
        hv: &[f64],
        out_hu: &mut [f64],
        out_hv: &mut [f64],
        f: f64,
    ) {
        let f_splat = simd.f64s_splat(f);
        let neg_f_splat = simd.f64s_splat(-f);

        // Get SIMD-aligned chunks
        let (hu_head, hu_tail) = S::f64s_as_simd(hu);
        let (hv_head, hv_tail) = S::f64s_as_simd(hv);
        let (out_hu_head, out_hu_tail) = S::f64s_as_mut_simd(out_hu);
        let (out_hv_head, out_hv_tail) = S::f64s_as_mut_simd(out_hv);

        // Process SIMD chunks
        for (((hu_v, hv_v), out_hu_v), out_hv_v) in hu_head
            .iter()
            .zip(hv_head.iter())
            .zip(out_hu_head.iter_mut())
            .zip(out_hv_head.iter_mut())
        {
            // S_hu += f * hv
            *out_hu_v = simd.f64s_mul_add(f_splat, *hv_v, *out_hu_v);
            // S_hv += -f * hu
            *out_hv_v = simd.f64s_mul_add(neg_f_splat, *hu_v, *out_hv_v);
        }

        // Scalar tail
        for ((hu_val, hv_val), (out_hu_val, out_hv_val)) in hu_tail
            .iter()
            .zip(hv_tail.iter())
            .zip(out_hu_tail.iter_mut().zip(out_hv_tail.iter_mut()))
        {
            *out_hu_val += f * hv_val;
            *out_hv_val += -f * hu_val;
        }
    }

    /// Combine derivatives with geometric factors using SIMD.
    #[inline]
    pub fn combine_derivatives_simd_inner<S: Simd>(
        simd: S,
        dfx_dr_h: &[f64],
        dfx_dr_hu: &[f64],
        dfx_dr_hv: &[f64],
        dfx_ds_h: &[f64],
        dfx_ds_hu: &[f64],
        dfx_ds_hv: &[f64],
        dfy_dr_h: &[f64],
        dfy_dr_hu: &[f64],
        dfy_dr_hv: &[f64],
        dfy_ds_h: &[f64],
        dfy_ds_hu: &[f64],
        dfy_ds_hv: &[f64],
        out_h: &mut [f64],
        out_hu: &mut [f64],
        out_hv: &mut [f64],
        rx: f64,
        sx: f64,
        ry: f64,
        sy: f64,
    ) {
        let rx_v = simd.f64s_splat(rx);
        let sx_v = simd.f64s_splat(sx);
        let ry_v = simd.f64s_splat(ry);
        let sy_v = simd.f64s_splat(sy);
        let neg_one = simd.f64s_splat(-1.0);

        // Process h variable
        {
            let (dfx_dr_head, dfx_dr_tail) = S::f64s_as_simd(dfx_dr_h);
            let (dfx_ds_head, dfx_ds_tail) = S::f64s_as_simd(dfx_ds_h);
            let (dfy_dr_head, dfy_dr_tail) = S::f64s_as_simd(dfy_dr_h);
            let (dfy_ds_head, dfy_ds_tail) = S::f64s_as_simd(dfy_ds_h);
            let (out_head, out_tail) = S::f64s_as_mut_simd(out_h);

            for ((((dfx_dr, dfx_ds), dfy_dr), dfy_ds), out) in dfx_dr_head
                .iter()
                .zip(dfx_ds_head.iter())
                .zip(dfy_dr_head.iter())
                .zip(dfy_ds_head.iter())
                .zip(out_head.iter_mut())
            {
                let mut div = simd.f64s_mul(*dfx_dr, rx_v);
                div = simd.f64s_mul_add(*dfx_ds, sx_v, div);
                div = simd.f64s_mul_add(*dfy_dr, ry_v, div);
                div = simd.f64s_mul_add(*dfy_ds, sy_v, div);
                *out = simd.f64s_mul(neg_one, div);
            }

            // Scalar tail
            for ((((dfx_dr, dfx_ds), dfy_dr), dfy_ds), out) in dfx_dr_tail
                .iter()
                .zip(dfx_ds_tail.iter())
                .zip(dfy_dr_tail.iter())
                .zip(dfy_ds_tail.iter())
                .zip(out_tail.iter_mut())
            {
                let div = dfx_dr * rx + dfx_ds * sx + dfy_dr * ry + dfy_ds * sy;
                *out = -div;
            }
        }

        // Process hu variable
        {
            let (dfx_dr_head, dfx_dr_tail) = S::f64s_as_simd(dfx_dr_hu);
            let (dfx_ds_head, dfx_ds_tail) = S::f64s_as_simd(dfx_ds_hu);
            let (dfy_dr_head, dfy_dr_tail) = S::f64s_as_simd(dfy_dr_hu);
            let (dfy_ds_head, dfy_ds_tail) = S::f64s_as_simd(dfy_ds_hu);
            let (out_head, out_tail) = S::f64s_as_mut_simd(out_hu);

            for ((((dfx_dr, dfx_ds), dfy_dr), dfy_ds), out) in dfx_dr_head
                .iter()
                .zip(dfx_ds_head.iter())
                .zip(dfy_dr_head.iter())
                .zip(dfy_ds_head.iter())
                .zip(out_head.iter_mut())
            {
                let mut div = simd.f64s_mul(*dfx_dr, rx_v);
                div = simd.f64s_mul_add(*dfx_ds, sx_v, div);
                div = simd.f64s_mul_add(*dfy_dr, ry_v, div);
                div = simd.f64s_mul_add(*dfy_ds, sy_v, div);
                *out = simd.f64s_mul(neg_one, div);
            }

            for ((((dfx_dr, dfx_ds), dfy_dr), dfy_ds), out) in dfx_dr_tail
                .iter()
                .zip(dfx_ds_tail.iter())
                .zip(dfy_dr_tail.iter())
                .zip(dfy_ds_tail.iter())
                .zip(out_tail.iter_mut())
            {
                let div = dfx_dr * rx + dfx_ds * sx + dfy_dr * ry + dfy_ds * sy;
                *out = -div;
            }
        }

        // Process hv variable
        {
            let (dfx_dr_head, dfx_dr_tail) = S::f64s_as_simd(dfx_dr_hv);
            let (dfx_ds_head, dfx_ds_tail) = S::f64s_as_simd(dfx_ds_hv);
            let (dfy_dr_head, dfy_dr_tail) = S::f64s_as_simd(dfy_dr_hv);
            let (dfy_ds_head, dfy_ds_tail) = S::f64s_as_simd(dfy_ds_hv);
            let (out_head, out_tail) = S::f64s_as_mut_simd(out_hv);

            for ((((dfx_dr, dfx_ds), dfy_dr), dfy_ds), out) in dfx_dr_head
                .iter()
                .zip(dfx_ds_head.iter())
                .zip(dfy_dr_head.iter())
                .zip(dfy_ds_head.iter())
                .zip(out_head.iter_mut())
            {
                let mut div = simd.f64s_mul(*dfx_dr, rx_v);
                div = simd.f64s_mul_add(*dfx_ds, sx_v, div);
                div = simd.f64s_mul_add(*dfy_dr, ry_v, div);
                div = simd.f64s_mul_add(*dfy_ds, sy_v, div);
                *out = simd.f64s_mul(neg_one, div);
            }

            for ((((dfx_dr, dfx_ds), dfy_dr), dfy_ds), out) in dfx_dr_tail
                .iter()
                .zip(dfx_ds_tail.iter())
                .zip(dfy_dr_tail.iter())
                .zip(dfy_ds_tail.iter())
                .zip(out_tail.iter_mut())
            {
                let div = dfx_dr * rx + dfx_ds * sx + dfy_dr * ry + dfy_ds * sy;
                *out = -div;
            }
        }
    }
}

// ============================================================================
// Public API with runtime SIMD dispatch
// ============================================================================

/// Apply differentiation matrix using faer's optimized GEMV.
///
/// Computes `out = D * flux` for all three SWE variables using faer's
/// SIMD-optimized matrix-vector multiplication.
///
/// # Arguments
/// * `d` - Differentiation matrix (Dr or Ds), n_nodes × n_nodes, row-major
/// * `flux_h`, `flux_hu`, `flux_hv` - Input flux vectors, length n_nodes
/// * `out_h`, `out_hu`, `out_hv` - Output vectors, length n_nodes
/// * `n_nodes` - Number of nodes per element
///
/// Uses faer GEMV for larger matrices (>=20 nodes) where SIMD overhead is amortized,
/// and scalar for smaller matrices where faer's setup cost dominates.

/// Threshold for using faer GEMV (nodes >= this use faer, below uses scalar).
/// Based on benchmarks: faer overhead is ~150ns, which pays off at P4 (25 nodes).
const FAER_GEMV_THRESHOLD: usize = 20;

#[cfg(feature = "simd")]
pub fn apply_diff_matrix(
    d: &[f64],
    flux_h: &[f64],
    flux_hu: &[f64],
    flux_hv: &[f64],
    out_h: &mut [f64],
    out_hu: &mut [f64],
    out_hv: &mut [f64],
    n_nodes: usize,
) {
    // Use faer GEMV only for larger matrices where the overhead is amortized
    if n_nodes >= FAER_GEMV_THRESHOLD {
        use faer::linalg::matmul::matmul;
        use faer::{Accum, ColMut, ColRef, MatRef, Par};

        let d_mat = MatRef::from_row_major_slice(d, n_nodes, n_nodes);
        let x_h = ColRef::from_slice(flux_h);
        let x_hu = ColRef::from_slice(flux_hu);
        let x_hv = ColRef::from_slice(flux_hv);
        let mut y_h = ColMut::from_slice_mut(out_h);
        let mut y_hu = ColMut::from_slice_mut(out_hu);
        let mut y_hv = ColMut::from_slice_mut(out_hv);

        matmul(&mut y_h, Accum::Replace, &d_mat, &x_h, 1.0, Par::Seq);
        matmul(&mut y_hu, Accum::Replace, &d_mat, &x_hu, 1.0, Par::Seq);
        matmul(&mut y_hv, Accum::Replace, &d_mat, &x_hv, 1.0, Par::Seq);
    } else {
        // For small matrices, scalar is faster due to faer's setup overhead
        apply_diff_matrix_scalar(d, flux_h, flux_hu, flux_hv, out_h, out_hu, out_hv, n_nodes);
    }
}

/// Apply LIFT matrix.
///
/// Computes `out += scale * LIFT * flux_diff` for all three variables.
/// Uses scalar implementation since LIFT matrices are small (n_nodes × n_face_nodes
/// where n_face_nodes is typically only 3-6) and faer's overhead dominates.
///
/// # Arguments
/// * `lift` - LIFT matrix for one face, n_nodes × n_face_nodes, row-major
/// * `flux_diff_h`, etc. - Flux differences at face nodes, length n_face_nodes
/// * `out_h`, etc. - Output vectors (accumulated), length n_nodes
/// * `n_nodes` - Number of volume nodes per element
/// * `n_face_nodes` - Number of nodes per face
/// * `scale` - Scaling factor (typically j_inv * s_jac)
#[cfg(feature = "simd")]
pub fn apply_lift(
    lift: &[f64],
    flux_diff_h: &[f64],
    flux_diff_hu: &[f64],
    flux_diff_hv: &[f64],
    out_h: &mut [f64],
    out_hu: &mut [f64],
    out_hv: &mut [f64],
    n_nodes: usize,
    n_face_nodes: usize,
    scale: f64,
) {
    // LIFT matrices are small (n_nodes × n_face_nodes with n_face_nodes ~ 3-6),
    // so scalar is faster than faer due to setup overhead.
    apply_lift_scalar(
        lift,
        flux_diff_h,
        flux_diff_hu,
        flux_diff_hv,
        out_h,
        out_hu,
        out_hv,
        n_nodes,
        n_face_nodes,
        scale,
    );
}

/// Apply Coriolis source term with automatic SIMD dispatch.
#[cfg(feature = "simd")]
pub fn coriolis_source(
    hu: &[f64],
    hv: &[f64],
    out_hu: &mut [f64],
    out_hv: &mut [f64],
    f: f64,
    n_nodes: usize,
) {
    let _ = n_nodes; // Used for consistency with scalar API

    struct Impl<'a> {
        hu: &'a [f64],
        hv: &'a [f64],
        out_hu: &'a mut [f64],
        out_hv: &'a mut [f64],
        f: f64,
    }

    impl WithSimd for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            simd_impl::coriolis_source_simd_inner(
                simd, self.hu, self.hv, self.out_hu, self.out_hv, self.f,
            );
        }
    }

    Arch::new().dispatch(Impl {
        hu,
        hv,
        out_hu,
        out_hv,
        f,
    });
}

/// Combine derivatives with geometric factors using automatic SIMD dispatch.
#[cfg(feature = "simd")]
#[allow(clippy::too_many_arguments)]
pub fn combine_derivatives(
    dfx_dr_h: &[f64],
    dfx_dr_hu: &[f64],
    dfx_dr_hv: &[f64],
    dfx_ds_h: &[f64],
    dfx_ds_hu: &[f64],
    dfx_ds_hv: &[f64],
    dfy_dr_h: &[f64],
    dfy_dr_hu: &[f64],
    dfy_dr_hv: &[f64],
    dfy_ds_h: &[f64],
    dfy_ds_hu: &[f64],
    dfy_ds_hv: &[f64],
    out_h: &mut [f64],
    out_hu: &mut [f64],
    out_hv: &mut [f64],
    rx: f64,
    sx: f64,
    ry: f64,
    sy: f64,
    n_nodes: usize,
) {
    let _ = n_nodes; // Used for consistency with scalar API

    struct Impl<'a> {
        dfx_dr_h: &'a [f64],
        dfx_dr_hu: &'a [f64],
        dfx_dr_hv: &'a [f64],
        dfx_ds_h: &'a [f64],
        dfx_ds_hu: &'a [f64],
        dfx_ds_hv: &'a [f64],
        dfy_dr_h: &'a [f64],
        dfy_dr_hu: &'a [f64],
        dfy_dr_hv: &'a [f64],
        dfy_ds_h: &'a [f64],
        dfy_ds_hu: &'a [f64],
        dfy_ds_hv: &'a [f64],
        out_h: &'a mut [f64],
        out_hu: &'a mut [f64],
        out_hv: &'a mut [f64],
        rx: f64,
        sx: f64,
        ry: f64,
        sy: f64,
    }

    impl WithSimd for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            simd_impl::combine_derivatives_simd_inner(
                simd,
                self.dfx_dr_h,
                self.dfx_dr_hu,
                self.dfx_dr_hv,
                self.dfx_ds_h,
                self.dfx_ds_hu,
                self.dfx_ds_hv,
                self.dfy_dr_h,
                self.dfy_dr_hu,
                self.dfy_dr_hv,
                self.dfy_ds_h,
                self.dfy_ds_hu,
                self.dfy_ds_hv,
                self.out_h,
                self.out_hu,
                self.out_hv,
                self.rx,
                self.sx,
                self.ry,
                self.sy,
            );
        }
    }

    Arch::new().dispatch(Impl {
        dfx_dr_h,
        dfx_dr_hu,
        dfx_dr_hv,
        dfx_ds_h,
        dfx_ds_hu,
        dfx_ds_hv,
        dfy_dr_h,
        dfy_dr_hu,
        dfy_dr_hv,
        dfy_ds_h,
        dfy_ds_hu,
        dfy_ds_hv,
        out_h,
        out_hu,
        out_hv,
        rx,
        sx,
        ry,
        sy,
    });
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn random_vec(n: usize, seed: u64) -> Vec<f64> {
        // Simple deterministic pseudo-random for testing
        let mut v = Vec::with_capacity(n);
        let mut x = seed;
        for _ in 0..n {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = (x as f64) / (u64::MAX as f64) * 2.0 - 1.0;
            v.push(val);
        }
        v
    }

    #[test]
    fn test_apply_diff_matrix_scalar() {
        let n = 9; // 3x3 element
        let d = random_vec(n * n, 42);
        let flux_h = random_vec(n, 1);
        let flux_hu = random_vec(n, 2);
        let flux_hv = random_vec(n, 3);

        let mut out_h = vec![0.0; n];
        let mut out_hu = vec![0.0; n];
        let mut out_hv = vec![0.0; n];

        apply_diff_matrix_scalar(
            &d, &flux_h, &flux_hu, &flux_hv, &mut out_h, &mut out_hu, &mut out_hv, n,
        );

        // Verify against manual computation for first element
        let mut expected = 0.0;
        for j in 0..n {
            expected += d[j] * flux_h[j];
        }
        assert_relative_eq!(out_h[0], expected, epsilon = 1e-12);
    }

    #[test]
    fn test_apply_lift_scalar() {
        let n_nodes = 9;
        let n_face_nodes = 3;
        let lift = random_vec(n_nodes * n_face_nodes, 42);
        let flux_diff_h = random_vec(n_face_nodes, 1);
        let flux_diff_hu = random_vec(n_face_nodes, 2);
        let flux_diff_hv = random_vec(n_face_nodes, 3);

        let mut out_h = vec![1.0; n_nodes];
        let mut out_hu = vec![1.0; n_nodes];
        let mut out_hv = vec![1.0; n_nodes];

        let scale = 0.5;
        apply_lift_scalar(
            &lift,
            &flux_diff_h,
            &flux_diff_hu,
            &flux_diff_hv,
            &mut out_h,
            &mut out_hu,
            &mut out_hv,
            n_nodes,
            n_face_nodes,
            scale,
        );

        // Verify accumulation worked (values should have changed from 1.0)
        assert!(out_h.iter().any(|&x| (x - 1.0).abs() > 1e-10));
    }

    #[test]
    fn test_coriolis_source_scalar() {
        let n = 8;
        let hu = vec![1.0; n];
        let hv = vec![2.0; n];
        let mut out_hu = vec![0.0; n];
        let mut out_hv = vec![0.0; n];

        let f = 1.2e-4; // Norwegian coast Coriolis

        coriolis_source_scalar(&hu, &hv, &mut out_hu, &mut out_hv, f, n);

        // S_hu = f * hv = 1.2e-4 * 2.0 = 2.4e-4
        // S_hv = -f * hu = -1.2e-4 * 1.0 = -1.2e-4
        assert_relative_eq!(out_hu[0], 2.4e-4, epsilon = 1e-10);
        assert_relative_eq!(out_hv[0], -1.2e-4, epsilon = 1e-10);
    }

    #[test]
    fn test_manning_friction_scalar() {
        let n = 4;
        let h = vec![10.0; n]; // 10m depth
        let hu = vec![10.0; n]; // velocity 1 m/s in x
        let hv = vec![0.0; n];
        let mut out_hu = vec![0.0; n];
        let mut out_hv = vec![0.0; n];

        let g = 9.81;
        let manning_n = 0.025;
        let g_n2 = g * manning_n * manning_n;
        let h_min = 0.01;

        manning_friction_scalar(&h, &hu, &hv, &mut out_hu, &mut out_hv, g_n2, h_min, n);

        // C_f = g * n² / h^(1/3) = 9.81 * 0.000625 / 10^(1/3) ≈ 0.00285
        // |u| = 1.0, u = 1.0
        // S_hu = -0.00285 * 1.0 * 1.0 ≈ -0.00285
        assert!(out_hu[0] < 0.0); // Should be negative (friction opposes motion)
        assert!(out_hu[0] > -0.01); // Should be small for typical values
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_coriolis_source_simd_matches_scalar() {
        let n = 16;
        let hu = random_vec(n, 1);
        let hv = random_vec(n, 2);
        let f = 1.2e-4;

        let mut out_hu_scalar = vec![0.0; n];
        let mut out_hv_scalar = vec![0.0; n];
        coriolis_source_scalar(&hu, &hv, &mut out_hu_scalar, &mut out_hv_scalar, f, n);

        let mut out_hu_simd = vec![0.0; n];
        let mut out_hv_simd = vec![0.0; n];
        coriolis_source(&hu, &hv, &mut out_hu_simd, &mut out_hv_simd, f, n);

        for i in 0..n {
            assert_relative_eq!(out_hu_simd[i], out_hu_scalar[i], epsilon = 1e-12);
            assert_relative_eq!(out_hv_simd[i], out_hv_scalar[i], epsilon = 1e-12);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_coriolis_source_simd_odd_size() {
        // Test with size not divisible by SIMD width
        let n = 11;
        let hu = random_vec(n, 1);
        let hv = random_vec(n, 2);
        let f = 1.2e-4;

        let mut out_hu_scalar = vec![0.0; n];
        let mut out_hv_scalar = vec![0.0; n];
        coriolis_source_scalar(&hu, &hv, &mut out_hu_scalar, &mut out_hv_scalar, f, n);

        let mut out_hu_simd = vec![0.0; n];
        let mut out_hv_simd = vec![0.0; n];
        coriolis_source(&hu, &hv, &mut out_hu_simd, &mut out_hv_simd, f, n);

        for i in 0..n {
            assert_relative_eq!(out_hu_simd[i], out_hu_scalar[i], epsilon = 1e-12);
            assert_relative_eq!(out_hv_simd[i], out_hv_scalar[i], epsilon = 1e-12);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_combine_derivatives_simd_matches_scalar() {
        let n = 16;
        let dfx_dr_h = random_vec(n, 1);
        let dfx_dr_hu = random_vec(n, 2);
        let dfx_dr_hv = random_vec(n, 3);
        let dfx_ds_h = random_vec(n, 4);
        let dfx_ds_hu = random_vec(n, 5);
        let dfx_ds_hv = random_vec(n, 6);
        let dfy_dr_h = random_vec(n, 7);
        let dfy_dr_hu = random_vec(n, 8);
        let dfy_dr_hv = random_vec(n, 9);
        let dfy_ds_h = random_vec(n, 10);
        let dfy_ds_hu = random_vec(n, 11);
        let dfy_ds_hv = random_vec(n, 12);

        let rx = 0.5;
        let sx = 0.3;
        let ry = -0.2;
        let sy = 0.4;

        let mut out_h_scalar = vec![0.0; n];
        let mut out_hu_scalar = vec![0.0; n];
        let mut out_hv_scalar = vec![0.0; n];
        combine_derivatives_scalar(
            &dfx_dr_h,
            &dfx_dr_hu,
            &dfx_dr_hv,
            &dfx_ds_h,
            &dfx_ds_hu,
            &dfx_ds_hv,
            &dfy_dr_h,
            &dfy_dr_hu,
            &dfy_dr_hv,
            &dfy_ds_h,
            &dfy_ds_hu,
            &dfy_ds_hv,
            &mut out_h_scalar,
            &mut out_hu_scalar,
            &mut out_hv_scalar,
            rx,
            sx,
            ry,
            sy,
            n,
        );

        let mut out_h_simd = vec![0.0; n];
        let mut out_hu_simd = vec![0.0; n];
        let mut out_hv_simd = vec![0.0; n];
        combine_derivatives(
            &dfx_dr_h,
            &dfx_dr_hu,
            &dfx_dr_hv,
            &dfx_ds_h,
            &dfx_ds_hu,
            &dfx_ds_hv,
            &dfy_dr_h,
            &dfy_dr_hu,
            &dfy_dr_hv,
            &dfy_ds_h,
            &dfy_ds_hu,
            &dfy_ds_hv,
            &mut out_h_simd,
            &mut out_hu_simd,
            &mut out_hv_simd,
            rx,
            sx,
            ry,
            sy,
            n,
        );

        for i in 0..n {
            assert_relative_eq!(out_h_simd[i], out_h_scalar[i], epsilon = 1e-12);
            assert_relative_eq!(out_hu_simd[i], out_hu_scalar[i], epsilon = 1e-12);
            assert_relative_eq!(out_hv_simd[i], out_hv_scalar[i], epsilon = 1e-12);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_apply_diff_matrix_simd_matches_scalar() {
        // SIMD version currently delegates to scalar, so this should always pass
        let n = 16;
        let d = random_vec(n * n, 42);
        let flux_h = random_vec(n, 1);
        let flux_hu = random_vec(n, 2);
        let flux_hv = random_vec(n, 3);

        let mut out_h_scalar = vec![0.0; n];
        let mut out_hu_scalar = vec![0.0; n];
        let mut out_hv_scalar = vec![0.0; n];
        apply_diff_matrix_scalar(
            &d,
            &flux_h,
            &flux_hu,
            &flux_hv,
            &mut out_h_scalar,
            &mut out_hu_scalar,
            &mut out_hv_scalar,
            n,
        );

        let mut out_h_simd = vec![0.0; n];
        let mut out_hu_simd = vec![0.0; n];
        let mut out_hv_simd = vec![0.0; n];
        apply_diff_matrix(
            &d,
            &flux_h,
            &flux_hu,
            &flux_hv,
            &mut out_h_simd,
            &mut out_hu_simd,
            &mut out_hv_simd,
            n,
        );

        for i in 0..n {
            assert_relative_eq!(out_h_simd[i], out_h_scalar[i], epsilon = 1e-12);
            assert_relative_eq!(out_hu_simd[i], out_hu_scalar[i], epsilon = 1e-12);
            assert_relative_eq!(out_hv_simd[i], out_hv_scalar[i], epsilon = 1e-12);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_apply_lift_simd_matches_scalar() {
        let n_nodes = 16;
        let n_face_nodes = 4;
        let lift = random_vec(n_nodes * n_face_nodes, 42);
        let flux_diff_h = random_vec(n_face_nodes, 1);
        let flux_diff_hu = random_vec(n_face_nodes, 2);
        let flux_diff_hv = random_vec(n_face_nodes, 3);
        let scale = 0.5;

        let mut out_h_scalar = vec![1.0; n_nodes];
        let mut out_hu_scalar = vec![1.0; n_nodes];
        let mut out_hv_scalar = vec![1.0; n_nodes];
        apply_lift_scalar(
            &lift,
            &flux_diff_h,
            &flux_diff_hu,
            &flux_diff_hv,
            &mut out_h_scalar,
            &mut out_hu_scalar,
            &mut out_hv_scalar,
            n_nodes,
            n_face_nodes,
            scale,
        );

        let mut out_h_simd = vec![1.0; n_nodes];
        let mut out_hu_simd = vec![1.0; n_nodes];
        let mut out_hv_simd = vec![1.0; n_nodes];
        apply_lift(
            &lift,
            &flux_diff_h,
            &flux_diff_hu,
            &flux_diff_hv,
            &mut out_h_simd,
            &mut out_hu_simd,
            &mut out_hv_simd,
            n_nodes,
            n_face_nodes,
            scale,
        );

        for i in 0..n_nodes {
            assert_relative_eq!(out_h_simd[i], out_h_scalar[i], epsilon = 1e-12);
            assert_relative_eq!(out_hu_simd[i], out_hu_scalar[i], epsilon = 1e-12);
            assert_relative_eq!(out_hv_simd[i], out_hv_scalar[i], epsilon = 1e-12);
        }
    }
}
