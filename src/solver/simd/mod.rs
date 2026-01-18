//! SIMD-optimized data structures and kernels.
//!
//! - [`SWESoABuffer`]: Structure-of-arrays layout for SIMD operations
//! - [`SIMDWorkspace`]: Workspace for SIMD-optimized RHS computation
//! - Scalar and SIMD kernel functions

mod buffers;
mod kernels;

// SIMD data structures
pub use buffers::{FaceWorkspace, SIMDWorkspace, SWESoABuffer};

// Scalar kernels (always available)
pub use kernels::{
    apply_diff_matrix_scalar, apply_lift_scalar, combine_derivatives_scalar,
    coriolis_source_scalar, manning_friction_scalar,
};

// SIMD kernels (feature-gated)
#[cfg(feature = "simd")]
pub use kernels::{apply_diff_matrix, apply_lift, combine_derivatives, coriolis_source};
