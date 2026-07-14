//! DG solver components.
//!
//! # Submodules
//!
//! - [`core`]: Core solution containers (DGSolution1D, DGSolution2D, SystemSolution)
//! - [`state`]: Physical state types (SWEState, TracerState, etc.)
//! - [`rhs`]: Right-hand side computation for various equations
//! - [`limiters`]: Slope limiters for oscillation control
//! - [`algorithms`]: Specialized algorithms (wetting/drying)
//! - [`diagnostics`]: Runtime diagnostics and progress tracking
//! - [`simd`]: SIMD-optimized data structures and kernels
//!
//! # Limiter Traits
//!
//! The [`Limiter2D`] trait provides a generic interface for slope limiters.
//! This enables composable limiters and runtime selection:
//!
//! - [`TVBLimiter2D`]: Total Variation Bounded slope limiter
//! - [`KuzminLimiter2D`]: Vertex-based Kuzmin limiter
//! - [`PositivityLimiter2D`]: Positivity-preserving limiter
//! - [`LimiterChain2D`]: Chain multiple limiters together
//! - [`StandardLimiter2D`]: Enum for zero-cost dispatch

pub mod algorithms;
#[cfg(feature = "burn")]
pub mod burn;
pub mod core;
pub mod diagnostics;
pub mod limiters;
pub mod rhs;
pub mod simd;
pub mod state;

// Re-export core solution types
pub use core::{DGSolution1D, DGSolution2D, SystemSolution, SystemSolution2D};

// Re-export state types
pub use state::{
    ConservativeTracerState, SWESolution, SWESolution2D, SWEState, SWEState2D, TracerSolution2D,
    TracerState,
};

// Re-export RHS types and functions
pub use rhs::{
    // 2D Advection
    AdvectionBoundaryCondition2D,
    AdvectionFluxType,
    // 1D
    BoundaryCondition,
    ConstantBC2D,
    DirichletBC2D,
    // 2D Tracer
    ExtrapolationTracerBC,
    // 3D RHS
    ExtrapolationTracerBC3D,
    FixedTracerBC,
    FixedTracerBC3D,
    PeriodicBC2D,
    Rhs3DConfig,
    // 2D SWE
    SWE2DRhsConfig,
    SWEFluxType,
    SWERhsConfig,
    Tracer2DRhsConfig,
    TracerBCContext2D,
    TracerBCContext3D,
    TracerBoundaryCondition2D,
    TracerBoundaryCondition3D,
    TracerSourceTerm2D,
    UpwindTracerBC,
    UpwindTracerBC3D,
    compute_dt_advection_2d,
    compute_dt_swe,
    compute_dt_swe_2d,
    compute_dt_tracer_2d,
    compute_dt_viscosity,
    compute_rhs,
    compute_rhs_3d,
    compute_rhs_advection_2d,
    compute_rhs_swe,
    compute_rhs_swe_2d,
    compute_rhs_tracer_2d,
};

// Re-export limiter types
pub use limiters::{
    // Traits and context
    BoxedLimiter2D,
    // 2D SWE limiters
    KuzminLimiter2D,
    // 2D Tracer limiters
    KuzminParameter2D,
    Limiter2D,
    LimiterChain2D,
    LimiterContext2D,
    NoLimiter2D,
    PositivityLimiter2D,
    StandardLimiter2D,
    // 1D limiters
    TVBParameter,
    TVBParameter2D,
    TracerAveragePolicy3D,
    TracerBounds,
    TracerLimiter3DConfig,
    TracerLimiter3DStats,
    TracerLimiterType3D,
    apply_swe_limiters,
    apply_swe_limiters_kuzmin_2d,
    apply_tracer_limiters_2d,
    apply_tracer_limiters_3d,
    apply_tracer_limiters_kuzmin_2d,
    create_limiter,
    positivity_limiter,
    swe_cell_averages_2d,
    swe_kuzmin_limiter_2d,
    swe_positivity_limiter_2d,
    tracer_cell_averages_2d,
    tracer_kuzmin_limiter_2d,
    tracer_positivity_limiter_2d,
    tracer_tvb_limiter_2d,
    tvb_limiter,
};

// Re-export algorithms
pub use algorithms::{WetDryConfig, apply_wet_dry_correction, apply_wet_dry_correction_all};

#[cfg(feature = "parallel")]
pub use algorithms::apply_wet_dry_correction_all_parallel;

// Re-export diagnostics
pub use diagnostics::{
    DiagnosticsTracker, ProgressReporter, SWEDiagnostics2D, current_cfl_2d, total_energy_2d,
    total_mass_2d, total_momentum_2d,
};

// Re-export SIMD types
pub use simd::{FaceWorkspace, SIMDWorkspace, SWESoABuffer};
pub use simd::{
    apply_diff_matrix_scalar, apply_lift_scalar, combine_derivatives_scalar,
    coriolis_source_scalar, manning_friction_scalar,
};

#[cfg(feature = "simd")]
pub use simd::{apply_diff_matrix, apply_lift, combine_derivatives, coriolis_source};

#[cfg(feature = "parallel")]
pub use rhs::{compute_rhs_parallel, compute_rhs_tracer_2d_parallel};

#[cfg(feature = "parallel")]
pub use limiters::{
    apply_swe_limiters_kuzmin_2d_parallel, swe_cell_averages_2d_parallel,
    swe_kuzmin_limiter_2d_parallel, swe_positivity_limiter_2d_parallel,
};

#[cfg(all(feature = "parallel", feature = "simd"))]
pub use rhs::{compute_dt_swe_2d_parallel, compute_rhs_swe_2d_parallel};
