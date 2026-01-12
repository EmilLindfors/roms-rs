//! DG solver components.

mod dg1d;
mod dg2d;
mod diagnostics;
mod limiters;
mod limiters_swe_2d;
mod limiters_tracer_2d;
mod rhs;
mod rhs_advection_2d;
mod rhs_swe_2d;
mod rhs_system;
mod rhs_tracer_2d;
mod state;
mod state_2d;
mod tracer_state;
mod wetting_drying;

// SIMD-optimized modules
pub mod simd_kernels;
pub mod simd_swe_2d;

pub use dg1d::DGSolution1D;
pub use dg2d::DGSolution2D;
pub use limiters::{TVBParameter, apply_swe_limiters, positivity_limiter, tvb_limiter};
pub use limiters_swe_2d::{
    apply_swe_limiters_2d, apply_swe_limiters_kuzmin_2d, swe_cell_averages_2d,
    swe_kuzmin_limiter_2d, swe_positivity_limiter_2d, swe_tvb_limiter_2d,
};
pub use limiters_tracer_2d::{
    KuzminParameter2D, TVBParameter2D, TracerBounds, apply_tracer_limiters_2d,
    apply_tracer_limiters_kuzmin_2d, tracer_cell_averages_2d, tracer_kuzmin_limiter_2d,
    tracer_positivity_limiter_2d, tracer_tvb_limiter_2d,
};
pub use rhs::{BoundaryCondition, compute_rhs};
pub use rhs_advection_2d::{
    AdvectionBoundaryCondition2D, AdvectionFluxType, ConstantBC2D, DirichletBC2D, PeriodicBC2D,
    compute_dt_advection_2d, compute_rhs_advection_2d,
};
pub use rhs_swe_2d::{SWE2DRhsConfig, compute_dt_swe_2d, compute_rhs_swe_2d};
#[cfg(feature = "simd")]
pub use rhs_swe_2d::compute_rhs_swe_2d_simd;
pub use rhs_system::{SWEFluxType, SWERhsConfig, compute_dt_swe, compute_rhs_swe};
pub use rhs_tracer_2d::{
    ExtrapolationTracerBC, FixedTracerBC, Tracer2DRhsConfig, TracerBCContext2D,
    TracerBoundaryCondition2D, TracerSourceTerm2D, UpwindTracerBC, compute_dt_tracer_2d,
    compute_rhs_tracer_2d,
};
pub use state::{SWESolution, SWEState, SystemSolution};
pub use state_2d::{SWESolution2D, SWEState2D, SystemSolution2D};
pub use tracer_state::{ConservativeTracerState, TracerSolution2D, TracerState};
pub use wetting_drying::{WetDryConfig, apply_wet_dry_correction, apply_wet_dry_correction_all};
pub use diagnostics::{
    DiagnosticsTracker, ProgressReporter, SWEDiagnostics2D, current_cfl_2d, total_energy_2d,
    total_mass_2d, total_momentum_2d,
};

#[cfg(feature = "parallel")]
pub use rhs::compute_rhs_parallel;
#[cfg(feature = "parallel")]
pub use rhs_swe_2d::compute_rhs_swe_2d_parallel;
#[cfg(feature = "parallel")]
pub use rhs_tracer_2d::compute_rhs_tracer_2d_parallel;

// SIMD exports
pub use simd_swe_2d::{FaceWorkspace, SIMDWorkspace, SWESoABuffer};

// SIMD kernel exports (scalar versions always available)
pub use simd_kernels::{
    apply_diff_matrix_scalar, apply_lift_scalar, combine_derivatives_scalar,
    coriolis_source_scalar, manning_friction_scalar,
};

#[cfg(feature = "simd")]
pub use simd_kernels::{apply_diff_matrix, apply_lift, combine_derivatives, coriolis_source};
