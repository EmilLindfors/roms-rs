//! Right-hand side computation for DG discretizations.
//!
//! Computes the semi-discrete RHS for various equation systems:
//! - Scalar advection (1D and 2D)
//! - Shallow water equations (1D and 2D)
//! - Tracer transport (2D)

mod advection_2d;
mod scalar_1d;
mod swe_1d;
mod swe_2d;
mod tracer_2d;

// 1D scalar
pub use scalar_1d::{BoundaryCondition, compute_rhs};
#[cfg(feature = "parallel")]
pub use scalar_1d::compute_rhs_parallel;

// 1D SWE
pub use swe_1d::{SWEFluxType, SWERhsConfig, compute_dt_swe, compute_rhs_swe};

// 2D advection
pub use advection_2d::{
    AdvectionBoundaryCondition2D, AdvectionFluxType, ConstantBC2D, DirichletBC2D, PeriodicBC2D,
    compute_dt_advection_2d, compute_rhs_advection_2d,
};

// 2D SWE
pub use swe_2d::{SWE2DRhsConfig, compute_dt_swe_2d, compute_rhs_swe_2d};
#[cfg(feature = "parallel")]
pub use swe_2d::{compute_dt_swe_2d_parallel, compute_rhs_swe_2d_parallel};
#[cfg(feature = "simd")]
pub use swe_2d::compute_rhs_swe_2d_simd;
#[cfg(all(feature = "parallel", feature = "simd"))]
pub use swe_2d::compute_rhs_swe_2d_parallel_simd;

// 2D Tracer
pub use tracer_2d::{
    ExtrapolationTracerBC, FixedTracerBC, Tracer2DRhsConfig, TracerBCContext2D,
    TracerBoundaryCondition2D, TracerSourceTerm2D, UpwindTracerBC, compute_dt_tracer_2d,
    compute_rhs_tracer_2d,
};
#[cfg(feature = "parallel")]
pub use tracer_2d::compute_rhs_tracer_2d_parallel;
