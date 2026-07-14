//! Right-hand side computation for DG discretizations.
//!
//! Computes the semi-discrete RHS for various equation systems:
//! - Scalar advection (1D and 2D)
//! - Shallow water equations (1D and 2D)
//! - Tracer transport (2D)

mod advection_2d;
pub mod baroclinic;
mod diffusion_2d;
mod scalar_1d;
mod swe_1d;
mod swe_2d;
mod tracer_2d;

// 1D scalar
#[cfg(feature = "parallel")]
pub use scalar_1d::compute_rhs_parallel;
pub use scalar_1d::{BoundaryCondition, compute_rhs};

// 1D SWE
pub use swe_1d::{SWEFluxType, SWERhsConfig, compute_dt_swe, compute_rhs_swe};

// 2D advection
pub use advection_2d::{
    AdvectionBoundaryCondition2D, AdvectionFluxType, ConstantBC2D, DirichletBC2D, PeriodicBC2D,
    compute_dt_advection_2d, compute_rhs_advection_2d,
};

// 2D SWE
pub use swe_2d::{SWE2DRhsConfig, compute_dt_swe_2d, compute_dt_viscosity, compute_rhs_swe_2d};
#[cfg(all(feature = "parallel", feature = "simd"))]
pub use swe_2d::{compute_dt_swe_2d_parallel, compute_rhs_swe_2d_parallel};

// 2D Tracer
#[cfg(feature = "parallel")]
pub use tracer_2d::compute_rhs_tracer_2d_parallel;
pub use tracer_2d::{
    ExtrapolationTracerBC, FixedTracerBC, Tracer2DRhsConfig, TracerBCContext2D,
    TracerBoundaryCondition2D, TracerSourceTerm2D, UpwindTracerBC, compute_dt_tracer_2d,
    compute_rhs_tracer_2d,
};

// 3D Baroclinic
pub use baroclinic::compute_pressure_gradient;

pub mod coriolis_3d;
pub use coriolis_3d::apply_coriolis_3d;

pub mod advection_3d;
pub use advection_3d::{
    ExtrapolationTracerBC3D, FixedTracerBC3D, TracerBCContext3D, TracerBoundaryCondition3D,
    UpwindTracerBC3D, apply_horizontal_advection_3d, apply_tracer_advection_3d,
    apply_vertical_advection_3d,
};

pub mod rhs_3d;
pub use rhs_3d::{Rhs3DConfig, compute_rhs_3d};
