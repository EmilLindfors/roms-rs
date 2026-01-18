//! Numerical flux functions.
//!
//! Provides numerical fluxes for DG discretizations:
//! - Scalar fluxes: upwind, Lax-Friedrichs
//! - Shallow water fluxes: Roe, HLL/HLLC
//! - 2D shallow water fluxes with rotation to face-aligned coordinates
//! - Tracer fluxes: upwind-based for temperature and salinity
//!
//! # Flux Trait
//!
//! The [`NumericalFlux2D`] trait provides a generic interface for numerical fluxes.
//! This enables extending the solver with custom fluxes while maintaining performance
//! through both compile-time and runtime dispatch options.
//!
//! ## Built-in Flux Types
//! - [`RoeFlux2D`]: Roe approximate Riemann solver
//! - [`HLLFlux2D`]: HLL solver (robust for strong shocks)
//! - [`RusanovFlux2D`]: Local Lax-Friedrichs (simple and robust)
//! - [`StandardFlux2D`]: Enum for zero-cost dispatch when type is known at compile time

mod hll;
mod roe;
mod swe_2d;
mod tracer_2d;
pub mod traits;
mod upwind;

pub use hll::{hll_flux_jump_swe, hll_flux_swe, hll_flux_swe_normal, hllc_flux_swe};
pub use roe::{roe_flux_jump_swe, roe_flux_swe, roe_flux_swe_normal};
pub use swe_2d::{
    SWEFluxType2D, compute_flux_swe_2d, hll_flux_swe_2d, roe_flux_swe_2d, rusanov_flux_swe_2d,
};
pub use tracer_2d::{
    TracerFluxType, lax_friedrichs_tracer_flux, roe_tracer_flux, tracer_numerical_flux,
    upwind_tracer_flux,
};
pub use upwind::{lax_friedrichs_flux, upwind_flux};

// Re-export trait-based flux types
pub use traits::{
    BoxedFlux2D, FluxContext2D, HLLFlux2D, NumericalFlux2D, RoeFlux2D, RusanovFlux2D,
    StandardFlux2D, create_flux,
};
