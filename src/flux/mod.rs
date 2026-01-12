//! Numerical flux functions.
//!
//! Provides numerical fluxes for DG discretizations:
//! - Scalar fluxes: upwind, Lax-Friedrichs
//! - Shallow water fluxes: Roe, HLL/HLLC
//! - 2D shallow water fluxes with rotation to face-aligned coordinates
//! - Tracer fluxes: upwind-based for temperature and salinity

mod hll;
mod roe;
mod swe_2d;
mod tracer_2d;
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
