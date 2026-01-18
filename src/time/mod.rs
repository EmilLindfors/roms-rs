//! Time integration methods.
//!
//! # Integrator Traits
//!
//! The [`TimeIntegrator`] trait provides a generic interface for time integrators.
//! This enables composable integrators and runtime selection:
//!
//! - [`SSPRK3`]: Strong Stability Preserving RK3 (optimal for hyperbolic systems)
//! - [`ForwardEuler`]: Simple 1st order (for testing)
//! - [`StandardIntegrator`]: Enum for zero-cost dispatch
//!
//! The [`Integrable`] trait marks solution types that can be time-stepped.

mod coupled_swe_tracer;
pub mod integrator;
mod ssp_rk3;
mod ssp_rk3_2d;
mod ssp_rk3_swe;
mod ssp_rk3_swe_2d;

#[cfg(feature = "burn")]
mod burn_ssp_rk3;

pub use coupled_swe_tracer::{
    CoupledRhs2D, CoupledState2D, CoupledTimeConfig, TracerLimiterType, compute_coupled_rhs,
    compute_coupled_rhs_baroclinic, compute_dt_coupled, run_coupled_simulation,
    run_coupled_simulation_limited, ssp_rk3_coupled_step, ssp_rk3_coupled_step_limited,
    ssp_rk3_coupled_step_timed, total_mass as total_mass_coupled, total_tracer,
};
pub use ssp_rk3::{compute_dt, ssp_rk3_step, ssp_rk3_step_timed};
pub use ssp_rk3_2d::{run_advection_2d, ssp_rk3_step_2d, ssp_rk3_step_2d_timed};
pub use ssp_rk3_swe::{
    SWETimeConfig, run_swe_simulation, ssp_rk3_swe_step, ssp_rk3_swe_step_timed, total_energy,
    total_mass, total_momentum,
};
pub use ssp_rk3_swe_2d::{
    SWE2DTimeConfig, SWELimiterType, run_swe_2d_simulation, ssp_rk3_swe_2d_step_limited,
};

// Integrator trait exports
pub use integrator::{
    BoxedIntegratorInfo, ForwardEuler, Integrable, IntegratorInfo, SSPRK3, StandardIntegrator,
    TimeIntegrator, create_integrator_info,
};

// Burn GPU time integration exports
#[cfg(feature = "burn")]
pub use burn_ssp_rk3::{
    BurnTimeConfig, compute_dt_burn, run_swe_2d_burn, ssp_rk3_step_burn,
};
