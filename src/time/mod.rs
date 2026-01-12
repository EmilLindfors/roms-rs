//! Time integration methods.

mod coupled_swe_tracer;
mod ssp_rk3;
mod ssp_rk3_2d;
mod ssp_rk3_swe;
mod ssp_rk3_swe_2d;

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
