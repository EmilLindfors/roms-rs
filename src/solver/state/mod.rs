//! Physical state types for shallow water and tracer equations.
//!
//! - [`SWEState`], [`SWESolution`]: 1D shallow water (h, hu)
//! - [`SWEState2D`], [`SWESolution2D`]: 2D shallow water (h, hu, hv)
//! - [`TracerState`], [`TracerSolution2D`]: Temperature and salinity

mod swe_1d;
mod swe_2d;
mod tracer;
mod state_3d;

pub use swe_1d::{SWESolution, SWEState};
pub use swe_2d::{SWESolution2D, SWEState2D, SWE_VAR_H, SWE_VAR_HU, SWE_VAR_HV};
pub use tracer::{ConservativeTracerState, TracerSolution2D, TracerState};
pub use state_3d::Solution3D;
