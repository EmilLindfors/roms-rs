//! Tracer-related source terms.
//!
//! - Baroclinic effects (density gradients)
//! - Surface heat flux
//! - River inputs

mod baroclinic;
mod surface_flux;

pub use baroclinic::{
    BaroclinicSource2D, LinearBaroclinicSource2D, TracerSourceContext2D, TracerSourceTerm2D,
    compute_tracer_gradients,
};
pub use surface_flux::*;
