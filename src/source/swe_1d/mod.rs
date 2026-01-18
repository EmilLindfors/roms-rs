//! 1D shallow water source terms.
//!
//! Physical forces for 1D shallow water equations:
//! - Bathymetry (bottom topography)
//! - Bottom friction (Manning, Chezy)

mod bathymetry;
mod friction;

pub use bathymetry::{BathymetrySource, HydrostaticReconstruction};
pub use friction::{ChezyFriction, ManningFriction};
