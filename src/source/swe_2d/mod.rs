//! 2D shallow water source terms.
//!
//! Physical forces acting on 2D shallow water flow:
//! - Coriolis effect from Earth's rotation
//! - Bottom friction (Manning, Chezy)
//! - Wind stress at the surface
//! - Atmospheric pressure gradients
//! - Tidal potential forcing
//! - Bathymetry gradients

mod atmospheric;
mod bathymetry;
mod coriolis;
mod friction;
mod tidal;
mod wind;

pub use atmospheric::{AtmosphericPressure2D, P_STANDARD, RHO_WATER_PRESSURE};
pub use bathymetry::BathymetrySource2D;
pub use coriolis::CoriolisSource2D;
pub use friction::{ChezyFriction2D, ManningFriction2D, SpatiallyVaryingManning2D};
pub use tidal::{TidalPotential, TidalPotentialConstituent};
pub use wind::{DragCoefficient, WindStress2D, RHO_AIR, RHO_WATER};
