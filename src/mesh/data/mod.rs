//! Mesh-associated data structures.
//!
//! This module contains data that lives on meshes:
//! - [`Bathymetry1D`], [`Bathymetry2D`]: Bottom topography
//! - [`LandMask2D`]: Wet/dry cell classification
//! - [`BoundaryTag`]: Boundary condition labels

mod bathymetry;
mod bathymetry_2d;
mod boundary_tags;
mod land_mask;

pub use bathymetry::{profiles as bathymetry_profiles, Bathymetry1D};
pub use bathymetry_2d::{profiles as bathymetry_profiles_2d, Bathymetry2D};
pub use boundary_tags::BoundaryTag;
pub use land_mask::{LandMask2D, LandMaskStatistics};
