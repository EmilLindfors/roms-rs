//! Mesh representation.
//!
//! Provides mesh data structures for DG discretizations:
//! - 1D mesh with element connectivity
//! - 2D mesh for quadrilateral elements with edge-based connectivity
//! - Bathymetry storage for shallow water (1D and 2D)
//! - Land masking for wet/dry cell classification
//! - Gmsh mesh file I/O

mod bathymetry;
mod bathymetry_2d;
mod boundary_tags;
pub mod gmsh;
mod land_mask;
mod mesh1d;
mod mesh2d;

pub use bathymetry::{Bathymetry1D, profiles as bathymetry_profiles};
pub use bathymetry_2d::{Bathymetry2D, profiles as bathymetry_profiles_2d};
pub use boundary_tags::BoundaryTag;
pub use gmsh::{GmshError, read_gmsh_mesh, write_gmsh_mesh};
pub use land_mask::{LandMask2D, LandMaskStatistics};
pub use mesh1d::{BoundaryFace, Mesh1D};
pub use mesh2d::{Edge, ElementFace, Mesh2D};
