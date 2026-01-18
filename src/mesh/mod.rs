//! Mesh representation for DG discretizations.
//!
//! This module provides mesh data structures organized into submodules:
//!
//! # Submodules
//!
//! - [`core`]: Core mesh implementations (Mesh1D, Mesh2D, builders)
//! - [`traits`]: Abstract traits for dimension-independent operations
//! - [`data`]: Mesh-associated data (bathymetry, land masks, boundary tags)
//! - [`io`]: Mesh file I/O (Gmsh format)
//!
//! # Mesh Traits
//!
//! The [`traits`] module provides abstract interfaces for mesh operations:
//! - [`Point`]: Coordinate type abstraction for 1D, 2D, 3D
//! - [`MeshTopology`]: Element and face connectivity
//! - [`MeshGeometry`]: Coordinate mappings and Jacobians
//! - [`MeshGeometryExt`]: Face normals and surface Jacobians (2D/3D)
//! - [`MeshCFL`]: CFL time step computation
//!
//! These traits enable generic code that works across mesh dimensions.

pub mod core;
pub mod data;
pub mod io;
pub mod traits;

// Re-export core mesh types
pub use core::{BoundaryFace, BoundaryConfig, Edge, ElementFace, Mesh1D, Mesh2D, Mesh2DBuilder};

// Re-export data types
pub use data::{
    bathymetry_profiles, bathymetry_profiles_2d, Bathymetry1D, Bathymetry2D, BoundaryTag,
    LandMask2D, LandMaskStatistics,
};

// Re-export I/O types
pub use io::{read_gmsh_mesh, write_gmsh_mesh, GmshError};

// Re-export trait types
pub use traits::{
    FaceConnection, Mesh1DGeometry, Mesh2DGeometry, MeshCFL, MeshGeometry, MeshGeometryExt,
    MeshGPUData, MeshIter, MeshTopology, Neighbor, Point,
};
