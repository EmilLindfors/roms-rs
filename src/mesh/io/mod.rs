//! Mesh I/O functionality.
//!
//! This module provides reading and writing of mesh files:
//! - [`read_gmsh_mesh`], [`write_gmsh_mesh`]: Gmsh format support

mod gmsh;

pub use gmsh::{read_gmsh_mesh, write_gmsh_mesh, GmshError};
