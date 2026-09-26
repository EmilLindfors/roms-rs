//! Mesh I/O functionality.
//!
//! This module provides reading and writing of mesh files:
//! - [`read_gmsh_mesh`], [`parse_gmsh_mesh`], [`write_gmsh_mesh`]: Gmsh MSH 4.1 and 2.2

mod gmsh;

pub use gmsh::{GmshError, parse_gmsh_mesh, read_gmsh_mesh, write_gmsh_mesh};
