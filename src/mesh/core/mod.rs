//! Core mesh implementations.
//!
//! This module contains the actual mesh data structures:
//! - [`Mesh1D`]: 1D mesh for interval elements
//! - [`Mesh2D`]: 2D mesh for quadrilateral elements
//! - [`Mesh2DBuilder`]: Fluent builder for constructing 2D meshes

mod mesh1d;
mod mesh2d;
mod mesh2d_builder;

pub use mesh1d::{BoundaryFace, Mesh1D};
pub use mesh2d::{Edge, ElementFace, Mesh2D};
pub use mesh2d_builder::{BoundaryConfig, Mesh2DBuilder};
