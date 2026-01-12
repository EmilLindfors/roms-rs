//! Polynomial basis representations.
//!
//! This module provides Vandermonde matrices for nodal-modal transformations:
//! - 1D Vandermonde for interval elements
//! - 2D Vandermonde for tensor-product quadrilateral elements

mod vandermonde;
mod vandermonde_2d;

pub use vandermonde::Vandermonde;
pub use vandermonde_2d::Vandermonde2D;
