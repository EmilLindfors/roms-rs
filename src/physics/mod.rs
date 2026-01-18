//! Physics module abstraction for DG simulations.
//!
//! This module provides a high-level interface for configuring and running
//! physics simulations. It abstracts over the details of:
//! - Numerical flux selection
//! - Boundary condition handling
//! - Source term composition
//! - Limiter application
//! - Time stepping
//!
//! # Key Traits
//!
//! - [`PhysicsModule`]: Core trait for physics computations (RHS, dt, post-processing)
//! - [`PhysicsConfig`]: Configuration for building physics modules
//!
//! # Example
//! ```ignore
//! use dg_rs::physics::{SWEPhysics2D, PhysicsBuilder};
//!
//! let physics = PhysicsBuilder::swe_2d()
//!     .with_flux(StandardFlux2D::Roe)
//!     .with_limiter(StandardLimiter2D::TvbWithPositivity { ... })
//!     .with_bathymetry(&bathymetry)
//!     .with_source(&combined_sources)
//!     .build();
//! ```

pub mod builder;
pub mod traits;

pub use builder::{PhysicsBuilder, SWEPhysics2D, SWEPhysics2DBuilder};
pub use traits::{PhysicsConfig, PhysicsModule, PhysicsModuleInfo};
