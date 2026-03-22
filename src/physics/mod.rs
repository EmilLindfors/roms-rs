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
pub mod vertical_diffusion;
pub mod vertical_mixing;
pub mod eos;
pub mod hydrostatic_3d;
pub mod vertical_velocity;

pub use builder::{PhysicsBuilder, SWEPhysics2D, SWEPhysics2DBuilder};
pub use traits::{PhysicsConfig, PhysicsModule, PhysicsModuleInfo};
pub use vertical_diffusion::apply_vertical_diffusion;
pub use vertical_mixing::{VerticalMixing, ConstantMixing, PacanowskiPhilanderMixing, Column, Forcing};
pub use eos::{EquationOfState, LinearEOS, UnescoEOS};
pub use hydrostatic_3d::Hydrostatic3D;
pub use vertical_velocity::compute_vertical_velocity;
