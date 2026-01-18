//! Boundary treatment source terms.
//!
//! Special handling for boundaries and localized features:
//! - Sponge layers for absorbing boundaries
//! - Enhanced friction in straits
//! - Sill overflow parameterization

mod sill;
mod sponge;
mod strait;

pub use sill::{SillDetector, SillOverflow2D, SillWithFriction};
pub use sponge::{
    RectangularBoundary, SpongeDistanceFn, SpongeLayer2D, SpongeProfile, rectangular_sponge_fn,
};
pub use strait::{StraitFriction2D, bathymetry_based_width, rectangular_strait, tapered_strait};
