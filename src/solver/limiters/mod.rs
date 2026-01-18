//! Slope limiters for DG methods.
//!
//! Limiters control oscillations near discontinuities while preserving
//! high-order accuracy in smooth regions.
//!
//! - [`Limiter2D`]: Trait for 2D slope limiters
//! - [`TVBLimiter2D`], [`KuzminLimiter2D`]: Concrete implementations
//! - [`LimiterChain2D`]: Compose multiple limiters
//! - [`StandardLimiter2D`]: Zero-cost dispatch enum

mod limiters_1d;
mod standard;
mod swe_2d;
mod tracer_2d;
mod traits;

// Traits
pub use traits::{BoxedLimiter2D, Limiter2D, LimiterContext2D};

// Standard limiter types (chain, enum, factory)
pub use standard::{
    KuzminLimiter2D, LimiterChain2D, NoLimiter2D, PositivityLimiter2D, StandardLimiter2D,
    TVBLimiter2D, create_limiter,
};

// 1D limiters
pub use limiters_1d::{TVBParameter, apply_swe_limiters, positivity_limiter, tvb_limiter};

// 2D SWE limiters
pub use swe_2d::{
    apply_swe_limiters_2d, apply_swe_limiters_kuzmin_2d, swe_cell_averages_2d,
    swe_kuzmin_limiter_2d, swe_positivity_limiter_2d, swe_tvb_limiter_2d,
};

// Parallel 2D SWE limiters
#[cfg(feature = "parallel")]
pub use swe_2d::{
    apply_swe_limiters_kuzmin_2d_parallel, swe_cell_averages_2d_parallel,
    swe_kuzmin_limiter_2d_parallel, swe_positivity_limiter_2d_parallel,
};

// 2D Tracer limiters
pub use tracer_2d::{
    KuzminParameter2D, TVBParameter2D, TracerBounds, apply_tracer_limiters_2d,
    apply_tracer_limiters_kuzmin_2d, tracer_cell_averages_2d, tracer_kuzmin_limiter_2d,
    tracer_positivity_limiter_2d, tracer_tvb_limiter_2d,
};
