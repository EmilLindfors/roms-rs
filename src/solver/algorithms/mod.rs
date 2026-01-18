//! Specialized algorithms for DG methods.
//!
//! - [`WetDryConfig`]: Configuration for wetting/drying treatment
//! - [`apply_wet_dry_correction`]: Wet/dry corrections for shallow water

mod wetting_drying;

pub use wetting_drying::{WetDryConfig, apply_wet_dry_correction, apply_wet_dry_correction_all};

#[cfg(feature = "parallel")]
pub use wetting_drying::apply_wet_dry_correction_all_parallel;
