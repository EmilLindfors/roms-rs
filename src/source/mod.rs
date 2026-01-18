//! Source terms for shallow water equations.
//!
//! Source terms represent forces that are not part of the hyperbolic flux:
//! - Bathymetry (bottom topography)
//! - Bottom friction (Manning/Chezy)
//! - Coriolis effect
//! - Wind stress
//! - Atmospheric pressure
//! - Tidal potential
//! - Sponge layers (boundary damping)
//!
//! # Submodules
//!
//! - [`traits`]: Abstract source term traits (SourceTerm, SourceTerm2D)
//! - [`swe_1d`]: 1D shallow water source terms
//! - [`swe_2d`]: 2D shallow water source terms
//! - [`tracer`]: Tracer-related source terms (baroclinic, heat flux)
//! - [`boundary`]: Boundary treatment sources (sponge, strait friction)
//! - [`wellbalanced`]: Well-balanced scheme components

pub mod boundary;
pub mod swe_1d;
pub mod swe_2d;
pub mod tracer;
pub mod traits;
pub mod wellbalanced;

// Re-export traits
pub use traits::{CombinedSource, CombinedSource2D, SourceContext2D, SourceTerm, SourceTerm2D};

// Re-export 1D source terms
pub use swe_1d::{BathymetrySource, ChezyFriction, HydrostaticReconstruction, ManningFriction};

// Re-export 2D SWE source terms
pub use swe_2d::{
    AtmosphericPressure2D, BathymetrySource2D, ChezyFriction2D, CoriolisSource2D, DragCoefficient,
    ManningFriction2D, SpatiallyVaryingManning2D, TidalPotential, TidalPotentialConstituent,
    WindStress2D, P_STANDARD, RHO_AIR, RHO_WATER, RHO_WATER_PRESSURE,
};

// Re-export tracer source terms
pub use tracer::{
    BaroclinicSource2D, LinearBaroclinicSource2D, TracerSourceContext2D, TracerSourceTerm2D,
    compute_tracer_gradients,
};

// Re-export boundary sources
pub use boundary::{
    RectangularBoundary, SillDetector, SillOverflow2D, SillWithFriction, SpongeDistanceFn,
    SpongeLayer2D, SpongeProfile, StraitFriction2D, bathymetry_based_width, rectangular_sponge_fn,
    rectangular_strait, tapered_strait,
};

// Re-export well-balanced components
pub use wellbalanced::HydrostaticReconstruction2D;
