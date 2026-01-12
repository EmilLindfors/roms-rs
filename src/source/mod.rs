//! Source terms for shallow water equations.
//!
//! Source terms represent forces that are not part of the hyperbolic flux:
//! - Bathymetry (bottom topography): S = (0, -gh ∂B/∂x)
//! - Bottom friction (Manning): S = (0, -g n² |u| u h^{-1/3})
//! - Coriolis: S = (0, fhv, -fhu) (2D only)
//! - Wind stress: S = (0, τ_x/ρ, τ_y/ρ) (2D only)
//! - Atmospheric pressure: S = (0, -h/ρ ∂P/∂x, -h/ρ ∂P/∂y) (2D only)
//! - Tidal potential: S = (0, -gh ∂Φ/∂x, -gh ∂Φ/∂y) (2D only)
//! - Sponge layer: S = γ(q_ref - q)
//!
//! The bathymetry source term requires special treatment (well-balanced schemes)
//! to correctly preserve the lake-at-rest steady state.

// 1D source terms
mod bathymetry;
mod friction;

// 2D source terms
mod atmospheric_pressure;
mod baroclinic;
mod bathymetry_source_2d;
mod coriolis_2d;
mod friction_2d;
mod hydrostatic_reconstruction_2d;
mod sill_overflow;
mod source_2d;
mod sponge_2d;
mod strait_friction;
mod tidal_potential;
mod wind_stress;
pub mod tracer_source_2d;

// 1D exports
pub use bathymetry::{BathymetrySource, HydrostaticReconstruction};
pub use friction::{ChezyFriction, ManningFriction};

// 2D exports
pub use atmospheric_pressure::{AtmosphericPressure2D, P_STANDARD, RHO_WATER_PRESSURE};
pub use baroclinic::{
    BaroclinicSource2D, LinearBaroclinicSource2D, TracerSourceContext2D, TracerSourceTerm2D,
    compute_tracer_gradients,
};
pub use bathymetry_source_2d::BathymetrySource2D;
pub use coriolis_2d::CoriolisSource2D;
pub use hydrostatic_reconstruction_2d::HydrostaticReconstruction2D;
pub use friction_2d::{ChezyFriction2D, ManningFriction2D, SpatiallyVaryingManning2D};
pub use sill_overflow::{SillDetector, SillOverflow2D, SillWithFriction};
pub use source_2d::{CombinedSource2D, SourceContext2D, SourceTerm2D};
pub use sponge_2d::{
    RectangularBoundary, SpongeDistanceFn, SpongeLayer2D, SpongeProfile, rectangular_sponge_fn,
};
pub use strait_friction::{
    StraitFriction2D, bathymetry_based_width, rectangular_strait, tapered_strait,
};
pub use tidal_potential::{TidalPotential, TidalPotentialConstituent};
pub use wind_stress::{DragCoefficient, WindStress2D, RHO_AIR, RHO_WATER};

use crate::solver::SWEState;

/// Trait for source terms in shallow water equations.
///
/// Source terms modify the RHS of the equations:
/// dq/dt = -∂F/∂x + S(q, x, t)
pub trait SourceTerm: Send + Sync {
    /// Evaluate the source term contribution at a single node.
    ///
    /// # Arguments
    /// * `state` - Current state (h, hu)
    /// * `db_dx` - Local bathymetry gradient
    /// * `position` - Physical position x
    /// * `time` - Current time
    ///
    /// # Returns
    /// Source contribution as SWEState (S_h, S_hu)
    fn evaluate(&self, state: &SWEState, db_dx: f64, position: f64, time: f64) -> SWEState;

    /// Name of this source term for debugging.
    fn name(&self) -> &'static str;
}

/// Combine multiple source terms.
pub struct CombinedSource<'a> {
    sources: Vec<&'a dyn SourceTerm>,
}

impl<'a> CombinedSource<'a> {
    /// Create a new combined source from a list of source terms.
    pub fn new(sources: Vec<&'a dyn SourceTerm>) -> Self {
        Self { sources }
    }
}

impl<'a> SourceTerm for CombinedSource<'a> {
    fn evaluate(&self, state: &SWEState, db_dx: f64, position: f64, time: f64) -> SWEState {
        let mut total = SWEState::zero();
        for source in &self.sources {
            let contrib = source.evaluate(state, db_dx, position, time);
            total = total + contrib;
        }
        total
    }

    fn name(&self) -> &'static str {
        "combined"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ZeroSource;

    impl SourceTerm for ZeroSource {
        fn evaluate(&self, _: &SWEState, _: f64, _: f64, _: f64) -> SWEState {
            SWEState::zero()
        }

        fn name(&self) -> &'static str {
            "zero"
        }
    }

    struct ConstantSource {
        value: SWEState,
    }

    impl SourceTerm for ConstantSource {
        fn evaluate(&self, _: &SWEState, _: f64, _: f64, _: f64) -> SWEState {
            self.value
        }

        fn name(&self) -> &'static str {
            "constant"
        }
    }

    #[test]
    fn test_combined_source() {
        let s1 = ConstantSource {
            value: SWEState::new(1.0, 2.0),
        };
        let s2 = ConstantSource {
            value: SWEState::new(0.5, 1.0),
        };

        let combined = CombinedSource::new(vec![&s1, &s2]);

        let result = combined.evaluate(&SWEState::zero(), 0.0, 0.0, 0.0);

        assert!((result.h - 1.5).abs() < 1e-14);
        assert!((result.hu - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_zero_source() {
        let zero = ZeroSource;
        let result = zero.evaluate(&SWEState::new(1.0, 2.0), 0.5, 3.0, 1.0);

        assert!(result.h.abs() < 1e-14);
        assert!(result.hu.abs() < 1e-14);
    }
}
