//! Boundary conditions for shallow water equations.
//!
//! Boundary conditions specify how to compute the "ghost" state outside
//! the domain for flux evaluation at boundary faces.
//!
//! # Available Boundary Conditions
//!
//! | BC Type | Description |
//! |---------|-------------|
//! | `Reflective2D` | Wall (no-flux), tangential velocity preserved |
//! | `Radiation2D` | Sommerfeld absorbing condition |
//! | `Chapman2D` | Radiation for sea surface height |
//! | `Flather2D` | Characteristic-based for velocity |
//! | `ChapmanFlather2D` | Combined Chapman + Flather |
//! | `HarmonicTidal2D` | Harmonic constituents (Dirichlet) |
//! | `HarmonicFlather2D` | Harmonic constituents + Flather |
//! | `Discharge2D` | Prescribed flow rate |
//! | `NestingBC2D` | One-way nesting from parent model |
//!
//! # BC Selection Guide for Tidal Simulations
//!
//! | Scenario | Recommended BC | Rationale |
//! |----------|----------------|-----------|
//! | Open ocean | `HarmonicFlather2D` | Best wave absorption |
//! | Closed/semi-closed basin | `HarmonicTidal2D` + sponge | Avoids velocity feedback resonance |
//! | Fjord mouth | `ChapmanFlather2D` | Separates incoming/outgoing waves |
//! | Nesting from parent | `NestingBC2D` (Flather mode) | Smooth transition |
//!
//! # IMPORTANT: Bathymetry Convention for Flather BCs
//!
//! Flather-type BCs (`Flather2D`, `HarmonicFlather2D`, `ChapmanFlather2D`) use
//! surface elevation η = h + B where B is bathymetry.
//!
//! **You MUST set bathymetry correctly** to avoid spurious velocities:
//!
//! ```text
//! For a domain with mean depth h0 (e.g., 50m):
//!   bathymetry = Bathymetry2D::constant(n_elements, n_nodes, -h0);
//!
//! This ensures:
//!   η = h + B = 50 + (-50) = 0 (surface at MSL)
//! ```
//!
//! **Without correct bathymetry** (B = 0):
//! - Interior η = h = 50m (surface 50m above MSL!)
//! - Tidal η ≈ 0m
//! - Flather computes spurious velocity ~20 m/s → blow-up!
//!
//! The Flather BCs will emit a one-time warning if misconfiguration is detected.
//!
//! # Sponge Layers for Closed Basins
//!
//! In closed or semi-closed basins (fjords, bays), Flather BC velocity feedback
//! can amplify reflected waves, causing instability.
//!
//! **Recommended pattern:**
//! ```ignore
//! // Use Dirichlet BC (no velocity feedback)
//! let tidal_bc = HarmonicTidal2D::new(constituents).with_ramp_up(3600.0);
//!
//! // Add sponge layer to absorb outgoing waves
//! let sponge = SpongeLayer2D::rectangular(
//!     |_, _, _| SWEState2D::from_primitives(h0, 0.0, 0.0),
//!     0.01,  // gamma_max
//!     SpongeProfile::Cosine,
//!     (x_min, x_max), (y_min, y_max),
//!     5000.0,  // sponge width
//!     [false, true, false, false],  // right boundary only
//! );
//! ```
//!
//! Or use the `TidalSimulationBuilder`:
//! ```ignore
//! let builder = TidalSimulationBuilder::closed_basin_stable(0.5, 50.0, 5000.0);
//! let bc = builder.build_bc();
//! let sponge = builder.build_sponge((x_min, x_max), (y_min, y_max));
//! ```

pub mod bathymetry_validation;
mod boundary_2d;
mod chapman;
mod multi_bc_2d;
mod nesting_bc;
#[cfg(feature = "netcdf")]
mod ocean_nesting;
mod radiation;
mod reflective;
mod tidal;
mod tidal_config;
mod tst_obc;

// 1D boundary conditions
pub use radiation::{FlatherBC, RadiationBC, SpongeLayer};
pub use reflective::{PartialSlipBC, ReflectiveBC};
pub use tidal::{InterpolatedTidalBC, TidalBC, TidalConstituent};

// 2D boundary conditions
pub use boundary_2d::{
    BCContext2D, ConstantDischarge2D, Discharge2D, Extrapolation2D, FixedState2D, Flather2D,
    HarmonicFlather2D, HarmonicTidal2D, Radiation2D, Reflective2D, SWEBoundaryCondition2D, Tidal2D,
};
pub use chapman::{Chapman2D, ChapmanFlather2D};
pub use multi_bc_2d::MultiBoundaryCondition2D;
pub use nesting_bc::NestingBC2D;
#[cfg(feature = "netcdf")]
pub use ocean_nesting::OceanNestingBC2D;
pub use tidal_config::{SpongeConfig, TidalBCType, TidalSimulationBuilder};
pub use tst_obc::{TSTConfig, TSTConstituent, TSTOBC2D};

// Re-export validation types
pub use bathymetry_validation::{
    BathymetryValidationConfig, BathymetryValidationResult, format_bathymetry_warning,
    validate_bathymetry_convention, warn_once_if_misconfigured,
};

use crate::solver::SWEState;

/// Context for boundary condition evaluation.
///
/// Provides all information needed to compute the ghost state at a boundary.
#[derive(Clone, Copy, Debug)]
pub struct BCContext {
    /// Current simulation time
    pub time: f64,
    /// Physical position of the boundary face
    pub position: f64,
    /// Interior state at the boundary
    pub interior_state: SWEState,
    /// Bathymetry at the boundary
    pub bathymetry: f64,
    /// Outward normal direction (-1 for left boundary, +1 for right boundary)
    pub normal: f64,
}

impl BCContext {
    /// Create a new boundary condition context.
    pub fn new(
        time: f64,
        position: f64,
        interior_state: SWEState,
        bathymetry: f64,
        normal: f64,
    ) -> Self {
        Self {
            time,
            position,
            interior_state,
            bathymetry,
            normal,
        }
    }

    /// Water surface elevation at the interior: η = h + B
    pub fn interior_surface_elevation(&self) -> f64 {
        self.interior_state.h + self.bathymetry
    }

    /// Interior velocity (with protection for dry cells).
    pub fn interior_velocity(&self, h_min: f64) -> f64 {
        self.interior_state.velocity_simple(h_min)
    }

    /// Check if this is a left boundary (normal = -1).
    pub fn is_left_boundary(&self) -> bool {
        self.normal < 0.0
    }

    /// Check if this is a right boundary (normal = +1).
    pub fn is_right_boundary(&self) -> bool {
        self.normal > 0.0
    }
}

/// Trait for shallow water boundary conditions.
///
/// Implementations compute a "ghost" state that represents the exterior
/// state at a boundary face. This ghost state is then used in the
/// numerical flux computation.
pub trait SWEBoundaryCondition: Send + Sync {
    /// Compute the ghost state for flux evaluation.
    ///
    /// # Arguments
    /// * `ctx` - Boundary condition context with interior state and metadata
    ///
    /// # Returns
    /// The ghost (exterior) state to use in flux computation
    fn ghost_state(&self, ctx: &BCContext) -> SWEState;

    /// Name of this boundary condition for debugging/logging.
    fn name(&self) -> &'static str;

    /// Check if this BC allows inflow (used for validation).
    fn allows_inflow(&self) -> bool {
        true
    }

    /// Check if this BC allows outflow (used for validation).
    fn allows_outflow(&self) -> bool {
        true
    }
}

/// Extrapolation boundary condition (zero-gradient).
///
/// Simply copies the interior state to the exterior.
/// This is equivalent to a zero-gradient condition.
#[derive(Clone, Debug, Default)]
pub struct ExtrapolationBC;

impl SWEBoundaryCondition for ExtrapolationBC {
    fn ghost_state(&self, ctx: &BCContext) -> SWEState {
        ctx.interior_state
    }

    fn name(&self) -> &'static str {
        "extrapolation"
    }
}

/// Fixed state boundary condition (Dirichlet).
///
/// Sets a fixed state at the boundary, regardless of interior state.
#[derive(Clone, Debug)]
pub struct FixedStateBC {
    /// Fixed state to impose
    pub state: SWEState,
}

impl FixedStateBC {
    /// Create a new fixed state BC.
    pub fn new(h: f64, hu: f64) -> Self {
        Self {
            state: SWEState::new(h, hu),
        }
    }

    /// Create from primitive variables.
    pub fn from_primitives(h: f64, u: f64) -> Self {
        Self {
            state: SWEState::from_primitives(h, u),
        }
    }
}

impl SWEBoundaryCondition for FixedStateBC {
    fn ghost_state(&self, _ctx: &BCContext) -> SWEState {
        self.state
    }

    fn name(&self) -> &'static str {
        "fixed_state"
    }
}

/// Discharge boundary condition.
///
/// Prescribes the flow rate Q = h * u at the boundary.
/// The depth is determined from interior information.
#[derive(Clone, Debug)]
pub struct DischargeBC {
    /// Prescribed discharge (m²/s in 1D)
    pub discharge: f64,
    /// Minimum depth
    pub h_min: f64,
}

impl DischargeBC {
    /// Create a new discharge BC.
    pub fn new(discharge: f64) -> Self {
        Self {
            discharge,
            h_min: 1e-6,
        }
    }
}

impl SWEBoundaryCondition for DischargeBC {
    fn ghost_state(&self, ctx: &BCContext) -> SWEState {
        // Use interior depth, prescribed discharge
        let h = ctx.interior_state.h.max(self.h_min);
        SWEState::new(h, self.discharge)
    }

    fn name(&self) -> &'static str {
        "discharge"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bc_context() {
        let ctx = BCContext::new(1.0, 0.0, SWEState::new(2.0, 3.0), 0.5, -1.0);

        assert!(ctx.is_left_boundary());
        assert!(!ctx.is_right_boundary());
        assert!((ctx.interior_surface_elevation() - 2.5).abs() < 1e-14);
    }

    #[test]
    fn test_extrapolation_bc() {
        let bc = ExtrapolationBC;
        let ctx = BCContext::new(0.0, 0.0, SWEState::new(2.0, 3.0), 0.0, 1.0);

        let ghost = bc.ghost_state(&ctx);
        assert!((ghost.h - 2.0).abs() < 1e-14);
        assert!((ghost.hu - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_fixed_state_bc() {
        let bc = FixedStateBC::new(1.5, 2.0);
        let ctx = BCContext::new(0.0, 0.0, SWEState::new(2.0, 3.0), 0.0, 1.0);

        let ghost = bc.ghost_state(&ctx);
        assert!((ghost.h - 1.5).abs() < 1e-14);
        assert!((ghost.hu - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_discharge_bc() {
        let bc = DischargeBC::new(5.0);
        let ctx = BCContext::new(0.0, 0.0, SWEState::new(2.0, 3.0), 0.0, 1.0);

        let ghost = bc.ghost_state(&ctx);
        assert!((ghost.h - 2.0).abs() < 1e-14); // Uses interior depth
        assert!((ghost.hu - 5.0).abs() < 1e-14); // Prescribed discharge
    }
}
