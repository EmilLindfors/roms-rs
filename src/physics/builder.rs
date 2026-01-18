//! Physics module builders.
//!
//! This module provides builder patterns for constructing physics modules.

use std::sync::Arc;

use crate::boundary::SWEBoundaryCondition2D;
use crate::equations::ShallowWater2D;
use crate::flux::StandardFlux2D;
use crate::mesh::{Bathymetry2D, Mesh2D};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::{
    LimiterContext2D, SWESolution2D, StandardLimiter2D, WetDryConfig,
    apply_wet_dry_correction_all, Limiter2D,
};
use crate::source::SourceTerm2D;

use super::traits::{PhysicsModule, PhysicsModuleInfo};

// =============================================================================
// SWE Physics 2D
// =============================================================================

/// 2D Shallow Water Equations physics module.
///
/// This module encapsulates all components needed to simulate the 2D SWE:
/// - Mesh and geometric data
/// - DG operators
/// - Numerical flux
/// - Boundary conditions
/// - Source terms
/// - Limiters
pub struct SWEPhysics2D<BC: SWEBoundaryCondition2D> {
    /// The mesh
    pub mesh: Arc<Mesh2D>,
    /// DG operators
    pub ops: Arc<DGOperators2D>,
    /// Geometric factors
    pub geom: Arc<GeometricFactors2D>,
    /// Shallow water equation parameters
    pub equation: ShallowWater2D,
    /// Numerical flux
    pub flux: StandardFlux2D,
    /// Boundary condition
    pub bc: BC,
    /// Optional source terms
    pub source: Option<Arc<dyn SourceTerm2D>>,
    /// Optional bathymetry
    pub bathymetry: Option<Arc<Bathymetry2D>>,
    /// Limiter
    pub limiter: StandardLimiter2D,
    /// Whether to use well-balanced scheme
    pub well_balanced: bool,
    /// Whether to apply wet/dry correction
    pub wet_dry_correction: bool,
    /// Polynomial order
    pub order: usize,
}

impl<BC: SWEBoundaryCondition2D> PhysicsModuleInfo for SWEPhysics2D<BC> {
    fn name(&self) -> &'static str {
        "swe-2d"
    }

    fn description(&self) -> &str {
        "2D Shallow Water Equations"
    }

    fn n_variables(&self) -> usize {
        3
    }

    fn variable_names(&self) -> &[&'static str] {
        &["h", "hu", "hv"]
    }
}

impl<BC: SWEBoundaryCondition2D + 'static> PhysicsModule<SWESolution2D> for SWEPhysics2D<BC> {
    fn compute_rhs(&self, state: &SWESolution2D, time: f64) -> SWESolution2D {
        use crate::solver::compute_rhs_swe_2d;
        use crate::solver::SWE2DRhsConfig;

        // Build the RHS config
        let mut config = SWE2DRhsConfig::new(&self.equation, &self.bc)
            .with_flux_type(self.flux.into())
            .with_coriolis(false); // Use source terms instead

        if let Some(ref source) = self.source {
            config.source_terms = Some(source.as_ref());
        }

        if let Some(ref bathy) = self.bathymetry {
            config = config.with_bathymetry(bathy.as_ref());
            if self.well_balanced {
                config = config.with_well_balanced(true);
            }
        }

        compute_rhs_swe_2d(state, &self.mesh, &self.ops, &self.geom, &config, time)
    }

    fn compute_dt(&self, state: &SWESolution2D, cfl: f64) -> f64 {
        use crate::solver::compute_dt_swe_2d;

        compute_dt_swe_2d(state, &self.mesh, &self.geom, &self.equation, self.order, cfl)
    }

    fn post_process(&self, state: &mut SWESolution2D) {
        // Apply limiter
        let ctx = LimiterContext2D::new(&self.mesh, &self.ops);
        self.limiter.apply(state, &ctx);

        // Apply wet/dry correction
        if self.wet_dry_correction {
            let config = WetDryConfig::new(self.equation.h_min, self.equation.g);
            apply_wet_dry_correction_all(state, &config);
        }
    }

    fn mesh(&self) -> &Mesh2D {
        &self.mesh
    }

    fn operators(&self) -> &DGOperators2D {
        &self.ops
    }

    fn geometry(&self) -> &GeometricFactors2D {
        &self.geom
    }

    fn order(&self) -> usize {
        self.order
    }
}

// =============================================================================
// SWE Physics 2D Builder
// =============================================================================

/// Builder for 2D Shallow Water Equations physics module.
///
/// # Example
/// ```ignore
/// use dg_rs::physics::SWEPhysics2DBuilder;
/// use dg_rs::flux::StandardFlux2D;
/// use dg_rs::solver::StandardLimiter2D;
///
/// let physics = SWEPhysics2DBuilder::new(mesh, ops, geom, equation, bc)
///     .with_flux(StandardFlux2D::Roe)
///     .with_limiter(StandardLimiter2D::TvbWithPositivity { tvb, h_min })
///     .with_bathymetry(bathymetry)
///     .with_well_balanced(true)
///     .build();
/// ```
pub struct SWEPhysics2DBuilder<BC: SWEBoundaryCondition2D> {
    mesh: Arc<Mesh2D>,
    ops: Arc<DGOperators2D>,
    geom: Arc<GeometricFactors2D>,
    equation: ShallowWater2D,
    bc: BC,
    flux: StandardFlux2D,
    source: Option<Arc<dyn SourceTerm2D>>,
    bathymetry: Option<Arc<Bathymetry2D>>,
    limiter: StandardLimiter2D,
    well_balanced: bool,
    wet_dry_correction: bool,
    order: usize,
}

impl<BC: SWEBoundaryCondition2D> SWEPhysics2DBuilder<BC> {
    /// Create a new builder with required components.
    pub fn new(
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
        equation: ShallowWater2D,
        bc: BC,
    ) -> Self {
        let order = ops.order;
        Self {
            mesh,
            ops,
            geom,
            equation,
            bc,
            flux: StandardFlux2D::default(),
            source: None,
            bathymetry: None,
            limiter: StandardLimiter2D::default(),
            well_balanced: false,
            wet_dry_correction: false,
            order,
        }
    }

    /// Set the numerical flux.
    pub fn with_flux(mut self, flux: StandardFlux2D) -> Self {
        self.flux = flux;
        self
    }

    /// Set the source terms.
    pub fn with_source<S: SourceTerm2D + 'static>(mut self, source: S) -> Self {
        self.source = Some(Arc::new(source));
        self
    }

    /// Set the source terms from an Arc.
    pub fn with_source_arc(mut self, source: Arc<dyn SourceTerm2D>) -> Self {
        self.source = Some(source);
        self
    }

    /// Set the bathymetry.
    pub fn with_bathymetry(mut self, bathymetry: Arc<Bathymetry2D>) -> Self {
        self.bathymetry = Some(bathymetry);
        self
    }

    /// Set the limiter.
    pub fn with_limiter(mut self, limiter: StandardLimiter2D) -> Self {
        self.limiter = limiter;
        self
    }

    /// Enable well-balanced scheme for bathymetry.
    pub fn with_well_balanced(mut self, enabled: bool) -> Self {
        self.well_balanced = enabled;
        self
    }

    /// Enable wet/dry correction.
    pub fn with_wet_dry_correction(mut self, enabled: bool) -> Self {
        self.wet_dry_correction = enabled;
        self
    }

    /// Build the physics module.
    pub fn build(self) -> SWEPhysics2D<BC> {
        SWEPhysics2D {
            mesh: self.mesh,
            ops: self.ops,
            geom: self.geom,
            equation: self.equation,
            flux: self.flux,
            bc: self.bc,
            source: self.source,
            bathymetry: self.bathymetry,
            limiter: self.limiter,
            well_balanced: self.well_balanced,
            wet_dry_correction: self.wet_dry_correction,
            order: self.order,
        }
    }
}

// =============================================================================
// Generic Physics Builder
// =============================================================================

/// Generic entry point for physics builders.
///
/// Provides factory methods for creating specialized physics builders.
pub struct PhysicsBuilder;

impl PhysicsBuilder {
    /// Create a builder for 2D Shallow Water Equations.
    pub fn swe_2d<BC: SWEBoundaryCondition2D>(
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
        equation: ShallowWater2D,
        bc: BC,
    ) -> SWEPhysics2DBuilder<BC> {
        SWEPhysics2DBuilder::new(mesh, ops, geom, equation, bc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::Reflective2D;
    use crate::types::{Depth, ElementIndex};

    fn k(idx: usize) -> ElementIndex {
        ElementIndex::new(idx)
    }

    fn create_test_components() -> (Arc<Mesh2D>, Arc<DGOperators2D>, Arc<GeometricFactors2D>) {
        let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2));
        let ops = Arc::new(DGOperators2D::new(2));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh));
        (mesh, ops, geom)
    }

    #[test]
    fn test_builder_basic() {
        let (mesh, ops, geom) = create_test_components();
        let equation = ShallowWater2D::with_h_min(9.81, Depth::new(1e-6));
        let bc = Reflective2D::default();

        let physics = PhysicsBuilder::swe_2d(mesh, ops, geom, equation, bc).build();

        assert_eq!(physics.name(), "swe-2d");
        assert_eq!(physics.n_variables(), 3);
        assert_eq!(physics.order(), 2);
    }

    #[test]
    fn test_builder_with_options() {
        let (mesh, ops, geom) = create_test_components();
        let equation = ShallowWater2D::with_h_min(9.81, Depth::new(1e-6));
        let bc = Reflective2D::default();

        let physics = PhysicsBuilder::swe_2d(mesh, ops, geom, equation, bc)
            .with_flux(StandardFlux2D::HLL)
            .with_limiter(StandardLimiter2D::None)
            .with_well_balanced(true)
            .with_wet_dry_correction(true)
            .build();

        assert!(physics.well_balanced);
        assert!(physics.wet_dry_correction);
    }

    #[test]
    fn test_physics_module_info() {
        let (mesh, ops, geom) = create_test_components();
        let equation = ShallowWater2D::with_h_min(9.81, Depth::new(1e-6));
        let bc = Reflective2D::default();

        let physics = PhysicsBuilder::swe_2d(mesh, ops, geom, equation, bc).build();

        let info: &dyn PhysicsModuleInfo = &physics;
        assert_eq!(info.name(), "swe-2d");
        assert_eq!(info.variable_names(), &["h", "hu", "hv"]);
    }

    #[test]
    fn test_compute_dt() {
        let (mesh, ops, geom) = create_test_components();
        let equation = ShallowWater2D::with_h_min(9.81, Depth::new(1e-6));
        let bc = Reflective2D::default();

        let physics = PhysicsBuilder::swe_2d(mesh.clone(), ops.clone(), geom, equation, bc).build();

        // Create a state with some water
        let mut state = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                state.set_state(k(ki), i, crate::solver::SWEState2D::new(1.0, 0.0, 0.0));
            }
        }

        let dt = physics.compute_dt(&state, 0.5);
        assert!(dt > 0.0);
        assert!(dt < 1.0); // Should be a reasonable time step
    }
}
