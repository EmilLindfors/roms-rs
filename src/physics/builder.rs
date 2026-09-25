//! Physics module builders.
//!
//! This module provides builder patterns for constructing physics modules.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::boundary::SWEBoundaryCondition2D;
use crate::equations::ShallowWater2D;
use crate::flux::StandardFlux2D;
use crate::mesh::{Bathymetry2D, Mesh2D};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::{
    ImplicitDamping2D, Limiter2D, LimiterContext2D, SWEFormulation2D, SWESolution2D,
    StandardLimiter2D, WetDryConfig, positivity_cfl_swe_2d,
};
#[cfg(not(feature = "parallel"))]
use crate::solver::{
    apply_implicit_damping_2d as implicit_damping,
    apply_wet_dry_correction_all as wet_dry_correction,
};
#[cfg(feature = "parallel")]
use crate::solver::{
    apply_implicit_damping_2d_parallel as implicit_damping,
    apply_wet_dry_correction_all_parallel as wet_dry_correction,
};
use crate::source::{BottomFriction2D, SourceTerm2D};

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
/// - Wetting/drying and point-implicit bottom friction
///
/// # Wetting and drying
///
/// A run has wetting/drying when it has a positivity limiter
/// (`StandardLimiter2D::Positivity`/`KuzminWithPositivity`) or a
/// [`WetDryConfig`] (`with_wet_dry`). Then:
/// - the formulation defaults to `SWEFormulation2D::WetDry`, and the interface
///   flux of `Standard` to HLL (Roe is not positivity preserving);
/// - `max_cfl` reports `positivity_cfl_swe_2d(N)`, which `Simulation` enforces;
/// - after every RK stage, depths are limited to h ≥ 0 and near-dry velocities
///   desingularized ([`crate::solver::apply_wet_dry_correction_all`]);
/// - in every RK stage, bottom friction (`with_implicit_friction`)
///   and the thin-layer relaxation are applied point-implicitly
///   ([`crate::solver::apply_implicit_damping_2d`]).
///
/// Elements whose mean depth went negative anyway are emptied (creating mass)
/// and counted in [`SWEPhysics2D::negative_depth_clips`].
///
/// `SWEFormulation2D::WetDry` is exactly well-balanced at shorelines and
/// more accurate on moving shorelines than `Standard` with hydrostatic
/// reconstruction; its dry-node threshold is `WetDryConfig::h_dry`. It
/// includes the bed slope, so the source terms must not contain
/// `BathymetrySource2D`. `EntropyStable`/`EntropyConservative` have no wet/dry
/// interface treatment.
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
    /// Spatial formulation of the SWE operator
    pub formulation: SWEFormulation2D,
    /// Wetting/drying treatment, if enabled
    pub wet_dry: Option<WetDryConfig>,
    /// Bottom friction applied point-implicitly, if any
    pub friction: Option<Arc<dyn BottomFriction2D>>,
    /// Polynomial order
    pub order: usize,
    /// Elements emptied because their mean depth was negative
    negative_depth_clips: AtomicUsize,
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

impl<BC: SWEBoundaryCondition2D> SWEPhysics2D<BC> {
    /// Whether this run has wetting/drying (a positivity limiter or a
    /// [`WetDryConfig`]).
    pub fn has_wetting_drying(&self) -> bool {
        self.wet_dry.is_some() || self.limiter.preserves_positivity()
    }

    /// Number of elements (summed over all stages so far) whose mean depth
    /// was negative and that were emptied, creating mass.
    ///
    /// Zero under the positivity CFL with HLL or Rusanov; anything else means
    /// the time step or the flux broke the positivity guarantee.
    pub fn negative_depth_clips(&self) -> usize {
        self.negative_depth_clips.load(Ordering::Relaxed)
    }

    fn damping(&self) -> ImplicitDamping2D<'_> {
        ImplicitDamping2D {
            friction: self.friction.as_deref(),
            wet_dry: self.wet_dry.as_ref(),
            h_min: self.equation.h_min,
        }
    }

    /// RHS configuration for this module's components.
    fn rhs_config(&self) -> crate::solver::SWE2DRhsConfig<'_, BC> {
        use crate::solver::SWE2DRhsConfig;

        let mut config = SWE2DRhsConfig::new(&self.equation, &self.bc)
            .with_formulation(self.formulation)
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
        if let Some(ref wet_dry) = self.wet_dry {
            config = config.with_dry_threshold(wet_dry.h_dry.meters());
        }
        config
    }
}

impl<BC: SWEBoundaryCondition2D + 'static> PhysicsModule<SWESolution2D> for SWEPhysics2D<BC> {
    fn compute_rhs(&self, state: &SWESolution2D, time: f64) -> SWESolution2D {
        let mut out = SWESolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
        self.compute_rhs_into(state, time, &mut out);
        out
    }

    /// Parallel element kernel when `parallel` is enabled (identical result to
    /// the serial one); allocation-free after the first call.
    fn compute_rhs_into(&self, state: &SWESolution2D, time: f64, out: &mut SWESolution2D) {
        #[cfg(not(feature = "parallel"))]
        use crate::solver::compute_rhs_swe_2d_into as rhs_into;
        #[cfg(feature = "parallel")]
        use crate::solver::compute_rhs_swe_2d_parallel_into as rhs_into;

        let config = self.rhs_config();
        rhs_into(state, &self.mesh, &self.ops, &self.geom, &config, time, out);
    }

    fn compute_dt(&self, state: &SWESolution2D, cfl: f64) -> f64 {
        #[cfg(not(feature = "parallel"))]
        use crate::solver::compute_dt_swe_2d as dt;
        #[cfg(feature = "parallel")]
        use crate::solver::compute_dt_swe_2d_parallel as dt;

        dt(
            state,
            &self.mesh,
            &self.geom,
            &self.equation,
            self.order,
            cfl,
        )
    }

    /// Limiter, then positivity and velocity desingularization (wet/dry).
    fn post_process(&self, state: &mut SWESolution2D) {
        let ctx = LimiterContext2D::new(&self.mesh, &self.ops);
        let mut clips = self.limiter.apply_counting(state, &ctx);
        if let Some(ref config) = self.wet_dry {
            clips += wet_dry_correction(state, &self.ops, config);
        }
        if clips > 0 {
            self.negative_depth_clips
                .fetch_add(clips, Ordering::Relaxed);
        }
    }

    /// Point-implicit bottom friction and thin-layer relaxation.
    fn implicit_damping(&self, stage: &mut SWESolution2D, from: &SWESolution2D, dt: f64) {
        implicit_damping(stage, from, dt, &self.damping());
    }

    /// The DGSEM positivity bound when the run has wetting/drying.
    fn max_cfl(&self) -> Option<f64> {
        self.has_wetting_drying()
            .then(|| positivity_cfl_swe_2d(self.order))
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
    flux: Option<StandardFlux2D>,
    source: Option<Arc<dyn SourceTerm2D>>,
    bathymetry: Option<Arc<Bathymetry2D>>,
    limiter: StandardLimiter2D,
    well_balanced: bool,
    formulation: Option<SWEFormulation2D>,
    wet_dry: Option<WetDryConfig>,
    friction: Option<Arc<dyn BottomFriction2D>>,
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
            flux: None,
            source: None,
            bathymetry: None,
            limiter: StandardLimiter2D::default(),
            well_balanced: false,
            formulation: None,
            wet_dry: None,
            friction: None,
            order,
        }
    }

    /// Set the numerical flux.
    ///
    /// Default: HLL for runs with wetting/drying (a positivity limiter or
    /// [`Self::with_wet_dry`]), Roe otherwise. Roe is not positivity preserving
    /// (Einfeldt et al. 1991), so wet/dry runs should keep HLL or use Rusanov.
    pub fn with_flux(mut self, flux: StandardFlux2D) -> Self {
        self.flux = Some(flux);
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

    /// Set the spatial formulation of the SWE operator.
    ///
    /// Default: `WetDry` for runs with wetting/drying (a positivity limiter or
    /// [`Self::with_wet_dry`]), `Standard` otherwise. The split-form
    /// formulations include the bed slope in the operator, so the source terms
    /// must not contain `BathymetrySource2D`; choose `Standard` (with
    /// [`Self::with_well_balanced`]) to keep it.
    pub fn with_formulation(mut self, formulation: SWEFormulation2D) -> Self {
        self.formulation = Some(formulation);
        self
    }

    /// Enable wetting/drying with the given configuration (see
    /// [`SWEPhysics2D`]): positivity, velocity desingularization and thin-layer
    /// relaxation below `config.h_dry`.
    pub fn with_wet_dry(mut self, config: WetDryConfig) -> Self {
        self.wet_dry = Some(config);
        self
    }

    /// Enable (with [`WetDryConfig::default`], h_dry = 1 mm) or disable
    /// wetting/drying.
    pub fn with_wet_dry_correction(mut self, enabled: bool) -> Self {
        self.wet_dry = enabled.then(WetDryConfig::default);
        self
    }

    /// Apply a bottom friction law point-implicitly in every RK stage
    /// (`TimeIntegrator::step_with_relaxation`), instead of as an explicit
    /// source term.
    ///
    /// Unconditionally stable and sign-preserving: explicit friction flips the
    /// momentum once Δt·C_f|u|/h > 2.5, which happens at every wet/dry front.
    /// Do not also put the same law in the source terms.
    pub fn with_implicit_friction<F: BottomFriction2D + 'static>(mut self, friction: F) -> Self {
        self.friction = Some(Arc::new(friction));
        self
    }

    /// Build the physics module.
    ///
    /// # Panics
    /// If a wet/dry run leaves the formulation at its default (`WetDry`) but
    /// its source terms contain `BathymetrySource2D`, which would count the
    /// bed slope twice.
    pub fn build(self) -> SWEPhysics2D<BC> {
        let wetting_drying = self.wet_dry.is_some() || self.limiter.preserves_positivity();
        let formulation = self.formulation.unwrap_or_else(|| {
            if !wetting_drying {
                return SWEFormulation2D::Standard;
            }
            assert!(
                !self
                    .source
                    .as_ref()
                    .is_some_and(|s| s.includes_bathymetry_slope()),
                "wet/dry runs default to SWEFormulation2D::WetDry, which includes the \
                 bed slope: remove BathymetrySource2D from the sources, or keep it with \
                 .with_formulation(SWEFormulation2D::Standard)"
            );
            SWEFormulation2D::WetDry
        });
        let flux = self.flux.unwrap_or(if wetting_drying {
            StandardFlux2D::HLL
        } else {
            StandardFlux2D::Roe
        });
        if wetting_drying && flux == StandardFlux2D::Roe {
            eprintln!(
                "warning: SWEPhysics2D with wetting/drying and the Roe flux: \
                 Roe is not positivity preserving; use HLL or Rusanov"
            );
        }

        SWEPhysics2D {
            mesh: self.mesh,
            ops: self.ops,
            geom: self.geom,
            equation: self.equation,
            flux,
            bc: self.bc,
            source: self.source,
            bathymetry: self.bathymetry,
            limiter: self.limiter,
            well_balanced: self.well_balanced,
            formulation,
            wet_dry: self.wet_dry,
            friction: self.friction,
            order: self.order,
            negative_depth_clips: AtomicUsize::new(0),
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
            .with_formulation(SWEFormulation2D::EntropyStable)
            .with_wet_dry_correction(true)
            .build();

        assert!(physics.well_balanced);
        assert_eq!(physics.formulation, SWEFormulation2D::EntropyStable);
        assert!(physics.wet_dry.is_some());
    }

    #[test]
    fn test_wet_dry_defaults() {
        let (mesh, ops, geom) = create_test_components();
        let builder = || {
            PhysicsBuilder::swe_2d(
                mesh.clone(),
                ops.clone(),
                geom.clone(),
                ShallowWater2D::new(9.81),
                Reflective2D::default(),
            )
        };

        // Fully wet run: Standard with Roe, no CFL cap
        let wet = builder().build();
        assert_eq!(wet.formulation, SWEFormulation2D::Standard);
        assert_eq!(wet.flux, StandardFlux2D::Roe);
        assert!(!wet.has_wetting_drying());
        assert_eq!(wet.max_cfl(), None);

        // Wetting/drying via the wet/dry treatment or a positivity limiter:
        // WetDry, HLL (for Standard) and the positivity CFL (order 2)
        for physics in [
            builder().with_wet_dry_correction(true).build(),
            builder()
                .with_limiter(StandardLimiter2D::Positivity(1e-3))
                .build(),
        ] {
            assert_eq!(physics.formulation, SWEFormulation2D::WetDry);
            assert_eq!(physics.flux, StandardFlux2D::HLL);
            assert_eq!(physics.max_cfl(), Some(positivity_cfl_swe_2d(2)));
        }

        // An explicit choice wins
        let rusanov = builder()
            .with_wet_dry_correction(true)
            .with_formulation(SWEFormulation2D::Standard)
            .with_source(crate::source::BathymetrySource2D::new(9.81))
            .with_flux(StandardFlux2D::Rusanov)
            .build();
        assert_eq!(rusanov.formulation, SWEFormulation2D::Standard);
        assert_eq!(rusanov.flux, StandardFlux2D::Rusanov);
        assert_eq!(
            rusanov.wet_dry.unwrap().h_dry,
            Depth::new(WetDryConfig::DEFAULT_H_DRY)
        );
    }

    #[test]
    #[should_panic(expected = "remove BathymetrySource2D")]
    fn test_wet_dry_default_rejects_bathymetry_source() {
        let (mesh, ops, geom) = create_test_components();
        PhysicsBuilder::swe_2d(
            mesh,
            ops,
            geom,
            ShallowWater2D::new(9.81),
            Reflective2D::default(),
        )
        .with_wet_dry_correction(true)
        .with_source(crate::source::BathymetrySource2D::new(9.81))
        .build();
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
