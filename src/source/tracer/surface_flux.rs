//! Tracer source terms for 2D shallow water equations.
//!
//! Source terms for temperature and salinity transport:
//! - Surface heat flux: Solar radiation and atmospheric heat exchange
//! - River discharge: Freshwater input with prescribed T, S
//! - Relaxation: Nudging towards climatological values
//!
//! # Physical Background
//!
//! The tracer transport equations are:
//!   ∂(hT)/∂t + ∇·(hT**u**) = κ_T ∇²T + S_T
//!   ∂(hS)/∂t + ∇·(hS**u**) = κ_S ∇²S + S_S
//!
//! Source terms S_T and S_S represent:
//! - Heat exchange at the surface (S_T only)
//! - Freshwater/salt input from rivers (both S_T and S_S)
//! - Relaxation to external data (both)
//!
//! # Norwegian Coast Context
//!
//! Key tracer sources in Norwegian coastal waters:
//! - Surface heat flux: -100 to +200 W/m² (seasonal)
//! - River discharge: ~3700 m³/s total for Norwegian coast
//! - Atlantic water intrusion: T ≈ 7-8°C, S ≈ 35 PSU

use crate::solver::{ConservativeTracerState, SWEState2D, TracerSourceTerm2D, TracerState};

// ============================================================================
// Physical Constants
// ============================================================================

/// Specific heat capacity of seawater (J/(kg·K))
pub const CP_SEAWATER: f64 = 3985.0;

/// Reference density of seawater (kg/m³)
pub const RHO_SEAWATER: f64 = 1025.0;

/// Heat capacity per unit volume: ρ·c_p (J/(m³·K))
pub const RHO_CP: f64 = RHO_SEAWATER * CP_SEAWATER;

// ============================================================================
// Surface Heat Flux
// ============================================================================

/// Surface heat flux source term.
///
/// Models heat exchange at the water surface from:
/// - Solar (shortwave) radiation: heating
/// - Longwave radiation: cooling
/// - Sensible heat: air-sea temperature difference
/// - Latent heat: evaporation (cooling)
///
/// The source term converts heat flux Q (W/m²) to temperature rate:
///   d(hT)/dt = Q / (ρ·c_p)
///
/// where positive Q represents heating.
///
/// # Units
///
/// - Heat flux: W/m² = J/(s·m²)
/// - Source term: K·m/s (rate of change of depth-integrated temperature)
///
/// # Example
///
/// ```
/// use dg_rs::source::tracer::SurfaceHeatFlux;
/// use dg_rs::solver::TracerSourceTerm2D;
/// use dg_rs::solver::{SWEState2D, ConservativeTracerState};
///
/// // Summer daytime: 200 W/m² net heating
/// let heat_flux = SurfaceHeatFlux::constant(200.0);
///
/// let swe = SWEState2D::new(10.0, 0.0, 0.0);
/// let tracers = ConservativeTracerState::new(100.0, 350.0); // hT=100, hS=350
///
/// let source = heat_flux.evaluate(0.0, (0.0, 0.0), &swe, &tracers);
/// // source.h_t > 0 (heating), source.h_s = 0 (no salt change)
/// ```
#[derive(Debug)]
pub struct SurfaceHeatFlux {
    /// Net heat flux function Q(x, y, t) in W/m²
    flux_fn: HeatFluxFunction,
    /// Minimum depth for source application
    h_min: f64,
}

/// Heat flux function type.
enum HeatFluxFunction {
    /// Constant heat flux
    Constant(f64),
    /// Spatially uniform, time-varying
    TimeVarying(Box<dyn Fn(f64) -> f64 + Send + Sync>),
    /// Spatially and temporally varying
    SpatioTemporal(Box<dyn Fn(f64, f64, f64) -> f64 + Send + Sync>),
}

impl std::fmt::Debug for HeatFluxFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HeatFluxFunction::Constant(q) => write!(f, "Constant({q})"),
            HeatFluxFunction::TimeVarying(_) => write!(f, "TimeVarying(...)"),
            HeatFluxFunction::SpatioTemporal(_) => write!(f, "SpatioTemporal(...)"),
        }
    }
}

impl SurfaceHeatFlux {
    /// Create a constant heat flux source.
    ///
    /// # Arguments
    /// * `q_net` - Net heat flux in W/m² (positive = heating)
    pub fn constant(q_net: f64) -> Self {
        Self {
            flux_fn: HeatFluxFunction::Constant(q_net),
            h_min: 1e-6,
        }
    }

    /// Create a time-varying heat flux (uniform in space).
    ///
    /// Useful for diurnal cycles or seasonal forcing.
    ///
    /// # Arguments
    /// * `flux_fn` - Function Q(t) returning heat flux in W/m²
    pub fn time_varying<F>(flux_fn: F) -> Self
    where
        F: Fn(f64) -> f64 + Send + Sync + 'static,
    {
        Self {
            flux_fn: HeatFluxFunction::TimeVarying(Box::new(flux_fn)),
            h_min: 1e-6,
        }
    }

    /// Create a spatially and temporally varying heat flux.
    ///
    /// # Arguments
    /// * `flux_fn` - Function Q(x, y, t) returning heat flux in W/m²
    pub fn spatio_temporal<F>(flux_fn: F) -> Self
    where
        F: Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
    {
        Self {
            flux_fn: HeatFluxFunction::SpatioTemporal(Box::new(flux_fn)),
            h_min: 1e-6,
        }
    }

    /// Set minimum depth threshold.
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }

    /// Norwegian coast summer day (typical July noon).
    ///
    /// Net heating of ~150 W/m² (solar minus losses).
    pub fn norwegian_summer_day() -> Self {
        Self::constant(150.0)
    }

    /// Norwegian coast winter (typical January).
    ///
    /// Net cooling of ~-80 W/m² (longwave + latent losses).
    pub fn norwegian_winter() -> Self {
        Self::constant(-80.0)
    }

    /// Diurnal cycle with sinusoidal variation.
    ///
    /// # Arguments
    /// * `q_mean` - Mean heat flux (W/m²)
    /// * `q_amp` - Amplitude of diurnal variation (W/m²)
    /// * `period` - Diurnal period in seconds (typically 86400)
    pub fn diurnal_cycle(q_mean: f64, q_amp: f64, period: f64) -> Self {
        Self::time_varying(move |t| {
            let phase = 2.0 * std::f64::consts::PI * t / period;
            q_mean + q_amp * phase.cos()
        })
    }

    /// Evaluate heat flux at a point.
    fn evaluate_flux(&self, time: f64, position: (f64, f64)) -> f64 {
        match &self.flux_fn {
            HeatFluxFunction::Constant(q) => *q,
            HeatFluxFunction::TimeVarying(f) => f(time),
            HeatFluxFunction::SpatioTemporal(f) => f(position.0, position.1, time),
        }
    }
}

impl TracerSourceTerm2D for SurfaceHeatFlux {
    fn evaluate(
        &self,
        time: f64,
        position: (f64, f64),
        swe: &SWEState2D,
        _tracers: &ConservativeTracerState,
    ) -> ConservativeTracerState {
        if swe.h < self.h_min {
            return ConservativeTracerState::zero();
        }

        let q = self.evaluate_flux(time, position);

        // Convert W/m² to K·m/s:
        // d(hT)/dt = Q / (ρ·c_p)
        let s_ht = q / RHO_CP;

        ConservativeTracerState {
            h_t: s_ht,
            h_s: 0.0, // Heat flux doesn't affect salinity
        }
    }

    fn name(&self) -> &'static str {
        "surface_heat_flux"
    }
}

// ============================================================================
// River Tracer Source
// ============================================================================

/// River discharge tracer source term.
///
/// Models freshwater input at localized regions (river mouths).
/// The discharge brings water with prescribed temperature and salinity
/// (typically S ≈ 0 for rivers).
///
/// # Physical Model
///
/// The source term represents volume dilution:
///   S_hT = Q_river / A_cell * (T_river - T_local)
///   S_hS = Q_river / A_cell * (S_river - S_local)
///
/// where Q_river is discharge rate and A_cell is the cell area.
///
/// This is best used with a localized function that concentrates the
/// discharge at river mouth cells.
///
/// # Example
///
/// ```
/// use dg_rs::source::tracer::RiverTracerSource;
/// use dg_rs::solver::TracerSourceTerm2D;
/// use dg_rs::solver::{SWEState2D, ConservativeTracerState, TracerState};
///
/// // Glomma river: 700 m³/s, 8°C, freshwater
/// let river = RiverTracerSource::new(
///     700.0,                           // discharge rate m³/s
///     TracerState::new(8.0, 0.0),      // river water properties
///     |x, y| if x < 1000.0 && y < 500.0 { 1.0 } else { 0.0 }, // localization
///     500_000.0,                       // source region area m²
/// );
/// ```
#[derive(Clone)]
pub struct RiverTracerSource<F: Fn(f64, f64) -> f64 + Send + Sync> {
    /// Total discharge rate (m³/s)
    discharge: f64,
    /// River water tracer properties
    tracers: TracerState,
    /// Localization function: weight(x, y) in [0, 1]
    localization: F,
    /// Total area of the source region (m²)
    source_area: f64,
    /// Minimum depth for source application
    h_min: f64,
}

impl<F: Fn(f64, f64) -> f64 + Send + Sync> RiverTracerSource<F> {
    /// Create a new river tracer source.
    ///
    /// # Arguments
    /// * `discharge` - Total discharge rate (m³/s)
    /// * `tracers` - River water properties (T, S)
    /// * `localization` - Weight function identifying source cells
    /// * `source_area` - Total area over which discharge is distributed (m²)
    pub fn new(discharge: f64, tracers: TracerState, localization: F, source_area: f64) -> Self {
        Self {
            discharge,
            tracers,
            localization,
            source_area,
            h_min: 1e-6,
        }
    }

    /// Set minimum depth threshold.
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }
}

impl<F: Fn(f64, f64) -> f64 + Send + Sync> TracerSourceTerm2D for RiverTracerSource<F> {
    fn evaluate(
        &self,
        _time: f64,
        position: (f64, f64),
        swe: &SWEState2D,
        tracers: &ConservativeTracerState,
    ) -> ConservativeTracerState {
        if swe.h < self.h_min {
            return ConservativeTracerState::zero();
        }

        let (x, y) = position;
        let weight = (self.localization)(x, y);

        if weight < 1e-10 {
            return ConservativeTracerState::zero();
        }

        // Volume flux per unit area: Q / A [m/s]
        let volume_flux = self.discharge * weight / self.source_area;

        // Local tracer concentrations
        let local = tracers.to_concentrations(swe.h, self.h_min);

        // Source: dilution/concentration effect
        // d(hT)/dt = volume_flux * (T_river - T_local)
        ConservativeTracerState {
            h_t: volume_flux * (self.tracers.temperature - local.temperature),
            h_s: volume_flux * (self.tracers.salinity - local.salinity),
        }
    }

    fn name(&self) -> &'static str {
        "river_tracer"
    }
}

/// Gaussian localization for river sources.
///
/// Creates a Gaussian distribution centered at (x0, y0) with width sigma.
/// The weight integrates to approximately 1 over the domain.
///
/// # Arguments
/// * `x0`, `y0` - Center of the Gaussian (river mouth location)
/// * `sigma` - Width of the Gaussian (m)
pub fn gaussian_river_localization(x0: f64, y0: f64, sigma: f64) -> impl Fn(f64, f64) -> f64 {
    let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
    move |x: f64, y: f64| {
        let r2 = (x - x0).powi(2) + (y - y0).powi(2);
        (-r2 * inv_2sigma2).exp()
    }
}

/// Rectangular localization for river sources.
///
/// Returns 1 inside the rectangle, 0 outside.
///
/// # Arguments
/// * `x_min`, `x_max` - X bounds of source region
/// * `y_min`, `y_max` - Y bounds of source region
pub fn rectangular_river_localization(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> impl Fn(f64, f64) -> f64 {
    move |x: f64, y: f64| {
        if x >= x_min && x <= x_max && y >= y_min && y <= y_max {
            1.0
        } else {
            0.0
        }
    }
}

// ============================================================================
// Relaxation Source
// ============================================================================

/// Uniform relaxation (nudging) source term for tracers.
///
/// Nudges tracer values towards constant target values with uniform rate:
///   S_hT = -γ_T * h * (T - T_target)
///   S_hS = -γ_S * h * (S - S_target)
///
/// where γ is the relaxation rate (1/s).
///
/// # Usage
///
/// Commonly used for:
/// - Simple climatological relaxation
/// - Initial spinup
///
/// # Example
///
/// ```
/// use dg_rs::source::tracer::UniformRelaxationTracerSource;
/// use dg_rs::solver::TracerSourceTerm2D;
/// use dg_rs::solver::{SWEState2D, ConservativeTracerState, TracerState};
///
/// // Relax towards climatology with 1-day timescale
/// let relax = UniformRelaxationTracerSource::new(
///     TracerState::new(10.0, 34.5), // target T=10°C, S=34.5 PSU
///     1.0 / 86400.0,                // γ = 1/day
/// );
/// ```
#[derive(Clone, Copy, Debug)]
pub struct UniformRelaxationTracerSource {
    /// Target tracer values
    target: TracerState,
    /// Relaxation coefficient (1/s)
    gamma: f64,
    /// Relaxation coefficient scale for temperature
    gamma_t_scale: f64,
    /// Relaxation coefficient scale for salinity
    gamma_s_scale: f64,
    /// Minimum depth for source application
    h_min: f64,
}

impl UniformRelaxationTracerSource {
    /// Create relaxation towards uniform target values with constant rate.
    ///
    /// # Arguments
    /// * `target` - Target tracer state
    /// * `gamma` - Relaxation rate (1/s). Use 1/τ where τ is timescale.
    pub fn new(target: TracerState, gamma: f64) -> Self {
        Self {
            target,
            gamma,
            gamma_t_scale: 1.0,
            gamma_s_scale: 1.0,
            h_min: 1e-6,
        }
    }

    /// Set different relaxation rates for T and S.
    pub fn with_separate_rates(mut self, gamma_t_scale: f64, gamma_s_scale: f64) -> Self {
        self.gamma_t_scale = gamma_t_scale;
        self.gamma_s_scale = gamma_s_scale;
        self
    }

    /// Set minimum depth threshold.
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }
}

impl TracerSourceTerm2D for UniformRelaxationTracerSource {
    fn evaluate(
        &self,
        _time: f64,
        _position: (f64, f64),
        swe: &SWEState2D,
        tracers: &ConservativeTracerState,
    ) -> ConservativeTracerState {
        if swe.h < self.h_min {
            return ConservativeTracerState::zero();
        }

        let local = tracers.to_concentrations(swe.h, self.h_min);

        let gamma_t = self.gamma * self.gamma_t_scale;
        let gamma_s = self.gamma * self.gamma_s_scale;

        ConservativeTracerState {
            h_t: -gamma_t * swe.h * (local.temperature - self.target.temperature),
            h_s: -gamma_s * swe.h * (local.salinity - self.target.salinity),
        }
    }

    fn name(&self) -> &'static str {
        "uniform_relaxation_tracer"
    }
}

/// Spatially varying relaxation (nudging) source term for tracers.
///
/// Nudges tracer values towards prescribed target values:
///   S_hT = -γ_T * h * (T - T_target)
///   S_hS = -γ_S * h * (S - S_target)
///
/// where γ is the relaxation rate (1/s) and both target and rate
/// can vary in space and time.
///
/// # Usage
///
/// Commonly used for:
/// - Boundary buffer zones (sponge layers for tracers)
/// - Assimilating observations
/// - Preventing drift from climatology
///
/// # Example
///
/// ```
/// use dg_rs::source::tracer::RelaxationTracerSource;
/// use dg_rs::solver::TracerSourceTerm2D;
/// use dg_rs::solver::{SWEState2D, ConservativeTracerState, TracerState};
///
/// // Spatially varying relaxation
/// let relax = RelaxationTracerSource::spatially_varying(
///     |x, y, t| TracerState::new(10.0 + 0.001 * x, 34.5),
///     |x, y| if x < 1000.0 { 0.001 } else { 0.0 },
/// );
/// ```
#[derive(Clone)]
pub struct RelaxationTracerSource<F, G>
where
    F: Fn(f64, f64, f64) -> TracerState + Send + Sync,
    G: Fn(f64, f64) -> f64 + Send + Sync,
{
    /// Target tracer values T_target(x, y, t), S_target(x, y, t)
    target_fn: F,
    /// Relaxation coefficient γ(x, y) (1/s)
    gamma_fn: G,
    /// Relaxation coefficient for temperature (1/s), if different from gamma_fn
    gamma_t_scale: f64,
    /// Relaxation coefficient for salinity (1/s), if different from gamma_fn
    gamma_s_scale: f64,
    /// Minimum depth for source application
    h_min: f64,
}

impl<F, G> RelaxationTracerSource<F, G>
where
    F: Fn(f64, f64, f64) -> TracerState + Send + Sync,
    G: Fn(f64, f64) -> f64 + Send + Sync,
{
    /// Create relaxation with spatially varying target and rate.
    ///
    /// # Arguments
    /// * `target_fn` - Target values T_target(x, y, t), S_target(x, y, t)
    /// * `gamma_fn` - Relaxation rate γ(x, y) (1/s)
    pub fn spatially_varying(target_fn: F, gamma_fn: G) -> Self {
        Self {
            target_fn,
            gamma_fn,
            gamma_t_scale: 1.0,
            gamma_s_scale: 1.0,
            h_min: 1e-6,
        }
    }

    /// Set different relaxation rates for T and S.
    ///
    /// The actual rates are gamma_fn(x,y) * scale.
    pub fn with_separate_rates(mut self, gamma_t_scale: f64, gamma_s_scale: f64) -> Self {
        self.gamma_t_scale = gamma_t_scale;
        self.gamma_s_scale = gamma_s_scale;
        self
    }

    /// Set minimum depth threshold.
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }
}

impl<F, G> TracerSourceTerm2D for RelaxationTracerSource<F, G>
where
    F: Fn(f64, f64, f64) -> TracerState + Send + Sync,
    G: Fn(f64, f64) -> f64 + Send + Sync,
{
    fn evaluate(
        &self,
        time: f64,
        position: (f64, f64),
        swe: &SWEState2D,
        tracers: &ConservativeTracerState,
    ) -> ConservativeTracerState {
        if swe.h < self.h_min {
            return ConservativeTracerState::zero();
        }

        let (x, y) = position;
        let target = (self.target_fn)(x, y, time);
        let gamma = (self.gamma_fn)(x, y);

        // Local tracer concentrations
        let local = tracers.to_concentrations(swe.h, self.h_min);

        // Relaxation: -γ * h * (C - C_target)
        // In conservative form: -γ * (hC - h*C_target)
        let gamma_t = gamma * self.gamma_t_scale;
        let gamma_s = gamma * self.gamma_s_scale;

        ConservativeTracerState {
            h_t: -gamma_t * swe.h * (local.temperature - target.temperature),
            h_s: -gamma_s * swe.h * (local.salinity - target.salinity),
        }
    }

    fn name(&self) -> &'static str {
        "relaxation_tracer"
    }
}

/// Sponge-like relaxation that increases towards domain boundaries.
///
/// Creates a relaxation coefficient that is zero in the interior and
/// increases towards specified boundaries.
///
/// # Arguments
/// * `target` - Target tracer state
/// * `gamma_max` - Maximum relaxation rate at boundary (1/s)
/// * `width` - Width of the sponge region (m)
/// * `x_bounds` - Domain bounds (x_min, x_max)
/// * `y_bounds` - Domain bounds (y_min, y_max)
pub fn sponge_relaxation(
    target: TracerState,
    gamma_max: f64,
    width: f64,
    x_bounds: (f64, f64),
    y_bounds: (f64, f64),
) -> RelaxationTracerSource<
    impl Fn(f64, f64, f64) -> TracerState + Send + Sync,
    impl Fn(f64, f64) -> f64 + Send + Sync,
> {
    let (x_min, x_max) = x_bounds;
    let (y_min, y_max) = y_bounds;

    RelaxationTracerSource::spatially_varying(
        move |_, _, _| target,
        move |x, y| {
            // Distance from boundaries
            let dx_min = (x - x_min) / width;
            let dx_max = (x_max - x) / width;
            let dy_min = (y - y_min) / width;
            let dy_max = (y_max - y) / width;

            let d = dx_min.min(dx_max).min(dy_min).min(dy_max).clamp(0.0, 1.0);

            // Quadratic ramp-up from boundary
            if d < 1.0 {
                gamma_max * (1.0 - d).powi(2)
            } else {
                0.0
            }
        },
    )
}

// ============================================================================
// Combined Tracer Source
// ============================================================================

/// Combine multiple tracer source terms.
///
/// Evaluates all source terms and sums their contributions.
///
/// # Example
///
/// ```
/// use dg_rs::source::tracer::{CombinedTracerSource2D, SurfaceHeatFlux};
/// use dg_rs::solver::{SWEState2D, ConservativeTracerState, TracerSourceTerm2D};
///
/// let heat = SurfaceHeatFlux::constant(100.0);
/// let combined = CombinedTracerSource2D::new(vec![Box::new(heat)]);
/// ```
pub struct CombinedTracerSource2D {
    sources: Vec<Box<dyn TracerSourceTerm2D>>,
}

impl CombinedTracerSource2D {
    /// Create a combined source from a list of boxed source terms.
    pub fn new(sources: Vec<Box<dyn TracerSourceTerm2D>>) -> Self {
        Self { sources }
    }

    /// Create an empty combined source.
    pub fn empty() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    /// Add a source term to this combined source.
    #[allow(clippy::should_implement_trait)] // We use builder pattern, not std::ops::Add
    pub fn add<S: TracerSourceTerm2D + 'static>(mut self, source: S) -> Self {
        self.sources.push(Box::new(source));
        self
    }

    /// Number of source terms.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}

impl TracerSourceTerm2D for CombinedTracerSource2D {
    fn evaluate(
        &self,
        time: f64,
        position: (f64, f64),
        swe: &SWEState2D,
        tracers: &ConservativeTracerState,
    ) -> ConservativeTracerState {
        let mut total = ConservativeTracerState::zero();

        for source in &self.sources {
            let contrib = source.evaluate(time, position, swe, tracers);
            total = ConservativeTracerState {
                h_t: total.h_t + contrib.h_t,
                h_s: total.h_s + contrib.h_s,
            };
        }

        total
    }

    fn name(&self) -> &'static str {
        "combined_tracer"
    }
}

// ============================================================================
// Reference-based combined source (avoids Box allocation)
// ============================================================================

/// Combined tracer source using references (no allocation).
///
/// Useful when source terms have known types at compile time.
pub struct CombinedTracerSourceRef<'a> {
    sources: Vec<&'a dyn TracerSourceTerm2D>,
}

impl<'a> CombinedTracerSourceRef<'a> {
    /// Create from a vector of source references.
    pub fn new(sources: Vec<&'a dyn TracerSourceTerm2D>) -> Self {
        Self { sources }
    }

    /// Create an empty combined source.
    pub fn empty() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    /// Number of source terms.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}

impl<'a> TracerSourceTerm2D for CombinedTracerSourceRef<'a> {
    fn evaluate(
        &self,
        time: f64,
        position: (f64, f64),
        swe: &SWEState2D,
        tracers: &ConservativeTracerState,
    ) -> ConservativeTracerState {
        let mut total = ConservativeTracerState::zero();

        for source in &self.sources {
            let contrib = source.evaluate(time, position, swe, tracers);
            total = ConservativeTracerState {
                h_t: total.h_t + contrib.h_t,
                h_s: total.h_s + contrib.h_s,
            };
        }

        total
    }

    fn name(&self) -> &'static str {
        "combined_tracer_ref"
    }
}

// ============================================================================
// Norwegian Coast Presets
// ============================================================================

/// Major Norwegian rivers with typical properties.
pub mod norwegian_rivers {
    use super::*;

    /// Glomma river (largest in Norway).
    ///
    /// Mean discharge: ~700 m³/s
    /// Temperature: 4-18°C (seasonal)
    /// Salinity: ~0 PSU
    pub fn glomma<F: Fn(f64, f64) -> f64 + Send + Sync>(
        localization: F,
        source_area: f64,
        temperature: f64,
    ) -> RiverTracerSource<F> {
        RiverTracerSource::new(
            700.0,
            TracerState::new(temperature, 0.0),
            localization,
            source_area,
        )
    }

    /// Drammenselva river.
    ///
    /// Mean discharge: ~300 m³/s
    pub fn drammenselva<F: Fn(f64, f64) -> f64 + Send + Sync>(
        localization: F,
        source_area: f64,
        temperature: f64,
    ) -> RiverTracerSource<F> {
        RiverTracerSource::new(
            300.0,
            TracerState::new(temperature, 0.0),
            localization,
            source_area,
        )
    }

    /// Numedalslågen river.
    ///
    /// Mean discharge: ~120 m³/s
    pub fn numedalslågen<F: Fn(f64, f64) -> f64 + Send + Sync>(
        localization: F,
        source_area: f64,
        temperature: f64,
    ) -> RiverTracerSource<F> {
        RiverTracerSource::new(
            120.0,
            TracerState::new(temperature, 0.0),
            localization,
            source_area,
        )
    }

    /// Generic Norwegian river with custom discharge.
    pub fn generic<F: Fn(f64, f64) -> f64 + Send + Sync>(
        discharge: f64,
        localization: F,
        source_area: f64,
        temperature: f64,
    ) -> RiverTracerSource<F> {
        RiverTracerSource::new(
            discharge,
            TracerState::new(temperature, 0.0),
            localization,
            source_area,
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    const G: f64 = 9.81;
    #[allow(dead_code)]
    const H_MIN: f64 = 1e-6;

    fn make_swe(h: f64) -> SWEState2D {
        SWEState2D::new(h, 0.0, 0.0)
    }

    fn make_tracers(h: f64, t: f64, s: f64) -> ConservativeTracerState {
        ConservativeTracerState::from_depth_and_tracers(h, TracerState::new(t, s))
    }

    // -------------------------------------------------------------------------
    // Surface Heat Flux Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_heat_flux_constant_heating() {
        let q_net = 200.0; // W/m² heating
        let flux = SurfaceHeatFlux::constant(q_net);

        let swe = make_swe(10.0);
        let tracers = make_tracers(10.0, 10.0, 35.0);

        let source = flux.evaluate(0.0, (0.0, 0.0), &swe, &tracers);

        // Expected: d(hT)/dt = Q / (ρ·c_p)
        let expected = q_net / RHO_CP;
        assert!((source.h_t - expected).abs() < 1e-12);
        assert!(source.h_s.abs() < 1e-14); // No salt change
    }

    #[test]
    fn test_heat_flux_constant_cooling() {
        let q_net = -100.0; // W/m² cooling
        let flux = SurfaceHeatFlux::constant(q_net);

        let swe = make_swe(5.0);
        let tracers = make_tracers(5.0, 15.0, 34.0);

        let source = flux.evaluate(0.0, (0.0, 0.0), &swe, &tracers);

        assert!(source.h_t < 0.0); // Cooling
        assert!(source.h_s.abs() < 1e-14);
    }

    #[test]
    fn test_heat_flux_dry_cell() {
        let flux = SurfaceHeatFlux::constant(200.0).with_h_min(0.01);

        let swe = make_swe(0.001); // Dry
        let tracers = make_tracers(0.001, 10.0, 35.0);

        let source = flux.evaluate(0.0, (0.0, 0.0), &swe, &tracers);

        assert!(source.h_t.abs() < 1e-14);
        assert!(source.h_s.abs() < 1e-14);
    }

    #[test]
    fn test_heat_flux_time_varying() {
        let period = 86400.0;
        let flux = SurfaceHeatFlux::diurnal_cycle(100.0, 50.0, period);

        // At t=0, Q = 100 + 50*cos(0) = 150
        let swe = make_swe(10.0);
        let tracers = make_tracers(10.0, 10.0, 35.0);

        let source_0 = flux.evaluate(0.0, (0.0, 0.0), &swe, &tracers);
        let expected_0 = 150.0 / RHO_CP;
        assert!((source_0.h_t - expected_0).abs() < 1e-12);

        // At t=period/2, Q = 100 + 50*cos(π) = 50
        let source_half = flux.evaluate(period / 2.0, (0.0, 0.0), &swe, &tracers);
        let expected_half = 50.0 / RHO_CP;
        assert!((source_half.h_t - expected_half).abs() < 1e-12);
    }

    #[test]
    fn test_heat_flux_spatio_temporal() {
        // Flux that varies with position and time
        let flux = SurfaceHeatFlux::spatio_temporal(|x, y, t| {
            100.0 * (1.0 + 0.1 * x + 0.2 * y + 0.01 * t)
        });

        let swe = make_swe(10.0);
        let tracers = make_tracers(10.0, 10.0, 35.0);

        let source = flux.evaluate(10.0, (5.0, 3.0), &swe, &tracers);
        // Q = 100 * (1 + 0.5 + 0.6 + 0.1) = 220
        let expected = 220.0 / RHO_CP;
        assert!((source.h_t - expected).abs() < 1e-10);
    }

    #[test]
    fn test_norwegian_presets() {
        let summer = SurfaceHeatFlux::norwegian_summer_day();
        let winter = SurfaceHeatFlux::norwegian_winter();

        let swe = make_swe(10.0);
        let tracers = make_tracers(10.0, 10.0, 35.0);

        let summer_source = summer.evaluate(0.0, (0.0, 0.0), &swe, &tracers);
        let winter_source = winter.evaluate(0.0, (0.0, 0.0), &swe, &tracers);

        assert!(summer_source.h_t > 0.0); // Heating
        assert!(winter_source.h_t < 0.0); // Cooling
    }

    // -------------------------------------------------------------------------
    // River Tracer Source Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_river_source_freshening() {
        // River with 100 m³/s, 10°C, 0 PSU
        let river = RiverTracerSource::new(
            100.0,
            TracerState::new(10.0, 0.0),
            |_, _| 1.0, // Uniform in source region
            10000.0,    // 10000 m² source area
        );

        let swe = make_swe(5.0);
        let tracers = make_tracers(5.0, 15.0, 35.0); // Ocean water

        let source = river.evaluate(0.0, (0.0, 0.0), &swe, &tracers);

        // Volume flux = 100 / 10000 = 0.01 m/s
        // S_hT = 0.01 * (10 - 15) = -0.05
        // S_hS = 0.01 * (0 - 35) = -0.35
        assert!((source.h_t - (-0.05)).abs() < 1e-10);
        assert!((source.h_s - (-0.35)).abs() < 1e-10);
    }

    #[test]
    fn test_river_source_localization() {
        let river = RiverTracerSource::new(
            100.0,
            TracerState::new(10.0, 0.0),
            rectangular_river_localization(0.0, 100.0, 0.0, 100.0),
            10000.0,
        );

        let swe = make_swe(5.0);
        let tracers = make_tracers(5.0, 15.0, 35.0);

        // Inside source region
        let source_inside = river.evaluate(0.0, (50.0, 50.0), &swe, &tracers);
        assert!(source_inside.h_s.abs() > 1e-10);

        // Outside source region
        let source_outside = river.evaluate(0.0, (200.0, 200.0), &swe, &tracers);
        assert!(source_outside.h_s.abs() < 1e-14);
    }

    #[test]
    fn test_river_gaussian_localization() {
        let loc = gaussian_river_localization(0.0, 0.0, 100.0);

        // At center, weight = 1
        let w_center = loc(0.0, 0.0);
        assert!((w_center - 1.0).abs() < 1e-10);

        // At sigma distance, weight = exp(-0.5) ≈ 0.606
        let w_sigma = loc(100.0, 0.0);
        assert!((w_sigma - (-0.5_f64).exp()).abs() < 1e-10);

        // Far away, weight ≈ 0
        let w_far = loc(1000.0, 0.0);
        assert!(w_far < 0.01);
    }

    #[test]
    fn test_river_dry_cell() {
        let river = RiverTracerSource::new(100.0, TracerState::new(10.0, 0.0), |_, _| 1.0, 10000.0)
            .with_h_min(0.1);

        let swe = make_swe(0.01); // Dry
        let tracers = make_tracers(0.01, 15.0, 35.0);

        let source = river.evaluate(0.0, (0.0, 0.0), &swe, &tracers);

        assert!(source.h_t.abs() < 1e-14);
        assert!(source.h_s.abs() < 1e-14);
    }

    // -------------------------------------------------------------------------
    // Relaxation Source Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_relaxation_towards_target() {
        let target = TracerState::new(10.0, 35.0);
        let gamma = 0.001; // 1/s
        let relax = UniformRelaxationTracerSource::new(target, gamma);

        let swe = make_swe(5.0);
        let tracers = make_tracers(5.0, 15.0, 30.0); // T too high, S too low

        let source = relax.evaluate(0.0, (0.0, 0.0), &swe, &tracers);

        // S_hT = -γ * h * (T - T_target) = -0.001 * 5 * (15 - 10) = -0.025
        // S_hS = -γ * h * (S - S_target) = -0.001 * 5 * (30 - 35) = +0.025
        assert!((source.h_t - (-0.025)).abs() < 1e-10);
        assert!((source.h_s - 0.025).abs() < 1e-10);
    }

    #[test]
    fn test_relaxation_at_target() {
        let target = TracerState::new(10.0, 35.0);
        let relax = UniformRelaxationTracerSource::new(target, 0.001);

        let swe = make_swe(5.0);
        let tracers = make_tracers(5.0, 10.0, 35.0); // At target

        let source = relax.evaluate(0.0, (0.0, 0.0), &swe, &tracers);

        assert!(source.h_t.abs() < 1e-12);
        assert!(source.h_s.abs() < 1e-12);
    }

    #[test]
    fn test_relaxation_separate_rates() {
        let target = TracerState::new(10.0, 35.0);
        let gamma = 0.001;
        let relax = UniformRelaxationTracerSource::new(target, gamma).with_separate_rates(2.0, 0.5);

        let swe = make_swe(5.0);
        let tracers = make_tracers(5.0, 15.0, 30.0);

        let source = relax.evaluate(0.0, (0.0, 0.0), &swe, &tracers);

        // S_hT = -γ*2 * h * (15 - 10) = -0.05
        // S_hS = -γ*0.5 * h * (30 - 35) = +0.0125
        assert!((source.h_t - (-0.05)).abs() < 1e-10);
        assert!((source.h_s - 0.0125).abs() < 1e-10);
    }

    #[test]
    fn test_sponge_relaxation() {
        let target = TracerState::new(10.0, 35.0);
        let sponge = sponge_relaxation(
            target,
            0.01,           // gamma_max
            1000.0,         // width
            (0.0, 10000.0), // x_bounds
            (0.0, 10000.0), // y_bounds
        );

        let swe = make_swe(5.0);
        let tracers = make_tracers(5.0, 15.0, 30.0);

        // At boundary: maximum relaxation
        let source_boundary = sponge.evaluate(0.0, (0.0, 5000.0), &swe, &tracers);
        assert!(source_boundary.h_t.abs() > 0.01); // Strong relaxation

        // In interior: no relaxation
        let source_interior = sponge.evaluate(0.0, (5000.0, 5000.0), &swe, &tracers);
        assert!(source_interior.h_t.abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // Combined Source Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_combined_source_sum() {
        let heat = SurfaceHeatFlux::constant(100.0);
        let relax = UniformRelaxationTracerSource::new(TracerState::new(10.0, 35.0), 0.001);

        let combined = CombinedTracerSource2D::empty().add(heat).add(relax);

        assert_eq!(combined.len(), 2);

        let swe = make_swe(5.0);
        let tracers = make_tracers(5.0, 15.0, 30.0);

        let source = combined.evaluate(0.0, (0.0, 0.0), &swe, &tracers);

        // Heat flux contribution: 100 / RHO_CP
        let heat_contrib = 100.0 / RHO_CP;
        // Relaxation contribution: -0.001 * 5 * (15 - 10) = -0.025
        let relax_contrib_t = -0.025;
        let relax_contrib_s = 0.025;

        assert!((source.h_t - (heat_contrib + relax_contrib_t)).abs() < 1e-10);
        assert!((source.h_s - relax_contrib_s).abs() < 1e-10);
    }

    #[test]
    fn test_combined_source_ref() {
        let heat = SurfaceHeatFlux::constant(100.0);

        let combined = CombinedTracerSourceRef::new(vec![&heat]);

        assert_eq!(combined.len(), 1);

        let swe = make_swe(10.0);
        let tracers = make_tracers(10.0, 10.0, 35.0);

        let source = combined.evaluate(0.0, (0.0, 0.0), &swe, &tracers);

        let expected = 100.0 / RHO_CP;
        assert!((source.h_t - expected).abs() < 1e-12);
    }

    #[test]
    fn test_combined_source_empty() {
        let combined = CombinedTracerSource2D::empty();

        assert!(combined.is_empty());

        let swe = make_swe(10.0);
        let tracers = make_tracers(10.0, 10.0, 35.0);

        let source = combined.evaluate(0.0, (0.0, 0.0), &swe, &tracers);

        assert!(source.h_t.abs() < 1e-14);
        assert!(source.h_s.abs() < 1e-14);
    }

    // -------------------------------------------------------------------------
    // Norwegian Rivers Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_glomma_preset() {
        let glomma = norwegian_rivers::glomma(|_, _| 1.0, 100000.0, 10.0);

        let swe = make_swe(5.0);
        let tracers = make_tracers(5.0, 12.0, 35.0);

        let source = glomma.evaluate(0.0, (0.0, 0.0), &swe, &tracers);

        // Glomma is fresh (S=0), so should reduce salinity
        assert!(source.h_s < 0.0);
        // Temperature difference: 10 - 12 = -2, so slight cooling
        assert!(source.h_t < 0.0);
    }

    // -------------------------------------------------------------------------
    // Physical Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_physical_constants() {
        // Verify physical constants are reasonable
        assert!((CP_SEAWATER - 3985.0).abs() < 1.0);
        assert!((RHO_SEAWATER - 1025.0).abs() < 1.0);
        assert!((RHO_CP - 4084625.0).abs() < 1000.0);
    }
}
