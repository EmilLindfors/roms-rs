//! # dg-rs
//!
//! A Discontinuous Galerkin library for solving hyperbolic PDEs.
//!
//! This crate provides the core building blocks for DG methods:
//! - Polynomial basis functions (Legendre)
//! - Quadrature rules (Gauss-Lobatto)
//! - DG operators (mass, differentiation, LIFT)
//! - Mesh representation
//! - Numerical fluxes
//! - Time integration (SSP-RK3)
//! - Conservation law abstractions (advection, shallow water)
//! - Harmonic analysis for tidal time series

pub mod analysis;
pub mod basis;
pub mod boundary;
pub mod equations;
pub mod flux;
pub mod io;
pub mod mesh;
pub mod operators;
pub mod polynomial;
pub mod solver;
pub mod source;
pub mod time;

// Re-export main types for convenience
// 1D types
pub use basis::Vandermonde;
pub use equations::{Advection1D, ConservationLaw, ShallowWater1D, ShallowWater2D};
pub use flux::{
    SWEFluxType2D, compute_flux_swe_2d, hll_flux_swe_2d, lax_friedrichs_flux, roe_flux_swe_2d,
    rusanov_flux_swe_2d, upwind_flux,
};
pub use mesh::Mesh1D;
pub use operators::DGOperators1D;
pub use solver::{
    BoundaryCondition, DGSolution1D, SWEFluxType, SWERhsConfig, SWESolution, SWEState,
    TVBParameter, apply_swe_limiters, compute_dt_swe, compute_rhs, compute_rhs_swe,
    positivity_limiter, tvb_limiter,
};

#[cfg(feature = "parallel")]
pub use solver::compute_rhs_parallel;
pub use time::{
    SWETimeConfig, compute_dt, run_swe_simulation, ssp_rk3_step, ssp_rk3_step_timed,
    ssp_rk3_swe_step, ssp_rk3_swe_step_timed, total_energy, total_mass, total_momentum,
};

// 2D types
pub use basis::Vandermonde2D;
pub use boundary::{
    BCContext2D, BathymetryValidationConfig, BathymetryValidationResult, Chapman2D,
    ChapmanFlather2D, ConstantDischarge2D, Discharge2D, Extrapolation2D, FixedState2D, Flather2D,
    HarmonicFlather2D, HarmonicTidal2D, NestingBC2D, Radiation2D, Reflective2D,
    SWEBoundaryCondition2D, SpongeConfig, TSTConfig, TSTConstituent, TSTOBC2D, Tidal2D,
    TidalBCType, TidalConstituent, TidalSimulationBuilder, format_bathymetry_warning,
    validate_bathymetry_convention,
};
#[cfg(feature = "netcdf")]
pub use boundary::OceanNestingBC2D;
pub use equations::Advection2D;
pub use mesh::{BoundaryTag, Mesh2D};
pub use operators::{DGOperators2D, GeometricFactors2D};
pub use solver::{
    AdvectionBoundaryCondition2D,
    AdvectionFluxType,
    ConservativeTracerState,
    ConstantBC2D,
    DGSolution2D,
    DirichletBC2D,
    ExtrapolationTracerBC,
    FixedTracerBC,
    PeriodicBC2D,
    SWE2DRhsConfig,
    SWESolution2D,
    SWEState2D,
    SystemSolution2D,
    Tracer2DRhsConfig,
    TracerBCContext2D,
    TracerBoundaryCondition2D,
    TracerSolution2D,
    TracerSourceTerm2D,
    TracerState,
    UpwindTracerBC,
    compute_dt_advection_2d,
    compute_dt_swe_2d,
    // Tracer transport
    compute_dt_tracer_2d,
    compute_rhs_advection_2d,
    compute_rhs_swe_2d,
    compute_rhs_tracer_2d,
    // SWE 2D limiters
    apply_swe_limiters_2d,
    apply_swe_limiters_kuzmin_2d,
    swe_positivity_limiter_2d,
    swe_tvb_limiter_2d,
    swe_kuzmin_limiter_2d,
    // Wetting/drying
    WetDryConfig,
    apply_wet_dry_correction,
    apply_wet_dry_correction_all,
    // Diagnostics
    DiagnosticsTracker,
    ProgressReporter,
    SWEDiagnostics2D,
    current_cfl_2d,
    total_energy_2d,
    total_mass_2d,
    total_momentum_2d,
};
#[cfg(feature = "parallel")]
pub use solver::{compute_rhs_swe_2d_parallel, compute_rhs_tracer_2d_parallel};
#[cfg(feature = "simd")]
pub use solver::compute_rhs_swe_2d_simd;
pub use source::{
    AtmosphericPressure2D, CombinedSource2D, CoriolisSource2D, DragCoefficient,
    HydrostaticReconstruction2D, P_STANDARD, RectangularBoundary, SourceContext2D, SourceTerm2D,
    SpongeLayer2D, SpongeProfile, TidalPotential, TidalPotentialConstituent, WindStress2D,
};
pub use time::{
    CoupledRhs2D,
    CoupledState2D,
    CoupledTimeConfig,
    // Coupled SWE + tracer integration
    compute_coupled_rhs,
    compute_dt_coupled,
    run_advection_2d,
    run_coupled_simulation,
    ssp_rk3_coupled_step,
    ssp_rk3_coupled_step_timed,
    ssp_rk3_step_2d,
    ssp_rk3_step_2d_timed,
};

// Analysis types
pub use analysis::{
    ComparisonMetrics, ConstituentComparison, ConstituentComparisonSummary, ConstituentResult,
    HarmonicAnalysis, HarmonicResult, TimeSeries, TimeSeriesPoint, compare_harmonics,
    // Stability monitoring
    StabilityMonitor, StabilityStatus, StabilityThresholds, StabilityWarning,
    // Tide gauge validation
    ModelExtractor, PrecomputedExtractor, StationValidationResult, TideGaugeStation,
    ValidationSummary, norwegian_stations, validate_stations,
};

// I/O types
pub use io::{
    BathymetryStatistics, BoundaryTimeSeries, CoastlineData, CoastlineError,
    CoastlineStatistics, ConstituentData, ConstituentEntry, ConstituentFileError,
    CoordinateProjection, FROYA_BBOX, GeoBoundingBox, GeoTiffBathymetry, GeoTiffError,
    LocalProjection, NORWAY_BBOX, TimeSeriesFileError, TimeSeriesRecord, UtmProjection,
    VtkError, constituent_period, parse_constituents, parse_timeseries, read_constituent_file,
    read_timeseries_file, write_vtk_coupled, write_vtk_series, write_vtk_swe,
    // Tide gauge I/O
    TideGaugeFile, TideGaugeFileError, files_to_observation_map, read_tide_gauge_directory,
    read_tide_gauge_file, write_tide_gauge_file,
};
#[cfg(feature = "netcdf")]
pub use io::{
    ForcingDataPoint, ForcingReader, NetCDFError, NetCDFMeshInfo, NetCDFWriter,
    NetCDFWriterConfig, OceanGridType, OceanModelReader, OceanState,
    FILL_VALUE_F32, FILL_VALUE_F64, is_valid_f32, is_valid_f64,
};

// Mesh types (additional exports)
pub use mesh::{LandMask2D, LandMaskStatistics};
