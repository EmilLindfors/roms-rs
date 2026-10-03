//! Frøya–Smøla–Hitra: tides on real bathymetry.
//!
//! The 2D production path (`Simulation` + `SSPRK3` + `SWEPhysics2D`) with the
//! wet/dry defaults: the `WetDry` split form, HLL, positivity limiting, velocity
//! desingularization, point-implicit Manning friction, and the positivity CFL.
//!
//! 1. **Domain.** A bed raster (`BedRaster`): Kartverket's topobathy model
//!    (`dem=`, land heights and depths in one 50 m grid) when present; else
//!    the GeoTIFF bathymetry (0 at its dry pixels) with a land mask from
//!    GSHHS, four times finer, where the bed is `land_elevation`. The raster
//!    is L2-projected onto the nodes of a rectangular grid
//!    (`Bathymetry2D::project`; `bed=point` samples it at the nodes), and only elements
//!    with a node below mean sea level are kept (`Mesh2D::retain_elements`);
//!    their faces to dropped elements are coastline walls. Land nodes inside
//!    kept elements are dry shore for `WetDry`. The sides of the rectangle
//!    are open where they cross water.
//! 2. **Lake at rest.** Walls everywhere, no forcing: the largest spurious
//!    current and surface error after `rest_hours` (exact balance keeps both
//!    at round-off; the collocated scheme reached m/s on steep beds).
//! 3. **Tides.** A characteristic open boundary (`CharacteristicOBC`) forced
//!    by NorKyst-800 boundary tides (`BoundaryTides` from the tidal atlas
//!    `tides=<file>`, made by `examples/norkyst_boundary_tides.rs`: η, ū, v̄ of
//!    10 constituents plus P1 and K2, varying along the boundary), on a
//!    `ModelClock` starting at `start`. `tide_transport=3` scales the atlas
//!    velocity to carry NorKyst's transport over the child's bed, by up to a
//!    factor 3 (`BoundaryTides::with_transport_scaling`; off by default). Coriolis, Manning friction, optionally
//!    wind and an atmospheric pressure gradient (with the inverse-barometer
//!    level at the open boundaries): uniform (`wind`) or from a weather model
//!    (`met=<file,…>`: MET Nordic, MEPS, AROME-Arctic or ERA5 NetCDF, e.g.
//!    from `met_forcing_subset`; `GriddedAtmosphere2D`, ramped like the tide).
//! 4. **Nesting** (`norkyst=<file>`, e.g. from `norkyst_nesting_subset`):
//!    instead of the atlas tides, NorKyst-800's hourly ζ, ū and v̄ at the open
//!    boundary (`OceanModelState`: tides plus the coastal current and the
//!    wind-driven flow), with the velocity scaled to conserve the parent's
//!    transport, a relaxation band `band_km` wide with timescale
//!    `band_minutes` at the boundary, and the bed blended to NorKyst's across
//!    the band (`blend=0` to keep it), ramped up from rest over `ramp_hours`.
//!    Open faces NorKyst does not cover become walls. NorKyst's ζ is shifted
//!    by minus the mean level `Z0` of the boundary atlas (its datum sits
//!    0.28 m below mean sea level here), or by `nest_level=`. NorKyst's
//!    `gauge_gains` and `gauge_ratios` constituents are replaced by the
//!    gauge-corrected ones (below) by adding the difference between the
//!    corrected and the raw atlas, which is NorKyst's own harmonic fit, so
//!    the residual flow is kept (`OceanModelState::with_tidal_correction`;
//!    `nest_tides=raw` to keep NorKyst's tides). The parent ζ is taken as is; `ib=1` adds the
//!    inverse-barometer level of `met=` to it (for a parent run without
//!    pressure forcing).
//!    Without an atlas: M2 of `M2_AMPLITUDE` in one phase along the boundary.
//!    NorKyst-800 has too little N2 here (0.027 m at Mausund against 0.156 m
//!    observed) and about twice the Q1, and its diurnals are off (K1 1.14×,
//!    +8.6°; O1 0.95×, −17.3° at Mausund). The atlas is corrected with the
//!    first gauge's whole-record fit (`TidalAtlas::infer`): `gauge_gains=K1,O1`
//!    (the default) scales each by the gauge's constant over NorKyst's at the
//!    gauge (`station_atlas=`), keeping NorKyst's spatial structure; then
//!    `gauge_ratios=N2,Q1,P1` (the default) re-infers N2, Q1 and P1 from M2,
//!    the corrected O1 and the corrected K1 with the gauge's ratios.
//! 5. **Validation.** Every `station_minutes` the surface and the
//!    depth-averaged current (east/north) are sampled at the tide gauges
//!    `gauges=` (files as written by `scripts/kartverket_gauge.sh`) and at
//!    the current records `currents=` (ADCP file format, depth-averaged). The
//!    DG polynomial is evaluated at the station itself (`PointLocator2D`,
//!    `Probe2D`) if its element is submerged (every node at least
//!    `STATION_MIN_DEPTH` deep), else at the nearest point of a submerged
//!    element: a coarse mesh leaves perched pockets in shoreline elements
//!    that hold water above the tide. After the run
//!    each station's series are written to the output directory
//!    (`station_*.txt`, `currents_*.txt`). Currents are fitted for tidal
//!    ellipses and compared with NorKyst-800's (from the velocity constants
//!    of `station_atlas=`; each gauge gets a second station at its nearest
//!    atlas point, where NorKyst's currents apply) and with the record's, by
//!    constituent and by the complex difference |ΔW|, plus time-series
//!    skill against the record. For the surface, the part after
//!    `spinup_hours` is fitted for reference constants and compared with the
//!    gauge (its whole record fitted, which also supplies the ratios of the
//!    constituents the run is too short to resolve) and with NorKyst-800 at
//!    the gauge (`station_atlas=`, from `norkyst_boundary_tides points=`):
//!    amplitude ratio, phase difference and complex difference per
//!    constituent, and RMSE against the observations and against the gauge's
//!    tidal prediction.
//!
//! Without the data files it runs a synthetic basin with an island and a
//! beach.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example froya_real_data -- [nx=120] [ny=90] [order=2] \
//!     [hours=12.42] [rest_hours=1] [ramp_hours=1] [output_minutes=60] \
//!     [snapshot_minutes=0] [wind] \
//!     [start=2025-06-15T00:00:00Z] [tides=data/froya_boundary_tides.txt] [norkyst=<file>] \
//!     [gauges=data/tide_gauges/mausund_obs.txt] [currents=<file,…>] \
//!     [station_atlas=data/froya_station_tides.txt] \
//!     [station_minutes=10] [spinup_hours=24] [gauge_gains=K1,O1] \
//!     [gauge_ratios=N2,Q1,P1] [land_elevation=5] \
//!     [bed=projected|point] [dem=data/froya_topobathy.tif|none] [rx0=] [rx0_min_depth=3] \
//!     [bbox=8.0,63.6,9.2,64.0] [lts=0] [rk=43|3] [cfl=] [output=output/froya] \
//!     [met=<file,…>] [band_km=3] [band_minutes=30] [blend=1] [ib=0] [nest_level=] \
//!     [nest_tides=corrected|raw] [waves=0] [wave_grid=25,36] [turning=]
//! ```
//!
//! `waves=N` (N > 0) times N steps of the spectral wave model on the domain
//! instead of the tidal run (`wave_grid=frequencies,directions`, refraction
//! capped at `turning=` rad/s): the step, what sets it, and the cost per model
//! hour.
//!
//! `snapshot_minutes=N` (N > 0) also writes the state every N minutes to
//! `<output>/froya.dgsnap` (`io::SnapshotWriter`: f32 η, u, v per node, with
//! the mesh, bed, clock and stations), a tenth of the VTU frames' size. The
//! viewer replays it without rebuilding the domain:
//! `cd viz && cargo run --release -- --replay ../output/froya/froya.dgsnap`.
//! With stations, N is rounded to a multiple of `station_minutes`.
//!
//! `rx0=r` smooths the bed, keeping its volume, until the slope factor
//! r_x0 = |h₁ − h₂|/(h₁ + h₂) between neighbouring nodes at least
//! `rx0_min_depth` (3 m) deep is at most r (`Bathymetry2D::smooth_rx0`):
//! shoals a node wide otherwise carry spurious m/s currents.
//!
//! `bbox=west,south,east,north` runs a smaller box, e.g. the 22 × 19 km
//! around Mausund (a quarter of the cost), with its own mesh and boundary
//! atlas (use `tide_transport=3` there): see "Mausund sub-domain" in
//! `docs/gmsh-meshes.md`.
//!
//! `lts=N` (N > 0) steps the tidal run with local time stepping
//! (`Multirate`, up to N levels of halved time steps): every element at
//! the largest power-of-two fraction of the step its own CFL allows.
//!
//! `rk=43` (the default) steps with SSP-RK(4,3), `rk=3` with SSP-RK3. The
//! positivity bound of the wet/dry scheme sets the step where elements may
//! run dry, and SSP-RK(4,3)'s SSP coefficient of 2 doubles it for 4/3 of the
//! RHS work per step. The bound is relaxed per element by how its water is
//! distributed, up to `cfl=`, by default 0.9 of the integrator's linear
//! stability limit (`linear_cfl_swe_2d`: 1.36 at P2 with SSP-RK(4,3)), and
//! in shoreline elements up to 0.9 of the subcells' lower limit
//! (`linear_cfl_subcells_swe_2d`: 0.99 at P2).
//!
//! Harmonic validation needs the record after spin-up to resolve the main
//! constituents: 15 days separate M2/S2 and K1/O1 (`hours=384` with the
//! default spin-up); N2 needs 28.
//!
//! ## Data files in ./data/
//!
//! - froya_topobathy.tif (Kartverket's topobathy model: land heights and
//!   depths at 50 m, from `scripts/kartverket_topobathy.sh`; used when present)
//! - froya_smola_hitra.tif (bathymetry) and GSHHS_f_L1.shp (coastline):
//!   without the topobathy model, or with `dem=none`
//! - froya_boundary_tides.txt (tidal atlas, optional)
//! - tide_gauges/mausund_obs.txt, froya_station_tides.txt (validation, optional)

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use dg_rs::analysis::{
    ADCPStation, ADCPValidationResult, ConstituentComparison, CurrentTimeSeries, EllipseFit,
    Inference, ReferenceConstant, ReferenceFit, StationValidationResult, TidalEllipse,
    TideGaugeStation, TimeSeries, fit_reference_constants, fit_tidal_ellipses,
    resolvable_constituents,
};
#[cfg(feature = "netcdf")]
use dg_rs::boundary::NestingOptions;
use dg_rs::boundary::{
    AtlasPoint, BCContext2D, BoundaryLevel, BoundaryTides, CharacteristicOBC,
    ExternalStateProvider, HarmonicTide, InverseBarometer, MultiBoundaryCondition2D,
    OceanModelState, Reflective2D, SWEBoundaryCondition2D, TidalAtlas,
};
use dg_rs::equations::ShallowWater2D;
#[cfg(feature = "netcdf")]
use dg_rs::io::{
    AtmosphereReader, NetCDFMeshInfo, NetCDFWriter, NetCDFWriterConfig, OceanModelReader,
};
use dg_rs::io::{
    BedRaster, CoastlineData, CoordinateProjection, GeoBoundingBox, GeoTiffBathymetry,
    LocalProjection, SnapshotWriter, TideGaugeFile, east_axis, read_adcp_file,
    read_tide_gauge_file, write_adcp_file, write_tide_gauge_file, write_vtk_swe,
};
use dg_rs::mesh::{
    Bathymetry2D, BoundaryTag, Mesh2D, MeshPoint, PointLocator2D, inverse_bilinear, read_gmsh_mesh,
};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::{PhysicsBuilder, PhysicsModule, SWEPhysics2D, SWEPhysics2DBuilder};
use dg_rs::simulation::Simulation;
use dg_rs::solver::{
    LINEAR_CFL_SAFETY, Probe2D, SWESolution2D, SWEState2D, StandardLimiter2D, WetDryConfig,
    linear_cfl_swe_2d, positivity_cfl_swe_2d,
};
use dg_rs::source::{
    AtmosphericPressure2D, CoriolisSource2D, DragCoefficient, GriddedAtmosphere2D,
    ManningFriction2D, WindStress2D,
};
use dg_rs::tides::canonical_name;
use dg_rs::time::{IntegratorInfo, ModelClock, Multirate, SSPRK3, SspScheme, StandardIntegrator};
#[cfg(feature = "netcdf")]
use dg_rs::types::Depth;
use dg_rs::types::ElementIndex;

/// Gravitational acceleration (m/s²)
const G: f64 = 9.81;
/// Coriolis parameter at 63.8°N (s⁻¹)
const F_CORIOLIS: f64 = 1.31e-4;
/// Manning roughness (s/m^{1/3})
const MANNING_N: f64 = 0.025;
/// Default bed elevation given to land nodes of shoreline elements (m above
/// MSL; `land_elevation=`)
const LAND_ELEVATION: f64 = 5.0;
/// Default elevation model: Kartverket's topobathy model at 50 m
/// (`scripts/kartverket_topobathy.sh 8.0 63.6 9.2 64.0 50 data/froya_topobathy.tif`)
const DEM: &str = "data/froya_topobathy.tif";
/// Frøya–Smøla–Hitra: west, south, east, north (°)
const FROYA_BBOX: [f64; 4] = [8.0, 63.6, 9.2, 64.0];
/// Default depth (m) below which `rx0=` leaves the bed alone: the shore and
/// the dry area keep their shape
const RX0_MIN_DEPTH: f64 = 3.0;
/// Land-mask cells per bathymetry pixel and direction: the coastline is
/// rasterised at ≈ 25 × 58 m
const LAND_MASK_REFINEMENT: usize = 4;

/// M2 period (s) and a typical amplitude on this coast (m)
const M2_PERIOD: f64 = 12.420_601 * 3600.0;
const M2_AMPLITUDE: f64 = 0.8;
/// Default tidal ramp-up (h)
const TIDAL_RAMP_HOURS: f64 = 1.0;
/// Largest distance (m) from an open-boundary node to a tidal-atlas point
const ATLAS_COVERAGE: f64 = 5000.0;

/// Wind (m/s, from °) and atmospheric pressure gradient (Pa/m, from °)
const WIND_SPEED: f64 = 8.0;
const WIND_DIRECTION: f64 = 225.0;
const PRESSURE_GRADIENT: f64 = 1.5e-3;
const PRESSURE_DIRECTION: f64 = 225.0;

/// A station samples the model at its position if every node of its element
/// is at least this deep (m below MSL), so that it is in open water that
/// stays wet through the tide; else at the nearest point of such an element
const STATION_MIN_DEPTH: f64 = 3.0;
/// ... no farther than this from the station (m)
const STATION_MAX_OFFSET: f64 = 3000.0;
/// Constituents fitted at the stations, in priority order (a short record
/// keeps the first ones): the forcing's, without the long-period ones
const STATION_CONSTITUENTS: [&str; 12] = [
    "M2", "S2", "K1", "O1", "N2", "Q1", "K2", "P1", "M4", "MS4", "MN4", "M6",
];
/// Constituents fitted to a gauge's whole record: long-period ones too, so
/// that the seasonal and fortnightly signal does not leak into the tides
const GAUGE_CONSTITUENTS: [&str; 15] = [
    "M2", "S2", "K1", "O1", "N2", "Q1", "K2", "P1", "M4", "MS4", "MN4", "M6", "Mf", "Mm", "Ssa",
];

/// Command-line options (`key=value`, or a bare flag)
struct Options {
    nx: usize,
    ny: usize,
    order: usize,
    hours: f64,
    rest_hours: f64,
    ramp_hours: f64,
    output_minutes: f64,
    /// Minutes between frames of the snapshot file (`snapshot_minutes=`; 0: none)
    snapshot_minutes: f64,
    wind: bool,
    start: String,
    tides: String,
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    norkyst: Option<String>,
    /// Weather-model files (`met=a.nc,b.nc`), joined in time
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    met: Vec<String>,
    /// Nesting: relaxation band width (km) and timescale at the boundary
    /// (min), whether to blend the bed to the parent's across the band, and
    /// whether to add the inverse-barometer level to the parent's ζ
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    band_km: f64,
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    band_minutes: f64,
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    blend_bed: bool,
    nesting_ib: bool,
    /// Nesting: replace NorKyst's `gauge_gains` and `gauge_ratios`
    /// constituents by the gauge-corrected ones (`nest_tides=corrected`, the
    /// default) or keep its tides (`nest_tides=raw`)
    correct_nested_tides: bool,
    /// Added to NorKyst's ζ (m); default: minus the mean level `Z0` of the
    /// boundary atlas
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    nesting_level: Option<f64>,
    gauges: Vec<String>,
    /// Current records (`currents=a.txt,b.txt`, ADCP file format:
    /// depth-averaged east/north velocity)
    currents: Vec<String>,
    station_atlas: String,
    station_minutes: f64,
    spinup_hours: f64,
    /// Atlas constituents scaled by the gauge's constants over NorKyst's at
    /// the gauge (`gauge_gains=K1,O1`), before the ratio inference
    gauge_gains: Vec<&'static str>,
    /// Atlas constituents re-inferred from their neighbour with the gauge's
    /// ratio (`gauge_ratios=N2,Q1,P1`)
    gauge_ratios: Vec<&'static str>,
    land_elevation: f64,
    /// Elevation model with land heights (`dem=`, used if the file exists;
    /// `dem=none` for the GeoTIFF bathymetry and GSHHS coastline)
    dem: Option<String>,
    /// L2-project the bed onto the nodes (`bed=projected`, the default) or
    /// sample it there (`bed=point`). Point samples of the 50 m raster at
    /// nodes 500 m apart land on skerries at random and close sounds: at
    /// 1 km the M2 current near Mausund was 0.41 of NorKyst's with point
    /// samples and 0.63 projected, the gauge's centred RMSE 4.4 and 4.0 cm
    project_bed: bool,
    /// Coastline-fitted Gmsh mesh (`mesh=`, from
    /// `scripts/gmsh_coastline_mesh.py`) instead of the `nx` × `ny` grid
    mesh: Option<String>,
    /// Local time stepping with up to this many levels (`lts=`; 0: global
    /// steps)
    lts: usize,
    /// SSP-RK(4,3) (`rk=43`, default) or SSP-RK3 (`rk=3`), globally or as
    /// the base of local time stepping
    integrator: StandardIntegrator,
    /// CFL number of the tidal run (`cfl=`): its linear stability bound. The
    /// default is `LINEAR_CFL_SAFETY` (0.9) times the integrator's linear
    /// limit (`linear_cfl_swe_2d`); the wet/dry positivity bound caps it
    /// where elements may run dry
    cfl: f64,
    /// Smooth the bed to a slope factor r_x0 ≤ this between neighbouring
    /// nodes deeper than `rx0_min_depth` (`rx0=`, e.g. 0.3; off by default)
    rx0: Option<f64>,
    rx0_min_depth: f64,
    /// Domain box `west,south,east,north` (°; `bbox=`). A smaller box needs
    /// its own boundary atlas (`tides=`, from `norkyst_boundary_tides bbox=`)
    bbox: [f64; 4],
    /// Scale the atlas velocity by NorKyst's total depth over the child's,
    /// by up to this factor (`tide_transport=`; 0: off)
    tide_transport: f64,
    profile: usize,
    /// Time this many steps of the spectral wave model on the domain instead
    /// of the tidal run (`waves=N`; 0: off), on `wave_grid=frequencies,directions`
    /// with refraction capped at `turning=` rad/s (none by default)
    waves: usize,
    wave_grid: [usize; 2],
    turning: Option<f64>,
    output: Option<PathBuf>,
}

impl Options {
    fn parse() -> Result<Self, String> {
        let args: HashMap<String, String> = std::env::args()
            .skip(1)
            .map(|a| match a.split_once('=') {
                Some((k, v)) => (k.to_string(), v.to_string()),
                None => (a, String::new()),
            })
            .collect();
        let get = |key: &str, default: f64| -> Result<f64, String> {
            args.get(key).map_or(Ok(default), |v| {
                v.parse().map_err(|_| format!("bad {key}={v}"))
            })
        };
        let constituents = |key: &str, default: &str| -> Result<Vec<&'static str>, String> {
            args.get(key)
                .map_or(default, String::as_str)
                .split(',')
                .filter(|n| !n.is_empty())
                .map(|n| canonical_name(n).ok_or(format!("{key}: unknown constituent {n}")))
                .collect()
        };
        let mut opts = Self {
            nx: get("nx", 120.0)? as usize,
            ny: get("ny", 90.0)? as usize,
            order: get("order", 2.0)? as usize,
            hours: get("hours", M2_PERIOD / 3600.0)?,
            rest_hours: get("rest_hours", 1.0)?,
            ramp_hours: get("ramp_hours", TIDAL_RAMP_HOURS)?,
            output_minutes: get("output_minutes", 60.0)?,
            snapshot_minutes: get("snapshot_minutes", 0.0)?,
            land_elevation: get("land_elevation", LAND_ELEVATION)?,
            dem: match args.get("dem").map_or(DEM, String::as_str) {
                "none" => None,
                path => Some(path.to_string()),
            },
            project_bed: match args.get("bed").map_or("projected", String::as_str) {
                "projected" => true,
                "point" => false,
                other => return Err(format!("bad bed={other}: projected or point")),
            },
            profile: get("profile", 0.0)? as usize,
            waves: get("waves", 0.0)? as usize,
            wave_grid: {
                let text = args.get("wave_grid").map_or("25,36", String::as_str);
                let parts: Vec<usize> = text
                    .split(',')
                    .map(|v| v.parse().map_err(|_| format!("bad wave_grid={text}")))
                    .collect::<Result<_, _>>()?;
                match parts[..] {
                    [nf, nd] => [nf, nd],
                    _ => return Err(format!("wave_grid=frequencies,directions, not {text}")),
                }
            },
            turning: args
                .get("turning")
                .map(|v| v.parse().map_err(|_| format!("bad turning={v}")))
                .transpose()?,
            lts: get("lts", 0.0)? as usize,
            cfl: get("cfl", f64::NAN)?,
            integrator: match args.get("rk").map_or("43", String::as_str) {
                "43" => StandardIntegrator::SSPRK43,
                "3" => StandardIntegrator::SSPRK3,
                other => return Err(format!("bad rk={other}: 43 or 3")),
            },
            rx0: args
                .get("rx0")
                .map(|v| v.parse().map_err(|_| format!("bad rx0={v}")))
                .transpose()?,
            rx0_min_depth: get("rx0_min_depth", RX0_MIN_DEPTH)?,
            bbox: match args.get("bbox") {
                None => FROYA_BBOX,
                Some(v) => v
                    .split(',')
                    .map(|c| c.trim().parse::<f64>())
                    .collect::<Result<Vec<_>, _>>()
                    .ok()
                    .and_then(|c| <[f64; 4]>::try_from(c).ok())
                    .filter(|[w, s, e, n]| w < e && s < n)
                    .ok_or(format!("bad bbox={v}: west,south,east,north"))?,
            },
            tide_transport: get("tide_transport", 0.0)?,
            mesh: args.get("mesh").cloned(),
            output: args.get("output").map(PathBuf::from),
            wind: args.contains_key("wind"),
            start: args
                .get("start")
                .cloned()
                .unwrap_or("2025-06-15T00:00:00Z".into()),
            tides: args
                .get("tides")
                .cloned()
                .unwrap_or("data/froya_boundary_tides.txt".into()),
            norkyst: args.get("norkyst").cloned(),
            met: args
                .get("met")
                .map_or("", String::as_str)
                .split(',')
                .filter(|f| !f.is_empty())
                .map(String::from)
                .collect(),
            band_km: get("band_km", 3.0)?,
            band_minutes: get("band_minutes", 30.0)?,
            blend_bed: get("blend", 1.0)? != 0.0,
            nesting_ib: get("ib", 0.0)? != 0.0,
            correct_nested_tides: match args.get("nest_tides").map_or("corrected", String::as_str) {
                "corrected" => true,
                "raw" => false,
                other => return Err(format!("bad nest_tides={other}: corrected or raw")),
            },
            nesting_level: args
                .get("nest_level")
                .map(|v| v.parse().map_err(|_| format!("bad nest_level={v}")))
                .transpose()?,
            gauges: args
                .get("gauges")
                .map_or("data/tide_gauges/mausund_obs.txt", String::as_str)
                .split(',')
                .filter(|g| !g.is_empty())
                .map(String::from)
                .collect(),
            currents: args
                .get("currents")
                .map_or("", String::as_str)
                .split(',')
                .filter(|c| !c.is_empty())
                .map(String::from)
                .collect(),
            station_atlas: args
                .get("station_atlas")
                .cloned()
                .unwrap_or("data/froya_station_tides.txt".into()),
            station_minutes: get("station_minutes", 10.0)?,
            spinup_hours: get("spinup_hours", 24.0)?,
            gauge_gains: constituents("gauge_gains", "K1,O1")?,
            gauge_ratios: constituents("gauge_ratios", "N2,Q1,P1")?,
        };
        if opts.cfl.is_nan() {
            let scheme = opts.integrator.ssp_scheme().expect("an SSP integrator");
            opts.cfl = linear_cfl_swe_2d(opts.order, scheme)
                .map_or(1.0, |linear| LINEAR_CFL_SAFETY * linear);
        }
        Ok(opts)
    }
}

/// Surface range (and where the highest wet node is: x, y, h), largest speed
/// where h > 10 cm (and where: x, y, h), the same in open water (still-water
/// depth at least `STATION_MIN_DEPTH`, away from the foreshore films), and
/// the number of wet nodes (h > 1 mm).
struct Stats {
    eta: (f64, f64),
    highest: (f64, f64, f64),
    speed: f64,
    fastest: (f64, f64, f64),
    open_speed: f64,
    open_fastest: (f64, f64, f64),
    wet: usize,
}

/// A tide gauge's record (finite samples, Unix times) and its reference
/// constants.
struct Gauge {
    station: TideGaugeStation,
    observed: TimeSeries,
    observed_fit: Result<ReferenceFit, String>,
}

/// A current record (Unix times, east/north) and its tidal ellipses.
struct CurrentRecord {
    station: ADCPStation,
    observed: CurrentTimeSeries,
    observed_fit: Result<EllipseFit, String>,
}

/// A station, the point that samples the model there, what it is compared
/// with, and the sampled series.
struct Station {
    name: String,
    longitude: f64,
    latitude: f64,
    /// Tide gauge record (η) and current record, if any
    gauge: Option<Gauge>,
    current: Option<CurrentRecord>,
    /// The DG solution evaluated at the sampling point
    probe: Probe2D,
    /// Distance from the station to the sampling point (m; 0 at the station)
    offset: f64,
    /// Still-water depth at the sampling point (m)
    depth: f64,
    /// Local east in the mesh axes, `(cos θ, sin θ)`
    east: (f64, f64),
    /// Unix times, surface elevation, depth-averaged east and north velocity
    times: Vec<f64>,
    eta: Vec<f64>,
    u_east: Vec<f64>,
    v_north: Vec<f64>,
}

impl Station {
    /// Record the state of `q` at Unix time `t`.
    fn sample(&mut self, q: &SWESolution2D, bathymetry: &Bathymetry2D, t: f64) {
        let p = self
            .probe
            .sample_swe(q, Some(bathymetry), WetDryConfig::DEFAULT_H_DRY);
        let (c, s) = self.east;
        self.times.push(t);
        self.eta.push(p.eta);
        self.u_east.push(p.u * c + p.v * s);
        self.v_north.push(-p.u * s + p.v * c);
    }
}

/// Water-only mesh with nodal bathymetry.
struct Domain {
    name: &'static str,
    mesh: Arc<Mesh2D>,
    ops: Arc<DGOperators2D>,
    geom: Arc<GeometricFactors2D>,
    bathymetry: Arc<Bathymetry2D>,
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    projection: Option<LocalProjection>,
}

impl Domain {
    /// Grid `nx` × `ny` over `[x0, x1] × [y0, y1]` with open sides, bed
    /// elevation `bed(x, y)` (positive on land) sampled at the nodes (or
    /// projected onto them at `resolution` with `bed=projected`); keep the elements
    /// with a node below mean sea level.
    fn build(
        name: &'static str,
        grid: Mesh2D,
        opts: &Options,
        bed: impl Fn(f64, f64) -> f64,
        resolution: f64,
        projection: Option<LocalProjection>,
    ) -> Self {
        let ops = DGOperators2D::new(opts.order);
        let grid_geom = GeometricFactors2D::compute(&grid, &ops);
        let mut grid_bed = if opts.project_bed {
            Bathymetry2D::project(&grid, &ops, &grid_geom, bed, resolution)
        } else {
            Bathymetry2D::from_function(&grid, &ops, &grid_geom, bed)
        };
        // Water one node wide is unresolved: make it shore
        let raised = grid_bed.raise_isolated_wet_nodes(&grid, &ops, &grid_geom, 0.0);
        println!("  {raised} isolated wet nodes raised to the lowest of their neighbours");
        // Shoals and pits a node wide: the depth-averaged velocity spikes over
        // them (a 6 m shoal among 15–45 m nodes carried 2.2–2.8 m/s)
        let min_depth = opts.rx0_min_depth;
        let describe = |bed: &Bathymetry2D| match bed.max_rx0(&grid, &ops, &grid_geom, min_depth) {
            Some(rx0) => {
                let at = |node: usize| {
                    let (k, i) = (ElementIndex::new(node / ops.n_nodes), node % ops.n_nodes);
                    let [x, y] = grid.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                    (x / 1000.0, y / 1000.0, -bed.data[node])
                };
                let (s, d) = (at(rx0.nodes[0]), at(rx0.nodes[1]));
                format!(
                    "{:.2} between ({:.1}, {:.1}) km, {:.1} m and ({:.1}, {:.1}) km, {:.1} m",
                    rx0.value, s.0, s.1, s.2, d.0, d.1, d.2
                )
            }
            None => "none".into(),
        };
        println!(
            "  Largest slope factor r_x0 (nodes ≥ {min_depth} m deep): {}",
            describe(&grid_bed)
        );
        if let Some(r_max) = opts.rx0 {
            let report = grid_bed.smooth_rx0(&grid, &ops, &grid_geom, r_max, min_depth);
            println!(
                "  Smoothed to r_x0 ≤ {r_max} ({} sweeps): {} nodes changed, by up to {:.1} m; now {}",
                report.sweeps,
                report.changed,
                report.max_change,
                describe(&grid_bed)
            );
        }
        let has_water = |k: ElementIndex| grid_bed.element(k).iter().any(|&b| b < 0.0);
        let (mesh, kept) = grid.retain_elements(has_water, BoundaryTag::Wall);
        let bathymetry = grid_bed.select_elements(&kept);
        let geom = GeometricFactors2D::compute(&mesh, &ops);

        Self {
            name,
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
            bathymetry: Arc::new(bathymetry),
            projection,
        }
    }

    /// Frøya–Smøla–Hitra from the data files, or `None` if they are missing.
    fn froya(opts: &Options) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let [west, south, east, north] = opts.bbox;
        let bbox = GeoBoundingBox::new(west, south, east, north);
        let (lat0, lon0) = bbox.center();
        let projection = LocalProjection::new(lat0, lon0);
        let dem_path = opts.dem.as_deref().map(Path::new).filter(|p| p.exists());
        let raster = if let Some(dem_path) = dem_path {
            // Kartverket's topobathy model: land heights and depths in one grid
            let dem = GeoTiffBathymetry::load(dem_path)?;
            let mut raster = BedRaster::elevation_model(&dem, &bbox)?;
            let (width, height) = raster.dimensions();
            // The 1 m level of the service fills unsurveyed sea with a flat 0
            // (`scripts/kartverket_topobathy.sh` refuses the cells that get it)
            let flat = (0..height)
                .flat_map(|row| (0..width).map(move |col| (row, col)))
                .filter(|&(row, col)| raster.pixel(row, col) == 0.0)
                .count();
            if flat * 20 > width * height {
                return Err(format!(
                    "{}: {:.0} % of the pixels are exactly 0, the flat sea of the 1 m level; \
                     fetch it at 50 m cells",
                    dem_path.display(),
                    100.0 * flat as f64 / (width * height) as f64
                )
                .into());
            }
            // Missing depths are an exact 0 too (the water surface): fill them
            // from their surroundings. Land above `land_elevation` is never
            // wet: cap it
            let filled = raster.fill_holes(|b| b == 0.0);
            println!(
                "  Elevation model: {} ({filled} zero pixels filled, land capped at {} m)",
                dem_path.display(),
                opts.land_elevation
            );
            raster.clamp_land(opts.land_elevation)
        } else {
            let bathy_path = Path::new("data/froya_smola_hitra.tif");
            let coast_path = Path::new("data/GSHHS_f_L1.shp");
            if !bathy_path.exists() || !coast_path.exists() {
                return Ok(None);
            }
            let geotiff = GeoTiffBathymetry::load(bathy_path)?;
            let coastline = CoastlineData::load(coast_path, &bbox)?;
            println!("  Bathymetry: {}", geotiff.statistics());
            println!(
                "  Coastline: {} polygons",
                coastline.statistics().polygon_count
            );
            BedRaster::from_geotiff(
                &geotiff,
                Some(&coastline),
                opts.land_elevation,
                &bbox,
                LAND_MASK_REFINEMENT,
            )
        };
        let (width, height) = raster.dimensions();
        println!(
            "  Bed raster: {width} × {height} pixels, {:.0} m resolution, {:.1} % water; {} onto the nodes",
            raster.pixel_size(),
            100.0 * raster.water_fraction(),
            if opts.project_bed {
                "projected"
            } else {
                "sampled"
            }
        );

        let grid = match &opts.mesh {
            Some(path) => {
                let mesh = read_gmsh_mesh(Path::new(path))?;
                println!(
                    "  Mesh: {path}, {} quadrilaterals, sizes {:.0}–{:.0} m",
                    mesh.n_elements,
                    mesh.h_min(),
                    mesh.h_max()
                );
                mesh
            }
            None => {
                // The sides are open sea wherever they cross water (land is
                // not meshed)
                let (x0, y0) = projection.geo_to_xy(bbox.min_lat, bbox.min_lon);
                let (x1, y1) = projection.geo_to_xy(bbox.max_lat, bbox.max_lon);
                Mesh2D::uniform_rectangle_with_bc(
                    x0,
                    x1,
                    y0,
                    y1,
                    opts.nx,
                    opts.ny,
                    BoundaryTag::Open,
                )
            }
        };
        Ok(Some(Self::build(
            "froya",
            grid,
            opts,
            raster.sampler(&projection),
            raster.pixel_size(),
            Some(projection),
        )))
    }

    /// 50 × 40 km basin shoaling to the east, with a beach along the east
    /// side and a round island.
    fn synthetic(opts: &Options) -> Self {
        let (lx, ly) = (50_000.0, 40_000.0);
        let bed = |x: f64, y: f64| {
            let island = (x - 0.4 * lx).hypot(y - 0.5 * ly) < 4_000.0;
            if island {
                opts.land_elevation
            } else {
                -100.0 + 105.0 * (x / lx).powi(2)
            }
        };
        let grid = Mesh2D::uniform_rectangle_with_bc(
            0.0,
            lx,
            0.0,
            ly,
            opts.nx,
            opts.ny,
            BoundaryTag::Open,
        );
        Self::build("synthetic", grid, opts, bed, 100.0, None)
    }

    /// Make walls of the open-boundary faces with a node farther than
    /// `ATLAS_COVERAGE` from the tidal atlas: water the atlas's parent model
    /// does not resolve (a narrow fjord arm crossing the domain edge) has no
    /// tide to force.
    fn close_uncovered_open_faces(
        &mut self,
        opts: &Options,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let atlas_path = Path::new(&opts.tides);
        if !atlas_path.exists() || opts.norkyst.is_some() {
            return Ok(());
        }
        let atlas = TidalAtlas::read(atlas_path)?;
        let closed = self.close_open_faces(|lon, lat| {
            atlas
                .nearest(lon, lat)
                .is_some_and(|(_, d)| d <= ATLAS_COVERAGE)
        });
        if closed > 0 {
            println!(
                "  {closed} open-boundary faces lie farther than {:.0} km from the tidal atlas: walls",
                ATLAS_COVERAGE / 1000.0
            );
        }
        Ok(())
    }

    /// Make walls of the open-boundary faces with a node where `covered(lon,
    /// lat)` is false; returns how many.
    fn close_open_faces(&mut self, covered: impl Fn(f64, f64) -> bool) -> usize {
        let Some(projection) = self.projection else {
            return 0;
        };
        let (mesh, ops) = (
            Arc::get_mut(&mut self.mesh).expect("mesh not shared yet"),
            &self.ops,
        );
        let mut closed = 0;
        for e in 0..mesh.edges.len() {
            let edge = &mesh.edges[e];
            if edge.right.is_some() || edge.boundary_tag != Some(BoundaryTag::Open) {
                continue;
            }
            let (k, face) = (ElementIndex::new(edge.left.element), edge.left.face);
            let uncovered = ops.face_nodes[face].iter().any(|&i| {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let (lat, lon) = projection.xy_to_geo(x, y);
                !covered(lon, lat)
            });
            if uncovered {
                mesh.edges[e].boundary_tag = Some(BoundaryTag::Wall);
                closed += 1;
            }
        }
        closed
    }

    /// NorKyst nesting (`norkyst=`): read the parent, close the open faces it
    /// does not cover, sample it at the open boundary and the relaxation
    /// band, and blend the bed to its depth across the band.
    #[cfg(feature = "netcdf")]
    fn nesting(
        &mut self,
        opts: &Options,
    ) -> Result<Option<OceanModelState>, Box<dyn std::error::Error>> {
        let (Some(path), Some(projection)) = (&opts.norkyst, self.projection) else {
            return Ok(None);
        };
        let reader = Arc::new(OceanModelReader::from_file(Path::new(path))?);
        println!("  NorKyst: {}", reader.summary());
        // NorKyst's ζ is not referenced to mean sea level: its 30-day mean
        // along the Frøya boundary is −0.28 m (−0.26 m at Mausund, where the
        // gauge's is 0). Shift it by the atlas's mean level.
        let level = match opts.nesting_level {
            Some(level) => level,
            None => {
                let atlas = Path::new(&opts.tides);
                let means: Vec<f64> = if atlas.exists() {
                    TidalAtlas::read(atlas)?
                        .points
                        .iter()
                        .filter_map(|p| p.mean)
                        .collect()
                } else {
                    Vec::new()
                };
                if means.is_empty() {
                    0.0
                } else {
                    -means.iter().sum::<f64>() / means.len() as f64
                }
            }
        };
        println!("  NorKyst ζ shifted by {level:+.3} m to mean sea level");
        let options = NestingOptions::default()
            .with_band(1000.0 * opts.band_km)
            .with_ramp_up(3600.0 * opts.ramp_hours)
            .with_reference_level(level);
        let wet = reader.wet_mask();
        let closed = self.close_open_faces(|lon, lat| {
            reader.stencil(lon, lat).is_some()
                || reader
                    .grid
                    .nearest(lon, lat, options.max_snap, |k| wet[k])
                    .is_some()
        });
        if closed > 0 {
            println!("  {closed} open-boundary faces have no wet NorKyst point nearby: walls");
        }
        let clock = ModelClock::parse(&opts.start)?;
        let parent = OceanModelState::new(
            reader,
            &self.mesh,
            &self.ops,
            &projection,
            BoundaryTag::Open,
            clock,
            &options,
        )?;
        println!(
            "  Nesting: {} open-boundary nodes ({} snapped to a wet NorKyst point), {} nodes in a {} km band",
            parent.n_boundary_nodes(),
            parent.n_snapped(),
            parent.n_band_nodes(),
            opts.band_km
        );
        if let Some((lo, median, hi)) = parent.depth_ratios(&self.bathymetry) {
            println!(
                "  NorKyst/child depth at the open boundary: {lo:.2}–{hi:.2} (median {median:.2})"
            );
        }
        if opts.blend_bed && opts.band_km > 0.0 {
            let bathymetry = Arc::get_mut(&mut self.bathymetry).expect("bed not shared yet");
            let blended = parent.blend_bathymetry(bathymetry, &self.ops, &self.geom);
            println!("  Bed blended to NorKyst's at {blended} nodes of the band");
        }
        Ok(Some(parent))
    }

    #[cfg(not(feature = "netcdf"))]
    fn nesting(
        &mut self,
        _opts: &Options,
    ) -> Result<Option<OceanModelState>, Box<dyn std::error::Error>> {
        Ok(None)
    }

    fn builder<BC: SWEBoundaryCondition2D>(&self, bc: BC) -> SWEPhysics2DBuilder<BC> {
        PhysicsBuilder::swe_2d(
            self.mesh.clone(),
            self.ops.clone(),
            self.geom.clone(),
            ShallowWater2D::new(G),
            bc,
        )
        .with_bathymetry(self.bathymetry.clone())
        .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
        .with_wet_dry(WetDryConfig::default())
        .with_implicit_friction(ManningFriction2D::new(G, MANNING_N))
        .with_source(CoriolisSource2D::f_plane(F_CORIOLIS))
    }

    /// Still water at mean sea level: h = max(0, −B).
    fn at_rest(&self) -> SWESolution2D {
        let mut q = SWESolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
        for k in ElementIndex::iter(self.mesh.n_elements) {
            for i in 0..self.ops.n_nodes {
                let h = (-self.bathymetry.get(k, i)).max(0.0);
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }
        q
    }

    /// Stations for the tide-gauge files `gauges` and the current records
    /// `currents` that lie in the domain (see [`Domain::probe`]), and one at
    /// each gauge's nearest point of the station atlas `atlas`: NorKyst's
    /// currents are compared there, at NorKyst's own position, since they
    /// vary over a few hundred metres among islands.
    fn stations(
        &self,
        gauges: &[String],
        currents: &[String],
        atlas: Option<&TidalAtlas>,
    ) -> Vec<Station> {
        let Some(projection) = &self.projection else {
            return Vec::new();
        };
        let locator = PointLocator2D::new(&self.mesh);
        let days = |t: &[f64]| {
            t.last()
                .zip(t.first())
                .map_or(0.0, |(b, a)| (b - a) / 86_400.0)
        };
        let mut stations = Vec::new();
        let mut add = |name: String,
                       longitude: f64,
                       latitude: f64,
                       gauge: Option<Gauge>,
                       current: Option<CurrentRecord>,
                       record_days: f64| {
            let (x, y) = projection.geo_to_xy(latitude, longitude);
            let Some((probe, offset, depth)) = self.probe(&locator, [x, y]) else {
                println!("  Station {name}: outside the domain (skipped)");
                return;
            };
            println!(
                "  Station {name}: sampled {}, {depth:.1} m deep; record {record_days:.0} days",
                if offset > 0.0 {
                    format!("{offset:.0} m from the station (its element is not submerged)")
                } else {
                    "at the station".to_string()
                }
            );
            stations.push(Station {
                name,
                longitude,
                latitude,
                gauge,
                current,
                probe,
                offset,
                depth,
                east: east_axis(projection, latitude, longitude),
                times: Vec::new(),
                eta: Vec::new(),
                u_east: Vec::new(),
                v_north: Vec::new(),
            });
        };
        for path in gauges {
            let gauge = match read_tide_gauge_file(Path::new(path)) {
                Ok(gauge) => gauge,
                Err(e) => {
                    println!("  Gauge {path}: {e} (skipped)");
                    continue;
                }
            };
            let Some(station) = gauge.station.clone() else {
                println!("  Gauge {path}: no station position (skipped)");
                continue;
            };
            let (times, values): (Vec<f64>, Vec<f64>) = gauge
                .time_series
                .times()
                .into_iter()
                .zip(gauge.time_series.values())
                .filter(|(_, v)| v.is_finite())
                .unzip();
            let observed_fit = fit_record(&times, &values, &GAUGE_CONSTITUENTS, None);
            let norkyst = atlas
                .and_then(|a| a.nearest(station.longitude, station.latitude))
                .filter(|&(p, d)| {
                    d <= STATION_MAX_OFFSET
                        && d > 0.0
                        && p.constituents.iter().all(|c| c.velocity.is_some())
                });
            let name = station.name.clone();
            add(
                name.clone(),
                station.longitude,
                station.latitude,
                Some(Gauge {
                    station,
                    observed: TimeSeries::new(&times, &values),
                    observed_fit,
                }),
                None,
                days(&times),
            );
            if let Some((p, _)) = norkyst {
                add(
                    format!("{name} NorKyst point"),
                    p.lon,
                    p.lat,
                    None,
                    None,
                    0.0,
                );
            }
        }
        for path in currents {
            let record = match read_adcp_file(Path::new(path)) {
                Ok(record) => record,
                Err(e) => {
                    println!("  Current record {path}: {e} (skipped)");
                    continue;
                }
            };
            let observed = record.time_series;
            let (times, u, v) = (observed.times(), observed.u_values(), observed.v_values());
            let observed_fit = fit_current_record(&times, &u, &v, &GAUGE_CONSTITUENTS, None);
            let station = record.station;
            add(
                station.name.clone(),
                station.longitude,
                station.latitude,
                None,
                Some(CurrentRecord {
                    station,
                    observed,
                    observed_fit,
                }),
                days(&times),
            );
        }
        stations
    }

    /// Where a station at `p` samples the model: at `p` if its element is
    /// submerged (every node at least `STATION_MIN_DEPTH` below mean sea
    /// level), else at the nearest point of a submerged element within
    /// `STATION_MAX_OFFSET`. Shoreline elements are avoided because
    /// a coarse mesh leaves pockets there that hold water above the tide.
    /// The probe, its distance from `p` and its still-water depth.
    fn probe(&self, locator: &PointLocator2D, p: [f64; 2]) -> Option<(Probe2D, f64, f64)> {
        let submerged = |k: ElementIndex| {
            self.bathymetry
                .element(k)
                .iter()
                .all(|&b| b <= -STATION_MIN_DEPTH)
        };
        let depth = |probe: &Probe2D| -probe.evaluate(self.bathymetry.element(probe.element()));
        if let Some(probe) = Probe2D::at(locator, &self.ops, p)
            && submerged(probe.element())
        {
            let d = depth(&probe);
            return Some((probe, 0.0, d));
        }
        // The station's reference coordinates clamped to the element: its
        // nearest point on a rectangle (the Frøya grid), close to it on any
        // convex quadrilateral
        let (distance, point) = ElementIndex::iter(self.mesh.n_elements)
            .filter(|&k| submerged(k))
            .filter_map(|k| {
                let [r, s] = inverse_bilinear(&self.mesh.element_vertices(k), p)
                    .unwrap_or([0.0, 0.0])
                    .map(|c| c.clamp(-1.0, 1.0));
                let [x, y] = self.mesh.reference_to_physical(k, r, s);
                let distance = (x - p[0]).hypot(y - p[1]);
                (distance <= STATION_MAX_OFFSET)
                    .then_some((distance, MeshPoint { element: k, r, s }))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))?;
        let probe = Probe2D::new(&self.mesh, &self.ops, point);
        let d = depth(&probe);
        Some((probe, distance, d))
    }

    fn volume(&self, q: &SWESolution2D) -> f64 {
        ElementIndex::iter(self.mesh.n_elements)
            .map(|k| {
                (0..self.ops.n_nodes)
                    .map(|i| {
                        self.geom.mass[self.geom.node_index(k.as_usize(), i)] * q.get_state(k, i).h
                    })
                    .sum::<f64>()
            })
            .sum()
    }

    fn stats(&self, q: &SWESolution2D) -> Stats {
        let mut stats = Stats {
            eta: (f64::MAX, f64::MIN),
            highest: (0.0, 0.0, 0.0),
            speed: 0.0,
            fastest: (0.0, 0.0, 0.0),
            open_speed: 0.0,
            open_fastest: (0.0, 0.0, 0.0),
            wet: 0,
        };
        for k in ElementIndex::iter(self.mesh.n_elements) {
            for i in 0..self.ops.n_nodes {
                let s = q.get_state(k, i);
                if s.h > WetDryConfig::DEFAULT_H_DRY {
                    stats.wet += 1;
                    let eta = s.h + self.bathymetry.get(k, i);
                    if eta > stats.eta.1 {
                        let [x, y] = self.mesh.reference_to_physical(
                            k,
                            self.ops.nodes_r[i],
                            self.ops.nodes_s[i],
                        );
                        stats.highest = (x, y, s.h);
                    }
                    stats.eta = (stats.eta.0.min(eta), stats.eta.1.max(eta));
                }
                let speed = s.hu.hypot(s.hv) / s.h;
                let at = || {
                    let [x, y] = self.mesh.reference_to_physical(
                        k,
                        self.ops.nodes_r[i],
                        self.ops.nodes_s[i],
                    );
                    (x, y, s.h)
                };
                if s.h > 0.1 && speed > stats.speed {
                    stats.speed = speed;
                    stats.fastest = at();
                }
                if -self.bathymetry.get(k, i) >= STATION_MIN_DEPTH && speed > stats.open_speed {
                    stats.open_speed = speed;
                    stats.open_fastest = at();
                }
            }
        }
        stats
    }

    fn print_summary(&self) {
        let depths: Vec<f64> = self.bathymetry.data.iter().map(|b| -b).collect();
        let n_nodes = depths.len();
        let dry = depths.iter().filter(|&&d| d <= 0.0).count();
        let shoreline = ElementIndex::iter(self.mesh.n_elements)
            .filter(|&k| (0..self.ops.n_nodes).any(|i| self.bathymetry.get(k, i) >= 0.0))
            .count();
        println!(
            "  Water-only mesh: {} elements (P{}), {} boundary faces; {shoreline} shoreline elements",
            self.mesh.n_elements, self.ops.order, self.mesh.n_boundary_edges
        );
        println!(
            "  Depth: max {:.0} m; {:.1} % of nodes dry at mean sea level",
            depths.iter().copied().fold(0.0, f64::max),
            100.0 * dry as f64 / n_nodes as f64
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = Options::parse()?;
    println!("Frøya–Smøla–Hitra tidal run (WetDry split form, Simulation)\n");
    println!("Setting up the domain...");
    let mut domain = match Domain::froya(&opts)? {
        Some(domain) => domain,
        None => {
            println!("  Data files not found in ./data/: synthetic basin instead");
            Domain::synthetic(&opts)
        }
    };
    domain.print_summary();
    let parent = domain.nesting(&opts)?;
    domain.close_uncovered_open_faces(&opts)?;
    if opts.waves > 0 {
        wave_cost(&domain, &opts);
        return Ok(());
    }

    lake_at_rest(&domain, opts.rest_hours);
    tidal_run(&domain, &opts, parent)
}

/// Walls everywhere and no forcing: the largest spurious current and surface
/// deviation where h > 10 cm.
fn lake_at_rest(domain: &Domain, hours: f64) {
    println!("\nLake at rest ({hours} h, walls only)...");
    let physics = domain.builder(Reflective2D::new()).build();
    let mut q = domain.at_rest();
    let rhs = physics.compute_rhs(&q, 0.0);
    println!("  max |dq/dt| at t = 0: {:.2e}", rhs.max_abs());

    let sim = Simulation::new(physics, SSPRK3).with_cfl(1.0);
    let result = sim.run(&mut q, 0.0, hours * 3600.0);
    let stats = domain.stats(&q);
    println!(
        "  after {} steps ({:.1} s wall): max |u| {:.2e} m/s, η in [{:.2e}, {:.2e}] m{}",
        result.n_steps,
        result.wall_time,
        stats.speed,
        stats.eta.0,
        stats.eta.1,
        if result.success { "" } else { " (FAILED)" }
    );
}

fn tidal_run(
    domain: &Domain,
    opts: &Options,
    parent: Option<OceanModelState>,
) -> Result<(), Box<dyn std::error::Error>> {
    let t_end = opts.hours * 3600.0;
    let wall = Reflective2D::new();
    let station_atlas = Path::new(&opts.station_atlas);
    let station_atlas = station_atlas
        .exists()
        .then(|| TidalAtlas::read(station_atlas))
        .transpose()?;
    let mut stations = domain.stations(&opts.gauges, &opts.currents, station_atlas.as_ref());
    let clock = ModelClock::parse(&opts.start)?;
    println!("  Clock: t = 0 at {} UTC", clock.format(0.0));
    let weather = weather(domain, opts, &clock, t_end)?;
    let level = match &weather {
        Some(gridded) => Some(Level::Gridded(gridded.clone())),
        None => opts.wind.then(|| Level::Uniform(pressure())),
    };
    let reference = AtlasReference::new(&stations, station_atlas.as_ref());
    let parent = match parent {
        Some(parent) if opts.correct_nested_tides => Some(correct_parent_tides(
            domain,
            opts,
            reference.as_ref(),
            parent,
            t_end,
        )?),
        other => other,
    };
    let band = parent
        .as_ref()
        .filter(|_| opts.band_km > 0.0)
        .map(|p| p.relaxation(60.0 * opts.band_minutes));
    let (open, forcing) = open_boundary(
        domain,
        opts,
        &clock,
        t_end,
        reference.as_ref(),
        parent,
        level,
    )?;
    let bc = MultiBoundaryCondition2D::new(&wall).with_open(open.as_ref());

    let mut builder = domain.builder(bc);
    if let Some(band) = band {
        builder = builder.with_source(band);
    }
    if let Some(weather) = weather {
        builder = builder.with_source(weather);
    } else if opts.wind {
        builder = builder
            .with_source(
                WindStress2D::from_direction(WIND_SPEED, WIND_DIRECTION)
                    .with_drag(DragCoefficient::LargePond),
            )
            .with_source(pressure());
    }
    let physics: SWEPhysics2D<_> = builder.build();
    if opts.profile > 0 {
        profile_phases(domain, &physics, opts.profile);
        return Ok(());
    }

    let output_dir = opts
        .output
        .clone()
        .unwrap_or_else(|| Path::new("output").join(domain.name));
    fs::create_dir_all(&output_dir)?;
    #[cfg(feature = "netcdf")]
    let mut netcdf = match &domain.projection {
        Some(projection) => Some(NetCDFWriter::create(
            NetCDFWriterConfig::new(output_dir.join("froya.nc").to_string_lossy())
                .with_title("Frøya–Smøla–Hitra tidal run")
                .with_institution("dg-rs")
                .with_clock(clock),
            &netcdf_mesh_info(domain, projection),
        )?),
        None => None,
    };

    println!(
        "\nTides: {forcing}, {:.2} h{} → {}",
        opts.hours,
        if !opts.met.is_empty() {
            ", gridded wind and pressure"
        } else if opts.wind {
            ", wind and pressure"
        } else {
            ""
        },
        output_dir.display()
    );
    println!(
        "  Integrator: {}{}, CFL {:.3}, capped at {:.3} where elements may run dry",
        opts.integrator.name(),
        if opts.lts > 0 {
            format!(", local time stepping up to {} levels", opts.lts)
        } else {
            String::new()
        },
        opts.cfl,
        opts.integrator.ssp_coefficient() * positivity_cfl_swe_2d(opts.order),
    );
    println!(
        "  time  |  η range (m)     | η max at (x, y km; h m) | max |u| (m/s) at (x, y km; h m) | open water ≥ 3 m (m/s)         | wet nodes | volume change"
    );

    let mut q = domain.at_rest();
    let volume0 = domain.volume(&q);
    // Callbacks sample the stations; every `output_every`-th also writes
    // output, every `snapshot_every`-th a snapshot frame
    let snapshot_minutes = (opts.snapshot_minutes > 0.0).then_some(opts.snapshot_minutes);
    let base_minutes = if stations.is_empty() {
        opts.output_minutes
            .min(snapshot_minutes.unwrap_or(f64::INFINITY))
    } else {
        opts.station_minutes
    };
    let every = |minutes: f64| (minutes / base_minutes).round().max(1.0) as usize;
    let (interval, output_every) = (base_minutes * 60.0, every(opts.output_minutes));
    let snapshot_every = snapshot_minutes.map(every);
    let mut snapshot = match snapshot_every {
        Some(every) => {
            let path = output_dir.join(format!("{}.dgsnap", domain.name));
            println!(
                "  Snapshot file: {} every {} min",
                path.display(),
                every as f64 * base_minutes
            );
            // The stations in mesh coordinates, for the viewer
            let station_lines: Vec<String> = domain
                .projection
                .as_ref()
                .map(|projection| {
                    stations
                        .iter()
                        .map(|s| {
                            let (x, y) = projection.geo_to_xy(s.latitude, s.longitude);
                            format!("{},{x:.1},{y:.1}", s.name)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let title = match domain.projection {
                // The date is the header's clock
                Some(_) => "Frøya–Smøla–Hitra",
                None => "Synthetic basin with an island and a beach",
            };
            let mut metadata = vec![("title", title), ("source", "froya_real_data")];
            metadata.extend(station_lines.iter().map(|s| ("station", s.as_str())));
            Some(SnapshotWriter::create(
                path,
                &domain.mesh,
                &domain.ops,
                &domain.bathymetry.data,
                Some(&clock),
                &metadata,
                WetDryConfig::DEFAULT_H_DRY,
            )?)
        }
        None => None,
    };
    let mut n_callbacks = 0;
    let mut frame = 0;
    let mut write_error = None;
    let start = Instant::now();
    let mut callback = |q: &SWESolution2D, t: f64| {
        for s in &mut stations {
            s.sample(q, &domain.bathymetry, clock.unix(t));
        }
        n_callbacks += 1;
        if let (Some(writer), Some(every)) = (snapshot.as_mut(), snapshot_every)
            && (n_callbacks - 1) % every == 0
            && let Err(e) = writer.write_state(t, q)
        {
            write_error.get_or_insert(e.to_string());
        }
        if (n_callbacks - 1) % output_every != 0 {
            return;
        }
        let stats = domain.stats(q);
        let (x, y, h) = stats.fastest;
        let (xo, yo, ho) = stats.open_fastest;
        println!(
            "{:6.2} h | [{:+.3}, {:+.3}] | ({:6.1}, {:6.1}; {:.3}) | {:5.2} at ({:6.1}, {:6.1}; {h:6.1}) | {:5.2} at ({:6.1}, {:6.1}; {ho:6.1}) | {:9} | {:+.3e}",
            t / 3600.0,
            stats.eta.0,
            stats.eta.1,
            stats.highest.0 / 1e3,
            stats.highest.1 / 1e3,
            stats.highest.2,
            stats.speed,
            x / 1e3,
            y / 1e3,
            stats.open_speed,
            xo / 1e3,
            yo / 1e3,
            stats.wet,
            domain.volume(q) / volume0 - 1.0
        );
        let path = output_dir.join(format!("{}_{frame:04}.vtu", domain.name));
        let written = write_vtk_swe(
            &path,
            &domain.mesh,
            &domain.ops,
            q,
            Some(&domain.bathymetry),
            t,
            WetDryConfig::DEFAULT_H_DRY,
        );
        if let Err(e) = written {
            write_error.get_or_insert(e.to_string());
        }
        #[cfg(feature = "netcdf")]
        if let Some(writer) = netcdf.as_mut() {
            let (h, eta, u, v) = netcdf_fields(domain, q);
            if let Err(e) = writer.write_timestep(t, &h, &eta, Some(&u), Some(&v)) {
                write_error.get_or_insert(e.to_string());
            }
        }
        frame += 1;
    };
    let (result, clips) = if opts.lts > 0 {
        let sim = Simulation::new(physics, Multirate::with_base(opts.integrator, opts.lts))
            .with_cfl(opts.cfl)
            .with_callback_interval(interval);
        let result = sim.run_with_callback(&mut q, 0.0, t_end, &mut callback);
        (result, sim.physics().negative_depth_clips())
    } else {
        let sim = Simulation::new(physics, opts.integrator)
            .with_cfl(opts.cfl)
            .with_callback_interval(interval);
        let result = sim.run_with_callback(&mut q, 0.0, t_end, &mut callback);
        (result, sim.physics().negative_depth_clips())
    };
    if let Some(e) = write_error {
        return Err(e.into());
    }

    let steps = result.n_steps.max(1);
    println!(
        "\n{} after {:.2} h: {} steps (mean dt {:.2} s), {:.1} s wall ({:.1} ms/step), {} negative-depth clips",
        if result.success { "Done" } else { "FAILED" },
        result.final_time / 3600.0,
        result.n_steps,
        result.final_time / steps as f64,
        start.elapsed().as_secs_f64(),
        1e3 * start.elapsed().as_secs_f64() / steps as f64,
        clips
    );
    if let Some(stats) = result.local_time_stepping {
        println!(
            "  Local time stepping: time-step ratio up to 2^{}, {:.2}x less RHS work than global steps at the finest step",
            stats.finest_level,
            stats.speedup()
        );
    }
    let analysis_start = clock.unix(opts.spinup_hours * 3600.0);
    for s in &stations {
        let slug = slug(&s.name);
        let path = output_dir.join(format!("station_{slug}.txt"));
        let series = TimeSeries::new(&s.times, &s.eta).with_name(s.name.clone());
        let mut file =
            TideGaugeFile::from_time_series(series).with_station(s.gauge.as_ref().map_or_else(
                || TideGaugeStation::new(s.name.clone(), s.longitude, s.latitude),
                |g| g.station.clone(),
            ));
        file.datum = Some("MSL".into());
        file.units = Some("m".into());
        write_tide_gauge_file(&path, &file)?;
        let current_path = output_dir.join(format!("currents_{slug}.txt"));
        let currents = CurrentTimeSeries::new(&s.times, &s.u_east, &s.v_north);
        let station =
            ADCPStation::new(s.name.clone(), s.longitude, s.latitude).with_water_depth(s.depth);
        write_adcp_file(&current_path, &station, &currents)?;
        println!(
            "\nStation {} → {}, {}",
            s.name,
            path.display(),
            current_path.display()
        );
        let sampled = if s.offset > 0.0 {
            format!("{:.0} m from the station", s.offset)
        } else {
            "at the station".into()
        };
        println!("  sampled {sampled}, {:.1} m deep", s.depth);
        let atlas_point = station_atlas
            .as_ref()
            .and_then(|a| a.nearest(s.longitude, s.latitude))
            .filter(|&(_, d)| d <= STATION_MAX_OFFSET);
        if let Some((p, d)) = atlas_point {
            println!(
                "  NorKyst: atlas point {d:.0} m from the station, {:.0} m deep",
                p.depth
            );
        }
        if let Some(gauge) = &s.gauge
            && let Err(e) = report_station(s, gauge, analysis_start, atlas_point.map(|(p, _)| p))
        {
            println!("  no harmonic comparison: {e}");
        }
        if let Err(e) = report_currents(s, analysis_start, atlas_point.map(|(p, _)| p)) {
            println!("  no current comparison: {e}");
        }
    }
    if let Some(e) = result.error {
        return Err(e.into());
    }
    println!("Visualize with ParaView: {}/*.vtu", output_dir.display());
    Ok(())
}

/// Wall time of each phase of an SSP-RK3 step (`profile=N`): after 15 min of
/// spin-up (so the flow and the wet/dry state are realistic), `n` calls of
/// each. A step is 3 RHS, 3 post-processing and 3 implicit-damping calls and
/// one dt.
fn profile_phases<P: PhysicsModule<SWESolution2D>>(domain: &Domain, physics: &P, n: usize) {
    let mut q = domain.at_rest();
    let spin_up = 900.0;
    let mut t = 0.0;
    let mut stages = dg_rs::time::StageWorkspace::new();
    while t < spin_up {
        let dt = physics
            .compute_dt_ssp(&q, 1.0, Some(SspScheme::Rk3))
            .min(spin_up - t);
        dg_rs::time::TimeIntegrator::step_with_relaxation(
            &SSPRK3,
            &mut q,
            dt,
            t,
            |s, time, out| physics.compute_rhs_into(s, time, out),
            |stage, from, dt| physics.implicit_damping(stage, from, dt),
            |s| physics.post_process(s),
            &mut stages,
        );
        t += dt;
    }
    let dt = physics.compute_dt_ssp(&q, 1.0, Some(SspScheme::Rk3));

    // Median of `n` calls (after a warm-up): robust to the boost and thermal
    // swings of a laptop
    let time = |f: &mut dyn FnMut()| {
        f();
        let mut ms: Vec<f64> = (0..n)
            .map(|_| {
                let start = Instant::now();
                f();
                1e3 * start.elapsed().as_secs_f64()
            })
            .collect();
        ms.sort_by(f64::total_cmp);
        ms[n / 2]
    };
    let measure = |threads: usize| {
        let mut out = q.clone();
        let mut scratch = q.clone();
        let rhs = time(&mut || physics.compute_rhs_into(&q, t, &mut out));
        let copy = time(&mut || scratch.clone_from(&q));
        let post = time(&mut || {
            scratch.clone_from(&q);
            physics.post_process(&mut scratch);
        }) - copy;
        let damping = time(&mut || {
            scratch.clone_from(&q);
            physics.implicit_damping(&mut scratch, &q, dt);
        }) - copy;
        let dt_ms = time(&mut || {
            std::hint::black_box(physics.compute_dt_ssp(&q, 1.0, Some(SspScheme::Rk3)));
        });
        let step = 3.0 * (rhs + post + damping) + dt_ms;
        println!("\nPhases after {spin_up} s, {threads} threads: ms per call, share of a step");
        for (name, ms, calls) in [
            ("RHS", rhs, 3.0),
            ("post_process (limiter, wet/dry)", post, 3.0),
            ("implicit damping", damping, 3.0),
            ("dt", dt_ms, 1.0),
        ] {
            println!(
                "  {name:32} {ms:7.3} ms  {:5.1} %",
                100.0 * calls * ms / step
            );
        }
        println!("  step (sum, without RK combinations) {step:.2} ms");
    };

    // One thread, then all of them
    let all = std::thread::available_parallelism().map_or(1, |n| n.get());
    for threads in [1, all] {
        #[cfg(feature = "parallel")]
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool")
            .install(|| measure(threads));
        #[cfg(not(feature = "parallel"))]
        {
            measure(1);
            break;
        }
    }
}

/// The cost of the spectral wave model on this domain (`waves=N`, TODO F.4):
/// SWAN's default sources (Komen, the DIA, JONSWAP friction, Battjes–Janssen)
/// under a 10 m/s wind to the east, a JONSWAP sea of H_s 2.5 m and T_p 10 s from
/// the west-north-west through the open boundaries and, to start with, over the
/// whole domain. The median wall time of N steps after a warm-up, split into the
/// propagation and the sources, the step and what sets it, and the cost per
/// model hour.
fn wave_cost(domain: &Domain, opts: &Options) {
    use dg_rs::waves::{
        SourceTerms, SpectralGrid, WaveModel2D, WaveWorkspace, Wind, group_velocity, wavenumber,
    };
    let [nf, nd] = opts.wave_grid;
    let grid = SpectralGrid::new(0.04, 0.5, nf, nd);
    let n_components = grid.n_components();
    // Travelling to the east-south-east
    let direction = (-15f64).to_radians();
    let sea = grid.jonswap(2.5, 10.0, 3.3, direction, 4.0);
    let model = WaveModel2D::new(
        domain.mesh.clone(),
        domain.ops.clone(),
        domain.geom.clone(),
        &domain.bathymetry,
        grid,
        G,
    )
    .with_sources(SourceTerms::swan_defaults(G))
    .with_wind(Wind {
        u10: 10.0,
        direction: 0.0,
    })
    .with_boundary_spectrum(&sea)
    .with_turning_limit(opts.turning);
    let np = model.n_points();
    println!(
        "\nSpectral wave model: {} elements, {np} nodes (P{}), {nf} frequencies (0.04–0.5 Hz) × \
         {nd} directions = {n_components} components, {:.1} M unknowns",
        domain.mesh.n_elements,
        domain.ops.order,
        (np * n_components) as f64 / 1e6
    );

    // The step, and the part of it the geographic propagation alone allows
    let cfl = 0.5;
    let dt = model.compute_dt(cfl);
    let sigma_min = model.grid.sigma[0];
    let nn = domain.ops.n_nodes;
    let order_factor = (2 * domain.ops.order + 1) as f64;
    let element_dt: Vec<f64> = (0..domain.mesh.n_elements)
        .map(|k| {
            let cg = (k * nn..(k + 1) * nn)
                .map(|p| {
                    let d = model.depth()[p];
                    group_velocity(sigma_min, wavenumber(sigma_min, d, G), d)
                })
                .fold(0.0, f64::max);
            cfl * domain.geom.element_size(k) / (order_factor * cg)
        })
        .collect();
    let dt_geographic = element_dt.iter().cloned().fold(f64::INFINITY, f64::min);
    // Every element at its own step against all at the smallest: the most
    // local time stepping could save (power-of-two levels save a little less)
    let multirate = element_dt.len() as f64
        / element_dt
            .iter()
            .map(|dt| dt_geographic / dt)
            .sum::<f64>();
    println!(
        "  elements' own geographic steps {dt_geographic:.3}–{:.1} s: local time stepping could \
         save up to {multirate:.1}×",
        element_dt.iter().cloned().fold(0.0, f64::max)
    );
    println!(
        "  step {dt:.3} s at CFL {cfl}; the geographic propagation alone would allow {dt_geographic:.3} s{}",
        if dt < 0.99 * dt_geographic {
            " (refraction or frequency shifting sets it)"
        } else {
            ""
        }
    );

    let mut n = model.uniform_state(&sea);
    let mut ws = WaveWorkspace::default();
    let mut t = 0.0;
    let median = |times: &mut Vec<f64>| {
        times.sort_by(f64::total_cmp);
        times[times.len() / 2]
    };
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    // Warm-up, then the steps timed
    for _ in 0..2 {
        model.step(&mut n, t, dt, &mut ws);
        t += dt;
    }
    let mut steps = Vec::with_capacity(opts.waves);
    let mut sources = Vec::with_capacity(opts.waves);
    for _ in 0..opts.waves {
        let start = Instant::now();
        model.step(&mut n, t, dt, &mut ws);
        steps.push(start.elapsed().as_secs_f64());
        t += dt;
        let mut scratch = n.clone();
        let start = Instant::now();
        model.apply_sources(&mut scratch, dt, &mut ws);
        sources.push(start.elapsed().as_secs_f64());
    }
    let (step, source) = (median(&mut steps), median(&mut sources));
    let per_hour = 3600.0 / dt * step;
    println!(
        "  {:.1} ms per step on {threads} threads: propagation {:.1} ms ({:.0} %), sources {:.1} ms \
         ({:.0} %)",
        1e3 * step,
        1e3 * (step - source),
        100.0 * (step - source) / step,
        1e3 * source,
        100.0 * source / step
    );
    println!(
        "  {:.0} ns per component and node per step ({:.1} ns of it propagation, three RK stages)",
        1e9 * step / (np * n_components) as f64,
        1e9 * (step - source) / (np * n_components) as f64,
    );
    println!(
        "  {per_hour:.0} s of wall time per model hour ({:.2}× real time)",
        3600.0 / per_hour
    );
    let params = model.parameters(&n);
    let hs_max = params.iter().map(|p| p.hs).fold(0.0, f64::max);
    println!(
        "  after {:.0} s: H_s ≤ {hs_max:.2} m, total action {:.3e}",
        t,
        model.total_action(&n)
    );
}

/// Lower-case ASCII file-name form of a station name.
fn slug(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// Reference constants of the constituents a record resolves; the others of
/// P1, K2, N2 and Q1 are inferred with the ratios of `reference` (a longer
/// record at the same place) where it has them, else at equilibrium.
fn fit_record(
    times: &[f64],
    values: &[f64],
    candidates: &[&'static str],
    reference: Option<&ReferenceFit>,
) -> Result<ReferenceFit, String> {
    let (Some(&t0), Some(&t1)) = (times.first(), times.last()) else {
        return Err("empty record".into());
    };
    let names = resolvable_constituents(candidates, t1 - t0, 1.0);
    let inferred: Vec<Inference> = Inference::EQUILIBRIUM
        .into_iter()
        .filter(|i| !names.contains(&i.name) && names.contains(&i.from))
        .map(|i| {
            reference
                .and_then(|r| Inference::from_reference(i.name, i.from, r))
                .unwrap_or(i)
        })
        .collect();
    fit_reference_constants(times, values, &names, &inferred)
}

/// Tidal ellipses of the constituents a current record resolves; the others
/// of P1, K2, N2 and Q1 are inferred with the major-axis ratio and lag
/// difference of `reference` (a longer record at the same place) where it
/// has them, else at equilibrium, for both components alike.
fn fit_current_record(
    times: &[f64],
    u: &[f64],
    v: &[f64],
    candidates: &[&'static str],
    reference: Option<&EllipseFit>,
) -> Result<EllipseFit, String> {
    let (Some(&t0), Some(&t1)) = (times.first(), times.last()) else {
        return Err("empty record".into());
    };
    let names = resolvable_constituents(candidates, t1 - t0, 1.0);
    let inferred: Vec<Inference> = Inference::EQUILIBRIUM
        .into_iter()
        .filter(|i| !names.contains(&i.name) && names.contains(&i.from))
        .map(|i| {
            let from_reference = reference.and_then(|r| {
                let (a, b) = (r.get(i.name)?, r.get(i.from)?);
                (b.major > 0.0).then(|| Inference {
                    amplitude_ratio: a.major / b.major,
                    lag_offset_deg: a.lag_deg - b.lag_deg,
                    ..i
                })
            });
            from_reference.unwrap_or(i)
        })
        .collect();
    fit_tidal_ellipses(times, u, v, &names, &inferred)
}

/// The model's reference constants at a station after spin-up (from
/// `analysis_start`, Unix s) against the gauge's (whole record) and
/// NorKyst's (`norkyst`, the station atlas); RMSE against the observations
/// and the gauge's tidal prediction.
fn report_station(
    s: &Station,
    gauge_record: &Gauge,
    analysis_start: f64,
    norkyst: Option<&AtlasPoint>,
) -> Result<(), String> {
    let first = s.times.partition_point(|&t| t < analysis_start);
    let (times, eta) = (&s.times[first..], &s.eta[first..]);
    let gauge_times = gauge_record.observed.times();
    let gauge = gauge_record
        .observed_fit
        .as_ref()
        .map_err(|e| format!("gauge fit: {e}"))?;
    let model = fit_record(times, eta, &STATION_CONSTITUENTS, Some(gauge))?;

    let days = |t: &[f64]| (t[t.len() - 1] - t[0]) / 86_400.0;
    println!(
        "  Surface: model {:.1} days after spin-up (R² {:.4}), gauge {:.1} days (R² {:.4})",
        days(times),
        model.r_squared,
        days(&gauge_times),
        gauge.r_squared
    );
    println!(
        "  name |  model H   G    |  gauge H   G    | ratio   ΔG (°)  |ΔZ| (m) | NorKyst H   G    |ΔZ| (m)"
    );
    let cmp = |a: &ReferenceConstant, h: f64, g: f64| {
        ConstituentComparison::new(
            a.name,
            a.amplitude,
            a.lag_deg.to_radians(),
            h,
            g.to_radians(),
        )
    };
    let (mut rss_gauge, mut rss_norkyst) = (0.0, 0.0);
    for c in &model.constants {
        let Some(g) = gauge.get(c.name) else { continue };
        let vs_gauge = cmp(c, g.amplitude, g.lag_deg);
        rss_gauge += vs_gauge.complex_difference().powi(2);
        let mut line = format!(
            "  {:4} | {:7.4} {:6.1}{} | {:7.4} {:6.1} | {:5.3} {:+7.1}   {:.4}",
            c.name,
            c.amplitude,
            c.lag_deg,
            if c.inferred { "*" } else { " " },
            g.amplitude,
            g.lag_deg,
            vs_gauge.amplitude_ratio,
            vs_gauge.phase_error_degrees(),
            vs_gauge.complex_difference()
        );
        if let Some(n) = norkyst.and_then(|p| p.constituents.iter().find(|n| n.name == c.name)) {
            let vs_norkyst = cmp(c, n.eta.0, n.eta.1);
            rss_norkyst += vs_norkyst.complex_difference().powi(2);
            line += &format!(
                "  | {:7.4} {:6.1}   {:.4}",
                n.eta.0,
                n.eta.1,
                vs_norkyst.complex_difference()
            );
        }
        println!("{line}");
    }
    println!("  (* inferred with the gauge's ratio)");
    print!(
        "  root sum of squares of |ΔZ|: {:.4} m vs the gauge",
        rss_gauge.sqrt()
    );
    if norkyst.is_some() {
        print!(", {:.4} m vs NorKyst", rss_norkyst.sqrt());
    }
    println!();

    // Time series after spin-up. The forcing has no mean level, so the bias
    // is the gauge's mean over the window: see the centred RMSE
    let model_series = TimeSeries::new(times, eta);
    let prediction = TimeSeries::new(times, &gauge.predict(times));
    for (what, reference) in [
        ("observations", &gauge_record.observed),
        ("gauge tidal prediction", &prediction),
    ] {
        let v = StationValidationResult::compute(&gauge_record.station, &model_series, reference);
        let centred = (v.metrics.rmse.powi(2) - v.metrics.bias.powi(2))
            .max(0.0)
            .sqrt();
        println!(
            "  vs {what}: RMSE {:.3} m (centred {centred:.3}), bias {:+.3} m, correlation {:.4}, \
             std model {:.3} / reference {:.3} m ({} samples)",
            v.metrics.rmse,
            v.metrics.bias,
            v.metrics.correlation,
            v.model_std,
            v.obs_std,
            v.metrics.n_points
        );
    }
    Ok(())
}

/// The model's tidal ellipses of the depth-averaged current at a station
/// after spin-up (from `analysis_start`, Unix s) against NorKyst's (the
/// station atlas, if it has currents) and a current record's (its whole
/// record fitted); time-series skill against the record.
fn report_currents(
    s: &Station,
    analysis_start: f64,
    norkyst: Option<&AtlasPoint>,
) -> Result<(), String> {
    let first = s.times.partition_point(|&t| t < analysis_start);
    let (times, u, v) = (&s.times[first..], &s.u_east[first..], &s.v_north[first..]);
    let observed = s
        .current
        .as_ref()
        .map(|c| {
            c.observed_fit
                .as_ref()
                .map_err(|e| format!("current record fit: {e}"))
        })
        .transpose()?;
    let model = fit_current_record(times, u, v, &STATION_CONSTITUENTS, observed)?;
    let norkyst: Vec<TidalEllipse> = norkyst
        .map(|p| {
            p.constituents
                .iter()
                .filter_map(|c| {
                    Some(TidalEllipse::from_components(
                        c.name,
                        c.velocity?.0,
                        c.velocity?.1,
                    ))
                })
                .collect()
        })
        .unwrap_or_default();
    println!(
        "  Depth-averaged current: model {:.1} days after spin-up (R² {:.4}), mean {:+.3} m/s east, \
         {:+.3} m/s north",
        (times[times.len() - 1] - times[0]) / 86_400.0,
        model.r_squared(u, v),
        model.mean.0,
        model.mean.1
    );
    let references: Vec<(&str, &[TidalEllipse])> = [
        (!norkyst.is_empty()).then_some(("NorKyst", norkyst.as_slice())),
        observed.map(|o| ("record", o.ellipses.as_slice())),
    ]
    .into_iter()
    .flatten()
    .collect();
    let mut header = "  name | model major  minor   inc     G   ".to_string();
    for (name, _) in &references {
        header += &format!("| {name:>7} major  minor   inc     G    |ΔW| ");
    }
    println!("{header}  (m/s, degrees: inclination from east, Greenwich lag)");
    let mut rss = vec![0.0; references.len()];
    for e in &model.ellipses {
        let mut line = format!(
            "  {:4} | {:11.4} {:+7.4} {:5.1} {:6.1}{} ",
            e.name,
            e.major,
            e.minor,
            e.inclination_deg,
            e.lag_deg,
            if e.inferred { "*" } else { " " }
        );
        for ((_, reference), rss) in references.iter().zip(&mut rss) {
            match reference.iter().find(|r| r.name == e.name) {
                Some(r) => {
                    let d = e.complex_difference(r);
                    *rss += d * d;
                    line += &format!(
                        "| {:13.4} {:+7.4} {:5.1} {:6.1}  {d:.4} ",
                        r.major, r.minor, r.inclination_deg, r.lag_deg
                    );
                }
                None => line += &format!("| {:>46} ", "-"),
            }
        }
        println!("{line}");
    }
    if !references.is_empty() {
        let summary: Vec<String> = references
            .iter()
            .zip(&rss)
            .map(|((name, _), rss)| format!("{:.4} m/s vs {name}", rss.sqrt()))
            .collect();
        println!(
            "  (* inferred) root sum of squares of |ΔW|: {}",
            summary.join(", ")
        );
    }

    // Time series against the record, paired by time
    if let Some(record) = &s.current {
        let model_series = CurrentTimeSeries::new(times, u, v);
        let (paired, _) = CurrentTimeSeries::paired_by_time(&model_series, &record.observed);
        if paired.len() < 2 {
            return Err("the current record does not overlap the run after spin-up".into());
        }
        let r = ADCPValidationResult::compute(&record.station, &model_series, &record.observed);
        let m = &r.metrics;
        println!(
            "  vs current record ({} samples): RMSE u {:.3}, v {:.3}, speed {:.3} m/s; \
             bias u {:+.3}, v {:+.3} m/s; vector correlation {:.3}; direction RMSE {:.1}°",
            m.u_metrics.n_points,
            m.u_metrics.rmse,
            m.v_metrics.rmse,
            m.speed_metrics.rmse,
            m.u_metrics.bias,
            m.v_metrics.bias,
            m.vector_correlation,
            m.direction_rmse
        );
    }
    Ok(())
}

/// The atmospheric pressure gradient of the `wind` option.
fn pressure() -> AtmosphericPressure2D {
    AtmosphericPressure2D::from_direction(PRESSURE_GRADIENT, PRESSURE_DIRECTION)
}

/// The inverse-barometer level of the pressure forcing.
enum Level {
    /// The uniform gradient of the `wind` option
    Uniform(AtmosphericPressure2D),
    /// A weather model's pressure (`met=`)
    Gridded(GriddedAtmosphere2D),
}

impl BoundaryLevel for Level {
    fn level(&self, ctx: &BCContext2D) -> f64 {
        match self {
            Self::Uniform(p) => p.inverse_barometer(ctx.position.0, ctx.position.1, ctx.time),
            Self::Gridded(g) => g.level(ctx),
        }
    }
}

/// Characteristic OBC with external data from `provider`, raised by the
/// inverse-barometer level of the pressure forcing, if any.
fn characteristic<P: ExternalStateProvider + 'static>(
    provider: P,
    level: Option<Level>,
) -> Box<dyn SWEBoundaryCondition2D> {
    match level {
        Some(level) => Box::new(CharacteristicOBC::new(InverseBarometer::new(
            provider, level,
        ))),
        None => Box::new(CharacteristicOBC::new(provider)),
    }
}

/// Weather-model wind and pressure (`met=`) on the mesh, ramped up like the
/// tide.
#[cfg(feature = "netcdf")]
fn weather(
    domain: &Domain,
    opts: &Options,
    clock: &ModelClock,
    t_end: f64,
) -> Result<Option<GriddedAtmosphere2D>, Box<dyn std::error::Error>> {
    let Some(projection) = domain.projection.filter(|_| !opts.met.is_empty()) else {
        return Ok(None);
    };
    let reader = Arc::new(AtmosphereReader::from_files(&opts.met)?);
    println!("  {}", reader.summary());
    let atmosphere =
        GriddedAtmosphere2D::new(reader, &domain.mesh, &domain.ops, projection, *clock)?
            .with_ramp_up(3600.0 * opts.ramp_hours);
    atmosphere.check_time_coverage(0.0, t_end)?;
    Ok(Some(atmosphere))
}

#[cfg(not(feature = "netcdf"))]
fn weather(
    _domain: &Domain,
    _opts: &Options,
    _clock: &ModelClock,
    _t_end: f64,
) -> Result<Option<GriddedAtmosphere2D>, Box<dyn std::error::Error>> {
    Ok(None)
}

type OpenBoundary = (Box<dyn SWEBoundaryCondition2D>, String);

/// What the boundary atlas is corrected with: the first gauge with a
/// whole-record fit, and NorKyst at it (the nearest `station_atlas` point
/// within `STATION_MAX_OFFSET`, and its distance in m).
struct AtlasReference<'a> {
    station: &'a str,
    fit: &'a ReferenceFit,
    norkyst: Option<(&'a AtlasPoint, f64)>,
}

impl<'a> AtlasReference<'a> {
    fn new(stations: &'a [Station], station_atlas: Option<&'a TidalAtlas>) -> Option<Self> {
        stations.iter().find_map(|s| {
            let fit = s.gauge.as_ref()?.observed_fit.as_ref().ok()?;
            let norkyst = station_atlas
                .and_then(|a| a.nearest(s.longitude, s.latitude))
                .filter(|&(_, d)| d <= STATION_MAX_OFFSET);
            Some(Self {
                station: &s.name,
                fit,
                norkyst,
            })
        })
    }
}

/// `atlas` corrected at the reference gauge (`TidalAtlas::infer`): first the
/// `gauge_gains` constituents are scaled by the complex ratio of the gauge's
/// constant to NorKyst's there, which keeps NorKyst's spatial structure and
/// fixes it at the gauge (the diurnals: NorKyst's K1 is 1.14×, +8.6° and its
/// O1 0.95×, −17.3° at Mausund); then the `gauge_ratios` constituents are
/// re-inferred from their neighbours with the gauge's ratios. Unchanged
/// without a gauge fit, and without the gains if NorKyst is not known at
/// the gauge.
fn corrected_atlas(
    atlas: &TidalAtlas,
    opts: &Options,
    reference: Option<&AtlasReference>,
) -> Result<TidalAtlas, Box<dyn std::error::Error>> {
    let mut atlas = atlas.clone();
    let signed = |lag: f64| (lag + 180.0).rem_euclid(360.0) - 180.0;
    let Some(&AtlasReference {
        station,
        fit,
        norkyst,
    }) = reference
    else {
        if !opts.gauge_gains.is_empty() || !opts.gauge_ratios.is_empty() {
            println!(
                "  No gauge fit: the atlas keeps its own {:?}",
                [&opts.gauge_gains[..], &opts.gauge_ratios[..]].concat()
            );
        }
        return Ok(atlas);
    };
    match norkyst {
        None if !opts.gauge_gains.is_empty() => println!(
            "  No NorKyst constants at {station} (station_atlas=): the atlas keeps its own {:?}",
            opts.gauge_gains
        ),
        None => {}
        Some((point, distance)) => {
            for &name in &opts.gauge_gains {
                let (Some(gauge), Some(source)) = (
                    fit.get(name),
                    point.constituents.iter().find(|c| c.name == name),
                ) else {
                    return Err(format!("gauge_gains: {station} or NorKyst lacks {name}").into());
                };
                let gain = gauge.amplitude / source.eta.0;
                let lag = signed(gauge.lag_deg - source.eta.1);
                atlas.infer(name, name, gain, lag)?;
                println!(
                    "  Atlas {name} × {gain:.3}, lag {lag:+.1}° (gauge {station} over NorKyst \
                     {distance:.0} m from it)"
                );
            }
        }
    }
    for &name in &opts.gauge_ratios {
        let Some(inference) = Inference::EQUILIBRIUM.iter().find(|i| i.name == name) else {
            return Err(format!("gauge_ratios: {name} is not P1, K2, N2 or Q1").into());
        };
        let from = inference.from;
        let i = Inference::from_reference(name, from, fit)
            .ok_or(format!("gauge_ratios: {station} lacks {name} or {from}"))?;
        atlas.infer(name, from, i.amplitude_ratio, i.lag_offset_deg)?;
        println!(
            "  Atlas {name} = {:.3} × {from}, lag {:+.1}° (ratio at {station})",
            i.amplitude_ratio,
            signed(i.lag_offset_deg)
        );
    }
    Ok(atlas)
}

/// NorKyst with its tides corrected (`nest_tides=corrected`): the boundary
/// atlas is NorKyst's own harmonic fit, so adding (corrected atlas − atlas)
/// keeps NorKyst's residual (coastal current, surge) and replaces its
/// `gauge_gains` and `gauge_ratios` constituents by the gauge-corrected
/// ones, at the boundary
/// and across the relaxation band.
fn correct_parent_tides(
    domain: &Domain,
    opts: &Options,
    reference: Option<&AtlasReference>,
    parent: OceanModelState,
    t_end: f64,
) -> Result<OceanModelState, Box<dyn std::error::Error>> {
    let atlas_path = Path::new(&opts.tides);
    let (Some(projection), true) = (&domain.projection, atlas_path.exists()) else {
        println!(
            "  No tidal atlas at {}: NorKyst's tides as they are",
            opts.tides
        );
        return Ok(parent);
    };
    let raw = TidalAtlas::read(atlas_path)?;
    let correction = corrected_atlas(&raw, opts, reference)?.difference(&raw)?;
    let largest = correction
        .points
        .iter()
        .flat_map(|p| p.constituents.iter().map(|c| (c.eta.0, c.name)))
        .fold((0.0, ""), |a, b| if b.0 > a.0 { b } else { a });
    if largest.0 == 0.0 {
        return Ok(parent);
    }
    println!(
        "  NorKyst tides corrected by (corrected − raw) atlas, largest {:.3} m ({})",
        largest.0, largest.1
    );
    Ok(parent.with_tidal_correction(
        &correction,
        projection,
        t_end,
        ATLAS_COVERAGE + 1000.0 * opts.band_km,
    )?)
}

/// Open-boundary condition and a description of the forcing: NorKyst
/// nesting (`norkyst=`), else atlas tides (`tides=`), else uniform M2; raised
/// by the inverse-barometer `level` of the pressure forcing (for nesting only
/// with `ib=1`).
fn open_boundary(
    domain: &Domain,
    opts: &Options,
    clock: &ModelClock,
    t_end: f64,
    reference: Option<&AtlasReference>,
    parent: Option<OceanModelState>,
    level: Option<Level>,
) -> Result<OpenBoundary, Box<dyn std::error::Error>> {
    if let Some(parent) = parent {
        parent.check_time_coverage(0.0, t_end)?;
        let description = format!(
            "NorKyst nesting ({})",
            parent
                .reader()
                .velocity_source
                .as_deref()
                .unwrap_or("no currents")
        );
        let level = level.filter(|_| opts.nesting_ib);
        return Ok((characteristic(parent, level), description));
    }

    let clock = *clock;
    let atlas_path = Path::new(&opts.tides);
    if let (Some(projection), true) = (&domain.projection, atlas_path.exists()) {
        let atlas = corrected_atlas(&TidalAtlas::read(atlas_path)?, opts, reference)?;
        let tides = atlas
            .boundary_tides(
                &domain.mesh,
                &domain.ops,
                projection,
                BoundaryTag::Open,
                &clock,
                t_end,
                ATLAS_COVERAGE,
            )?
            .with_ramp_up(3600.0 * opts.ramp_hours);
        let tides = if opts.tide_transport > 0.0 {
            tides.with_transport_scaling(opts.tide_transport)
        } else {
            tides
        };
        let mut description = describe(&tides, atlas_path);
        if opts.tide_transport > 0.0 {
            description.push_str(&format!(
                ", velocity transport-scaled (≤ {}×)",
                opts.tide_transport
            ));
        }
        return Ok((characteristic(tides, level), description));
    }

    println!("  No tidal atlas at {}: uniform M2", opts.tides);
    let tide = HarmonicTide::m2(M2_AMPLITUDE, 0.0)
        .with_ramp_up(3600.0 * opts.ramp_hours)
        .with_nodal_corrections(&clock, 0.5 * t_end);
    let description = format!("M2 of {M2_AMPLITUDE} m in one phase at the open boundaries");
    Ok((characteristic(tide, level), description))
}

/// Constituents, forced nodes, and the M2 amplitude range and phase spread
/// along the open boundary.
fn describe(tides: &BoundaryTides, path: &Path) -> String {
    let mut text = format!(
        "{} constituents from {} at {} open-boundary nodes",
        tides.names().len(),
        path.display(),
        tides.n_nodes()
    );
    if let Some(m2) = tides.names().iter().position(|&n| n == "M2") {
        let constants: Vec<(f64, f64)> = (0..tides.n_nodes())
            .map(|slot| tides.elevation_constants(slot, m2))
            .collect();
        let (lo, hi) = constants.iter().fold((f64::MAX, f64::MIN), |(lo, hi), c| {
            (lo.min(c.0), hi.max(c.0))
        });
        // Phases relative to the first node, in (−180°, 180°]
        let relative = constants
            .iter()
            .map(|c| (c.1 - constants[0].1 + 180.0).rem_euclid(360.0) - 180.0);
        let (p_lo, p_hi) =
            relative.fold((f64::MAX, f64::MIN), |(lo, hi), p| (lo.min(p), hi.max(p)));
        text += &format!("; M2 {lo:.3}–{hi:.3} m, phase spread {:.1}°", p_hi - p_lo);
    }
    text
}

#[cfg(feature = "netcdf")]
fn netcdf_mesh_info(domain: &Domain, projection: &LocalProjection) -> NetCDFMeshInfo {
    let (mut x, mut y, mut lat, mut lon) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for k in ElementIndex::iter(domain.mesh.n_elements) {
        for i in 0..domain.ops.n_nodes {
            let [xi, yi] =
                domain
                    .mesh
                    .reference_to_physical(k, domain.ops.nodes_r[i], domain.ops.nodes_s[i]);
            let (la, lo) = projection.xy_to_geo(xi, yi);
            x.push(xi);
            y.push(yi);
            lat.push(la);
            lon.push(lo);
        }
    }
    NetCDFMeshInfo::from_xy(x, y).with_latlon(lat, lon)
}

/// Depth, surface elevation and velocities per node
#[cfg(feature = "netcdf")]
fn netcdf_fields(domain: &Domain, q: &SWESolution2D) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut h, mut eta, mut u, mut v) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for k in ElementIndex::iter(domain.mesh.n_elements) {
        for i in 0..domain.ops.n_nodes {
            let s = q.get_state(k, i);
            let (ui, vi) = s.velocity_simple(Depth::new(WetDryConfig::DEFAULT_H_DRY));
            h.push(s.h);
            eta.push(s.h + domain.bathymetry.get(k, i));
            u.push(ui);
            v.push(vi);
        }
    }
    (h, eta, u, v)
}
