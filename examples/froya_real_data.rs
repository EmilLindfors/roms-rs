//! Frøya–Smøla–Hitra: tides on real bathymetry.
//!
//! The 2D production path (`Simulation` + `SSPRK3` + `SWEPhysics2D`) with the
//! wet/dry defaults: the `WetDry` split form, HLL, positivity limiting, velocity
//! desingularization, point-implicit Manning friction, and the positivity CFL.
//!
//! 1. **Domain.** Bathymetry from the GeoTIFF (EPSG:4326) and land from GSHHS:
//!    a node is water where both say so. A rectangular grid covers the
//!    domain, and only elements with a water node are kept
//!    (`Mesh2D::retain_elements`); their faces to dropped elements are
//!    coastline walls. Land nodes inside kept elements get the bed elevation
//!    `land_elevation`, so `WetDry` treats them as dry shore. The sides of the
//!    rectangle are open where they cross water.
//! 2. **Lake at rest.** Walls everywhere, no forcing: the largest spurious
//!    current and surface error after `rest_hours` (exact balance keeps both
//!    at round-off; the collocated scheme reached m/s on steep beds).
//! 3. **Tides.** A characteristic open boundary (`CharacteristicOBC`) forced
//!    by NorKyst-800 boundary tides (`BoundaryTides` from the tidal atlas
//!    `tides=<file>`, made by `examples/norkyst_boundary_tides.rs`: η, ū, v̄ of
//!    10 constituents plus P1 and K2, varying along the boundary), on a
//!    `ModelClock` starting at `start`. Coriolis, Manning friction, optionally
//!    wind and an atmospheric pressure gradient (with the inverse-barometer
//!    level at the open boundaries), or NorKyst nesting (`norkyst=<file>`).
//!    Without an atlas: M2 of `M2_AMPLITUDE` in one phase along the boundary.
//!    NorKyst-800 has too little N2 here (0.027 m at Mausund against 0.156 m
//!    observed) and about twice the Q1, so `gauge_ratios=N2,Q1` (the
//!    default) re-infers them in the atlas from M2 and O1 with the ratios of
//!    the first gauge's whole-record fit (`TidalAtlas::infer`).
//! 4. **Validation.** Every `station_minutes` the surface is sampled at the
//!    tide gauges `gauges=` (files as written by `scripts/kartverket_gauge.sh`;
//!    the nearest node at least `STATION_MIN_DEPTH` deep). After the run each
//!    station series is written to the output directory, and the part after
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
//!     [hours=12.42] [rest_hours=1] [ramp_hours=1] [output_minutes=60] [wind] \
//!     [start=2025-06-15T00:00:00Z] [tides=data/froya_boundary_tides.txt] [norkyst=<file>] \
//!     [gauges=data/tide_gauges/mausund_obs.txt] [station_atlas=data/froya_station_tides.txt] \
//!     [station_minutes=10] [spinup_hours=24] [gauge_ratios=N2,Q1] [land_elevation=5] \
//!     [output=output/froya]
//! ```
//!
//! Harmonic validation needs the record after spin-up to resolve the main
//! constituents: 15 days separate M2/S2 and K1/O1 (`hours=384` with the
//! default spin-up); N2 needs 28.
//!
//! ## Data files in ./data/
//!
//! - froya_smola_hitra.tif (bathymetry)
//! - GSHHS_f_L1.shp (coastline)
//! - froya_boundary_tides.txt (tidal atlas, optional)
//! - tide_gauges/mausund_obs.txt, froya_station_tides.txt (validation, optional)

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use dg_rs::analysis::{
    ConstituentComparison, Inference, ReferenceConstant, ReferenceFit, StationValidationResult,
    TideGaugeStation, TimeSeries, fit_reference_constants, resolvable_constituents,
};
#[cfg(feature = "netcdf")]
use dg_rs::boundary::OceanModelState;
use dg_rs::boundary::{
    BoundaryTides, CharacteristicOBC, ExternalStateProvider, HarmonicTide, InverseBarometer,
    MultiBoundaryCondition2D, Reflective2D, SWEBoundaryCondition2D, TidalAtlas,
};
use dg_rs::equations::ShallowWater2D;
use dg_rs::io::{
    CoastlineData, CoordinateProjection, GeoBoundingBox, GeoTiffBathymetry, LocalProjection,
    TideGaugeFile, read_tide_gauge_file, write_tide_gauge_file, write_vtk_swe,
};
#[cfg(feature = "netcdf")]
use dg_rs::io::{NetCDFMeshInfo, NetCDFWriter, NetCDFWriterConfig, OceanModelReader};
use dg_rs::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::{PhysicsBuilder, PhysicsModule, SWEPhysics2D, SWEPhysics2DBuilder};
use dg_rs::simulation::Simulation;
use dg_rs::solver::{SWESolution2D, SWEState2D, StandardLimiter2D, WetDryConfig};
use dg_rs::source::{
    AtmosphericPressure2D, CoriolisSource2D, DragCoefficient, ManningFriction2D, WindStress2D,
};
use dg_rs::tides::canonical_name;
use dg_rs::time::{ModelClock, SSPRK3};
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

/// A station samples the nearest node at least this deep (m below MSL), so
/// it stays wet through the tide
const STATION_MIN_DEPTH: f64 = 3.0;
/// ... and no farther than this from the gauge (m)
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
    wind: bool,
    start: String,
    tides: String,
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    norkyst: Option<String>,
    gauges: Vec<String>,
    station_atlas: String,
    station_minutes: f64,
    spinup_hours: f64,
    gauge_ratios: Vec<&'static str>,
    land_elevation: f64,
    profile: usize,
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
        Ok(Self {
            nx: get("nx", 120.0)? as usize,
            ny: get("ny", 90.0)? as usize,
            order: get("order", 2.0)? as usize,
            hours: get("hours", M2_PERIOD / 3600.0)?,
            rest_hours: get("rest_hours", 1.0)?,
            ramp_hours: get("ramp_hours", TIDAL_RAMP_HOURS)?,
            output_minutes: get("output_minutes", 60.0)?,
            land_elevation: get("land_elevation", LAND_ELEVATION)?,
            profile: get("profile", 0.0)? as usize,
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
            gauges: args
                .get("gauges")
                .map_or("data/tide_gauges/mausund_obs.txt", String::as_str)
                .split(',')
                .filter(|g| !g.is_empty())
                .map(String::from)
                .collect(),
            station_atlas: args
                .get("station_atlas")
                .cloned()
                .unwrap_or("data/froya_station_tides.txt".into()),
            station_minutes: get("station_minutes", 10.0)?,
            spinup_hours: get("spinup_hours", 24.0)?,
            gauge_ratios: args
                .get("gauge_ratios")
                .map_or("N2,Q1", String::as_str)
                .split(',')
                .filter(|n| !n.is_empty())
                .map(|n| canonical_name(n).ok_or(format!("unknown constituent {n}")))
                .collect::<Result<_, _>>()?,
        })
    }
}

/// Surface range (and where the highest wet node is: x, y, h), largest speed
/// where h > 10 cm (and where: x, y, h), and the number of wet nodes
/// (h > 1 mm).
struct Stats {
    eta: (f64, f64),
    highest: (f64, f64, f64),
    speed: f64,
    fastest: (f64, f64, f64),
    wet: usize,
}

/// A tide gauge, the node that samples it, and the sampled series.
struct Station {
    station: TideGaugeStation,
    /// Observed record (finite samples, Unix times) and its reference
    /// constants
    observed: TimeSeries,
    observed_fit: Result<ReferenceFit, String>,
    element: ElementIndex,
    node: usize,
    /// Distance from the gauge to the node (m)
    offset: f64,
    /// Still-water depth at the node (m)
    depth: f64,
    /// Unix times and surface elevation
    times: Vec<f64>,
    eta: Vec<f64>,
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
    /// Grid `nx` × `ny` over `[x0, x1] × [y0, y1]` with open sides; keep
    /// elements with a water node, where `bed(x, y)` is `Some(B)`.
    fn build(
        name: &'static str,
        (x0, x1, y0, y1): (f64, f64, f64, f64),
        opts: &Options,
        bed: impl Fn(f64, f64) -> Option<f64>,
        projection: Option<LocalProjection>,
    ) -> Self {
        // The sides are open sea wherever they cross water (land is not meshed)
        let grid =
            Mesh2D::uniform_rectangle_with_bc(x0, x1, y0, y1, opts.nx, opts.ny, BoundaryTag::Open);

        let ops = DGOperators2D::new(opts.order);
        let n = ops.n_nodes;
        let beds: Vec<Option<f64>> = ElementIndex::iter(grid.n_elements)
            .flat_map(|k| {
                let grid = &grid;
                let ops = &ops;
                let bed = &bed;
                (0..n).map(move |i| {
                    let [x, y] = grid.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                    bed(x, y)
                })
            })
            .collect();
        let has_water = |k: ElementIndex| beds[k.as_usize() * n..][..n].iter().any(Option::is_some);
        let (mesh, old) = grid.retain_elements(has_water, BoundaryTag::Wall);

        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let mut bathymetry = Bathymetry2D::flat(mesh.n_elements, n);
        for (k, &o) in old.iter().enumerate() {
            for i in 0..n {
                let b = beds[o * n + i].unwrap_or(opts.land_elevation);
                bathymetry.set(ElementIndex::new(k), i, b);
            }
        }
        bathymetry.compute_gradients(&ops, &geom);

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
        let bathy_path = Path::new("data/froya_smola_hitra.tif");
        let coast_path = Path::new("data/GSHHS_f_L1.shp");
        if !bathy_path.exists() || !coast_path.exists() {
            return Ok(None);
        }
        let bbox = GeoBoundingBox::new(8.0, 63.6, 9.2, 64.0);
        let (lat0, lon0) = bbox.center();
        let projection = LocalProjection::new(lat0, lon0);
        let geotiff = GeoTiffBathymetry::load(bathy_path)?;
        let coastline = CoastlineData::load(coast_path, &bbox)?;
        println!("  Bathymetry: {}", geotiff.statistics());
        println!(
            "  Coastline: {} polygons",
            coastline.statistics().polygon_count
        );

        let (x0, y0) = projection.geo_to_xy(bbox.min_lat, bbox.min_lon);
        let (x1, y1) = projection.geo_to_xy(bbox.max_lat, bbox.max_lon);
        let bed = |x: f64, y: f64| {
            let (lat, lon) = projection.xy_to_geo(x, y);
            coastline
                .is_water(lat, lon)
                .then(|| geotiff.get_depth_bilinear(lat, lon))
                .flatten()
                .filter(|&b| b < 0.0)
        };
        Ok(Some(Self::build(
            "froya",
            (x0, x1, y0, y1),
            opts,
            bed,
            Some(projection),
        )))
    }

    /// 50 × 40 km basin shoaling to the east, with a beach along the east
    /// side and a round island.
    fn synthetic(opts: &Options) -> Self {
        let (lx, ly) = (50_000.0, 40_000.0);
        let bed = |x: f64, y: f64| {
            let island = (x - 0.4 * lx).hypot(y - 0.5 * ly) < 4_000.0;
            let b = -100.0 + 105.0 * (x / lx).powi(2);
            (!island && b < 0.0).then_some(b)
        };
        Self::build("synthetic", (0.0, lx, 0.0, ly), opts, bed, None)
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

    /// Stations for the gauge files that lie in the domain: each samples the
    /// nearest node at least `STATION_MIN_DEPTH` deep.
    fn stations(&self, paths: &[String]) -> Vec<Station> {
        let Some(projection) = &self.projection else {
            return Vec::new();
        };
        let mut stations = Vec::new();
        for path in paths {
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
            let (gx, gy) = projection.geo_to_xy(station.latitude, station.longitude);
            let nearest = ElementIndex::iter(self.mesh.n_elements)
                .flat_map(|k| (0..self.ops.n_nodes).map(move |i| (k, i)))
                .filter(|&(k, i)| self.bathymetry.get(k, i) <= -STATION_MIN_DEPTH)
                .map(|(k, i)| {
                    let [x, y] = self.mesh.reference_to_physical(
                        k,
                        self.ops.nodes_r[i],
                        self.ops.nodes_s[i],
                    );
                    ((x - gx).hypot(y - gy), k, i)
                })
                .min_by(|a, b| a.0.total_cmp(&b.0));
            match nearest {
                Some((offset, element, node)) if offset <= STATION_MAX_OFFSET => {
                    let depth = -self.bathymetry.get(element, node);
                    let (times, values): (Vec<f64>, Vec<f64>) = gauge
                        .time_series
                        .times()
                        .into_iter()
                        .zip(gauge.time_series.values())
                        .filter(|(_, v)| v.is_finite())
                        .unzip();
                    let observed_fit = fit_record(&times, &values, &GAUGE_CONSTITUENTS, None);
                    println!(
                        "  Station {}: node {offset:.0} m from the gauge, {depth:.1} m deep; \
                         gauge record {:.0} days",
                        station.name,
                        times
                            .last()
                            .zip(times.first())
                            .map_or(0.0, |(b, a)| (b - a) / 86_400.0)
                    );
                    stations.push(Station {
                        station,
                        observed: TimeSeries::new(&times, &values),
                        observed_fit,
                        element,
                        node,
                        offset,
                        depth,
                        times: Vec::new(),
                        eta: Vec::new(),
                    });
                }
                _ => println!("  Gauge {}: outside the domain (skipped)", station.name),
            }
        }
        stations
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
                if s.h > 0.1 && speed > stats.speed {
                    let [x, y] = self.mesh.reference_to_physical(
                        k,
                        self.ops.nodes_r[i],
                        self.ops.nodes_s[i],
                    );
                    stats.speed = speed;
                    stats.fastest = (x, y, s.h);
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
    let domain = match Domain::froya(&opts)? {
        Some(domain) => domain,
        None => {
            println!("  Data files not found in ./data/: synthetic basin instead");
            Domain::synthetic(&opts)
        }
    };
    domain.print_summary();

    lake_at_rest(&domain, opts.rest_hours);
    tidal_run(&domain, &opts)
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

fn tidal_run(domain: &Domain, opts: &Options) -> Result<(), Box<dyn std::error::Error>> {
    let t_end = opts.hours * 3600.0;
    let wall = Reflective2D::new();
    let mut stations = domain.stations(&opts.gauges);
    let (open, clock, forcing) = open_boundary(domain, opts, t_end, &stations)?;
    println!("  Clock: t = 0 at {} UTC", clock.format(0.0));
    let bc = MultiBoundaryCondition2D::new(&wall).with_open(open.as_ref());

    let mut builder = domain.builder(bc);
    if opts.wind {
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
        if opts.wind { ", wind and pressure" } else { "" },
        output_dir.display()
    );
    println!(
        "  time  |  η range (m)     | η max at (x, y km; h m) | max |u| (m/s) at (x, y km; h m) | wet nodes | volume change"
    );

    let mut q = domain.at_rest();
    let volume0 = domain.volume(&q);
    // Callbacks sample the stations; every `output_every`-th also writes output
    let (interval, output_every) = if stations.is_empty() {
        (opts.output_minutes * 60.0, 1)
    } else {
        let every = (opts.output_minutes / opts.station_minutes)
            .round()
            .max(1.0);
        (opts.station_minutes * 60.0, every as usize)
    };
    let mut n_callbacks = 0;
    let mut frame = 0;
    let mut write_error = None;
    let sim = Simulation::new(physics, SSPRK3)
        .with_cfl(1.0)
        .with_callback_interval(interval);
    let start = Instant::now();
    let result = sim.run_with_callback(&mut q, 0.0, t_end, |q, t| {
        for s in &mut stations {
            let state = q.get_state(s.element, s.node);
            s.times.push(clock.unix(t));
            s.eta.push(state.h + domain.bathymetry.get(s.element, s.node));
        }
        n_callbacks += 1;
        if (n_callbacks - 1) % output_every != 0 {
            return;
        }
        let stats = domain.stats(q);
        let (x, y, h) = stats.fastest;
        println!(
            "{:6.2} h | [{:+.3}, {:+.3}] | ({:6.1}, {:6.1}; {:.3}) | {:5.2} at ({:6.1}, {:6.1}; {h:6.1}) | {:9} | {:+.3e}",
            t / 3600.0,
            stats.eta.0,
            stats.eta.1,
            stats.highest.0 / 1e3,
            stats.highest.1 / 1e3,
            stats.highest.2,
            stats.speed,
            x / 1e3,
            y / 1e3,
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
    });
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
        sim.physics().negative_depth_clips()
    );
    let station_atlas = Path::new(&opts.station_atlas);
    let station_atlas = station_atlas
        .exists()
        .then(|| TidalAtlas::read(station_atlas))
        .transpose()?;
    for s in &stations {
        let path = output_dir.join(format!("station_{}.txt", slug(&s.station.name)));
        let series = TimeSeries::new(&s.times, &s.eta).with_name(s.station.name.clone());
        let mut file = TideGaugeFile::from_time_series(series).with_station(s.station.clone());
        file.datum = Some("MSL".into());
        file.units = Some("m".into());
        write_tide_gauge_file(&path, &file)?;
        println!("\nStation {} → {}", s.station.name, path.display());
        if let Err(e) = report_station(
            s,
            clock.unix(opts.spinup_hours * 3600.0),
            station_atlas.as_ref(),
        ) {
            println!("  no harmonic comparison: {e}");
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
            .compute_dt(&q, physics.max_cfl().unwrap_or(1.0))
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
    let dt = physics.compute_dt(&q, physics.max_cfl().unwrap_or(1.0));

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
            std::hint::black_box(physics.compute_dt(&q, 1.0));
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

/// The model's reference constants at a station after spin-up (from
/// `analysis_start`, Unix s) against the gauge's (whole record) and the
/// station atlas's; RMSE against the observations and the gauge's tidal
/// prediction.
fn report_station(
    s: &Station,
    analysis_start: f64,
    atlas: Option<&TidalAtlas>,
) -> Result<(), String> {
    let first = s.times.partition_point(|&t| t < analysis_start);
    let (times, eta) = (&s.times[first..], &s.eta[first..]);
    let gauge_times = s.observed.times();
    let gauge = s
        .observed_fit
        .as_ref()
        .map_err(|e| format!("gauge fit: {e}"))?;
    let model = fit_record(times, eta, &STATION_CONSTITUENTS, Some(gauge))?;
    let norkyst = atlas
        .and_then(|a| a.nearest(s.station.longitude, s.station.latitude))
        .filter(|&(_, d)| d <= STATION_MAX_OFFSET);

    let days = |t: &[f64]| (t[t.len() - 1] - t[0]) / 86_400.0;
    println!(
        "  node {:.0} m from the gauge, {:.1} m deep; model {:.1} days after spin-up (R² {:.4}), \
         gauge {:.1} days (R² {:.4})",
        s.offset,
        s.depth,
        days(times),
        model.r_squared,
        days(&gauge_times),
        gauge.r_squared
    );
    if let Some((p, d)) = norkyst {
        println!(
            "  NorKyst: atlas point {d:.0} m from the gauge, {:.0} m deep",
            p.depth
        );
    }
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
        if let Some(n) = norkyst.and_then(|(p, _)| p.constituents.iter().find(|n| n.name == c.name))
        {
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
        ("observations", &s.observed),
        ("gauge tidal prediction", &prediction),
    ] {
        let v = StationValidationResult::compute(&s.station, &model_series, reference);
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

/// The atmospheric pressure gradient of the `wind` option.
fn pressure() -> AtmosphericPressure2D {
    AtmosphericPressure2D::from_direction(PRESSURE_GRADIENT, PRESSURE_DIRECTION)
}

/// Characteristic OBC with external data from `provider`, raised by the
/// inverse-barometer level of the pressure forcing when `wind` is on.
fn characteristic<P: ExternalStateProvider + 'static>(
    provider: P,
    wind: bool,
) -> Box<dyn SWEBoundaryCondition2D> {
    if wind {
        let p = pressure();
        let level = move |x, y, t| p.inverse_barometer(x, y, t);
        Box::new(CharacteristicOBC::new(InverseBarometer::new(
            provider, level,
        )))
    } else {
        Box::new(CharacteristicOBC::new(provider))
    }
}

type OpenBoundary = (Box<dyn SWEBoundaryCondition2D>, ModelClock, String);

/// Open-boundary condition, model clock and a description of the forcing:
/// NorKyst nesting (`norkyst=`), else atlas tides (`tides=`), else uniform M2.
fn open_boundary(
    domain: &Domain,
    opts: &Options,
    t_end: f64,
    stations: &[Station],
) -> Result<OpenBoundary, Box<dyn std::error::Error>> {
    #[cfg(feature = "netcdf")]
    if let (Some(path), Some(projection)) = (&opts.norkyst, &domain.projection) {
        let reader = Arc::new(OceanModelReader::from_file(Path::new(path))?);
        println!("  NorKyst: {}", reader.summary());
        let clock = OceanModelState::<LocalProjection>::first_snapshot(&reader);
        let parent = OceanModelState::new(reader, *projection, clock);
        parent.check_time_coverage(0.0, t_end)?;
        return Ok((
            characteristic(parent, opts.wind),
            clock,
            "NorKyst nesting".into(),
        ));
    }

    let clock = ModelClock::parse(&opts.start)?;
    let atlas_path = Path::new(&opts.tides);
    if let (Some(projection), true) = (&domain.projection, atlas_path.exists()) {
        let mut atlas = TidalAtlas::read(atlas_path)?;
        let reference = stations
            .iter()
            .find_map(|s| Some((&s.station.name, s.observed_fit.as_ref().ok()?)));
        if reference.is_none() && !opts.gauge_ratios.is_empty() {
            println!(
                "  No gauge fit: the atlas keeps its own {:?}",
                opts.gauge_ratios
            );
        }
        for &name in opts.gauge_ratios.iter().filter(|_| reference.is_some()) {
            let (Some(inference), Some((station, fit))) = (
                Inference::EQUILIBRIUM.iter().find(|i| i.name == name),
                reference,
            ) else {
                return Err(format!("gauge_ratios: {name} is not P1, K2, N2 or Q1").into());
            };
            let from = inference.from;
            let i = Inference::from_reference(name, from, fit)
                .ok_or(format!("gauge_ratios: {station} lacks {name} or {from}"))?;
            atlas.infer(name, from, i.amplitude_ratio, i.lag_offset_deg)?;
            println!(
                "  Atlas {name} = {:.3} × {from}, lag {:+.1}° (ratio at {station})",
                i.amplitude_ratio,
                (i.lag_offset_deg + 180.0).rem_euclid(360.0) - 180.0
            );
        }
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
        let description = describe(&tides, atlas_path);
        return Ok((characteristic(tides, opts.wind), clock, description));
    }

    println!("  No tidal atlas at {}: uniform M2", opts.tides);
    let tide = HarmonicTide::m2(M2_AMPLITUDE, 0.0)
        .with_ramp_up(3600.0 * opts.ramp_hours)
        .with_nodal_corrections(&clock, 0.5 * t_end);
    let description = format!("M2 of {M2_AMPLITUDE} m in one phase at the open boundaries");
    Ok((characteristic(tide, opts.wind), clock, description))
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
