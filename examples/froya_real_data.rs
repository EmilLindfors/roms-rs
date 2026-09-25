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
//!    `LAND_ELEVATION`, so `WetDry` treats them as dry shore. The sides of the
//!    rectangle are open where they cross water.
//! 2. **Lake at rest.** Walls everywhere, no forcing: the largest spurious
//!    current and surface error after `rest_hours` (exact balance keeps both
//!    at round-off; the collocated scheme reached m/s on steep beds).
//! 3. **Tides.** M2 at the open boundaries (`HarmonicFlather2D`, one phase
//!    along the whole boundary: see TODO P1.4), Coriolis, Manning friction,
//!    optionally wind and an atmospheric pressure gradient, or NorKyst nesting
//!    (`--features netcdf`, `norkyst=<file>`).
//!
//! Without the data files it runs a synthetic basin with an island and a
//! beach.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example froya_real_data -- [nx=120] [ny=90] [order=2] \
//!     [hours=12.42] [rest_hours=1] [output_minutes=60] [wind] [norkyst=<file>]
//! ```
//!
//! ## Data files in ./data/
//!
//! - froya_smola_hitra.tif (bathymetry)
//! - GSHHS_f_L1.shp (coastline)

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "netcdf")]
use dg_rs::boundary::OceanNestingBC2D;
use dg_rs::boundary::{
    HarmonicFlather2D, MultiBoundaryCondition2D, Reflective2D, SWEBoundaryCondition2D,
};
use dg_rs::equations::ShallowWater2D;
use dg_rs::io::{
    CoastlineData, CoordinateProjection, GeoBoundingBox, GeoTiffBathymetry, LocalProjection,
    write_vtk_swe,
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
use dg_rs::time::SSPRK3;
#[cfg(feature = "netcdf")]
use dg_rs::types::Depth;
use dg_rs::types::ElementIndex;

/// Gravitational acceleration (m/s²)
const G: f64 = 9.81;
/// Coriolis parameter at 63.8°N (s⁻¹)
const F_CORIOLIS: f64 = 1.31e-4;
/// Manning roughness (s/m^{1/3})
const MANNING_N: f64 = 0.025;
/// Bed elevation given to land nodes of shoreline elements (m above MSL)
const LAND_ELEVATION: f64 = 5.0;

/// M2 period (s) and a typical amplitude on this coast (m)
const M2_PERIOD: f64 = 12.420_601 * 3600.0;
const M2_AMPLITUDE: f64 = 0.8;
/// Tidal ramp-up (s)
const TIDAL_RAMP: f64 = 3600.0;

/// Wind (m/s, from °) and atmospheric pressure gradient (Pa/m, from °)
const WIND_SPEED: f64 = 8.0;
const WIND_DIRECTION: f64 = 225.0;
const PRESSURE_GRADIENT: f64 = 1.5e-3;
const PRESSURE_DIRECTION: f64 = 225.0;

/// Command-line options (`key=value`, or a bare flag)
struct Options {
    nx: usize,
    ny: usize,
    order: usize,
    hours: f64,
    rest_hours: f64,
    output_minutes: f64,
    wind: bool,
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    norkyst: Option<String>,
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
            output_minutes: get("output_minutes", 60.0)?,
            wind: args.contains_key("wind"),
            norkyst: args.get("norkyst").cloned(),
        })
    }
}

/// Surface range, largest speed where h > 10 cm (and where: x, y, h), and
/// the number of wet nodes (h > 1 mm).
struct Stats {
    eta: (f64, f64),
    speed: f64,
    fastest: (f64, f64, f64),
    wet: usize,
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

        let geom = GeometricFactors2D::compute(&mesh);
        let mut bathymetry = Bathymetry2D::flat(mesh.n_elements, n);
        for (k, &o) in old.iter().enumerate() {
            for i in 0..n {
                let b = beds[o * n + i].unwrap_or(LAND_ELEVATION);
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

    fn volume(&self, q: &SWESolution2D) -> f64 {
        ElementIndex::iter(self.mesh.n_elements)
            .map(|k| {
                let j = self.geom.det_j[k.as_usize()];
                (0..self.ops.n_nodes)
                    .map(|i| self.ops.weights[i] * j * q.get_state(k, i).h)
                    .sum::<f64>()
            })
            .sum()
    }

    fn stats(&self, q: &SWESolution2D) -> Stats {
        let mut stats = Stats {
            eta: (f64::MAX, f64::MIN),
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
    let tide = HarmonicFlather2D::m2_only(M2_AMPLITUDE, 0.0, 0.0).with_ramp_up(TIDAL_RAMP);

    #[cfg(feature = "netcdf")]
    let nesting = match (&opts.norkyst, &domain.projection) {
        (Some(path), Some(projection)) => {
            let reader = Arc::new(OceanModelReader::from_file(Path::new(path))?);
            println!("  NorKyst: {}", reader.summary());
            let bc = OceanNestingBC2D::new(reader, *projection).with_reference_level(0.0);
            bc.check_time_coverage(0.0, t_end)?;
            Some(bc)
        }
        _ => None,
    };
    #[cfg(feature = "netcdf")]
    let bc = match &nesting {
        Some(nesting) => MultiBoundaryCondition2D::new(&wall).with_open(nesting),
        None => MultiBoundaryCondition2D::new(&wall).with_open(&tide),
    };
    #[cfg(not(feature = "netcdf"))]
    let bc = MultiBoundaryCondition2D::new(&wall).with_open(&tide);

    let mut builder = domain.builder(bc);
    if opts.wind {
        builder = builder
            .with_source(
                WindStress2D::from_direction(WIND_SPEED, WIND_DIRECTION)
                    .with_drag(DragCoefficient::LargePond),
            )
            .with_source(AtmosphericPressure2D::from_direction(
                PRESSURE_GRADIENT,
                PRESSURE_DIRECTION,
            ));
    }
    let physics: SWEPhysics2D<_> = builder.build();

    let output_dir = Path::new("output").join(domain.name);
    fs::create_dir_all(&output_dir)?;
    #[cfg(feature = "netcdf")]
    let mut netcdf = match &domain.projection {
        Some(projection) => Some(NetCDFWriter::create(
            NetCDFWriterConfig::new(output_dir.join("froya.nc").to_string_lossy())
                .with_title("Frøya–Smøla–Hitra tidal run")
                .with_institution("dg-rs"),
            &netcdf_mesh_info(domain, projection),
        )?),
        None => None,
    };

    println!(
        "\nTides: M2 {M2_AMPLITUDE} m at the open boundaries, {:.2} h{} → {}",
        opts.hours,
        if opts.wind { ", wind and pressure" } else { "" },
        output_dir.display()
    );
    println!(
        "  time  |  η range (m)     | max |u| (m/s) at (x, y km; h m) | wet nodes | volume change"
    );

    let mut q = domain.at_rest();
    let volume0 = domain.volume(&q);
    let interval = opts.output_minutes * 60.0;
    let mut frame = 0;
    let mut write_error = None;
    let sim = Simulation::new(physics, SSPRK3)
        .with_cfl(1.0)
        .with_callback_interval(interval);
    let start = Instant::now();
    let result = sim.run_with_callback(&mut q, 0.0, t_end, |q, t| {
        let stats = domain.stats(q);
        let (x, y, h) = stats.fastest;
        println!(
            "{:6.2} h | [{:+.3}, {:+.3}] | {:5.2} at ({:6.1}, {:6.1}; {h:6.1}) | {:9} | {:+.3e}",
            t / 3600.0,
            stats.eta.0,
            stats.eta.1,
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
    if let Some(e) = result.error {
        return Err(e.into());
    }
    println!("Visualize with ParaView: {}/*.vtu", output_dir.display());
    Ok(())
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
