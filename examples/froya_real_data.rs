//! Froya-Smola-Hitra Real Data Simulation
//!
//! Demonstrates loading real bathymetry and coastline data for shallow water
//! simulation on the Norwegian coast.
//!
//! ## Features
//!
//! - Real bathymetry from GeoTIFF file
//! - Coastline-based land masking from GSHHS shapefile
//! - DG discretization with configurable polynomial order
//! - SSP-RK3 time integration
//! - VTK output for visualization
//!
//! ## Known Limitations
//!
//! For production-quality long-duration simulations, enable:
//! - Kuzmin or TVB limiters (see `norwegian_fjord.rs` example)
//! - Lower CFL numbers for complex bathymetry
//! - Well-balanced schemes for steep bathymetry
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example froya_real_data
//! ```
//!
//! ## Required data files in ./data/
//!
//! - froya_smola_hitra.tif (bathymetry)
//! - GSHHS_f_L1.shp (coastline)

use std::fs;
use std::path::Path;

use dg_rs::boundary::{HarmonicFlather2D, MultiBoundaryCondition2D, Reflective2D};
#[cfg(feature = "netcdf")]
use dg_rs::boundary::OceanNestingBC2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::io::{
    CoastlineData, CoordinateProjection, GeoBoundingBox, GeoTiffBathymetry, LocalProjection,
    write_vtk_swe,
};
#[cfg(feature = "netcdf")]
use dg_rs::io::{NetCDFMeshInfo, NetCDFWriter, NetCDFWriterConfig, OceanModelReader};
#[cfg(feature = "netcdf")]
use std::sync::Arc;
use dg_rs::mesh::{Bathymetry2D, BoundaryTag, LandMask2D, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::solver::{SWE2DRhsConfig, SWESolution2D, compute_dt_swe_2d, compute_rhs_swe_2d};
#[cfg(feature = "simd")]
use dg_rs::solver::compute_rhs_swe_2d_simd;
use dg_rs::source::{
    AtmosphericPressure2D, BathymetrySource2D, CombinedSource2D, CoriolisSource2D,
    DragCoefficient, ManningFriction2D, WindStress2D,
};
use dg_rs::time::{SWE2DTimeConfig, ssp_rk3_swe_2d_step_limited};
use dg_rs::{SWEFluxType2D, SWEState2D};

// ============================================================================
// Physical Parameters
// ============================================================================

/// Gravitational acceleration (m/s²)
const G: f64 = 9.81;

/// Minimum depth for wetting/drying (m)
/// Note: Higher value needed for stability without limiters
const H_MIN: f64 = 5.0;

/// Coriolis parameter at 64°N (s⁻¹)
/// f = 2 * Omega * sin(lat), where Omega = 7.292e-5 rad/s
const F_CORIOLIS: f64 = 1.26e-4;

/// Manning roughness coefficient (s/m^{1/3})
const MANNING_N: f64 = 0.025;

// ============================================================================
// Tidal Parameters
// ============================================================================

/// M2 tidal period (s) - approximately 12.42 hours
const M2_PERIOD: f64 = 12.42 * 3600.0;

/// M2 tidal amplitude (m) - typical for Norwegian coast
const M2_AMPLITUDE: f64 = 0.8;

/// Tidal forcing ramp-up duration (s) - 1 hour for smooth spin-up
const TIDAL_RAMP_DURATION: f64 = 3600.0;

// ============================================================================
// Wind Parameters
// ============================================================================

/// Wind speed (m/s) - moderate southwesterly
const WIND_SPEED: f64 = 8.0;

/// Wind direction (degrees, meteorological convention: 0=N, 90=E, 180=S, 270=W)
/// 225 = from southwest (typical Norwegian coast winter wind)
const WIND_DIRECTION: f64 = 225.0;

/// Enable wind forcing
const ENABLE_WIND: bool = true;

// ============================================================================
// Atmospheric Pressure Parameters
// ============================================================================

/// Enable atmospheric pressure forcing
const ENABLE_PRESSURE: bool = true;

/// Pressure gradient magnitude (Pa/m)
/// 1.0e-3 Pa/m = 1 hPa per 100 km (moderate gradient)
/// 2.0e-3 Pa/m = 2 hPa per 100 km (strong storm)
const PRESSURE_GRADIENT: f64 = 1.5e-3;

/// Pressure gradient direction (degrees, direction gradient is FROM)
/// 225 = from southwest (typical Norwegian storm)
const PRESSURE_DIRECTION: f64 = 225.0;

// ============================================================================
// Domain Configuration
// ============================================================================

/// Polynomial order for DG basis (lower order is more stable without limiters)
const N_POLY: usize = 2;

/// Number of elements in x-direction
const NX: usize = 30;

/// Number of elements in y-direction
const NY: usize = 20;

// ============================================================================
// Simulation Configuration
// ============================================================================

/// Simulation duration (s) - 30 minutes for quick testing
const T_END: f64 = 0.5 * 3600.0;

/// Output interval (s) - output every 5 minutes
const OUTPUT_INTERVAL: f64 = 300.0;

/// Enable NetCDF output (requires --features netcdf)
#[cfg(feature = "netcdf")]
const ENABLE_NETCDF: bool = true;

/// Enable ocean model nesting (requires --features netcdf and NorKyst data file)
#[cfg(feature = "netcdf")]
const ENABLE_OCEAN_NESTING: bool = true;

/// Path to NorKyst v3 data file for ocean nesting
#[cfg(feature = "netcdf")]
const NORKYST_PATH: &str = "/home/emil/aqc/aqc-h3o/data/norkyst_v3/norkyst_v3_norkystv3_800m_m00_be_20240130_0600_stuv_bbox_7p8_62p8_8p9_64p1.nc";

/// CFL number for time stepping
const CFL: f64 = 0.1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=================================================================");
    println!("  Froya-Smola-Hitra Real Data Simulation");
    println!("=================================================================");
    println!();

    // ========================================================================
    // Define geographic domain
    // ========================================================================
    let geo_bbox = GeoBoundingBox::new(
        8.0,   // min_lon (western edge)
        63.6,  // min_lat (southern edge)
        9.2,   // max_lon (eastern edge)
        64.0,  // max_lat (northern edge)
    );

    println!("Geographic domain:");
    println!("  Longitude: [{:.2}°E, {:.2}°E]", geo_bbox.min_lon, geo_bbox.max_lon);
    println!("  Latitude:  [{:.2}°N, {:.2}°N]", geo_bbox.min_lat, geo_bbox.max_lat);

    // Create projection centered on domain
    let (center_lat, center_lon) = geo_bbox.center();
    let projection = LocalProjection::new(center_lat, center_lon);

    println!("\nProjection:");
    println!("  Reference: ({:.4}°N, {:.4}°E)", projection.ref_lat(), projection.ref_lon());

    // ========================================================================
    // Load geographic data
    // ========================================================================
    println!("\nLoading data files...");

    let bathy_path = Path::new("data/froya_smola_hitra.tif");
    let coastline_path = Path::new("data/GSHHS_f_L1.shp");

    // Check if files exist
    if !bathy_path.exists() || !coastline_path.exists() {
        println!("Warning: Data files not found.");
        println!("  Bathymetry: {} (exists: {})", bathy_path.display(), bathy_path.exists());
        println!("  Coastline: {} (exists: {})", coastline_path.display(), coastline_path.exists());
        println!("\nRunning with synthetic data instead.\n");
        return run_synthetic_simulation();
    }

    // Load GeoTIFF bathymetry with bbox hint
    let bathy_bbox = GeoBoundingBox::new(7.5, 63.3, 10.0, 64.2);
    let geotiff = GeoTiffBathymetry::load_with_bbox(bathy_path, Some(bathy_bbox))?;
    let bathy_stats = geotiff.statistics();
    println!("  Bathymetry: {}x{} pixels, depth range [{:.0}, {:.0}] m",
        bathy_stats.width, bathy_stats.height, bathy_stats.min_depth, bathy_stats.max_depth);

    // Load coastline data
    let coastline = CoastlineData::load(coastline_path, &geo_bbox)?;
    let coast_stats = coastline.statistics();
    println!("  Coastline: {} polygons, {} vertices",
        coast_stats.polygon_count, coast_stats.total_vertices);

    // ========================================================================
    // Create mesh
    // ========================================================================
    println!("\nSetting up mesh...");

    let (x_min, y_min) = projection.geo_to_xy(geo_bbox.min_lat, geo_bbox.min_lon);
    let (x_max, y_max) = projection.geo_to_xy(geo_bbox.max_lat, geo_bbox.max_lon);

    let mut mesh = Mesh2D::uniform_rectangle_with_bc(
        x_min, x_max, y_min, y_max,
        NX, NY,
        BoundaryTag::Wall,  // Default to wall
    );

    // Tag open boundaries (west and north edges = ocean)
    tag_boundaries(&mut mesh, x_min, y_min, x_max, y_max);

    println!("  Elements: {} ({}x{})", mesh.n_elements, NX, NY);
    println!("  Domain: {:.1} km x {:.1} km",
        (x_max - x_min) / 1000.0, (y_max - y_min) / 1000.0);

    let dx = (x_max - x_min) / NX as f64;
    let dy = (y_max - y_min) / NY as f64;
    println!("  Resolution: {:.0} m x {:.0} m", dx, dy);

    // ========================================================================
    // Set up DG operators
    // ========================================================================
    let ops = DGOperators2D::new(N_POLY);
    let geom = GeometricFactors2D::compute(&mesh);

    println!("  Polynomial order: {}", N_POLY);
    println!("  Nodes per element: {}", ops.n_nodes);
    println!("  Total DOFs: {}", mesh.n_elements * ops.n_nodes * 3);

    // ========================================================================
    // Initialize bathymetry from GeoTIFF
    // ========================================================================
    println!("\nSetting up bathymetry...");

    // Use REAL bathymetry from GeoTIFF with well-balanced scheme
    let mut bathymetry = Bathymetry2D::from_geotiff(&mesh, &ops, &geom, &geotiff, &projection);
    println!("  Original range: [{:.0}, {:.0}] m", bathymetry.min(), bathymetry.max());
    println!("  Original max gradient: {:.4}", bathymetry.max_gradient_magnitude());

    // CRITICAL: Use cell-average bathymetry for well-balanced property
    // This makes bathymetry constant within each element (zero gradients).
    // The hydrostatic reconstruction handles bathymetry jumps at interfaces.
    bathymetry.to_cell_average();
    println!("  Cell-average range: [{:.0}, {:.0}] m", bathymetry.min(), bathymetry.max());
    println!("  Cell-average max gradient: {:.4} (should be 0)", bathymetry.max_gradient_magnitude());

    // ========================================================================
    // Create land mask
    // ========================================================================
    println!("\nCreating land mask...");

    let land_mask = LandMask2D::from_coastline_and_bathymetry(
        &mesh, &ops, &coastline, &geotiff, &projection, H_MIN
    );
    let mask_stats = land_mask.statistics();
    println!("  Wet elements: {} ({:.1}%)",
        mask_stats.wet_elements,
        100.0 * mask_stats.wet_elements as f64 / mask_stats.total_elements as f64);
    println!("  Dry elements: {} ({:.1}%)",
        mask_stats.dry_elements,
        100.0 * mask_stats.dry_elements as f64 / mask_stats.total_elements as f64);

    // ========================================================================
    // Initialize state
    // ========================================================================
    println!("\nSetting up initial conditions...");

    let mut state = create_initial_state(&mesh, &ops, &bathymetry, &land_mask);
    println!("  Initial depth range: [{:.1}, {:.1}] m",
        state.min_depth(), state.max_depth());

    // ========================================================================
    // Set up physics
    // ========================================================================
    println!("\nSetting up physics...");

    // Source terms - Coriolis and friction only
    // Note: BathymetrySource2D not needed with cell-average bathymetry
    // because gradients are zero. Bathymetry effects come from interface reconstruction.
    let coriolis = CoriolisSource2D::f_plane(F_CORIOLIS);
    let friction = ManningFriction2D::new(G, MANNING_N);
    let wind = WindStress2D::from_direction(WIND_SPEED, WIND_DIRECTION)
        .with_drag(DragCoefficient::LargePond);
    let pressure = AtmosphericPressure2D::from_direction(PRESSURE_GRADIENT, PRESSURE_DIRECTION);

    // Combine sources based on enabled physics
    let sources: CombinedSource2D = match (ENABLE_WIND, ENABLE_PRESSURE) {
        (true, true) => CombinedSource2D::new(vec![&coriolis, &friction, &wind, &pressure]),
        (true, false) => CombinedSource2D::new(vec![&coriolis, &friction, &wind]),
        (false, true) => CombinedSource2D::new(vec![&coriolis, &friction, &pressure]),
        (false, false) => CombinedSource2D::new(vec![&coriolis, &friction]),
    };

    println!("  Coriolis: f = {:.2e} s⁻¹ (64°N)", F_CORIOLIS);
    println!("  Manning: n = {}", MANNING_N);
    if ENABLE_WIND {
        println!("  Wind: {:.1} m/s from {:.0}° (Large-Pond drag)", WIND_SPEED, WIND_DIRECTION);
    } else {
        println!("  Wind: disabled");
    }
    if ENABLE_PRESSURE {
        println!("  Pressure: {:.1} hPa/100km from {:.0}°", PRESSURE_GRADIENT * 1e5, PRESSURE_DIRECTION);
    } else {
        println!("  Pressure: disabled");
    }

    // ========================================================================
    // Load ocean model for nesting (if enabled and available)
    // ========================================================================
    #[cfg(feature = "netcdf")]
    let ocean_reader: Option<Arc<OceanModelReader>> = if ENABLE_OCEAN_NESTING {
        let norkyst_path = Path::new(NORKYST_PATH);
        if norkyst_path.exists() {
            match OceanModelReader::from_file(norkyst_path) {
                Ok(reader) => {
                    println!("\nOcean model loaded:");
                    println!("  {}", reader.summary());
                    Some(Arc::new(reader))
                }
                Err(e) => {
                    println!("\nWarning: Failed to load ocean model: {}", e);
                    println!("  Falling back to tidal BC");
                    None
                }
            }
        } else {
            println!("\nWarning: Ocean model file not found: {}", norkyst_path.display());
            println!("  Falling back to tidal BC");
            None
        }
    } else {
        None
    };

    // Boundary conditions - ocean nesting or tidal forcing at open boundaries
    let wall_bc = Reflective2D::new();
    let tidal_bc = HarmonicFlather2D::m2_only(M2_AMPLITUDE, 0.0, 0.0)
        .with_ramp_up(TIDAL_RAMP_DURATION);

    // Create ocean nesting BC if reader is available
    #[cfg(feature = "netcdf")]
    let ocean_bc: Option<OceanNestingBC2D<LocalProjection>> = ocean_reader.as_ref().map(|reader| {
        OceanNestingBC2D::new(Arc::clone(reader), projection.clone())
            .with_reference_level(0.0)
            .with_flather(true)
            .with_flather_weight(0.8)  // Blend with some radiation
    });

    // Use ocean BC if available, otherwise fall back to tidal
    #[cfg(feature = "netcdf")]
    let bc = if let Some(ref obc) = ocean_bc {
        println!("  Boundary: Ocean model nesting (NorKyst v3) with Flather blending");
        MultiBoundaryCondition2D::new(&wall_bc).with_open(obc)
    } else {
        println!("  Boundary: M2 tidal forcing with {:.0} min ramp-up", TIDAL_RAMP_DURATION / 60.0);
        MultiBoundaryCondition2D::new(&wall_bc).with_open(&tidal_bc)
    };

    #[cfg(not(feature = "netcdf"))]
    let bc = {
        println!("  Boundary: M2 tidal forcing with {:.0} min ramp-up", TIDAL_RAMP_DURATION / 60.0);
        MultiBoundaryCondition2D::new(&wall_bc).with_open(&tidal_bc)
    };

    // ========================================================================
    // Time integration setup
    // ========================================================================
    let equation = ShallowWater2D::new(G);

    // Time configuration with Kuzmin limiters and improved wetting/drying
    let time_config = SWE2DTimeConfig::new(CFL, G, H_MIN)
        .with_kuzmin_limiters(1.0)     // Strict limiting for steep bathymetry
        .with_wet_dry_treatment();     // Improved wetting/drying (velocity cap, thin-layer damping)

    println!("\nSimulation parameters:");
    println!("  Duration: {:.1} hours ({:.2} M2 cycles)", T_END / 3600.0, T_END / M2_PERIOD);
    println!("  CFL: {}", CFL);
    println!("  Limiter: Kuzmin (vertex-based)");
    println!("  Output interval: {:.0} min", OUTPUT_INTERVAL / 60.0);

    // Create output directory
    let output_dir = Path::new("output/froya");
    fs::create_dir_all(output_dir)?;
    println!("  Output directory: {}", output_dir.display());

    // Create NetCDF writer (if feature enabled)
    #[cfg(feature = "netcdf")]
    let mut nc_writer = if ENABLE_NETCDF {
        let mesh_info = create_netcdf_mesh_info(&mesh, &ops, &projection);
        let nc_config = NetCDFWriterConfig::new(output_dir.join("froya.nc").to_string_lossy())
            .with_title("Froya-Smola-Hitra Simulation")
            .with_institution("dg-rs")
            .with_comment(format!(
                "M2 amplitude: {} m, Wind: {} m/s from {}°",
                M2_AMPLITUDE, WIND_SPEED, WIND_DIRECTION
            ));
        Some(NetCDFWriter::create(nc_config, &mesh_info)?)
    } else {
        None
    };
    #[cfg(feature = "netcdf")]
    println!("  NetCDF output: {}", output_dir.join("froya.nc").display());

    // ========================================================================
    // Run simulation
    // ========================================================================
    println!("\n=================================================================");
    println!("  Starting simulation...");
    println!("=================================================================\n");

    // Print TRUE initial state before any time stepping
    println!("Initial state (before time stepping):");
    print_diagnostics(0.0, &state, &land_mask);

    // Test lake-at-rest: compute RHS with no tidal forcing
    // Uses only Coriolis (zero for still water) - no friction, no bathymetry source
    {
        let wall_bc_test = Reflective2D::new();
        let bc_test = MultiBoundaryCondition2D::new(&wall_bc_test);
        let coriolis_only = CombinedSource2D::new(vec![&coriolis]);
        let config_test = SWE2DRhsConfig::new(&equation, &bc_test)
            .with_flux_type(SWEFluxType2D::Rusanov)
            .with_source_terms(&coriolis_only)
            .with_bathymetry(&bathymetry)
            .with_well_balanced(true);
        let rhs_test = compute_rhs_swe_2d(&state, &mesh, &ops, &geom, &config_test, 0.0);

        // Compute max RHS (should be ~0 for lake-at-rest)
        let mut max_rhs_h = 0.0_f64;
        let mut max_rhs_hu = 0.0_f64;
        let mut max_rhs_hv = 0.0_f64;
        for k in 0..state.n_elements {
            if !land_mask.is_wet(k) { continue; }
            for i in 0..state.n_nodes {
                let rhs_state = rhs_test.get_state(k, i);
                max_rhs_h = max_rhs_h.max(rhs_state.h.abs());
                max_rhs_hu = max_rhs_hu.max(rhs_state.hu.abs());
                max_rhs_hv = max_rhs_hv.max(rhs_state.hv.abs());
            }
        }
        println!("\nLake-at-rest test (reflective BC, well_balanced=true):");
        println!("  Max |RHS_h|:  {:.2e}", max_rhs_h);
        println!("  Max |RHS_hu|: {:.2e}", max_rhs_hu);
        println!("  Max |RHS_hv|: {:.2e}", max_rhs_hv);

        if max_rhs_hu > 1.0 || max_rhs_hv > 1.0 {
            println!("  WARNING: Large RHS indicates well-balanced scheme not fully effective");
        } else {
            println!("  OK: Lake-at-rest is well-balanced");
        }
    }
    println!();

    let mut t = 0.0;
    let mut step = 0;
    let mut last_output = -OUTPUT_INTERVAL;  // Force initial output
    let mut vtk_frame = 0;

    while t < T_END {
        // Compute time step
        let dt = compute_dt_swe_2d(&state, &mesh, &geom, &equation, N_POLY, CFL);
        let dt = dt.min(T_END - t);

        // RHS function with full physics and well-balanced scheme
        let rhs_fn = |s: &SWESolution2D, time: f64| {
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_flux_type(SWEFluxType2D::Rusanov)
                .with_source_terms(&sources)
                .with_bathymetry(&bathymetry)
                .with_well_balanced(true);  // Enable hydrostatic reconstruction
            #[cfg(feature = "simd")]
            { compute_rhs_swe_2d_simd(s, &mesh, &ops, &geom, &config, time) }
            #[cfg(not(feature = "simd"))]
            { compute_rhs_swe_2d(s, &mesh, &ops, &geom, &config, time) }
        };

        // SSP-RK3 step with limiters applied after each stage
        ssp_rk3_swe_2d_step_limited(&mut state, dt, t, &mesh, &ops, rhs_fn, &time_config);

        t += dt;
        step += 1;

        // Output
        if t - last_output >= OUTPUT_INTERVAL || t >= T_END - 1e-6 {
            print_diagnostics(t, &state, &land_mask);

            let vtk_path = output_dir.join(format!("froya_{:04}.vtu", vtk_frame));
            write_vtk_swe(&vtk_path, &mesh, &ops, &state, Some(&bathymetry), t, H_MIN)?;

            // Write NetCDF timestep
            #[cfg(feature = "netcdf")]
            if let Some(ref mut writer) = nc_writer {
                let (h_data, eta_data, u_data, v_data) = extract_solution_data(&state, &bathymetry);
                writer.write_timestep(t, &h_data, &eta_data, Some(&u_data), Some(&v_data))?;
            }

            vtk_frame += 1;
            last_output = t;
        }
    }

    // ========================================================================
    // Final summary
    // ========================================================================
    println!("\n=================================================================");
    println!("  Simulation complete!");
    println!("=================================================================");
    println!("  Final time: {:.2} hours", t / 3600.0);
    println!("  Total steps: {}", step);
    println!("  Average dt: {:.2} s", if step > 0 { t / step as f64 } else { 0.0 });
    println!("  VTK frames: {}", vtk_frame);

    print_final_diagnostics(&state, &land_mask);

    println!("\nVisualize with ParaView:");
    println!("  paraview output/froya/froya_*.vtu");

    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Tag domain boundaries as open (ocean) or wall.
fn tag_boundaries(mesh: &mut Mesh2D, x_min: f64, _y_min: f64, _x_max: f64, y_max: f64) {
    let tolerance = 100.0;  // meters

    for edge in mesh.edges.iter_mut() {
        if edge.is_boundary() {
            let (v0, v1) = edge.vertices;
            let (x0, y0) = mesh.vertices[v0];
            let (x1, y1) = mesh.vertices[v1];
            let x_mid = (x0 + x1) / 2.0;
            let y_mid = (y0 + y1) / 2.0;

            // West edge (x = x_min) and north edge (y = y_max) are open ocean
            let on_west = (x_mid - x_min).abs() < tolerance;
            let on_north = (y_mid - y_max).abs() < tolerance;

            if on_west || on_north {
                edge.boundary_tag = Some(BoundaryTag::Open);
            }
            // Other boundaries remain as Wall (default)
        }
    }
}

/// Create initial state with water at rest (constant η = h + B).
///
/// For lake-at-rest, water surface elevation η must be constant.
/// We set η = 0 (mean sea level) for all wet areas.
fn create_initial_state(
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    bathymetry: &Bathymetry2D,
    _land_mask: &LandMask2D,
) -> SWESolution2D {
    let mut swe = SWESolution2D::new(mesh.n_elements, ops.n_nodes);

    // Water surface at mean sea level (η = 0)
    let eta = 0.0;

    for k in 0..mesh.n_elements {
        for i in 0..ops.n_nodes {
            let b = bathymetry.get(k, i);

            // h = η - B for lake-at-rest
            // h must be positive, so dry cells get H_MIN
            let h = (eta - b).max(H_MIN);

            swe.set_state(k, i, SWEState2D::from_primitives(h, 0.0, 0.0));
        }
    }

    swe
}

/// Print simulation diagnostics.
fn print_diagnostics(t: f64, state: &SWESolution2D, land_mask: &LandMask2D) {
    let hours = t / 3600.0;
    let m2_phase = (t / M2_PERIOD) % 1.0;

    // Compute statistics only for wet cells
    let mut h_min = f64::INFINITY;
    let mut h_max = f64::NEG_INFINITY;
    let mut u_max = 0.0_f64;
    let mut v_max = 0.0_f64;

    for k in 0..state.n_elements {
        if !land_mask.is_wet(k) {
            continue;
        }

        for i in 0..state.n_nodes {
            let s = state.get_state(k, i);
            h_min = h_min.min(s.h);
            h_max = h_max.max(s.h);

            let (u, v) = s.velocity_simple(H_MIN);
            u_max = u_max.max(u.abs());
            v_max = v_max.max(v.abs());
        }
    }

    let speed = (u_max * u_max + v_max * v_max).sqrt();

    println!("t = {:5.1} h ({:.2} M2) | h: [{:6.1}, {:6.1}] m | |u|: {:.3} m/s",
        hours, m2_phase, h_min, h_max, speed);
}

/// Print final diagnostics.
fn print_final_diagnostics(state: &SWESolution2D, land_mask: &LandMask2D) {
    println!("\nFinal State Analysis (wet cells only):");

    let mut h_min = f64::INFINITY;
    let mut h_max = f64::NEG_INFINITY;
    let mut h_sum = 0.0;
    let mut u_max = 0.0_f64;
    let mut v_max = 0.0_f64;
    let mut count = 0;

    for k in 0..state.n_elements {
        if !land_mask.is_wet(k) {
            continue;
        }

        for i in 0..state.n_nodes {
            let s = state.get_state(k, i);
            h_min = h_min.min(s.h);
            h_max = h_max.max(s.h);
            h_sum += s.h;
            count += 1;

            let (u, v) = s.velocity_simple(H_MIN);
            u_max = u_max.max(u.abs());
            v_max = v_max.max(v.abs());
        }
    }

    let h_mean = if count > 0 { h_sum / count as f64 } else { 0.0 };
    let speed_max = (u_max * u_max + v_max * v_max).sqrt();

    println!("  Water depth: min={:.1} m, mean={:.1} m, max={:.1} m", h_min, h_mean, h_max);
    println!("  Max velocity: {:.3} m/s (u_max={:.3}, v_max={:.3})", speed_max, u_max, v_max);
}

/// Run with synthetic data when real files are not available.
fn run_synthetic_simulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("=================================================================");
    println!("  Running Synthetic Simulation");
    println!("=================================================================\n");

    // Create simple rectangular domain
    let x_min = 0.0;
    let x_max = 50_000.0;  // 50 km
    let y_min = 0.0;
    let y_max = 40_000.0;  // 40 km

    let mut mesh = Mesh2D::uniform_rectangle_with_bc(
        x_min, x_max, y_min, y_max,
        NX, NY,
        BoundaryTag::Wall,
    );

    // Tag west and north as open
    tag_boundaries(&mut mesh, x_min, y_min, x_max, y_max);

    let ops = DGOperators2D::new(N_POLY);
    let geom = GeometricFactors2D::compute(&mesh);

    // Synthetic bathymetry: deeper toward west (offshore)
    let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| {
        let t = x / x_max;
        -100.0 * (1.0 - 0.7 * t)  // -100m to -30m west to east
    });

    // All wet (no land mask)
    let land_mask = LandMask2D::all_wet(mesh.n_elements);

    // Initialize state
    let mut state = create_initial_state(&mesh, &ops, &bathymetry, &land_mask);

    // Physics
    let coriolis = CoriolisSource2D::f_plane(F_CORIOLIS);
    let friction = ManningFriction2D::new(G, MANNING_N);
    let bathy_source = BathymetrySource2D::new(G);
    let wind = WindStress2D::from_direction(WIND_SPEED, WIND_DIRECTION)
        .with_drag(DragCoefficient::LargePond);
    let pressure = AtmosphericPressure2D::from_direction(PRESSURE_GRADIENT, PRESSURE_DIRECTION);

    // Combine sources based on enabled physics
    let sources: CombinedSource2D = match (ENABLE_WIND, ENABLE_PRESSURE) {
        (true, true) => CombinedSource2D::new(vec![&coriolis, &friction, &bathy_source, &wind, &pressure]),
        (true, false) => CombinedSource2D::new(vec![&coriolis, &friction, &bathy_source, &wind]),
        (false, true) => CombinedSource2D::new(vec![&coriolis, &friction, &bathy_source, &pressure]),
        (false, false) => CombinedSource2D::new(vec![&coriolis, &friction, &bathy_source]),
    };

    // h_ref = 0 because bathymetry is negative below MSL
    // Add ramp-up period for smooth spin-up
    let tidal_bc = HarmonicFlather2D::m2_only(M2_AMPLITUDE, 0.0, 0.0)
        .with_ramp_up(TIDAL_RAMP_DURATION);
    let wall_bc = Reflective2D::new();
    let bc = MultiBoundaryCondition2D::new(&wall_bc).with_open(&tidal_bc);

    let equation = ShallowWater2D::new(G);

    // Time config with Kuzmin limiters and improved wetting/drying
    let time_config = SWE2DTimeConfig::new(CFL, G, H_MIN)
        .with_kuzmin_limiters(1.0)
        .with_wet_dry_treatment();

    // Create output directory
    let output_dir = Path::new("output/froya_synthetic");
    fs::create_dir_all(output_dir)?;

    println!("Domain: {:.0} km x {:.0} km", (x_max - x_min) / 1000.0, (y_max - y_min) / 1000.0);
    println!("Elements: {}", mesh.n_elements);
    println!("Bathymetry: [{:.0}, {:.0}] m", bathymetry.min(), bathymetry.max());
    println!("Limiter: Kuzmin (vertex-based)");
    println!("Output: {}", output_dir.display());
    println!();

    // Full tidal cycle simulation for synthetic case
    let t_end_synthetic = M2_PERIOD;  // 1 full M2 tidal cycle (~12.4 hours)

    let mut t = 0.0;
    let mut step = 0;
    let mut last_output = -OUTPUT_INTERVAL;
    let mut vtk_frame = 0;

    while t < t_end_synthetic {
        let dt = compute_dt_swe_2d(&state, &mesh, &geom, &equation, N_POLY, CFL);
        let dt = dt.min(t_end_synthetic - t);

        let rhs_fn = |s: &SWESolution2D, time: f64| {
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_flux_type(SWEFluxType2D::Rusanov)
                .with_source_terms(&sources)
                .with_bathymetry(&bathymetry)
                .with_well_balanced(true);  // Enable hydrostatic reconstruction
            #[cfg(feature = "simd")]
            { compute_rhs_swe_2d_simd(s, &mesh, &ops, &geom, &config, time) }
            #[cfg(not(feature = "simd"))]
            { compute_rhs_swe_2d(s, &mesh, &ops, &geom, &config, time) }
        };

        // SSP-RK3 step with limiters applied after each stage
        ssp_rk3_swe_2d_step_limited(&mut state, dt, t, &mesh, &ops, rhs_fn, &time_config);

        t += dt;
        step += 1;

        if t - last_output >= OUTPUT_INTERVAL / 2.0 || t >= t_end_synthetic - 1e-6 {
            print_diagnostics(t, &state, &land_mask);

            let vtk_path = output_dir.join(format!("synthetic_{:04}.vtu", vtk_frame));
            write_vtk_swe(&vtk_path, &mesh, &ops, &state, Some(&bathymetry), t, H_MIN)?;

            vtk_frame += 1;
            last_output = t;
        }
    }

    println!("\nSynthetic simulation complete!");
    println!("  Steps: {}, VTK frames: {}", step, vtk_frame);
    println!("\nTo use real data, place files in ./data/:");
    println!("  - froya_smola_hitra.tif");
    println!("  - GSHHS_f_L1.shp");

    Ok(())
}

/// Create mesh info for NetCDF output.
#[cfg(feature = "netcdf")]
fn create_netcdf_mesh_info(
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    projection: &LocalProjection,
) -> NetCDFMeshInfo {
    let n_total = mesh.n_elements * ops.n_nodes;
    let mut x_coords = Vec::with_capacity(n_total);
    let mut y_coords = Vec::with_capacity(n_total);
    let mut lat_coords = Vec::with_capacity(n_total);
    let mut lon_coords = Vec::with_capacity(n_total);

    for k in 0..mesh.n_elements {
        for i in 0..ops.n_nodes {
            let r = ops.nodes_r[i];
            let s = ops.nodes_s[i];
            let (x, y) = mesh.reference_to_physical(k, r, s);
            x_coords.push(x);
            y_coords.push(y);
            let (lat, lon) = projection.xy_to_geo(x, y);
            lat_coords.push(lat);
            lon_coords.push(lon);
        }
    }

    NetCDFMeshInfo::from_xy(x_coords, y_coords)
        .with_latlon(lat_coords, lon_coords)
}

/// Extract solution data for NetCDF output.
#[cfg(feature = "netcdf")]
fn extract_solution_data(
    state: &SWESolution2D,
    bathymetry: &Bathymetry2D,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_total = state.n_elements * state.n_nodes;
    let mut h_data = Vec::with_capacity(n_total);
    let mut eta_data = Vec::with_capacity(n_total);
    let mut u_data = Vec::with_capacity(n_total);
    let mut v_data = Vec::with_capacity(n_total);

    for k in 0..state.n_elements {
        for i in 0..state.n_nodes {
            let s = state.get_state(k, i);
            let b = bathymetry.get(k, i);

            h_data.push(s.h);
            eta_data.push(s.h + b);  // η = h + B

            let (u, v) = s.velocity_simple(H_MIN);
            u_data.push(u);
            v_data.push(v);
        }
    }

    (h_data, eta_data, u_data, v_data)
}
