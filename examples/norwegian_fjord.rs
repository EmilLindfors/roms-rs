//! Norwegian Fjord Simulation Example
//!
//! Demonstrates a coupled shallow water + tracer transport simulation
//! of a simplified Norwegian fjord with:
//!
//! - Tidal forcing at the open boundary (M2 tide)
//! - River discharge with freshwater inflow
//! - Surface heat flux (summer conditions)
//! - Coriolis force (60°N latitude)
//! - Bottom friction (Manning)
//! - Tracer transport (temperature and salinity)
//!
//! Run with: `cargo run --release --example norwegian_fjord`

use std::cell::Cell;
use std::fs;
use std::path::Path;

use dg_rs::boundary::{HarmonicFlather2D, MultiBoundaryCondition2D, Reflective2D};
use dg_rs::equations::ShallowWater2D;
use dg_rs::io::write_vtk_coupled;
use dg_rs::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::solver::{
    SWE2DRhsConfig, SWESolution2D, Tracer2DRhsConfig, TracerSolution2D, TracerState,
    UpwindTracerBC, compute_rhs_swe_2d, compute_rhs_tracer_2d,
};
// Parallel imports (only beneficial for meshes with 1000+ elements)
use dg_rs::source::tracer_source_2d::{
    CombinedTracerSource2D, RiverTracerSource, SurfaceHeatFlux, gaussian_river_localization,
};
use dg_rs::source::{BathymetrySource2D, CombinedSource2D, CoriolisSource2D, ManningFriction2D};
use dg_rs::time::{
    CoupledRhs2D, CoupledState2D, CoupledTimeConfig, compute_dt_coupled,
    run_coupled_simulation_limited,
};
use dg_rs::{SWEFluxType2D, SWEState2D};
#[cfg(feature = "parallel")]
use dg_rs::{compute_rhs_swe_2d_parallel, compute_rhs_tracer_2d_parallel};

// ============================================================================
// Physical Parameters
// ============================================================================

/// Gravitational acceleration (m/s²)
const G: f64 = 9.81;

/// Minimum depth for wetting/drying (m)
const H_MIN: f64 = 0.01;

/// Coriolis parameter at 60°N (s⁻¹)
const F_CORIOLIS: f64 = 1.2e-4;

/// Manning roughness coefficient (s/m^{1/3})
const MANNING_N: f64 = 0.025;

/// Horizontal diffusivity for tracers (m²/s)
const KAPPA: f64 = 10.0;

/// M2 tidal period (s)
const M2_PERIOD: f64 = 12.42 * 3600.0;

/// M2 tidal amplitude (m)
const M2_AMPLITUDE: f64 = 0.5;

/// River discharge (m³/s)
const RIVER_DISCHARGE: f64 = 100.0;

/// River temperature (°C)
const RIVER_TEMPERATURE: f64 = 8.0;

/// Atlantic water temperature (°C)
const ATLANTIC_TEMPERATURE: f64 = 7.5;

/// Atlantic water salinity (PSU)
const ATLANTIC_SALINITY: f64 = 35.0;

/// Surface heat flux in summer (W/m²)
const HEAT_FLUX: f64 = 100.0;

// ============================================================================
// Domain Configuration
// ============================================================================

/// Fjord length (m)
const L_X: f64 = 20_000.0;

/// Fjord width (m)
const L_Y: f64 = 5_000.0;

/// Mean depth (m)
const H_MEAN: f64 = 50.0;

/// Number of elements in x-direction
const NX: usize = 20;

/// Number of elements in y-direction
const NY: usize = 5;

/// Polynomial order
const ORDER: usize = 3;

// ============================================================================
// Simulation Configuration
// ============================================================================

/// Simulation duration (s) - 1 tidal cycle
const T_END: f64 = M2_PERIOD;

/// Output interval (s)
const OUTPUT_INTERVAL: f64 = 3600.0;

/// CFL number
const CFL: f64 = 0.3;

fn main() {
    println!("Norwegian Fjord Simulation");
    println!("==========================");
    println!();
    println!("Domain: {:.0} km x {:.0} km", L_X / 1000.0, L_Y / 1000.0);
    println!("Mesh: {} x {} elements, order {}", NX, NY, ORDER);
    println!("Mean depth: {} m", H_MEAN);
    println!(
        "Simulation time: {:.1} hours ({:.1} M2 cycles)",
        T_END / 3600.0,
        T_END / M2_PERIOD
    );
    println!();

    // ========================================================================
    // Setup mesh and operators
    // ========================================================================
    println!("Setting up mesh and operators...");

    let mesh = create_fjord_mesh();
    let n_elements = mesh.n_elements;

    let ops = DGOperators2D::new(ORDER);
    let n_nodes = ops.n_nodes;
    let geom = GeometricFactors2D::compute(&mesh);

    println!("  Elements: {}", n_elements);
    println!("  Nodes per element: {}", n_nodes);
    println!("  Total DOFs: {}", n_elements * n_nodes * 3);

    // ========================================================================
    // Setup bathymetry
    // ========================================================================
    println!("Setting up bathymetry...");

    let bathymetry = create_bathymetry(&mesh, &ops, &geom);

    // ========================================================================
    // Setup initial conditions
    // ========================================================================
    println!("Setting up initial conditions...");

    let mut state = create_initial_state(n_elements, n_nodes, &mesh);

    println!("  Initial water depth: {:.1} m (uniform)", H_MEAN);
    println!(
        "  Initial tracers: T={:.1} C, S={:.1} PSU",
        ATLANTIC_TEMPERATURE, ATLANTIC_SALINITY
    );

    // ========================================================================
    // Setup source terms
    // ========================================================================
    println!("Setting up physics...");

    // SWE source terms
    let coriolis = CoriolisSource2D::f_plane(F_CORIOLIS);
    let friction = ManningFriction2D::new(G, MANNING_N);
    let bathy_source = BathymetrySource2D::new(G);

    println!("  Coriolis: f = {:.2e} s^-1", F_CORIOLIS);
    println!("  Manning: n = {}", MANNING_N);

    // Tracer source terms
    let heat_flux = SurfaceHeatFlux::constant(HEAT_FLUX);

    // River at x=1km, y=L_Y/2
    let river_x = 1000.0;
    let river_y = L_Y / 2.0;
    let river_sigma = 200.0;
    let river_area = std::f64::consts::PI * river_sigma * river_sigma;

    let river = RiverTracerSource::new(
        RIVER_DISCHARGE,
        TracerState::new(RIVER_TEMPERATURE, 0.0),
        gaussian_river_localization(river_x, river_y, river_sigma),
        river_area,
    );

    let tracer_sources = CombinedTracerSource2D::empty().add(heat_flux).add(river);

    println!("  Heat flux: {} W/m^2", HEAT_FLUX);
    println!(
        "  River: Q={} m^3/s at ({:.0}, {:.0}) km",
        RIVER_DISCHARGE,
        river_x / 1000.0,
        river_y / 1000.0
    );

    // ========================================================================
    // Setup boundary conditions
    // ========================================================================
    println!("Setting up boundary conditions...");

    // SWE: Tidal forcing at y=L_Y (open sea), walls elsewhere
    let tidal_bc = HarmonicFlather2D::m2_only(M2_AMPLITUDE, 0.0, H_MEAN);
    let wall_bc = Reflective2D::new();

    // Tracer: Atlantic water on inflow, extrapolation on outflow
    let tracer_bc = UpwindTracerBC::atlantic();

    println!("  Open boundary: M2 tide, amplitude = {} m", M2_AMPLITUDE);
    println!("  Walls: reflective");

    // ========================================================================
    // Time integration config
    // ========================================================================
    let time_config = CoupledTimeConfig::new(CFL, G, H_MIN)
        .with_baroclinic(false)
        .with_diffusivity(KAPPA)
        .with_kuzmin_limiters(1.0); // Enable Kuzmin vertex-based limiters

    let equation = ShallowWater2D::new(G);

    // ========================================================================
    // Run simulation
    // ========================================================================
    println!();
    println!("Starting simulation...");

    // Create output directory for VTK files
    let output_dir = Path::new("output");
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).expect("Failed to create output directory");
    }
    println!("VTK output directory: {}", output_dir.display());
    println!();

    let last_output = Cell::new(0.0);
    let vtk_frame = Cell::new(0usize);

    // Create combined source for SWE
    let swe_sources = CombinedSource2D::new(vec![&coriolis, &friction, &bathy_source]);

    // Multi-BC dispatcher
    let swe_bc = MultiBoundaryCondition2D::new(&wall_bc).with_open(&tidal_bc);

    // RHS function - uses parallel when enabled AND mesh has 1000+ elements
    // (parallel overhead dominates for smaller meshes)
    #[cfg(feature = "parallel")]
    let use_parallel = n_elements >= 1000;
    #[cfg(not(feature = "parallel"))]
    let _use_parallel = false;

    let rhs_fn = |state: &CoupledState2D, time: f64| {
        let swe_config = SWE2DRhsConfig::new(&equation, &swe_bc)
            .with_flux_type(SWEFluxType2D::Roe)
            .with_coriolis(false)
            .with_source_terms(&swe_sources)
            .with_bathymetry(&bathymetry);

        let tracer_config = Tracer2DRhsConfig::new(&tracer_bc, G, H_MIN)
            .with_diffusivity(KAPPA)
            .with_source_terms(&tracer_sources);

        #[cfg(feature = "parallel")]
        if use_parallel {
            let swe_rhs =
                compute_rhs_swe_2d_parallel(&state.swe, &mesh, &ops, &geom, &swe_config, time);
            let tracer_rhs = compute_rhs_tracer_2d_parallel(
                &state.tracers,
                &state.swe,
                &mesh,
                &ops,
                &geom,
                &tracer_config,
                time,
            );
            return CoupledRhs2D::new(swe_rhs, tracer_rhs);
        }

        // Serial version (used when parallel disabled or mesh too small)
        let swe_rhs = compute_rhs_swe_2d(&state.swe, &mesh, &ops, &geom, &swe_config, time);
        let tracer_rhs = compute_rhs_tracer_2d(
            &state.tracers,
            &state.swe,
            &mesh,
            &ops,
            &geom,
            &tracer_config,
            time,
        );
        CoupledRhs2D::new(swe_rhs, tracer_rhs)
    };

    // dt function
    let dt_fn =
        |state: &CoupledState2D| compute_dt_coupled(state, &mesh, &geom, &time_config, ORDER);

    // Callback for output
    let callback = |t: f64, state: &CoupledState2D| {
        if t - last_output.get() >= OUTPUT_INTERVAL || t >= T_END - 1e-6 {
            last_output.set(t);
            print_diagnostics(t, state);

            // Write VTK output
            let frame = vtk_frame.get();
            let vtk_path = output_dir.join(format!("fjord_{:04}.vtu", frame));
            if let Err(e) = write_vtk_coupled(
                &vtk_path,
                &mesh,
                &ops,
                &state.swe,
                &state.tracers,
                Some(&bathymetry),
                t,
                H_MIN,
            ) {
                eprintln!("Warning: Failed to write VTK file: {}", e);
            }
            vtk_frame.set(frame + 1);
        }
    };

    let (t_final, n_steps) = run_coupled_simulation_limited(
        &mut state,
        T_END,
        rhs_fn,
        dt_fn,
        &mesh,
        &ops,
        &time_config,
        Some(callback),
    );

    // ========================================================================
    // Final summary
    // ========================================================================
    println!();
    println!("Simulation complete!");
    println!("  Final time: {:.1} hours", t_final / 3600.0);
    println!("  Total steps: {}", n_steps);
    println!(
        "  Average dt: {:.2} s",
        if n_steps > 0 {
            t_final / n_steps as f64
        } else {
            0.0
        }
    );
    println!("  VTK frames written: {}", vtk_frame.get());

    print_final_diagnostics(&state);
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create the fjord mesh with appropriate boundary tags.
fn create_fjord_mesh() -> Mesh2D {
    // Create uniform mesh with wall boundaries
    let mut mesh = Mesh2D::uniform_rectangle_with_bc(0.0, L_X, 0.0, L_Y, NX, NY, BoundaryTag::Wall);

    // Tag the open boundary (y = L_Y) as Open for tidal forcing
    for edge in mesh.edges.iter_mut() {
        if edge.is_boundary() {
            // Get edge vertex coordinates
            let (v0, v1) = edge.vertices;
            let y0 = mesh.vertices[v0].1;
            let y1 = mesh.vertices[v1].1;
            let y_center = 0.5 * (y0 + y1);

            // If edge is at y = L_Y, tag as Open
            if (y_center - L_Y).abs() < 1.0 {
                edge.boundary_tag = Some(BoundaryTag::Open);
            }
        }
    }

    mesh
}

/// Create bathymetry with a gentle slope towards the fjord head.
fn create_bathymetry(
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
) -> Bathymetry2D {
    // Bathymetry: deeper at open boundary, shallower at fjord head
    // B(x, y) = -H_mean * (0.8 + 0.4 * y / L_Y)
    let bathymetry_fn = |_x: f64, y: f64| -H_MEAN * (0.8 + 0.4 * y / L_Y);

    Bathymetry2D::from_function(mesh, ops, geom, bathymetry_fn)
}

/// Create initial state with uniform depth and Atlantic water properties.
fn create_initial_state(n_elements: usize, n_nodes: usize, mesh: &Mesh2D) -> CoupledState2D {
    let mut swe = SWESolution2D::new(n_elements, n_nodes);
    let mut tracers = TracerSolution2D::new(n_elements, n_nodes);

    for k in 0..n_elements {
        // Get element center
        let elem = &mesh.elements[k];
        let y_center: f64 = elem.iter().map(|&v| mesh.vertices[v].1).sum::<f64>() / 4.0;

        // Depth varies with bathymetry
        let b = -H_MEAN * (0.8 + 0.4 * y_center / L_Y);
        let h = -b;

        for i in 0..n_nodes {
            swe.set_state(k, i, SWEState2D::from_primitives(h, 0.0, 0.0));

            let tracer = TracerState::new(ATLANTIC_TEMPERATURE, ATLANTIC_SALINITY);
            tracers.set_from_concentrations(k, i, h, tracer);
        }
    }

    CoupledState2D::new(swe, tracers)
}

/// Print simulation diagnostics.
fn print_diagnostics(t: f64, state: &CoupledState2D) {
    let hours = t / 3600.0;
    let m2_phase = (t / M2_PERIOD) % 1.0;

    let h_min_v = state.swe.min_depth();
    let h_max_v = state.swe.max_depth();
    let (u_max, v_max) = velocity_max(&state.swe);
    let (t_min, t_max) = state.tracers.temperature_range(&state.swe, H_MIN);
    let (s_min, s_max) = state.tracers.salinity_range(&state.swe, H_MIN);

    println!(
        "t = {:5.1} h ({:.2} M2) | h: [{:5.1}, {:5.1}] m | |u|: {:.3} m/s | T: [{:.1}, {:.1}] C | S: [{:.1}, {:.1}] PSU",
        hours,
        m2_phase,
        h_min_v,
        h_max_v,
        (u_max * u_max + v_max * v_max).sqrt(),
        t_min,
        t_max,
        s_min,
        s_max,
    );
}

/// Get maximum velocity components.
fn velocity_max(swe: &SWESolution2D) -> (f64, f64) {
    let mut u_max = 0.0_f64;
    let mut v_max = 0.0_f64;

    for k in 0..swe.n_elements {
        for i in 0..swe.n_nodes {
            let state = swe.get_state(k, i);
            let (u, v) = state.velocity_simple(H_MIN);
            u_max = u_max.max(u.abs());
            v_max = v_max.max(v.abs());
        }
    }

    (u_max, v_max)
}

/// Print final diagnostics.
fn print_final_diagnostics(state: &CoupledState2D) {
    println!();
    println!("Final State Analysis");
    println!("--------------------");

    let h_min_v = state.swe.min_depth();
    let h_max_v = state.swe.max_depth();
    println!("  Water depth: min={:.2} m, max={:.2} m", h_min_v, h_max_v);

    let (u_max, v_max) = velocity_max(&state.swe);
    let speed_max = (u_max * u_max + v_max * v_max).sqrt();
    println!(
        "  Max velocity: {:.3} m/s (u_max={:.3}, v_max={:.3})",
        speed_max, u_max, v_max
    );

    let (t_min, t_max) = state.tracers.temperature_range(&state.swe, H_MIN);
    let (s_min, s_max) = state.tracers.salinity_range(&state.swe, H_MIN);
    println!("  Temperature: min={:.2} C, max={:.2} C", t_min, t_max);
    println!("  Salinity: min={:.2} PSU, max={:.2} PSU", s_min, s_max);

    // Check for freshwater influence
    let mut fresh_nodes = 0;
    let total_nodes = state.tracers.n_elements * state.tracers.n_nodes;
    for k in 0..state.tracers.n_elements {
        for i in 0..state.tracers.n_nodes {
            let h = state.swe.get_state(k, i).h;
            let tracer = state.tracers.get_concentrations(k, i, h, H_MIN);
            if tracer.salinity < 30.0 {
                fresh_nodes += 1;
            }
        }
    }

    println!(
        "  Freshwater influence: {:.1}% of domain (S < 30 PSU)",
        100.0 * fresh_nodes as f64 / total_nodes as f64
    );
}
