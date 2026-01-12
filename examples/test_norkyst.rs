//! Test loading NorKyst v3 data and using it for boundary conditions.
//!
//! Run with: cargo run --example test_norkyst --features netcdf

#[cfg(feature = "netcdf")]
fn main() {
    use dg_rs::boundary::{BCContext2D, OceanNestingBC2D, SWEBoundaryCondition2D};
    use dg_rs::io::{LocalProjection, OceanModelReader};
    use dg_rs::solver::SWEState2D;
    use std::sync::Arc;

    // Try stuv file (salinity, temperature, u, v) with larger bbox
    let path = "/home/emil/aqc/aqc-h3o/data/norkyst_v3/norkyst_v3_norkystv3_800m_m00_be_20240130_0600_stuv_bbox_7p8_62p8_8p9_64p1.nc";

    println!("Loading NorKyst v3 file: {}", path);

    match OceanModelReader::from_file(path) {
        Ok(reader) => {
            println!("{}", reader.summary());

            // Try to get a state at Frøya coordinates
            let lat = 63.75;
            let lon = 8.5;

            if let Some(state) = reader.get_state(lon, lat, 0) {
                println!("\nState at ({}, {}):", lat, lon);
                println!("  SSH: {:.3} m", state.ssh);
                println!("  u: {:.3} m/s", state.u);
                println!("  v: {:.3} m/s", state.v);
                if let Some(t) = state.temperature {
                    println!("  T: {:.2} °C", t);
                }
                if let Some(s) = state.salinity {
                    println!("  S: {:.2} PSU", s);
                }
            } else {
                println!("No data at ({}, {})", lat, lon);
            }

            // Test time interpolation if there are multiple time steps
            if reader.time.len() > 1 {
                let t_mid = (reader.time[0] + reader.time[1]) / 2.0;
                println!("\nTime interpolation test at t={:.1} s:", t_mid);
                if let Some(state) = reader.get_state_interpolated(lon, lat, t_mid) {
                    println!("  SSH: {:.3} m", state.ssh);
                    println!("  u: {:.3} m/s", state.u);
                    println!("  v: {:.3} m/s", state.v);
                }
            }

            // =====================================================
            // Demonstrate OceanNestingBC2D for boundary conditions
            // =====================================================
            println!("\n{}", "=".repeat(60));
            println!("OceanNestingBC2D Boundary Condition Demo");
            println!("{}", "=".repeat(60));

            // Create projection centered on Frøya
            let projection = LocalProjection::new(lat, lon);

            // Wrap reader in Arc for sharing
            let reader = Arc::new(reader);

            // Create ocean nesting BC with Flather blending
            let bc = OceanNestingBC2D::new(Arc::clone(&reader), projection)
                .with_reference_level(0.0) // MSL reference
                .with_flather(true)
                .with_flather_weight(1.0);

            println!("BC name: {}", bc.name());
            if let Some((t0, t1)) = bc.time_range() {
                println!("Time range: {:.0} - {:.0} s", t0, t1);
            }
            println!("Spatial bounds: {:?}", bc.spatial_bounds());

            // Simulate a boundary node query
            // At origin (0, 0) in local coords = center lat/lon
            let bathymetry = 50.0; // 50m depth
            let interior_state = SWEState2D::from_primitives(50.0, 0.0, 0.0); // Still water
            let time = reader.time[0]; // First timestep

            let ctx = BCContext2D::new(
                time,
                (0.0, 0.0),    // Position at origin (= lat, lon center)
                interior_state,
                bathymetry,
                (1.0, 0.0),   // Outward normal in +x direction
                9.81,         // g
                1e-6,         // h_min
            );

            let ghost = bc.ghost_state(&ctx);
            println!("\nBoundary ghost state at origin:");
            println!("  h: {:.3} m", ghost.h);
            println!("  u: {:.4} m/s", ghost.hu / ghost.h.max(1e-6));
            println!("  v: {:.4} m/s", ghost.hv / ghost.h.max(1e-6));

            // Test at a point 5km to the east
            let ctx_east = BCContext2D::new(
                time,
                (5000.0, 0.0), // 5km east of center
                interior_state,
                bathymetry,
                (1.0, 0.0),
                9.81,
                1e-6,
            );

            if bc.contains_position(5000.0, 0.0) {
                let ghost_east = bc.ghost_state(&ctx_east);
                println!("\nBoundary ghost state 5km east:");
                println!("  h: {:.3} m", ghost_east.h);
                println!("  u: {:.4} m/s", ghost_east.hu / ghost_east.h.max(1e-6));
                println!("  v: {:.4} m/s", ghost_east.hv / ghost_east.h.max(1e-6));
            } else {
                println!("\nPoint 5km east is outside ocean model domain");
            }

            println!("\nOceanNestingBC2D ready for use in simulations!");
        }
        Err(e) => {
            eprintln!("Failed to load: {}", e);
        }
    }
}

#[cfg(not(feature = "netcdf"))]
fn main() {
    eprintln!("This example requires the 'netcdf' feature.");
    eprintln!("Run with: cargo run --example test_norkyst --features netcdf");
}
