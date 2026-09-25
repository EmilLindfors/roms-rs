//! Ocean model nesting boundary condition.
//!
//! Provides spatially-varying boundary conditions from ocean model output (NorKyst, ROMS).
//! Unlike the simple `NestingBC2D` which applies uniform conditions, this BC queries
//! the ocean model at each boundary node's geographic location.
//!
//! # Features
//!
//! - Spatially-varying boundary conditions from parent ocean model
//! - Automatic coordinate transformation (mesh coords → lat/lon)
//! - Time interpolation within the ocean model data
//! - Flather-type open boundary (wave absorption) via the Riemann solver
//! - Handles SSH → water depth conversion with bathymetry
//!
//! # Weak (ghost-state) formulation
//!
//! The ghost state is the parent state, `h = η_ext − B` with the parent
//! velocity. The upwind Riemann solver at the boundary face takes the outgoing
//! Riemann invariant from the interior and the incoming one from the ghost,
//! which is the Flather (1976) condition. Adding the Flather correction
//! `sqrt(g/h)(η_int − η_ext)` to the ghost normal velocity as well would count
//! it twice and reflect outgoing waves, so `with_flather` and
//! `with_flather_weight` no longer change the ghost state.
//!
//! # Time base
//!
//! The reader decodes the parent's CF time to Unix seconds. Simulation time
//! `t` (seconds) maps to the parent instant `epoch + t`; by default the epoch is
//! the first parent snapshot, so `t = 0` starts the run there. Set it with
//! [`OceanNestingBC2D::with_epoch`] to start elsewhere in the file. Asking for a
//! time outside the file panics instead of freezing the forcing at the first or
//! last snapshot; check a run up front with
//! [`OceanNestingBC2D::check_time_coverage`].
//!
//! # Example
//!
//! ```ignore
//! use dg_rs::boundary::OceanNestingBC2D;
//! use dg_rs::io::{OceanModelReader, LocalProjection};
//!
//! // Load ocean model data
//! let reader = OceanModelReader::from_file("norkyst.nc")?;
//!
//! // Create projection (mesh uses local coords centered at this point)
//! let projection = LocalProjection::new(63.8, 8.9);
//!
//! // Create nesting BC
//! let bc = OceanNestingBC2D::new(reader, projection)
//!     .with_reference_level(0.0)  // MSL reference
//!     .with_epoch(1_706_594_400.0); // 2024-01-30 06:00 UTC
//! bc.check_time_coverage(0.0, 6.0 * 3600.0)?;
//! ```

use crate::boundary::{BCContext2D, SWEBoundaryCondition2D};
use crate::io::{CoordinateProjection, OceanModelReader, OceanState};
use crate::solver::SWEState2D;
use std::sync::Arc;

/// Ocean model nesting boundary condition.
///
/// Provides spatially-varying boundary forcing from a parent ocean model
/// (e.g., NorKyst v3, ROMS) by querying the model at each boundary node's
/// geographic location.
pub struct OceanNestingBC2D<P: CoordinateProjection> {
    /// Ocean model data reader
    reader: Arc<OceanModelReader>,
    /// Projection to convert mesh (x, y) to (lat, lon)
    projection: P,
    /// Reference sea level (η₀) - typically 0 for MSL
    reference_level: f64,
    /// Use Flather blending for wave absorption.
    /// Only affects [`name`](SWEBoundaryCondition2D::name).
    use_flather: bool,
    /// Flather blending weight (unused, retained for API compatibility)
    flather_weight: f64,
    /// Minimum depth threshold
    h_min: f64,
    /// Fallback state when ocean model has no data at location
    fallback_state: Option<SWEState2D>,
    /// Parent-model instant (Unix seconds) of simulation time t = 0
    epoch_unix: f64,
}

impl<P: CoordinateProjection> OceanNestingBC2D<P> {
    /// Create a new ocean nesting BC.
    ///
    /// # Arguments
    /// * `reader` - Ocean model data reader (shared via Arc for efficiency)
    /// * `projection` - Coordinate projection to convert mesh coords to lat/lon
    pub fn new(reader: Arc<OceanModelReader>, projection: P) -> Self {
        let epoch_unix = reader.time.first().copied().unwrap_or(0.0);
        Self {
            reader,
            projection,
            reference_level: 0.0,
            use_flather: true,
            flather_weight: 1.0,
            h_min: 1e-6,
            fallback_state: None,
            epoch_unix,
        }
    }

    /// Set the parent-model instant (Unix seconds, UTC) of simulation time 0.
    ///
    /// Defaults to the first snapshot in the file.
    pub fn with_epoch(mut self, epoch_unix: f64) -> Self {
        self.epoch_unix = epoch_unix;
        self
    }

    /// Parent-model instant (Unix seconds) of simulation time 0.
    pub fn epoch(&self) -> f64 {
        self.epoch_unix
    }

    /// Parent-model instant (Unix seconds) of simulation time `t`.
    fn model_time(&self, t: f64) -> f64 {
        self.epoch_unix + t
    }

    /// Check that simulation times `[t_start, t_end]` lie within the parent
    /// file, so a run cannot hit the out-of-range panic in `ghost_state`.
    pub fn check_time_coverage(&self, t_start: f64, t_end: f64) -> Result<(), String> {
        for t in [t_start, t_end] {
            if !self.reader.covers_time(self.model_time(t)) {
                return Err(self.coverage_message(t));
            }
        }
        Ok(())
    }

    fn coverage_message(&self, t: f64) -> String {
        match self.simulation_time_range() {
            Some((t0, t1)) => format!(
                "OceanNestingBC2D: simulation time {t} s is outside the parent file, which \
                 covers simulation times [{t0}, {t1}] s (epoch {} Unix s)",
                self.epoch_unix
            ),
            None => "OceanNestingBC2D: the parent file has no time steps".to_string(),
        }
    }

    /// Set the reference sea level.
    ///
    /// The water depth h is computed as: h = η - B + η₀
    /// where η is SSH from ocean model, B is local bathymetry, η₀ is reference.
    pub fn with_reference_level(mut self, level: f64) -> Self {
        self.reference_level = level;
        self
    }

    /// Enable or disable Flather blending.
    ///
    /// No longer affects the ghost state: the parent-state ghost combined
    /// with the upwind Riemann solver already lets outgoing waves exit cleanly
    /// (see the module docs). Retained for API compatibility.
    pub fn with_flather(mut self, enable: bool) -> Self {
        self.use_flather = enable;
        self
    }

    /// Set Flather blending weight.
    ///
    /// No longer affects the ghost state (see [`with_flather`](Self::with_flather)).
    /// Retained for API compatibility.
    pub fn with_flather_weight(mut self, weight: f64) -> Self {
        self.flather_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set minimum depth threshold.
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }

    /// Set fallback state when ocean model has no data.
    ///
    /// If not set, the interior state is mirrored when data is unavailable.
    pub fn with_fallback(mut self, state: SWEState2D) -> Self {
        self.fallback_state = Some(state);
        self
    }

    /// Get the ocean state at a position and simulation time.
    fn get_ocean_state(&self, x: f64, y: f64, time: f64) -> Option<OceanState> {
        // Convert mesh coordinates to geographic
        let (lat, lon) = self.projection.xy_to_geo(x, y);

        // Query ocean model with time interpolation
        self.reader
            .get_state_interpolated(lon, lat, self.model_time(time))
    }

    /// Convert ocean state to SWE state.
    ///
    /// # Arguments
    /// * `ocean` - Ocean state (SSH, u, v)
    /// * `bathymetry` - Local bathymetry B (negative = below sea level, oceanographic convention)
    fn ocean_to_swe(&self, ocean: &OceanState, bathymetry: f64) -> SWEState2D {
        // Water depth: h = η - B
        // where η = SSH (sea surface height above geoid)
        //       B = bathymetry (negative for underwater, oceanographic convention)
        //
        // Example: SSH = 0.5m, B = -100m → h = 0.5 - (-100) = 100.5m
        let h = (ocean.ssh - bathymetry + self.reference_level).max(self.h_min);

        // Momentum: hu = h * u, hv = h * v
        let hu = h * ocean.u;
        let hv = h * ocean.v;

        SWEState2D::new(h, hu, hv)
    }

    /// Get the time range covered by the ocean model, in Unix seconds.
    pub fn time_range(&self) -> Option<(f64, f64)> {
        self.reader.time_range()
    }

    /// Get the time range covered by the ocean model, in simulation time
    /// (seconds after the epoch).
    pub fn simulation_time_range(&self) -> Option<(f64, f64)> {
        let (t0, t1) = self.reader.time_range()?;
        Some((t0 - self.epoch_unix, t1 - self.epoch_unix))
    }

    /// Get the spatial bounding box of the ocean model.
    ///
    /// Returns (min_lon, min_lat, max_lon, max_lat).
    pub fn spatial_bounds(&self) -> (f64, f64, f64, f64) {
        self.reader.bbox
    }

    /// Check if a position is within the ocean model domain.
    pub fn contains_position(&self, x: f64, y: f64) -> bool {
        let (lat, lon) = self.projection.xy_to_geo(x, y);
        let (min_lon, min_lat, max_lon, max_lat) = self.reader.bbox;
        lon >= min_lon && lon <= max_lon && lat >= min_lat && lat <= max_lat
    }
}

impl<P: CoordinateProjection + Send + Sync> SWEBoundaryCondition2D for OceanNestingBC2D<P> {
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        let (x, y) = ctx.position;
        let t = ctx.time;

        // Out-of-range times are a configuration error. Falling back (or
        // clamping to the nearest snapshot) would silently freeze the forcing.
        if !self.reader.covers_time(self.model_time(t)) {
            panic!("{}", self.coverage_message(t));
        }

        // Try to get ocean state at this position and time
        let ocean_state = match self.get_ocean_state(x, y, t) {
            Some(state) => state,
            None => {
                // No data - use fallback or mirror interior
                return self.fallback_state.unwrap_or(ctx.interior_state);
            }
        };

        // Ghost = parent state, h = η_ext − B (clamped to h_min). The Riemann
        // solver applies the Flather relation; adding
        // sqrt(g/h)(η_int − η_ext) to u_n here would apply it twice.
        self.ocean_to_swe(&ocean_state, ctx.bathymetry)
    }

    fn name(&self) -> &'static str {
        if self.use_flather {
            "ocean_nesting_flather"
        } else {
            "ocean_nesting_dirichlet"
        }
    }
}

// Implement Clone manually since Arc is Clone but P might need Clone
impl<P: CoordinateProjection + Clone> Clone for OceanNestingBC2D<P> {
    fn clone(&self) -> Self {
        Self {
            reader: Arc::clone(&self.reader),
            projection: self.projection.clone(),
            reference_level: self.reference_level,
            use_flather: self.use_flather,
            flather_weight: self.flather_weight,
            h_min: self.h_min,
            fallback_state: self.fallback_state,
            epoch_unix: self.epoch_unix,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::LocalProjection;

    const G: f64 = 9.81;
    const H_MIN: f64 = 1e-6;

    fn make_context(
        x: f64,
        y: f64,
        h: f64,
        hu: f64,
        hv: f64,
        bathy: f64,
        time: f64,
    ) -> BCContext2D {
        BCContext2D::new(
            time,
            (x, y),
            SWEState2D::new(h, hu, hv),
            bathy,
            (1.0, 0.0), // Normal in +x
            G,
            H_MIN,
        )
    }

    #[test]
    fn test_ocean_to_swe_conversion() {
        // Create a mock scenario
        // Ocean: SSH = 0.5m, u = 0.1 m/s, v = 0.05 m/s
        // Bathymetry: 50m depth
        // Expected: h = 0.5 + 50 = 50.5m

        let ocean = OceanState {
            ssh: 0.5,
            u: 0.1,
            v: 0.05,
            temperature: None,
            salinity: None,
        };

        // We can't easily test the full BC without a real OceanModelReader,
        // but we can verify the conversion logic
        let h = ocean.ssh + 50.0; // bathymetry = 50
        let hu = h * ocean.u;
        let hv = h * ocean.v;

        assert!((h - 50.5).abs() < 1e-10);
        assert!((hu - 5.05).abs() < 1e-10);
        assert!((hv - 2.525).abs() < 1e-10);
    }

    #[test]
    fn test_projection_integration() {
        // Test that LocalProjection correctly converts coordinates
        let proj = LocalProjection::new(63.8, 8.9);

        // At origin, should return center lat/lon
        let (lat, lon) = proj.xy_to_geo(0.0, 0.0);
        assert!((lat - 63.8).abs() < 1e-6);
        assert!((lon - 8.9).abs() < 1e-6);

        // 1km east should increase longitude
        let (lat2, lon2) = proj.xy_to_geo(1000.0, 0.0);
        assert!(lon2 > lon);
        assert!((lat2 - lat).abs() < 0.01); // Latitude roughly same
    }

    mod parent_file {
        use super::*;
        use crate::io::test_files::{self, CENTRE, T0, Zeta};

        /// Nesting BC on a parent with hourly SSH 0.1, 0.3, 0.7 m, centred on
        /// the grid. The reader loads everything, so the temp dir can go.
        fn bc() -> OceanNestingBC2D<LocalProjection> {
            let dir = tempfile::tempdir().unwrap();
            let path = test_files::write(
                dir.path(),
                Some("hours since 2024-01-30 06:00:00"),
                &[0.1, 0.3, 0.7],
                Zeta::Float,
            );
            let reader = Arc::new(OceanModelReader::from_file(path).unwrap());
            OceanNestingBC2D::new(reader, LocalProjection::new(CENTRE.0, CENTRE.1))
        }

        fn ghost_ssh(bc: &OceanNestingBC2D<LocalProjection>, t: f64) -> f64 {
            let ctx = make_context(0.0, 0.0, 50.0, 0.0, 0.0, -50.0, t);
            bc.ghost_state(&ctx).h - 50.0
        }

        #[test]
        fn simulation_time_starts_at_first_snapshot_by_default() {
            let bc = bc();
            assert_eq!(bc.epoch(), T0);
            assert!((ghost_ssh(&bc, 0.0) - 0.1).abs() < 1e-6);
            assert!((ghost_ssh(&bc, 1800.0) - 0.2).abs() < 1e-6);
            assert!((ghost_ssh(&bc, 5400.0) - 0.5).abs() < 1e-6);
            assert_eq!(bc.simulation_time_range(), Some((0.0, 7200.0)));
        }

        #[test]
        fn epoch_shifts_the_parent_time() {
            let bc = bc().with_epoch(T0 + 3600.0);
            assert!((ghost_ssh(&bc, 0.0) - 0.3).abs() < 1e-6);
            assert_eq!(bc.simulation_time_range(), Some((-3600.0, 3600.0)));
        }

        #[test]
        fn coverage_check_reports_out_of_range_runs() {
            let bc = bc();
            assert!(bc.check_time_coverage(0.0, 7200.0).is_ok());
            assert!(bc.check_time_coverage(0.0, 7201.0).is_err());
            assert!(bc.check_time_coverage(-1.0, 3600.0).is_err());
        }

        #[test]
        #[should_panic(expected = "outside the parent file")]
        fn forcing_past_the_last_snapshot_panics() {
            // P0.20: this used to return the last snapshot forever.
            ghost_ssh(&bc(), 7200.0 + 60.0);
        }
    }
}
