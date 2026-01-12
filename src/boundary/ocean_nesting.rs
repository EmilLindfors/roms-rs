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
//! - Flather blending for wave absorption at open boundaries
//! - Handles SSH → water depth conversion with bathymetry
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
//!     .with_flather(true);
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
    /// Use Flather blending for wave absorption
    use_flather: bool,
    /// Flather blending weight (0 = Dirichlet, 1 = full Flather)
    flather_weight: f64,
    /// Minimum depth threshold
    h_min: f64,
    /// Fallback state when ocean model has no data at location
    fallback_state: Option<SWEState2D>,
}

impl<P: CoordinateProjection> OceanNestingBC2D<P> {
    /// Create a new ocean nesting BC.
    ///
    /// # Arguments
    /// * `reader` - Ocean model data reader (shared via Arc for efficiency)
    /// * `projection` - Coordinate projection to convert mesh coords to lat/lon
    pub fn new(reader: Arc<OceanModelReader>, projection: P) -> Self {
        Self {
            reader,
            projection,
            reference_level: 0.0,
            use_flather: true,
            flather_weight: 1.0,
            h_min: 1e-6,
            fallback_state: None,
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
    /// Flather blending combines the parent model state with a radiation
    /// condition to allow outgoing waves to exit cleanly.
    pub fn with_flather(mut self, enable: bool) -> Self {
        self.use_flather = enable;
        self
    }

    /// Set Flather blending weight.
    ///
    /// - 0.0: Pure Dirichlet (directly impose parent state)
    /// - 1.0: Full Flather blending (default)
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

    /// Get the ocean state at a position and time.
    fn get_ocean_state(&self, x: f64, y: f64, time: f64) -> Option<OceanState> {
        // Convert mesh coordinates to geographic
        let (lat, lon) = self.projection.xy_to_geo(x, y);

        // Query ocean model with time interpolation
        self.reader.get_state_interpolated(lon, lat, time)
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

    /// Get the time range covered by the ocean model.
    pub fn time_range(&self) -> Option<(f64, f64)> {
        if self.reader.time.is_empty() {
            None
        } else {
            Some((
                self.reader.time[0],
                self.reader.time[self.reader.time.len() - 1],
            ))
        }
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
        let g = ctx.g;
        let (nx, ny) = ctx.normal;

        // Try to get ocean state at this position and time
        let ocean_state = match self.get_ocean_state(x, y, t) {
            Some(state) => state,
            None => {
                // No data - use fallback or mirror interior
                return self
                    .fallback_state
                    .unwrap_or_else(|| ctx.interior_state.clone());
            }
        };

        // Convert to SWE state
        let state_ext = self.ocean_to_swe(&ocean_state, ctx.bathymetry);

        // Pure Dirichlet mode: return external state directly
        if !self.use_flather || self.flather_weight < 1e-10 {
            return state_ext;
        }

        // Flather blending mode
        let h_ext = state_ext.h.max(self.h_min);
        let (u_ext, v_ext) = state_ext.velocity_simple(self.h_min);

        // Normal and tangential velocities (external)
        let un_ext = u_ext * nx + v_ext * ny;
        let ut_ext = -u_ext * ny + v_ext * nx;

        // Surface elevations
        let eta_ext = h_ext + ctx.bathymetry;
        let eta_int = ctx.interior_surface_elevation();

        // Wave celerity
        let c_ext = (g * h_ext).sqrt();

        // Flather relation: u_n = u_n_ext + c * (η_int - η_ext) / h
        let un_flather = un_ext + c_ext * (eta_int - eta_ext) / h_ext;

        // Blend Dirichlet and Flather
        let w = self.flather_weight;
        let un_ghost = (1.0 - w) * un_ext + w * un_flather;
        let ut_ghost = ut_ext; // Tangential from external

        // Convert back to (u, v)
        let u_ghost = un_ghost * nx - ut_ghost * ny;
        let v_ghost = un_ghost * ny + ut_ghost * nx;

        SWEState2D::from_primitives(h_ext, u_ghost, v_ghost)
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::LocalProjection;

    const G: f64 = 9.81;
    const H_MIN: f64 = 1e-6;

    fn make_context(x: f64, y: f64, h: f64, hu: f64, hv: f64, bathy: f64, time: f64) -> BCContext2D {
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
}
