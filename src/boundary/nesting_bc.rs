//! Nesting boundary condition for one-way coupling from parent models.
//!
//! Provides boundary forcing from a parent (coarser resolution) model via
//! time-interpolated state data. Supports both pure Dirichlet and
//! Flather-blended modes.
//!
//! # One-Way Nesting
//!
//! In one-way nesting, the child model receives boundary conditions from
//! a parent model but does not feed back. This is appropriate when:
//! - The parent domain is much larger than the child
//! - Boundary effects on the parent are negligible
//! - Computational efficiency is important
//!
//! # Modes
//!
//! - **Dirichlet**: Directly imposes the interpolated state from the parent
//! - **Flather**: Blends parent data with characteristic radiation for waves
//!
//! # Example
//!
//! ```ignore
//! use dg::boundary::NestingBC2D;
//! use dg::io::{BoundaryTimeSeries, TimeSeriesRecord};
//! use dg::solver::SWEState2D;
//!
//! // Create time series from parent model data
//! let records = vec![
//!     TimeSeriesRecord { time: 0.0, state: SWEState2D::from_primitives(50.0, 0.1, 0.0) },
//!     TimeSeriesRecord { time: 3600.0, state: SWEState2D::from_primitives(50.5, 0.2, 0.0) },
//! ];
//! let ts = BoundaryTimeSeries::from_records(records).unwrap();
//!
//! // Create nesting BC with Flather blending (default)
//! let bc = NestingBC2D::new(ts).with_h_ref(50.0);
//!
//! // Or pure Dirichlet mode
//! let bc_dirichlet = NestingBC2D::new(
//!     BoundaryTimeSeries::from_records(vec![
//!         TimeSeriesRecord { time: 0.0, state: SWEState2D::from_primitives(50.0, 0.0, 0.0) },
//!     ]).unwrap()
//! ).without_flather();
//! ```

use crate::boundary::bathymetry_validation::warn_once_if_misconfigured;
use crate::boundary::{BCContext2D, SWEBoundaryCondition2D};
use crate::io::BoundaryTimeSeries;
use crate::solver::SWEState2D;
use crate::types::Depth;

/// Nesting boundary condition from parent model.
///
/// Interpolates time series data from a parent model and applies it
/// as a boundary condition, optionally with Flather blending for
/// better wave absorption.
#[derive(Clone, Debug)]
pub struct NestingBC2D {
    /// Time series from parent model
    time_series: BoundaryTimeSeries,
    /// Use Flather blending (true) or pure Dirichlet (false)
    use_flather: bool,
    /// Reference depth for Flather mode
    h_ref: f64,
    /// Minimum depth threshold
    h_min: f64,
    /// Blending weight for Flather mode (0 = pure Dirichlet, 1 = full Flather)
    flather_weight: f64,
}

impl NestingBC2D {
    /// Create a new nesting BC from time series data.
    ///
    /// Default: Flather blending enabled with h_ref = 50.0
    pub fn new(time_series: BoundaryTimeSeries) -> Self {
        Self {
            time_series,
            use_flather: true,
            h_ref: 50.0,
            h_min: 1e-6,
            flather_weight: 1.0,
        }
    }

    /// Disable Flather blending (pure Dirichlet mode).
    ///
    /// The interpolated state is directly imposed without modification.
    pub fn without_flather(mut self) -> Self {
        self.use_flather = false;
        self
    }

    /// Enable Flather blending (default).
    pub fn with_flather(mut self) -> Self {
        self.use_flather = true;
        self
    }

    /// Set the reference depth for Flather mode.
    pub fn with_h_ref(mut self, h_ref: f64) -> Self {
        self.h_ref = h_ref;
        self
    }

    /// Set the minimum depth threshold.
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }

    /// Set the Flather blending weight.
    ///
    /// - 0.0: Pure Dirichlet (ignores internal state)
    /// - 1.0: Full Flather blending (default)
    /// - 0.5: Half Flather (smoother transition)
    pub fn with_flather_weight(mut self, weight: f64) -> Self {
        self.flather_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Get the time range covered by the time series.
    ///
    /// Returns (start_time, end_time).
    pub fn time_range(&self) -> (f64, f64) {
        self.time_series.time_range()
    }

    /// Get the interpolated state at time t.
    pub fn interpolate(&self, t: f64) -> SWEState2D {
        self.time_series.interpolate(t)
    }

    /// Check if the given time is within the time series range.
    pub fn is_time_valid(&self, t: f64) -> bool {
        let (t_start, t_end) = self.time_range();
        t >= t_start && t <= t_end
    }

    /// Get reference to the underlying time series.
    pub fn time_series(&self) -> &BoundaryTimeSeries {
        &self.time_series
    }
}

impl SWEBoundaryCondition2D for NestingBC2D {
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        use std::sync::atomic::AtomicBool;
        static WARNED: AtomicBool = AtomicBool::new(false);

        let t = ctx.time;
        let (nx, ny) = ctx.normal;
        let g = ctx.g;

        // Interpolate parent model state at current time
        let state_ext = self.time_series.interpolate(t);

        // Pure Dirichlet mode: return interpolated state directly
        if !self.use_flather || self.flather_weight < 1e-10 {
            return state_ext;
        }

        // Flather blending mode
        // Combines parent state with characteristic-based adjustment

        // External (parent) quantities
        let h_ext = state_ext.h.max(self.h_min);
        let (u_ext, v_ext) = state_ext.velocity_simple(Depth::new(self.h_min));
        let un_ext = u_ext * nx + v_ext * ny;
        let ut_ext = -u_ext * ny + v_ext * nx;
        let eta_ext = h_ext + ctx.bathymetry;

        // Validate bathymetry configuration for Flather mode (warns once if misconfigured)
        // Use eta_ext as the expected elevation since it comes from the parent model
        warn_once_if_misconfigured(
            &WARNED,
            "NestingBC2D (Flather mode)",
            ctx.interior_state.h,
            ctx.bathymetry,
            eta_ext - h_ext, // Expected surface elevation relative to MSL
        );

        // Interior surface elevation for Flather correction
        let eta_int = ctx.interior_surface_elevation();

        // Wave celerity at external depth
        let c_ext = (g * h_ext).sqrt();

        // Flather relation for normal velocity:
        // u_n_ghost = u_n_ext + c * (η_int - η_ext) / h_ext
        let un_flather = un_ext + c_ext * (eta_int - eta_ext) / h_ext;

        // Blend between Dirichlet and Flather
        let w = self.flather_weight;
        let un_ghost = (1.0 - w) * un_ext + w * un_flather;

        // Keep tangential velocity from external state
        let ut_ghost = ut_ext;

        // Convert back to (u, v)
        let u_ghost = un_ghost * nx - ut_ghost * ny;
        let v_ghost = un_ghost * ny + ut_ghost * nx;

        // Use external depth (from parent model)
        SWEState2D::from_primitives(h_ext, u_ghost, v_ghost)
    }

    fn name(&self) -> &'static str {
        if self.use_flather {
            "nesting_flather_2d"
        } else {
            "nesting_dirichlet_2d"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::TimeSeriesRecord;

    const TOL: f64 = 1e-12;
    const G: f64 = 9.81;
    const H_MIN: f64 = 1e-6;

    fn make_simple_ts() -> BoundaryTimeSeries {
        let records = vec![
            TimeSeriesRecord {
                time: 0.0,
                state: SWEState2D::from_primitives(50.0, 0.1, 0.0),
            },
            TimeSeriesRecord {
                time: 100.0,
                state: SWEState2D::from_primitives(51.0, 0.2, 0.0),
            },
        ];
        BoundaryTimeSeries::from_records(records).unwrap()
    }

    fn make_context(h: f64, hu: f64, hv: f64, bathymetry: f64, time: f64) -> BCContext2D {
        BCContext2D::new(
            time,
            (0.0, 0.0),
            SWEState2D::new(h, hu, hv),
            bathymetry,
            (1.0, 0.0), // Normal pointing in +x
            G,
            H_MIN,
        )
    }

    #[test]
    fn test_nesting_bc_creation() {
        let ts = make_simple_ts();
        let bc = NestingBC2D::new(ts);

        assert!(bc.use_flather);
        assert!((bc.h_ref - 50.0).abs() < TOL);
        assert!((bc.h_min - 1e-6).abs() < TOL);
        assert!((bc.flather_weight - 1.0).abs() < TOL);
    }

    #[test]
    fn test_nesting_bc_builders() {
        let ts = make_simple_ts();
        let bc = NestingBC2D::new(ts)
            .without_flather()
            .with_h_ref(100.0)
            .with_h_min(1e-8)
            .with_flather_weight(0.5);

        assert!(!bc.use_flather);
        assert!((bc.h_ref - 100.0).abs() < TOL);
        assert!((bc.h_min - 1e-8).abs() < TOL);
        assert!((bc.flather_weight - 0.5).abs() < TOL);

        // Re-enable flather
        let bc2 = bc.with_flather();
        assert!(bc2.use_flather);
    }

    #[test]
    fn test_time_range() {
        let ts = make_simple_ts();
        let bc = NestingBC2D::new(ts);

        let (t_start, t_end) = bc.time_range();
        assert!((t_start - 0.0).abs() < TOL);
        assert!((t_end - 100.0).abs() < TOL);
    }

    #[test]
    fn test_is_time_valid() {
        let ts = make_simple_ts();
        let bc = NestingBC2D::new(ts);

        assert!(bc.is_time_valid(0.0));
        assert!(bc.is_time_valid(50.0));
        assert!(bc.is_time_valid(100.0));
        assert!(!bc.is_time_valid(-1.0));
        assert!(!bc.is_time_valid(101.0));
    }

    #[test]
    fn test_interpolate() {
        let ts = make_simple_ts();
        let bc = NestingBC2D::new(ts);

        // At t=0: h=50, u=0.1, hu=5
        let state0 = bc.interpolate(0.0);
        assert!((state0.h - 50.0).abs() < TOL);
        assert!((state0.hu - 5.0).abs() < TOL); // hu = h * u = 50 * 0.1

        // At t=50 (midpoint): linear interpolation of conserved variables
        // h = (50 + 51) / 2 = 50.5
        // hu = (5 + 10.2) / 2 = 7.6 where 10.2 = 51 * 0.2
        let state50 = bc.interpolate(50.0);
        assert!((state50.h - 50.5).abs() < TOL);
        assert!((state50.hu - 7.6).abs() < TOL);

        // At t=100: h=51, u=0.2, hu=10.2
        let state100 = bc.interpolate(100.0);
        assert!((state100.h - 51.0).abs() < TOL);
        assert!((state100.hu - 10.2).abs() < TOL);
    }

    #[test]
    fn test_dirichlet_mode() {
        let ts = make_simple_ts();
        let bc = NestingBC2D::new(ts).without_flather();

        // Context with different interior state
        let ctx = make_context(60.0, 12.0, 0.0, 0.0, 50.0);
        let ghost = bc.ghost_state(&ctx);

        // Should return interpolated state exactly, ignoring interior
        // At t=50: h=50.5, hu=7.6 (see test_interpolate)
        assert!((ghost.h - 50.5).abs() < TOL);
        assert!((ghost.hu - 7.6).abs() < TOL);
    }

    #[test]
    fn test_flather_matching_state() {
        // When interior matches external, Flather should give same velocity
        let ts = make_simple_ts();
        let bc = NestingBC2D::new(ts).with_h_ref(50.0);

        // Interior matches external at t=0 (h=50, u=0.1)
        // η_int = η_ext, so the Flather correction is zero
        let state_ext = bc.interpolate(0.0);
        let h_ext = state_ext.h;

        let ctx = make_context(h_ext, h_ext * 0.1, 0.0, 0.0, 0.0);
        let ghost = bc.ghost_state(&ctx);

        // Velocity should match external
        assert!((ghost.hu / ghost.h - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_flather_elevated_interior() {
        // When interior is elevated, Flather should add outward velocity
        let records = vec![
            TimeSeriesRecord {
                time: 0.0,
                state: SWEState2D::from_primitives(50.0, 0.0, 0.0),
            },
            TimeSeriesRecord {
                time: 100.0,
                state: SWEState2D::from_primitives(50.0, 0.0, 0.0),
            },
        ];
        let ts = BoundaryTimeSeries::from_records(records).unwrap();
        let bc = NestingBC2D::new(ts);

        // Interior is 1m higher than external
        // η_ext = 50, η_int = 51
        // c = sqrt(9.81 * 50) ≈ 22.1
        // u_n_ghost = 0 + 22.1 * 1 / 50 ≈ 0.44
        let ctx = make_context(51.0, 0.0, 0.0, 0.0, 0.0);
        let ghost = bc.ghost_state(&ctx);

        // Should have outward velocity
        assert!(ghost.hu / ghost.h > 0.3);
    }

    #[test]
    fn test_flather_weight_zero() {
        // flather_weight = 0 should behave like Dirichlet
        let ts = make_simple_ts();
        let bc = NestingBC2D::new(ts).with_flather_weight(0.0);

        let ctx = make_context(60.0, 12.0, 0.0, 0.0, 0.0);
        let ghost = bc.ghost_state(&ctx);

        // Should return external state's velocity, ignoring interior mismatch
        assert!((ghost.hu / ghost.h - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_flather_weight_half() {
        // flather_weight = 0.5 should blend Dirichlet and Flather
        let records = vec![TimeSeriesRecord {
            time: 0.0,
            state: SWEState2D::from_primitives(50.0, 0.0, 0.0),
        }];
        let ts = BoundaryTimeSeries::from_records(records).unwrap();
        let bc = NestingBC2D::new(ts).with_flather_weight(0.5);

        // Interior elevated by 1m
        let ctx = make_context(51.0, 0.0, 0.0, 0.0, 0.0);
        let ghost_half = bc.ghost_state(&ctx);

        // Compare with full Flather
        let bc_full = NestingBC2D::new(
            BoundaryTimeSeries::from_records(vec![TimeSeriesRecord {
                time: 0.0,
                state: SWEState2D::from_primitives(50.0, 0.0, 0.0),
            }])
            .unwrap(),
        )
        .with_flather_weight(1.0);
        let ghost_full = bc_full.ghost_state(&ctx);

        // Half weight should give about half the velocity
        assert!((ghost_half.hu / ghost_half.h - ghost_full.hu / ghost_full.h * 0.5).abs() < 0.01);
    }

    #[test]
    fn test_tangential_velocity_preserved() {
        let records = vec![TimeSeriesRecord {
            time: 0.0,
            state: SWEState2D::from_primitives(50.0, 0.1, 0.5),
        }];
        let ts = BoundaryTimeSeries::from_records(records).unwrap();
        let bc = NestingBC2D::new(ts);

        let ctx = make_context(50.0, 5.0, 25.0, 0.0, 0.0);
        let ghost = bc.ghost_state(&ctx);

        // Tangential velocity (v) should be from external, not interior
        // For normal (1,0), tangential is v component
        assert!((ghost.hv / ghost.h - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bc_name() {
        let ts = make_simple_ts();

        let bc_flather = NestingBC2D::new(ts.clone());
        assert_eq!(bc_flather.name(), "nesting_flather_2d");

        let bc_dirichlet = NestingBC2D::new(ts).without_flather();
        assert_eq!(bc_dirichlet.name(), "nesting_dirichlet_2d");
    }

    #[test]
    fn test_time_series_accessor() {
        let ts = make_simple_ts();
        let bc = NestingBC2D::new(ts);

        let ts_ref = bc.time_series();
        let (t_start, t_end) = ts_ref.time_range();
        assert!((t_start - 0.0).abs() < TOL);
        assert!((t_end - 100.0).abs() < TOL);
    }

    #[test]
    fn test_clamping_before_start() {
        let ts = make_simple_ts();
        let bc = NestingBC2D::new(ts).without_flather();

        // Before time range, should clamp to first value
        let ctx = make_context(60.0, 0.0, 0.0, 0.0, -50.0);
        let ghost = bc.ghost_state(&ctx);

        assert!((ghost.h - 50.0).abs() < TOL);
    }

    #[test]
    fn test_clamping_after_end() {
        let ts = make_simple_ts();
        let bc = NestingBC2D::new(ts).without_flather();

        // After time range, should clamp to last value
        let ctx = make_context(60.0, 0.0, 0.0, 0.0, 200.0);
        let ghost = bc.ghost_state(&ctx);

        assert!((ghost.h - 51.0).abs() < TOL);
    }
}
