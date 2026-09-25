//! Parent ocean-model output (NorKyst, ROMS) as external state of a
//! characteristic open boundary.
//!
//! [`OceanModelState`] queries the parent at each boundary node's geographic
//! location and time and hands `(η, u, v)` to a
//! [`CharacteristicOBC`](crate::boundary::CharacteristicOBC), which takes the
//! incoming Riemann invariant from it and the outgoing one from the interior
//! (the nonlinear Flather condition).
//!
//! # Time base
//!
//! The reader decodes the parent's CF time to Unix seconds, and simulation
//! time `t` maps to the parent instant `clock.unix(t)`. Asking for a time
//! outside the file panics instead of freezing the forcing at the first or
//! last snapshot; check a run up front with
//! [`OceanModelState::check_time_coverage`].
//!
//! # Example
//!
//! ```ignore
//! use dg_rs::boundary::{CharacteristicOBC, OceanModelState};
//! use dg_rs::io::{LocalProjection, OceanModelReader};
//! use dg_rs::time::ModelClock;
//!
//! let reader = Arc::new(OceanModelReader::from_file("norkyst.nc")?);
//! let clock = ModelClock::parse("2024-01-30T06:00:00Z")?;
//! let parent = OceanModelState::new(reader, LocalProjection::new(63.8, 8.9), clock);
//! parent.check_time_coverage(0.0, 6.0 * 3600.0)?;
//! let bc = CharacteristicOBC::new(parent);
//! ```

use crate::boundary::{BCContext2D, ExternalState, ExternalStateProvider};
use crate::io::{CoordinateProjection, OceanModelReader};
use crate::time::ModelClock;
use std::sync::Arc;

/// Parent ocean-model output as the external state of an open boundary.
#[derive(Clone)]
pub struct OceanModelState<P: CoordinateProjection> {
    reader: Arc<OceanModelReader>,
    projection: P,
    clock: ModelClock,
    /// Added to the parent's SSH (m)
    reference_level: f64,
}

impl<P: CoordinateProjection> OceanModelState<P> {
    /// Parent output from `reader`; mesh coordinates map to geographic ones
    /// with `projection`, simulation time to UTC with `clock`.
    pub fn new(reader: Arc<OceanModelReader>, projection: P, clock: ModelClock) -> Self {
        Self {
            reader,
            projection,
            clock,
            reference_level: 0.0,
        }
    }

    /// A clock whose `t = 0` is the first snapshot of `reader`.
    pub fn first_snapshot(reader: &OceanModelReader) -> ModelClock {
        ModelClock::new(reader.time.first().copied().unwrap_or(0.0))
    }

    /// Add `level` (m) to the parent's SSH, e.g. to move it to the child's datum.
    pub fn with_reference_level(mut self, level: f64) -> Self {
        self.reference_level = level;
        self
    }

    /// The model clock.
    pub fn clock(&self) -> &ModelClock {
        &self.clock
    }

    /// Check that simulation times `[t_start, t_end]` lie within the parent
    /// file, so a run cannot hit the out-of-range panic.
    pub fn check_time_coverage(&self, t_start: f64, t_end: f64) -> Result<(), String> {
        for t in [t_start, t_end] {
            if !self.reader.covers_time(self.clock.unix(t)) {
                return Err(self.coverage_message(t));
            }
        }
        Ok(())
    }

    fn coverage_message(&self, t: f64) -> String {
        match self.simulation_time_range() {
            Some((t0, t1)) => format!(
                "OceanModelState: simulation time {t} s is outside the parent file, which \
                 covers simulation times [{t0}, {t1}] s (epoch {})",
                self.clock.format(0.0)
            ),
            None => "OceanModelState: the parent file has no time steps".to_string(),
        }
    }

    /// Time range of the parent file in Unix seconds.
    pub fn time_range(&self) -> Option<(f64, f64)> {
        self.reader.time_range()
    }

    /// Time range of the parent file in simulation time.
    pub fn simulation_time_range(&self) -> Option<(f64, f64)> {
        let (t0, t1) = self.reader.time_range()?;
        Some((self.clock.model_time(t0), self.clock.model_time(t1)))
    }

    /// Bounding box of the parent grid, `(min_lon, min_lat, max_lon, max_lat)`.
    pub fn spatial_bounds(&self) -> (f64, f64, f64, f64) {
        self.reader.bbox
    }
}

impl<P: CoordinateProjection + Send + Sync> ExternalStateProvider for OceanModelState<P> {
    /// Parent `(η, u, v)`; where the parent has no data (land, outside its
    /// grid) the interior state, i.e. zero-gradient.
    fn external_state(&self, ctx: &BCContext2D) -> ExternalState {
        let t = self.clock.unix(ctx.time);
        // Out-of-range times are a configuration error. Falling back (or
        // clamping to the nearest snapshot) would silently freeze the forcing.
        if !self.reader.covers_time(t) {
            panic!("{}", self.coverage_message(ctx.time));
        }
        let (x, y) = ctx.position;
        let (lat, lon) = self.projection.xy_to_geo(x, y);
        match self.reader.get_state_interpolated(lon, lat, t) {
            Some(parent) => {
                ExternalState::new(parent.ssh + self.reference_level, parent.u, parent.v)
            }
            None => {
                let (u, v) = ctx.interior_velocity();
                ExternalState::new(ctx.interior_surface_elevation(), u, v)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::LocalProjection;
    use crate::io::test_files::{self, CENTRE, T0, Zeta};
    use crate::solver::SWEState2D;

    /// Parent with hourly SSH 0.1, 0.3, 0.7 m, centred on the grid, on a
    /// clock starting at `epoch` (default: the first snapshot). The reader
    /// loads everything, so the temp dir can go.
    fn parent(epoch: Option<f64>) -> OceanModelState<LocalProjection> {
        let dir = tempfile::tempdir().unwrap();
        let path = test_files::write(
            dir.path(),
            Some("hours since 2024-01-30 06:00:00"),
            &[0.1, 0.3, 0.7],
            Zeta::Float,
        );
        let reader = Arc::new(OceanModelReader::from_file(path).unwrap());
        let clock = epoch.map_or_else(
            || OceanModelState::<LocalProjection>::first_snapshot(&reader),
            ModelClock::new,
        );
        OceanModelState::new(reader, LocalProjection::new(CENTRE.0, CENTRE.1), clock)
    }

    fn ssh(parent: &OceanModelState<LocalProjection>, t: f64) -> f64 {
        let interior = SWEState2D::new(50.0, 0.0, 0.0);
        let ctx = BCContext2D::new(t, (0.0, 0.0), interior, -50.0, (1.0, 0.0), 9.81, 1e-6);
        parent.external_state(&ctx).eta
    }

    #[test]
    fn first_snapshot_clock_starts_the_run_there() {
        let parent = parent(None);
        assert_eq!(parent.clock().epoch_unix, T0);
        assert!((ssh(&parent, 0.0) - 0.1).abs() < 1e-6);
        assert!((ssh(&parent, 1800.0) - 0.2).abs() < 1e-6);
        assert!((ssh(&parent, 5400.0) - 0.5).abs() < 1e-6);
        assert_eq!(parent.simulation_time_range(), Some((0.0, 7200.0)));
    }

    #[test]
    fn clock_epoch_shifts_the_parent_time() {
        let parent = parent(Some(T0 + 3600.0)).with_reference_level(0.05);
        assert!((ssh(&parent, 0.0) - 0.35).abs() < 1e-6);
        assert_eq!(parent.simulation_time_range(), Some((-3600.0, 3600.0)));
    }

    #[test]
    fn coverage_check_reports_out_of_range_runs() {
        let parent = parent(None);
        assert!(parent.check_time_coverage(0.0, 7200.0).is_ok());
        assert!(parent.check_time_coverage(0.0, 7201.0).is_err());
        assert!(parent.check_time_coverage(-1.0, 3600.0).is_err());
    }

    #[test]
    #[should_panic(expected = "outside the parent file")]
    fn forcing_past_the_last_snapshot_panics() {
        // P0.20: this used to return the last snapshot forever.
        ssh(&parent(None), 7200.0 + 60.0);
    }
}
