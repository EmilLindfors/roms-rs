//! Model clock: the absolute (UTC) instant of simulation time `t = 0`.
//!
//! Simulation time `t` is elapsed seconds, and everything that refers to real
//! time converts through one [`ModelClock`]:
//!
//! - boundary forcing: tidal harmonics (equilibrium argument `V₀` at the
//!   epoch, nodal `f`, `u` at the run midpoint) and parent-model output (CF
//!   time in Unix seconds);
//! - output (CF `units = "seconds since <epoch>"`);
//! - validation against observations (Unix-time series).
//!
//! # Nodal corrections
//!
//! A constituent with reference constants `(H, G)` predicts
//!
//! ```text
//! η(t) = f · H · cos(ω t + V₀ + u − G)
//! ```
//!
//! `V₀` must be the equilibrium argument at `t = 0` (it carries the phase of
//! the astronomical forcing, and `ω t` advances it exactly). The nodal terms
//! `f` and `u` change over the 18.61-year lunar node cycle and are held
//! constant over a run, so they are best taken at the middle of the run
//! (or of an analysed record) rather than at its start: over a year `u` for M2
//! drifts by up to ≈ 0.7° and `f` by ≈ 1 %. [`ModelClock::nodal_correction`]
//! does exactly that.

use crate::io::datetime::{format_utc, parse_datetime_seconds};
use crate::tides::{AstronomicalArguments, NodalCorrection, nodal_correction};

/// Julian Date of the Unix epoch (1970-01-01 00:00:00 UTC).
const UNIX_EPOCH_JD: f64 = 2_440_587.5;

/// Maps simulation time `t` (seconds) to UTC: `t = 0` is `epoch_unix`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelClock {
    /// Unix time (seconds since 1970-01-01 00:00:00 UTC) of `t = 0`.
    pub epoch_unix: f64,
}

impl ModelClock {
    /// A clock whose `t = 0` is the Unix time `epoch_unix`.
    pub fn new(epoch_unix: f64) -> Self {
        assert!(epoch_unix.is_finite(), "non-finite model epoch");
        Self { epoch_unix }
    }

    /// A clock starting at a UTC date and time, e.g. `"2025-06-01T00:00:00Z"`
    /// or `"2025-06-01 00:00:00"` (RFC 3339 offsets are applied).
    pub fn parse(epoch: &str) -> Result<Self, String> {
        parse_datetime_seconds(epoch).map(Self::new)
    }

    /// A clock starting at a UTC calendar instant.
    pub fn at_datetime(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> Self {
        let s = format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:00");
        Self::parse(&s).unwrap_or_else(|e| panic!("invalid model epoch {s}: {e}"))
    }

    /// Unix time of simulation time `t`.
    #[inline]
    pub fn unix(&self, t: f64) -> f64 {
        self.epoch_unix + t
    }

    /// Simulation time of the Unix time `unix`.
    #[inline]
    pub fn model_time(&self, unix: f64) -> f64 {
        unix - self.epoch_unix
    }

    /// Julian Date (UTC) of simulation time `t`.
    pub fn julian_date(&self, t: f64) -> f64 {
        UNIX_EPOCH_JD + self.unix(t) / 86_400.0
    }

    /// Astronomical arguments at simulation time `t`.
    pub fn astronomical_arguments(&self, t: f64) -> AstronomicalArguments {
        AstronomicalArguments::at_julian_date(self.julian_date(t))
    }

    /// Nodal correction of constituent `name` for a run starting at `t = 0`:
    /// the equilibrium argument `V₀` at `t = 0` and the nodal factor `f` and
    /// phase `u` at `t_mid` (the middle of the run or record). `None` for a
    /// constituent without tabulated corrections.
    pub fn nodal_correction(&self, name: &str, t_mid: f64) -> Option<NodalCorrection> {
        let at_epoch = nodal_correction(name, &self.astronomical_arguments(0.0))?;
        let at_mid = nodal_correction(name, &self.astronomical_arguments(t_mid))?;
        Some(NodalCorrection {
            f: at_mid.f,
            u_deg: at_mid.u_deg,
            v0_deg: at_epoch.v0_deg,
        })
    }

    /// `YYYY-MM-DD HH:MM:SS` (UTC) of simulation time `t`.
    pub fn format(&self, t: f64) -> String {
        format_utc(self.unix(t))
    }

    /// CF time units for output in simulation seconds:
    /// `"seconds since YYYY-MM-DD HH:MM:SS"`.
    pub fn cf_time_units(&self) -> String {
        format!("seconds since {}", self.format(0.0))
    }
}

impl Default for ModelClock {
    /// `t = 0` at the Unix epoch, so simulation time is Unix time.
    fn default() -> Self {
        Self::new(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tides::julian_date;

    #[test]
    fn unix_and_model_time_are_inverse() {
        let clock = ModelClock::parse("2025-06-01T00:00:00Z").unwrap();
        assert_eq!(clock.epoch_unix, 1_748_736_000.0);
        assert_eq!(clock.unix(3600.0), 1_748_739_600.0);
        assert_eq!(clock.model_time(clock.unix(123.5)), 123.5);
        assert_eq!(clock, ModelClock::at_datetime(2025, 6, 1, 0, 0));
    }

    #[test]
    fn julian_date_matches_calendar() {
        let clock = ModelClock::at_datetime(2024, 1, 30, 6, 0);
        let jd = julian_date(2024, 1, 30, 7, 0, 0.0);
        assert!((clock.julian_date(3600.0) - jd).abs() < 1e-9);
    }

    #[test]
    fn cf_units_name_the_epoch() {
        let clock = ModelClock::parse("2025-06-01 12:30:00").unwrap();
        assert_eq!(clock.cf_time_units(), "seconds since 2025-06-01 12:30:00");
        assert_eq!(clock.format(90.0), "2025-06-01 12:31:30");
    }

    /// V₀ comes from the epoch and f, u from the midpoint, so shifting the
    /// midpoint changes f and u but not V₀.
    #[test]
    fn nodal_correction_takes_v0_at_epoch_and_f_u_at_midpoint() {
        let clock = ModelClock::at_datetime(2025, 1, 1, 0, 0);
        let year = 365.25 * 86_400.0;
        let at_start = clock.nodal_correction("M2", 0.0).unwrap();
        let mid_year = clock.nodal_correction("M2", 0.5 * year).unwrap();
        assert_eq!(at_start.v0_deg, mid_year.v0_deg);
        assert_eq!(
            at_start,
            nodal_correction("M2", &clock.astronomical_arguments(0.0)).unwrap()
        );
        let later = nodal_correction("M2", &clock.astronomical_arguments(0.5 * year)).unwrap();
        assert_eq!((mid_year.f, mid_year.u_deg), (later.f, later.u_deg));
        // Half a year of the nodal cycle moves u (M2) measurably
        assert!((mid_year.u_deg - at_start.u_deg).abs() > 0.1);
        assert!(clock.nodal_correction("XX9", 0.0).is_none());
    }
}
