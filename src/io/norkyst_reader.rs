//! Reader for `norkyst-client --format text` point/site output.
//!
//! [`norkyst-client`](https://github.com/EmilLindfors/nordkyst-client) extracts
//! NorKyst-800 ocean state over OPeNDAP and, in point/site mode with
//! `--format text`, writes a plain-text dump to stdout (or a file). This module
//! parses that dump into [`TimeSeries`] so the validation harness
//! (`analysis::tide_gauge`) can compare model tides against NorKyst without a
//! NetCDF round-trip. It is `std`-only — it does **not** require the `netcdf`
//! feature.
//!
//! # Text format
//!
//! Each snapshot is one block: an optional `site_id:` line, a `time=…` line, a
//! `surface …` line carrying sea-surface height (`zeta`) and bottom depth, then
//! one `depth=…` line per vertical level:
//!
//! ```text
//! site_id: 10362
//! time=2024-01-01 00:00:00 UTC lat=63.44 lon=10.39
//! surface sea_surface_height=Some(0.42) bottom_depth=Some(120.0)
//! depth=0 temperature=Some(8.5) salinity=Some(34.1) u_current=Some(0.12) v_current=Some(-0.03)
//! depth=3 temperature=Some(8.4) salinity=Some(34.2) u_current=Some(0.10) v_current=Some(-0.02)
//! time=2024-01-01 01:00:00 UTC lat=63.44 lon=10.39
//! surface sea_surface_height=Some(0.55) bottom_depth=Some(120.0)
//! depth=0 temperature=Some(8.5) salinity=Some(34.1) u_current=Some(0.15) v_current=Some(-0.01)
//! ```
//!
//! `Option<f64>` fields are rendered with Rust's `Debug` (`Some(1.23)` / `None`).
//! Times are parsed to seconds since the Unix epoch (1970-01-01 UTC) via an exact
//! proleptic-Gregorian conversion, so the elapsed-time spacing a harmonic fit
//! needs is preserved. Lines that match none of the block keys (e.g. the
//! `Target URL: …` banner text mode prints) are ignored.
//!
//! Note: the `surface` line requires norkyst-client's text writer to emit it.
//! Older builds that omitted it yield records with `sea_surface_height = None`;
//! [`NorKystTextData::sea_surface_height_series`] then returns an empty series.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use thiserror::Error;

use crate::analysis::TimeSeries;

/// Error type for parsing norkyst-client text output.
#[derive(Debug, Error)]
pub enum NorKystTextError {
    /// File I/O error.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Parse error with line number and context.
    #[error("Parse error at line {line}: {message}")]
    ParseError {
        /// 1-based line number.
        line: usize,
        /// Human-readable description.
        message: String,
    },

    /// No time snapshots were found in the input.
    #[error("no NorKyst records found in input")]
    Empty,
}

/// A single vertical level within a snapshot.
#[derive(Clone, Copy, Debug)]
pub struct NorKystLevel {
    /// Depth below surface (m, positive down).
    pub depth: f64,
    /// Temperature (°C), if present.
    pub temperature: Option<f64>,
    /// Salinity (PSU), if present.
    pub salinity: Option<f64>,
    /// Eastward current (m/s), if present.
    pub u_current: Option<f64>,
    /// Northward current (m/s), if present.
    pub v_current: Option<f64>,
}

/// A single NorKyst snapshot at one location and time.
#[derive(Clone, Debug)]
pub struct NorKystRecord {
    /// Seconds since the Unix epoch (1970-01-01 00:00:00 UTC).
    pub time: f64,
    /// Latitude (degrees North).
    pub lat: f64,
    /// Longitude (degrees East).
    pub lon: f64,
    /// Site identifier, if the input carried one.
    pub site_id: Option<i64>,
    /// Sea-surface height / free-surface elevation `zeta` (m), if present.
    pub sea_surface_height: Option<f64>,
    /// Still-water bottom depth `h` (m, positive down), if present.
    pub bottom_depth: Option<f64>,
    /// Vertical profile, ordered as written (shallowest first in NorKyst output).
    pub profile: Vec<NorKystLevel>,
}

impl NorKystRecord {
    /// The shallowest level carrying both current components, if any.
    fn surface_level(&self) -> Option<&NorKystLevel> {
        self.profile
            .iter()
            .filter(|l| l.u_current.is_some() && l.v_current.is_some())
            .min_by(|a, b| {
                a.depth
                    .partial_cmp(&b.depth)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

/// Parsed norkyst-client text output: a sequence of snapshots.
///
/// Typical use is a single point extraction (one location, many times), but a
/// site-mode file can interleave several locations; use [`site_ids`] /
/// [`records_for_site`] to split when needed.
///
/// [`site_ids`]: NorKystTextData::site_ids
/// [`records_for_site`]: NorKystTextData::records_for_site
#[derive(Clone, Debug, Default)]
pub struct NorKystTextData {
    /// All snapshots, in file order.
    pub records: Vec<NorKystRecord>,
}

impl NorKystTextData {
    /// Number of snapshots.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether there are no snapshots.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Distinct site identifiers present, in first-seen order.
    pub fn site_ids(&self) -> Vec<i64> {
        let mut ids = Vec::new();
        for r in &self.records {
            if let Some(id) = r.site_id
                && !ids.contains(&id)
            {
                ids.push(id);
            }
        }
        ids
    }

    /// Snapshots belonging to a given site.
    pub fn records_for_site(&self, site_id: i64) -> impl Iterator<Item = &NorKystRecord> {
        self.records
            .iter()
            .filter(move |r| r.site_id == Some(site_id))
    }

    /// Sea-surface height (`zeta`) time series over all snapshots that have it.
    ///
    /// This is the series to feed the tidal validation path
    /// (`HarmonicAnalysis::fit` → `reference_constants` → catalogue compare).
    /// Assumes a single location; for multi-site input, filter with
    /// [`records_for_site`](Self::records_for_site) first.
    pub fn sea_surface_height_series(&self) -> TimeSeries {
        let (times, values): (Vec<f64>, Vec<f64>) = self
            .records
            .iter()
            .filter_map(|r| r.sea_surface_height.map(|z| (r.time, z)))
            .unzip();
        TimeSeries::new(&times, &values)
    }

    /// Surface (shallowest) current time series `(u, v)` in m/s.
    ///
    /// Each snapshot contributes its shallowest level that has both components;
    /// snapshots without currents are skipped. Useful for a first comparison
    /// against ADCP surface currents.
    pub fn surface_current_series(&self) -> (TimeSeries, TimeSeries) {
        let mut times = Vec::new();
        let mut us = Vec::new();
        let mut vs = Vec::new();
        for r in &self.records {
            if let Some(level) = r.surface_level() {
                // surface_level guarantees both are Some.
                times.push(r.time);
                us.push(level.u_current.unwrap());
                vs.push(level.v_current.unwrap());
            }
        }
        (TimeSeries::new(&times, &us), TimeSeries::new(&times, &vs))
    }
}

/// Read norkyst-client text output from a file.
pub fn read_norkyst_text_file(path: &Path) -> Result<NorKystTextData, NorKystTextError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = Vec::new();
    for line in reader.lines() {
        lines.push(line?);
    }
    parse_norkyst_text(lines.iter().map(|s| s.as_str()))
}

/// Parse norkyst-client text output from an in-memory string.
pub fn parse_norkyst_text_str(content: &str) -> Result<NorKystTextData, NorKystTextError> {
    parse_norkyst_text(content.lines())
}

/// Core parser over a sequence of lines.
fn parse_norkyst_text<'a>(
    lines: impl IntoIterator<Item = &'a str>,
) -> Result<NorKystTextData, NorKystTextError> {
    let mut records: Vec<NorKystRecord> = Vec::new();
    let mut pending_site_id: Option<i64> = None;
    let mut current: Option<NorKystRecord> = None;

    for (idx, raw) in lines.into_iter().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix("site_id:") {
            pending_site_id =
                Some(
                    rest.trim()
                        .parse()
                        .map_err(|_| NorKystTextError::ParseError {
                            line: line_no,
                            message: format!("invalid site_id: {rest:?}"),
                        })?,
                );
            continue;
        }

        if line.starts_with("time=") {
            if let Some(rec) = current.take() {
                records.push(rec);
            }
            let (time, lat, lon) = parse_time_line(line, line_no)?;
            current = Some(NorKystRecord {
                time,
                lat,
                lon,
                site_id: pending_site_id.take(),
                sea_surface_height: None,
                bottom_depth: None,
                profile: Vec::new(),
            });
            continue;
        }

        if line.starts_with("surface") && line.contains("sea_surface_height=") {
            let rec = current
                .as_mut()
                .ok_or_else(|| NorKystTextError::ParseError {
                    line: line_no,
                    message: "'surface' line before any 'time=' line".into(),
                })?;
            let (ssh, bottom) = parse_surface_line(line, line_no)?;
            rec.sea_surface_height = ssh;
            rec.bottom_depth = bottom;
            continue;
        }

        if line.starts_with("depth=") {
            let rec = current
                .as_mut()
                .ok_or_else(|| NorKystTextError::ParseError {
                    line: line_no,
                    message: "'depth=' line before any 'time=' line".into(),
                })?;
            rec.profile.push(parse_depth_line(line, line_no)?);
            continue;
        }

        // Unrecognized (banner text, comments): ignore.
    }

    if let Some(rec) = current.take() {
        records.push(rec);
    }

    if records.is_empty() {
        return Err(NorKystTextError::Empty);
    }

    Ok(NorKystTextData { records })
}

/// Parse `time=<datetime> lat=<f64> lon=<f64>`.
///
/// The datetime may contain spaces (chrono's `2024-01-01 00:00:00 UTC`), so we
/// anchor on the ` lat=` / ` lon=` markers rather than tokenizing on whitespace.
fn parse_time_line(line: &str, line_no: usize) -> Result<(f64, f64, f64), NorKystTextError> {
    let err = |message: String| NorKystTextError::ParseError {
        line: line_no,
        message,
    };

    let rest = line
        .strip_prefix("time=")
        .ok_or_else(|| err("expected 'time=' prefix".into()))?;
    let lon_pos = rest
        .rfind(" lon=")
        .ok_or_else(|| err("missing ' lon=' field".into()))?;
    let lon_str = rest[lon_pos + " lon=".len()..].trim();
    let before_lon = &rest[..lon_pos];
    let lat_pos = before_lon
        .rfind(" lat=")
        .ok_or_else(|| err("missing ' lat=' field".into()))?;
    let lat_str = before_lon[lat_pos + " lat=".len()..].trim();
    let time_str = before_lon[..lat_pos].trim();

    let time = parse_datetime_seconds(time_str)
        .map_err(|e| err(format!("invalid time {time_str:?}: {e}")))?;
    let lat = lat_str
        .parse::<f64>()
        .map_err(|_| err(format!("invalid lat {lat_str:?}")))?;
    let lon = lon_str
        .parse::<f64>()
        .map_err(|_| err(format!("invalid lon {lon_str:?}")))?;
    Ok((time, lat, lon))
}

/// Parse `surface sea_surface_height=<opt> bottom_depth=<opt>`.
fn parse_surface_line(
    line: &str,
    line_no: usize,
) -> Result<(Option<f64>, Option<f64>), NorKystTextError> {
    let mut ssh = None;
    let mut bottom = None;
    for (key, value) in kv_tokens(line) {
        match key {
            "sea_surface_height" => ssh = parse_opt_f64(value, "sea_surface_height", line_no)?,
            "bottom_depth" => bottom = parse_opt_f64(value, "bottom_depth", line_no)?,
            _ => {} // "surface" bare token or future fields
        }
    }
    Ok((ssh, bottom))
}

/// Parse `depth=<f64> temperature=<opt> salinity=<opt> u_current=<opt> v_current=<opt>`.
fn parse_depth_line(line: &str, line_no: usize) -> Result<NorKystLevel, NorKystTextError> {
    let err = |message: String| NorKystTextError::ParseError {
        line: line_no,
        message,
    };
    let mut depth = None;
    let mut temperature = None;
    let mut salinity = None;
    let mut u_current = None;
    let mut v_current = None;
    for (key, value) in kv_tokens(line) {
        match key {
            "depth" => {
                depth = Some(
                    value
                        .parse::<f64>()
                        .map_err(|_| err(format!("invalid depth {value:?}")))?,
                )
            }
            "temperature" => temperature = parse_opt_f64(value, "temperature", line_no)?,
            "salinity" => salinity = parse_opt_f64(value, "salinity", line_no)?,
            "u_current" => u_current = parse_opt_f64(value, "u_current", line_no)?,
            "v_current" => v_current = parse_opt_f64(value, "v_current", line_no)?,
            _ => {}
        }
    }
    Ok(NorKystLevel {
        depth: depth.ok_or_else(|| err("missing 'depth=' field".into()))?,
        temperature,
        salinity,
        u_current,
        v_current,
    })
}

/// Split a whitespace-delimited line into `(key, value)` pairs on the first `=`.
///
/// Only valid for lines whose values contain no spaces (surface/depth lines).
fn kv_tokens(line: &str) -> impl Iterator<Item = (&str, &str)> {
    line.split_whitespace()
        .filter_map(|tok| tok.split_once('='))
}

/// Parse an `Option<f64>` rendered by Rust `Debug`: `Some(1.23)`, `None`, or a
/// bare numeric literal (tolerant of a future plain-number format).
fn parse_opt_f64(s: &str, field: &str, line_no: usize) -> Result<Option<f64>, NorKystTextError> {
    let s = s.trim();
    if s == "None" {
        return Ok(None);
    }
    let inner = s
        .strip_prefix("Some(")
        .and_then(|rest| rest.strip_suffix(')'))
        .unwrap_or(s);
    if inner == "NaN" {
        return Ok(None);
    }
    inner
        .parse::<f64>()
        .map(Some)
        .map_err(|_| NorKystTextError::ParseError {
            line: line_no,
            message: format!("invalid {field} value {s:?}"),
        })
}

/// Convert a datetime string to seconds since the Unix epoch.
///
/// Accepts a bare number (already seconds), chrono's
/// `YYYY-MM-DD HH:MM:SS[.f] [UTC]`, or RFC3339 `YYYY-MM-DDTHH:MM:SS[.f]Z`.
/// The date→days conversion is exact proleptic Gregorian.
fn parse_datetime_seconds(s: &str) -> Result<f64, String> {
    let s = s.trim();
    if let Ok(v) = s.parse::<f64>() {
        return Ok(v);
    }

    // Strip timezone markers (only UTC is emitted).
    let s = s.trim_end_matches('Z').trim();
    let s = s.strip_suffix("UTC").map(str::trim).unwrap_or(s);

    let (date_part, time_part) = if let Some(pos) = s.find('T') {
        (&s[..pos], &s[pos + 1..])
    } else if let Some(pos) = s.find(' ') {
        (&s[..pos], &s[pos + 1..])
    } else {
        (s, "")
    };

    let d: Vec<&str> = date_part.split('-').collect();
    if d.len() != 3 {
        return Err(format!("unrecognized date {date_part:?}"));
    }
    let year: i64 = d[0].parse().map_err(|_| "bad year".to_string())?;
    let month: i64 = d[1].parse().map_err(|_| "bad month".to_string())?;
    let day: i64 = d[2].parse().map_err(|_| "bad day".to_string())?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return Err(format!("date out of range {date_part:?}"));
    }

    let (mut hour, mut minute, mut second) = (0i64, 0i64, 0.0f64);
    if !time_part.is_empty() {
        let t: Vec<&str> = time_part.split(':').collect();
        if t.len() < 2 {
            return Err(format!("unrecognized time {time_part:?}"));
        }
        hour = t[0].parse().map_err(|_| "bad hour".to_string())?;
        minute = t[1].parse().map_err(|_| "bad minute".to_string())?;
        if t.len() > 2 {
            second = t[2].parse().map_err(|_| "bad second".to_string())?;
        }
    }

    let days = days_from_civil(year, month, day);
    Ok(days as f64 * 86_400.0 + hour as f64 * 3600.0 + minute as f64 * 60.0 + second)
}

/// Days from 1970-01-01 to the given proleptic-Gregorian date.
///
/// Howard Hinnant's `days_from_civil` (public-domain), exact for all valid dates.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 }; // [0, 11]
    let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    #[test]
    fn days_from_civil_reference_points() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1970, 1, 2), 1);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        // 2000-01-01 is 30 years after the epoch, spanning leap years.
        assert_eq!(days_from_civil(2000, 1, 1), 10_957);
        // 2024-01-01
        assert_eq!(days_from_civil(2024, 1, 1), 19_723);
    }

    #[test]
    fn parse_datetime_formats_agree() {
        let secs = 19_723.0 * 86_400.0 + 3600.0; // 2024-01-01 01:00:00 UTC
        let chrono_disp = parse_datetime_seconds("2024-01-01 01:00:00 UTC").unwrap();
        let rfc3339 = parse_datetime_seconds("2024-01-01T01:00:00Z").unwrap();
        let numeric = parse_datetime_seconds(&format!("{secs}")).unwrap();
        assert!((chrono_disp - secs).abs() < TOL);
        assert!((rfc3339 - secs).abs() < TOL);
        assert!((numeric - secs).abs() < TOL);
    }

    #[test]
    fn parse_opt_f64_variants() {
        assert_eq!(parse_opt_f64("None", "x", 1).unwrap(), None);
        assert_eq!(parse_opt_f64("NaN", "x", 1).unwrap(), None);
        assert_eq!(parse_opt_f64("Some(NaN)", "x", 1).unwrap(), None);
        assert_eq!(parse_opt_f64("Some(1.5)", "x", 1).unwrap(), Some(1.5));
        assert_eq!(parse_opt_f64("-0.03", "x", 1).unwrap(), Some(-0.03));
        assert!(parse_opt_f64("Some(oops)", "x", 1).is_err());
    }

    const SAMPLE: &str = "\
Target URL: https://thredds.met.no/thredds/dodsC/norkyst
site_id: 10362
time=2024-01-01 00:00:00 UTC lat=63.44 lon=10.39
surface sea_surface_height=Some(0.42) bottom_depth=Some(120.0)
depth=0 temperature=Some(8.5) salinity=Some(34.1) u_current=Some(0.12) v_current=Some(-0.03)
depth=3 temperature=Some(8.4) salinity=Some(34.2) u_current=Some(0.10) v_current=Some(-0.02)
site_id: 10362
time=2024-01-01 01:00:00 UTC lat=63.44 lon=10.39
surface sea_surface_height=Some(0.55) bottom_depth=Some(120.0)
depth=0 temperature=Some(8.6) salinity=Some(34.0) u_current=Some(0.15) v_current=None
";

    #[test]
    fn parse_sample_block() {
        let data = parse_norkyst_text_str(SAMPLE).unwrap();
        assert_eq!(data.len(), 2);

        let r0 = &data.records[0];
        assert_eq!(r0.site_id, Some(10362));
        assert!((r0.lat - 63.44).abs() < TOL);
        assert!((r0.lon - 10.39).abs() < TOL);
        assert_eq!(r0.sea_surface_height, Some(0.42));
        assert_eq!(r0.bottom_depth, Some(120.0));
        assert_eq!(r0.profile.len(), 2);
        assert_eq!(r0.profile[0].u_current, Some(0.12));

        // One hour apart, exactly.
        let dt = data.records[1].time - data.records[0].time;
        assert!((dt - 3600.0).abs() < TOL, "dt = {dt}");
    }

    #[test]
    fn sea_surface_height_series_extracts_zeta() {
        let data = parse_norkyst_text_str(SAMPLE).unwrap();
        let ts = data.sea_surface_height_series();
        assert_eq!(ts.len(), 2);
        assert!((ts.values()[0] - 0.42).abs() < TOL);
        assert!((ts.values()[1] - 0.55).abs() < TOL);
        assert!((ts.duration() - 3600.0).abs() < TOL);
    }

    #[test]
    fn surface_current_uses_shallowest_level_and_skips_missing() {
        let data = parse_norkyst_text_str(SAMPLE).unwrap();
        let (u, v) = data.surface_current_series();
        // Second snapshot's shallowest level has v_current=None, so it is skipped.
        assert_eq!(u.len(), 1);
        assert_eq!(v.len(), 1);
        assert!((u.values()[0] - 0.12).abs() < TOL);
        assert!((v.values()[0] - (-0.03)).abs() < TOL);
    }

    #[test]
    fn site_ids_and_filtering() {
        let text = "\
site_id: 1
time=2024-01-01 00:00:00 UTC lat=60.0 lon=5.0
surface sea_surface_height=Some(0.1) bottom_depth=Some(50.0)
site_id: 2
time=2024-01-01 00:00:00 UTC lat=61.0 lon=6.0
surface sea_surface_height=Some(0.2) bottom_depth=Some(80.0)
";
        let data = parse_norkyst_text_str(text).unwrap();
        assert_eq!(data.site_ids(), vec![1, 2]);
        assert_eq!(data.records_for_site(2).count(), 1);
        assert_eq!(
            data.records_for_site(2).next().unwrap().sea_surface_height,
            Some(0.2)
        );
    }

    #[test]
    fn missing_surface_line_yields_empty_zeta_series() {
        // Simulates output from an older client that omitted the surface line.
        let text = "\
time=2024-01-01 00:00:00 UTC lat=60.0 lon=5.0
depth=0 temperature=Some(8.0) salinity=Some(34.0) u_current=Some(0.1) v_current=Some(0.2)
";
        let data = parse_norkyst_text_str(text).unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data.records[0].sea_surface_height, None);
        assert!(data.sea_surface_height_series().is_empty());
    }

    #[test]
    fn empty_input_errors() {
        assert!(matches!(
            parse_norkyst_text_str("Target URL: foo\n# nothing else"),
            Err(NorKystTextError::Empty)
        ));
    }

    #[test]
    fn malformed_time_line_reports_line_number() {
        let text = "time=not-a-date lat=x lon=5.0\n";
        let err = parse_norkyst_text_str(text).unwrap_err();
        assert!(matches!(err, NorKystTextError::ParseError { line: 1, .. }));
    }
}
