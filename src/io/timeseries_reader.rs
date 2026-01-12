//! Reader for boundary time series files (nesting data).
//!
//! Parses time series files containing state data from parent models
//! for use in nesting boundary conditions.
//!
//! # File Format
//!
//! ```text
//! # Nesting data from parent model
//! # location: 5.0 60.0
//! # columns: time(s) h(m) hu(m2/s) hv(m2/s)
//! 0.0 10.5 1.2 0.3
//! 3600.0 10.8 1.1 0.4
//! 7200.0 10.3 0.9 0.2
//! ```
//!
//! Time values must be monotonically increasing.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use thiserror::Error;

use crate::solver::SWEState2D;

/// Error type for time series file parsing.
#[derive(Debug, Error)]
pub enum TimeSeriesFileError {
    /// File I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Parse error with line number
    #[error("Parse error at line {line}: {message}")]
    ParseError { line: usize, message: String },

    /// Empty file (no data records)
    #[error("Time series file contains no data")]
    EmptyFile,

    /// Non-monotonic time values
    #[error("Non-monotonic time at line {line}")]
    NonMonotonic { line: usize },
}

/// A single time series record.
#[derive(Clone, Copy, Debug)]
pub struct TimeSeriesRecord {
    /// Time in seconds
    pub time: f64,
    /// SWE state (h, hu, hv) at this time
    pub state: SWEState2D,
}

impl TimeSeriesRecord {
    /// Create a new record.
    pub fn new(time: f64, state: SWEState2D) -> Self {
        Self { time, state }
    }

    /// Create from primitive variables.
    pub fn from_primitives(time: f64, h: f64, u: f64, v: f64) -> Self {
        Self {
            time,
            state: SWEState2D::from_primitives(h, u, v),
        }
    }
}

/// Boundary time series data with interpolation capability.
///
/// Stores a sequence of state records at discrete times and provides
/// linear interpolation between them.
#[derive(Clone, Debug)]
pub struct BoundaryTimeSeries {
    /// Optional location (x, y) for this time series
    pub location: Option<(f64, f64)>,
    /// Time series records sorted by time
    records: Vec<TimeSeriesRecord>,
}

impl BoundaryTimeSeries {
    /// Create a new boundary time series from records.
    ///
    /// Records should be sorted by time (this is not verified).
    pub fn new(records: Vec<TimeSeriesRecord>) -> Self {
        Self {
            location: None,
            records,
        }
    }

    /// Create a boundary time series from records with validation.
    ///
    /// # Errors
    /// - `EmptyFile` if no records provided
    /// - `NonMonotonic` if time values are not strictly increasing
    pub fn from_records(records: Vec<TimeSeriesRecord>) -> Result<Self, TimeSeriesFileError> {
        if records.is_empty() {
            return Err(TimeSeriesFileError::EmptyFile);
        }

        // Check monotonicity
        for i in 1..records.len() {
            if records[i].time <= records[i - 1].time {
                return Err(TimeSeriesFileError::NonMonotonic { line: i + 1 });
            }
        }

        Ok(Self {
            location: None,
            records,
        })
    }

    /// Create an empty time series.
    pub fn empty() -> Self {
        Self {
            location: None,
            records: Vec::new(),
        }
    }

    /// Set the location.
    pub fn with_location(mut self, x: f64, y: f64) -> Self {
        self.location = Some((x, y));
        self
    }

    /// Add a record (must maintain time ordering).
    pub fn push(&mut self, record: TimeSeriesRecord) {
        self.records.push(record);
    }

    /// Get number of records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Get the time range covered by this series.
    ///
    /// Returns (t_start, t_end) or (0, 0) if empty.
    pub fn time_range(&self) -> (f64, f64) {
        if self.records.is_empty() {
            return (0.0, 0.0);
        }
        (
            self.records.first().unwrap().time,
            self.records.last().unwrap().time,
        )
    }

    /// Get duration of the time series.
    pub fn duration(&self) -> f64 {
        let (t0, t1) = self.time_range();
        t1 - t0
    }

    /// Access the underlying records.
    pub fn records(&self) -> &[TimeSeriesRecord] {
        &self.records
    }

    /// Interpolate state at time t using linear interpolation.
    ///
    /// - If t < first time: returns first state (clamped)
    /// - If t > last time: returns last state (clamped)
    /// - Otherwise: linear interpolation between bracketing records
    ///
    /// # Returns
    /// The interpolated SWEState2D, or zero state if empty.
    pub fn interpolate(&self, t: f64) -> SWEState2D {
        if self.records.is_empty() {
            return SWEState2D::zero();
        }

        // Handle before first record (clamp)
        if t <= self.records[0].time {
            return self.records[0].state;
        }

        // Handle after last record (clamp)
        let last = self.records.last().unwrap();
        if t >= last.time {
            return last.state;
        }

        // Binary search for bracketing interval
        let idx = self
            .records
            .binary_search_by(|r| r.time.partial_cmp(&t).unwrap())
            .unwrap_or_else(|i| i.saturating_sub(1));

        // Ensure we have a valid interval
        if idx + 1 >= self.records.len() {
            return self.records[idx].state;
        }

        let r0 = &self.records[idx];
        let r1 = &self.records[idx + 1];

        // Linear interpolation factor
        let dt = r1.time - r0.time;
        let alpha = if dt > 1e-14 { (t - r0.time) / dt } else { 0.0 };

        // Interpolate each component
        SWEState2D {
            h: r0.state.h + alpha * (r1.state.h - r0.state.h),
            hu: r0.state.hu + alpha * (r1.state.hu - r0.state.hu),
            hv: r0.state.hv + alpha * (r1.state.hv - r0.state.hv),
        }
    }

    /// Interpolate and return primitive variables (h, u, v).
    ///
    /// Uses velocity desingularization for shallow depths.
    pub fn interpolate_primitives(&self, t: f64, h_min: f64) -> (f64, f64, f64) {
        let state = self.interpolate(t);
        let (u, v) = state.velocity_simple(h_min);
        (state.h, u, v)
    }

    /// Get the state at a specific index.
    pub fn get(&self, index: usize) -> Option<&TimeSeriesRecord> {
        self.records.get(index)
    }

    /// Check if time t is within the covered range.
    pub fn contains_time(&self, t: f64) -> bool {
        let (t0, t1) = self.time_range();
        t >= t0 && t <= t1
    }
}

/// Read a time series file.
///
/// # Arguments
/// * `path` - Path to the time series file
///
/// # Returns
/// * `Ok(BoundaryTimeSeries)` - Parsed time series data
/// * `Err(TimeSeriesFileError)` - If reading or parsing fails
///
/// # Example
///
/// ```ignore
/// use dg::io::read_timeseries_file;
/// use std::path::Path;
///
/// let ts = read_timeseries_file(Path::new("parent_bc.txt"))?;
/// let (t0, t1) = ts.time_range();
/// println!("Time series from {} to {} seconds", t0, t1);
///
/// // Interpolate at specific time
/// let state = ts.interpolate(1800.0);
/// println!("h = {}, hu = {}, hv = {}", state.h, state.hu, state.hv);
/// ```
pub fn read_timeseries_file(path: &Path) -> Result<BoundaryTimeSeries, TimeSeriesFileError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut location: Option<(f64, f64)> = None;
    let mut records = Vec::new();
    let mut last_time: Option<f64> = None;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        // Parse metadata comments
        if line.starts_with('#') {
            let comment = line.trim_start_matches('#').trim();
            if let Some(loc_str) = comment.strip_prefix("location:") {
                let parts: Vec<&str> = loc_str.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let (Ok(x), Ok(y)) = (parts[0].parse(), parts[1].parse()) {
                        location = Some((x, y));
                    }
                }
            }
            continue;
        }

        // Parse data line: time h hu hv
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(TimeSeriesFileError::ParseError {
                line: line_num + 1,
                message: "Expected: time h hu hv".into(),
            });
        }

        let time: f64 = parts[0]
            .parse()
            .map_err(|_| TimeSeriesFileError::ParseError {
                line: line_num + 1,
                message: "Invalid time value".into(),
            })?;
        let h: f64 = parts[1]
            .parse()
            .map_err(|_| TimeSeriesFileError::ParseError {
                line: line_num + 1,
                message: "Invalid h value".into(),
            })?;
        let hu: f64 = parts[2]
            .parse()
            .map_err(|_| TimeSeriesFileError::ParseError {
                line: line_num + 1,
                message: "Invalid hu value".into(),
            })?;
        let hv: f64 = parts[3]
            .parse()
            .map_err(|_| TimeSeriesFileError::ParseError {
                line: line_num + 1,
                message: "Invalid hv value".into(),
            })?;

        // Check monotonicity
        if let Some(prev_time) = last_time {
            if time <= prev_time {
                return Err(TimeSeriesFileError::NonMonotonic { line: line_num + 1 });
            }
        }
        last_time = Some(time);

        records.push(TimeSeriesRecord {
            time,
            state: SWEState2D::new(h, hu, hv),
        });
    }

    if records.is_empty() {
        return Err(TimeSeriesFileError::EmptyFile);
    }

    Ok(BoundaryTimeSeries { location, records })
}

/// Parse time series from a string.
///
/// Same format as file, useful for testing or embedded data.
pub fn parse_timeseries(content: &str) -> Result<BoundaryTimeSeries, TimeSeriesFileError> {
    let mut location: Option<(f64, f64)> = None;
    let mut records = Vec::new();
    let mut last_time: Option<f64> = None;

    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        if line.starts_with('#') {
            let comment = line.trim_start_matches('#').trim();
            if let Some(loc_str) = comment.strip_prefix("location:") {
                let parts: Vec<&str> = loc_str.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let (Ok(x), Ok(y)) = (parts[0].parse(), parts[1].parse()) {
                        location = Some((x, y));
                    }
                }
            }
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(TimeSeriesFileError::ParseError {
                line: line_num + 1,
                message: "Expected: time h hu hv".into(),
            });
        }

        let time: f64 = parts[0]
            .parse()
            .map_err(|_| TimeSeriesFileError::ParseError {
                line: line_num + 1,
                message: "Invalid time".into(),
            })?;
        let h: f64 = parts[1]
            .parse()
            .map_err(|_| TimeSeriesFileError::ParseError {
                line: line_num + 1,
                message: "Invalid h".into(),
            })?;
        let hu: f64 = parts[2]
            .parse()
            .map_err(|_| TimeSeriesFileError::ParseError {
                line: line_num + 1,
                message: "Invalid hu".into(),
            })?;
        let hv: f64 = parts[3]
            .parse()
            .map_err(|_| TimeSeriesFileError::ParseError {
                line: line_num + 1,
                message: "Invalid hv".into(),
            })?;

        if let Some(prev) = last_time {
            if time <= prev {
                return Err(TimeSeriesFileError::NonMonotonic { line: line_num + 1 });
            }
        }
        last_time = Some(time);

        records.push(TimeSeriesRecord::new(time, SWEState2D::new(h, hu, hv)));
    }

    if records.is_empty() {
        return Err(TimeSeriesFileError::EmptyFile);
    }

    Ok(BoundaryTimeSeries { location, records })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_parse_simple_timeseries() {
        let content = "0.0 10.0 1.0 0.5\n100.0 12.0 2.0 1.0";
        let ts = parse_timeseries(content).unwrap();

        assert_eq!(ts.len(), 2);
        assert!((ts.records[0].time - 0.0).abs() < TOL);
        assert!((ts.records[0].state.h - 10.0).abs() < TOL);
        assert!((ts.records[1].time - 100.0).abs() < TOL);
    }

    #[test]
    fn test_parse_with_location() {
        let content = "# location: 5.0 60.0\n0.0 10.0 1.0 0.5";
        let ts = parse_timeseries(content).unwrap();

        assert_eq!(ts.location, Some((5.0, 60.0)));
    }

    #[test]
    fn test_parse_with_comments() {
        let content = r#"
# Nesting data
# location: 5.0 60.0
# columns: time h hu hv
0.0 10.0 1.0 0.5
100.0 12.0 2.0 1.0
"#;
        let ts = parse_timeseries(content).unwrap();
        assert_eq!(ts.len(), 2);
    }

    #[test]
    fn test_parse_empty_error() {
        let content = "# just comments\n# no data";
        let result = parse_timeseries(content);
        assert!(matches!(result, Err(TimeSeriesFileError::EmptyFile)));
    }

    #[test]
    fn test_parse_non_monotonic_error() {
        let content = "0.0 10.0 1.0 0.5\n0.0 12.0 2.0 1.0"; // Same time
        let result = parse_timeseries(content);
        assert!(matches!(
            result,
            Err(TimeSeriesFileError::NonMonotonic { .. })
        ));

        let content2 = "100.0 10.0 1.0 0.5\n50.0 12.0 2.0 1.0"; // Decreasing
        let result2 = parse_timeseries(content2);
        assert!(matches!(
            result2,
            Err(TimeSeriesFileError::NonMonotonic { .. })
        ));
    }

    #[test]
    fn test_parse_missing_fields() {
        let content = "0.0 10.0 1.0"; // Missing hv
        let result = parse_timeseries(content);
        assert!(matches!(
            result,
            Err(TimeSeriesFileError::ParseError { .. })
        ));
    }

    #[test]
    fn test_interpolate_exact_times() {
        let records = vec![
            TimeSeriesRecord::new(0.0, SWEState2D::new(10.0, 1.0, 0.5)),
            TimeSeriesRecord::new(100.0, SWEState2D::new(12.0, 2.0, 1.0)),
        ];
        let ts = BoundaryTimeSeries::new(records);

        let s0 = ts.interpolate(0.0);
        assert!((s0.h - 10.0).abs() < TOL);
        assert!((s0.hu - 1.0).abs() < TOL);

        let s1 = ts.interpolate(100.0);
        assert!((s1.h - 12.0).abs() < TOL);
    }

    #[test]
    fn test_interpolate_midpoint() {
        let records = vec![
            TimeSeriesRecord::new(0.0, SWEState2D::new(10.0, 0.0, 0.0)),
            TimeSeriesRecord::new(100.0, SWEState2D::new(20.0, 10.0, 5.0)),
        ];
        let ts = BoundaryTimeSeries::new(records);

        let s = ts.interpolate(50.0);
        assert!((s.h - 15.0).abs() < TOL);
        assert!((s.hu - 5.0).abs() < TOL);
        assert!((s.hv - 2.5).abs() < TOL);
    }

    #[test]
    fn test_interpolate_clamp_before() {
        let records = vec![
            TimeSeriesRecord::new(100.0, SWEState2D::new(10.0, 1.0, 0.5)),
            TimeSeriesRecord::new(200.0, SWEState2D::new(12.0, 2.0, 1.0)),
        ];
        let ts = BoundaryTimeSeries::new(records);

        // Before first time should clamp to first value
        let s = ts.interpolate(0.0);
        assert!((s.h - 10.0).abs() < TOL);
    }

    #[test]
    fn test_interpolate_clamp_after() {
        let records = vec![
            TimeSeriesRecord::new(0.0, SWEState2D::new(10.0, 1.0, 0.5)),
            TimeSeriesRecord::new(100.0, SWEState2D::new(12.0, 2.0, 1.0)),
        ];
        let ts = BoundaryTimeSeries::new(records);

        // After last time should clamp to last value
        let s = ts.interpolate(1000.0);
        assert!((s.h - 12.0).abs() < TOL);
    }

    #[test]
    fn test_interpolate_empty() {
        let ts = BoundaryTimeSeries::empty();
        let s = ts.interpolate(50.0);
        assert!((s.h - 0.0).abs() < TOL);
    }

    #[test]
    fn test_time_range() {
        let records = vec![
            TimeSeriesRecord::new(10.0, SWEState2D::zero()),
            TimeSeriesRecord::new(50.0, SWEState2D::zero()),
            TimeSeriesRecord::new(100.0, SWEState2D::zero()),
        ];
        let ts = BoundaryTimeSeries::new(records);

        let (t0, t1) = ts.time_range();
        assert!((t0 - 10.0).abs() < TOL);
        assert!((t1 - 100.0).abs() < TOL);
    }

    #[test]
    fn test_duration() {
        let records = vec![
            TimeSeriesRecord::new(10.0, SWEState2D::zero()),
            TimeSeriesRecord::new(100.0, SWEState2D::zero()),
        ];
        let ts = BoundaryTimeSeries::new(records);
        assert!((ts.duration() - 90.0).abs() < TOL);
    }

    #[test]
    fn test_contains_time() {
        let records = vec![
            TimeSeriesRecord::new(10.0, SWEState2D::zero()),
            TimeSeriesRecord::new(100.0, SWEState2D::zero()),
        ];
        let ts = BoundaryTimeSeries::new(records);

        assert!(!ts.contains_time(5.0));
        assert!(ts.contains_time(10.0));
        assert!(ts.contains_time(50.0));
        assert!(ts.contains_time(100.0));
        assert!(!ts.contains_time(150.0));
    }

    #[test]
    fn test_interpolate_primitives() {
        let records = vec![
            TimeSeriesRecord::new(0.0, SWEState2D::new(10.0, 20.0, 10.0)), // h=10, hu=20, hv=10
            TimeSeriesRecord::new(100.0, SWEState2D::new(10.0, 20.0, 10.0)),
        ];
        let ts = BoundaryTimeSeries::new(records);

        let (h, u, v) = ts.interpolate_primitives(50.0, 1e-6);
        assert!((h - 10.0).abs() < TOL);
        assert!((u - 2.0).abs() < TOL); // u = hu/h = 20/10 = 2
        assert!((v - 1.0).abs() < TOL); // v = hv/h = 10/10 = 1
    }

    #[test]
    fn test_read_timeseries_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "# location: 5.0 60.0").unwrap();
        writeln!(file, "0.0 10.0 1.0 0.5").unwrap();
        writeln!(file, "3600.0 11.0 1.5 0.7").unwrap();
        writeln!(file, "7200.0 10.5 1.2 0.6").unwrap();

        let ts = read_timeseries_file(file.path()).unwrap();

        assert_eq!(ts.location, Some((5.0, 60.0)));
        assert_eq!(ts.len(), 3);
        assert!((ts.duration() - 7200.0).abs() < TOL);
    }

    #[test]
    fn test_many_records_interpolation() {
        // Test with many records to verify binary search
        let records: Vec<_> = (0..100)
            .map(|i| {
                let t = i as f64 * 10.0;
                let h = 10.0 + (t / 100.0);
                TimeSeriesRecord::new(t, SWEState2D::new(h, 0.0, 0.0))
            })
            .collect();
        let ts = BoundaryTimeSeries::new(records);

        // Test interpolation at various points
        for t in [55.0, 123.0, 456.0, 789.0, 950.0] {
            let s = ts.interpolate(t);
            let expected_h = 10.0 + (t.min(990.0) / 100.0);
            assert!(
                (s.h - expected_h).abs() < 0.2,
                "t={}, h={}, expected={}",
                t,
                s.h,
                expected_h
            );
        }
    }
}
