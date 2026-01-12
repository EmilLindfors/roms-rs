//! ADCP current observation file reader.
//!
//! Reads ADCP velocity observations from various file formats:
//! - Simple CSV/text format with u, v components
//! - Generic time series format
//!
//! # File Formats
//!
//! ## Simple Text Format
//!
//! ```text
//! # ADCP current observations
//! # station: Froya
//! # longitude: 8.85
//! # latitude: 63.75
//! # depth: 25.0
//! # units: m/s
//! # columns: time(s) u(m/s) v(m/s)
//! 0.0 0.15 0.08
//! 3600.0 0.22 0.12
//! 7200.0 0.18 0.05
//! ```
//!
//! ## CSV Format
//!
//! ```text
//! time,u,v
//! 0.0,0.15,0.08
//! 3600.0,0.22,0.12
//! 7200.0,0.18,0.05
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use thiserror::Error;

use crate::analysis::{ADCPStation, CurrentTimeSeries};

/// Error type for ADCP file operations.
#[derive(Debug, Error)]
pub enum ADCPFileError {
    /// IO error reading file
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Parse error in file content
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Missing required metadata
    #[error("Missing metadata: {0}")]
    MissingMetadata(String),

    /// Invalid file format
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
}

/// Parsed ADCP observation file.
#[derive(Clone, Debug)]
pub struct ADCPFile {
    /// Station metadata
    pub station: ADCPStation,
    /// Current time series
    pub time_series: CurrentTimeSeries,
    /// Additional metadata from file comments
    pub metadata: HashMap<String, String>,
}

impl ADCPFile {
    /// Get number of observations.
    pub fn len(&self) -> usize {
        self.time_series.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.time_series.is_empty()
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f64 {
        self.time_series.duration()
    }
}

/// Read an ADCP file in simple text format.
///
/// # Arguments
/// * `path` - Path to the ADCP data file
///
/// # Returns
/// Parsed ADCP file with station metadata and time series
pub fn read_adcp_file(path: &Path) -> Result<ADCPFile, ADCPFileError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut metadata: HashMap<String, String> = HashMap::new();
    let mut times: Vec<f64> = Vec::new();
    let mut u_values: Vec<f64> = Vec::new();
    let mut v_values: Vec<f64> = Vec::new();
    let mut is_csv = false;
    let mut has_header = false;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        // Parse comments/metadata
        if line.starts_with('#') {
            let content = line[1..].trim();
            if let Some((key, value)) = content.split_once(':') {
                metadata.insert(key.trim().to_lowercase(), value.trim().to_string());
            }
            continue;
        }

        // Detect CSV format
        if line_num == 0 && line.contains(',') && !line.chars().next().unwrap().is_ascii_digit() {
            is_csv = true;
            has_header = true;
            continue;
        }

        // Skip CSV header
        if is_csv && !has_header && line.contains(',') && line.starts_with("time") {
            has_header = true;
            continue;
        }

        // Parse data line
        let parts: Vec<&str> = if is_csv || line.contains(',') {
            line.split(',').map(|s| s.trim()).collect()
        } else {
            line.split_whitespace().collect()
        };

        if parts.len() < 3 {
            return Err(ADCPFileError::ParseError(format!(
                "Line {} needs at least 3 columns (time, u, v): {}",
                line_num + 1,
                line
            )));
        }

        let time: f64 = parts[0].parse().map_err(|_| {
            ADCPFileError::ParseError(format!("Invalid time at line {}: {}", line_num + 1, parts[0]))
        })?;

        let u: f64 = parts[1].parse().map_err(|_| {
            ADCPFileError::ParseError(format!(
                "Invalid u-velocity at line {}: {}",
                line_num + 1,
                parts[1]
            ))
        })?;

        let v: f64 = parts[2].parse().map_err(|_| {
            ADCPFileError::ParseError(format!(
                "Invalid v-velocity at line {}: {}",
                line_num + 1,
                parts[2]
            ))
        })?;

        times.push(time);
        u_values.push(u);
        v_values.push(v);
    }

    if times.is_empty() {
        return Err(ADCPFileError::InvalidFormat(
            "No data records found in file".to_string(),
        ));
    }

    // Extract station info from metadata
    let station_name = metadata
        .get("station")
        .cloned()
        .unwrap_or_else(|| "Unknown".to_string());

    let longitude: f64 = metadata
        .get("longitude")
        .or_else(|| metadata.get("lon"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    let latitude: f64 = metadata
        .get("latitude")
        .or_else(|| metadata.get("lat"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    let depth: Option<f64> = metadata.get("depth").and_then(|s| s.parse().ok());

    let water_depth: Option<f64> = metadata
        .get("water_depth")
        .or_else(|| metadata.get("waterdepth"))
        .and_then(|s| s.parse().ok());

    let mut station = ADCPStation::new(station_name, longitude, latitude);
    if let Some(d) = depth {
        station = station.with_depth(d);
    }
    if let Some(wd) = water_depth {
        station = station.with_water_depth(wd);
    }

    let time_series = CurrentTimeSeries::new(&times, &u_values, &v_values)
        .with_name(station.name.clone());

    Ok(ADCPFile {
        station,
        time_series,
        metadata,
    })
}

/// Write an ADCP file in simple text format.
///
/// # Arguments
/// * `path` - Output file path
/// * `station` - Station metadata
/// * `time_series` - Current time series to write
pub fn write_adcp_file(
    path: &Path,
    station: &ADCPStation,
    time_series: &CurrentTimeSeries,
) -> Result<(), ADCPFileError> {
    let file = File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);

    // Write metadata
    writeln!(writer, "# ADCP current observations")?;
    writeln!(writer, "# station: {}", station.name)?;
    writeln!(writer, "# longitude: {}", station.longitude)?;
    writeln!(writer, "# latitude: {}", station.latitude)?;
    if let Some(depth) = station.depth {
        writeln!(writer, "# depth: {}", depth)?;
    }
    if let Some(water_depth) = station.water_depth {
        writeln!(writer, "# water_depth: {}", water_depth)?;
    }
    writeln!(writer, "# units: m/s")?;
    writeln!(writer, "# columns: time(s) u(m/s) v(m/s)")?;

    // Write data
    for point in &time_series.data {
        writeln!(writer, "{} {} {}", point.time, point.u, point.v)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_simple_format() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"# ADCP observations
# station: TestStation
# longitude: 8.5
# latitude: 63.5
# depth: 25.0
# units: m/s
0.0 0.1 0.05
3600.0 0.15 0.08
7200.0 0.12 0.03"#
        )
        .unwrap();

        let adcp = read_adcp_file(file.path()).unwrap();

        assert_eq!(adcp.station.name, "TestStation");
        assert_eq!(adcp.station.longitude, 8.5);
        assert_eq!(adcp.station.latitude, 63.5);
        assert_eq!(adcp.station.depth, Some(25.0));
        assert_eq!(adcp.len(), 3);
        assert!((adcp.duration() - 7200.0).abs() < 1e-10);
    }

    #[test]
    fn test_read_csv_format() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"time,u,v
0.0,0.1,0.05
3600.0,0.15,0.08
7200.0,0.12,0.03"#
        )
        .unwrap();

        let adcp = read_adcp_file(file.path()).unwrap();

        assert_eq!(adcp.len(), 3);
        assert!((adcp.time_series.data[0].u - 0.1).abs() < 1e-10);
        assert!((adcp.time_series.data[0].v - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_write_and_read_roundtrip() {
        let station = ADCPStation::new("RoundtripTest", 9.0, 64.0)
            .with_depth(30.0)
            .with_water_depth(80.0);

        let times = vec![0.0, 3600.0, 7200.0];
        let u = vec![0.2, 0.25, 0.18];
        let v = vec![0.1, 0.12, 0.08];
        let ts = CurrentTimeSeries::new(&times, &u, &v);

        let file = NamedTempFile::new().unwrap();
        write_adcp_file(file.path(), &station, &ts).unwrap();

        let read_back = read_adcp_file(file.path()).unwrap();

        assert_eq!(read_back.station.name, "RoundtripTest");
        assert_eq!(read_back.station.longitude, 9.0);
        assert_eq!(read_back.station.latitude, 64.0);
        assert_eq!(read_back.station.depth, Some(30.0));
        assert_eq!(read_back.len(), 3);

        for (i, point) in read_back.time_series.data.iter().enumerate() {
            assert!((point.time - times[i]).abs() < 1e-10);
            assert!((point.u - u[i]).abs() < 1e-10);
            assert!((point.v - v[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_empty_file_error() {
        let file = NamedTempFile::new().unwrap();
        writeln!(&file, "# Just comments").unwrap();

        let result = read_adcp_file(file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_data_error() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "0.0 invalid 0.1").unwrap();

        let result = read_adcp_file(file.path());
        assert!(result.is_err());
    }
}
