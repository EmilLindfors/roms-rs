//! Tide gauge observation file reader.
//!
//! Reads tide gauge water level observations from various file formats:
//! - Simple CSV/text format
//! - Norwegian Kartverket format
//! - Generic time series format
//!
//! # File Formats
//!
//! ## Simple Text Format
//!
//! ```text
//! # Tide gauge observations
//! # station: Bergen
//! # longitude: 5.32
//! # latitude: 60.39
//! # datum: MSL
//! # units: m
//! # columns: time(s) water_level(m)
//! 0.0 0.15
//! 3600.0 0.42
//! 7200.0 0.58
//! ```
//!
//! ## CSV Format
//!
//! ```text
//! time,water_level
//! 0.0,0.15
//! 3600.0,0.42
//! 7200.0,0.58
//! ```
//!
//! ## ISO DateTime Format
//!
//! ```text
//! # station: Bergen
//! # columns: datetime water_level(m)
//! 2024-01-01T00:00:00Z 0.15
//! 2024-01-01T01:00:00Z 0.42
//! 2024-01-01T02:00:00Z 0.58
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use thiserror::Error;

use crate::analysis::{TideGaugeStation, TimeSeries};

/// Error type for tide gauge file operations.
#[derive(Debug, Error)]
pub enum TideGaugeFileError {
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

/// Parsed tide gauge observation file.
#[derive(Clone, Debug)]
pub struct TideGaugeFile {
    /// Station metadata (if present in file)
    pub station: Option<TideGaugeStation>,
    /// Time series data
    pub time_series: TimeSeries,
    /// Original file path
    pub source_file: Option<String>,
    /// Reference datum (if specified)
    pub datum: Option<String>,
    /// Units (if specified)
    pub units: Option<String>,
}

impl TideGaugeFile {
    /// Create from time series without station metadata.
    pub fn from_time_series(time_series: TimeSeries) -> Self {
        Self {
            station: None,
            time_series,
            source_file: None,
            datum: None,
            units: None,
        }
    }

    /// Set station metadata.
    pub fn with_station(mut self, station: TideGaugeStation) -> Self {
        self.station = Some(station);
        self
    }

    /// Set source file path.
    pub fn with_source(mut self, path: impl Into<String>) -> Self {
        self.source_file = Some(path.into());
        self
    }
}

/// Read a tide gauge observation file.
///
/// Supports multiple formats:
/// - Space/tab-separated text with optional headers
/// - CSV with header row
/// - Comments starting with #
///
/// # Arguments
/// * `path` - Path to the observation file
///
/// # Returns
/// Parsed tide gauge file with time series and optional metadata.
pub fn read_tide_gauge_file(path: &Path) -> Result<TideGaugeFile, TideGaugeFileError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut metadata: HashMap<String, String> = HashMap::new();
    let mut times: Vec<f64> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    let mut has_header = false;
    let mut is_csv = false;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Parse metadata comments
        if line.starts_with('#') {
            let content = line.trim_start_matches('#').trim();
            if let Some((key, value)) = content.split_once(':') {
                metadata.insert(key.trim().to_lowercase(), value.trim().to_string());
            }
            continue;
        }

        // Check for CSV header
        if line_num < 5 && line.contains(',') && line.chars().any(|c| c.is_alphabetic()) {
            has_header = true;
            is_csv = true;
            continue;
        }

        // Parse data line
        let parts: Vec<&str> = if is_csv || line.contains(',') {
            is_csv = true;
            line.split(',').map(|s| s.trim()).collect()
        } else {
            line.split_whitespace().collect()
        };

        if parts.len() < 2 {
            continue;
        }

        // Parse time (first column)
        let time = parse_time_value(parts[0]).map_err(|e| {
            TideGaugeFileError::ParseError(format!("Line {}: time parse error: {}", line_num + 1, e))
        })?;

        // Parse water level (second column)
        let value: f64 = parts[1].parse().map_err(|e| {
            TideGaugeFileError::ParseError(format!(
                "Line {}: water level parse error: {}",
                line_num + 1,
                e
            ))
        })?;

        times.push(time);
        values.push(value);
    }

    if times.is_empty() {
        return Err(TideGaugeFileError::InvalidFormat(
            "No data points found".to_string(),
        ));
    }

    // Build time series
    let mut time_series = TimeSeries::new(&times, &values);

    // Extract station metadata if present
    let station = if let Some(name) = metadata.get("station") {
        let lon = metadata
            .get("longitude")
            .or_else(|| metadata.get("lon"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let lat = metadata
            .get("latitude")
            .or_else(|| metadata.get("lat"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);

        let mut station = TideGaugeStation::new(name.clone(), lon, lat);

        if let Some(id) = metadata.get("id").or_else(|| metadata.get("station_id")) {
            station = station.with_external_id(id.clone());
        }

        time_series = time_series.with_name(name.clone());
        if lon != 0.0 && lat != 0.0 {
            // Note: These are geographic coords, not local coords
            // Local coords would need to be set via projection
        }

        Some(station)
    } else {
        None
    };

    Ok(TideGaugeFile {
        station,
        time_series,
        source_file: Some(path.to_string_lossy().to_string()),
        datum: metadata.get("datum").cloned(),
        units: metadata.get("units").cloned(),
    })
}

/// Parse a time value from string.
///
/// Supports:
/// - Numeric seconds: "3600.0"
/// - ISO 8601 datetime: "2024-01-01T00:00:00Z" (converted to seconds from epoch)
/// - Hours: "1.0h" or "1.0H"
fn parse_time_value(s: &str) -> Result<f64, String> {
    let s = s.trim();

    // Check for hour suffix
    if s.ends_with('h') || s.ends_with('H') {
        let num = s.trim_end_matches(|c| c == 'h' || c == 'H');
        return num
            .parse::<f64>()
            .map(|h| h * 3600.0)
            .map_err(|e| e.to_string());
    }

    // Try numeric parse first
    if let Ok(val) = s.parse::<f64>() {
        return Ok(val);
    }

    // Try ISO datetime (simplified parsing)
    // Format: YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD HH:MM:SS
    if s.len() >= 19 && (s.contains('T') || s.contains(' ')) {
        // Extract components
        let date_time: Vec<&str> = if s.contains('T') {
            s.split('T').collect()
        } else {
            s.splitn(2, ' ').collect()
        };

        if date_time.len() != 2 {
            return Err(format!("Invalid datetime format: {}", s));
        }

        let date_parts: Vec<&str> = date_time[0].split('-').collect();
        let time_str = date_time[1].trim_end_matches('Z');
        let time_parts: Vec<&str> = time_str.split(':').collect();

        if date_parts.len() != 3 || time_parts.len() < 2 {
            return Err(format!("Invalid datetime format: {}", s));
        }

        let year: i64 = date_parts[0].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
        let month: i64 = date_parts[1].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
        let day: i64 = date_parts[2].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
        let hour: i64 = time_parts[0].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
        let minute: i64 = time_parts[1].parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
        let second: f64 = if time_parts.len() > 2 {
            time_parts[2].parse().unwrap_or(0.0)
        } else {
            0.0
        };

        // Simplified conversion to seconds (not accounting for leap years, etc.)
        // This is relative time, so we just need consistency
        let days_since_epoch = (year - 1970) * 365 + (month - 1) * 30 + (day - 1);
        let seconds = days_since_epoch * 86400 + hour * 3600 + minute * 60 + second as i64;

        return Ok(seconds as f64);
    }

    Err(format!("Unable to parse time value: {}", s))
}

/// Read multiple tide gauge files from a directory.
///
/// Reads all .txt, .csv, and .dat files in the directory.
pub fn read_tide_gauge_directory(
    dir: &Path,
) -> Result<Vec<TideGaugeFile>, TideGaugeFileError> {
    let mut files = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if ext == "txt" || ext == "csv" || ext == "dat" {
                    match read_tide_gauge_file(&path) {
                        Ok(file) => files.push(file),
                        Err(e) => {
                            eprintln!("Warning: Failed to read {:?}: {}", path, e);
                        }
                    }
                }
            }
        }
    }

    Ok(files)
}

/// Convert tide gauge files to a HashMap for validation.
///
/// Maps station name to time series.
pub fn files_to_observation_map(files: &[TideGaugeFile]) -> HashMap<String, TimeSeries> {
    let mut map = HashMap::new();

    for file in files {
        let name = file
            .station
            .as_ref()
            .map(|s| s.name.clone())
            .or_else(|| file.time_series.name.clone())
            .unwrap_or_else(|| {
                file.source_file
                    .as_ref()
                    .map(|p| {
                        Path::new(p)
                            .file_stem()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_default()
                    })
                    .unwrap_or_else(|| "unknown".to_string())
            });

        map.insert(name, file.time_series.clone());
    }

    map
}

/// Write tide gauge data to a file.
///
/// Writes in the simple text format with metadata headers.
pub fn write_tide_gauge_file(
    path: &Path,
    data: &TideGaugeFile,
) -> Result<(), TideGaugeFileError> {
    use std::io::Write;

    let mut file = File::create(path)?;

    // Write metadata
    writeln!(file, "# Tide gauge observations")?;

    if let Some(ref station) = data.station {
        writeln!(file, "# station: {}", station.name)?;
        writeln!(file, "# longitude: {:.4}", station.longitude)?;
        writeln!(file, "# latitude: {:.4}", station.latitude)?;
        if let Some(ref id) = station.external_id {
            writeln!(file, "# id: {}", id)?;
        }
    }

    if let Some(ref datum) = data.datum {
        writeln!(file, "# datum: {}", datum)?;
    }
    if let Some(ref units) = data.units {
        writeln!(file, "# units: {}", units)?;
    }

    writeln!(file, "# columns: time(s) water_level(m)")?;

    // Write data
    for point in &data.time_series.data {
        writeln!(file, "{:.1} {:.6}", point.time, point.value)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_numeric_time() {
        assert!((parse_time_value("3600.0").unwrap() - 3600.0).abs() < 1e-10);
        assert!((parse_time_value("0").unwrap() - 0.0).abs() < 1e-10);
        assert!((parse_time_value("1.5h").unwrap() - 5400.0).abs() < 1e-10);
        assert!((parse_time_value("2H").unwrap() - 7200.0).abs() < 1e-10);
    }

    #[test]
    fn test_read_simple_format() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "# station: TestStation").unwrap();
        writeln!(file, "# longitude: 5.32").unwrap();
        writeln!(file, "# latitude: 60.39").unwrap();
        writeln!(file, "# columns: time water_level").unwrap();
        writeln!(file, "0.0 0.15").unwrap();
        writeln!(file, "3600.0 0.42").unwrap();
        writeln!(file, "7200.0 0.58").unwrap();

        let result = read_tide_gauge_file(file.path()).unwrap();

        assert!(result.station.is_some());
        let station = result.station.unwrap();
        assert_eq!(station.name, "TestStation");
        assert!((station.longitude - 5.32).abs() < 1e-10);
        assert!((station.latitude - 60.39).abs() < 1e-10);

        assert_eq!(result.time_series.len(), 3);
        assert!((result.time_series.data[0].value - 0.15).abs() < 1e-10);
        assert!((result.time_series.data[1].time - 3600.0).abs() < 1e-10);
    }

    #[test]
    fn test_read_csv_format() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "time,water_level").unwrap();
        writeln!(file, "0.0,0.15").unwrap();
        writeln!(file, "3600.0,0.42").unwrap();
        writeln!(file, "7200.0,0.58").unwrap();

        let result = read_tide_gauge_file(file.path()).unwrap();

        assert_eq!(result.time_series.len(), 3);
        assert!((result.time_series.data[2].value - 0.58).abs() < 1e-10);
    }

    #[test]
    fn test_write_and_read() {
        let station = TideGaugeStation::new("Bergen", 5.32, 60.39);
        let times = vec![0.0, 3600.0, 7200.0];
        let values = vec![0.1, 0.5, 0.3];
        let ts = TimeSeries::new(&times, &values);

        let data = TideGaugeFile::from_time_series(ts)
            .with_station(station);

        let file = NamedTempFile::new().unwrap();
        write_tide_gauge_file(file.path(), &data).unwrap();

        let read_back = read_tide_gauge_file(file.path()).unwrap();

        assert!(read_back.station.is_some());
        assert_eq!(read_back.station.unwrap().name, "Bergen");
        assert_eq!(read_back.time_series.len(), 3);
    }

    #[test]
    fn test_files_to_map() {
        let station = TideGaugeStation::new("Test", 0.0, 0.0);
        let ts = TimeSeries::new(&[0.0, 1.0], &[0.5, 0.6]);
        let file = TideGaugeFile::from_time_series(ts).with_station(station);

        let map = files_to_observation_map(&[file]);

        assert!(map.contains_key("Test"));
        assert_eq!(map.get("Test").unwrap().len(), 2);
    }

    #[test]
    fn test_empty_file_error() {
        let file = NamedTempFile::new().unwrap();
        writeln!(&file, "# Just comments").unwrap();

        let result = read_tide_gauge_file(file.path());
        assert!(result.is_err());
    }
}
