//! Reader for tidal constituent files.
//!
//! Parses simple text files containing tidal harmonic constituents
//! (amplitude and phase) for use in boundary conditions.
//!
//! # File Format
//!
//! ```text
//! # Tidal constituents for Bergen
//! # location: 5.32 60.39
//! # reference_level: 0.0
//! # columns: name amplitude(m) phase(deg)
//! M2 0.45 125.3
//! S2 0.15 158.7
//! K1 0.08 45.2
//! O1 0.06 67.8
//! N2 0.09 112.4
//! ```
//!
//! Lines starting with `#` are comments. Metadata can be specified
//! in comments with `key: value` format.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use thiserror::Error;

/// Error type for constituent file parsing.
#[derive(Debug, Error)]
pub enum ConstituentFileError {
    /// File I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Parse error with line number
    #[error("Parse error at line {line}: {message}")]
    ParseError { line: usize, message: String },

    /// Unknown constituent name
    #[error("Unknown constituent: {0}")]
    UnknownConstituent(String),
}

/// A single tidal constituent entry from a file.
#[derive(Clone, Debug)]
pub struct ConstituentEntry {
    /// Constituent name (e.g., "M2", "S2", "K1")
    pub name: String,
    /// Amplitude in meters
    pub amplitude: f64,
    /// Phase in degrees (as read from file)
    pub phase_degrees: f64,
    /// Period in seconds (looked up from name)
    pub period: f64,
}

impl ConstituentEntry {
    /// Convert phase from degrees to radians.
    pub fn phase_radians(&self) -> f64 {
        self.phase_degrees.to_radians()
    }

    /// Get angular frequency omega = 2*pi/period in rad/s.
    pub fn omega(&self) -> f64 {
        2.0 * std::f64::consts::PI / self.period
    }
}

/// Full constituent data including metadata.
#[derive(Clone, Debug)]
pub struct ConstituentData {
    /// Optional location (x, y) or (lon, lat)
    pub location: Option<(f64, f64)>,
    /// Reference level for elevations (mean sea level)
    pub reference_level: f64,
    /// List of constituents
    pub constituents: Vec<ConstituentEntry>,
}

impl ConstituentData {
    /// Create empty constituent data.
    pub fn new() -> Self {
        Self {
            location: None,
            reference_level: 0.0,
            constituents: Vec::new(),
        }
    }

    /// Add a constituent.
    pub fn add(&mut self, entry: ConstituentEntry) {
        self.constituents.push(entry);
    }

    /// Get number of constituents.
    pub fn len(&self) -> usize {
        self.constituents.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.constituents.is_empty()
    }

    /// Find constituent by name.
    pub fn get(&self, name: &str) -> Option<&ConstituentEntry> {
        let name_upper = name.to_uppercase();
        self.constituents.iter().find(|c| c.name == name_upper)
    }

    /// Evaluate tidal elevation at time t.
    ///
    /// η(t) = reference_level + Σ Aᵢ cos(ωᵢt + φᵢ)
    pub fn evaluate(&self, t: f64) -> f64 {
        let mut eta = self.reference_level;
        for c in &self.constituents {
            eta += c.amplitude * (c.omega() * t + c.phase_radians()).cos();
        }
        eta
    }
}

impl Default for ConstituentData {
    fn default() -> Self {
        Self::new()
    }
}

/// Standard tidal constituent periods in seconds.
///
/// Returns the period for common tidal constituents, or None if unknown.
///
/// # Supported Constituents
///
/// ## Semidiurnal (period ~12 hours)
/// - M2: Principal lunar (12.42 h) - dominant in most locations
/// - S2: Principal solar (12.00 h)
/// - N2: Larger lunar elliptic (12.66 h)
/// - K2: Luni-solar semidiurnal (11.97 h)
///
/// ## Diurnal (period ~24 hours)
/// - K1: Luni-solar diurnal (23.93 h)
/// - O1: Principal lunar diurnal (25.82 h)
/// - P1: Principal solar diurnal (24.07 h)
/// - Q1: Larger lunar elliptic diurnal (26.87 h)
///
/// ## Shallow water (overtides)
/// - M4: Principal lunar shallow water (6.21 h)
/// - MS4: Luni-solar shallow water (6.10 h)
/// - MN4: Lunar shallow water (6.27 h)
pub fn constituent_period(name: &str) -> Option<f64> {
    // Periods in hours, converted to seconds
    let hours_to_seconds = 3600.0;

    match name.to_uppercase().as_str() {
        // Semidiurnal constituents
        "M2" => Some(12.4206012 * hours_to_seconds),
        "S2" => Some(12.0 * hours_to_seconds),
        "N2" => Some(12.6583482 * hours_to_seconds),
        "K2" => Some(11.9672348 * hours_to_seconds),

        // Diurnal constituents
        "K1" => Some(23.9344697 * hours_to_seconds),
        "O1" => Some(25.8193417 * hours_to_seconds),
        "P1" => Some(24.0658902 * hours_to_seconds),
        "Q1" => Some(26.8683567 * hours_to_seconds),

        // Shallow water (overtides)
        "M4" => Some(6.2103006 * hours_to_seconds),
        "MS4" => Some(6.1033392 * hours_to_seconds),
        "MN4" => Some(6.2691739 * hours_to_seconds),
        "M6" => Some(4.1402004 * hours_to_seconds),

        // Long period
        "MF" => Some(327.8599387 * hours_to_seconds),
        "MM" => Some(661.3111655 * hours_to_seconds),
        "SSA" => Some(4382.9052083 * hours_to_seconds),

        _ => None,
    }
}

/// Read a tidal constituent file.
///
/// # Arguments
/// * `path` - Path to the constituent file
///
/// # Returns
/// * `Ok(ConstituentData)` - Parsed constituent data
/// * `Err(ConstituentFileError)` - If reading or parsing fails
///
/// # Example
///
/// ```ignore
/// use dg::io::read_constituent_file;
/// use std::path::Path;
///
/// let data = read_constituent_file(Path::new("tides.txt"))?;
/// println!("Loaded {} constituents", data.len());
/// for c in &data.constituents {
///     println!("{}: A={:.3}m, φ={:.1}°", c.name, c.amplitude, c.phase_degrees);
/// }
/// ```
pub fn read_constituent_file(path: &Path) -> Result<ConstituentData, ConstituentFileError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut data = ConstituentData::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Parse comments for metadata
        if line.starts_with('#') {
            let comment = line.trim_start_matches('#').trim();

            if let Some(loc_str) = comment.strip_prefix("location:") {
                let parts: Vec<&str> = loc_str.split_whitespace().collect();
                if parts.len() >= 2 {
                    let x =
                        parts[0]
                            .parse::<f64>()
                            .map_err(|_| ConstituentFileError::ParseError {
                                line: line_num + 1,
                                message: "Invalid location x coordinate".into(),
                            })?;
                    let y =
                        parts[1]
                            .parse::<f64>()
                            .map_err(|_| ConstituentFileError::ParseError {
                                line: line_num + 1,
                                message: "Invalid location y coordinate".into(),
                            })?;
                    data.location = Some((x, y));
                }
            } else if let Some(ref_str) = comment.strip_prefix("reference_level:") {
                data.reference_level = ref_str.trim().parse::<f64>().map_err(|_| {
                    ConstituentFileError::ParseError {
                        line: line_num + 1,
                        message: "Invalid reference level".into(),
                    }
                })?;
            }
            continue;
        }

        // Parse constituent line: name amplitude phase
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(ConstituentFileError::ParseError {
                line: line_num + 1,
                message: "Expected: name amplitude phase".into(),
            });
        }

        let name = parts[0].to_uppercase();
        let amplitude = parts[1]
            .parse::<f64>()
            .map_err(|_| ConstituentFileError::ParseError {
                line: line_num + 1,
                message: "Invalid amplitude value".into(),
            })?;
        let phase_degrees =
            parts[2]
                .parse::<f64>()
                .map_err(|_| ConstituentFileError::ParseError {
                    line: line_num + 1,
                    message: "Invalid phase value".into(),
                })?;

        let period = constituent_period(&name)
            .ok_or_else(|| ConstituentFileError::UnknownConstituent(name.clone()))?;

        data.constituents.push(ConstituentEntry {
            name,
            amplitude,
            phase_degrees,
            period,
        });
    }

    Ok(data)
}

/// Parse constituent data from a string.
///
/// Same format as file, useful for testing or embedded data.
pub fn parse_constituents(content: &str) -> Result<ConstituentData, ConstituentFileError> {
    let mut data = ConstituentData::new();

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
                        data.location = Some((x, y));
                    }
                }
            } else if let Some(ref_str) = comment.strip_prefix("reference_level:") {
                if let Ok(level) = ref_str.trim().parse() {
                    data.reference_level = level;
                }
            }
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(ConstituentFileError::ParseError {
                line: line_num + 1,
                message: "Expected: name amplitude phase".into(),
            });
        }

        let name = parts[0].to_uppercase();
        let amplitude: f64 = parts[1]
            .parse()
            .map_err(|_| ConstituentFileError::ParseError {
                line: line_num + 1,
                message: "Invalid amplitude".into(),
            })?;
        let phase_degrees: f64 =
            parts[2]
                .parse()
                .map_err(|_| ConstituentFileError::ParseError {
                    line: line_num + 1,
                    message: "Invalid phase".into(),
                })?;

        let period = constituent_period(&name)
            .ok_or_else(|| ConstituentFileError::UnknownConstituent(name.clone()))?;

        data.constituents.push(ConstituentEntry {
            name,
            amplitude,
            phase_degrees,
            period,
        });
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_constituent_period_m2() {
        let period = constituent_period("M2").unwrap();
        // M2 period is approximately 12.42 hours
        let hours = period / 3600.0;
        assert!((hours - 12.42).abs() < 0.01);
    }

    #[test]
    fn test_constituent_period_case_insensitive() {
        assert!(constituent_period("m2").is_some());
        assert!(constituent_period("M2").is_some());
        assert!(constituent_period("s2").is_some());
        assert!(constituent_period("K1").is_some());
    }

    #[test]
    fn test_constituent_period_unknown() {
        assert!(constituent_period("UNKNOWN").is_none());
        assert!(constituent_period("XYZ").is_none());
    }

    #[test]
    fn test_parse_simple_file() {
        let content = "M2 0.45 125.3\nS2 0.15 158.7";
        let data = parse_constituents(content).unwrap();

        assert_eq!(data.len(), 2);
        assert_eq!(data.constituents[0].name, "M2");
        assert!((data.constituents[0].amplitude - 0.45).abs() < TOL);
        assert!((data.constituents[0].phase_degrees - 125.3).abs() < TOL);
    }

    #[test]
    fn test_parse_with_metadata() {
        let content = r#"
# location: 5.32 60.39
# reference_level: 0.5
M2 0.45 125.3
"#;
        let data = parse_constituents(content).unwrap();

        assert_eq!(data.location, Some((5.32, 60.39)));
        assert!((data.reference_level - 0.5).abs() < TOL);
        assert_eq!(data.len(), 1);
    }

    #[test]
    fn test_parse_with_comments() {
        let content = r#"
# This is a comment
# columns: name amplitude phase
M2 0.45 125.3
# Another comment
S2 0.15 158.7
"#;
        let data = parse_constituents(content).unwrap();
        assert_eq!(data.len(), 2);
    }

    #[test]
    fn test_parse_empty_lines() {
        let content = "M2 0.45 125.3\n\n\nS2 0.15 158.7\n";
        let data = parse_constituents(content).unwrap();
        assert_eq!(data.len(), 2);
    }

    #[test]
    fn test_parse_unknown_constituent() {
        let content = "UNKNOWN 0.1 45.0";
        let result = parse_constituents(content);
        assert!(matches!(
            result,
            Err(ConstituentFileError::UnknownConstituent(_))
        ));
    }

    #[test]
    fn test_parse_invalid_amplitude() {
        let content = "M2 invalid 45.0";
        let result = parse_constituents(content);
        assert!(matches!(
            result,
            Err(ConstituentFileError::ParseError { .. })
        ));
    }

    #[test]
    fn test_parse_missing_fields() {
        let content = "M2 0.45";
        let result = parse_constituents(content);
        assert!(matches!(
            result,
            Err(ConstituentFileError::ParseError { .. })
        ));
    }

    #[test]
    fn test_constituent_entry_phase_radians() {
        let entry = ConstituentEntry {
            name: "M2".into(),
            amplitude: 0.5,
            phase_degrees: 180.0,
            period: 44712.0,
        };
        assert!((entry.phase_radians() - std::f64::consts::PI).abs() < TOL);
    }

    #[test]
    fn test_constituent_entry_omega() {
        let entry = ConstituentEntry {
            name: "TEST".into(),
            amplitude: 1.0,
            phase_degrees: 0.0,
            period: 2.0 * std::f64::consts::PI,
        };
        assert!((entry.omega() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_constituent_data_evaluate() {
        let mut data = ConstituentData::new();
        data.reference_level = 1.0;
        data.constituents.push(ConstituentEntry {
            name: "TEST".into(),
            amplitude: 0.5,
            phase_degrees: 0.0,
            period: 2.0 * std::f64::consts::PI, // omega = 1
        });

        // At t=0: cos(0) = 1, so eta = 1.0 + 0.5*1 = 1.5
        assert!((data.evaluate(0.0) - 1.5).abs() < TOL);

        // At t=pi: cos(pi) = -1, so eta = 1.0 + 0.5*(-1) = 0.5
        assert!((data.evaluate(std::f64::consts::PI) - 0.5).abs() < TOL);
    }

    #[test]
    fn test_constituent_data_get() {
        let content = "M2 0.45 125.3\nS2 0.15 158.7";
        let data = parse_constituents(content).unwrap();

        assert!(data.get("M2").is_some());
        assert!(data.get("m2").is_some()); // case insensitive
        assert!(data.get("K1").is_none());
    }

    #[test]
    fn test_read_constituent_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "# location: 5.0 60.0").unwrap();
        writeln!(file, "# reference_level: 0.0").unwrap();
        writeln!(file, "M2 0.45 125.3").unwrap();
        writeln!(file, "S2 0.15 158.7").unwrap();

        let data = read_constituent_file(file.path()).unwrap();

        assert_eq!(data.location, Some((5.0, 60.0)));
        assert_eq!(data.len(), 2);
        assert!((data.constituents[0].amplitude - 0.45).abs() < TOL);
    }

    #[test]
    fn test_norwegian_constituents() {
        // Test standard Norwegian coast constituents
        let content = r#"
# Norwegian coast constituents (typical values)
M2 0.50 120.0
S2 0.15 150.0
N2 0.10 100.0
K1 0.05 45.0
O1 0.04 30.0
P1 0.02 40.0
"#;
        let data = parse_constituents(content).unwrap();
        assert_eq!(data.len(), 6);

        // Check M2 is dominant
        let m2 = data.get("M2").unwrap();
        for c in &data.constituents {
            assert!(m2.amplitude >= c.amplitude);
        }
    }
}
