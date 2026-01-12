//! Tidal harmonic analysis, comparison tools, and tide gauge validation.
//!
//! This module provides tools for:
//! - Decomposing tidal time series into constituent harmonics using least-squares fitting
//! - Comparing model results against observations
//! - Validating model output against tide gauge observations
//!
//! # Mathematical Background
//!
//! Tidal signal decomposition:
//! ```text
//! η(t) = η₀ + Σᵢ [Aᵢ cos(ωᵢt) + Bᵢ sin(ωᵢt)]
//! ```
//!
//! Where:
//! - ωᵢ = 2π/Tᵢ (angular frequency from constituent period)
//! - Amplitude: Hᵢ = √(Aᵢ² + Bᵢ²)
//! - Phase: φᵢ = atan2(-Bᵢ, Aᵢ)
//!
//! This is a linear least-squares problem.
//!
//! # Example - Harmonic Analysis
//!
//! ```ignore
//! use dg_rs::analysis::{HarmonicAnalysis, TimeSeries};
//!
//! // Create time series from observations
//! let times: Vec<f64> = (0..1000).map(|i| i as f64 * 3600.0).collect();
//! let values: Vec<f64> = times.iter().map(|&t| /* your data */).collect();
//! let series = TimeSeries::new(&times, &values);
//!
//! // Fit tidal constituents
//! let analysis = HarmonicAnalysis::standard();
//! let result = analysis.fit(&series);
//!
//! println!("M2 amplitude: {:.3} m", result.constituents[0].amplitude);
//! println!("M2 phase: {:.1} deg", result.constituents[0].phase.to_degrees());
//! ```
//!
//! # Example - Tide Gauge Validation
//!
//! ```ignore
//! use dg_rs::analysis::{TideGaugeStation, StationValidationResult, TimeSeries};
//!
//! let station = TideGaugeStation::new("Bergen", 5.32, 60.39);
//! let model = TimeSeries::new(&times, &model_values);
//! let obs = TimeSeries::new(&times, &obs_values);
//!
//! let result = StationValidationResult::compute(&station, &model, &obs);
//! println!("RMSE: {:.3} m", result.metrics.rmse);
//! println!("Skill: {:.2}", result.metrics.skill_score);
//! ```

mod adcp;
mod harmonic;
mod metrics;
mod stability;
mod tide_gauge;

pub use adcp::{
    ADCPStation, ADCPValidationResult, ADCPValidationSummary, CurrentPoint, CurrentTimeSeries,
    CurrentValidationMetrics, norwegian_stations as adcp_norwegian_stations,
};
pub use harmonic::{ConstituentResult, HarmonicAnalysis, HarmonicResult};
pub use metrics::{
    ComparisonMetrics, ConstituentComparison, ConstituentComparisonSummary, compare_harmonics,
};
pub use stability::{StabilityMonitor, StabilityStatus, StabilityThresholds, StabilityWarning};
pub use tide_gauge::{
    ModelExtractor, PrecomputedExtractor, StationValidationResult, TideGaugeStation,
    ValidationSummary, norwegian_stations, validate_stations,
};

use std::f64::consts::PI;

/// A single time series data point.
#[derive(Clone, Copy, Debug)]
pub struct TimeSeriesPoint {
    /// Time in seconds
    pub time: f64,
    /// Value (elevation in m, velocity in m/s, etc.)
    pub value: f64,
}

/// Time series with optional metadata.
///
/// Stores a sequence of (time, value) pairs along with optional
/// location and name information.
#[derive(Clone, Debug)]
pub struct TimeSeries {
    /// The time series data points
    pub data: Vec<TimeSeriesPoint>,
    /// Optional (x, y) location coordinates
    pub location: Option<(f64, f64)>,
    /// Optional name/identifier
    pub name: Option<String>,
}

impl TimeSeries {
    /// Create a new time series from parallel arrays of times and values.
    ///
    /// # Panics
    ///
    /// Panics if `times` and `values` have different lengths.
    pub fn new(times: &[f64], values: &[f64]) -> Self {
        assert_eq!(
            times.len(),
            values.len(),
            "times and values must have same length"
        );

        let data = times
            .iter()
            .zip(values.iter())
            .map(|(&time, &value)| TimeSeriesPoint { time, value })
            .collect();

        Self {
            data,
            location: None,
            name: None,
        }
    }

    /// Create a time series with location information.
    pub fn with_location(mut self, x: f64, y: f64) -> Self {
        self.location = Some((x, y));
        self
    }

    /// Create a time series with a name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Number of data points.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the time series is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Duration of the time series (last time - first time).
    pub fn duration(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.last().unwrap().time - self.data.first().unwrap().time
    }

    /// Get times as a vector.
    pub fn times(&self) -> Vec<f64> {
        self.data.iter().map(|p| p.time).collect()
    }

    /// Get values as a vector.
    pub fn values(&self) -> Vec<f64> {
        self.data.iter().map(|p| p.value).collect()
    }

    /// Compute the mean value.
    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().map(|p| p.value).sum::<f64>() / self.data.len() as f64
    }

    /// Compute the variance.
    pub fn variance(&self) -> f64 {
        if self.data.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let sum_sq: f64 = self.data.iter().map(|p| (p.value - mean).powi(2)).sum();
        sum_sq / (self.data.len() - 1) as f64
    }

    /// Compute the standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Wrap a phase angle to the range [0, 2π).
pub fn wrap_phase(phase: f64) -> f64 {
    let mut p = phase % (2.0 * PI);
    if p < 0.0 {
        p += 2.0 * PI;
    }
    p
}

/// Compute phase difference wrapped to [-π, π].
pub fn phase_difference(phase1: f64, phase2: f64) -> f64 {
    let diff = phase1 - phase2;
    let mut wrapped = diff % (2.0 * PI);
    if wrapped > PI {
        wrapped -= 2.0 * PI;
    } else if wrapped < -PI {
        wrapped += 2.0 * PI;
    }
    wrapped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_series_creation() {
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![1.0, 2.0, 1.5, 2.5];
        let ts = TimeSeries::new(&times, &values);

        assert_eq!(ts.len(), 4);
        assert!((ts.duration() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_series_statistics() {
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let ts = TimeSeries::new(&times, &values);

        assert!((ts.mean() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_wrap_phase() {
        assert!((wrap_phase(0.0) - 0.0).abs() < 1e-10);
        assert!((wrap_phase(PI) - PI).abs() < 1e-10);
        assert!((wrap_phase(-PI) - PI).abs() < 1e-10);
        assert!((wrap_phase(3.0 * PI) - PI).abs() < 1e-10);
    }

    #[test]
    fn test_phase_difference() {
        assert!((phase_difference(0.1, 0.0) - 0.1).abs() < 1e-10);
        assert!((phase_difference(0.0, 0.1) - (-0.1)).abs() < 1e-10);

        // Wrap around
        let diff = phase_difference(0.1, 2.0 * PI - 0.1);
        assert!((diff - 0.2).abs() < 1e-10);
    }
}
