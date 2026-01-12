//! Tide gauge validation for comparing model output against observations.
//!
//! This module provides tools for validating 2D shallow water model results
//! against tide gauge observations from the Norwegian coast.
//!
//! # Features
//!
//! - Station metadata management (location, name, datum)
//! - Model value extraction at station locations
//! - Time series comparison (RMSE, bias, correlation, skill score)
//! - Harmonic constituent comparison (amplitude, phase errors)
//! - Multi-station validation summaries
//!
//! # Example
//!
//! ```ignore
//! use dg_rs::analysis::{TideGaugeStation, TideGaugeValidation, StationValidationResult};
//! use dg_rs::analysis::TimeSeries;
//!
//! // Define stations
//! let bergen = TideGaugeStation::new("Bergen", 5.32, 60.39)
//!     .with_datum_offset(0.0);
//!
//! // Create observed time series
//! let obs = TimeSeries::new(&times, &water_levels)
//!     .with_name("Bergen");
//!
//! // Create model time series (extracted at station location)
//! let model = TimeSeries::new(&times, &model_eta)
//!     .with_name("Bergen");
//!
//! // Validate
//! let result = StationValidationResult::compute(&bergen, &model, &obs);
//! println!("RMSE: {:.3} m", result.metrics.rmse);
//! println!("Skill: {:.2}", result.metrics.skill_score);
//! ```

use super::{
    ComparisonMetrics, ConstituentComparison, ConstituentComparisonSummary, HarmonicAnalysis,
    HarmonicResult, TimeSeries, compare_harmonics,
};

/// Metadata for a tide gauge station.
///
/// Stores location, name, and reference datum information.
#[derive(Clone, Debug)]
pub struct TideGaugeStation {
    /// Station identifier/name
    pub name: String,
    /// Longitude (degrees East)
    pub longitude: f64,
    /// Latitude (degrees North)
    pub latitude: f64,
    /// Local x-coordinate in model domain (computed from projection)
    pub x: Option<f64>,
    /// Local y-coordinate in model domain (computed from projection)
    pub y: Option<f64>,
    /// Vertical datum offset (m) - add to observations to match model datum
    pub datum_offset: f64,
    /// Station ID in external database (e.g., PSMSL, UHSLC)
    pub external_id: Option<String>,
}

impl TideGaugeStation {
    /// Create a new tide gauge station.
    ///
    /// # Arguments
    /// * `name` - Station name/identifier
    /// * `longitude` - Longitude in degrees East
    /// * `latitude` - Latitude in degrees North
    pub fn new(name: impl Into<String>, longitude: f64, latitude: f64) -> Self {
        Self {
            name: name.into(),
            longitude,
            latitude,
            x: None,
            y: None,
            datum_offset: 0.0,
            external_id: None,
        }
    }

    /// Set local model coordinates.
    pub fn with_local_coords(mut self, x: f64, y: f64) -> Self {
        self.x = Some(x);
        self.y = Some(y);
        self
    }

    /// Set vertical datum offset.
    ///
    /// This offset is added to observations to match the model's vertical datum.
    pub fn with_datum_offset(mut self, offset: f64) -> Self {
        self.datum_offset = offset;
        self
    }

    /// Set external database ID.
    pub fn with_external_id(mut self, id: impl Into<String>) -> Self {
        self.external_id = Some(id.into());
        self
    }

    /// Check if local coordinates are set.
    pub fn has_local_coords(&self) -> bool {
        self.x.is_some() && self.y.is_some()
    }

    /// Get local coordinates (x, y).
    ///
    /// # Panics
    /// Panics if local coordinates are not set.
    pub fn local_coords(&self) -> (f64, f64) {
        (
            self.x.expect("Local x-coordinate not set"),
            self.y.expect("Local y-coordinate not set"),
        )
    }
}

/// Norwegian tide gauge stations.
///
/// Standard stations for validation along the Norwegian coast.
/// Coordinates are approximate and should be verified against official sources.
pub mod norwegian_stations {
    use super::TideGaugeStation;

    /// Bergen tide gauge (Kartverket station).
    pub fn bergen() -> TideGaugeStation {
        TideGaugeStation::new("Bergen", 5.32, 60.39)
    }

    /// Stavanger tide gauge.
    pub fn stavanger() -> TideGaugeStation {
        TideGaugeStation::new("Stavanger", 5.73, 58.97)
    }

    /// Trondheim tide gauge.
    pub fn trondheim() -> TideGaugeStation {
        TideGaugeStation::new("Trondheim", 10.39, 63.44)
    }

    /// Kristiansund tide gauge.
    pub fn kristiansund() -> TideGaugeStation {
        TideGaugeStation::new("Kristiansund", 7.73, 63.11)
    }

    /// Ålesund tide gauge.
    pub fn alesund() -> TideGaugeStation {
        TideGaugeStation::new("Ålesund", 6.15, 62.47)
    }

    /// Bodø tide gauge.
    pub fn bodo() -> TideGaugeStation {
        TideGaugeStation::new("Bodø", 14.39, 67.29)
    }

    /// Tromsø tide gauge.
    pub fn tromso() -> TideGaugeStation {
        TideGaugeStation::new("Tromsø", 18.96, 69.65)
    }

    /// Hammerfest tide gauge.
    pub fn hammerfest() -> TideGaugeStation {
        TideGaugeStation::new("Hammerfest", 23.68, 70.66)
    }

    /// Vardo tide gauge.
    pub fn vardo() -> TideGaugeStation {
        TideGaugeStation::new("Vardø", 31.11, 70.37)
    }

    /// Heimsjø tide gauge (Frøya region).
    pub fn heimsjo() -> TideGaugeStation {
        TideGaugeStation::new("Heimsjø", 9.10, 63.43)
    }

    /// Get all standard Norwegian coast stations.
    pub fn all_stations() -> Vec<TideGaugeStation> {
        vec![
            bergen(),
            stavanger(),
            trondheim(),
            kristiansund(),
            alesund(),
            bodo(),
            tromso(),
            hammerfest(),
            vardo(),
        ]
    }

    /// Get stations in the Trøndelag region (around Frøya).
    pub fn trondelag_stations() -> Vec<TideGaugeStation> {
        vec![trondheim(), kristiansund(), heimsjo()]
    }
}

/// Validation result for a single tide gauge station.
#[derive(Clone, Debug)]
pub struct StationValidationResult {
    /// Station metadata
    pub station: TideGaugeStation,
    /// Time series comparison metrics
    pub metrics: ComparisonMetrics,
    /// Harmonic analysis of model time series
    pub model_harmonics: Option<HarmonicResult>,
    /// Harmonic analysis of observed time series
    pub obs_harmonics: Option<HarmonicResult>,
    /// Constituent-by-constituent comparison
    pub constituent_comparisons: Vec<ConstituentComparison>,
    /// Summary of constituent comparisons
    pub constituent_summary: Option<ConstituentComparisonSummary>,
    /// Model mean water level (m)
    pub model_mean: f64,
    /// Observed mean water level (m)
    pub obs_mean: f64,
    /// Model standard deviation (m)
    pub model_std: f64,
    /// Observed standard deviation (m)
    pub obs_std: f64,
}

impl StationValidationResult {
    /// Compute validation result from model and observation time series.
    ///
    /// # Arguments
    /// * `station` - Station metadata
    /// * `model` - Model water level time series
    /// * `obs` - Observed water level time series (will have datum_offset applied)
    ///
    /// # Panics
    /// Panics if time series have different lengths.
    pub fn compute(station: &TideGaugeStation, model: &TimeSeries, obs: &TimeSeries) -> Self {
        assert_eq!(
            model.len(),
            obs.len(),
            "Model and observation time series must have same length"
        );

        // Apply datum offset to observations
        let obs_adjusted: Vec<f64> = obs.values().iter().map(|&v| v + station.datum_offset).collect();

        let model_values = model.values();
        let metrics = ComparisonMetrics::compute(&model_values, &obs_adjusted);

        // Compute means and standard deviations
        let model_mean = model.mean();
        let obs_mean = obs_adjusted.iter().sum::<f64>() / obs_adjusted.len() as f64;
        let model_std = model.std_dev();
        let obs_std = {
            let var: f64 = obs_adjusted
                .iter()
                .map(|&v| (v - obs_mean).powi(2))
                .sum::<f64>()
                / (obs_adjusted.len() - 1) as f64;
            var.sqrt()
        };

        Self {
            station: station.clone(),
            metrics,
            model_harmonics: None,
            obs_harmonics: None,
            constituent_comparisons: Vec::new(),
            constituent_summary: None,
            model_mean,
            obs_mean,
            model_std,
            obs_std,
        }
    }

    /// Compute validation with harmonic analysis.
    ///
    /// Performs harmonic analysis on both time series and compares constituents.
    pub fn compute_with_harmonics(
        station: &TideGaugeStation,
        model: &TimeSeries,
        obs: &TimeSeries,
        analysis: &HarmonicAnalysis,
    ) -> Self {
        let mut result = Self::compute(station, model, obs);

        // Perform harmonic analysis
        let model_harmonics = analysis.fit(model);

        // Create adjusted observation time series
        let obs_adjusted_values: Vec<f64> = obs
            .values()
            .iter()
            .map(|&v| v + station.datum_offset)
            .collect();
        let obs_adjusted = TimeSeries::new(&obs.times(), &obs_adjusted_values);
        let obs_harmonics = analysis.fit(&obs_adjusted);

        // Compare constituents
        let comparisons = compare_harmonics(&model_harmonics, &obs_harmonics);
        let summary = ConstituentComparisonSummary::from_comparisons(&comparisons);

        result.model_harmonics = Some(model_harmonics);
        result.obs_harmonics = Some(obs_harmonics);
        result.constituent_comparisons = comparisons;
        result.constituent_summary = Some(summary);

        result
    }

    /// Check if model passes basic validation criteria.
    ///
    /// Criteria:
    /// - Correlation > 0.9
    /// - Skill score > 0.8
    /// - Bias < 0.1 m
    pub fn passes_basic_validation(&self) -> bool {
        self.metrics.correlation > 0.9
            && self.metrics.skill_score > 0.8
            && self.metrics.bias.abs() < 0.1
    }

    /// Check if model passes strict validation criteria.
    ///
    /// Criteria:
    /// - Correlation > 0.95
    /// - Skill score > 0.9
    /// - Bias < 0.05 m
    /// - RMSE < 0.1 m
    pub fn passes_strict_validation(&self) -> bool {
        self.metrics.correlation > 0.95
            && self.metrics.skill_score > 0.9
            && self.metrics.bias.abs() < 0.05
            && self.metrics.rmse < 0.1
    }

    /// Format validation result as a summary string.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "Station: {}\n\
             Location: ({:.2}°E, {:.2}°N)\n\
             --------------------------\n\
             Time Series Metrics:\n\
             RMSE:        {:.4} m\n\
             Bias:        {:.4} m\n\
             MAE:         {:.4} m\n\
             Correlation: {:.4}\n\
             Skill Score: {:.4}\n\
             Max Error:   {:.4} m\n\
             N Points:    {}\n\
             --------------------------\n\
             Statistics:\n\
             Model Mean:  {:.4} m\n\
             Obs Mean:    {:.4} m\n\
             Model Std:   {:.4} m\n\
             Obs Std:     {:.4} m\n",
            self.station.name,
            self.station.longitude,
            self.station.latitude,
            self.metrics.rmse,
            self.metrics.bias,
            self.metrics.mae,
            self.metrics.correlation,
            self.metrics.skill_score,
            self.metrics.max_error,
            self.metrics.n_points,
            self.model_mean,
            self.obs_mean,
            self.model_std,
            self.obs_std,
        );

        if let Some(ref summary) = self.constituent_summary {
            s.push_str(&format!(
                "--------------------------\n\
                 Harmonic Comparison:\n\
                 N Constituents:      {}\n\
                 Mean Amp Ratio:      {:.3}\n\
                 RMS Amp Error:       {:.4} m\n\
                 RMS Phase Error:     {:.1}°\n",
                summary.n_constituents,
                summary.mean_amplitude_ratio,
                summary.rms_amplitude_error,
                summary.rms_phase_error_degrees(),
            ));
        }

        s
    }
}

/// Summary of validation across multiple stations.
#[derive(Clone, Debug)]
pub struct ValidationSummary {
    /// Number of stations validated
    pub n_stations: usize,
    /// Mean RMSE across all stations
    pub mean_rmse: f64,
    /// Mean bias across all stations
    pub mean_bias: f64,
    /// Mean correlation across all stations
    pub mean_correlation: f64,
    /// Mean skill score across all stations
    pub mean_skill_score: f64,
    /// Worst (highest) RMSE
    pub worst_rmse: f64,
    /// Station with worst RMSE
    pub worst_station: String,
    /// Best (lowest) RMSE
    pub best_rmse: f64,
    /// Station with best RMSE
    pub best_station: String,
    /// Number of stations passing basic validation
    pub n_passing_basic: usize,
    /// Number of stations passing strict validation
    pub n_passing_strict: usize,
}

impl ValidationSummary {
    /// Compute summary from a list of station validation results.
    pub fn from_results(results: &[StationValidationResult]) -> Self {
        if results.is_empty() {
            return Self {
                n_stations: 0,
                mean_rmse: 0.0,
                mean_bias: 0.0,
                mean_correlation: 0.0,
                mean_skill_score: 0.0,
                worst_rmse: 0.0,
                worst_station: String::new(),
                best_rmse: 0.0,
                best_station: String::new(),
                n_passing_basic: 0,
                n_passing_strict: 0,
            };
        }

        let n = results.len() as f64;

        let mean_rmse = results.iter().map(|r| r.metrics.rmse).sum::<f64>() / n;
        let mean_bias = results.iter().map(|r| r.metrics.bias).sum::<f64>() / n;
        let mean_correlation = results.iter().map(|r| r.metrics.correlation).sum::<f64>() / n;
        let mean_skill_score = results.iter().map(|r| r.metrics.skill_score).sum::<f64>() / n;

        // Use partial_cmp with unwrap_or to handle NaN values gracefully
        let worst = results
            .iter()
            .max_by(|a, b| a.metrics.rmse.partial_cmp(&b.metrics.rmse).unwrap_or(std::cmp::Ordering::Equal))
            .expect("results is not empty");
        let best = results
            .iter()
            .min_by(|a, b| a.metrics.rmse.partial_cmp(&b.metrics.rmse).unwrap_or(std::cmp::Ordering::Equal))
            .expect("results is not empty");

        let n_passing_basic = results.iter().filter(|r| r.passes_basic_validation()).count();
        let n_passing_strict = results.iter().filter(|r| r.passes_strict_validation()).count();

        Self {
            n_stations: results.len(),
            mean_rmse,
            mean_bias,
            mean_correlation,
            mean_skill_score,
            worst_rmse: worst.metrics.rmse,
            worst_station: worst.station.name.clone(),
            best_rmse: best.metrics.rmse,
            best_station: best.station.name.clone(),
            n_passing_basic,
            n_passing_strict,
        }
    }

    /// Format summary as a string.
    pub fn to_string(&self) -> String {
        format!(
            "Validation Summary ({} stations)\n\
             ================================\n\
             Mean RMSE:        {:.4} m\n\
             Mean Bias:        {:.4} m\n\
             Mean Correlation: {:.4}\n\
             Mean Skill:       {:.4}\n\
             --------------------------------\n\
             Best RMSE:   {:.4} m ({})\n\
             Worst RMSE:  {:.4} m ({})\n\
             --------------------------------\n\
             Passing Basic:  {}/{}\n\
             Passing Strict: {}/{}\n",
            self.n_stations,
            self.mean_rmse,
            self.mean_bias,
            self.mean_correlation,
            self.mean_skill_score,
            self.best_rmse,
            self.best_station,
            self.worst_rmse,
            self.worst_station,
            self.n_passing_basic,
            self.n_stations,
            self.n_passing_strict,
            self.n_stations,
        )
    }
}

/// Extract model values at tide gauge stations.
///
/// This trait allows extracting water level time series from different
/// model output formats at specific station locations.
pub trait ModelExtractor {
    /// Extract water level at a single point for all time steps.
    ///
    /// # Arguments
    /// * `x` - Local x-coordinate
    /// * `y` - Local y-coordinate
    /// * `times` - Time values (seconds)
    ///
    /// # Returns
    /// Water level (surface elevation) at each time step, or None if point is outside domain.
    fn extract_at_point(&self, x: f64, y: f64, times: &[f64]) -> Option<Vec<f64>>;

    /// Extract time series at a tide gauge station.
    fn extract_at_station(&self, station: &TideGaugeStation, times: &[f64]) -> Option<TimeSeries> {
        let (x, y) = station.local_coords();
        self.extract_at_point(x, y, times).map(|values| {
            TimeSeries::new(times, &values)
                .with_name(&station.name)
                .with_location(x, y)
        })
    }
}

/// Simple model extractor that stores pre-computed time series at station locations.
///
/// Use this when model output has already been extracted during simulation.
#[derive(Clone, Debug, Default)]
pub struct PrecomputedExtractor {
    /// Map from station name to time series
    data: std::collections::HashMap<String, TimeSeries>,
}

impl PrecomputedExtractor {
    /// Create a new empty extractor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a time series for a station.
    pub fn add_series(&mut self, station_name: impl Into<String>, series: TimeSeries) {
        self.data.insert(station_name.into(), series);
    }

    /// Get time series for a station.
    pub fn get_series(&self, station_name: &str) -> Option<&TimeSeries> {
        self.data.get(station_name)
    }
}

impl ModelExtractor for PrecomputedExtractor {
    fn extract_at_point(&self, _x: f64, _y: f64, _times: &[f64]) -> Option<Vec<f64>> {
        // This extractor doesn't support arbitrary point extraction
        None
    }

    fn extract_at_station(&self, station: &TideGaugeStation, _times: &[f64]) -> Option<TimeSeries> {
        self.data.get(&station.name).cloned()
    }
}

/// Run validation for multiple stations.
///
/// # Arguments
/// * `stations` - List of tide gauge stations
/// * `model_extractor` - Model output extractor
/// * `observations` - Map from station name to observed time series
/// * `times` - Common time vector for model extraction
/// * `analysis` - Optional harmonic analysis configuration
///
/// # Returns
/// Validation results for each station and overall summary.
pub fn validate_stations(
    stations: &[TideGaugeStation],
    model_extractor: &dyn ModelExtractor,
    observations: &std::collections::HashMap<String, TimeSeries>,
    times: &[f64],
    analysis: Option<&HarmonicAnalysis>,
) -> (Vec<StationValidationResult>, ValidationSummary) {
    let mut results = Vec::new();

    for station in stations {
        // Get model time series
        let model_ts = match model_extractor.extract_at_station(station, times) {
            Some(ts) => ts,
            None => {
                eprintln!("Warning: Could not extract model data for station {}", station.name);
                continue;
            }
        };

        // Get observation time series
        let obs_ts = match observations.get(&station.name) {
            Some(ts) => ts,
            None => {
                eprintln!("Warning: No observations for station {}", station.name);
                continue;
            }
        };

        // Compute validation
        let result = if let Some(ha) = analysis {
            StationValidationResult::compute_with_harmonics(station, &model_ts, obs_ts, ha)
        } else {
            StationValidationResult::compute(station, &model_ts, obs_ts)
        };

        results.push(result);
    }

    let summary = ValidationSummary::from_results(&results);
    (results, summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_station_creation() {
        let station = TideGaugeStation::new("Bergen", 5.32, 60.39)
            .with_datum_offset(0.05)
            .with_local_coords(1000.0, 2000.0);

        assert_eq!(station.name, "Bergen");
        assert!((station.longitude - 5.32).abs() < TOL);
        assert!((station.latitude - 60.39).abs() < TOL);
        assert!((station.datum_offset - 0.05).abs() < TOL);
        assert!(station.has_local_coords());
        assert_eq!(station.local_coords(), (1000.0, 2000.0));
    }

    #[test]
    fn test_norwegian_stations() {
        let stations = norwegian_stations::all_stations();
        assert!(!stations.is_empty());

        let bergen = norwegian_stations::bergen();
        assert_eq!(bergen.name, "Bergen");
    }

    #[test]
    fn test_perfect_validation() {
        let station = TideGaugeStation::new("Test", 0.0, 0.0);
        let times: Vec<f64> = (0..100).map(|i| i as f64 * 3600.0).collect();
        let values: Vec<f64> = times.iter().map(|&t| (t / 44712.0).sin()).collect();

        let model = TimeSeries::new(&times, &values);
        let obs = TimeSeries::new(&times, &values);

        let result = StationValidationResult::compute(&station, &model, &obs);

        assert!(result.metrics.rmse.abs() < TOL);
        assert!(result.metrics.correlation > 0.99);
        assert!(result.metrics.skill_score > 0.99);
        assert!(result.passes_basic_validation());
        assert!(result.passes_strict_validation());
    }

    #[test]
    fn test_validation_with_bias() {
        let station = TideGaugeStation::new("Test", 0.0, 0.0);
        let times: Vec<f64> = (0..100).map(|i| i as f64 * 3600.0).collect();
        let values: Vec<f64> = times.iter().map(|&t| (t / 44712.0).sin()).collect();
        let biased: Vec<f64> = values.iter().map(|&v| v + 0.5).collect();

        let model = TimeSeries::new(&times, &biased);
        let obs = TimeSeries::new(&times, &values);

        let result = StationValidationResult::compute(&station, &model, &obs);

        assert!((result.metrics.bias - 0.5).abs() < TOL);
        assert!(result.metrics.correlation > 0.99); // High correlation despite bias
    }

    #[test]
    fn test_datum_offset_applied() {
        let station = TideGaugeStation::new("Test", 0.0, 0.0).with_datum_offset(0.1);
        let times: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let model_values = vec![1.1; 10]; // Model at 1.1
        let obs_values = vec![1.0; 10]; // Obs at 1.0, will become 1.1 with offset

        let model = TimeSeries::new(&times, &model_values);
        let obs = TimeSeries::new(&times, &obs_values);

        let result = StationValidationResult::compute(&station, &model, &obs);

        // After applying datum offset, model and adjusted obs should match
        assert!(result.metrics.rmse.abs() < TOL);
    }

    #[test]
    fn test_validation_summary() {
        let station1 = TideGaugeStation::new("Good", 0.0, 0.0);
        let station2 = TideGaugeStation::new("Bad", 1.0, 1.0);

        let times: Vec<f64> = (0..100).map(|i| i as f64 * 3600.0).collect();
        let good_values: Vec<f64> = times.iter().map(|&t| (t / 44712.0).sin()).collect();
        let bad_model: Vec<f64> = good_values.iter().map(|&v| v + 0.2).collect();

        let model1 = TimeSeries::new(&times, &good_values);
        let obs1 = TimeSeries::new(&times, &good_values);
        let model2 = TimeSeries::new(&times, &bad_model);
        let obs2 = TimeSeries::new(&times, &good_values);

        let result1 = StationValidationResult::compute(&station1, &model1, &obs1);
        let result2 = StationValidationResult::compute(&station2, &model2, &obs2);

        let summary = ValidationSummary::from_results(&[result1, result2]);

        assert_eq!(summary.n_stations, 2);
        assert_eq!(summary.best_station, "Good");
        assert_eq!(summary.worst_station, "Bad");
    }

    #[test]
    fn test_precomputed_extractor() {
        let mut extractor = PrecomputedExtractor::new();

        let times: Vec<f64> = vec![0.0, 1.0, 2.0];
        let values: Vec<f64> = vec![1.0, 1.5, 1.2];
        let series = TimeSeries::new(&times, &values);

        extractor.add_series("Bergen", series.clone());

        let station = TideGaugeStation::new("Bergen", 5.32, 60.39);
        let extracted = extractor.extract_at_station(&station, &times);

        assert!(extracted.is_some());
        assert_eq!(extracted.unwrap().len(), 3);
    }

    #[test]
    fn test_validation_with_harmonics() {
        let station = TideGaugeStation::new("Test", 0.0, 0.0);

        // Create synthetic M2 tidal signal
        let m2_period = 12.42 * 3600.0; // seconds
        let m2_omega = 2.0 * PI / m2_period;
        let amplitude = 0.5;
        let phase = 0.3;

        // 48 hours of data at 10-minute intervals
        let n_points = 288;
        let dt = 600.0; // 10 minutes
        let times: Vec<f64> = (0..n_points).map(|i| i as f64 * dt).collect();
        let values: Vec<f64> = times
            .iter()
            .map(|&t| amplitude * (m2_omega * t + phase).cos())
            .collect();

        let model = TimeSeries::new(&times, &values);
        let obs = TimeSeries::new(&times, &values);

        let analysis = HarmonicAnalysis::standard();
        let result = StationValidationResult::compute_with_harmonics(
            &station, &model, &obs, &analysis
        );

        assert!(result.model_harmonics.is_some());
        assert!(result.obs_harmonics.is_some());
        assert!(!result.constituent_comparisons.is_empty());
        assert!(result.constituent_summary.is_some());
    }

    #[test]
    fn test_summary_output() {
        let station = TideGaugeStation::new("Test", 5.0, 60.0);
        let times: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let values = vec![1.0; 10];

        let model = TimeSeries::new(&times, &values);
        let obs = TimeSeries::new(&times, &values);

        let result = StationValidationResult::compute(&station, &model, &obs);
        let summary = result.summary();

        assert!(summary.contains("Test"));
        assert!(summary.contains("RMSE"));
        assert!(summary.contains("Skill Score"));
    }
}
