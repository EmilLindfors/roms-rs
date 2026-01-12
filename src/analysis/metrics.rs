//! Comparison metrics for time series validation.
//!
//! Provides statistical metrics for comparing model results against observations,
//! including constituent-level comparisons for harmonic analysis results.

use super::{HarmonicResult, phase_difference};

/// Statistical comparison metrics between two time series.
///
/// All metrics assume the first series is the model and the second is the observation.
#[derive(Clone, Copy, Debug)]
pub struct ComparisonMetrics {
    /// Root mean square error: sqrt(mean((model - obs)²))
    pub rmse: f64,
    /// Mean absolute error: mean(|model - obs|)
    pub mae: f64,
    /// Bias (mean error): mean(model - obs)
    pub bias: f64,
    /// Pearson correlation coefficient [-1, 1]
    pub correlation: f64,
    /// Murphy skill score: 1 - MSE / Var(obs)
    pub skill_score: f64,
    /// Maximum absolute error: max(|model - obs|)
    pub max_error: f64,
    /// Number of data points
    pub n_points: usize,
}

impl ComparisonMetrics {
    /// Compute comparison metrics between model and observation series.
    ///
    /// # Panics
    ///
    /// Panics if the series have different lengths or are empty.
    pub fn compute(model: &[f64], observation: &[f64]) -> Self {
        assert_eq!(
            model.len(),
            observation.len(),
            "Model and observation must have same length"
        );
        assert!(!model.is_empty(), "Series must not be empty");

        let n = model.len();

        // Basic statistics
        let model_mean: f64 = model.iter().sum::<f64>() / n as f64;
        let obs_mean: f64 = observation.iter().sum::<f64>() / n as f64;

        // Errors
        let errors: Vec<f64> = model
            .iter()
            .zip(observation.iter())
            .map(|(&m, &o)| m - o)
            .collect();

        let bias = errors.iter().sum::<f64>() / n as f64;

        let mse: f64 = errors.iter().map(|e| e * e).sum::<f64>() / n as f64;
        let rmse = mse.sqrt();

        let mae: f64 = errors.iter().map(|e| e.abs()).sum::<f64>() / n as f64;

        let max_error = errors.iter().map(|e| e.abs()).fold(0.0, f64::max);

        // Variance of observations
        let obs_variance: f64 = observation
            .iter()
            .map(|&o| (o - obs_mean).powi(2))
            .sum::<f64>()
            / n as f64;

        // Murphy skill score
        let skill_score = if obs_variance > 1e-10 {
            1.0 - mse / obs_variance
        } else if mse < 1e-10 {
            1.0
        } else {
            f64::NEG_INFINITY
        };

        // Pearson correlation
        let model_variance: f64 =
            model.iter().map(|&m| (m - model_mean).powi(2)).sum::<f64>() / n as f64;

        let covariance: f64 = model
            .iter()
            .zip(observation.iter())
            .map(|(&m, &o)| (m - model_mean) * (o - obs_mean))
            .sum::<f64>()
            / n as f64;

        let correlation = if model_variance > 1e-10 && obs_variance > 1e-10 {
            covariance / (model_variance.sqrt() * obs_variance.sqrt())
        } else if model_variance < 1e-10 && obs_variance < 1e-10 {
            1.0 // Both constant, consider perfectly correlated
        } else {
            0.0 // One constant, no correlation defined
        };

        Self {
            rmse,
            mae,
            bias,
            correlation,
            skill_score,
            max_error,
            n_points: n,
        }
    }

    /// Check if correlation is significant (> 0.95).
    pub fn is_highly_correlated(&self) -> bool {
        self.correlation > 0.95
    }

    /// Check if skill score indicates good model performance (> 0.9).
    pub fn is_skillful(&self) -> bool {
        self.skill_score > 0.9
    }
}

/// Comparison between fitted constituents from model and observation.
#[derive(Clone, Copy, Debug)]
pub struct ConstituentComparison {
    /// Constituent name
    pub name: &'static str,
    /// Absolute amplitude error: |A_model - A_obs|
    pub amplitude_error: f64,
    /// Amplitude ratio: A_model / A_obs
    pub amplitude_ratio: f64,
    /// Phase error in radians, wrapped to [-π, π]
    pub phase_error: f64,
    /// Model amplitude
    pub model_amplitude: f64,
    /// Observation amplitude
    pub obs_amplitude: f64,
    /// Model phase
    pub model_phase: f64,
    /// Observation phase
    pub obs_phase: f64,
}

impl ConstituentComparison {
    /// Create a comparison between model and observation constituents.
    pub fn new(
        name: &'static str,
        model_amplitude: f64,
        model_phase: f64,
        obs_amplitude: f64,
        obs_phase: f64,
    ) -> Self {
        let amplitude_error = (model_amplitude - obs_amplitude).abs();
        let amplitude_ratio = if obs_amplitude > 1e-10 {
            model_amplitude / obs_amplitude
        } else {
            f64::INFINITY
        };
        let phase_error = phase_difference(model_phase, obs_phase);

        Self {
            name,
            amplitude_error,
            amplitude_ratio,
            phase_error,
            model_amplitude,
            obs_amplitude,
            model_phase,
            obs_phase,
        }
    }

    /// Phase error in degrees.
    pub fn phase_error_degrees(&self) -> f64 {
        self.phase_error.to_degrees()
    }

    /// Check if amplitude is within tolerance (ratio between 0.9 and 1.1).
    pub fn amplitude_within_10_percent(&self) -> bool {
        self.amplitude_ratio > 0.9 && self.amplitude_ratio < 1.1
    }

    /// Check if phase is within tolerance (error < 10 degrees).
    pub fn phase_within_10_degrees(&self) -> bool {
        self.phase_error.abs() < 10.0_f64.to_radians()
    }
}

/// Compare harmonic analysis results from model and observation.
///
/// Matches constituents by name and computes comparison metrics.
pub fn compare_harmonics(
    model: &HarmonicResult,
    obs: &HarmonicResult,
) -> Vec<ConstituentComparison> {
    let mut comparisons = Vec::new();

    for model_c in &model.constituents {
        if let Some(obs_c) = obs.get_constituent(model_c.name) {
            comparisons.push(ConstituentComparison::new(
                model_c.name,
                model_c.amplitude,
                model_c.phase,
                obs_c.amplitude,
                obs_c.phase,
            ));
        }
    }

    comparisons
}

/// Summary statistics for a set of constituent comparisons.
#[derive(Clone, Copy, Debug)]
pub struct ConstituentComparisonSummary {
    /// Mean amplitude ratio across all constituents
    pub mean_amplitude_ratio: f64,
    /// RMS of amplitude errors
    pub rms_amplitude_error: f64,
    /// RMS of phase errors (radians)
    pub rms_phase_error: f64,
    /// Number of constituents compared
    pub n_constituents: usize,
}

impl ConstituentComparisonSummary {
    /// Compute summary from a list of constituent comparisons.
    pub fn from_comparisons(comparisons: &[ConstituentComparison]) -> Self {
        if comparisons.is_empty() {
            return Self {
                mean_amplitude_ratio: 0.0,
                rms_amplitude_error: 0.0,
                rms_phase_error: 0.0,
                n_constituents: 0,
            };
        }

        let n = comparisons.len() as f64;

        let mean_amplitude_ratio = comparisons.iter().map(|c| c.amplitude_ratio).sum::<f64>() / n;

        let rms_amplitude_error = (comparisons
            .iter()
            .map(|c| c.amplitude_error.powi(2))
            .sum::<f64>()
            / n)
            .sqrt();

        let rms_phase_error = (comparisons
            .iter()
            .map(|c| c.phase_error.powi(2))
            .sum::<f64>()
            / n)
            .sqrt();

        Self {
            mean_amplitude_ratio,
            rms_amplitude_error,
            rms_phase_error,
            n_constituents: comparisons.len(),
        }
    }

    /// RMS phase error in degrees.
    pub fn rms_phase_error_degrees(&self) -> f64 {
        self.rms_phase_error.to_degrees()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_perfect_match() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let metrics = ComparisonMetrics::compute(&data, &data);

        assert!(metrics.rmse.abs() < TOL);
        assert!(metrics.mae.abs() < TOL);
        assert!(metrics.bias.abs() < TOL);
        assert!((metrics.correlation - 1.0).abs() < TOL);
        assert!((metrics.skill_score - 1.0).abs() < TOL);
        assert!(metrics.max_error.abs() < TOL);
    }

    #[test]
    fn test_constant_bias() {
        let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let model: Vec<f64> = obs.iter().map(|&x| x + 0.5).collect();
        let metrics = ComparisonMetrics::compute(&model, &obs);

        assert!((metrics.bias - 0.5).abs() < TOL);
        assert!((metrics.mae - 0.5).abs() < TOL);
        assert!((metrics.rmse - 0.5).abs() < TOL);
        assert!((metrics.correlation - 1.0).abs() < TOL); // Perfect correlation despite bias
    }

    #[test]
    fn test_negative_correlation() {
        let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let model = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let metrics = ComparisonMetrics::compute(&model, &obs);

        assert!((metrics.correlation - (-1.0)).abs() < TOL);
    }

    #[test]
    fn test_skill_score() {
        let obs = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let obs_mean = obs.iter().sum::<f64>() / obs.len() as f64; // = 0.4
        let model = vec![obs_mean; 5]; // Predicts exact mean
        let metrics = ComparisonMetrics::compute(&model, &obs);

        // Predicting the exact mean gives skill score of 0
        assert!(
            metrics.skill_score.abs() < 0.01,
            "Skill score should be ~0, got {:.4}",
            metrics.skill_score
        );
    }

    #[test]
    fn test_constituent_comparison() {
        let comp = ConstituentComparison::new("M2", 1.0, 0.5, 1.1, 0.6);

        assert!((comp.amplitude_error - 0.1).abs() < TOL);
        assert!((comp.amplitude_ratio - 1.0 / 1.1).abs() < TOL);
        assert!((comp.phase_error - (-0.1)).abs() < TOL);
    }

    #[test]
    fn test_phase_error_wrap() {
        // Phase near 0 vs phase near 2π should have small error
        let comp = ConstituentComparison::new("M2", 1.0, 0.1, 1.0, 2.0 * PI - 0.1);

        assert!(
            comp.phase_error.abs() < 0.3,
            "Phase error should be ~0.2, got {}",
            comp.phase_error
        );
    }

    #[test]
    fn test_comparison_summary() {
        let comparisons = vec![
            ConstituentComparison::new("M2", 1.0, 0.0, 1.0, 0.0),
            ConstituentComparison::new("S2", 0.5, 0.5, 0.5, 0.5),
        ];

        let summary = ConstituentComparisonSummary::from_comparisons(&comparisons);

        assert_eq!(summary.n_constituents, 2);
        assert!((summary.mean_amplitude_ratio - 1.0).abs() < TOL);
        assert!(summary.rms_amplitude_error.abs() < TOL);
        assert!(summary.rms_phase_error.abs() < TOL);
    }

    #[test]
    fn test_is_highly_correlated() {
        let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Perfect correlation
        let metrics = ComparisonMetrics::compute(&obs, &obs);
        assert!(metrics.is_highly_correlated());

        // Low correlation
        let uncorrelated = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let metrics2 = ComparisonMetrics::compute(&uncorrelated, &obs);
        assert!(!metrics2.is_highly_correlated());
    }

    #[test]
    fn test_amplitude_and_phase_tolerance() {
        // Within tolerance
        let comp1 = ConstituentComparison::new("M2", 1.0, 0.1, 1.05, 0.12);
        assert!(comp1.amplitude_within_10_percent());
        assert!(comp1.phase_within_10_degrees());

        // Outside tolerance
        let comp2 = ConstituentComparison::new("M2", 1.0, 0.5, 1.5, 0.1);
        assert!(!comp2.amplitude_within_10_percent());
        assert!(!comp2.phase_within_10_degrees());
    }
}
