//! Harmonic analysis for tidal time series.
//!
//! Decomposes tidal signals into constituent harmonics using least-squares fitting.
//!
//! # Mathematical Background
//!
//! The tidal signal is modeled as:
//! ```text
//! η(t) = η₀ + Σᵢ [Aᵢ cos(ωᵢt) + Bᵢ sin(ωᵢt)]
//! ```
//!
//! This is rewritten as a linear least-squares problem:
//! ```text
//! y = X * β  where  β = [η₀, A₁, B₁, A₂, B₂, ...]ᵀ
//! ```
//!
//! After solving, the amplitude and phase are recovered:
//! ```text
//! Hᵢ = √(Aᵢ² + Bᵢ²)
//! φᵢ = atan2(-Bᵢ, Aᵢ)
//! ```

use super::TimeSeries;
use crate::boundary::TidalConstituent;
use faer::{Mat, linalg::solvers::Solve};
use std::f64::consts::PI;

/// Result for a single tidal constituent after fitting.
#[derive(Clone, Copy, Debug)]
pub struct ConstituentResult {
    /// Name of the constituent (e.g., "M2", "S2")
    pub name: &'static str,
    /// Period in seconds
    pub period: f64,
    /// Fitted amplitude in meters
    pub amplitude: f64,
    /// Fitted phase in radians [0, 2π)
    pub phase: f64,
}

impl ConstituentResult {
    /// Create a TidalConstituent from this result.
    pub fn to_tidal_constituent(&self) -> TidalConstituent {
        TidalConstituent::new(self.name, self.amplitude, self.period, self.phase)
    }

    /// Evaluate the constituent at time t.
    pub fn evaluate(&self, t: f64) -> f64 {
        let omega = 2.0 * PI / self.period;
        self.amplitude * (omega * t + self.phase).cos()
    }
}

/// Full result from harmonic analysis.
#[derive(Clone, Debug)]
pub struct HarmonicResult {
    /// Mean value (η₀)
    pub mean: f64,
    /// Fitted constituents
    pub constituents: Vec<ConstituentResult>,
    /// Residual variance (unexplained variance)
    pub residual_variance: f64,
    /// Coefficient of determination R²
    pub r_squared: f64,
}

impl HarmonicResult {
    /// Evaluate the fitted harmonic signal at time t.
    pub fn evaluate(&self, t: f64) -> f64 {
        let mut eta = self.mean;
        for c in &self.constituents {
            eta += c.evaluate(t);
        }
        eta
    }

    /// Reconstruct the time series for given times.
    pub fn reconstruct(&self, times: &[f64]) -> Vec<f64> {
        times.iter().map(|&t| self.evaluate(t)).collect()
    }

    /// Get a constituent by name.
    pub fn get_constituent(&self, name: &str) -> Option<&ConstituentResult> {
        self.constituents.iter().find(|c| c.name == name)
    }
}

/// Harmonic analysis configuration and fitting.
///
/// Performs least-squares fitting of tidal constituents to time series data.
#[derive(Clone, Debug)]
pub struct HarmonicAnalysis {
    /// Constituents to fit (with zero amplitude as placeholders)
    constituent_templates: Vec<TidalConstituent>,
}

impl HarmonicAnalysis {
    /// Create analyzer with standard constituents (M2, S2, K1, O1).
    ///
    /// These are the four most common tidal constituents.
    pub fn standard() -> Self {
        Self {
            constituent_templates: vec![
                TidalConstituent::m2(0.0, 0.0),
                TidalConstituent::s2(0.0, 0.0),
                TidalConstituent::k1(0.0, 0.0),
                TidalConstituent::o1(0.0, 0.0),
            ],
        }
    }

    /// Create analyzer with Norwegian coast constituents (M2, S2, N2, K1, O1, P1).
    ///
    /// Includes N2 and P1 which are significant along the Norwegian coast.
    pub fn norwegian_coast() -> Self {
        Self {
            constituent_templates: vec![
                TidalConstituent::m2(0.0, 0.0),
                TidalConstituent::s2(0.0, 0.0),
                TidalConstituent::n2(0.0, 0.0),
                TidalConstituent::k1(0.0, 0.0),
                TidalConstituent::o1(0.0, 0.0),
                TidalConstituent::p1(0.0, 0.0),
            ],
        }
    }

    /// Create analyzer with custom constituents.
    ///
    /// The amplitude and phase values in the constituents are ignored;
    /// only the name and period are used.
    pub fn new(constituents: Vec<TidalConstituent>) -> Self {
        Self {
            constituent_templates: constituents,
        }
    }

    /// Create analyzer with a single constituent.
    pub fn single(constituent: TidalConstituent) -> Self {
        Self {
            constituent_templates: vec![constituent],
        }
    }

    /// Get the constituent periods being analyzed.
    pub fn periods(&self) -> Vec<f64> {
        self.constituent_templates
            .iter()
            .map(|c| c.period)
            .collect()
    }

    /// Get the constituent names being analyzed.
    pub fn names(&self) -> Vec<&'static str> {
        self.constituent_templates.iter().map(|c| c.name).collect()
    }

    /// Minimum record length needed for constituent separation (Rayleigh criterion).
    ///
    /// For reliable separation of two constituents with periods T₁ and T₂,
    /// the record length should satisfy: T > 1 / |1/T₁ - 1/T₂|
    ///
    /// Returns the minimum length needed to separate all constituent pairs.
    pub fn minimum_record_length(&self) -> f64 {
        let periods = self.periods();
        let mut min_length = 0.0;

        for i in 0..periods.len() {
            for j in (i + 1)..periods.len() {
                let f1 = 1.0 / periods[i];
                let f2 = 1.0 / periods[j];
                let df = (f1 - f2).abs();
                if df > 1e-10 {
                    let length = 1.0 / df;
                    if length > min_length {
                        min_length = length;
                    }
                }
            }
        }

        min_length
    }

    /// Fit constituents to time series using least-squares.
    ///
    /// Builds the design matrix and solves the overdetermined system
    /// using the normal equations.
    ///
    /// # Panics
    ///
    /// Panics if the time series has fewer data points than unknowns
    /// (1 + 2 * number of constituents).
    pub fn fit(&self, series: &TimeSeries) -> HarmonicResult {
        let n_data = series.len();
        let n_constituents = self.constituent_templates.len();
        let n_unknowns = 1 + 2 * n_constituents; // mean + (A, B) per constituent

        assert!(
            n_data >= n_unknowns,
            "Need at least {} data points to fit {} constituents, got {}",
            n_unknowns,
            n_constituents,
            n_data
        );

        let times = series.times();
        let values = series.values();

        // Build design matrix A
        // A = [1, cos(ω₁t), sin(ω₁t), cos(ω₂t), sin(ω₂t), ...]
        let mut a = Mat::<f64>::zeros(n_data, n_unknowns);
        for (i, &t) in times.iter().enumerate() {
            a[(i, 0)] = 1.0; // mean
            for (j, c) in self.constituent_templates.iter().enumerate() {
                let omega = c.angular_frequency();
                a[(i, 1 + 2 * j)] = (omega * t).cos();
                a[(i, 2 + 2 * j)] = (omega * t).sin();
            }
        }

        // Solve using normal equations: (A'A) x = A' y
        // First compute A' A (symmetric, n_unknowns × n_unknowns)
        let mut ata = Mat::<f64>::zeros(n_unknowns, n_unknowns);
        for i in 0..n_unknowns {
            for j in 0..n_unknowns {
                let mut sum = 0.0;
                for k in 0..n_data {
                    sum += a[(k, i)] * a[(k, j)];
                }
                ata[(i, j)] = sum;
            }
        }

        // Compute A' y
        let mut aty = Mat::<f64>::zeros(n_unknowns, 1);
        for i in 0..n_unknowns {
            let mut sum = 0.0;
            for k in 0..n_data {
                sum += a[(k, i)] * values[k];
            }
            aty[(i, 0)] = sum;
        }

        // Solve (A'A) x = A'y using LU decomposition
        let lu = ata.as_ref().full_piv_lu();
        let x = lu.solve(&aty);

        // Extract results
        let mean = x[(0, 0)];

        let mut constituents = Vec::with_capacity(n_constituents);
        for (j, c) in self.constituent_templates.iter().enumerate() {
            let a_coef = x[(1 + 2 * j, 0)]; // cosine coefficient
            let b_coef = x[(2 + 2 * j, 0)]; // sine coefficient

            // Amplitude: H = sqrt(A² + B²)
            let amplitude = (a_coef * a_coef + b_coef * b_coef).sqrt();

            // Phase: φ = atan2(-B, A), wrapped to [0, 2π)
            let mut phase = (-b_coef).atan2(a_coef);
            if phase < 0.0 {
                phase += 2.0 * PI;
            }

            constituents.push(ConstituentResult {
                name: c.name,
                period: c.period,
                amplitude,
                phase,
            });
        }

        // Compute residuals and statistics
        let fitted: Vec<f64> = times
            .iter()
            .map(|&t| {
                let mut val = mean;
                for (j, c) in self.constituent_templates.iter().enumerate() {
                    let omega = c.angular_frequency();
                    val += x[(1 + 2 * j, 0)] * (omega * t).cos();
                    val += x[(2 + 2 * j, 0)] * (omega * t).sin();
                }
                val
            })
            .collect();

        let residuals: Vec<f64> = values
            .iter()
            .zip(fitted.iter())
            .map(|(&obs, &fit)| obs - fit)
            .collect();

        let residual_variance = if n_data > 1 {
            residuals.iter().map(|r| r * r).sum::<f64>() / (n_data - 1) as f64
        } else {
            0.0
        };

        // R² = 1 - SS_res / SS_tot
        let total_variance = series.variance();
        let r_squared = if total_variance > 1e-10 {
            1.0 - residual_variance / total_variance
        } else {
            1.0
        };

        HarmonicResult {
            mean,
            constituents,
            residual_variance,
            r_squared,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn make_test_series(times: &[f64], values: &[f64]) -> TimeSeries {
        TimeSeries::new(times, values)
    }

    #[test]
    fn test_single_constituent_recovery() {
        // Generate M2 tide with known amplitude and phase
        let amplitude = 1.5;
        let phase = 0.5;
        let m2 = TidalConstituent::m2(amplitude, phase);

        // Generate 30 days of hourly data (720 points)
        let times: Vec<f64> = (0..720).map(|i| i as f64 * 3600.0).collect();
        let values: Vec<f64> = times.iter().map(|&t| m2.evaluate(t)).collect();

        let series = make_test_series(&times, &values);
        let analysis = HarmonicAnalysis::single(TidalConstituent::m2(0.0, 0.0));
        let result = analysis.fit(&series);

        // Should recover amplitude and phase accurately
        assert!(
            (result.constituents[0].amplitude - amplitude).abs() < 0.01,
            "Amplitude error: expected {}, got {}",
            amplitude,
            result.constituents[0].amplitude
        );
        assert!(
            (result.constituents[0].phase - phase).abs() < 0.01,
            "Phase error: expected {}, got {}",
            phase,
            result.constituents[0].phase
        );

        // Mean should be zero
        assert!(result.mean.abs() < TOL);

        // R² should be very close to 1
        assert!(result.r_squared > 0.9999);
    }

    #[test]
    fn test_mean_plus_constituent() {
        // Signal with mean offset plus M2
        let mean_level = 2.5;
        let amplitude = 1.0;
        let phase = 0.0;
        let m2 = TidalConstituent::m2(amplitude, phase);

        let times: Vec<f64> = (0..500).map(|i| i as f64 * 3600.0).collect();
        let values: Vec<f64> = times.iter().map(|&t| mean_level + m2.evaluate(t)).collect();

        let series = make_test_series(&times, &values);
        let analysis = HarmonicAnalysis::single(TidalConstituent::m2(0.0, 0.0));
        let result = analysis.fit(&series);

        assert!(
            (result.mean - mean_level).abs() < 0.01,
            "Mean error: expected {}, got {}",
            mean_level,
            result.mean
        );
        assert!(
            (result.constituents[0].amplitude - amplitude).abs() < 0.01,
            "Amplitude error"
        );
    }

    #[test]
    fn test_multiple_constituents() {
        // M2 + S2 signal
        let m2 = TidalConstituent::m2(1.0, 0.3);
        let s2 = TidalConstituent::s2(0.4, 0.7);

        // Need at least 15 days to separate M2 and S2 (Rayleigh criterion)
        let times: Vec<f64> = (0..400).map(|i| i as f64 * 3600.0).collect(); // ~16.7 days
        let values: Vec<f64> = times
            .iter()
            .map(|&t| m2.evaluate(t) + s2.evaluate(t))
            .collect();

        let series = make_test_series(&times, &values);
        let analysis = HarmonicAnalysis::new(vec![
            TidalConstituent::m2(0.0, 0.0),
            TidalConstituent::s2(0.0, 0.0),
        ]);
        let result = analysis.fit(&series);

        // M2 recovery
        let m2_result = result.get_constituent("M2").unwrap();
        assert!(
            (m2_result.amplitude - 1.0).abs() < 0.02,
            "M2 amplitude error"
        );
        assert!((m2_result.phase - 0.3).abs() < 0.02, "M2 phase error");

        // S2 recovery
        let s2_result = result.get_constituent("S2").unwrap();
        assert!(
            (s2_result.amplitude - 0.4).abs() < 0.02,
            "S2 amplitude error"
        );
        assert!((s2_result.phase - 0.7).abs() < 0.02, "S2 phase error");
    }

    #[test]
    fn test_reconstruction() {
        let m2 = TidalConstituent::m2(1.0, 0.0);

        let times: Vec<f64> = (0..500).map(|i| i as f64 * 3600.0).collect();
        let values: Vec<f64> = times.iter().map(|&t| 2.0 + m2.evaluate(t)).collect();

        let series = make_test_series(&times, &values);
        let analysis = HarmonicAnalysis::single(TidalConstituent::m2(0.0, 0.0));
        let result = analysis.fit(&series);

        // Reconstruct at original times
        let reconstructed = result.reconstruct(&times);

        // Should match original closely
        for (orig, recon) in values.iter().zip(reconstructed.iter()) {
            assert!(
                (orig - recon).abs() < 0.001,
                "Reconstruction error too large"
            );
        }
    }

    #[test]
    fn test_minimum_record_length() {
        let analysis = HarmonicAnalysis::new(vec![
            TidalConstituent::m2(0.0, 0.0),
            TidalConstituent::s2(0.0, 0.0),
        ]);

        let min_length = analysis.minimum_record_length();

        // M2 = 12.42 hours, S2 = 12.00 hours
        // |1/12.42 - 1/12.00| ≈ 0.00282 cycles/hour
        // T > 1/0.00282 ≈ 355 hours ≈ 14.8 days
        assert!(
            min_length > 300.0 * 3600.0 && min_length < 400.0 * 3600.0,
            "Minimum record length should be ~355 hours, got {} hours",
            min_length / 3600.0
        );
    }

    #[test]
    fn test_standard_analysis() {
        let analysis = HarmonicAnalysis::standard();
        let names = analysis.names();

        assert_eq!(names.len(), 4);
        assert!(names.contains(&"M2"));
        assert!(names.contains(&"S2"));
        assert!(names.contains(&"K1"));
        assert!(names.contains(&"O1"));
    }

    #[test]
    fn test_norwegian_coast_analysis() {
        let analysis = HarmonicAnalysis::norwegian_coast();
        let names = analysis.names();

        assert_eq!(names.len(), 6);
        assert!(names.contains(&"M2"));
        assert!(names.contains(&"N2"));
        assert!(names.contains(&"P1"));
    }

    #[test]
    fn test_constituent_result_to_tidal() {
        let result = ConstituentResult {
            name: "M2",
            period: 12.42 * 3600.0,
            amplitude: 1.5,
            phase: 0.5,
        };

        let constituent = result.to_tidal_constituent();

        assert_eq!(constituent.name, "M2");
        assert!((constituent.amplitude - 1.5).abs() < TOL);
        assert!((constituent.phase - 0.5).abs() < TOL);
    }

    #[test]
    fn test_phase_wrap() {
        // Test that phase is always in [0, 2π)
        let amplitude = 1.0;
        // Use a phase that would naturally result in negative atan2
        let m2 = TidalConstituent::m2(amplitude, 0.0);

        let times: Vec<f64> = (0..500).map(|i| i as f64 * 3600.0).collect();
        let values: Vec<f64> = times.iter().map(|&t| m2.evaluate(t)).collect();

        let series = make_test_series(&times, &values);
        let analysis = HarmonicAnalysis::single(TidalConstituent::m2(0.0, 0.0));
        let result = analysis.fit(&series);

        let phase = result.constituents[0].phase;
        assert!(
            phase >= 0.0 && phase < 2.0 * PI,
            "Phase {} not in [0, 2π)",
            phase
        );
    }
}
