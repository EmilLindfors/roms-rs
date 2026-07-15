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
use crate::tides::{AstronomicalArguments, NodalCorrection, nodal_correction};
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

    /// Remove a nodal correction from this *apparent* constituent, recovering the
    /// nodal-corrected *reference* constants (`H_ref`, `φ_ref = −G`).
    ///
    /// Harmonic analysis of a finite record fits the *apparent* amplitude and
    /// phase seen during that record: `H_app = f·H_ref` and, in the crate's
    /// internal phase convention (`η = H·cos(ω t + φ)`, so `φ = −G`),
    /// `φ_app = φ_ref + (V₀ + u)` — the equilibrium argument `V₀` and nodal phase
    /// `u` are baked in. This is the inverse of [`crate::tides::correct_amplitude_phase`]:
    ///
    /// ```text
    /// H_ref = H_app / f,   φ_ref = φ_app − (V₀ + u)
    /// ```
    ///
    /// so the result is comparable to a published catalogue's `(H, G)` pairs.
    /// A degenerate `f` (≈ 0) leaves the amplitude untouched.
    pub fn remove_nodal_correction(&self, correction: &NodalCorrection) -> ConstituentResult {
        let amplitude = if correction.f.abs() > 1e-12 {
            self.amplitude / correction.f
        } else {
            self.amplitude
        };
        let phase = (self.phase - correction.phase_offset_rad()).rem_euclid(2.0 * PI);
        ConstituentResult {
            name: self.name,
            period: self.period,
            amplitude,
            phase,
        }
    }

    /// Convert this apparent constituent into reference constants at `epoch`.
    ///
    /// `epoch` is the astronomical state at the analysed record's time origin
    /// (`t = 0`); see [`Self::remove_nodal_correction`]. Constituents with no
    /// tabulated nodal correction (see [`crate::tides::nodal_correction`]) are
    /// returned unchanged.
    pub fn to_reference(&self, epoch: &AstronomicalArguments) -> ConstituentResult {
        match nodal_correction(self.name, epoch) {
            Some(correction) => self.remove_nodal_correction(&correction),
            None => *self,
        }
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

    /// Nodal-correct every fitted constituent to reference constants at `epoch`.
    ///
    /// `epoch` must be the astronomical state at the analysed record's time
    /// origin (`t = 0`), i.e. `AstronomicalArguments::at_datetime(...)` for the
    /// UTC instant that the first sample was taken. The returned constituents
    /// carry reference amplitude `H` and internal phase `φ = −G`, directly
    /// comparable to a published harmonic catalogue (Kartverket, ROMS, …).
    ///
    /// The mean and fit statistics are not returned because reference constants
    /// no longer reconstruct the original series (they omit the nodal factors);
    /// see [`ConstituentResult::to_reference`].
    pub fn reference_constants(&self, epoch: &AstronomicalArguments) -> Vec<ConstituentResult> {
        self.constituents
            .iter()
            .map(|c| c.to_reference(epoch))
            .collect()
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

    /// Smallest signed angular difference `a − b`, wrapped to `(−π, π]`.
    fn angle_diff(a: f64, b: f64) -> f64 {
        let mut d = (a - b) % (2.0 * PI);
        if d > PI {
            d -= 2.0 * PI;
        } else if d <= -PI {
            d += 2.0 * PI;
        }
        d
    }

    /// End-to-end nodal-correction inference: start from reference constants,
    /// apply the forward Schureman correction at a chosen epoch to get the
    /// *apparent* signal, synthesize and fit it, then `to_reference` must recover
    /// the original reference amplitude and phase. This ties the harmonic-analysis
    /// inference to `tides::correct_amplitude_phase` (its exact inverse).
    #[test]
    fn test_reference_constants_round_trip() {
        use crate::tides::{AstronomicalArguments, correct_amplitude_phase, nodal_correction};

        // Epoch = record's t = 0 instant.
        let epoch = AstronomicalArguments::at_datetime(2024, 6, 1, 0, 0, 0.0);

        // Reference K1: H = 0.10 m, Greenwich lag G = 45° → internal φ = −G.
        let h_ref = 0.10_f64;
        let phi_ref = (-45.0_f64.to_radians()).rem_euclid(2.0 * PI);

        // Forward correction → apparent constants seen during the record.
        let corr = nodal_correction("K1", &epoch).unwrap();
        let (h_app, phi_app) = correct_amplitude_phase(h_ref, phi_ref, &corr);
        // The nodal factor genuinely moves the amplitude (guards a no-op epoch).
        assert!(
            (h_app - h_ref).abs() > 1e-3,
            "f(K1) too close to 1 at epoch"
        );

        // Synthesize ~30 days of hourly apparent K1 and fit it.
        let apparent = TidalConstituent::k1(h_app, phi_app);
        let times: Vec<f64> = (0..720).map(|i| i as f64 * 3600.0).collect();
        let values: Vec<f64> = times.iter().map(|&t| apparent.evaluate(t)).collect();
        let series = make_test_series(&times, &values);
        let result = HarmonicAnalysis::single(TidalConstituent::k1(0.0, 0.0)).fit(&series);

        // The fit recovers the apparent constants...
        let fitted = &result.constituents[0];
        assert!(
            (fitted.amplitude - h_app).abs() < 1e-3,
            "apparent amplitude"
        );
        assert!(
            angle_diff(fitted.phase, phi_app).abs() < 1e-3,
            "apparent phase"
        );

        // ...and inference recovers the reference constants we started from.
        let reference = result.reference_constants(&epoch);
        assert!(
            (reference[0].amplitude - h_ref).abs() < 1e-3,
            "reference amplitude: expected {h_ref}, got {}",
            reference[0].amplitude
        );
        assert!(
            angle_diff(reference[0].phase, phi_ref).abs() < 1e-3,
            "reference phase: expected {phi_ref}, got {}",
            reference[0].phase
        );
    }

    /// Constituents with no tabulated nodal correction pass through untouched.
    #[test]
    fn test_reference_constants_unknown_constituent_unchanged() {
        use crate::tides::AstronomicalArguments;

        let epoch = AstronomicalArguments::at_datetime(2024, 1, 1, 0, 0, 0.0);
        let c = ConstituentResult {
            name: "XYZ",
            period: 12.0 * 3600.0,
            amplitude: 0.5,
            phase: 1.2,
        };
        let r = c.to_reference(&epoch);
        assert!((r.amplitude - 0.5).abs() < TOL);
        assert!((r.phase - 1.2).abs() < TOL);
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
