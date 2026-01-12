//! Integration tests for harmonic analysis module.
//!
//! Tests the full analysis pipeline including fitting, reconstruction,
//! and comparison metrics.

use dg_rs::{ComparisonMetrics, HarmonicAnalysis, TidalConstituent, TimeSeries, compare_harmonics};
use std::f64::consts::PI;

/// Generate a synthetic tidal signal from multiple constituents.
fn generate_tidal_signal(constituents: &[TidalConstituent], mean: f64, times: &[f64]) -> Vec<f64> {
    times
        .iter()
        .map(|&t| {
            let mut eta = mean;
            for c in constituents {
                eta += c.evaluate(t);
            }
            eta
        })
        .collect()
}

#[test]
fn test_m2_recovery() {
    // Generate M2 tide: A=1.0m, φ=0.5rad
    let m2 = TidalConstituent::m2(1.0, 0.5);

    // 1000 hours of hourly data (~42 days)
    let times: Vec<f64> = (0..1000).map(|i| i as f64 * 3600.0).collect();
    let values = generate_tidal_signal(&[m2.clone()], 0.0, &times);

    let series = TimeSeries::new(&times, &values);
    let analysis = HarmonicAnalysis::standard();
    let result = analysis.fit(&series);

    let m2_result = result.get_constituent("M2").unwrap();

    // Should recover amplitude within 1%
    assert!(
        (m2_result.amplitude - 1.0).abs() < 0.01,
        "M2 amplitude error: expected 1.0, got {:.4}",
        m2_result.amplitude
    );

    // Should recover phase within 0.01 radians
    assert!(
        (m2_result.phase - 0.5).abs() < 0.01,
        "M2 phase error: expected 0.5, got {:.4}",
        m2_result.phase
    );

    // Other constituents should have near-zero amplitude
    let s2_result = result.get_constituent("S2").unwrap();
    assert!(
        s2_result.amplitude < 0.01,
        "S2 should be near-zero, got {:.4}",
        s2_result.amplitude
    );
}

#[test]
fn test_m2_s2_separation() {
    // Test Rayleigh criterion: need ~15 days to separate M2 and S2
    let m2 = TidalConstituent::m2(1.0, 0.0);
    let s2 = TidalConstituent::s2(0.5, PI / 4.0);

    // 400 hours (~17 days, enough for M2/S2 separation)
    let times: Vec<f64> = (0..400).map(|i| i as f64 * 3600.0).collect();
    let values = generate_tidal_signal(&[m2.clone(), s2.clone()], 2.0, &times);

    let series = TimeSeries::new(&times, &values);
    let analysis = HarmonicAnalysis::new(vec![
        TidalConstituent::m2(0.0, 0.0),
        TidalConstituent::s2(0.0, 0.0),
    ]);
    let result = analysis.fit(&series);

    // Mean should be recovered
    assert!(
        (result.mean - 2.0).abs() < 0.01,
        "Mean error: expected 2.0, got {:.4}",
        result.mean
    );

    // M2 recovery
    let m2_result = result.get_constituent("M2").unwrap();
    assert!(
        (m2_result.amplitude - 1.0).abs() < 0.02,
        "M2 amplitude error"
    );

    // S2 recovery
    let s2_result = result.get_constituent("S2").unwrap();
    assert!(
        (s2_result.amplitude - 0.5).abs() < 0.02,
        "S2 amplitude error"
    );

    // Phase recovery for S2
    assert!(
        (s2_result.phase - PI / 4.0).abs() < 0.05,
        "S2 phase error: expected {:.4}, got {:.4}",
        PI / 4.0,
        s2_result.phase
    );
}

#[test]
fn test_norwegian_coast_constituents() {
    // Test full Norwegian coast analysis with 6 constituents
    let constituents = vec![
        TidalConstituent::m2(1.2, 0.1),  // Dominant
        TidalConstituent::s2(0.4, 0.3),  // Second
        TidalConstituent::n2(0.2, 0.5),  // Third
        TidalConstituent::k1(0.15, 0.7), // Diurnal
        TidalConstituent::o1(0.1, 0.9),  // Diurnal
        TidalConstituent::p1(0.05, 1.1), // Diurnal
    ];

    // Need longer record for diurnal/semidiurnal separation (~30 days)
    let times: Vec<f64> = (0..800).map(|i| i as f64 * 3600.0).collect();
    let values = generate_tidal_signal(&constituents, 1.5, &times);

    let series = TimeSeries::new(&times, &values);
    let analysis = HarmonicAnalysis::norwegian_coast();
    let result = analysis.fit(&series);

    // Check that all constituents are recovered reasonably
    for c in &constituents {
        let fitted = result.get_constituent(c.name).unwrap();
        let amp_error = (fitted.amplitude - c.amplitude).abs();
        assert!(
            amp_error < 0.05,
            "{} amplitude error too large: expected {:.3}, got {:.3}",
            c.name,
            c.amplitude,
            fitted.amplitude
        );
    }

    // R² should be very high
    assert!(
        result.r_squared > 0.99,
        "R² should be > 0.99, got {:.4}",
        result.r_squared
    );
}

#[test]
fn test_reconstruction() {
    let m2 = TidalConstituent::m2(1.0, 0.0);
    let mean_level = 5.0;

    let times: Vec<f64> = (0..500).map(|i| i as f64 * 3600.0).collect();
    let original = generate_tidal_signal(&[m2], mean_level, &times);

    let series = TimeSeries::new(&times, &original);
    let analysis = HarmonicAnalysis::single(TidalConstituent::m2(0.0, 0.0));
    let result = analysis.fit(&series);

    // Reconstruct
    let reconstructed = result.reconstruct(&times);

    // Compare original and reconstructed
    let metrics = ComparisonMetrics::compute(&reconstructed, &original);

    assert!(
        metrics.rmse < 0.001,
        "Reconstruction RMSE too large: {:.6}",
        metrics.rmse
    );
    assert!(
        (metrics.correlation - 1.0).abs() < 0.0001,
        "Correlation not perfect: {:.6}",
        metrics.correlation
    );
}

#[test]
fn test_comparison_metrics() {
    // Perfect model
    let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let model_perfect = obs.clone();

    let metrics = ComparisonMetrics::compute(&model_perfect, &obs);
    assert!(metrics.rmse < 1e-10);
    assert!((metrics.correlation - 1.0).abs() < 1e-10);
    assert!((metrics.skill_score - 1.0).abs() < 1e-10);

    // Biased model
    let model_biased: Vec<f64> = obs.iter().map(|&x| x + 0.1).collect();
    let metrics_biased = ComparisonMetrics::compute(&model_biased, &obs);
    assert!((metrics_biased.bias - 0.1).abs() < 1e-10);
    assert!((metrics_biased.correlation - 1.0).abs() < 1e-10); // Still perfect correlation
}

#[test]
fn test_harmonic_comparison() {
    // Create "model" and "observation" harmonic results
    let m2_model = TidalConstituent::m2(1.0, 0.5);
    let m2_obs = TidalConstituent::m2(1.1, 0.52);

    let times: Vec<f64> = (0..500).map(|i| i as f64 * 3600.0).collect();

    let model_values = generate_tidal_signal(&[m2_model], 0.0, &times);
    let obs_values = generate_tidal_signal(&[m2_obs], 0.0, &times);

    let model_series = TimeSeries::new(&times, &model_values);
    let obs_series = TimeSeries::new(&times, &obs_values);

    let analysis = HarmonicAnalysis::single(TidalConstituent::m2(0.0, 0.0));
    let model_result = analysis.fit(&model_series);
    let obs_result = analysis.fit(&obs_series);

    let comparisons = compare_harmonics(&model_result, &obs_result);

    assert_eq!(comparisons.len(), 1);
    let m2_comp = &comparisons[0];

    // Amplitude error should be ~0.1
    assert!(
        (m2_comp.amplitude_error - 0.1).abs() < 0.02,
        "Amplitude error: expected ~0.1, got {:.4}",
        m2_comp.amplitude_error
    );

    // Amplitude ratio should be ~0.91
    assert!(
        (m2_comp.amplitude_ratio - 1.0 / 1.1).abs() < 0.02,
        "Amplitude ratio: expected ~0.91, got {:.4}",
        m2_comp.amplitude_ratio
    );

    // Phase error should be ~0.02 radians
    assert!(
        m2_comp.phase_error.abs() < 0.05,
        "Phase error too large: {:.4}",
        m2_comp.phase_error
    );
}

#[test]
fn test_minimum_record_length() {
    // Standard analysis (M2, S2, K1, O1)
    let analysis = HarmonicAnalysis::standard();
    let min_length = analysis.minimum_record_length();

    // M2 vs S2 separation requires ~355 hours
    // Convert to hours for easier verification
    let min_hours = min_length / 3600.0;
    assert!(
        min_hours > 300.0 && min_hours < 400.0,
        "Minimum record length should be ~355 hours, got {:.1} hours",
        min_hours
    );
}

#[test]
fn test_time_series_methods() {
    let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let series = TimeSeries::new(&times, &values)
        .with_location(100.0, 200.0)
        .with_name("Test Station");

    assert_eq!(series.len(), 5);
    assert!((series.duration() - 4.0).abs() < 1e-10);
    assert!((series.mean() - 3.0).abs() < 1e-10);
    assert!(series.location.is_some());
    assert_eq!(series.name.as_deref(), Some("Test Station"));

    // Variance of [1,2,3,4,5] = 2.5 (sample variance)
    let expected_variance = 2.5;
    assert!(
        (series.variance() - expected_variance).abs() < 1e-10,
        "Variance error"
    );
}

#[test]
fn test_noisy_signal_recovery() {
    // Add small noise to tidal signal
    let m2 = TidalConstituent::m2(1.0, 0.0);

    let times: Vec<f64> = (0..1000).map(|i| i as f64 * 3600.0).collect();
    let mut values = generate_tidal_signal(&[m2], 0.0, &times);

    // Add deterministic "noise" pattern (small amplitude)
    for (i, v) in values.iter_mut().enumerate() {
        *v += 0.01 * (i as f64 * 0.1).sin();
    }

    let series = TimeSeries::new(&times, &values);
    let analysis = HarmonicAnalysis::single(TidalConstituent::m2(0.0, 0.0));
    let result = analysis.fit(&series);

    // Should still recover M2 accurately
    let m2_result = result.get_constituent("M2").unwrap();
    assert!(
        (m2_result.amplitude - 1.0).abs() < 0.02,
        "M2 amplitude with noise: expected 1.0, got {:.4}",
        m2_result.amplitude
    );

    // R² should still be high
    assert!(
        result.r_squared > 0.99,
        "R² with noise: {:.4}",
        result.r_squared
    );
}

#[test]
fn test_constituent_evaluate() {
    let result = dg_rs::ConstituentResult {
        name: "M2",
        period: 12.42 * 3600.0,
        amplitude: 1.5,
        phase: 0.0,
    };

    // At t=0, should equal amplitude (cos(0) = 1)
    assert!((result.evaluate(0.0) - 1.5).abs() < 1e-10);

    // At t=half_period, should be -amplitude (cos(π) = -1)
    let half_period = result.period / 2.0;
    assert!((result.evaluate(half_period) - (-1.5)).abs() < 1e-10);
}

#[test]
fn test_phase_wraparound() {
    // Test that phases near 2π and 0 are handled correctly
    let m2 = TidalConstituent::m2(1.0, 2.0 * PI - 0.1);

    let times: Vec<f64> = (0..500).map(|i| i as f64 * 3600.0).collect();
    let values = generate_tidal_signal(&[m2], 0.0, &times);

    let series = TimeSeries::new(&times, &values);
    let analysis = HarmonicAnalysis::single(TidalConstituent::m2(0.0, 0.0));
    let result = analysis.fit(&series);

    let fitted_phase = result.constituents[0].phase;

    // Phase should be in [0, 2π)
    assert!(
        fitted_phase >= 0.0 && fitted_phase < 2.0 * PI,
        "Phase {} not in [0, 2π)",
        fitted_phase
    );

    // Should match original phase (near 2π - 0.1)
    let phase_diff = (fitted_phase - (2.0 * PI - 0.1)).abs();
    let phase_diff_wrapped = phase_diff.min(2.0 * PI - phase_diff);
    assert!(
        phase_diff_wrapped < 0.01,
        "Phase recovery error: expected {:.4}, got {:.4}",
        2.0 * PI - 0.1,
        fitted_phase
    );
}
