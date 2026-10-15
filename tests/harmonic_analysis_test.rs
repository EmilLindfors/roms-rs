//! Integration tests for the harmonic analysis: fitting reference constants,
//! prediction, inference, the Rayleigh guard, and comparison metrics.

use dg_rs::{
    AnalysisError, ComparisonMetrics, HarmonicAnalysis, Inference, ReferenceConstant, ReferenceFit,
    TimeSeries, compare_harmonics,
};

/// 2025-06-01 00:00 UTC
const T0: f64 = 1_748_736_000.0;

/// `hours` hourly Unix times from `T0`.
fn hourly(hours: usize) -> Vec<f64> {
    (0..hours).map(|i| T0 + i as f64 * 3600.0).collect()
}

/// The tide of reference constants `(name, H, G°)` about `mean` at `times`,
/// in the fit's own convention (`ReferenceFit::predict`).
fn tide(times: &[f64], mean: f64, constants: &[(&'static str, f64, f64)]) -> Vec<f64> {
    ReferenceFit {
        mean,
        constants: constants
            .iter()
            .map(|&(name, amplitude, lag_deg)| ReferenceConstant {
                name,
                amplitude,
                lag_deg,
                inferred: false,
            })
            .collect(),
        r_squared: 1.0,
        residual_rms: 0.0,
    }
    .predict(times)
}

fn lag_error(a: f64, b: f64) -> f64 {
    ((a - b + 180.0).rem_euclid(360.0) - 180.0).abs()
}

#[test]
fn test_m2_recovery() {
    // ~42 days of M2 alone; the other standard constituents fit to nothing
    let times = hourly(1000);
    let values = tide(&times, 0.0, &[("M2", 1.0, 28.6)]);

    let fit = HarmonicAnalysis::standard().fit(&times, &values).unwrap();
    let m2 = fit.get("M2").unwrap();
    assert!((m2.amplitude - 1.0).abs() < 1e-9, "M2 H {}", m2.amplitude);
    assert!(lag_error(m2.lag_deg, 28.6) < 1e-6, "M2 G {}", m2.lag_deg);
    for name in ["S2", "K1", "O1"] {
        assert!(fit.get(name).unwrap().amplitude < 1e-9, "{name}");
    }
}

#[test]
fn test_m2_s2_separation() {
    // 400 hours (~17 days) separate M2 and S2 (14.8 days)
    let times = hourly(400);
    let values = tide(&times, 2.0, &[("M2", 1.0, 0.0), ("S2", 0.5, 45.0)]);

    let fit = HarmonicAnalysis::new(&["M2", "S2"])
        .fit(&times, &values)
        .unwrap();
    assert!((fit.mean - 2.0).abs() < 1e-9, "mean {}", fit.mean);
    let (m2, s2) = (fit.get("M2").unwrap(), fit.get("S2").unwrap());
    assert!((m2.amplitude - 1.0).abs() < 1e-9);
    assert!((s2.amplitude - 0.5).abs() < 1e-9);
    assert!(lag_error(s2.lag_deg, 45.0) < 1e-6, "S2 G {}", s2.lag_deg);
}

/// Ten days cannot separate M2 from S2: an error, where the old
/// record-time fit returned an ill-conditioned answer.
#[test]
fn test_short_record_is_an_error() {
    let times = hourly(240);
    let values = tide(&times, 0.0, &[("M2", 1.0, 0.0), ("S2", 0.5, 45.0)]);
    let err = HarmonicAnalysis::standard()
        .fit(&times, &values)
        .unwrap_err();
    assert!(
        matches!(
            err,
            AnalysisError::NotSeparable {
                a: "M2",
                b: "S2",
                ..
            }
        ),
        "{err}"
    );
    // What 10 days do resolve
    let analysis =
        HarmonicAnalysis::resolvable(&["M2", "S2", "K1", "O1"], times[239] - times[0], 1.0, &[]);
    assert_eq!(analysis.names(), ["M2", "K1"]);
}

#[test]
fn test_norwegian_coast_constituents() {
    // 33 days: M2, S2, N2, K1, O1 fitted, P1 inferred from K1 at its
    // equilibrium ratio (separating it needs half a year)
    let truth = [
        ("M2", 1.2, 5.7),
        ("S2", 0.4, 17.2),
        ("N2", 0.2, 28.6),
        ("K1", 0.15, 40.1),
        ("O1", 0.1, 51.6),
        ("P1", 0.331 * 0.15, 40.1),
    ];
    let times = hourly(800);
    let values = tide(&times, 1.5, &truth);

    let fit = HarmonicAnalysis::norwegian_coast()
        .fit(&times, &values)
        .unwrap();
    for (name, h, g) in truth {
        let c = fit.get(name).unwrap();
        assert!(
            (c.amplitude - h).abs() < 1e-8,
            "{name}: H {} vs {h}",
            c.amplitude
        );
        assert!(
            lag_error(c.lag_deg, g) < 1e-6,
            "{name}: G {} vs {g}",
            c.lag_deg
        );
    }
    assert!(fit.get("P1").unwrap().inferred);
    assert!(fit.r_squared > 1.0 - 1e-12, "R² {}", fit.r_squared);
}

#[test]
fn test_reconstruction() {
    let times = hourly(500);
    let original = tide(&times, 5.0, &[("M2", 1.0, 0.0)]);

    let fit = HarmonicAnalysis::new(&["M2"])
        .fit_series(&TimeSeries::new(&times, &original))
        .unwrap();
    let metrics = ComparisonMetrics::compute(&fit.predict(&times), &original);
    assert!(metrics.rmse < 1e-9, "RMSE {}", metrics.rmse);
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

/// Model and observation over different spans compare constituent by
/// constituent: reference constants do not depend on the record.
#[test]
fn test_harmonic_comparison() {
    let model_times = hourly(500);
    let obs_times: Vec<f64> = hourly(700)
        .into_iter()
        .map(|t| t + 9.0 * 86_400.0)
        .collect();
    let model = tide(&model_times, 0.0, &[("M2", 1.0, 30.0)]);
    let obs = tide(&obs_times, 0.0, &[("M2", 1.1, 31.0)]);

    let analysis = HarmonicAnalysis::new(&["M2"]);
    let comparisons = compare_harmonics(
        &analysis.fit(&model_times, &model).unwrap(),
        &analysis.fit(&obs_times, &obs).unwrap(),
    );
    assert_eq!(comparisons.len(), 1);
    let m2 = &comparisons[0];
    assert!((m2.amplitude_error - 0.1).abs() < 1e-9);
    assert!((m2.amplitude_ratio - 1.0 / 1.1).abs() < 1e-9);
    assert!((m2.phase_error_degrees() + 1.0).abs() < 1e-6, "{m2:?}");
}

#[test]
fn test_minimum_record_length() {
    // M2 and S2 set it for the standard four: 354.4 hours
    let min_hours = HarmonicAnalysis::standard().minimum_record_length() / 3600.0;
    assert!(
        (min_hours - 354.37).abs() < 0.1,
        "Minimum record length should be ~354 hours, got {min_hours:.1} hours"
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
    assert_eq!(series.times(), times.as_slice());

    // Variance of [1,2,3,4,5] = 2.5 (sample variance)
    assert!((series.variance() - 2.5).abs() < 1e-10, "Variance error");

    let shifted = series.map_values(|v| v + 0.5);
    assert!((shifted.mean() - 3.5).abs() < 1e-10);
    assert_eq!(shifted.name.as_deref(), Some("Test Station"));
}

#[test]
fn test_noisy_signal_recovery() {
    let times = hourly(1000);
    let mut values = tide(&times, 0.0, &[("M2", 1.0, 0.0)]);
    // Deterministic "noise" (small amplitude)
    for (i, v) in values.iter_mut().enumerate() {
        *v += 0.01 * (i as f64 * 0.1).sin();
    }

    let fit = HarmonicAnalysis::new(&["M2"]).fit(&times, &values).unwrap();
    let m2 = fit.get("M2").unwrap();
    assert!((m2.amplitude - 1.0).abs() < 0.02, "M2 H {}", m2.amplitude);
    assert!(fit.r_squared > 0.99, "R² {}", fit.r_squared);
}

#[test]
fn test_phase_wraparound() {
    // Lags near 360° and 0° are the same; the fit returns [0, 360)
    let times = hourly(500);
    let values = tide(&times, 0.0, &[("M2", 1.0, 354.3)]);

    let fit = HarmonicAnalysis::new(&["M2"]).fit(&times, &values).unwrap();
    let g = fit.get("M2").unwrap().lag_deg;
    assert!((0.0..360.0).contains(&g), "G {g} not in [0, 360)");
    assert!(lag_error(g, 354.3) < 1e-6, "G {g}");
}

/// An inference whose ratio comes from a reference fit (a longer record at
/// the same place) reproduces that reference's ratio in a short fit.
#[test]
fn test_inference_from_a_reference_fit() {
    let truth = [("M2", 0.74, 301.2), ("N2", 0.154, 279.1)];
    let year = hourly(24 * 366);
    let reference = HarmonicAnalysis::new(&["M2", "N2"])
        .fit(&year, &tide(&year, 0.0, &truth))
        .unwrap();

    let fortnight = hourly(24 * 15);
    let values = tide(&fortnight, 0.0, &truth);
    let fit = HarmonicAnalysis::resolvable(
        &["M2", "N2"],
        fortnight[fortnight.len() - 1] - fortnight[0],
        1.0,
        &[Inference::N2_FROM_M2.or_from(Some(&reference))],
    )
    .fit(&fortnight, &values)
    .unwrap();
    let n2 = fit.get("N2").unwrap();
    assert!(n2.inferred);
    assert!((n2.amplitude - 0.154).abs() < 1e-8, "N2 H {}", n2.amplitude);
    assert!(lag_error(n2.lag_deg, 279.1) < 1e-6, "N2 G {}", n2.lag_deg);
}
