//! End-to-end tidal validation against NorKyst-800, via the text-ingest reader.
//!
//! This wires the full P1.5 comparison path:
//!
//! 1. Read `norkyst-client --format text` point output with
//!    [`dg_rs::io::read_norkyst_text_file`] → a sea-surface-height (`zeta`) series.
//! 2. Fit tidal constituents with [`HarmonicAnalysis`].
//! 3. Strip the nodal modulation with [`HarmonicResult::reference_constants`] so the
//!    fitted *apparent* constants become catalogue-comparable *reference* constants.
//! 4. Compare `(H, G)` against a published harmonic catalogue and print skill numbers.
//!
//! ## Run
//!
//! Synthetic self-check (no data needed) — synthesizes a tide from a known
//! catalogue, applies the forward Schureman nodal correction at a chosen epoch,
//! writes it as a NorKyst text file, reads it back through the real reader, and
//! confirms the inference recovers the catalogue:
//!
//! ```bash
//! cargo run --example norkyst_tidal_validation --no-default-features --features parallel,simd
//! ```
//!
//! Real data. Point series (text) fetched with
//! `norkyst-client … --format text -o series.txt`:
//!
//! ```bash
//! cargo run --example norkyst_tidal_validation -- series.txt
//! ```
//!
//! Grid parquet (the reliable OPeNDAP path) fetched with
//! `norkyst-client … --bbox … --format parquet -o dataset/` — pass the dataset
//! directory (or a single `.parquet`) plus optional `lon lat` to pick the
//! nearest cell (defaults to Bergen). Requires the `parquet` feature:
//!
//! ```bash
//! cargo run --example norkyst_tidal_validation \
//!   --features parquet -- dataset/ 5.32 60.39
//! ```
//!
//! In real-data mode the record's first sample time is taken as the analysis
//! epoch, so the reference constants it prints are directly comparable to a
//! catalogue (Kartverket / ROMS / NorKyst harmonic constants).

use std::f64::consts::PI;

use dg_rs::analysis::{HarmonicAnalysis, HarmonicResult, TimeSeries};
use dg_rs::boundary::TidalConstituent;
use dg_rs::io::read_norkyst_text_file;
use dg_rs::tides::{AstronomicalArguments, correct_amplitude_phase, julian_date, nodal_correction};

/// Julian Date of the Unix epoch (1970-01-01 00:00:00 UTC).
const UNIX_EPOCH_JD: f64 = 2_440_587.5;

/// A published harmonic constant: amplitude `h` (m) and Greenwich phase lag `g` (deg).
struct Catalogue {
    name: &'static str,
    h: f64,
    g_deg: f64,
}

/// Synthetic Bergen-like west-coast catalogue (NOT official values — for the
/// self-check only). Amplitudes in metres, Greenwich lags in degrees.
const BERGEN_LIKE: &[Catalogue] = &[
    Catalogue {
        name: "M2",
        h: 0.55,
        g_deg: 175.0,
    },
    Catalogue {
        name: "S2",
        h: 0.18,
        g_deg: 200.0,
    },
    Catalogue {
        name: "N2",
        h: 0.11,
        g_deg: 155.0,
    },
    Catalogue {
        name: "K1",
        h: 0.065,
        g_deg: 100.0,
    },
    Catalogue {
        name: "O1",
        h: 0.055,
        g_deg: 330.0,
    },
    Catalogue {
        name: "P1",
        h: 0.020,
        g_deg: 95.0,
    },
];

fn main() {
    match std::env::args().nth(1) {
        Some(path) => {
            // Optional station coords (used to pick the nearest grid cell for
            // gridded parquet input); default to Bergen.
            let lon = std::env::args()
                .nth(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(5.32);
            let lat = std::env::args()
                .nth(3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(60.39);
            real_data_mode(&path, lon, lat);
        }
        None => synthetic_self_check(),
    }
}

// ---------------------------------------------------------------------------
// Real-data mode
// ---------------------------------------------------------------------------

fn real_data_mode(path: &str, lon: f64, lat: f64) {
    let zeta = match load_zeta(path, lon, lat) {
        Some(z) => z,
        None => std::process::exit(1),
    };

    let analysis = HarmonicAnalysis::norwegian_coast();
    let (rel, epoch) = match to_relative_series_and_epoch(&zeta, &analysis) {
        Some(x) => x,
        None => std::process::exit(1),
    };

    let fit = analysis.fit(&rel);
    let refs = fit.reference_constants(&epoch);

    println!("\nInferred reference constants (nodal correction removed):");
    println!(
        "  fit R² = {:.4}, record length = {:.1} days",
        fit.r_squared,
        rel.duration() / 86_400.0
    );
    println!(
        "  {:<4} {:>10} {:>10} {:>12}",
        "name", "H (m)", "G (deg)", "period (h)"
    );
    for c in &refs {
        println!(
            "  {:<4} {:>10.4} {:>10.1} {:>12.3}",
            c.name,
            c.amplitude,
            greenwich_lag_deg(c.phase),
            c.period / 3600.0,
        );
    }
    println!(
        "\nCompare these against the station's published catalogue to score the model \
         (RMS amplitude error, RMS phase error)."
    );
}

/// Load a sea-surface-height series from either the text point format or, with
/// the `parquet` feature, a norkyst-client **grid** parquet file/dataset
/// (picking the wet cell nearest `(lon, lat)`). Returns `None` (after printing
/// why) if nothing usable is found.
#[cfg_attr(not(feature = "parquet"), allow(unused_variables))]
fn load_zeta(path: &str, lon: f64, lat: f64) -> Option<TimeSeries> {
    let p = std::path::Path::new(path);
    let looks_parquet = p.is_dir() || p.extension().is_some_and(|e| e == "parquet");

    if looks_parquet {
        #[cfg(feature = "parquet")]
        {
            println!("Reading NorKyst grid parquet: {path}");
            let data = match dg_rs::io::read_norkyst_parquet_glob(p) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("error: {e}");
                    return None;
                }
            };
            let zeta = data.sea_surface_height_series_nearest(lon, lat);
            println!(
                "  {} grid rows; nearest wet cell to ({lon}, {lat}) → {} zeta samples",
                data.len(),
                zeta.len()
            );
            if zeta.is_empty() {
                eprintln!("error: no wet cell with sea-surface height near ({lon}, {lat}).");
                return None;
            }
            return Some(zeta);
        }
        #[cfg(not(feature = "parquet"))]
        {
            eprintln!(
                "error: {path} looks like parquet, but this build lacks the `parquet` feature.\n\
                 Rebuild with `--features parquet` to read grid parquet natively."
            );
            return None;
        }
    }

    println!("Reading NorKyst text output: {path}");
    let data = match read_norkyst_text_file(p) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            return None;
        }
    };
    let zeta = data.sea_surface_height_series();
    println!(
        "  {} snapshots, {} with sea-surface height",
        data.len(),
        zeta.len()
    );
    if zeta.is_empty() {
        eprintln!(
            "error: no sea-surface height in the record. This needs a norkyst-client build \
             whose text writer emits the `surface …` line."
        );
        return None;
    }
    Some(zeta)
}

// ---------------------------------------------------------------------------
// Synthetic self-check
// ---------------------------------------------------------------------------

fn synthetic_self_check() {
    println!("Synthetic round-trip self-check (no external data)\n");

    // Analysis epoch = the record's first-sample instant.
    let (year, month, day) = (2024, 6, 1);
    let epoch_jd = julian_date(year, month, day, 0, 0, 0.0);
    let epoch = AstronomicalArguments::at_julian_date(epoch_jd);
    let epoch_unix = ((epoch_jd - UNIX_EPOCH_JD) * 86_400.0).round() as i64;

    // Forward-correct each catalogue constituent to the *apparent* constants a
    // finite record starting at `epoch` would actually show.
    let mean = 0.30_f64; // arbitrary datum offset; harmonic inference ignores the mean
    let apparent: Vec<TidalConstituent> = BERGEN_LIKE
        .iter()
        .map(|c| {
            let phi_ref = internal_phase(c.g_deg);
            let (h_app, phi_app) = match nodal_correction(c.name, &epoch) {
                Some(corr) => correct_amplitude_phase(c.h, phi_ref, &corr),
                None => (c.h, phi_ref),
            };
            named_constituent(c.name, h_app, phi_app)
        })
        .collect();

    // Synthesize 30 days of hourly zeta and write it in the norkyst-client text
    // format, so the real reader (not a shortcut) parses it back.
    let n_hours = 30 * 24;
    let mut text = String::new();
    for i in 0..n_hours {
        let t_rel = i as f64 * 3600.0;
        let zeta: f64 = mean + apparent.iter().map(|c| c.evaluate(t_rel)).sum::<f64>();
        let stamp = format_utc(epoch_unix + i as i64 * 3600);
        text.push_str("site_id: 999\n");
        text.push_str(&format!("time={stamp} lat=60.39 lon=5.32\n"));
        text.push_str(&format!(
            "surface sea_surface_height=Some({zeta}) bottom_depth=Some(300.0)\n"
        ));
    }

    // Round-trip through a real file and the real reader.
    let file = std::env::temp_dir().join("norkyst_selfcheck_series.txt");
    std::fs::write(&file, &text).expect("write temp series");
    println!("Wrote synthetic series: {}", file.display());
    let data = read_norkyst_text_file(&file).expect("read back series");
    let zeta_series = data.sea_surface_height_series();
    println!(
        "Read back {} snapshots ({} with zeta)\n",
        data.len(),
        zeta_series.len()
    );

    // Fit and infer reference constants at the epoch.
    let analysis = HarmonicAnalysis::norwegian_coast();
    let (rel, epoch2) =
        to_relative_series_and_epoch(&zeta_series, &analysis).expect("record long enough");
    let fit = analysis.fit(&rel);
    let refs = fit.reference_constants(&epoch2);

    print_comparison(&fit, &refs);
    let _ = std::fs::remove_file(&file);
}

/// Print the inferred reference constants next to the catalogue they came from.
fn print_comparison(fit: &HarmonicResult, refs: &[dg_rs::analysis::ConstituentResult]) {
    println!("Recovered catalogue (fit R² = {:.6}):", fit.r_squared);
    println!(
        "  {:<4} {:>8} {:>8} {:>8}   {:>8} {:>8} {:>8}",
        "name", "H_cat", "H_fit", "ΔH", "G_cat", "G_fit", "ΔG"
    );

    let mut sum_dh2 = 0.0;
    let mut sum_dg2 = 0.0;
    let mut n = 0.0;
    for cat in BERGEN_LIKE {
        let Some(c) = refs.iter().find(|r| r.name == cat.name) else {
            continue;
        };
        let h_fit = c.amplitude;
        let g_fit = greenwich_lag_deg(c.phase);
        let dh = h_fit - cat.h;
        let dg = wrap_deg_signed(g_fit - cat.g_deg);
        sum_dh2 += dh * dh;
        sum_dg2 += dg * dg;
        n += 1.0;
        println!(
            "  {:<4} {:>8.4} {:>8.4} {:>8.4}   {:>8.1} {:>8.1} {:>8.2}",
            cat.name, cat.h, h_fit, dh, cat.g_deg, g_fit, dg
        );
    }

    if n > 0.0 {
        println!(
            "\n  RMS amplitude error: {:.5} m    RMS phase error: {:.3}°",
            (sum_dh2 / n).sqrt(),
            (sum_dg2 / n).sqrt()
        );
        println!(
            "  (Both should be ~0: the reader + harmonic inference reproduce the catalogue \
             the synthetic tide was built from.)"
        );
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Shift an absolute-time (Unix-second) series to start at `t = 0` and return the
/// astronomical epoch for that origin, so `reference_constants` is self-consistent.
///
/// Returns `None` (after printing why) if the record is too short to fit.
fn to_relative_series_and_epoch(
    series: &TimeSeries,
    analysis: &HarmonicAnalysis,
) -> Option<(TimeSeries, AstronomicalArguments)> {
    let n_unknowns = 1 + 2 * analysis.names().len();
    if series.len() < n_unknowns {
        eprintln!(
            "error: need ≥{} samples to fit {} constituents, got {}",
            n_unknowns,
            analysis.names().len(),
            series.len()
        );
        return None;
    }

    let times = series.times();
    let values = series.values();
    let t0 = times[0];
    let rel_times: Vec<f64> = times.iter().map(|&t| t - t0).collect();
    let rel = TimeSeries::new(&rel_times, &values);

    let min_len = analysis.minimum_record_length();
    if rel.duration() < min_len {
        eprintln!(
            "warning: record spans {:.1} days but Rayleigh separation of these constituents \
             needs {:.1} days — constituents may be aliased.",
            rel.duration() / 86_400.0,
            min_len / 86_400.0
        );
    }

    let epoch = AstronomicalArguments::at_julian_date(t0 / 86_400.0 + UNIX_EPOCH_JD);
    Some((rel, epoch))
}

/// Internal phase convention `φ = −G` (radians), wrapped to `[0, 2π)`.
fn internal_phase(g_deg: f64) -> f64 {
    (-g_deg.to_radians()).rem_euclid(2.0 * PI)
}

/// Greenwich phase lag `G = −φ` in degrees, wrapped to `[0, 360)`.
fn greenwich_lag_deg(phase_rad: f64) -> f64 {
    (-phase_rad.to_degrees()).rem_euclid(360.0)
}

/// Wrap a degree difference to `(−180, 180]`.
fn wrap_deg_signed(d: f64) -> f64 {
    let mut x = d.rem_euclid(360.0);
    if x > 180.0 {
        x -= 360.0;
    }
    x
}

/// Build a named tidal constituent from `(amplitude, internal_phase)`.
fn named_constituent(name: &str, amplitude: f64, phase: f64) -> TidalConstituent {
    match name {
        "M2" => TidalConstituent::m2(amplitude, phase),
        "S2" => TidalConstituent::s2(amplitude, phase),
        "N2" => TidalConstituent::n2(amplitude, phase),
        "K1" => TidalConstituent::k1(amplitude, phase),
        "O1" => TidalConstituent::o1(amplitude, phase),
        "P1" => TidalConstituent::p1(amplitude, phase),
        other => panic!("unhandled constituent {other}"),
    }
}

/// Format Unix seconds as `YYYY-MM-DDTHH:MM:SSZ` (the reader accepts RFC3339).
fn format_utc(unix_s: i64) -> String {
    let days = unix_s.div_euclid(86_400);
    let secs = unix_s.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Inverse of Howard Hinnant's `days_from_civil`: days-since-Unix-epoch → `(y, m, d)`.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}
