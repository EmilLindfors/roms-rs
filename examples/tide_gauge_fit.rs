//! Harmonic reference constants of a tide-gauge record, optionally compared
//! with a tidal atlas (e.g. NorKyst-800 at the gauge).
//!
//! 1. Read a gauge file (`dg_rs::io::read_tide_gauge_file`, ISO times, e.g.
//!    from `scripts/kartverket_gauge.sh`), keep `[from, to)`.
//! 2. Fit reference constants (`fit_reference_constants`) of every constituent
//!    the record resolves (`resolvable_constituents`, Rayleigh factor
//!    `rayleigh`); unresolved P1, K2, N2 and Q1 are inferred at their
//!    equilibrium ratios.
//! 3. With `atlas=<file>`: the atlas point nearest the gauge and, per
//!    constituent, the amplitude ratio, phase difference and complex
//!    difference, plus their root sum of squares.
//!
//! ## Run
//!
//! ```bash
//! ./scripts/kartverket_gauge.sh Mausund 63.869331 8.665231 2024-07-01 2025-07-01
//! cargo run --release --example tide_gauge_fit -- data/tide_gauges/mausund_obs.txt \
//!     [from=2025-06-01] [to=2025-07-01] [rayleigh=1] [atlas=data/froya_station_tides.txt]
//! ```

use std::collections::HashMap;
use std::path::Path;

use dg_rs::analysis::{
    ConstituentComparison, Inference, ReferenceFit, fit_reference_constants,
    resolvable_constituents,
};
use dg_rs::boundary::TidalAtlas;
use dg_rs::io::read_tide_gauge_file;
use dg_rs::time::ModelClock;

/// Candidates in priority order: a short record keeps the first ones.
const CANDIDATES: [&str; 15] = [
    "M2", "S2", "K1", "O1", "N2", "Q1", "K2", "P1", "M4", "MS4", "MN4", "M6", "Mf", "Mm", "Ssa",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: HashMap<String, String> = HashMap::new();
    let mut file = None;
    for a in std::env::args().skip(1) {
        match a.split_once('=') {
            Some((k, v)) => {
                args.insert(k.into(), v.into());
            }
            None => file = Some(a),
        }
    }
    let file =
        file.ok_or("usage: tide_gauge_fit <gauge file> [from=] [to=] [rayleigh=] [atlas=]")?;
    let bound = |key: &str| -> Result<Option<f64>, String> {
        args.get(key)
            .map(|v| ModelClock::parse(v).map(|c| c.epoch_unix))
            .transpose()
    };
    let (from, to) = (bound("from")?, bound("to")?);
    let rayleigh: f64 = args.get("rayleigh").map_or(Ok(1.0), |v| v.parse())?;

    let gauge = read_tide_gauge_file(Path::new(&file))?;
    let (times, values): (Vec<f64>, Vec<f64>) = gauge
        .time_series
        .times()
        .into_iter()
        .zip(gauge.time_series.values())
        .filter(|&(t, v)| v.is_finite() && from.is_none_or(|f| t >= f) && to.is_none_or(|e| t < e))
        .unzip();
    let (Some(&t0), Some(&t1)) = (times.first(), times.last()) else {
        return Err("no samples in the window".into());
    };
    let name = gauge
        .station
        .as_ref()
        .map_or(file.clone(), |s| s.name.clone());
    println!(
        "{name}: {} samples, {} to {} ({:.1} days)",
        times.len(),
        ModelClock::new(t0).format(0.0),
        ModelClock::new(t1).format(0.0),
        (t1 - t0) / 86_400.0
    );

    let fit = fit_gauge(&times, &values, rayleigh)?;
    println!(
        "  Z0 {:+.3} m, R² {:.4}, residual RMS {:.3} m\n",
        fit.mean, fit.r_squared, fit.residual_rms
    );
    println!("  name    H (m)    G (°)");
    for c in &fit.constants {
        println!(
            "  {:4}  {:7.4}  {:7.2}{}",
            c.name,
            c.amplitude,
            c.lag_deg,
            if c.inferred { "  (inferred)" } else { "" }
        );
    }

    if let Some(path) = args.get("atlas") {
        let station = gauge
            .station
            .as_ref()
            .ok_or("the gauge file has no position")?;
        let atlas = TidalAtlas::read(Path::new(path))?;
        let (point, distance) = atlas
            .nearest(station.longitude, station.latitude)
            .ok_or("empty atlas")?;
        println!(
            "\nvs {path}: point ({:.4}, {:.4}), {:.0} m from the gauge, depth {:.0} m",
            point.lon, point.lat, distance, point.depth
        );
        println!("  name  H atlas  H gauge  ratio   ΔG (°)   |ΔZ| (m)");
        let mut rss = 0.0;
        for c in &point.constituents {
            let Some(g) = fit.get(c.name) else { continue };
            let cmp = ConstituentComparison::new(
                c.name,
                c.eta.0,
                c.eta.1.to_radians(),
                g.amplitude,
                g.lag_deg.to_radians(),
            );
            rss += cmp.complex_difference().powi(2);
            println!(
                "  {:4}  {:7.4}  {:7.4}  {:5.3}  {:+7.2}   {:.4}{}",
                c.name,
                c.eta.0,
                g.amplitude,
                cmp.amplitude_ratio,
                cmp.phase_error_degrees(),
                cmp.complex_difference(),
                if g.inferred { "  (gauge inferred)" } else { "" }
            );
        }
        println!("  root sum of squares of |ΔZ|: {:.4} m", rss.sqrt());
    }
    Ok(())
}

/// Reference constants of every constituent the record resolves, with the
/// equilibrium inferences for the rest.
fn fit_gauge(times: &[f64], values: &[f64], rayleigh: f64) -> Result<ReferenceFit, String> {
    let length = times.last().unwrap_or(&0.0) - times.first().unwrap_or(&0.0);
    let names = resolvable_constituents(&CANDIDATES, length, rayleigh);
    let inferred: Vec<Inference> = Inference::EQUILIBRIUM
        .into_iter()
        .filter(|i| !names.contains(&i.name) && names.contains(&i.from))
        .collect();
    fit_reference_constants(times, values, &names, &inferred)
}
