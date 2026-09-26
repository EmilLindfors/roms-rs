//! Least-squares fit of tidal *reference* constants `(H, G)` to a series on
//! an absolute (Unix) time axis, with inference of unresolved constituents.
//!
//! [`HarmonicAnalysis`](super::HarmonicAnalysis) fits apparent constants in
//! record time and leaves the astronomy to a separate step. Here the basis
//! functions carry it, so the unknowns are the reference constants directly:
//!
//! ```text
//! η(t) = Z₀ + Σⱼ fⱼ Hⱼ cos(ωⱼ (t − t₀) + V₀ⱼ + uⱼ − Gⱼ)
//! ```
//!
//! with `V₀` at the first sample `t₀` and the nodal `f`, `u` at the record
//! midpoint ([`ModelClock::nodal_correction`]).
//!
//! # Inference
//!
//! Two constituents are separable only if the record is longer than their
//! synodic period `2π/|ω₁ − ω₂|` (Rayleigh criterion). P1 and K1 need 182
//! days, as do K2 and S2, so a one- or two-month record cannot fit them
//! independently: their energy leaks into each other and the fit becomes
//! ill-conditioned. An [`Inference`] ties such a constituent to a resolved one
//! with a fixed amplitude ratio and phase-lag offset (the equilibrium-tide
//! values by default, as in Foreman's `t_tide`), so both enter the basis but
//! share one pair of unknowns. A request that is not separable and not
//! inferred is an error.

use faer::{Mat, linalg::solvers::Solve};

use crate::tides::constituent_period;
use crate::time::ModelClock;

/// A constituent inferred from a resolved one: `H = ratio · H_from`,
/// `G = G_from + lag_offset`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Inference {
    /// Inferred constituent.
    pub name: &'static str,
    /// Resolved constituent it follows.
    pub from: &'static str,
    /// Amplitude ratio `H / H_from`.
    pub amplitude_ratio: f64,
    /// Phase-lag offset `G − G_from` (degrees).
    pub lag_offset_deg: f64,
}

impl Inference {
    /// P1 from K1 with the equilibrium amplitude ratio 0.331 (Cartwright &
    /// Tayler 1971) and equal lag.
    pub const P1_FROM_K1: Self = Self {
        name: "P1",
        from: "K1",
        amplitude_ratio: 0.331,
        lag_offset_deg: 0.0,
    };

    /// K2 from S2 with the equilibrium amplitude ratio 0.272 and equal lag.
    pub const K2_FROM_S2: Self = Self {
        name: "K2",
        from: "S2",
        amplitude_ratio: 0.272,
        lag_offset_deg: 0.0,
    };
}

/// A fitted or inferred reference constant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferenceConstant {
    /// Constituent name.
    pub name: &'static str,
    /// Amplitude `H` (units of the series).
    pub amplitude: f64,
    /// Greenwich phase lag `G` (degrees, `[0, 360)`).
    pub lag_deg: f64,
    /// Whether the constant was inferred rather than fitted.
    pub inferred: bool,
}

/// Result of [`fit_reference_constants`].
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceFit {
    /// Mean `Z₀` over the record.
    pub mean: f64,
    /// Fitted constituents in the requested order, then the inferred ones.
    pub constants: Vec<ReferenceConstant>,
    /// Fraction of the variance explained.
    pub r_squared: f64,
    /// RMS of the residual.
    pub residual_rms: f64,
}

impl ReferenceFit {
    /// The constant of constituent `name`.
    pub fn get(&self, name: &str) -> Option<&ReferenceConstant> {
        self.constants.iter().find(|c| c.name == name)
    }
}

/// Fit reference constants of `names` (plus `inferred`) to the samples
/// `(times_unix[i], values[i])`; see the module docs.
///
/// Errors on an unknown constituent, an inference whose source is not fitted,
/// fewer samples than unknowns, or a pair of fitted constituents that the
/// record cannot separate.
pub fn fit_reference_constants(
    times_unix: &[f64],
    values: &[f64],
    names: &[&'static str],
    inferred: &[Inference],
) -> Result<ReferenceFit, String> {
    let n = times_unix.len();
    assert_eq!(n, values.len(), "times and values differ in length");
    let n_unknowns = 1 + 2 * names.len();
    if n < n_unknowns {
        return Err(format!("{n} samples for {n_unknowns} unknowns"));
    }
    let (t0, t1) = (times_unix[0], times_unix[n - 1]);
    let length = t1 - t0;
    let clock = ModelClock::new(t0);
    let t_mid = 0.5 * length;

    // (ω, f, V₀ + u) of a constituent on this record
    let astronomy = |name: &str| -> Result<(f64, f64, f64), String> {
        let period =
            constituent_period(name).ok_or_else(|| format!("unknown constituent {name}"))?;
        let n = clock
            .nodal_correction(name, t_mid)
            .ok_or_else(|| format!("no nodal correction for {name}"))?;
        Ok((
            2.0 * std::f64::consts::PI / period,
            n.f,
            n.phase_offset_rad(),
        ))
    };
    let fitted: Vec<(f64, f64, f64)> = names
        .iter()
        .map(|n| astronomy(n))
        .collect::<Result<_, _>>()?;

    // Rayleigh criterion between the fitted constituents
    for (i, a) in fitted.iter().enumerate() {
        for (j, b) in fitted.iter().enumerate().skip(i + 1) {
            let synodic = 2.0 * std::f64::consts::PI / (a.0 - b.0).abs();
            if length < synodic {
                return Err(format!(
                    "{} and {} need a {:.1}-day record to separate (have {:.1} days); \
                     infer one of them instead",
                    names[i],
                    names[j],
                    synodic / 86_400.0,
                    length / 86_400.0
                ));
            }
        }
    }

    // Inferred constituents, attached to the index of their source
    let attached: Vec<(usize, Inference, (f64, f64, f64))> = inferred
        .iter()
        .map(|inf| {
            let source = names.iter().position(|&n| n == inf.from).ok_or_else(|| {
                format!(
                    "{} is inferred from {}, which is not fitted",
                    inf.name, inf.from
                )
            })?;
            Ok((source, *inf, astronomy(inf.name)?))
        })
        .collect::<Result<_, String>>()?;

    // Design matrix: [1, (f cos(arg), f sin(arg)) per fitted constituent]
    let mut a = Mat::<f64>::zeros(n, n_unknowns);
    for (row, &t_unix) in times_unix.iter().enumerate() {
        let t = t_unix - t0;
        a[(row, 0)] = 1.0;
        for (j, &(omega, f, phase)) in fitted.iter().enumerate() {
            let arg = omega * t + phase;
            a[(row, 1 + 2 * j)] = f * arg.cos();
            a[(row, 2 + 2 * j)] = f * arg.sin();
        }
        for &(j, inf, (omega, f, phase)) in &attached {
            let arg = omega * t + phase - inf.lag_offset_deg.to_radians();
            a[(row, 1 + 2 * j)] += inf.amplitude_ratio * f * arg.cos();
            a[(row, 2 + 2 * j)] += inf.amplitude_ratio * f * arg.sin();
        }
    }
    let y = Mat::<f64>::from_fn(n, 1, |i, _| values[i]);
    let ata = a.transpose() * &a;
    let aty = a.transpose() * &y;
    let x = ata.as_ref().full_piv_lu().solve(&aty);

    // H cos(arg − G) = H cos G cos(arg) + H sin G sin(arg)
    let constant = |name, j: usize, ratio: f64, offset: f64, inferred| {
        let (c, s) = (x[(1 + 2 * j, 0)], x[(2 + 2 * j, 0)]);
        ReferenceConstant {
            name,
            amplitude: ratio * c.hypot(s),
            lag_deg: (s.atan2(c).to_degrees() + offset).rem_euclid(360.0),
            inferred,
        }
    };
    let mut constants: Vec<ReferenceConstant> = names
        .iter()
        .enumerate()
        .map(|(j, &name)| constant(name, j, 1.0, 0.0, false))
        .collect();
    constants.extend(
        attached.iter().map(|&(j, inf, _)| {
            constant(inf.name, j, inf.amplitude_ratio, inf.lag_offset_deg, true)
        }),
    );

    let fitted_values = &a * &x;
    let mean_value = values.iter().sum::<f64>() / n as f64;
    let (mut ss_res, mut ss_tot) = (0.0, 0.0);
    for (i, &v) in values.iter().enumerate() {
        ss_res += (v - fitted_values[(i, 0)]).powi(2);
        ss_tot += (v - mean_value).powi(2);
    }
    Ok(ReferenceFit {
        mean: x[(0, 0)],
        constants,
        r_squared: if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            1.0
        },
        residual_rms: (ss_res / n as f64).sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesize a series from reference constants on `clock` with the
    /// corrections the fit uses, `hours` hourly samples from `t0`.
    fn synthesize(t0: f64, hours: usize, mean: f64, constants: &[(&str, f64, f64)]) -> Vec<f64> {
        let clock = ModelClock::new(t0);
        let t_mid = 0.5 * (hours - 1) as f64 * 3600.0;
        (0..hours)
            .map(|i| {
                let t = i as f64 * 3600.0;
                mean + constants
                    .iter()
                    .map(|&(name, h, g)| {
                        let n = clock.nodal_correction(name, t_mid).unwrap();
                        let omega = 2.0 * std::f64::consts::PI / constituent_period(name).unwrap();
                        n.f * h * (omega * t + n.phase_offset_rad() - g.to_radians()).cos()
                    })
                    .sum::<f64>()
            })
            .collect()
    }

    fn lag_error(a: f64, b: f64) -> f64 {
        ((a - b + 180.0).rem_euclid(360.0) - 180.0).abs()
    }

    /// A 30-day record with P1 and K2 at their equilibrium ratios: the fit
    /// with inference recovers every constant, inferred ones included.
    #[test]
    fn recovers_reference_constants_with_inference() {
        let t0 = ModelClock::at_datetime(2025, 6, 1, 0, 0).epoch_unix;
        let truth = [
            ("M2", 0.80, 283.0),
            ("S2", 0.28, 318.0),
            ("N2", 0.17, 262.0),
            ("K1", 0.07, 190.0),
            ("O1", 0.05, 45.0),
            ("M4", 0.02, 110.0),
            ("P1", 0.331 * 0.07, 190.0),
            ("K2", 0.272 * 0.28, 318.0),
        ];
        let hours = 30 * 24;
        let values = synthesize(t0, hours, 0.12, &truth);
        let times: Vec<f64> = (0..hours).map(|i| t0 + i as f64 * 3600.0).collect();
        let fit = fit_reference_constants(
            &times,
            &values,
            &["M2", "S2", "N2", "K1", "O1", "M4"],
            &[Inference::P1_FROM_K1, Inference::K2_FROM_S2],
        )
        .unwrap();

        assert!((fit.mean - 0.12).abs() < 1e-9);
        assert!(fit.r_squared > 1.0 - 1e-12);
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
    }

    /// K1 and P1 cannot be fitted independently from a month.
    #[test]
    fn unresolvable_pair_is_an_error() {
        let t0 = 1.7e9;
        let times: Vec<f64> = (0..720).map(|i| t0 + i as f64 * 3600.0).collect();
        let values = vec![0.0; 720];
        let err = fit_reference_constants(&times, &values, &["K1", "P1"], &[]).unwrap_err();
        assert!(err.contains("K1 and P1"), "{err}");
        let err = fit_reference_constants(&times, &values, &["M2"], &[Inference::P1_FROM_K1])
            .unwrap_err();
        assert!(err.contains("not fitted"), "{err}");
    }
}
