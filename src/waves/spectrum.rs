//! The discrete wave spectrum: frequencies and directions, parametric spectra,
//! and the integrated quantities other modules need (significant wave height,
//! periods, direction, Stokes drift, radiation stress, bed orbital velocity).
//!
//! Frequencies are logarithmically spaced, `σ_i = σ_0 γ^i`, each the centre of the
//! bin `[σ_i γ^(−½), σ_i γ^(½)]` of width `Δσ_i = σ_i (γ^½ − γ^(−½))`, as in SWAN
//! and WAM: the resolution is relative, constant over the spectrum. Directions
//! cover the circle uniformly, `θ_j = j Δθ`, the direction the waves travel *to*,
//! counter-clockwise from mesh +x (the nautical "coming from" convention of
//! observations is converted at the boundaries).
//!
//! A spectrum is stored as variance density `E(σ, θ)` (m²/(rad/s)/rad) or action
//! density `N = E/σ`, one value per component `c = i n_θ + j` (frequency-major), so
//! that `m_n = Σ σⁿ E Δσ Δθ` and `H_s = 4 √m_0`.

use std::f64::consts::{PI, TAU};

use super::dispersion::group_velocity_ratio;

/// Frequencies and directions of a spectral wave model.
#[derive(Clone, Debug)]
pub struct SpectralGrid {
    /// Radian frequencies σ_i (rad/s), increasing
    pub sigma: Vec<f64>,
    /// Bin widths Δσ_i (rad/s)
    pub d_sigma: Vec<f64>,
    /// Directions θ_j (rad), travelling to, counter-clockwise from +x
    pub theta: Vec<f64>,
    /// Direction bin width Δθ (rad)
    pub d_theta: f64,
    /// cos θ_j and sin θ_j
    pub cos_theta: Vec<f64>,
    pub sin_theta: Vec<f64>,
}

/// Integrated parameters of a spectrum.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaveParameters {
    /// Variance m_0 (m²)
    pub m0: f64,
    /// Significant wave height H_s = 4 √m_0 (m)
    pub hs: f64,
    /// Mean periods T_m01 = 2π m_0/m_1 and T_m02 = 2π √(m_0/m_2) (s)
    pub tm01: f64,
    pub tm02: f64,
    /// Peak period (s), of the directionally integrated spectrum, refined by a
    /// parabola through the peak bin and its neighbours in log σ
    pub tp: f64,
    /// Mean direction (rad, travelling to), energy weighted
    pub direction: f64,
    /// Directional spread (rad), Kuik et al. (1988): √(2 (1 − r)), r the mean
    /// resultant length
    pub spread: f64,
}

impl SpectralGrid {
    /// `n_freq` frequencies from `f_min` to `f_max` (Hz), log-spaced, and `n_dir`
    /// directions over the circle.
    pub fn new(f_min: f64, f_max: f64, n_freq: usize, n_dir: usize) -> Self {
        assert!(
            f_min > 0.0 && f_max > f_min && n_freq >= 2 && n_dir >= 4,
            "a spectral grid needs 0 < f_min < f_max, 2+ frequencies and 4+ directions"
        );
        let gamma = (f_max / f_min).powf(1.0 / (n_freq - 1) as f64);
        let sigma: Vec<f64> = (0..n_freq)
            .map(|i| TAU * f_min * gamma.powi(i as i32))
            .collect();
        let width = gamma.sqrt() - 1.0 / gamma.sqrt();
        let d_sigma = sigma.iter().map(|s| s * width).collect();
        let d_theta = TAU / n_dir as f64;
        let theta: Vec<f64> = (0..n_dir).map(|j| j as f64 * d_theta).collect();
        Self {
            d_sigma,
            cos_theta: theta.iter().map(|t| t.cos()).collect(),
            sin_theta: theta.iter().map(|t| t.sin()).collect(),
            theta,
            d_theta,
            sigma,
        }
    }

    pub fn n_freq(&self) -> usize {
        self.sigma.len()
    }

    pub fn n_dir(&self) -> usize {
        self.theta.len()
    }

    /// Components (frequency × direction).
    pub fn n_components(&self) -> usize {
        self.n_freq() * self.n_dir()
    }

    /// Component of frequency `i` and direction `j`.
    #[inline]
    pub fn component(&self, i: usize, j: usize) -> usize {
        i * self.n_dir() + j
    }

    /// The ratio γ = σ_{i+1}/σ_i of neighbouring frequencies.
    pub fn frequency_ratio(&self) -> f64 {
        self.sigma[1] / self.sigma[0]
    }

    /// `m_n = Σ σⁿ E Δσ Δθ` of the variance density `e`.
    pub fn moment(&self, e: &[f64], n: i32) -> f64 {
        let nd = self.n_dir();
        (0..self.n_freq())
            .map(|i| {
                let row: f64 = e[i * nd..(i + 1) * nd].iter().sum();
                self.sigma[i].powi(n) * row * self.d_sigma[i]
            })
            .sum::<f64>()
            * self.d_theta
    }

    /// Integrated parameters of the variance density `e` (all zero for no energy).
    pub fn parameters(&self, e: &[f64]) -> WaveParameters {
        let nd = self.n_dir();
        let m0 = self.moment(e, 0);
        if m0 <= 0.0 {
            return WaveParameters::default();
        }
        let (m1, m2) = (self.moment(e, 1), self.moment(e, 2));
        // Directionally integrated E(σ), its peak refined in log σ
        let row = |i: usize| e[i * nd..(i + 1) * nd].iter().sum::<f64>() * self.d_theta;
        let peak = (0..self.n_freq())
            .max_by(|&a, &b| row(a).total_cmp(&row(b)))
            .unwrap_or(0);
        let mut log_sigma_p = self.sigma[peak].ln();
        if peak > 0 && peak + 1 < self.n_freq() {
            let (a, b, c) = (row(peak - 1), row(peak), row(peak + 1));
            let curvature = a - 2.0 * b + c;
            if curvature < 0.0 {
                let shift = 0.5 * (a - c) / curvature;
                log_sigma_p += shift.clamp(-0.5, 0.5) * self.frequency_ratio().ln();
            }
        }
        let (mut a1, mut b1) = (0.0, 0.0);
        for i in 0..self.n_freq() {
            for j in 0..nd {
                let w = e[i * nd + j] * self.d_sigma[i];
                a1 += w * self.cos_theta[j];
                b1 += w * self.sin_theta[j];
            }
        }
        let (a1, b1) = (a1 * self.d_theta / m0, b1 * self.d_theta / m0);
        let r = a1.hypot(b1).min(1.0);
        WaveParameters {
            m0,
            hs: 4.0 * m0.sqrt(),
            tm01: TAU * m0 / m1,
            tm02: TAU * (m0 / m2).sqrt(),
            tp: TAU / log_sigma_p.exp(),
            direction: b1.atan2(a1).rem_euclid(TAU),
            spread: (2.0 * (1.0 - r)).sqrt(),
        }
    }

    /// A JONSWAP spectrum (Hasselmann et al. 1973) of significant wave height `hs`
    /// (m), peak period `tp` (s) and peak enhancement `gamma` (3.3 for a young wind
    /// sea, 1 for Pierson–Moskowitz), spread over directions as `cos^m(θ − θ_m)`
    /// within ±90° of `direction` (rad, travelling to). Both shapes are normalised
    /// on this grid, so the variance density's `H_s` is `hs` exactly.
    pub fn jonswap(&self, hs: f64, tp: f64, gamma: f64, direction: f64, m: f64) -> Vec<f64> {
        let fp = 1.0 / tp;
        let shape: Vec<f64> = self
            .sigma
            .iter()
            .map(|&s| {
                let f = s / TAU;
                let width = if f <= fp { 0.07 } else { 0.09 };
                let r = (-(f - fp).powi(2) / (2.0 * width * width * fp * fp)).exp();
                f.powi(-5) * (-1.25 * (f / fp).powi(-4)).exp() * gamma.powf(r)
            })
            .collect();
        let spreading: Vec<f64> = self
            .theta
            .iter()
            .map(|&t| {
                let c = (t - direction).cos();
                if c > 0.0 { c.powf(m) } else { 0.0 }
            })
            .collect();
        let mut e: Vec<f64> = shape
            .iter()
            .flat_map(|&s| spreading.iter().map(move |&d| s * d))
            .collect();
        let m0 = self.moment(&e, 0);
        if m0 > 0.0 {
            let scale = (hs / 4.0).powi(2) / m0;
            e.iter_mut().for_each(|x| *x *= scale);
        }
        e
    }

    /// Stokes drift (m/s, mesh x and y) at height `z` (m, ≤ 0, below the mean
    /// surface) of the variance density `e` in depth `depth`, with `k[i]` the
    /// wavenumber of frequency i there:
    /// `u_s(z) = Σ σ k cosh(2k(z + d)) / sinh²(kd) E Δσ Δθ e_θ`
    /// (`2 σ k e^{2kz} E` in deep water; Kenyon 1969).
    pub fn stokes_drift(&self, e: &[f64], k: &[f64], depth: f64, z: f64) -> [f64; 2] {
        let nd = self.n_dir();
        let (mut ux, mut uy) = (0.0, 0.0);
        for i in 0..self.n_freq() {
            let (ki, s) = (k[i], self.sigma[i]);
            // cosh(2k(z+d)) / sinh²(kd), written to stay finite for large kd
            let decay = 2.0 * (2.0 * ki * z).exp() * (1.0 + (-4.0 * ki * (z + depth)).exp())
                / (1.0 - (-2.0 * ki * depth).exp()).powi(2);
            let w = s * ki * decay * self.d_sigma[i] * self.d_theta;
            for j in 0..nd {
                let ej = e[i * nd + j] * w;
                ux += ej * self.cos_theta[j];
                uy += ej * self.sin_theta[j];
            }
        }
        [ux, uy]
    }

    /// Radiation stress per ρg (m², Longuet-Higgins & Stewart 1964):
    /// `[S_xx, S_xy, S_yy] = Σ E (n (e_θ e_θ + I) − ½ I) Δσ Δθ`, with `k[i]` the
    /// wavenumbers in depth `depth`.
    pub fn radiation_stress(&self, e: &[f64], k: &[f64], depth: f64) -> [f64; 3] {
        let nd = self.n_dir();
        let mut s = [0.0; 3];
        for i in 0..self.n_freq() {
            let n = group_velocity_ratio(k[i] * depth);
            let w = self.d_sigma[i] * self.d_theta;
            for j in 0..nd {
                let ej = e[i * nd + j] * w;
                let (c, sn) = (self.cos_theta[j], self.sin_theta[j]);
                s[0] += ej * (n * (c * c + 1.0) - 0.5);
                s[1] += ej * n * c * sn;
                s[2] += ej * (n * (sn * sn + 1.0) - 0.5);
            }
        }
        s
    }

    /// Root-mean-square orbital velocity at the bed (m/s):
    /// `U_rms² = Σ σ² / sinh²(kd) E Δσ Δθ`.
    pub fn bed_orbital_velocity(&self, e: &[f64], k: &[f64], depth: f64) -> f64 {
        let nd = self.n_dir();
        let mut u2 = 0.0;
        for i in 0..self.n_freq() {
            let kd = k[i] * depth;
            if kd > 30.0 {
                continue;
            }
            let row: f64 = e[i * nd..(i + 1) * nd].iter().sum();
            u2 += (self.sigma[i] / kd.sinh()).powi(2) * row * self.d_sigma[i];
        }
        (u2 * self.d_theta).sqrt()
    }
}

/// Angle `a` (rad) folded into (−π, π].
pub fn wrap_angle(a: f64) -> f64 {
    let a = a.rem_euclid(TAU);
    if a > PI { a - TAU } else { a }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waves::dispersion::wavenumber;

    const G: f64 = 9.81;

    fn grid() -> SpectralGrid {
        SpectralGrid::new(0.04, 1.0, 36, 36)
    }

    #[test]
    fn a_jonswap_spectrum_has_its_parameters() {
        let grid = grid();
        let direction = 1.2;
        let e = grid.jonswap(2.5, 9.0, 3.3, direction, 4.0);
        let p = grid.parameters(&e);
        assert!((p.hs - 2.5).abs() < 1e-12, "H_s {}", p.hs);
        // The peak, refined between bins (γ ≈ 1.095 here)
        assert!((p.tp - 9.0).abs() < 0.15, "T_p {}", p.tp);
        // A peaked JONSWAP: T_m01 ≈ 0.8 T_p, T_m02 ≈ 0.75 T_p
        assert!((0.7..0.9).contains(&(p.tm01 / 9.0)), "T_m01 {}", p.tm01);
        assert!(p.tm02 < p.tm01);
        // 1.2 rad is not a bin centre: the sampled cos⁴ is a little lopsided
        assert!(
            wrap_angle(p.direction - direction).abs() < 2e-3,
            "θ {}",
            p.direction
        );
        // cos⁴: r = ∫cos⁵/∫cos⁴ = 128/(45π), spread √(2(1 − r)) = 0.435 rad
        let r = 128.0 / (45.0 * PI);
        assert!(
            (p.spread - (2.0 * (1.0 - r)).sqrt()).abs() < 0.01,
            "spread {}",
            p.spread
        );
    }

    #[test]
    fn stokes_drift_and_radiation_stress_of_a_deep_water_swell() {
        // One component: a 10 s wave along +x in deep water, variance 1/16 m²
        let grid = SpectralGrid::new(0.05, 0.5, 31, 36);
        let i = 10;
        let mut e = vec![0.0; grid.n_components()];
        let m0 = 1.0 / 16.0;
        e[grid.component(i, 0)] = m0 / (grid.d_sigma[i] * grid.d_theta);
        let depth = 1000.0;
        let k: Vec<f64> = grid
            .sigma
            .iter()
            .map(|&s| wavenumber(s, depth, G))
            .collect();
        // u_s(z) = 2 σ k m0 e^{2kz} (a² = 2 m0: σ k a² e^{2kz})
        let (s, ki) = (grid.sigma[i], k[i]);
        for z in [0.0, -5.0, -20.0] {
            let [ux, uy] = grid.stokes_drift(&e, &k, depth, z);
            let expected = 2.0 * s * ki * m0 * (2.0 * ki * z).exp();
            assert!(
                (ux - expected).abs() < 1e-12 * expected,
                "z {z}: {ux} against {expected}"
            );
            assert!(uy.abs() < 1e-15);
        }
        // S_xx = m0 (2n − ½) = ½ m0 deep, S_yy = m0 (n − ½) = 0, S_xy = 0
        let [sxx, sxy, syy] = grid.radiation_stress(&e, &k, depth);
        assert!((sxx / m0 - 0.5).abs() < 1e-12);
        assert!(syy.abs() < 1e-12 * m0 && sxy.abs() < 1e-15);
        // A spectrum's Stokes drift weighs σ³: its tail adds to the peak's share
        let swell = grid.jonswap(1.0, 10.0, 3.3, 0.0, 200.0);
        let [ux, _] = grid.stokes_drift(&swell, &k, depth, 0.0);
        let peak = (TAU / 10.0).powi(3) / (8.0 * G);
        assert!(ux > peak, "{ux} against the peak's {peak}");
    }

    #[test]
    fn shallow_water_radiation_stress_and_orbital_velocity() {
        // Long waves in 2 m: n → 1, S_xx = 1.5 m0 · ... = m0 (2·1 − ½), S_yy = ½ m0
        let grid = SpectralGrid::new(0.02, 0.2, 30, 36);
        let e = grid.jonswap(0.2, 40.0, 20.0, 0.0, 200.0);
        let depth = 2.0;
        let k: Vec<f64> = grid
            .sigma
            .iter()
            .map(|&s| wavenumber(s, depth, G))
            .collect();
        let m0 = grid.moment(&e, 0);
        let [sxx, _, syy] = grid.radiation_stress(&e, &k, depth);
        assert!((sxx / m0 - 1.5).abs() < 0.03, "S_xx/m0 {}", sxx / m0);
        assert!((syy / m0 - 0.5).abs() < 0.03, "S_yy/m0 {}", syy / m0);
        // Shallow: U_rms² = m0 σ² / (kd)² = m0 g / d
        let u = grid.bed_orbital_velocity(&e, &k, depth);
        assert!(
            (u / (m0 * G / depth).sqrt() - 1.0).abs() < 0.02,
            "U_rms {u}"
        );
    }
}
