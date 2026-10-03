//! Source terms of the wave action balance, per node: wind input, whitecapping,
//! quadruplet interactions, bottom friction and depth-induced breaking (SWAN's
//! "GEN3 KOMEN" set; triads are not included), and the diagnostic tail.
//!
//! Every term is written as `S(σ, θ) = A + B E` in variance density, with `A` the
//! linear (Phillips-type) wind input and `B` a rate (1/s) that depends on the
//! spectrum only through integrated quantities:
//!
//! - **Wind input** (Komen et al. 1984; Snyder et al. 1981):
//!   `B_in = max(0, ¼ (ρ_a/ρ_w) (28 u*/c cos(θ − θ_w) − 1)) σ`, with the friction
//!   velocity `u* = √C_D U_10` (Wu 1982 for `C_D`) and phase speed `c = σ/k`; and the
//!   linear growth of Cavaleri & Malanotte-Rizzoli (1981) that starts a sea from
//!   calm, `A = 1.5e-3/(2π g²) (u* max(0, cos(θ − θ_w)))⁴ exp(−(σ/σ*_PM)^(−4))`,
//!   `σ*_PM = 2π · 0.13 g / (28 u*)` (Tolman's filter keeps it off the low
//!   frequencies).
//! - **Whitecapping** (Komen et al. 1984, with δ = 1 as SWAN since Rogers et al.
//!   2003): `B_wc = −C_ds ((1 − δ) + δ k/k̃) (s̃²/s̃²_PM)² σ̃ k/k̃`, steepness
//!   `s̃ = k̃ √m_0`, `s̃²_PM = 3.02e-3`, `C_ds = 2.36e-5`, mean frequency
//!   `σ̃ = (⟨σ⁻¹⟩)⁻¹` and wavenumber `k̃ = (⟨k^(−½)⟩)⁻²` energy-weighted.
//! - **Bottom friction** (JONSWAP; Hasselmann et al. 1973, Bouws & Komen 1983):
//!   `B_bf = −C_b σ² / (g² sinh²(kd))`, `C_b = 0.038 m²/s³`.
//! - **Depth-induced breaking** (Battjes & Janssen 1978): the bore dissipation
//!   `D = −¼ α Q_b (σ̃/2π) H_max²`, `H_max = γ d`, `α = 1`, `γ = 0.73`, with the
//!   fraction of breaking waves from `(1 − Q_b)/ln Q_b = −(H_rms/H_max)²`, spread
//!   over the spectrum in proportion to `E`: `B_br = D / m_0` (`σ̃ = m_1/m_0`).
//! - **Quadruplets** by the DIA ([`super::nonlinear`]), scaled for finite depth by
//!   `R(0.75 k̃ d)`. They are not of the form `A + B E`: each component's gain
//!   joins `A` and its loss `B` (as the rate `S_nl/E`), both frozen over the step.
//! - **Diagnostic tail** (Komen et al. 1994, as WAM): above the cut-off
//!   `σ_c = max(2.5 σ̃, 4 σ_PM)` (`σ_PM` = 2π · 0.13 g/(28 u*), with wind) the
//!   spectrum is not prognostic but `E(σ_c, θ)(σ_c/σ)^p` after every step, with
//!   `p` = 4 (SWAN's for the Komen terms) by default; the DIA reads it above the
//!   grid too.
//!
//! [`SourceTerms::integrate`] advances `N = E/σ` over a step with the rates frozen
//! (an exponential integrator: exact for constant `A`, `B`, and unconditionally
//! stable for the dissipative terms, which are stiff in the surf zone). The
//! explicit gains are not: the DIA's grow as σ¹¹E³ at high frequencies, and
//! without a limit a 2 Hz grid blew up in a 14 h fetch run at 18 s steps.
//! Ris's (1997) limiter, as SWAN's, caps the growth of `N` over a step at a
//! fraction γ = 0.1 of the Phillips level, `ΔN ≤ γ α_PM/(2σk³c_g)` (`α_PM` =
//! 0.0081); losses are not limited (they are stable, and the surf zone needs them
//! whole).

use std::f64::consts::{PI, TAU};

use super::dispersion::group_velocity;
use super::nonlinear::{Quadruplets, shallow_water_factor};
use super::spectrum::SpectralGrid;

/// Air and water densities (kg/m³) for the wind input.
const RHO_AIR: f64 = 1.225;
const RHO_WATER: f64 = 1025.0;

/// SWAN's power of the diagnostic tail with the Komen terms.
pub const DEFAULT_TAIL_POWER: f64 = 4.0;

/// SWAN's fraction γ of the Phillips level in Ris's limiter.
pub const DEFAULT_LIMITER: f64 = 0.1;

/// Phillips' constant α_PM of the equilibrium range (Pierson & Moskowitz 1964).
const PHILLIPS: f64 = 0.0081;

/// Wind at a node: speed at 10 m (m/s) and the direction it blows *to* (rad,
/// counter-clockwise from mesh +x).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Wind {
    pub u10: f64,
    pub direction: f64,
}

impl Wind {
    /// Friction velocity u* = √C_D U_10 (m/s), C_D of Wu (1982).
    pub fn friction_velocity(&self) -> f64 {
        let cd = if self.u10 < 7.5 {
            1.2875e-3
        } else {
            (0.8 + 0.065 * self.u10) * 1e-3
        };
        cd.sqrt() * self.u10
    }
}

/// Which source terms act, and their coefficients.
#[derive(Clone, Debug)]
pub struct SourceTerms {
    pub g: f64,
    /// Wind input (Komen) with linear growth (Cavaleri & Malanotte-Rizzoli)
    pub wind: bool,
    /// Whitecapping (Komen): `C_ds`, `δ`, `s̃²_PM`
    pub whitecapping: Option<(f64, f64, f64)>,
    /// JONSWAP bottom friction `C_b` (m²/s³)
    pub bottom_friction: Option<f64>,
    /// Battjes–Janssen breaking: `α`, `γ`
    pub breaking: Option<(f64, f64)>,
    /// Quadruplet interactions (DIA)
    pub quadruplets: Option<Quadruplets>,
    /// Power `p` of the diagnostic `σ^(−p)` tail, or no tail
    pub tail: Option<f64>,
    /// Ris's limiter: the largest growth of `N` over a step, as a fraction γ of
    /// the Phillips level `α_PM/(2σk³c_g)`, or none
    pub limiter: Option<f64>,
}

impl SourceTerms {
    /// No sources: pure propagation.
    pub fn none(g: f64) -> Self {
        Self {
            g,
            wind: false,
            whitecapping: None,
            bottom_friction: None,
            breaking: None,
            quadruplets: None,
            tail: None,
            limiter: None,
        }
    }

    /// SWAN's defaults for these terms (wind, Komen whitecapping, the DIA, JONSWAP
    /// friction with `C_b = 0.038`, Battjes–Janssen with `α = 1`, `γ = 0.73`, an
    /// f⁻⁴ tail, Ris's limiter with γ = 0.1).
    pub fn swan_defaults(g: f64) -> Self {
        Self {
            g,
            wind: true,
            whitecapping: Some((2.36e-5, 1.0, 3.02e-3)),
            bottom_friction: Some(0.038),
            breaking: Some((1.0, 0.73)),
            quadruplets: Some(Quadruplets::default()),
            tail: Some(DEFAULT_TAIL_POWER),
            limiter: Some(DEFAULT_LIMITER),
        }
    }

    pub fn with_wind(mut self, on: bool) -> Self {
        self.wind = on;
        self
    }

    pub fn with_whitecapping(mut self, on: bool) -> Self {
        self.whitecapping = on.then_some((2.36e-5, 1.0, 3.02e-3));
        self
    }

    pub fn with_bottom_friction(mut self, cb: Option<f64>) -> Self {
        self.bottom_friction = cb;
        self
    }

    pub fn with_breaking(mut self, alpha_gamma: Option<(f64, f64)>) -> Self {
        self.breaking = alpha_gamma;
        self
    }

    /// Quadruplet interactions by the DIA with these coefficients, or none.
    pub fn with_quadruplets(mut self, quadruplets: Option<Quadruplets>) -> Self {
        self.quadruplets = quadruplets;
        self
    }

    /// The diagnostic `σ^(−p)` tail of power `p`, or none.
    pub fn with_tail(mut self, power: Option<f64>) -> Self {
        self.tail = power;
        self
    }

    /// Ris's growth limiter with fraction `gamma` of the Phillips level, or none.
    pub fn with_limiter(mut self, gamma: Option<f64>) -> Self {
        self.limiter = gamma;
        self
    }

    /// Whether any term acts.
    pub fn any(&self) -> bool {
        self.wind
            || self.whitecapping.is_some()
            || self.bottom_friction.is_some()
            || self.breaking.is_some()
            || self.quadruplets.is_some()
            || self.tail.is_some()
    }

    /// The linear input `a[c]` (m²/(rad/s)/rad/s) and rates `b[c]` (1/s) of the
    /// variance density `e` at a node of depth `depth` with wavenumbers `k[i]`,
    /// under `wind`.
    #[allow(clippy::too_many_arguments)]
    pub fn rates(
        &self,
        grid: &SpectralGrid,
        e: &[f64],
        k: &[f64],
        depth: f64,
        wind: Wind,
        a: &mut [f64],
        b: &mut [f64],
    ) {
        let (nf, nd) = (grid.n_freq(), grid.n_dir());
        a.fill(0.0);
        b.fill(0.0);
        let g = self.g;
        let means = Means::of(grid, e, k);

        // First, while `b` is free: the DIA's transfer, split into gains and losses
        if let (Some(quadruplets), Some(means)) = (self.quadruplets, means) {
            let scale = shallow_water_factor(0.75 * means.k * depth);
            quadruplets.source(grid, e, self.tail, g, scale, b);
            for ((a, b), &e) in a.iter_mut().zip(b.iter_mut()).zip(e) {
                let s = std::mem::take(b);
                if s >= 0.0 || e <= 0.0 {
                    *a = s;
                } else {
                    *b = s / e;
                }
            }
        }
        if self.wind && wind.u10 > 0.0 {
            let us = wind.friction_velocity();
            let sigma_pm = TAU * 0.13 * g / (28.0 * us);
            for (i, &ki) in k.iter().enumerate().take(nf) {
                let (s, c) = (grid.sigma[i], grid.sigma[i] / ki);
                let filter = (-(s / sigma_pm).powi(-4)).exp();
                for j in 0..nd {
                    let cos = (grid.theta[j] - wind.direction).cos();
                    let comp = i * nd + j;
                    b[comp] +=
                        (0.25 * RHO_AIR / RHO_WATER * (28.0 * us / c * cos - 1.0)).max(0.0) * s;
                    a[comp] += 1.5e-3 / (TAU * g * g) * (us * cos.max(0.0)).powi(4) * filter;
                }
            }
        }
        let Some(means) = means else {
            return;
        };
        let m0 = means.m0;
        if let Some((cds, delta, s_pm2)) = self.whitecapping {
            let (sigma_m, k_m) = (means.sigma, means.k);
            let steepness2 = k_m * k_m * m0;
            let gamma = cds * (steepness2 / s_pm2).powi(2) * sigma_m;
            for i in 0..nf {
                let r = k[i] / k_m;
                let rate = gamma * ((1.0 - delta) + delta * r) * r;
                b[i * nd..(i + 1) * nd].iter_mut().for_each(|x| *x -= rate);
            }
        }
        if let Some(cb) = self.bottom_friction {
            for i in 0..nf {
                let kd = k[i] * depth;
                if kd > 30.0 {
                    continue;
                }
                let rate = cb * (grid.sigma[i] / (g * kd.sinh())).powi(2);
                b[i * nd..(i + 1) * nd].iter_mut().for_each(|x| *x -= rate);
            }
        }
        if let Some((alpha, gamma)) = self.breaking {
            let h_max = gamma * depth;
            let q_b = breaking_fraction((8.0 * m0).sqrt() / h_max);
            if q_b > 0.0 {
                let sigma_mean = grid.moment(e, 1) / m0;
                let dissipation = 0.25 * alpha * q_b * sigma_mean / (2.0 * PI) * h_max * h_max;
                let rate = dissipation / m0;
                b.iter_mut().for_each(|x| *x -= rate);
            }
        }
    }

    /// Advance the action density `n` (one node's spectrum) over `dt` with the rates
    /// of its present state frozen: `N ← N e^{BΔt} + (A/σ)(e^{BΔt} − 1)/B`.
    /// `e`, `a`, `b` are scratch (one value per component).
    #[allow(clippy::too_many_arguments)]
    pub fn integrate(
        &self,
        grid: &SpectralGrid,
        n: &mut [f64],
        k: &[f64],
        depth: f64,
        wind: Wind,
        dt: f64,
        e: &mut [f64],
        a: &mut [f64],
        b: &mut [f64],
    ) {
        let nd = grid.n_dir();
        for (c, (e, n)) in e.iter_mut().zip(n.iter()).enumerate() {
            *e = grid.sigma[c / nd] * n;
        }
        self.rates(grid, e, k, depth, wind, a, b);
        for (i, (&sigma, &ki)) in grid.sigma.iter().zip(k).enumerate() {
            // Ris's limit on the growth over a step, or none
            let max_growth = self.limiter.map_or(f64::INFINITY, |gamma| {
                gamma * PHILLIPS / (2.0 * sigma * ki.powi(3) * group_velocity(sigma, ki, depth))
            });
            for c in i * nd..(i + 1) * nd {
                let x = b[c] * dt;
                let growth = x.exp();
                // (e^x − 1)/x → 1 as x → 0
                let phi = if x.abs() < 1e-8 {
                    1.0 + 0.5 * x
                } else {
                    (growth - 1.0) / x
                };
                let next = n[c] * growth + a[c] / sigma * dt * phi;
                n[c] = next.min(n[c] + max_growth).max(0.0);
            }
        }
        if let Some(power) = self.tail {
            self.apply_tail(grid, n, k, wind, power, e);
        }
    }

    /// Replace the action density above the cut-off `max(2.5 σ̃, 4 σ_PM)` by the
    /// `σ^(−power)` tail (in variance density) from the last bin below it. `e` is
    /// scratch.
    fn apply_tail(
        &self,
        grid: &SpectralGrid,
        n: &mut [f64],
        k: &[f64],
        wind: Wind,
        power: f64,
        e: &mut [f64],
    ) {
        let nd = grid.n_dir();
        for (c, (e, n)) in e.iter_mut().zip(n.iter()).enumerate() {
            *e = grid.sigma[c / nd] * n;
        }
        let Some(means) = Means::of(grid, e, k) else {
            return;
        };
        let mut cut = 2.5 * means.sigma;
        if self.wind && wind.u10 > 0.0 {
            cut = cut.max(4.0 * TAU * 0.13 * self.g / (28.0 * wind.friction_velocity()));
        }
        // The last prognostic bin: σ_ic ≤ σ_c
        let Some(ic) = grid.sigma.iter().rposition(|&s| s <= cut) else {
            return;
        };
        let sigma_c = grid.sigma[ic];
        for i in ic + 1..grid.n_freq() {
            let ratio = (sigma_c / grid.sigma[i]).powf(power) / grid.sigma[i];
            for j in 0..nd {
                n[i * nd + j] = e[ic * nd + j] * ratio;
            }
        }
    }
}

/// Energy-weighted means of a spectrum: `m_0`, `σ̃ = ⟨σ⁻¹⟩⁻¹`, `k̃ = ⟨k^(−½)⟩⁻²`
/// (Komen et al. 1984), or none for no energy.
#[derive(Clone, Copy, Debug)]
struct Means {
    m0: f64,
    sigma: f64,
    k: f64,
}

impl Means {
    fn of(grid: &SpectralGrid, e: &[f64], k: &[f64]) -> Option<Self> {
        let nd = grid.n_dir();
        let (mut m0, mut inv_sigma, mut inv_sqrt_k) = (0.0, 0.0, 0.0);
        for i in 0..grid.n_freq() {
            let row: f64 = e[i * nd..(i + 1) * nd].iter().sum::<f64>() * grid.d_sigma[i];
            m0 += row;
            inv_sigma += row / grid.sigma[i];
            inv_sqrt_k += row / k[i].sqrt();
        }
        if m0 <= 0.0 {
            return None;
        }
        Some(Self {
            m0: m0 * grid.d_theta,
            sigma: m0 / inv_sigma,
            k: (inv_sqrt_k / m0).powi(-2),
        })
    }
}

/// Fraction of breaking waves `Q_b` for `β = H_rms/H_max`, solving
/// `(1 − Q_b)/ln Q_b = −β²` (Battjes & Janssen 1978): 0 for small β, 1 for β ≥ 1.
pub fn breaking_fraction(beta: f64) -> f64 {
    if beta < 0.2 {
        return 0.0;
    }
    if beta >= 1.0 {
        return 1.0;
    }
    // Newton on f(Q) = 1 − Q + β² ln Q from SWAN's explicit first guess
    let b2 = beta * beta;
    let q0 = if beta > 0.5 {
        (2.0 * beta - 1.0).powi(2)
    } else {
        0.0
    };
    let ex = ((q0 - 1.0) / b2).exp();
    let mut q = (q0 - b2 * (q0 - ex) / (b2 - ex)).clamp(1e-300, 1.0);
    for _ in 0..50 {
        let f = 1.0 - q + b2 * q.ln();
        let df = -1.0 + b2 / q;
        let next = (q - f / df).clamp(q * 1e-3, 1.0);
        if (next - q).abs() <= 1e-14 * q {
            q = next;
            break;
        }
        q = next;
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waves::dispersion::wavenumber;

    const G: f64 = 9.81;

    #[test]
    fn the_breaking_fraction_solves_battjes_janssen() {
        for &beta in &[0.25, 0.4, 0.6, 0.8, 0.95] {
            let q = breaking_fraction(beta);
            assert!((0.0..=1.0).contains(&q));
            let residual = (1.0 - q) / q.ln() + beta * beta;
            assert!(
                residual.abs() < 1e-10,
                "β = {beta}: Q_b {q}, residual {residual:e}"
            );
        }
        assert_eq!(breaking_fraction(0.1), 0.0);
        assert_eq!(breaking_fraction(1.2), 1.0);
        // Monotonic
        assert!(breaking_fraction(0.7) > breaking_fraction(0.6));
    }

    #[test]
    fn bottom_friction_decays_each_component_at_its_rate() {
        let grid = SpectralGrid::new(0.05, 0.4, 20, 12);
        let depth = 8.0;
        let k: Vec<f64> = grid
            .sigma
            .iter()
            .map(|&s| wavenumber(s, depth, G))
            .collect();
        let sources = SourceTerms::none(G).with_bottom_friction(Some(0.038));
        let e0 = grid.jonswap(1.0, 8.0, 3.3, 0.3, 2.0);
        let mut n: Vec<f64> = (0..e0.len())
            .map(|c| e0[c] / grid.sigma[c / grid.n_dir()])
            .collect();
        let m = n.len();
        let (mut e, mut a, mut b) = (vec![0.0; m], vec![0.0; m], vec![0.0; m]);
        let (dt, steps) = (30.0, 40);
        for _ in 0..steps {
            sources.integrate(
                &grid,
                &mut n,
                &k,
                depth,
                Wind::default(),
                dt,
                &mut e,
                &mut a,
                &mut b,
            );
        }
        // Exactly exponential: the rate depends on the component only
        let t = dt * steps as f64;
        for c in 0..m {
            let i = c / grid.n_dir();
            let rate = 0.038 * (grid.sigma[i] / (G * (k[i] * depth).sinh())).powi(2);
            let expected = e0[c] / grid.sigma[i] * (-rate * t).exp();
            assert!((n[c] - expected).abs() <= 1e-12 * e0[c].max(1e-300) / grid.sigma[i] + 1e-300);
        }
    }

    #[test]
    fn wind_grows_a_sea_from_calm_downwind_only() {
        let grid = SpectralGrid::new(0.05, 1.0, 30, 24);
        let depth = 100.0;
        let k: Vec<f64> = grid
            .sigma
            .iter()
            .map(|&s| wavenumber(s, depth, G))
            .collect();
        let sources = SourceTerms::none(G).with_wind(true);
        let wind = Wind {
            u10: 15.0,
            direction: 0.0,
        };
        let m = grid.n_components();
        let mut n = vec![0.0; m];
        let (mut e, mut a, mut b) = (vec![0.0; m], vec![0.0; m], vec![0.0; m]);
        sources.integrate(&grid, &mut n, &k, depth, wind, 60.0, &mut e, &mut a, &mut b);
        // From calm only the linear term acts: A Δt / σ
        let us = wind.friction_velocity();
        let sigma_pm = TAU * 0.13 * G / (28.0 * us);
        for i in 0..grid.n_freq() {
            let s = grid.sigma[i];
            let a0 = 1.5e-3 / (TAU * G * G) * us.powi(4) * (-(s / sigma_pm).powi(-4)).exp();
            // The exponential input acts from the start too: (e^{bΔt} − 1)/b
            let c = s / k[i];
            let b0 = (0.25 * RHO_AIR / RHO_WATER * (28.0 * us / c - 1.0)).max(0.0) * s;
            let phi = if b0 > 0.0 {
                ((b0 * 60.0).exp() - 1.0) / b0
            } else {
                60.0
            };
            let downwind = n[grid.component(i, 0)];
            let expected = a0 / s * phi;
            assert!(
                (downwind - expected).abs() <= 1e-10 * expected + 1e-300,
                "{downwind} against {expected}"
            );
            // Nothing against the wind
            assert_eq!(n[grid.component(i, grid.n_dir() / 2)], 0.0);
        }
    }

    #[test]
    fn breaking_bounds_the_wave_height_in_shallow_water() {
        // A 2 m sea in 1.5 m of water loses its excess within minutes
        let grid = SpectralGrid::new(0.05, 0.5, 20, 12);
        let depth = 1.5;
        let k: Vec<f64> = grid
            .sigma
            .iter()
            .map(|&s| wavenumber(s, depth, G))
            .collect();
        let sources = SourceTerms::none(G).with_breaking(Some((1.0, 0.73)));
        let e0 = grid.jonswap(2.0, 8.0, 3.3, 0.0, 2.0);
        let mut n: Vec<f64> = (0..e0.len())
            .map(|c| e0[c] / grid.sigma[c / grid.n_dir()])
            .collect();
        let m = n.len();
        let (mut e, mut a, mut b) = (vec![0.0; m], vec![0.0; m], vec![0.0; m]);
        let mut hs = Vec::new();
        for _ in 0..600 {
            sources.integrate(
                &grid,
                &mut n,
                &k,
                depth,
                Wind::default(),
                1.0,
                &mut e,
                &mut a,
                &mut b,
            );
            let e: Vec<f64> = (0..m)
                .map(|c| n[c] * grid.sigma[c / grid.n_dir()])
                .collect();
            hs.push(grid.parameters(&e).hs);
        }
        // Monotonic decay towards H_s ≈ γ d √2 ... below the breaking limit
        assert!(hs.windows(2).all(|w| w[1] <= w[0]));
        let last = *hs.last().unwrap();
        assert!(last < 0.73 * depth * 2f64.sqrt(), "H_s {last}");
        assert!(
            last > 0.3 * depth,
            "H_s {last}: breaking only removes the excess"
        );
    }
}
