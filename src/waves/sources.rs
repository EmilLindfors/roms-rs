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
//!   Its dissipation saturates at `Q_b = 1`, `H_rms = H_max`, where it is
//!   `¼ α (σ̃/2π) H_max²` whatever the energy, so in a few decimetres of
//!   water it cannot remove a sea that the propagation brings in (2 m of H_s
//!   in 0.11 m loses ≈ 6e-4 of its energy per second). The depth limit
//!   ([`SourceTerms::with_depth_limit`]) scales the spectrum down to that
//!   state after every step, `m_0 ≤ (γ d)²/8`: no more than every wave
//!   breaking, the edge of Battjes and Janssen's model.
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
//! whole). A cap per step makes the growth depend on the step, so
//! [`GrowthLimiter::Rate`] caps it per unit time instead, as WAM since cycle 4
//! (Hersbach & Janssen 1999; ECMWF IFS Part VII §5.2):
//! `ΔF ≤ C g ũ* f⁻⁴ f_c Δt` in `F(f, θ)`, `C = 5·10⁻⁷` (ecWAM's), `f_c` the larger
//! of the mean frequencies `σ̃/2π` of the wind sea and of the whole spectrum
//! (ecWAM's `implsch.F90`: the wind sea is the components the wind feeds), and
//! `ũ* = max(u*, g f*_PM/f_c)` with `f*_PM = 5.6·10⁻³` (without wind the floor
//! of a fully developed sea at `f_c`). Under swell the whole spectrum's mean
//! frequency is the swell's, and would cap the young wind sea's growth by
//! several times too much. It suits [`SourceIntegration::Implicit`], whose
//! steps are long. ecWAM caps the losses by the same limit
//! ([`GrowthLimiter::RateBothSigns`]), which keeps energy where a sink would
//! remove more than the limit in a step; depth-induced breaking is left out
//! of that cap ([`super::WaveModel2D`] gives it its own pass), and here the
//! cap on bottom friction keeps waves on floored land nodes (see the
//! variant).

use std::f64::consts::{PI, TAU};

use super::dispersion::group_velocity;
#[cfg(feature = "simd")]
use super::nonlinear::LANES;
use super::nonlinear::{Quadruplets, shallow_water_factor};
use super::spectrum::SpectralGrid;

/// Air and water densities (kg/m³) for the wind input.
const RHO_AIR: f64 = 1.225;
const RHO_WATER: f64 = 1025.0;

/// SWAN's power of the diagnostic tail with the Komen terms.
pub const DEFAULT_TAIL_POWER: f64 = 4.0;

/// SWAN's fraction γ of the Phillips level in Ris's limiter.
pub const DEFAULT_LIMITER: f64 = 0.1;

/// The coefficient `C` of [`GrowthLimiter::Rate`]: ecWAM's present `COEF4`
/// (Hersbach & Janssen 1999 and ECMWF's IFS Cy33r1 documentation have
/// 3·10⁻⁷). With the Komen terms it keeps the fetch-limited growth within
/// 8 % of Ris's limiter's (`examples/wave_growth.rs`), 3·10⁻⁷ within 22 %.
pub const DEFAULT_RATE_LIMITER: f64 = 5.0e-7;

/// The dimensionless Pierson–Moskowitz peak frequency `f_PM u*/g` of the floor
/// of `ũ*` in [`GrowthLimiter::Rate`].
const PM_PEAK: f64 = 5.6e-3;

/// How the sources' growth of the action density over a step is limited
/// (losses only by [`Self::RateBothSigns`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GrowthLimiter {
    /// Ris's (SWAN's): at most a fraction γ of the Phillips level per step,
    /// `ΔN ≤ γ α_PM/(2σk³c_g)`, whatever the step
    PerStep(f64),
    /// Hersbach & Janssen's (WAM's): at most `C g ũ* f⁻⁴ f_c Δt` of `F(f, θ)`,
    /// proportional to the step (see the module docs)
    Rate(f64),
    /// [`Self::Rate`]'s limit on the losses too, `|ΔF| ≤ C g ũ* f⁻⁴ f_c Δt`, as
    /// ecWAM's `implsch.F90` (`SIGN(MIN(|ΔF|, limit), ΔF)`). Depth-induced
    /// breaking must stay outside it: [`super::WaveModel2D`] steps breaking on
    /// a pass of its own under this limiter, but [`SourceTerms::integrate`]
    /// caps every term it is given.
    ///
    /// Not recommended here (measured 2026-10-09, the 24 h Frøya storm at
    /// a 56 s step): against [`Self::Rate`] it moves H_s at MET's points by
    /// ≤ 2 cm, but it caps the bottom friction, which is what removes the
    /// waves on the depth floor of land nodes (0.1 m: a 2 m sea keeps
    /// 1.8 m over a step, against 0.17 m), so seas of up to 19 m pile up
    /// against cliff coasts. Also less step-independent: at 900 s steps
    /// a duration-limited sea is 8 % low at 96 h (60 s: 3.08e-3).
    RateBothSigns(f64),
}

impl GrowthLimiter {
    /// Whether the limit caps the losses as well as the growth.
    pub fn caps_losses(&self) -> bool {
        matches!(self, Self::RateBothSigns(_))
    }
}

/// Phillips' constant α_PM of the equilibrium range (Pierson & Moskowitz 1964).
const PHILLIPS: f64 = 0.0081;

/// Directions whose wind-input cosines [`SourceTerms::rates`] tabulates on the
/// stack; it computes those of any further directions at every frequency.
const MAX_DIRECTIONS: usize = 72;

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

/// What the sinks take from a node's spectrum per second
/// ([`SourceTerms::dissipation`]).
///
/// The variances (m²/s) times ρg are the energy each sink gives the water
/// (W/m²): whitecapping and breaking at the surface, friction at the bed.
/// `force` is the momentum the waves lose with them, per ρ (m²/s²):
///
/// ```text
/// F = g ∫∫ (k/σ) (cos θ, sin θ) D dσ dθ,
/// ```
///
/// each component's momentum `E k/σ` per unit energy. It is the force of
/// Dingemans et al. (1987) on the mean flow, SWAN's alternative to the
/// radiation stress's divergence: Longuet-Higgins's (1970) longshore force on
/// a uniform coast, and zero where the waves only shoal or refract. It drives
/// the currents and sets the water up where the waves break, but does not
/// set it down where they shoal: it leaves out the part of `−∇·S` from
/// shoaling and refraction without loss, which is nearly a gradient.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Dissipation {
    /// The variance (m²/s) whitecapping removes per second
    pub whitecapping: f64,
    /// The variance (m²/s) bottom friction removes per second
    pub bottom_friction: f64,
    /// The variance (m²/s) depth-induced breaking removes per second
    pub breaking: f64,
    /// The momentum all three remove per second, per ρ (m²/s²)
    pub force: [f64; 2],
}

impl Dissipation {
    /// The variance (m²/s) all the sinks remove per second.
    pub fn total(&self) -> f64 {
        self.whitecapping + self.bottom_friction + self.breaking
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
    /// With breaking, the spectrum scaled down to `H_rms ≤ γ d` after every
    /// step ([`Self::with_depth_limit`])
    pub depth_limit: bool,
    /// Quadruplet interactions (DIA)
    pub quadruplets: Option<Quadruplets>,
    /// Power `p` of the diagnostic `σ^(−p)` tail, or no tail
    pub tail: Option<f64>,
    /// The limit on the growth of `N` over a step, or none
    pub limiter: Option<GrowthLimiter>,
    /// How [`Self::integrate`] advances the action over a step
    pub integration: SourceIntegration,
}

/// How [`SourceTerms::integrate`] advances the action over a step: with the
/// exponential update `N ← N e^{BΔt} + (A/σ)(e^{BΔt} − 1)/B` (exact for
/// constant `A`, `B`), with the rates `A`, `B` of
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SourceIntegration {
    /// the state at the start of the step: first order in the step, since
    /// the rates depend on the spectrum
    #[default]
    FrozenRates,
    /// the state half a step on, predicted with the rates of the start (the
    /// exponential midpoint rule): second order, for twice the rates' cost
    Midpoint,
    /// the start, with the DIA implicit in each component's own energy: its
    /// transfer linearised about the start by its diagonal derivative `Λ`
    /// ([`Quadruplets::source_and_diagonal`]), `Λ` joining the rate `B`
    /// where it is negative (WAM's scheme, ECMWF IFS Part VII §5.2,
    /// Hersbach & Janssen 1999, with the exponential update in place of
    /// implicit Euler). The high-frequency transfer, which the explicit
    /// gains overshoot at long steps, relaxes to its equilibrium instead.
    Implicit,
}

/// The nodes of [`SourceTerms::integrate_at`]: node `l` has the
/// wavenumbers `k[l n_σ..(l + 1) n_σ]`, the depth `depths[l]` and the wind
/// `winds[l]`.
#[derive(Clone, Copy)]
pub(crate) struct Nodes<'a> {
    pub k: &'a [f64],
    pub depths: &'a [f64],
    pub winds: &'a [Wind],
}

impl Nodes<'_> {
    fn node(&self, l: usize, n_freq: usize) -> (&[f64], f64, Wind) {
        (
            &self.k[l * n_freq..(l + 1) * n_freq],
            self.depths[l],
            self.winds[l],
        )
    }
}

/// Whether [`SourceTerms::rates_at`] runs the DIA at its nodes at once, with
/// the storage of its interleaved spectra and transfers.
enum Lanes<'a> {
    None,
    /// The interleaved spectra, transfers and diagonals
    #[cfg(feature = "simd")]
    Dia(&'a mut Vec<f64>, &'a mut Vec<f64>, &'a mut Vec<f64>),
    #[cfg(not(feature = "simd"))]
    #[allow(dead_code)]
    Never(std::marker::PhantomData<&'a ()>),
}

thread_local! {
    /// The midpoint's predictor in [`SourceTerms::integrate`]
    static PREDICTOR: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
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
            depth_limit: false,
            quadruplets: None,
            tail: None,
            limiter: None,
            integration: SourceIntegration::FrozenRates,
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
            depth_limit: false,
            quadruplets: Some(Quadruplets::default()),
            tail: Some(DEFAULT_TAIL_POWER),
            limiter: Some(GrowthLimiter::PerStep(DEFAULT_LIMITER)),
            integration: SourceIntegration::FrozenRates,
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

    /// Scale each node's spectrum down after every step where its root-mean-
    /// square height exceeds breaking's `H_max = γ d`, to `m_0 = (γ d)²/8`
    /// (its shape kept): off by default, and only with breaking. Battjes–
    /// Janssen's dissipation stops growing with the energy at `H_rms = H_max`
    /// (see the module docs), so without the limit a few decimetres of water
    /// can hold metres of H_s where the propagation brings them, such as a
    /// node near the shore of a coarse element whose other nodes are deep.
    /// The energy it removes is not in [`Self::dissipation`].
    pub fn with_depth_limit(mut self, on: bool) -> Self {
        self.depth_limit = on;
        self
    }

    /// [`Self::with_depth_limit`]'s scaling of the action density `n` at a
    /// node of depth `depth`, if on and with breaking.
    fn limit_to_depth(&self, grid: &SpectralGrid, n: &mut [f64], depth: f64) {
        let Some((_, gamma)) = self.breaking.filter(|_| self.depth_limit) else {
            return;
        };
        let nd = grid.n_dir();
        let m0 = n
            .chunks_exact(nd)
            .enumerate()
            .map(|(i, row)| row.iter().sum::<f64>() * grid.sigma[i] * grid.d_sigma[i])
            .sum::<f64>()
            * grid.d_theta;
        let h_max = gamma * depth;
        let m0_max = 0.125 * h_max * h_max;
        if m0 > m0_max {
            let scale = m0_max / m0;
            n.iter_mut().for_each(|x| *x *= scale);
        }
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

    /// How [`Self::integrate`] advances the action over a step.
    pub fn with_integration(mut self, integration: SourceIntegration) -> Self {
        self.integration = integration;
        self
    }

    /// The limit on the growth over a step, or none.
    pub fn with_limiter(mut self, limiter: Option<GrowthLimiter>) -> Self {
        self.limiter = limiter;
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
        a.fill(0.0);
        b.fill(0.0);
        let means = Means::of(grid, e, k);
        // First, while `b` is free: the DIA's transfer, split into gains and losses
        if let (Some(quadruplets), Some(means)) = (self.quadruplets, means) {
            let scale = quadruplet_scale(means, depth);
            if self.integration == SourceIntegration::Implicit {
                // The transfer into `b`, its diagonal into `a`
                quadruplets.source_and_diagonal(grid, e, self.tail, self.g, scale, b, a);
                linearise_transfer(e, a, b);
            } else {
                quadruplets.source(grid, e, self.tail, self.g, scale, b);
                split_transfer(e, a, b);
            }
        }
        self.add_local_rates(grid, e, k, depth, wind, means, a, b);
    }

    /// The terms of [`Self::rates`] besides the DIA, added to `a` and `b`;
    /// `means` are `e`'s.
    fn add_local_rates(
        &self,
        grid: &SpectralGrid,
        e: &[f64],
        k: &[f64],
        depth: f64,
        wind: Wind,
        means: Option<Means>,
        a: &mut [f64],
        b: &mut [f64],
    ) {
        let (nf, nd) = (grid.n_freq(), grid.n_dir());
        let g = self.g;
        if self.wind && wind.u10 > 0.0 {
            let us = wind.friction_velocity();
            let sigma_pm = TAU * 0.13 * g / (28.0 * us);
            // cos(θ − θ_w) once per direction (on the stack up to
            // `MAX_DIRECTIONS`), not at every frequency
            let mut table = [0.0; MAX_DIRECTIONS];
            let cosine = |j: usize| (grid.theta[j] - wind.direction).cos();
            for (j, cos) in table.iter_mut().enumerate().take(nd) {
                *cos = cosine(j);
            }
            for (i, &ki) in k.iter().enumerate().take(nf) {
                let (s, c) = (grid.sigma[i], grid.sigma[i] / ki);
                let filter = (-(s / sigma_pm).powi(-4)).exp();
                for j in 0..nd {
                    let cos = table.get(j).copied().unwrap_or_else(|| cosine(j));
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

    /// What the sinks (whitecapping, bottom friction, depth-induced breaking)
    /// take per second from the variance density `e` at a node of depth
    /// `depth` with wavenumbers `k[i]`, over a step `dt` (s) with the rates
    /// frozen, as [`Self::integrate`] steps them: each component loses
    /// `E (1 − e^{BΔt})` to the sinks' total rate `B`, shared among them in
    /// proportion to their rates, and per second
    ///
    /// ```text
    /// D_term = (B_term/B) E (1 − e^{BΔt})/Δt,
    /// ```
    ///
    /// integrated over the spectrum, with the momentum they take with it
    /// (see [`Dissipation`]). `dt = 0` gives the instantaneous `D = −B E`.
    /// Over a step no sink removes more than the energy there is (`E/Δt`):
    /// a stiff sink, such as whitecapping on a steep sea in a few
    /// decimetres of water, takes what the step brings it, not its rate
    /// times the energy the propagation has just brought.
    ///
    /// The rates are those of [`Self::rates`]; the wind input and the DIA
    /// (which only moves energy) are left out. `rates`, `term` are scratch
    /// (one value per component).
    pub fn dissipation(
        &self,
        grid: &SpectralGrid,
        e: &[f64],
        k: &[f64],
        depth: f64,
        dt: f64,
        rates: &mut [f64],
        term: &mut [f64],
    ) -> Dissipation {
        assert!(dt >= 0.0, "a step of {dt} s");
        let mut out = Dissipation::default();
        let Some(means) = Means::of(grid, e, k) else {
            return out;
        };
        let nd = grid.n_dir();
        let none = Self::none(self.g);
        let sinks = [
            Self {
                whitecapping: self.whitecapping,
                ..none.clone()
            },
            Self {
                bottom_friction: self.bottom_friction,
                ..none.clone()
            },
            Self {
                breaking: self.breaking,
                ..none
            },
        ];
        // Without wind no term has a linear input `A`
        let calm = Wind::default();
        // The sinks' total rate, and each component's share of a step's loss
        // per unit rate: `(1 − e^{BΔt})/(−BΔt)` (1 for an instant)
        rates.fill(0.0);
        for sink in &sinks {
            sink.add_local_rates(grid, e, k, depth, calm, Some(means), &mut [], rates);
        }
        for r in rates.iter_mut() {
            let x = *r * dt;
            *r = if x < -1e-12 { -(x.exp_m1()) / -x } else { 1.0 };
        }
        for (index, sink) in sinks.iter().enumerate() {
            term.fill(0.0);
            sink.add_local_rates(grid, e, k, depth, calm, Some(means), &mut [], term);
            let mut variance = 0.0;
            for (i, &ki) in k.iter().enumerate().take(grid.n_freq()) {
                let w = grid.d_sigma[i] * grid.d_theta;
                let momentum = self.g * ki / grid.sigma[i];
                for j in 0..nd {
                    let c = i * nd + j;
                    let lost = -term[c] * rates[c] * e[c] * w;
                    variance += lost;
                    out.force[0] += momentum * grid.cos_theta[j] * lost;
                    out.force[1] += momentum * grid.sin_theta[j] * lost;
                }
            }
            match index {
                0 => out.whitecapping = variance,
                1 => out.bottom_friction = variance,
                _ => out.breaking = variance,
            }
        }
        out
    }

    /// Advance the action density `n` (one node's spectrum) over `dt` by
    /// [`Self::integration`]. `e`, `a`, `b` are scratch (one value per
    /// component).
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
        PREDICTOR.with_borrow_mut(|predictor| {
            let nodes = Nodes {
                k,
                depths: &[depth],
                winds: &[wind],
            };
            self.integrate_nodes(grid, n, nodes, dt, e, a, b, predictor, Lanes::None);
        });
    }

    /// [`Self::integrate`] at the nodes (at most [`LANES`] with `lanes`)
    /// whose spectra lie one after the other in `spectra`, with the DIA's
    /// transfer at all of them at once if `lanes` ([`Quadruplets::source_lanes`],
    /// bit for bit as at each alone).
    pub(crate) fn integrate_at(
        &self,
        grid: &SpectralGrid,
        spectra: &mut [f64],
        nodes: Nodes,
        dt: f64,
        scratch: &mut SourceScratch,
        lanes: bool,
    ) {
        let SourceScratch {
            e,
            a,
            b,
            predictor,
            #[cfg(feature = "simd")]
            e_lanes,
            #[cfg(feature = "simd")]
            s_lanes,
            #[cfg(feature = "simd")]
            d_lanes,
        } = scratch;
        a.resize(spectra.len(), 0.0);
        b.resize(spectra.len(), 0.0);
        #[cfg(feature = "simd")]
        let lanes = if lanes {
            Lanes::Dia(e_lanes, s_lanes, d_lanes)
        } else {
            Lanes::None
        };
        #[cfg(not(feature = "simd"))]
        let lanes = {
            let _ = lanes;
            Lanes::None
        };
        self.integrate_nodes(grid, spectra, nodes, dt, e, a, b, predictor, lanes);
    }

    /// The step of [`Self::integrate`] at the nodes of `spectra`; `a`, `b`
    /// hold their rates (one after the other, as `spectra`), `predictor` the
    /// midpoint's half step.
    fn integrate_nodes(
        &self,
        grid: &SpectralGrid,
        spectra: &mut [f64],
        nodes: Nodes,
        dt: f64,
        e: &mut [f64],
        a: &mut [f64],
        b: &mut [f64],
        predictor: &mut Vec<f64>,
        mut lanes: Lanes,
    ) {
        let nc = grid.n_components();
        let advance = |spectra: &mut [f64], dt: f64, e: &mut [f64], a: &[f64], b: &[f64]| {
            for (l, n) in spectra.chunks_exact_mut(nc).enumerate() {
                let (k, depth, wind) = nodes.node(l, grid.n_freq());
                let rows = l * nc..(l + 1) * nc;
                self.advance(grid, n, k, depth, wind, dt, e, &a[rows.clone()], &b[rows]);
            }
        };
        match self.integration {
            SourceIntegration::FrozenRates | SourceIntegration::Implicit => {
                self.rates_at(grid, spectra, nodes, e, a, b, &mut lanes);
                advance(spectra, dt, e, a, b);
            }
            SourceIntegration::Midpoint => {
                predictor.clear();
                predictor.extend_from_slice(spectra);
                self.rates_at(grid, spectra, nodes, e, a, b, &mut lanes);
                advance(predictor, 0.5 * dt, e, a, b);
                self.rates_at(grid, predictor, nodes, e, a, b, &mut lanes);
                advance(spectra, dt, e, a, b);
            }
        }
    }

    /// [`Self::rates`] of the action densities `spectra` (one node's after
    /// another) into `a`, `b` (the same layout); `e` is scratch.
    fn rates_at(
        &self,
        grid: &SpectralGrid,
        spectra: &[f64],
        nodes: Nodes,
        e: &mut [f64],
        a: &mut [f64],
        b: &mut [f64],
        lanes: &mut Lanes,
    ) {
        let (nf, nc) = (grid.n_freq(), grid.n_components());
        #[cfg(feature = "simd")]
        if let (Lanes::Dia(e_lanes, s_lanes, d_lanes), Some(quadruplets)) =
            (lanes, self.quadruplets)
        {
            let implicit = self.integration == SourceIntegration::Implicit;
            // The nodes' variance densities interleaved, empty lanes beyond them
            e_lanes.clear();
            e_lanes.resize(nc * LANES, 0.0);
            s_lanes.resize(nc * LANES, 0.0);
            let mut means = [None; LANES];
            let mut scales = [1.0; LANES];
            for (l, n) in spectra.chunks_exact(nc).enumerate() {
                let (k, depth, _) = nodes.node(l, nf);
                variance(grid, n, e);
                means[l] = Means::of(grid, e, k);
                if let Some(means) = means[l] {
                    scales[l] = quadruplet_scale(means, depth);
                }
                for (c, x) in e.iter().enumerate() {
                    e_lanes[c * LANES + l] = *x;
                }
            }
            let diag = implicit.then(|| {
                d_lanes.resize(nc * LANES, 0.0);
                &mut d_lanes[..]
            });
            quadruplets.source_lanes(grid, e_lanes, self.tail, self.g, scales, s_lanes, diag);
            for (l, n) in spectra.chunks_exact(nc).enumerate() {
                let (k, depth, wind) = nodes.node(l, nf);
                let (a, b) = (&mut a[l * nc..(l + 1) * nc], &mut b[l * nc..(l + 1) * nc]);
                variance(grid, n, e);
                a.fill(0.0);
                b.fill(0.0);
                if means[l].is_some() {
                    for (c, b) in b.iter_mut().enumerate() {
                        *b = s_lanes[c * LANES + l];
                    }
                    if implicit {
                        for (c, a) in a.iter_mut().enumerate() {
                            *a = d_lanes[c * LANES + l];
                        }
                        linearise_transfer(e, a, b);
                    } else {
                        split_transfer(e, a, b);
                    }
                }
                self.add_local_rates(grid, e, k, depth, wind, means[l], a, b);
            }
            return;
        }
        #[cfg(not(feature = "simd"))]
        let _ = lanes;
        for (l, n) in spectra.chunks_exact(nc).enumerate() {
            let (k, depth, wind) = nodes.node(l, nf);
            let (a, b) = (&mut a[l * nc..(l + 1) * nc], &mut b[l * nc..(l + 1) * nc]);
            variance(grid, n, e);
            self.rates(grid, e, k, depth, wind, a, b);
        }
    }

    /// The update of [`Self::integrate`] from the rates `a`, `b` of `n`, then
    /// the tail. `e` is scratch.
    fn advance(
        &self,
        grid: &SpectralGrid,
        n: &mut [f64],
        k: &[f64],
        depth: f64,
        wind: Wind,
        dt: f64,
        e: &mut [f64],
        a: &[f64],
        b: &[f64],
    ) {
        let nd = grid.n_dir();
        // Hersbach & Janssen's limit: `C g ũ* f_c Δt` (the factor of
        // `f⁻⁴` in `F`), from the state at the start
        let rate_limit = match self.limiter {
            Some(GrowthLimiter::Rate(coefficient) | GrowthLimiter::RateBothSigns(coefficient)) => {
                variance(grid, n, e);
                Means::of(grid, e, k).map(|means| {
                    let f_c = limiter_frequency(grid, e, k, wind, means);
                    let floor = self.g * PM_PEAK / f_c;
                    coefficient * self.g * wind.friction_velocity().max(floor) * f_c * dt
                })
            }
            _ => None,
        };
        // The exponentials first (in `e`, free until the tail), so that the
        // loop around them is free of calls
        let growths = &mut *e;
        for (growth, b) in growths.iter_mut().zip(b) {
            *growth = (b * dt).exp();
        }
        for (i, (&sigma, &ki)) in grid.sigma.iter().zip(k).enumerate() {
            // The limit on the growth of N over the step, or none
            let max_growth = match self.limiter {
                Some(GrowthLimiter::PerStep(gamma)) => {
                    gamma * PHILLIPS / (2.0 * sigma * ki.powi(3) * group_velocity(sigma, ki, depth))
                }
                // ΔF(f) per 2πσ: N = E/σ, E(σ) = F(f)/2π
                Some(GrowthLimiter::Rate(_) | GrowthLimiter::RateBothSigns(_)) => rate_limit
                    .map_or(f64::INFINITY, |limit| {
                        limit * (sigma / TAU).powi(-4) / (TAU * sigma)
                    }),
                None => f64::INFINITY,
            };
            let max_loss = if self.limiter.is_some_and(|l| l.caps_losses()) {
                max_growth
            } else {
                f64::INFINITY
            };
            let row = i * nd..(i + 1) * nd;
            let rows = n[row.clone()]
                .iter_mut()
                .zip(&growths[row.clone()])
                .zip(&a[row.clone()])
                .zip(&b[row]);
            for (((n, &growth), &a), &b) in rows {
                let x = b * dt;
                // (e^x − 1)/x → 1 as x → 0
                let phi = if x.abs() < 1e-8 {
                    1.0 + 0.5 * x
                } else {
                    (growth - 1.0) / x
                };
                let next = *n * growth + a / sigma * dt * phi;
                *n = next.max(*n - max_loss).min(*n + max_growth).max(0.0);
            }
        }
        if let Some(power) = self.tail {
            self.apply_tail(grid, n, k, wind, power, e);
        }
        self.limit_to_depth(grid, n, depth);
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
        variance(grid, n, e);
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

/// `e = σ N`: the variance density of the action density `n`.
fn variance(grid: &SpectralGrid, n: &[f64], e: &mut [f64]) {
    let nd = grid.n_dir();
    for (c, (e, n)) in e.iter_mut().zip(n).enumerate() {
        *e = grid.sigma[c / nd] * n;
    }
}

/// The frequency `f_c` (Hz) of Hersbach & Janssen's limit for the variance
/// density `e` with wavenumbers `k` under `wind`: the larger of the mean
/// frequencies `m₀/m₋₁` of the wind sea and of the whole spectrum (`means`),
/// as ecWAM's `implsch.F90` (`USFM = u* max(FMEANWS, FMEAN)`). The wind sea is
/// the components with a positive wind input, `28 u*/c cos(θ − θ_w) > 1`
/// (Komen's; ecWAM's `XLLWS` marks those of Janssen's).
fn limiter_frequency(grid: &SpectralGrid, e: &[f64], k: &[f64], wind: Wind, means: Means) -> f64 {
    let nd = grid.n_dir();
    let us = wind.friction_velocity();
    let (mut m0, mut inv_sigma) = (0.0, 0.0);
    if us > 0.0 {
        let mut table = [0.0; MAX_DIRECTIONS];
        let cosine = |j: usize| (grid.theta[j] - wind.direction).cos();
        for (j, cos) in table.iter_mut().enumerate().take(nd) {
            *cos = cosine(j);
        }
        for (i, &ki) in k.iter().enumerate().take(grid.n_freq()) {
            // Fed where cos(θ − θ_w) > c/(28 u*)
            let threshold = grid.sigma[i] / ki / (28.0 * us);
            let row: f64 = (0..nd)
                .filter(|&j| table.get(j).copied().unwrap_or_else(|| cosine(j)) > threshold)
                .map(|j| e[i * nd + j])
                .sum::<f64>()
                * grid.d_sigma[i];
            m0 += row;
            inv_sigma += row / grid.sigma[i];
        }
    }
    let wind_sea = if m0 > 0.0 { m0 / inv_sigma } else { 0.0 };
    wind_sea.max(means.sigma) / TAU
}

/// The DIA's finite-depth factor at a node of depth `depth` with the
/// spectrum's `means`: `R(0.75 k̃ d)`.
fn quadruplet_scale(means: Means, depth: f64) -> f64 {
    shallow_water_factor(0.75 * means.k * depth)
}

/// Split the DIA's transfer in `b` (of the variance density `e`) into the
/// gains, which join the linear input `a`, and the losses, which stay in `b`
/// as the rate `S_nl/E`.
fn split_transfer(e: &[f64], a: &mut [f64], b: &mut [f64]) {
    for ((a, b), &e) in a.iter_mut().zip(b.iter_mut()).zip(e) {
        let s = std::mem::take(b);
        if s >= 0.0 || e <= 0.0 {
            *a = s;
        } else {
            *b = s / e;
        }
    }
}

/// The DIA's transfer `s` (in `b`) of `e`, linearised about it by its
/// diagonal `Λ` (in `a`) where that is negative: the rate `B = min(Λ, 0)`
/// and the input `A = s − B e` (`A + B E` is `s` at `E = e`).
fn linearise_transfer(e: &[f64], a: &mut [f64], b: &mut [f64]) {
    for ((a, b), &e) in a.iter_mut().zip(b.iter_mut()).zip(e) {
        let (s, rate) = (*b, a.min(0.0));
        *a = s - rate * e;
        *b = rate;
    }
}

/// Per-worker storage of [`SourceTerms::integrate_at`]: one node's
/// variance density, the nodes' input and rates, the midpoint's predictor,
/// and the interleaved spectra and transfers of the DIA's lanes.
#[derive(Default)]
pub(crate) struct SourceScratch {
    e: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    predictor: Vec<f64>,
    #[cfg(feature = "simd")]
    e_lanes: Vec<f64>,
    #[cfg(feature = "simd")]
    s_lanes: Vec<f64>,
    #[cfg(feature = "simd")]
    d_lanes: Vec<f64>,
}

impl SourceScratch {
    /// Storage for `n_components` components.
    pub fn new(n_components: usize) -> Self {
        Self {
            e: vec![0.0; n_components],
            a: vec![0.0; n_components],
            b: vec![0.0; n_components],
            predictor: Vec::new(),
            #[cfg(feature = "simd")]
            e_lanes: Vec::new(),
            #[cfg(feature = "simd")]
            s_lanes: Vec::new(),
            #[cfg(feature = "simd")]
            d_lanes: Vec::new(),
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

    /// Hersbach & Janssen's limit takes the wind sea's mean frequency when it
    /// is above the whole spectrum's (ecWAM's `max(FMEANWS, FMEAN)`): a swell
    /// against the wind leaves it as the wind sea alone sets it, where the
    /// whole spectrum's mean frequency would be the swell's.
    #[test]
    fn a_swell_does_not_tighten_the_growth_limit_of_the_wind_sea() {
        let grid = SpectralGrid::new(0.04, 0.6, 25, 36);
        let depth = 100.0;
        let k: Vec<f64> = grid
            .sigma
            .iter()
            .map(|&s| wavenumber(s, depth, G))
            .collect();
        let wind = Wind {
            u10: 15.0,
            direction: 0.0,
        };
        let sea = grid.jonswap(1.0, 4.0, 3.3, 0.0, 4.0);
        let swell = grid.jonswap(3.0, 14.0, 3.3, PI, 20.0);
        let both: Vec<f64> = sea.iter().zip(&swell).map(|(a, b)| a + b).collect();
        let f_c =
            |e: &[f64]| limiter_frequency(&grid, e, &k, wind, Means::of(&grid, e, &k).unwrap());
        let mean = |e: &[f64]| Means::of(&grid, e, &k).unwrap().sigma / TAU;
        let (alone, with_swell) = (f_c(&sea), f_c(&both));
        println!(
            "f_c: wind sea {alone:.4} Hz, with the swell {with_swell:.4} Hz; the mean frequency \
             of both {:.4} Hz",
            mean(&both)
        );
        // The swell travels against the wind: not fed, and outside the mean
        assert!((with_swell / alone - 1.0).abs() < 1e-12);
        assert!(with_swell > 2.0 * mean(&both));
        // Without wind, or a sea the wind does not feed: the whole mean
        let calm = Wind::default();
        assert_eq!(
            limiter_frequency(&grid, &both, &k, calm, Means::of(&grid, &both, &k).unwrap()),
            mean(&both)
        );
        assert_eq!(f_c(&swell), mean(&swell));
    }

    /// [`GrowthLimiter::RateBothSigns`] caps a loss by the growth's limit:
    /// a steep swell without wind under whitecapping at a 5-minute step,
    /// where `ũ* = g f*_PM/f_c` makes the limit `C g² f*_PM f⁻⁴ Δt` in
    /// `F(f, θ)`. [`GrowthLimiter::Rate`] lets the same loss through whole.
    #[test]
    fn the_two_sided_limit_caps_the_losses_by_the_growths_limit() {
        let grid = SpectralGrid::new(0.05, 0.6, 24, 18);
        let depth = 200.0;
        let k: Vec<f64> = grid
            .sigma
            .iter()
            .map(|&s| wavenumber(s, depth, G))
            .collect();
        let e0 = grid.jonswap(4.0, 5.0, 3.3, 0.0, 4.0);
        let n0: Vec<f64> = (0..e0.len())
            .map(|c| e0[c] / grid.sigma[c / grid.n_dir()])
            .collect();
        let (m, dt) = (n0.len(), 300.0);
        let run = |limiter| {
            let sources = SourceTerms::none(G)
                .with_whitecapping(true)
                .with_limiter(Some(limiter));
            let (mut e, mut a, mut b) = (vec![0.0; m], vec![0.0; m], vec![0.0; m]);
            let mut n = n0.clone();
            let wind = Wind::default();
            sources.integrate(&grid, &mut n, &k, depth, wind, dt, &mut e, &mut a, &mut b);
            n
        };
        let (one, both) = (
            run(GrowthLimiter::Rate(DEFAULT_RATE_LIMITER)),
            run(GrowthLimiter::RateBothSigns(DEFAULT_RATE_LIMITER)),
        );
        let (mut capped, mut free) = (0, 0);
        for c in 0..m {
            let sigma = grid.sigma[c / grid.n_dir()];
            let f = sigma / TAU;
            let limit = DEFAULT_RATE_LIMITER * G * G * PM_PEAK * f.powi(-4) * dt / (TAU * sigma);
            // Whitecapping only removes
            assert!(one[c] <= n0[c] && both[c] <= n0[c]);
            if n0[c] - one[c] > limit {
                // The loss beyond the limit is cut to it
                assert!(
                    (n0[c] - both[c] - limit).abs() <= 1e-12 * (n0[c] + limit),
                    "component {c}: loss {:e}, limit {limit:e}",
                    n0[c] - both[c]
                );
                capped += 1;
            } else {
                assert_eq!(both[c], one[c]);
                free += 1;
            }
        }
        println!("{capped} components capped, {free} free");
        assert!(capped > 0 && free > 0, "{capped} capped, {free} free");
    }

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

    /// The exponential midpoint converges at second order in the step and
    /// the frozen rates at first: a young sea under SWAN's sources (wind,
    /// whitecapping, the DIA, friction) for 10 minutes at one node, against
    /// the midpoint at a step of 1.875 s.
    #[test]
    fn the_midpoint_rule_is_second_order() {
        let grid = SpectralGrid::new(0.05, 0.6, 24, 18);
        let depth = 30.0;
        let k: Vec<f64> = grid
            .sigma
            .iter()
            .map(|&s| wavenumber(s, depth, G))
            .collect();
        let wind = Wind {
            u10: 15.0,
            direction: 0.0,
        };
        // No limiter: it caps the growth per step, whatever the step
        let sources = SourceTerms::swan_defaults(G).with_limiter(None);
        let e0 = grid.jonswap(0.8, 4.0, 3.3, 0.0, 2.0);
        let n0: Vec<f64> = (0..e0.len())
            .map(|c| e0[c] / grid.sigma[c / grid.n_dir()])
            .collect();
        let m = n0.len();
        let run = |integration: SourceIntegration, dt: f64| {
            let sources = sources.clone().with_integration(integration);
            let (mut e, mut a, mut b) = (vec![0.0; m], vec![0.0; m], vec![0.0; m]);
            let mut n = n0.clone();
            for _ in 0..(600.0 / dt).round() as usize {
                sources.integrate(&grid, &mut n, &k, depth, wind, dt, &mut e, &mut a, &mut b);
            }
            let e: Vec<f64> = (0..m)
                .map(|c| n[c] * grid.sigma[c / grid.n_dir()])
                .collect();
            grid.moment(&e, 0)
        };
        let reference = run(SourceIntegration::Midpoint, 1.875);
        // The sea grows: the sources act
        let m0 = grid.moment(&e0, 0);
        assert!(reference > 1.5 * m0, "m0 {m0} → {reference}");
        for (integration, order) in [
            (SourceIntegration::FrozenRates, 1.0),
            (SourceIntegration::Midpoint, 2.0),
        ] {
            let errors: Vec<f64> = [60.0, 30.0, 15.0]
                .iter()
                .map(|&dt| (run(integration, dt) - reference).abs() / reference)
                .collect();
            for pair in errors.windows(2) {
                let rate = (pair[0] / pair[1]).log2();
                assert!(
                    (rate - order).abs() < 0.25,
                    "{integration:?}: errors {errors:?}, rate {rate:.2}"
                );
            }
        }
    }

    /// The dissipation diagnostic against what the sinks do over a step
    /// (frozen rates, as `integrate` steps them): the variance each removes
    /// per second, and with all three the waves' momentum `g Σ (k/σ) e_θ E`
    /// they remove, to round-off, at a short step and at a long one where
    /// the instantaneous `−B E` would claim more than there is. A spread sea
    /// at 30° in 3 m of water, where whitecapping, friction and breaking all
    /// act.
    #[test]
    fn the_dissipation_is_what_the_sinks_remove() {
        let grid = SpectralGrid::new(0.05, 0.5, 20, 24);
        let depth = 3.0;
        let k: Vec<f64> = grid
            .sigma
            .iter()
            .map(|&s| wavenumber(s, depth, G))
            .collect();
        let e0 = grid.jonswap(2.0, 7.0, 3.3, 0.5, 4.0);
        let nd = grid.n_dir();
        let m = e0.len();
        let all = SourceTerms::swan_defaults(G);
        let none = SourceTerms::none(G);
        let sinks = none
            .clone()
            .with_whitecapping(true)
            .with_bottom_friction(all.bottom_friction)
            .with_breaking(all.breaking);
        let (mut e, mut a, mut b) = (vec![0.0; m], vec![0.0; m], vec![0.0; m]);
        // Variance and momentum (per ρ) of a spectrum
        let integrals = |e: &[f64]| {
            let (mut m0, mut momentum) = (0.0, [0.0; 2]);
            for c in 0..m {
                let (i, j) = (c / nd, c % nd);
                let w = e[c] * grid.d_sigma[i] * grid.d_theta;
                m0 += w;
                momentum[0] += G * k[i] / grid.sigma[i] * grid.cos_theta[j] * w;
                momentum[1] += G * k[i] / grid.sigma[i] * grid.sin_theta[j] * w;
            }
            (m0, momentum)
        };
        let mut removed = |sources: &SourceTerms, dt: f64| {
            let mut n: Vec<f64> = (0..m).map(|c| e0[c] / grid.sigma[c / nd]).collect();
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
            let after: Vec<f64> = (0..m).map(|c| n[c] * grid.sigma[c / nd]).collect();
            let ((m0, p0), (m1, p1)) = (integrals(&e0), integrals(&after));
            ((m0 - m1) / dt, [(p0[0] - p1[0]) / dt, (p0[1] - p1[1]) / dt])
        };
        let (mut ra, mut rb) = (vec![0.0; m], vec![0.0; m]);
        let instant = all.dissipation(&grid, &e0, &k, depth, 0.0, &mut ra, &mut rb);
        assert!(
            instant.whitecapping > 0.0 && instant.bottom_friction > 0.0 && instant.breaking > 0.0,
            "{instant:?}"
        );
        let m0 = integrals(&e0).0;
        for dt in [1e-3, 600.0] {
            let d = all.dissipation(&grid, &e0, &k, depth, dt, &mut ra, &mut rb);
            // Each sink alone: the diagnostic over the step of that sink alone
            type Get = fn(&Dissipation) -> f64;
            for (name, sink, got) in [
                (
                    "whitecapping",
                    none.clone().with_whitecapping(true),
                    (|d: &Dissipation| d.whitecapping) as Get,
                ),
                (
                    "friction",
                    none.clone().with_bottom_friction(all.bottom_friction),
                    |d: &Dissipation| d.bottom_friction,
                ),
                (
                    "breaking",
                    none.clone().with_breaking(all.breaking),
                    |d: &Dissipation| d.breaking,
                ),
            ] {
                let (variance, _) = removed(&sink, dt);
                let alone = got(&sink.dissipation(&grid, &e0, &k, depth, dt, &mut ra, &mut rb));
                assert!(
                    (variance - alone).abs() <= 1e-12 * m0 / dt.min(1.0),
                    "{name} at {dt} s: {variance} against {alone}"
                );
            }
            // All three together, shared among them by their rates
            let (variance, force) = removed(&sinks, dt);
            assert!(
                (variance - d.total()).abs() <= 1e-12 * m0 / dt.min(1.0),
                "{dt} s: {variance} against {}",
                d.total()
            );
            let size = d.force[0].hypot(d.force[1]);
            for (x, y) in force.iter().zip(d.force) {
                assert!(
                    (x - y).abs() <= 1e-10 * size,
                    "{dt} s: {force:?} against {:?}",
                    d.force
                );
            }
            // Along the waves, about 30° off x
            let angle = d.force[1].atan2(d.force[0]);
            assert!((angle - 0.5).abs() < 0.05, "{angle}");
            // Never more than there is (to round-off: at 600 s it is all of it)
            assert!(
                d.total() * dt <= m0 * (1.0 + 1e-12),
                "{dt} s: {} of {m0}",
                d.total() * dt
            );
        }
        // An instant is the rate times the energy: a short step's limit
        let short = all.dissipation(&grid, &e0, &k, depth, 1e-3, &mut ra, &mut rb);
        assert!((short.total() / instant.total() - 1.0).abs() < 1e-4);
        let long = all.dissipation(&grid, &e0, &k, depth, 600.0, &mut ra, &mut rb);
        assert!(
            long.total() < 0.5 * instant.total(),
            "{long:?} against {instant:?}"
        );
        // No energy, no dissipation
        let calm = all.dissipation(&grid, &vec![0.0; m], &k, depth, 60.0, &mut ra, &mut rb);
        assert_eq!(calm, Dissipation::default());
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

    #[test]
    fn the_depth_limit_caps_a_sea_breaking_cannot_remove() {
        // A 2 m sea in 0.11 m of water: breaking saturates and barely touches
        // it, the depth limit scales it to `H_rms = γ d` in one step
        let grid = SpectralGrid::new(0.05, 0.5, 20, 12);
        let nd = grid.n_dir();
        let state = |hs: f64, depth: f64| {
            let e0 = grid.jonswap(hs, 8.0, 3.3, 0.0, 2.0);
            let n: Vec<f64> = (0..e0.len()).map(|c| e0[c] / grid.sigma[c / nd]).collect();
            let k: Vec<f64> = grid
                .sigma
                .iter()
                .map(|&s| wavenumber(s, depth, G))
                .collect();
            (n, k)
        };
        let m0 = |n: &[f64]| {
            let e: Vec<f64> = (0..n.len()).map(|c| n[c] * grid.sigma[c / nd]).collect();
            grid.moment(&e, 0)
        };
        let breaking = SourceTerms::none(G).with_breaking(Some((1.0, 0.73)));
        let limited = breaking.clone().with_depth_limit(true);
        let step = |sources: &SourceTerms, n: &mut [f64], k: &[f64], depth: f64| {
            let m = n.len();
            let (mut e, mut a, mut b) = (vec![0.0; m], vec![0.0; m], vec![0.0; m]);
            let wind = Wind::default();
            sources.integrate(&grid, n, k, depth, wind, 10.0, &mut e, &mut a, &mut b);
        };
        let depth = 0.11;
        let (n0, k) = state(2.0, depth);
        let (mut free, mut capped) = (n0.clone(), n0.clone());
        step(&breaking, &mut free, &k, depth);
        step(&limited, &mut capped, &k, depth);
        let bound = (0.73 * depth).powi(2) / 8.0;
        // Breaking alone keeps 99.4 % of the energy
        assert!(m0(&free) > 0.99 * m0(&n0), "{} of {}", m0(&free), m0(&n0));
        assert!(
            (m0(&capped) / bound - 1.0).abs() < 1e-12,
            "{} against {bound}",
            m0(&capped)
        );
        // The shape is breaking's (which scales every component alike)
        let ratio = capped[0] / free[0];
        for (c, f) in capped.iter().zip(&free) {
            assert!((c - f * ratio).abs() <= 1e-12 * f);
        }
        // No other sink on: the limit without breaking does nothing
        let mut none = n0.clone();
        let unbroken = SourceTerms::none(G).with_depth_limit(true);
        step(&unbroken, &mut none, &k, depth);
        assert_eq!(none, n0);
        // Below the limit, bit for bit as without it
        let depth = 3.0;
        let (n0, k) = state(1.0, depth);
        let (mut free, mut capped) = (n0.clone(), n0);
        step(&breaking, &mut free, &k, depth);
        step(&limited, &mut capped, &k, depth);
        assert_eq!(free, capped);
    }
}
