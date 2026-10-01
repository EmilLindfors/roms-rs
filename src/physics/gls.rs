//! Generic length scale (GLS) turbulence closure: two transport equations
//! for the turbulent kinetic energy `k` and a length-scale variable `ψ`
//! (Umlauf & Burchard 2003; ROMS: Warner et al. 2005), per water column.
//!
//! # Equations
//!
//! ```text
//!     ∂k/∂t = ∂/∂z(ν/σ_k ∂k/∂z) + P + B − ε,
//!     ∂ψ/∂t = ∂/∂z(ν/σ_ψ ∂ψ/∂z) + (ψ/k)(c₁P + c₃B − c₂ε),
//!     ψ = (c_μ⁰)^p k^m l^n,    ε = (c_μ⁰)^(3+p/n) k^(3/2+m/n) ψ^(−1/n),
//! ```
//!
//! with shear production `P = ν S²`, buoyancy production `B = −ν′ N²`, the
//! length `l = (c_μ⁰)³ k^(3/2)/ε` and the eddy viscosity and diffusivity
//! `ν = c_μ √k l`, `ν′ = c_μ′ √k l`. The exponents `(p, m, n)` select the
//! model: `(3, 3/2, −1)` is k-ε (`ψ = ε`), `(−1, 1/2, −1)` k-ω, `(2, 1, −2/3)`
//! Umlauf & Burchard's "generic" model ([`GlsParameters`]).
//!
//! The stability functions `c_μ`, `c_μ′` are Canuto et al.'s (2001), version
//! A or B, in their quasi-equilibrium form (Umlauf & Burchard 2005; GOTM's
//! `cmue_d`): functions of `α_N = (k/ε)² N²` alone, with `α_N` bounded below
//! where convection would make them singular. `c_μ⁰` is their neutral value,
//! so that a log layer has `k = u*²/(c_μ⁰)²` exactly.
//!
//! Three constants follow from the others, as in GOTM:
//! - `c₃` in stable stratification (`B < 0`) from the steady-state
//!   Richardson number `Ri_st` (0.25): homogeneous shear turbulence at
//!   `Ri = Ri_st` neither grows nor decays ([`GlsMixing::c3_minus`]);
//! - `c₃` in unstable stratification such that the ε equation the model
//!   implies has k-ε's `c₃ε⁺ = 1` (in the presets, [`GlsParameters`]);
//! - von Kármán's constant `κ = c_μ⁰ √(σ_ψ (c₂ − c₁)/n²)`, the one the ψ
//!   equation keeps in a log layer ([`GlsMixing::kappa`]).
//!
//! Galperin et al.'s (1988) limit `l ≤ c_lim √(2k)/N` (stable water) bounds
//! the length.
//!
//! # Discretisation (Umlauf & Burchard 2005; GOTM `tkeeq`, `genericeq`)
//!
//! `k` and `ψ` live at the w-points. Once per baroclinic step, before the
//! implicit vertical diffusion of momentum and tracers that uses the new `ν`,
//! `ν′`:
//! 1. `S²`, `N²` at the interior w-points from the layer-centre velocity and
//!    density, `P`, `B` from the old `ν`, `ν′`, `ε`;
//! 2. `k`, then `ψ`, by one backward-Euler step with implicit diffusion. The
//!    sinks are linearised in the new value (Patankar 1980: a negative
//!    `B` is a sink when `P + B < 0`), so both stay positive for any `Δt`;
//! 3. `ε` from the new `k`, `ψ`, bounded below by `ε_min` and the Galperin
//!    limit, and `ψ` reset to the bounded `ε`.
//!
//! The boundaries are logarithmic layers with flux conditions at the
//! boundary layers' centres: no flux of `k`, and the flux of `ψ` of a log
//! layer, `−n (c_μ⁰)^(p+1) κ^(n+1)/σ_ψ k^(m+1/2) (z′ + z₀)^n`, at a height
//! `z′` of half the end layer over roughness `z₀`. The boundary w-points
//! themselves carry `k = u*²/(c_μ⁰)²`, `l = κ z₀` (diagnostic only).
//!
//! The model is local to each column: `k` and `ψ` are not advected
//! horizontally or vertically (as in GOTM; ROMS advects them).
//!
//! # Verification
//!
//! - Wind entrainment into linear stratification (Kato & Phillips 1969):
//!   the mixed layer is within 1.5 % of Price's (1979) `1.05 u* √(t/N₀)`
//!   at 12–30 h with k-ε, at Δt = 10 and 60 s alike (gate in
//!   `physics::vertical_diffusion`).
//! - Open-channel flow: bottom stress `u*²` at steady state, `k` in local
//!   equilibrium with the linear stress to 3 %, the wall length `κz` at the
//!   first w-point (0.96 at 80 levels).
//! - Homogeneous shear turbulence is steady at `Ri_st` (growth rate
//!   ≈ 10⁻⁵ S, against ±10⁻² S at Ri = 0.2 and 0.3).
//! - Convection: an unstable column overturns from the minimum turbulence
//!   within hours, with no convective adjustment.

use crate::physics::vertical_mixing::{Column, Forcing, Turbulence, VerticalMixing};
use crate::solver::algorithms::tridiagonal::solve_tridiagonal;

/// Exponents and constants of a GLS model (see the module docs). The
/// presets are ROMS's (`ocean.in`), except for `c₃`: `c₃⁻` is computed
/// ([`GlsMixing::c3_minus`]), and `c₃⁺` is GOTM's, `(3/2 − c₃ε⁺)·n + m`
/// for the `c₃ε⁺ = 1` of k-ε's ε equation (ROMS has 1 for every model,
/// which in k-ω is `c₃ε⁺ = 2`: k-ω then cannot start convection from
/// weak turbulence).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlsParameters {
    /// Exponents of `ψ = (c_μ⁰)^p k^m l^n`
    pub p: f64,
    pub m: f64,
    pub n: f64,
    /// Schmidt numbers of `k` and `ψ`
    pub sigma_k: f64,
    pub sigma_psi: f64,
    /// `c₁`, `c₂` of the `ψ` equation, and `c₃` in unstable stratification
    /// (`B > 0`)
    pub c1: f64,
    pub c2: f64,
    pub c3_plus: f64,
}

impl GlsParameters {
    /// k-ε (Rodi 1987).
    pub const K_EPSILON: Self = Self {
        p: 3.0,
        m: 1.5,
        n: -1.0,
        sigma_k: 1.0,
        sigma_psi: 1.3,
        c1: 1.44,
        c2: 1.92,
        c3_plus: 1.0,
    };

    /// k-ω (Wilcox 1988): `ψ = ω = ε/((c_μ⁰)⁴ k)`.
    pub const K_OMEGA: Self = Self {
        p: -1.0,
        m: 0.5,
        n: -1.0,
        sigma_k: 2.0,
        sigma_psi: 2.0,
        c1: 0.555,
        c2: 0.833,
        c3_plus: 0.0,
    };

    /// Umlauf & Burchard's (2003) "generic" model.
    pub const GENERIC: Self = Self {
        p: 2.0,
        m: 1.0,
        n: -0.67,
        sigma_k: 0.8,
        sigma_psi: 1.07,
        c1: 1.0,
        c2: 1.22,
        c3_plus: 0.665,
    };
}

/// Stability functions of [`GlsMixing`]: Canuto et al. (2001), quasi-equilibrium.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum StabilityFunctions {
    /// Version A (GOTM's and ROMS's default).
    #[default]
    CanutoA,
    /// Version B.
    CanutoB,
}

/// The quasi-equilibrium stability functions' polynomial coefficients
/// (Umlauf & Burchard 2005, App. A; GOTM `cmue_d`), `c_μ⁰`, and the lowest
/// `α_N` they are evaluated at.
#[derive(Clone, Copy, Debug)]
struct Stability {
    d: [f64; 6],
    n: [f64; 3],
    nt: [f64; 3],
    cm0: f64,
    an_min: f64,
}

impl Stability {
    fn new(functions: StabilityFunctions) -> Self {
        // Pressure-strain and pressure-scrambling constants (GOTM `init_scnd`)
        let [cc1, cc2, cc3, cc4, cc5, cc6, ct1, ct2, ct3, ct4, ct5, ctt]: [f64; 12] =
            match functions {
                StabilityFunctions::CanutoA => [
                    5.0, 0.8, 1.968, 1.136, 0.0, 0.4, 5.95, 0.6, 1.0, 0.0, 0.3333, 0.72,
                ],
                StabilityFunctions::CanutoB => [
                    5.0, 0.6983, 1.9664, 1.094, 0.0, 0.495, 5.6, 0.6, 1.0, 0.0, 0.3333, 0.477,
                ],
            };
        let a1 = 2.0 / 3.0 - cc2 / 2.0;
        let a2 = 1.0 - cc3 / 2.0;
        let a3 = 1.0 - cc4 / 2.0;
        let _a4 = cc5 / 2.0;
        let a5 = 0.5 - cc6 / 2.0;
        let at1 = 1.0 - ct2;
        let at2 = 1.0 - ct3;
        let at3 = 2.0 * (1.0 - ct4);
        let _at4 = 2.0 * (1.0 - ct5);
        let at5 = 2.0 * ctt * (1.0 - ct5);
        let nn = 0.5 * cc1;
        let nt = ct1;

        let d0 = 36.0 * nn.powi(3) * nt.powi(2);
        let d1 = 84.0 * a5 * at3 * nn.powi(2) * nt + 36.0 * at5 * nn.powi(3) * nt;
        let d2 = 9.0 * (at2.powi(2) - at1.powi(2)) * nn.powi(3)
            - 12.0 * (a2.powi(2) - 3.0 * a3.powi(2)) * nn * nt.powi(2);
        let d3 = 12.0 * a5 * at3 * (a2 * at1 - 3.0 * a3 * at2) * nn
            + 12.0 * a5 * at3 * (a3.powi(2) - a2.powi(2)) * nt
            + 12.0 * at5 * (3.0 * a3.powi(2) - a2.powi(2)) * nn * nt;
        let d4 = 48.0 * a5.powi(2) * at3.powi(2) * nn + 36.0 * a5 * at3 * at5 * nn.powi(2);
        let d5 = 3.0 * (a2.powi(2) - 3.0 * a3.powi(2)) * (at1.powi(2) - at2.powi(2)) * nn;
        let n0 = 36.0 * a1 * nn.powi(2) * nt.powi(2);
        let n1 = -12.0 * a5 * at3 * (at1 + at2) * nn.powi(2)
            + 8.0 * a5 * at3 * (6.0 * a1 - a2 - 3.0 * a3) * nn * nt
            + 36.0 * a1 * at5 * nn.powi(2) * nt;
        let n2 = 9.0 * a1 * (at2.powi(2) - at1.powi(2)) * nn.powi(2);
        let nt0 = 12.0 * at3 * nn.powi(3) * nt;
        let nt1 = 12.0 * a5 * at3.powi(2) * nn.powi(2);
        let nt2 = 9.0 * a1 * at3 * (at1 - at2) * nn.powi(2)
            + (6.0 * a1 * (a2 - 3.0 * a3) - 4.0 * (a2.powi(2) - 3.0 * a3.powi(2))) * at3 * nn * nt;

        // The neutral value (GOTM `compute_cm0`), and the convective limit
        // of α_N where the denominator vanishes
        let cm0 = ((a2.powi(2) - 3.0 * a3.powi(2) + 3.0 * a1 * nn) / (3.0 * nn.powi(2))).powf(0.25);
        let an_min = (-(d1 + nt0) + ((d1 + nt0).powi(2) - 4.0 * d0 * (d4 + nt1)).sqrt())
            / (2.0 * (d4 + nt1));
        Self {
            d: [d0, d1, d2, d3, d4, d5],
            n: [n0, n1, n2],
            nt: [nt0, nt1, nt2],
            cm0,
            an_min,
        }
    }

    /// `(c_μ, c_μ′)` at `α_N = (k/ε)² N²`, with `α_S` from `P + B = ε`.
    #[inline]
    fn eval(&self, an: f64) -> (f64, f64) {
        let [d0, d1, d2, d3, d4, d5] = self.d;
        let [n0, n1, n2] = self.n;
        let [nt0, nt1, nt2] = self.nt;
        let an = an.max(0.5 * self.an_min);
        let tmp0 = -d0 - (d1 + nt0) * an - (d4 + nt1) * an * an;
        let tmp1 = -d2 + n0 + (n1 - d3 - nt2) * an;
        let tmp2 = n2 - d5;
        let a_s = if tmp2.abs() < 1e-10 {
            -tmp0 / tmp1
        } else {
            (-tmp1 + (tmp1 * tmp1 - 4.0 * tmp0 * tmp2).sqrt()) / (2.0 * tmp2)
        };
        let dcm = d0 + d1 * an + d2 * a_s + d3 * an * a_s + d4 * an * an + d5 * a_s * a_s;
        let ncm = n0 + n1 * an + n2 * a_s;
        let ncmp = nt0 + nt1 * an + nt2 * a_s;
        let cm3_inv = 1.0 / self.cm0.powi(3);
        (cm3_inv * ncm / dcm, cm3_inv * ncmp / dcm)
    }

    /// `c₃` for which homogeneous shear turbulence at `Ri = ri` is steady:
    /// `c₂ + (c₁ − c₂) c_μ/(c_μ′ Ri)` at the `α_N` where
    /// `c_μ α_N/Ri − c_μ′ α_N = (c_μ⁰)⁻³` (GOTM `compute_cpsi3`).
    fn c3_for(&self, c1: f64, c2: f64, ri: f64) -> f64 {
        let residual = |an: f64| {
            let (cm, cmp) = self.eval(an);
            cm * an / ri - cmp * an - self.cm0.powi(-3)
        };
        let (mut an, e) = (5.0, 1e-8);
        for _ in 0..200 {
            let f = residual(an);
            let step = -f * e / (residual(an + e) - f);
            assert!(
                step.abs() <= 100.0,
                "steady-state Richardson number {ri} is out of the stability functions' range"
            );
            an += 0.5 * step;
            if step.abs() < 1e-10 {
                break;
            }
        }
        let (cm, cmp) = self.eval(an);
        c2 + (c1 - c2) / ri * cm / cmp
    }
}

/// GLS vertical mixing (see the module docs). The turbulence is stored in
/// `Solution3D::tke`, `Solution3D::gls` and stepped by the vertical
/// diffusion ([`crate::physics::apply_vertical_diffusion`]).
#[derive(Clone, Debug)]
pub struct GlsMixing {
    params: GlsParameters,
    stability: Stability,
    c3_minus: f64,
    kappa: f64,
    /// Lower bounds of `k` (m²/s²) and `ε` (m²/s³).
    k_min: f64,
    eps_min: f64,
    /// Galperin's `c_lim`, if the length is limited.
    length_limit: Option<f64>,
    /// Added to the turbulent viscosity and diffusivity (m²/s).
    background_viscosity: f64,
    background_diffusivity: f64,
    /// Roughness lengths of the surface and of the bed (m).
    z0_surface: f64,
    z0_bottom: f64,
}

impl GlsMixing {
    /// Default lower bound of `k` (GOTM's).
    pub const DEFAULT_K_MIN: f64 = 1e-8;
    /// Default lower bound of `ε` (GOTM's).
    pub const DEFAULT_EPS_MIN: f64 = 1e-12;
    /// Galperin et al.'s (1988) length limit.
    pub const DEFAULT_LENGTH_LIMIT: f64 = 0.53;
    /// Default steady-state Richardson number.
    pub const DEFAULT_RI_ST: f64 = 0.25;
    /// Default background viscosity and diffusivity (ROMS's usual `Akv_bak`,
    /// `Akt_bak`).
    pub const DEFAULT_BACKGROUND_VISCOSITY: f64 = 1e-5;
    pub const DEFAULT_BACKGROUND_DIFFUSIVITY: f64 = 1e-6;
    /// Default roughness of the surface and the bed (ROMS's `Zos`, `Zob`).
    pub const DEFAULT_ROUGHNESS: f64 = 0.02;

    /// A GLS model with `params` and `stability` functions; `c₃⁻` for
    /// [`Self::DEFAULT_RI_ST`] (see [`Self::with_steady_state_richardson`]).
    ///
    /// # Panics
    /// If `σ_ψ(c₂ − c₁)/n²` is not positive (no log layer), or `n ≥ 0`.
    pub fn new(params: GlsParameters, stability: StabilityFunctions) -> Self {
        let stability = Stability::new(stability);
        assert!(
            params.n < 0.0,
            "GLS exponent n must be negative, got {}",
            params.n
        );
        let radicand = params.sigma_psi * (params.c2 - params.c1) / params.n.powi(2);
        assert!(
            radicand > 0.0,
            "GLS constants have no log layer: σ_ψ(c₂ − c₁)/n² = {radicand}"
        );
        Self {
            params,
            stability,
            c3_minus: stability.c3_for(params.c1, params.c2, Self::DEFAULT_RI_ST),
            kappa: stability.cm0 * radicand.sqrt(),
            k_min: Self::DEFAULT_K_MIN,
            eps_min: Self::DEFAULT_EPS_MIN,
            length_limit: Some(Self::DEFAULT_LENGTH_LIMIT),
            background_viscosity: Self::DEFAULT_BACKGROUND_VISCOSITY,
            background_diffusivity: Self::DEFAULT_BACKGROUND_DIFFUSIVITY,
            z0_surface: Self::DEFAULT_ROUGHNESS,
            z0_bottom: Self::DEFAULT_ROUGHNESS,
        }
    }

    /// k-ε with Canuto A stability functions (GOTM's default model).
    pub fn k_epsilon() -> Self {
        Self::new(GlsParameters::K_EPSILON, StabilityFunctions::CanutoA)
    }

    /// k-ω with Canuto A stability functions.
    pub fn k_omega() -> Self {
        Self::new(GlsParameters::K_OMEGA, StabilityFunctions::CanutoA)
    }

    /// Umlauf & Burchard's generic model with Canuto A stability functions.
    pub fn generic() -> Self {
        Self::new(GlsParameters::GENERIC, StabilityFunctions::CanutoA)
    }

    /// `c₃⁻` from the steady-state Richardson number `ri` instead.
    pub fn with_steady_state_richardson(mut self, ri: f64) -> Self {
        assert!(
            ri > 0.0,
            "steady-state Richardson number must be positive, got {ri}"
        );
        self.c3_minus = self.stability.c3_for(self.params.c1, self.params.c2, ri);
        self
    }

    /// Roughness lengths `z0_surface`, `z0_bottom` (m) of the boundary log
    /// layers. Match the bed's to the drag's ([`crate::physics::BottomDrag3D`]).
    pub fn with_roughness(mut self, z0_surface: f64, z0_bottom: f64) -> Self {
        assert!(
            z0_surface > 0.0 && z0_bottom > 0.0,
            "roughness lengths must be positive, got {z0_surface}, {z0_bottom}"
        );
        self.z0_surface = z0_surface;
        self.z0_bottom = z0_bottom;
        self
    }

    /// Background viscosity and diffusivity (m²/s), added to the turbulent ones.
    pub fn with_background(mut self, viscosity: f64, diffusivity: f64) -> Self {
        assert!(
            viscosity >= 0.0 && diffusivity >= 0.0,
            "background mixing must be non-negative, got {viscosity}, {diffusivity}"
        );
        self.background_viscosity = viscosity;
        self.background_diffusivity = diffusivity;
        self
    }

    /// Galperin's length limit `l ≤ c_lim √(2k)/N` with `c_lim`, or none.
    pub fn with_length_limit(mut self, c_lim: Option<f64>) -> Self {
        self.length_limit = c_lim;
        self
    }

    /// Lower bounds of `k` (m²/s²) and `ε` (m²/s³).
    pub fn with_minimum(mut self, k_min: f64, eps_min: f64) -> Self {
        assert!(
            k_min > 0.0 && eps_min > 0.0,
            "minimum k and ε must be positive, got {k_min}, {eps_min}"
        );
        self.k_min = k_min;
        self.eps_min = eps_min;
        self
    }

    /// The model's exponents and constants.
    pub fn parameters(&self) -> GlsParameters {
        self.params
    }

    /// Neutral stability function `c_μ⁰`.
    pub fn cm0(&self) -> f64 {
        self.stability.cm0
    }

    /// `c₃` in stable stratification.
    pub fn c3_minus(&self) -> f64 {
        self.c3_minus
    }

    /// The von Kármán constant of the model's log layer.
    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    /// Stability functions `(c_μ, c_μ′)` at `α_N = (k/ε)² N²`.
    pub fn stability_functions(&self, an: f64) -> (f64, f64) {
        self.stability.eval(an)
    }

    /// `ε` of `k` and `ψ`.
    #[inline]
    pub fn dissipation(&self, k: f64, psi: f64) -> f64 {
        let GlsParameters { p, m, n, .. } = self.params;
        self.stability.cm0.powf(3.0 + p / n) * k.powf(1.5 + m / n) * psi.powf(-1.0 / n)
    }

    /// `ψ` of `k` and `ε`.
    #[inline]
    pub fn psi(&self, k: f64, eps: f64) -> f64 {
        let GlsParameters { p, m, n, .. } = self.params;
        let cm0 = self.stability.cm0;
        let length = cm0.powi(3) * k.powf(1.5) / eps;
        cm0.powf(p) * k.powf(m) * length.powf(n)
    }

    /// Turbulent viscosity and diffusivity `(c_μ, c_μ′)·√k l` of `k`, `ε`
    /// and `N²`.
    #[inline]
    fn coefficients(&self, k: f64, eps: f64, n2: f64) -> (f64, f64) {
        let tau = k / eps;
        let (cm, cmp) = self.stability.eval(tau * tau * n2);
        let x = self.stability.cm0.powi(3) * k * k / eps;
        (cm * x, cmp * x)
    }

    /// `ε` bounded below by `ε_min` and, in stable water, the length limit.
    #[inline]
    fn bounded_dissipation(&self, k: f64, eps: f64, n2: f64) -> f64 {
        let eps = eps.max(self.eps_min);
        match self.length_limit {
            Some(c_lim) if n2 > 0.0 => {
                let cm0 = self.stability.cm0;
                eps.max(cm0.powi(3) / (std::f64::consts::SQRT_2 * c_lim) * k * n2.sqrt())
            }
            _ => eps,
        }
    }

    /// `ψ` at a wall with friction-velocity `k` and roughness `z0`.
    #[inline]
    fn wall_psi(&self, k: f64, z0: f64) -> f64 {
        let GlsParameters { p, m, n, .. } = self.params;
        self.stability.cm0.powf(p) * k.powf(m) * (self.kappa * z0).powf(n)
    }

    /// Flux of `ψ` into the water of a log layer at `z′ + z₀` from a wall,
    /// with `k` there.
    #[inline]
    fn wall_psi_flux(&self, k: f64, height: f64) -> f64 {
        let GlsParameters {
            p, m, n, sigma_psi, ..
        } = self.params;
        -n * self.stability.cm0.powf(p + 1.0) * self.kappa.powf(n + 1.0) / sigma_psi
            * k.powf(m + 0.5)
            * height.powf(n)
    }
}

/// Per-w-point buffers of one column step, in [`Turbulence::scratch`].
struct Fields<'a> {
    /// Layer thicknesses (the first N)
    h: &'a mut [f64],
    n2: &'a mut [f64],
    s2: &'a mut [f64],
    /// Viscosity, `P`, `B`, `ε` of the old turbulence
    num: &'a mut [f64],
    production: &'a mut [f64],
    buoyancy: &'a mut [f64],
    eps: &'a mut [f64],
    k_old: &'a mut [f64],
    psi_old: &'a mut [f64],
}

/// Tridiagonal system of the interior w-points, in [`Turbulence::scratch`].
struct Solver<'a> {
    a: &'a mut [f64],
    b: &'a mut [f64],
    c: &'a mut [f64],
    d: &'a mut [f64],
    x: &'a mut [f64],
    c_prime: &'a mut [f64],
    d_prime: &'a mut [f64],
}

/// `scratch` split into [`Fields`] and [`Solver`] for `nw` w-points.
fn work(scratch: &mut Vec<f64>, nw: usize) -> (Fields<'_>, Solver<'_>) {
    const N_FIELDS: usize = 9;
    const N_SOLVER: usize = 7;
    let len = (N_FIELDS + N_SOLVER) * nw;
    if scratch.len() < len {
        scratch.resize(len, 0.0);
    }
    let mut chunks = scratch[..len].chunks_exact_mut(nw);
    let mut next = || chunks.next().expect("sized above");
    (
        Fields {
            h: next(),
            n2: next(),
            s2: next(),
            num: next(),
            production: next(),
            buoyancy: next(),
            eps: next(),
            k_old: next(),
            psi_old: next(),
        },
        Solver {
            a: next(),
            b: next(),
            c: next(),
            d: next(),
            x: next(),
            c_prime: next(),
            d_prime: next(),
        },
    )
}

/// One backward-Euler step of `∂Y/∂t = ∂/∂z(D ∂Y/∂z) + Q + L·Y` at the
/// interior w-points `1..nw−1` of `y`: `diff(j)` is `D` at w-point `j`
/// (averaged to the layers between), `source(j) = (Q, L)` with `L ≤ 0`
/// taken at the new time, and `flux_bottom`, `flux_top` enter the water at
/// the boundary layers' centres, replacing the diffusion across those
/// layers (GOTM `diff_face` with Neumann conditions).
#[allow(clippy::too_many_arguments)]
fn solve_interior(
    y: &mut [f64],
    h: &[f64],
    diff: impl Fn(usize) -> f64,
    source: impl Fn(usize) -> (f64, f64),
    flux_bottom: f64,
    flux_top: f64,
    dt: f64,
    s: &mut Solver<'_>,
) {
    let nw = y.len();
    let m = nw - 2;
    for j in 1..nw - 1 {
        let row = j - 1;
        let volume = 0.5 * (h[j - 1] + h[j]);
        let lower = if j == 1 {
            0.0
        } else {
            dt * 0.5 * (diff(j) + diff(j - 1)) / h[j - 1] / volume
        };
        let upper = if j == nw - 2 {
            0.0
        } else {
            dt * 0.5 * (diff(j) + diff(j + 1)) / h[j] / volume
        };
        let (q, l) = source(j);
        s.a[row] = -lower;
        s.c[row] = -upper;
        s.b[row] = 1.0 + lower + upper - dt * l;
        s.d[row] = y[j] + dt * q;
        if j == 1 {
            s.d[row] += dt * flux_bottom / volume;
        }
        if j == nw - 2 {
            s.d[row] += dt * flux_top / volume;
        }
    }
    solve_tridiagonal(
        &s.a[..m],
        &s.b[..m],
        &s.c[..m],
        &s.d[..m],
        &mut s.x[..m],
        &mut s.c_prime[..m],
        &mut s.d_prime[..m],
    );
    y[1..nw - 1].copy_from_slice(&s.x[..m]);
}

/// Patankar's split of a source `prod + buoy − diss` of `Y` into `(Q, L)`
/// with `Q ≥ 0` explicit and `L·Y` (`L ≤ 0`) implicit: a negative `buoy` is a
/// sink when the sum of the productions is negative.
#[inline]
fn patankar(prod: f64, buoy: f64, diss: f64, y_old: f64) -> (f64, f64) {
    if prod + buoy > 0.0 {
        (prod + buoy, -diss / y_old)
    } else {
        (prod, -(diss - buoy) / y_old)
    }
}

impl VerticalMixing for GlsMixing {
    /// The mixing of the turbulence at its minimum (there is no state to
    /// step here).
    fn compute_mixing_into(
        &self,
        _column: &Column,
        _forcing: &Forcing,
        av: &mut [f64],
        kt: &mut [f64],
    ) {
        let (num, nuh) = self.coefficients(self.k_min, self.eps_min, 0.0);
        av.fill(num + self.background_viscosity);
        kt.fill(nuh + self.background_diffusivity);
    }

    /// `k = k_min`, `ε = ε_min`.
    fn initial_turbulence(&self) -> Option<[f64; 2]> {
        Some([self.k_min, self.psi(self.k_min, self.eps_min)])
    }

    fn step_mixing_into(
        &self,
        column: &Column,
        _forcing: &Forcing,
        dt: f64,
        turbulence: Turbulence<'_>,
        av: &mut [f64],
        kt: &mut [f64],
    ) {
        let Turbulence {
            tke,
            gls: psi,
            scratch,
        } = turbulence;
        let nl = column.u.len();
        let nw = nl + 1;
        assert!(
            tke.len() == nw && psi.len() == nw,
            "GLS turbulence must be at the {nw} w-points, got {} and {}",
            tke.len(),
            psi.len()
        );
        let (f, mut solver) = work(scratch, nw);
        let GlsParameters {
            sigma_k,
            sigma_psi,
            c1,
            c2,
            c3_plus,
            ..
        } = self.params;
        let cm0 = self.stability.cm0;

        for l in 0..nl {
            f.h[l] = column.z_w[l + 1] - column.z_w[l];
        }
        // Shear and stratification at the interior w-points, held to the ends
        let buoyancy_scale = column.g / column.rho0;
        for j in 1..nl {
            let dz = column.z_r[j] - column.z_r[j - 1];
            let (du, dv) = (column.u[j] - column.u[j - 1], column.v[j] - column.v[j - 1]);
            f.s2[j] = (du * du + dv * dv) / (dz * dz);
            f.n2[j] = -buoyancy_scale * (column.rho[j] - column.rho[j - 1]) / dz;
        }
        if nl >= 2 {
            (f.s2[0], f.n2[0]) = (f.s2[1], f.n2[1]);
            (f.s2[nl], f.n2[nl]) = (f.s2[nl - 1], f.n2[nl - 1]);
        } else {
            f.s2.fill(0.0);
            f.n2.fill(0.0);
        }

        if nl >= 2 {
            // P, B, ε of the old turbulence
            for j in 1..nl {
                let eps = self.dissipation(tke[j], psi[j]);
                let (num, nuh) = self.coefficients(tke[j], eps, f.n2[j]);
                f.eps[j] = eps;
                f.num[j] = num;
                f.production[j] = num * f.s2[j];
                f.buoyancy[j] = -nuh * f.n2[j];
            }
            f.k_old.copy_from_slice(tke);
            f.psi_old.copy_from_slice(psi);
            let Fields {
                h,
                n2,
                num,
                production,
                buoyancy,
                eps,
                k_old,
                psi_old,
                ..
            } = &f;

            // k, with no flux through the log layers
            solve_interior(
                tke,
                h,
                |j| num[j] / sigma_k,
                |j| patankar(production[j], buoyancy[j], eps[j], k_old[j]),
                0.0,
                0.0,
                dt,
                &mut solver,
            );
            for k in &mut tke[1..nl] {
                *k = k.max(self.k_min);
            }

            // ψ, with the log layers' fluxes at the end layers' centres
            let flux_bottom = self.wall_psi_flux(tke[1], 0.5 * h[0] + self.z0_bottom);
            let flux_top = self.wall_psi_flux(tke[nl - 1], 0.5 * h[nl - 1] + self.z0_surface);
            solve_interior(
                psi,
                h,
                |j| num[j] / sigma_psi,
                |j| {
                    let ratio = psi_old[j] / k_old[j];
                    let c3 = if buoyancy[j] > 0.0 {
                        c3_plus
                    } else {
                        self.c3_minus
                    };
                    patankar(
                        c1 * ratio * production[j],
                        c3 * ratio * buoyancy[j],
                        c2 * ratio * eps[j],
                        psi_old[j],
                    )
                },
                flux_bottom,
                flux_top,
                dt,
                &mut solver,
            );

            // ε bounded, and ψ consistent with it
            for j in 1..nl {
                let eps = self.bounded_dissipation(tke[j], self.dissipation(tke[j], psi[j]), n2[j]);
                psi[j] = self.psi(tke[j], eps);
            }
        }

        // The log layers at the boundary w-points
        tke[0] = (column.bottom_friction_velocity.powi(2) / (cm0 * cm0)).max(self.k_min);
        tke[nl] = (column.surface_friction_velocity.powi(2) / (cm0 * cm0)).max(self.k_min);
        psi[0] = self.wall_psi(tke[0], self.z0_bottom);
        psi[nl] = self.wall_psi(tke[nl], self.z0_surface);

        for j in 0..nw {
            let eps = self.bounded_dissipation(tke[j], self.dissipation(tke[j], psi[j]), f.n2[j]);
            let (num, nuh) = self.coefficients(tke[j], eps, f.n2[j]);
            av[j] = num + self.background_viscosity;
            kt[j] = nuh + self.background_diffusivity;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A column of `nl` uniform layers over `depth` with uniform shear `s`
    /// and `N² = ri·s²`, its w-point turbulence at `k`, `ε`.
    struct Homogeneous {
        z_r: Vec<f64>,
        z_w: Vec<f64>,
        u: Vec<f64>,
        v: Vec<f64>,
        rho: Vec<f64>,
        tke: Vec<f64>,
        gls: Vec<f64>,
    }

    const G: f64 = 9.81;
    const RHO0: f64 = 1025.0;

    impl Homogeneous {
        fn new(
            mixing: &GlsMixing,
            nl: usize,
            depth: f64,
            s: f64,
            ri: f64,
            k: f64,
            eps: f64,
        ) -> Self {
            let dz = depth / nl as f64;
            let z_w: Vec<f64> = (0..=nl).map(|j| -depth + j as f64 * dz).collect();
            let z_r: Vec<f64> = (0..nl).map(|l| -depth + (l as f64 + 0.5) * dz).collect();
            let n2 = ri * s * s;
            Self {
                u: z_r.iter().map(|z| s * z).collect(),
                v: vec![0.0; nl],
                rho: z_r.iter().map(|z| RHO0 * (1.0 - n2 / G * z)).collect(),
                tke: vec![k; nl + 1],
                gls: vec![mixing.psi(k, eps); nl + 1],
                z_r,
                z_w,
            }
        }

        fn step(&mut self, mixing: &GlsMixing, dt: f64, scratch: &mut Vec<f64>) -> Vec<f64> {
            let nw = self.z_w.len();
            let (mut av, mut kt) = (vec![0.0; nw], vec![0.0; nw]);
            mixing.step_mixing_into(
                &Column {
                    z_r: &self.z_r,
                    z_w: &self.z_w,
                    u: &self.u,
                    v: &self.v,
                    rho: &self.rho,
                    g: G,
                    rho0: RHO0,
                    surface_friction_velocity: 0.0,
                    bottom_friction_velocity: 0.0,
                },
                &Forcing {
                    surface_stress: [0.0; 2],
                    bottom_stress: [0.0; 2],
                    surface_buoyancy_flux: 0.0,
                },
                dt,
                Turbulence {
                    tke: &mut self.tke,
                    gls: &mut self.gls,
                    scratch,
                },
                &mut av,
                &mut kt,
            );
            av
        }
    }

    /// Canuto et al.'s neutral stability function `c_μ⁰` (Umlauf & Burchard
    /// 2005, Table 3: 0.5270 for A, 0.5540 for B), equal to the stability
    /// function at `α_N = 0` so that a log layer has `k = u*²/(c_μ⁰)²`; and
    /// the derived constants of the presets (ROMS: c₃⁻ = 0.05 for the generic
    /// model; κ near 0.4).
    #[test]
    fn canuto_stability_functions_have_the_published_neutral_values() {
        for (functions, published) in [
            (StabilityFunctions::CanutoA, 0.5270),
            (StabilityFunctions::CanutoB, 0.5540),
        ] {
            let gls = GlsMixing::new(GlsParameters::K_EPSILON, functions);
            let (cm, cmp) = gls.stability_functions(0.0);
            assert!(
                (gls.cm0() - published).abs() < 1e-3,
                "{functions:?}: c_μ⁰ {} against {published}",
                gls.cm0()
            );
            assert!(
                (cm - gls.cm0()).abs() < 1e-12,
                "{functions:?}: c_μ(0) {cm} against c_μ⁰ {}",
                gls.cm0()
            );
            assert!(
                cmp > cm,
                "{functions:?}: neutral Prandtl number {} ≥ 1",
                cm / cmp
            );
            // Stable stratification damps the mixing, momentum less than heat
            let (cm_s, cmp_s) = gls.stability_functions(5.0);
            assert!(cm_s < cm && cmp_s < cmp && cm_s / cmp_s > cm / cmp);
        }
        let generic = GlsMixing::generic();
        assert!(
            (generic.c3_minus() - 0.05).abs() < 0.01,
            "generic c₃⁻ {}",
            generic.c3_minus()
        );
        for gls in [GlsMixing::k_epsilon(), GlsMixing::k_omega(), generic] {
            assert!((0.37..0.43).contains(&gls.kappa()), "κ {}", gls.kappa());
        }
    }

    /// `c₃⁻` makes homogeneous shear turbulence steady at the steady-state
    /// Richardson number: in uniform shear and stratification, `k` grows
    /// below `Ri_st` = 0.25 and decays above it, and at `Ri_st` its growth
    /// rate is three orders of magnitude smaller than either (≈ 10⁻⁵ S).
    #[test]
    fn homogeneous_turbulence_is_steady_at_the_steady_state_richardson_number() {
        let (s, dt) = (0.05, 2.0);
        for gls in [
            GlsMixing::k_epsilon(),
            GlsMixing::k_omega(),
            GlsMixing::generic(),
        ] {
            let gls = gls.with_length_limit(None).with_background(0.0, 0.0);
            let rate = |ri: f64| {
                let mut column = Homogeneous::new(&gls, 200, 200.0, s, ri, 1e-4, 1e-6);
                let mut scratch = Vec::new();
                let (mid, mut k_half) = (100, 0.0);
                let steps = (60.0 / s / dt) as usize;
                for n in 0..steps {
                    if n == steps / 2 {
                        k_half = column.tke[mid];
                    }
                    column.step(&gls, dt, &mut scratch);
                }
                (column.tke[mid] / k_half).ln() / (steps / 2) as f64 / dt / s
            };
            let (below, at, above) = (rate(0.2), rate(GlsMixing::DEFAULT_RI_ST), rate(0.3));
            let name = format!("{:?}", gls.parameters());
            assert!(below > 0.0 && above < 0.0, "{name}: rates {below}, {above}");
            assert!(
                at.abs() < 0.01 * below.min(-above),
                "{name}: growth rate {at} at Ri_st against {below}, {above}"
            );
        }
    }

    /// The Patankar sinks keep `k` and `ψ` positive and finite for any step,
    /// in strong stratification and in convection.
    #[test]
    fn turbulence_stays_positive_for_any_time_step() {
        for gls in [
            GlsMixing::k_epsilon(),
            GlsMixing::k_omega(),
            GlsMixing::generic(),
        ] {
            for ri in [-2.0, 0.0, 10.0] {
                let mut column = Homogeneous::new(&gls, 30, 30.0, 0.1, ri, 1e-3, 1e-5);
                let mut scratch = Vec::new();
                for dt in [1e6, 1.0, 1e4] {
                    let av = column.step(&gls, dt, &mut scratch);
                    assert!(
                        column
                            .tke
                            .iter()
                            .chain(&column.gls)
                            .chain(&av)
                            .all(|x| x.is_finite() && *x > 0.0),
                        "Ri {ri}, dt {dt}: k {:?}",
                        column.tke
                    );
                }
            }
        }
    }

    /// A single layer has no interior w-point: only the boundary log layers.
    #[test]
    fn a_single_layer_has_only_the_boundary_layers() {
        let gls = GlsMixing::k_epsilon();
        let mut column = Homogeneous::new(&gls, 1, 0.5, 0.0, 0.0, 1e-4, 1e-6);
        let av = column.step(&gls, 10.0, &mut Vec::new());
        assert_eq!(column.tke, vec![GlsMixing::DEFAULT_K_MIN; 2]);
        assert!(av.iter().all(|x| x.is_finite() && *x > 0.0));
    }
}
