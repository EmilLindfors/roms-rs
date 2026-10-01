//! Behaviour of 3D particles: swimming in response to what they sense
//! (TODO F.2, Stage B).
//!
//! A [`ParticleBehaviour3D`] gives each particle, at the start of every
//! step, a swimming speed `w_s` (m/s, positive up) that it holds over the
//! step on top of its own constant vertical speed (sinking), so
//!
//! ```text
//! dσ/dt = Ω/D + (w_p + w_s)/D,
//! ```
//!
//! from what it senses there ([`Surroundings`]: time, position, depth,
//! temperature and salinity of the field), its development (e.g. degree-days)
//! and a uniform random number from its own stream. The behaviour also sets
//! the rate of development and when a particle dies of age
//! ([`super::ParticleStatus::Dead`]). The decision is explicit, at the start
//! of the step, as in LADiM and Johnsen et al. (2014); a step shorter than
//! the time to swim through a cue's gradient (e.g. `K/w_s²` for the light
//! threshold under mixing `K`) resolves it.
//!
//! # Salmon lice
//!
//! [`SalmonLice`] is the larval behaviour of *Lepeophtheirus salmonis* in
//! the IMR lice model (Johnsen et al. 2014; the operational LADiM
//! `salmon_lice` IBM). A larva
//! - develops in degree-days, `dA/dt = max(T, T_min)/86 400 s`, from a
//!   nauplius to an infective copepodid at `A_c` and dies at `A_max`;
//! - swims up at `w_s` when the light at its depth, `E₀(t) e^{−kζ}` (ζ the
//!   depth below the surface, `E₀` the surface irradiance, a
//!   [`SurfaceLight`]), is at least its stage's threshold;
//! - swims down at `w_s` when the salinity is below its stage's threshold,
//!   drawn uniformly from `[S_lo, S_hi]` every step (LADiM) or fixed
//!   (Johnsen et al.). Low salinity wins over light;
//! - otherwise does not swim: only the flow and the random walk move it.
//!
//! Survival is a weight, `e^{−μ t}` with the constant mortality
//! `μ = 0.17 d⁻¹` (Stien et al. 2005), not a removal
//! ([`SalmonLice::survival`]).
//!
//! | parameter | [`SalmonLice::ladim`] | [`SalmonLice::johnsen_2014`] |
//! |---|---|---|
//! | `w_s` | 0.5 mm/s | 0.5 mm/s |
//! | `k` | 0.2 m⁻¹ | 0.2 m⁻¹ |
//! | light threshold, nauplius / copepodid (µmol photons m⁻² s⁻¹) | 0.01 / 0.01 | 0.39 / 2.06·10⁻⁵ |
//! | salinity threshold, nauplius / copepodid | U[30, 32] / U[20, 28] | 20 / 20 |
//! | `A_c`, `A_max` (degree-days) | 40, 170 | 50, 150 |
//! | `T_min` | 5 °C | 0 °C |
//!
//! # Surface light
//!
//! [`ClearSkyLight`] is the clear-sky irradiance of Skartveit & Olseth
//! (1988) as LADiM computes it: the solar height `h` from the declination
//! and the true solar time `15°·(UTC hours) + λ`,
//! `E₀ = E_max sin h / sin h_noon + E_tw` for `h ≥ 0` (`E_max` = 1500,
//! `E_tw` = 5.76 µmol photons m⁻² s⁻¹), and twilight falling log-linearly in
//! three 6° bands to 1.15·10⁻⁵ below −18°. LADiM truncates the time to the
//! hour; here it is continuous.
//!
//! # References
//!
//! - Johnsen, I. A., Fiksen, Ø., Sandvik, A. D. & Asplin, L. (2014).
//!   Vertical salmon lice behaviour as a response to environmental
//!   conditions and its influence on regional dispersion in a fjord system.
//!   *Aquacult. Environ. Interact.* 5, 127–141.
//! - Ådlandsvik, B. & Sævik, P. N. LADiM, the Lagrangian Advection and
//!   Diffusion Model, and its `salmon_lice` IBM
//!   (github.com/pnsaevik/ladim_plugins).
//! - Stien, A., Bjørn, P. A., Heuch, P. A. & Elston, D. A. (2005).
//!   Population dynamics of salmon lice *Lepeophtheirus salmonis* on
//!   Atlantic salmon and sea trout. *Mar. Ecol. Prog. Ser.* 290, 263–275.
//! - Skartveit, A. & Olseth, J. A. (1988). Varighetstabeller for timevis
//!   belysning mot 5 flater på 16 norske stasjoner. Meteorological report
//!   series 1988-7, University of Bergen.

use crate::io::datetime::{civil_from_days, days_from_civil};
use crate::time::ModelClock;

/// What a particle senses at the start of a step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surroundings {
    /// Simulation time (s).
    pub time: f64,
    /// Horizontal position (mesh coordinates).
    pub position: [f64; 2],
    /// Depth below the surface (m).
    pub depth: f64,
    /// Water depth of the column (m).
    pub column_depth: f64,
    /// Temperature (°C), if the field has it.
    pub temperature: Option<f64>,
    /// Salinity, if the field has it.
    pub salinity: Option<f64>,
}

/// How a 3D particle swims, develops and dies (see the module docs).
pub trait ParticleBehaviour3D: Sync {
    /// Whether the behaviour uses the surroundings. Without (`false`), the
    /// tracker samples nothing and draws no random number for it.
    fn senses(&self) -> bool {
        true
    }

    /// Swimming speed (m/s, positive up) that a particle at `development`
    /// holds over the next step, with `uniform` ∈ (0, 1) from the
    /// particle's own random stream.
    fn swimming_speed(&self, development: f64, surroundings: &Surroundings, uniform: f64) -> f64;

    /// Rate of development (units of development per second, e.g.
    /// degree-days per second) in `surroundings`; 0 by default.
    fn development_rate(&self, _surroundings: &Surroundings) -> f64 {
        0.0
    }

    /// Whether a particle at `development` and `age` (s) has died; never by
    /// default.
    fn expired(&self, _development: f64, _age: f64) -> bool {
        false
    }
}

/// No behaviour: the particle moves with the flow, its own vertical speed
/// and the random walk only.
#[derive(Clone, Copy, Debug, Default)]
pub struct Passive;

impl ParticleBehaviour3D for Passive {
    fn senses(&self) -> bool {
        false
    }

    fn swimming_speed(&self, _: f64, _: &Surroundings, _: f64) -> f64 {
        0.0
    }
}

/// Downwelling irradiance just below the surface.
pub trait SurfaceLight: Sync {
    /// Irradiance (µmol photons m⁻² s⁻¹) at simulation time `t` at the
    /// horizontal `position` (mesh coordinates).
    fn surface_irradiance(&self, t: f64, position: [f64; 2]) -> f64;
}

impl<F: Fn(f64, [f64; 2]) -> f64 + Sync> SurfaceLight for F {
    fn surface_irradiance(&self, t: f64, position: [f64; 2]) -> f64 {
        self(t, position)
    }
}

/// The same irradiance everywhere at all times (µmol photons m⁻² s⁻¹).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstantLight(pub f64);

impl SurfaceLight for ConstantLight {
    fn surface_irradiance(&self, _: f64, _: [f64; 2]) -> f64 {
        self.0
    }
}

/// Clear-sky surface light at one site (Skartveit & Olseth 1988, as in
/// LADiM; see the module docs). A domain of tens of kilometres shares the
/// site's sun: one degree of longitude shifts it by 4 minutes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClearSkyLight {
    clock: ModelClock,
    longitude: f64,
    latitude: f64,
}

/// Irradiance at noon under a clear sky (µmol photons m⁻² s⁻¹), less the
/// twilight value.
const CLEAR_SKY_MAX: f64 = 1500.0;
/// Irradiance at sunrise and sunset (µmol photons m⁻² s⁻¹).
const TWILIGHT: f64 = 5.76;
/// Irradiance at −6°, −12° and −18° of solar height (µmol photons m⁻² s⁻¹).
const CIVIL: f64 = 0.048;
const NAUTICAL: f64 = 1.15e-4;
const NIGHT: f64 = 1.15e-5;

impl ClearSkyLight {
    /// The light at `longitude`, `latitude` (degrees) on `clock`'s time.
    pub fn new(clock: ModelClock, longitude: f64, latitude: f64) -> Self {
        assert!(
            longitude.is_finite() && (-90.0..=90.0).contains(&latitude),
            "site ({longitude}, {latitude}) is not a longitude and latitude"
        );
        Self {
            clock,
            longitude,
            latitude,
        }
    }

    /// Sines of the solar height at simulation time `t` and at that day's
    /// noon.
    fn sine_heights(&self, t: f64) -> (f64, f64) {
        let unix = self.clock.unix(t);
        let days = (unix / 86_400.0).floor();
        let (year, _, _) = civil_from_days(days as i64);
        let yday = (days as i64 - days_from_civil(year, 1, 1) + 1) as f64;
        let hours = (unix - 86_400.0 * days) / 3600.0;
        let rad = std::f64::consts::PI / 180.0;
        // Declination (Skartveit & Olseth; LADiM's coefficients)
        let a1 = 0.9856 * rad;
        let sin_delta =
            0.3979 * (a1 * (yday - 80.0) + 1.9171 * rad * ((a1 * yday).sin() - 0.98112)).sin();
        let cos_delta = (1.0 - sin_delta * sin_delta).sqrt();
        let phi = self.latitude * rad;
        // True solar time, 0 with the sun in the north
        let solar_time = (15.0 * hours + self.longitude) * rad;
        let sin_h = sin_delta * phi.sin() - cos_delta * phi.cos() * solar_time.cos();
        let sin_noon = sin_delta * phi.sin() + cos_delta * phi.cos();
        (sin_h, sin_noon)
    }

    /// Solar height (degrees) at simulation time `t`.
    pub fn solar_height(&self, t: f64) -> f64 {
        self.sine_heights(t).0.clamp(-1.0, 1.0).asin().to_degrees()
    }

    /// Irradiance just below the surface (µmol photons m⁻² s⁻¹) at time
    /// `t`.
    pub fn irradiance(&self, t: f64) -> f64 {
        let (sin_h, sin_noon) = self.sine_heights(t);
        let h = sin_h.clamp(-1.0, 1.0).asin().to_degrees();
        // Linear in the solar height within each 6° band of twilight
        let band = |upper: f64, lower: f64, top: f64| (upper - lower) / 6.0 * (top + h) + lower;
        if h >= 0.0 {
            CLEAR_SKY_MAX * sin_h / sin_noon + TWILIGHT
        } else if h >= -6.0 {
            band(TWILIGHT, CIVIL, 6.0)
        } else if h >= -12.0 {
            band(CIVIL, NAUTICAL, 12.0)
        } else if h >= -18.0 {
            band(NAUTICAL, NIGHT, 18.0)
        } else {
            NIGHT
        }
    }
}

impl SurfaceLight for ClearSkyLight {
    fn surface_irradiance(&self, t: f64, _: [f64; 2]) -> f64 {
        self.irradiance(t)
    }
}

/// Larval stage of a salmon louse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiceStage {
    /// Nauplius I–II: not yet infective.
    Nauplius,
    /// Copepodid: infective until it dies.
    Copepodid,
}

/// Larval salmon lice (see the module docs): light-seeking, avoiding low
/// salinity, developing in degree-days.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SalmonLice<L> {
    /// Surface light.
    pub light: L,
    /// Light attenuation coefficient `k` (1/m).
    pub attenuation: f64,
    /// Swimming speed `w_s` (m/s), up or down.
    pub swimming_speed: f64,
    /// Light at depth (µmol photons m⁻² s⁻¹) from which nauplii and
    /// copepodids swim up.
    pub light_threshold: [f64; 2],
    /// Salinity below which nauplii and copepodids swim down: drawn
    /// uniformly from `[lo, hi]` every step (equal bounds: fixed).
    pub salinity_threshold: [[f64; 2]; 2],
    /// Development (degree-days) at which a nauplius becomes a copepodid.
    pub copepodid_at: f64,
    /// Development (degree-days) at which a larva dies.
    pub lifespan: f64,
    /// Lowest temperature (°C) counted in the degree-days.
    pub min_temperature: f64,
    /// Mortality (1/day) of the survival weight.
    pub mortality: f64,
}

impl<L: SurfaceLight> SalmonLice<L> {
    /// The operational IMR lice model's parameters (LADiM's `salmon_lice`
    /// IBM; see the module docs).
    pub fn ladim(light: L) -> Self {
        Self {
            light,
            attenuation: 0.2,
            swimming_speed: 5e-4,
            light_threshold: [0.01, 0.01],
            salinity_threshold: [[30.0, 32.0], [20.0, 28.0]],
            copepodid_at: 40.0,
            lifespan: 170.0,
            min_temperature: 5.0,
            mortality: 0.17,
        }
    }

    /// Johnsen et al.'s (2014) parameters (their "Light" experiment).
    pub fn johnsen_2014(light: L) -> Self {
        Self {
            light_threshold: [0.39, 2.06e-5],
            salinity_threshold: [[20.0, 20.0], [20.0, 20.0]],
            copepodid_at: 50.0,
            lifespan: 150.0,
            min_temperature: 0.0,
            ..Self::ladim(light)
        }
    }

    /// Stage of a larva at `development` (degree-days).
    pub fn stage(&self, development: f64) -> LiceStage {
        if development < self.copepodid_at {
            LiceStage::Nauplius
        } else {
            LiceStage::Copepodid
        }
    }

    /// Probability that a larva of `age` (s) is alive: `e^{−μ t}`.
    pub fn survival(&self, age: f64) -> f64 {
        (-self.mortality * age / 86_400.0).exp()
    }

    /// Light (µmol photons m⁻² s⁻¹) at `depth` below the surface.
    pub fn light_at(&self, t: f64, position: [f64; 2], depth: f64) -> f64 {
        self.light.surface_irradiance(t, position) * (-self.attenuation * depth.max(0.0)).exp()
    }
}

impl<L: SurfaceLight> ParticleBehaviour3D for SalmonLice<L> {
    fn swimming_speed(&self, development: f64, s: &Surroundings, uniform: f64) -> f64 {
        let stage = self.stage(development) as usize;
        if let Some(salinity) = s.salinity {
            let [lo, hi] = self.salinity_threshold[stage];
            if salinity < hi - uniform * (hi - lo) {
                return -self.swimming_speed;
            }
        }
        if self.light_at(s.time, s.position, s.depth) >= self.light_threshold[stage] {
            self.swimming_speed
        } else {
            0.0
        }
    }

    /// `max(T, T_min)` per day; nothing where the field has no temperature.
    fn development_rate(&self, s: &Surroundings) -> f64 {
        s.temperature
            .map_or(0.0, |t| t.max(self.min_temperature) / 86_400.0)
    }

    fn expired(&self, development: f64, _age: f64) -> bool {
        development >= self.lifespan
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surroundings(depth: f64, salinity: Option<f64>) -> Surroundings {
        Surroundings {
            time: 0.0,
            position: [0.0, 0.0],
            depth,
            column_depth: 50.0,
            temperature: Some(10.0),
            salinity,
        }
    }

    /// LADiM's `surface_light` (its Python code run at whole hours) at
    /// 60°N, 0°E on 2014-06-23, its own self-test case: noon, evening
    /// (sun up), late evening (civil twilight) and midnight (nautical); and
    /// the polar night in Tromsø at noon on 21 December (civil twilight).
    #[test]
    fn clear_sky_light_matches_ladim() {
        let clock = ModelClock::parse("2014-06-23T00:00:00Z").unwrap();
        let light = ClearSkyLight::new(clock, 0.0, 60.0);
        for (hour, expect) in [
            (12.0, 1505.76),
            (18.0, 649.136_924_252_445_5),
            (21.0, 43.412_938_470_471_91),
            (0.0, 0.043_553_061_949_000_62),
        ] {
            let e = light.irradiance(hour * 3600.0);
            assert!(
                (e - expect).abs() < 1e-9 * expect,
                "{hour} h: {e} against LADiM's {expect}"
            );
        }
        let tromso = ClearSkyLight::new(
            ModelClock::parse("2025-12-21T12:00:00Z").unwrap(),
            19.0,
            69.65,
        );
        let e = tromso.irradiance(0.0);
        assert!(
            (e - 1.866_675_921_347_331_6).abs() < 1e-9,
            "Tromsø at noon: {e}"
        );
        assert!((-6.0..0.0).contains(&tromso.solar_height(0.0)));
    }

    /// Up in the light, down in fresh water (fresh water wins), still in
    /// the dark; the stage switches the thresholds.
    #[test]
    fn lice_swim_up_to_light_and_down_from_fresh_water() {
        let lice = SalmonLice::johnsen_2014(ConstantLight(100.0));
        let w = lice.swimming_speed;
        // 100·e^{−0.2ζ} ≥ 0.39 above ζ = 27.7 m (nauplii); ≥ 2.06e-5 above
        // 77.0 m (copepodids)
        assert_eq!(
            lice.swimming_speed(0.0, &surroundings(27.0, Some(33.0)), 0.5),
            w
        );
        assert_eq!(
            lice.swimming_speed(0.0, &surroundings(28.5, Some(33.0)), 0.5),
            0.0
        );
        assert_eq!(
            lice.swimming_speed(60.0, &surroundings(28.5, Some(33.0)), 0.5),
            w
        );
        assert_eq!(
            lice.swimming_speed(60.0, &surroundings(77.5, None), 0.5),
            0.0
        );
        assert_eq!(
            lice.swimming_speed(0.0, &surroundings(1.0, Some(19.0)), 0.5),
            -w
        );
        // LADiM: a copepodid avoids S < 28 − 8U
        let ladim = SalmonLice::ladim(ConstantLight(100.0));
        let s = surroundings(1.0, Some(24.0));
        assert_eq!(ladim.swimming_speed(45.0, &s, 0.4), -w);
        assert_eq!(ladim.swimming_speed(45.0, &s, 0.6), w);
        assert_eq!(ladim.stage(39.9), LiceStage::Nauplius);
        assert_eq!(ladim.stage(40.0), LiceStage::Copepodid);
    }

    /// Degree-days with LADiM's 5 °C floor; death at the lifespan; the
    /// survival weight after one day.
    #[test]
    fn lice_develop_in_degree_days() {
        let lice = SalmonLice::ladim(ConstantLight(0.0));
        let mut s = surroundings(1.0, None);
        assert!((lice.development_rate(&s) * 86_400.0 - 10.0).abs() < 1e-12);
        s.temperature = Some(2.0);
        assert!((lice.development_rate(&s) * 86_400.0 - 5.0).abs() < 1e-12);
        s.temperature = None;
        assert_eq!(lice.development_rate(&s), 0.0);
        assert!(!lice.expired(169.9, 0.0) && lice.expired(170.0, 0.0));
        assert!((lice.survival(86_400.0) - (-0.17f64).exp()).abs() < 1e-15);
    }
}
