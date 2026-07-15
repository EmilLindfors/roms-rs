//! Tidal-Subtidal Open Boundary Condition (TST-OBC).
//!
//! TST-OBC separates the boundary signal into tidal and subtidal components:
//! - **Tidal**: Prescribed from harmonic constituents (Flather-type)
//! - **Subtidal**: Radiates freely (Chapman-type)
//!
//! This allows the model to correctly handle:
//! - Known tidal forcing from harmonic analysis
//! - Slow variations (storm surge, seasonal) that should radiate out
//!
//! # Mathematical Formulation
//!
//! Surface elevation is decomposed:
//! ```text
//! η = η_tidal + η_subtidal
//!
//! η_tidal(t) = η₀ + Σᵢ Aᵢcos(ωᵢt + φᵢ)
//! η_subtidal = η_interior - η_tidal
//! ```
//!
//! Boundary conditions:
//! - Tidal: Flather with prescribed η_tidal and characteristic velocity
//! - Subtidal: Chapman radiation allowing slow variations to exit
//!
//! # Example
//!
//! ```ignore
//! use dg::boundary::{TSTConfig, TSTConstituent, TSTOBC2D};
//! use std::f64::consts::PI;
//!
//! // Define M2 and S2 constituents
//! let constituents = vec![
//!     TSTConstituent::new("M2".to_string(), 0.45, 2.0 * PI / 44714.0, 2.17),
//!     TSTConstituent::new("S2".to_string(), 0.15, 2.0 * PI / 43200.0, 2.77),
//! ];
//!
//! let config = TSTConfig {
//!     mean_elevation: 0.0,
//!     constituents,
//!     h_ref: 50.0,       // 50m reference depth
//!     dx: 800.0,         // Grid spacing for radiation
//!     subtidal_weight: 1.0,  // Full subtidal radiation
//!     h_min: 1e-6,
//! };
//!
//! let bc = TSTOBC2D::new(config);
//!
//! // Predict tidal elevation at any time
//! let eta = bc.predict_tidal_elevation(0.0);
//! ```

use crate::boundary::{BCContext2D, SWEBoundaryCondition2D};
use crate::io::ConstituentData;
use crate::solver::SWEState2D;
use std::f64::consts::PI;

/// A single tidal constituent for TST-OBC.
#[derive(Clone, Debug)]
pub struct TSTConstituent {
    /// Name of the constituent (e.g., "M2", "S2")
    pub name: String,
    /// Amplitude in meters
    pub amplitude: f64,
    /// Angular frequency in rad/s (ω = 2π/T)
    pub omega: f64,
    /// Phase in radians
    pub phase: f64,
}

impl TSTConstituent {
    /// Create a new tidal constituent.
    ///
    /// # Arguments
    /// * `name` - Constituent name (e.g., "M2")
    /// * `amplitude` - Amplitude in meters
    /// * `omega` - Angular frequency in rad/s
    /// * `phase` - Phase in radians
    pub fn new(name: String, amplitude: f64, omega: f64, phase: f64) -> Self {
        Self {
            name,
            amplitude,
            omega,
            phase,
        }
    }

    /// Create from amplitude and Greenwich phase **lag** in degrees.
    ///
    /// `phase_degrees` is the Greenwich phase lag G, the standard tidal-harmonic
    /// convention in which the physical elevation is `A·cos(ωt − G)`. The internal
    /// phase used by [`evaluate`](Self::evaluate) (which adds, `cos(ωt + φ)`) is
    /// therefore the negated lag, `φ = −G`. Negating here rather than adding the
    /// lag is what makes the tide advance in time instead of running backwards.
    pub fn from_degrees(name: String, amplitude: f64, omega: f64, phase_degrees: f64) -> Self {
        Self {
            name,
            amplitude,
            omega,
            phase: -phase_degrees * PI / 180.0,
        }
    }

    /// Evaluate the constituent at time t.
    ///
    /// Returns: A * cos(ωt + φ), where φ is the internal phase (the negated
    /// Greenwich phase lag when constructed via [`from_degrees`](Self::from_degrees)).
    pub fn evaluate(&self, t: f64) -> f64 {
        self.amplitude * (self.omega * t + self.phase).cos()
    }

    /// Evaluate the time derivative at time t.
    ///
    /// Returns: -Aω * sin(ωt + φ)
    pub fn evaluate_rate(&self, t: f64) -> f64 {
        -self.amplitude * self.omega * (self.omega * t + self.phase).sin()
    }
}

/// Configuration for TST-OBC.
#[derive(Clone, Debug)]
pub struct TSTConfig {
    /// Mean sea level elevation (η₀)
    pub mean_elevation: f64,
    /// Tidal constituents
    pub constituents: Vec<TSTConstituent>,
    /// Reference depth below mean sea level
    pub h_ref: f64,
    /// Grid spacing for radiation term (dx)
    pub dx: f64,
    /// Subtidal radiation weight (0-1)
    /// - 0: Pure tidal (no subtidal radiation)
    /// - 1: Full subtidal radiation
    pub subtidal_weight: f64,
    /// Minimum depth threshold
    pub h_min: f64,
}

impl TSTConfig {
    /// Create a TST configuration from constituent file data.
    ///
    /// # Arguments
    /// * `data` - Parsed constituent data from file
    /// * `h_ref` - Reference depth below mean sea level
    /// * `dx` - Grid spacing for radiation term
    pub fn from_constituent_data(data: &ConstituentData, h_ref: f64, dx: f64) -> Self {
        let constituents = data
            .constituents
            .iter()
            .map(|c| TSTConstituent {
                name: c.name.clone(),
                amplitude: c.amplitude,
                omega: 2.0 * PI / c.period,
                // c.phase_degrees is the Greenwich phase lag G; the internal phase
                // is the negated lag so that elevation is A·cos(ωt − G). See
                // `TSTConstituent::from_degrees`.
                phase: -c.phase_degrees * PI / 180.0,
            })
            .collect();

        Self {
            mean_elevation: data.reference_level,
            constituents,
            h_ref,
            dx,
            subtidal_weight: 1.0,
            h_min: 1e-6,
        }
    }

    /// Create a TST configuration from constituent file data, applying nodal
    /// corrections and equilibrium arguments at a prediction epoch.
    ///
    /// This is the astronomically complete counterpart to
    /// [`from_constituent_data`](Self::from_constituent_data): each constituent's
    /// amplitude is scaled by its nodal factor `f`, and its internal phase gains
    /// the equilibrium argument plus nodal phase `(V₀ + u)`, so the prediction is
    ///
    /// ```text
    /// η(t) = Σ fᵢ Aᵢ cos(ωᵢ t + (V₀ + u)ᵢ − Gᵢ)
    /// ```
    ///
    /// where `t` is elapsed simulation time (seconds) measured from `epoch_jd`.
    /// Constituents whose names are not recognised by
    /// [`nodal_correction`](crate::tides::nodal_correction) are carried through
    /// uncorrected (identity `f = 1`, `V₀ = u = 0`).
    ///
    /// # Arguments
    /// * `data` - Parsed constituent data (amplitude, Greenwich phase lag `G`)
    /// * `h_ref` - Reference depth below mean sea level
    /// * `dx` - Grid spacing for the radiation term
    /// * `epoch_jd` - Julian Date (UTC) of simulation time `t = 0`
    ///   (see [`julian_date`](crate::tides::julian_date))
    pub fn from_constituent_data_at_epoch(
        data: &ConstituentData,
        h_ref: f64,
        dx: f64,
        epoch_jd: f64,
    ) -> Self {
        use crate::tides::{AstronomicalArguments, NodalCorrection, nodal_correction};

        let astro = AstronomicalArguments::at_julian_date(epoch_jd);
        let constituents = data
            .constituents
            .iter()
            .map(|c| {
                let correction =
                    nodal_correction(&c.name, &astro).unwrap_or(NodalCorrection::IDENTITY);
                // Internal phase φ = −G; add (V₀ + u), scale amplitude by f.
                TSTConstituent {
                    name: c.name.clone(),
                    amplitude: correction.f * c.amplitude,
                    omega: 2.0 * PI / c.period,
                    phase: -c.phase_degrees * PI / 180.0 + correction.phase_offset_rad(),
                }
            })
            .collect();

        Self {
            mean_elevation: data.reference_level,
            constituents,
            h_ref,
            dx,
            subtidal_weight: 1.0,
            h_min: 1e-6,
        }
    }

    /// Builder: Set subtidal radiation weight.
    pub fn with_subtidal_weight(mut self, weight: f64) -> Self {
        self.subtidal_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Builder: Set minimum depth.
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }
}

/// Tidal-Subtidal Open Boundary Condition.
///
/// Separates boundary forcing into:
/// - Tidal: Prescribed from harmonic constituents (Flather-type)
/// - Subtidal: Radiates freely (Chapman-type)
#[derive(Clone, Debug)]
pub struct TSTOBC2D {
    config: TSTConfig,
}

impl TSTOBC2D {
    /// Create a new TST-OBC from configuration.
    pub fn new(config: TSTConfig) -> Self {
        Self { config }
    }

    /// Predict tidal surface elevation at time t.
    ///
    /// η_tidal(t) = η₀ + Σᵢ Aᵢcos(ωᵢt + φᵢ)
    pub fn predict_tidal_elevation(&self, t: f64) -> f64 {
        let mut eta = self.config.mean_elevation;
        for constituent in &self.config.constituents {
            eta += constituent.evaluate(t);
        }
        eta
    }

    /// Predict tidal elevation rate of change at time t.
    ///
    /// dη_tidal/dt = Σᵢ -Aᵢωᵢsin(ωᵢt + φᵢ)
    pub fn predict_tidal_rate(&self, t: f64) -> f64 {
        let mut rate = 0.0;
        for constituent in &self.config.constituents {
            rate += constituent.evaluate_rate(t);
        }
        rate
    }

    /// Get tidal water depth at time t with explicit bathymetry.
    ///
    /// h_tidal = η_tidal - bathymetry
    ///
    /// where bathymetry is the bed elevation (negative below MSL).
    pub fn tidal_depth_with_bathy(&self, t: f64, bathymetry: f64) -> f64 {
        (self.predict_tidal_elevation(t) - bathymetry).max(self.config.h_min)
    }

    /// Get tidal water depth at time t using h_ref (deprecated).
    ///
    /// h_tidal = h_ref + η_tidal
    ///
    /// **Deprecated**: Use [`tidal_depth_with_bathy`] instead. This method uses
    /// `h_ref` as a proxy for depth, which double-counts when bathymetry is set.
    #[deprecated(
        note = "Use tidal_depth_with_bathy instead; h_ref is ignored in ghost_state depth computation"
    )]
    pub fn tidal_depth(&self, t: f64) -> f64 {
        (self.config.h_ref + self.predict_tidal_elevation(t)).max(self.config.h_min)
    }

    /// Get the configuration.
    pub fn config(&self) -> &TSTConfig {
        &self.config
    }
}

impl SWEBoundaryCondition2D for TSTOBC2D {
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        let t = ctx.time;
        let g = ctx.g;
        let (nx, ny) = ctx.normal;

        // 1. Predict tidal elevation and corresponding depth
        let eta_tidal = self.predict_tidal_elevation(t);
        let h_tidal = (eta_tidal - ctx.bathymetry).max(self.config.h_min);

        // 2. Interior state
        let h_int = ctx.interior_state.h;
        let eta_int = ctx.interior_surface_elevation();

        // 3. Compute subtidal residual
        // η_subtidal = η_interior - η_tidal
        let eta_subtidal = eta_int - eta_tidal;

        // 4. Wave celerities
        let c_tidal = (g * h_tidal).sqrt();
        let c_int = (g * h_int.max(self.config.h_min)).sqrt();

        // 5. Tidal component: Flather relation
        // u_n_tidal = c * (η_int - η_tidal) / h_tidal
        // This gives zero when interior matches tidal prediction
        let un_tidal = c_tidal * (eta_int - eta_tidal) / h_tidal;

        // 6. Subtidal component: Chapman radiation
        // The Chapman condition radiates subtidal residuals
        // ∂η_subtidal/∂t + c * ∂η_subtidal/∂n = 0
        //
        // For outgoing radiation, positive subtidal elevation should
        // produce outward flow (positive normal velocity) to carry
        // the perturbation out of the domain.
        //
        // u_n_subtidal = c * η_subtidal / dx (radiation velocity)
        let un_subtidal = c_int * eta_subtidal / self.config.dx;

        // 7. Blend tidal and subtidal velocities
        let w = self.config.subtidal_weight;
        let un_ghost = un_tidal + w * un_subtidal;

        // 8. Preserve tangential velocity from interior
        let ut_ghost = ctx.interior_tangential_velocity();

        // 9. Convert (un, ut) back to (u, v) in Cartesian coordinates
        // u = un * nx - ut * ny
        // v = un * ny + ut * nx
        let u_ghost = un_ghost * nx - ut_ghost * ny;
        let v_ghost = un_ghost * ny + ut_ghost * nx;

        // 10. Use tidal depth for ghost state
        // The subtidal adjustment affects velocity, not depth
        SWEState2D::from_primitives(h_tidal, u_ghost, v_ghost)
    }

    fn name(&self) -> &'static str {
        "tst_obc_2d"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;
    const G: f64 = 9.81;
    const H_MIN: f64 = 1e-6;

    // M2 period in seconds (12.42 hours)
    const M2_PERIOD: f64 = 44714.0;
    const M2_OMEGA: f64 = 2.0 * PI / M2_PERIOD;

    fn make_context(h: f64, hu: f64, hv: f64, bathymetry: f64, time: f64) -> BCContext2D {
        BCContext2D::new(
            time,
            (0.0, 0.0),
            SWEState2D::new(h, hu, hv),
            bathymetry,
            (1.0, 0.0), // Normal pointing in +x
            G,
            H_MIN,
        )
    }

    #[test]
    fn test_tst_constituent_evaluate() {
        let m2 = TSTConstituent::new("M2".to_string(), 0.5, M2_OMEGA, 0.0);

        // At t=0, cos(0) = 1
        assert!((m2.evaluate(0.0) - 0.5).abs() < TOL);

        // At t=T/4, cos(π/2) = 0
        let t_quarter = M2_PERIOD / 4.0;
        assert!(m2.evaluate(t_quarter).abs() < 1e-10);

        // At t=T/2, cos(π) = -1
        let t_half = M2_PERIOD / 2.0;
        assert!((m2.evaluate(t_half) - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_tst_constituent_rate() {
        let m2 = TSTConstituent::new("M2".to_string(), 0.5, M2_OMEGA, 0.0);

        // At t=0, -sin(0) = 0
        assert!(m2.evaluate_rate(0.0).abs() < TOL);

        // At t=T/4, -sin(π/2) = -1
        let t_quarter = M2_PERIOD / 4.0;
        let expected_rate = -0.5 * M2_OMEGA;
        assert!((m2.evaluate_rate(t_quarter) - expected_rate).abs() < 1e-10);
    }

    #[test]
    fn test_tst_constituent_from_degrees() {
        // Regression (TODO P0.9): a 90° Greenwich phase lag G gives internal
        // phase φ = −G, so η = 0.5·cos(ωt − π/2) = 0.5·sin(ωt).
        let m2 = TSTConstituent::from_degrees("M2".to_string(), 0.5, M2_OMEGA, 90.0);

        // At t=0, cos(−π/2) = 0 (sign-agnostic).
        assert!(m2.evaluate(0.0).abs() < 1e-10);

        // The tide peaks a quarter period *after* t=0 (t = G/ω = T/4), reaching
        // +A. The old, mirrored convention cos(ωt + π/2) would give −A here.
        let t_quarter = M2_PERIOD / 4.0;
        assert!(
            (m2.evaluate(t_quarter) - 0.5).abs() < 1e-10,
            "expected peak +0.5 at T/4, got {} (mirrored tide?)",
            m2.evaluate(t_quarter)
        );
    }

    #[test]
    fn test_predict_tidal_elevation() {
        let config = TSTConfig {
            mean_elevation: 0.5,
            constituents: vec![
                TSTConstituent::new("M2".to_string(), 0.3, M2_OMEGA, 0.0),
                TSTConstituent::new("S2".to_string(), 0.1, 2.0 * PI / 43200.0, PI / 4.0),
            ],
            h_ref: 50.0,
            dx: 800.0,
            subtidal_weight: 1.0,
            h_min: 1e-6,
        };
        let bc = TSTOBC2D::new(config);

        // At t=0
        // M2: 0.3 * cos(0) = 0.3
        // S2: 0.1 * cos(π/4) = 0.1 * 0.7071...
        let expected = 0.5 + 0.3 + 0.1 * (PI / 4.0).cos();
        assert!((bc.predict_tidal_elevation(0.0) - expected).abs() < TOL);
    }

    #[test]
    fn test_predict_tidal_rate() {
        let config = TSTConfig {
            mean_elevation: 0.0,
            constituents: vec![TSTConstituent::new("M2".to_string(), 0.5, M2_OMEGA, 0.0)],
            h_ref: 50.0,
            dx: 800.0,
            subtidal_weight: 1.0,
            h_min: 1e-6,
        };
        let bc = TSTOBC2D::new(config);

        // At t=0, rate = -A*ω*sin(0) = 0
        assert!(bc.predict_tidal_rate(0.0).abs() < TOL);

        // At t=T/4, rate = -A*ω*sin(π/2) = -A*ω
        let t_quarter = M2_PERIOD / 4.0;
        let expected = -0.5 * M2_OMEGA;
        assert!((bc.predict_tidal_rate(t_quarter) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_tst_obc_pure_tidal() {
        // When subtidal_weight = 0, should behave like Flather
        let config = TSTConfig {
            mean_elevation: 0.0,
            constituents: vec![TSTConstituent::new("M2".to_string(), 0.5, M2_OMEGA, 0.0)],
            h_ref: 50.0,
            dx: 800.0,
            subtidal_weight: 0.0, // No subtidal radiation
            h_min: 1e-6,
        };
        let bc = TSTOBC2D::new(config);

        // Interior matches tidal prediction at t=0
        // η_tidal(0) = mean_elevation + A*cos(0) = 0 + 0.5 = 0.5
        //
        // With bathymetry = -50 (bed 50m below MSL):
        //   h_tidal = η_tidal - bathymetry = 0.5 - (-50) = 50.5 ✓
        //   η_int = h_int + B = h_int + (-50) = h_int - 50
        //   For zero velocity: η_int = η_tidal = 0.5 → h_int = 50.5
        let h_int = 50.5; // h such that η = h + B = 50.5 - 50 = 0.5 = η_tidal
        let ctx = make_context(h_int, 0.0, 0.0, -50.0, 0.0);

        let ghost = bc.ghost_state(&ctx);

        // Ghost depth = h_tidal = η_tidal - B = 0.5 - (-50) = 50.5
        assert!((ghost.h - 50.5).abs() < 0.01);

        // Velocity should be near zero when interior elevation matches tidal
        // un_tidal = c * (η_int - η_tidal) / h_tidal = c * (0.5 - 0.5) / 50.5 = 0
        assert!(ghost.hu.abs() < 1e-10);
    }

    #[test]
    fn test_tst_obc_subtidal_radiation() {
        let config = TSTConfig {
            mean_elevation: 0.0,
            constituents: vec![], // No tidal constituents
            h_ref: 50.0,
            dx: 800.0,
            subtidal_weight: 1.0, // Full subtidal radiation
            h_min: 1e-6,
        };
        let bc = TSTOBC2D::new(config);

        // Interior has elevated water (storm surge scenario)
        // With bathymetry = -50 (bed 50m below MSL):
        //   η_tidal = 0, η_int = h_int + B = 51 + (-50) = 1.0m
        //   η_subtidal = η_int - η_tidal = 1.0 - 0.0 = 1.0m
        let h_int = 51.0; // 1m above MSL
        let ctx = make_context(h_int, 0.0, 0.0, -50.0, 0.0);

        let ghost = bc.ghost_state(&ctx);

        // Should generate outward velocity to radiate the surge
        // h_tidal = η_tidal - B = 0 - (-50) = 50
        // c = sqrt(9.81 * 51) ≈ 22.4 m/s
        // u_n = c * η_subtidal / dx ≈ 22.4 * 1.0 / 800 ≈ 0.028 m/s (outward)
        assert!(
            ghost.hu / ghost.h > 0.0,
            "Expected positive (outward) velocity"
        );
    }

    #[test]
    fn test_tst_obc_tangential_preserved() {
        let config = TSTConfig {
            mean_elevation: 0.0,
            constituents: vec![],
            h_ref: 50.0,
            dx: 800.0,
            subtidal_weight: 1.0,
            h_min: 1e-6,
        };
        let bc = TSTOBC2D::new(config);

        // Interior with tangential velocity, bathymetry = -50 (bed 50m below MSL)
        // η_int = h + B = 50 + (-50) = 0 = η_tidal (no constituents)
        let h = 50.0;
        let ctx = BCContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::from_primitives(h, 0.0, 2.0), // v = 2 m/s tangential
            -50.0,
            (1.0, 0.0),
            G,
            H_MIN,
        );

        let ghost = bc.ghost_state(&ctx);

        // Tangential velocity should be preserved
        assert!((ghost.hv / ghost.h - 2.0).abs() < TOL);
    }

    #[test]
    fn test_from_constituent_data() {
        use crate::io::ConstituentData;

        let data = ConstituentData {
            location: Some((5.32, 60.39)),
            reference_level: 0.1,
            constituents: vec![
                crate::io::ConstituentEntry {
                    name: "M2".to_string(),
                    amplitude: 0.45,
                    phase_degrees: 125.3,
                    period: M2_PERIOD,
                },
                crate::io::ConstituentEntry {
                    name: "S2".to_string(),
                    amplitude: 0.15,
                    phase_degrees: 158.7,
                    period: 43200.0,
                },
            ],
        };

        let config = TSTConfig::from_constituent_data(&data, 50.0, 800.0);

        assert!((config.mean_elevation - 0.1).abs() < TOL);
        assert_eq!(config.constituents.len(), 2);
        assert!((config.h_ref - 50.0).abs() < TOL);
        assert!((config.dx - 800.0).abs() < TOL);
        assert!((config.subtidal_weight - 1.0).abs() < TOL);

        // Check M2 constituent
        let m2 = &config.constituents[0];
        assert_eq!(m2.name, "M2");
        assert!((m2.amplitude - 0.45).abs() < TOL);
        assert!((m2.omega - 2.0 * PI / M2_PERIOD).abs() < 1e-10);
        // Internal phase is the negated Greenwich phase lag (φ = −G) so that
        // elevation is A·cos(ωt − G) rather than a time-reversed A·cos(ωt + G).
        assert!((m2.phase - (-125.3 * PI / 180.0)).abs() < 1e-10);
    }

    #[test]
    fn test_from_constituent_data_at_epoch_applies_nodal_correction() {
        use crate::io::ConstituentData;
        use crate::tides::{AstronomicalArguments, nodal_correction};

        // K1 carries the large diurnal nodal modulation (11–19%), so it is the
        // clearest witness that corrections were applied vs the uncorrected path.
        let data = ConstituentData {
            location: None,
            reference_level: 0.0,
            constituents: vec![crate::io::ConstituentEntry {
                name: "K1".to_string(),
                amplitude: 0.10,
                phase_degrees: 60.0,
                period: 23.9344697 * 3600.0,
            }],
        };

        // Epoch: 2024-01-01 00:00 UTC.
        let epoch_jd = crate::tides::julian_date(2024, 1, 1, 0, 0, 0.0);
        let astro = AstronomicalArguments::at_julian_date(epoch_jd);
        let corr = nodal_correction("K1", &astro).unwrap();

        let corrected = TSTConfig::from_constituent_data_at_epoch(&data, 50.0, 800.0, epoch_jd);
        let uncorrected = TSTConfig::from_constituent_data(&data, 50.0, 800.0);

        let c_corr = &corrected.constituents[0];
        let c_raw = &uncorrected.constituents[0];

        // Amplitude scaled by the nodal factor f (materially different from raw).
        assert!((c_corr.amplitude - corr.f * 0.10).abs() < 1e-12);
        assert!(
            (c_corr.amplitude - c_raw.amplitude).abs() > 1e-3,
            "nodal factor should shift K1 amplitude appreciably (f = {})",
            corr.f
        );

        // Internal phase = −G + (V₀ + u).
        let expected_phase = -60.0 * PI / 180.0 + corr.phase_offset_rad();
        assert!((c_corr.phase - expected_phase).abs() < 1e-12);
        // The equilibrium argument is a real, nonzero astronomical offset.
        assert!((c_corr.phase - c_raw.phase).abs() > 1e-3);

        // Frequency and mean level are untouched by the correction.
        assert!((c_corr.omega - c_raw.omega).abs() < 1e-15);
        assert!((corrected.mean_elevation - uncorrected.mean_elevation).abs() < TOL);
    }

    #[test]
    fn test_from_constituent_data_at_epoch_passes_through_unknown() {
        use crate::io::ConstituentData;

        // An unrecognised constituent name is carried through with identity
        // correction (period still drives omega).
        let data = ConstituentData {
            location: None,
            reference_level: 0.2,
            constituents: vec![crate::io::ConstituentEntry {
                name: "SA".to_string(), // solar annual — not in the nodal table
                amplitude: 0.05,
                phase_degrees: 30.0,
                period: 365.25 * 24.0 * 3600.0,
            }],
        };
        let epoch_jd = crate::tides::julian_date(2024, 1, 1, 0, 0, 0.0);
        let config = TSTConfig::from_constituent_data_at_epoch(&data, 50.0, 800.0, epoch_jd);
        let c = &config.constituents[0];
        assert!((c.amplitude - 0.05).abs() < 1e-12); // f = 1
        assert!((c.phase - (-30.0 * PI / 180.0)).abs() < 1e-12); // V0 = u = 0
    }

    #[test]
    fn test_config_builders() {
        let config = TSTConfig {
            mean_elevation: 0.0,
            constituents: vec![],
            h_ref: 50.0,
            dx: 800.0,
            subtidal_weight: 1.0,
            h_min: 1e-6,
        }
        .with_subtidal_weight(0.5)
        .with_h_min(1e-8);

        assert!((config.subtidal_weight - 0.5).abs() < TOL);
        assert!((config.h_min - 1e-8).abs() < TOL);
    }

    #[test]
    fn test_config_weight_clamping() {
        let config = TSTConfig {
            mean_elevation: 0.0,
            constituents: vec![],
            h_ref: 50.0,
            dx: 800.0,
            subtidal_weight: 0.5,
            h_min: 1e-6,
        }
        .with_subtidal_weight(2.0); // Should clamp to 1.0

        assert!((config.subtidal_weight - 1.0).abs() < TOL);

        let config2 = config.with_subtidal_weight(-0.5); // Should clamp to 0.0
        assert!(config2.subtidal_weight.abs() < TOL);
    }

    #[test]
    fn test_tidal_depth_with_bathy() {
        let config = TSTConfig {
            mean_elevation: 0.0,
            constituents: vec![TSTConstituent::new("M2".to_string(), 0.5, M2_OMEGA, 0.0)],
            h_ref: 50.0,
            dx: 800.0,
            subtidal_weight: 1.0,
            h_min: 1e-6,
        };
        let bc = TSTOBC2D::new(config);

        // With bathymetry = -50 (bed 50m below MSL):
        // At t=0, η = 0.5, h = η - B = 0.5 - (-50) = 50.5
        assert!((bc.tidal_depth_with_bathy(0.0, -50.0) - 50.5).abs() < TOL);

        // At t=T/2, η = -0.5, h = -0.5 - (-50) = 49.5
        let t_half = M2_PERIOD / 2.0;
        assert!((bc.tidal_depth_with_bathy(t_half, -50.0) - 49.5).abs() < 1e-10);
    }

    #[test]
    #[allow(deprecated)]
    fn test_tidal_depth_deprecated() {
        // Verify deprecated method still works with h_ref
        let config = TSTConfig {
            mean_elevation: 0.0,
            constituents: vec![TSTConstituent::new("M2".to_string(), 0.5, M2_OMEGA, 0.0)],
            h_ref: 50.0,
            dx: 800.0,
            subtidal_weight: 1.0,
            h_min: 1e-6,
        };
        let bc = TSTOBC2D::new(config);

        // Deprecated: h = h_ref + η = 50 + 0.5 = 50.5
        assert!((bc.tidal_depth(0.0) - 50.5).abs() < TOL);
    }

    #[test]
    fn test_bc_name() {
        let config = TSTConfig {
            mean_elevation: 0.0,
            constituents: vec![],
            h_ref: 50.0,
            dx: 800.0,
            subtidal_weight: 1.0,
            h_min: 1e-6,
        };
        let bc = TSTOBC2D::new(config);

        assert_eq!(bc.name(), "tst_obc_2d");
    }
}
