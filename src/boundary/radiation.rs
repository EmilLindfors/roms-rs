//! Radiation (absorbing) boundary condition for shallow water equations.
//!
//! Radiation boundaries allow waves to exit the domain without reflection.
//! This is achieved by using characteristic-based conditions that ensure
//! outgoing characteristics carry information from the interior, while
//! incoming characteristics are set to a reference state.
//!
//! Common formulations:
//! - Sommerfeld radiation: ∂u/∂t + c ∂u/∂n = 0
//! - Flather radiation: combines elevation and velocity conditions
//! - Orlanski radiation: uses local phase speed estimation

use super::{BCContext, SWEBoundaryCondition};
use crate::solver::SWEState;

/// Simple radiation (absorbing) boundary condition.
///
/// Uses a characteristic-based approach where:
/// - Outgoing characteristics use interior values
/// - Incoming characteristics use reference state values
///
/// This BC is approximate but simple and effective for many applications.
#[derive(Clone, Debug)]
pub struct RadiationBC {
    /// Reference water depth (typically undisturbed depth)
    pub h_ref: f64,
    /// Reference velocity (typically zero)
    pub u_ref: f64,
    /// Gravitational acceleration
    pub g: f64,
    /// Minimum depth
    pub h_min: f64,
}

impl RadiationBC {
    /// Create a new radiation BC with given reference depth.
    pub fn new(h_ref: f64, g: f64) -> Self {
        Self {
            h_ref,
            u_ref: 0.0,
            g,
            h_min: 1e-6,
        }
    }

    /// Create with reference depth and velocity.
    pub fn with_reference_velocity(h_ref: f64, u_ref: f64, g: f64) -> Self {
        Self {
            h_ref,
            u_ref,
            g,
            h_min: 1e-6,
        }
    }

    /// Standard gravity (9.81 m/s²).
    pub fn standard(h_ref: f64) -> Self {
        Self::new(h_ref, 9.81)
    }
}

impl SWEBoundaryCondition for RadiationBC {
    fn ghost_state(&self, ctx: &BCContext) -> SWEState {
        let h_int = ctx.interior_state.h;
        let u_int = ctx.interior_state.velocity_simple(self.h_min);

        // Wave celerity
        let c_int = (self.g * h_int.max(self.h_min)).sqrt();
        let c_ref = (self.g * self.h_ref.max(self.h_min)).sqrt();

        // Characteristic variables (Riemann invariants for SWE)
        // R+ = u + 2c (right-going)
        // R- = u - 2c (left-going)

        // At right boundary (normal = +1):
        // - R+ is outgoing (use interior)
        // - R- is incoming (use reference)
        //
        // At left boundary (normal = -1):
        // - R- is outgoing (use interior)
        // - R+ is incoming (use reference)

        let (r_plus, r_minus) = if ctx.normal > 0.0 {
            // Right boundary
            let r_plus_out = u_int + 2.0 * c_int; // From interior
            let r_minus_in = self.u_ref - 2.0 * c_ref; // From exterior
            (r_plus_out, r_minus_in)
        } else {
            // Left boundary
            let r_plus_in = self.u_ref + 2.0 * c_ref; // From exterior
            let r_minus_out = u_int - 2.0 * c_int; // From interior
            (r_plus_in, r_minus_out)
        };

        // Reconstruct state from characteristic variables
        // u = (R+ + R-) / 2
        // c = (R+ - R-) / 4
        // h = c² / g
        let u_ghost = 0.5 * (r_plus + r_minus);
        let c_ghost = 0.25 * (r_plus - r_minus);
        let h_ghost = (c_ghost * c_ghost / self.g).max(0.0);

        SWEState::from_primitives(h_ghost, u_ghost)
    }

    fn name(&self) -> &'static str {
        "radiation"
    }
}

/// Flather radiation boundary condition.
///
/// A widely-used open boundary condition in ocean modeling that combines
/// information about both sea surface height and normal velocity.
///
/// η_b = η_ext + (u_int - u_ext) * sqrt(h/g)
///
/// This is particularly effective for tidal simulations where external
/// forcing (η_ext, u_ext) is known.
///
/// Reference: Flather (1976), "A tidal model of the north-west European
/// continental shelf"
#[derive(Clone, Debug)]
pub struct FlatherBC {
    /// External (reference) water surface elevation
    pub eta_ext: f64,
    /// External (reference) normal velocity
    pub u_ext: f64,
    /// Bathymetry at boundary (for computing depth from eta)
    pub bathymetry: f64,
    /// Gravitational acceleration
    pub g: f64,
    /// Minimum depth
    pub h_min: f64,
}

impl FlatherBC {
    /// Create a new Flather BC.
    pub fn new(eta_ext: f64, u_ext: f64, bathymetry: f64, g: f64) -> Self {
        Self {
            eta_ext,
            u_ext,
            bathymetry,
            g,
            h_min: 1e-6,
        }
    }

    /// Create with zero external velocity (pure elevation forcing).
    pub fn elevation_only(eta_ext: f64, bathymetry: f64, g: f64) -> Self {
        Self::new(eta_ext, 0.0, bathymetry, g)
    }

    /// Update the external elevation (for time-varying forcing).
    pub fn set_elevation(&mut self, eta_ext: f64) {
        self.eta_ext = eta_ext;
    }

    /// Update the external velocity.
    pub fn set_velocity(&mut self, u_ext: f64) {
        self.u_ext = u_ext;
    }
}

impl SWEBoundaryCondition for FlatherBC {
    fn ghost_state(&self, ctx: &BCContext) -> SWEState {
        // Use depth at boundary for wave speed
        let h_boundary = (self.eta_ext - self.bathymetry).max(self.h_min);

        // Flather condition modifies the boundary velocity
        // The sign depends on boundary orientation
        let u_ghost = if ctx.normal > 0.0 {
            // Right boundary: outflow is positive
            self.u_ext
                + (ctx.interior_surface_elevation() - self.eta_ext) * (self.g / h_boundary).sqrt()
        } else {
            // Left boundary: outflow is negative
            self.u_ext
                - (ctx.interior_surface_elevation() - self.eta_ext) * (self.g / h_boundary).sqrt()
        };

        // Depth from external elevation
        let h_ghost = (self.eta_ext - ctx.bathymetry).max(0.0);

        SWEState::from_primitives(h_ghost, u_ghost)
    }

    fn name(&self) -> &'static str {
        "flather"
    }
}

/// Sponge layer boundary treatment.
///
/// Rather than a sharp boundary condition, this applies a gradual
/// relaxation toward a reference state over a specified distance.
/// Typically used in conjunction with other BCs to reduce reflections.
///
/// u_damped = u + dt * γ(x) * (u_ref - u)
///
/// where γ(x) varies from 0 (interior) to some maximum at the boundary.
#[derive(Clone, Debug)]
pub struct SpongeLayer {
    /// Reference state to relax toward
    pub reference_state: SWEState,
    /// Width of sponge layer (in physical units)
    pub width: f64,
    /// Maximum relaxation coefficient (1/time)
    pub gamma_max: f64,
    /// Position where sponge starts (interior edge)
    pub start_position: f64,
    /// Whether sponge extends inward (true) or outward (false)
    pub inward: bool,
}

impl SpongeLayer {
    /// Create a new sponge layer.
    pub fn new(
        reference_state: SWEState,
        width: f64,
        gamma_max: f64,
        start_position: f64,
        inward: bool,
    ) -> Self {
        Self {
            reference_state,
            width,
            gamma_max,
            start_position,
            inward,
        }
    }

    /// Compute the damping coefficient at a given position.
    ///
    /// Returns 0 outside the sponge layer, increasing to gamma_max at the boundary.
    pub fn damping_coefficient(&self, position: f64) -> f64 {
        let dist = if self.inward {
            position - self.start_position
        } else {
            self.start_position - position
        };

        if dist < 0.0 || dist > self.width {
            return 0.0;
        }

        // Smooth ramp using cosine profile
        let xi = dist / self.width;
        self.gamma_max * 0.5 * (1.0 - (std::f64::consts::PI * xi).cos())
    }

    /// Apply sponge damping to a state.
    pub fn apply(&self, state: &SWEState, position: f64, dt: f64) -> SWEState {
        let gamma = self.damping_coefficient(position);
        if gamma < 1e-10 {
            return *state;
        }

        let factor = 1.0 / (1.0 + gamma * dt);
        let h_new = factor * (state.h + gamma * dt * self.reference_state.h);
        let hu_new = factor * (state.hu + gamma * dt * self.reference_state.hu);

        SWEState::new(h_new, hu_new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 10.0;
    const TOL: f64 = 1e-10;

    fn make_context(h: f64, hu: f64, bath: f64, normal: f64) -> BCContext {
        BCContext::new(0.0, 0.0, SWEState::new(h, hu), bath, normal)
    }

    #[test]
    fn test_radiation_still_water() {
        let h_ref = 2.0;
        let bc = RadiationBC::new(h_ref, G);

        // Still water at reference depth - should stay still
        let ctx = make_context(h_ref, 0.0, 0.0, 1.0);
        let ghost = bc.ghost_state(&ctx);

        assert!(
            (ghost.h - h_ref).abs() < 0.1,
            "Depth should be near reference"
        );
        assert!(ghost.hu.abs() < 0.1, "Should be nearly still");
    }

    #[test]
    fn test_radiation_outgoing_wave() {
        let h_ref = 2.0;
        let bc = RadiationBC::new(h_ref, G);

        // Outgoing wave (positive velocity at right boundary)
        let ctx = make_context(2.5, 5.0, 0.0, 1.0); // h=2.5, u=2
        let ghost = bc.ghost_state(&ctx);

        // Ghost state should allow wave to exit
        // Velocity should be similar to interior (outgoing characteristic preserved)
        let u_ghost = ghost.velocity_simple(1e-6);
        assert!(u_ghost > 0.0, "Outgoing velocity should be preserved");
    }

    #[test]
    fn test_radiation_symmetric() {
        let h_ref = 2.0;
        let bc = RadiationBC::new(h_ref, G);

        // Same perturbation, opposite boundaries
        let ctx_right = make_context(2.5, 2.5, 0.0, 1.0);
        let ctx_left = make_context(2.5, -2.5, 0.0, -1.0);

        let ghost_right = bc.ghost_state(&ctx_right);
        let ghost_left = bc.ghost_state(&ctx_left);

        // Depths should be similar
        assert!(
            (ghost_right.h - ghost_left.h).abs() < 0.5,
            "Depths should be similar for symmetric setup"
        );
    }

    #[test]
    fn test_flather_still_water() {
        let eta_ext = 2.0;
        let bath = 0.0;
        let bc = FlatherBC::elevation_only(eta_ext, bath, G);

        // Interior at same elevation, still
        let ctx = make_context(2.0, 0.0, bath, 1.0);
        let ghost = bc.ghost_state(&ctx);

        // Should be nearly still at reference elevation
        assert!((ghost.h - 2.0).abs() < 0.1);
        assert!(ghost.hu.abs() < 1.0);
    }

    #[test]
    fn test_flather_elevation_difference() {
        let eta_ext = 2.0;
        let bath = 0.0;
        let bc = FlatherBC::elevation_only(eta_ext, bath, G);

        // Interior higher than external
        let ctx = make_context(2.5, 0.0, bath, 1.0);
        let ghost = bc.ghost_state(&ctx);

        // Should induce outward flow (positive at right boundary)
        let u_ghost = ghost.velocity_simple(1e-6);
        assert!(u_ghost > 0.0, "Higher interior should cause outflow");
    }

    #[test]
    fn test_sponge_layer_outside() {
        let sponge = SpongeLayer::new(SWEState::new(2.0, 0.0), 1.0, 1.0, 0.0, true);

        // Outside sponge layer
        let gamma = sponge.damping_coefficient(-0.5);
        assert!(gamma.abs() < TOL);
    }

    #[test]
    fn test_sponge_layer_inside() {
        let sponge = SpongeLayer::new(SWEState::new(2.0, 0.0), 1.0, 1.0, 0.0, true);

        // At boundary (full damping)
        let gamma = sponge.damping_coefficient(1.0);
        assert!((gamma - 1.0).abs() < TOL);

        // Halfway (intermediate damping)
        let gamma_mid = sponge.damping_coefficient(0.5);
        assert!(gamma_mid > 0.0 && gamma_mid < 1.0);
    }

    #[test]
    fn test_sponge_apply() {
        let ref_state = SWEState::new(2.0, 0.0);
        let sponge = SpongeLayer::new(ref_state, 1.0, 10.0, 0.0, true);

        let state = SWEState::new(3.0, 6.0);
        let damped = sponge.apply(&state, 1.0, 0.1);

        // Should move toward reference
        assert!(damped.h < state.h && damped.h > ref_state.h);
        assert!(damped.hu.abs() < state.hu.abs());
    }
}
