//! Bottom friction source terms for shallow water equations.
//!
//! Bottom friction dissipates momentum and is particularly important in
//! shallow coastal areas. The most common formulation is Manning's equation.
//!
//! Manning friction:
//! S_hu = -g * n² * |u| * u / h^{1/3}
//!
//! where n is the Manning coefficient (typical values 0.01-0.05).
//!
//! Friction becomes stiff in shallow water, so implicit or semi-implicit
//! treatment may be needed for stability.

use crate::solver::SWEState;
use crate::source::SourceTerm;

/// Manning bottom friction source term.
///
/// S = (0, -g n² |u| u h^{-1/3})
///
/// Manning coefficient n has units of s/m^{1/3} and depends on bed roughness:
/// - Smooth concrete: n ≈ 0.012
/// - Natural channels: n ≈ 0.03-0.05
/// - Floodplains with vegetation: n ≈ 0.1-0.15
#[derive(Clone, Debug)]
pub struct ManningFriction {
    /// Gravitational acceleration
    pub g: f64,
    /// Manning coefficient (s/m^{1/3})
    pub manning_n: f64,
    /// Minimum depth for friction calculation
    pub h_min: f64,
}

impl ManningFriction {
    /// Create a new Manning friction source term.
    pub fn new(g: f64, manning_n: f64) -> Self {
        Self {
            g,
            manning_n,
            h_min: 1e-6,
        }
    }

    /// Create with custom minimum depth.
    pub fn with_h_min(g: f64, manning_n: f64, h_min: f64) -> Self {
        Self {
            g,
            manning_n,
            h_min,
        }
    }

    /// Standard gravity (9.81 m/s²) with given Manning coefficient.
    pub fn standard(manning_n: f64) -> Self {
        Self::new(9.81, manning_n)
    }

    /// Compute the friction coefficient C_f.
    ///
    /// τ = ρ C_f |u| u, where C_f = g n² h^{-1/3}
    pub fn friction_coefficient(&self, h: f64) -> f64 {
        if h > self.h_min {
            self.g * self.manning_n * self.manning_n / h.powf(1.0 / 3.0)
        } else {
            // Limit friction coefficient for very shallow water
            self.g * self.manning_n * self.manning_n / self.h_min.powf(1.0 / 3.0)
        }
    }

    /// Compute friction source term explicitly.
    ///
    /// S_hu = -g n² |u| u / h^{1/3} = -C_f |u| u
    pub fn explicit_source(&self, state: &SWEState) -> SWEState {
        if state.h < self.h_min {
            return SWEState::zero();
        }

        let u = state.hu / state.h;
        let c_f = self.friction_coefficient(state.h);
        let s_hu = -c_f * u.abs() * u;

        SWEState::new(0.0, s_hu)
    }

    /// Semi-implicit friction update.
    ///
    /// For stiff friction (shallow water), explicit treatment may require
    /// very small time steps. Semi-implicit treatment is unconditionally
    /// stable for friction.
    ///
    /// Given: du/dt = -C_f |u| u / h
    /// Linearize: du/dt ≈ -C_f |u^n| u^{n+1} / h
    /// Solution: u^{n+1} = u^n / (1 + dt * C_f * |u^n| / h)
    ///
    /// # Arguments
    /// * `state` - Current state
    /// * `dt` - Time step
    ///
    /// # Returns
    /// Updated state after friction
    pub fn semi_implicit_update(&self, state: &SWEState, dt: f64) -> SWEState {
        if state.h < self.h_min {
            return *state;
        }

        let u = state.hu / state.h;
        let c_f = self.friction_coefficient(state.h);

        // Damping factor
        let denom = 1.0 + dt * c_f * u.abs() / state.h;
        let u_new = u / denom;

        SWEState::new(state.h, state.h * u_new)
    }

    /// Check if friction will be stiff for given state and time step.
    ///
    /// Friction is considered stiff if dt * C_f * |u| / h > 1
    pub fn is_stiff(&self, state: &SWEState, dt: f64) -> bool {
        if state.h < self.h_min {
            return false;
        }

        let u = (state.hu / state.h).abs();
        let c_f = self.friction_coefficient(state.h);

        dt * c_f * u / state.h > 1.0
    }
}

impl SourceTerm for ManningFriction {
    fn evaluate(&self, state: &SWEState, _db_dx: f64, _position: f64, _time: f64) -> SWEState {
        self.explicit_source(state)
    }

    fn name(&self) -> &'static str {
        "manning_friction"
    }
}

/// Chezy friction formulation.
///
/// Alternative to Manning, with constant friction coefficient:
/// S_hu = -C_D |u| u / h
///
/// where C_D is the drag coefficient (dimensionless).
#[derive(Clone, Debug)]
pub struct ChezyFriction {
    /// Drag coefficient (dimensionless)
    pub c_d: f64,
    /// Minimum depth
    pub h_min: f64,
}

impl ChezyFriction {
    /// Create a new Chezy friction source term.
    pub fn new(c_d: f64) -> Self {
        Self { c_d, h_min: 1e-6 }
    }

    /// Compute friction source explicitly.
    pub fn explicit_source(&self, state: &SWEState) -> SWEState {
        if state.h < self.h_min {
            return SWEState::zero();
        }

        let u = state.hu / state.h;
        let s_hu = -self.c_d * u.abs() * u / state.h;

        SWEState::new(0.0, s_hu)
    }
}

impl SourceTerm for ChezyFriction {
    fn evaluate(&self, state: &SWEState, _db_dx: f64, _position: f64, _time: f64) -> SWEState {
        self.explicit_source(state)
    }

    fn name(&self) -> &'static str {
        "chezy_friction"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 10.0;
    const TOL: f64 = 1e-10;

    #[test]
    fn test_manning_zero_velocity() {
        let friction = ManningFriction::new(G, 0.03);
        let state = SWEState::new(2.0, 0.0); // u = 0

        let s = friction.evaluate(&state, 0.0, 0.0, 0.0);

        assert!(s.h.abs() < TOL);
        assert!(s.hu.abs() < TOL);
    }

    #[test]
    fn test_manning_direction() {
        let friction = ManningFriction::new(G, 0.03);

        // Positive velocity
        let state_pos = SWEState::new(2.0, 4.0); // u = 2
        let s_pos = friction.evaluate(&state_pos, 0.0, 0.0, 0.0);

        // Friction should oppose motion (negative source)
        assert!(s_pos.hu < 0.0, "Friction should oppose positive velocity");

        // Negative velocity
        let state_neg = SWEState::new(2.0, -4.0); // u = -2
        let s_neg = friction.evaluate(&state_neg, 0.0, 0.0, 0.0);

        // Friction should oppose motion (positive source)
        assert!(s_neg.hu > 0.0, "Friction should oppose negative velocity");

        // Magnitude should be the same
        assert!((s_pos.hu.abs() - s_neg.hu.abs()).abs() < TOL);
    }

    #[test]
    fn test_manning_shallow_water_stronger() {
        let friction = ManningFriction::new(G, 0.03);

        // Same velocity, different depths
        let deep = SWEState::new(4.0, 8.0); // h=4, u=2
        let shallow = SWEState::new(1.0, 2.0); // h=1, u=2

        let s_deep = friction.evaluate(&deep, 0.0, 0.0, 0.0);
        let s_shallow = friction.evaluate(&shallow, 0.0, 0.0, 0.0);

        // Friction should be stronger in shallow water
        assert!(
            s_shallow.hu.abs() > s_deep.hu.abs(),
            "Friction should be stronger in shallow water: shallow={}, deep={}",
            s_shallow.hu.abs(),
            s_deep.hu.abs()
        );
    }

    #[test]
    fn test_manning_coefficient_dependency() {
        let friction_smooth = ManningFriction::new(G, 0.01);
        let friction_rough = ManningFriction::new(G, 0.05);

        let state = SWEState::new(2.0, 4.0);

        let s_smooth = friction_smooth.evaluate(&state, 0.0, 0.0, 0.0);
        let s_rough = friction_rough.evaluate(&state, 0.0, 0.0, 0.0);

        // Rough surface should have more friction
        assert!(
            s_rough.hu.abs() > s_smooth.hu.abs(),
            "Rough surface should have more friction"
        );

        // Friction scales as n²
        let ratio = s_rough.hu / s_smooth.hu;
        let expected_ratio = (0.05 / 0.01_f64).powi(2);
        assert!(
            (ratio - expected_ratio).abs() < 0.01,
            "Friction should scale as n²: ratio={}, expected={}",
            ratio,
            expected_ratio
        );
    }

    #[test]
    fn test_semi_implicit_update() {
        let friction = ManningFriction::new(G, 0.03);
        let state = SWEState::new(2.0, 4.0); // h=2, u=2
        let dt = 0.1;

        let state_new = friction.semi_implicit_update(&state, dt);

        // Depth should be unchanged
        assert!((state_new.h - state.h).abs() < TOL);

        // Velocity should decrease
        let u_old = state.hu / state.h;
        let u_new = state_new.hu / state_new.h;
        assert!(u_new.abs() < u_old.abs(), "Velocity should decrease");

        // Velocity should maintain sign
        assert!(u_new > 0.0, "Velocity should maintain sign");
    }

    #[test]
    fn test_semi_implicit_stability() {
        let friction = ManningFriction::new(G, 0.1); // Strong friction
        let state = SWEState::new(0.1, 0.2); // Shallow, fast flow
        let dt = 10.0; // Very large time step

        let state_new = friction.semi_implicit_update(&state, dt);

        // Should not blow up or change sign
        assert!(state_new.h > 0.0);
        assert!(state_new.hu.is_finite());
        assert!(state_new.hu >= 0.0, "Should not overshoot to negative");
    }

    #[test]
    fn test_is_stiff() {
        let friction = ManningFriction::new(G, 0.1); // High friction coefficient

        // Deep water, moderate flow - not stiff
        let deep = SWEState::new(10.0, 10.0);
        assert!(!friction.is_stiff(&deep, 0.01));

        // Shallow water, fast flow, large dt - stiff
        let shallow = SWEState::new(0.01, 0.1); // h=0.01, u=10
        assert!(friction.is_stiff(&shallow, 1.0));
    }

    #[test]
    fn test_chezy_friction() {
        let chezy = ChezyFriction::new(0.002);
        let state = SWEState::new(2.0, 4.0);

        let s = chezy.evaluate(&state, 0.0, 0.0, 0.0);

        assert!(s.h.abs() < TOL);
        assert!(s.hu < 0.0, "Friction should oppose motion");
    }

    #[test]
    fn test_dry_cell_no_friction() {
        let friction = ManningFriction::new(G, 0.03);
        let state = SWEState::new(1e-10, 1e-10);

        let s = friction.evaluate(&state, 0.0, 0.0, 0.0);

        assert!(s.h.abs() < TOL);
        assert!(s.hu.abs() < TOL);
    }
}
