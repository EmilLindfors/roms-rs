//! 2D bottom friction source terms for shallow water equations.
//!
//! Bottom friction dissipates momentum and is particularly important in
//! shallow coastal areas and narrow straits. The most common formulation
//! is Manning's equation.
//!
//! Manning friction (2D):
//!   S_hu = -g n² |u| u / h^{1/3}
//!   S_hv = -g n² |u| v / h^{1/3}
//!
//! where n is the Manning coefficient and |u| = sqrt(u² + v²).
//!
//! Friction becomes stiff in shallow water, so implicit or semi-implicit
//! treatment may be needed for stability.

use crate::solver::SWEState2D;
use crate::source::{SourceContext2D, SourceTerm2D};

/// Manning bottom friction source term for 2D shallow water equations.
///
/// S = (0, -g n² |u| u / h^{1/3}, -g n² |u| v / h^{1/3})
///
/// where |u| = sqrt(u² + v²) is the velocity magnitude.
///
/// Manning coefficient n has units of s/m^{1/3} and depends on bed roughness:
/// - Smooth concrete: n ≈ 0.012
/// - Natural channels: n ≈ 0.03-0.05
/// - Floodplains with vegetation: n ≈ 0.1-0.15
/// - Rocky fjord bottoms: n ≈ 0.025-0.035
///
/// # Example
///
/// ```ignore
/// use dg::source::{ManningFriction2D, CombinedSource2D, CoriolisSource2D};
///
/// let friction = ManningFriction2D::standard(0.025);  // Rocky fjord bottom
/// let coriolis = CoriolisSource2D::norwegian_coast();
///
/// let combined = CombinedSource2D::new(vec![&friction, &coriolis]);
/// ```
#[derive(Clone, Debug)]
pub struct ManningFriction2D {
    /// Gravitational acceleration (m/s²)
    pub g: f64,
    /// Manning coefficient (s/m^{1/3})
    pub manning_n: f64,
    /// Minimum depth for friction calculation (m)
    pub h_min: f64,
}

impl ManningFriction2D {
    /// Create a new Manning friction source term.
    ///
    /// # Arguments
    /// * `g` - Gravitational acceleration (m/s²)
    /// * `manning_n` - Manning coefficient (s/m^{1/3})
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

    /// Typical values for natural water bodies.
    pub fn natural_channel() -> Self {
        Self::standard(0.035)
    }

    /// Typical values for rocky fjord bottoms.
    pub fn fjord() -> Self {
        Self::standard(0.025)
    }

    /// Compute the friction coefficient C_f.
    ///
    /// τ = ρ C_f |u| u, where C_f = g n² h^{-1/3}
    #[inline]
    pub fn friction_coefficient(&self, h: f64) -> f64 {
        let h_eff = h.max(self.h_min);
        // Use cbrt() instead of powf(1.0/3.0) - 2-3x faster
        self.g * self.manning_n * self.manning_n / h_eff.cbrt()
    }

    /// Compute friction source term explicitly.
    #[inline]
    pub fn explicit_source(&self, state: &SWEState2D, h_min: f64) -> SWEState2D {
        if state.h < h_min {
            return SWEState2D::zero();
        }

        let h_inv = 1.0 / state.h;
        let u = state.hu * h_inv;
        let v = state.hv * h_inv;
        let speed_sq = u * u + v * v;

        if speed_sq < 1e-28 {
            return SWEState2D::zero();
        }

        let speed = speed_sq.sqrt();
        let c_f = self.friction_coefficient(state.h);
        let factor = -c_f * speed;

        // S = (0, -C_f |u| u, -C_f |u| v)
        SWEState2D {
            h: 0.0,
            hu: factor * u,
            hv: factor * v,
        }
    }

    /// Semi-implicit friction update for 2D.
    ///
    /// For stiff friction (shallow water), explicit treatment may require
    /// very small time steps. Semi-implicit treatment is unconditionally stable.
    ///
    /// Given: du/dt = -C_f |u| u / h (component-wise)
    /// Linearize: du/dt ≈ -C_f |u^n| u^{n+1} / h
    /// Solution: u^{n+1} = u^n / (1 + dt * C_f * |u^n| / h)
    ///
    /// # Arguments
    /// * `state` - Current state
    /// * `dt` - Time step
    ///
    /// # Returns
    /// Updated state after friction
    pub fn semi_implicit_update(&self, state: &SWEState2D, dt: f64) -> SWEState2D {
        if state.h < self.h_min {
            return *state;
        }

        let u = state.hu / state.h;
        let v = state.hv / state.h;
        let speed = (u * u + v * v).sqrt();

        if speed < 1e-14 {
            return *state;
        }

        let c_f = self.friction_coefficient(state.h);

        // Damping factor
        let denom = 1.0 + dt * c_f * speed / state.h;
        let u_new = u / denom;
        let v_new = v / denom;

        SWEState2D {
            h: state.h,
            hu: state.h * u_new,
            hv: state.h * v_new,
        }
    }

    /// Check if friction will be stiff for given state and time step.
    ///
    /// Friction is considered stiff if dt * C_f * |u| / h > 1
    pub fn is_stiff_for_state(&self, state: &SWEState2D, dt: f64) -> bool {
        if state.h < self.h_min {
            return false;
        }

        let u = state.hu / state.h;
        let v = state.hv / state.h;
        let speed = (u * u + v * v).sqrt();
        let c_f = self.friction_coefficient(state.h);

        dt * c_f * speed / state.h > 1.0
    }
}

impl SourceTerm2D for ManningFriction2D {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        self.explicit_source(&ctx.state, ctx.h_min)
    }

    fn name(&self) -> &'static str {
        "manning_friction_2d"
    }

    fn is_stiff(&self) -> bool {
        // Manning friction can be stiff in shallow water
        true
    }
}

/// Chezy friction formulation for 2D.
///
/// Alternative to Manning, with constant friction coefficient:
/// S_hu = -C_D |u| u / h
/// S_hv = -C_D |u| v / h
///
/// where C_D is the drag coefficient (dimensionless).
#[derive(Clone, Debug)]
pub struct ChezyFriction2D {
    /// Drag coefficient (dimensionless)
    pub c_d: f64,
    /// Minimum depth (m)
    pub h_min: f64,
}

impl ChezyFriction2D {
    /// Create a new Chezy friction source term.
    pub fn new(c_d: f64) -> Self {
        Self { c_d, h_min: 1e-6 }
    }

    /// Compute friction source explicitly.
    pub fn explicit_source(&self, state: &SWEState2D) -> SWEState2D {
        if state.h < self.h_min {
            return SWEState2D::zero();
        }

        let u = state.hu / state.h;
        let v = state.hv / state.h;
        let speed = (u * u + v * v).sqrt();

        if speed < 1e-14 {
            return SWEState2D::zero();
        }

        SWEState2D {
            h: 0.0,
            hu: -self.c_d * speed * u / state.h,
            hv: -self.c_d * speed * v / state.h,
        }
    }
}

impl SourceTerm2D for ChezyFriction2D {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        self.explicit_source(&ctx.state)
    }

    fn name(&self) -> &'static str {
        "chezy_friction_2d"
    }

    fn is_stiff(&self) -> bool {
        true
    }
}

/// Spatially-varying Manning friction.
///
/// Allows the Manning coefficient to vary with position,
/// useful for domains with different bed types.
///
/// # Example
///
/// ```ignore
/// // Higher friction in shallow coastal areas
/// let friction = SpatiallyVaryingManning2D::new(9.81, |x, y| {
///     let depth_proxy = y;  // Assume depth increases with y
///     if depth_proxy < 10.0 {
///         0.05  // Higher friction in shallows
///     } else {
///         0.025  // Lower friction in deep water
///     }
/// });
/// ```
pub struct SpatiallyVaryingManning2D<F>
where
    F: Fn(f64, f64) -> f64 + Send + Sync,
{
    /// Gravitational acceleration (m/s²)
    pub g: f64,
    /// Function returning Manning coefficient n(x, y)
    pub manning_fn: F,
    /// Minimum depth (m)
    pub h_min: f64,
}

impl<F> SpatiallyVaryingManning2D<F>
where
    F: Fn(f64, f64) -> f64 + Send + Sync,
{
    /// Create a new spatially-varying Manning friction.
    pub fn new(g: f64, manning_fn: F) -> Self {
        Self {
            g,
            manning_fn,
            h_min: 1e-6,
        }
    }

    /// Standard gravity with spatially-varying coefficient.
    pub fn standard(manning_fn: F) -> Self {
        Self::new(9.81, manning_fn)
    }
}

impl<F> SourceTerm2D for SpatiallyVaryingManning2D<F>
where
    F: Fn(f64, f64) -> f64 + Send + Sync,
{
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        if ctx.state.h < self.h_min {
            return SWEState2D::zero();
        }

        let (x, y) = ctx.position;
        let n = (self.manning_fn)(x, y);

        let u = ctx.state.hu / ctx.state.h;
        let v = ctx.state.hv / ctx.state.h;
        let speed = (u * u + v * v).sqrt();

        if speed < 1e-14 {
            return SWEState2D::zero();
        }

        let h_eff = ctx.state.h.max(self.h_min);
        // Use cbrt() instead of powf(1.0/3.0) - 2-3x faster
        let c_f = self.g * n * n / h_eff.cbrt();

        SWEState2D {
            h: 0.0,
            hu: -c_f * speed * u,
            hv: -c_f * speed * v,
        }
    }

    fn name(&self) -> &'static str {
        "spatially_varying_manning_2d"
    }

    fn is_stiff(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 10.0;
    const TOL: f64 = 1e-10;

    fn make_context(h: f64, hu: f64, hv: f64) -> SourceContext2D {
        SourceContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(h, hu, hv),
            0.0,
            (0.0, 0.0),
            G,
            1e-6,
        )
    }

    #[test]
    fn test_manning_zero_velocity() {
        let friction = ManningFriction2D::new(G, 0.03);
        let ctx = make_context(2.0, 0.0, 0.0);
        let s = friction.evaluate(&ctx);

        assert!(s.h.abs() < TOL);
        assert!(s.hu.abs() < TOL);
        assert!(s.hv.abs() < TOL);
    }

    #[test]
    fn test_manning_x_direction() {
        let friction = ManningFriction2D::new(G, 0.03);

        // Positive x-velocity
        let ctx = make_context(2.0, 4.0, 0.0); // h=2, u=2, v=0
        let s = friction.evaluate(&ctx);

        assert!(s.h.abs() < TOL);
        assert!(s.hu < 0.0, "Friction should oppose positive x-velocity");
        assert!(s.hv.abs() < TOL, "No friction in y when v=0");
    }

    #[test]
    fn test_manning_y_direction() {
        let friction = ManningFriction2D::new(G, 0.03);

        // Positive y-velocity
        let ctx = make_context(2.0, 0.0, 4.0); // h=2, u=0, v=2
        let s = friction.evaluate(&ctx);

        assert!(s.h.abs() < TOL);
        assert!(s.hu.abs() < TOL, "No friction in x when u=0");
        assert!(s.hv < 0.0, "Friction should oppose positive y-velocity");
    }

    #[test]
    fn test_manning_diagonal_flow() {
        let friction = ManningFriction2D::new(G, 0.03);

        // Diagonal flow: u = v = 2
        let h = 2.0;
        let ctx = make_context(h, 2.0 * h, 2.0 * h); // h=2, u=2, v=2
        let s = friction.evaluate(&ctx);

        // Both components should be negative (opposing flow)
        assert!(s.hu < 0.0, "hu friction should be negative");
        assert!(s.hv < 0.0, "hv friction should be negative");

        // Components should be equal for equal velocities
        assert!(
            (s.hu - s.hv).abs() < TOL,
            "Equal velocities should give equal friction"
        );
    }

    #[test]
    fn test_manning_shallow_stronger() {
        let friction = ManningFriction2D::new(G, 0.03);

        // Same velocity, different depths
        let u = 2.0;
        let h_deep = 4.0;
        let h_shallow = 1.0;

        let ctx_deep = make_context(h_deep, u * h_deep, 0.0);
        let ctx_shallow = make_context(h_shallow, u * h_shallow, 0.0);

        let s_deep = friction.evaluate(&ctx_deep);
        let s_shallow = friction.evaluate(&ctx_shallow);

        assert!(
            s_shallow.hu.abs() > s_deep.hu.abs(),
            "Friction should be stronger in shallow water: shallow={}, deep={}",
            s_shallow.hu.abs(),
            s_deep.hu.abs()
        );
    }

    #[test]
    fn test_manning_coefficient_scaling() {
        let friction_smooth = ManningFriction2D::new(G, 0.01);
        let friction_rough = ManningFriction2D::new(G, 0.05);

        let ctx = make_context(2.0, 4.0, 0.0);

        let s_smooth = friction_smooth.evaluate(&ctx);
        let s_rough = friction_rough.evaluate(&ctx);

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
        let friction = ManningFriction2D::new(G, 0.03);
        let state = SWEState2D::new(2.0, 4.0, 2.0); // h=2, u=2, v=1
        let dt = 0.1;

        let state_new = friction.semi_implicit_update(&state, dt);

        // Depth unchanged
        assert!((state_new.h - state.h).abs() < TOL);

        // Velocities should decrease
        let u_old = state.hu / state.h;
        let v_old = state.hv / state.h;
        let u_new = state_new.hu / state_new.h;
        let v_new = state_new.hv / state_new.h;

        assert!(u_new.abs() < u_old.abs(), "u should decrease");
        assert!(v_new.abs() < v_old.abs(), "v should decrease");

        // Signs should be preserved
        assert!(u_new > 0.0, "u should stay positive");
        assert!(v_new > 0.0, "v should stay positive");
    }

    #[test]
    fn test_semi_implicit_stability() {
        let friction = ManningFriction2D::new(G, 0.1); // Strong friction
        let state = SWEState2D::new(0.1, 0.2, 0.3); // Shallow, fast flow
        let dt = 10.0; // Very large time step

        let state_new = friction.semi_implicit_update(&state, dt);

        // Should not blow up
        assert!(state_new.h > 0.0);
        assert!(state_new.hu.is_finite());
        assert!(state_new.hv.is_finite());
        // Should not overshoot
        assert!(state_new.hu >= 0.0);
        assert!(state_new.hv >= 0.0);
    }

    #[test]
    fn test_is_stiff_for_state() {
        let friction = ManningFriction2D::new(G, 0.1);

        // Deep water - not stiff
        let deep = SWEState2D::new(10.0, 10.0, 0.0);
        assert!(!friction.is_stiff_for_state(&deep, 0.01));

        // Shallow water, large dt - stiff
        let shallow = SWEState2D::new(0.01, 0.1, 0.0);
        assert!(friction.is_stiff_for_state(&shallow, 1.0));
    }

    #[test]
    fn test_chezy_friction() {
        let chezy = ChezyFriction2D::new(0.002);
        let ctx = make_context(2.0, 4.0, 2.0);
        let s = chezy.evaluate(&ctx);

        assert!(s.h.abs() < TOL);
        assert!(s.hu < 0.0, "hu friction should oppose flow");
        assert!(s.hv < 0.0, "hv friction should oppose flow");
    }

    #[test]
    fn test_dry_cell_no_friction() {
        let friction = ManningFriction2D::new(G, 0.03);
        let ctx = make_context(1e-10, 1e-10, 1e-10);
        let s = friction.evaluate(&ctx);

        assert!(s.h.abs() < TOL);
        assert!(s.hu.abs() < TOL);
        assert!(s.hv.abs() < TOL);
    }

    #[test]
    fn test_spatially_varying_manning() {
        // Higher friction for x < 5
        let friction = SpatiallyVaryingManning2D::new(G, |x, _y| if x < 5.0 { 0.05 } else { 0.02 });

        let state = SWEState2D::new(2.0, 4.0, 0.0);

        let ctx_rough = SourceContext2D::new(0.0, (0.0, 0.0), state, 0.0, (0.0, 0.0), G, 1e-6);

        let ctx_smooth = SourceContext2D::new(0.0, (10.0, 0.0), state, 0.0, (0.0, 0.0), G, 1e-6);

        let s_rough = friction.evaluate(&ctx_rough);
        let s_smooth = friction.evaluate(&ctx_smooth);

        assert!(
            s_rough.hu.abs() > s_smooth.hu.abs(),
            "Should have more friction in rough region"
        );
    }

    #[test]
    fn test_preset_friction_values() {
        let natural = ManningFriction2D::natural_channel();
        assert!((natural.manning_n - 0.035).abs() < 1e-10);

        let fjord = ManningFriction2D::fjord();
        assert!((fjord.manning_n - 0.025).abs() < 1e-10);
    }

    #[test]
    fn test_is_stiff_trait() {
        let manning = ManningFriction2D::standard(0.03);
        assert!(manning.is_stiff());

        let chezy = ChezyFriction2D::new(0.002);
        assert!(chezy.is_stiff());
    }
}
