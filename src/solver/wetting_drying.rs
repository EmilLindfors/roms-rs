//! Improved wetting/drying treatment for shallow water equations.
//!
//! The standard approach uses a hard cutoff at h_min, which can cause:
//! - Sharp transitions and numerical oscillations
//! - Unrealistic velocities at wet/dry fronts
//! - Momentum discontinuities
//!
//! This module provides improved treatment:
//! - **Thin-layer blending**: Gradual flux reduction as h → h_min
//! - **Velocity capping**: Maximum velocity based on shallow water physics
//! - **Smooth momentum damping**: Continuous transition to zero momentum
//!
//! # References
//! - Kärnä et al. (2011), "A non-hydrostatic version of SELFE"
//! - Medeiros & Hagen (2013), "Review of wetting and drying algorithms for
//!   numerical tidal flow models"

use crate::solver::state_2d::SWEState2D;

/// Configuration for wetting/drying treatment.
#[derive(Clone, Debug)]
pub struct WetDryConfig {
    /// Minimum depth threshold (cells with h < h_min are considered dry)
    pub h_min: f64,
    /// Depth at which thin-layer blending begins (h_thin > h_min)
    pub h_thin: f64,
    /// Maximum allowed velocity magnitude (m/s)
    pub max_velocity: f64,
    /// Gravitational acceleration
    pub g: f64,
}

impl WetDryConfig {
    /// Create configuration with standard parameters.
    ///
    /// # Arguments
    /// * `h_min` - Minimum depth (typically 0.001 - 0.1 m)
    /// * `g` - Gravitational acceleration
    pub fn new(h_min: f64, g: f64) -> Self {
        Self {
            h_min,
            h_thin: 10.0 * h_min, // Start blending at 10x h_min
            max_velocity: 20.0,   // Cap at 20 m/s (very fast tidal current)
            g,
        }
    }

    /// Create with custom thin-layer threshold.
    pub fn with_h_thin(mut self, h_thin: f64) -> Self {
        self.h_thin = h_thin.max(self.h_min);
        self
    }

    /// Create with custom maximum velocity.
    pub fn with_max_velocity(mut self, max_velocity: f64) -> Self {
        self.max_velocity = max_velocity;
        self
    }

    /// Check if depth is in the thin-layer regime.
    #[inline]
    pub fn is_thin_layer(&self, h: f64) -> bool {
        h > self.h_min && h < self.h_thin
    }

    /// Check if cell is dry.
    #[inline]
    pub fn is_dry(&self, h: f64) -> bool {
        h <= self.h_min
    }

    /// Check if cell is fully wet (above thin-layer).
    #[inline]
    pub fn is_wet(&self, h: f64) -> bool {
        h >= self.h_thin
    }

    /// Compute blending factor for thin-layer regime.
    ///
    /// Returns:
    /// - 0 when h <= h_min (dry)
    /// - 1 when h >= h_thin (fully wet)
    /// - Smooth transition in between (cubic Hermite)
    #[inline(always)]
    pub fn blending_factor(&self, h: f64) -> f64 {
        if h <= self.h_min {
            0.0
        } else if h >= self.h_thin {
            1.0
        } else {
            // Smooth Hermite interpolation: 3t² - 2t³
            // Precompute inverse for efficiency
            let inv_range = 1.0 / (self.h_thin - self.h_min);
            let t = (h - self.h_min) * inv_range;
            t * t * (3.0 - 2.0 * t)
        }
    }

    /// Precomputed inverse range for blending factor (call once per timestep).
    #[inline(always)]
    pub fn inv_blend_range(&self) -> f64 {
        1.0 / (self.h_thin - self.h_min)
    }

    /// Fast blending factor with precomputed inverse range.
    #[inline(always)]
    pub fn blending_factor_fast(&self, h: f64, inv_range: f64) -> f64 {
        if h <= self.h_min {
            0.0
        } else if h >= self.h_thin {
            1.0
        } else {
            let t = (h - self.h_min) * inv_range;
            t * t * (3.0 - 2.0 * t)
        }
    }

    /// Compute desingularized velocity with thin-layer blending.
    ///
    /// Returns velocity that:
    /// - Goes to zero smoothly as h → h_min
    /// - Is capped at max_velocity
    /// - Uses desingularization for numerical stability
    pub fn compute_velocity(&self, h: f64, hu: f64, hv: f64) -> (f64, f64) {
        if h <= self.h_min {
            return (0.0, 0.0);
        }

        // Desingularization formula: u = 2h·hu / (h² + h_reg²)
        let h_reg = h.max(self.h_min);
        let denom = h * h + h_reg * h_reg;
        let u = 2.0 * h * hu / denom;
        let v = 2.0 * h * hv / denom;

        // Apply velocity cap
        let speed = (u * u + v * v).sqrt();
        if speed > self.max_velocity {
            let scale = self.max_velocity / speed;
            (u * scale, v * scale)
        } else {
            (u, v)
        }
    }

    /// Apply thin-layer damping to momentum.
    ///
    /// In the thin-layer regime, momentum is gradually reduced to prevent
    /// unrealistic velocities as depth decreases.
    pub fn damp_momentum(&self, state: &SWEState2D) -> SWEState2D {
        let alpha = self.blending_factor(state.h);

        if alpha >= 1.0 {
            // Fully wet: no damping, but still apply velocity cap
            return self.apply_velocity_cap(state);
        }

        if alpha <= 0.0 {
            // Dry: zero momentum
            return SWEState2D::new(state.h.max(0.0), 0.0, 0.0);
        }

        // Thin layer: blend momentum toward zero
        let hu_damped = alpha * state.hu;
        let hv_damped = alpha * state.hv;

        // Also apply velocity cap
        let result = SWEState2D::new(state.h, hu_damped, hv_damped);
        self.apply_velocity_cap(&result)
    }

    /// Apply velocity cap to state, preserving direction.
    pub fn apply_velocity_cap(&self, state: &SWEState2D) -> SWEState2D {
        if state.h <= self.h_min {
            return SWEState2D::new(state.h.max(0.0), 0.0, 0.0);
        }

        let (u, v) = self.compute_velocity(state.h, state.hu, state.hv);
        let speed = (u * u + v * v).sqrt();

        if speed <= self.max_velocity {
            *state
        } else {
            // Cap velocity while preserving direction
            let scale = self.max_velocity / speed;
            SWEState2D::new(state.h, state.h * u * scale, state.h * v * scale)
        }
    }

    /// Determine wet/dry status of an interface.
    ///
    /// Returns (left_wet, right_wet) tuple.
    pub fn interface_wet_status(&self, h_l: f64, h_r: f64) -> (bool, bool) {
        (h_l > self.h_min, h_r > self.h_min)
    }

    /// Compute interface flux factor for wet/dry interfaces.
    ///
    /// At wet-dry interfaces, we need to:
    /// 1. Allow water to flow from wet to dry (wetting)
    /// 2. Allow water to flow from thin to wet (draining)
    /// 3. Prevent spurious flow into very dry regions
    ///
    /// Returns a factor in [0, 1] to multiply the numerical flux.
    pub fn interface_flux_factor(&self, h_l: f64, h_r: f64) -> f64 {
        let (l_wet, r_wet) = self.interface_wet_status(h_l, h_r);

        match (l_wet, r_wet) {
            (false, false) => 0.0, // Both dry: no flux
            (true, true) => {
                // Both wet: use minimum blending factor
                let alpha_l = self.blending_factor(h_l);
                let alpha_r = self.blending_factor(h_r);
                alpha_l.min(alpha_r)
            }
            (true, false) => {
                // Left wet, right dry: wetting front
                // Allow outflow from wet side, scaled by blending
                self.blending_factor(h_l)
            }
            (false, true) => {
                // Left dry, right wet: wetting front (reversed)
                self.blending_factor(h_r)
            }
        }
    }
}

impl Default for WetDryConfig {
    fn default() -> Self {
        Self::new(0.01, 9.81) // 1cm minimum depth
    }
}

/// Apply wetting/drying correction to SWE state.
///
/// This function should be called after each RK stage to:
/// 1. Ensure h >= 0
/// 2. Damp momentum in thin-layer regime
/// 3. Cap unrealistic velocities
///
/// # Arguments
/// * `state` - State to correct (modified in place)
/// * `config` - Wet/dry configuration
pub fn apply_wet_dry_correction(state: &mut SWEState2D, config: &WetDryConfig) {
    // Ensure non-negative depth
    state.h = state.h.max(0.0);

    // Apply thin-layer damping and velocity cap
    *state = config.damp_momentum(state);
}

/// Apply wetting/drying correction to entire solution.
///
/// Optimized version that:
/// - Skips fully-wet elements (no correction needed)
/// - Fuses blending and velocity cap into single pass
/// - Precomputes inverse range for blending factor
pub fn apply_wet_dry_correction_all(
    solution: &mut crate::solver::SWESolution2D,
    config: &WetDryConfig,
) {
    let n_elements = solution.n_elements;
    let n_nodes = solution.n_nodes;
    let inv_range = config.inv_blend_range();
    let max_vel_sq = config.max_velocity * config.max_velocity;

    for k in 0..n_elements {
        // Fast path: check if element is fully wet (skip correction)
        let mut min_h = f64::INFINITY;
        let mut max_h = f64::NEG_INFINITY;
        let elem_data = solution.element_data(k);

        // Scan for min/max depth in element (stride of 3 for [h, hu, hv])
        for i in 0..n_nodes {
            let h = elem_data[i * 3];
            min_h = min_h.min(h);
            max_h = max_h.max(h);
        }

        // If all nodes are fully wet and depths are non-negative, check velocity cap only
        if min_h >= config.h_thin {
            // Fully wet: only need to check velocity cap
            apply_velocity_cap_element(solution, k, config, max_vel_sq);
            continue;
        }

        // If all nodes are dry, zero out momentum
        if max_h <= config.h_min {
            let elem_data = solution.element_data_mut(k);
            for i in 0..n_nodes {
                let base = i * 3;
                elem_data[base] = elem_data[base].max(0.0); // h >= 0
                elem_data[base + 1] = 0.0; // hu = 0
                elem_data[base + 2] = 0.0; // hv = 0
            }
            continue;
        }

        // Mixed wet/dry: apply full correction per-node
        apply_wet_dry_correction_element_fused(solution, k, config, inv_range, max_vel_sq);
    }
}

/// Apply velocity cap only (for fully wet elements).
#[inline]
fn apply_velocity_cap_element(
    solution: &mut crate::solver::SWESolution2D,
    k: usize,
    config: &WetDryConfig,
    max_vel_sq: f64,
) {
    let n_nodes = solution.n_nodes;
    let elem_data = solution.element_data_mut(k);

    for i in 0..n_nodes {
        let base = i * 3;
        let h = elem_data[base];
        let hu = elem_data[base + 1];
        let hv = elem_data[base + 2];

        if h <= config.h_min {
            continue;
        }

        // Compute velocity squared to check cap
        let h_inv = 1.0 / h;
        let u = hu * h_inv;
        let v = hv * h_inv;
        let speed_sq = u * u + v * v;

        if speed_sq > max_vel_sq {
            let scale = (max_vel_sq / speed_sq).sqrt();
            elem_data[base + 1] = h * u * scale;
            elem_data[base + 2] = h * v * scale;
        }
    }
}

/// Fused wet/dry correction for mixed wet/dry elements.
#[inline]
fn apply_wet_dry_correction_element_fused(
    solution: &mut crate::solver::SWESolution2D,
    k: usize,
    config: &WetDryConfig,
    inv_range: f64,
    max_vel_sq: f64,
) {
    let n_nodes = solution.n_nodes;
    let elem_data = solution.element_data_mut(k);

    for i in 0..n_nodes {
        let base = i * 3;
        let mut h = elem_data[base];
        let mut hu = elem_data[base + 1];
        let mut hv = elem_data[base + 2];

        // Ensure non-negative depth
        h = h.max(0.0);

        // Compute blending factor
        let alpha = config.blending_factor_fast(h, inv_range);

        if alpha <= 0.0 {
            // Dry: zero momentum
            elem_data[base] = h;
            elem_data[base + 1] = 0.0;
            elem_data[base + 2] = 0.0;
            continue;
        }

        // Apply thin-layer damping
        if alpha < 1.0 {
            hu *= alpha;
            hv *= alpha;
        }

        // Apply velocity cap
        if h > config.h_min {
            let h_inv = 1.0 / h;
            let u = hu * h_inv;
            let v = hv * h_inv;
            let speed_sq = u * u + v * v;

            if speed_sq > max_vel_sq {
                let scale = (max_vel_sq / speed_sq).sqrt();
                hu = h * u * scale;
                hv = h * v * scale;
            }
        }

        elem_data[base] = h;
        elem_data[base + 1] = hu;
        elem_data[base + 2] = hv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_blending_factor() {
        let config = WetDryConfig::new(0.01, 9.81); // h_min=0.01, h_thin=0.1

        // Dry: factor = 0
        assert!((config.blending_factor(0.0) - 0.0).abs() < TOL);
        assert!((config.blending_factor(0.005) - 0.0).abs() < TOL);
        assert!((config.blending_factor(0.01) - 0.0).abs() < TOL);

        // Wet: factor = 1
        assert!((config.blending_factor(0.1) - 1.0).abs() < TOL);
        assert!((config.blending_factor(1.0) - 1.0).abs() < TOL);

        // Thin layer: 0 < factor < 1
        let mid = config.blending_factor(0.055); // midpoint
        assert!(mid > 0.0 && mid < 1.0);

        // Hermite interpolation is smooth: f(0.5) = 0.5 for t=0.5
        // t = (0.055 - 0.01) / (0.1 - 0.01) = 0.045/0.09 = 0.5
        // f(0.5) = 3*0.25 - 2*0.125 = 0.75 - 0.25 = 0.5
        assert!((mid - 0.5).abs() < TOL);
    }

    #[test]
    fn test_velocity_cap() {
        let config = WetDryConfig::new(0.01, 9.81).with_max_velocity(10.0);

        // Normal velocity (not capped)
        let (u, v) = config.compute_velocity(1.0, 5.0, 0.0);
        assert!((u - 5.0).abs() < 1e-8);
        assert!(v.abs() < TOL);

        // High velocity (should be capped)
        let (u, v) = config.compute_velocity(1.0, 50.0, 0.0);
        let speed = (u * u + v * v).sqrt();
        assert!((speed - 10.0).abs() < 1e-8); // Capped at 10 m/s
    }

    #[test]
    fn test_thin_layer_damping() {
        let config = WetDryConfig::new(0.01, 9.81);

        // Fully wet: no damping
        let state_wet = SWEState2D::new(1.0, 5.0, 3.0);
        let damped_wet = config.damp_momentum(&state_wet);
        assert!((damped_wet.hu - 5.0).abs() < 1e-8);
        assert!((damped_wet.hv - 3.0).abs() < 1e-8);

        // Dry: zero momentum
        let state_dry = SWEState2D::new(0.005, 0.1, 0.05);
        let damped_dry = config.damp_momentum(&state_dry);
        assert!(damped_dry.hu.abs() < TOL);
        assert!(damped_dry.hv.abs() < TOL);

        // Thin layer: reduced momentum
        let state_thin = SWEState2D::new(0.055, 1.0, 0.5); // h at midpoint
        let damped_thin = config.damp_momentum(&state_thin);
        // At midpoint, alpha ≈ 0.5
        assert!((damped_thin.hu - 0.5).abs() < 0.1);
        assert!((damped_thin.hv - 0.25).abs() < 0.05);
    }

    #[test]
    fn test_interface_flux_factor() {
        let config = WetDryConfig::new(0.01, 9.81);

        // Both dry: no flux
        let factor = config.interface_flux_factor(0.005, 0.003);
        assert!(factor.abs() < TOL);

        // Both fully wet: full flux
        let factor = config.interface_flux_factor(1.0, 2.0);
        assert!((factor - 1.0).abs() < TOL);

        // Wet-dry interface: reduced flux
        let factor = config.interface_flux_factor(0.5, 0.005);
        assert!((factor - 1.0).abs() < TOL); // Wet side is fully wet

        // Thin layer interface: reduced flux
        let factor = config.interface_flux_factor(0.055, 0.055);
        assert!(factor > 0.0 && factor < 1.0);
    }

    #[test]
    fn test_wet_dry_status() {
        let config = WetDryConfig::new(0.01, 9.81);

        assert!(config.is_dry(0.005));
        assert!(!config.is_dry(0.02));

        assert!(config.is_thin_layer(0.05));
        assert!(!config.is_thin_layer(0.005));
        assert!(!config.is_thin_layer(0.5));

        assert!(config.is_wet(1.0));
        assert!(!config.is_wet(0.05));
    }

    #[test]
    fn test_apply_correction() {
        let config = WetDryConfig::new(0.01, 9.81);

        // Negative depth should become zero
        let mut state = SWEState2D::new(-0.1, 1.0, 0.5);
        apply_wet_dry_correction(&mut state, &config);
        assert!(state.h >= 0.0);
        assert!(state.hu.abs() < TOL); // Dry, so zero momentum

        // Normal state should be unchanged (except velocity cap)
        let mut state = SWEState2D::new(1.0, 5.0, 3.0);
        apply_wet_dry_correction(&mut state, &config);
        assert!((state.h - 1.0).abs() < TOL);
        assert!((state.hu - 5.0).abs() < 1e-8);
    }
}
