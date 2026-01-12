//! Bathymetry validation for Flather-type boundary conditions.
//!
//! Flather BCs use surface elevation η = h + B where B is bathymetry (negative below MSL).
//! A common misconfiguration is forgetting to set bathymetry, leaving B = 0, which causes
//! spurious velocities from the large discrepancy between interior η and tidal η.
//!
//! This module provides shared validation logic used by all Flather BC variants.

use std::sync::atomic::{AtomicBool, Ordering};

/// Configuration for bathymetry validation thresholds.
#[derive(Debug, Clone, Copy)]
pub struct BathymetryValidationConfig {
    /// Maximum allowed discrepancy between interior and expected surface elevation (m).
    /// Default: 10.0 m
    pub max_discrepancy: f64,

    /// Minimum interior water depth to trigger validation (m).
    /// Shallow depths are less likely to cause issues.
    /// Default: 5.0 m
    pub min_interior_depth: f64,

    /// Maximum bathymetry magnitude considered "near zero" (m).
    /// Values below this suggest bathymetry wasn't set.
    /// Default: 1.0 m
    pub max_bathymetry_near_zero: f64,
}

impl Default for BathymetryValidationConfig {
    fn default() -> Self {
        Self {
            max_discrepancy: 10.0,
            min_interior_depth: 5.0,
            max_bathymetry_near_zero: 1.0,
        }
    }
}

/// Result of bathymetry validation check.
#[derive(Debug, Clone, Copy)]
pub struct BathymetryValidationResult {
    /// Whether the bathymetry configuration appears valid.
    pub is_valid: bool,

    /// Interior water depth (h).
    pub interior_h: f64,

    /// Bathymetry value at the boundary (B).
    pub bathymetry: f64,

    /// Computed interior surface elevation (η = h + B).
    pub surface_elevation: f64,

    /// Expected surface elevation (e.g., tidal elevation).
    pub expected_elevation: f64,

    /// Discrepancy between interior and expected surface elevation.
    pub discrepancy: f64,
}

impl BathymetryValidationResult {
    /// Suggest the correct bathymetry value to use.
    pub fn suggested_bathymetry(&self) -> f64 {
        -self.interior_h
    }
}

/// Validate that bathymetry is correctly configured for a Flather-type BC.
///
/// Returns a validation result indicating whether the configuration appears correct.
/// A configuration is flagged as potentially incorrect when:
/// 1. Bathymetry is near zero (|B| < threshold)
/// 2. Interior water depth is significant (h > threshold)
/// 3. Discrepancy between interior and expected elevation exceeds threshold
///
/// # Arguments
///
/// * `interior_h` - Interior water depth (h)
/// * `bathymetry` - Bathymetry value at boundary (B, typically negative below MSL)
/// * `expected_elevation` - Expected surface elevation (e.g., tidal elevation)
/// * `config` - Validation thresholds
pub fn validate_bathymetry_convention(
    interior_h: f64,
    bathymetry: f64,
    expected_elevation: f64,
    config: &BathymetryValidationConfig,
) -> BathymetryValidationResult {
    let surface_elevation = interior_h + bathymetry;
    let discrepancy = (surface_elevation - expected_elevation).abs();

    let is_valid = !(bathymetry.abs() < config.max_bathymetry_near_zero
        && interior_h > config.min_interior_depth
        && discrepancy > config.max_discrepancy);

    BathymetryValidationResult {
        is_valid,
        interior_h,
        bathymetry,
        surface_elevation,
        expected_elevation,
        discrepancy,
    }
}

/// Format a warning message for bathymetry misconfiguration.
///
/// # Arguments
///
/// * `bc_name` - Name of the boundary condition (e.g., "HarmonicFlather2D")
/// * `result` - Validation result from `validate_bathymetry_convention`
pub fn format_bathymetry_warning(bc_name: &str, result: &BathymetryValidationResult) -> String {
    format!(
        "WARNING [{bc_name}]: Possible bathymetry misconfiguration detected!\n\
         Interior h = {:.1}m, bathymetry = {:.2}m → surface elevation η = {:.1}m\n\
         But expected elevation = {:.2}m (difference = {:.1}m)\n\
         \n\
         This likely causes spurious velocities. Set bathymetry correctly:\n\
           let bathymetry = Bathymetry2D::constant(n_elements, n_nodes, {:.1});\n\
           let config = config.with_bathymetry(&bathymetry);\n\
         \n\
         See {bc_name} docs for details.",
        result.interior_h,
        result.bathymetry,
        result.surface_elevation,
        result.expected_elevation,
        result.discrepancy,
        result.suggested_bathymetry()
    )
}

/// Check bathymetry and emit a one-time warning if misconfigured.
///
/// This function handles the common pattern of warning once per BC type using
/// an `AtomicBool` flag. It validates the configuration and prints to stderr
/// if a problem is detected.
///
/// # Arguments
///
/// * `warned` - Atomic flag to ensure warning is only emitted once
/// * `bc_name` - Name of the boundary condition for the warning message
/// * `interior_h` - Interior water depth
/// * `bathymetry` - Bathymetry value at boundary
/// * `expected_elevation` - Expected surface elevation (e.g., tidal)
///
/// # Returns
///
/// The validation result (useful for additional handling if needed)
pub fn warn_once_if_misconfigured(
    warned: &AtomicBool,
    bc_name: &str,
    interior_h: f64,
    bathymetry: f64,
    expected_elevation: f64,
) -> BathymetryValidationResult {
    let config = BathymetryValidationConfig::default();
    let result = validate_bathymetry_convention(interior_h, bathymetry, expected_elevation, &config);

    if !result.is_valid && !warned.swap(true, Ordering::Relaxed) {
        eprintln!("{}", format_bathymetry_warning(bc_name, &result));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_bathymetry_convention() {
        // Correct setup: h = 50m, B = -50m → η = 0m (at MSL)
        let config = BathymetryValidationConfig::default();
        let result = validate_bathymetry_convention(50.0, -50.0, 0.5, &config);

        assert!(result.is_valid);
        assert!((result.surface_elevation - 0.0).abs() < 1e-10);
        assert!((result.discrepancy - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_zero_bathymetry() {
        // Incorrect setup: h = 50m, B = 0m → η = 50m (should be ~0)
        let config = BathymetryValidationConfig::default();
        let result = validate_bathymetry_convention(50.0, 0.0, 0.5, &config);

        assert!(!result.is_valid);
        assert!((result.surface_elevation - 50.0).abs() < 1e-10);
        assert!((result.discrepancy - 49.5).abs() < 1e-10);
        assert!((result.suggested_bathymetry() - (-50.0)).abs() < 1e-10);
    }

    #[test]
    fn test_threshold_edge_cases() {
        let config = BathymetryValidationConfig::default();

        // Just below threshold - should be valid (shallow water)
        let result = validate_bathymetry_convention(4.9, 0.0, 0.0, &config);
        assert!(result.is_valid);

        // Just below threshold - should be valid (small discrepancy)
        let result = validate_bathymetry_convention(50.0, 0.0, 40.5, &config);
        assert!(result.is_valid);

        // Just above threshold - should be valid (bathymetry set)
        let result = validate_bathymetry_convention(50.0, -1.1, 0.0, &config);
        assert!(result.is_valid);
    }

    #[test]
    fn test_warning_message_format() {
        let result = BathymetryValidationResult {
            is_valid: false,
            interior_h: 50.0,
            bathymetry: 0.0,
            surface_elevation: 50.0,
            expected_elevation: 0.5,
            discrepancy: 49.5,
        };

        let warning = format_bathymetry_warning("TestBC", &result);

        assert!(warning.contains("TestBC"));
        assert!(warning.contains("50.0m"));
        assert!(warning.contains("bathymetry = 0.00m"));
        assert!(warning.contains("-50.0"));
    }
}
