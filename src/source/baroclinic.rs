//! Baroclinic pressure source term for 2D shallow water equations.
//!
//! In a depth-averaged model, density variations (from temperature and salinity)
//! create horizontal pressure gradients that drive baroclinic circulation.
//!
//! # Physical Background
//!
//! The depth-integrated momentum equations include a baroclinic pressure term:
//!
//! ∂(hu)/∂t + ... = ... - (1/ρ₀) ∫₋ₕ⁰ ∂p'/∂x dz
//!
//! where p' is the pressure anomaly from density variations.
//!
//! For a depth-averaged model with density ρ(x, y, t), the baroclinic source is:
//!
//! S_hu = -g h² / (2ρ₀) * ∂ρ/∂x
//! S_hv = -g h² / (2ρ₀) * ∂ρ/∂y
//!
//! The h²/2 factor arises from vertical integration assuming a well-mixed water column.
//!
//! # Norwegian Coast Context
//!
//! Baroclinic effects are crucial in Norwegian fjords where:
//! - Fresh river water (low density) overlays saline Atlantic water
//! - Density fronts drive estuarine circulation
//! - Seasonal stratification affects mixing and current patterns
//!
//! Typical density variations:
//! - Fresh fjord surface: ρ ≈ 1005-1015 kg/m³
//! - Atlantic water: ρ ≈ 1027 kg/m³
//! - Density gradient: Δρ ≈ 10-20 kg/m³ over a few km

use crate::equations::{EquationOfState, LinearEquationOfState};
use crate::solver::{SWESolution2D, SWEState2D, TracerSolution2D, TracerState};

/// Baroclinic pressure source term.
///
/// Computes the horizontal pressure gradient from density variations
/// due to temperature and salinity differences.
///
/// # Example
/// ```ignore
/// use dg_rs::source::BaroclinicSource2D;
/// use dg_rs::equations::EquationOfState;
///
/// let eos = EquationOfState::new();
/// let baroclinic = BaroclinicSource2D::new(eos, 9.81, 1025.0, 1e-6);
///
/// // In RHS computation, you need to provide tracers and density gradients
/// ```
pub struct BaroclinicSource2D {
    /// Equation of state for density calculation
    eos: EquationOfState,
    /// Gravitational acceleration (m/s²)
    g: f64,
    /// Reference density (kg/m³)
    rho_0: f64,
    /// Minimum depth for wet/dry treatment
    h_min: f64,
}

impl BaroclinicSource2D {
    /// Create a new baroclinic source term.
    ///
    /// # Arguments
    /// * `eos` - Equation of state for computing density
    /// * `g` - Gravitational acceleration (typically 9.81 m/s²)
    /// * `rho_0` - Reference density (typically 1025 kg/m³)
    /// * `h_min` - Minimum depth threshold
    pub fn new(eos: EquationOfState, g: f64, rho_0: f64, h_min: f64) -> Self {
        Self {
            eos,
            g,
            rho_0,
            h_min,
        }
    }

    /// Create with default parameters for Norwegian coast.
    pub fn norwegian_coast() -> Self {
        Self {
            eos: EquationOfState::new(),
            g: 9.81,
            rho_0: 1025.0,
            h_min: 1e-3,
        }
    }

    /// Compute density at a point from tracer values.
    pub fn compute_density(&self, tracers: TracerState) -> f64 {
        self.eos.density_from_tracers(tracers)
    }

    /// Compute the baroclinic source term given density gradients.
    ///
    /// # Arguments
    /// * `h` - Water depth
    /// * `drho_dx` - Density gradient in x-direction (kg/m⁴)
    /// * `drho_dy` - Density gradient in y-direction (kg/m⁴)
    ///
    /// # Returns
    /// Source term (S_h, S_hu, S_hv)
    pub fn compute_source(&self, h: f64, drho_dx: f64, drho_dy: f64) -> SWEState2D {
        if h < self.h_min {
            return SWEState2D::zero();
        }

        // Baroclinic pressure gradient:
        // S_hu = -g h² / (2ρ₀) * ∂ρ/∂x
        // S_hv = -g h² / (2ρ₀) * ∂ρ/∂y
        let coeff = -self.g * h * h / (2.0 * self.rho_0);

        SWEState2D {
            h: 0.0,
            hu: coeff * drho_dx,
            hv: coeff * drho_dy,
        }
    }

    /// Compute baroclinic source from tracer values and their gradients.
    ///
    /// This is a convenience method that computes density gradients from
    /// tracer gradients using the equation of state, then applies the
    /// baroclinic pressure formula.
    ///
    /// # Arguments
    /// * `h` - Water depth
    /// * `temperature` - Temperature at this point (°C)
    /// * `salinity` - Salinity at this point (PSU)
    /// * `tracer_grads` - (dT/dx, dT/dy, dS/dx, dS/dy)
    pub fn compute_source_from_tracers(
        &self,
        h: f64,
        temperature: f64,
        salinity: f64,
        tracer_grads: (f64, f64, f64, f64),
    ) -> SWEState2D {
        if h < self.h_min {
            return SWEState2D::zero();
        }

        let (dt_dx, dt_dy, ds_dx, ds_dy) = tracer_grads;

        // Compute density derivatives from tracer derivatives using EOS
        // drho/dx = (∂ρ/∂T) * dT/dx + (∂ρ/∂S) * dS/dx
        let alpha = self.eos.thermal_expansion(temperature, salinity);
        let beta = self.eos.haline_contraction(temperature, salinity);

        // Note: alpha = -(1/ρ) ∂ρ/∂T, so ∂ρ/∂T = -α*ρ
        //       beta = (1/ρ) ∂ρ/∂S, so ∂ρ/∂S = β*ρ
        let rho = self
            .eos
            .density_from_tracers(TracerState::new(temperature, salinity));

        let drho_dx = -alpha * rho * dt_dx + beta * rho * ds_dx;
        let drho_dy = -alpha * rho * dt_dy + beta * rho * ds_dy;

        self.compute_source(h, drho_dx, drho_dy)
    }
}

/// Linear baroclinic source using simplified equation of state.
///
/// Uses the linear EOS: ρ = ρ₀(1 - α(T - T₀) + β(S - S₀))
///
/// This allows direct computation without full EOS evaluation:
/// - ∂ρ/∂x = ρ₀(-α ∂T/∂x + β ∂S/∂x)
///
/// Faster and often sufficiently accurate for coastal modeling.
pub struct LinearBaroclinicSource2D {
    /// Linear equation of state
    eos: LinearEquationOfState,
    /// Gravitational acceleration (m/s²)
    g: f64,
    /// Minimum depth for wet/dry treatment
    h_min: f64,
}

impl LinearBaroclinicSource2D {
    /// Create a new linear baroclinic source term.
    pub fn new(eos: LinearEquationOfState, g: f64, h_min: f64) -> Self {
        Self { eos, g, h_min }
    }

    /// Create with default parameters for Norwegian coast.
    pub fn norwegian_coast() -> Self {
        Self {
            eos: LinearEquationOfState::new(),
            g: 9.81,
            h_min: 1e-3,
        }
    }

    /// Compute the baroclinic source term from tracer gradients.
    ///
    /// Uses the linear EOS to directly compute density gradient from
    /// temperature and salinity gradients.
    ///
    /// # Arguments
    /// * `h` - Water depth
    /// * `dt_dx` - Temperature gradient in x (°C/m)
    /// * `dt_dy` - Temperature gradient in y (°C/m)
    /// * `ds_dx` - Salinity gradient in x (PSU/m)
    /// * `ds_dy` - Salinity gradient in y (PSU/m)
    pub fn compute_source_from_gradients(
        &self,
        h: f64,
        dt_dx: f64,
        dt_dy: f64,
        ds_dx: f64,
        ds_dy: f64,
    ) -> SWEState2D {
        if h < self.h_min {
            return SWEState2D::zero();
        }

        // From linear EOS: ∂ρ/∂x = ρ₀(-α ∂T/∂x + β ∂S/∂x)
        let drho_dx = self.eos.rho_0 * (-self.eos.alpha * dt_dx + self.eos.beta * ds_dx);
        let drho_dy = self.eos.rho_0 * (-self.eos.alpha * dt_dy + self.eos.beta * ds_dy);

        // Baroclinic pressure gradient
        let coeff = -self.g * h * h / (2.0 * self.eos.rho_0);

        SWEState2D {
            h: 0.0,
            hu: coeff * drho_dx,
            hv: coeff * drho_dy,
        }
    }
}

/// Extended source context that includes tracer information.
///
/// Used when evaluating coupled sources that need both SWE state and tracers.
#[derive(Clone, Copy, Debug)]
pub struct TracerSourceContext2D {
    /// Current simulation time
    pub time: f64,
    /// Physical position (x, y)
    pub position: (f64, f64),
    /// Current SWE state (h, hu, hv)
    pub swe_state: SWEState2D,
    /// Current tracer values (T, S)
    pub tracers: TracerState,
    /// Bathymetry (bottom elevation) at this point
    pub bathymetry: f64,
    /// Bathymetry gradients (∂B/∂x, ∂B/∂y)
    pub bathymetry_gradient: (f64, f64),
    /// Tracer gradients (∂T/∂x, ∂T/∂y, ∂S/∂x, ∂S/∂y)
    pub tracer_gradients: (f64, f64, f64, f64),
    /// Gravitational acceleration
    pub g: f64,
    /// Minimum depth threshold
    pub h_min: f64,
}

impl TracerSourceContext2D {
    /// Get velocity (u, v) with desingularization.
    pub fn velocity(&self) -> (f64, f64) {
        self.swe_state.velocity_simple(self.h_min)
    }

    /// Get water surface elevation (η = h + B).
    pub fn surface_elevation(&self) -> f64 {
        self.swe_state.h + self.bathymetry
    }
}

/// Trait for source terms that depend on tracer fields.
///
/// Extends the basic `SourceTerm2D` to include tracer information.
pub trait TracerSourceTerm2D: Send + Sync {
    /// Evaluate the source term contribution at a single node.
    fn evaluate(&self, ctx: &TracerSourceContext2D) -> SWEState2D;

    /// Name of this source term for debugging.
    fn name(&self) -> &'static str;
}

impl TracerSourceTerm2D for LinearBaroclinicSource2D {
    fn evaluate(&self, ctx: &TracerSourceContext2D) -> SWEState2D {
        let (dt_dx, dt_dy, ds_dx, ds_dy) = ctx.tracer_gradients;
        self.compute_source_from_gradients(ctx.swe_state.h, dt_dx, dt_dy, ds_dx, ds_dy)
    }

    fn name(&self) -> &'static str {
        "linear_baroclinic"
    }
}

/// Compute tracer gradients at a single node for baroclinic pressure calculation.
///
/// Uses the DG differentiation matrices to compute ∂T/∂x, ∂T/∂y, ∂S/∂x, ∂S/∂y.
///
/// # Arguments
/// * `tracer_sol` - Tracer solution (hT, hS)
/// * `swe_sol` - SWE solution (h, hu, hv) for depth
/// * `ops` - DG operators (differentiation matrices)
/// * `geom` - Geometric factors (Jacobian, etc.)
/// * `k` - Element index
/// * `i` - Node index within element
/// * `h_min` - Minimum depth threshold
///
/// # Returns
/// (∂T/∂x, ∂T/∂y, ∂S/∂x, ∂S/∂y)
#[allow(dead_code)]
pub fn compute_tracer_gradients_node(
    tracer_sol: &TracerSolution2D,
    swe_sol: &SWESolution2D,
    ops: &crate::operators::DGOperators2D,
    geom: &crate::operators::GeometricFactors2D,
    k: usize,
    i: usize,
    h_min: f64,
) -> (f64, f64, f64, f64) {
    // Get tracer concentrations at all nodes in this element
    let n_nodes = ops.n_nodes;

    // Compute T and S at each node
    let mut t_vals = vec![0.0; n_nodes];
    let mut s_vals = vec![0.0; n_nodes];

    for j in 0..n_nodes {
        let h = swe_sol.get_var(k, j, 0);
        let tracers = tracer_sol.get_concentrations(k, j, h, h_min);
        t_vals[j] = tracers.temperature;
        s_vals[j] = tracers.salinity;
    }

    // Compute derivatives in reference coordinates
    let mut dt_dr = 0.0;
    let mut dt_ds = 0.0;
    let mut ds_dr = 0.0;
    let mut ds_ds_val = 0.0;

    for j in 0..n_nodes {
        dt_dr += ops.dr[(i, j)] * t_vals[j];
        dt_ds += ops.ds[(i, j)] * t_vals[j];
        ds_dr += ops.dr[(i, j)] * s_vals[j];
        ds_ds_val += ops.ds[(i, j)] * s_vals[j];
    }

    // Transform to physical coordinates using geometric factors
    // ∂/∂x = rx * ∂/∂r + sx * ∂/∂s
    // ∂/∂y = ry * ∂/∂r + sy * ∂/∂s
    let rx = geom.rx[k];
    let ry = geom.ry[k];
    let sx = geom.sx[k];
    let sy = geom.sy[k];

    let dt_dx = rx * dt_dr + sx * dt_ds;
    let dt_dy = ry * dt_dr + sy * dt_ds;
    let ds_dx = rx * ds_dr + sx * ds_ds_val;
    let ds_dy = ry * ds_dr + sy * ds_ds_val;

    (dt_dx, dt_dy, ds_dx, ds_dy)
}

/// Compute tracer gradients for the entire domain.
///
/// Returns four vectors containing dT/dx, dT/dy, dS/dx, dS/dy at every node.
/// The vectors are indexed as `grad[k * n_nodes + i]` for element k, node i.
///
/// # Arguments
/// * `tracer_sol` - Tracer solution (hT, hS)
/// * `swe_sol` - SWE solution (h, hu, hv) for depth
/// * `mesh` - Computational mesh
/// * `ops` - DG operators (differentiation matrices)
/// * `geom` - Geometric factors (Jacobian, etc.)
/// * `h_min` - Minimum depth threshold
///
/// # Returns
/// (dT_dx, dT_dy, dS_dx, dS_dy) vectors for all nodes
pub fn compute_tracer_gradients(
    tracer_sol: &TracerSolution2D,
    swe_sol: &SWESolution2D,
    mesh: &crate::mesh::Mesh2D,
    ops: &crate::operators::DGOperators2D,
    geom: &crate::operators::GeometricFactors2D,
    h_min: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_nodes = ops.n_nodes;
    let total_nodes = mesh.n_elements * n_nodes;

    let mut dt_dx = vec![0.0; total_nodes];
    let mut dt_dy = vec![0.0; total_nodes];
    let mut ds_dx = vec![0.0; total_nodes];
    let mut ds_dy = vec![0.0; total_nodes];

    for k in 0..mesh.n_elements {
        // Get tracer concentrations at all nodes in this element
        let mut t_vals = vec![0.0; n_nodes];
        let mut s_vals = vec![0.0; n_nodes];

        for j in 0..n_nodes {
            let h = swe_sol.get_var(k, j, 0);
            let tracers = tracer_sol.get_concentrations(k, j, h, h_min);
            t_vals[j] = tracers.temperature;
            s_vals[j] = tracers.salinity;
        }

        // Geometric factors for this element
        let rx = geom.rx[k];
        let ry = geom.ry[k];
        let sx = geom.sx[k];
        let sy = geom.sy[k];

        // Compute gradients at each node
        for i in 0..n_nodes {
            let mut dt_dr = 0.0;
            let mut dt_ds_ref = 0.0;
            let mut ds_dr = 0.0;
            let mut ds_ds_ref = 0.0;

            for j in 0..n_nodes {
                dt_dr += ops.dr[(i, j)] * t_vals[j];
                dt_ds_ref += ops.ds[(i, j)] * t_vals[j];
                ds_dr += ops.dr[(i, j)] * s_vals[j];
                ds_ds_ref += ops.ds[(i, j)] * s_vals[j];
            }

            // Transform to physical coordinates
            let idx = k * n_nodes + i;
            dt_dx[idx] = rx * dt_dr + sx * dt_ds_ref;
            dt_dy[idx] = ry * dt_dr + sy * dt_ds_ref;
            ds_dx[idx] = rx * ds_dr + sx * ds_ds_ref;
            ds_dy[idx] = ry * ds_dr + sy * ds_ds_ref;
        }
    }

    (dt_dx, dt_dy, ds_dx, ds_dy)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_baroclinic_source_zero_gradient() {
        let source = BaroclinicSource2D::norwegian_coast();

        // Zero density gradient should give zero source
        let result = source.compute_source(10.0, 0.0, 0.0);

        assert!(result.h.abs() < TOL);
        assert!(result.hu.abs() < TOL);
        assert!(result.hv.abs() < TOL);
    }

    #[test]
    fn test_baroclinic_source_dry_cell() {
        let source = BaroclinicSource2D::norwegian_coast();

        // Dry cell should give zero source
        let result = source.compute_source(1e-6, 1.0, 1.0);

        assert!(result.h.abs() < TOL);
        assert!(result.hu.abs() < TOL);
        assert!(result.hv.abs() < TOL);
    }

    #[test]
    fn test_baroclinic_source_direction() {
        let source = BaroclinicSource2D::norwegian_coast();

        // Positive density gradient in x (heavier water to the right)
        // Should cause flow towards the denser water (positive hu source)
        // Actually: S_hu = -g h² / (2ρ₀) * ∂ρ/∂x
        // With positive ∂ρ/∂x, S_hu is negative
        let h = 10.0;
        let drho_dx = 0.001; // 1 kg/m³ per km

        let result = source.compute_source(h, drho_dx, 0.0);

        // Coefficient is negative, so positive drho_dx gives negative hu source
        assert!(result.hu < 0.0);
        assert!(result.hv.abs() < TOL);
    }

    #[test]
    fn test_baroclinic_source_magnitude() {
        let source = BaroclinicSource2D::norwegian_coast();

        let h = 10.0;
        let drho_dx = 0.01; // Strong gradient

        let result = source.compute_source(h, drho_dx, 0.0);

        // Expected: -9.81 * 100 / (2 * 1025) * 0.01 ≈ -0.0048
        let expected_hu = -9.81 * h * h / (2.0 * 1025.0) * drho_dx;
        assert!((result.hu - expected_hu).abs() < TOL);
    }

    #[test]
    fn test_linear_baroclinic_from_temperature() {
        let source = LinearBaroclinicSource2D::norwegian_coast();

        let h = 10.0;
        // Temperature increasing in x (warm anomaly to the right)
        // Warmer water is lighter, so this is like negative density gradient
        let dt_dx = 0.001; // 1°C per km

        let result = source.compute_source_from_gradients(h, dt_dx, 0.0, 0.0, 0.0);

        // With thermal expansion, warmer to right means lighter to right
        // This drives flow towards the light water (negative x), so positive hu source
        // Actually: drho/dx = -α ρ₀ dT/dx < 0 (negative)
        // S_hu = -g h² / (2ρ₀) * drho/dx > 0 (positive)
        assert!(result.hu > 0.0);
    }

    #[test]
    fn test_linear_baroclinic_from_salinity() {
        let source = LinearBaroclinicSource2D::norwegian_coast();

        let h = 10.0;
        // Salinity increasing in x (salty to the right)
        // Saltier water is heavier, so positive density gradient
        let ds_dx = 0.001; // 1 PSU per km

        let result = source.compute_source_from_gradients(h, 0.0, 0.0, ds_dx, 0.0);

        // drho/dx = β ρ₀ dS/dx > 0 (positive)
        // S_hu = -g h² / (2ρ₀) * drho/dx < 0 (negative)
        assert!(result.hu < 0.0);
    }

    #[test]
    fn test_linear_vs_full_baroclinic() {
        let full = BaroclinicSource2D::norwegian_coast();
        let linear = LinearBaroclinicSource2D::norwegian_coast();

        let h = 10.0;

        // At reference state, both should give similar results
        let tracers = TracerState::norwegian_coastal(); // T=8, S=34
        let rho = full.compute_density(tracers);

        // Small perturbations in T and S
        let dt = 1.0;
        let ds = 1.0;
        let dx = 1000.0; // 1 km

        let tracers_warm = TracerState::new(tracers.temperature + dt, tracers.salinity);
        let tracers_salty = TracerState::new(tracers.temperature, tracers.salinity + ds);

        let rho_warm = full.compute_density(tracers_warm);
        let rho_salty = full.compute_density(tracers_salty);

        let drho_dx_from_t = (rho_warm - rho) / dx;
        let drho_dx_from_s = (rho_salty - rho) / dx;

        let full_result_t = full.compute_source(h, drho_dx_from_t, 0.0);
        let full_result_s = full.compute_source(h, drho_dx_from_s, 0.0);

        let linear_result_t = linear.compute_source_from_gradients(h, dt / dx, 0.0, 0.0, 0.0);
        let linear_result_s = linear.compute_source_from_gradients(h, 0.0, 0.0, ds / dx, 0.0);

        // Both methods should give results with the correct sign
        // (linear is an approximation, so we don't expect exact agreement)
        assert!(
            full_result_t.hu * linear_result_t.hu > 0.0,
            "T contribution should have same sign"
        );
        assert!(
            full_result_s.hu * linear_result_s.hu > 0.0,
            "S contribution should have same sign"
        );

        // Both should give values in reasonable range (not orders of magnitude off)
        let ratio_t = full_result_t.hu / linear_result_t.hu;
        let ratio_s = full_result_s.hu / linear_result_s.hu;
        assert!(
            ratio_t > 0.3 && ratio_t < 3.0,
            "T ratio should be within factor of 3"
        );
        assert!(
            ratio_s > 0.3 && ratio_s < 3.0,
            "S ratio should be within factor of 3"
        );
    }
}
