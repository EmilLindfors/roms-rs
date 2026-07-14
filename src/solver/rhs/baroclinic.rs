//! Baroclinic pressure gradient computation.
//!
//! Implements the pressure gradient force in sigma coordinates using the standard formulation.
//!
//! The pressure gradient force is given by:
//!
//! ```text
//! F = -1/ρ₀ ∇p|_z
//! ```
//!
//! In sigma coordinates, using the chain rule:
//!
//! ```text
//! ∇p|_z = ∇p|_σ - (∂p/∂σ) (∇z|_σ / ∂z/∂σ)
//! ```
//!
//! Using the hydrostatic relation ∂p/∂z = -gρ:
//!
//! ```text
//! ∂p/∂σ = (∂p/∂z) (∂z/∂σ) = -gρ (∂z/∂σ)
//! ```
//!
//! Substituting this back:
//!
//! ```text
//! ∇p|_z = ∇p|_σ + gρ ∇z|_σ
//! ```
//!
//! where ∇z|_σ is the gradient of geopotential height along sigma surfaces.

use crate::mesh::data::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::state::Solution3D;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// Compute the sigma-coordinate pressure gradient force.
///
/// Output is stored in `grad_px` and `grad_py` with layout `[element][node][level]`.
/// The result is the force per unit mass: -1/ρ₀ ∇p.
///
/// The density used in the hydrostatic pressure and its gradient is `ρ − rho_ref`.
/// This lets the caller select which part of the PGF is computed:
///
/// - `rho_ref = 0` → **full** PGF (barotropic + baroclinic). The barotropic part
///   is exactly the free-surface term `−g∇η`.
/// - `rho_ref = ρ₀` → **baroclinic-only** PGF. The ρ₀ contribution integrates to
///   exactly `−g∇η`, so subtracting ρ₀ removes the barotropic term. Under mode
///   splitting the barotropic `−g∇η` is supplied by the 2D sub-model's `½gh²`
///   flux, so the 3D internal mode must use the baroclinic-only PGF to avoid
///   double-counting the surface pressure gradient (see `ModeSplitIntegrator`).
///
/// # Arguments
/// * `state` - 3D solution state (contains η and ρ)
/// * `bathymetry` - Bathymetry data (contains bed elevation B and grad B)
/// * `sigma` - Vertical grid configuration
/// * `ops` - 2D DG operators (for gradients)
/// * `geom` - Geometric factors (Jacobians)
/// * `g` - Gravitational acceleration (m/s²)
/// * `rho_0` - Reference density ρ₀ used in the `-1/ρ₀` normalization (kg/m³)
/// * `rho_ref` - Density subtracted from ρ before integrating (0 = full PGF, ρ₀ = baroclinic-only)
/// * `grad_px` - Output x-component of PGF (m/s²)
/// * `grad_py` - Output y-component of PGF (m/s²)
pub fn compute_pressure_gradient(
    state: &Solution3D,
    bathymetry: &Bathymetry2D,
    sigma: &SigmaGrid,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    g: f64,
    rho_0: f64,
    rho_ref: f64,
    grad_px: &mut [f64],
    grad_py: &mut [f64],
) {
    let n_elements = state.n_elements;
    let n_nodes = state.n_nodes;
    let n_levels = sigma.n_levels();

    // Workspaces for element-local calculations
    // P stored as [node * n_levels + level]
    let mut p_elem = vec![0.0; n_nodes * n_levels];

    // Eta gradients for the element
    let mut d_eta_dx = vec![0.0; n_nodes];
    let mut d_eta_dy = vec![0.0; n_nodes];

    // Workspace for derivatives (nodal values)
    let mut nodal_vals = vec![0.0; n_nodes];
    let mut dr_vals = vec![0.0; n_nodes];
    let mut ds_vals = vec![0.0; n_nodes];

    for k in 0..n_elements {
        let el_idx = ElementIndex::new(k);
        let k_base_idx = k * n_nodes; // Index into 2D arrays

        // 1. Compute gradients of eta for this element
        let eta_k = state.eta.element(el_idx);

        // Compute reference derivatives of eta
        for i in 0..n_nodes {
            let mut sum_r = 0.0;
            let mut sum_s = 0.0;
            for j in 0..n_nodes {
                sum_r += ops.dr[(i, j)] * eta_k[j];
                sum_s += ops.ds[(i, j)] * eta_k[j];
            }
            dr_vals[i] = sum_r;
            ds_vals[i] = sum_s;
        }

        // Transform to physical gradients
        // We need geometric factors for this element
        // geom arrays are per-element (affine assumption)
        let rx = geom.rx[k];
        let ry = geom.ry[k];
        let sx = geom.sx[k];
        let sy = geom.sy[k];

        for i in 0..n_nodes {
            d_eta_dx[i] = rx * dr_vals[i] + sx * ds_vals[i];
            d_eta_dy[i] = ry * dr_vals[i] + sy * ds_vals[i];
        }

        // 2. Compute Hydrostatic Pressure P(z) column by column
        for i in 0..n_nodes {
            let eta = eta_k[i];
            let total_depth = bathymetry.water_depth(el_idx, i, eta);

            // Get density column
            let rho_col = state.rho_column(el_idx, i);

            let mut p_cum = 0.0;

            // Integrate from top down
            // Using midpoint rule for each layer
            for level in (0..n_levels).rev() {
                // Layer thickness at this node
                // Note: sigma.d_sigma() returns slice of layer thicknesses in sigma space
                let dz = total_depth * sigma.d_sigma()[level];

                // Subtract the reference density so the caller can select the full
                // (rho_ref = 0) or baroclinic-only (rho_ref = ρ₀) pressure gradient.
                let rho = rho_col[level] - rho_ref;

                // Pressure at layer center
                // P_center = P_cumulative_above + weight_of_half_layer
                let p_local = p_cum + g * rho * dz * 0.5;

                // Store in p_elem [node_i, level]
                // Layout: i * n_levels + level
                p_elem[i * n_levels + level] = p_local;

                // Add full layer weight to cumulative pressure
                p_cum += g * rho * dz;
            }
        }

        // 3. Compute Gradients level by level
        for level in 0..n_levels {
            // Extract P at this level for all nodes
            for i in 0..n_nodes {
                nodal_vals[i] = p_elem[i * n_levels + level];
            }

            // Compute reference derivatives of P
            for i in 0..n_nodes {
                let mut sum_r = 0.0;
                let mut sum_s = 0.0;
                for j in 0..n_nodes {
                    sum_r += ops.dr[(i, j)] * nodal_vals[j];
                    sum_s += ops.ds[(i, j)] * nodal_vals[j];
                }
                dr_vals[i] = sum_r;
                ds_vals[i] = sum_s;
            }

            // Sigma coordinate at this level
            // Use rho-points (centers) for pressure gradient calculation
            let sigma_val = sigma.sigma_rho()[level];

            for i in 0..n_nodes {
                // Physical gradient of P along sigma surface
                let dp_dx_sigma = rx * dr_vals[i] + sx * ds_vals[i];
                let dp_dy_sigma = ry * dr_vals[i] + sy * ds_vals[i];

                // Gradient of z along sigma surface
                // z = eta + sigma * D, D = eta - B
                // grad z = grad eta + sigma * grad D
                let (d_depth_dx, d_depth_dy) =
                    bathymetry.water_depth_gradient(el_idx, i, d_eta_dx[i], d_eta_dy[i]);

                let dz_dx = d_eta_dx[i] + sigma_val * d_depth_dx;
                let dz_dy = d_eta_dy[i] + sigma_val * d_depth_dy;

                let rho = state.rho_column(el_idx, i)[level] - rho_ref;

                // PGF components: F = -1/rho0 * (∇p|_σ + gρ ∇z|_σ), with ρ here being
                // ρ − rho_ref so that rho_ref = ρ₀ yields the baroclinic-only PGF.
                let fx = -1.0 / rho_0 * (dp_dx_sigma + g * rho * dz_dx);
                let fy = -1.0 / rho_0 * (dp_dy_sigma + g * rho * dz_dy);

                // Store in output
                // Output layout: [element][node][level] which is same as Solution3D
                let out_idx = (k_base_idx + i) * n_levels + level;
                grad_px[out_idx] = fx;
                grad_py[out_idx] = fy;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh2D;
    use crate::vertical::UniformStretching;

    fn create_test_setup(
        nx: usize,
        ny: usize,
        n_levels: usize,
        order: usize,
    ) -> (
        Solution3D,
        Bathymetry2D,
        SigmaGrid,
        DGOperators2D,
        GeometricFactors2D,
        Mesh2D,
    ) {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1000.0, 0.0, 1000.0, nx, ny);
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh);
        let sigma = SigmaGrid::new(n_levels, UniformStretching);
        let state = Solution3D::new(mesh.n_elements, ops.n_nodes, n_levels);
        // Default flat bed elevation B = -100.0, i.e. 100 m depth at eta = 0.
        let bathymetry = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -100.0);

        (state, bathymetry, sigma, ops, geom, mesh)
    }

    #[test]
    fn test_pgf_constant_density_sloping_bottom() {
        let n_levels = 5;
        let (mut state, mut bathymetry, sigma, ops, geom, mesh) =
            create_test_setup(2, 1, n_levels, 1);

        // 1. Set sloping bed elevation B(x) = -100 - 0.1 * x.
        // Depth at eta = 0 is D(x) = 100 + 0.1 * x, and grad B = -0.1.
        let n_nodes = ops.n_nodes;
        for k in 0..mesh.n_elements {
            let k_idx = k * n_nodes;
            let el = ElementIndex::new(k);
            for i in 0..n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [x, _y] = mesh.reference_to_physical(el, r, s);
                bathymetry.data[k_idx + i] = -100.0 - 0.1 * x;
                bathymetry.gradient_x[k_idx + i] = -0.1;
                bathymetry.gradient_y[k_idx + i] = 0.0;
            }
        }

        // 2. Set constant density
        for val in state.rho.iter_mut() {
            *val = 1000.0;
        }
        state.eta.fill(0.0);

        // 3. Compute PGF
        let mut grad_px = vec![0.0; state.n_elements * n_nodes * n_levels];
        let mut grad_py = vec![0.0; state.n_elements * n_nodes * n_levels];

        compute_pressure_gradient(
            &state,
            &bathymetry,
            &sigma,
            &ops,
            &geom,
            9.81,
            1000.0,
            0.0, // full PGF
            &mut grad_px,
            &mut grad_py,
        );

        // 4. Verify PGF is zero
        let mut max_err = 0.0;
        for val in grad_px.iter().chain(grad_py.iter()) {
            if val.abs() > max_err {
                max_err = val.abs();
            }
        }
        // With standard sigma coordinate PGF, constant density + sloping bottom should be zero.
        assert!(max_err < 1e-12, "Max PGF error: {:.4e}", max_err);
    }

    #[test]
    fn baroclinic_only_pgf_excludes_surface_slope() {
        // Regression (TODO P0.6): with constant density ρ = ρ₀ and a tilted free
        // surface over a flat bottom, the FULL PGF must be the barotropic term
        // −g∇η, while the BAROCLINIC-ONLY PGF (rho_ref = ρ₀) must vanish. If the
        // baroclinic path still carried −g∇η, the mode-split G-term would double
        // the surface pressure gradient already supplied by the 2D ½gh² flux.
        let n_levels = 4;
        let rho0 = 1000.0;
        let g = 9.81;
        let slope = 1.0e-4; // eta rises 0.1 m over the 1000 m domain
        let (mut state, bathymetry, sigma, ops, geom, mesh) = create_test_setup(4, 1, n_levels, 1);
        let n_nodes = ops.n_nodes;

        // Tilted free surface eta = slope * x; constant density = rho0.
        for k in 0..mesh.n_elements {
            let el = ElementIndex::new(k);
            for i in 0..n_nodes {
                let [x, _y] = mesh.reference_to_physical(el, ops.nodes_r[i], ops.nodes_s[i]);
                state.eta.data[k * n_nodes + i] = slope * x;
            }
        }
        for val in state.rho.iter_mut() {
            *val = rho0;
        }

        let n = state.n_elements * n_nodes * n_levels;
        let mut full_x = vec![0.0; n];
        let mut full_y = vec![0.0; n];
        let mut bcl_x = vec![0.0; n];
        let mut bcl_y = vec![0.0; n];

        // Full PGF (rho_ref = 0) must equal -g*slope in x, ~0 in y.
        compute_pressure_gradient(
            &state,
            &bathymetry,
            &sigma,
            &ops,
            &geom,
            g,
            rho0,
            0.0,
            &mut full_x,
            &mut full_y,
        );
        let expected_fx = -g * slope;
        let full_x_err = full_x
            .iter()
            .map(|v| (v - expected_fx).abs())
            .fold(0.0_f64, f64::max);
        let full_y_err = full_y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(
            full_x_err < 1e-9,
            "full PGF should be -g∇η = {expected_fx:.3e}; max x error {full_x_err:.3e}"
        );
        assert!(full_y_err < 1e-9, "full PGF y should vanish, got {full_y_err:.3e}");

        // Baroclinic-only PGF (rho_ref = rho0) must vanish (density anomaly is 0).
        compute_pressure_gradient(
            &state,
            &bathymetry,
            &sigma,
            &ops,
            &geom,
            g,
            rho0,
            rho0,
            &mut bcl_x,
            &mut bcl_y,
        );
        let bcl_err = bcl_x
            .iter()
            .chain(bcl_y.iter())
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            bcl_err < 1e-12,
            "baroclinic-only PGF must exclude the barotropic -g∇η; max {bcl_err:.3e}"
        );
    }
}
