//! Implicit vertical diffusion solver.
//!
//! Solves the vertical diffusion equation using a tridiagonal solver (Thomas algorithm).
//!
//! Equation:
//! ∂ϕ/∂t = ∂/∂z (K ∂ϕ/∂z)
//!
//! Discretization (Backward Euler in time, Centered in space):
//! (ϕ_k^{n+1} - ϕ_k^n) / Δt = 1/Δz_k [ K_{k+1/2} (ϕ_{k+1}^{n+1} - ϕ_k^{n+1}) / Δz_{k+1/2}
//!                                   - K_{k-1/2} (ϕ_k^{n+1} - ϕ_{k-1}^{n+1}) / Δz_{k-1/2} ]

use crate::mesh::data::Bathymetry2D;
use crate::physics::vertical_mixing::{Column, Forcing, VerticalMixing};
use crate::solver::algorithms::tridiagonal::solve_tridiagonal;
use crate::solver::state::Solution3D;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// Apply vertical mixing and diffusion to the 3D state.
pub fn apply_vertical_diffusion<M: VerticalMixing + ?Sized>(
    state: &mut Solution3D,
    sigma: &SigmaGrid,
    bathymetry: &Bathymetry2D,
    dt: f64,
    mixing: &M,
    forcing: &Forcing,
) {
    let n_levels = state.n_levels;
    let mut a = vec![0.0; n_levels];
    let mut b = vec![0.0; n_levels];
    let mut c = vec![0.0; n_levels];
    let mut d = vec![0.0; n_levels]; // RHS
    let mut x = vec![0.0; n_levels]; // Solution
    let mut c_prime = vec![0.0; n_levels];
    let mut d_prime = vec![0.0; n_levels];

    let mut z_r = vec![0.0; n_levels];
    let mut z_w = vec![0.0; n_levels + 1];
    let mut dz = vec![0.0; n_levels];

    for k in 0..state.n_elements {
        for i in 0..state.n_nodes {
            let elem_idx = ElementIndex::new(k);

            // 1. Prepare Column data
            let u = state.u_column(elem_idx, i);
            let v = state.v_column(elem_idx, i);
            let rho = state.rho_column(elem_idx, i);

            let eta = state.eta.get(k, i);
            // Still-water depth h = -B (bathymetry stores bed elevation B, negative
            // under water). The sigma routines form the total column as eta + h, so
            // eta + h = eta - B = water_depth, matching the PGF/advection paths.
            let h = -bathymetry.get(elem_idx, i);

            sigma.z_at_levels_into(eta, h, &mut z_r);
            sigma.z_at_faces_into(eta, h, &mut z_w);
            sigma.layer_thicknesses_into(eta, h, &mut dz);

            let column = Column {
                z_r: &z_r,
                z_w: &z_w,
                u,
                v,
                rho,
            };

            // 2. Compute mixing coefficients
            let (av, kt) = mixing.compute_mixing(&column, forcing);
            drop(column);

            // Store diagnostics
            state
                .eddy_viscosity_column_mut(elem_idx, i)
                .copy_from_slice(&av[0..n_levels]);
            state
                .eddy_diffusivity_column_mut(elem_idx, i)
                .copy_from_slice(&kt[0..n_levels]);

            // 3. Solve diffusion

            // U-momentum
            solve_diffusion_column(
                state.u_column_mut(elem_idx, i),
                &av,
                &dz,
                dt,
                forcing.surface_stress[0] / 1025.0, // kinematic
                forcing.bottom_stress[0] / 1025.0,
                &mut a,
                &mut b,
                &mut c,
                &mut d,
                &mut x,
                &mut c_prime,
                &mut d_prime,
            );

            // V-momentum
            solve_diffusion_column(
                state.v_column_mut(elem_idx, i),
                &av,
                &dz,
                dt,
                forcing.surface_stress[1] / 1025.0,
                forcing.bottom_stress[1] / 1025.0,
                &mut a,
                &mut b,
                &mut c,
                &mut d,
                &mut x,
                &mut c_prime,
                &mut d_prime,
            );

            // Temp
            solve_diffusion_column(
                state.temp_column_mut(elem_idx, i),
                &kt,
                &dz,
                dt,
                forcing.surface_buoyancy_flux,
                0.0,
                &mut a,
                &mut b,
                &mut c,
                &mut d,
                &mut x,
                &mut c_prime,
                &mut d_prime,
            );

            // Salt
            solve_diffusion_column(
                state.salt_column_mut(elem_idx, i),
                &kt,
                &dz,
                dt,
                0.0,
                0.0,
                &mut a,
                &mut b,
                &mut c,
                &mut d,
                &mut x,
                &mut c_prime,
                &mut d_prime,
            );
        }
    }
}

fn solve_diffusion_column(
    phi: &mut [f64],
    nu: &[f64],
    dz: &[f64],
    dt: f64,
    flux_top: f64,
    flux_bot: f64,
    a: &mut [f64],
    b: &mut [f64],
    c: &mut [f64],
    d: &mut [f64],
    x: &mut [f64],
    c_prime: &mut [f64],
    d_prime: &mut [f64],
) {
    let n = phi.len();

    // Reset arrays
    a.fill(0.0);
    b.fill(0.0);
    c.fill(0.0);

    for k in 0..n {
        let lambda = dt / dz[k];
        d[k] = phi[k]; // RHS starts with old value

        // Lower flux term: lambda * nu_{k-1/2} * (phi_k - phi_{k-1}) / dist
        // Corresponds to index k in nu (bottom is 0)
        let val_lower = if k == 0 {
            0.0 // Boundary
        } else {
            let dist = 0.5 * (dz[k] + dz[k - 1]);
            lambda * nu[k] / dist
        };

        // Upper flux term: lambda * nu_{k+1/2} * (phi_{k+1} - phi_k) / dist
        // Corresponds to index k+1 in nu
        let val_upper = if k == n - 1 {
            0.0 // Boundary
        } else {
            let dist = 0.5 * (dz[k] + dz[k + 1]);
            lambda * nu[k + 1] / dist
        };

        a[k] = -val_lower;
        c[k] = -val_upper;
        b[k] = 1.0 + val_lower + val_upper;
    }

    // Apply Boundary Conditions to RHS
    // Bottom: - lambda * Flux_{bot}
    let lambda_bot = dt / dz[0];
    d[0] -= lambda_bot * flux_bot;

    // Top: + lambda * Flux_{top}
    let lambda_top = dt / dz[n - 1];
    d[n - 1] += lambda_top * flux_top;

    solve_tridiagonal(a, b, c, d, x, c_prime, d_prime);

    // Copy result back
    for i in 0..n {
        phi[i] = x[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::vertical_mixing::ConstantMixing;

    #[test]
    fn test_diffusion_constant() {
        // Test that constant profile with zero flux remains constant
        let n = 10;
        let mut phi = vec![1.0; n];
        let nu = vec![0.1; n + 1];
        let dz = vec![1.0; n];
        let dt = 1.0;
        let flux_top = 0.0;
        let flux_bot = 0.0;

        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut c = vec![0.0; n];
        let mut d = vec![0.0; n];
        let mut x = vec![0.0; n];
        let mut c_prime = vec![0.0; n];
        let mut d_prime = vec![0.0; n];

        solve_diffusion_column(
            &mut phi,
            &nu,
            &dz,
            dt,
            flux_top,
            flux_bot,
            &mut a,
            &mut b,
            &mut c,
            &mut d,
            &mut x,
            &mut c_prime,
            &mut d_prime,
        );

        for v in phi {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_diffusion_linear_flux() {
        // Linear profile phi = z.
        // dphi/dz = 1.
        // Flux = -nu * dphi/dz = -0.1 * 1 = -0.1.
        // If we apply flux BCs consistent with this, should stay linear?
        // Wait, steady state of diffusion equation d/dz(nu dphi/dz) = 0 is linear profile (for constant nu).
        // So if we initialize with linear profile and apply correct fluxes, it should be steady.

        let n = 5;
        let dz_val = 1.0;
        // Centers at 0.5, 1.5, 2.5, 3.5, 4.5
        let mut phi: Vec<f64> = (0..n).map(|k| (k as f64 + 0.5) * dz_val).collect();
        let nu = vec![1.0; n + 1];
        let dz = vec![dz_val; n];
        let dt = 0.1;

        // Flux = - nu * dphi/dz.
        // dphi/dz = 1.0.
        // Flux = -1.0.
        // Flux is upward positive in code convention?
        // Eq: dphi/dt = d/dz(nu dphi/dz).
        // Flux = nu dphi/dz.
        // If phi increasing upwards, flux is positive upwards.
        // So Flux = 1.0 * 1.0 = 1.0.

        let flux_top = 1.0; // Positive upward flux
        let flux_bot = 1.0; // Positive upward flux

        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut c = vec![0.0; n];
        let mut d = vec![0.0; n];
        let mut x = vec![0.0; n];
        let mut c_prime = vec![0.0; n];
        let mut d_prime = vec![0.0; n];

        let phi_orig = phi.clone();

        solve_diffusion_column(
            &mut phi,
            &nu,
            &dz,
            dt,
            flux_top,
            flux_bot,
            &mut a,
            &mut b,
            &mut c,
            &mut d,
            &mut x,
            &mut c_prime,
            &mut d_prime,
        );

        for i in 0..n {
            assert!(
                (phi[i] - phi_orig[i]).abs() < 1e-10,
                "Changed at {}: {} -> {}",
                i,
                phi_orig[i],
                phi[i]
            );
        }
    }

    #[test]
    fn diffusion_layer_thickness_is_depth_dependent() {
        // Regression (REVIEW.md §2.1 / TODO P0.5): `apply_vertical_diffusion`
        // must build layer thicknesses from the actual water depth (eta - B),
        // not a hardcoded 100 m. Two columns driven by identical surface stress
        // but with 50 m vs 500 m of water respond differently, because the
        // top-layer thickness (and thus the implicit-solve coefficients) scale
        // with depth. Before the fix both used h = 100 and were identical.
        use crate::mesh::data::Bathymetry2D;
        use crate::types::ElementIndex;
        use crate::vertical::SigmaGrid;

        let n_levels = 10;
        let sigma = SigmaGrid::uniform(n_levels);
        let mixing = ConstantMixing::new(0.1, 0.01);
        let forcing = Forcing {
            surface_stress: [0.1, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_buoyancy_flux: 0.0,
        };
        let dt = 100.0;

        let run = |bed_elevation: f64| {
            let mut state = Solution3D::new(1, 1, n_levels);
            let bathymetry = Bathymetry2D::constant(1, 1, bed_elevation);
            apply_vertical_diffusion(&mut state, &sigma, &bathymetry, dt, &mixing, &forcing);
            state.u_column(ElementIndex::new(0), 0).to_vec()
        };

        let shallow = run(-50.0); // 50 m water column
        let deep = run(-500.0); // 500 m water column

        let max_diff = shallow
            .iter()
            .zip(deep.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        assert!(
            max_diff > 1e-6,
            "50 m and 500 m columns gave identical velocity profiles \
             (max diff {max_diff:.3e}); depth is likely still hardcoded"
        );
    }
}
