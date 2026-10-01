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
use crate::solver::core::blocks::{Pooled, for_each_block};
use crate::solver::state::Solution3D;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// Apply vertical mixing and diffusion to the 3D state.
///
/// The surface and bottom stresses enter as the momentum fluxes `τ/ρ₀` at the
/// column ends, so over `dt` the depth-integrated velocity of every column
/// changes by `dt·(τ_s − τ_b)/ρ₀`.
///
/// `bottom_drag`, if given, holds a linear drag rate `r` (m/s) per column
/// (`[element][node]`): the bottom flux `r·u_b` of the new bottom-layer
/// velocity joins `τ_b`, implicitly, so it cannot reverse the flow for any
/// `dt` (the quadratic drag linearised with `r = C_d|u_b|`, see
/// [`crate::physics::BottomDrag3D`]).
///
/// Columns shallower than `min_column_depth` (m) are left alone: they carry
/// no vertical structure (3D wetting and drying, see
/// [`crate::physics::Hydrostatic3D::with_min_column_depth`]), and the solve
/// would divide by vanishing layers.
#[allow(clippy::too_many_arguments)]
pub fn apply_vertical_diffusion<M: VerticalMixing + ?Sized>(
    state: &mut Solution3D,
    sigma: &SigmaGrid,
    bathymetry: &Bathymetry2D,
    dt: f64,
    mixing: &M,
    forcing: &Forcing,
    rho0: f64,
    min_column_depth: f64,
    bottom_drag: Option<&[f64]>,
) {
    let (nn, nl) = (state.n_nodes, state.n_levels);
    let Solution3D {
        eta,
        u,
        v,
        temp,
        salt,
        rho,
        eddy_viscosity,
        eddy_diffusivity,
        ..
    } = state;
    let (eta, rho): (&[f64], &[f64]) = (&eta.data, rho);
    let n = state.n_elements * nn * nl;
    for_each_block(
        state.n_elements,
        [
            &mut u[..n],
            &mut v[..n],
            &mut temp[..n],
            &mut salt[..n],
            &mut eddy_viscosity[..n],
            &mut eddy_diffusivity[..n],
        ],
        || {
            Pooled::take(
                |s: &ColumnScratch| s.z_r.len() == nl,
                || ColumnScratch::new(nl),
            )
        },
        |scratch, k, [u, v, temp, salt, eddy_viscosity, eddy_diffusivity]| {
            let ColumnScratch {
                a,
                b,
                c,
                d,
                x,
                c_prime,
                d_prime,
                z_r,
                z_w,
                dz,
                av,
                kt,
            } = &mut **scratch;
            for i in 0..nn {
                let idx = k * nn + i;
                // Still-water depth h = -B (bathymetry stores bed elevation B,
                // negative under water). The sigma routines form the total
                // column as eta + h, so eta + h = eta - B = water_depth,
                // matching the PGF/advection paths.
                let h = -bathymetry.get(ElementIndex::new(k), i);
                if eta[idx] + h < min_column_depth {
                    continue;
                }
                let (local, column) = (i * nl..(i + 1) * nl, idx * nl..(idx + 1) * nl);

                // 1. Prepare Column data
                sigma.z_at_levels_into(eta[idx], h, z_r);
                sigma.z_at_faces_into(eta[idx], h, z_w);
                sigma.layer_thicknesses_into(eta[idx], h, dz);

                // 2. Compute mixing coefficients
                mixing.compute_mixing_into(
                    &Column {
                        z_r,
                        z_w,
                        u: &u[local.clone()],
                        v: &v[local.clone()],
                        rho: &rho[column],
                    },
                    forcing,
                    av,
                    kt,
                );

                // Store diagnostics
                eddy_viscosity[local.clone()].copy_from_slice(&av[0..nl]);
                eddy_diffusivity[local.clone()].copy_from_slice(&kt[0..nl]);

                // 3. Solve diffusion: u, v with the stresses (kinematic) and
                // the drag, T with the surface buoyancy flux, S without
                let drag = bottom_drag.map_or(0.0, |rate| rate[idx]);
                let [tau_sx, tau_sy] = forcing.surface_stress;
                let [tau_bx, tau_by] = forcing.bottom_stress;
                for (phi, nu, flux_top, flux_bot, drag) in [
                    (
                        &mut u[local.clone()],
                        &*av,
                        tau_sx / rho0,
                        tau_bx / rho0,
                        drag,
                    ),
                    (
                        &mut v[local.clone()],
                        &*av,
                        tau_sy / rho0,
                        tau_by / rho0,
                        drag,
                    ),
                    (
                        &mut temp[local.clone()],
                        &*kt,
                        forcing.surface_buoyancy_flux,
                        0.0,
                        0.0,
                    ),
                    (&mut salt[local], &*kt, 0.0, 0.0, 0.0),
                ] {
                    solve_diffusion_column(
                        phi, nu, dz, dt, flux_top, flux_bot, drag, a, b, c, d, x, c_prime, d_prime,
                    );
                }
            }
        },
    );
}

/// One column's buffers of [`apply_vertical_diffusion`].
struct ColumnScratch {
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    /// Right-hand side
    d: Vec<f64>,
    /// Solution
    x: Vec<f64>,
    c_prime: Vec<f64>,
    d_prime: Vec<f64>,
    z_r: Vec<f64>,
    z_w: Vec<f64>,
    dz: Vec<f64>,
    /// Eddy viscosity and diffusivity at the w-points
    av: Vec<f64>,
    kt: Vec<f64>,
}

impl ColumnScratch {
    fn new(n_levels: usize) -> Self {
        let levels = || vec![0.0; n_levels];
        Self {
            a: levels(),
            b: levels(),
            c: levels(),
            d: levels(),
            x: levels(),
            c_prime: levels(),
            d_prime: levels(),
            z_r: levels(),
            z_w: vec![0.0; n_levels + 1],
            dz: levels(),
            av: vec![0.0; n_levels + 1],
            kt: vec![0.0; n_levels + 1],
        }
    }
}

/// One backward-Euler step of `∂φ/∂t = ∂/∂z(ν ∂φ/∂z)` in a column, with the
/// upward fluxes `flux_top` at the surface and `flux_bot + drag_bot·φ₀` at the
/// bed (`drag_bot·φ₀` at the new time).
#[allow(clippy::too_many_arguments)]
fn solve_diffusion_column(
    phi: &mut [f64],
    nu: &[f64],
    dz: &[f64],
    dt: f64,
    flux_top: f64,
    flux_bot: f64,
    drag_bot: f64,
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
    // Bottom: - lambda * Flux_{bot}, the drag part implicit
    let lambda_bot = dt / dz[0];
    d[0] -= lambda_bot * flux_bot;
    b[0] += lambda_bot * drag_bot;

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
            0.0,
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
            0.0,
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
        // Regression (2026-07-08 review §2.1 / TODO P0.5): `apply_vertical_diffusion`
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
            apply_vertical_diffusion(
                &mut state,
                &sigma,
                &bathymetry,
                dt,
                &mixing,
                &forcing,
                1025.0,
                0.0,
                None,
            );
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

    /// The stresses are momentum fluxes at the column ends: one implicit step
    /// changes the depth-integrated velocity by exactly `dt·(τ_s − τ_b)/ρ₀`,
    /// on a stretched grid and with any viscosity.
    #[test]
    fn column_momentum_changes_by_the_stress_impulse() {
        use crate::mesh::data::Bathymetry2D;
        use crate::types::ElementIndex;
        use crate::vertical::{SigmaGrid, SongHaidvogelStretching};

        let sigma = SigmaGrid::new(
            12,
            SongHaidvogelStretching {
                theta_s: 5.0,
                theta_b: 0.4,
                hc: 10.0,
            },
        );
        let mixing = ConstantMixing::new(0.02, 0.01);
        let forcing = Forcing {
            surface_stress: [0.12, -0.05],
            bottom_stress: [0.03, 0.01],
            surface_buoyancy_flux: 0.0,
        };
        let (dt, rho0, depth) = (300.0, 1027.0, 40.0);

        let mut state = Solution3D::new(1, 1, 12);
        let bathymetry = Bathymetry2D::constant(1, 1, -depth);
        let mut dz = vec![0.0; 12];
        sigma.layer_thicknesses_into(0.0, depth, &mut dz);
        let transport = |col: &[f64]| -> f64 { col.iter().zip(&dz).map(|(u, h)| u * h).sum() };
        apply_vertical_diffusion(
            &mut state,
            &sigma,
            &bathymetry,
            dt,
            &mixing,
            &forcing,
            rho0,
            0.0,
            None,
        );

        let el = ElementIndex::new(0);
        let du = transport(state.u_column(el, 0));
        let dv = transport(state.v_column(el, 0));
        let expect_u = dt * (0.12 - 0.03) / rho0;
        let expect_v = dt * (-0.05 - 0.01) / rho0;
        assert!(
            (du - expect_u).abs() < 1e-12 * expect_u.abs(),
            "{du} vs {expect_u}"
        );
        assert!(
            (dv - expect_v).abs() < 1e-12 * expect_v.abs(),
            "{dv} vs {expect_v}"
        );
    }

    /// The implicit bottom drag is a flux `r·u_b` of the *new* bottom-layer
    /// velocity: one step changes the column transport by exactly
    /// `−dt·r·u_bⁿ⁺¹`, and even at `r·dt/Δz_b = 10⁴` the bottom layer only
    /// slows down, it does not reverse.
    #[test]
    fn implicit_bottom_drag_removes_r_times_the_new_bottom_velocity() {
        use crate::mesh::data::Bathymetry2D;
        use crate::types::ElementIndex;
        use crate::vertical::{SigmaGrid, SongHaidvogelStretching};

        let n_levels = 8;
        let sigma = SigmaGrid::new(
            n_levels,
            SongHaidvogelStretching {
                theta_s: 3.0,
                theta_b: 0.4,
                hc: 10.0,
            },
        );
        let mixing = ConstantMixing::new(0.01, 0.0);
        let forcing = Forcing {
            surface_stress: [0.0, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_buoyancy_flux: 0.0,
        };
        let depth = 20.0;
        let mut dz = vec![0.0; n_levels];
        sigma.layer_thicknesses_into(0.0, depth, &mut dz);
        let transport = |col: &[f64]| -> f64 { col.iter().zip(&dz).map(|(u, h)| u * h).sum() };
        let bathymetry = Bathymetry2D::constant(1, 1, -depth);
        let el = ElementIndex::new(0);

        for (dt, rate) in [(60.0, 2.5e-3), (1e4, 1.0)] {
            let mut state = Solution3D::new(1, 1, n_levels);
            state.u.fill(0.8);
            state.v.fill(-0.3);
            let before = (transport(&state.u), transport(&state.v));
            apply_vertical_diffusion(
                &mut state,
                &sigma,
                &bathymetry,
                dt,
                &mixing,
                &forcing,
                1025.0,
                0.0,
                Some(&[rate]),
            );
            let (u, v) = (state.u_column(el, 0), state.v_column(el, 0));
            for (after, before, bottom) in [
                (transport(u), before.0, u[0]),
                (transport(v), before.1, v[0]),
            ] {
                let expect = -dt * rate * bottom;
                assert!(
                    (after - before - expect).abs() < 1e-12 * before.abs(),
                    "dt {dt}: transport change {} vs −dt·r·u_b = {expect}",
                    after - before
                );
                assert!(
                    bottom * before > 0.0 && bottom.abs() < before.abs() / depth,
                    "dt {dt}: bottom velocity {bottom} reversed or grew"
                );
            }
        }
    }
}
