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
use crate::physics::vertical_mixing::{Column, Forcing, Turbulence, VerticalMixing};
use crate::solver::algorithms::tridiagonal::solve_tridiagonal;
use crate::solver::core::blocks::{Pooled, for_each_block};
use crate::solver::state::Solution3D;
use crate::time::StepDrag;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// Apply vertical mixing and diffusion to the 3D state.
///
/// The surface and bottom stresses enter as the momentum fluxes `τ/ρ₀` at the
/// column ends, so over `dt` the depth-integrated velocity of every column
/// changes by `dt·(τ_s − τ_b)/ρ₀`. The surface stress of a column is
/// `forcing.surface_stress` plus, if given, its entry of `surface_stress`
/// (`[τ_x, τ_y]`, N/m², one per column, `[element][node]`); the closure
/// receives the column's own [`Forcing`] and friction velocity.
///
/// `drag.bottom`, if given, holds a linear drag rate `r` (m/s) per column
/// (`[element][node]`): the bottom flux `r·u_b` of the new bottom-layer
/// velocity joins `τ_b`, implicitly, so it cannot reverse the flow for any
/// `dt` (the quadratic drag linearised with `r = C_d|u_b|`, see
/// [`crate::physics::BottomDrag3D`]). `drag.layers`, if given, holds a
/// linear drag rate `λ_l` (1/s) per layer (`[element][node][level]`): the
/// sink `−λ_l u_l` of the new velocity, implicitly (net cages, see
/// [`crate::physics::cage_drag`]).
///
/// Columns shallower than `min_column_depth` (m) are left alone: they carry
/// no vertical structure (3D wetting and drying, see
/// [`crate::physics::Hydrostatic3D::with_min_column_depth`]), and the solve
/// would divide by vanishing layers.
///
/// A closure with prognostic turbulence ([`VerticalMixing::initial_turbulence`],
/// e.g. [`crate::physics::GlsMixing`]) steps `state.tke` and `state.gls` over
/// `dt` first, in each column, from the shear and stratification before the
/// solve and the friction velocities of the surface stress and of the bottom
/// stress plus drag (`g` enters `N²`); the solve then uses the new viscosity
/// and diffusivity. The turbulence is allocated, at the closure's initial
/// value, on the first call.
#[allow(clippy::too_many_arguments)]
pub fn apply_vertical_diffusion<M: VerticalMixing + ?Sized>(
    state: &mut Solution3D,
    sigma: &SigmaGrid,
    bathymetry: &Bathymetry2D,
    dt: f64,
    mixing: &M,
    forcing: &Forcing,
    surface_stress: Option<[&[f64]; 2]>,
    g: f64,
    rho0: f64,
    min_column_depth: f64,
    drag: StepDrag<'_>,
) {
    let (nn, nl) = (state.n_nodes, state.n_levels);
    let n_w = state.n_elements * nn * (nl + 1);
    if let Some([k0, psi0]) = mixing.initial_turbulence()
        && state.tke.len() != n_w
    {
        state.tke = vec![k0; n_w];
        state.gls = vec![psi0; n_w];
    }
    let Solution3D {
        eta,
        u,
        v,
        temp,
        salt,
        rho,
        eddy_viscosity,
        eddy_diffusivity,
        tke,
        gls,
        ..
    } = state;
    let (eta, rho): (&[f64], &[f64]) = (&eta.data, rho);
    let n = state.n_elements * nn * nl;
    let [tau_bx, tau_by] = forcing.bottom_stress;
    for_each_block(
        state.n_elements,
        [
            &mut u[..n],
            &mut v[..n],
            &mut temp[..n],
            &mut salt[..n],
            &mut eddy_viscosity[..n_w],
            &mut eddy_diffusivity[..n_w],
            &mut tke[..],
            &mut gls[..],
        ],
        || {
            Pooled::take(
                |s: &ColumnScratch| s.z_r.len() == nl,
                || ColumnScratch::new(nl),
            )
        },
        |scratch, k, [u, v, temp, salt, eddy_viscosity, eddy_diffusivity, tke, gls]| {
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
                work,
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

                // 2. Step the turbulence, if any, and compute the mixing
                // coefficients. The surface stress: the uniform one plus the
                // column's; the bottom stress: the prescribed one plus the
                // drag of the bottom-layer velocity
                let [mut tau_sx, mut tau_sy] = forcing.surface_stress;
                if let Some([field_x, field_y]) = surface_stress {
                    tau_sx += field_x[idx];
                    tau_sy += field_y[idx];
                }
                let forcing = Forcing {
                    surface_stress: [tau_sx, tau_sy],
                    ..*forcing
                };
                let surface_friction_velocity = (tau_sx.hypot(tau_sy) / rho0).sqrt();
                let bottom_drag = drag.bottom.map_or(0.0, |rate| rate[idx]);
                let layer_drag = drag.layers.map(|rate| &rate[column.clone()]);
                let bottom_friction_velocity = ((tau_bx / rho0 + bottom_drag * u[i * nl])
                    .hypot(tau_by / rho0 + bottom_drag * v[i * nl]))
                .sqrt();
                let turbulence_column = if tke.is_empty() {
                    0..0
                } else {
                    i * (nl + 1)..(i + 1) * (nl + 1)
                };
                mixing.step_mixing_into(
                    &Column {
                        z_r,
                        z_w,
                        u: &u[local.clone()],
                        v: &v[local.clone()],
                        rho: &rho[column],
                        g,
                        rho0,
                        surface_friction_velocity,
                        bottom_friction_velocity,
                    },
                    &forcing,
                    dt,
                    Turbulence {
                        tke: &mut tke[turbulence_column.clone()],
                        gls: &mut gls[turbulence_column],
                        scratch: work,
                    },
                    av,
                    kt,
                );

                // Store diagnostics, at the w-points
                let w_points = i * (nl + 1)..(i + 1) * (nl + 1);
                eddy_viscosity[w_points.clone()].copy_from_slice(av);
                eddy_diffusivity[w_points].copy_from_slice(kt);

                // 3. Solve diffusion: u, v with the stresses (kinematic) and
                // the drags, T with the surface buoyancy flux, S without
                for (phi, nu, flux_top, flux_bot, bottom_drag, layer_drag) in [
                    (
                        &mut u[local.clone()],
                        &*av,
                        tau_sx / rho0,
                        tau_bx / rho0,
                        bottom_drag,
                        layer_drag,
                    ),
                    (
                        &mut v[local.clone()],
                        &*av,
                        tau_sy / rho0,
                        tau_by / rho0,
                        bottom_drag,
                        layer_drag,
                    ),
                    (
                        &mut temp[local.clone()],
                        &*kt,
                        forcing.surface_buoyancy_flux,
                        0.0,
                        0.0,
                        None,
                    ),
                    (&mut salt[local], &*kt, 0.0, 0.0, 0.0, None),
                ] {
                    solve_diffusion_column(
                        phi,
                        nu,
                        dz,
                        dt,
                        flux_top,
                        flux_bot,
                        bottom_drag,
                        layer_drag,
                        [a, b, c, d, x, c_prime, d_prime],
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
    /// The closure's work space ([`Turbulence::scratch`])
    work: Vec<f64>,
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
            work: Vec::new(),
        }
    }
}

/// One backward-Euler step of `∂φ/∂t = ∂/∂z(ν ∂φ/∂z) − λ_l φ` in a column,
/// with the upward fluxes `flux_top` at the surface and `flux_bot +
/// drag_bot·φ₀` at the bed (`drag_bot·φ₀` and `λ_l φ` at the new time,
/// `λ_l` from `drag_layers`, zero without). `work` holds the tridiagonal
/// system and the solver's buffers.
#[allow(clippy::too_many_arguments)]
fn solve_diffusion_column(
    phi: &mut [f64],
    nu: &[f64],
    dz: &[f64],
    dt: f64,
    flux_top: f64,
    flux_bot: f64,
    drag_bot: f64,
    drag_layers: Option<&[f64]>,
    [a, b, c, d, x, c_prime, d_prime]: [&mut [f64]; 7],
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
        b[k] = 1.0 + val_lower + val_upper + drag_layers.map_or(0.0, |rate| dt * rate[k]);
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

    phi.copy_from_slice(x);
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
            None,
            [
                &mut a,
                &mut b,
                &mut c,
                &mut d,
                &mut x,
                &mut c_prime,
                &mut d_prime,
            ],
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
            None,
            [
                &mut a,
                &mut b,
                &mut c,
                &mut d,
                &mut x,
                &mut c_prime,
                &mut d_prime,
            ],
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
                None,
                9.81,
                1025.0,
                0.0,
                StepDrag::NONE,
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
            None,
            9.81,
            rho0,
            0.0,
            StepDrag::NONE,
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
                None,
                9.81,
                1025.0,
                0.0,
                StepDrag::bottom(&[rate]),
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

/// Gates of the GLS closure ([`crate::physics::GlsMixing`]) in single
/// columns stepped by [`apply_vertical_diffusion`].
#[cfg(test)]
mod gls_gates {
    use super::*;
    use crate::physics::bottom_drag::BottomDrag3D;
    use crate::physics::eos::{EquationOfState, LinearEOS};
    use crate::physics::gls::GlsMixing;

    const G: f64 = 9.81;

    /// Wind entrainment into a linearly stratified layer (Kato & Phillips
    /// 1969): the mixed layer deepens as `D = 1.05 u* √(t/N₀)` (Price 1979),
    /// the classic GLS benchmark (Umlauf & Burchard 2005, §6.2). `D` is the
    /// depth where `k` first falls below 10⁻⁵ m²/s² (Burchard & Bolding
    /// 2001). Measured with k-ε at 30 h: 1.014 × Price's depth at Δt = 10 s
    /// and 60 s alike (k-ω 1.000, generic 1.014). The entrainment is shear
    /// driven: breaking waves with a wave height's roughness (z₀ = 0.5 m)
    /// leave it at 1.000 / 1.000 / 1.014. Heat is conserved and stays within
    /// its initial range.
    #[test]
    fn wind_mixed_layer_deepens_at_the_kato_phillips_rate() {
        let (depth, n_levels, u_star, n0, dt) = (50.0, 100, 0.01, 0.01, 60.0);
        let eos = LinearEOS::default();
        let rho0 = eos.rho0;
        let sigma = SigmaGrid::uniform(n_levels);
        let bathymetry = Bathymetry2D::constant(1, 1, -depth);
        let forcing = Forcing {
            surface_stress: [rho0 * u_star * u_star, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_buoyancy_flux: 0.0,
        };
        let mut z_w = vec![0.0; n_levels + 1];
        sigma.z_at_faces_into(0.0, depth, &mut z_w);
        let dtdz = n0 * n0 / (G * eos.alpha);

        let waves = |gls: GlsMixing| {
            gls.with_roughness(0.5, GlsMixing::DEFAULT_ROUGHNESS)
                .with_wave_breaking(GlsMixing::DEFAULT_WAVE_BREAKING)
        };
        for (name, gls) in [
            ("k-ε", GlsMixing::k_epsilon()),
            ("k-ω", GlsMixing::k_omega()),
            ("generic", GlsMixing::generic()),
            ("k-ε, waves", waves(GlsMixing::k_epsilon())),
            ("k-ω, waves", waves(GlsMixing::k_omega())),
            ("generic, waves", waves(GlsMixing::generic())),
        ] {
            let mut state = Solution3D::new(1, 1, n_levels);
            for (l, &s) in sigma.sigma_rho().iter().enumerate() {
                state.temp[l] = eos.t0 + dtdz * s * depth;
                state.salt[l] = eos.s0;
            }
            eos.update_density(&mut state);
            let heat = |state: &Solution3D| -> f64 { state.temp.iter().sum() };
            let heat0 = heat(&state);
            let (t_min, t_max) = (state.temp[0], state.temp[n_levels - 1]);

            let mut t = 0.0;
            for hours in [12.0, 24.0, 30.0] {
                while t < hours * 3600.0 - 1e-9 {
                    apply_vertical_diffusion(
                        &mut state,
                        &sigma,
                        &bathymetry,
                        dt,
                        &gls,
                        &forcing,
                        None,
                        G,
                        rho0,
                        0.0,
                        StepDrag::NONE,
                    );
                    eos.update_density(&mut state);
                    t += dt;
                }
                let mixed = (0..=n_levels)
                    .rev()
                    .find(|&j| state.tke[j] < 1e-5)
                    .map_or(depth, |j| -z_w[j]);
                let price = 1.05 * u_star * (t / n0).sqrt();
                assert!(
                    (mixed / price - 1.0).abs() < 0.05,
                    "{name} at {hours} h: mixed layer {mixed} m against Price's {price:.2} m"
                );
            }
            assert!(
                (heat(&state) - heat0).abs() < 1e-12 * heat0,
                "{name}: heat changed by {}",
                heat(&state) - heat0
            );
            assert!(
                state
                    .temp
                    .iter()
                    .all(|&t| t > t_min - 1e-12 && t < t_max + 1e-12),
                "{name}: temperature left its range [{t_min}, {t_max}]"
            );
            assert!(
                state.tke.iter().chain(&state.gls).all(|&x| x > 0.0),
                "{name}: k or ψ not positive"
            );
        }
    }

    /// Breaking waves (Craig & Banner 1994) under wind against a log-layer
    /// drag, 10 m deep on 20 levels of 0.5 m, to steady state, against the
    /// surface log layer:
    /// - With a wave height's roughness (z₀ = 0.5 m) the wave-affected layer
    ///   reaches the top interior w-points. k-ω and the generic model:
    ///   measured `k` 4.4 / 4.3× the log layer's at the top one, 2.1 / 2.3×
    ///   at the next, mid-depth within 3 %; the diffusivity there 2.9 / 2.3×
    ///   the log surface's; the top w-point at 90 % of `u*²/(c_μ⁰)²` after
    ///   one step instead of 15 / 31 min. k-ε's shear-free length is short
    ///   (`l ≈ 0.09 s` against κs), so its diffusivity there drops to
    ///   0.35× with 1.3× the `k`: use k-ω or the generic model with waves.
    /// - With Charnock's roughness at u* = 1 cm/s (1.4 cm, so the 2 cm
    ///   minimum), the wave-affected layer (≈ 2–4 z₀) is thinner than half
    ///   the top layer and the column is the log layer's (`k` to 0.6 %, the
    ///   diffusivity to 3 %).
    #[test]
    fn breaking_waves_raise_the_near_surface_mixing_under_wind() {
        let (depth, n_levels, u_star, z0b, dt, rho0) = (10.0, 20, 0.01, 0.02, 60.0, 1025.0);
        let sigma = SigmaGrid::uniform(n_levels);
        let bathymetry = Bathymetry2D::constant(1, 1, -depth);
        let z_b = (1.0 + sigma.sigma_rho()[0]) * depth;
        let forcing = Forcing {
            surface_stress: [rho0 * u_star * u_star, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_buoyancy_flux: 0.0,
        };
        let drag = BottomDrag3D::log_layer(z0b).with_bounds(0.0, 1.0);
        let top = n_levels - 1;
        // The steady column, and the steps the top interior w-point takes to
        // 90 % of the log layer's k
        let run = |gls: &GlsMixing| {
            let k_log = u_star * u_star / gls.cm0().powi(2);
            let mut state = Solution3D::new(1, 1, n_levels);
            state.rho.fill(rho0);
            let mut spin_up = None;
            for step in 1..=(24.0 * 3600.0 / dt) as usize {
                let rate = drag.rate(z_b, state.u[0].abs());
                apply_vertical_diffusion(
                    &mut state,
                    &sigma,
                    &bathymetry,
                    dt,
                    gls,
                    &forcing,
                    None,
                    G,
                    rho0,
                    0.0,
                    StepDrag::bottom(&[rate]),
                );
                if spin_up.is_none() && state.tke[top] > 0.9 * k_log {
                    spin_up = Some(step);
                }
            }
            (
                state,
                spin_up.expect("the top never reached the log layer's k"),
            )
        };
        for gls in [GlsMixing::k_omega(), GlsMixing::generic()] {
            let gls = gls.with_background(0.0, 0.0);
            let name = format!("{:?}", gls.parameters());
            let k_log = u_star * u_star / gls.cm0().powi(2);
            let (log, log_spin_up) = run(&gls);
            let (waves, spin_up) = run(&gls
                .clone()
                .with_roughness(0.5, GlsMixing::DEFAULT_ROUGHNESS)
                .with_wave_breaking(GlsMixing::DEFAULT_WAVE_BREAKING));
            assert!(
                waves.tke[top] > 4.0 * k_log && waves.tke[top - 1] > 2.0 * k_log,
                "{name}: k/k_log {} and {} at the top w-points",
                waves.tke[top] / k_log,
                waves.tke[top - 1] / k_log
            );
            assert!(
                (waves.tke[n_levels / 2] / k_log - 1.0).abs() < 0.03,
                "{name}: mid-depth k/k_log {}",
                waves.tke[n_levels / 2] / k_log
            );
            assert!(
                waves.eddy_diffusivity[top] > 2.0 * log.eddy_diffusivity[top],
                "{name}: diffusivity {} under waves against {}",
                waves.eddy_diffusivity[top],
                log.eddy_diffusivity[top]
            );
            assert!(
                spin_up == 1 && log_spin_up >= 10,
                "{name}: spin-up {spin_up} steps under waves, {log_spin_up} without"
            );

            let (charnock, _) = run(&gls
                .clone()
                .with_charnock_roughness(GlsMixing::DEFAULT_CHARNOCK)
                .with_wave_breaking(GlsMixing::DEFAULT_WAVE_BREAKING));
            for j in 1..n_levels {
                let (k_ratio, kt_ratio) = (
                    charnock.tke[j] / log.tke[j],
                    charnock.eddy_diffusivity[j] / log.eddy_diffusivity[j],
                );
                assert!(
                    (k_ratio - 1.0).abs() < 0.01 && (kt_ratio - 1.0).abs() < 0.05,
                    "{name}, w-point {j}: k {k_ratio}, diffusivity {kt_ratio} × the log layer's"
                );
            }
        }
        let gls = GlsMixing::k_epsilon().with_background(0.0, 0.0);
        let (log, _) = run(&gls);
        let (waves, _) = run(&gls
            .with_roughness(0.5, GlsMixing::DEFAULT_ROUGHNESS)
            .with_wave_breaking(GlsMixing::DEFAULT_WAVE_BREAKING));
        let ratio = waves.eddy_diffusivity[top] / log.eddy_diffusivity[top];
        assert!(
            ratio < 0.5 && waves.tke[top] > log.tke[top],
            "k-ε: diffusivity {ratio} × the log layer's under waves"
        );
    }

    /// Open-channel flow driven by a body force against a log-layer drag,
    /// to steady state: the bottom stress is `u*²`, `k` is in local
    /// equilibrium with the linear stress, `u*²(1 − z/H)/(c_μ⁰)²`, and the
    /// length at the first w-point approaches the wall's `κz` as the levels
    /// refine (0.86, 0.90, 0.96 at 20, 40, 80 levels). Further out k-ε's
    /// viscosity is ≈ 0.8 of the parabolic `κu*z(1 − z/H)`, as k-ε has in
    /// open channels; that is not gated.
    #[test]
    fn open_channel_turbulence_reaches_the_log_layer_equilibrium() {
        let (depth, u_star, z0, dt, rho0) = (10.0, 0.03, 0.01, 30.0, 1025.0);
        let body_force = u_star * u_star / depth;
        let bathymetry = Bathymetry2D::constant(1, 1, -depth);
        let forcing = Forcing {
            surface_stress: [0.0, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_buoyancy_flux: 0.0,
        };
        let drag = BottomDrag3D::log_layer(z0).with_bounds(0.0, 1.0);
        let gls = GlsMixing::k_epsilon()
            .with_roughness(GlsMixing::DEFAULT_ROUGHNESS, z0)
            .with_background(0.0, 0.0);
        let (cm0, kappa) = (gls.cm0(), gls.kappa());

        let mut wall_length = 0.0;
        for n_levels in [20, 40, 80] {
            let sigma = SigmaGrid::uniform(n_levels);
            let mut z_w = vec![0.0; n_levels + 1];
            sigma.z_at_faces_into(0.0, depth, &mut z_w);
            let z_b = (1.0 + sigma.sigma_rho()[0]) * depth;
            let mut state = Solution3D::new(1, 1, n_levels);
            state.rho.fill(rho0);
            for _ in 0..(24.0 * 3600.0 / dt) as usize {
                // The drag rate at tⁿ, as the mode splitter freezes it
                let rate = drag.rate(z_b, state.u[0].abs());
                for u in &mut state.u {
                    *u += dt * body_force;
                }
                apply_vertical_diffusion(
                    &mut state,
                    &sigma,
                    &bathymetry,
                    dt,
                    &gls,
                    &forcing,
                    None,
                    G,
                    rho0,
                    0.0,
                    StepDrag::bottom(&[rate]),
                );
            }

            let stress = drag.rate(z_b, state.u[0].abs()) * state.u[0];
            assert!(
                (stress / (u_star * u_star) - 1.0).abs() < 1e-6,
                "{n_levels} levels: bottom stress {stress} against u*² {}",
                u_star * u_star
            );
            for (k, z_w) in state.tke.iter().zip(&z_w).take(n_levels / 2 + 1).skip(1) {
                let z = z_w + depth;
                let equilibrium = u_star * u_star * (1.0 - z / depth) / (cm0 * cm0);
                assert!(
                    (k / equilibrium - 1.0).abs() < 0.03,
                    "{n_levels} levels, z = {z} m: k {k} against {equilibrium}"
                );
            }
            // k-ε: ψ = ε
            let length = cm0.powi(3) * state.tke[1].powf(1.5) / state.gls[1];
            let ratio = length / (kappa * (z_w[1] + depth));
            assert!(
                ratio > wall_length && ratio < 1.0,
                "{n_levels} levels: l/κz {ratio} at the first w-point (coarser: {wall_length})"
            );
            wall_length = ratio;
        }
        assert!(wall_length > 0.95, "l/κz {wall_length} at 80 levels");
    }

    /// Wind against a log-layer drag, unstratified, to steady state: a layer
    /// of constant stress `u*²` from the surface to the bed. The surface's
    /// friction velocity only sets the diagnostic `k` at the boundary
    /// w-points; the wind reaches the turbulence through the shear
    /// production of its momentum flux. At equilibrium the interior must
    /// hold the log layer's `k = u*²/(c_μ⁰)²` and carry the stress as
    /// `ν ∂u/∂z = u*²` at every w-point: measured to 2.9e-11 / 2.9e-11 (k-ε),
    /// 1.4e-11 / 1.3e-11 (k-ω), 2.4e-10 / 2.3e-10 (generic) after 48 h
    /// (7e-3 to 1.2e-2 after 12 h).
    ///
    /// From rest (`k = k_min`) the top interior w-point needs ≈ 36 h/u* to
    /// reach 90 % of that `k`, for any Δt (h the top layer's thickness:
    /// 3 h at 3 m layers and u* = 1 cm/s, 20 min at 0.3 m); a surface
    /// condition with `u*` in it (wave injection, Charnock roughness) would
    /// shorten that spin-up.
    #[test]
    fn a_constant_stress_layer_holds_the_log_layer_turbulence() {
        let (depth, n_levels, u_star, z0, dt, rho0) = (10.0, 20, 0.01, 0.02, 60.0, 1025.0);
        let sigma = SigmaGrid::uniform(n_levels);
        let bathymetry = Bathymetry2D::constant(1, 1, -depth);
        let dz = depth / n_levels as f64;
        let z_b = (1.0 + sigma.sigma_rho()[0]) * depth;
        let forcing = Forcing {
            surface_stress: [rho0 * u_star * u_star, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_buoyancy_flux: 0.0,
        };
        let drag = BottomDrag3D::log_layer(z0).with_bounds(0.0, 1.0);
        for gls in [
            GlsMixing::k_epsilon(),
            GlsMixing::k_omega(),
            GlsMixing::generic(),
        ] {
            let gls = gls.with_roughness(z0, z0).with_background(0.0, 0.0);
            let k_log = u_star * u_star / gls.cm0().powi(2);
            let mut state = Solution3D::new(1, 1, n_levels);
            state.rho.fill(rho0);
            for _ in 0..(48.0 * 3600.0 / dt) as usize {
                let rate = drag.rate(z_b, state.u[0].abs());
                apply_vertical_diffusion(
                    &mut state,
                    &sigma,
                    &bathymetry,
                    dt,
                    &gls,
                    &forcing,
                    None,
                    G,
                    rho0,
                    0.0,
                    StepDrag::bottom(&[rate]),
                );
            }
            let name = format!("{:?}", gls.parameters());
            let k_err = state
                .tke
                .iter()
                .map(|k| (k / k_log - 1.0).abs())
                .fold(0.0, f64::max);
            // ν at w-point j, between layers j − 1 and j
            let stress_err = (1..n_levels)
                .map(|j| {
                    let shear = (state.u[j] - state.u[j - 1]) / dz;
                    (state.eddy_viscosity[j] * shear / (u_star * u_star) - 1.0).abs()
                })
                .fold(0.0, f64::max);
            assert!(k_err < 1e-8, "{name}: k off u*²/(c_μ⁰)² by {k_err:.2e}");
            assert!(
                stress_err < 1e-8,
                "{name}: ν ∂u/∂z off u*² by {stress_err:.2e}"
            );
        }
    }

    /// Convection without wind or shear: cold water over warm (unstable,
    /// ΔT = 0.5 °C over the top 10 m of a 40 m column) mixes itself through
    /// buoyancy production (`c₃⁺`), from the minimum turbulence, with no
    /// convective adjustment: after 3 h the largest inversion left is below
    /// 2e-4 of ΔT (5e-5 °C measured) and the column is at its mean
    /// temperature to 1e-3 °C, for all three presets; heat is conserved.
    /// With ROMS's `c₃⁺ = 1` k-ω stayed unstable (0.45 °C after 12 h, k at
    /// its minimum).
    #[test]
    fn an_unstable_column_overturns_by_buoyancy_production() {
        let (depth, n_levels, dt) = (40.0, 40, 60.0);
        let eos = LinearEOS::default();
        let sigma = SigmaGrid::uniform(n_levels);
        let bathymetry = Bathymetry2D::constant(1, 1, -depth);
        let forcing = Forcing {
            surface_stress: [0.0, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_buoyancy_flux: 0.0,
        };
        for gls in [
            GlsMixing::k_epsilon(),
            GlsMixing::k_omega(),
            GlsMixing::generic(),
        ] {
            let mut state = Solution3D::new(1, 1, n_levels);
            for (l, &s) in sigma.sigma_rho().iter().enumerate() {
                state.temp[l] = eos.t0 + if s * depth > -10.0 { -0.5 } else { 0.0 };
                state.salt[l] = eos.s0;
            }
            eos.update_density(&mut state);
            let heat0: f64 = state.temp.iter().sum();
            let instability = |state: &Solution3D| {
                state
                    .temp
                    .windows(2)
                    .map(|t| t[0] - t[1])
                    .fold(0.0_f64, f64::max)
            };
            for _ in 0..(3.0 * 3600.0 / dt) as usize {
                apply_vertical_diffusion(
                    &mut state,
                    &sigma,
                    &bathymetry,
                    dt,
                    &gls,
                    &forcing,
                    None,
                    G,
                    eos.rho0,
                    0.0,
                    StepDrag::NONE,
                );
                eos.update_density(&mut state);
            }
            let name = format!("{:?}", gls.parameters());
            let inversion = instability(&state);
            assert!(
                inversion < 1e-4,
                "{name}: inversion {inversion} °C after 3 h"
            );
            let mean = heat0 / n_levels as f64;
            assert!(
                state.temp.iter().all(|t| (t - mean).abs() < 1e-3),
                "{name}: not mixed, T {:?}",
                state.temp
            );
            let heat: f64 = state.temp.iter().sum();
            assert!((heat - heat0).abs() < 1e-12 * heat0, "{name}: heat changed");
        }
    }

    /// A surface stress field reaches every column on its own: the momentum
    /// flux, and the friction velocity of the GLS surface boundary values
    /// (diagnostic, see the constant-stress gate above). Three columns of a
    /// stratified layer under the uniform stress (0.05, 0) Pa plus their
    /// entries of the field, (0.05, 0), (−0.05, 0) and (0, 0.08):
    /// each ends bit for bit where a lone column under its total stress does,
    /// in velocity, temperature, `k` and `ψ`.
    #[test]
    fn each_column_feels_its_own_surface_stress() {
        let (depth, n_levels, dt, steps) = (30.0, 30, 60.0, 360);
        let eos = LinearEOS::default();
        let sigma = SigmaGrid::uniform(n_levels);
        let gls = GlsMixing::k_epsilon();
        let forcing = |tau: [f64; 2]| Forcing {
            surface_stress: tau,
            bottom_stress: [0.0, 0.0],
            surface_buoyancy_flux: 0.0,
        };
        let stratified = |n_columns: usize| {
            let mut state = Solution3D::new(1, n_columns, n_levels);
            for column in state.temp.chunks_exact_mut(n_levels) {
                for (t, &s) in column.iter_mut().zip(sigma.sigma_rho()) {
                    *t = eos.t0 + 0.1 * s * depth;
                }
            }
            state.salt.fill(eos.s0);
            eos.update_density(&mut state);
            state
        };
        let run = |state: &mut Solution3D, forcing: &Forcing, field: Option<[&[f64]; 2]>| {
            let bathymetry = Bathymetry2D::constant(1, state.n_nodes, -depth);
            for _ in 0..steps {
                apply_vertical_diffusion(
                    state,
                    &sigma,
                    &bathymetry,
                    dt,
                    &gls,
                    forcing,
                    field,
                    G,
                    eos.rho0,
                    0.0,
                    StepDrag::NONE,
                );
                eos.update_density(state);
            }
        };

        let mut columns = stratified(3);
        let (field_x, field_y) = ([0.05, -0.05, 0.0], [0.0, 0.0, 0.08]);
        run(
            &mut columns,
            &forcing([0.05, 0.0]),
            Some([&field_x, &field_y]),
        );
        assert!(columns.u[n_levels - 1] > 0.01, "the wind drove no current");
        for (i, total) in [[0.1, 0.0], [0.0, 0.0], [0.05, 0.08]]
            .into_iter()
            .enumerate()
        {
            let mut lone = stratified(1);
            run(&mut lone, &forcing(total), None);
            let (levels, faces) = (
                i * n_levels..(i + 1) * n_levels,
                i * (n_levels + 1)..(i + 1) * (n_levels + 1),
            );
            assert_eq!(&columns.u[levels.clone()], &lone.u[..], "column {i}: u");
            assert_eq!(&columns.v[levels.clone()], &lone.v[..], "column {i}: v");
            assert_eq!(&columns.temp[levels], &lone.temp[..], "column {i}: T");
            assert_eq!(&columns.tke[faces.clone()], &lone.tke[..], "column {i}: k");
            assert_eq!(&columns.gls[faces], &lone.gls[..], "column {i}: ψ");
        }
    }
}
