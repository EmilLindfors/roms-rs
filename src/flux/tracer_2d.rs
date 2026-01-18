//! Numerical fluxes for 2D tracer transport.
//!
//! Tracers (temperature, salinity) are advected by the flow field from the
//! shallow water solution. The numerical flux is computed using upwinding
//! based on the normal velocity at element interfaces.
//!
//! # Transport Equation
//!
//! For a tracer C (concentration), the conservative form is:
//!
//! ∂(hC)/∂t + ∇·(hC**u**) = sources
//!
//! where h is water depth and **u** = (u, v) is velocity.
//!
//! At element interfaces, we need a numerical flux for hC·**u**·**n**.

use crate::solver::{ConservativeTracerState, SWEState2D};
use crate::types::Depth;

/// Upwind flux for tracer transport.
///
/// The flux is determined by the sign of the normal velocity:
/// - If u·n > 0 (outflow): use interior tracer values
/// - If u·n < 0 (inflow): use exterior tracer values
///
/// This is first-order upwinding, which is diffusive but stable.
///
/// # Arguments
/// * `tracer_int` - Tracer state from interior element (hT, hS)
/// * `tracer_ext` - Tracer state from exterior element (hT, hS)
/// * `swe_int` - SWE state from interior (h, hu, hv)
/// * `swe_ext` - SWE state from exterior (h, hu, hv)
/// * `normal` - Outward unit normal (nx, ny)
/// * `h_min` - Minimum depth threshold
///
/// # Returns
/// Numerical flux (F_hT·n, F_hS·n)
pub fn upwind_tracer_flux(
    tracer_int: &ConservativeTracerState,
    tracer_ext: &ConservativeTracerState,
    swe_int: &SWEState2D,
    swe_ext: &SWEState2D,
    normal: (f64, f64),
    h_min: f64,
) -> ConservativeTracerState {
    let (nx, ny) = normal;
    let h_min = Depth::new(h_min);

    // Compute velocities
    let (u_int, v_int) = swe_int.velocity_simple(h_min);
    let (u_ext, v_ext) = swe_ext.velocity_simple(h_min);

    // Normal velocities
    let un_int = u_int * nx + v_int * ny;
    let un_ext = u_ext * nx + v_ext * ny;

    // Average normal velocity for upwinding decision
    let un_avg = 0.5 * (un_int + un_ext);

    // Upwind selection
    if un_avg >= 0.0 {
        // Flow is outward: use interior values
        ConservativeTracerState {
            h_t: tracer_int.h_t * un_int,
            h_s: tracer_int.h_s * un_int,
        }
    } else {
        // Flow is inward: use exterior values
        ConservativeTracerState {
            h_t: tracer_ext.h_t * un_ext,
            h_s: tracer_ext.h_s * un_ext,
        }
    }
}

/// Central flux for tracer transport (not recommended - unstable).
///
/// Provided for comparison/testing. Uses arithmetic average of fluxes.
pub fn central_tracer_flux(
    tracer_int: &ConservativeTracerState,
    tracer_ext: &ConservativeTracerState,
    swe_int: &SWEState2D,
    swe_ext: &SWEState2D,
    normal: (f64, f64),
    h_min: f64,
) -> ConservativeTracerState {
    let (nx, ny) = normal;
    let h_min = Depth::new(h_min);

    let (u_int, v_int) = swe_int.velocity_simple(h_min);
    let (u_ext, v_ext) = swe_ext.velocity_simple(h_min);

    let un_int = u_int * nx + v_int * ny;
    let un_ext = u_ext * nx + v_ext * ny;

    // Central average of fluxes
    ConservativeTracerState {
        h_t: 0.5 * (tracer_int.h_t * un_int + tracer_ext.h_t * un_ext),
        h_s: 0.5 * (tracer_int.h_s * un_int + tracer_ext.h_s * un_ext),
    }
}

/// Lax-Friedrichs flux for tracer transport.
///
/// More dissipative than pure upwind but handles discontinuities better.
///
/// F* = 0.5 * (F_int + F_ext) - 0.5 * λ_max * (q_ext - q_int)
///
/// where λ_max is based on the maximum wave speed |u| + c.
pub fn lax_friedrichs_tracer_flux(
    tracer_int: &ConservativeTracerState,
    tracer_ext: &ConservativeTracerState,
    swe_int: &SWEState2D,
    swe_ext: &SWEState2D,
    normal: (f64, f64),
    h_min: f64,
    g: f64,
) -> ConservativeTracerState {
    let (nx, ny) = normal;
    let h_min = Depth::new(h_min);

    let (u_int, v_int) = swe_int.velocity_simple(h_min);
    let (u_ext, v_ext) = swe_ext.velocity_simple(h_min);

    let un_int = u_int * nx + v_int * ny;
    let un_ext = u_ext * nx + v_ext * ny;

    // Central flux
    let flux_central_h_t = 0.5 * (tracer_int.h_t * un_int + tracer_ext.h_t * un_ext);
    let flux_central_h_s = 0.5 * (tracer_int.h_s * un_int + tracer_ext.h_s * un_ext);

    // Maximum wave speed for dissipation
    let c_int = (g * swe_int.h.max(0.0)).sqrt();
    let c_ext = (g * swe_ext.h.max(0.0)).sqrt();
    let speed_int = (u_int * u_int + v_int * v_int).sqrt();
    let speed_ext = (u_ext * u_ext + v_ext * v_ext).sqrt();
    let lambda_max = (speed_int + c_int).max(speed_ext + c_ext);

    // Lax-Friedrichs flux
    ConservativeTracerState {
        h_t: flux_central_h_t - 0.5 * lambda_max * (tracer_ext.h_t - tracer_int.h_t),
        h_s: flux_central_h_s - 0.5 * lambda_max * (tracer_ext.h_s - tracer_int.h_s),
    }
}

/// Roe-type flux for tracer transport.
///
/// Uses Roe-averaged velocity for upwinding, providing a good balance
/// between accuracy and stability.
pub fn roe_tracer_flux(
    tracer_int: &ConservativeTracerState,
    tracer_ext: &ConservativeTracerState,
    swe_int: &SWEState2D,
    swe_ext: &SWEState2D,
    normal: (f64, f64),
    h_min: f64,
) -> ConservativeTracerState {
    let (nx, ny) = normal;
    let h_min = Depth::new(h_min);

    // Compute Roe-averaged velocity
    let h_int = swe_int.h.max(h_min.meters());
    let h_ext = swe_ext.h.max(h_min.meters());

    let sqrt_h_int = h_int.sqrt();
    let sqrt_h_ext = h_ext.sqrt();
    let denom = sqrt_h_int + sqrt_h_ext;

    let (u_int, v_int) = swe_int.velocity_simple(h_min);
    let (u_ext, v_ext) = swe_ext.velocity_simple(h_min);

    let u_roe = if denom > 1e-10 {
        (sqrt_h_int * u_int + sqrt_h_ext * u_ext) / denom
    } else {
        0.0
    };

    let v_roe = if denom > 1e-10 {
        (sqrt_h_int * v_int + sqrt_h_ext * v_ext) / denom
    } else {
        0.0
    };

    // Roe-averaged normal velocity
    let un_roe = u_roe * nx + v_roe * ny;

    // Physical fluxes
    let un_int = u_int * nx + v_int * ny;
    let un_ext = u_ext * nx + v_ext * ny;

    let flux_int_h_t = tracer_int.h_t * un_int;
    let flux_int_h_s = tracer_int.h_s * un_int;
    let flux_ext_h_t = tracer_ext.h_t * un_ext;
    let flux_ext_h_s = tracer_ext.h_s * un_ext;

    // Central + upwind correction
    let jump_h_t = tracer_ext.h_t - tracer_int.h_t;
    let jump_h_s = tracer_ext.h_s - tracer_int.h_s;

    ConservativeTracerState {
        h_t: 0.5 * (flux_int_h_t + flux_ext_h_t) - 0.5 * un_roe.abs() * jump_h_t,
        h_s: 0.5 * (flux_int_h_s + flux_ext_h_s) - 0.5 * un_roe.abs() * jump_h_s,
    }
}

/// Tracer flux type selector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum TracerFluxType {
    /// First-order upwind (default, stable)
    #[default]
    Upwind,
    /// Roe-type flux (better accuracy)
    Roe,
    /// Lax-Friedrichs (most dissipative)
    LaxFriedrichs,
    /// Central flux (unstable, for testing)
    Central,
}

/// Compute tracer numerical flux using the specified method.
pub fn tracer_numerical_flux(
    flux_type: TracerFluxType,
    tracer_int: &ConservativeTracerState,
    tracer_ext: &ConservativeTracerState,
    swe_int: &SWEState2D,
    swe_ext: &SWEState2D,
    normal: (f64, f64),
    h_min: f64,
    g: f64,
) -> ConservativeTracerState {
    match flux_type {
        TracerFluxType::Upwind => {
            upwind_tracer_flux(tracer_int, tracer_ext, swe_int, swe_ext, normal, h_min)
        }
        TracerFluxType::Roe => {
            roe_tracer_flux(tracer_int, tracer_ext, swe_int, swe_ext, normal, h_min)
        }
        TracerFluxType::LaxFriedrichs => {
            lax_friedrichs_tracer_flux(tracer_int, tracer_ext, swe_int, swe_ext, normal, h_min, g)
        }
        TracerFluxType::Central => {
            central_tracer_flux(tracer_int, tracer_ext, swe_int, swe_ext, normal, h_min)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::TracerState;

    const TOL: f64 = 1e-12;
    const H_MIN: f64 = 1e-6;

    fn make_test_states() -> (
        ConservativeTracerState,
        ConservativeTracerState,
        SWEState2D,
        SWEState2D,
    ) {
        let h = 10.0;
        let tracers_int = TracerState::new(10.0, 34.0);
        let tracers_ext = TracerState::new(8.0, 35.0);

        let tracer_int = ConservativeTracerState::from_depth_and_tracers(h, tracers_int);
        let tracer_ext = ConservativeTracerState::from_depth_and_tracers(h, tracers_ext);

        let swe_int = SWEState2D::from_primitives(h, 1.0, 0.0); // Flow in +x
        let swe_ext = SWEState2D::from_primitives(h, 1.0, 0.0);

        (tracer_int, tracer_ext, swe_int, swe_ext)
    }

    #[test]
    fn test_upwind_outflow() {
        let (tracer_int, tracer_ext, swe_int, swe_ext) = make_test_states();

        // Normal pointing outward (+x), flow is outward
        let normal = (1.0, 0.0);
        let flux = upwind_tracer_flux(&tracer_int, &tracer_ext, &swe_int, &swe_ext, normal, H_MIN);

        // Should use interior values since flow is outward
        // Flux = hT * u = 100 * 1 = 100 for temperature
        assert!((flux.h_t - 100.0).abs() < TOL);
        // Flux = hS * u = 340 * 1 = 340 for salinity
        assert!((flux.h_s - 340.0).abs() < TOL);
    }

    #[test]
    fn test_upwind_inflow() {
        let (tracer_int, tracer_ext, _, _) = make_test_states();

        // Reverse the flow
        let h = 10.0;
        let swe_int = SWEState2D::from_primitives(h, -1.0, 0.0); // Flow in -x
        let swe_ext = SWEState2D::from_primitives(h, -1.0, 0.0);

        // Normal pointing outward (+x), but flow is inward (-x)
        let normal = (1.0, 0.0);
        let flux = upwind_tracer_flux(&tracer_int, &tracer_ext, &swe_int, &swe_ext, normal, H_MIN);

        // Should use exterior values since flow is inward
        // Flux = hT_ext * u_ext = 80 * (-1) = -80 for temperature
        assert!((flux.h_t - (-80.0)).abs() < TOL);
        // Flux = hS_ext * u_ext = 350 * (-1) = -350 for salinity
        assert!((flux.h_s - (-350.0)).abs() < TOL);
    }

    #[test]
    fn test_flux_continuous_solution() {
        let h = 10.0;
        let tracers = TracerState::new(10.0, 34.0);
        let tracer_state = ConservativeTracerState::from_depth_and_tracers(h, tracers);
        let swe_state = SWEState2D::from_primitives(h, 1.0, 0.5);

        let normal = (1.0, 0.0);

        // All flux types should give same result for continuous solution
        let flux_upwind = upwind_tracer_flux(
            &tracer_state,
            &tracer_state,
            &swe_state,
            &swe_state,
            normal,
            H_MIN,
        );

        let flux_roe = roe_tracer_flux(
            &tracer_state,
            &tracer_state,
            &swe_state,
            &swe_state,
            normal,
            H_MIN,
        );

        let flux_lf = lax_friedrichs_tracer_flux(
            &tracer_state,
            &tracer_state,
            &swe_state,
            &swe_state,
            normal,
            H_MIN,
            9.81,
        );

        // For continuous solution, flux = hC * u·n
        let expected_h_t = 100.0 * 1.0; // hT * u
        let expected_h_s = 340.0 * 1.0; // hS * u

        assert!((flux_upwind.h_t - expected_h_t).abs() < TOL);
        assert!((flux_roe.h_t - expected_h_t).abs() < TOL);
        assert!((flux_lf.h_t - expected_h_t).abs() < TOL);

        assert!((flux_upwind.h_s - expected_h_s).abs() < TOL);
        assert!((flux_roe.h_s - expected_h_s).abs() < TOL);
        assert!((flux_lf.h_s - expected_h_s).abs() < TOL);
    }

    #[test]
    fn test_flux_zero_velocity() {
        let h = 10.0;
        let tracers_int = TracerState::new(10.0, 34.0);
        let tracers_ext = TracerState::new(8.0, 35.0);

        let tracer_int = ConservativeTracerState::from_depth_and_tracers(h, tracers_int);
        let tracer_ext = ConservativeTracerState::from_depth_and_tracers(h, tracers_ext);

        // Zero velocity
        let swe_int = SWEState2D::from_primitives(h, 0.0, 0.0);
        let swe_ext = SWEState2D::from_primitives(h, 0.0, 0.0);

        let normal = (1.0, 0.0);

        let flux = upwind_tracer_flux(&tracer_int, &tracer_ext, &swe_int, &swe_ext, normal, H_MIN);

        // Zero flux for zero velocity
        assert!(flux.h_t.abs() < TOL);
        assert!(flux.h_s.abs() < TOL);
    }

    #[test]
    fn test_flux_y_direction() {
        let h = 10.0;
        let tracers = TracerState::new(10.0, 34.0);
        let tracer_state = ConservativeTracerState::from_depth_and_tracers(h, tracers);

        // Flow in +y direction
        let swe_state = SWEState2D::from_primitives(h, 0.0, 2.0);

        // Normal in y direction
        let normal = (0.0, 1.0);

        let flux = upwind_tracer_flux(
            &tracer_state,
            &tracer_state,
            &swe_state,
            &swe_state,
            normal,
            H_MIN,
        );

        // Flux = hT * v = 100 * 2 = 200
        assert!((flux.h_t - 200.0).abs() < TOL);
        // Flux = hS * v = 340 * 2 = 680
        assert!((flux.h_s - 680.0).abs() < TOL);
    }

    #[test]
    fn test_flux_type_selector() {
        let (tracer_int, tracer_ext, swe_int, swe_ext) = make_test_states();
        let normal = (1.0, 0.0);
        let g = 9.81;

        // Test that selector returns correct flux type
        let flux_upwind = tracer_numerical_flux(
            TracerFluxType::Upwind,
            &tracer_int,
            &tracer_ext,
            &swe_int,
            &swe_ext,
            normal,
            H_MIN,
            g,
        );

        let flux_direct =
            upwind_tracer_flux(&tracer_int, &tracer_ext, &swe_int, &swe_ext, normal, H_MIN);

        assert!((flux_upwind.h_t - flux_direct.h_t).abs() < TOL);
        assert!((flux_upwind.h_s - flux_direct.h_s).abs() < TOL);
    }
}
