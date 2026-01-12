//! Chapman radiation boundary condition for sea surface height.
//!
//! The Chapman condition provides a radiation BC that allows long gravity waves
//! to exit the domain without reflection. It's based on the characteristic
//! wave equation:
//!
//! ∂η/∂t ± c ∂η/∂n = 0
//!
//! where c = √(gh) is the shallow water wave speed and the sign depends on
//! whether the boundary is outgoing (+) or incoming (-).
//!
//! In discrete form, the Chapman condition gives:
//!
//! η_ghost = α·η_ext + (1-α)·η_int
//!
//! where α = 1/(1 + c·dt/dx) is the radiation coefficient.
//!
//! # Reference
//!
//! Chapman, D.C. (1985): "Numerical treatment of cross-shelf open boundaries
//! in a barotropic coastal ocean model", Journal of Physical Oceanography.

use super::bathymetry_validation::warn_once_if_misconfigured;
use super::{BCContext2D, SWEBoundaryCondition2D};
use crate::solver::SWEState2D;

/// Chapman radiation boundary condition for 2D.
///
/// Allows gravity waves to propagate out of the domain with minimal reflection.
/// Typically combined with a velocity condition (such as Flather) for complete
/// specification of the open boundary.
///
/// # Type Parameters
///
/// * `F` - External elevation function η_ext(x, y, t)
///
/// # Example
///
/// ```
/// use dg_rs::boundary::{Chapman2D, BCContext2D, SWEBoundaryCondition2D};
/// use dg_rs::SWEState2D;
///
/// // Create Chapman BC with constant external elevation
/// let chapman = Chapman2D::new(|_x, _y, _t| 0.0, 100.0);
///
/// // Evaluate at boundary
/// let ctx = BCContext2D::new(
///     0.0, (0.0, 0.0),
///     SWEState2D::new(10.0, 5.0, 0.0), // h=10, hu=5
///     0.0, (1.0, 0.0), // outward normal in x
///     9.81, 1e-6,
/// );
/// let ghost = chapman.ghost_state(&ctx);
/// ```
#[derive(Clone)]
pub struct Chapman2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// External (far-field) surface elevation function η_ext(x, y, t)
    pub external_elevation: F,
    /// Characteristic length scale (grid spacing) for radiation
    pub dx: f64,
    /// Time step for implicit treatment (optional, enhances stability)
    pub dt: Option<f64>,
    /// Minimum depth threshold
    pub h_min: f64,
}

impl<F> Chapman2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// Create Chapman BC with external elevation function.
    ///
    /// # Arguments
    /// * `external_elevation` - Function returning η_ext(x, y, t)
    /// * `dx` - Grid spacing (characteristic length for radiation)
    pub fn new(external_elevation: F, dx: f64) -> Self {
        Self {
            external_elevation,
            dx,
            dt: None,
            h_min: 1e-6,
        }
    }

    /// Create with time step for improved radiation accuracy.
    pub fn with_dt(external_elevation: F, dx: f64, dt: f64) -> Self {
        Self {
            external_elevation,
            dx,
            dt: Some(dt),
            h_min: 1e-6,
        }
    }

    /// Set the time step.
    pub fn set_dt(&mut self, dt: f64) {
        self.dt = Some(dt);
    }

    /// Set minimum depth threshold.
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }
}

impl<F> SWEBoundaryCondition2D for Chapman2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        let (x, y) = ctx.position;
        let t = ctx.time;
        let g = ctx.g;

        // External surface elevation
        let eta_ext = (self.external_elevation)(x, y, t);

        // Interior surface elevation
        let eta_int = ctx.interior_surface_elevation();

        // Wave celerity at interior
        let c = (g * ctx.interior_state.h.max(self.h_min)).sqrt();

        // Chapman radiation coefficient
        // α = 1/(1 + c·dt/dx) blends between exterior (α=1) and interior (α=0)
        let cfl = self.dt.map_or(1.0, |dt| c * dt / self.dx);
        let alpha = 1.0 / (1.0 + cfl);

        // Radiation condition for elevation
        // η_ghost = α·η_ext + (1-α)·η_int
        let eta_ghost = alpha * eta_ext + (1.0 - alpha) * eta_int;
        let h_ghost = (eta_ghost - ctx.bathymetry).max(self.h_min);

        // Extrapolate velocity from interior (tangential preserved)
        let (u, v) = ctx.interior_velocity();

        SWEState2D::from_primitives(h_ghost, u, v)
    }

    fn name(&self) -> &'static str {
        "chapman_2d"
    }
}

/// Combined Chapman (elevation) + Flather (velocity) boundary condition.
///
/// This is the recommended open boundary condition for tidal simulations:
/// - Chapman condition for sea surface height (radiation)
/// - Flather condition for normal velocity (characteristic-based)
/// - Tangential velocity extrapolated from interior
///
/// The combination provides both wave absorption and correct tidal forcing.
///
/// # Type Parameters
///
/// * `F` - External state function returning (η_ext, u_n_ext, u_t_ext) at (x, y, t)
///
/// # Example
///
/// ```
/// use dg_rs::boundary::{ChapmanFlather2D, BCContext2D, SWEBoundaryCondition2D};
/// use dg_rs::SWEState2D;
/// use std::f64::consts::PI;
///
/// // M2 tidal forcing
/// let omega = 2.0 * PI / (12.42 * 3600.0);
/// let amplitude = 1.0;
/// let cf = ChapmanFlather2D::new(
///     move |_x, _y, t| (amplitude * (omega * t).cos(), 0.0, 0.0),
///     100.0, // dx
/// );
/// ```
#[derive(Clone)]
pub struct ChapmanFlather2D<F>
where
    F: Fn(f64, f64, f64) -> (f64, f64, f64) + Send + Sync,
{
    /// External state function returning (η_ext, u_n_ext, u_t_ext) at (x, y, t)
    pub external_state: F,
    /// Grid spacing for Chapman radiation
    pub dx: f64,
    /// Time step (optional)
    pub dt: Option<f64>,
    /// Reference depth for Flather
    pub h_ref: f64,
    /// Minimum depth
    pub h_min: f64,
}

impl<F> ChapmanFlather2D<F>
where
    F: Fn(f64, f64, f64) -> (f64, f64, f64) + Send + Sync,
{
    /// Create a new ChapmanFlather BC.
    ///
    /// # Arguments
    /// * `external_state` - Function returning (η, u_n, u_t) at (x, y, t)
    /// * `dx` - Grid spacing for Chapman radiation
    pub fn new(external_state: F, dx: f64) -> Self {
        Self {
            external_state,
            dx,
            dt: None,
            h_ref: 10.0,
            h_min: 1e-6,
        }
    }

    /// Set reference depth for Flather velocity condition.
    pub fn with_h_ref(mut self, h_ref: f64) -> Self {
        self.h_ref = h_ref;
        self
    }

    /// Set time step for Chapman.
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = Some(dt);
        self
    }

    /// Set minimum depth.
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }
}

impl<F> SWEBoundaryCondition2D for ChapmanFlather2D<F>
where
    F: Fn(f64, f64, f64) -> (f64, f64, f64) + Send + Sync,
{
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        use std::sync::atomic::AtomicBool;
        static WARNED: AtomicBool = AtomicBool::new(false);

        let (x, y) = ctx.position;
        let t = ctx.time;
        let (nx, ny) = ctx.normal;
        let g = ctx.g;

        // External state: (η_ext, u_n_ext, u_t_ext)
        let (eta_ext, un_ext, _ut_ext) = (self.external_state)(x, y, t);

        // Validate bathymetry configuration (warns once if misconfigured)
        warn_once_if_misconfigured(
            &WARNED,
            "ChapmanFlather2D",
            ctx.interior_state.h,
            ctx.bathymetry,
            eta_ext,
        );

        // Interior state
        let eta_int = ctx.interior_surface_elevation();
        let h_int = ctx.interior_state.h;

        // Wave celerity at interior
        let c = (g * h_int.max(self.h_min)).sqrt();

        // Chapman for elevation
        let cfl = self.dt.map_or(1.0, |dt| c * dt / self.dx);
        let alpha = 1.0 / (1.0 + cfl);
        let eta_ghost = alpha * eta_ext + (1.0 - alpha) * eta_int;
        let h_ghost = (eta_ghost - ctx.bathymetry).max(self.h_min);

        // Flather for normal velocity
        // u_n_ghost = u_n_ext + (c/h_ref) * (η_int - η_ext)
        let h_ref = self.h_ref.max(self.h_min);
        let c_ref = (g * h_ref).sqrt();
        let un_ghost = un_ext + (c_ref / h_ref) * (eta_int - eta_ext);

        // Tangential velocity from interior (zero-gradient)
        let ut_ghost = ctx.interior_tangential_velocity();

        // Convert (un, ut) back to (u, v)
        // u = un * nx - ut * ny
        // v = un * ny + ut * nx
        let u_ghost = un_ghost * nx - ut_ghost * ny;
        let v_ghost = un_ghost * ny + ut_ghost * nx;

        SWEState2D::from_primitives(h_ghost, u_ghost, v_ghost)
    }

    fn name(&self) -> &'static str {
        "chapman_flather_2d"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 10.0;
    const TOL: f64 = 1e-10;

    fn make_context(h: f64, hu: f64, hv: f64, normal: (f64, f64)) -> BCContext2D {
        BCContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(h, hu, hv),
            0.0, // bathymetry
            normal,
            G,
            1e-6,
        )
    }

    #[test]
    fn test_chapman_steady_state() {
        // When interior matches external, ghost should match both
        let eta_ext = 5.0;
        let chapman = Chapman2D::new(|_, _, _| eta_ext, 100.0);

        let ctx = make_context(5.0, 0.0, 0.0, (1.0, 0.0)); // h=5, eta=h+B=5
        let ghost = chapman.ghost_state(&ctx);

        assert!(
            (ghost.h - 5.0).abs() < TOL,
            "Steady state should maintain h=5, got {}",
            ghost.h
        );
    }

    #[test]
    fn test_chapman_radiation_blending() {
        // With dt specified, should blend between external and internal
        let chapman = Chapman2D::with_dt(|_, _, _| 0.0, 100.0, 0.1);

        // Interior elevation = 10 (h=10, B=0)
        // External elevation = 0
        // With some CFL, ghost should be between 0 and 10
        let ctx = make_context(10.0, 0.0, 0.0, (1.0, 0.0));
        let ghost = chapman.ghost_state(&ctx);

        assert!(ghost.h > 0.0, "Ghost depth should be positive");
        assert!(ghost.h < 10.0, "Ghost should blend toward external");
    }

    #[test]
    fn test_chapman_velocity_extrapolation() {
        // Velocity should be extrapolated from interior
        let chapman = Chapman2D::new(|_, _, _| 10.0, 100.0);

        let ctx = make_context(10.0, 20.0, 30.0, (1.0, 0.0)); // u=2, v=3
        let ghost = chapman.ghost_state(&ctx);

        // Velocity should be preserved from interior
        let (u, v) = (ghost.hu / ghost.h, ghost.hv / ghost.h);
        assert!((u - 2.0).abs() < TOL, "u should be extrapolated: got {}", u);
        assert!((v - 3.0).abs() < TOL, "v should be extrapolated: got {}", v);
    }

    #[test]
    fn test_chapman_flather_steady() {
        // Test combined condition in steady state
        let cf = ChapmanFlather2D::new(|_, _, _| (10.0, 0.0, 0.0), 100.0).with_h_ref(10.0);

        let ctx = make_context(10.0, 0.0, 0.0, (1.0, 0.0));
        let ghost = cf.ghost_state(&ctx);

        // In steady state matching external, ghost should match
        assert!(
            (ghost.h - 10.0).abs() < TOL,
            "Steady state h: got {}",
            ghost.h
        );
        assert!(
            ghost.hu.abs() < TOL,
            "Steady state hu should be ~0: got {}",
            ghost.hu
        );
    }

    #[test]
    fn test_chapman_flather_inflow() {
        // When interior elevation is higher, Flather should reduce normal velocity
        let cf = ChapmanFlather2D::new(|_, _, _| (8.0, 0.0, 0.0), 100.0).with_h_ref(10.0);

        // Interior: eta_int = 10, External: eta_ext = 8
        // Flather: u_n = u_n_ext + c/h_ref * (eta_int - eta_ext)
        //        = 0 + sqrt(g*h_ref)/h_ref * (10-8)
        //        = sqrt(g/h_ref) * 2
        let ctx = make_context(10.0, 0.0, 0.0, (1.0, 0.0));
        let ghost = cf.ghost_state(&ctx);

        let expected_un = (G / 10.0).sqrt() * 2.0;
        let u_ghost = ghost.hu / ghost.h;

        assert!(
            (u_ghost - expected_un).abs() < 0.1,
            "Flather normal velocity: expected ~{}, got {}",
            expected_un,
            u_ghost
        );
    }

    #[test]
    fn test_chapman_preserves_tangential() {
        // Tangential velocity should be preserved in ChapmanFlather
        let cf = ChapmanFlather2D::new(|_, _, _| (10.0, 0.0, 0.0), 100.0).with_h_ref(10.0);

        // Flow tangential to boundary (normal is x, flow is y)
        let ctx = make_context(10.0, 0.0, 30.0, (1.0, 0.0)); // u_n=0, u_t=3
        let ghost = cf.ghost_state(&ctx);

        let v_ghost = ghost.hv / ghost.h;
        assert!(
            (v_ghost - 3.0).abs() < TOL,
            "Tangential velocity should be preserved: got {}",
            v_ghost
        );
    }

    #[test]
    fn test_chapman_different_normals() {
        // Test with different normal directions
        let chapman = Chapman2D::new(|_, _, _| 10.0, 100.0);

        for normal in [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)] {
            let ctx = make_context(10.0, 5.0, 5.0, normal);
            let ghost = chapman.ghost_state(&ctx);

            assert!(
                ghost.h > 0.0,
                "Ghost depth should be positive for normal {:?}",
                normal
            );
        }
    }
}
