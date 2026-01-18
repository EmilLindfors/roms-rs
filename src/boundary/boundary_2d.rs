//! 2D Boundary conditions for shallow water equations.
//!
//! Extends the 1D boundary conditions to 2D, accounting for:
//! - 2D position (x, y)
//! - 2D normal vectors (nx, ny)
//! - Tangential velocity preservation
//!
//! Available boundary conditions:
//! - Reflective (wall): no-flux through boundary, tangential velocity preserved
//! - Radiation (absorbing): Sommerfeld radiation condition
//! - Flather: characteristic-based open boundary (combines radiation and tidal)
//! - Tidal: prescribed water surface elevation
//! - Discharge: prescribed normal flow rate

use crate::mesh::BoundaryTag;
use crate::solver::SWEState2D;
use crate::types::Depth;

use super::bathymetry_validation::warn_once_if_misconfigured;

/// Context for 2D boundary condition evaluation.
///
/// Provides all information needed to compute the ghost state at a boundary.
#[derive(Clone, Copy, Debug)]
pub struct BCContext2D {
    /// Current simulation time
    pub time: f64,
    /// Physical position of the boundary face (x, y)
    pub position: (f64, f64),
    /// Interior state at the boundary
    pub interior_state: SWEState2D,
    /// Bathymetry at the boundary
    pub bathymetry: f64,
    /// Outward unit normal direction (nx, ny)
    pub normal: (f64, f64),
    /// Gravitational acceleration
    pub g: f64,
    /// Minimum depth threshold
    pub h_min: f64,
    /// Optional boundary tag for multi-BC dispatch
    pub boundary_tag: Option<BoundaryTag>,
}

impl BCContext2D {
    /// Create a new 2D boundary condition context.
    ///
    /// For multi-BC dispatch, use `with_tag` to add the boundary tag.
    pub fn new(
        time: f64,
        position: (f64, f64),
        interior_state: SWEState2D,
        bathymetry: f64,
        normal: (f64, f64),
        g: f64,
        h_min: f64,
    ) -> Self {
        Self {
            time,
            position,
            interior_state,
            bathymetry,
            normal,
            g,
            h_min,
            boundary_tag: None,
        }
    }

    /// Create a context with a boundary tag for multi-BC dispatch.
    pub fn with_tag(
        time: f64,
        position: (f64, f64),
        interior_state: SWEState2D,
        bathymetry: f64,
        normal: (f64, f64),
        g: f64,
        h_min: f64,
        boundary_tag: BoundaryTag,
    ) -> Self {
        Self {
            time,
            position,
            interior_state,
            bathymetry,
            normal,
            g,
            h_min,
            boundary_tag: Some(boundary_tag),
        }
    }

    /// Water surface elevation at the interior: η = h + B
    pub fn interior_surface_elevation(&self) -> f64 {
        self.interior_state.h + self.bathymetry
    }

    /// Interior velocity components (u, v).
    pub fn interior_velocity(&self) -> (f64, f64) {
        self.interior_state.velocity_simple(Depth::new(self.h_min))
    }

    /// Normal velocity component: u·n = u*nx + v*ny
    pub fn interior_normal_velocity(&self) -> f64 {
        let (u, v) = self.interior_velocity();
        let (nx, ny) = self.normal;
        u * nx + v * ny
    }

    /// Tangential velocity component: u×n = -u*ny + v*nx
    pub fn interior_tangential_velocity(&self) -> f64 {
        let (u, v) = self.interior_velocity();
        let (nx, ny) = self.normal;
        -u * ny + v * nx
    }

    /// Wave celerity at interior: c = sqrt(g*h)
    pub fn interior_celerity(&self) -> f64 {
        (self.g * self.interior_state.h.max(0.0)).sqrt()
    }
}

/// Trait for 2D shallow water boundary conditions.
///
/// Implementations compute a "ghost" state that represents the exterior
/// state at a boundary face. This ghost state is then used in the
/// numerical flux computation.
pub trait SWEBoundaryCondition2D: Send + Sync {
    /// Compute the ghost state for flux evaluation.
    ///
    /// # Arguments
    /// * `ctx` - Boundary condition context with interior state and metadata
    ///
    /// # Returns
    /// The ghost (exterior) state to use in flux computation
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D;

    /// Name of this boundary condition for debugging/logging.
    fn name(&self) -> &'static str;

    /// Check if this BC allows inflow (used for validation).
    fn allows_inflow(&self) -> bool {
        true
    }

    /// Check if this BC allows outflow (used for validation).
    fn allows_outflow(&self) -> bool {
        true
    }
}

/// Reflective (wall) boundary condition for 2D.
///
/// Creates a mirror state with reversed normal velocity and preserved
/// tangential velocity, ensuring zero mass flux through the boundary.
///
/// # Mathematical Formulation
///
/// For a boundary with outward normal n = (nx, ny):
/// - h_ghost = h_interior
/// - (u·n)_ghost = -(u·n)_interior  (normal velocity reversed)
/// - (u×n)_ghost = (u×n)_interior   (tangential velocity preserved)
///
/// This gives:
/// - u_ghost = u - 2*(u·n)*nx
/// - v_ghost = v - 2*(u·n)*ny
#[derive(Clone, Debug, Default)]
pub struct Reflective2D {
    /// Minimum depth for velocity computation
    pub h_min: f64,
}

impl Reflective2D {
    /// Create a new reflective BC.
    pub fn new() -> Self {
        Self { h_min: 1e-6 }
    }

    /// Create with custom minimum depth.
    pub fn with_h_min(h_min: f64) -> Self {
        Self { h_min }
    }
}

impl SWEBoundaryCondition2D for Reflective2D {
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        let h = ctx.interior_state.h;
        let (nx, ny) = ctx.normal;

        if h <= self.h_min {
            return SWEState2D::new(h, 0.0, 0.0);
        }

        let (u, v) = ctx.interior_velocity();
        let un = u * nx + v * ny;

        // Reflect normal component: u_ghost = u - 2*(u·n)*n
        let u_ghost = u - 2.0 * un * nx;
        let v_ghost = v - 2.0 * un * ny;

        SWEState2D::from_primitives(h, u_ghost, v_ghost)
    }

    fn name(&self) -> &'static str {
        "reflective_2d"
    }

    fn allows_inflow(&self) -> bool {
        false
    }

    fn allows_outflow(&self) -> bool {
        false
    }
}

/// Radiation (absorbing) boundary condition for 2D.
///
/// Implements a Sommerfeld-type radiation condition that allows waves
/// to exit the domain without reflection. Based on characteristics.
///
/// u_n = ±c * (h - h_ext) / h_ext
///
/// where c = sqrt(gh) is the wave speed and h_ext is the external depth.
#[derive(Clone, Debug)]
pub struct Radiation2D {
    /// External (far-field) water depth
    pub h_external: f64,
    /// External (far-field) velocity (typically zero)
    pub u_external: (f64, f64),
    /// Minimum depth
    pub h_min: f64,
}

impl Radiation2D {
    /// Create a new radiation BC with specified external depth.
    pub fn new(h_external: f64) -> Self {
        Self {
            h_external,
            u_external: (0.0, 0.0),
            h_min: 1e-6,
        }
    }

    /// Create with external depth and velocity.
    pub fn with_velocity(h_external: f64, u_external: (f64, f64)) -> Self {
        Self {
            h_external,
            u_external,
            h_min: 1e-6,
        }
    }
}

impl SWEBoundaryCondition2D for Radiation2D {
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        let h_int = ctx.interior_state.h;
        let (nx, ny) = ctx.normal;
        let g = ctx.g;

        // Use characteristic-based approach
        // For outgoing waves, use interior state
        // For incoming waves, use external state
        let c_int = (g * h_int.max(0.0)).sqrt();
        let (u_int, v_int) = ctx.interior_velocity();
        let un_int = u_int * nx + v_int * ny;

        // Outgoing characteristic: un + 2c (positive for outflow)
        // Incoming characteristic: un - 2c (negative for inflow)
        if un_int + c_int > 0.0 {
            // Outgoing - extrapolate interior
            ctx.interior_state
        } else {
            // Incoming - use external with radiation condition
            let c_ext = (g * self.h_external.max(0.0)).sqrt();
            let (u_ext, v_ext) = self.u_external;
            let un_ext = u_ext * nx + v_ext * ny;
            let _ut_ext = -u_ext * ny + v_ext * nx;

            // Sommerfeld condition: incoming characteristic from exterior
            let un_ghost =
                un_ext - c_ext * (self.h_external - h_int) / self.h_external.max(self.h_min);

            // Preserve tangential velocity from interior
            let ut_ghost = ctx.interior_tangential_velocity();

            // Convert back to (u, v)
            let u_ghost = un_ghost * nx - ut_ghost * ny;
            let v_ghost = un_ghost * ny + ut_ghost * nx;

            SWEState2D::from_primitives(self.h_external, u_ghost, v_ghost)
        }
    }

    fn name(&self) -> &'static str {
        "radiation_2d"
    }
}

/// Flather boundary condition for 2D.
///
/// Combines tidal forcing with characteristic-based radiation.
/// This is the standard open boundary condition for coastal models.
///
/// The Flather condition sets:
/// - h from prescribed tidal elevation
/// - u_n from characteristic relation: u_n = u_n_tidal + c * (η - η_tidal) / h_tidal
///
/// This allows prescribed tides while permitting free outflow of waves.
#[derive(Clone, Debug)]
pub struct Flather2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// Function returning tidal elevation η(x, y, t)
    pub tidal_elevation: F,
    /// External reference depth (below tidal datum)
    pub h_ref: f64,
    /// External velocity (typically zero)
    pub u_external: (f64, f64),
    /// Minimum depth
    pub h_min: f64,
}

impl<F> Flather2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// Create a new Flather BC.
    ///
    /// # Arguments
    /// * `tidal_elevation` - Function η(x, y, t) returning surface elevation
    /// * `h_ref` - Reference depth below mean sea level
    pub fn new(tidal_elevation: F, h_ref: f64) -> Self {
        Self {
            tidal_elevation,
            h_ref,
            u_external: (0.0, 0.0),
            h_min: 1e-6,
        }
    }
}

impl<F> SWEBoundaryCondition2D for Flather2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        use std::sync::atomic::AtomicBool;
        static WARNED: AtomicBool = AtomicBool::new(false);

        let (x, y) = ctx.position;
        let t = ctx.time;
        let (nx, ny) = ctx.normal;
        let g = ctx.g;

        // Prescribed tidal surface elevation
        let eta_tidal = (self.tidal_elevation)(x, y, t);
        let h_tidal = (eta_tidal - ctx.bathymetry).max(self.h_min);

        // Validate bathymetry configuration (warns once if misconfigured)
        warn_once_if_misconfigured(
            &WARNED,
            "Flather2D",
            ctx.interior_state.h,
            ctx.bathymetry,
            eta_tidal,
        );

        // Wave celerity at tidal state
        let c_tidal = (g * h_tidal).sqrt();

        // Interior surface elevation
        let eta_int = ctx.interior_surface_elevation();

        // Flather relation for normal velocity
        // u_n_ghost = u_n_ext + c * (η_int - η_tidal) / h_tidal
        let (u_ext, v_ext) = self.u_external;
        let un_ext = u_ext * nx + v_ext * ny;
        let un_ghost = un_ext + c_tidal * (eta_int - eta_tidal) / h_tidal;

        // Preserve tangential velocity from interior
        let ut_ghost = ctx.interior_tangential_velocity();

        // Convert back to (u, v)
        let u_ghost = un_ghost * nx - ut_ghost * ny;
        let v_ghost = un_ghost * ny + ut_ghost * nx;

        SWEState2D::from_primitives(h_tidal, u_ghost, v_ghost)
    }

    fn name(&self) -> &'static str {
        "flather_2d"
    }
}

/// Tidal boundary condition for 2D (Dirichlet for elevation).
///
/// Prescribes water surface elevation, extrapolates velocity.
/// Use Flather BC for better wave absorption.
#[derive(Clone, Debug)]
pub struct Tidal2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// Function returning tidal elevation η(x, y, t)
    pub tidal_elevation: F,
    /// Minimum depth
    pub h_min: f64,
}

impl<F> Tidal2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// Create a new tidal BC.
    pub fn new(tidal_elevation: F) -> Self {
        Self {
            tidal_elevation,
            h_min: 1e-6,
        }
    }
}

impl<F> SWEBoundaryCondition2D for Tidal2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        let (x, y) = ctx.position;
        let t = ctx.time;

        // Prescribed surface elevation
        let eta_tidal = (self.tidal_elevation)(x, y, t);
        let h_ghost = (eta_tidal - ctx.bathymetry).max(self.h_min);

        // Extrapolate velocity from interior
        let (u, v) = ctx.interior_velocity();

        SWEState2D::from_primitives(h_ghost, u, v)
    }

    fn name(&self) -> &'static str {
        "tidal_2d"
    }
}

/// Discharge (river inflow) boundary condition for 2D.
///
/// Prescribes normal flow rate per unit width Q = h * u_n.
/// The depth is either prescribed or taken from interior.
#[derive(Clone, Debug)]
pub struct Discharge2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// Function returning discharge per unit width Q(x, y, t)
    pub discharge: F,
    /// Optional prescribed depth (if None, use interior depth)
    pub prescribed_depth: Option<f64>,
    /// Minimum depth
    pub h_min: f64,
}

impl<F> Discharge2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// Create a new discharge BC with interior depth.
    pub fn new(discharge: F) -> Self {
        Self {
            discharge,
            prescribed_depth: None,
            h_min: 1e-6,
        }
    }

    /// Create with prescribed depth.
    pub fn with_depth(discharge: F, depth: f64) -> Self {
        Self {
            discharge,
            prescribed_depth: Some(depth),
            h_min: 1e-6,
        }
    }
}

impl<F> SWEBoundaryCondition2D for Discharge2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        let (x, y) = ctx.position;
        let t = ctx.time;
        let (nx, ny) = ctx.normal;

        // Get depth
        let h = self
            .prescribed_depth
            .unwrap_or(ctx.interior_state.h)
            .max(self.h_min);

        // Prescribed discharge gives normal velocity
        let q = (self.discharge)(x, y, t);
        let un = -q / h; // Negative because discharge flows into domain (against outward normal)

        // Zero tangential velocity for river inflow
        let ut = 0.0;

        // Convert to (u, v)
        let u = un * nx - ut * ny;
        let v = un * ny + ut * nx;

        SWEState2D::from_primitives(h, u, v)
    }

    fn name(&self) -> &'static str {
        "discharge_2d"
    }

    fn allows_outflow(&self) -> bool {
        false
    }
}

/// Extrapolation (zero-gradient) boundary condition.
///
/// Simply copies the interior state to the exterior.
#[derive(Clone, Debug, Default)]
pub struct Extrapolation2D;

impl SWEBoundaryCondition2D for Extrapolation2D {
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        ctx.interior_state
    }

    fn name(&self) -> &'static str {
        "extrapolation_2d"
    }
}

/// Fixed state (Dirichlet) boundary condition.
#[derive(Clone, Debug)]
pub struct FixedState2D {
    /// Fixed state to impose
    pub state: SWEState2D,
}

impl FixedState2D {
    /// Create a new fixed state BC.
    pub fn new(h: f64, hu: f64, hv: f64) -> Self {
        Self {
            state: SWEState2D::new(h, hu, hv),
        }
    }

    /// Create from primitive variables.
    pub fn from_primitives(h: f64, u: f64, v: f64) -> Self {
        Self {
            state: SWEState2D::from_primitives(h, u, v),
        }
    }
}

impl SWEBoundaryCondition2D for FixedState2D {
    fn ghost_state(&self, _ctx: &BCContext2D) -> SWEState2D {
        self.state
    }

    fn name(&self) -> &'static str {
        "fixed_state_2d"
    }
}

// ============================================================================
// Convenience Boundary Conditions (Non-Generic)
// ============================================================================

/// Harmonic tidal constituent for 2D boundaries.
///
/// This is a re-export for convenience; see [`super::TidalConstituent`] for details.
pub use super::TidalConstituent;

/// Flather radiation BC with harmonic tidal forcing (non-generic version).
///
/// This is a convenience struct that stores tidal constituents directly,
/// avoiding the need for closures in common cases.
///
/// Supports smooth ramp-up to prevent initial impulse from tidal forcing.
///
/// # IMPORTANT: Bathymetry Convention
///
/// This BC uses surface elevation η = h + B where B is bathymetry (negative below MSL).
/// **You MUST set bathymetry correctly** in `SWE2DRhsConfig` for this BC to work properly.
///
/// For a domain with mean depth h0 (e.g., 50m), set:
/// ```text
/// bathymetry = Bathymetry2D::constant(n_elements, n_nodes, -h0);
/// ```
///
/// This ensures η = h + B = h0 + (-h0) = 0 at rest (surface at MSL).
///
/// **Without bathymetry**: η = h = 50m, causing the Flather relation to compute
/// spurious velocities of ~20 m/s, leading to immediate blow-up!
///
/// # Stability Note
///
/// `HarmonicFlather2D` includes velocity feedback from interior elevation.
/// In semi-closed basins (with reflective walls), this can create resonance.
/// Consider using:
/// - [`HarmonicTidal2D`] for simpler Dirichlet-type forcing (more stable)
/// - Sponge layers (`SpongeLayer2D`) to absorb reflected energy
/// - [`ChapmanFlather2D`] for combined Chapman-Flather treatment
///
/// # Example
///
/// ```
/// use dg_rs::boundary::{HarmonicFlather2D, TidalConstituent};
///
/// // Create Flather BC with M2 tide and 6-hour ramp-up
/// let m2 = TidalConstituent::m2(0.5, 0.0); // 0.5m amplitude, 0 phase
/// let bc = HarmonicFlather2D::new(vec![m2], 50.0) // 50m reference depth
///     .with_ramp_up(6.0 * 3600.0);
///
/// // IMPORTANT: Also set bathymetry in config:
/// // let bathymetry = Bathymetry2D::constant(n_elements, n_nodes, -50.0);
/// // let config = SWE2DRhsConfig::new(&eq, &bc).with_bathymetry(&bathymetry);
/// ```
#[derive(Clone, Debug)]
pub struct HarmonicFlather2D {
    /// Mean surface elevation
    pub mean_elevation: f64,
    /// Tidal constituents
    pub constituents: Vec<TidalConstituent>,
    /// Reference depth (below mean sea level)
    pub h_ref: f64,
    /// External velocity (typically zero)
    pub u_external: (f64, f64),
    /// Minimum depth
    pub h_min: f64,
    /// Ramp-up duration in seconds (None = no ramp-up)
    pub ramp_duration: Option<f64>,
}

impl HarmonicFlather2D {
    /// Create a new harmonic Flather BC.
    ///
    /// # Arguments
    /// * `constituents` - List of tidal constituents
    /// * `h_ref` - Reference depth below mean sea level
    pub fn new(constituents: Vec<TidalConstituent>, h_ref: f64) -> Self {
        Self {
            mean_elevation: 0.0,
            constituents,
            h_ref,
            u_external: (0.0, 0.0),
            h_min: 1e-6,
            ramp_duration: None,
        }
    }

    /// Create with mean elevation offset.
    pub fn with_mean_elevation(mut self, mean: f64) -> Self {
        self.mean_elevation = mean;
        self
    }

    /// Create with external velocity.
    pub fn with_external_velocity(mut self, u: f64, v: f64) -> Self {
        self.u_external = (u, v);
        self
    }

    /// Create M2-only tidal forcing.
    ///
    /// # Arguments
    /// * `amplitude` - M2 tidal amplitude (m)
    /// * `phase` - M2 tidal phase (radians)
    /// * `h_ref` - Reference depth (m)
    pub fn m2_only(amplitude: f64, phase: f64, h_ref: f64) -> Self {
        Self::new(vec![TidalConstituent::m2(amplitude, phase)], h_ref)
    }

    /// Create with M2 and S2 constituents (spring-neap cycle).
    pub fn m2_s2(m2_amp: f64, s2_amp: f64, h_ref: f64) -> Self {
        Self::new(
            vec![
                TidalConstituent::m2(m2_amp, 0.0),
                TidalConstituent::s2(s2_amp, 0.0),
            ],
            h_ref,
        )
    }

    /// Enable smooth ramp-up of tidal forcing.
    ///
    /// The ramp gradually increases tidal amplitudes from 0 to full amplitude
    /// over the specified duration. This prevents initial impulse from
    /// causing spurious oscillations.
    ///
    /// # Arguments
    /// * `duration` - Ramp-up period in seconds (typically 1-3 tidal periods)
    pub fn with_ramp_up(mut self, duration: f64) -> Self {
        self.ramp_duration = Some(duration);
        self
    }

    /// Compute ramp factor at time t.
    ///
    /// Returns:
    /// - 0 at t=0
    /// - 1 for t >= ramp_duration
    /// - Smooth Hermite interpolation in between: 3t² - 2t³
    pub fn ramp_factor(&self, t: f64) -> f64 {
        match self.ramp_duration {
            None => 1.0,
            Some(duration) if duration <= 0.0 => 1.0,
            Some(duration) => {
                if t <= 0.0 {
                    0.0
                } else if t >= duration {
                    1.0
                } else {
                    let tau = t / duration;
                    tau * tau * (3.0 - 2.0 * tau)
                }
            }
        }
    }

    /// Evaluate tidal elevation at time t.
    ///
    /// If ramp-up is enabled, the tidal constituents are scaled by the
    /// ramp factor. The mean elevation is NOT ramped.
    pub fn elevation(&self, t: f64) -> f64 {
        let ramp = self.ramp_factor(t);
        let mut eta = self.mean_elevation;
        for c in &self.constituents {
            eta += ramp * c.evaluate(t);
        }
        eta
    }
}

impl SWEBoundaryCondition2D for HarmonicFlather2D {
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        use std::sync::atomic::AtomicBool;
        static WARNED: AtomicBool = AtomicBool::new(false);

        let t = ctx.time;
        let (nx, ny) = ctx.normal;
        let g = ctx.g;

        // Compute tidal elevation
        let eta_tidal = self.elevation(t);
        let h_tidal = (eta_tidal - ctx.bathymetry + self.h_ref).max(self.h_min);

        // Validate bathymetry configuration (warns once if misconfigured)
        warn_once_if_misconfigured(
            &WARNED,
            "HarmonicFlather2D",
            ctx.interior_state.h,
            ctx.bathymetry,
            eta_tidal,
        );

        // Wave celerity
        let c_tidal = (g * h_tidal).sqrt();

        // Interior surface elevation
        let eta_int = ctx.interior_surface_elevation();

        // Flather relation
        let (u_ext, v_ext) = self.u_external;
        let un_ext = u_ext * nx + v_ext * ny;
        let un_ghost = un_ext + c_tidal * (eta_int - eta_tidal) / h_tidal;

        // Preserve tangential velocity
        let ut_ghost = ctx.interior_tangential_velocity();

        // Convert back to (u, v)
        let u_ghost = un_ghost * nx - ut_ghost * ny;
        let v_ghost = un_ghost * ny + ut_ghost * nx;

        SWEState2D::from_primitives(h_tidal, u_ghost, v_ghost)
    }

    fn name(&self) -> &'static str {
        "harmonic_flather_2d"
    }

    fn allows_outflow(&self) -> bool {
        true
    }
}

/// Tidal BC with harmonic forcing (non-generic, Dirichlet for elevation).
///
/// Prescribes water surface elevation from harmonic constituents.
/// Extrapolates velocity from interior.
///
/// Supports smooth ramp-up to prevent initial impulse from tidal forcing.
///
/// # Stability
///
/// This BC is more stable than [`HarmonicFlather2D`] because it doesn't include
/// velocity feedback. It works well in semi-closed basins where Flather can
/// cause resonance. However, it doesn't absorb outgoing waves, which can lead
/// to phase errors from reflections.
///
/// # IMPORTANT: Bathymetry Convention
///
/// This BC computes depth as `h_ghost = η_tidal - bathymetry`. For proper
/// behavior, ensure bathymetry is set correctly in `SWE2DRhsConfig`:
///
/// ```text
/// // For mean depth h0 = 50m:
/// let bathymetry = Bathymetry2D::constant(n_elements, n_nodes, -50.0);
/// let config = config.with_bathymetry(&bathymetry);
/// ```
///
/// This ensures h_ghost = 0 - (-50) = 50m when η_tidal = 0 (surface at MSL).
#[derive(Clone, Debug)]
pub struct HarmonicTidal2D {
    /// Mean surface elevation
    pub mean_elevation: f64,
    /// Tidal constituents
    pub constituents: Vec<TidalConstituent>,
    /// Minimum depth
    pub h_min: f64,
    /// Ramp-up duration in seconds (None = no ramp-up)
    pub ramp_duration: Option<f64>,
}

impl HarmonicTidal2D {
    /// Create a new harmonic tidal BC.
    pub fn new(constituents: Vec<TidalConstituent>) -> Self {
        Self {
            mean_elevation: 0.0,
            constituents,
            h_min: 1e-6,
            ramp_duration: None,
        }
    }

    /// Create with mean elevation.
    pub fn with_mean_elevation(mut self, mean: f64) -> Self {
        self.mean_elevation = mean;
        self
    }

    /// Create M2-only.
    pub fn m2_only(amplitude: f64, phase: f64) -> Self {
        Self::new(vec![TidalConstituent::m2(amplitude, phase)])
    }

    /// Set minimum depth threshold.
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }

    /// Enable smooth ramp-up of tidal forcing.
    ///
    /// The ramp gradually increases tidal amplitudes from 0 to full amplitude
    /// over the specified duration. This prevents initial impulse from
    /// causing spurious oscillations.
    ///
    /// # Arguments
    /// * `duration` - Ramp-up period in seconds (typically 1-3 tidal periods)
    pub fn with_ramp_up(mut self, duration: f64) -> Self {
        self.ramp_duration = Some(duration);
        self
    }

    /// Compute ramp factor at time t.
    pub fn ramp_factor(&self, t: f64) -> f64 {
        match self.ramp_duration {
            None => 1.0,
            Some(duration) if duration <= 0.0 => 1.0,
            Some(duration) => {
                if t <= 0.0 {
                    0.0
                } else if t >= duration {
                    1.0
                } else {
                    let tau = t / duration;
                    tau * tau * (3.0 - 2.0 * tau)
                }
            }
        }
    }

    /// Evaluate tidal elevation.
    ///
    /// If ramp-up is enabled, the tidal constituents are scaled by the
    /// ramp factor. The mean elevation is NOT ramped.
    pub fn elevation(&self, t: f64) -> f64 {
        let ramp = self.ramp_factor(t);
        let mut eta = self.mean_elevation;
        for c in &self.constituents {
            eta += ramp * c.evaluate(t);
        }
        eta
    }
}

impl SWEBoundaryCondition2D for HarmonicTidal2D {
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        let eta = self.elevation(ctx.time);
        let h_ghost = (eta - ctx.bathymetry).max(self.h_min);
        let (u, v) = ctx.interior_velocity();
        SWEState2D::from_primitives(h_ghost, u, v)
    }

    fn name(&self) -> &'static str {
        "harmonic_tidal_2d"
    }
}

/// Constant discharge BC (non-generic convenience).
///
/// Prescribes a constant flow rate per unit width.
#[derive(Clone, Debug)]
pub struct ConstantDischarge2D {
    /// Discharge per unit width (m²/s)
    pub discharge: f64,
    /// Optional prescribed depth
    pub prescribed_depth: Option<f64>,
    /// Minimum depth
    pub h_min: f64,
}

impl ConstantDischarge2D {
    /// Create with given discharge.
    ///
    /// # Arguments
    /// * `discharge` - Flow rate per unit width (m²/s), positive = into domain
    pub fn new(discharge: f64) -> Self {
        Self {
            discharge,
            prescribed_depth: None,
            h_min: 1e-6,
        }
    }

    /// Create with prescribed depth.
    pub fn with_depth(discharge: f64, depth: f64) -> Self {
        Self {
            discharge,
            prescribed_depth: Some(depth),
            h_min: 1e-6,
        }
    }
}

impl SWEBoundaryCondition2D for ConstantDischarge2D {
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        let (nx, ny) = ctx.normal;
        let h = self
            .prescribed_depth
            .unwrap_or(ctx.interior_state.h)
            .max(self.h_min);

        // Normal velocity from discharge (negative = inflow)
        let un = -self.discharge / h;
        let ut = 0.0;

        let u = un * nx - ut * ny;
        let v = un * ny + ut * nx;

        SWEState2D::from_primitives(h, u, v)
    }

    fn name(&self) -> &'static str {
        "constant_discharge_2d"
    }

    fn allows_outflow(&self) -> bool {
        self.discharge < 0.0 // Negative discharge means outflow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;
    const G: f64 = 10.0;
    const H_MIN: f64 = 1e-6;

    fn make_context(h: f64, hu: f64, hv: f64, normal: (f64, f64)) -> BCContext2D {
        BCContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(h, hu, hv),
            0.0,
            normal,
            G,
            H_MIN,
        )
    }

    #[test]
    fn test_reflective_still_water() {
        let bc = Reflective2D::new();
        let ctx = make_context(2.0, 0.0, 0.0, (1.0, 0.0));

        let ghost = bc.ghost_state(&ctx);

        assert!((ghost.h - 2.0).abs() < TOL);
        assert!(ghost.hu.abs() < TOL);
        assert!(ghost.hv.abs() < TOL);
    }

    #[test]
    fn test_reflective_normal_flow() {
        let bc = Reflective2D::new();

        // Flow towards x-boundary (normal = (1, 0))
        // h=2, u=3, v=0
        let ctx = make_context(2.0, 6.0, 0.0, (1.0, 0.0));
        let ghost = bc.ghost_state(&ctx);

        // Normal velocity should be reversed
        assert!((ghost.h - 2.0).abs() < TOL);
        assert!((ghost.hu - (-6.0)).abs() < TOL); // u reversed
        assert!(ghost.hv.abs() < TOL); // v unchanged
    }

    #[test]
    fn test_reflective_tangential_preserved() {
        let bc = Reflective2D::new();

        // Pure tangential flow (v=3, normal = (1, 0))
        let ctx = make_context(2.0, 0.0, 6.0, (1.0, 0.0));
        let ghost = bc.ghost_state(&ctx);

        // Tangential velocity should be preserved
        assert!((ghost.h - 2.0).abs() < TOL);
        assert!(ghost.hu.abs() < TOL); // No normal velocity
        assert!((ghost.hv - 6.0).abs() < TOL); // Tangential preserved
    }

    #[test]
    fn test_reflective_diagonal_normal() {
        let bc = Reflective2D::new();

        // 45-degree normal
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        let ctx = make_context(1.0, 2.0, 0.0, (sqrt2_inv, sqrt2_inv));
        let ghost = bc.ghost_state(&ctx);

        // Normal component: u·n = 2 * 1/√2 = √2
        // Reflected: u_ghost = u - 2*(u·n)*n
        let un = 2.0 * sqrt2_inv;
        let u_expected = 2.0 - 2.0 * un * sqrt2_inv;
        let v_expected = 0.0 - 2.0 * un * sqrt2_inv;

        assert!((ghost.h - 1.0).abs() < TOL);
        assert!((ghost.hu / ghost.h - u_expected).abs() < TOL);
        assert!((ghost.hv / ghost.h - v_expected).abs() < TOL);
    }

    #[test]
    fn test_reflective_zero_normal_flux() {
        let bc = Reflective2D::new();

        // Various flow directions
        for (u, v) in [(3.0, 0.0), (0.0, 3.0), (2.0, 1.0)] {
            let h = 2.0;
            let ctx = make_context(h, h * u, h * v, (1.0, 0.0));
            let ghost = bc.ghost_state(&ctx);

            // Average normal velocity should be zero
            let un_int = u;
            let un_ghost = ghost.hu / ghost.h;
            let un_avg = 0.5 * (un_int + un_ghost);

            assert!(
                un_avg.abs() < TOL,
                "Average normal velocity should be zero, got {}",
                un_avg
            );
        }
    }

    #[test]
    fn test_extrapolation() {
        let bc = Extrapolation2D;
        let ctx = make_context(2.0, 3.0, 4.0, (1.0, 0.0));

        let ghost = bc.ghost_state(&ctx);

        assert!((ghost.h - 2.0).abs() < TOL);
        assert!((ghost.hu - 3.0).abs() < TOL);
        assert!((ghost.hv - 4.0).abs() < TOL);
    }

    #[test]
    fn test_fixed_state() {
        let bc = FixedState2D::from_primitives(1.5, 0.5, 0.25);
        let ctx = make_context(2.0, 6.0, 4.0, (1.0, 0.0));

        let ghost = bc.ghost_state(&ctx);

        assert!((ghost.h - 1.5).abs() < TOL);
        assert!((ghost.hu - 0.75).abs() < TOL); // 1.5 * 0.5
        assert!((ghost.hv - 0.375).abs() < TOL); // 1.5 * 0.25
    }

    #[test]
    fn test_tidal_elevation() {
        // Constant tidal elevation
        let bc = Tidal2D::new(|_x, _y, _t| 0.5);
        let ctx = BCContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::from_primitives(2.0, 1.0, 0.5),
            -0.5, // Bathymetry
            (1.0, 0.0),
            G,
            H_MIN,
        );

        let ghost = bc.ghost_state(&ctx);

        // h = η - B = 0.5 - (-0.5) = 1.0
        assert!((ghost.h - 1.0).abs() < TOL);
        // Velocity extrapolated from interior
        assert!((ghost.hu / ghost.h - 1.0).abs() < TOL);
        assert!((ghost.hv / ghost.h - 0.5).abs() < TOL);
    }

    #[test]
    fn test_discharge() {
        // Constant discharge of 5 m²/s
        let bc = Discharge2D::with_depth(|_x, _y, _t| 5.0, 2.0);
        let ctx = make_context(2.0, 0.0, 0.0, (1.0, 0.0));

        let ghost = bc.ghost_state(&ctx);

        // h = 2.0 (prescribed)
        assert!((ghost.h - 2.0).abs() < TOL);
        // u_n = -Q/h = -5/2 = -2.5 (flowing into domain)
        assert!((ghost.hu / ghost.h - (-2.5)).abs() < TOL);
        // v = 0
        assert!(ghost.hv.abs() < TOL);
    }

    #[test]
    fn test_context_methods() {
        let ctx = BCContext2D::new(
            1.0,
            (5.0, 10.0),
            SWEState2D::from_primitives(2.0, 3.0, 1.0),
            0.5,
            (0.6, 0.8),
            G,
            H_MIN,
        );

        // Surface elevation
        assert!((ctx.interior_surface_elevation() - 2.5).abs() < TOL);

        // Normal velocity: u·n = 3*0.6 + 1*0.8 = 2.6
        assert!((ctx.interior_normal_velocity() - 2.6).abs() < TOL);

        // Tangential velocity: -u*ny + v*nx = -3*0.8 + 1*0.6 = -1.8
        assert!((ctx.interior_tangential_velocity() - (-1.8)).abs() < TOL);

        // Celerity
        assert!((ctx.interior_celerity() - (G * 2.0_f64).sqrt()).abs() < TOL);
    }
}
