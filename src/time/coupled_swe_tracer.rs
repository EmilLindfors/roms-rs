//! Coupled time integration for SWE and tracer transport.
//!
//! This module provides time integrators that advance both the shallow water
//! equations and tracer transport equations together. The coupling is:
//!
//! - **SWE → Tracer**: Velocity field from SWE advects tracers
//! - **Tracer → SWE**: Density from T,S creates baroclinic pressure gradient
//!
//! The integration uses SSP-RK3 with careful handling of stage times
//! for time-dependent boundary conditions.
//!
//! # Usage
//!
//! ```ignore
//! // Create coupled state
//! let mut state = CoupledState2D::new(swe, tracers);
//!
//! // Create coupled RHS functions
//! let rhs = CoupledRhs::new(&config, &tracer_config, baroclinic_source);
//!
//! // Run simulation
//! run_coupled_simulation(&mut state, t_end, &rhs, ...);
//! ```

use crate::boundary::SWEBoundaryCondition2D;
use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::types::ElementIndex;
use crate::solver::{
    KuzminParameter2D, SWE2DRhsConfig, SWESolution2D, SWEState2D, TVBParameter2D,
    Tracer2DRhsConfig, TracerBoundaryCondition2D, TracerBounds, TracerSolution2D, TracerState,
    apply_tracer_limiters_2d, apply_tracer_limiters_kuzmin_2d, compute_rhs_swe_2d,
    compute_rhs_tracer_2d,
};
use crate::source::{BaroclinicSource2D, compute_tracer_gradients};

/// Combined state for coupled SWE-tracer system.
#[derive(Clone)]
pub struct CoupledState2D {
    /// Shallow water solution (h, hu, hv)
    pub swe: SWESolution2D,
    /// Tracer solution (hT, hS)
    pub tracers: TracerSolution2D,
}

impl CoupledState2D {
    /// Create a new coupled state.
    pub fn new(swe: SWESolution2D, tracers: TracerSolution2D) -> Self {
        assert_eq!(swe.n_elements, tracers.n_elements);
        assert_eq!(swe.n_nodes, tracers.n_nodes);
        Self { swe, tracers }
    }

    /// Create with uniform initial conditions.
    pub fn uniform(
        n_elements: usize,
        n_nodes: usize,
        h: f64,
        u: f64,
        v: f64,
        temperature: f64,
        salinity: f64,
    ) -> Self {
        let mut swe = SWESolution2D::new(n_elements, n_nodes);
        for k in ElementIndex::iter(n_elements) {
            for i in 0..n_nodes {
                swe.set_state(k, i, SWEState2D::from_primitives(h, u, v));
            }
        }

        let tracers = TracerSolution2D::uniform(
            n_elements,
            n_nodes,
            h,
            TracerState::new(temperature, salinity),
        );

        Self { swe, tracers }
    }

    /// Scale both solutions.
    pub fn scale(&mut self, c: f64) {
        self.swe.scale(c);
        self.tracers.scale(c);
    }

    /// Axpy operation on both solutions.
    pub fn axpy(&mut self, c: f64, other: &Self) {
        self.swe.axpy(c, &other.swe);
        self.tracers.axpy(c, &other.tracers);
    }

    /// Copy from another state.
    pub fn copy_from(&mut self, other: &Self) {
        self.swe.copy_from(&other.swe);
        self.tracers.copy_from(&other.tracers);
    }

    /// Get total tracer content (for conservation checks).
    pub fn total_tracer_content(
        &self,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
    ) -> (f64, f64) {
        let total_h_t = self.tracers.integrate_h_t(ops, geom);
        let total_h_s = self.tracers.integrate_h_s(ops, geom);
        (total_h_t, total_h_s)
    }
}

/// RHS for the coupled system.
///
/// Contains both SWE RHS and tracer RHS.
#[derive(Clone)]
pub struct CoupledRhs2D {
    /// SWE RHS
    pub swe: SWESolution2D,
    /// Tracer RHS
    pub tracers: TracerSolution2D,
}

impl CoupledRhs2D {
    /// Create a new coupled RHS.
    pub fn new(swe: SWESolution2D, tracers: TracerSolution2D) -> Self {
        Self { swe, tracers }
    }

    /// Create zero RHS.
    pub fn zero(n_elements: usize, n_nodes: usize) -> Self {
        Self {
            swe: SWESolution2D::new(n_elements, n_nodes),
            tracers: TracerSolution2D::new(n_elements, n_nodes),
        }
    }
}

/// Type of tracer limiter to use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TracerLimiterType {
    /// No limiting (not recommended for production)
    None,
    /// TVB slope limiter + Zhang-Shu positivity
    Tvb,
    /// Kuzmin vertex-based limiter + Zhang-Shu positivity
    Kuzmin,
}

impl Default for TracerLimiterType {
    fn default() -> Self {
        Self::None
    }
}

/// Configuration for coupled time integration.
#[derive(Clone)]
pub struct CoupledTimeConfig {
    /// CFL number
    pub cfl: f64,
    /// Gravitational acceleration
    pub g: f64,
    /// Minimum depth
    pub h_min: f64,
    /// Whether to include baroclinic coupling
    pub baroclinic: bool,
    /// Horizontal diffusivity for tracers
    pub tracer_diffusivity: f64,
    /// Whether to apply tracer limiters (legacy, use limiter_type instead)
    pub apply_tracer_limiters: bool,
    /// Type of tracer limiter to use
    pub limiter_type: TracerLimiterType,
    /// TVB parameter for tracer slope limiting
    pub tracer_tvb: TVBParameter2D,
    /// Kuzmin parameter for vertex-based limiting
    pub kuzmin_params: KuzminParameter2D,
    /// Physical bounds for tracers
    pub tracer_bounds: TracerBounds,
}

impl CoupledTimeConfig {
    /// Create a new configuration.
    pub fn new(cfl: f64, g: f64, h_min: f64) -> Self {
        Self {
            cfl,
            g,
            h_min,
            baroclinic: true,
            tracer_diffusivity: 0.0,
            apply_tracer_limiters: false,
            limiter_type: TracerLimiterType::None,
            tracer_tvb: TVBParameter2D::default(),
            kuzmin_params: KuzminParameter2D::default(),
            tracer_bounds: TracerBounds::default(),
        }
    }

    /// Set baroclinic coupling.
    pub fn with_baroclinic(mut self, enabled: bool) -> Self {
        self.baroclinic = enabled;
        self
    }

    /// Set tracer diffusivity.
    pub fn with_diffusivity(mut self, kappa: f64) -> Self {
        self.tracer_diffusivity = kappa;
        self
    }

    /// Enable TVB tracer limiters with specified parameter and domain size.
    ///
    /// This enables both TVB slope limiting and positivity-preserving
    /// limiting for tracer fields.
    ///
    /// # Arguments
    /// * `tvb_m` - TVB parameter M (typically 10-100; larger = less limiting)
    /// * `domain_size` - Reference length scale for normalization (e.g., domain diagonal)
    pub fn with_tracer_limiters(mut self, tvb_m: f64, domain_size: f64) -> Self {
        self.apply_tracer_limiters = true;
        self.limiter_type = TracerLimiterType::Tvb;
        self.tracer_tvb = TVBParameter2D::with_domain_size(tvb_m, domain_size);
        self
    }

    /// Enable Kuzmin vertex-based tracer limiters.
    ///
    /// This enables vertex-based slope limiting and positivity-preserving
    /// limiting for tracer fields. The Kuzmin limiter uses vertex-patch
    /// stencils for better oscillation control than TVB.
    ///
    /// # Arguments
    /// * `relaxation` - Relaxation factor for bounds (1.0 = strict, 1.1 = 10% wider)
    pub fn with_kuzmin_limiters(mut self, relaxation: f64) -> Self {
        self.apply_tracer_limiters = true;
        self.limiter_type = TracerLimiterType::Kuzmin;
        self.kuzmin_params = KuzminParameter2D::relaxed(relaxation);
        self
    }

    /// Set custom tracer bounds (default: T ∈ [-2, 40]°C, S ∈ [0, 42] PSU).
    pub fn with_tracer_bounds(mut self, bounds: TracerBounds) -> Self {
        self.tracer_bounds = bounds;
        self
    }
}

/// Compute the coupled RHS for SWE and tracers.
///
/// This computes both the SWE RHS (with optional baroclinic source) and
/// the tracer RHS (advected by SWE velocity field).
#[allow(clippy::too_many_arguments)]
pub fn compute_coupled_rhs<SweBc, TracerBc>(
    state: &CoupledState2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    swe_config: &SWE2DRhsConfig<SweBc>,
    tracer_config: &Tracer2DRhsConfig<TracerBc>,
    time: f64,
) -> CoupledRhs2D
where
    SweBc: SWEBoundaryCondition2D,
    TracerBc: TracerBoundaryCondition2D,
{
    // Compute SWE RHS (including any configured source terms)
    let swe_rhs = compute_rhs_swe_2d(&state.swe, mesh, ops, geom, swe_config, time);

    // Compute tracer RHS using SWE velocity field
    let tracer_rhs = compute_rhs_tracer_2d(
        &state.tracers,
        &state.swe,
        mesh,
        ops,
        geom,
        tracer_config,
        time,
    );

    CoupledRhs2D::new(swe_rhs, tracer_rhs)
}

/// Compute the coupled RHS with baroclinic pressure gradient.
///
/// This is a more involved computation that:
/// 1. Computes tracer gradients
/// 2. Computes density from T,S
/// 3. Adds baroclinic source to SWE
/// 4. Computes tracer advection
#[allow(clippy::too_many_arguments)]
pub fn compute_coupled_rhs_baroclinic<SweBc, TracerBc>(
    state: &CoupledState2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    swe_config: &SWE2DRhsConfig<SweBc>,
    tracer_config: &Tracer2DRhsConfig<TracerBc>,
    baroclinic: &BaroclinicSource2D,
    time: f64,
) -> CoupledRhs2D
where
    SweBc: SWEBoundaryCondition2D,
    TracerBc: TracerBoundaryCondition2D,
{
    let h_min = swe_config.equation.h_min.meters();

    // Compute tracer gradients for baroclinic term
    let (d_t_dx, d_t_dy, d_s_dx, d_s_dy) =
        compute_tracer_gradients(&state.tracers, &state.swe, mesh, ops, geom, h_min);

    // Compute SWE RHS first (without baroclinic - we'll add it)
    let mut swe_rhs = compute_rhs_swe_2d(&state.swe, mesh, ops, geom, swe_config, time);

    // Add baroclinic source term
    for k in ElementIndex::iter(mesh.n_elements) {
        let ki = k.as_usize();
        for i in 0..ops.n_nodes {
            let swe_state = state.swe.get_state(k, i);
            let tracers = state.tracers.get_concentrations(k, i, swe_state.h, h_min);

            // Get tracer gradients at this node
            let grad_idx = ki * ops.n_nodes + i;
            let tracer_grads = (
                d_t_dx[grad_idx],
                d_t_dy[grad_idx],
                d_s_dx[grad_idx],
                d_s_dy[grad_idx],
            );

            // Compute baroclinic source
            let baroclinic_source = baroclinic.compute_source_from_tracers(
                swe_state.h,
                tracers.temperature,
                tracers.salinity,
                tracer_grads,
            );

            // Add to RHS
            let current = swe_rhs.get_state(k, i);
            swe_rhs.set_state(k, i, current + baroclinic_source);
        }
    }

    // Compute tracer RHS
    let tracer_rhs = compute_rhs_tracer_2d(
        &state.tracers,
        &state.swe,
        mesh,
        ops,
        geom,
        tracer_config,
        time,
    );

    CoupledRhs2D::new(swe_rhs, tracer_rhs)
}

/// Perform one step of SSP-RK3 for the coupled system.
///
/// This advances both SWE and tracer solutions together.
#[allow(clippy::too_many_arguments)]
pub fn ssp_rk3_coupled_step<F>(state: &mut CoupledState2D, rhs_fn: F, dt: f64)
where
    F: Fn(&CoupledState2D) -> CoupledRhs2D,
{
    let n_elements = state.swe.n_elements;
    let n_nodes = state.swe.n_nodes;

    // Stage 1: q1 = q + dt * L(q)
    let rhs = rhs_fn(state);
    let mut state1 = state.clone();
    state1.swe.axpy(dt, &rhs.swe);
    state1.tracers.axpy(dt, &rhs.tracers);

    // Stage 2: q2 = 3/4 * q + 1/4 * q1 + 1/4 * dt * L(q1)
    let rhs1 = rhs_fn(&state1);
    let mut state2 = CoupledState2D::new(
        SWESolution2D::new(n_elements, n_nodes),
        TracerSolution2D::new(n_elements, n_nodes),
    );
    state2.copy_from(state);
    state2.scale(0.75);
    state2.axpy(0.25, &state1);
    state2.swe.axpy(0.25 * dt, &rhs1.swe);
    state2.tracers.axpy(0.25 * dt, &rhs1.tracers);

    // Stage 3: q_new = 1/3 * q + 2/3 * q2 + 2/3 * dt * L(q2)
    let rhs2 = rhs_fn(&state2);
    state.scale(1.0 / 3.0);
    state.axpy(2.0 / 3.0, &state2);
    state.swe.axpy(2.0 / 3.0 * dt, &rhs2.swe);
    state.tracers.axpy(2.0 / 3.0 * dt, &rhs2.tracers);
}

/// Perform one step of SSP-RK3 for the coupled system with time-dependent RHS.
///
/// Stage times for SSP-RK3:
/// - Stage 1: t
/// - Stage 2: t + dt
/// - Stage 3: t + dt/2
#[allow(clippy::too_many_arguments)]
pub fn ssp_rk3_coupled_step_timed<F>(state: &mut CoupledState2D, rhs_fn: F, t: f64, dt: f64)
where
    F: Fn(&CoupledState2D, f64) -> CoupledRhs2D,
{
    let n_elements = state.swe.n_elements;
    let n_nodes = state.swe.n_nodes;

    // Stage 1: q1 = q + dt * L(q, t)
    let rhs = rhs_fn(state, t);
    let mut state1 = state.clone();
    state1.swe.axpy(dt, &rhs.swe);
    state1.tracers.axpy(dt, &rhs.tracers);

    // Stage 2: q2 = 3/4 * q + 1/4 * q1 + 1/4 * dt * L(q1, t + dt)
    let t1 = t + dt;
    let rhs1 = rhs_fn(&state1, t1);
    let mut state2 = CoupledState2D::new(
        SWESolution2D::new(n_elements, n_nodes),
        TracerSolution2D::new(n_elements, n_nodes),
    );
    state2.copy_from(state);
    state2.scale(0.75);
    state2.axpy(0.25, &state1);
    state2.swe.axpy(0.25 * dt, &rhs1.swe);
    state2.tracers.axpy(0.25 * dt, &rhs1.tracers);

    // Stage 3: q_new = 1/3 * q + 2/3 * q2 + 2/3 * dt * L(q2, t + dt/2)
    let t2 = t + 0.5 * dt;
    let rhs2 = rhs_fn(&state2, t2);
    state.scale(1.0 / 3.0);
    state.axpy(2.0 / 3.0, &state2);
    state.swe.axpy(2.0 / 3.0 * dt, &rhs2.swe);
    state.tracers.axpy(2.0 / 3.0 * dt, &rhs2.tracers);
}

/// Apply the appropriate tracer limiter based on configuration.
fn apply_configured_limiter(
    tracers: &mut TracerSolution2D,
    swe: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    config: &CoupledTimeConfig,
) {
    match config.limiter_type {
        TracerLimiterType::None => {}
        TracerLimiterType::Tvb => {
            apply_tracer_limiters_2d(
                tracers,
                swe,
                mesh,
                ops,
                &config.tracer_tvb,
                &config.tracer_bounds,
                config.h_min,
            );
        }
        TracerLimiterType::Kuzmin => {
            apply_tracer_limiters_kuzmin_2d(
                tracers,
                swe,
                mesh,
                ops,
                &config.kuzmin_params,
                &config.tracer_bounds,
                config.h_min,
            );
        }
    }
}

/// Perform one step of SSP-RK3 for the coupled system with tracer limiters.
///
/// This variant applies tracer limiters after each RK3 stage to maintain
/// physical bounds and prevent oscillations.
///
/// Stage times for SSP-RK3:
/// - Stage 1: t
/// - Stage 2: t + dt
/// - Stage 3: t + dt/2
///
/// After each stage, if `config.apply_tracer_limiters` is true, the configured
/// limiter type is applied:
/// - `TracerLimiterType::Tvb`: TVB slope limiter + positivity
/// - `TracerLimiterType::Kuzmin`: Vertex-based Kuzmin limiter + positivity
#[allow(clippy::too_many_arguments)]
pub fn ssp_rk3_coupled_step_limited<F>(
    state: &mut CoupledState2D,
    rhs_fn: F,
    t: f64,
    dt: f64,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    config: &CoupledTimeConfig,
) where
    F: Fn(&CoupledState2D, f64) -> CoupledRhs2D,
{
    let n_elements = state.swe.n_elements;
    let n_nodes = state.swe.n_nodes;

    // Stage 1: q1 = q + dt * L(q, t)
    let rhs = rhs_fn(state, t);
    let mut state1 = state.clone();
    state1.swe.axpy(dt, &rhs.swe);
    state1.tracers.axpy(dt, &rhs.tracers);

    // Apply limiters after stage 1
    if config.apply_tracer_limiters {
        apply_configured_limiter(&mut state1.tracers, &state1.swe, mesh, ops, config);
    }

    // Stage 2: q2 = 3/4 * q + 1/4 * q1 + 1/4 * dt * L(q1, t + dt)
    let t1 = t + dt;
    let rhs1 = rhs_fn(&state1, t1);
    let mut state2 = CoupledState2D::new(
        SWESolution2D::new(n_elements, n_nodes),
        TracerSolution2D::new(n_elements, n_nodes),
    );
    state2.copy_from(state);
    state2.scale(0.75);
    state2.axpy(0.25, &state1);
    state2.swe.axpy(0.25 * dt, &rhs1.swe);
    state2.tracers.axpy(0.25 * dt, &rhs1.tracers);

    // Apply limiters after stage 2
    if config.apply_tracer_limiters {
        apply_configured_limiter(&mut state2.tracers, &state2.swe, mesh, ops, config);
    }

    // Stage 3: q_new = 1/3 * q + 2/3 * q2 + 2/3 * dt * L(q2, t + dt/2)
    let t2 = t + 0.5 * dt;
    let rhs2 = rhs_fn(&state2, t2);
    state.scale(1.0 / 3.0);
    state.axpy(2.0 / 3.0, &state2);
    state.swe.axpy(2.0 / 3.0 * dt, &rhs2.swe);
    state.tracers.axpy(2.0 / 3.0 * dt, &rhs2.tracers);

    // Apply limiters after final stage
    if config.apply_tracer_limiters {
        apply_configured_limiter(&mut state.tracers, &state.swe, mesh, ops, config);
    }
}

/// Compute stable time step for the coupled system.
///
/// Takes the minimum of SWE and tracer time steps.
pub fn compute_dt_coupled(
    state: &CoupledState2D,
    mesh: &Mesh2D,
    geom: &GeometricFactors2D,
    config: &CoupledTimeConfig,
    order: usize,
) -> f64 {
    use crate::equations::ShallowWater2D;
    use crate::solver::compute_dt_swe_2d;

    let equation = ShallowWater2D::new(config.g);

    let dt_swe = compute_dt_swe_2d(&state.swe, mesh, geom, &equation, order, config.cfl);

    let dt_tracer = crate::solver::compute_dt_tracer_2d(
        &state.swe,
        mesh,
        geom,
        config.g,
        config.h_min,
        config.tracer_diffusivity,
        order,
        config.cfl,
    );

    dt_swe.min(dt_tracer)
}

/// Run a complete coupled simulation.
///
/// # Arguments
/// * `state` - Initial coupled state (modified in place)
/// * `t_end` - End time
/// * `rhs_fn` - RHS function (state, time) -> RHS
/// * `dt_fn` - Function to compute dt given current state
/// * `callback` - Optional callback(t, state) called after each step
///
/// # Returns
/// Final time reached and number of steps taken.
#[allow(clippy::too_many_arguments)]
pub fn run_coupled_simulation<F, D, C>(
    state: &mut CoupledState2D,
    t_end: f64,
    rhs_fn: F,
    dt_fn: D,
    mut callback: Option<C>,
) -> (f64, usize)
where
    F: Fn(&CoupledState2D, f64) -> CoupledRhs2D,
    D: Fn(&CoupledState2D) -> f64,
    C: FnMut(f64, &CoupledState2D),
{
    let mut t = 0.0;
    let mut n_steps = 0;

    while t < t_end {
        let mut dt = dt_fn(state);

        if t + dt > t_end {
            dt = t_end - t;
        }

        ssp_rk3_coupled_step_timed(state, &rhs_fn, t, dt);

        t += dt;
        n_steps += 1;

        if let Some(ref mut cb) = callback {
            cb(t, state);
        }
    }

    (t, n_steps)
}

/// Run a complete coupled simulation with tracer limiters.
///
/// This variant applies tracer limiters after each RK3 stage to maintain
/// physical bounds and prevent oscillations.
///
/// # Arguments
/// * `state` - Initial coupled state (modified in place)
/// * `t_end` - End time
/// * `rhs_fn` - RHS function (state, time) -> RHS
/// * `dt_fn` - Function to compute dt given current state
/// * `mesh` - 2D mesh
/// * `ops` - DG operators
/// * `config` - Time configuration (includes limiter settings)
/// * `callback` - Optional callback(t, state) called after each step
///
/// # Returns
/// Final time reached and number of steps taken.
#[allow(clippy::too_many_arguments)]
pub fn run_coupled_simulation_limited<F, D, C>(
    state: &mut CoupledState2D,
    t_end: f64,
    rhs_fn: F,
    dt_fn: D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    config: &CoupledTimeConfig,
    mut callback: Option<C>,
) -> (f64, usize)
where
    F: Fn(&CoupledState2D, f64) -> CoupledRhs2D,
    D: Fn(&CoupledState2D) -> f64,
    C: FnMut(f64, &CoupledState2D),
{
    let mut t = 0.0;
    let mut n_steps = 0;

    while t < t_end {
        let mut dt = dt_fn(state);

        if t + dt > t_end {
            dt = t_end - t;
        }

        ssp_rk3_coupled_step_limited(state, &rhs_fn, t, dt, mesh, ops, config);

        t += dt;
        n_steps += 1;

        if let Some(ref mut cb) = callback {
            cb(t, state);
        }
    }

    (t, n_steps)
}

/// Compute total mass in the coupled system.
pub fn total_mass(state: &CoupledState2D, ops: &DGOperators2D, geom: &GeometricFactors2D) -> f64 {
    let mut mass = 0.0;
    for k in ElementIndex::iter(state.swe.n_elements) {
        let ki = k.as_usize();
        let j = geom.det_j[ki];
        for (i, &w) in ops.weights.iter().enumerate() {
            let h = state.swe.get_var(k, i, 0);
            mass += w * h * j;
        }
    }
    mass
}

/// Compute total tracer content (hT, hS) in the coupled system.
pub fn total_tracer(
    state: &CoupledState2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
) -> (f64, f64) {
    state.total_tracer_content(ops, geom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::Reflective2D;
    use crate::equations::ShallowWater2D;
    use crate::solver::{ExtrapolationTracerBC, Tracer2DRhsConfig};
    use crate::types::ElementIndex;

    fn k(idx: usize) -> ElementIndex {
        ElementIndex::new(idx)
    }

    const G: f64 = 9.81;
    const H_MIN: f64 = 1e-6;
    const TOL: f64 = 1e-10;

    fn create_test_setup() -> (Mesh2D, DGOperators2D, GeometricFactors2D) {
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        (mesh, ops, geom)
    }

    #[test]
    fn test_coupled_state() {
        let n_elem = 4;
        let n_nodes = 9;

        let state = CoupledState2D::uniform(n_elem, n_nodes, 10.0, 1.0, 0.5, 8.0, 34.0);

        assert_eq!(state.swe.n_elements, n_elem);
        assert_eq!(state.tracers.n_elements, n_elem);

        // Check values
        let swe = state.swe.get_state(k(0), 0);
        assert!((swe.h - 10.0).abs() < TOL);

        let tracers = state.tracers.get_concentrations(k(0), 0, 10.0, H_MIN);
        assert!((tracers.temperature - 8.0).abs() < TOL);
        assert!((tracers.salinity - 34.0).abs() < TOL);
    }

    #[test]
    fn test_coupled_rhs_uniform() {
        let (mesh, ops, geom) = create_test_setup();

        let equation = ShallowWater2D::new(G);
        let swe_bc = Reflective2D::new();
        let swe_config = SWE2DRhsConfig::new(&equation, &swe_bc).with_coriolis(false);

        let tracer_bc = ExtrapolationTracerBC;
        let tracer_config = Tracer2DRhsConfig::new(&tracer_bc, G, H_MIN);

        // Uniform state should give zero RHS
        let state =
            CoupledState2D::uniform(mesh.n_elements, ops.n_nodes, 10.0, 1.0, 0.0, 8.0, 34.0);

        let rhs = compute_coupled_rhs(&state, &mesh, &ops, &geom, &swe_config, &tracer_config, 0.0);

        // Check RHS is approximately zero
        let max_swe = rhs.swe.max_abs();
        let mut max_tracer: f64 = 0.0;
        for elem in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let t = rhs.tracers.get_conservative(elem, i);
                max_tracer = max_tracer.max(t.h_t.abs()).max(t.h_s.abs());
            }
        }

        assert!(max_swe < TOL, "SWE RHS should be ~0, got {}", max_swe);
        assert!(
            max_tracer < TOL,
            "Tracer RHS should be ~0, got {}",
            max_tracer
        );
    }

    #[test]
    fn test_ssp_rk3_coupled_still_water() {
        let (mesh, ops, geom) = create_test_setup();

        let mut state =
            CoupledState2D::uniform(mesh.n_elements, ops.n_nodes, 10.0, 0.0, 0.0, 8.0, 34.0);

        let initial_mass = total_mass(&state, &ops, &geom);
        let (initial_h_t, initial_h_s) = total_tracer(&state, &ops, &geom);

        // Zero RHS for still water
        let rhs_fn = |_: &CoupledState2D| CoupledRhs2D::zero(mesh.n_elements, ops.n_nodes);

        ssp_rk3_coupled_step(&mut state, rhs_fn, 0.01);

        // Should be unchanged
        let final_mass = total_mass(&state, &ops, &geom);
        let (final_h_t, final_h_s) = total_tracer(&state, &ops, &geom);

        assert!((initial_mass - final_mass).abs() < TOL);
        assert!((initial_h_t - final_h_t).abs() < TOL);
        assert!((initial_h_s - final_h_s).abs() < TOL);
    }

    #[test]
    fn test_tracer_conservation() {
        let (mesh, ops, geom) = create_test_setup();

        let equation = ShallowWater2D::new(G);
        let swe_bc = Reflective2D::new();
        let swe_config = SWE2DRhsConfig::new(&equation, &swe_bc).with_coriolis(false);

        let tracer_bc = ExtrapolationTracerBC;
        let tracer_config = Tracer2DRhsConfig::new(&tracer_bc, G, H_MIN);

        // Non-uniform tracers with uniform flow
        let mut state =
            CoupledState2D::uniform(mesh.n_elements, ops.n_nodes, 10.0, 1.0, 0.0, 8.0, 34.0);

        // Add perturbation to tracers
        for elem in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [x, _y] = mesh.reference_to_physical(elem, r, s);
                let t_perturb = 0.5 * (2.0 * std::f64::consts::PI * x).sin();
                let h = state.swe.get_state(elem, i).h;
                let cons = state.tracers.get_conservative(elem, i);
                state.tracers.set_conservative(
                    elem,
                    i,
                    crate::solver::ConservativeTracerState::new(cons.h_t + h * t_perturb, cons.h_s),
                );
            }
        }

        let (_initial_h_t, _initial_h_s) = total_tracer(&state, &ops, &geom);

        // Compute RHS
        let rhs = compute_coupled_rhs(&state, &mesh, &ops, &geom, &swe_config, &tracer_config, 0.0);

        // Integrate tracer RHS - should be zero for conservation
        let mut integral_h_t = 0.0;
        let mut integral_h_s = 0.0;
        for elem in ElementIndex::iter(mesh.n_elements) {
            let ki = elem.as_usize();
            let j = geom.det_j[ki];
            for (i, &w) in ops.weights.iter().enumerate() {
                let t = rhs.tracers.get_conservative(elem, i);
                integral_h_t += w * t.h_t * j;
                integral_h_s += w * t.h_s * j;
            }
        }

        assert!(
            integral_h_t.abs() < 1e-10,
            "hT should be conserved: d(total)/dt = {:.2e}",
            integral_h_t
        );
        assert!(
            integral_h_s.abs() < 1e-10,
            "hS should be conserved: d(total)/dt = {:.2e}",
            integral_h_s
        );
    }

    #[test]
    fn test_dt_coupled() {
        let (mesh, ops, geom) = create_test_setup();

        let state =
            CoupledState2D::uniform(mesh.n_elements, ops.n_nodes, 10.0, 2.0, 1.0, 8.0, 34.0);

        let config = CoupledTimeConfig::new(0.5, G, H_MIN);
        let dt = compute_dt_coupled(&state, &mesh, &geom, &config, ops.order);

        assert!(dt > 0.0);
        assert!(dt < f64::INFINITY);
        assert!(dt < 0.1, "dt should be reasonably small");
    }
}
