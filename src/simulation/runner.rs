//! Simulation runner implementation.
//!
//! Provides a high-level interface for running time-dependent simulations.

use crate::physics::PhysicsModule;
use crate::time::{Integrable, StageWorkspace, TimeIntegrator};

// =============================================================================
// Simulation Configuration
// =============================================================================

/// Configuration for a simulation run.
#[derive(Clone, Debug)]
pub struct SimulationConfig {
    /// CFL number for time step calculation (capped by `PhysicsModule::max_cfl`).
    pub cfl: f64,
    /// Maximum time step (overrides CFL if smaller).
    pub dt_max: Option<f64>,
    /// Minimum time step (simulation fails if dt drops below this).
    pub dt_min: Option<f64>,
    /// Interval for calling callbacks (in simulation time units).
    pub callback_interval: Option<f64>,
    /// Maximum number of time steps.
    pub max_steps: Option<usize>,
    /// Whether to print progress to stdout.
    pub verbose: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            cfl: 0.5,
            dt_max: None,
            dt_min: None,
            callback_interval: None,
            max_steps: None,
            verbose: false,
        }
    }
}

// =============================================================================
// Simulation Result
// =============================================================================

/// Result of a simulation run.
#[derive(Clone, Debug)]
pub struct SimulationResult {
    /// Final simulation time reached.
    pub final_time: f64,
    /// Total number of time steps taken.
    pub n_steps: usize,
    /// Minimum time step used.
    pub dt_min: f64,
    /// Maximum time step used.
    pub dt_max: f64,
    /// Total wall-clock time in seconds.
    pub wall_time: f64,
    /// Whether the simulation completed successfully.
    pub success: bool,
    /// Error message if simulation failed.
    pub error: Option<String>,
}

impl SimulationResult {
    /// Create a successful result.
    pub fn success(
        final_time: f64,
        n_steps: usize,
        dt_min: f64,
        dt_max: f64,
        wall_time: f64,
    ) -> Self {
        Self {
            final_time,
            n_steps,
            dt_min,
            dt_max,
            wall_time,
            success: true,
            error: None,
        }
    }

    /// Create a failed result.
    pub fn failure(final_time: f64, n_steps: usize, error: String) -> Self {
        Self {
            final_time,
            n_steps,
            dt_min: f64::INFINITY,
            dt_max: 0.0,
            wall_time: 0.0,
            success: false,
            error: Some(error),
        }
    }
}

// =============================================================================
// Simulation Runner
// =============================================================================

/// High-level simulation runner.
///
/// Ties together physics modules and time integrators into a complete
/// simulation workflow with diagnostics and callbacks.
///
/// # Type Parameters
///
/// * `S` - Solution type (must implement [`Integrable`])
/// * `P` - Physics module (must implement [`PhysicsModule<S>`])
/// * `I` - Time integrator (must implement [`TimeIntegrator<S>`])
pub struct Simulation<S, P, I>
where
    S: Integrable,
    P: PhysicsModule<S>,
    I: TimeIntegrator<S>,
{
    physics: P,
    integrator: I,
    config: SimulationConfig,
    _marker: std::marker::PhantomData<S>,
}

impl<S, P, I> Simulation<S, P, I>
where
    S: Integrable,
    P: PhysicsModule<S>,
    I: TimeIntegrator<S>,
{
    /// Create a new simulation with the given physics and integrator.
    pub fn new(physics: P, integrator: I) -> Self {
        Self {
            physics,
            integrator,
            config: SimulationConfig::default(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the CFL number.
    pub fn with_cfl(mut self, cfl: f64) -> Self {
        self.config.cfl = cfl;
        self
    }

    /// Set the maximum time step.
    pub fn with_dt_max(mut self, dt_max: f64) -> Self {
        self.config.dt_max = Some(dt_max);
        self
    }

    /// Set the minimum time step (simulation fails if dt drops below).
    pub fn with_dt_min(mut self, dt_min: f64) -> Self {
        self.config.dt_min = Some(dt_min);
        self
    }

    /// Call back at `t_start + k·interval` exactly (the step before each
    /// callback is shortened to land on it), e.g. for regularly sampled
    /// station series and output times. Without an interval, every step.
    pub fn with_callback_interval(mut self, interval: f64) -> Self {
        assert!(interval > 0.0, "callback interval must be positive");
        self.config.callback_interval = Some(interval);
        self
    }

    /// Set the maximum number of steps.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.config.max_steps = Some(max_steps);
        self
    }

    /// Enable verbose output.
    pub fn verbose(mut self) -> Self {
        self.config.verbose = true;
        self
    }

    /// Get a reference to the physics module.
    pub fn physics(&self) -> &P {
        &self.physics
    }

    /// Get a reference to the time integrator.
    pub fn integrator(&self) -> &I {
        &self.integrator
    }

    /// Run the simulation from `t_start` to `t_end`.
    ///
    /// # Arguments
    /// * `state` - Initial solution state (modified in place)
    /// * `t_start` - Starting time
    /// * `t_end` - Ending time
    ///
    /// # Returns
    /// Simulation result with timing and step statistics.
    pub fn run(&self, state: &mut S, t_start: f64, t_end: f64) -> SimulationResult {
        self.run_with_callback(state, t_start, t_end, |_, _| {})
    }

    /// Run the simulation with a callback function.
    ///
    /// The callback is called at the configured interval (or every step if not set).
    ///
    /// # Arguments
    /// * `state` - Initial solution state (modified in place)
    /// * `t_start` - Starting time
    /// * `t_end` - Ending time
    /// * `callback` - Function called with (state, time) at each callback point
    pub fn run_with_callback<F>(
        &self,
        state: &mut S,
        t_start: f64,
        t_end: f64,
        mut callback: F,
    ) -> SimulationResult
    where
        F: FnMut(&S, f64),
    {
        let start_wall = std::time::Instant::now();

        let mut t = t_start;
        let mut n_steps = 0;
        let mut dt_min_used = f64::INFINITY;
        let mut dt_max_used: f64 = 0.0;
        // Callbacks fall exactly on t_start + k·interval: the step before
        // one is shortened to land on it (k counts, so no drift accumulates)
        let mut callbacks_done = 0_u64;
        let next_callback = |k: u64| {
            self.config
                .callback_interval
                .map(|interval| t_start + (k + 1) as f64 * interval)
        };
        // Stage buffers reused for the whole run (no per-step allocation)
        let mut stages = StageWorkspace::new();
        // e.g. the positivity bound of a wet/dry scheme
        let cfl = self
            .physics
            .max_cfl()
            .map_or(self.config.cfl, |max| self.config.cfl.min(max));

        // Call initial callback
        callback(state, t);

        if self.config.verbose {
            println!(
                "Starting simulation: {} with {} integrator",
                self.physics.name(),
                self.integrator.name()
            );
            println!("  t_start = {:.4}, t_end = {:.4}", t_start, t_end);
            if cfl < self.config.cfl {
                println!(
                    "  CFL {} capped at {cfl:.3} by {}",
                    self.config.cfl,
                    self.physics.name()
                );
            }
        }

        while t < t_end {
            // Check step limit
            if let Some(max_steps) = self.config.max_steps
                && n_steps >= max_steps
            {
                return SimulationResult::failure(
                    t,
                    n_steps,
                    format!("Maximum step limit ({}) reached", max_steps),
                );
            }

            // Compute time step
            let mut dt = self.physics.compute_dt(state, cfl);

            // Apply dt limits
            if let Some(dt_max) = self.config.dt_max {
                dt = dt.min(dt_max);
            }

            // Check minimum dt
            if let Some(dt_min) = self.config.dt_min
                && dt < dt_min
            {
                return SimulationResult::failure(
                    t,
                    n_steps,
                    format!("Time step ({:.2e}) below minimum ({:.2e})", dt, dt_min),
                );
            }

            // Don't overshoot the end time or the next callback time
            let target = next_callback(callbacks_done).map_or(t_end, |c| c.min(t_end));
            let landed = t + dt >= target;
            if landed {
                dt = target - t;
            }

            // Track dt statistics
            dt_min_used = dt_min_used.min(dt);
            dt_max_used = dt_max_used.max(dt);

            // Advance the solution. Stiff damping (friction, wet/dry
            // relaxation) is implicit in every RK stage; limiters
            // and wet/dry treatment run after every RK stage, so no RHS
            // evaluation sees an unlimited state.
            self.integrator.step_with_relaxation(
                state,
                dt,
                t,
                |s, time, out| self.physics.compute_rhs_into(s, time, out),
                |stage, from, dt| self.physics.implicit_damping(stage, from, dt),
                |s| self.physics.post_process(s),
                &mut stages,
            );

            // Exact landing: no round-off drift from t + (target − t)
            t = if landed { target } else { t + dt };
            n_steps += 1;

            // Callback at the configured interval, or every step
            match next_callback(callbacks_done) {
                Some(c) if t >= c => {
                    callback(state, t);
                    callbacks_done += 1;
                }
                Some(_) => {}
                None => callback(state, t),
            }

            // Progress output
            if self.config.verbose && n_steps % 100 == 0 {
                println!("  Step {}: t = {:.4}, dt = {:.2e}", n_steps, t, dt);
            }
        }

        let wall_time = start_wall.elapsed().as_secs_f64();

        if self.config.verbose {
            println!("Simulation complete:");
            println!("  Steps: {}", n_steps);
            println!("  Wall time: {:.2}s", wall_time);
            println!("  dt range: [{:.2e}, {:.2e}]", dt_min_used, dt_max_used);
        }

        SimulationResult::success(t, n_steps, dt_min_used, dt_max_used, wall_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::boundary::Reflective2D;
    use crate::equations::ShallowWater2D;
    use crate::mesh::Mesh2D;
    use crate::operators::{DGOperators2D, GeometricFactors2D};
    use crate::physics::PhysicsBuilder;
    use crate::solver::SWESolution2D;
    use crate::types::{Depth, ElementIndex};

    fn k(idx: usize) -> ElementIndex {
        ElementIndex::new(idx)
    }
    use crate::time::SSPRK3;

    fn create_test_setup() -> (crate::physics::SWEPhysics2D<Reflective2D>, SWESolution2D) {
        let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2));
        let ops = Arc::new(DGOperators2D::new(2));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let equation = ShallowWater2D::with_h_min(9.81, Depth::new(1e-6));
        let bc = Reflective2D::default();

        let physics = PhysicsBuilder::swe_2d(mesh.clone(), ops.clone(), geom, equation, bc).build();

        let mut state = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                state.set_state(k(ki), i, crate::solver::SWEState2D::new(1.0, 0.0, 0.0));
            }
        }

        (physics, state)
    }

    #[test]
    fn test_simulation_basic() {
        let (physics, mut state) = create_test_setup();
        let integrator = SSPRK3;

        let sim = Simulation::new(physics, integrator).with_cfl(0.5);

        let result = sim.run(&mut state, 0.0, 0.01);

        assert!(result.success);
        assert!(result.n_steps > 0);
        assert!(result.final_time >= 0.01 - 1e-10);
    }

    #[test]
    fn test_simulation_with_callback() {
        let (physics, mut state) = create_test_setup();
        let integrator = SSPRK3;

        let sim = Simulation::new(physics, integrator).with_cfl(0.5);

        let mut callback_count = 0;
        let result = sim.run_with_callback(&mut state, 0.0, 0.01, |_state, _time| {
            callback_count += 1;
        });

        assert!(result.success);
        assert!(callback_count > 0);
    }

    /// An interval that is not a multiple of dt: callbacks land exactly on
    /// t_start + k·interval (they used to fire at the first step past it and
    /// drift by up to one dt per interval), and the run still ends at t_end.
    #[test]
    fn callbacks_land_on_the_interval() {
        let (physics, mut state) = create_test_setup();
        let sim = Simulation::new(physics, SSPRK3)
            .with_cfl(0.5)
            .with_callback_interval(0.0037);
        let (t0, t_end) = (0.5, 0.52);
        let mut times = Vec::new();
        let result = sim.run_with_callback(&mut state, t0, t_end, |_, t| times.push(t));

        assert!(result.success);
        assert_eq!(result.final_time, t_end);
        assert_eq!(times.len(), 6, "{times:?}"); // t0 and 5 intervals
        for (k, &t) in times.iter().enumerate() {
            let expected = t0 + k as f64 * 0.0037;
            assert!(
                (t - expected).abs() < 1e-15,
                "callback {k} at {t}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_simulation_max_steps() {
        let (physics, mut state) = create_test_setup();
        let integrator = SSPRK3;

        let sim = Simulation::new(physics, integrator)
            .with_cfl(0.5)
            .with_max_steps(5);

        let result = sim.run(&mut state, 0.0, 100.0);

        assert!(!result.success);
        assert_eq!(result.n_steps, 5);
    }

    /// dh/dt = −1 everywhere; `post_process` clamps h ≥ 0. Records the smallest
    /// depth any RHS evaluation saw and how often `post_process` ran.
    struct DrainingPhysics {
        mesh: Mesh2D,
        ops: DGOperators2D,
        geom: GeometricFactors2D,
        min_h_seen: std::sync::Mutex<f64>,
        post_process_calls: std::sync::atomic::AtomicUsize,
    }

    impl crate::physics::PhysicsModuleInfo for DrainingPhysics {
        fn name(&self) -> &'static str {
            "draining"
        }
        fn description(&self) -> &str {
            "test: uniform drain with a positivity clamp"
        }
        fn n_variables(&self) -> usize {
            3
        }
        fn variable_names(&self) -> &[&'static str] {
            &["h", "hu", "hv"]
        }
    }

    impl PhysicsModule<SWESolution2D> for DrainingPhysics {
        fn compute_rhs(&self, state: &SWESolution2D, _time: f64) -> SWESolution2D {
            let seen = state.h_data().iter().cloned().fold(f64::INFINITY, f64::min);
            let mut min_h = self.min_h_seen.lock().unwrap();
            *min_h = min_h.min(seen);
            let mut rhs = SWESolution2D::new(state.n_elements, state.n_nodes);
            rhs.h_data_mut().fill(-1.0);
            rhs
        }
        fn compute_dt(&self, _state: &SWESolution2D, _cfl: f64) -> f64 {
            1.0
        }
        fn post_process(&self, state: &mut SWESolution2D) {
            self.post_process_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for h in state.h_data_mut() {
                *h = h.max(0.0);
            }
        }
        fn mesh(&self) -> &Mesh2D {
            &self.mesh
        }
        fn operators(&self) -> &DGOperators2D {
            &self.ops
        }
        fn geometry(&self) -> &GeometricFactors2D {
            &self.geom
        }
        fn order(&self) -> usize {
            self.ops.order
        }
    }

    #[test]
    fn test_simulation_limits_every_stage() {
        // P0.19 regression: `Simulation` called `post_process` once per step, so
        // the 2nd and 3rd SSP-RK3 stages evaluated the RHS on unlimited states
        // (here h = 0.5 − 1 = −0.5), voiding the Zhang–Shu positivity guarantee.
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        let ops = DGOperators2D::new(1);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let mut state = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        state.h_data_mut().fill(0.5);
        let physics = DrainingPhysics {
            mesh,
            ops,
            geom,
            min_h_seen: std::sync::Mutex::new(f64::INFINITY),
            post_process_calls: std::sync::atomic::AtomicUsize::new(0),
        };

        let sim = Simulation::new(physics, SSPRK3);
        let result = sim.run(&mut state, 0.0, 2.0);

        assert!(result.success);
        assert_eq!(result.n_steps, 2);
        let physics = sim.physics();
        let calls = physics
            .post_process_calls
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            calls,
            3 * result.n_steps,
            "post_process must run after every stage"
        );
        let min_h = *physics.min_h_seen.lock().unwrap();
        assert!(min_h >= 0.0, "an RHS evaluation saw h = {min_h}");
        assert!(state.h_data().iter().all(|&h| h >= 0.0));
    }

    #[test]
    fn test_simulation_config() {
        let config = SimulationConfig::default();
        assert_eq!(config.cfl, 0.5);
        assert!(config.dt_max.is_none());
        assert!(config.dt_min.is_none());
    }

    #[test]
    fn test_simulation_result() {
        let result = SimulationResult::success(10.0, 100, 0.001, 0.01, 1.5);
        assert!(result.success);
        assert!(result.error.is_none());

        let result = SimulationResult::failure(5.0, 50, "Test error".to_string());
        assert!(!result.success);
        assert_eq!(result.error.as_deref(), Some("Test error"));
    }
}
