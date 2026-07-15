//! 3D Simulation Runner.
//!
//! Specialized runner for 3D hydrostatic simulations using mode splitting.

use crate::boundary::SWEBoundaryCondition2D;
use crate::physics::eos::EquationOfState;
use crate::physics::hydrostatic_3d::Hydrostatic3D;
use crate::physics::traits::PhysicsModule; // For 2D trait bounds
use crate::physics::vertical_mixing::VerticalMixing;
use crate::simulation::{SimulationConfig, SimulationResult};
use crate::solver::state::Solution3D;
use crate::time::mode_split::ModeSplitIntegrator;

/// High-level simulation runner for 3D models.
pub struct Simulation3D<EOS, MIX, BC>
where
    EOS: EquationOfState,
    MIX: VerticalMixing,
    BC: SWEBoundaryCondition2D,
{
    physics: Hydrostatic3D<EOS, MIX, BC>,
    integrator: ModeSplitIntegrator,
    config: SimulationConfig,
}

impl<EOS, MIX, BC> Simulation3D<EOS, MIX, BC>
where
    EOS: EquationOfState,
    MIX: VerticalMixing,
    crate::physics::SWEPhysics2D<BC>: PhysicsModule<crate::solver::SWESolution2D>,
    BC: Clone + Send + Sync + SWEBoundaryCondition2D,
{
    /// Create a new 3D simulation.
    pub fn new(physics: Hydrostatic3D<EOS, MIX, BC>, integrator: ModeSplitIntegrator) -> Self {
        Self {
            physics,
            integrator,
            config: SimulationConfig::default(),
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

    /// Set the minimum time step.
    pub fn with_dt_min(mut self, dt_min: f64) -> Self {
        self.config.dt_min = Some(dt_min);
        self
    }

    /// Set the callback interval.
    pub fn with_callback_interval(mut self, interval: f64) -> Self {
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

    /// Run the simulation.
    pub fn run(&mut self, state: &mut Solution3D, t_start: f64, t_end: f64) -> SimulationResult {
        self.run_with_callback(state, t_start, t_end, |_, _| {})
    }

    /// Run with callback.
    pub fn run_with_callback<F>(
        &mut self,
        state: &mut Solution3D,
        t_start: f64,
        t_end: f64,
        mut callback: F,
    ) -> SimulationResult
    where
        F: FnMut(&Solution3D, f64),
    {
        let start_wall = std::time::Instant::now();
        // ... implementation
        let mut t = t_start;
        let mut n_steps = 0;
        let mut dt_min_used = f64::INFINITY;
        let mut dt_max_used: f64 = 0.0;
        let mut last_callback_time = t_start;

        // Initial callback
        callback(state, t);

        if self.config.verbose {
            println!("Starting 3D simulation...");
            println!("  t_start = {:.4}, t_end = {:.4}", t_start, t_end);
        }

        while t < t_end {
            // Check step limit
            if let Some(max_steps) = self.config.max_steps {
                if n_steps >= max_steps {
                    return SimulationResult::failure(
                        t,
                        n_steps,
                        format!("Maximum step limit ({}) reached", max_steps),
                    );
                }
            }

            // Compute time step (3D CFL)
            let mut dt = self.physics.compute_dt(state, self.config.cfl);

            // Apply dt limits
            if let Some(dt_max) = self.config.dt_max {
                dt = dt.min(dt_max);
            }

            if let Some(dt_min) = self.config.dt_min {
                if dt < dt_min {
                    return SimulationResult::failure(
                        t,
                        n_steps,
                        format!("Time step ({:.2e}) below minimum ({:.2e})", dt, dt_min),
                    );
                }
            }

            // Don't overshoot end time
            if t + dt > t_end {
                dt = t_end - t;
            }

            // Track statistics
            dt_min_used = dt_min_used.min(dt);
            dt_max_used = dt_max_used.max(dt);

            // Update density based on current T, S before computing forces
            self.physics.update_density(state);

            // Step using Mode Split Integrator
            self.integrator.step_with_stage_hook(
                state,
                &self.physics.sigma,
                &self.physics.bathymetry,
                dt,
                t,
                &self.physics.forcing,
                &self.physics.mixing,
                // 2D RHS Closure
                |eta, ubar, vbar, t_loc| self.physics.compute_rhs_2d(eta, ubar, vbar, t_loc),
                // 3D RHS Closure
                |s, t_loc| self.physics.compute_rhs_3d(s, t_loc),
                // Stage limiter hook
                |s| {
                    self.physics.apply_tracer_limiters(s);
                },
            );

            t += dt;
            n_steps += 1;

            // Post-process
            self.physics.post_process(state);

            // Callback
            if let Some(interval) = self.config.callback_interval {
                if t - last_callback_time >= interval {
                    callback(state, t);
                    last_callback_time = t;
                }
            } else {
                callback(state, t);
                last_callback_time = t;
            }

            // Progress
            if self.config.verbose && n_steps % 100 == 0 {
                println!("  Step {}: t = {:.4}, dt = {:.2e}", n_steps, t, dt);
            }
        }

        let wall_time = start_wall.elapsed().as_secs_f64();

        if self.config.verbose {
            println!("Simulation complete.");
            println!("  Wall time: {:.2}s", wall_time);
        }

        SimulationResult::success(t, n_steps, dt_min_used, dt_max_used, wall_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::Reflective2D;
    use crate::equations::ShallowWater2D;
    use crate::mesh::Mesh2D;
    use crate::mesh::data::Bathymetry2D;
    use crate::operators::{DGOperators2D, GeometricFactors2D};
    use crate::physics::vertical_mixing::{ConstantMixing, Forcing};
    use crate::physics::{Hydrostatic3D, LinearEOS, PhysicsBuilder};
    use crate::solver::DGSolution2D;
    use crate::source::CoriolisSource2D;
    use crate::time::ModeSplitIntegrator;
    use crate::types::ElementIndex;
    use crate::vertical::{SigmaGrid, UniformStretching};
    use std::sync::Arc;

    #[test]
    fn seiche_period_matches_analytic() {
        // Regression (TODO P0.6): a barotropic seiche in a closed basin must
        // oscillate at the analytic period T = 2L/√(gH). If the mode-split G-term
        // double-counted the barotropic pressure gradient (−g∇η present in both
        // the 2D ½gh² flux and the depth-averaged 3D PGF), the effective gravity
        // would roughly double and the period would collapse by ~1/√2 (or the run
        // would go unstable). With the baroclinic-only PGF the period is correct.
        let g = 9.81_f64;
        let length = 1000.0_f64;
        let depth = 10.0_f64; // bed at -10 m, eta ≈ 0
        let t_analytic = 2.0 * length / (g * depth).sqrt();

        // Long, thin closed basin.
        let nx = 10;
        let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, length, 0.0, 100.0, nx, 1));
        let ops = Arc::new(DGOperators2D::new(1));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh));
        let sigma = Arc::new(SigmaGrid::new(3, UniformStretching));

        let bathymetry = Arc::new(Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -depth));
        let coriolis = Arc::new(CoriolisSource2D::f_plane(0.0));
        let mixing = ConstantMixing::new(1e-4, 1e-4);
        let forcing = Forcing {
            surface_stress: [0.0, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_buoyancy_flux: 0.0,
        };

        let swe_physics = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(g),
            Reflective2D::default(),
        )
        .with_bathymetry(bathymetry.clone())
        .build();

        let physics = Hydrostatic3D::new(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            sigma.clone(),
            bathymetry.clone(),
            coriolis,
            LinearEOS::default(),
            mixing,
            swe_physics,
            forcing,
            g,
            1025.0,
        );

        // Fundamental seiche mode: eta = A cos(pi x / L), u = 0.
        let amp = 0.01;
        let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, 3);
        // Track the node nearest the left wall (x = 0) to sample the time series.
        let mut left_node = (0usize, 0usize);
        let mut left_x = f64::INFINITY;
        for k in 0..mesh.n_elements {
            let el = ElementIndex::new(k);
            for i in 0..ops.n_nodes {
                let [x, _y] = mesh.reference_to_physical(el, ops.nodes_r[i], ops.nodes_s[i]);
                state.eta.data[k * ops.n_nodes + i] =
                    amp * (std::f64::consts::PI * x / length).cos();
                if x < left_x {
                    left_x = x;
                    left_node = (k, i);
                }
            }
        }
        physics.update_density(&mut state);

        let integrator =
            ModeSplitIntegrator::new(20, &DGSolution2D::new(mesh.n_elements, ops.n_nodes));
        let mut sim = Simulation3D::new(physics, integrator)
            .with_cfl(0.4)
            .with_dt_max(4.0) // resolve the ~200 s period with many samples
            .with_max_steps(200);

        // Sample eta at the left wall each step.
        let mut samples: Vec<(f64, f64)> = Vec::new();
        let (lk, li) = left_node;
        let idx = lk * ops.n_nodes + li;
        // Run just over half a period so we capture the downward zero crossing (T/4).
        let result = sim.run_with_callback(&mut state, 0.0, 0.6 * t_analytic, |s, t| {
            samples.push((t, s.eta.data[idx]));
        });
        assert!(result.success, "seiche run failed: {:?}", result.error);

        // The left-wall elevation follows A cos(2π t / T); the first downward zero
        // crossing is at t = T/4. Find it and linearly interpolate the time.
        let mut t_cross = None;
        for w in samples.windows(2) {
            let (t0, e0) = w[0];
            let (t1, e1) = w[1];
            if e0 >= 0.0 && e1 < 0.0 {
                t_cross = Some(t0 + (t1 - t0) * e0 / (e0 - e1));
                break;
            }
        }
        let t_cross = t_cross.expect("no downward zero crossing observed within 0.6 T");
        let period_measured = 4.0 * t_cross;
        let rel_err = (period_measured - t_analytic).abs() / t_analytic;
        assert!(
            rel_err < 0.15,
            "seiche period {period_measured:.1}s vs analytic {t_analytic:.1}s (rel err {rel_err:.2}); \
             barotropic PGF double-counted?"
        );
    }
}
