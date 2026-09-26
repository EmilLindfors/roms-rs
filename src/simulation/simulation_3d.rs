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

    /// The mode-split integrator (e.g. for
    /// [`ModeSplitIntegrator::last_substeps`]).
    pub fn integrator(&self) -> &ModeSplitIntegrator {
        &self.integrator
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
            if let Some(max_steps) = self.config.max_steps
                && n_steps >= max_steps
            {
                return SimulationResult::failure(
                    t,
                    n_steps,
                    format!("Maximum step limit ({}) reached", max_steps),
                );
            }

            // Compute time step (3D CFL)
            let mut dt = self.physics.compute_dt(state, self.config.cfl);

            // Apply dt limits
            if let Some(dt_max) = self.config.dt_max {
                dt = dt.min(dt_max);
            }

            if let Some(dt_min) = self.config.dt_min
                && dt < dt_min
            {
                return SimulationResult::failure(
                    t,
                    n_steps,
                    format!("Time step ({:.2e}) below minimum ({:.2e})", dt, dt_min),
                );
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

            // One barotropic pass and one 3D SSP-RK3 step
            self.integrator.step(
                state,
                &self.physics.sigma,
                &self.physics.bathymetry,
                dt,
                t,
                &self.physics.forcing,
                &self.physics.mixing,
                &self.physics.swe_physics,
                |s, t_loc, out| self.physics.compute_rhs_3d_into(s, t_loc, out),
                // Limit tracers and refresh the density at every stage, so the
                // baroclinic pressure gradient of the next stage is current
                |s| {
                    if !self.physics.apply_tracer_limiters(s).changed() {
                        self.physics.update_density(s);
                    }
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
    use crate::physics::{Hydrostatic3D, LinearEOS, PhysicsBuilder, SWEPhysics2D};
    use crate::simulation::Simulation;
    use crate::solver::state::{SWE_VAR_H, SWE_VAR_HU, SWE_VAR_HV};
    use crate::solver::{DGSolution2D, SWESolution2D};
    use crate::source::CoriolisSource2D;
    use crate::time::{ModeSplitIntegrator, SSPRK3};
    use crate::types::ElementIndex;
    use crate::vertical::{SigmaGrid, UniformStretching};
    use std::sync::Arc;

    const G: f64 = 9.81;

    type Physics = Hydrostatic3D<LinearEOS, ConstantMixing, Reflective2D>;

    /// Closed basin `L × width`, 10 m deep, one element across, with the
    /// fundamental seiche `η = A cos(πx/L)` at rest. Period `T = 2L/√(gH)`.
    struct Seiche {
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
        bathymetry: Arc<Bathymetry2D>,
        length: f64,
        width: f64,
        depth: f64,
        amplitude: f64,
    }

    impl Seiche {
        fn new(nx: usize, order: usize, width: f64) -> Self {
            let length = 1000.0;
            let depth = 10.0;
            let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, length, 0.0, width, nx, 1));
            let ops = Arc::new(DGOperators2D::new(order));
            let geom = Arc::new(GeometricFactors2D::compute(&mesh));
            let bathymetry = Arc::new(Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -depth));
            Self {
                mesh,
                ops,
                geom,
                bathymetry,
                length,
                width,
                depth,
                amplitude: 0.01,
            }
        }

        fn period(&self) -> f64 {
            2.0 * self.length / (G * self.depth).sqrt()
        }

        fn swe(&self) -> SWEPhysics2D<Reflective2D> {
            PhysicsBuilder::swe_2d(
                self.mesh.clone(),
                self.ops.clone(),
                self.geom.clone(),
                ShallowWater2D::new(G),
                Reflective2D::default(),
            )
            .with_bathymetry(self.bathymetry.clone())
            .build()
        }

        fn hydrostatic(&self) -> Physics {
            Hydrostatic3D::new(
                self.mesh.clone(),
                self.ops.clone(),
                self.geom.clone(),
                Arc::new(SigmaGrid::new(3, UniformStretching)),
                self.bathymetry.clone(),
                Arc::new(CoriolisSource2D::f_plane(0.0)),
                // Density independent of T and S: the 3D tracers are not yet
                // constancy-preserving (TODO P4.2), so under the tide T and S
                // drift with η and would feed a spurious baroclinic PGF into G.
                LinearEOS {
                    alpha: 0.0,
                    beta: 0.0,
                    ..LinearEOS::default()
                },
                ConstantMixing::new(1e-4, 1e-4),
                self.swe(),
                Forcing {
                    surface_stress: [0.0, 0.0],
                    bottom_stress: [0.0, 0.0],
                    surface_buoyancy_flux: 0.0,
                },
                G,
                1025.0,
            )
        }

        /// Initial elevation at every node, `[k * n_nodes + i]`.
        fn eta0(&self) -> Vec<f64> {
            let mut eta = Vec::with_capacity(self.mesh.n_elements * self.ops.n_nodes);
            for k in 0..self.mesh.n_elements {
                for i in 0..self.ops.n_nodes {
                    let [x, _] = self.mesh.reference_to_physical(
                        ElementIndex::new(k),
                        self.ops.nodes_r[i],
                        self.ops.nodes_s[i],
                    );
                    eta.push(self.amplitude * (std::f64::consts::PI * x / self.length).cos());
                }
            }
            eta
        }

        fn state_3d(&self, physics: &Physics) -> Solution3D {
            let mut state = Solution3D::new(self.mesh.n_elements, self.ops.n_nodes, 3);
            state.eta.data = self.eta0();
            // Reference T and S, so rho = rho0 and the baroclinic PGF is zero. At
            // T = S = 0 the linear EOS gives 0.975 rho0, and the "baroclinic" PGF
            // returns 2.5 % of the surface gradient through the frozen G.
            let eos = LinearEOS::default();
            state.temp.fill(eos.t0);
            state.salt.fill(eos.s0);
            physics.update_density(&mut state);
            state
        }

        fn state_2d(&self) -> SWESolution2D {
            let mut q = SWESolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
            for (h, eta) in q.data[SWE_VAR_H].iter_mut().zip(self.eta0()) {
                *h = self.depth + eta;
            }
            q
        }

        /// `∫ ½gη² + ½(hu² + hv²)/h dA`.
        fn energy(&self, eta: &[f64], hu: &[f64], hv: &[f64]) -> f64 {
            let mut density = DGSolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
            for (idx, e) in density.data.iter_mut().enumerate() {
                let h = self.depth + eta[idx];
                *e = 0.5 * G * eta[idx] * eta[idx] + 0.5 * (hu[idx].powi(2) + hv[idx].powi(2)) / h;
            }
            density.integrate(&self.ops, &self.geom)
        }

        fn energy_3d(&self, s: &Solution3D) -> f64 {
            let h = |i: usize| self.depth + s.eta.data[i];
            let n = s.eta.data.len();
            let hu: Vec<f64> = (0..n).map(|i| h(i) * s.ubar.data[i]).collect();
            let hv: Vec<f64> = (0..n).map(|i| h(i) * s.vbar.data[i]).collect();
            self.energy(&s.eta.data, &hu, &hv)
        }

        fn energy_2d(&self, q: &SWESolution2D) -> f64 {
            let eta: Vec<f64> = q.data[SWE_VAR_H].iter().map(|h| h - self.depth).collect();
            self.energy(&eta, &q.data[SWE_VAR_HU], &q.data[SWE_VAR_HV])
        }
    }

    #[test]
    fn seiche_period_matches_analytic() {
        // Regression (TODO P0.6): a barotropic seiche in a closed basin must
        // oscillate at the analytic period T = 2L/√(gH). If the mode-split G-term
        // double-counted the barotropic pressure gradient (−g∇η present in both
        // the 2D ½gh² flux and the depth-averaged 3D PGF), the effective gravity
        // would roughly double and the period would collapse by ~1/√2 (or the run
        // would go unstable). With the baroclinic-only PGF the period is correct.
        let seiche = Seiche::new(10, 1, 100.0);
        let t_analytic = seiche.period();
        let physics = seiche.hydrostatic();
        let mut state = seiche.state_3d(&physics);
        // Node 0 of element 0 sits on the left wall (x = 0).
        let idx = 0;

        // The 3D dt estimate assumes a 2 m/s internal wave (TODO P4.5); dt_max
        // sets the step here.
        let mut sim = Simulation3D::new(physics, ModeSplitIntegrator::new())
            .with_cfl(10.0)
            .with_dt_max(4.0) // resolve the ~200 s period with many samples
            .with_max_steps(200);

        let mut samples: Vec<(f64, f64)> = Vec::new();
        // Run just over half a period so we capture the downward zero crossing (T/4).
        let result = sim.run_with_callback(&mut state, 0.0, 0.6 * t_analytic, |s, t| {
            samples.push((t, s.eta.data[idx]));
        });
        assert!(result.success, "seiche run failed: {:?}", result.error);

        // The left-wall elevation follows A cos(2π t / T); the first downward zero
        // crossing is at t = T/4. Find it and linearly interpolate the time.
        let t_cross = samples
            .windows(2)
            .find(|w| w[0].1 >= 0.0 && w[1].1 < 0.0)
            .map(|w| w[0].0 + (w[1].0 - w[0].0) * w[0].1 / (w[0].1 - w[1].1))
            .expect("no downward zero crossing observed within 0.6 T");
        let period_measured = 4.0 * t_cross;
        let rel_err = (period_measured - t_analytic).abs() / t_analytic;
        assert!(
            rel_err < 0.02,
            "seiche period {period_measured:.1}s vs analytic {t_analytic:.1}s (rel err {rel_err:.3}); \
             barotropic PGF double-counted?"
        );
    }

    /// TODO P4.1 gate: the mode splitter must not damp resolved barotropic
    /// motion. Before the one-pass rewrite, SSP-RK3 around a Forward Euler
    /// subcycle lost about 23 % of a seiche's amplitude per period.
    ///
    /// 10 periods at 50 baroclinic steps per period (n_bt = 23), against the
    /// pure 2D model on the same mesh, which isolates the damping of the
    /// splitting from the (identical) DG dissipation. The filter predicts
    /// ≈ 0.035 % of the amplitude per period here (see `BarotropicFilter`),
    /// and that is what is measured. Total volume is conserved to round-off
    /// (≈ 1e-12 relative).
    #[test]
    fn seiche_amplitude_is_kept_by_the_mode_split() {
        // A narrow basin: the cross-basin CFL makes the barotropic step small,
        // so a few elements give a realistic n_bt cheaply.
        let seiche = Seiche::new(10, 2, 10.0);
        let period = seiche.period();
        let periods = 10.0;
        let t_end = periods * period;
        let dt = period / 50.0;

        // Pure 2D reference
        let mut q = seiche.state_2d();
        let e0_2d = seiche.energy_2d(&q);
        let reference = Simulation::new(seiche.swe(), SSPRK3).with_cfl(0.5);
        let result = reference.run(&mut q, 0.0, t_end);
        assert!(result.success, "2D reference failed: {:?}", result.error);
        let decay_2d = (seiche.energy_2d(&q) / e0_2d).sqrt();

        // Mode split
        let physics = seiche.hydrostatic();
        let mut state = seiche.state_3d(&physics);
        let e0 = seiche.energy_3d(&state);
        let volume0 = state.eta.integrate(&seiche.ops, &seiche.geom);
        let mut sim = Simulation3D::new(physics, ModeSplitIntegrator::new())
            .with_cfl(10.0)
            .with_dt_max(dt);
        let result = sim.run(&mut state, 0.0, t_end);
        assert!(result.success, "mode-split run failed: {:?}", result.error);

        // Substeps of a full-length step (the last step of the run is shortened)
        let mut probe = state.clone();
        sim.run(&mut probe, t_end, t_end + dt);
        let n_bt = sim.integrator().last_substeps();
        assert!(
            n_bt >= 20,
            "test regime: expected ≥ 20 substeps, got {n_bt}"
        );

        let decay_split = (seiche.energy_3d(&state) / e0).sqrt();
        let loss_per_period = 1.0 - (decay_split / decay_2d).powf(1.0 / periods);
        assert!(
            loss_per_period.abs() < 5e-4,
            "mode split loses {:.4} % of the amplitude per period more than 2D \
             (2D keeps {decay_2d:.6}, split {decay_split:.6} after {periods} periods, n_bt = {n_bt})",
            100.0 * loss_per_period
        );

        let volume = state.eta.integrate(&seiche.ops, &seiche.geom);
        let area = seiche.length * seiche.width;
        assert!(
            (volume - volume0).abs() / (seiche.depth * area) < 1e-11,
            "volume drift {:.3e} m³",
            volume - volume0
        );
    }
}
