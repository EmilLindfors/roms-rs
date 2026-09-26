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
            self.integrator.step(state, &self.physics, dt, t);

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
    use crate::mesh::data::Bathymetry2D;
    use crate::mesh::{Mesh2D, Mesh2DBuilder};
    use crate::operators::{DGOperators2D, GeometricFactors2D};
    use crate::physics::vertical_mixing::{ConstantMixing, Forcing};
    use crate::physics::{Hydrostatic3D, LinearEOS, PhysicsBuilder, SWEPhysics2D};
    use crate::simulation::Simulation;
    use crate::solver::state::{SWE_VAR_H, SWE_VAR_HU, SWE_VAR_HV};
    use crate::solver::{DGSolution2D, SWEFormulation2D, SWESolution2D};
    use crate::source::CoriolisSource2D;
    use crate::time::{ModeSplitIntegrator, SSPRK3};
    use crate::types::ElementIndex;
    use crate::vertical::{SigmaGrid, UniformStretching};
    use std::sync::Arc;

    const G: f64 = 9.81;
    const RHO0: f64 = 1025.0;

    type Physics = Hydrostatic3D<LinearEOS, ConstantMixing, Reflective2D>;

    fn no_stress() -> Forcing {
        Forcing {
            surface_stress: [0.0, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_buoyancy_flux: 0.0,
        }
    }

    /// Three uniform σ-levels, Coriolis `f`, constant viscosity.
    #[allow(clippy::too_many_arguments)]
    fn hydrostatic(
        mesh: &Arc<Mesh2D>,
        ops: &Arc<DGOperators2D>,
        geom: &Arc<GeometricFactors2D>,
        bathymetry: &Arc<Bathymetry2D>,
        swe: SWEPhysics2D<Reflective2D>,
        forcing: Forcing,
        viscosity: f64,
        f: f64,
    ) -> Physics {
        Hydrostatic3D::new(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            Arc::new(SigmaGrid::new(3, UniformStretching)),
            bathymetry.clone(),
            Arc::new(CoriolisSource2D::f_plane(f)),
            // Density independent of T and S: the 3D tracers are not yet
            // constancy-preserving (TODO P4.2), so under the tide T and S
            // drift with η and would feed a spurious baroclinic PGF into G.
            LinearEOS {
                alpha: 0.0,
                beta: 0.0,
                ..LinearEOS::default()
            },
            ConstantMixing::new(viscosity, viscosity),
            swe,
            forcing,
            G,
            RHO0,
        )
    }

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
            let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
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
            self.swe_with(SWEFormulation2D::Standard)
        }

        fn swe_with(&self, formulation: SWEFormulation2D) -> SWEPhysics2D<Reflective2D> {
            PhysicsBuilder::swe_2d(
                self.mesh.clone(),
                self.ops.clone(),
                self.geom.clone(),
                ShallowWater2D::new(G),
                Reflective2D::default(),
            )
            .with_bathymetry(self.bathymetry.clone())
            .with_formulation(formulation)
            .build()
        }

        fn hydrostatic(&self) -> Physics {
            self.hydrostatic_forced(no_stress(), 1e-4)
        }

        fn hydrostatic_forced(&self, forcing: Forcing, viscosity: f64) -> Physics {
            hydrostatic(
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.bathymetry,
                self.swe(),
                forcing,
                viscosity,
                0.0,
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

    /// TODO P4.1 gate: without vertical shear or stratification the mode-split
    /// model is the 2D model. A nonlinear seiche (η/H = 0.05) must match the
    /// pure 2D run over two periods: G must not count the mean-flow advection
    /// a second time.
    #[test]
    fn unsheared_flow_matches_the_2d_model() {
        let mut seiche = Seiche::new(10, 2, 10.0);
        seiche.amplitude = 0.5;
        let period = seiche.period();
        let t_end = 2.0 * period;
        let dt = period / 50.0;

        let mut q = seiche.state_2d();
        let result = Simulation::new(seiche.swe(), SSPRK3)
            .with_cfl(0.5)
            .run(&mut q, 0.0, t_end);
        assert!(result.success, "2D reference failed: {:?}", result.error);

        let physics = seiche.hydrostatic();
        let mut state = seiche.state_3d(&physics);
        let mut sim = Simulation3D::new(physics, ModeSplitIntegrator::new())
            .with_cfl(10.0)
            .with_dt_max(dt);
        let result = sim.run(&mut state, 0.0, t_end);
        assert!(result.success, "mode-split run failed: {:?}", result.error);

        let max_diff = q.data[SWE_VAR_H]
            .iter()
            .zip(&state.eta.data)
            .map(|(h, eta)| (h - seiche.depth - eta).abs())
            .fold(0.0_f64, f64::max);
        // 1.8e-3 of the amplitude at 50 steps per period, falling with dt (the
        // filter acting on the harmonics of the steepening wave); PR 1's G,
        // which counted the mean-flow advection twice: 0.14.
        assert!(
            max_diff < 4e-3 * seiche.amplitude,
            "η differs from the 2D model by {max_diff:.3e} m ({:.2e} of the amplitude)",
            max_diff / seiche.amplitude
        );
    }

    /// TODO P4.1 gate: the wind stress reaches the depth mean. In a closed
    /// basin the steady surface slope is `∂η/∂x = τ/(ρ₀ g D)`. Starting from
    /// rest, the basin seiches about the setup, so η is averaged over whole
    /// seiche periods (every basin mode's period divides the fundamental one).
    /// Before, the 3D stress never reached the barotropic mode: no setup.
    #[test]
    fn wind_setup_balances_the_surface_stress() {
        let mut seiche = Seiche::new(10, 2, 10.0);
        seiche.amplitude = 0.0;
        let period = seiche.period();
        let tau = 0.02;
        let forcing = Forcing {
            surface_stress: [tau, 0.0],
            ..no_stress()
        };
        // Viscous enough that the wind-driven shear, and with it the momentum
        // dispersion −∇·⟨u′u′⟩ at the end walls, stays small
        let physics = seiche.hydrostatic_forced(forcing, 0.05);
        let mut state = seiche.state_3d(&physics);

        let dt = period / 50.0;
        let (t_avg0, t_avg1) = (2.0 * period, 6.0 * period);
        let mut eta_sum = vec![0.0; state.eta.data.len()];
        let mut weight = 0.0;
        let mut sim = Simulation3D::new(physics, ModeSplitIntegrator::new())
            .with_cfl(10.0)
            .with_dt_max(dt);
        let result = sim.run_with_callback(&mut state, 0.0, t_avg1, |s, t| {
            if t > t_avg0 + 0.5 * dt {
                for (sum, eta) in eta_sum.iter_mut().zip(&s.eta.data) {
                    *sum += eta;
                }
                weight += 1.0;
            }
        });
        assert!(result.success, "wind run failed: {:?}", result.error);

        // Least-squares slope of the mean η against x
        let xs: Vec<f64> = (0..seiche.mesh.n_elements)
            .flat_map(|k| {
                let mesh = &seiche.mesh;
                let ops = &seiche.ops;
                (0..ops.n_nodes).map(move |i| {
                    mesh.reference_to_physical(ElementIndex::new(k), ops.nodes_r[i], ops.nodes_s[i])
                        [0]
                })
            })
            .collect();
        let n = xs.len() as f64;
        let x_mean = xs.iter().sum::<f64>() / n;
        let eta_mean: Vec<f64> = eta_sum.iter().map(|e| e / weight).collect();
        let e_mean = eta_mean.iter().sum::<f64>() / n;
        let slope = xs
            .iter()
            .zip(&eta_mean)
            .map(|(x, e)| (x - x_mean) * (e - e_mean))
            .sum::<f64>()
            / xs.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>();

        let expected = tau / (RHO0 * G * seiche.depth);
        // Measured 1.0001 × the expected slope; 0 without the stress in G
        assert!(
            (slope - expected).abs() < 5e-3 * expected,
            "wind setup slope {slope:.4e} vs τ/(ρ₀gD) = {expected:.4e}"
        );
    }

    /// TODO P4.1 gate: wind over a rotating, horizontally uniform ocean. The
    /// depth-integrated transport obeys `dU/dt = fV + τ/ρ₀`, `dV/dt = −fU`:
    /// from rest `U = (τ/ρ₀f) sin ft`, `V = −(τ/ρ₀f)(1 − cos ft)`, whose
    /// average is the Ekman transport `τ/(ρ₀f)` to the right of the wind. The
    /// 2D module carries the Coriolis force of the mean flow and G the stress;
    /// before, the stress never reached the depth mean.
    #[test]
    fn wind_drives_the_ekman_inertial_transport() {
        let (f, tau, depth) = (1.2e-4, 0.1, 50.0);
        let mesh = Arc::new(
            Mesh2DBuilder::new(0.0, 40e3, 0.0, 40e3)
                .with_resolution(4, 4)
                .fully_periodic()
                .build(),
        );
        let ops = Arc::new(DGOperators2D::new(1));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -depth));
        let swe = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::default(),
        )
        .with_bathymetry(bathymetry.clone())
        .with_source(CoriolisSource2D::f_plane(f))
        .build();
        let forcing = Forcing {
            surface_stress: [tau, 0.0],
            ..no_stress()
        };
        let physics = hydrostatic(&mesh, &ops, &geom, &bathymetry, swe, forcing, 0.01, f);
        let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, 3);
        physics.update_density(&mut state);

        let inertial = 2.0 * std::f64::consts::PI / f;
        let scale = tau / (RHO0 * f);
        let mut max_err: f64 = 0.0;
        let mut sim = Simulation3D::new(physics, ModeSplitIntegrator::new().with_min_substeps(20))
            .with_cfl(10.0)
            .with_dt_max(600.0);
        let result = sim.run_with_callback(&mut state, 0.0, 1.5 * inertial, |s, t| {
            let (u, v) = ((f * t).sin(), -(1.0 - (f * t).cos()));
            for idx in 0..s.eta.data.len() {
                let d = depth + s.eta.data[idx];
                max_err = max_err
                    .max((d * s.ubar.data[idx] / scale - u).abs())
                    .max((d * s.vbar.data[idx] / scale - v).abs());
            }
        });
        assert!(result.success, "Ekman run failed: {:?}", result.error);
        // Measured 3.9e-4; 2.0 without the stress in G
        assert!(
            max_err < 1e-3,
            "transport off the inertial-Ekman solution by {max_err:.2e} of τ/(ρ₀f)"
        );
    }

    /// TODO P4.1 gate (PR 3): the barotropic transport of each step (DU_avg2)
    /// is exactly the transport that moved the free surface,
    /// `η̄ − ηⁿ = −Δt ∇·DU_avg2` at every node, for the collocated and the
    /// entropy-stable split form (nonlinear seiche, walls).
    #[test]
    fn barotropic_transport_moves_the_free_surface() {
        for formulation in [SWEFormulation2D::Standard, SWEFormulation2D::EntropyStable] {
            let mut seiche = Seiche::new(10, 2, 10.0);
            seiche.amplitude = 0.5;
            let physics = hydrostatic(
                &seiche.mesh,
                &seiche.ops,
                &seiche.geom,
                &seiche.bathymetry,
                seiche.swe_with(formulation),
                no_stress(),
                1e-4,
                0.0,
            );
            let mut state = seiche.state_3d(&physics);
            let mut integrator = ModeSplitIntegrator::new();
            let dt = seiche.period() / 50.0;
            let mut div = DGSolution2D::new(seiche.mesh.n_elements, seiche.ops.n_nodes);

            for n in 0..10 {
                let eta0 = state.eta.data.clone();
                integrator.step(&mut state, &physics, dt, n as f64 * dt);
                let transport = integrator.barotropic_transport().expect("after a step");
                transport.divergence_into(&seiche.ops, &seiche.geom, &mut div);

                let (mut change, mut residual) = (0.0_f64, 0.0_f64);
                for ((eta, eta0), d) in state.eta.data.iter().zip(&eta0).zip(&div.data) {
                    change = change.max((eta - eta0).abs());
                    residual = residual.max((eta - eta0 + dt * d).abs());
                }
                assert!(
                    residual < 1e-11 * change,
                    "{formulation:?}, step {n}: η̄ − ηⁿ + Δt∇·DU_avg2 = {residual:.2e} \
                     (η change {change:.2e})"
                );
            }
        }
    }
}
