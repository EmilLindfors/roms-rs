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

            // Don't overshoot the end time or leave a sliver of it (see
            // `Simulation::run_with_callback`)
            let landed = t + dt * (1.0 + super::runner::LANDING_SLACK) >= t_end;
            if landed {
                dt = t_end - t;
            }

            // Track statistics
            dt_min_used = dt_min_used.min(dt);
            dt_max_used = dt_max_used.max(dt);

            // Update density based on current T, S before computing forces
            self.physics.update_density(state);

            // One barotropic pass and one 3D SSP-RK3 step
            self.integrator.step(state, &self.physics, dt, t);

            t = if landed { t_end } else { t + dt };
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
    use crate::boundary::{
        BCContext2D, BoundaryState, CharacteristicOBC, ExternalState, HarmonicTide, Reflective2D,
        SWEBoundaryCondition2D,
    };
    use crate::equations::ShallowWater2D;
    use crate::mesh::data::Bathymetry2D;
    use crate::mesh::data::BoundaryTag;
    use crate::mesh::{Mesh2D, Mesh2DBuilder};
    use crate::operators::{DGOperators2D, GeometricFactors2D};
    use crate::physics::vertical_mixing::{ConstantMixing, Forcing};
    use crate::physics::{BottomDrag3D, Hydrostatic3D, LinearEOS, PhysicsBuilder, SWEPhysics2D};
    use crate::simulation::Simulation;
    use crate::solver::state::{SWE_VAR_H, SWE_VAR_HU, SWE_VAR_HV};
    use crate::solver::{DGSolution2D, SWEFormulation2D, SWESolution2D, SWEState2D};
    use crate::source::CoriolisSource2D;
    use crate::time::{ModeSplitIntegrator, SSPRK3};
    use crate::types::ElementIndex;
    use crate::vertical::{SigmaGrid, UniformStretching};
    use std::sync::Arc;

    const G: f64 = 9.81;
    const RHO0: f64 = 1025.0;

    type Physics = Hydrostatic3D<LinearEOS, ConstantMixing, Reflective2D>;

    /// The largest value, or NaN if any is NaN (`f64::max` drops NaN, which
    /// would let a blown-up run pass).
    fn max_or_nan(values: impl IntoIterator<Item = f64>) -> f64 {
        values
            .into_iter()
            .fold(0.0, |m, x| if x.is_nan() || x > m { x } else { m })
    }

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
            // T/S-dependent: uniform tracers stay uniform under the tide
            // (P4.2), so they feed no baroclinic PGF into G. Before, T and S
            // drifted with η and damped a seiche by 0.6 % per period.
            LinearEOS::default(),
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

    /// A closed channel over a bed rising from 12 m to 4 m, a 0.5 m seiche
    /// (so the layers thin and thicken by up to an eighth), sheared columns,
    /// and the T/S-dependent linear EOS.
    struct SlopingTide {
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
        bathymetry: Arc<Bathymetry2D>,
        sigma: SigmaGrid,
        length: f64,
        /// The coordinate the tide runs along (0: x, 1: y).
        axis: usize,
    }

    impl SlopingTide {
        fn new() -> Self {
            let length = 1000.0;
            let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, length, 0.0, 100.0, 8, 2));
            let ops = Arc::new(DGOperators2D::new(2));
            let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
            let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
                -12.0 + 8.0 * x / length + 0.5 * (std::f64::consts::PI * y / 100.0).cos()
            }));
            Self {
                mesh,
                ops,
                geom,
                bathymetry,
                sigma: SigmaGrid::new(3, UniformStretching),
                length,
                axis: 0,
            }
        }

        /// A 200 m × 1 km channel, periodic in x, with the tide across it
        /// (along y) over a bed sloping in y alone: every field is uniform in
        /// x.
        fn across_channel() -> Self {
            let length = 1000.0;
            let mesh = Arc::new(Mesh2D::channel_periodic_x(0.0, 200.0, 0.0, length, 2, 8));
            let ops = Arc::new(DGOperators2D::new(2));
            let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
            let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |_, y| {
                -12.0 + 8.0 * y / length
            }));
            Self {
                mesh,
                ops,
                geom,
                bathymetry,
                sigma: SigmaGrid::new(3, UniformStretching),
                length,
                axis: 1,
            }
        }

        fn physics(&self) -> Physics {
            self.physics_with(ConstantMixing::new(1e-3, 1e-4))
        }

        fn physics_with(&self, mixing: ConstantMixing) -> Physics {
            let swe = PhysicsBuilder::swe_2d(
                self.mesh.clone(),
                self.ops.clone(),
                self.geom.clone(),
                ShallowWater2D::new(G),
                Reflective2D::default(),
            )
            .with_bathymetry(self.bathymetry.clone())
            .with_formulation(SWEFormulation2D::EntropyStable)
            .build();
            Hydrostatic3D::new(
                self.mesh.clone(),
                self.ops.clone(),
                self.geom.clone(),
                Arc::new(self.sigma.clone()),
                self.bathymetry.clone(),
                Arc::new(CoriolisSource2D::f_plane(0.0)),
                LinearEOS::default(),
                mixing,
                swe,
                no_stress(),
                G,
                RHO0,
            )
        }

        /// `η = 0.5 cos(πx/L)` (x along the tide), zero-mean shear
        /// `u = 0.1(σ + ½)` in x, and the tracers from `tracer(x, σ)`.
        fn state(&self, physics: &Physics, tracer: impl Fn(f64, f64) -> (f64, f64)) -> Solution3D {
            let (nn, nl) = (self.ops.n_nodes, self.sigma.n_levels());
            let mut state = Solution3D::new(self.mesh.n_elements, nn, nl);
            for k in 0..self.mesh.n_elements {
                for i in 0..nn {
                    let x = self.mesh.reference_to_physical(
                        ElementIndex::new(k),
                        self.ops.nodes_r[i],
                        self.ops.nodes_s[i],
                    )[self.axis];
                    let idx = k * nn + i;
                    state.eta.data[idx] = 0.5 * (std::f64::consts::PI * x / self.length).cos();
                    for (l, &s) in self.sigma.sigma_rho().iter().enumerate() {
                        state.u[idx * nl + l] = 0.1 * (s + 0.5);
                        (state.temp[idx * nl + l], state.salt[idx * nl + l]) = tracer(x, s);
                    }
                }
            }
            physics.update_density(&mut state);
            state
        }

        /// `∫ Σ_l H_z C dA` of a tracer.
        fn inventory(&self, state: &Solution3D, tracer: &[f64]) -> f64 {
            let (nn, nl) = (self.ops.n_nodes, self.sigma.n_levels());
            let mut column = DGSolution2D::new(self.mesh.n_elements, nn);
            for (idx, c) in column.data.iter_mut().enumerate() {
                let depth = state.eta.data[idx] - self.bathymetry.data[idx];
                *c = (0..nl)
                    .map(|l| depth * self.sigma.d_sigma()[l] * tracer[idx * nl + l])
                    .sum();
            }
            column.integrate(&self.ops, &self.geom)
        }

        /// One seiche period at 40 steps per period, calling `check` after
        /// every step.
        fn run(
            &self,
            physics: &Physics,
            state: &mut Solution3D,
            mut check: impl FnMut(&Solution3D, &Physics),
        ) {
            let period = 2.0 * self.length / (G * 8.0).sqrt();
            let dt = period / 40.0;
            let mut integrator = ModeSplitIntegrator::new();
            for n in 0..40 {
                physics.update_density(state);
                integrator.step(state, physics, dt, n as f64 * dt);
                physics.post_process(state);
                assert!(
                    max_or_nan(state.u.iter().chain(&state.temp).copied()).is_finite(),
                    "step {n}: the run blew up"
                );
                check(state, physics);
            }
        }
    }

    /// TODO P4.2/P4.6 gate: uniform T and S stay uniform under a large tide
    /// over a sloping bed. Before, the tracers were stepped as
    /// concentrations with an inventory tendency, and Ω closed at the surface
    /// with the 3D velocities' own ∂η/∂t instead of the barotropic pass's: a
    /// 1 m tide over 20 m pumped salinity by about ±1.7 psu, and through a
    /// T/S-dependent EOS fed a spurious baroclinic pressure gradient into G.
    #[test]
    fn uniform_tracers_stay_uniform_under_a_tide_over_a_sloping_bed() {
        let case = SlopingTide::new();
        let physics = case.physics();
        let eos = LinearEOS::default();
        let mut state = case.state(&physics, |_, _| (eos.t0 + 2.3, eos.s0 - 0.9));
        let (mut drift, mut residual, mut omega_scale) = (0.0_f64, 0.0_f64, 0.0_f64);
        case.run(&physics, &mut state, |s, p| {
            for t in &s.temp {
                drift = drift.max((t - (eos.t0 + 2.3)).abs());
            }
            for salt in &s.salt {
                drift = drift.max((salt - (eos.s0 - 0.9)).abs());
            }
            residual = residual.max(p.last_surface_residual());
            omega_scale = omega_scale.max(s.w.iter().fold(0.0, |m, w| m.max(w.abs())));
        });
        // Measured 9.3e-13 after one period; before, 12.8 (psu and °C)
        assert!(
            drift < 1e-11 * eos.s0,
            "uniform tracers drifted by {drift:.3e} under the tide"
        );
        // Measured 3.0e-15 against Ω of 6.8e-3 m/s
        // The barotropic pass's nodal identity closes Ω at the surface
        assert!(omega_scale > 1e-5, "test regime: Ω {omega_scale:.2e}");
        assert!(
            residual < 1e-10 * omega_scale,
            "Ω surface residual {residual:.2e} (Ω scale {omega_scale:.2e})"
        );
    }

    /// TODO P4.2/P4.6 gate: the tracer inventories `∫ Σ H_z C dA` are
    /// conserved to round-off in a closed basin, with horizontal and vertical
    /// gradients, the tide, the sheared flow, the baroclinic flow they drive
    /// and implicit vertical diffusion.
    #[test]
    fn tracer_inventories_are_conserved_under_a_tide() {
        let case = SlopingTide::new();
        let physics = case.physics();
        let mut state = case.state(&physics, |x, s| {
            (
                10.0 + 3.0 * x / case.length - 2.0 * s,
                33.0 + x / case.length + s,
            )
        });
        let t0 = case.inventory(&state, &state.temp);
        let s0 = case.inventory(&state, &state.salt);
        let mut max_err = 0.0_f64;
        case.run(&physics, &mut state, |s, _| {
            max_err = max_err
                .max((case.inventory(s, &s.temp) - t0).abs() / t0)
                .max((case.inventory(s, &s.salt) - s0).abs() / s0);
        });
        // Measured 3.1e-15; before, 1.1e-2 within one period
        assert!(
            max_err < 1e-12,
            "tracer inventory drifted by {max_err:.3e} (relative)"
        );
    }

    /// TODO P4.2 gate: the layer momentum moves with the layer transports. A
    /// tide sloshes across a channel over a bed sloping across it, carrying a
    /// sheared along-channel flow `u = 0.1(σ + ½)`; everything is uniform
    /// along the channel. Without Coriolis, baroclinic pressure or viscosity,
    /// the layers then exchange no volume (`Ω = 0`: each layer's transport
    /// divergence is its share of `∂η/∂t`), and the along-channel momentum of
    /// a layer changes only by the cross-channel advection, which conserves
    /// it, and by its share `Δσ_l` of the depth-mean change. So each layer's
    /// anomaly `∫ H_z,l (u_l − ū) dA` is conserved.
    ///
    /// The old velocity form `∂u/∂t = −∇·(u u)` moved `H_z u` by
    /// `−Δσ_l D u ∂v/∂y` more than the flux form `−∂(H_z v u)/∂y`: the anomaly
    /// drifted by 7.4e-3 of the layer transport within the period.
    #[test]
    fn layer_momentum_anomalies_are_conserved_across_a_tide() {
        let case = SlopingTide::across_channel();
        let physics = case.physics_with(ConstantMixing::new(0.0, 0.0));
        let eos = LinearEOS::default();
        let mut state = case.state(&physics, |_, _| (eos.t0, eos.s0));
        let (nn, nl) = (case.ops.n_nodes, case.sigma.n_levels());
        let anomalies = |s: &Solution3D| -> Vec<f64> {
            let mut column = DGSolution2D::new(case.mesh.n_elements, nn);
            (0..nl)
                .map(|l| {
                    let ds = case.sigma.d_sigma()[l];
                    for (idx, c) in column.data.iter_mut().enumerate() {
                        let depth = s.eta.data[idx] - case.bathymetry.data[idx];
                        *c = depth * ds * (s.u[idx * nl + l] - s.ubar.data[idx]);
                    }
                    column.integrate(&case.ops, &case.geom)
                })
                .collect()
        };
        let initial = anomalies(&state);
        // The layer's transport over the channel (mean depth 8 m, u′ ≈ 0.1)
        let scale = 8.0 / nl as f64 * 0.1 * 200.0 * case.length;
        let (mut max_err, mut max_v) = (0.0_f64, 0.0_f64);
        case.run(&physics, &mut state, |s, p| {
            for (a, b) in anomalies(s).iter().zip(&initial) {
                max_err = max_err.max((a - b).abs() / scale);
            }
            max_v = max_v.max(s.v.iter().fold(0.0, |m, v| m.max(v.abs())));
            let omega = s.w.iter().fold(0.0_f64, |m, w| m.max(w.abs()));
            assert!(omega < 1e-12, "the layers exchanged volume: Ω {omega:.2e}");
            assert!(p.last_surface_residual() < 1e-12);
        });
        assert!(max_v > 1e-2, "test regime: no tide ({max_v:.2e} m/s)");
        // Measured 9.6e-16; before, 7.4e-3
        assert!(
            max_err < 1e-12,
            "layer momentum anomaly drifted by {max_err:.3e} of the layer transport"
        );
    }

    /// TODO P4.3/P4.6 gate: a stratified fjord at rest stays at rest in the
    /// whole mode-split model. The bed drops from 10 to 150 m within ≈ 400 m,
    /// and the stratification is linear in z (temperature, T/S-dependent EOS,
    /// no vertical tracer diffusion). The PGF is exact for it, so the only
    /// currents are round-off.
    #[test]
    fn stratified_fjord_at_rest_stays_at_rest() {
        let length = 1000.0;
        let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, length, 0.0, 100.0, 8, 1));
        let ops = Arc::new(DGOperators2D::new(2));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _| {
            -(80.0 - 70.0 * ((x - 500.0) / 150.0).tanh())
        }));
        let sigma = SigmaGrid::new(10, UniformStretching);
        let swe = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::default(),
        )
        .with_bathymetry(bathymetry.clone())
        .with_formulation(SWEFormulation2D::EntropyStable)
        .build();
        let physics = Hydrostatic3D::new(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            Arc::new(sigma.clone()),
            bathymetry.clone(),
            Arc::new(CoriolisSource2D::f_plane(1.2e-4)),
            LinearEOS::default(),
            ConstantMixing::new(1e-3, 0.0),
            swe,
            no_stress(),
            G,
            RHO0,
        );
        let (nn, nl) = (ops.n_nodes, sigma.n_levels());
        let mut state = Solution3D::new(mesh.n_elements, nn, nl);
        let eos = LinearEOS::default();
        for idx in 0..mesh.n_elements * nn {
            let depth = -bathymetry.data[idx];
            for (l, &s) in sigma.sigma_rho().iter().enumerate() {
                // 6 °C warmer at the surface than at 150 m: N² ≈ 7e-5 s⁻²
                state.temp[idx * nl + l] = eos.t0 + 0.04 * (s * depth + 75.0);
                state.salt[idx * nl + l] = eos.s0;
            }
        }
        physics.update_density(&mut state);

        let mut integrator = ModeSplitIntegrator::new();
        let dt = 60.0;
        for n in 0..60 {
            physics.update_density(&mut state);
            integrator.step(&mut state, &physics, dt, n as f64 * dt);
            physics.post_process(&mut state);
        }
        let max_speed = max_or_nan(state.u.iter().zip(&state.v).map(|(u, v)| u.hypot(*v)));
        // Measured 2.8e-11 m/s after 1 h: round-off forcing (≈ 1e-14 m/s²)
        // accumulating. The σ form started at 1.9e-4 m/s² and blew up (NaN)
        // within 16 steps.
        assert!(
            max_speed < 1e-9,
            "a fjord at rest spun up {max_speed:.3e} m/s in 1 h"
        );
    }

    /// A 3D beach: the bed rises from −4 m to +2 m along 1 km, water sloshing
    /// up it with `amplitude`, `WetDry` 2D module, T/S-dependent EOS,
    /// stratified linearly in z. With `amplitude > 0` a 0.05 Pa wind and
    /// vertical tracer diffusion; at rest neither (diffusion bends the
    /// profile at the bed and surface differently in columns of different
    /// depth, which drives a real boundary flow on a slope: Phillips 1970,
    /// Wunsch 1970; 1.9e-4 m/s here within 1000 s).
    fn beach_3d(amplitude: f64) -> (Physics, Solution3D, Arc<Bathymetry2D>) {
        let length = 1000.0;
        let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, length, 0.0, 100.0, 10, 1));
        let ops = Arc::new(DGOperators2D::new(2));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _| {
            -4.0 + 6.0 * x / length
        }));
        let swe = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::default(),
        )
        .with_bathymetry(bathymetry.clone())
        .with_formulation(SWEFormulation2D::WetDry)
        .with_wet_dry_correction(true)
        .build();
        let sigma = SigmaGrid::new(4, UniformStretching);
        let physics = Hydrostatic3D::new(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            Arc::new(sigma.clone()),
            bathymetry.clone(),
            Arc::new(CoriolisSource2D::f_plane(1.2e-4)),
            LinearEOS::default(),
            ConstantMixing::new(1e-3, if amplitude > 0.0 { 1e-4 } else { 0.0 }),
            swe,
            Forcing {
                surface_stress: [if amplitude > 0.0 { 0.05 } else { 0.0 }, 0.0],
                ..no_stress()
            },
            G,
            RHO0,
        );
        let (nn, nl) = (ops.n_nodes, sigma.n_levels());
        let mut state = Solution3D::new(mesh.n_elements, nn, nl);
        let eos = LinearEOS::default();
        for k in 0..mesh.n_elements {
            for i in 0..nn {
                let el = ElementIndex::new(k);
                let [x, _] = mesh.reference_to_physical(el, ops.nodes_r[i], ops.nodes_s[i]);
                let idx = k * nn + i;
                let b = bathymetry.data[idx];
                let eta = (amplitude * (std::f64::consts::PI * x / length).cos()).max(b);
                state.eta.data[idx] = eta;
                for (l, &s) in sigma.sigma_rho().iter().enumerate() {
                    // Linear in z (isopycnals level at rest): 0.5 °C/m
                    let z = eta + s * (eta - b);
                    state.temp[idx * nl + l] = eos.t0 + 0.5 * (z + 2.0);
                    state.salt[idx * nl + l] = eos.s0;
                }
            }
        }
        physics.update_density(&mut state);
        (physics, state, bathymetry)
    }

    /// Runs `steps` mode-split steps of 10 s on `state`, asserting every
    /// step stays finite, calling `check` after each.
    fn run_beach(
        physics: &Physics,
        state: &mut Solution3D,
        steps: usize,
        mut check: impl FnMut(&Solution3D),
    ) {
        let mut integrator = ModeSplitIntegrator::new();
        let dt = 10.0;
        for n in 0..steps {
            physics.update_density(state);
            integrator.step(state, physics, dt, n as f64 * dt);
            physics.post_process(state);
            assert!(
                max_or_nan(state.u.iter().chain(&state.temp).map(|x| x.abs())).is_finite(),
                "step {n}: the run blew up"
            );
            check(state);
        }
    }

    /// `∫ Σ_l H_z C dA` of a tracer on the beach (dry columns hold none).
    fn beach_inventory(physics: &Physics, state: &Solution3D, tracer: &[f64]) -> f64 {
        let nl = state.n_levels;
        let mut column = DGSolution2D::new(state.n_elements, state.n_nodes);
        for (idx, c) in column.data.iter_mut().enumerate() {
            let depth = state.eta.data[idx] - physics.bathymetry.data[idx];
            *c = (0..nl)
                .map(|l| depth * physics.sigma.d_sigma()[l] * tracer[idx * nl + l])
                .sum();
        }
        column.integrate(&physics.ops, &physics.geom)
    }

    /// TODO P4.5 gate: 3D wetting and drying. Water sloshes up and down a
    /// beach (bed from −4 m to +2 m, 0.3 m amplitude, wind, stratified) for
    /// ≈ 4 periods through the `WetDry` 2D module. Before, the first dry node
    /// gave NaN in the first step (vertical diffusion over a zero-thickness
    /// column); then, once that was masked, films at the 2D velocity cap
    /// (20 m/s) blew up the 3D momentum advection within 44 steps. Now: no
    /// clips, and the temperature stays inside its initial range: exactly in
    /// water shallower than 1 m, elsewhere to 2e-3 °C. The layer means may
    /// leave it: the water at the bed is 0.26 °C colder than the bed layer's
    /// mean, and rising, it cools the layer (1.0e-3 °C below the initial
    /// range with Akima and limited Akima, 1.6e-3 °C with TVD). Upwind
    /// freezes the bed layer at the wall and stays inside.
    #[test]
    fn a_beach_wets_and_dries_in_3d() {
        let (physics, mut state, _) = beach_3d(0.3);
        let (t_min, t_max) = (
            state.temp.iter().copied().fold(f64::MAX, f64::min),
            state.temp.iter().copied().fold(f64::MIN, f64::max),
        );
        let mut thin_seen = 0;
        run_beach(&physics, &mut state, 200, |s| {
            for (idx, column) in s.temp.chunks_exact(s.n_levels).enumerate() {
                // Measured 1.0e-3 °C below, at the bed of the deepest column
                // (see above)
                let depth = s.eta.data[idx] - physics.bathymetry.data[idx];
                let slack = if depth < 1.0 { 1e-9 } else { 2e-3 };
                for &t in column {
                    assert!(
                        t >= t_min - slack && t <= t_max + slack,
                        "temperature {t} left [{t_min}, {t_max}] in {depth:.3} m of water"
                    );
                }
            }
            thin_seen += (0..s.eta.data.len())
                .filter(|&idx| {
                    let d = s.eta.data[idx] - physics.bathymetry.data[idx];
                    d > 0.0 && d < physics.min_column_depth
                })
                .count();
        });
        assert!(thin_seen > 0, "test regime: no thin wet column");
        assert_eq!(physics.swe_physics.negative_depth_clips(), 0);
    }

    /// TODO P4.5 gate: on the beach, uniform T and S stay uniform (constancy
    /// in the shoreline elements, whose tracers are element means per level)
    /// and stratified tracers keep their inventories.
    #[test]
    fn beach_tracers_are_constant_and_conserved() {
        let (physics, mut state, _) = beach_3d(0.3);
        state.temp.fill(12.3);
        state.salt.fill(33.1);
        let mut drift = 0.0_f64;
        run_beach(&physics, &mut state, 200, |s| {
            drift = max_or_nan(
                s.temp
                    .iter()
                    .map(|t| (t - 12.3).abs())
                    .chain(s.salt.iter().map(|x| (x - 33.1).abs()))
                    .chain([drift]),
            );
        });
        // Measured 4.7e-9 (4e-10 of the values): the 2D pass balances nearly
        // dry elements to round-off of the domain's η change, which their
        // tiny volumes amplify. Wet elements hold to ≈ 1e-13 (the P4.2 gate).
        assert!(drift < 5e-8, "uniform tracers drifted by {drift:.3e}");

        let (physics, mut state, _) = beach_3d(0.3);
        let (t0, s0) = (
            beach_inventory(&physics, &state, &state.temp),
            beach_inventory(&physics, &state, &state.salt),
        );
        let mut max_err = 0.0_f64;
        run_beach(&physics, &mut state, 200, |s| {
            max_err = max_or_nan([
                max_err,
                (beach_inventory(&physics, s, &s.temp) - t0).abs() / t0,
                (beach_inventory(&physics, s, &s.salt) - s0).abs() / s0,
            ]);
        });
        // Measured 2.3e-11: elements too dry to define a concentration keep
        // their last one (3e-15 without wetting and drying, the P4.2 gate)
        assert!(max_err < 5e-11, "inventories drifted by {max_err:.3e}");
    }

    /// TODO P4.5 gate: a stratified lake at rest with dry land stays at
    /// rest: the thin columns at the shoreline exert no baroclinic pressure.
    #[test]
    fn stratified_lake_with_dry_land_stays_at_rest() {
        let (physics, mut state, _) = beach_3d(0.0);
        let mut max_speed = 0.0_f64;
        run_beach(&physics, &mut state, 100, |s| {
            max_speed = max_or_nan(
                s.u.iter()
                    .zip(&s.v)
                    .map(|(u, v)| u.hypot(*v))
                    .chain([max_speed]),
            );
        });
        // Measured 1.3e-13 m/s
        assert!(max_speed < 1e-10, "the lake spun up {max_speed:.3e} m/s");
    }

    /// A flat, doubly periodic ocean `depth` deep, 40 km square (4 × 4 P1),
    /// without rotation, on `n_levels` uniform σ-levels, with the reference
    /// T and S (no baroclinic pressure) at rest, and its 2D module (no
    /// friction of its own).
    fn flat_periodic_3d(
        depth: f64,
        n_levels: usize,
        forcing: Forcing,
        viscosity: f64,
    ) -> (Physics, Solution3D) {
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
        .build();
        let physics = Hydrostatic3D::new(
            mesh.clone(),
            ops.clone(),
            geom,
            Arc::new(SigmaGrid::new(n_levels, UniformStretching)),
            bathymetry,
            Arc::new(CoriolisSource2D::f_plane(0.0)),
            LinearEOS::default(),
            ConstantMixing::new(viscosity, viscosity),
            swe,
            forcing,
            G,
            RHO0,
        );
        let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, n_levels);
        let eos = LinearEOS::default();
        state.temp.fill(eos.t0);
        state.salt.fill(eos.s0);
        physics.update_density(&mut state);
        (physics, state)
    }

    /// TODO P4.4 gate: wind against quadratic bottom drag over a flat,
    /// periodic, non-rotating ocean. The steady state passes the wind stress
    /// through the column unchanged: a linear profile of shear `τ/(ρ₀ν)`,
    /// and a bottom layer with `C_d u_b² = τ/ρ₀`. The depth mean gets there
    /// only if the pass (`−r·ū`) and `G` (`−r·(u_b − ū)`) together apply the
    /// drag of the bottom layer, not of the depth mean.
    #[test]
    fn wind_against_bottom_drag_reaches_the_quadratic_balance() {
        let (depth, tau, cd, nu, n_levels) = (10.0, 0.1, 2.5e-3, 0.01, 10);
        let forcing = Forcing {
            surface_stress: [tau, 0.0],
            ..no_stress()
        };
        let (physics, mut state) = flat_periodic_3d(depth, n_levels, forcing, nu);
        let physics = physics.with_bottom_drag(BottomDrag3D::quadratic(cd));
        // The depth mean relaxes at 2C_d u_b/D ≈ 1e-4 /s: 40 e-foldings
        let mut sim = Simulation3D::new(physics, ModeSplitIntegrator::new())
            .with_cfl(10.0)
            .with_dt_max(600.0);
        let result = sim.run(&mut state, 0.0, 4e5);
        assert!(result.success, "drag run failed: {:?}", result.error);

        let u_b = (tau / (RHO0 * cd)).sqrt();
        let step = tau / (RHO0 * nu) * depth / n_levels as f64;
        let (mut bottom_err, mut shear_err, mut v_max) = (0.0_f64, 0.0_f64, 0.0_f64);
        for col in 0..state.eta.data.len() {
            let u = &state.u[col * n_levels..(col + 1) * n_levels];
            let v = &state.v[col * n_levels..(col + 1) * n_levels];
            bottom_err = max_or_nan([bottom_err, (u[0] - u_b).abs() / u_b]);
            for pair in u.windows(2) {
                shear_err = max_or_nan([shear_err, (pair[1] - pair[0] - step).abs() / step]);
            }
            v_max = max_or_nan(v.iter().map(|v| v.abs()).chain([v_max]));
        }
        // Measured 1.2e-13 and 2.1e-13. With the pass's −r·ū alone (no shear
        // part in G) the bottom layer settles 9.4 % off, the shear 16 %.
        assert!(
            bottom_err < 1e-10,
            "bottom-layer velocity off C_d u_b² = τ/ρ₀ by {bottom_err:.3e} of u_b = {u_b:.4}"
        );
        assert!(
            shear_err < 1e-10,
            "layer shear off τ/(ρ₀ν) by {shear_err:.3e}"
        );
        assert!(v_max < 1e-12, "cross-wind flow {v_max:.3e} m/s");
    }

    /// TODO P4.4 gate: without vertical shear the 3D bottom drag is the 2D
    /// quadratic friction `C_d|ū|ū/h`. A uniform flow over a flat, periodic
    /// ocean on one σ-level decays as `u₀/(1 + C_d u₀ t/D)`; the mode split
    /// (rate frozen over each step) converges to it at first order in Δt.
    #[test]
    fn unsheared_bottom_drag_decays_like_the_2d_quadratic_friction() {
        let (depth, cd, u0, t_end) = (10.0, 2.5e-3, 1.0, 8000.0);
        let exact = u0 / (1.0 + cd * u0 * t_end / depth);

        let mut errors = Vec::new();
        for dt in [400.0, 200.0, 100.0] {
            let (physics, mut state) = flat_periodic_3d(depth, 1, no_stress(), 0.0);
            let physics = physics.with_bottom_drag(BottomDrag3D::quadratic(cd));
            state.u.fill(u0);
            state.ubar.data.fill(u0);
            let mut integrator = ModeSplitIntegrator::new();
            let steps = (t_end / dt).round() as usize;
            for n in 0..steps {
                integrator.step(&mut state, &physics, dt, n as f64 * dt);
            }
            let err = max_or_nan(state.ubar.data.iter().map(|u| (u - exact).abs()));
            let spread = max_or_nan(state.u.iter().map(|u| (u - state.ubar.data[0]).abs()));
            assert!(
                spread < 1e-14,
                "dt {dt}: the uniform flow lost uniformity by {spread:.3e}"
            );
            errors.push(err);
        }

        // The 2D model with the same C_d (Chézy), implicit friction
        let (physics, _) = flat_periodic_3d(depth, 1, no_stress(), 0.0);
        let swe = PhysicsBuilder::swe_2d(
            physics.mesh.clone(),
            physics.ops.clone(),
            physics.geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::default(),
        )
        .with_bathymetry(physics.bathymetry.clone())
        .with_implicit_friction(crate::source::ChezyFriction2D::new(cd))
        .build();
        let mut q = SWESolution2D::new(physics.mesh.n_elements, physics.ops.n_nodes);
        q.data[SWE_VAR_H].fill(depth);
        q.data[SWE_VAR_HU].fill(depth * u0);
        let result = Simulation::new(swe, SSPRK3)
            .with_cfl(0.5)
            .run(&mut q, 0.0, t_end);
        assert!(result.success, "2D reference failed: {:?}", result.error);
        let err_2d = max_or_nan(
            q.data[SWE_VAR_HU]
                .iter()
                .map(|hu| (hu / depth - exact).abs()),
        );

        // Measured 4.8e-3, 2.4e-3, 1.2e-3 at Δt = 400, 200, 100 s (rates
        // 1.01, 1.01); the 2D model at CFL 0.5: 2.1e-3
        assert!(err_2d < 1e-2 * exact, "2D reference off by {err_2d:.3e}");
        for pair in errors.windows(2) {
            let rate = (pair[0] / pair[1]).log2();
            assert!(
                (0.9..1.3).contains(&rate),
                "convergence rate {rate:.2} (errors {errors:?})"
            );
        }
        assert!(errors[2] < 1e-2 * exact, "error {:.3e}", errors[2]);
    }

    /// TODO P4.4 gate: the beach of the P4.5 gates with log-layer drag. Its
    /// thin bottom layers put `C_d` at its upper bound (0.1), so in the
    /// shallows `r·Δt/D` is far beyond an explicit drag's limit (≈ 2); the
    /// implicit drag keeps the run finite, without clips, and takes energy
    /// out of the slosh.
    #[test]
    fn a_beach_wets_and_dries_with_log_layer_drag() {
        let speed = |s: &Solution3D| max_or_nan(s.ubar.data.iter().map(|u| u.abs()));
        let (physics, mut state, _) = beach_3d(0.3);
        let mut free = 0.0_f64;
        run_beach(&physics, &mut state, 200, |s| free = free.max(speed(s)));

        let (physics, mut state, _) = beach_3d(0.3);
        let physics = physics.with_bottom_drag(BottomDrag3D::log_layer(0.005));
        let mut dragged = 0.0_f64;
        run_beach(&physics, &mut state, 200, |s| {
            dragged = dragged.max(speed(s))
        });
        assert_eq!(physics.swe_physics.negative_depth_clips(), 0);
        // Measured 3.15 m/s free, 0.72 m/s with drag
        assert!(dragged < 0.5 * free, "drag {dragged} vs free {free}");
    }

    /// Uniform far field `(η = 0, ū = 0, v̄)` for a characteristic OBC.
    #[derive(Clone, Copy, Debug)]
    struct UniformFarField(f64);

    impl crate::boundary::ExternalStateProvider for UniformFarField {
        fn external_state(&self, _ctx: &crate::boundary::BCContext2D) -> ExternalState {
            ExternalState::new(0.0, 0.0, self.0)
        }
    }

    type OpenPhysics = Hydrostatic3D<LinearEOS, ConstantMixing, CharacteristicOBC<UniformFarField>>;

    /// TODO P4.2 gate: open boundaries pass a sheared flow through. A
    /// channel, periodic in x and open at y = 0 and y = 40 km, carries the
    /// steady wind-against-drag flow of
    /// `wind_against_bottom_drag_reaches_the_quadratic_balance` (along y,
    /// linear shear, `C_d v_b² = τ/ρ₀`), with a characteristic OBC whose far
    /// field is that flow's depth mean. The flow must stay exactly as it is:
    /// every layer leaves through one open end and enters through the other
    /// with its own velocity. Treating the open faces as walls for the 3D
    /// kernels (the behaviour before) gives the layers a depth-uniform share
    /// of the boundary flux and reflects their momentum there.
    #[test]
    fn a_sheared_flow_passes_through_open_boundaries() {
        let (depth, tau, cd, nu, n_levels) = (10.0, 0.1, 2.5e-3, 0.01, 10);
        let v_b = (tau / (RHO0 * cd)).sqrt();
        let step = tau / (RHO0 * nu) * depth / n_levels as f64;
        let profile: Vec<f64> = (0..n_levels).map(|l| v_b + l as f64 * step).collect();
        let v_mean = profile.iter().sum::<f64>() / n_levels as f64;

        let run = |walls: &[BoundaryTag]| -> (f64, f64) {
            let mut mesh = Mesh2D::channel_periodic_x(0.0, 40e3, 0.0, 40e3, 4, 4);
            for edge in &mut mesh.edges {
                if edge.right.is_none() {
                    edge.boundary_tag = Some(BoundaryTag::Open);
                }
            }
            let mesh = Arc::new(mesh);
            let ops = Arc::new(DGOperators2D::new(1));
            let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
            let bathymetry = Arc::new(Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -depth));
            let swe = PhysicsBuilder::swe_2d(
                mesh.clone(),
                ops.clone(),
                geom.clone(),
                ShallowWater2D::new(G),
                CharacteristicOBC::new(UniformFarField(v_mean)),
            )
            .with_bathymetry(bathymetry.clone())
            .build();
            let physics: OpenPhysics = Hydrostatic3D::new(
                mesh.clone(),
                ops.clone(),
                geom,
                Arc::new(SigmaGrid::new(n_levels, UniformStretching)),
                bathymetry,
                Arc::new(CoriolisSource2D::f_plane(0.0)),
                LinearEOS::default(),
                ConstantMixing::new(nu, nu),
                swe,
                Forcing {
                    surface_stress: [0.0, tau],
                    ..no_stress()
                },
                G,
                RHO0,
            )
            .with_bottom_drag(BottomDrag3D::quadratic(cd))
            .with_wall_tags(walls.iter().copied());

            let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, n_levels);
            let eos = LinearEOS::default();
            state.temp.fill(eos.t0);
            state.salt.fill(eos.s0);
            for column in state.v.chunks_exact_mut(n_levels) {
                column.copy_from_slice(&profile);
            }
            state.vbar.data.fill(v_mean);
            physics.update_density(&mut state);

            let mut sim = Simulation3D::new(physics, ModeSplitIntegrator::new())
                .with_cfl(10.0)
                .with_dt_max(600.0);
            let result = sim.run(&mut state, 0.0, 86400.0);
            assert!(
                result.success,
                "open-channel run failed: {:?}",
                result.error
            );
            let velocity_err = max_or_nan(
                state
                    .v
                    .chunks_exact(n_levels)
                    .flat_map(|col| col.iter().zip(&profile).map(|(v, p)| (v - p).abs()))
                    .chain(state.u.iter().map(|u| u.abs())),
            );
            let eta = max_or_nan(state.eta.data.iter().map(|e| e.abs()));
            (velocity_err / v_mean, eta)
        };

        let (err, eta) = run(&[BoundaryTag::Wall]);
        let (closed_err, _) = run(&[BoundaryTag::Wall, BoundaryTag::Open]);
        // Measured 5.7e-14 of v̄ and 3.6e-15 m; with the open faces as walls
        // the flow is 2.6e-4 of v̄ off after a day (1.9e-2 with the old
        // velocity-form advection, which also mirrored the velocity there)
        assert!(err < 1e-10, "the flow changed by {err:.3e} of v̄ in a day");
        assert!(eta < 1e-10, "η moved by {eta:.3e} m");
        assert!(
            closed_err > 1e-4,
            "test regime: walls changed it by only {closed_err:.3e}"
        );
    }

    /// A characteristic OBC on faces tagged [`BoundaryTag::Open`], walls
    /// elsewhere.
    #[derive(Clone, Debug)]
    struct OpenOrWall<P>(CharacteristicOBC<P>);

    impl<P: crate::boundary::ExternalStateProvider> SWEBoundaryCondition2D for OpenOrWall<P> {
        fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
            match ctx.boundary_tag {
                Some(BoundaryTag::Open) => self.0.ghost_state(ctx),
                _ => Reflective2D::default().ghost_state(ctx),
            }
        }

        fn boundary_state(&self, ctx: &BCContext2D) -> BoundaryState {
            match ctx.boundary_tag {
                Some(BoundaryTag::Open) => self.0.boundary_state(ctx),
                _ => Reflective2D::default().boundary_state(ctx),
            }
        }

        fn name(&self) -> &'static str {
            "open_or_wall"
        }

        fn is_wall(&self, tag: Option<BoundaryTag>) -> Option<bool> {
            Some(tag != Some(BoundaryTag::Open))
        }
    }

    /// TODO P4.2 gate: a tide enters the 3D model through an open boundary as
    /// it enters the 2D model. A 30 km channel shoaling from 20 to 10 m, open
    /// at its west end to an M2 tide (characteristic OBC, 3 h ramp), walls
    /// elsewhere, run for a day: without stresses or density differences the
    /// flow has no shear, and the mode-split η must follow the 2D model's.
    #[test]
    fn a_tide_enters_through_an_open_boundary_as_in_the_2d_model() {
        let (length, amplitude) = (30e3, 0.5);
        let mesh = Arc::new(Mesh2D::uniform_rectangle_with_sides(
            0.0,
            length,
            0.0,
            3e3,
            10,
            1,
            [
                BoundaryTag::Wall,
                BoundaryTag::Wall,
                BoundaryTag::Wall,
                BoundaryTag::Open,
            ],
        ));
        let ops = Arc::new(DGOperators2D::new(2));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _| {
            -20.0 + 10.0 * x / length
        }));
        let swe = || {
            let tide = OpenOrWall(CharacteristicOBC::new(
                HarmonicTide::m2(amplitude, 0.0).with_ramp_up(3.0 * 3600.0),
            ));
            PhysicsBuilder::swe_2d(
                mesh.clone(),
                ops.clone(),
                geom.clone(),
                ShallowWater2D::new(G),
                tide,
            )
            .with_bathymetry(bathymetry.clone())
            // Balanced over the slope (`Standard` would need a
            // `BathymetrySource2D`)
            .with_formulation(SWEFormulation2D::EntropyStable)
            .build()
        };
        let t_end = 86400.0;

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for (h, b) in q.data[SWE_VAR_H].iter_mut().zip(&bathymetry.data) {
            *h = -b;
        }
        let result = Simulation::new(swe(), SSPRK3)
            .with_cfl(0.5)
            .run(&mut q, 0.0, t_end);
        assert!(result.success, "2D reference failed: {:?}", result.error);

        let physics = Hydrostatic3D::new(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            Arc::new(SigmaGrid::new(4, UniformStretching)),
            bathymetry.clone(),
            Arc::new(CoriolisSource2D::f_plane(0.0)),
            LinearEOS::default(),
            ConstantMixing::new(1e-3, 1e-3),
            swe(),
            no_stress(),
            G,
            RHO0,
        );
        let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, 4);
        let eos = LinearEOS::default();
        state.temp.fill(eos.t0);
        state.salt.fill(eos.s0);
        physics.update_density(&mut state);
        let mut sim = Simulation3D::new(physics, ModeSplitIntegrator::new())
            .with_cfl(10.0)
            .with_dt_max(300.0);
        let result = sim.run(&mut state, 0.0, t_end);
        assert!(result.success, "mode-split run failed: {:?}", result.error);

        let max_diff = max_or_nan(
            q.data[SWE_VAR_H]
                .iter()
                .zip(&bathymetry.data)
                .zip(&state.eta.data)
                .map(|((h, b), eta)| (h + b - eta).abs()),
        );
        let max_eta = max_or_nan(state.eta.data.iter().map(|e| e.abs()));
        assert!(
            max_eta > 0.3 * amplitude,
            "test regime: no tide in the channel"
        );
        // Measured 3.9e-6 m (|η| up to 0.38 m)
        assert!(
            max_diff < 1e-4 * amplitude,
            "η differs from the 2D model by {max_diff:.3e} m"
        );
    }

    /// A parent exchange flow for the 3D nesting: out at the surface, in at
    /// the bed (`u = −0.2 (σ + ½)` m/s, no depth mean), temperature
    /// `T₀ + ΔT (σ + ½)`.
    struct ExchangeParent {
        delta_t: f64,
    }

    impl crate::boundary::ParentColumns3D for ExchangeParent {
        fn supplies(&self) -> crate::boundary::Supplied {
            crate::boundary::Supplied {
                velocity: true,
                tracers: true,
            }
        }

        fn column(
            &self,
            _ctx: &crate::boundary::ColumnContext3D,
            sigma: &SigmaGrid,
            out: crate::boundary::ParentColumn<'_>,
        ) -> bool {
            let eos = LinearEOS::default();
            for (l, &s) in sigma.sigma_rho().iter().enumerate() {
                out.u[l] = -0.2 * (s + 0.5);
                out.v[l] = 0.0;
                out.temp[l] = eos.t0 + self.delta_t * (s + 0.5);
                out.salt[l] = eos.s0;
            }
            true
        }
    }

    fn exchange(delta_t: f64) -> Option<Arc<dyn crate::boundary::ParentColumns3D>> {
        Some(Arc::new(ExchangeParent { delta_t }))
    }

    type NestedPhysics =
        Hydrostatic3D<LinearEOS, ConstantMixing, OpenOrWall<crate::boundary::StillWater>>;

    /// An 8 km × 1 km channel, 10 m deep, open at its west end to still
    /// water, with five levels, nested (or not) in the parent `parent` gives
    /// for its mesh and operators, over a 2 km band with 1 h time scales,
    /// run for 12 h from rest at `T₀`. Returns the physics, the state and the
    /// x of every node.
    fn run_nested_channel(
        parent: impl FnOnce(
            &Mesh2D,
            &DGOperators2D,
        ) -> Option<Arc<dyn crate::boundary::ParentColumns3D>>,
    ) -> (NestedPhysics, Solution3D, Vec<f64>) {
        use crate::boundary::{Nesting3D, NestingBand3D, StillWater};
        let mesh = Arc::new(Mesh2D::uniform_rectangle_with_sides(
            0.0,
            8e3,
            0.0,
            1e3,
            8,
            1,
            [
                BoundaryTag::Wall,
                BoundaryTag::Wall,
                BoundaryTag::Wall,
                BoundaryTag::Open,
            ],
        ));
        let ops = Arc::new(DGOperators2D::new(2));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -10.0));
        let n_levels = 5;
        let swe = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            OpenOrWall(CharacteristicOBC::new(StillWater::default())),
        )
        .with_bathymetry(bathymetry.clone())
        .build();
        let mut physics = Hydrostatic3D::new(
            mesh.clone(),
            ops.clone(),
            geom,
            Arc::new(SigmaGrid::new(n_levels, UniformStretching)),
            bathymetry,
            Arc::new(CoriolisSource2D::f_plane(0.0)),
            LinearEOS::default(),
            ConstantMixing::new(1e-4, 1e-5),
            swe,
            no_stress(),
            G,
            RHO0,
        );
        if let Some(parent) = parent(&mesh, &ops) {
            let band = NestingBand3D {
                width: 2e3,
                velocity_timescale: Some(3600.0),
                tracer_timescale: Some(3600.0),
                ..NestingBand3D::default()
            };
            let nesting =
                Nesting3D::new(parent, &mesh, &ops, n_levels, &[BoundaryTag::Open], &band)
                    .expect("an open boundary to nest");
            physics = physics.with_nesting(nesting);
        }
        let eos = LinearEOS::default();
        let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, n_levels);
        state.temp.fill(eos.t0);
        state.salt.fill(eos.s0);
        let x: Vec<f64> = (0..mesh.n_elements * ops.n_nodes)
            .map(|flat| {
                let (k, i) = (flat / ops.n_nodes, flat % ops.n_nodes);
                mesh.reference_to_physical(ElementIndex::new(k), ops.nodes_r[i], ops.nodes_s[i])[0]
            })
            .collect();
        let dt = 120.0;
        let mut integrator = ModeSplitIntegrator::new();
        for n in 0..360 {
            physics.update_density(&mut state);
            integrator.step(&mut state, &physics, dt, n as f64 * dt);
            physics.post_process(&mut state);
            assert!(
                max_or_nan(state.u.iter().chain(&state.temp).copied()).is_finite(),
                "step {n}: the run blew up"
            );
        }
        assert_eq!(physics.swe_physics.negative_depth_clips(), 0);
        (physics, state, x)
    }

    /// TODO P4.2 gate: 3D nesting with the parent's own water keeps uniform
    /// tracers uniform. The parent's exchange flow shears the boundary
    /// fluxes and relaxes the band's shear; its tracer is the interior's, so
    /// neither the inflow nor the relaxation may change it.
    #[test]
    fn nesting_in_the_same_water_keeps_tracers_uniform() {
        let (_, state, _) = run_nested_channel(|_, _| exchange(0.0));
        let eos = LinearEOS::default();
        let drift = max_or_nan(state.temp.iter().map(|t| (t - eos.t0).abs()));
        let shear = max_or_nan(
            state
                .u
                .chunks_exact(state.n_levels)
                .zip(&state.ubar.data)
                .flat_map(|(column, ubar)| column.iter().map(move |u| (u - ubar).abs())),
        );
        // Measured 3.8e-12 °C under a 0.10 m/s exchange shear
        assert!(
            shear > 1e-2,
            "test regime: no exchange flow ({shear:.2e} m/s)"
        );
        assert!(drift < 1e-10, "uniform temperature drifted by {drift:.3e}");
    }

    /// TODO P4.2 gate: the band takes on the parent's stratification and
    /// shear. An exchange flow of warm surface water out and cold bottom
    /// water in (ΔT = 4 °C) is nested at the west end of a channel at rest;
    /// after 12 h (12 relaxation times) the boundary columns hold the
    /// parent's profiles. Without nesting nothing happens.
    #[test]
    fn nesting_imposes_the_parent_profiles_at_the_boundary() {
        let delta_t = 4.0;
        let (physics, state, x) = run_nested_channel(|_, _| exchange(delta_t));
        let sigma = physics.sigma.clone();
        let eos = LinearEOS::default();
        let nl = state.n_levels;
        let (mut t_err, mut u_err) = (0.0_f64, 0.0_f64);
        for (flat, &x) in x.iter().enumerate() {
            let column = flat * nl..(flat + 1) * nl;
            let ubar = state.ubar.data[flat];
            for (l, &s) in sigma.sigma_rho().iter().enumerate() {
                let t = state.temp[column.start + l];
                if x == 0.0 {
                    t_err = t_err.max((t - (eos.t0 + delta_t * (s + 0.5))).abs() / delta_t);
                    let shear = state.u[column.start + l] - ubar;
                    u_err = u_err.max((shear + 0.2 * (s + 0.5)).abs() / 0.08);
                }
            }
        }
        // Measured 2.3e-2 of ΔT and 5.2e-2 of the shear amplitude: the
        // gravity current the stratification drives pulls against the
        // relaxation
        assert!(
            t_err < 0.05,
            "boundary temperature off the parent's by {t_err:.3} of ΔT"
        );
        assert!(
            u_err < 0.1,
            "boundary shear off the parent's by {u_err:.3} of its amplitude"
        );

        let (_, still, _) = run_nested_channel(|_, _| None);
        let moved = max_or_nan(
            still
                .temp
                .iter()
                .map(|t| (t - eos.t0).abs())
                .chain(still.u.iter().map(|u| u.abs())),
        );
        // Measured 5.7e-12: round-off
        assert!(
            moved < 1e-10,
            "without nesting the channel moved: {moved:.3e}"
        );
    }

    /// TODO P4.2 gate: a parent read from gridded profiles nests like the
    /// analytic one. The exchange flow of
    /// `nesting_imposes_the_parent_profiles_at_the_boundary`, written as an
    /// [`crate::io::OceanModelReader`] on z-levels every 2 m around the
    /// channel and sampled by [`crate::boundary::OceanModelColumns`]
    /// (horizontal stencils, time, depths of the child layers), drives the
    /// same run. The profiles are linear in depth, so the z-levels hold them
    /// exactly; what differs is that the reader's profile follows depth
    /// below the surface and the analytic one σ, by a relative `η/D`.
    #[test]
    fn a_parent_read_on_z_levels_nests_like_the_analytic_profiles() {
        use crate::boundary::{OceanColumnsOptions, OceanModelColumns};
        use crate::io::{
            CoordinateProjection, FieldSeries, GeoGrid, LocalProjection, OceanModelReader,
            ProfileLevels, ProfileSeries,
        };
        use crate::time::ModelClock;
        let delta_t = 4.0;
        let t0 = 1_706_594_400.0;
        let eos = LinearEOS::default();
        let reader_parent = |mesh: &Mesh2D, ops: &DGOperators2D| {
            let projection = LocalProjection::new(63.5, 8.5);
            let (lat0, lon0) = projection.xy_to_geo(-2000.0, -2000.0);
            let (lat1, lon1) = projection.xy_to_geo(10_000.0, 3000.0);
            let axis = |a: f64, b: f64| (0..7).map(|i| a + (b - a) * i as f64 / 6.0).collect();
            let grid = GeoGrid::regular(axis(lon0, lon1), axis(lat0, lat1)).unwrap();
            let m = grid.len();
            let levels: Vec<f64> = (0..7).map(|i| 2.0 * i as f64).collect();
            let profiles = |f: &dyn Fn(f64) -> f64| {
                let data = (0..2 * m)
                    .flat_map(|_| levels.iter().map(|&d| f(d) as f32))
                    .collect();
                ProfileSeries::new(m, ProfileLevels::Depth(levels.clone()), data)
            };
            let reader = OceanModelReader::new(grid, vec![t0, t0 + 13.0 * 3600.0])
                .unwrap()
                .with_ssh(FieldSeries::new(m, vec![0.0; 2 * m]))
                .with_profiles(
                    Some((profiles(&|d| -0.2 * (0.5 - d / 10.0)), profiles(&|_| 0.0))),
                    Some(profiles(&|d| eos.t0 + delta_t * (0.5 - d / 10.0))),
                    Some(profiles(&|_| eos.s0)),
                );
            let parent: Arc<dyn crate::boundary::ParentColumns3D> =
                Arc::new(OceanModelColumns::new(
                    Arc::new(reader),
                    mesh,
                    ops,
                    projection,
                    ModelClock::new(t0),
                    OceanColumnsOptions::default(),
                ));
            Some(parent)
        };
        let (_, read, _) = run_nested_channel(reader_parent);
        let (_, analytic, _) = run_nested_channel(|_, _| exchange(delta_t));
        let t_diff = max_or_nan(
            read.temp
                .iter()
                .zip(&analytic.temp)
                .map(|(a, b)| (a - b).abs() / delta_t),
        );
        let u_diff = max_or_nan(
            read.u
                .iter()
                .zip(&analytic.u)
                .map(|(a, b)| (a - b).abs() / 0.08),
        );
        // Measured 1.9e-5 of ΔT and 3.7e-5 of the shear, with |η| up to
        // 6.5e-4 m: the relative η/D
        assert!(t_diff < 2e-4, "T differs by {t_diff:.3e} of ΔT");
        assert!(
            u_diff < 2e-4,
            "u differs by {u_diff:.3e} of the shear amplitude"
        );
    }

    /// A mode-1 internal-wave pulse in the middle of a 20 km channel, 20 m
    /// deep, linearly stratified (N = 0.05 s⁻¹, `c₁ = NH/π` ≈ 0.32 m/s), 1 km
    /// wide and 0.8 m high, with the ends `ends`: walls, or open to still
    /// water through a characteristic OBC, relaxed over `sponge` (if any) to
    /// the stratification at rest ([`crate::boundary::ReferenceColumns`]).
    /// P1 on 500 m, six levels, no tracer diffusion. Returns the largest
    /// temperature anomaly `|T − T_bg(z)|` (°C) over the middle four levels
    /// after every hour.
    fn internal_wave_pulse(
        ends: BoundaryTag,
        hours: usize,
        sponge: Option<crate::boundary::NestingBand3D>,
    ) -> Vec<f64> {
        use crate::boundary::{Nesting3D, ReferenceColumns, StillWater};
        let (length, depth, width, height) = (20e3, 20.0, 1e3, 0.8);
        let mesh = Arc::new(Mesh2D::uniform_rectangle_with_sides(
            0.0,
            length,
            0.0,
            1e3,
            40,
            1,
            [BoundaryTag::Wall, ends, BoundaryTag::Wall, ends],
        ));
        let ops = Arc::new(DGOperators2D::new(1));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -depth));
        let sigma = SigmaGrid::new(6, UniformStretching);
        let swe = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            OpenOrWall(CharacteristicOBC::new(StillWater::default())),
        )
        .with_bathymetry(bathymetry.clone())
        .build();
        let mut physics = Hydrostatic3D::new(
            mesh.clone(),
            ops.clone(),
            geom,
            Arc::new(sigma.clone()),
            bathymetry.clone(),
            Arc::new(CoriolisSource2D::f_plane(0.0)),
            LinearEOS::default(),
            ConstantMixing::new(1e-4, 0.0),
            swe,
            no_stress(),
            G,
            RHO0,
        );
        let eos = LinearEOS::default();
        // N² = gα dT/dz
        let gradient = 0.05_f64.powi(2) / (G * eos.alpha);
        let background = |z: f64| eos.t0 + gradient * (z + 0.5 * depth);
        let (nn, nl) = (ops.n_nodes, sigma.n_levels());
        let mut state = Solution3D::new(mesh.n_elements, nn, nl);
        state.salt.fill(eos.s0);
        for column in state.temp.chunks_exact_mut(nl) {
            for (t, &s) in column.iter_mut().zip(sigma.sigma_rho()) {
                *t = background(s * depth);
            }
        }
        if let Some(band) = sponge {
            let rest = Arc::new(ReferenceColumns::from_state(&state));
            let nesting = Nesting3D::new(rest, &mesh, &ops, nl, &[ends], &band)
                .expect("open ends to relax at");
            physics = physics.with_nesting(nesting);
        }
        for idx in 0..mesh.n_elements * nn {
            let (k, i) = (idx / nn, idx % nn);
            let [x, _] =
                mesh.reference_to_physical(ElementIndex::new(k), ops.nodes_r[i], ops.nodes_s[i]);
            let pulse = height * (-((x - 0.5 * length) / width).powi(2)).exp();
            for (l, &s) in sigma.sigma_rho().iter().enumerate() {
                let displacement = pulse * (std::f64::consts::PI * s).sin().abs();
                state.temp[idx * nl + l] = background(s * depth - displacement);
            }
        }
        physics.update_density(&mut state);
        let sigma_rho = sigma.sigma_rho();
        let anomaly = |s: &Solution3D| {
            max_or_nan((0..s.eta.data.len()).flat_map(|idx| {
                let (eta, d) = (s.eta.data[idx], s.eta.data[idx] - bathymetry.data[idx]);
                (1..5).map(move |l| {
                    let z = eta + sigma_rho[l] * d;
                    (s.temp[idx * nl + l] - background(z)).abs()
                })
            }))
        };
        let dt = 240.0;
        let steps_per_hour = (3600.0 / dt) as usize;
        let mut integrator = ModeSplitIntegrator::new();
        let mut history = vec![anomaly(&state)];
        for n in 0..hours * steps_per_hour {
            physics.update_density(&mut state);
            integrator.step(&mut state, &physics, dt, n as f64 * dt);
            physics.post_process(&mut state);
            if (n + 1) % steps_per_hour == 0 {
                history.push(anomaly(&state));
            }
        }
        history
    }

    /// TODO P4.2 gate: an internal wave leaves through a relaxed open
    /// boundary. A mode-1 pulse splits into two, each half the initial
    /// amplitude, which reach the open ends after ≈ 9 h. A 4 km band relaxing
    /// to the stratification at rest (30 min on the boundary) lets them out:
    /// what remains in the channel at 16 h is 4.5 % of the outgoing pulse.
    /// With walls the pulses come back.
    ///
    /// With first-order upwind vertical advection (before P4.5's Akima) 36 %
    /// remained, but most of it was not reflection: the scheme's mixing left
    /// an anomaly of 0.2 °C along the pulses' path in the middle of the
    /// channel from hour 4 on, before anything could return from the ends
    /// (0.02 °C with Akima).
    ///
    /// Without the band the open faces extrapolate, and that is unstable for
    /// stratified flow: a pulse like this (P2, 50 m, N = 0.02 s⁻¹) drove an
    /// exchange flow at the boundary that displaced its isopycnals without
    /// bound (the anomaly grew from 0.24 to 6 °C and |u| to 0.26 m/s within
    /// 8 h). Boundary values of the state at rest without a band are stable
    /// but reflect the pulse completely.
    #[test]
    fn an_internal_wave_leaves_through_a_relaxed_open_boundary() {
        use crate::boundary::NestingBand3D;
        let band = NestingBand3D {
            width: 4e3,
            velocity_timescale: Some(1800.0),
            tracer_timescale: Some(1800.0),
            ..NestingBand3D::default()
        };
        let open = internal_wave_pulse(BoundaryTag::Open, 16, Some(band));
        let (initial, split, outgoing, reflected) = (open[0], open[1], open[6], open[16]);
        assert!(
            (split / initial - 0.5).abs() < 0.05,
            "test regime: the pulse should split into halves ({split:.4} of {initial:.4})"
        );
        // Measured 4.5 %. 2 km bands: 4.9 % (30 min), 12 % (1 h); 4 km at
        // 1 h 4.4 %, at 2 h 14 %; 6 km at 1 h 2.5 %; 1 km at 30 min 18 %
        let reflection = reflected / outgoing;
        assert!(
            reflection < 0.08,
            "{:.1} % of the pulse reflected at the relaxed open ends",
            100.0 * reflection
        );
        // The P1 channel damps the 1 km pulse: 0.86 of it is back at 16 h
        let walls = internal_wave_pulse(BoundaryTag::Wall, 16, None);
        assert!(
            walls[16] / walls[6] > 0.7,
            "test regime: walls should reflect the pulse ({:.4} of {:.4})",
            walls[16],
            walls[6]
        );
    }

    /// Mode-1 internal seiche in a closed basin `L` = 5 km long, `H` = 20 m
    /// deep, linearly stratified with N = 0.05 s⁻¹: isopycnals displaced by
    /// `a cos(πx/L) sin(π(z + H)/H)` (a = 0.5 m) at rest. P2 on 500 m,
    /// `levels` uniform levels, 480 s steps (65 per period), `f = 0`, no
    /// mixing.
    ///
    /// The hydrostatic mode-1 speed under a free surface solves
    /// `tan(NH/c₁) = N c₁/g` (`w'' + (N/c)² w = 0`, `w(−H) = 0`,
    /// `w' = (g/c²) w` at the surface): `c₁` = 0.318 m/s, 0.05 % below the
    /// rigid lid's `NH/π`, and the period `T = 2L/c₁` = 8.7 h.
    ///
    /// Runs 1.1 T and returns the measured period over `T` and the amplitude
    /// lost in that period, from the displacement's projection on the
    /// vertical mode at the wall column `x = 0`: the period from its zero
    /// crossings at T/4 and 3T/4, the loss from its crest at T.
    fn internal_seiche(levels: usize, scheme: crate::solver::rhs::VerticalAdvection) -> (f64, f64) {
        internal_seiche_on(SigmaGrid::new(levels, UniformStretching), scheme)
    }

    /// [`internal_seiche`] on the levels `sigma`.
    fn internal_seiche_on(
        sigma: SigmaGrid,
        scheme: crate::solver::rhs::VerticalAdvection,
    ) -> (f64, f64) {
        let (length, depth, amplitude, n_buoyancy) = (5e3, 20.0, 0.5, 0.05);
        let mut c1 = n_buoyancy * depth / std::f64::consts::PI;
        for _ in 0..20 {
            c1 = n_buoyancy * depth / (std::f64::consts::PI + (n_buoyancy * c1 / G).atan());
        }
        let period = 2.0 * length / c1;
        let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, length, 0.0, 500.0, 10, 1));
        let ops = Arc::new(DGOperators2D::new(2));
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
        .build();
        let physics = Hydrostatic3D::new(
            mesh.clone(),
            ops.clone(),
            geom,
            Arc::new(sigma.clone()),
            bathymetry,
            Arc::new(CoriolisSource2D::f_plane(0.0)),
            LinearEOS::default(),
            ConstantMixing::new(0.0, 0.0),
            swe,
            no_stress(),
            G,
            RHO0,
        )
        .with_vertical_advection(scheme);
        let eos = LinearEOS::default();
        // N² = gα dT/dz
        let gradient = n_buoyancy.powi(2) / (G * eos.alpha);
        let background = |z: f64| eos.t0 + gradient * (z + 0.5 * depth);
        let (nn, nl) = (ops.n_nodes, sigma.n_levels());
        let sigma_rho = sigma.sigma_rho();
        let shape: Vec<f64> = sigma_rho
            .iter()
            .map(|s| (std::f64::consts::PI * (s + 1.0)).sin())
            .collect();
        let mut state = Solution3D::new(mesh.n_elements, nn, nl);
        state.salt.fill(eos.s0);
        for idx in 0..mesh.n_elements * nn {
            let (k, i) = (idx / nn, idx % nn);
            let [x, _] =
                mesh.reference_to_physical(ElementIndex::new(k), ops.nodes_r[i], ops.nodes_s[i]);
            let crest = amplitude * (std::f64::consts::PI * x / length).cos();
            for (l, &s) in sigma_rho.iter().enumerate() {
                state.temp[idx * nl + l] = background(s * depth - crest * shape[l]);
            }
        }
        // Node 0 of element 0 is the corner (0, 0)
        let d_sigma = sigma.d_sigma();
        let norm: f64 = shape.iter().zip(d_sigma).map(|(m, ds)| m * m * ds).sum();
        let mode = |s: &Solution3D| {
            let (eta, d) = (s.eta.data[0], s.eta.data[0] + depth);
            (0..nl)
                .map(|l| {
                    let anomaly = s.temp[l] - background(eta + sigma_rho[l] * d);
                    -anomaly / gradient * shape[l] * d_sigma[l]
                })
                .sum::<f64>()
                / norm
        };
        let dt = 480.0;
        let mut integrator = ModeSplitIntegrator::new();
        let mut history = vec![(0.0, mode(&state))];
        for n in 0..(1.1 * period / dt).round() as usize {
            physics.update_density(&mut state);
            integrator.step(&mut state, &physics, dt, n as f64 * dt);
            physics.post_process(&mut state);
            history.push(((n + 1) as f64 * dt, mode(&state)));
        }
        let crossings: Vec<f64> = history
            .windows(2)
            .filter(|w| (w[0].1 >= 0.0) != (w[1].1 >= 0.0))
            .map(|w| w[0].0 + (w[1].0 - w[0].0) * w[0].1 / (w[0].1 - w[1].1))
            .collect();
        assert_eq!(
            crossings.len(),
            2,
            "test regime: zero crossings {crossings:?}"
        );
        // The crest, from the parabola through the highest sample after the
        // trough and its neighbours
        let highest = (history.len() / 2..history.len() - 1)
            .max_by(|&a, &b| history[a].1.total_cmp(&history[b].1))
            .expect("samples");
        let [y0, y1, y2] = [highest - 1, highest, highest + 1].map(|n| history[n].1);
        let crest = y1 - (y2 - y0).powi(2) / (8.0 * (y2 - 2.0 * y1 + y0));
        let measured = 2.0 * (crossings[1] - crossings[0]);
        (measured / period, 1.0 - crest / history[0].1)
    }

    /// TODO P4.6 gate: the mode-1 internal-wave speed. The internal seiche
    /// ([`internal_seiche`]) oscillates at the free-surface hydrostatic
    /// period `2L/c₁`: +0.04 % on 20 levels, +0.37 % on 10 (−0.04 % on 40,
    /// −0.06 % on 80). The differences fall 3.9× and 4.4× per halving of the
    /// spacing, second order, down to a floor that is the wave's own
    /// nonlinearity (a/H = 0.025): on 80 levels −0.056 % at a = 0.5 m,
    /// −0.011 % at 0.25 m, +0.004 % at 0.05 m. Smaller steps add a small
    /// splitting error at any amplitude (−0.04 % at 60 s steps, 524 per
    /// period), and 20 elements instead of 10 change it by < 0.01 %. The
    /// seiche loses 0.13 % of its amplitude per period.
    ///
    /// The free surface matters at stronger stratification: at N = 0.1 and
    /// 0.2 s⁻¹ the rigid lid's `NH/π` is 0.21 % and 0.83 % too fast, and the
    /// model's period converges to 0.19 % and 0.7–0.8 % above `2πL/(NH)`.
    /// P1 on the same mesh is 0.5 % slow and loses 1.3 % per period.
    #[test]
    fn a_mode_1_internal_seiche_has_the_internal_wave_speed() {
        use crate::solver::rhs::VerticalAdvection::Akima;
        let (fine, loss) = internal_seiche(20, Akima);
        assert!(
            (fine - 1.0).abs() < 1.5e-3,
            "the internal seiche's period is {:+.3} % off 2L/c₁",
            100.0 * (fine - 1.0)
        );
        assert!(
            (0.0..5e-3).contains(&loss),
            "the internal seiche loses {:.3} % of its amplitude per period",
            100.0 * loss
        );
        let (coarse, _) = internal_seiche(10, Akima);
        assert!(
            coarse - 1.0 > 3.0 * (fine - 1.0).abs(),
            "period error {:+.3} % on 10 levels, {:+.3} % on 20: not second order",
            100.0 * (coarse - 1.0),
            100.0 * (fine - 1.0)
        );
    }

    /// TODO P4.5 gate: the Akima vertical advection neither slows nor damps
    /// internal waves as first-order upwind does. On 10 levels the internal
    /// seiche is 1.06 % slow with upwind (0.40 % on 20 levels, 0.15 % on 40:
    /// about first order in the spacing) and loses 1.2 % of its amplitude
    /// per period; with Akima 0.37 % and 0.14 %.
    #[test]
    fn akima_vertical_advection_keeps_internal_waves() {
        use crate::solver::rhs::VerticalAdvection::{Akima, Upwind};
        let (akima, akima_loss) = internal_seiche(10, Akima);
        let (upwind, upwind_loss) = internal_seiche(10, Upwind);
        assert!(
            upwind - 1.0 > 2.0 * (akima - 1.0).abs(),
            "period error: upwind {:+.3} %, Akima {:+.3} %",
            100.0 * (upwind - 1.0),
            100.0 * (akima - 1.0)
        );
        assert!(
            upwind_loss > 5.0 * akima_loss.abs(),
            "amplitude loss per period: upwind {:.3} %, Akima {:.3} %",
            100.0 * upwind_loss,
            100.0 * akima_loss
        );
    }

    /// What [`lock_exchange`] measures.
    struct LockExchange {
        /// The Froude numbers `U/√(g′H)` of the dense and the light front
        /// over the window (after the collapse of the lock).
        froude: [f64; 2],
        /// How far T left its initial range at the end (°C).
        t_excess: f64,
        /// The largest horizontal viscosity of each element at the end
        /// (m²/s).
        largest_viscosity: Vec<f64>,
    }

    /// The lock exchange (Ilıcak et al. 2012, scaled): an 8 km channel,
    /// 20 m deep, 5 °C colder left of the middle, at rest (`g′ = gαΔT` =
    /// 8.3e-3 m/s², `√(g′H)` = 0.41 m/s), with the horizontal Kuzmin tracer
    /// limiter, vertical viscosity 1e-4 m²/s and `f = 0`. At `order` on `n_x`
    /// elements along the channel and `levels` levels, with the horizontal
    /// `viscosity` of the shear, the tracers' `vertical_advection` and steps
    /// of `dt` (s), measured over the times `window` (s; see
    /// [`LockExchange`]).
    ///
    /// The fronts are where the bed (surface) layer's dense (light) water
    /// ends: the middle plus its length there, from its fraction integrated
    /// along the channel.
    fn lock_exchange(
        order: usize,
        n_x: usize,
        levels: usize,
        viscosity: crate::solver::rhs::HorizontalViscosity3D,
        vertical_advection: crate::solver::rhs::VerticalAdvection,
        dt: f64,
        window: [f64; 2],
    ) -> LockExchange {
        use crate::solver::rhs::{ViscosityScratch3D, largest_horizontal_viscosity_3d};
        use crate::solver::{TracerLimiter3DConfig, TracerLimiterType3D};
        let (length, width, depth, delta_t) = (8e3, 500.0, 20.0, 5.0);
        let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, length, 0.0, width, n_x, 1));
        let ops = Arc::new(DGOperators2D::new(order));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -depth));
        let sigma = SigmaGrid::new(levels, UniformStretching);
        let swe = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::default(),
        )
        .with_bathymetry(bathymetry.clone())
        .build();
        let physics = Hydrostatic3D::new(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            Arc::new(sigma.clone()),
            bathymetry,
            Arc::new(CoriolisSource2D::f_plane(0.0)),
            LinearEOS::default(),
            ConstantMixing::new(1e-4, 0.0),
            swe,
            no_stress(),
            G,
            RHO0,
        )
        .with_horizontal_viscosity(viscosity.background)
        .with_smagorinsky_viscosity(viscosity.smagorinsky)
        .with_vertical_advection(vertical_advection)
        .with_tracer_limiter(TracerLimiter3DConfig {
            limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.0 },
            ..TracerLimiter3DConfig::default()
        });
        let eos = LinearEOS::default();
        let speed_scale = (G * eos.alpha * delta_t * depth).sqrt();
        let (nn, nl) = (ops.n_nodes, sigma.n_levels());
        let x_of = |idx: usize| {
            let (k, i) = (idx / nn, idx % nn);
            mesh.reference_to_physical(ElementIndex::new(k), ops.nodes_r[i], ops.nodes_s[i])[0]
        };
        let (cold, warm) = (eos.t0 - 0.5 * delta_t, eos.t0 + 0.5 * delta_t);
        let mut state = Solution3D::new(mesh.n_elements, nn, nl);
        state.salt.fill(eos.s0);
        for idx in 0..mesh.n_elements * nn {
            let t = if x_of(idx) < 0.5 * length { cold } else { warm };
            state.temp[idx * nl..(idx + 1) * nl].fill(t);
        }
        // Element k lies right of the middle if its centre does
        let right = |k: usize| x_of(k * nn) + x_of(k * nn + nn - 1) > length;
        let fronts = |s: &Solution3D| {
            let dense = |t: f64| ((warm - t) / delta_t).clamp(0.0, 1.0);
            let mut bed = DGSolution2D::new(mesh.n_elements, nn);
            let mut surface = DGSolution2D::new(mesh.n_elements, nn);
            for idx in 0..mesh.n_elements * nn {
                if right(idx / nn) {
                    bed.data[idx] = dense(s.temp[idx * nl]);
                } else {
                    surface.data[idx] = 1.0 - dense(s.temp[idx * nl + nl - 1]);
                }
            }
            [
                0.5 * length + bed.integrate(&ops, &geom) / width,
                0.5 * length - surface.integrate(&ops, &geom) / width,
            ]
        };

        let mut integrator = ModeSplitIntegrator::new();
        let mut positions = Vec::new();
        let [first, last] = window.map(|t| (t / dt).round() as usize);
        for n in 0..last {
            physics.update_density(&mut state);
            integrator.step(&mut state, &physics, dt, n as f64 * dt);
            physics.post_process(&mut state);
            if n + 1 == first || n + 1 == last {
                positions.push(fronts(&state));
            }
        }
        let t_excess = max_or_nan(
            state
                .temp
                .iter()
                .map(|&t| (t - cold).max(warm - t) - delta_t),
        );
        // The largest horizontal viscosity of each element at the end
        let mut largest_viscosity = vec![0.0; mesh.n_elements];
        largest_horizontal_viscosity_3d(
            &mut largest_viscosity,
            &state,
            viscosity,
            &mesh,
            &ops,
            &geom,
            &physics.bathymetry,
            &sigma,
            &physics.boundaries,
            physics.min_column_depth,
            &mut ViscosityScratch3D::new(mesh.n_elements, &ops),
        );
        let span = (last - first) as f64 * dt;
        let froude = [0, 1].map(|f| (positions[1][f] - positions[0][f]).abs() / span / speed_scale);
        LockExchange {
            froude,
            t_excess,
            largest_viscosity,
        }
    }

    /// TODO P4.6 gate: the lock exchange ([`lock_exchange`]) at P1 on
    /// 250 m, ten levels, 30 s steps, without horizontal viscosity, fronts
    /// over hours 1–3.
    ///
    /// The dense water runs right along the bed and the light water left
    /// along the surface, each at the Froude number 0.483 (0.475 at 125 m on
    /// 20 levels, 0.473 at 62 m), a little below Benjamin's (1968) ½ for an
    /// energy-conserving current, as dissipative currents in the laboratory
    /// and in hydrostatic models are. At the middle the two layers move at
    /// ±0.20 m/s, Benjamin's ½√(g′H). The interface between them thickens
    /// to ≈ 8 m (where the shear's Richardson number reaches ≈ ¼), so the
    /// exchange transport is only 0.40 of `(H/2)·½√(g′H)`. The limiter keeps
    /// T inside its initial range.
    #[test]
    fn a_lock_exchange_runs_at_the_gravity_current_speed() {
        use crate::solver::rhs::VerticalAdvection::Akima;
        let LockExchange {
            froude: [dense, light],
            t_excess,
            ..
        } = lock_exchange(
            1,
            32,
            10,
            Default::default(),
            Akima,
            30.0,
            [3600.0, 10800.0],
        );
        assert!(
            t_excess < 1e-9,
            "T left its initial range by {t_excess:.3e} °C"
        );
        // Measured 0.483 for both
        for (front, froude) in [("dense", dense), ("light", light)] {
            assert!(
                (0.44..0.5).contains(&froude),
                "the {front} front runs at Fr = {froude:.4}"
            );
        }
        assert!(
            (dense - light).abs() < 0.01,
            "the fronts run at Fr = {dense:.4} and {light:.4}"
        );
    }

    /// TODO P4.5 gate: the lock exchange ([`lock_exchange`]) at P2 needs
    /// horizontal viscosity, and runs with it. On 250 m, ten levels, 40 s
    /// steps, ν = 10 m²/s and TVD vertical advection the fronts run at
    /// Fr = 0.500 (dense) and 0.494 (light) over hours 1–2 (0.499 and 0.494
    /// over hours 1–3, the same at 30 s steps; P1 without viscosity 0.483),
    /// and T stays in its range to 2e-12 °C. Before hour 1 the fronts are
    /// still accelerating (0.48 over 0.5–1.5 h).
    ///
    /// Without horizontal viscosity the interface's shear instability grows
    /// at the grid scale, fastest in a hydrostatic model (its growth rate
    /// rises with the wavenumber, `≈ kΔU/2`), and P2 has too little numerical
    /// dissipation to stop it: NaN after ≈ 1500 s, at 10 and 2 s steps
    /// alike; at ν = 5 the mid-depth interface at an element vertex grows
    /// from 0.3 to 4.6 m/s within 450 s (≈ 6e-3 s⁻¹, as `kΔU/2` at the node
    /// spacing) after an hour. The smallest ν that holds shrinks with the
    /// spacing: between 5 and 10 m²/s at 500 m, 5 and 7 at 250 m, 2.5 and 5
    /// at 125 m, at most 2.5 at 62 m (fronts 0.491 and 0.490). In grid
    /// Reynolds numbers `ΔU·Δx/ν` (ΔU = 0.4 m/s, Δx = element size / N) the
    /// bound is 7–10 at 250 m and below, 10–20 at 500 m.
    ///
    /// With unlimited Akima vertical advection the same runs pass, but at
    /// 125 m and ν = 10 the bed layers undershoot by 0.23 °C next to the
    /// sharp interface, in their element means. TVD and the default limited
    /// Akima keep it to 4e-12 °C ([`a_sharp_lock_exchange_interface_stays_in_range`]).
    #[test]
    fn a_p2_lock_exchange_runs_with_horizontal_viscosity() {
        use crate::solver::rhs::{HorizontalViscosity3D, VerticalAdvection::Tvd};
        let LockExchange {
            froude: [dense, light],
            t_excess,
            ..
        } = lock_exchange(
            2,
            32,
            10,
            HorizontalViscosity3D::constant(10.0),
            Tvd,
            40.0,
            [3600.0, 7200.0],
        );
        assert!(
            t_excess < 1e-9,
            "T left its initial range by {t_excess:.3e} °C"
        );
        // Measured 0.500 and 0.494
        for (front, froude) in [("dense", dense), ("light", light)] {
            assert!(
                (0.46..0.52).contains(&froude),
                "the {front} front runs at Fr = {froude:.4}"
            );
        }
        assert!(
            (dense - light).abs() < 0.01,
            "the fronts run at Fr = {dense:.4} and {light:.4}"
        );
    }

    /// TODO P4.5 gate: Smagorinsky's viscosity holds the P2 lock exchange
    /// ([`lock_exchange`]) without a constant one, and only where the flow
    /// needs it. On 250 m, ten levels, 40 s steps, `C_s` = 0.7 and the
    /// default (limited Akima) vertical advection the fronts run at
    /// Fr = 0.504 (dense) and 0.499 (light) over hours 1–2, T stays in its
    /// range to 2e-12 °C, and after 2 h ν reaches 19 m²/s at the fronts' heads
    /// and ≈ 5 m²/s along the interface between them (the constant ν that
    /// holds it is 5–7 m²/s everywhere), while the water at rest beyond the
    /// fronts gets ≤ 0.1 m²/s (the median element 0.13 m²/s).
    ///
    /// `C_s` = 0.5 also holds it (Fr 0.503 and 0.497; ν ≤ 13), 0.4 does not
    /// (NaN within 2 h). On 125 m `C_s` = 0.4 suffices: the strain at the
    /// node spacing grows as the spacing shrinks, so ν falls with the grid
    /// as the constant ν had to by hand (0.5: ν ≈ 1–3 m²/s along the
    /// interface, ≤ 10 at the heads; Fr 0.505 and 0.505).
    #[test]
    fn a_p2_lock_exchange_runs_with_smagorinsky_viscosity() {
        use crate::solver::rhs::{HorizontalViscosity3D, VerticalAdvection};
        let LockExchange {
            froude: [dense, light],
            t_excess,
            largest_viscosity,
        } = lock_exchange(
            2,
            32,
            10,
            HorizontalViscosity3D::smagorinsky(0.7),
            VerticalAdvection::default(),
            40.0,
            [3600.0, 7200.0],
        );
        assert!(
            t_excess < 1e-9,
            "T left its initial range by {t_excess:.3e} °C"
        );
        // Measured 0.504 and 0.499
        for (front, froude) in [("dense", dense), ("light", light)] {
            assert!(
                (0.46..0.52).contains(&froude),
                "the {front} front runs at Fr = {froude:.4}"
            );
        }
        assert!(
            (dense - light).abs() < 0.01,
            "the fronts run at Fr = {dense:.4} and {light:.4}"
        );
        // Measured 18.9 at the heads, ≤ 0.1 within 2 km of the ends
        let n = largest_viscosity.len();
        let largest = max_or_nan(largest_viscosity.iter().copied());
        let beyond = max_or_nan(
            largest_viscosity[..n / 4]
                .iter()
                .chain(&largest_viscosity[3 * n / 4..])
                .copied(),
        );
        assert!(largest > 10.0, "ν reaches only {largest:.3} m²/s");
        assert!(
            beyond < 0.5,
            "ν reaches {beyond:.3} m²/s in the water beyond the fronts"
        );
    }

    /// TODO P4.5 gate: the vertical advection keeps a sharp interface's
    /// layer means in range. The P2 lock exchange ([`lock_exchange`]) on
    /// 125 m, ten levels, ν = 10 m²/s, 40 s steps, over 3 h: with Akima the
    /// bed layers undershoot by 0.23 °C next to the interface, in their
    /// element means (one layer of an element at 7.27 °C in a 7.5–12.5 °C
    /// range), which the horizontal Kuzmin limiter cannot change: it bounds
    /// the nodes around those means. Akima under a TVD limiter (the default,
    /// `LimitedAkima`) keeps T in range to 4e-12 °C (TVD: the same), with the
    /// fronts at Fr 0.502 and 0.501 (Akima: 0.502 and 0.500).
    ///
    /// Akima's undershoot comes and goes with the interface: with ν = 15 it
    /// is 0.016 °C after 2 h, with ν = 5–10 on 125–250 m ≤ 2e-8 °C at 2 h.
    #[test]
    fn a_sharp_lock_exchange_interface_stays_in_range() {
        use crate::solver::rhs::{HorizontalViscosity3D, VerticalAdvection};
        let LockExchange {
            froude: [dense, light],
            t_excess,
            ..
        } = lock_exchange(
            2,
            64,
            10,
            HorizontalViscosity3D::constant(10.0),
            VerticalAdvection::default(),
            40.0,
            [3600.0, 10800.0],
        );
        assert!(
            t_excess < 1e-9,
            "T left its initial range by {t_excess:.3e} °C"
        );
        // Measured 0.502 and 0.501
        for (front, froude) in [("dense", dense), ("light", light)] {
            assert!(
                (0.46..0.52).contains(&froude),
                "the {front} front runs at Fr = {froude:.4}"
            );
        }
    }

    /// TODO P4.5 gate: the horizontal viscosity of the 3D shear in the mode
    /// split. A shear `u = p(σ)·cos(ky)` along a doubly periodic box (P3,
    /// flat, no vertical mixing, no rotation, uniform T/S) is divergence-free
    /// in every layer, so nothing but the viscosity acts on it: it decays as
    /// `e^(−νk²t)` with its profile, and the depth mean stays at rest (the
    /// column sum of the viscosity is zero, so `G` is).
    #[test]
    fn a_shear_mode_decays_at_the_viscous_rate_and_leaves_the_depth_mean() {
        let (length, depth, nu, levels) = (10e3, 20.0, 50.0, 6);
        let mesh = Arc::new(Mesh2D::uniform_periodic(0.0, length, 0.0, length, 2, 8));
        let ops = Arc::new(DGOperators2D::new(3));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -depth));
        let sigma = SigmaGrid::new(levels, UniformStretching);
        let swe = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::default(),
        )
        .with_bathymetry(bathymetry.clone())
        .build();
        let physics = Hydrostatic3D::new(
            mesh.clone(),
            ops.clone(),
            geom,
            Arc::new(sigma.clone()),
            bathymetry,
            Arc::new(CoriolisSource2D::f_plane(0.0)),
            LinearEOS::default(),
            ConstantMixing::new(0.0, 0.0),
            swe,
            no_stress(),
            G,
            RHO0,
        )
        .with_horizontal_viscosity(nu);
        let k = 2.0 * std::f64::consts::PI / length;
        // A profile with zero depth mean
        let profile: Vec<f64> = sigma
            .sigma_rho()
            .iter()
            .map(|s| 0.1 * (2.0 * s + 1.0))
            .collect();
        let (nn, nl) = (ops.n_nodes, levels);
        let eos = LinearEOS::default();
        let mut state = Solution3D::new(mesh.n_elements, nn, nl);
        state.temp.fill(eos.t0);
        state.salt.fill(eos.s0);
        let shape: Vec<f64> = (0..mesh.n_elements * nn)
            .map(|idx| {
                let (e, i) = (idx / nn, idx % nn);
                let [_, y] = mesh.reference_to_physical(
                    ElementIndex::new(e),
                    ops.nodes_r[i],
                    ops.nodes_s[i],
                );
                (k * y).cos()
            })
            .collect();
        for (idx, c) in shape.iter().enumerate() {
            for (l, p) in profile.iter().enumerate() {
                state.u[idx * nl + l] = p * c;
            }
        }
        physics.update_density(&mut state);
        let ops_mass = |idx: usize| physics.geom.node_mass(idx / nn, idx % nn);

        // e-folding 5.1e4 s; the viscous step limit is ≈ 870 s
        let (dt, n_steps) = (500.0, 40);
        let mut integrator = ModeSplitIntegrator::new();
        for n in 0..n_steps {
            integrator.step(&mut state, &physics, dt, n as f64 * dt);
            physics.post_process(&mut state);
        }
        let decay = (-nu * k * k * dt * n_steps as f64).exp();
        // The mode's amplitude, by mass-weighted projection, and the rest
        let (mut projected, mut norm, mut rest) = (0.0, 0.0, 0.0_f64);
        for (idx, c) in shape.iter().enumerate() {
            let mass = ops_mass(idx);
            for (l, p) in profile.iter().enumerate() {
                projected += mass * state.u[idx * nl + l] * p * c;
                norm += mass * (p * c).powi(2);
            }
        }
        let amplitude = projected / norm;
        for (idx, c) in shape.iter().enumerate() {
            for (l, p) in profile.iter().enumerate() {
                rest = max_or_nan([rest, (state.u[idx * nl + l] - amplitude * p * c).abs()]);
            }
        }
        let mean = max_or_nan(
            state
                .ubar
                .data
                .iter()
                .chain(&state.vbar.data)
                .chain(&state.eta.data)
                .chain(&state.v)
                .map(|x| x.abs()),
        );
        // Measured: the amplitude 1.1e-6 off, the rest 5.5e-5 m/s (the
        // nodal cos(ky) is not exactly the discrete mode), the mean 1.7e-12
        assert!(
            (amplitude / decay - 1.0).abs() < 1e-5,
            "the shear decayed to {amplitude:.6} of its amplitude, not {decay:.6}"
        );
        assert!(
            rest < 1e-3 * decay,
            "the shear left its profile by {rest:.3e} m/s"
        );
        assert!(mean < 1e-11, "the depth mean (or v, η) moved by {mean:.3e}");
    }

    /// TODO P4.5 gate: Smagorinsky's viscosity of the shear reaches the
    /// depth mean through `G`, and the momentum is conserved. A shear
    /// `u = p(σ)·cos(ky)` along a doubly periodic box (as in the shear-mode
    /// gate) whose profile is lopsided (`Σ_l Δσ_l |p_l| p_l ≠ 0`): `ν_l`
    /// follows `|p_l|`, so the column sum of the layers' stresses,
    /// `∂_y(D Σ_l Δσ_l ν_l ∂_y u′_l)`, is not zero, and the depth mean, at
    /// rest at first, moves. Its integral `∫ D ū`, the total momentum, stays
    /// zero, and `ū` stays the layers' depth mean. The step bound follows ν:
    /// at rest it is the advective one, and where the viscous one is the
    /// tighter it falls as `1/C_s²`.
    #[test]
    fn smagorinsky_shear_stress_reaches_the_depth_mean_and_conserves_momentum() {
        let (length, depth, levels) = (10e3, 20.0, 6);
        let mesh = Arc::new(Mesh2D::uniform_periodic(0.0, length, 0.0, length, 2, 8));
        let ops = Arc::new(DGOperators2D::new(3));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -depth));
        let sigma = SigmaGrid::new(levels, UniformStretching);
        let swe = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::default(),
        )
        .with_bathymetry(bathymetry.clone())
        .build();
        let mut physics = Hydrostatic3D::new(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            Arc::new(sigma.clone()),
            bathymetry,
            Arc::new(CoriolisSource2D::f_plane(0.0)),
            LinearEOS::default(),
            ConstantMixing::new(0.0, 0.0),
            swe,
            no_stress(),
            G,
            RHO0,
        )
        .with_smagorinsky_viscosity(0.5);
        let k = 2.0 * std::f64::consts::PI / length;
        // p = 0.1·((2σ + 1)² − ⅓): zero depth mean, lopsided
        let raw: Vec<f64> = sigma
            .sigma_rho()
            .iter()
            .map(|s| (2.0 * s + 1.0).powi(2))
            .collect();
        let offset = sigma.depth_average(&raw);
        let profile: Vec<f64> = raw.iter().map(|r| 0.1 * (r - offset)).collect();
        let (nn, nl) = (ops.n_nodes, levels);
        let eos = LinearEOS::default();
        let mut state = Solution3D::new(mesh.n_elements, nn, nl);
        state.temp.fill(eos.t0);
        state.salt.fill(eos.s0);
        for idx in 0..mesh.n_elements * nn {
            let (e, i) = (idx / nn, idx % nn);
            let [_, y] =
                mesh.reference_to_physical(ElementIndex::new(e), ops.nodes_r[i], ops.nodes_s[i]);
            for (l, p) in profile.iter().enumerate() {
                state.u[idx * nl + l] = p * (k * y).cos();
            }
        }
        physics.update_density(&mut state);
        let momentum = |s: &Solution3D| {
            let mut column = DGSolution2D::new(mesh.n_elements, nn);
            for (idx, c) in column.data.iter_mut().enumerate() {
                *c = (s.eta.data[idx] + depth) * s.ubar.data[idx];
            }
            column.integrate(&ops, &geom)
        };

        // The step bound follows Smagorinsky's ν: none at rest, ∝ 1/C_s²
        // where the viscous bound is the tighter one
        let mut rest = state.clone();
        rest.u.fill(0.0);
        let advective = physics.compute_dt(&rest, 1.0);
        let viscous = |physics: &mut Physics, cs: f64| {
            physics.horizontal_viscosity.smagorinsky = cs;
            physics.compute_dt(&state, 1.0)
        };
        let (fifty, hundred) = (viscous(&mut physics, 50.0), viscous(&mut physics, 100.0));
        assert!(fifty < 0.1 * advective, "{fifty} s against {advective} s");
        assert!(
            (fifty / hundred - 4.0).abs() < 1e-10,
            "the bound fell by {} from C_s 50 to 100",
            fifty / hundred
        );
        physics.horizontal_viscosity.smagorinsky = 0.5;

        // ν ≤ 9 m²/s (C_s Δ = 0.5·833 m, |S| ≤ 5e-5 s⁻¹)
        let dt = 500.0;
        let mut integrator = ModeSplitIntegrator::new();
        for n in 0..40 {
            integrator.step(&mut state, &physics, dt, n as f64 * dt);
            physics.post_process(&mut state);
        }
        let moved = max_or_nan(state.ubar.data.iter().map(|u| u.abs()));
        // The momentum that moved, `∫ D|ū|`, and what is left of it in total
        let mut speed = DGSolution2D::new(mesh.n_elements, nn);
        for (idx, s) in speed.data.iter_mut().enumerate() {
            *s = (state.eta.data[idx] + depth) * state.ubar.data[idx].abs();
        }
        let (total, moved_momentum) = (momentum(&state), speed.integrate(&ops, &geom));
        let consistency = max_or_nan((0..mesh.n_elements * nn).map(|idx| {
            (sigma.depth_average(&state.u[idx * nl..(idx + 1) * nl]) - state.ubar.data[idx]).abs()
        }));
        // Measured: ū up to 1.1e-4 m/s; the total 9e-10 of the moved
        // momentum, growing with it step by step (round-off of the BR1
        // sums, ≈ 7e-12 of ∫|G| per step); consistency 3e-18
        assert!(
            moved > 1e-5,
            "test regime: the depth mean moved by {moved:.3e}"
        );
        assert!(
            total.abs() < 1e-8 * moved_momentum,
            "the momentum changed by {:.3e} of what moved",
            total / moved_momentum
        );
        assert!(
            consistency < 1e-12,
            "ū left the layers' depth mean by {consistency:.3e}"
        );
    }

    /// TODO P4.5 gate: the parallel 3D kernels give the serial result bit for
    /// bit. A stratified basin over a sloping bed with every 3D term on
    /// (Coriolis, wind, log-layer bottom drag, Pacanowski–Philander mixing,
    /// Smagorinsky viscosity of the shear, Akima momentum advection, the
    /// horizontal Kuzmin limiter), stepped on one thread and on four: every
    /// field of the state is identical. The kernels write disjoint element
    /// blocks and reduce only with exact operations
    /// (`solver::core::blocks`), so the thread count cannot change the
    /// result.
    #[cfg(feature = "parallel")]
    #[test]
    fn the_3d_step_does_not_depend_on_the_thread_count() {
        use crate::physics::PacanowskiPhilanderMixing;
        use crate::solver::rhs::VerticalAdvection;
        use crate::solver::{TracerLimiter3DConfig, TracerLimiterType3D};
        use crate::vertical::SongHaidvogelStretching;

        let run = || {
            let (length, width) = (6e3, 3e3);
            let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, length, 0.0, width, 6, 3));
            let ops = Arc::new(DGOperators2D::new(2));
            let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
            let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
                -(15.0 + 60.0 * x / length + 10.0 * y / width)
            }));
            let sigma = SigmaGrid::new(8, SongHaidvogelStretching::new(3.0, 0.4, 10.0));
            let swe = PhysicsBuilder::swe_2d(
                mesh.clone(),
                ops.clone(),
                geom.clone(),
                ShallowWater2D::new(G),
                Reflective2D::default(),
            )
            .with_bathymetry(bathymetry.clone())
            .with_formulation(SWEFormulation2D::EntropyStable)
            .with_source(CoriolisSource2D::f_plane(1.2e-4))
            .build();
            let forcing = Forcing {
                surface_stress: [0.2, -0.1],
                bottom_stress: [0.0, 0.0],
                surface_buoyancy_flux: 0.0,
            };
            let physics = Hydrostatic3D::new(
                mesh.clone(),
                ops.clone(),
                geom.clone(),
                Arc::new(sigma.clone()),
                bathymetry.clone(),
                Arc::new(CoriolisSource2D::f_plane(1.2e-4)),
                LinearEOS::default(),
                PacanowskiPhilanderMixing::new(1e-2, 1e-5, 1e-5, G, RHO0),
                swe,
                forcing,
                G,
                RHO0,
            )
            .with_bottom_drag(BottomDrag3D::log_layer(0.005))
            .with_smagorinsky_viscosity(0.5)
            .with_momentum_vertical_advection(VerticalAdvection::Akima)
            .with_tracer_limiter(TracerLimiter3DConfig {
                limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.0 },
                ..TracerLimiter3DConfig::default()
            });
            let (nn, nl) = (ops.n_nodes, sigma.n_levels());
            let eos = LinearEOS::default();
            let mut state = Solution3D::new(mesh.n_elements, nn, nl);
            for idx in 0..mesh.n_elements * nn {
                let (k, i) = (idx / nn, idx % nn);
                let [x, _] = mesh.reference_to_physical(
                    ElementIndex::new(k),
                    ops.nodes_r[i],
                    ops.nodes_s[i],
                );
                let depth = -bathymetry.data[idx];
                state.eta.data[idx] = 0.05 * (x / length - 0.5);
                for (l, &s) in sigma.sigma_rho().iter().enumerate() {
                    let z = s * depth;
                    // A pycnocline that tilts along the basin: a front
                    state.temp[idx * nl + l] =
                        eos.t0 + 3.0 * ((z + 10.0 + 5.0 * x / length) / 4.0).tanh();
                    state.salt[idx * nl + l] = eos.s0;
                }
            }
            let mut integrator = ModeSplitIntegrator::new();
            let dt = 30.0;
            for n in 0..8 {
                physics.update_density(&mut state);
                integrator.step(&mut state, &physics, dt, n as f64 * dt);
                physics.post_process(&mut state);
            }
            state
        };
        let on = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("a thread pool")
                .install(run)
        };
        let (serial, parallel) = (on(1), on(4));
        let speed = max_or_nan(serial.u.iter().zip(&serial.v).map(|(u, v)| u.hypot(*v)));
        assert!(speed > 1e-3, "the flow is too weak to test: {speed:.3e} m/s");
        for (name, a, b) in [
            ("u", &serial.u, &parallel.u),
            ("v", &serial.v, &parallel.v),
            ("w", &serial.w, &parallel.w),
            ("T", &serial.temp, &parallel.temp),
            ("S", &serial.salt, &parallel.salt),
            ("rho", &serial.rho, &parallel.rho),
            ("Av", &serial.eddy_viscosity, &parallel.eddy_viscosity),
            ("eta", &serial.eta.data, &parallel.eta.data),
            ("ubar", &serial.ubar.data, &parallel.ubar.data),
            ("vbar", &serial.vbar.data, &parallel.vbar.data),
        ] {
            let differing = a
                .iter()
                .zip(b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(differing, 0, "{name} differs at {differing} values");
        }
    }
}
