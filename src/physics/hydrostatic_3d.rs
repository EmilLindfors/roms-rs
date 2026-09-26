//! Hydrostatic 3D Physics Module.
//!
//! Handles the full 3D primitive equations with hydrostatic approximation.
//! Contains the 2D barotropic physics module as a sub-component.

use std::sync::{Arc, Mutex};

use crate::boundary::SWEBoundaryCondition2D;
use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::physics::SWEPhysics2D;
use crate::physics::eos::EquationOfState;
use crate::physics::traits::PhysicsModule; // For SWEPhysics2D
use crate::physics::vertical_mixing::{Forcing, VerticalMixing};
use crate::physics::vertical_velocity::compute_vertical_velocity;
use crate::solver::rhs::{
    ExtrapolationTracerBC3D, Rhs3DConfig, TracerBoundaryCondition3D, compute_rhs_3d,
};
use crate::solver::state::Solution3D;
use crate::solver::{TracerLimiter3DConfig, TracerLimiter3DStats, apply_tracer_limiters_3d};
use crate::source::CoriolisSource2D;
use crate::vertical::SigmaGrid;

/// Hydrostatic 3D Physics Module.
///
/// Bundles all configuration and static data for the 3D solver.
pub struct Hydrostatic3D<EOS, MIX, BC>
where
    EOS: EquationOfState,
    MIX: VerticalMixing,
    BC: SWEBoundaryCondition2D,
{
    pub mesh: Arc<Mesh2D>,
    pub ops: Arc<DGOperators2D>,
    pub geom: Arc<GeometricFactors2D>,
    pub sigma: Arc<SigmaGrid>,
    pub bathymetry: Arc<Bathymetry2D>,
    pub coriolis: Arc<CoriolisSource2D>,
    pub eos: EOS,
    pub mixing: MIX,
    pub swe_physics: SWEPhysics2D<BC>, // 2D sub-model
    pub forcing: Forcing,
    pub g: f64,
    pub rho0: f64,
    /// Temperature BC. Not consulted yet: every physical boundary is a closed
    /// wall in 3D (see `TracerBoundaryCondition3D`).
    pub temp_bc: Arc<dyn TracerBoundaryCondition3D>,
    /// Salinity BC. Not consulted yet (see `temp_bc`).
    pub salt_bc: Arc<dyn TracerBoundaryCondition3D>,
    pub tracer_limiter: TracerLimiter3DConfig,
    pub w_scratch: Mutex<Vec<f64>>,
}

impl<EOS, MIX, BC> Hydrostatic3D<EOS, MIX, BC>
where
    EOS: EquationOfState,
    MIX: VerticalMixing,
    SWEPhysics2D<BC>: PhysicsModule<crate::solver::SWESolution2D>,
    BC: Clone + Send + Sync + SWEBoundaryCondition2D,
{
    pub fn new(
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
        sigma: Arc<SigmaGrid>,
        bathymetry: Arc<Bathymetry2D>,
        coriolis: Arc<CoriolisSource2D>,
        eos: EOS,
        mixing: MIX,
        swe_physics: SWEPhysics2D<BC>,
        forcing: Forcing,
        g: f64,
        rho0: f64,
    ) -> Self {
        // Omega lives at the w-points: n_levels + 1 interfaces per column.
        let n_w = mesh.n_elements * ops.n_nodes * (sigma.n_levels() + 1);
        Self {
            mesh,
            ops,
            geom,
            sigma,
            bathymetry,
            coriolis,
            eos,
            mixing,
            swe_physics,
            forcing,
            g,
            rho0,
            temp_bc: Arc::new(ExtrapolationTracerBC3D),
            salt_bc: Arc::new(ExtrapolationTracerBC3D),
            tracer_limiter: TracerLimiter3DConfig::none(),
            w_scratch: Mutex::new(vec![0.0; n_w]),
        }
    }

    /// Override scalar tracer boundary conditions used by the high-level RHS path.
    pub fn with_tracer_boundary_conditions(
        mut self,
        temp_bc: Arc<dyn TracerBoundaryCondition3D>,
        salt_bc: Arc<dyn TracerBoundaryCondition3D>,
    ) -> Self {
        self.temp_bc = temp_bc;
        self.salt_bc = salt_bc;
        self
    }

    /// Set scalar tracer boundary conditions used by the high-level RHS path.
    pub fn set_tracer_boundary_conditions(
        &mut self,
        temp_bc: Arc<dyn TracerBoundaryCondition3D>,
        salt_bc: Arc<dyn TracerBoundaryCondition3D>,
    ) {
        self.temp_bc = temp_bc;
        self.salt_bc = salt_bc;
    }

    /// Override 3D tracer limiter configuration.
    pub fn with_tracer_limiter(mut self, tracer_limiter: TracerLimiter3DConfig) -> Self {
        self.tracer_limiter = tracer_limiter;
        self
    }

    /// Set 3D tracer limiter configuration.
    pub fn set_tracer_limiter(&mut self, tracer_limiter: TracerLimiter3DConfig) {
        self.tracer_limiter = tracer_limiter;
    }

    /// Apply configured 3D tracer limiters and refresh density if tracers changed.
    pub fn apply_tracer_limiters(&self, state: &mut Solution3D) -> TracerLimiter3DStats {
        let stats = apply_tracer_limiters_3d(
            state,
            &self.mesh,
            &self.ops,
            &self.geom,
            &self.bathymetry,
            &self.sigma,
            &self.tracer_limiter,
        );

        if stats.changed() {
            self.eos.update_density(state);
        }

        stats
    }

    /// Compute the 3D Right-Hand Side.
    pub fn compute_rhs_3d(&self, state: &Solution3D, time: f64) -> Solution3D {
        let mut rhs = Solution3D::new(state.n_elements, state.n_nodes, state.n_levels);
        self.compute_rhs_3d_into(state, time, &mut rhs);
        rhs
    }

    /// [`Self::compute_rhs_3d`] into `rhs`, which is overwritten: tendencies of
    /// `u, v, temp, salt`, and zero for the barotropic fields, `w` and `rho`
    /// (the mode splitter supplies the barotropic rates; `w` and `rho` are
    /// diagnostics).
    pub fn compute_rhs_3d_into(&self, state: &Solution3D, _time: f64, rhs: &mut Solution3D) {
        rhs.eta.fill(0.0);
        rhs.ubar.fill(0.0);
        rhs.vbar.fill(0.0);
        rhs.w.fill(0.0);
        rhs.rho.fill(0.0);

        let config = Rhs3DConfig {
            mesh: &self.mesh,
            ops: &self.ops,
            geom: &self.geom,
            bathymetry: &self.bathymetry,
            sigma: &self.sigma,
            coriolis: &self.coriolis,
            eos: &self.eos,
            temp_bc: &*self.temp_bc,
            salt_bc: &*self.salt_bc,
            g: self.g,
            rho0: self.rho0,
        };

        // Compute Vertical Velocity (Diagnostic)
        let mut w_vel = self.w_scratch.lock().expect("Failed to lock w_scratch");
        compute_vertical_velocity(
            &mut *w_vel,
            state,
            &self.mesh,
            &self.ops,
            &self.sigma,
            &self.bathymetry,
            &self.geom,
            self.g,
        );

        compute_rhs_3d(rhs, state, &w_vel, &config);
    }

    /// Update density field based on current temperature and salinity.
    pub fn update_density(&self, state: &mut Solution3D) {
        self.eos.update_density(state);
    }

    /// Compute permissible time step based on 3D CFL condition.
    ///
    /// Limited by horizontal advection speed + gravity wave speed (if explicit)
    /// or just advection (if split).
    /// Since we use mode splitting, the 3D step is limited by:
    /// 1. Internal wave speed (baroclinic modes)
    /// 2. 3D Advection velocity
    pub fn compute_dt(&self, state: &Solution3D, cfl: f64) -> f64 {
        // Simplified estimate: use 2D CFL but scaled for internal waves?
        // Or just use advection speed.
        // For mode splitting, dt_3d can be much larger than dt_2d (barotropic).
        // Typically dt_3d is limited by internal gravity waves c_n ~ sqrt(g' H).

        // For now, let's delegate to SWEPhysics2D compute_dt and multiply by a factor?
        // No, better to compute explicit advection limit.

        // Placeholder: Return a conservative estimate
        // Min(dx / (|u| + c_internal))

        // Let's assume c_internal << c_external
        // So we can take a larger step.
        // For this prototype, return 0.1s or something safe.
        // Or better: use the 2D dt computation but with a larger CFL factor.

        // We really should iterate over elements and find max(|u| + c_bc) / dx.
        // c_bc approx NH * N * H / pi?

        // Let's use the 2D physics dt as a baseline.
        // But 2D physics uses sqrt(gH), which is fast.
        // We want to skip that.

        // Let's iterate elements.
        let mut min_dt = f64::INFINITY;

        for k in 0..self.mesh.n_elements {
            let j_inv = self.geom.det_j_inv[k];
            // length scale h ~ 1/sqrt(J_inv) ?
            // For parallelogram: Area = J. h ~ sqrt(Area).
            let h_len = 1.0 / j_inv.sqrt(); // Approx element size

            // Max velocity in column
            let mut max_vel = 0.0;
            let el = crate::types::ElementIndex::new(k);
            for i in 0..state.n_nodes {
                for l in 0..state.n_levels {
                    let u = state.u_column(el, i)[l];
                    let v = state.v_column(el, i)[l];
                    let vel = (u * u + v * v).sqrt();
                    if vel > max_vel {
                        max_vel = vel;
                    }
                }
            }

            // Internal wave speed approximation: c ~ 2.0 m/s (typical)
            let c_internal = 2.0;
            let wave_speed = max_vel + c_internal;

            if wave_speed > 1e-6 {
                let dt_loc = cfl * h_len / wave_speed / (self.ops.order as f64 + 1.0).powi(2);
                if dt_loc < min_dt {
                    min_dt = dt_loc;
                }
            }
        }

        if min_dt == f64::INFINITY { 1.0 } else { min_dt }
    }

    pub fn post_process(&self, state: &mut Solution3D) {
        self.apply_tracer_limiters(state);

        // Update Vertical Velocity for output
        let mut w_vel = self.w_scratch.lock().expect("Failed to lock w_scratch");

        // Re-compute vertical velocity
        compute_vertical_velocity(
            &mut *w_vel,
            state,
            &self.mesh,
            &self.ops,
            &self.sigma,
            &self.bathymetry,
            &self.geom,
            self.g,
        );

        // state.w is an output field at the layer centres: average the two
        // bounding interface values of each layer.
        let n_levels = state.n_levels;
        for (w_col, omega_col) in state
            .w
            .chunks_exact_mut(n_levels)
            .zip(w_vel.chunks_exact(n_levels + 1))
        {
            for (l, w) in w_col.iter_mut().enumerate() {
                *w = 0.5 * (omega_col[l] + omega_col[l + 1]);
            }
        }

        // Could also apply equation of state update here to ensure rho is fresh
        // self.eos.update_density(state);
    }
}
