//! Mode-split time integration for 3D ocean models.
//!
//! Separates the fast barotropic (2D) mode from the slow baroclinic (3D) mode.
//!
//! # Method
//!
//! The mode splitting technique solves the 2D barotropic equations (free surface and depth-averaged velocity)
//! with a small time step `dt_bt` and the 3D baroclinic equations with a larger time step `dt_bc`.
//!
//! The coupling involves:
//! 1.  Forcing the 2D mode with "slow" terms from the 3D mode (advection, diffusion, baroclinic pressure gradient).
//! 2.  Accumulating time-averaged 2D fields over the barotropic sub-steps.
//! 3.  Replacing the depth-averaged part of the 3D velocity with the time-averaged 2D velocity (reconciliation).
//!
//! # Split Methods
//!
//! - **ROMS**: Predictor-corrector approach.
//! - **Thetis**: G-term coupling (simpler implementation).

use crate::solver::state::Solution3D;
use crate::time::{IntegratorInfo, TimeIntegrator, SSPRK3, Integrable};
use crate::solver::DGSolution2D;
use crate::vertical::SigmaGrid;
use crate::types::ElementIndex;
use crate::physics::vertical_mixing::{VerticalMixing, Forcing};
use crate::physics::vertical_diffusion::apply_vertical_diffusion;

/// Method for coupling barotropic and baroclinic modes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SplitMethod {
    /// Thetis-style G-term coupling.
    ///
    /// Computes slow terms once per 3D step and applies them as forcing to the 2D mode.
    GTerm,
}

/// Mode-split time integrator.
pub struct ModeSplitIntegrator {
    /// Number of barotropic steps per baroclinic step.
    pub n_bt_steps: usize,
    /// Coupling method.
    pub split_method: SplitMethod,
    /// Internal 3D integrator (usually SSP-RK3).
    bc_integrator: SSPRK3,
    /// Temporary storage for 2D RHS forcing (G-terms).
    g_term_u: DGSolution2D,
    g_term_v: DGSolution2D,
}

impl ModeSplitIntegrator {
    /// Create a new mode-split integrator.
    pub fn new(n_bt_steps: usize, template_2d: &DGSolution2D) -> Self {
        Self {
            n_bt_steps,
            split_method: SplitMethod::GTerm,
            bc_integrator: SSPRK3,
            g_term_u: template_2d.zeros_like(),
            g_term_v: template_2d.zeros_like(),
        }
    }

    /// Perform one baroclinic (3D) time step.
    ///
    /// # Arguments
    /// * `state` - Full 3D solution (modified in place).
    /// * `sigma` - Vertical grid for depth averaging.
    /// * `dt` - Baroclinic time step (large step).
    /// * `t` - Current time.
    /// * `forcing` - Surface and bottom forcing.
    /// * `mixing` - Vertical mixing closure.
    /// * `rhs_2d` - Function computing 2D RHS: f(eta, ubar, vbar, t) -> (d_eta, d_ubar, d_vbar).
    /// * `rhs_3d` - Function computing 3D RHS: f(state_3d, t) -> state_3d.
    pub fn step<F2, F3>(
        &mut self,
        state: &mut Solution3D,
        sigma: &SigmaGrid,
        dt: f64,
        t: f64,
        forcing: &Forcing,
        mixing: &dyn VerticalMixing,
        mut rhs_2d: F2,
        mut rhs_3d: F3,
    ) where
        F2: FnMut(&DGSolution2D, &DGSolution2D, &DGSolution2D, f64) -> (DGSolution2D, DGSolution2D, DGSolution2D),
        F3: FnMut(&Solution3D, f64) -> Solution3D,
    {
        // Extract fields to avoid capturing self
        let bc_integrator = &self.bc_integrator;
        let g_term_u = &mut self.g_term_u;
        let g_term_v = &mut self.g_term_v;
        let n_bt_steps = self.n_bt_steps;
        
        // 1. Explicit 3D Step with Mode Splitting
        bc_integrator.step(state, dt, t, |s, t_| {
            // 1. Calculate 3D RHS (Slow terms)
            let rhs_slow = rhs_3d(s, t_);
            
            // 2. Calculate depth-averaged slow forcing (G-terms)
            Self::compute_depth_average_field_static(sigma, &rhs_slow.u, g_term_u);
            Self::compute_depth_average_field_static(sigma, &rhs_slow.v, g_term_v);
            
            // 3. Sub-cycle 2D equations with forcing G
            // We need to clone the 2D state to advance it
            let mut eta_sub = s.eta.clone();
            let mut ubar_sub = s.ubar.clone();
            let mut vbar_sub = s.vbar.clone();
            
            let dt_bt = dt / n_bt_steps as f64;
            let mut t_bt = t_;
            
            // Accumulators for time averaging
            let mut eta_sum = s.eta.zeros_like();
            let mut ubar_sum = s.ubar.zeros_like();
            let mut vbar_sum = s.vbar.zeros_like();
            
            for _ in 0..n_bt_steps {
                let (d_eta, d_ubar, d_vbar) = rhs_2d(&eta_sub, &ubar_sub, &vbar_sub, t_bt);
                
                // Update state (Forward Euler for now for sub-steps)
                eta_sub.axpy(dt_bt, &d_eta);
                
                ubar_sub.axpy(dt_bt, &d_ubar);
                ubar_sub.axpy(dt_bt, g_term_u); // Add baroclinic forcing
                
                vbar_sub.axpy(dt_bt, &d_vbar);
                vbar_sub.axpy(dt_bt, g_term_v); // Add baroclinic forcing
                
                t_bt += dt_bt;
                
                // Accumulate for time averaging (simple arithmetic mean)
                eta_sum.axpy(1.0, &eta_sub);
                ubar_sum.axpy(1.0, &ubar_sub);
                vbar_sum.axpy(1.0, &vbar_sub);
            }
            
            // Compute time average
            let scale = 1.0 / n_bt_steps as f64;
            eta_sum.scale(scale);
            ubar_sum.scale(scale);
            vbar_sum.scale(scale);
            
            // 4. Construct total RHS
            let mut rhs_total = rhs_slow.clone();
            
            // Calculate effective 2D rates: (avg_new - old) / dt
            let mut d_ubar_eff = ubar_sum.clone();
            d_ubar_eff.axpy(-1.0, &s.ubar);
            d_ubar_eff.scale(1.0 / dt);
            
            let mut d_vbar_eff = vbar_sum.clone();
            d_vbar_eff.axpy(-1.0, &s.vbar);
            d_vbar_eff.scale(1.0 / dt);
            
            let mut d_eta_eff = eta_sum.clone();
            d_eta_eff.axpy(-1.0, &s.eta);
            d_eta_eff.scale(1.0 / dt);
            
            // Apply correction to 3D fields:
            // rhs_total = rhs_slow - G + d_bar_eff
            Self::apply_barotropic_correction_static(sigma, &mut rhs_total.u, g_term_u, &d_ubar_eff);
            Self::apply_barotropic_correction_static(sigma, &mut rhs_total.v, g_term_v, &d_vbar_eff);
            
            // Update 2D fields in rhs_total
            rhs_total.eta = d_eta_eff;
            rhs_total.ubar = d_ubar_eff;
            rhs_total.vbar = d_vbar_eff;
            
            rhs_total
        });
        
        // 2. Implicit Vertical Diffusion
        // This updates u, v, temp, salt in place
        apply_vertical_diffusion(state, sigma, dt, mixing, forcing);
        
        // 3. Reconcile 2D/3D momentum (again)
        // Diffusion changes vertical profile and thus depth average (due to stress).
        // We must ensure that depth average matches state.ubar (which came from 2D mode).
        // 2D mode included stress.
        // 3D diffusion included stress.
        // Ideally they match. But due to numerics, they might drift.
        // We enforce: u_final = u_diffused - ubar_diffused + ubar_state
        
        // Compute ubar_diffused
        Self::compute_depth_average_field_static(sigma, &state.u, g_term_u); // reuse buffer
        Self::compute_depth_average_field_static(sigma, &state.v, g_term_v);
        
        // Apply correction: u += (ubar_state - ubar_diffused)
        // Correction term = ubar_state - ubar_diffused
        // My function `apply_barotropic_correction_static` does: u += (d_eff - G)
        // We want u += (ubar_state - ubar_diffused)
        // So pass d_eff = ubar_state, G = ubar_diffused.
        
        Self::apply_barotropic_correction_static(sigma, &mut state.u, g_term_u, &state.ubar);
        Self::apply_barotropic_correction_static(sigma, &mut state.v, g_term_v, &state.vbar);
    }

    fn compute_depth_average_field_static(sigma: &SigmaGrid, u_3d: &[f64], u_2d: &mut DGSolution2D) {
        let n_levels = sigma.n_levels();
        for k in 0..u_2d.n_elements {
            for i in 0..u_2d.n_nodes {
                let col = Solution3D::get_column(u_3d, u_2d.n_nodes, n_levels, ElementIndex::new(k), i);
                u_2d.data[k * u_2d.n_nodes + i] = sigma.depth_average(col);
            }
        }
    }

    fn apply_barotropic_correction_static(sigma: &SigmaGrid, u_3d: &mut [f64], g_term: &DGSolution2D, d_bar_eff: &DGSolution2D) {
        let n_levels = sigma.n_levels();
        for k in 0..g_term.n_elements {
            for i in 0..g_term.n_nodes {
                let correction = d_bar_eff.data[k * g_term.n_nodes + i] - g_term.data[k * g_term.n_nodes + i];
                let col = Solution3D::get_column_mut(u_3d, g_term.n_nodes, n_levels, ElementIndex::new(k), i);
                for l in 0..n_levels {
                    col[l] += correction;
                }
            }
        }
    }
}

impl IntegratorInfo for ModeSplitIntegrator {
    fn name(&self) -> &'static str {
        "mode-split-ssp-rk3"
    }

    fn order(&self) -> usize {
        3
    }

    fn n_stages(&self) -> usize {
        3
    }

    fn is_ssp(&self) -> bool {
        true
    }

    fn stage_times(&self, dt: f64) -> Vec<f64> {
        self.bc_integrator.stage_times(dt)
    }
}
