//! Mode-split time integration for 3D ocean models.
//!
//! Separates the fast barotropic (2D) mode from the slow baroclinic (3D) mode.
//!
//! # Method
//!
//! One baroclinic step `tⁿ → tⁿ⁺¹ = tⁿ + Δt` (Shchepetkin & McWilliams 2005, with
//! SSP-RK3 in place of their forward-backward barotropic stepping):
//!
//! 1. **Slow forcing.** The 3D RHS `R₃D` at `tⁿ`; its depth integral
//!    `G = Dⁿ·⟨R₃D,u⟩` is a transport tendency, frozen over the step.
//! 2. **One barotropic pass.** The 2D shallow-water module steps the transport
//!    `(h, hu, hv)` plus `G` with SSP-RK3 substeps of `Δt/n_bt` over
//!    `[tⁿ, tⁿ + M*·Δt/n_bt]`, `M* ≈ 1.3·n_bt`. `n_bt` follows from the 2D CFL
//!    every step. The substep states are averaged with the power-law weights of
//!    [`BarotropicFilter`], centred on `tⁿ⁺¹`; the average `(η̄, D̄ū)` is the new
//!    barotropic state, with `ū = D̄ū / D̄`.
//! 3. **3D stages.** SSP-RK3 on the 3D fields. In every stage the depth mean of
//!    the velocity tendency is replaced by the constant rate `(ūⁿ⁺¹ − ūⁿ)/Δt`,
//!    and `η, ū, v̄` get the same constant rates. SSP-RK3 reproduces a
//!    constant-rate solution exactly, so each stage sees the barotropic state
//!    linearly interpolated to its stage time, and the depth mean of `u` stays
//!    equal to `ū`. Stage 1 reuses the `R₃D` of step 1.
//! 4. **Implicit vertical diffusion**, then the depth mean of `u` is reset to `ū`.
//!
//! The 2D positivity limiter and wet/dry treatment run after every barotropic
//! RK stage (and on the filtered state, since the filter has small negative
//! weights), and stiff 2D damping (implicit friction) is applied per stage, as
//! in [`crate::simulation::Simulation`].
//!
//! # Accuracy and known gaps (TODO P4.1)
//!
//! - The filter adds almost no damping to resolved barotropic motion
//!   (≈ 0.03 % amplitude per period at 50 baroclinic steps per period; see
//!   [`BarotropicFilter`]).
//! - `G` is frozen at `tⁿ`, so the slow coupling is first order in `Δt`.
//!   AB3 extrapolation to `tⁿ⁺½` comes with the `rufrc` construction.
//! - `G` still contains the 3D advection and Coriolis terms that the 2D module
//!   computes too (double count), and no surface/bottom stress; both are fixed
//!   by `G = ∫R₃D dz + (τ_s − τ_b)/ρ₀ − R₂D(q̄ⁿ)`.
//! - `R₃D` is still evaluated into a freshly allocated state (TODO P4.5).

use crate::mesh::data::Bathymetry2D;
use crate::physics::PhysicsModule;
use crate::physics::vertical_diffusion::apply_vertical_diffusion;
use crate::physics::vertical_mixing::{Forcing, VerticalMixing};
use crate::solver::state::Solution3D;
use crate::solver::state::{SWE_VAR_H, SWE_VAR_HU, SWE_VAR_HV};
use crate::solver::{DGSolution2D, SWESolution2D};
use crate::time::{Integrable, IntegratorInfo, SSPRK3, StageWorkspace, TimeIntegrator};
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// Fewest barotropic substeps per baroclinic step. Below four the filter
/// weights cannot be centred on `tⁿ⁺¹`.
pub const MIN_BAROTROPIC_SUBSTEPS: usize = 4;

/// Method for coupling barotropic and baroclinic modes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SplitMethod {
    /// Depth-integrated 3D tendency as forcing of one filtered barotropic pass.
    GTerm,
}

/// Primary weights of the barotropic time filter.
///
/// The power-law shape of Shchepetkin & McWilliams (2005, §2.3; ROMS
/// `set_weights.F`, `POWER_LAW`):
///
/// ```text
///     A(τ) = τ^p (1 − τ^q) − r·τ,   p = 2, q = 4, r = 0.284,   τ = m·s
/// ```
///
/// for substeps `m = 1 … M*`, where `M*` is the last positive weight. The weights
/// are normalised to sum to one, and the scale `s` is iterated until their
/// centroid is exactly `n_bt` (time `tⁿ⁺¹`). The window is `M* ≈ 1.3·n_bt`
/// substeps long.
///
/// The `−r·τ` term makes the first few weights slightly negative and brings the
/// second moment about `tⁿ⁺¹` close to zero. For a wave of frequency `ω` the
/// filter's amplitude response is `|Σ w_m e^{iω(t_m − tⁿ⁺¹)}| ≈ 1 − ω²μ₂/2`, and
/// the barotropic state is restarted from the filtered value every step, so
/// `μ₂` sets the damping of resolved barotropic motion. At 50 baroclinic steps
/// per period:
///
/// | Window | μ₂ / Δt² | amplitude lost per period |
/// |---|---|---|
/// | Hann over `2Δt` (the previous filter) | 0.131 | 5.0 % |
/// | power law, `r = 0` | 0.084 | 3.2 % |
/// | power law, `r = 0.284` | ≈ 0.001 | 0.03 % |
///
/// `μ₂` stays positive, so no frequency is amplified (`r = 0.3` would amplify).
#[derive(Clone, Debug)]
pub struct BarotropicFilter {
    n_bt: usize,
    weights: Vec<f64>,
}

impl BarotropicFilter {
    const R: f64 = 0.284;

    /// Weights for `n_bt` substeps per baroclinic step (`n_bt ≥ 4`).
    pub fn new(n_bt: usize) -> Self {
        assert!(
            n_bt >= MIN_BAROTROPIC_SUBSTEPS,
            "the barotropic filter needs at least {MIN_BAROTROPIC_SUBSTEPS} substeps per \
             baroclinic step, got {n_bt}"
        );
        let n = n_bt as f64;
        let shape = |x: f64| x * x * (1.0 - x.powi(4)) - Self::R * x;

        let mut weights = Vec::with_capacity(2 * n_bt);
        let mut scale = 0.6 / n;
        for _ in 0..200 {
            // A(x) < 0 for x ≥ 1, so the last positive weight has m·s < 1
            let m_max = (1.0 / scale).floor() as usize;
            weights.clear();
            weights.extend((1..=m_max).map(|m| shape(m as f64 * scale)));
            while weights.last().is_some_and(|&w| w <= 0.0) {
                weights.pop();
            }
            let sum: f64 = weights.iter().sum();
            weights.iter_mut().for_each(|w| *w /= sum);
            let centroid = Self::centroid(&weights);
            if (centroid - n).abs() <= 1e-13 * n {
                break;
            }
            scale *= centroid / n;
        }
        Self { n_bt, weights }
    }

    /// Substeps per baroclinic step (`Δt/Δt_bt`).
    pub fn n_bt(&self) -> usize {
        self.n_bt
    }

    /// Weight of the state after substep `m`, stored at index `m − 1`. The
    /// window length `M*` is `weights().len()`.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    fn centroid(weights: &[f64]) -> f64 {
        weights
            .iter()
            .enumerate()
            .map(|(i, w)| (i + 1) as f64 * w)
            .sum()
    }
}

/// Field buffers sized on the first step and reused afterwards.
struct Buffers {
    /// `R₃D` at `tⁿ`, reused as the first 3D stage.
    rhs_n: Solution3D,
    /// Barotropic transport during the pass.
    q: SWESolution2D,
    /// Filtered transport.
    q_avg: SWESolution2D,
    /// Slow forcing `G` of the transport (zero for `h`).
    forcing: SWESolution2D,
    /// Depth means of the u/v tendency (or of u/v after diffusion).
    mean_u: DGSolution2D,
    mean_v: DGSolution2D,
    /// Constant barotropic rates over the step.
    rate_eta: DGSolution2D,
    rate_ubar: DGSolution2D,
    rate_vbar: DGSolution2D,
}

impl Buffers {
    fn new(state: &Solution3D) -> Self {
        let (ne, nn) = (state.n_elements, state.n_nodes);
        Self {
            rhs_n: Solution3D::new(ne, nn, state.n_levels),
            q: SWESolution2D::new(ne, nn),
            q_avg: SWESolution2D::new(ne, nn),
            forcing: SWESolution2D::new(ne, nn),
            mean_u: DGSolution2D::new(ne, nn),
            mean_v: DGSolution2D::new(ne, nn),
            rate_eta: DGSolution2D::new(ne, nn),
            rate_ubar: DGSolution2D::new(ne, nn),
            rate_vbar: DGSolution2D::new(ne, nn),
        }
    }
}

/// Mode-split time integrator: one filtered barotropic pass per baroclinic
/// SSP-RK3 step (see the module docs).
pub struct ModeSplitIntegrator {
    /// Coupling method.
    pub split_method: SplitMethod,
    barotropic_cfl: f64,
    min_substeps: usize,
    filter: Option<BarotropicFilter>,
    buffers: Option<Buffers>,
    stages_3d: StageWorkspace<Solution3D>,
    stages_2d: StageWorkspace<SWESolution2D>,
}

impl Default for ModeSplitIntegrator {
    fn default() -> Self {
        Self::new()
    }
}

impl ModeSplitIntegrator {
    /// Barotropic CFL used unless set with [`Self::with_barotropic_cfl`].
    pub const DEFAULT_BAROTROPIC_CFL: f64 = 0.5;

    /// Integrator with the default barotropic CFL and at least
    /// [`MIN_BAROTROPIC_SUBSTEPS`] substeps per step.
    pub fn new() -> Self {
        Self {
            split_method: SplitMethod::GTerm,
            barotropic_cfl: Self::DEFAULT_BAROTROPIC_CFL,
            min_substeps: MIN_BAROTROPIC_SUBSTEPS,
            filter: None,
            buffers: None,
            stages_3d: StageWorkspace::new(),
            stages_2d: StageWorkspace::new(),
        }
    }

    /// CFL number of the barotropic substeps, capped by the 2D module's
    /// [`PhysicsModule::max_cfl`] (e.g. its wet/dry positivity bound).
    pub fn with_barotropic_cfl(mut self, cfl: f64) -> Self {
        assert!(cfl > 0.0, "barotropic CFL must be positive, got {cfl}");
        self.barotropic_cfl = cfl;
        self
    }

    /// Use at least `n` barotropic substeps per baroclinic step, even when the
    /// 2D CFL allows fewer (`n ≥ 4`).
    pub fn with_min_substeps(mut self, n: usize) -> Self {
        assert!(
            n >= MIN_BAROTROPIC_SUBSTEPS,
            "at least {MIN_BAROTROPIC_SUBSTEPS} barotropic substeps are needed, got {n}"
        );
        self.min_substeps = n;
        self
    }

    /// Barotropic substeps per baroclinic step used by the last step (0 before
    /// the first step).
    pub fn last_substeps(&self) -> usize {
        self.filter.as_ref().map_or(0, BarotropicFilter::n_bt)
    }

    /// Perform one baroclinic (3D) time step.
    ///
    /// # Arguments
    /// * `state` - Full 3D solution (modified in place).
    /// * `sigma` - Vertical grid for depth averaging.
    /// * `bathymetry` - Bed elevation `B`; the depth is `η − B`.
    /// * `dt` - Baroclinic time step.
    /// * `t` - Current time.
    /// * `forcing` - Surface and bottom forcing (vertical diffusion).
    /// * `mixing` - Vertical mixing closure.
    /// * `barotropic` - 2D shallow-water module for the fast mode, in transport
    ///   form `(h, hu, hv)`.
    /// * `rhs_3d` - `f(state, t, out)`: overwrites `out` with the 3D tendency.
    ///   Only `u, v, temp, salt` are used; the barotropic entries are replaced.
    /// * `stage_hook` - Runs on every 3D stage value, including the last:
    ///   limiters and the density update.
    #[allow(clippy::too_many_arguments)]
    pub fn step<P, F3, H>(
        &mut self,
        state: &mut Solution3D,
        sigma: &SigmaGrid,
        bathymetry: &Bathymetry2D,
        dt: f64,
        t: f64,
        forcing: &Forcing,
        mixing: &dyn VerticalMixing,
        barotropic: &P,
        mut rhs_3d: F3,
        stage_hook: H,
    ) where
        P: PhysicsModule<SWESolution2D>,
        F3: FnMut(&Solution3D, f64, &mut Solution3D),
        H: FnMut(&mut Solution3D),
    {
        let Buffers {
            rhs_n,
            q,
            q_avg,
            forcing: g_term,
            mean_u,
            mean_v,
            rate_eta,
            rate_ubar,
            rate_vbar,
        } = self.buffers.get_or_insert_with(|| Buffers::new(state));

        // 1. Slow forcing: G = Dⁿ·⟨R₃D⟩ at tⁿ, a transport tendency
        rhs_3d(state, t, rhs_n);
        depth_average(sigma, &rhs_n.u, mean_u);
        depth_average(sigma, &rhs_n.v, mean_v);
        to_transport(state, bathymetry, q);
        g_term.data[SWE_VAR_H].fill(0.0);
        for (idx, &h) in q.data[SWE_VAR_H].iter().enumerate() {
            g_term.data[SWE_VAR_HU][idx] = h * mean_u.data[idx];
            g_term.data[SWE_VAR_HV][idx] = h * mean_v.data[idx];
        }

        // 2. One filtered barotropic pass
        let cfl = barotropic
            .max_cfl()
            .map_or(self.barotropic_cfl, |max| self.barotropic_cfl.min(max));
        let dt_bt_max = barotropic.compute_dt(q, cfl);
        assert!(
            dt_bt_max > 0.0,
            "barotropic time step {dt_bt_max} is not positive"
        );
        let n_bt = ((dt / dt_bt_max).ceil() as usize).max(self.min_substeps);
        if self.filter.as_ref().is_none_or(|f| f.n_bt() != n_bt) {
            self.filter = Some(BarotropicFilter::new(n_bt));
        }
        let filter = self.filter.as_ref().expect("filter was just set");
        let dt_bt = dt / n_bt as f64;

        q_avg.fill(0.0);
        for (m, &w) in filter.weights().iter().enumerate() {
            SSPRK3.step_with_relaxation(
                q,
                dt_bt,
                t + m as f64 * dt_bt,
                |s, time, out| {
                    barotropic.compute_rhs_into(s, time, out);
                    out.axpy(1.0, g_term);
                },
                |stage, from, dt_stage| barotropic.implicit_damping(stage, from, dt_stage),
                |s| barotropic.post_process(s),
                &mut self.stages_2d,
            );
            q_avg.axpy(w, q);
        }
        // The early weights are negative: restore positivity of the average
        barotropic.post_process(q_avg);

        // Constant barotropic rates over the step
        for k in 0..state.n_elements {
            let bed = bathymetry.element(ElementIndex::new(k));
            for (i, &b) in bed.iter().enumerate() {
                let idx = k * state.n_nodes + i;
                let h = q_avg.data[SWE_VAR_H][idx];
                let (ubar, vbar) = if h > 0.0 {
                    (
                        q_avg.data[SWE_VAR_HU][idx] / h,
                        q_avg.data[SWE_VAR_HV][idx] / h,
                    )
                } else {
                    (0.0, 0.0)
                };
                rate_eta.data[idx] = (h + b - state.eta.data[idx]) / dt;
                rate_ubar.data[idx] = (ubar - state.ubar.data[idx]) / dt;
                rate_vbar.data[idx] = (vbar - state.vbar.data[idx]) / dt;
            }
        }

        // 3. 3D stages with the barotropic state prescribed
        let mut first_stage = true;
        SSPRK3.step_with_workspace(
            state,
            dt,
            t,
            |s, time, out| {
                if first_stage {
                    out.copy_from(rhs_n);
                    first_stage = false;
                } else {
                    rhs_3d(s, time, out);
                }
                depth_average(sigma, &out.u, mean_u);
                depth_average(sigma, &out.v, mean_v);
                shift_columns(&mut out.u, out.n_levels, mean_u, rate_ubar);
                shift_columns(&mut out.v, out.n_levels, mean_v, rate_vbar);
                out.eta.copy_from(rate_eta);
                out.ubar.copy_from(rate_ubar);
                out.vbar.copy_from(rate_vbar);
            },
            stage_hook,
            &mut self.stages_3d,
        );

        // 4. Implicit vertical diffusion changes the depth mean through the
        // surface and bottom stresses; reset it to the barotropic ū.
        apply_vertical_diffusion(state, sigma, bathymetry, dt, mixing, forcing);
        depth_average(sigma, &state.u, mean_u);
        depth_average(sigma, &state.v, mean_v);
        shift_columns(&mut state.u, state.n_levels, mean_u, &state.ubar);
        shift_columns(&mut state.v, state.n_levels, mean_v, &state.vbar);
    }
}

/// Barotropic transport `(h, hu, hv)` with `h = η − B`.
fn to_transport(state: &Solution3D, bathymetry: &Bathymetry2D, q: &mut SWESolution2D) {
    for k in 0..state.n_elements {
        let bed = bathymetry.element(ElementIndex::new(k));
        for (i, &b) in bed.iter().enumerate() {
            let idx = k * state.n_nodes + i;
            let h = state.eta.data[idx] - b;
            q.data[SWE_VAR_H][idx] = h;
            q.data[SWE_VAR_HU][idx] = h * state.ubar.data[idx];
            q.data[SWE_VAR_HV][idx] = h * state.vbar.data[idx];
        }
    }
}

/// Depth average of every column of a 3D field.
fn depth_average(sigma: &SigmaGrid, field: &[f64], out: &mut DGSolution2D) {
    for (mean, column) in out
        .data
        .iter_mut()
        .zip(field.chunks_exact(sigma.n_levels()))
    {
        *mean = sigma.depth_average(column);
    }
}

/// Shift every column uniformly so its depth mean goes from `from` to `to`.
/// The σ-layer fractions sum to one, so a uniform shift changes the depth mean
/// by exactly the shift.
fn shift_columns(field: &mut [f64], n_levels: usize, from: &DGSolution2D, to: &DGSolution2D) {
    for ((column, &a), &b) in field
        .chunks_exact_mut(n_levels)
        .zip(&from.data)
        .zip(&to.data)
    {
        let shift = b - a;
        column.iter_mut().for_each(|x| *x += shift);
    }
}

impl IntegratorInfo for ModeSplitIntegrator {
    fn name(&self) -> &'static str {
        "mode-split-filtered-ssprk3"
    }

    /// First order: the slow forcing is frozen at `tⁿ` over the step.
    fn order(&self) -> usize {
        1
    }

    fn n_stages(&self) -> usize {
        3
    }

    fn is_ssp(&self) -> bool {
        false
    }

    fn stage_times(&self, dt: f64) -> Vec<f64> {
        SSPRK3.stage_times(dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Moments of the filter about `tⁿ⁺¹`, in units of the baroclinic step.
    fn central_moment(filter: &BarotropicFilter, k: i32) -> f64 {
        let n = filter.n_bt() as f64;
        filter
            .weights()
            .iter()
            .enumerate()
            .map(|(i, w)| w * (((i + 1) as f64 - n) / n).powi(k))
            .sum()
    }

    /// `|Σ w_m e^{iω(t_m − tⁿ⁺¹)}|` at `ω·Δt = omega_dt`.
    fn response(filter: &BarotropicFilter, omega_dt: f64) -> f64 {
        let n = filter.n_bt() as f64;
        let (re, im) = filter
            .weights()
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(re, im), (i, w)| {
                let phase = omega_dt * ((i + 1) as f64 - n) / n;
                (re + w * phase.cos(), im + w * phase.sin())
            });
        (re * re + im * im).sqrt()
    }

    #[test]
    fn reports_coupled_split_accuracy_not_slow_integrator_accuracy() {
        let integrator = ModeSplitIntegrator::new();

        assert_eq!(integrator.name(), "mode-split-filtered-ssprk3");
        assert_eq!(integrator.order(), 1);
        assert_eq!(integrator.n_stages(), 3);
        assert!(!integrator.is_ssp());
        assert_eq!(integrator.last_substeps(), 0);
    }

    #[test]
    #[should_panic(expected = "at least 4 barotropic substeps")]
    fn rejects_too_few_barotropic_substeps() {
        let _ = ModeSplitIntegrator::new().with_min_substeps(3);
    }

    /// The filter must be normalised and centred on `tⁿ⁺¹` (P0.6: a filter
    /// that is off-centre biases the slow trend; an unnormalised one blows up),
    /// and its window must end within `2Δt`.
    #[test]
    fn filter_is_normalised_and_centred_on_the_baroclinic_endpoint() {
        for n_bt in MIN_BAROTROPIC_SUBSTEPS..=200 {
            let filter = BarotropicFilter::new(n_bt);
            let w = filter.weights();
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-13, "n_bt {n_bt}: Σw = {sum}");
            let centred = central_moment(&filter, 1);
            assert!(
                centred.abs() < 1e-6,
                "n_bt {n_bt}: centroid off tⁿ⁺¹ by {centred} Δt"
            );
            assert!(
                w.len() > n_bt && w.len() <= 2 * n_bt,
                "n_bt {n_bt}: window of {} substeps",
                w.len()
            );
        }
    }

    /// A constant-rate barotropic signal is reproduced exactly at `tⁿ⁺¹`: the
    /// value at the centroid, not at the end of the window.
    #[test]
    fn filter_reproduces_a_linear_signal_at_the_endpoint() {
        let n_bt = 12;
        let filter = BarotropicFilter::new(n_bt);
        let dt = 30.0;
        let rate = 0.7;
        let filtered: f64 = filter
            .weights()
            .iter()
            .enumerate()
            .map(|(i, w)| w * rate * (i + 1) as f64 * dt / n_bt as f64)
            .sum();
        assert!((filtered - rate * dt).abs() < 1e-9 * rate * dt);
    }

    /// Resolved barotropic motion is neither amplified nor noticeably damped,
    /// and the filter still damps what the baroclinic step cannot resolve.
    #[test]
    fn filter_barely_damps_resolved_waves_and_never_amplifies() {
        for n_bt in [4, 5, 6, 10, 20, 30, 60, 120] {
            let filter = BarotropicFilter::new(n_bt);
            // Second moment positive (no amplification) and small (little damping)
            let mu2 = central_moment(&filter, 2);
            assert!(mu2 > 0.0 && mu2 < 0.02, "n_bt {n_bt}: μ₂ = {mu2}");

            for j in 0..=400 {
                let omega_dt = std::f64::consts::PI * j as f64 / 400.0;
                let r = response(&filter, omega_dt);
                assert!(r <= 1.0 + 1e-12, "n_bt {n_bt}: |R({omega_dt})| = {r}");
            }

            // 50 baroclinic steps per period: < 0.7 % amplitude per period
            let per_step = response(&filter, 2.0 * std::f64::consts::PI / 50.0);
            let per_period = 1.0 - per_step.powi(50);
            assert!(
                per_period < 7e-3,
                "n_bt {n_bt}: {:.3} % lost per period",
                100.0 * per_period
            );
        }
        // n_bt ≥ 10: < 0.1 % per period
        let per_step = response(
            &BarotropicFilter::new(30),
            2.0 * std::f64::consts::PI / 50.0,
        );
        assert!(1.0 - per_step.powi(50) < 1e-3);
    }
}
