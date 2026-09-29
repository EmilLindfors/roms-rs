//! Multirate (local) time stepping: each element advances with its own
//! stable time step, a power-of-two fraction of the global one.
//!
//! On a locally refined mesh (a fish farm at 20 m inside a kilometre-scale
//! domain, fjord arms, deep channels next to shallow banks), global explicit
//! stepping lets the few smallest or fastest elements set the time step of
//! all. Here element k gets a level ℓₖ ∈ 0..=L and takes 2^ℓₖ substeps of
//! Δt_ℓ = Δt/2^ℓ per (coarse) step Δt, with Δt_L the smallest element time
//! step.
//!
//! # Scheme
//!
//! A multirate partitioned Runge–Kutta method in the construction of
//! Constantinescu & Sandu (2007, J. Sci. Comput. 33:239–278,
//! doi:10.1007/s10915-007-9151-y), which SLIM uses for DG ocean models (Seny
//! et al. 2013, IJNMF 71:41–64, doi:10.1002/fld.3646; 2014, JCP 256:135–160,
//! doi:10.1016/j.jcp.2013.07.041), here with SSP-RK3 as the base method.
//!
//! The coarse step is split into P = 2^L finest substeps. In finest substep p
//! every element evaluates one SSP-RK3 step of its own (Shu–Osher form):
//!
//! ```text
//! Y₂ = S + Δt_ℓ F₁
//! Y₃ = ¾ S + ¼ (Y₂ + Δt_ℓ F₂)
//! H  = ⅓ S + ⅔ (Y₃ + Δt_ℓ F₃)
//! ```
//!
//! S is the element's value at the start of its current substep, and Fₛ its
//! RHS with every face neighbour at *its* stage-s value of the same finest
//! substep p. An element's substep result is the mean of H over the finest
//! substeps it spans. Stage s of an element on level ℓ is evaluated at time
//! tₖ + cₛΔt_ℓ, with c = (0, 1, ½) and tₖ its substep start.
//!
//! - **SSP / positivity:** every element update is a convex combination of
//!   forward-Euler steps of size Δt_ℓ from limited stage values (limiters and
//!   wet/dry correction run on every stage value). The Zhang–Shu positivity
//!   bound therefore holds for each element at its own Δt_ℓ.
//! - **Conservation:** every element gives stage s of finest substep p the
//!   same weight, bₛΔt/P with b = (⅙, ⅙, ⅔). The flux through a face at
//!   (p, s) comes from both sides' (p, s) values, so it is the same for both
//!   sides. Mass is conserved to rounding.
//! - **Order:** SSP-RK3 inside each level. Across level interfaces the
//!   coupling satisfies the second-order partitioned conditions
//!   (Σ bᵢcᵢ = ½ for every level) but not the third-order ones
//!   (Σ bᵢ(A_coarse c_fine)ᵢ = 5/24 ≠ ⅙ for two levels). The gate
//!   (`tests/local_time_stepping_test.rs`) measures orders 2.6–2.8 against the
//!   global solution. The interface error dominates the time error, though:
//!   ≈ 50× SSP-RK3's at the same CFL for a gravity wave crossing four levels,
//!   1.5e-3 relative at CFL 0.8. It scales as (λΔt)², so it is negligible for
//!   tides (ωΔt ~ 1e-3).
//! - **One level** is SSP-RK3, bit for bit.
//!
//! # Levels
//!
//! An element's time step comes from its own wave speeds and those of its
//! face neighbours (`LocalTimeStepping::element_dt`). The coarse step is
//! 2^L times the smallest element time step, with L the most levels (up to the
//! maximum) that keep it at most the largest one. Neighbouring levels are then
//! made to differ by at most one, refining the coarser side. Together these
//! keep water that floods in from a neighbour within one substep from
//! outrunning an element's time step (a nearly dry element otherwise gets an
//! enormous one). Levels are reassigned every coarse step.
//!
//! **Cost.** An element's RHS reads the state of its stencil
//! ([`LocalTimeStepping::stencil`]: its face neighbours for the fluxes, and
//! with BR1 horizontal viscosity also the elements at their far corners), so
//! its stage-s values change only when those within s stencil steps change.
//! Its stage-s RHS is therefore evaluated only at the rate rₛ(k), the finest
//! level within s stencil steps, and reused in between. Near a level
//! interface the coarse side pays up to three stencil widths at the finer
//! rate; elsewhere every element runs at its own level.
//!
//! A reused RHS is the one the element would compute again: its inputs,
//! including its substep's stage time, are unchanged. So the flux through a
//! face at (p, s) is the same seen from both sides, also for a viscous flux
//! that reads the neighbours' gradients (each evaluated at its own
//! element's stage time). A stencil that is too narrow breaks exactly this,
//! and with it conservation (`tests/local_time_stepping_test.rs`).

use crate::mesh::Mesh2D;
use crate::types::ElementIndex;

use super::integrator::{Integrable, IntegratorInfo, SSPRK3, StageWorkspace, TimeIntegrator};

/// What one element does in one stage of [`LocalTimeStepping::stage_where`].
///
/// With `F = L(input)` the element's RHS at `time`, the new stage value is
/// `out = a·base + b·input + c·F` (a Shu–Osher stage: `c = β·Δt`), summed in
/// that order, with zero terms left out. The stiff damping is then applied
/// point-implicitly over `c` with rates from `input`
/// ([`PhysicsModule::implicit_damping`](crate::physics::PhysicsModule::implicit_damping)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementStage {
    /// Time of the RHS evaluation.
    pub time: f64,
    /// Weight of the base value (the substep start).
    pub a: f64,
    /// Weight of the input value (the RHS argument).
    pub b: f64,
    /// Weight of the RHS.
    pub c: f64,
    /// Then, if `Some((w, keep))`: `acc ← w·out + keep·acc` (the `acc` term
    /// only if `keep ≠ 0`). Otherwise `out` is
    /// [post-processed](crate::physics::PhysicsModule::post_process).
    pub accumulate: Option<(f64, f64)>,
}

/// The elements whose state the RHS of an element k reads
/// ([`LocalTimeStepping::stencil`]), besides k itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RhsStencil {
    /// Its face neighbours (numerical fluxes).
    Faces,
    /// Its face neighbours j and, for each, the elements across the two
    /// faces of j that meet the shared face at its corners: BR1 viscosity,
    /// whose face flux reads j's gradient at the shared face's nodes. With
    /// the diagonal GLL LIFT, that gradient reads beyond j only at the corner
    /// nodes. On a structured mesh: the 3 × 3 block around k.
    FacesAndCorners,
}

/// The element-subset operations that local time stepping needs from a
/// physics module (see
/// [`PhysicsModule::local_time_stepping`](crate::physics::PhysicsModule::local_time_stepping)).
///
/// Each works on a list of distinct elements in one pass, with work
/// proportional to the list (a multirate step makes many passes over small
/// subsets), gives them bit for bit what the whole-state operations give
/// them, and leaves the other elements untouched.
pub trait LocalTimeStepping<S>: Sync {
    /// Largest stable time step of every element for `cfl`, into `out` (one
    /// value per mesh element; `f64::INFINITY` where nothing limits it).
    fn element_dt(&self, state: &S, cfl: f64, out: &mut [f64]);

    /// The elements whose state an element's RHS reads.
    fn stencil(&self) -> RhsStencil {
        RhsStencil::Faces
    }

    /// Whether the element operations below work in this configuration
    /// (e.g. not with a limiter that reads the neighbours' means). `false`
    /// makes `Simulation` run global SSP-RK3 through the whole-state stages
    /// instead of the fused ones; local time stepping panics.
    fn is_element_local(&self) -> bool {
        true
    }

    /// One stage for every listed element k, with stage `plan(k)` (see
    /// [`ElementStage`]): its RHS from `input`, which may read the values
    /// of its [`Self::stencil`], nothing further away; the new stage value
    /// into its rows of `out`, and its sum into `acc` when the stage
    /// accumulates (`acc` must then be given). `plan` may also be called for
    /// the face neighbours of the listed elements (their stage time, e.g. for
    /// the boundary values of their gradients).
    fn stage_where(
        &self,
        base: &S,
        input: &S,
        elements: &[u32],
        plan: &(dyn Fn(usize) -> ElementStage + Sync),
        out: &mut S,
        acc: Option<&mut S>,
    );

    /// `state ← acc` and
    /// [`PhysicsModule::post_process`](crate::physics::PhysicsModule::post_process)
    /// on the listed elements: the end of their substep.
    fn finish_where(&self, state: &mut S, acc: &S, elements: &[u32]);
}

// =============================================================================
// Integrator
// =============================================================================

/// Multirate SSP-RK3 (see the [module documentation](self)): `Simulation`
/// gives every element its own time step, `Δt/2^ℓ` with `ℓ ≤ max_levels`.
///
/// It needs a physics module with
/// [`PhysicsModule::local_time_stepping`](crate::physics::PhysicsModule::local_time_stepping)
/// (`SWEPhysics2D`). As a plain [`TimeIntegrator`], without a physics module,
/// it takes global SSP-RK3 steps.
#[derive(Clone, Copy, Debug)]
pub struct MultirateSSPRK3 {
    max_levels: usize,
}

impl MultirateSSPRK3 {
    /// Largest supported number of levels below the coarsest (a time-step
    /// ratio of 2¹⁶).
    pub const MAX_LEVELS: usize = 16;

    /// At most `max_levels` levels below the coarsest, i.e. time steps from
    /// Δt down to Δt/2^max_levels. Zero gives global SSP-RK3.
    pub fn new(max_levels: usize) -> Self {
        assert!(
            max_levels <= Self::MAX_LEVELS,
            "at most {} levels",
            Self::MAX_LEVELS
        );
        Self { max_levels }
    }

    /// The largest number of levels below the coarsest.
    pub fn max_levels(&self) -> usize {
        self.max_levels
    }
}

impl IntegratorInfo for MultirateSSPRK3 {
    fn name(&self) -> &'static str {
        "multirate-ssp-rk3"
    }

    /// Second order across level interfaces (third inside a level).
    fn order(&self) -> usize {
        2
    }

    fn n_stages(&self) -> usize {
        3
    }

    fn is_ssp(&self) -> bool {
        true
    }

    fn stage_times(&self, dt: f64) -> Vec<f64> {
        vec![0.0, dt, 0.5 * dt]
    }

    fn max_local_levels(&self) -> Option<usize> {
        Some(self.max_levels)
    }
}

/// Without a physics module there are no elements: global SSP-RK3.
impl<S: Integrable> TimeIntegrator<S> for MultirateSSPRK3 {
    fn step_with_relaxation<F, R, H>(
        &self,
        state: &mut S,
        dt: f64,
        t: f64,
        rhs: F,
        relax: R,
        stage_hook: H,
        workspace: &mut StageWorkspace<S>,
    ) where
        F: FnMut(&S, f64, &mut S),
        R: FnMut(&mut S, &S, f64),
        H: FnMut(&mut S),
    {
        SSPRK3.step_with_relaxation(state, dt, t, rhs, relax, stage_hook, workspace);
    }
}

// =============================================================================
// Levels
// =============================================================================

/// Levels of the elements for element time steps `element_dt`: the coarse
/// step `Δt = 2^L·min(element_dt)`, with L the largest level count up to
/// `max_levels` that keeps `Δt ≤ max(element_dt)` and `Δt ≤ dt_cap`, and
/// `levels[k]` the smallest ℓ ≤ L with `Δt/2^ℓ ≤ element_dt[k]`.
///
/// Returns `(Δt, L)`; `(∞, 0)` when no element limits the step.
pub fn assign_levels(
    element_dt: &[f64],
    max_levels: usize,
    dt_cap: Option<f64>,
    levels: &mut Vec<u8>,
) -> (f64, usize) {
    assert!(max_levels <= MultirateSSPRK3::MAX_LEVELS);
    let finite = element_dt.iter().copied().filter(|dt| dt.is_finite());
    let (dt_min, dt_max) = finite.fold((f64::INFINITY, 0.0_f64), |(lo, hi), dt| {
        (lo.min(dt), hi.max(dt))
    });
    levels.clear();
    levels.resize(element_dt.len(), 0);
    if !dt_min.is_finite() {
        return (f64::INFINITY, 0);
    }

    let mut finest = 0;
    while finest < max_levels {
        // Powers of two scale exactly: level L steps exactly dt_min
        let coarser = dt_min * (2u64 << finest) as f64;
        if coarser > dt_max || dt_cap.is_some_and(|cap| coarser > cap) {
            break;
        }
        finest += 1;
    }
    let dt = dt_min * (1u64 << finest) as f64;

    for (level, &dt_k) in levels.iter_mut().zip(element_dt) {
        let (mut l, mut dt_l) = (0, dt);
        while l < finest && dt_l > dt_k {
            dt_l *= 0.5;
            l += 1;
        }
        *level = l as u8;
    }
    (dt, finest)
}

/// `to[k] = max(from[k], from[j] − drop)` over the face neighbours j of k;
/// whether any `to[k]` differs from `from[k]`.
fn spread_to_neighbours(mesh: &Mesh2D, from: &[u8], to: &mut [u8], drop: u8) -> bool {
    let element = |(k, out): (usize, &mut u8)| {
        let k_idx = ElementIndex::new(k);
        *out = (0..4)
            .filter_map(|face| mesh.neighbor(k_idx, face))
            .map(|nb| from[nb.element].saturating_sub(drop))
            .fold(from[k], u8::max);
        *out != from[k]
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        to.par_iter_mut()
            .enumerate()
            .map(element)
            .reduce(|| false, |a, b| a | b)
    }
    #[cfg(not(feature = "parallel"))]
    to.iter_mut()
        .enumerate()
        .map(element)
        .fold(false, |a, b| a | b)
}

/// `to[k]` = the largest `from` over k and its `stencil`.
fn spread_over_stencil(mesh: &Mesh2D, from: &[u8], to: &mut [u8], stencil: RhsStencil) {
    match stencil {
        RhsStencil::Faces => {
            spread_to_neighbours(mesh, from, to, 0);
        }
        RhsStencil::FacesAndCorners => {
            let element = |(k, out): (usize, &mut u8)| {
                let mut max = from[k];
                for face in 0..4 {
                    let Some(nb) = mesh.neighbor(ElementIndex::new(k), face) else {
                        continue;
                    };
                    max = max.max(from[nb.element]);
                    // Across the faces of the neighbour adjacent to the
                    // shared one (faces are numbered around the element)
                    let nb_idx = ElementIndex::new(nb.element);
                    for side in [1, 3] {
                        if let Some(corner) = mesh.neighbor(nb_idx, (nb.face + side) % 4) {
                            max = max.max(from[corner.element]);
                        }
                    }
                }
                *out = max;
            };
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                to.par_iter_mut().enumerate().for_each(element);
            }
            #[cfg(not(feature = "parallel"))]
            to.iter_mut().enumerate().for_each(element);
        }
    }
}

/// Raise levels until face neighbours differ by at most one (the coarser
/// side is refined; smaller steps stay stable). An element then sees at most
/// two neighbour substeps per substep of its own, so water from a neighbour
/// cannot outrun its time step. `scratch` is resized as needed.
fn limit_level_jumps(mesh: &Mesh2D, level: &mut Vec<u8>, scratch: &mut Vec<u8>) {
    scratch.resize(level.len(), 0);
    while spread_to_neighbours(mesh, level, scratch, 1) {
        std::mem::swap(level, scratch);
    }
}

/// Elements in decreasing order of `key` (increasing index within a key)
/// into `order`, and `count[d]` = the number with `key ≥ d` for `d` in
/// `0..=max_key + 1`: the elements with `key ≥ d` are `order[..count[d]]`.
fn sort_descending(key: &[u8], max_key: usize, order: &mut Vec<u32>, count: &mut Vec<usize>) {
    count.clear();
    count.resize(max_key + 2, 0);
    for &k in key {
        count[k as usize] += 1;
    }
    // count[d] ← Σ_{r ≥ d} hist[r]; the elements of key r start at count[r + 1]
    for d in (0..=max_key).rev() {
        count[d] += count[d + 1];
    }
    let mut next: Vec<usize> = (0..=max_key).map(|r| count[r + 1]).collect();
    order.clear();
    order.resize(key.len(), 0);
    for (k, &r) in key.iter().enumerate() {
        order[next[r as usize]] = k as u32;
        next[r as usize] += 1;
    }
}

/// Level of finest substep p's alignment: the coarsest level whose substeps
/// start at p (0 at the start and end of the coarse step).
#[inline]
fn depth(p: usize, finest: usize) -> usize {
    if p == 0 || p == 1 << finest {
        0
    } else {
        finest - p.trailing_zeros() as usize
    }
}

// =============================================================================
// Stepper
// =============================================================================

/// Work counts of a multirate run.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MultirateStats {
    /// Coarse steps taken.
    pub steps: u64,
    /// Element RHS evaluations.
    pub element_evaluations: u64,
    /// Element RHS evaluations that global SSP-RK3 at the finest time step
    /// would have needed for the same steps.
    pub global_evaluations: u64,
    /// Finest level used (the largest time-step ratio is 2^this).
    pub finest_level: usize,
}

impl MultirateStats {
    /// Saving in RHS work against global stepping at the finest time step.
    pub fn speedup(&self) -> f64 {
        self.global_evaluations as f64 / self.element_evaluations.max(1) as f64
    }
}

/// Levels, stage storage and work counts of a multirate SSP-RK3 run; one per
/// simulation (`Simulation` keeps it).
pub struct MultirateStepper<S> {
    max_levels: usize,
    /// Finest level L of the current coarse step
    finest: usize,
    element_dt: Vec<f64>,
    level: Vec<u8>,
    /// `rate[s][k]`: the level whose substeps element k's stage-(s + 1) RHS
    /// follows, the finest level within s + 1 steps of the RHS stencil
    rate: [Vec<u8>; 3],
    /// Elements by decreasing `rate[s]` and by decreasing level, with the
    /// counts at or above each depth (see [`sort_descending`]): the active
    /// elements of a pass are a prefix
    rate_order: [Vec<u32>; 3],
    rate_count: [Vec<usize>; 3],
    level_order: Vec<u32>,
    level_count: Vec<usize>,
    /// Element RHS evaluations of one coarse step with the current levels
    evaluations_per_step: u64,
    y2: Option<S>,
    y3: Option<S>,
    rhs: Option<S>,
    acc: Option<S>,
    stats: MultirateStats,
}

impl<S: Integrable> MultirateStepper<S> {
    /// Empty stepper with up to `max_levels` levels below the coarsest.
    pub fn new(max_levels: usize) -> Self {
        assert!(max_levels <= MultirateSSPRK3::MAX_LEVELS);
        Self {
            max_levels,
            finest: 0,
            element_dt: Vec::new(),
            level: Vec::new(),
            rate: Default::default(),
            rate_order: Default::default(),
            rate_count: Default::default(),
            level_order: Vec::new(),
            level_count: Vec::new(),
            evaluations_per_step: 0,
            y2: None,
            y3: None,
            rhs: None,
            acc: None,
            stats: MultirateStats::default(),
        }
    }

    /// Assign levels from the current element time steps, and return the
    /// coarse step (see [`assign_levels`]; at most `dt_cap`). Call it before
    /// every [`Self::step`].
    pub fn assign_levels(
        &mut self,
        local: &dyn LocalTimeStepping<S>,
        mesh: &Mesh2D,
        state: &S,
        cfl: f64,
        dt_cap: Option<f64>,
    ) -> f64 {
        let n = mesh.n_elements;
        self.element_dt.resize(n, 0.0);
        local.element_dt(state, cfl, &mut self.element_dt);
        let (dt, finest) =
            assign_levels(&self.element_dt, self.max_levels, dt_cap, &mut self.level);
        self.finest = finest;

        let stencil = local.stencil();
        let [r1, r2, r3] = &mut self.rate;
        for r in [&mut *r1, &mut *r2, &mut *r3] {
            r.resize(n, 0);
        }
        limit_level_jumps(mesh, &mut self.level, r1);
        spread_over_stencil(mesh, &self.level, r1, stencil);
        spread_over_stencil(mesh, r1, r2, stencil);
        spread_over_stencil(mesh, r2, r3, stencil);
        self.evaluations_per_step = self.rate.iter().flatten().map(|&r| 1u64 << r).sum();
        for s in 0..3 {
            sort_descending(
                &self.rate[s],
                finest,
                &mut self.rate_order[s],
                &mut self.rate_count[s],
            );
        }
        sort_descending(
            &self.level,
            finest,
            &mut self.level_order,
            &mut self.level_count,
        );
        dt
    }

    /// Put all `n_elements` elements on one level, for global SSP-RK3
    /// steps through the fused per-element stages (no element time steps
    /// needed: the caller takes the global step). Call it once; the levels
    /// stay until the next [`Self::assign_levels`].
    pub fn assign_one_level(&mut self, n_elements: usize) {
        self.finest = 0;
        self.level.clear();
        self.level.resize(n_elements, 0);
        for s in 0..3 {
            self.rate[s].clear();
            self.rate[s].resize(n_elements, 0);
            sort_descending(
                &self.rate[s],
                0,
                &mut self.rate_order[s],
                &mut self.rate_count[s],
            );
        }
        sort_descending(&self.level, 0, &mut self.level_order, &mut self.level_count);
        self.evaluations_per_step = 3 * n_elements as u64;
    }

    /// Level of every element from the last [`Self::assign_levels`].
    pub fn levels(&self) -> &[u8] {
        &self.level
    }

    /// Finest level of the last [`Self::assign_levels`].
    pub fn finest_level(&self) -> usize {
        self.finest
    }

    /// Work counts so far.
    pub fn stats(&self) -> MultirateStats {
        self.stats
    }

    /// One coarse step of `dt` from time `t`, with the levels of the last
    /// [`Self::assign_levels`]. `dt` may be shorter than the step it returned
    /// (all levels shrink with it).
    pub fn step(&mut self, local: &dyn LocalTimeStepping<S>, state: &mut S, t: f64, dt: f64) {
        let finest = self.finest;
        let (level, rate) = (&self.level, &self.rate);
        let (rate_order, rate_count) = (&self.rate_order, &self.rate_count);
        let (level_order, level_count) = (&self.level_order, &self.level_count);
        let y2 = self.y2.get_or_insert_with(|| state.clone());
        let y3 = self.y3.get_or_insert_with(|| state.clone());
        let f = self.rhs.get_or_insert_with(|| state.clone());
        let acc = self.acc.get_or_insert_with(|| state.clone());

        let step_of = |l: u8| dt / (1u64 << l) as f64;
        for p in 0..1usize << finest {
            let d = depth(p, finest);
            // Start time and step of element k's current substep
            let substep = |k: usize| {
                let l = level[k];
                let dt_l = step_of(l);
                let q = p >> (finest - l as usize);
                (t + q as f64 * dt_l, dt_l)
            };
            // The elements whose stage-(s + 1) RHS follows a level ≥ d
            let active = |s: usize| &rate_order[s][..rate_count[s][d]];

            // Stage 1: Y₂ = S + Δt_ℓ F(S)
            local.stage_where(
                state,
                state,
                active(0),
                &|k| {
                    let (t_k, dt_l) = substep(k);
                    ElementStage {
                        time: t_k,
                        a: 1.0,
                        b: 0.0,
                        c: dt_l,
                        accumulate: None,
                    }
                },
                y2,
                None,
            );

            // Stage 2: Y₃ = ¾S + ¼Y₂ + ¼Δt_ℓ F(Y₂)
            local.stage_where(
                state,
                y2,
                active(1),
                &|k| {
                    let (t_k, dt_l) = substep(k);
                    ElementStage {
                        time: t_k + dt_l,
                        a: 0.75,
                        b: 0.25,
                        c: 0.25 * dt_l,
                        accumulate: None,
                    }
                },
                y3,
                None,
            );

            // Stage 3: H = ⅓S + ⅔Y₃ + ⅔Δt_ℓ F(Y₃). The substep result is the
            // mean of H over its 2^(r₃ − ℓ) blocks; a block starting with the
            // substep restarts the sum
            local.stage_where(
                state,
                y3,
                active(2),
                &|k| {
                    let (t_k, dt_l) = substep(k);
                    let blocks = 1u64 << (rate[2][k] - level[k]);
                    let first = level[k] as usize >= d;
                    ElementStage {
                        time: t_k + 0.5 * dt_l,
                        a: 1.0 / 3.0,
                        b: 2.0 / 3.0,
                        c: 2.0 / 3.0 * dt_l,
                        accumulate: Some((1.0 / blocks as f64, if first { 0.0 } else { 1.0 })),
                    }
                },
                f,
                Some(&mut *acc),
            );

            // Substeps ending here
            let next = depth(p + 1, finest);
            local.finish_where(state, acc, &level_order[..level_count[next]]);
        }

        self.stats.steps += 1;
        self.stats.element_evaluations += self.evaluations_per_step;
        self.stats.global_evaluations += 3 * self.level.len() as u64 * (1u64 << finest);
        self.stats.finest_level = self.stats.finest_level.max(finest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_are_powers_of_two_below_the_coarse_step() {
        let element_dt = [1.0, 0.3, 0.26, 0.1, f64::INFINITY, 0.55];
        let mut levels = Vec::new();
        let (dt, finest) = assign_levels(&element_dt, 8, None, &mut levels);
        // 0.1·2³ = 0.8 ≤ 1.0 < 1.6
        assert_eq!(finest, 3);
        assert_eq!(dt, 0.8);
        assert_eq!(levels, [0, 2, 2, 3, 0, 1]);
        for (&l, &dt_k) in levels.iter().zip(&element_dt) {
            let dt_l = dt / (1 << l) as f64;
            assert!(dt_l <= dt_k, "level {l} steps {dt_l} > {dt_k}");
            // and it is the coarsest such level
            assert!(l == 0 || 2.0 * dt_l > dt_k);
        }
    }

    #[test]
    fn levels_respect_the_caps() {
        let element_dt = [1.0, 0.1];
        let mut levels = Vec::new();
        assert_eq!(assign_levels(&element_dt, 2, None, &mut levels), (0.4, 2));
        assert_eq!(levels, [0, 2]);
        assert_eq!(
            assign_levels(&element_dt, 8, Some(0.25), &mut levels),
            (0.2, 1)
        );
        assert_eq!(levels, [0, 1]);
        assert_eq!(assign_levels(&element_dt, 0, None, &mut levels), (0.1, 0));
        assert_eq!(levels, [0, 0]);
        let dry = [f64::INFINITY; 3];
        assert_eq!(
            assign_levels(&dry, 4, None, &mut levels),
            (f64::INFINITY, 0)
        );
        assert_eq!(levels, [0, 0, 0]);
    }

    #[test]
    fn descending_sort_gives_prefixes_by_key() {
        let key = [0u8, 2, 1, 2, 0, 3];
        let (mut order, mut count) = (Vec::new(), Vec::new());
        sort_descending(&key, 3, &mut order, &mut count);
        assert_eq!(order, [5, 1, 3, 2, 0, 4]);
        assert_eq!(count, [6, 4, 3, 1, 0]);
        for d in 0..=4 {
            let mut prefix: Vec<u32> = order[..count[d]].to_vec();
            prefix.sort();
            let expected: Vec<u32> = (0..6)
                .filter(|&k| key[k] as usize >= d)
                .map(|k| k as u32)
                .collect();
            assert_eq!(prefix, expected, "d = {d}");
        }
    }

    #[test]
    fn depth_is_the_coarsest_aligned_level() {
        let depths: Vec<_> = (0..=8).map(|p| depth(p, 3)).collect();
        assert_eq!(depths, [0, 3, 2, 3, 1, 3, 2, 3, 0]);
        assert_eq!(depth(0, 0), 0);
        assert_eq!(depth(1, 0), 0);
    }

    #[test]
    fn rates_spread_one_face_per_stage() {
        // A row of 6 elements; only the last one is on level 2
        let mesh = Mesh2D::uniform_rectangle(0.0, 6.0, 0.0, 1.0, 6, 1);
        let level = [0, 0, 0, 0, 0, 2];
        let mut r1 = vec![0; 6];
        let mut r2 = vec![0; 6];
        let mut r3 = vec![0; 6];
        spread_to_neighbours(&mesh, &level, &mut r1, 0);
        spread_to_neighbours(&mesh, &r1, &mut r2, 0);
        spread_to_neighbours(&mesh, &r2, &mut r3, 0);
        assert_eq!(r1, [0, 0, 0, 0, 2, 2]);
        assert_eq!(r2, [0, 0, 0, 2, 2, 2]);
        assert_eq!(r3, [0, 0, 2, 2, 2, 2]);
    }

    /// The viscous stencil of an element on a structured mesh is the 3 × 3
    /// block around it: face neighbours and diagonals, not two faces away.
    #[test]
    fn corner_stencil_is_the_block_around_an_element() {
        // 5 × 5 elements, row-major from the bottom left; the centre on level 1
        let mesh = Mesh2D::uniform_rectangle(0.0, 5.0, 0.0, 5.0, 5, 5);
        let at = |i: usize, j: usize| j * 5 + i;
        let mut level = vec![0u8; 25];
        level[at(2, 2)] = 1;
        let mut rate = vec![0u8; 25];
        spread_over_stencil(&mesh, &level, &mut rate, RhsStencil::FacesAndCorners);
        for j in 0..5_usize {
            for i in 0..5_usize {
                let in_block = i.abs_diff(2) <= 1 && j.abs_diff(2) <= 1;
                assert_eq!(rate[at(i, j)], in_block as u8, "element ({i}, {j})");
            }
        }
        spread_over_stencil(&mesh, &level, &mut rate, RhsStencil::Faces);
        for j in 0..5_usize {
            for i in 0..5_usize {
                let in_cross = i.abs_diff(2) + j.abs_diff(2) <= 1;
                assert_eq!(rate[at(i, j)], in_cross as u8, "element ({i}, {j})");
            }
        }
    }

    #[test]
    fn level_jumps_are_limited_to_one() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 6.0, 0.0, 1.0, 6, 1);
        let mut level = vec![0, 0, 0, 0, 0, 3];
        let mut scratch = Vec::new();
        limit_level_jumps(&mesh, &mut level, &mut scratch);
        assert_eq!(level, [0, 0, 0, 1, 2, 3]);
        let mut level = vec![2, 0, 0, 0, 1, 0];
        limit_level_jumps(&mesh, &mut level, &mut scratch);
        assert_eq!(level, [2, 1, 0, 0, 1, 0]);
    }
}
