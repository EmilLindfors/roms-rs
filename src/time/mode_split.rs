//! Mode-split time integration for 3D ocean models.
//!
//! Separates the fast barotropic (2D) mode from the slow baroclinic (3D) mode.
//!
//! # Method
//!
//! One baroclinic step `tⁿ → tⁿ⁺¹ = tⁿ + Δt` (Shchepetkin & McWilliams 2005, with
//! SSP-RK3 in place of their forward-backward barotropic stepping):
//!
//! 1. **Slow forcing.** The horizontal 3D momentum tendency `R₃D` at `tⁿ`
//!    ([`ModeSplitPhysics::momentum_rhs_into`]) and, from it, the slow forcing
//!    `Gⁿ` of the barotropic transport ([`ModeSplitPhysics::slow_forcing_into`]).
//!    The pass uses the average of `G` over the step, extrapolated from
//!    `Gⁿ, Gⁿ⁻¹, Gⁿ⁻²` (AB3 with variable steps; [`step_average_weights`]).
//! 2. **One barotropic pass.** The 2D shallow-water module steps the transport
//!    `(h, hu, hv)` plus `G` with SSP-RK3 substeps of `Δt/n_bt` over
//!    `[tⁿ, tⁿ + M*·Δt/n_bt]`, `M* ≈ 1.3·n_bt`. `n_bt` follows from the 2D CFL
//!    every step. The substep states are averaged with the power-law weights of
//!    [`BarotropicFilter`], centred on `tⁿ⁺¹`; the average `(η̄, D̄ū)` is the new
//!    barotropic state, with `ū = D̄ū / D̄`. The fluxes of every RK stage are
//!    accumulated with the secondary weights into the transport that moved
//!    `η` over the step, `η̄ − ηⁿ = −Δt·∇·DU_avg2` ([`BarotropicTransport`]).
//!    Rivers ([`ModeSplitPhysics::rivers`]) add their discharge to every
//!    stage and are averaged the same way, `+ Δt·Q̄/A_k` in their elements
//!    (see [`crate::source::river`]). After the 2D RHS, each stage's work
//!    (`+ G`, the combination, the drag, the sums) is one element-block
//!    pass.
//! 3. **3D stages.** SSP-RK3 on the 3D fields, with the velocity and the
//!    tracers stepped as inventories `H_z u`, `H_z C` and divided by the new
//!    `H_z` at the end of the step. `η, ū, v̄` get the constant rates of the
//!    pass, `(η̄ − ηⁿ)/Δt` etc.; SSP-RK3 reproduces a constant-rate solution
//!    exactly, so each stage sees the barotropic state linearly interpolated
//!    to its stage time. In every stage the column sum of the momentum
//!    tendency is replaced by the constant rate of the barotropic transport,
//!    `(D̄ūⁿ⁺¹ − Dⁿūⁿ)/Δt`, shared out by `Δσ_l`, so the depth mean of `u` is
//!    `ūⁿ⁺¹` at the end. Stage 1 reuses the `R₃D` of step 1. The terms that
//!    move with the layer volume fluxes
//!    ([`ModeSplitPhysics::transport_rhs_into`]: momentum and tracer
//!    advection) get the pass's `DU_avg2` and `∂η/∂t`, so that the layers
//!    carry exactly the water the free surface moved: constancy and
//!    conservation of the tracers and of the layer momentum (see
//!    [`crate::solver::rhs::transport_3d`]). The turbulence of a prognostic
//!    closure (`Solution3D::tke`, `gls`, at the w-points) is carried the
//!    same way, as inventories over the w-cells
//!    ([`crate::solver::rhs::w_cell_thicknesses`]).
//! 4. **Implicit vertical terms** (diffusion with the surface and bottom
//!    stresses), then the depth mean of `u` is reset to `ū`.
//!
//! A bottom drag `τ_b/ρ₀ = r·u_b` ([`ModeSplitPhysics::bottom_drag_into`])
//! is linearised with `r` frozen at `tⁿ`. Its depth-mean part `−r·ū` is
//! applied point-implicitly in every barotropic RK stage, like the 2D
//! friction, so no `r·Δt/D` destabilises it; `G` carries the part of the
//! vertical shear, `−r·(u_b − ū)`, and the vertical diffusion takes
//! `r·u_bⁿ⁺¹` as its bottom flux. A drag within the column (net cages,
//! `−λ_l u_l` on layer `l`, [`ModeSplitPhysics::layer_drag_into`]) is
//! treated the same way: `−Λ̄ ū` with `Λ̄ = Σ_l Δσ_l λ_l` in the pass,
//! `−D Σ_l Δσ_l λ_l (u_l − ū)` in `G`, `−λ_l u_lⁿ⁺¹` in the vertical solve.
//!
//! The 2D positivity limiter and wet/dry treatment run after every barotropic
//! RK stage (and on the filtered state, since the filter has small negative
//! weights), and stiff 2D damping (implicit friction) is applied per stage, as
//! in [`crate::simulation::Simulation`].
//!
//! # Division of labour
//!
//! The 2D module owns the depth-mean flow: the barotropic pressure gradient
//! (with the DG face coupling of `η`), advection of `ū`, Coriolis on `ū` and
//! its own bottom friction (none when the 3D model has a bottom drag, whose
//! depth-mean part the splitter applies). `G` carries only what depends on
//! the vertical structure: the baroclinic pressure gradient, the momentum
//! dispersion of the vertical shear and the surface/bottom stresses of the 3D
//! columns (see [`crate::physics::Hydrostatic3D`]).
//!
//! # Accuracy and known gaps (TODO P4.1)
//!
//! - The slow coupling is third order (AB3 step average of `G`).
//! - The filter adds almost no damping to resolved barotropic motion
//!   (≈ 0.03 % amplitude per period at 50 baroclinic steps per period; see
//!   [`BarotropicFilter`]). Strictly this term is first order, with a
//!   constant ≈ 1e-3 of the usual one.

use crate::mesh::data::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::physics::PhysicsModule;
use crate::solver::core::blocks::{for_each_block, update_values, update_with};
use crate::solver::rhs::subcells::subcell_interfaces;
use crate::solver::rhs::{
    BarotropicFlux, MetricForm, from_inventory, subcell_divergence_element, to_inventory,
    transport_divergence_element, w_cell_thicknesses,
};
use crate::solver::state::Solution3D;
use crate::solver::state::{SWE_VAR_H, SWE_VAR_HU, SWE_VAR_HV};
use crate::solver::{DGSolution2D, SWESolution2D};
use crate::source::{RiverInflow, RiverSources};
use crate::time::{IntegratorInfo, SSPRK3, SspScheme, StageWorkspace, TimeIntegrator};
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;
use std::cell::RefCell;

/// Fewest barotropic substeps per baroclinic step. Below four the filter
/// weights cannot be centred on `tⁿ⁺¹`.
pub const MIN_BAROTROPIC_SUBSTEPS: usize = 4;

/// Most barotropic substeps per baroclinic step before the step is taken as
/// blown up (a real run needs tens to hundreds).
const MAX_BAROTROPIC_SUBSTEPS: usize = 1_000_000;

/// Relative residual of `η̄ − ηⁿ = −Δt∇·DU_avg2` above which an element is
/// treated as balanced only as a whole (round-off is ≈ 1e-13).
const NODAL_IDENTITY_TOLERANCE: f64 = 1e-9;

/// ... and whose depth change over the step (relative to the element's
/// deepest column) is above round-off: a fluid at rest has a round-off
/// transport, whose relative residual is O(1).
const DEPTH_ROUND_OFF: f64 = 1e-12;

/// Relative error of a column's water, `|∂η/∂t + ∇·DU_avg2|·Δt / D`, above
/// which an element is treated as balanced only as a whole: a tracer's
/// constancy error at that node. Without it, a film at a wetting front (a
/// millimetre of water) turns the round-off of a subcell element's residual
/// into ≈ 1e-6 of its tracers.
const CONSTANCY_TOLERANCE: f64 = 1e-12;

/// Depth (m) below which a column counts as this deep for
/// [`CONSTANCY_TOLERANCE`] (its tracers are not defined below it, see
/// [`crate::solver::rhs::from_inventory`]).
const MIN_COLUMN_VOLUME_DEPTH: f64 = 1e-6;

/// The fast-mode module: a 2D shallow-water RHS in transport form that can
/// also report the numerical mass flux at every element face.
pub trait BarotropicPhysics: PhysicsModule<SWESolution2D> {
    /// [`PhysicsModule::compute_rhs_into`], also writing the mass component
    /// of `F*` at every element face node along its outward normal into
    /// `face_mass`, and the mass flux through every subcell interface of the
    /// wet/dry subcell elements into `subcell_mass` (NaN in other elements;
    /// layouts of [`crate::solver::compute_rhs_swe_2d_mass_fluxes_into`]).
    fn compute_rhs_mass_fluxes_into(
        &self,
        state: &SWESolution2D,
        time: f64,
        out: &mut SWESolution2D,
        face_mass: &mut [f64],
        subcell_mass: &mut [f64],
    );

    /// The volume form of the mass equation's divergence: the barotropic
    /// transport's divergence and the 3D layer continuity must be the same
    /// operator as the 2D module's (they differ on general quadrilaterals).
    fn metric_form(&self) -> MetricForm;
}

/// The barotropic transport of one baroclinic step: the fluxes that moved the
/// free surface, integrated over the barotropic pass with the filter's
/// secondary weights (`DU_avg2` of Shchepetkin & McWilliams 2005).
///
/// With the substep states averaged with primary weights `w_m`, and substep
/// `j` advancing `h` by `−Δt_bt Σ_s b_s ∇·F_{j,s}` (SSP-RK3 stages
/// `b = (1/6, 1/6, 2/3)`),
///
/// ```text
///     η̄ − ηⁿ = −Δt ∇·DU_avg2,    DU_avg2 = (1/n_bt) Σ_j W_j Σ_s b_s F_{j,s},
///     W_j = Σ_{m ≥ j} w_m.
/// ```
///
/// The DG divergence is linear in the nodal `(hu, hv)` and the face mass flux
/// `F*_h`, so both are accumulated ([`Self::divergence_into`]). The identity is
/// exact to round-off for the collocated and flux-differencing forms. In
/// `WetDry` elements with a shallow node the volume term is a subcell
/// finite-volume update; its interface mass fluxes are accumulated too
/// ([`Self::subcell`]), and where every stage of the pass took the subcells
/// the identity holds in their form ([`subcell_divergence_element`]). Two
/// cases only keep the element balance `∫ (η̄ − ηⁿ) = −Δt ∮ F*_h`, not the
/// nodal identity:
/// - elements that switched between the volume term and the subcells during
///   the pass;
/// - elements the positivity limiter or wet/dry correction changed (they
///   keep the element mean).
///
/// The rivers of the 3D model ([`ModeSplitPhysics::rivers`]) add their
/// discharge to every stage and are averaged the same way, into
/// [`Self::discharge`], so that with them
/// `η̄ − ηⁿ = −Δt·∇·DU_avg2 + Δt·Σ Q̄/A_k` (see [`crate::source::river`]).
/// Other mass sources in the 2D module are not part of the transport.
pub struct BarotropicTransport {
    /// Nodal transport `hu` (m²/s).
    pub hu: DGSolution2D,
    /// Nodal transport `hv` (m²/s).
    pub hv: DGSolution2D,
    /// Mass flux out of every element face node (m²/s), laid out as
    /// `(k · 4 + face) · n_face_nodes + fi`.
    pub face: Vec<f64>,
    /// Mass flux through every subcell interface of the wet/dry subcell
    /// elements, `k · n_sub + slot` ([`crate::solver::rhs::subcells`]); NaN
    /// in the elements where some stage of the pass took the
    /// flux-differencing volume term.
    pub subcell: Vec<f64>,
    /// Discharge of every river averaged with the same weights (m³/s).
    pub discharge: Vec<f64>,
}

impl BarotropicTransport {
    fn new(
        n_elements: usize,
        n_nodes: usize,
        n_face_values: usize,
        n_subcell_values: usize,
        n_rivers: usize,
    ) -> Self {
        Self {
            hu: DGSolution2D::new(n_elements, n_nodes),
            hv: DGSolution2D::new(n_elements, n_nodes),
            face: vec![0.0; n_face_values],
            subcell: vec![0.0; n_subcell_values],
            discharge: vec![0.0; n_rivers],
        }
    }

    fn clear(&mut self) {
        self.hu.fill(0.0);
        self.hv.fill(0.0);
        self.face.fill(0.0);
        self.subcell.fill(0.0);
        self.discharge.fill(0.0);
    }

    /// The subcell interface fluxes of element `k` if every stage of the pass
    /// moved its mass on the subcells, else `None`.
    pub fn subcells_of(&self, k: usize, n_1d: usize) -> Option<&[f64]> {
        let n_sub = subcell_interfaces(n_1d);
        let fluxes = &self.subcell[k * n_sub..(k + 1) * n_sub];
        fluxes.iter().all(|f| f.is_finite()).then_some(fluxes)
    }

    /// The divergence of the transport, in the 2D kernel's form: the strong
    /// form with its volume term in `metric`'s form (the 2D module's,
    /// [`BarotropicPhysics::metric_form`]; conservatively
    /// `J⁻¹[Dr·(J∇r·q) + Ds·(J∇s·q)] − J⁻¹ Σ_f LIFT_f sJ_f (q·n − F*_h)` with
    /// `q = (hu, hv)`), or in the subcell elements of the pass the subcell
    /// form ([`subcell_divergence_element`]).
    pub fn divergence_into(
        &self,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        metric: MetricForm,
        out: &mut DGSolution2D,
    ) {
        let (nn, nfn) = (ops.n_nodes, ops.n_face_nodes);
        for k in 0..out.n_elements {
            let (hu, hv) = (
                &self.hu.data[k * nn..(k + 1) * nn],
                &self.hv.data[k * nn..(k + 1) * nn],
            );
            let face = &self.face[k * 4 * nfn..(k + 1) * 4 * nfn];
            let div = &mut out.data[k * nn..(k + 1) * nn];
            match self.subcells_of(k, ops.n_1d) {
                Some(subcells) => {
                    subcell_divergence_element(ops, geom, k, hu, hv, subcells, face, div)
                }
                None => transport_divergence_element(ops, geom, metric, k, hu, hv, face, div),
            }
        }
    }
}

/// The 3D model as the mode splitter sees it.
pub trait ModeSplitPhysics {
    /// The 2D shallow-water module of the fast mode, in transport form.
    type Barotropic: BarotropicPhysics;

    /// The fast-mode module.
    fn barotropic(&self) -> &Self::Barotropic;

    /// Vertical grid (for depth averages).
    fn sigma(&self) -> &SigmaGrid;

    /// Bed elevation `B`; the depth is `η − B`.
    fn bathymetry(&self) -> &Bathymetry2D;

    /// The rivers, with their levels set, if any. The splitter adds them to
    /// every barotropic stage (they must not also be a source of the 2D
    /// module) and hands their step-mean discharge to the 3D stages in
    /// [`BarotropicFlux::rivers`] (see [`crate::source::river`]).
    fn rivers(&self) -> Option<&RiverSources> {
        None
    }

    /// Overwrite `out.u` and `out.v` with the explicit velocity tendency of
    /// `state` (`∂u/∂t`, m/s²): everything but the terms that move with the
    /// layer volume fluxes. The slow forcing is built from it. Other fields
    /// of `out` are overwritten by [`Self::transport_rhs_into`] or the
    /// splitter.
    fn momentum_rhs_into(&self, state: &Solution3D, t: f64, out: &mut Solution3D);

    /// Add the inventory tendency `∂(H_z u)/∂t` of the momentum terms that
    /// move with the layer volume fluxes (advection) to `out.u` and `out.v`,
    /// which the splitter has turned into inventory tendencies, and overwrite
    /// `out.temp` and `out.salt` with the tracers' inventory tendencies
    /// `∂(H_z C)/∂t`, and `out.tke` and `out.gls` (sized like `state`'s, empty
    /// without a prognostic closure) with the turbulence's over the w-cells,
    /// `∂(H_w φ)/∂t` (zero to leave it to the columns). The layer fluxes are
    /// corrected to the barotropic transport of the step, `barotropic` (see
    /// [`crate::solver::rhs::transport_3d`]). `state` holds velocities and
    /// concentrations.
    fn transport_rhs_into(
        &self,
        state: &Solution3D,
        t: f64,
        barotropic: BarotropicFlux,
        out: &mut Solution3D,
    );

    /// Overwrite `g` with the slow forcing of the barotropic transport at time
    /// `t`: `(0, G_hu, G_hv)` in m²/s², from `state` and its explicit
    /// velocity tendency `rhs` ([`Self::momentum_rhs_into`]; the advection,
    /// which needs layer transports, is the implementation's to add).
    ///
    /// `G` must hold exactly the depth-integrated terms that the 2D module does
    /// not compute itself, so that nothing is counted twice.
    fn slow_forcing_into(
        &self,
        state: &Solution3D,
        rhs: &Solution3D,
        t: f64,
        g: &mut SWESolution2D,
    );

    /// Bottom drag of the step: overwrite `rate` (one entry per column,
    /// `[element][node]`) with the linear rate `r` (m/s) of the bottom stress,
    /// `τ_b/ρ₀ = r·u_b`, from `state` at `t`, and return `true`; `false` (the
    /// default) for none.
    ///
    /// The splitter freezes `r` over the step. It applies `−r·ū` to the depth
    /// mean in the barotropic pass, point-implicitly, and hands the rates to
    /// [`Self::vertical_implicit`]. [`Self::slow_forcing_into`] must add the
    /// rest of the drag on the depth mean, `−r·(u_b − ū)`.
    fn bottom_drag_into(&self, _state: &Solution3D, _t: f64, _rate: &mut [f64]) -> bool {
        false
    }

    /// Drag within the water column (net cages, see
    /// [`crate::physics::cage_drag`]): overwrite `rate` (one entry per layer,
    /// `[element][node][level]`) with the linear rates `λ_l` (1/s) of the
    /// momentum sink `−λ_l u_l` from `state` at `t`, and return `true`;
    /// `false` (the default) for none.
    ///
    /// The splitter freezes `λ` over the step. It applies `−Λ̄ ū`,
    /// `Λ̄ = Σ_l Δσ_l λ_l`, to the depth mean in the barotropic pass,
    /// point-implicitly, and hands the rates to [`Self::vertical_implicit`].
    /// [`Self::slow_forcing_into`] must add the rest of the drag on the
    /// depth mean, `−D Σ_l Δσ_l λ_l (u_l − ū)`.
    fn layer_drag_into(&self, _state: &Solution3D, _t: f64, _rate: &mut [f64]) -> bool {
        false
    }

    /// Implicit vertical terms over `[t, t + dt]`: vertical diffusion, with
    /// the surface and bottom stresses as its boundary fluxes, the bottom
    /// drag `r·u_b` at the new time if `drag.bottom` holds the rates `r` of
    /// [`Self::bottom_drag_into`], and the layer drag `λ_l u_l` at the new
    /// time if `drag.layers` holds the rates of [`Self::layer_drag_into`].
    fn vertical_implicit(&self, state: &mut Solution3D, t: f64, dt: f64, drag: StepDrag<'_>);

    /// Advect the stage value `stage` (inventories, as the stages carry
    /// them, with the stage's `η`) over `dt` with the implicit part of the
    /// vertical flux of the last [`Self::transport_rhs_into`], if any: after
    /// every 3D stage `u⁽ⁱ⁾ = Σ_k α_ik u⁽ᵏ⁾ + β_i Δt L(u⁽ⁱ⁻¹⁾)`, with
    /// `dt = β_i Δt` (see [`crate::physics::implicit_advection`]). The
    /// elements marked in `element_means` carry their fields as element
    /// means for the step. By default nothing.
    fn implicit_vertical_advection(
        &self,
        _stage: &mut Solution3D,
        _dt: f64,
        _element_means: &[bool],
    ) {
    }

    /// Runs on every 3D stage value (with the tracers as concentrations),
    /// including the last: limiters, density.
    fn post_stage(&self, state: &mut Solution3D);
}

/// The linearised drags of one baroclinic step, frozen at `tⁿ`
/// ([`ModeSplitPhysics::bottom_drag_into`],
/// [`ModeSplitPhysics::layer_drag_into`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct StepDrag<'a> {
    /// Bottom-drag rate `r` (m/s) of every column, `[element][node]`:
    /// `τ_b/ρ₀ = r·u_b`.
    pub bottom: Option<&'a [f64]>,
    /// Drag rate `λ_l` (1/s) of every layer, `[element][node][level]`:
    /// `∂u_l/∂t = −λ_l u_l`.
    pub layers: Option<&'a [f64]>,
}

impl<'a> StepDrag<'a> {
    /// No drag.
    pub const NONE: Self = Self {
        bottom: None,
        layers: None,
    };

    /// A bottom drag only.
    pub fn bottom(rate: &'a [f64]) -> Self {
        Self {
            bottom: Some(rate),
            layers: None,
        }
    }

    /// A layer drag only.
    pub fn layers(rate: &'a [f64]) -> Self {
        Self {
            bottom: None,
            layers: Some(rate),
        }
    }
}

/// Weights `w_j` such that `Σ w_j G(t_j)` is the average over `[tⁿ, tⁿ + Δt]`
/// of the polynomial through the stored values of `G`.
///
/// `tau[j] = t_j − tⁿ` for the newest first (`tau[0] = 0`), one to three
/// entries: constant, linear, or quadratic (Adams–Bashforth 1–3 with variable
/// steps). For a constant step and three values this is AB3,
/// `(23, −16, 5)/12`.
pub fn step_average_weights(tau: &[f64], dt: f64) -> [f64; 3] {
    assert!(
        (1..=3).contains(&tau.len()),
        "one to three past values, got {}",
        tau.len()
    );
    // Mean over x ∈ [0, dt] of Π (x − a) over the given roots
    let mean = |roots: &[f64]| match *roots {
        [] => 1.0,
        [a] => 0.5 * dt - a,
        [a, b] => dt * dt / 3.0 - 0.5 * (a + b) * dt + a * b,
        _ => unreachable!(),
    };
    let mut w = [0.0; 3];
    let mut roots = [0.0; 2];
    for j in 0..tau.len() {
        let mut n = 0;
        let mut denominator = 1.0;
        for (m, &tm) in tau.iter().enumerate() {
            if m != j {
                roots[n] = tm;
                n += 1;
                denominator *= tau[j] - tm;
            }
        }
        w[j] = mean(&roots[..n]) / denominator;
    }
    w
}

/// `G` at the last (up to) three baroclinic steps, newest first.
struct SlowForcingHistory {
    g: [SWESolution2D; 3],
    t: [f64; 3],
    len: usize,
    /// Start time of the next step if the run continues, for detecting restarts.
    next_t: f64,
}

impl SlowForcingHistory {
    fn new(ne: usize, nn: usize) -> Self {
        Self {
            g: std::array::from_fn(|_| SWESolution2D::new(ne, nn)),
            t: [0.0; 3],
            len: 0,
            next_t: f64::NAN,
        }
    }

    /// Make room for `G(t)` at slot 0. The history restarts when `t` does not
    /// continue the previous step.
    fn push(&mut self, t: f64, dt: f64) -> &mut SWESolution2D {
        if self.len > 0 && (t - self.next_t).abs() > 1e-9 * dt {
            self.len = 0;
        }
        self.g.rotate_right(1);
        self.t.rotate_right(1);
        self.t[0] = t;
        self.len = (self.len + 1).min(3);
        &mut self.g[0]
    }

    /// The history as a restart carries it.
    fn record(&self) -> SlowForcingRecord {
        SlowForcingRecord {
            g: self.g[..self.len].to_vec(),
            t: self.t[..self.len].to_vec(),
            next_t: self.next_t,
        }
    }

    /// Continue from `record` (see [`ModeSplitIntegrator::restore_slow_forcing`]).
    fn restore(&mut self, record: SlowForcingRecord) {
        let len = record.g.len();
        assert!(
            len <= 3 && record.t.len() == len,
            "a slow-forcing record holds up to three G with their times, got {len} G and {} times",
            record.t.len()
        );
        for (slot, g) in self.g.iter_mut().zip(record.g) {
            assert!(
                g.n_elements == slot.n_elements && g.n_nodes == slot.n_nodes,
                "the slow-forcing record is of another mesh"
            );
            *slot = g;
        }
        self.t[..len].copy_from_slice(&record.t);
        self.len = len;
        self.next_t = record.next_t;
    }

    /// Step average of `G` over `[t, t + dt]` into `out`.
    fn step_average(&mut self, dt: f64, out: &mut SWESolution2D) {
        let mut tau = [0.0; 3];
        for (tau_j, t_j) in tau.iter_mut().zip(&self.t[..self.len]) {
            *tau_j = t_j - self.t[0];
        }
        let w = step_average_weights(&tau[..self.len], dt);
        out.fill(0.0);
        for (g, &wj) in self.g[..self.len].iter().zip(&w) {
            out.axpy(wj, g);
        }
        self.next_t = self.t[0] + dt;
    }
}

/// The slow forcing `G` of the last (up to three) baroclinic steps and their
/// start times, newest first: what [`ModeSplitIntegrator`] carries from one
/// step to the next (its AB3 step average), and so what a restart must hold
/// besides the state (see [`crate::io::Restart3D`]).
#[derive(Clone)]
pub struct SlowForcingRecord {
    /// `G` of each step (`h` component zero), newest first.
    pub g: Vec<SWESolution2D>,
    /// The steps' start times (s), newest first.
    pub t: Vec<f64>,
    /// Start time (s) of the step that continues the run: a step starting
    /// elsewhere restarts the history. NaN before the first step.
    pub next_t: f64,
}

impl SlowForcingRecord {
    /// No history: the next step starts the AB3 average afresh, as the first
    /// step of a run does.
    pub fn empty() -> Self {
        Self {
            g: Vec::new(),
            t: Vec::new(),
            next_t: f64::NAN,
        }
    }
}

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
    /// `W_j = Σ_{m ≥ j} w_m`, the weight of substep `j`'s fluxes in the
    /// filtered state.
    secondary: Vec<f64>,
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
        let mut secondary = weights.clone();
        for m in (0..secondary.len().saturating_sub(1)).rev() {
            secondary[m] += secondary[m + 1];
        }
        Self {
            n_bt,
            weights,
            secondary,
        }
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

    /// Secondary weight `W_j = Σ_{m ≥ j} w_m` of substep `j`, at index `j − 1`.
    pub fn secondary_weights(&self) -> &[f64] {
        &self.secondary
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
    /// Horizontal momentum tendency `R₃D` at `tⁿ`, reused as the first 3D stage.
    rhs_n: Solution3D,
    /// A 3D stage value with velocities and concentrations (the stages carry
    /// inventories `H_z u`, `H_z C`).
    concentrations: Solution3D,
    /// The last values of the inventory fields ([`InventoryFields`]): what a
    /// node or element without water keeps.
    last_values: InventoryFields,
    /// Elements where the pass kept only the element balance: their
    /// inventory fields are element means per level for the step.
    element_means: Vec<bool>,
    /// σ-thicknesses of the w-cells, for the turbulence's inventories.
    d_sigma_w: Vec<f64>,
    /// `∇·DU_avg2` of the step.
    transport_divergence: DGSolution2D,
    /// Barotropic transport during the pass.
    q: SWESolution2D,
    /// Its first two SSP-RK3 stage values, and a stage's 2D RHS.
    u1: SWESolution2D,
    u2: SWESolution2D,
    k_2d: SWESolution2D,
    /// Filtered transport.
    q_avg: SWESolution2D,
    /// Step-averaged slow forcing `G` of the transport (zero for `h`).
    forcing: SWESolution2D,
    /// `G` of the last steps, for the step average.
    history: SlowForcingHistory,
    /// DU_avg2 of the last step.
    transport: BarotropicTransport,
    /// Face mass fluxes of one 2D RHS evaluation.
    face_mass: Vec<f64>,
    /// Subcell interface mass fluxes of one 2D RHS evaluation.
    subcell_mass: Vec<f64>,
    /// Depth means of the u/v tendency (or of u/v after diffusion).
    mean_u: DGSolution2D,
    mean_v: DGSolution2D,
    /// Constant barotropic rates over the step: of `η`, `ū`, `v̄`, and of
    /// the transport `Dū`, `Dv̄`.
    rate_eta: DGSolution2D,
    rate_ubar: DGSolution2D,
    rate_vbar: DGSolution2D,
    rate_hu: DGSolution2D,
    rate_hv: DGSolution2D,
    /// Bottom-drag rate `r` of every column, frozen over the step.
    drag_rate: Vec<f64>,
    /// Layer-drag rate `λ_l` of every layer, frozen over the step (sized on
    /// the first step with a layer drag).
    layer_drag_rate: Vec<f64>,
    /// Its column mean `Λ̄ = Σ_l Δσ_l λ_l`.
    column_drag_rate: Vec<f64>,
}

impl Buffers {
    fn new(state: &Solution3D, ops: &DGOperators2D, n_rivers: usize) -> Self {
        let (ne, nn) = (state.n_elements, state.n_nodes);
        let n_face_values = ne * 4 * ops.n_face_nodes;
        let n_subcell_values = ne * subcell_interfaces(ops.n_1d);
        Self {
            rhs_n: Solution3D::new(ne, nn, state.n_levels),
            concentrations: Solution3D::new(ne, nn, state.n_levels),
            last_values: InventoryFields::of(state),
            element_means: vec![false; ne],
            d_sigma_w: vec![0.0; state.n_levels + 1],
            transport_divergence: DGSolution2D::new(ne, nn),
            q: SWESolution2D::new(ne, nn),
            u1: SWESolution2D::new(ne, nn),
            u2: SWESolution2D::new(ne, nn),
            k_2d: SWESolution2D::new(ne, nn),
            q_avg: SWESolution2D::new(ne, nn),
            forcing: SWESolution2D::new(ne, nn),
            history: SlowForcingHistory::new(ne, nn),
            transport: BarotropicTransport::new(ne, nn, n_face_values, n_subcell_values, n_rivers),
            face_mass: vec![0.0; n_face_values],
            subcell_mass: vec![0.0; n_subcell_values],
            mean_u: DGSolution2D::new(ne, nn),
            mean_v: DGSolution2D::new(ne, nn),
            rate_eta: DGSolution2D::new(ne, nn),
            rate_ubar: DGSolution2D::new(ne, nn),
            rate_vbar: DGSolution2D::new(ne, nn),
            rate_hu: DGSolution2D::new(ne, nn),
            rate_hv: DGSolution2D::new(ne, nn),
            drag_rate: vec![0.0; ne * nn],
            layer_drag_rate: Vec::new(),
            column_drag_rate: Vec::new(),
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
    /// A history to continue from, applied at the next step (see
    /// [`Self::restore_slow_forcing`]).
    restored_slow_forcing: Option<SlowForcingRecord>,
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
            restored_slow_forcing: None,
        }
    }

    /// CFL number of the barotropic substeps, capped by the 2D module's
    /// forward-Euler bounds (e.g. its wet/dry positivity bound, see
    /// [`PhysicsModule::compute_dt_ssp`]).
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

    /// The barotropic transport (DU_avg2) of the last step, if any.
    pub fn barotropic_transport(&self) -> Option<&BarotropicTransport> {
        self.buffers.as_ref().map(|b| &b.transport)
    }

    /// The slow forcing the next step's AB3 average continues from (empty
    /// before the first step): with the state, what a restart needs to
    /// continue the run bit for bit.
    pub fn slow_forcing_record(&self) -> SlowForcingRecord {
        match (&self.restored_slow_forcing, &self.buffers) {
            (Some(record), _) => record.clone(),
            (None, Some(buffers)) => buffers.history.record(),
            (None, None) => SlowForcingRecord::empty(),
        }
    }

    /// Continue the AB3 average of `G` from `record` (from
    /// [`Self::slow_forcing_record`] of the run being resumed) at the next
    /// step, in place of whatever history this integrator has.
    pub fn restore_slow_forcing(&mut self, record: SlowForcingRecord) {
        self.restored_slow_forcing = Some(record);
    }

    /// Barotropic substeps per baroclinic step used by the last step (0 before
    /// the first step).
    pub fn last_substeps(&self) -> usize {
        self.filter.as_ref().map_or(0, BarotropicFilter::n_bt)
    }

    /// Perform one baroclinic (3D) time step of `state` from `t` to `t + dt`.
    pub fn step<P: ModeSplitPhysics>(
        &mut self,
        state: &mut Solution3D,
        physics: &P,
        dt: f64,
        t: f64,
    ) {
        let sigma = physics.sigma();
        let bathymetry = physics.bathymetry();
        let barotropic = physics.barotropic();
        let rivers = physics.rivers();
        let n_rivers = rivers.map_or(0, RiverSources::len);
        let Buffers {
            rhs_n,
            concentrations,
            last_values: lent_values,
            element_means,
            d_sigma_w,
            transport_divergence,
            q,
            u1,
            u2,
            k_2d,
            q_avg,
            forcing: g_term,
            history,
            transport,
            face_mass,
            subcell_mass,
            mean_u,
            mean_v,
            rate_eta,
            rate_ubar,
            rate_vbar,
            rate_hu,
            rate_hv,
            drag_rate,
            layer_drag_rate,
            column_drag_rate,
        } = self
            .buffers
            .get_or_insert_with(|| Buffers::new(state, barotropic.operators(), n_rivers));
        assert_eq!(
            transport.discharge.len(),
            n_rivers,
            "the number of rivers changed between steps"
        );
        if let Some(record) = self.restored_slow_forcing.take() {
            history.restore(record);
        }
        let nn = state.n_nodes;

        // 1. Slow forcing: Gⁿ from R₃D at tⁿ, averaged over the step (AB3);
        // the drag rates of the step
        let bottom_drag = physics.bottom_drag_into(state, t, drag_rate);
        let drag_rate: &[f64] = drag_rate;
        let n_layer_values = state.u.len();
        if layer_drag_rate.len() != n_layer_values {
            layer_drag_rate.resize(n_layer_values, 0.0);
        }
        let layer_drag = physics.layer_drag_into(state, t, layer_drag_rate);
        if layer_drag {
            column_drag_rate.resize(state.eta.data.len(), 0.0);
            for (mean, rates) in column_drag_rate
                .iter_mut()
                .zip(layer_drag_rate.chunks_exact(state.n_levels))
            {
                *mean = rates
                    .iter()
                    .zip(sigma.d_sigma())
                    .map(|(r, ds)| r * ds)
                    .sum();
            }
        }
        let drag = StepDrag {
            bottom: bottom_drag.then_some(drag_rate),
            layers: layer_drag.then_some(&layer_drag_rate[..]),
        };
        let column_drag = layer_drag.then_some(&column_drag_rate[..]);
        physics.momentum_rhs_into(state, t, rhs_n);
        physics.slow_forcing_into(state, rhs_n, t, history.push(t, dt));
        history.step_average(dt, g_term);
        to_transport(state, bathymetry, q);

        // 2. One filtered barotropic pass
        let dt_bt_max = barotropic.compute_dt_ssp(q, self.barotropic_cfl, Some(SspScheme::Rk3));
        assert!(
            dt_bt_max > 0.0,
            "barotropic time step {dt_bt_max} is not positive"
        );
        // A blown-up state (wave speeds of 1e100 m/s) would ask for more
        // substeps than memory holds ("capacity overflow" in the filter)
        let substeps = (dt / dt_bt_max).ceil();
        assert!(
            substeps <= MAX_BAROTROPIC_SUBSTEPS as f64,
            "{substeps:.3e} barotropic substeps for a {dt} s step (barotropic step \
             {dt_bt_max:.3e} s): the state has blown up"
        );
        let n_bt = (substeps as usize).max(self.min_substeps);
        if self.filter.as_ref().is_none_or(|f| f.n_bt() != n_bt) {
            self.filter = Some(BarotropicFilter::new(n_bt));
        }
        let filter = self.filter.as_ref().expect("filter was just set");
        let dt_bt = dt / n_bt as f64;

        q_avg.fill(0.0);
        transport.clear();
        let stage_drag = StageDrag {
            bottom: drag.bottom,
            column: column_drag,
        };
        // The filter's weight of the state the next substep starts from
        let mut pending_weight = None;
        for (m, &w_secondary) in filter.secondary_weights().iter().enumerate() {
            let t_m = t + m as f64 * dt_bt;
            for stage in SSP_RK3_STAGES {
                let time = t_m + stage.time * dt_bt;
                // The stage's start value `x`, the value it writes, and `qⁿ`
                // for the second stage's combination. The third stage writes
                // `qⁿ⁺¹` in place, reading each node's `qⁿ` before
                let (x, target, start) = match stage.input {
                    StageInput::Start => (&*q, &mut *u1, None),
                    StageInput::First => (&*u1, &mut *u2, Some(&*q)),
                    StageInput::Second => (&*u2, &mut *q, None),
                };
                barotropic.compute_rhs_mass_fluxes_into(x, time, k_2d, face_mass, subcell_mass);
                let c = w_secondary * stage.weight / n_bt as f64;
                if let Some(rivers) = rivers {
                    for (i, mean) in transport.discharge.iter_mut().enumerate() {
                        let discharge = rivers.discharge(i, time);
                        let rate = discharge * rivers.inv_area(i);
                        let k = rivers.element(i).as_usize();
                        k_2d.data[SWE_VAR_H][k * nn..(k + 1) * nn]
                            .iter_mut()
                            .for_each(|h| *h += rate);
                        *mean += c * discharge;
                    }
                }
                let rates = StageRates {
                    rhs: k_2d,
                    forcing: g_term,
                    face_mass,
                    subcell_mass,
                    drag: stage_drag,
                };
                let weights = PassWeights {
                    transport: c,
                    average: pending_weight.take(),
                };
                barotropic_stage(
                    stage, x, start, rates, dt_bt, weights, target, transport, q_avg,
                );
                barotropic.implicit_damping(target, x, stage.dt_fraction * dt_bt);
                barotropic.post_process(target);
            }
            pending_weight = Some(filter.weights()[m]);
        }
        if let Some(w) = pending_weight {
            q_avg.axpy(w, q);
        }
        // The early weights are negative: restore positivity of the average
        barotropic.post_process(q_avg);

        // Constant barotropic rates over the step
        for k in 0..state.n_elements {
            let bed = bathymetry.element(ElementIndex::new(k));
            for (i, &b) in bed.iter().enumerate() {
                let idx = k * state.n_nodes + i;
                let (eta, ubar, vbar) = filtered_barotropic_state(q_avg, idx, b);
                let (depth, depth_n) = (eta - b, state.eta.data[idx] - b);
                rate_eta.data[idx] = (eta - state.eta.data[idx]) / dt;
                rate_ubar.data[idx] = (ubar - state.ubar.data[idx]) / dt;
                rate_vbar.data[idx] = (vbar - state.vbar.data[idx]) / dt;
                rate_hu.data[idx] = (depth * ubar - depth_n * state.ubar.data[idx]) / dt;
                rate_hv.data[idx] = (depth * vbar - depth_n * state.vbar.data[idx]) / dt;
            }
        }

        // 3. 3D stages with the barotropic state prescribed, the velocity and
        // the tracers as inventories H_z·u, H_z·C (the stage's η sets H_z)
        let river_inflow = rivers.map(|rivers| RiverInflow {
            rivers,
            discharge: &transport.discharge,
        });
        let barotropic_flux = BarotropicFlux {
            dt,
            hu: &transport.hu.data,
            hv: &transport.hv.data,
            face: &transport.face,
            subcell: &transport.subcell,
            eta_rate: &rate_eta.data,
            rivers: river_inflow,
        };
        let geom = barotropic.geometry();
        // Elements where the pass broke the nodal identity ∂η/∂t = −∇·DU_avg2
        // (WetDry subcells, positivity limiter) carry their tracers as element
        // means for the step
        transport.divergence_into(
            barotropic.operators(),
            geom,
            barotropic.metric_form(),
            transport_divergence,
        );
        for (k, mark) in element_means.iter_mut().enumerate() {
            let nodes = k * nn..(k + 1) * nn;
            let bed = bathymetry.element(ElementIndex::new(k));
            let source = river_inflow.map_or(0.0, |r| r.volume_rate(k));
            let (mut residual, mut scale, mut depth) = (0.0_f64, 0.0_f64, 0.0_f64);
            // The largest relative change of a column's volume that its layers
            // would not carry (a tracer's constancy error there)
            let mut constancy = 0.0_f64;
            for ((&rate, &div), (&eta, &b)) in rate_eta.data[nodes.clone()]
                .iter()
                .zip(&transport_divergence.data[nodes.clone()])
                .zip(state.eta.data[nodes].iter().zip(bed))
            {
                let r = (rate + div - source).abs();
                residual = residual.max(r);
                scale = scale.max(rate.abs()).max(div.abs()).max(source);
                depth = depth.max(eta - b);
                constancy = constancy.max(r * dt / (eta - b).max(MIN_COLUMN_VOLUME_DEPTH));
            }
            // Relative to the flow, and not round-off of a fluid at rest; or
            // large against a thin column's own water (a film at a wetting
            // front, whose tracers it would otherwise skew)
            *mark = (residual > NODAL_IDENTITY_TOLERANCE * scale
                && residual * dt > DEPTH_ROUND_OFF * depth)
                || constancy > CONSTANCY_TOLERANCE;
        }
        let means: &[bool] = element_means;
        w_cell_thicknesses(sigma.d_sigma(), d_sigma_w);
        // The σ-thicknesses of the cells of each inventory field
        let cells = [sigma.d_sigma(), &d_sigma_w[..]];
        let cells_of = |field: usize| cells[usize::from(field >= N_LAYER_FIELDS)];
        // Inventories of `s` → velocities and concentrations in `out` (which
        // holds the last values, kept where there is no water)
        let to_values = |s: &Solution3D, out: &mut InventoryFields| {
            let eta = &s.eta.data;
            let fields = inventory_fields(s).into_iter().zip(out.fields_mut());
            for (field, (q, out)) in fields.enumerate() {
                if !q.is_empty() {
                    from_inventory(q, eta, cells_of(field), bathymetry, geom, means, out);
                }
            }
        };
        let to_inventories = |s: &mut Solution3D| {
            let eta = &s.eta.data;
            let fields = [
                &mut s.u,
                &mut s.v,
                &mut s.temp,
                &mut s.salt,
                &mut s.tke,
                &mut s.gls,
            ];
            for (field, q) in fields.into_iter().enumerate() {
                if !q.is_empty() {
                    to_inventory(q, eta, cells_of(field), bathymetry);
                }
            }
        };
        // Lent to both stage closures for the step
        let last_values_cell = RefCell::new(std::mem::take(lent_values));
        last_values_cell.borrow_mut().copy_from_state(state);
        let last_values = &last_values_cell;
        to_inventories(state);
        let mut first_stage = true;
        SSPRK3.step_with_relaxation(
            state,
            dt,
            t,
            |s, time, out| {
                // The inventory fields are converted from `s` next
                copy_non_inventory_fields(s, concentrations);
                {
                    let mut last = last_values.borrow_mut();
                    to_values(s, &mut last);
                    last.copy_to_state(concentrations);
                }
                if first_stage {
                    // R₃D is u, v; the other fields of `out` are written below
                    update_with(&mut out.u, &rhs_n.u, |x, y| *x = y);
                    update_with(&mut out.v, &rhs_n.v, |x, y| *x = y);
                    first_stage = false;
                } else {
                    physics.momentum_rhs_into(concentrations, time, out);
                }
                // Velocity tendencies → inventory tendencies, then the
                // advection's
                to_inventory(&mut out.u, &s.eta.data, sigma.d_sigma(), bathymetry);
                to_inventory(&mut out.v, &s.eta.data, sigma.d_sigma(), bathymetry);
                out.tke.resize(s.tke.len(), 0.0);
                out.gls.resize(s.gls.len(), 0.0);
                physics.transport_rhs_into(concentrations, time, barotropic_flux, out);
                // w and rho are diagnostics, refreshed after the stages
                update_values(&mut out.w, |x| *x = 0.0);
                update_values(&mut out.rho, |x| *x = 0.0);
                set_column_sums(sigma, &mut out.u, rate_hu);
                set_column_sums(sigma, &mut out.v, rate_hv);
                out.eta.copy_from(rate_eta);
                out.ubar.copy_from(rate_ubar);
                out.vbar.copy_from(rate_vbar);
            },
            // The implicit part of the vertical advection, on the stage's
            // inventories and thicknesses
            |s, _, stage_dt| physics.implicit_vertical_advection(s, stage_dt, means),
            |s| {
                let mut last = last_values.borrow_mut();
                to_values(s, &mut last);
                last.copy_to_state(s);
                physics.post_stage(s);
                last.copy_from_state(s);
                to_inventories(s);
            },
            &mut self.stages_3d,
        );

        // The RK combination of the constant rates reproduces the filtered
        // state only up to round-off, which can leave a dry node a hair below
        // its bed (η < B) and the next pass with a negative depth. Take it
        // exactly: for h̄ ≥ 0, (h̄ + B) − B ≥ 0 in floating point.
        for k in 0..state.n_elements {
            let bed = bathymetry.element(ElementIndex::new(k));
            for (i, &b) in bed.iter().enumerate() {
                let idx = k * state.n_nodes + i;
                let (eta, ubar, vbar) = filtered_barotropic_state(q_avg, idx, b);
                state.eta.data[idx] = eta;
                state.ubar.data[idx] = ubar;
                state.vbar.data[idx] = vbar;
            }
        }
        // Inventories back to velocities and concentrations, with the new H_z
        {
            let mut last = last_values.borrow_mut();
            to_values(state, &mut last);
            last.copy_to_state(state);
        }
        *lent_values = last_values_cell.into_inner();

        // 4. The implicit vertical terms change the depth mean through the
        // surface and bottom stresses, which G has already given to the
        // barotropic mode: reset it to ū.
        physics.vertical_implicit(state, t, dt, drag);
        depth_average(sigma, &state.u, mean_u);
        depth_average(sigma, &state.v, mean_v);
        shift_columns(&mut state.u, state.n_levels, mean_u, &state.ubar);
        shift_columns(&mut state.v, state.n_levels, mean_v, &state.vbar);
    }
}

/// The start value of an SSP-RK3 stage of a barotropic substep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StageInput {
    /// `qⁿ`, the substep's start.
    Start,
    /// The first stage value `u₁`.
    First,
    /// The second stage value `u₂`.
    Second,
}

/// An SSP-RK3 stage of a barotropic substep (Shu–Osher form):
///
/// ```text
/// u₁   = qⁿ + Δt·L(qⁿ)
/// u₂   = ¾qⁿ + ¼u₁ + ¼Δt·L(u₁)
/// qⁿ⁺¹ = ⅓qⁿ + ⅔u₂ + ⅔Δt·L(u₂)
/// ```
#[derive(Clone, Copy, Debug)]
struct RkStage {
    input: StageInput,
    /// Stage time, as a fraction of the substep.
    time: f64,
    /// Weight of the stage's RHS in the substep (Butcher form,
    /// `qⁿ⁺¹ = qⁿ + Δt(k₁ + k₂ + 4k₃)/6`): the weight of its fluxes in the
    /// transport.
    weight: f64,
    /// Length of the stage's forward-Euler step, as a fraction of the
    /// substep: the step of its point-implicit damping.
    dt_fraction: f64,
}

/// The three stages of [`SSPRK3`], in order.
const SSP_RK3_STAGES: [RkStage; 3] = [
    RkStage {
        input: StageInput::Start,
        time: 0.0,
        weight: 1.0 / 6.0,
        dt_fraction: 1.0,
    },
    RkStage {
        input: StageInput::First,
        time: 1.0,
        weight: 1.0 / 6.0,
        dt_fraction: 0.25,
    },
    RkStage {
        input: StageInput::Second,
        time: 0.5,
        weight: 2.0 / 3.0,
        dt_fraction: 2.0 / 3.0,
    },
];

/// The depth-mean part of the 3D model's drags, frozen over the step:
/// `bottom` holds the bottom-drag rate `r` (m/s), `column` the column mean
/// `Λ̄` (1/s) of the layer drag, per column.
#[derive(Clone, Copy)]
struct StageDrag<'a> {
    bottom: Option<&'a [f64]>,
    column: Option<&'a [f64]>,
}

/// The rates of a barotropic stage: the 2D module's RHS (with the rivers),
/// the slow forcing `G` added to it, the face mass fluxes of the RHS, and
/// the drag.
#[derive(Clone, Copy)]
struct StageRates<'a> {
    rhs: &'a SWESolution2D,
    forcing: &'a SWESolution2D,
    face_mass: &'a [f64],
    subcell_mass: &'a [f64],
    drag: StageDrag<'a>,
}

/// What a barotropic stage adds to the pass's sums: its start value's
/// transport and its face fluxes with weight `transport` to DU_avg2, and,
/// at the first stage of a substep, its start value (the state after the
/// previous substep) with the filter's weight `average` to the average.
#[derive(Clone, Copy)]
struct PassWeights {
    transport: f64,
    average: Option<f64>,
}

/// One SSP-RK3 stage of a barotropic substep of `dt` from the stage's start
/// value `x` (and `qⁿ`, `start`, for the second stage), in one element-block
/// pass:
/// - the stage combination with the rate `L(x) = RHS + G` into `target`, as
///   [`SSPRK3`] evaluates it, bit for bit; the third stage writes `qⁿ⁺¹`
///   over `qⁿ` (`target` holds `qⁿ`);
/// - the depth-mean part of the drags, `∂(hu, hv)/∂t = −ρ·(hu, hv)` with
///   `ρ = r/h + Λ̄`, point-implicitly with the new depth: `(hu, hv) ←
///   (hu, hv)/(1 + dt_s·ρ)`, `dt_s` the stage's forward-Euler step (it only
///   ever shrinks the transport);
/// - the stage's contributions to DU_avg2 and to the filtered average
///   ([`PassWeights`]).
///
/// One pass instead of a serial sweep per operation: the 2D state is
/// streamed once per stage, on every thread.
fn barotropic_stage(
    stage: RkStage,
    x: &SWESolution2D,
    start: Option<&SWESolution2D>,
    rates: StageRates<'_>,
    dt: f64,
    weights: PassWeights,
    target: &mut SWESolution2D,
    transport: &mut BarotropicTransport,
    average: &mut SWESolution2D,
) {
    let (ne, nn) = (x.n_elements, x.n_nodes);
    let n_face = rates.face_mass.len() / ne.max(1);
    let n_sub = rates.subcell_mass.len() / ne.max(1);
    let dt_stage = stage.dt_fraction * dt;
    let [h, hu, hv] = &mut target.data;
    let [avg_h, avg_hu, avg_hv] = &mut average.data;
    let outputs = [
        &mut h[..],
        hu,
        hv,
        &mut transport.hu.data[..],
        &mut transport.hv.data[..],
        &mut transport.face[..],
        &mut transport.subcell[..],
        avg_h,
        avg_hu,
        avg_hv,
    ];
    for_each_block(
        ne,
        outputs,
        || (),
        |_,
         k,
         [
            h,
            hu,
            hv,
            tr_hu,
            tr_hv,
            tr_face,
            tr_subcell,
            avg_h,
            avg_hu,
            avg_hv,
        ]| {
            let nodes = k * nn..(k + 1) * nn;
            for (var, new) in [&mut *h, &mut *hu, &mut *hv].into_iter().enumerate() {
                let x = &x.data[var][nodes.clone()];
                let rhs = &rates.rhs.data[var][nodes.clone()];
                let forcing = &rates.forcing.data[var][nodes.clone()];
                let rate = |i: usize| rhs[i] + forcing[i];
                match stage.input {
                    StageInput::Start => {
                        for (i, (new, &x)) in new.iter_mut().zip(x).enumerate() {
                            *new = x + dt * rate(i);
                        }
                    }
                    StageInput::First => {
                        let start = start.expect("the second stage combines with qⁿ");
                        let start = &start.data[var][nodes.clone()];
                        for (i, (new, (&x, &q))) in
                            new.iter_mut().zip(x.iter().zip(start)).enumerate()
                        {
                            *new = q * 0.75 + 0.25 * x + 0.25 * dt * rate(i);
                        }
                    }
                    StageInput::Second => {
                        for (i, (new, &x)) in new.iter_mut().zip(x).enumerate() {
                            *new = *new * (1.0 / 3.0) + 2.0 / 3.0 * x + 2.0 / 3.0 * dt * rate(i);
                        }
                    }
                }
            }
            let drag = rates.drag;
            if drag.bottom.is_some() || drag.column.is_some() {
                let columns = h.iter().zip(hu.iter_mut()).zip(hv.iter_mut());
                for (i, ((&h, hu), hv)) in columns.enumerate() {
                    if h <= 0.0 {
                        continue;
                    }
                    let idx = k * nn + i;
                    let rate = drag.bottom.map_or(0.0, |r| r[idx] / h)
                        + drag.column.map_or(0.0, |c| c[idx]);
                    if rate > 0.0 {
                        let factor = 1.0 / (1.0 + dt_stage * rate);
                        *hu *= factor;
                        *hv *= factor;
                    }
                }
            }
            let c = weights.transport;
            for (sum, var) in [(tr_hu, SWE_VAR_HU), (tr_hv, SWE_VAR_HV)] {
                for (a, &b) in sum.iter_mut().zip(&x.data[var][nodes.clone()]) {
                    *a += c * b;
                }
            }
            let faces = &rates.face_mass[k * n_face..(k + 1) * n_face];
            for (a, &b) in tr_face.iter_mut().zip(faces) {
                *a += c * b;
            }
            // NaN (a stage without subcells) stays NaN
            let subcells = &rates.subcell_mass[k * n_sub..(k + 1) * n_sub];
            for (a, &b) in tr_subcell.iter_mut().zip(subcells) {
                *a += c * b;
            }
            if let Some(w) = weights.average {
                for (var, sum) in [avg_h, avg_hu, avg_hv].into_iter().enumerate() {
                    for (a, &b) in sum.iter_mut().zip(&x.data[var][nodes.clone()]) {
                        *a += w * b;
                    }
                }
            }
        },
    );
}

/// `(η, ū, v̄)` at node `idx` of the filtered transport, over bed `b`
/// (`ū = 0` where the filtered depth is not positive).
fn filtered_barotropic_state(q_avg: &SWESolution2D, idx: usize, b: f64) -> (f64, f64, f64) {
    let h = q_avg.data[SWE_VAR_H][idx];
    if h > 0.0 {
        let inv_h = 1.0 / h;
        (
            h + b,
            q_avg.data[SWE_VAR_HU][idx] * inv_h,
            q_avg.data[SWE_VAR_HV][idx] * inv_h,
        )
    } else {
        (h + b, 0.0, 0.0)
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
    let (nl, nn) = (sigma.n_levels(), out.n_nodes);
    for_each_block(
        out.n_elements,
        [&mut out.data[..]],
        || (),
        |_, k, [means]| {
            let columns = &field[k * nn * nl..(k + 1) * nn * nl];
            for (mean, column) in means.iter_mut().zip(columns.chunks_exact(nl)) {
                *mean = sigma.depth_average(column);
            }
        },
    );
}

/// Replace the sum of every column of an inventory tendency by `rate`,
/// sharing the difference out by the layer fractions `Δσ_l`.
fn set_column_sums(sigma: &SigmaGrid, field: &mut [f64], rate: &DGSolution2D) {
    let (d_sigma, nn) = (sigma.d_sigma(), rate.n_nodes);
    for_each_block(
        rate.n_elements,
        [field],
        || (),
        |_, k, [columns]| {
            let rates = &rate.data[k * nn..(k + 1) * nn];
            for (column, &r) in columns.chunks_exact_mut(d_sigma.len()).zip(rates) {
                let difference = r - column.iter().sum::<f64>();
                for (x, &ds) in column.iter_mut().zip(d_sigma) {
                    *x += ds * difference;
                }
            }
        },
    );
}

/// Copy the fields of `from` that the 3D stages do not carry as inventories
/// (the barotropic state, the diagnostics) into `to`, and size `to`'s
/// turbulence like `from`'s; `to`'s inventory fields are left as they are.
fn copy_non_inventory_fields(from: &Solution3D, to: &mut Solution3D) {
    to.eta.copy_from(&from.eta);
    to.ubar.copy_from(&from.ubar);
    to.vbar.copy_from(&from.vbar);
    for (x, y) in [
        (&mut to.w, &from.w),
        (&mut to.rho, &from.rho),
        (&mut to.eddy_viscosity, &from.eddy_viscosity),
        (&mut to.eddy_diffusivity, &from.eddy_diffusivity),
    ] {
        copy_field(x, y);
    }
    to.tke.resize(from.tke.len(), 0.0);
    to.gls.resize(from.gls.len(), 0.0);
}

/// `to = from`, in parallel when the lengths agree.
fn copy_field(to: &mut Vec<f64>, from: &[f64]) {
    if to.len() == from.len() {
        update_with(to, from, |x, y| *x = y);
    } else {
        to.clear();
        to.extend_from_slice(from);
    }
}

/// The fields a 3D step carries as inventories `H φ`: `u, v, temp, salt`
/// over the layers, then the turbulence `tke, gls` over the w-cells (empty
/// without a prognostic closure).
fn inventory_fields(state: &Solution3D) -> [&Vec<f64>; 6] {
    [
        &state.u,
        &state.v,
        &state.temp,
        &state.salt,
        &state.tke,
        &state.gls,
    ]
}

/// How many of [`inventory_fields`] live on the layers.
const N_LAYER_FIELDS: usize = 4;

/// Their values, kept between the stages of a step.
#[derive(Default)]
struct InventoryFields([Vec<f64>; 6]);

impl InventoryFields {
    fn of(state: &Solution3D) -> Self {
        let mut fields = Self::default();
        fields.copy_from_state(state);
        fields
    }

    fn fields_mut(&mut self) -> &mut [Vec<f64>; 6] {
        &mut self.0
    }

    /// The turbulence is allocated by the closure's first step, so its
    /// length can change between steps.
    fn copy_from_state(&mut self, state: &Solution3D) {
        for (a, b) in self.0.iter_mut().zip(inventory_fields(state)) {
            copy_field(a, b);
        }
    }

    fn copy_to_state(&self, state: &mut Solution3D) {
        let [u, v, temp, salt, tke, gls] = &self.0;
        let fields = [
            (&mut state.u, u),
            (&mut state.v, v),
            (&mut state.temp, temp),
            (&mut state.salt, salt),
            (&mut state.tke, tke),
            (&mut state.gls, gls),
        ];
        for (to, from) in fields {
            assert_eq!(to.len(), from.len(), "inventory field lengths");
            update_with(to, from, |x, y| *x = y);
        }
    }
}

/// Shift every column uniformly so its depth mean goes from `from` to `to`.
/// The σ-layer fractions sum to one, so a uniform shift changes the depth mean
/// by exactly the shift.
fn shift_columns(field: &mut [f64], n_levels: usize, from: &DGSolution2D, to: &DGSolution2D) {
    let nn = from.n_nodes;
    for_each_block(
        from.n_elements,
        [field],
        || (),
        |_, k, [columns]| {
            let nodes = k * nn..(k + 1) * nn;
            let shifts = from.data[nodes.clone()].iter().zip(&to.data[nodes]);
            for (column, (&a, &b)) in columns.chunks_exact_mut(n_levels).zip(shifts) {
                let shift = b - a;
                column.iter_mut().for_each(|x| *x += shift);
            }
        },
    );
}

impl IntegratorInfo for ModeSplitIntegrator {
    fn name(&self) -> &'static str {
        "mode-split-filtered-ssprk3"
    }

    /// Second order: the slow coupling is third order (AB3) and the 3D stages
    /// see the barotropic state interpolated linearly. The filter's residual
    /// barotropic damping is formally first order but ≈ 1e-3 times smaller
    /// than a plain average's (see [`BarotropicFilter`]).
    fn order(&self) -> usize {
        2
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
    use crate::time::Integrable;

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
        assert_eq!(integrator.order(), 2);
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

    /// AB3 for a constant step; exact step averages of quadratics for any
    /// steps; the lower orders at the start of a run.
    #[test]
    fn step_average_weights_integrate_polynomials_exactly() {
        let dt = 7.0;
        let w = step_average_weights(&[0.0, -dt, -2.0 * dt], dt);
        for (wj, ab3) in w.iter().zip([23.0, -16.0, 5.0]) {
            assert!((wj - ab3 / 12.0).abs() < 1e-14, "{w:?}");
        }
        let w = step_average_weights(&[0.0, -dt], dt);
        assert!((w[0] - 1.5).abs() < 1e-14 && (w[1] + 0.5).abs() < 1e-14);
        assert_eq!(step_average_weights(&[0.0], dt), [1.0, 0.0, 0.0]);

        // Variable steps: the step average of any quadratic is exact
        let tau = [0.0, -3.0, -11.0];
        let dt = 5.0;
        let w = step_average_weights(&tau, dt);
        let p = |x: f64| 2.0 - 0.3 * x + 0.07 * x * x;
        let exact = (2.0 * dt - 0.15 * dt * dt + 0.07 * dt.powi(3) / 3.0) / dt;
        let approx: f64 = tau.iter().zip(&w).map(|(t, w)| w * p(*t)).sum();
        assert!((approx - exact).abs() < 1e-12, "{approx} vs {exact}");
    }

    /// The fused barotropic stages ([`barotropic_stage`]) give, bit for bit,
    /// what [`SSPRK3`] with the drag as relaxation and separate sweeps for
    /// `+ G`, the transport and the filtered average give: over two substeps
    /// of a nonlinear rate, with a bottom and a layer drag and a dry node.
    #[test]
    fn fused_barotropic_stages_match_ssp_rk3_bit_for_bit() {
        let (ne, nn, n_face) = (3, 4, 8);
        let value = |seed: usize| ((seed as f64 * 0.618_034).fract() - 0.3) * 2.0;
        let mut q0 = SWESolution2D::new(ne, nn);
        let mut g = SWESolution2D::new(ne, nn);
        for var in 0..3 {
            for i in 0..ne * nn {
                q0.data[var][i] = value(7 * i + var) + if var == SWE_VAR_H { 1.0 } else { 0.0 };
                if var != SWE_VAR_H {
                    g.data[var][i] = 0.1 * value(11 * i + var);
                }
            }
        }
        q0.data[SWE_VAR_H][5] = 0.0;
        let bottom: Vec<f64> = (0..ne * nn).map(|i| 0.01 * value(3 * i).abs()).collect();
        let column: Vec<f64> = (0..ne * nn).map(|i| 1e-3 * value(5 * i).abs()).collect();
        let drag = StageDrag {
            bottom: Some(&bottom),
            column: Some(&column),
        };
        // A nonlinear 2D "RHS" and face fluxes depending on the state
        let rhs = |s: &SWESolution2D, time: f64, out: &mut SWESolution2D, face: &mut [f64]| {
            for var in 0..3 {
                for i in 0..ne * nn {
                    let x = s.data[var][i];
                    out.data[var][i] =
                        -0.3 * x * s.data[SWE_VAR_H][i] + 0.05 * (time + i as f64).sin();
                }
            }
            for (j, f) in face.iter_mut().enumerate() {
                *f = s.data[SWE_VAR_HU][j % (ne * nn)] - 0.5 * s.data[SWE_VAR_H][j % (ne * nn)];
            }
        };
        let (dt, w_avg, w_secondary) = (0.7, [0.4, 0.6], [1.0, 0.6]);

        // Reference: SSPRK3 with each operation as its own sweep
        let mut q = q0.clone();
        let mut avg = SWESolution2D::new(ne, nn);
        let mut transport = BarotropicTransport::new(ne, nn, ne * n_face, 0, 0);
        let mut face = vec![0.0; ne * n_face];
        let mut workspace = StageWorkspace::new();
        let damp = |s: &mut SWESolution2D, dt: f64| {
            let [h, hu, hv] = &mut s.data;
            for (idx, ((&h, hu), hv)) in h.iter().zip(hu.iter_mut()).zip(hv.iter_mut()).enumerate()
            {
                if h > 0.0 {
                    let factor = 1.0 / (1.0 + dt * (bottom[idx] / h + column[idx]));
                    *hu *= factor;
                    *hv *= factor;
                }
            }
        };
        for m in 0..2 {
            let mut stage = 0;
            SSPRK3.step_with_relaxation(
                &mut q,
                dt,
                m as f64 * dt,
                |s, time, out| {
                    rhs(s, time, out, &mut face);
                    out.axpy(1.0, &g);
                    let c = w_secondary[m] * SSP_RK3_STAGES[stage].weight;
                    let sums = [
                        (&mut transport.hu.data[..], &s.data[SWE_VAR_HU][..]),
                        (&mut transport.hv.data[..], &s.data[SWE_VAR_HV][..]),
                        (&mut transport.face[..], &face[..]),
                    ];
                    for (sum, values) in sums {
                        for (a, b) in sum.iter_mut().zip(values) {
                            *a += c * b;
                        }
                    }
                    stage += 1;
                },
                |s, _, dt_stage| damp(s, dt_stage),
                |_| {},
                &mut workspace,
            );
            avg.axpy(w_avg[m], &q);
        }

        // Fused
        let mut q_fused = q0.clone();
        let (mut u1, mut u2, mut k) = (q0.clone(), q0.clone(), q0.clone());
        let mut avg_fused = SWESolution2D::new(ne, nn);
        let mut transport_fused = BarotropicTransport::new(ne, nn, ne * n_face, 0, 0);
        let mut pending = None;
        for m in 0..2 {
            for stage in SSP_RK3_STAGES {
                let (x, target, start) = match stage.input {
                    StageInput::Start => (&q_fused, &mut u1, None),
                    StageInput::First => (&u1, &mut u2, Some(&q_fused)),
                    StageInput::Second => (&u2, &mut q_fused, None),
                };
                rhs(x, m as f64 * dt + stage.time * dt, &mut k, &mut face);
                let rates = StageRates {
                    rhs: &k,
                    forcing: &g,
                    face_mass: &face,
                    subcell_mass: &[],
                    drag,
                };
                let weights = PassWeights {
                    transport: w_secondary[m] * stage.weight,
                    average: pending.take(),
                };
                barotropic_stage(
                    stage,
                    x,
                    start,
                    rates,
                    dt,
                    weights,
                    target,
                    &mut transport_fused,
                    &mut avg_fused,
                );
            }
            pending = Some(w_avg[m]);
        }
        avg_fused.axpy(pending.unwrap(), &q_fused);

        let bits = |s: &[f64]| s.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for var in 0..3 {
            assert_eq!(
                bits(&q.data[var]),
                bits(&q_fused.data[var]),
                "state, var {var}"
            );
            assert_eq!(
                bits(&avg.data[var]),
                bits(&avg_fused.data[var]),
                "average, var {var}"
            );
        }
        assert_eq!(bits(&transport.hu.data), bits(&transport_fused.hu.data));
        assert_eq!(bits(&transport.hv.data), bits(&transport_fused.hv.data));
        assert_eq!(bits(&transport.face), bits(&transport_fused.face));
    }

    /// Phase of the prescribed forcing, so that `G′(0) ≠ 0` and the
    /// lower-order starting steps show.
    const PHASE: f64 = 1.0;

    /// A uniform ocean at rest, forced by a prescribed slow forcing
    /// `G = A cos(ωt + φ)` of the x-transport and nothing else.
    struct Forced {
        swe: crate::physics::SWEPhysics2D<crate::boundary::Reflective2D>,
        sigma: SigmaGrid,
        bathymetry: Bathymetry2D,
        omega: f64,
        amplitude: f64,
    }

    impl Forced {
        fn new(omega: f64, amplitude: f64) -> (Self, Solution3D) {
            use crate::boundary::Reflective2D;
            use crate::equations::ShallowWater2D;
            use crate::mesh::Mesh2DBuilder;
            use crate::operators::{DGOperators2D, GeometricFactors2D};
            use crate::physics::PhysicsBuilder;
            use std::sync::Arc;

            let mesh = Arc::new(
                Mesh2DBuilder::new(0.0, 20e3, 0.0, 20e3)
                    .with_resolution(2, 2)
                    .fully_periodic()
                    .build(),
            );
            let ops = Arc::new(DGOperators2D::new(1));
            let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
            let bathymetry = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -10.0);
            let swe = PhysicsBuilder::swe_2d(
                mesh.clone(),
                ops.clone(),
                geom,
                ShallowWater2D::new(9.81),
                Reflective2D::default(),
            )
            .with_bathymetry(Arc::new(bathymetry.clone()))
            .build();
            let state = Solution3D::new(mesh.n_elements, ops.n_nodes, 2);
            let physics = Self {
                swe,
                sigma: SigmaGrid::uniform(2),
                bathymetry,
                omega,
                amplitude,
            };
            (physics, state)
        }

        /// A beach: the bed rises from −2 m to +2 m along a 1 km channel,
        /// water sloshing up it from `η = 0.3 cos(πx/L)`, with the `WetDry`
        /// formulation, its positivity limiter and wet/dry correction. No
        /// slow forcing.
        fn beach() -> (Self, Solution3D) {
            use crate::boundary::Reflective2D;
            use crate::equations::ShallowWater2D;
            use crate::mesh::Mesh2D;
            use crate::operators::{DGOperators2D, GeometricFactors2D};
            use crate::physics::PhysicsBuilder;
            use std::sync::Arc;

            let length = 1000.0;
            let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, length, 0.0, 100.0, 10, 1));
            let ops = Arc::new(DGOperators2D::new(2));
            let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
            let bathymetry =
                Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _| -2.0 + 4.0 * x / length);
            let mut eta = Vec::with_capacity(mesh.n_elements * ops.n_nodes);
            for k in 0..mesh.n_elements {
                for i in 0..ops.n_nodes {
                    let [x, _] = mesh.reference_to_physical(
                        ElementIndex::new(k),
                        ops.nodes_r[i],
                        ops.nodes_s[i],
                    );
                    let b = bathymetry.get(ElementIndex::new(k), i);
                    eta.push((0.3 * (std::f64::consts::PI * x / length).cos()).max(b));
                }
            }
            let swe = PhysicsBuilder::swe_2d(
                mesh.clone(),
                ops.clone(),
                geom,
                ShallowWater2D::new(9.81),
                Reflective2D::default(),
            )
            .with_bathymetry(Arc::new(bathymetry.clone()))
            .with_wet_dry_correction(true)
            .build();
            let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, 2);
            state.eta.data = eta;
            let physics = Self {
                swe,
                sigma: SigmaGrid::uniform(2),
                bathymetry,
                omega: 0.0,
                amplitude: 0.0,
            };
            (physics, state)
        }
    }

    impl ModeSplitPhysics for Forced {
        type Barotropic = crate::physics::SWEPhysics2D<crate::boundary::Reflective2D>;

        fn barotropic(&self) -> &Self::Barotropic {
            &self.swe
        }

        fn sigma(&self) -> &SigmaGrid {
            &self.sigma
        }

        fn bathymetry(&self) -> &Bathymetry2D {
            &self.bathymetry
        }

        fn momentum_rhs_into(&self, _state: &Solution3D, _t: f64, out: &mut Solution3D) {
            out.scale(0.0);
        }

        fn transport_rhs_into(
            &self,
            _state: &Solution3D,
            _t: f64,
            _barotropic: BarotropicFlux,
            out: &mut Solution3D,
        ) {
            out.temp.fill(0.0);
            out.salt.fill(0.0);
        }

        fn slow_forcing_into(
            &self,
            _state: &Solution3D,
            _rhs: &Solution3D,
            t: f64,
            g: &mut SWESolution2D,
        ) {
            g.fill(0.0);
            g.data[SWE_VAR_HU].fill(self.amplitude * (self.omega * t + PHASE).cos());
        }

        fn vertical_implicit(
            &self,
            _state: &mut Solution3D,
            _t: f64,
            _dt: f64,
            _drag: StepDrag<'_>,
        ) {
        }

        fn post_stage(&self, _state: &mut Solution3D) {}
    }

    /// TODO P4.1 gate: the slow forcing is integrated to second order. The
    /// AB3 step averages are third order; the two starting steps (constant,
    /// then linear) leave a second-order error. Before, `G` was frozen at
    /// `tⁿ`: first order. The transport is linear in time within each
    /// barotropic pass, so the filter is exact here and only the time
    /// integration of `G` is measured.
    #[test]
    fn slow_forcing_is_integrated_to_second_order() {
        let omega = 2.0 * std::f64::consts::PI / 86_400.0;
        let amplitude = 1e-4;
        let t_end = 86_400.0;
        let errors: Vec<f64> = [3600.0, 1800.0, 900.0, 450.0]
            .iter()
            .map(|&dt| {
                let (physics, mut state) = Forced::new(omega, amplitude);
                let mut integrator = ModeSplitIntegrator::new();
                let steps = (t_end / dt) as usize;
                for n in 0..steps {
                    integrator.step(&mut state, &physics, dt, n as f64 * dt);
                }
                let exact = amplitude * ((omega * t_end + PHASE).sin() - PHASE.sin()) / omega;
                state
                    .ubar
                    .data
                    .iter()
                    .map(|u| (10.0 * u - exact).abs())
                    .fold(0.0, f64::max)
                    / (amplitude / omega)
            })
            .collect();
        // Measured 3.6e-2, 8.0e-3, 1.9e-3, 4.6e-4 of A/ω: ratios 4.5 → 4.1
        for pair in errors.windows(2) {
            let order = (pair[0] / pair[1]).log2();
            assert!(order > 1.9, "order {order:.2} (errors {errors:?})");
        }
        assert!(errors[3] < 1e-3, "errors {errors:?}");
    }

    /// TODO P4.1 gate (PR 3): with wetting and drying the transport still
    /// balances every element, `∫_K (η̄ − ηⁿ) = −Δt ∮_K F*_h`, although dry
    /// elements use subcell finite volumes and the positivity limiter and
    /// wet/dry correction change nodal depths (they keep element means).
    #[test]
    fn barotropic_transport_balances_every_element_with_wetting_and_drying() {
        let (physics, mut state) = Forced::beach();
        let ops = physics.swe.operators().clone();
        let geom = physics.swe.geometry().clone();
        let h_dry = crate::solver::WetDryConfig::DEFAULT_H_DRY;
        let mut integrator = ModeSplitIntegrator::new();
        let mut div = DGSolution2D::new(state.n_elements, state.n_nodes);
        let dt = 10.0;

        let element_integral = |field: &[f64], k: usize| -> f64 {
            geom.integrate_element(k, &field[k * ops.n_nodes..(k + 1) * ops.n_nodes])
        };
        let mut shoreline_elements = 0;
        for n in 0..10 {
            let eta0 = state.eta.data.clone();
            integrator.step(&mut state, &physics, dt, n as f64 * dt);
            integrator
                .barotropic_transport()
                .expect("after a step")
                .divergence_into(&ops, &geom, physics.swe.metric_form(), &mut div);

            let change: Vec<f64> = state
                .eta
                .data
                .iter()
                .zip(&eta0)
                .map(|(a, b)| a - b)
                .collect();
            let scale = (0..state.n_elements)
                .map(|k| element_integral(&change, k).abs())
                .fold(0.0, f64::max);
            for k in 0..state.n_elements {
                let residual = element_integral(&change, k) + dt * element_integral(&div.data, k);
                assert!(
                    residual.abs() < 1e-12 * scale,
                    "step {n}, element {k}: ∫(η̄ − ηⁿ) + Δt∮F* = {residual:.2e} (scale {scale:.2e})"
                );
                let bed = physics.bathymetry.element(ElementIndex::new(k));
                let eta = &state.eta.data[k * ops.n_nodes..(k + 1) * ops.n_nodes];
                if eta.iter().zip(bed).any(|(e, b)| e - b < h_dry)
                    && eta.iter().zip(bed).any(|(e, b)| e - b > 0.1)
                {
                    shoreline_elements += 1;
                }
            }
        }
        assert!(
            shoreline_elements > 0,
            "test regime: no element straddled the shoreline"
        );
        assert_eq!(physics.swe.negative_depth_clips(), 0);
    }
}
