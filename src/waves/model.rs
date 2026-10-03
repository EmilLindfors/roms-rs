//! The spectral wave model on the DG mesh: the wave action balance
//!
//! ```text
//! ∂N/∂t + ∇·((c_g e_θ + U) N) + ∂(c_σ N)/∂σ + ∂(c_θ N)/∂θ = S/σ
//! ```
//!
//! for the action density `N(x, y, σ, θ, t) = E/σ` (Komen et al. 1994; the SWAN
//! Scientific Documentation, §2.2), with
//!
//! - geographic propagation at the group velocity plus the current, discretised
//!   per spectral component by the strong-form DG of the rest of the crate
//!   (Hesthaven & Warburton 2008, §6): the volume term through the contravariant
//!   fluxes, an upwind flux on the faces (the neighbour's face nodes in reverse
//!   order), on the same mesh, order and quadrature;
//! - refraction by depth and current shear, `c_θ = −(1/k)(∂σ/∂d ∂d/∂m + k·∂U/∂m)`,
//!   `m` across the crest (θ + 90°): waves turn towards shallower water;
//! - frequency shifting by currents, `c_σ = ∂σ/∂d U·∇d − c_g k·∂U/∂s` (`s` along
//!   θ; the term of a changing depth `∂σ/∂d ∂d/∂t` is not included yet);
//! - both by finite volumes over the direction bins (periodic) and the frequency
//!   bins (energy leaves through the ends, none comes in): MUSCL, the upwind bin
//!   reconstructed linearly with van Leer's limiter (van Leer 1979; second order
//!   where the spectrum is smooth, TVD), or first-order upwind
//!   ([`SpectralAdvection`]). Every term is in flux form, so the total action
//!   `Σ Δσ Δθ ∫ N dA` is conserved up to what crosses the open boundaries and the
//!   spectral ends.
//!
//! Depth (`d = η − B`, floored at a minimum depth), its gradient, the current and
//! its gradient are nodal fields (gradients by the element's own derivative
//! matrices), set from a circulation run with [`WaveModel2D::set_water_level`] and
//! [`WaveModel2D::set_currents`]; the wavenumber, group velocity and `∂σ/∂d` of
//! every frequency at every node are kept with them.
//!
//! Boundaries: open faces let waves out and bring in the boundary spectrum
//! ([`WaveModel2D::with_boundary_spectrum`]); every other face absorbs (nothing
//! comes in), as SWAN's default coast.
//!
//! A step ([`WaveModel2D::step`]) is SSP-RK3 for the propagation, with a
//! positivity-preserving scaling of each component in each element towards its
//! mean after every stage (Zhang & Shu 2010; conservative), then the sources over
//! the whole step at every node ([`super::SourceTerms::integrate`]), split
//! first-order in time from the propagation.
//!
//! On steep slopes refraction turns the waves fast, and its explicit Courant
//! limit `|c_θ| Δt ≤ Δθ` sets a step far below the geographic one (100× at
//! Frøya). [`WaveModel2D::with_implicit_refraction`] takes the direction
//! advection out of the Runge–Kutta stages and steps it implicitly at every
//! node and frequency after them, as SWAN does: first-order upwind in θ,
//! backward Euler, a cyclic tridiagonal system per node and frequency. Its
//! matrix is an M-matrix whose columns sum to one, so the step is positive
//! and conserves the action to round-off at any Courant number. With
//! [`SpectralAdvection::VanLeer`] (the default) a deferred correction adds
//! MUSCL's flux less upwind's, of the state before the step, to the
//! right-hand side: the fixed point is MUSCL's steady state (second order in
//! the bins), and the correction is scaled down where it would make the
//! right-hand side negative (it sums to zero over the circle, so the action is
//! still kept). With [`SpectralAdvection::Upwind`] the steady state is the
//! explicit upwind scheme's (first order).

use std::sync::Arc;

use crate::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::time::{SSPRK3, StageWorkspace, TimeIntegrator};
use crate::types::ElementIndex;

use super::dispersion::{dsigma_ddepth, group_velocity, wavenumber};
use super::sources::{SourceTerms, Wind};
use super::spectrum::{SpectralGrid, WaveParameters};
use super::state::WaveSolution;

/// Default minimum depth (m): shallower water, and land, is taken this deep.
pub const DEFAULT_DEPTH_MIN: f64 = 0.1;

/// How refraction and frequency shifting move action between the spectral bins.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpectralAdvection {
    /// First-order upwind finite volumes
    Upwind,
    /// MUSCL: the upwind bin linearly reconstructed with van Leer's limiter,
    /// second order where the spectrum is smooth, TVD for Courant numbers ≤ ½
    #[default]
    VanLeer,
}

/// Reusable storage of [`WaveModel2D::step`].
#[derive(Default)]
pub struct WaveWorkspace {
    stages: StageWorkspace<WaveSolution>,
    /// The state node-major (`[point][component]`), for the sources
    node_major: Vec<f64>,
}

/// The spectral wave model (see the module docs).
pub struct WaveModel2D {
    pub mesh: Arc<Mesh2D>,
    pub ops: Arc<DGOperators2D>,
    pub geom: Arc<GeometricFactors2D>,
    pub grid: SpectralGrid,
    pub sources: SourceTerms,
    pub wind: Wind,
    g: f64,
    depth_min: f64,
    /// Bed elevation B per node (m)
    bed: Vec<f64>,
    /// Depth d per node (m, ≥ `depth_min`), its gradient
    depth: Vec<f64>,
    depth_grad: Vec<[f64; 2]>,
    /// Current per node (m/s) and its gradient `[∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y]`
    current: Vec<[f64; 2]>,
    current_grad: Vec<[f64; 4]>,
    /// Per frequency and node (`[i · n_points + p]`): wavenumber, group velocity
    /// and ∂σ/∂d
    k: Vec<f64>,
    cg: Vec<f64>,
    sigma_d: Vec<f64>,
    /// Action density entering through open faces, per component
    boundary: Option<Vec<f64>>,
    /// Largest turning rate |c_θ| (rad/s) refraction may have, or none
    turning_limit: Option<f64>,
    spectral_advection: SpectralAdvection,
    /// Refraction stepped implicitly after the Runge–Kutta stages
    implicit_refraction: bool,
}

impl WaveModel2D {
    /// A model on `mesh` over the bed `bathymetry` (still water, no current, no
    /// wind, no sources) with the spectral grid `grid`.
    pub fn new(
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
        bathymetry: &Bathymetry2D,
        grid: SpectralGrid,
        g: f64,
    ) -> Self {
        let n_points = mesh.n_elements * ops.n_nodes;
        assert_eq!(bathymetry.data.len(), n_points, "the bed must be nodal");
        let n_freq = grid.n_freq();
        let mut model = Self {
            sources: SourceTerms::none(g),
            wind: Wind::default(),
            g,
            depth_min: DEFAULT_DEPTH_MIN,
            bed: bathymetry.data.clone(),
            depth: vec![0.0; n_points],
            depth_grad: vec![[0.0; 2]; n_points],
            current: vec![[0.0; 2]; n_points],
            current_grad: vec![[0.0; 4]; n_points],
            k: vec![0.0; n_freq * n_points],
            cg: vec![0.0; n_freq * n_points],
            sigma_d: vec![0.0; n_freq * n_points],
            boundary: None,
            turning_limit: None,
            spectral_advection: SpectralAdvection::default(),
            implicit_refraction: false,
            mesh,
            ops,
            geom,
            grid,
        };
        model.set_water_level(&vec![0.0; n_points]);
        model
    }

    pub fn with_sources(mut self, sources: SourceTerms) -> Self {
        self.sources = sources;
        self
    }

    /// A uniform wind over the domain.
    pub fn with_wind(mut self, wind: Wind) -> Self {
        self.wind = wind;
        self
    }

    /// The variance density `e[c]` (m²/(rad/s)/rad) of the waves entering through
    /// open faces.
    pub fn with_boundary_spectrum(mut self, e: &[f64]) -> Self {
        assert_eq!(e.len(), self.grid.n_components());
        let nd = self.grid.n_dir();
        self.boundary = Some(
            e.iter()
                .enumerate()
                .map(|(c, e)| e / self.grid.sigma[c / nd])
                .collect(),
        );
        self
    }

    /// Water shallower than `depth` (m) is taken this deep.
    pub fn with_depth_min(mut self, depth: f64) -> Self {
        assert!(depth > 0.0);
        self.depth_min = depth;
        let eta: Vec<f64> = self
            .depth
            .iter()
            .zip(&self.bed)
            .map(|(d, b)| d + b)
            .collect();
        self.set_water_level(&eta);
        self
    }

    /// Cap the refraction rate |c_θ| at `rate` (rad/s): on steep, shallow slopes
    /// it can otherwise set a time step far below the geographic one. SWAN-type
    /// models limit it the same way (e.g. Dietrich et al. 2013). `None` (the
    /// default) leaves refraction exact.
    pub fn with_turning_limit(mut self, rate: Option<f64>) -> Self {
        self.turning_limit = rate;
        self
    }

    /// Step refraction implicitly (see the module docs): the step is no longer
    /// limited by the turning rate, only by the geographic propagation and the
    /// frequency shifting. Off by default.
    pub fn with_implicit_refraction(mut self, on: bool) -> Self {
        self.implicit_refraction = on;
        self
    }

    /// The scheme of the direction and frequency advection (van Leer's MUSCL by
    /// default). With implicit refraction it applies to the frequencies only.
    pub fn with_spectral_advection(mut self, scheme: SpectralAdvection) -> Self {
        self.spectral_advection = scheme;
        self
    }

    /// Gravitational acceleration (m/s²).
    pub fn g(&self) -> f64 {
        self.g
    }

    pub fn n_points(&self) -> usize {
        self.depth.len()
    }

    /// A state with no energy.
    pub fn zero_state(&self) -> WaveSolution {
        WaveSolution::new(self.grid.n_components(), self.n_points())
    }

    /// A state with the variance density `e[c]` at every node.
    pub fn uniform_state(&self, e: &[f64]) -> WaveSolution {
        let nd = self.grid.n_dir();
        let action: Vec<f64> = e
            .iter()
            .enumerate()
            .map(|(c, e)| e / self.grid.sigma[c / nd])
            .collect();
        let mut state = self.zero_state();
        state.fill_uniform(&action);
        state
    }

    /// Depth per node (m), floored at the minimum depth.
    pub fn depth(&self) -> &[f64] {
        &self.depth
    }

    /// Wavenumber of frequency `i` at node `p` (rad/m).
    pub fn wavenumber(&self, i: usize, p: usize) -> f64 {
        self.k[i * self.n_points() + p]
    }

    /// Set the surface elevation η (m, per node): the depth `η − B` (floored), its
    /// gradient and the wave kinematics follow.
    pub fn set_water_level(&mut self, eta: &[f64]) {
        let n_points = self.n_points();
        assert_eq!(eta.len(), n_points);
        for ((d, &eta), &bed) in self.depth.iter_mut().zip(eta).zip(&self.bed) {
            *d = (eta - bed).max(self.depth_min);
        }
        nodal_gradient(&self.ops, &self.geom, &self.depth, &mut self.depth_grad);
        let (g, grid, depth) = (self.g, &self.grid, &self.depth);
        let rows = self
            .k
            .chunks_mut(n_points)
            .zip(self.cg.chunks_mut(n_points))
            .zip(self.sigma_d.chunks_mut(n_points));
        for (i, ((k, cg), sd)) in rows.enumerate() {
            let s = grid.sigma[i];
            for p in 0..n_points {
                let kp = wavenumber(s, depth[p], g);
                k[p] = kp;
                cg[p] = group_velocity(s, kp, depth[p]);
                sd[p] = dsigma_ddepth(s, kp, depth[p]);
            }
        }
    }

    /// Set the current (m/s, per node, mesh x and y).
    pub fn set_currents(&mut self, u: &[f64], v: &[f64]) {
        let n_points = self.n_points();
        assert!(u.len() == n_points && v.len() == n_points);
        for p in 0..n_points {
            self.current[p] = [u[p], v[p]];
        }
        let (mut gu, mut gv) = (vec![[0.0; 2]; n_points], vec![[0.0; 2]; n_points]);
        nodal_gradient(&self.ops, &self.geom, u, &mut gu);
        nodal_gradient(&self.ops, &self.geom, v, &mut gv);
        for p in 0..n_points {
            self.current_grad[p] = [gu[p][0], gu[p][1], gv[p][0], gv[p][1]];
        }
    }

    /// Turning rate `c_θ` (rad/s) of frequency `i` in direction `theta` at node `p`.
    #[inline]
    fn c_theta(&self, i: usize, theta: f64, p: usize) -> f64 {
        let ip = i * self.n_points() + p;
        let (sn, cs) = theta.sin_cos();
        let [dx, dy] = self.depth_grad[p];
        let [ux, uy, vx, vy] = self.current_grad[p];
        let depth_term = self.sigma_d[ip] / self.k[ip] * (-sn * dx + cs * dy);
        let current_term = cs * (-sn * ux + cs * uy) + sn * (-sn * vx + cs * vy);
        let c = -depth_term - current_term;
        match self.turning_limit {
            Some(limit) => c.clamp(-limit, limit),
            None => c,
        }
    }

    /// Frequency shift `c_σ` (rad/s²) of frequency `i` in direction `theta` at node `p`.
    #[inline]
    fn c_sigma(&self, i: usize, theta: f64, p: usize) -> f64 {
        let ip = i * self.n_points() + p;
        let (sn, cs) = theta.sin_cos();
        let [dx, dy] = self.depth_grad[p];
        let [u, v] = self.current[p];
        let [ux, uy, vx, vy] = self.current_grad[p];
        let advected = self.sigma_d[ip] * (u * dx + v * dy);
        let strain = cs * (cs * ux + sn * uy) + sn * (cs * vx + sn * vy);
        advected - self.cg[ip] * self.k[ip] * strain
    }

    /// The largest stable step (s) for Courant number `cfl` (≤ 1 for SSP-RK3): DG
    /// propagation `Δt ≤ cfl h / ((2N + 1) |c_g e_θ + U|)` per element (h = √area),
    /// and the spectral advection `|c_θ| Δt ≤ cfl Δθ`, `|c_σ| Δt ≤ cfl Δσ`, with
    /// half of that for MUSCL (van Leer's reconstruction is TVD, so positive, for
    /// Courant numbers ≤ ½).
    pub fn compute_dt(&self, cfl: f64) -> f64 {
        let (n_nodes, n_points) = (self.ops.n_nodes, self.n_points());
        let order_factor = (2 * self.ops.order + 1) as f64;
        let mut dt = f64::INFINITY;
        for k in 0..self.mesh.n_elements {
            let h = self.geom.element_size(k);
            let mut speed: f64 = 0.0;
            for p in k * n_nodes..(k + 1) * n_nodes {
                let [u, v] = self.current[p];
                let cg_max = (0..self.grid.n_freq())
                    .map(|i| self.cg[i * n_points + p])
                    .fold(0.0, f64::max);
                speed = speed.max(cg_max + u.hypot(v));
            }
            if speed > 0.0 {
                dt = dt.min(cfl * h / (order_factor * speed));
            }
        }
        let (dtheta, nd) = (self.grid.d_theta, self.grid.n_dir());
        let cfl = match self.spectral_advection {
            SpectralAdvection::Upwind => cfl,
            SpectralAdvection::VanLeer => 0.5 * cfl,
        };
        for p in 0..n_points {
            for i in 0..self.grid.n_freq() {
                for j in 0..nd {
                    if !self.implicit_refraction {
                        let theta = self.grid.theta[j] + 0.5 * dtheta;
                        let ct = self.c_theta(i, theta, p).abs();
                        if ct > 0.0 {
                            dt = dt.min(cfl * dtheta / ct);
                        }
                    }
                    let cs = self.c_sigma(i, self.grid.theta[j], p).abs();
                    if cs > 0.0 {
                        dt = dt.min(cfl * self.grid.d_sigma[i] / cs);
                    }
                }
            }
        }
        dt
    }

    /// `out = −∇·((c_g e_θ + U) N) − ∂(c_θ N)/∂θ − ∂(c_σ N)/∂σ` for every
    /// component (no sources).
    pub fn propagation_rhs_into(&self, n: &WaveSolution, out: &mut WaveSolution) {
        let np = self.n_points();
        let nn = self.ops.n_nodes;
        for_each_chunk(
            &mut out.data,
            np,
            || vec![0.0; 2 * nn],
            |scratch, c, out_c| {
                self.geographic_rhs(n, c, out_c, scratch);
                self.spectral_rhs(n, c, out_c);
            },
        );
    }

    /// The DG geographic term of component `c` into `out` (overwritten).
    fn geographic_rhs(&self, n: &WaveSolution, c: usize, out: &mut [f64], scratch: &mut [f64]) {
        let (ops, geom, mesh) = (&*self.ops, &*self.geom, &*self.mesh);
        let (nn, nfn) = (ops.n_nodes, ops.n_face_nodes);
        let np = self.n_points();
        let nd = self.grid.n_dir();
        let (i, j) = (c / nd, c % nd);
        let (cos, sin) = (self.grid.cos_theta[j], self.grid.sin_theta[j]);
        let cg = &self.cg[i * np..(i + 1) * np];
        let field = n.component(c);
        let boundary = self.boundary.as_ref().map_or(0.0, |b| b[c]);
        let velocity = |p: usize| {
            let [u, v] = self.current[p];
            [cg[p] * cos + u, cg[p] * sin + v]
        };
        let (fr, fs) = scratch.split_at_mut(nn);
        for k in 0..mesh.n_elements {
            let base = k * nn;
            // Volume: −J⁻¹ (D_r F̃_r + D_s F̃_s), F̃ the contravariant fluxes
            for a in 0..nn {
                let p = base + a;
                let [vx, vy] = velocity(p);
                let ((jrx, jry), (jsx, jsy)) = geom.contravariant(k, a);
                fr[a] = (jrx * vx + jry * vy) * field[p];
                fs[a] = (jsx * vx + jsy * vy) * field[p];
            }
            for a in 0..nn {
                let (dr, ds) = (
                    &ops.dr_row_major[a * nn..(a + 1) * nn],
                    &ops.ds_row_major[a * nn..(a + 1) * nn],
                );
                let mut div = 0.0;
                for b in 0..nn {
                    div += dr[b] * fr[b] + ds[b] * fs[b];
                }
                out[base + a] = -div * geom.jacobian_inv(k, a);
            }
            // Faces: lift (F⁻·n − F*) with the upwind flux F*
            let element = ElementIndex::new(k);
            for face in 0..4 {
                let neighbour = mesh.neighbor(element, face);
                let open = neighbour.is_none()
                    && mesh.boundary_tag(element, face) == Some(BoundaryTag::Open);
                for fi in 0..nfn {
                    let a = ops.face_nodes[face][fi];
                    let p = base + a;
                    let (nx, ny) = geom.normal(k, face, fi);
                    let [vx, vy] = velocity(p);
                    let un_in = vx * nx + vy * ny;
                    let n_in = field[p];
                    let flux = match neighbour {
                        Some(nb) => {
                            let q = nb.element * nn + ops.face_nodes[nb.face][nfn - 1 - fi];
                            let [wx, wy] = velocity(q);
                            let un_out = wx * nx + wy * ny;
                            if un_in + un_out >= 0.0 {
                                un_in * n_in
                            } else {
                                un_out * field[q]
                            }
                        }
                        // Out through every boundary; in only through open ones
                        None if un_in >= 0.0 => un_in * n_in,
                        None if open => un_in * boundary,
                        None => 0.0,
                    };
                    let jump = (un_in * n_in - flux) * geom.surface_jacobian(k, face, fi);
                    let lift = &ops.lift_row_major[face];
                    for b in 0..nn {
                        let l = lift[b * nfn + fi];
                        if l != 0.0 {
                            out[base + b] += l * jump * geom.jacobian_inv(k, b);
                        }
                    }
                }
            }
        }
    }

    /// Subtract the direction and frequency flux divergences of component `c`.
    fn spectral_rhs(&self, n: &WaveSolution, c: usize, out: &mut [f64]) {
        let (nf, nd) = (self.grid.n_freq(), self.grid.n_dir());
        let (i, j) = (c / nd, c % nd);
        let grid = &self.grid;
        let second_order = self.spectral_advection == SpectralAdvection::VanLeer;
        let here = n.component(c);
        // Directions: periodic, faces at θ_j ± Δθ/2; the bins two away for the
        // reconstruction
        let dir = |offset: isize| {
            let jj = (j as isize + offset).rem_euclid(nd as isize) as usize;
            n.component(i * nd + jj)
        };
        let (below, above) = (dir(-1), dir(1));
        let (below2, above2) = (second_order.then(|| dir(-2)), second_order.then(|| dir(2)));
        let (theta_lo, theta_hi) = (
            grid.theta[j] - 0.5 * grid.d_theta,
            grid.theta[j] + 0.5 * grid.d_theta,
        );
        let inv_dtheta = 1.0 / grid.d_theta;
        // Frequencies: faces between bins; nothing enters at the ends
        let freq = |offset: isize| {
            let ii = i as isize + offset;
            (0..nf as isize)
                .contains(&ii)
                .then(|| n.component(ii as usize * nd + j))
        };
        let (lower, upper) = (freq(-1), freq(1));
        let (lower2, upper2) = (
            freq(-2).filter(|_| second_order),
            freq(2).filter(|_| second_order),
        );
        let inv_dsigma = 1.0 / grid.d_sigma[i];
        let theta = grid.theta[j];
        let at = |field: Option<&[f64]>, p: usize| field.map(|f| f[p]);
        let explicit_refraction = !self.implicit_refraction;
        for (p, out) in out.iter_mut().enumerate() {
            if explicit_refraction {
                let f_hi = face_flux(
                    self.c_theta(i, theta_hi, p),
                    at(below2.is_some().then_some(below), p),
                    here[p],
                    above[p],
                    at(above2, p),
                );
                let f_lo = face_flux(
                    self.c_theta(i, theta_lo, p),
                    at(below2, p),
                    below[p],
                    here[p],
                    at(above2.is_some().then_some(above), p),
                );
                *out -= (f_hi - f_lo) * inv_dtheta;
            }
            let cs = self.c_sigma(i, theta, p);
            let g_hi = match upper {
                Some(up) => face_flux(
                    0.5 * (cs + self.c_sigma(i + 1, theta, p)),
                    at(lower.filter(|_| second_order), p),
                    here[p],
                    up[p],
                    at(upper2, p),
                ),
                None => cs.max(0.0) * here[p],
            };
            let g_lo = match lower {
                Some(lo) => face_flux(
                    0.5 * (cs + self.c_sigma(i - 1, theta, p)),
                    at(lower2, p),
                    lo[p],
                    here[p],
                    at(upper.filter(|_| second_order), p),
                ),
                None => cs.min(0.0) * here[p],
            };
            *out -= (g_hi - g_lo) * inv_dsigma;
        }
    }

    /// Scale every component in every element towards its mean so that no node is
    /// negative (Zhang & Shu 2010): the mean, and so the total action, is kept; an
    /// element whose mean is negative is zeroed.
    pub fn limit_positivity(&self, n: &mut WaveSolution) {
        let (nn, np) = (self.ops.n_nodes, self.n_points());
        let geom = &*self.geom;
        let weights = &self.ops.weights;
        for_each_chunk(
            &mut n.data,
            np,
            || (),
            |_, _, field| {
                for (k, values) in field.chunks_exact_mut(nn).enumerate() {
                    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                    if min >= 0.0 {
                        continue;
                    }
                    let (mut mass, mut area) = (0.0, 0.0);
                    for (a, &x) in values.iter().enumerate() {
                        let w = weights[a] * geom.jacobian(k, a);
                        mass += w * x;
                        area += w;
                    }
                    let mean = mass / area;
                    if mean <= 0.0 {
                        values.fill(0.0);
                        continue;
                    }
                    let theta = mean / (mean - min);
                    values
                        .iter_mut()
                        .for_each(|x| *x = (mean + theta * (*x - mean)).max(0.0));
                }
            },
        );
    }

    /// Advance `n` from `t` by `dt`: SSP-RK3 propagation (positivity limited every
    /// stage), then the sources over the step. Implicit refraction takes half the
    /// step before the stages and half after (Strang), so its splitting error is
    /// second order in the step.
    pub fn step(&self, n: &mut WaveSolution, t: f64, dt: f64, ws: &mut WaveWorkspace) {
        // Strang: half of the implicit refraction on each side of the stages
        let refraction = self.implicit_refraction.then_some(0.5 * dt);
        if let Some(half) = refraction {
            self.node_pass(n, ws, Some(half), None);
        }
        SSPRK3.step_with_workspace(
            n,
            dt,
            t,
            |s, _, out| self.propagation_rhs_into(s, out),
            |s| self.limit_positivity(s),
            &mut ws.stages,
        );
        let sources = self.sources.any().then_some(dt);
        if refraction.is_some() || sources.is_some() {
            self.node_pass(n, ws, refraction, sources);
        }
    }

    /// The sources over `dt` at every node.
    pub fn apply_sources(&self, n: &mut WaveSolution, dt: f64, ws: &mut WaveWorkspace) {
        self.node_pass(n, ws, None, Some(dt));
    }

    /// Implicit refraction over `dt` at every node (see the module docs),
    /// whether or not the model steps it so.
    pub fn apply_implicit_refraction(&self, n: &mut WaveSolution, dt: f64, ws: &mut WaveWorkspace) {
        self.node_pass(n, ws, Some(dt), None);
    }

    /// At every node, on its spectrum: implicit refraction over the first
    /// interval, then the sources over the second, each if given.
    fn node_pass(
        &self,
        n: &mut WaveSolution,
        ws: &mut WaveWorkspace,
        refraction: Option<f64>,
        sources: Option<f64>,
    ) {
        let (np, nc, nf) = (
            self.n_points(),
            self.grid.n_components(),
            self.grid.n_freq(),
        );
        let nd = self.grid.n_dir();
        ws.node_major.resize(np * nc, 0.0);
        let data = &n.data;
        for_each_chunk(
            &mut ws.node_major,
            nc,
            || (),
            |_, p, spectrum| {
                for (c, x) in spectrum.iter_mut().enumerate() {
                    *x = data[c * np + p];
                }
            },
        );
        for_each_chunk(
            &mut ws.node_major,
            nc,
            || {
                (
                    vec![0.0; nf],
                    vec![0.0; nc],
                    vec![0.0; nc],
                    vec![0.0; nc],
                    CyclicScratch::new(nd),
                )
            },
            |(k, e, a, b, cyclic), p, spectrum| {
                if let Some(dt) = refraction {
                    for (i, row) in spectrum.chunks_exact_mut(nd).enumerate() {
                        self.refract_implicitly(i, p, row, dt, cyclic);
                    }
                }
                if let Some(dt) = sources {
                    for (i, k) in k.iter_mut().enumerate() {
                        *k = self.k[i * np + p];
                    }
                    self.sources.integrate(
                        &self.grid,
                        spectrum,
                        k,
                        self.depth[p],
                        self.wind,
                        dt,
                        e,
                        a,
                        b,
                    );
                }
            },
        );
        let node_major = &ws.node_major;
        for_each_chunk(
            &mut n.data,
            np,
            || (),
            |_, c, field| {
                for (p, x) in field.iter_mut().enumerate() {
                    *x = node_major[p * nc + c];
                }
            },
        );
    }

    /// Backward Euler over `dt` for the direction advection of frequency `i` at
    /// node `p`, on its action densities over the directions `row`: first-order
    /// upwind fluxes `c⁺ N_j + c⁻ N_{j+1}` through the faces `θ_j + Δθ/2`,
    /// periodic, so
    ///
    /// ```text
    /// N_j + λ (c⁺_{j+½} − c⁻_{j−½}) N_j + λ c⁻_{j+½} N_{j+1} − λ c⁺_{j−½} N_{j−1} = N*_j
    /// ```
    ///
    /// with `λ = Δt/Δθ`: a cyclic tridiagonal M-matrix whose columns sum to one.
    fn refract_implicitly(
        &self,
        i: usize,
        p: usize,
        row: &mut [f64],
        dt: f64,
        s: &mut CyclicScratch,
    ) {
        let nd = row.len();
        let (dtheta, lambda) = (self.grid.d_theta, dt / self.grid.d_theta);
        // The turning rate at the face above each bin
        for (j, c) in s.faces.iter_mut().enumerate() {
            *c = self.c_theta(i, self.grid.theta[j] + 0.5 * dtheta, p);
        }
        if s.faces.iter().all(|&c| c == 0.0) {
            return;
        }
        for j in 0..nd {
            let (above, below) = (s.faces[j], s.faces[(j + nd - 1) % nd]);
            s.diagonal[j] = 1.0 + lambda * (above.max(0.0) - below.min(0.0));
            s.upper[j] = lambda * above.min(0.0);
            s.lower[j] = -lambda * below.max(0.0);
        }
        if self.spectral_advection == SpectralAdvection::VanLeer {
            // Deferred correction: MUSCL's flux less upwind's, of the state now
            let at = |j: usize, offset: isize| {
                row[(j as isize + offset).rem_euclid(nd as isize) as usize]
            };
            for j in 0..nd {
                let (c, left, right) = (s.faces[j], at(j, 0), at(j, 1));
                let muscl = face_flux(c, Some(at(j, -1)), left, right, Some(at(j, 2)));
                s.correction[j] = muscl - (c.max(0.0) * left + c.min(0.0) * right);
            }
            // Limit each face's correction by its donor (Zalesak 1979): what
            // the corrections take out of a bin is at most what it holds, so
            // the right-hand side stays non-negative, and as fluxes they keep
            // the action
            for j in 0..nd {
                let below = s.correction[(j + nd - 1) % nd];
                let taken = lambda * (s.correction[j].max(0.0) + (-below).max(0.0));
                s.divergence[j] = if taken > row[j] { row[j] / taken } else { 1.0 };
            }
            for j in 0..nd {
                let c = s.correction[j];
                let donor = if c >= 0.0 { j } else { (j + 1) % nd };
                s.correction[j] = c * s.divergence[donor];
            }
            for j in 0..nd {
                let d = lambda * (s.correction[j] - s.correction[(j + nd - 1) % nd]);
                row[j] = (row[j] - d).max(0.0);
            }
        }
        solve_cyclic_tridiagonal(row, s);
    }

    /// The variance density `E = σ N` at node `p` into `e`.
    pub fn energy_spectrum_into(&self, n: &WaveSolution, p: usize, e: &mut [f64]) {
        n.spectrum_into(p, e);
        let nd = self.grid.n_dir();
        for (c, x) in e.iter_mut().enumerate() {
            *x *= self.grid.sigma[c / nd];
        }
    }

    /// Integrated parameters at every node.
    pub fn parameters(&self, n: &WaveSolution) -> Vec<WaveParameters> {
        let mut e = vec![0.0; self.grid.n_components()];
        (0..self.n_points())
            .map(|p| {
                self.energy_spectrum_into(n, p, &mut e);
                self.grid.parameters(&e)
            })
            .collect()
    }

    /// Stokes drift (m/s) at height `z` (m, ≤ 0) below the surface at every node.
    pub fn stokes_drift(&self, n: &WaveSolution, z: f64) -> Vec<[f64; 2]> {
        let (np, nf) = (self.n_points(), self.grid.n_freq());
        let mut e = vec![0.0; self.grid.n_components()];
        let mut k = vec![0.0; nf];
        (0..np)
            .map(|p| {
                self.energy_spectrum_into(n, p, &mut e);
                (0..nf).for_each(|i| k[i] = self.k[i * np + p]);
                self.grid
                    .stokes_drift(&e, &k, self.depth[p], z.max(-self.depth[p]))
            })
            .collect()
    }

    /// Surface roughness `alpha · H_s` (m) at every node, for the turbulence
    /// closure of the circulation (Terray et al. 1996: `alpha ≈ 0.6`; see
    /// [`crate::physics::Hydrostatic3D::with_surface_roughness`]).
    pub fn surface_roughness(&self, n: &WaveSolution, alpha: f64) -> Vec<f64> {
        self.parameters(n).iter().map(|p| alpha * p.hs).collect()
    }

    /// The waves' bed stress per ρ (m²/s²) at every node on a bed of roughness
    /// length `z0` (m), Soulsby's `½ f_w U_w²` ([`SpectralGrid::bed_wave_stress`]),
    /// for [`crate::source::WaveCurrentFriction2D`].
    pub fn bed_wave_stress(&self, n: &WaveSolution, z0: f64) -> Vec<f64> {
        let (np, nf) = (self.n_points(), self.grid.n_freq());
        let mut e = vec![0.0; self.grid.n_components()];
        let mut k = vec![0.0; nf];
        (0..np)
            .map(|p| {
                self.energy_spectrum_into(n, p, &mut e);
                (0..nf).for_each(|i| k[i] = self.k[i * np + p]);
                self.grid.bed_wave_stress(&e, &k, self.depth[p], z0)
            })
            .collect()
    }

    /// Radiation stress per ρg `[S_xx, S_xy, S_yy]` (m²) at every node.
    pub fn radiation_stress(&self, n: &WaveSolution) -> Vec<[f64; 3]> {
        let (np, nf) = (self.n_points(), self.grid.n_freq());
        let mut e = vec![0.0; self.grid.n_components()];
        let mut k = vec![0.0; nf];
        (0..np)
            .map(|p| {
                self.energy_spectrum_into(n, p, &mut e);
                (0..nf).for_each(|i| k[i] = self.k[i * np + p]);
                self.grid.radiation_stress(&e, &k, self.depth[p])
            })
            .collect()
    }

    /// Total action `Σ_c Δσ Δθ ∫ N_c dA` (m⁴·s/rad), conserved by the propagation
    /// up to boundary and spectral-end fluxes.
    pub fn total_action(&self, n: &WaveSolution) -> f64 {
        let (nn, nd) = (self.ops.n_nodes, self.grid.n_dir());
        let mut total = 0.0;
        for c in 0..self.grid.n_components() {
            let field = n.component(c);
            let mut integral = 0.0;
            for k in 0..self.mesh.n_elements {
                for a in 0..nn {
                    integral += self.ops.weights[a] * self.geom.jacobian(k, a) * field[k * nn + a];
                }
            }
            total += integral * self.grid.d_sigma[c / nd] * self.grid.d_theta;
        }
        total
    }
}

/// The upwind flux `speed · N_face` through the face between the bins `left` and
/// `right`. With the bins beyond them (`far_left`, `far_right`) the face value
/// is the upwind bin's, reconstructed linearly with van Leer's limited slope
/// (`2ab/(a + b)` of the one-sided differences, 0 at an extremum): second
/// order where smooth, TVD; without, first-order upwind.
#[inline]
fn face_flux(
    speed: f64,
    far_left: Option<f64>,
    left: f64,
    right: f64,
    far_right: Option<f64>,
) -> f64 {
    // Half of van Leer's slope from the one-sided differences a and b
    let half_slope = |a: f64, b: f64| if a * b > 0.0 { a * b / (a + b) } else { 0.0 };
    if speed >= 0.0 {
        let value = far_left.map_or(left, |ll| left + half_slope(left - ll, right - left));
        speed * value
    } else {
        let value = far_right.map_or(right, |rr| right - half_slope(right - left, rr - right));
        speed * value
    }
}

/// Storage of the implicit refraction at one node: the turning rates at the
/// faces, a cyclic tridiagonal system over the directions and its solve.
struct CyclicScratch {
    faces: Vec<f64>,
    /// MUSCL's flux less upwind's through each face, then its divergence
    correction: Vec<f64>,
    divergence: Vec<f64>,
    /// `lower[j] x[j−1] + diagonal[j] x[j] + upper[j] x[j+1]`, indices periodic
    lower: Vec<f64>,
    diagonal: Vec<f64>,
    upper: Vec<f64>,
    modified: Vec<f64>,
    c_prime: Vec<f64>,
    x: Vec<f64>,
    u: Vec<f64>,
    z: Vec<f64>,
}

impl CyclicScratch {
    fn new(n: usize) -> Self {
        Self {
            faces: vec![0.0; n],
            correction: vec![0.0; n],
            divergence: vec![0.0; n],
            lower: vec![0.0; n],
            diagonal: vec![0.0; n],
            upper: vec![0.0; n],
            modified: vec![0.0; n],
            c_prime: vec![0.0; n],
            x: vec![0.0; n],
            u: vec![0.0; n],
            z: vec![0.0; n],
        }
    }
}

/// The Thomas algorithm: `lower[j] x[j−1] + diagonal[j] x[j] + upper[j] x[j+1]
/// = rhs[j]` (`lower[0]` and `upper[n−1]` ignored) into `x`. Stable without
/// pivoting for diagonally dominant rows or columns.
fn thomas(
    lower: &[f64],
    diagonal: &[f64],
    upper: &[f64],
    rhs: &[f64],
    x: &mut [f64],
    c_prime: &mut [f64],
) {
    let n = diagonal.len();
    c_prime[0] = upper[0] / diagonal[0];
    x[0] = rhs[0] / diagonal[0];
    for j in 1..n {
        let m = 1.0 / (diagonal[j] - lower[j] * c_prime[j - 1]);
        c_prime[j] = upper[j] * m;
        x[j] = (rhs[j] - lower[j] * x[j - 1]) * m;
    }
    for j in (0..n - 1).rev() {
        x[j] -= c_prime[j] * x[j + 1];
    }
}

/// Solve the cyclic tridiagonal system of `s` (`lower[0]` couples to
/// `x[n−1]`, `upper[n−1]` to `x[0]`) for the right-hand side `rhs`, in place:
/// the Thomas algorithm on the system without its corners, corrected by
/// Sherman–Morrison (Press et al., "Numerical Recipes", §2.7).
fn solve_cyclic_tridiagonal(rhs: &mut [f64], s: &mut CyclicScratch) {
    let n = rhs.len();
    // The corners A[n−1][0] and A[0][n−1]
    let (alpha, beta) = (s.upper[n - 1], s.lower[0]);
    if alpha == 0.0 && beta == 0.0 {
        thomas(
            &s.lower,
            &s.diagonal,
            &s.upper,
            rhs,
            &mut s.x,
            &mut s.c_prime,
        );
        rhs.copy_from_slice(&s.x);
        return;
    }
    let gamma = -s.diagonal[0];
    s.modified.copy_from_slice(&s.diagonal);
    s.modified[0] -= gamma;
    s.modified[n - 1] -= alpha * beta / gamma;
    thomas(
        &s.lower,
        &s.modified,
        &s.upper,
        rhs,
        &mut s.x,
        &mut s.c_prime,
    );
    s.u.fill(0.0);
    s.u[0] = gamma;
    s.u[n - 1] = alpha;
    thomas(
        &s.lower,
        &s.modified,
        &s.upper,
        &s.u,
        &mut s.z,
        &mut s.c_prime,
    );
    let fact = (s.x[0] + beta * s.x[n - 1] / gamma) / (1.0 + s.z[0] + beta * s.z[n - 1] / gamma);
    for (r, (x, z)) in rhs.iter_mut().zip(s.x.iter().zip(&s.z)) {
        *r = x - fact * z;
    }
}

/// Physical gradient of the nodal field `f` by each element's derivative matrices.
pub(crate) fn nodal_gradient(
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    f: &[f64],
    out: &mut [[f64; 2]],
) {
    let nn = ops.n_nodes;
    for k in 0..f.len() / nn {
        let fk = &f[k * nn..(k + 1) * nn];
        for a in 0..nn {
            let (dr, ds) = (
                &ops.dr_row_major[a * nn..(a + 1) * nn],
                &ops.ds_row_major[a * nn..(a + 1) * nn],
            );
            let fr: f64 = dr.iter().zip(fk).map(|(d, f)| d * f).sum();
            let fs: f64 = ds.iter().zip(fk).map(|(d, f)| d * f).sum();
            let ((rx, ry), (sx, sy)) = (geom.grad_r(k, a), geom.grad_s(k, a));
            out[k * nn + a] = [rx * fr + sx * fs, ry * fr + sy * fs];
        }
    }
}

/// `f(scratch, index, chunk)` for every `size`-long chunk of `data`, in parallel
/// with the `parallel` feature, with scratch from `init` per worker.
fn for_each_chunk<T, S, I, F>(data: &mut [T], size: usize, init: I, f: F)
where
    T: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize, &mut [T]) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(size)
            .enumerate()
            .for_each_init(init, |s, (i, chunk)| f(s, i, chunk));
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut s = init();
        for (i, chunk) in data.chunks_mut(size).enumerate() {
            f(&mut s, i, chunk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh2D;

    /// The cyclic solve against the matrix it inverts: residual at round-off,
    /// with both corners set and with one (as the upwind refraction has).
    #[test]
    fn the_cyclic_solve_inverts_its_matrix() {
        let n = 9;
        for corners in [(0.3, -0.7), (0.0, -0.4), (0.25, 0.0)] {
            let mut s = CyclicScratch::new(n);
            for j in 0..n {
                let t = j as f64;
                s.lower[j] = -0.2 - 0.1 * (1.3 * t).sin().abs();
                s.upper[j] = -0.3 + 0.05 * (0.7 * t).cos();
                s.diagonal[j] = 1.5 + 0.2 * (0.4 * t).sin();
            }
            s.upper[n - 1] = corners.0;
            s.lower[0] = corners.1;
            let rhs: Vec<f64> = (0..n).map(|j| 1.0 + (j as f64).sqrt()).collect();
            let mut x = rhs.clone();
            let (lower, diagonal, upper) = (s.lower.clone(), s.diagonal.clone(), s.upper.clone());
            solve_cyclic_tridiagonal(&mut x, &mut s);
            for j in 0..n {
                let ax =
                    lower[j] * x[(j + n - 1) % n] + diagonal[j] * x[j] + upper[j] * x[(j + 1) % n];
                assert!(
                    (ax - rhs[j]).abs() < 1e-13 * rhs[j],
                    "row {j}: {ax} against {}",
                    rhs[j]
                );
            }
        }
    }

    /// Implicit refraction over a step 10⁴ times the geographic limit, on a
    /// steep slope: the action of every frequency at every node is kept to
    /// round-off, nothing goes negative, and the waves have turned.
    #[test]
    fn implicit_refraction_keeps_the_action_at_any_step() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 200.0, 0.0, 200.0, 2, 2);
        let ops = Arc::new(DGOperators2D::new(2));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry =
            Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| -(2.0 + 0.1 * y + 0.02 * x));
        let grid = SpectralGrid::new(0.06, 0.3, 6, 36);
        let model = WaveModel2D::new(Arc::new(mesh), ops, geom, &bathymetry, grid, 9.81)
            .with_implicit_refraction(true);
        let e = model.grid.jonswap(1.0, 8.0, 3.3, 0.3, 2.0);
        let mut n = model.uniform_state(&e);
        let before = n.clone();
        let explicit = model.compute_dt(1.0).min(1.0);
        let mut ws = WaveWorkspace::default();
        model.apply_implicit_refraction(&mut n, 1e4 * explicit, &mut ws);
        let (nf, nd, np) = (model.grid.n_freq(), model.grid.n_dir(), model.n_points());
        let mut turned = 0.0f64;
        for p in 0..np {
            for i in 0..nf {
                let sum =
                    |s: &WaveSolution| -> f64 { (0..nd).map(|j| s.component(i * nd + j)[p]).sum() };
                let (a, b) = (sum(&before), sum(&n));
                assert!(
                    (a - b).abs() <= 1e-13 * a.max(1e-300),
                    "node {p}, frequency {i}"
                );
                for j in 0..nd {
                    let c = i * nd + j;
                    assert!(n.component(c)[p] >= 0.0);
                    turned = turned.max((n.component(c)[p] - before.component(c)[p]).abs());
                }
            }
        }
        assert!(turned > 1e-3, "nothing turned ({turned})");
    }
}
