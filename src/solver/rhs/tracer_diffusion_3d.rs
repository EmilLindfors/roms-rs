//! Horizontal diffusion of the tracers' departure from a reference
//! stratification, along σ-surfaces.
//!
//! Along every σ-layer `l`, the inventory tendency of `C ∈ {T, S}` is
//!
//! ```text
//!     ∂(H_z C_l)/∂t = ∇·(κ H_z ∇C′_l),    C′_l = C_l − C_ref(z_l),    H_z = Δσ_l D,
//! ```
//!
//! with `C_ref(z)` a horizontally uniform reference profile evaluated at the
//! layer's height `z_l = η + σ_l D` (ROMS's `TS_MIX_CLIMA`, diffusion of the
//! departure from a climatology). A plain Laplacian of `C` along steep
//! σ-surfaces mixes across the stratification wherever the surfaces cross
//! it; the departure is zero in a fluid at rest in the reference, so this
//! diffusion leaves it alone however steep the σ-surfaces are, and damps
//! only the anomalies that motion (or the numerics) creates on them.
//!
//! **Why (TODO P1.3).** Over steep beds, where a σ-level crosses the
//! pycnocline between the nodes of one element, the stratified rest state of
//! the σ-pairs pressure gradient with split-form advection is unstable: the
//! advection deposits density along the level's chord, and round-off grows
//! (e-folding ≈ 15 min on the Frøya bed). The density anomalies it grows from
//! are the grid-scale `C′` this term damps.
//!
//! BR1 per layer and tracer (Bassi & Rebay 1997), as the 3D shear viscosity
//! ([`crate::solver::rhs::viscosity_3d`]), with the shared kernels of
//! `diffusion_2d`. The face flux is central and single-valued, so interior
//! faces conserve each layer's inventory; walls pass no flux and open faces
//! the interior's. Thin columns (`min_column_depth`) have no diffusivity and
//! enter their neighbours' gradients with no departure.

use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::TracerReferenceProfile;
use crate::solver::core::blocks::{Pooled, for_each_block};
use crate::solver::state::Solution3D;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

use super::boundary_3d::{Boundaries3D, FaceExterior};
use super::diffusion_2d::{
    DiffusionScratch, ScalarGradient2D, br1_diffusion_element, br1_gradient_element,
};

/// Horizontal diffusion of T and S about a reference stratification (see
/// the [module documentation](self)).
#[derive(Clone, Debug, PartialEq)]
pub struct TracerAnomalyDiffusion3D {
    /// Diffusivity `κ` (m²/s).
    pub kappa: f64,
    /// The stratification the departures are taken from.
    pub reference: TracerReferenceProfile,
}

impl TracerAnomalyDiffusion3D {
    /// Diffusivity `kappa` (m²/s) of the departures from `reference`.
    ///
    /// # Panics
    /// If `kappa` is negative or not finite.
    pub fn new(kappa: f64, reference: TracerReferenceProfile) -> Self {
        assert!(
            kappa >= 0.0 && kappa.is_finite(),
            "tracer diffusivity must be finite and non-negative, got {kappa}"
        );
        Self { kappa, reference }
    }
}

/// Buffers of [`apply_tracer_anomaly_diffusion_3d`], reused between calls.
pub struct TracerDiffusionScratch3D {
    /// `D` of every column, zero in thin ones, `[element][node]`.
    depth: Vec<f64>,
    /// One layer's departures `(T′, S′)`, `[element][node]`.
    departure: Vec<[f64; 2]>,
    /// Their BR1 gradients.
    gradient: Vec<[ScalarGradient2D; 2]>,
}

impl TracerDiffusionScratch3D {
    /// Buffers for `n_elements` elements of `ops`.
    pub fn new(n_elements: usize, ops: &DGOperators2D) -> Self {
        let n = n_elements * ops.n_nodes;
        Self {
            depth: vec![0.0; n],
            departure: vec![[0.0; 2]; n],
            gradient: vec![[ScalarGradient2D::default(); 2]; n],
        }
    }
}

/// One element's buffers of the BR1 passes.
struct ElementScratch {
    own: Vec<[f64; 2]>,
    gradient: [Vec<ScalarGradient2D>; 2],
    diffusion: DiffusionScratch<2>,
    out: [Vec<f64>; 2],
}

impl ElementScratch {
    fn take(nn: usize) -> Pooled<Self> {
        Pooled::take(
            |s: &Self| s.own.len() == nn,
            || Self {
                own: vec![[0.0; 2]; nn],
                gradient: std::array::from_fn(|_| vec![ScalarGradient2D::default(); nn]),
                diffusion: DiffusionScratch::new(nn),
                out: std::array::from_fn(|_| vec![0.0; nn]),
            },
        )
    }
}

/// Add the inventory tendencies `∇·(κ H_z ∇C′)` of the horizontal
/// `diffusion` of `state`'s T and S to `rhs_temp`, `rhs_salt`
/// (`[element][node][level]`; see the [module documentation](self)).
#[allow(clippy::too_many_arguments)]
pub fn apply_tracer_anomaly_diffusion_3d(
    rhs_temp: &mut [f64],
    rhs_salt: &mut [f64],
    state: &Solution3D,
    diffusion: &TracerAnomalyDiffusion3D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    bathymetry: &Bathymetry2D,
    sigma: &SigmaGrid,
    boundaries: &Boundaries3D,
    min_column_depth: f64,
    scratch: &mut TracerDiffusionScratch3D,
) {
    let kappa = diffusion.kappa;
    if kappa == 0.0 {
        return;
    }
    let reference = &diffusion.reference;
    let (ne, nn, nl) = (mesh.n_elements, ops.n_nodes, state.n_levels);
    let n = ne * nn * nl;
    let TracerDiffusionScratch3D {
        depth,
        departure,
        gradient,
    } = scratch;
    for (d, (&eta, &b)) in depth
        .iter_mut()
        .zip(state.eta.data.iter().zip(&bathymetry.data))
    {
        let h = eta - b;
        *d = if h < min_column_depth { 0.0 } else { h };
    }
    let depth: &[f64] = depth;

    for (level, &s) in sigma.sigma_rho().iter().enumerate() {
        let d_sigma = sigma.d_sigma()[level];
        for_each_block(
            ne,
            [&mut departure[..]],
            || (),
            |_, k, [departure_k]| {
                for (i, c) in departure_k.iter_mut().enumerate() {
                    let idx = k * nn + i;
                    *c = if depth[idx] == 0.0 {
                        [0.0; 2]
                    } else {
                        let z = state.eta.data[idx] + s * depth[idx];
                        let at = idx * nl + level;
                        [
                            state.temp[at] - reference.temperature(z),
                            state.salt[at] - reference.salinity(z),
                        ]
                    };
                }
            },
        );
        let departure: &[[f64; 2]] = departure;
        for_each_block(
            ne,
            [&mut gradient[..]],
            || ElementScratch::take(nn),
            |scratch, k, [gradient_k]| {
                let ElementScratch { own, gradient, .. } = &mut **scratch;
                let [grad_t, grad_s] = gradient;
                br1_gradient_element(
                    ElementIndex::new(k),
                    mesh,
                    ops,
                    geom,
                    |j, node| departure[j.as_usize() * nn + node],
                    |_, _, _, _, interior| interior,
                    own,
                    [grad_t, grad_s],
                );
                for (i, g) in gradient_k.iter_mut().enumerate() {
                    *g = [grad_t[i], grad_s[i]];
                }
            },
        );
        let gradient: &[[ScalarGradient2D; 2]] = gradient;
        // κ H_z ∇C′ at a node
        let flux = |idx: usize| {
            let coefficient = kappa * d_sigma * depth[idx];
            gradient[idx].map(|g| (coefficient * g.dx, coefficient * g.dy))
        };
        for_each_block(
            ne,
            [&mut rhs_temp[..n], &mut rhs_salt[..n]],
            || ElementScratch::take(nn),
            |scratch, k, rhs| {
                let ElementScratch { diffusion, out, .. } = &mut **scratch;
                let [out_t, out_s] = out;
                br1_diffusion_element(
                    ElementIndex::new(k),
                    mesh,
                    ops,
                    geom,
                    |j, node| flux(j.as_usize() * nn + node),
                    diffusion,
                    [out_t, out_s],
                );
                // The kernel takes the interior flux on boundary faces: walls
                // pass none
                let el = ElementIndex::new(k);
                for face in 0..4 {
                    if boundaries.exterior(mesh, el, face) != FaceExterior::Wall {
                        continue;
                    }
                    for (fi, &node) in ops.face_nodes[face].iter().enumerate() {
                        let (nx, ny) = geom.normal(k, face, fi);
                        let lift = ops.lift_row_major[face][node * ops.n_face_nodes + fi]
                            * geom.lift_scale(k, face, fi, node);
                        for (c, (fx, fy)) in flux(k * nn + node).into_iter().enumerate() {
                            out[c][node] -= lift * (fx * nx + fy * ny);
                        }
                    }
                }
                for (rhs_k, out) in rhs.into_iter().zip(out.iter()) {
                    for (i, &d) in out.iter().enumerate() {
                        rhs_k[i * nl + level] += d;
                    }
                }
            },
        );
    }
}
