//! Layer volume transport of the 3D stages, consistent with the barotropic mode.
//!
//! # Formulation
//!
//! In σ-coordinates the volume of layer `l` obeys
//!
//! ```text
//!     ∂H_z,l/∂t + ∇·Q_l + Ω_{l+1/2} − Ω_{l−1/2} = 0,    H_z,l = D·Δσ_l,
//! ```
//!
//! with `Q_l` the horizontal transport of the layer (m²/s) and `Ω` the volume
//! flux through the σ-surfaces (m/s). A tracer `C` is advected with the same
//! fluxes, `∂(H_z C)/∂t = −∇·(Q C) − δ(Ω C)`. Constant `C` stays constant, and
//! the inventory `Σ H_z C` is conserved, only if the fluxes that move the
//! tracer are exactly the fluxes that move the water (Shchepetkin &
//! McWilliams 2005, §3; ROMS `set_depth`/`omega`).
//!
//! Under mode splitting the free surface is moved by the barotropic pass:
//! `η̄ − ηⁿ = −Δt·∇·DU_avg2` ([`crate::time::BarotropicTransport`]). The layer
//! transports of every 3D stage are therefore built as (ROMS `set_massflux`,
//! corrector step):
//!
//! 1. `Q_l = H_z,l u_l` at the nodes, and the central average
//!    `{{Q_l}}·n` at the element faces (at open boundaries the interior's
//!    `Q_l·n`, at walls none; [`crate::solver::rhs::boundary_3d`]);
//! 2. corrected by the layer's share of the difference to the barotropic
//!    transport, `Q_l += Δσ_l (DU_avg2 − Σ_m Q_m)` at the nodes and on the
//!    faces, so that `Σ_l Q_l = DU_avg2` exactly. The dissipative part of the
//!    2D face flux is shared out the same way;
//! 3. `Ω` integrated up from the bed, with the layer's share of the barotropic
//!    `∂η/∂t` and its volume source `s_l` from rivers ([`crate::source::river`]):
//!    `Ω_{l+1/2} = Ω_{l−1/2} − ∇·Q_l − Δσ_l ∂η/∂t + s_l`.
//!
//! The DG divergence ([`transport_divergence_element`]) is linear in the nodal
//! transport and the face flux, so `Σ_l ∇·Q_l = ∇·DU_avg2 = Σ_l s_l − ∂η/∂t`
//! and `Ω` at the surface vanishes to round-off. That holds wherever the barotropic pass
//! keeps the nodal identity; where it only keeps element balances (`WetDry`
//! elements with a dry node, positivity-limited elements, see
//! `BarotropicTransport`), the surface residual is spread linearly over the
//! column. It integrates to zero over the element (the element balance), so
//! every layer's continuity still holds for the element as a whole, and the
//! mode splitter carries the tracers of those elements as element means per
//! level ([`from_inventory`]): constant and conservative there
//! too. This is what makes 3D wetting and drying work.
//!
//! [`apply_tracer_transport_3d`] then advects a tracer with these fluxes in
//! inventory form: in split form within the elements
//! ([`advective_divergence_element`], which the 3D model's energy balance
//! over sloping σ-levels needs), upwind in `C` on the face fluxes, and on
//! `Ω` by one of
//! the [`VerticalAdvection`] schemes (by default fourth-order Akima under a
//! TVD limiter, [`VerticalAdvection::LimitedAkima`]).
//!
//! # Momentum
//!
//! The horizontal velocity is advected with the same fluxes, also in
//! inventory form ([`apply_momentum_transport_3d`]; ROMS `rhs3d`/`step3d_uv`):
//!
//! ```text
//!     ∂(H_z u)_l/∂t = −∇·(Q_l u_l) − (Ω_{l+1/2} u_{l+1/2} − Ω_{l−1/2} u_{l−1/2}) + …
//! ```
//!
//! upwind in `u` on the face fluxes and (by default) centred on `Ω`. A velocity that is
//! uniform in space changes its inventory exactly as the layer thickness
//! changes, and the layer momentum `∫ H_z,l u_l` is changed by the advection
//! only through open boundaries.

use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::mesh::data::BoundaryTag;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::core::blocks::{Pooled, for_each_block, max_over_blocks};
use crate::solver::rhs::advection_3d::{TracerBCContext3D, TracerBoundaryCondition3D};
use crate::solver::rhs::boundary_3d::{Boundaries3D, Exterior3D, ExteriorField, FaceExterior};
use crate::solver::state::Solution3D;
use crate::source::RiverInflow;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// The DG divergence of a transport `q = (hu, hv)` on element `k`, in the
/// strong (conservative) form of the 2D shallow-water kernel:
///
/// ```text
///     J⁻¹[Dr·(J∇r·q) + Ds·(J∇s·q)] − J⁻¹ Σ_f LIFT_f sJ_f (q·n − F*)
/// ```
///
/// `hu`, `hv` are the element's nodal values; `face` holds the numerical flux
/// `F*` out of every face node, `face · n_face_nodes + fi`. The result is
/// linear in `(hu, hv, face)`, which the layer transports rely on.
pub fn transport_divergence_element(
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    k: usize,
    hu: &[f64],
    hv: &[f64],
    face: &[f64],
    div: &mut [f64],
) {
    let (nn, nfn) = (ops.n_nodes, ops.n_face_nodes);
    for (i, d) in div.iter_mut().enumerate() {
        let (mut dr, mut ds) = (0.0, 0.0);
        for j in 0..nn {
            let ((ar_x, ar_y), (as_x, as_y)) = geom.contravariant(k, j);
            let (fr, fs) = (ar_x * hu[j] + ar_y * hv[j], as_x * hu[j] + as_y * hv[j]);
            dr += ops.dr[(i, j)] * fr;
            ds += ops.ds[(i, j)] * fs;
        }
        *d = geom.jacobian_inv(k, i) * (dr + ds);
    }
    for f in 0..4 {
        let f_star = &face[f * nfn..(f + 1) * nfn];
        for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
            let (nx, ny) = geom.normal(k, f, fi);
            let scale = geom.lift_scale(k, f, fi, node);
            let jump = nx * hu[node] + ny * hv[node] - f_star[fi];
            for (i, d) in div.iter_mut().enumerate() {
                *d -= scale * ops.lift[f][(i, fi)] * jump;
            }
        }
    }
}

/// The DG divergence of the advective flux `q φ` of a field `φ` carried by
/// the transport `q = (hu, hv)` on element `k`, with the volume term in
/// split (flux-differencing) form:
///
/// ```text
///     J⁻¹ Σ_j [Dr_ij ({{J∇r·q}}_ij) + Ds_ij ({{J∇s·q}}_ij)] 2{{φ}}_ij
///         − J⁻¹ Σ_f LIFT_f sJ_f (q·n φ − F*),
/// ```
///
/// `{{a}}_ij = ½(a_i + a_j)`. It is `½[∇·(qφ) + q·∇φ + φ∇·q]` at the nodes
/// (Kennedy & Gruber 2008; Gassner 2013 for the GLL summation-by-parts
/// property that makes it conservative), where
/// [`transport_divergence_element`] of `qφ` is `∇·(qφ)` alone, and the two
/// differ by aliasing only. With `φ` constant both are `φ∇·q`, so the
/// continuity of the layers (which keeps the conservative form) still
/// keeps a constant field constant.
///
/// Why the split form: the 3D model's energy balance needs the advection of
/// the background stratification to be the negative adjoint of the
/// baroclinic pressure gradient's `ρ∇z` term (see
/// [`crate::solver::rhs::baroclinic`]), and under GLL summation by parts
/// that holds for this form, not for the conservative one. Over a
/// stratified seamount at rest the conservative form let round-off grow
/// tenfold every 6 h (`examples/seamount_3d.rs`).
///
/// `hu`, `hv` and `phi` are the element's nodal values, `face` the
/// numerical flux `F*` of `qφ` out of every face node, `fr`, `fs` scratch
/// of `n_nodes` values.
#[allow(clippy::too_many_arguments)]
pub fn advective_divergence_element(
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    k: usize,
    hu: &[f64],
    hv: &[f64],
    phi: &[f64],
    face: &[f64],
    fr: &mut [f64],
    fs: &mut [f64],
    div: &mut [f64],
) {
    let (nn, nfn) = (ops.n_nodes, ops.n_face_nodes);
    for j in 0..nn {
        let ((ar_x, ar_y), (as_x, as_y)) = geom.contravariant(k, j);
        fr[j] = ar_x * hu[j] + ar_y * hv[j];
        fs[j] = as_x * hu[j] + as_y * hv[j];
    }
    for (i, d) in div.iter_mut().enumerate() {
        let mut sum = 0.0;
        for j in 0..nn {
            let flux = ops.dr[(i, j)] * (fr[i] + fr[j]) + ops.ds[(i, j)] * (fs[i] + fs[j]);
            sum += flux * (phi[i] + phi[j]);
        }
        *d = 0.5 * geom.jacobian_inv(k, i) * sum;
    }
    for f in 0..4 {
        let f_star = &face[f * nfn..(f + 1) * nfn];
        for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
            let (nx, ny) = geom.normal(k, f, fi);
            let scale = geom.lift_scale(k, f, fi, node);
            let jump = (nx * hu[node] + ny * hv[node]) * phi[node] - f_star[fi];
            for (i, d) in div.iter_mut().enumerate() {
                *d -= scale * ops.lift[f][(i, fi)] * jump;
            }
        }
    }
}

/// The barotropic side of one baroclinic step, as the layer transports need
/// it: `DU_avg2` and the rate of the free surface over the step.
#[derive(Clone, Copy)]
pub struct BarotropicFlux<'a> {
    /// Nodal barotropic transport `DU_avg2` (m²/s), `[element][node]`.
    pub hu: &'a [f64],
    pub hv: &'a [f64],
    /// Barotropic mass flux out of every element face node (m²/s),
    /// `(k · 4 + face) · n_face_nodes + fi`.
    pub face: &'a [f64],
    /// `∂η/∂t = (η̄ − ηⁿ)/Δt` over the step (m/s), `[element][node]`.
    pub eta_rate: &'a [f64],
    /// The rivers of the step, if any: volume sources of the layers, so
    /// that `∂η/∂t = −∇·DU_avg2 + Σ Q̄/A_k` (see [`crate::source::river`]).
    pub rivers: Option<RiverInflow<'a>>,
}

/// Layer transports and `Ω` of one 3D stage (see the module docs).
pub struct LayerTransport {
    n_levels: usize,
    /// Layer transport `Q_l = H_z u` at the nodes (m²/s), corrected to
    /// `DU_avg2`; layout of [`Solution3D`], `[element][node][level]`.
    pub hu: Vec<f64>,
    pub hv: Vec<f64>,
    /// Layer volume flux out of every element face node (m²/s),
    /// `((k · 4 + face) · n_face_nodes + fi) · n_levels + level`.
    pub face: Vec<f64>,
    /// `Ω` at the w-points (m/s), `n_levels + 1` per column, bed first.
    pub omega: Vec<f64>,
    /// σ-thickness `Δσ_l` of the layers (uniform until the first
    /// [`Self::compute`]).
    pub d_sigma: Vec<f64>,
    /// Largest `|Ω|` at the surface before the residual was spread over the
    /// column (m/s): round-off where the barotropic pass keeps the nodal
    /// identity `η̄ − ηⁿ = −Δt∇·DU_avg2`.
    pub surface_residual: f64,
}

impl LayerTransport {
    /// Buffers for `n_elements` elements of `ops` and `n_levels` layers.
    pub fn new(n_elements: usize, ops: &DGOperators2D, n_levels: usize) -> Self {
        let (nn, nfn) = (ops.n_nodes, ops.n_face_nodes);
        Self {
            n_levels,
            hu: vec![0.0; n_elements * nn * n_levels],
            hv: vec![0.0; n_elements * nn * n_levels],
            face: vec![0.0; n_elements * 4 * nfn * n_levels],
            omega: vec![0.0; n_elements * nn * (n_levels + 1)],
            d_sigma: vec![1.0 / n_levels as f64; n_levels],
            surface_residual: 0.0,
        }
    }

    /// Number of layers.
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }

    /// Layer transports and `Ω` of `state` (its `η`, `u`, `v`), corrected to
    /// the barotropic transport `barotropic`, with the physical boundaries
    /// `boundaries`.
    ///
    /// With `barotropic = None` they are the state's own transports: `H_z u`
    /// at the nodes and the central average on the faces, uncorrected, with
    /// `Ω` closed by the free-surface rate their divergence implies,
    /// `∂η/∂t = −Σ_l ∇·Q_l`. (The slow forcing of the mode splitter uses them
    /// before the barotropic pass of a step exists.)
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        state: &Solution3D,
        barotropic: Option<BarotropicFlux>,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        sigma: &SigmaGrid,
        bathymetry: &Bathymetry2D,
        boundaries: &Boundaries3D,
        exterior: &Exterior3D,
    ) {
        let (nn, nfn, nl) = (ops.n_nodes, ops.n_face_nodes, self.n_levels);
        assert_eq!(state.n_levels, nl, "layer count of the state");
        let n_elements = state.n_elements;
        let d_sigma = sigma.d_sigma();
        self.d_sigma.copy_from_slice(d_sigma);
        let Self {
            hu: layer_hu,
            hv: layer_hv,
            face: layer_face,
            omega: layer_omega,
            ..
        } = self;

        // 1. Nodal layer transports, corrected to DU_avg2
        for_each_block(
            n_elements,
            [&mut layer_hu[..], &mut layer_hv[..]],
            || (),
            |_, k, [hu_k, hv_k]| {
                let bed = bathymetry.element(ElementIndex::new(k));
                for (i, &b) in bed.iter().enumerate() {
                    let idx = k * nn + i;
                    let depth = state.eta.data[idx] - b;
                    let column = idx * nl..(idx + 1) * nl;
                    let local = i * nl..(i + 1) * nl;
                    let (mut sum_u, mut sum_v) = (0.0, 0.0);
                    for (((hu, hv), (&u, &v)), &ds) in hu_k[local.clone()]
                        .iter_mut()
                        .zip(&mut hv_k[local.clone()])
                        .zip(state.u[column.clone()].iter().zip(&state.v[column]))
                        .zip(d_sigma)
                    {
                        *hu = depth * ds * u;
                        *hv = depth * ds * v;
                        sum_u += *hu;
                        sum_v += *hv;
                    }
                    let Some(barotropic) = barotropic else {
                        continue;
                    };
                    let (corr_u, corr_v) = (barotropic.hu[idx] - sum_u, barotropic.hv[idx] - sum_v);
                    for ((hu, hv), &ds) in hu_k[local.clone()]
                        .iter_mut()
                        .zip(&mut hv_k[local])
                        .zip(d_sigma)
                    {
                        *hu += ds * corr_u;
                        *hv += ds * corr_v;
                    }
                }
            },
        );
        let (layer_hu, layer_hv): (&[f64], &[f64]) = (layer_hu, layer_hv);

        // 2. Face fluxes: central average of the nodal transports, corrected
        // to the barotropic face flux
        for_each_block(
            n_elements,
            [&mut layer_face[..]],
            || (),
            |_, k, [face_k]| {
                let el = ElementIndex::new(k);
                for f in 0..4 {
                    let face_exterior = boundaries.exterior(mesh, el, f);
                    for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
                        let (nx, ny) = geom.normal(k, f, fi);
                        let flat = k * nn + node;
                        let interior = flat * nl;
                        // The profile of the layer fluxes: central between
                        // elements, the interior's at open boundaries (the
                        // shear leaves with the flow) or central with a
                        // nesting parent's, none at walls
                        let across = match face_exterior {
                            FaceExterior::Element(nb) => Across::Node(
                                (nb.element * nn + ops.face_nodes[nb.face][nfn - 1 - fi]) * nl,
                            ),
                            FaceExterior::Open(tag) => match exterior.velocity {
                                Some([u, v]) if u.at(tag, flat, 0).is_some() => {
                                    let depth = state.eta.data[flat] - bathymetry.data[flat];
                                    Across::Parent(u, v, tag, depth)
                                }
                                _ => Across::Node(interior),
                            },
                            FaceExterior::Wall => Across::Nothing,
                        };
                        let local = f * nfn + fi;
                        let fluxes = &mut face_k[local * nl..(local + 1) * nl];
                        let mut sum = 0.0;
                        for (l, flux) in fluxes.iter_mut().enumerate() {
                            let q_in = nx * layer_hu[interior + l] + ny * layer_hv[interior + l];
                            *flux = match across {
                                Across::Node(e) => {
                                    0.5 * (q_in + nx * layer_hu[e + l] + ny * layer_hv[e + l])
                                }
                                Across::Parent(u, v, tag, depth) => {
                                    let u = u.at(tag, flat, l).expect("checked");
                                    let v = v.at(tag, flat, l).expect("checked");
                                    0.5 * (q_in + depth * d_sigma[l] * (nx * u + ny * v))
                                }
                                Across::Nothing => 0.0,
                            };
                            sum += *flux;
                        }
                        let Some(barotropic) = barotropic else {
                            continue;
                        };
                        let corr = barotropic.face[(k * 4 + f) * nfn + fi] - sum;
                        for (flux, &ds) in fluxes.iter_mut().zip(d_sigma) {
                            *flux += ds * corr;
                        }
                    }
                }
            },
        );
        let layer_face: &[f64] = layer_face;

        // 3. Ω from the bed up
        let sigma_w = sigma.sigma_w();
        let residual = max_over_blocks(
            n_elements,
            [&mut layer_omega[..]],
            || {
                Pooled::take(
                    |s: &OmegaScratch| s.fits(nn, nfn, nl),
                    || OmegaScratch::new(nn, nfn, nl),
                )
            },
            |scratch, k, [omega_k]| {
                let OmegaScratch {
                    hu,
                    hv,
                    face,
                    div,
                    source,
                } = &mut **scratch;
                // The rivers' volume sources of the layers, the same at
                // every node of the element
                source.fill(0.0);
                if let Some(rivers) = barotropic.and_then(|b| b.rivers) {
                    rivers.add_layer_rates(k, source);
                }
                for l in 0..nl {
                    for i in 0..nn {
                        hu[i] = layer_hu[(k * nn + i) * nl + l];
                        hv[i] = layer_hv[(k * nn + i) * nl + l];
                    }
                    for (slot, flux) in face.iter_mut().enumerate() {
                        *flux = layer_face[((k * 4 * nfn) + slot) * nl + l];
                    }
                    transport_divergence_element(
                        ops,
                        geom,
                        k,
                        hu,
                        hv,
                        face,
                        &mut div[l * nn..(l + 1) * nn],
                    );
                }
                let mut largest = 0.0_f64;
                for i in 0..nn {
                    let idx = k * nn + i;
                    let rate = match barotropic {
                        Some(barotropic) => barotropic.eta_rate[idx],
                        None => -(0..nl).map(|l| div[l * nn + i]).sum::<f64>(),
                    };
                    let omega = &mut omega_k[i * (nl + 1)..(i + 1) * (nl + 1)];
                    omega[0] = 0.0;
                    for l in 0..nl {
                        omega[l + 1] = omega[l] - div[l * nn + i] - d_sigma[l] * rate + source[l];
                    }
                    let residual = omega[nl];
                    largest = largest.max(residual.abs());
                    for (w, &s) in omega.iter_mut().zip(sigma_w) {
                        *w -= (s + 1.0) * residual;
                    }
                }
                largest
            },
        );
        self.surface_residual = residual.max(0.0);
    }

    /// The transports of the control volumes around the w-points of `layers`
    /// (the w-cells), for advecting a field that lives at the w-points (the
    /// turbulence of [`crate::physics::GlsMixing`]; ROMS `gls_corstep`).
    /// `self` must have one level more than `layers`.
    ///
    /// The w-cell of an interior w-point spans from the centre of the layer
    /// below to the centre of the layer above; those of the bed and the
    /// surface are the lower half of the bed layer and the upper half of the
    /// top layer ([`w_cell_thicknesses`]). Each carries half of the
    /// horizontal transport of the two layers it overlaps, nodally and on
    /// the faces, and passes the mean of the layer's two `Ω` through a layer
    /// centre (none through the bed and the surface). Averaging the
    /// continuity of the two layers gives the w-cell's, so a field advected
    /// with these fluxes keeps the constancy and conservation of
    /// [`apply_tracer_transport_3d`].
    pub fn stagger_from(&mut self, layers: &LayerTransport) {
        let nl = layers.n_levels;
        assert_eq!(self.n_levels, nl + 1, "w-cells of {nl} layers");
        let nw = nl + 1;
        for (w_cells, layer_values) in [
            (&mut self.hu, &layers.hu),
            (&mut self.hv, &layers.hv),
            (&mut self.face, &layers.face),
        ] {
            for (w, l) in w_cells
                .chunks_exact_mut(nw)
                .zip(layer_values.chunks_exact(nl))
            {
                w[0] = 0.5 * l[0];
                for j in 1..nl {
                    w[j] = 0.5 * (l[j - 1] + l[j]);
                }
                w[nl] = 0.5 * l[nl - 1];
            }
        }
        for (w, omega) in self
            .omega
            .chunks_exact_mut(nw + 1)
            .zip(layers.omega.chunks_exact(nl + 1))
        {
            w[0] = 0.0;
            for l in 0..nl {
                w[l + 1] = 0.5 * (omega[l] + omega[l + 1]);
            }
            w[nw] = 0.0;
        }
        w_cell_thicknesses(&layers.d_sigma, &mut self.d_sigma);
        self.surface_residual = layers.surface_residual;
    }
}

/// σ-thicknesses of the w-cells of layers of σ-thickness `d_sigma` into
/// `out` (one more): half the bed layer, the mean of each pair of adjacent
/// layers, half the top layer (see [`LayerTransport::stagger_from`]). They
/// sum to one, and a w-cell's thickness is `D` times its value.
pub fn w_cell_thicknesses(d_sigma: &[f64], out: &mut [f64]) {
    let nl = d_sigma.len();
    assert_eq!(out.len(), nl + 1, "w-cells of {nl} layers");
    out[0] = 0.5 * d_sigma[0];
    for j in 1..nl {
        out[j] = 0.5 * (d_sigma[j - 1] + d_sigma[j]);
    }
    out[nl] = 0.5 * d_sigma[nl - 1];
}

/// One element's layers of [`LayerTransport::compute`]: a layer's nodal
/// transport and face fluxes, the divergence of every layer, and the
/// layers' volume sources.
struct OmegaScratch {
    hu: Vec<f64>,
    hv: Vec<f64>,
    face: Vec<f64>,
    div: Vec<f64>,
    source: Vec<f64>,
}

impl OmegaScratch {
    fn new(nn: usize, nfn: usize, nl: usize) -> Self {
        Self {
            hu: vec![0.0; nn],
            hv: vec![0.0; nn],
            face: vec![0.0; 4 * nfn],
            div: vec![0.0; nn * nl],
            source: vec![0.0; nl],
        }
    }

    fn fits(&self, nn: usize, nfn: usize, nl: usize) -> bool {
        self.hu.len() == nn
            && self.face.len() == 4 * nfn
            && self.div.len() == nn * nl
            && self.source.len() == nl
    }
}

/// What lies across a face for the layer volume fluxes.
#[derive(Clone, Copy)]
enum Across<'a> {
    /// A node's layer transports, at this offset.
    Node(usize),
    /// A nesting parent's velocity, over the interior's depth.
    Parent(
        ExteriorField<'a>,
        ExteriorField<'a>,
        crate::mesh::data::BoundaryTag,
        f64,
    ),
    /// A wall.
    Nothing,
}

/// Layer thickness `H_z = D·Δσ` (m), floored at the same small positive value
/// as the 3D advection kernels, so that dividing an inventory by it is safe.
#[inline]
pub(crate) fn layer_thickness_of(depth: f64, d_sigma: f64) -> f64 {
    (depth * d_sigma).max(crate::solver::rhs::advection_3d::MIN_LAYER_THICKNESS)
}

/// Depth (m) below which a node's (or, as a mean, an element's) tracers are
/// left as they were: it holds no water to define a concentration.
const DRY_DEPTH: f64 = 1e-6;

/// Multiply a field by the thickness of its cells under `η`: concentration
/// `C` → inventory `H_z C`, with `H_z = D·Δσ` for the σ-thicknesses
/// `d_sigma` of the cells in a column (the layers' [`SigmaGrid::d_sigma`],
/// or the w-cells' of [`w_cell_thicknesses`] for a field at the w-points).
pub fn to_inventory(tracer: &mut [f64], eta: &[f64], d_sigma: &[f64], bathymetry: &Bathymetry2D) {
    let (nn, nl) = (bathymetry.n_nodes, d_sigma.len());
    let n_elements = bathymetry.n_elements;
    for_each_block(
        n_elements,
        [&mut tracer[..n_elements * nn * nl]],
        || (),
        |_, k, [block]| {
            let nodes = k * nn..(k + 1) * nn;
            for ((column, &e), &b) in block
                .chunks_exact_mut(nl)
                .zip(&eta[nodes.clone()])
                .zip(&bathymetry.data[nodes])
            {
                for (c, &ds) in column.iter_mut().zip(d_sigma) {
                    *c *= layer_thickness_of(e - b, ds);
                }
            }
        },
    );
}

/// Inventory `q = H_z C` → concentration, written to `out`, for cells of
/// σ-thickness `d_sigma` in a column (as [`to_inventory`]).
///
/// - Elements not marked in `element_means`: `C = q / H_z` at every node.
/// - Marked elements: per level, the element's inventory over its volume,
///   `Σ w_i J_i q_il / Σ w_i J_i H_z,il`, at every node. The mode splitter
///   marks the elements where the barotropic pass kept only the element
///   balance, not the nodal identity `η̄ − ηⁿ = −Δt∇·DU_avg2` (`WetDry`
///   subcell and positivity-limited elements, see `BarotropicTransport`):
///   there the nodal quotient is not constant for a constant tracer, but the
///   element means are (and are conservative, see the module docs).
/// - A node with (almost) no water, shallower than `DRY_DEPTH`, and a level
///   of a marked element with a mean depth below it, keep the value `out`
///   already holds (the last concentration) instead of dividing by a
///   vanishing layer.
///
/// Converting back with [`to_inventory`] keeps each element's
/// inventory per level exactly.
#[allow(clippy::too_many_arguments)]
pub fn from_inventory(
    q: &[f64],
    eta: &[f64],
    d_sigma: &[f64],
    bathymetry: &Bathymetry2D,
    geom: &GeometricFactors2D,
    element_means: &[bool],
    out: &mut [f64],
) {
    let nl = d_sigma.len();
    let nn = bathymetry.n_nodes;
    let n_elements = bathymetry.n_elements;
    for_each_block(
        n_elements,
        [&mut out[..n_elements * nn * nl]],
        || (),
        |_, k, [out_k]| {
            let bed = bathymetry.element(ElementIndex::new(k));
            let eta_k = &eta[k * nn..(k + 1) * nn];
            let q_k = &q[k * nn * nl..(k + 1) * nn * nl];
            if !element_means[k] {
                for (i, (&e, &b)) in eta_k.iter().zip(bed).enumerate() {
                    if e - b < DRY_DEPTH {
                        continue;
                    }
                    for (l, &ds) in d_sigma.iter().enumerate() {
                        out_k[i * nl + l] = q_k[i * nl + l] / layer_thickness_of(e - b, ds);
                    }
                }
                return;
            }
            let area: f64 = (0..nn).map(|i| geom.node_mass(k, i)).sum();
            for (l, &ds) in d_sigma.iter().enumerate() {
                let (mut inventory, mut volume) = (0.0, 0.0);
                for (i, (&e, &b)) in eta_k.iter().zip(bed).enumerate() {
                    let mass = geom.node_mass(k, i);
                    inventory += mass * q_k[i * nl + l];
                    volume += mass * layer_thickness_of(e - b, ds);
                }
                if volume > area * ds * DRY_DEPTH {
                    let mean = inventory / volume;
                    for i in 0..nn {
                        out_k[i * nl + l] = mean;
                    }
                }
            }
        },
    );
}

/// Overwrite `rhs` with the inventory tendency `∂(H_z C)/∂t` of the tracer
/// concentration `tracer`, advected by the layer transports `transport`:
///
/// ```text
///     ∂(H_z C)_l/∂t = −∇·(Q_l C_l) − (Ω_{l+1/2} C_{l+1/2} − Ω_{l−1/2} C_{l−1/2})
/// ```
///
/// Horizontally the nodal flux is `Q_l C_l` and the face flux `F_l C↑`, with
/// `C↑` the concentration upwind of the layer's face flux `F_l`: the element's
/// own on outflow, the neighbour's on inflow, and `bc`'s exterior value on
/// inflow through a physical boundary (a nesting parent's value, from
/// `exterior`, where it has one). Vertically `C_{l±1/2}` is reconstructed
/// by `vertical` ([`VerticalAdvection`]).
///
/// For constant `C` the tendency is `C·(−∇·Q_l − δΩ_l) = C·Δσ_l ∂η/∂t`, the
/// change of the layer's thickness: constancy. Both fluxes are single-valued
/// on each face, so the inventory is conserved.
#[allow(clippy::too_many_arguments)]
pub fn apply_tracer_transport_3d(
    rhs: &mut [f64],
    tracer: &[f64],
    transport: &LayerTransport,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    bc: &dyn TracerBoundaryCondition3D,
    exterior: Option<ExteriorField>,
    boundaries: &Boundaries3D,
    vertical: VerticalAdvection,
) {
    let (nn, nl) = (ops.n_nodes, transport.n_levels);
    let context = LayerContext {
        transport,
        mesh,
        ops,
        geom,
        boundaries,
    };
    for_each_block(
        mesh.n_elements,
        [&mut rhs[..mesh.n_elements * nn * nl]],
        || TransportScratch::take(ops, nl),
        |scratch, k, [rhs_k]| {
            for l in 0..nl {
                let inflow = |f, fi, node, tag, interior, flux| {
                    exterior
                        .and_then(|e| e.at(tag, node, l))
                        .unwrap_or_else(|| {
                            bc.exterior_value(&TracerBCContext3D {
                                element: k,
                                face: f,
                                level: l,
                                face_node: fi,
                                boundary_tag: Some(tag),
                                interior_value: interior,
                                normal_velocity: flux,
                            })
                        })
                };
                let div = context.flux_divergence(k, l, tracer, true, inflow, scratch);
                for (i, &d) in div.iter().enumerate() {
                    rhs_k[i * nl + l] = -d;
                }
            }

            subtract_vertical_flux(rhs_k, tracer, transport, k, nn, vertical, scratch);
        },
    );
}

/// Subtract `δ(Ω φ)`, with `φ` at the σ-surfaces reconstructed by `scheme`,
/// from `rhs_k` (element `k`'s block, `[node][level]`) in every column of
/// element `k`.
fn subtract_vertical_flux(
    rhs_k: &mut [f64],
    field: &[f64],
    transport: &LayerTransport,
    k: usize,
    nn: usize,
    scheme: VerticalAdvection,
    scratch: &mut TransportScratch,
) {
    let nl = transport.n_levels;
    let TransportScratch {
        vertical: flux,
        slope,
        ..
    } = scratch;
    for i in 0..nn {
        let idx = k * nn + i;
        let omega = &transport.omega[idx * (nl + 1)..(idx + 1) * (nl + 1)];
        let column = &field[idx * nl..(idx + 1) * nl];
        scheme.surface_values(column, omega, &transport.d_sigma, slope, flux);
        for l in 0..=nl {
            flux[l] *= omega[l];
        }
        for (l, r) in rhs_k[i * nl..(i + 1) * nl].iter_mut().enumerate() {
            *r -= flux[l + 1] - flux[l];
        }
    }
}

/// Reconstruction of a tracer at the σ-surfaces `l ± 1/2` for its vertical
/// advection `−δ(Ω C)` ([`apply_tracer_transport_3d`]). All are
/// conservative, keep a constant tracer constant and take the layers'
/// thicknesses into account (`H_z ∝ Δσ` within a column), so a linear
/// profile is reconstructed exactly on stretched levels too, except by
/// `Upwind`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VerticalAdvection {
    /// Second-order centred, `C_{l−1/2} = ½(C_{l−1} + C_l)`: non-dissipative
    /// and linear, the momentum's default (ROMS `UV_C2VADVECTION`). It
    /// oscillates at fronts and is exact for a linear profile only on
    /// uniform levels. Stable with the SSP-RK3 stages up to a vertical
    /// Courant number `|Ω|Δt/H_z` ≈ 1.7 (√3).
    Centred,
    /// First-order upwind on `Ω`: monotone, but it diffuses with
    /// `κ ≈ |w|Δz/2`. A mode-1 internal seiche on 10 uniform levels runs
    /// 1.1 % slow and loses 1.2 % of its amplitude per period (Akima: 0.37 %
    /// and 0.14 %).
    Upwind,
    /// Fourth-order Akima (ROMS `TS_A4VADVECTION`; Shchepetkin & McWilliams
    /// 2005, §3): the surface value of the cubic that has the two layers'
    /// means and, at their centres, the harmonic-mean slopes
    /// `d = 2δ₋δ₊/(δ₋ + δ₊)` of the neighbouring gradients (zero at an
    /// extremum, one-sided at the bed and the surface). On layers `a` below
    /// and `b` above,
    ///
    /// ```text
    ///     C_{l−1/2} = (b C_{l−1} + a C_l)/(a + b) + ab (d_{l−1} − d_l) / (3(a + b)),
    /// ```
    ///
    /// ROMS's `½(C_{l−1} + C_l) − (d_l − d_{l−1})/6` on uniform levels (ROMS
    /// applies that in index space on stretched levels too). Centred: no
    /// numerical diffusion, but not monotone; it over- and undershoots
    /// slightly at extrema and fronts, where the zero slope leaves
    /// second-order centred.
    ///
    /// With the SSP-RK3 stages it is stable up to a vertical Courant number
    /// `|Ω|Δt/H_z` ≈ 1.2 (the fourth-order centred operator's eigenvalues
    /// reach 1.37 `|Ω|/H_z` on the imaginary axis, RK3's limit there is √3);
    /// upwind up to 1.
    Akima,
    /// Third-order upwind-biased with a TVD limiter: the surface value of the
    /// parabola with the means of the upwind layer, the one beyond it and the
    /// downwind one (`C_{l−1/2} = (5C_{l−1} + 2C_l − C_{l−2})/6` on uniform
    /// levels, upwind `C_{l−1}`), its increment over the upwind layer's mean
    /// limited to at most the upwind and downwind differences and zero at an
    /// extremum (Koren 1993). This is the spatial part of HSIMT (Wu & Zhu
    /// 2010, ROMS `TS_HSIMT`), whose Courant-number terms the SSP-RK3 stages
    /// replace. Out of the bed or the surface layer, where there is no layer
    /// beyond, the surface value is linear through the two end layers'
    /// means: first order there would freeze the end layer's mean where the
    /// horizontal flow vanishes (at a wall), and a mode-1 internal seiche
    /// with a thick bed layer drifted by a third of its amplitude per period.
    ///
    /// Monotone in the interior: no new extrema up to a vertical Courant
    /// number `|Ω|Δt/H_z` of ½ (SSP-RK3 inherits the forward-Euler bound).
    /// The end layers' means can leave the column's range slightly, as the
    /// water at the bed or the surface is carried into the layer above or
    /// below. The limiter adds some diffusion at extrema and fronts.
    Tvd,
    /// [`Self::Akima`]'s surface values under [`Self::Tvd`]'s limiter: the
    /// increment over the upwind layer's mean at most the upwind and
    /// downwind differences and zero at an extremum (out of an end layer,
    /// the value between the two layers'), i.e. bounded by the layers below
    /// and above. Where a profile is smooth and monotone Akima's value lies
    /// within those bounds and is kept. At the foot of a front, where the
    /// upwind layer's harmonic-mean slope is small and the next one steep,
    /// Akima's centred value lies far towards the downwind layer's, more
    /// than the upwind gradient allows, and the scheme oscillates there; it
    /// is clipped to the bound.
    ///
    /// Akima's undershoots at a sharp interface are in the layers' element
    /// means, which no conservative slope limiter of the nodal values can
    /// change: a P2 lock exchange on 125 m (ν = 10 m²/s) left 7.27 °C in a
    /// layer of a 7.5–12.5 °C range. The bounds have to be on the flux.
    /// Monotone in the interior up to a vertical Courant number of ½, as
    /// [`Self::Tvd`]. The tracers' default.
    #[default]
    LimitedAkima,
}

impl VerticalAdvection {
    /// `C` at the `nl + 1` σ-surfaces of `column` (bed first) into
    /// `surface`, for the volume fluxes `omega` (which decide the upwind
    /// side) through layers of σ-thickness `d_sigma`. The bed and surface
    /// pass no flux; their values are zero. `slope` holds `nl` values of
    /// scratch.
    fn surface_values(
        self,
        column: &[f64],
        omega: &[f64],
        d_sigma: &[f64],
        slope: &mut [f64],
        surface: &mut [f64],
    ) {
        let nl = column.len();
        surface[0] = 0.0;
        surface[nl] = 0.0;
        match self {
            Self::Centred => {
                for l in 1..nl {
                    surface[l] = 0.5 * (column[l - 1] + column[l]);
                }
            }
            Self::Upwind => {
                for l in 1..nl {
                    surface[l] = if omega[l] >= 0.0 {
                        column[l - 1]
                    } else {
                        column[l]
                    };
                }
            }
            Self::Akima if nl > 1 => akima_surface_values(column, d_sigma, slope, surface),
            Self::LimitedAkima if nl > 1 => {
                akima_surface_values(column, d_sigma, slope, surface);
                for l in 1..nl {
                    let (up, far, down) = upwind_stencil(l, nl, omega[l]);
                    let Some(far) = far else {
                        // Out of an end layer: between the two layers' means
                        let (lo, hi) = (column[up].min(column[down]), column[up].max(column[down]));
                        surface[l] = surface[l].max(lo).min(hi);
                        continue;
                    };
                    surface[l] = column[up]
                        + tvd_limited(
                            surface[l] - column[up],
                            column[up] - column[far],
                            column[down] - column[up],
                        );
                }
            }
            Self::Tvd => {
                for l in 1..nl {
                    let (up, far, down) = upwind_stencil(l, nl, omega[l]);
                    let Some(far) = far else {
                        // Out of an end layer: linear through the two layers'
                        // means (first order would freeze the end layer's
                        // mean against a wall, where u = 0)
                        let (a, b) = (d_sigma[up], d_sigma[down]);
                        surface[l] = column[up] + a / (a + b) * (column[down] - column[up]);
                        continue;
                    };
                    let (a, b, c) = (d_sigma[up], d_sigma[down], d_sigma[far]);
                    let (behind, ahead) = (column[up] - column[far], column[down] - column[up]);
                    let increment = a * b / ((a + c) * (a + b + c)) * behind
                        + a * (a + c) / ((a + b) * (a + b + c)) * ahead;
                    surface[l] = column[up] + tvd_limited(increment, behind, ahead);
                }
            }
            Self::Akima | Self::LimitedAkima => {}
        }
    }
}

/// The layers around σ-surface `l` of a column of `nl` for the volume flux
/// `omega` through it: the upwind layer, the one beyond it (if any) and the
/// downwind layer.
fn upwind_stencil(l: usize, nl: usize, omega: f64) -> (usize, Option<usize>, usize) {
    if omega >= 0.0 {
        (l - 1, l.checked_sub(2), l)
    } else {
        (l, Some(l + 1).filter(|&f| f < nl), l - 1)
    }
}

/// Koren's (1993) TVD bound on the `increment` of a surface value over its
/// upwind layer's mean: at most the differences `behind` (upwind layer
/// minus the one beyond) and `ahead` (downwind minus upwind), and zero at an
/// extremum or against the flow's gradient.
fn tvd_limited(increment: f64, behind: f64, ahead: f64) -> f64 {
    if behind * ahead > 0.0 && increment * ahead > 0.0 {
        increment.signum() * increment.abs().min(behind.abs()).min(ahead.abs())
    } else {
        0.0
    }
}

/// [`VerticalAdvection::Akima`]'s values at the interior σ-surfaces of
/// `column` (`nl > 1` layers of σ-thickness `d_sigma`) into `surface`, with
/// the harmonic-mean slopes in `slope`.
fn akima_surface_values(column: &[f64], d_sigma: &[f64], slope: &mut [f64], surface: &mut [f64]) {
    let nl = column.len();
    // The gradient across surface l (between the centres of layers l − 1
    // and l), repeated at the bed and the surface
    let gradient = |l: usize| {
        let l = l.clamp(1, nl - 1);
        2.0 * (column[l] - column[l - 1]) / (d_sigma[l - 1] + d_sigma[l])
    };
    for (l, d) in slope.iter_mut().enumerate() {
        let (below, above) = (gradient(l), gradient(l + 1));
        let product = below * above;
        *d = if product > 0.0 {
            2.0 * product / (below + above)
        } else {
            0.0
        };
    }
    for l in 1..nl {
        let (a, b) = (d_sigma[l - 1], d_sigma[l]);
        surface[l] =
            (b * column[l - 1] + a * column[l] + a * b * (slope[l - 1] - slope[l]) / 3.0) / (a + b);
    }
}

/// Add the inventory tendency `∂(H_z u)/∂t` of the momentum advection by the
/// layer transports `transport` to `rhs_u` and `rhs_v`:
///
/// ```text
///     ∂(H_z u)_l/∂t += −∇·(Q_l u_l) − (Ω_{l+1/2} u_{l+1/2} − Ω_{l−1/2} u_{l−1/2})
/// ```
///
/// for the advected velocity `(u, v)`; `Q` and `Ω` come from `transport`.
/// Horizontally the face flux is `F_l u↑`, with `u↑` upwind of the layer's
/// face flux as for the tracers. Flowing in through an open boundary it is
/// a nesting parent's velocity (`exterior`) where there is one, otherwise
/// the interior's (zero gradient, ROMS's "gradient" condition for the 3D
/// velocity; see [`crate::solver::rhs::boundary_3d`]). Walls carry no volume, so no
/// momentum. Vertically `u` at the σ-surfaces is reconstructed by `vertical`
/// (`Hydrostatic3D` uses [`VerticalAdvection::Centred`] by default).
///
/// A velocity uniform in space gets `u·Δσ_l ∂η/∂t`: divided by the new layer
/// thickness it stays uniform. The layer momentum `∫ H_z,l u_l` changes
/// only through open faces.
#[allow(clippy::too_many_arguments)]
pub fn apply_momentum_transport_3d(
    rhs_u: &mut [f64],
    rhs_v: &mut [f64],
    u: &[f64],
    v: &[f64],
    transport: &LayerTransport,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    boundaries: &Boundaries3D,
    exterior: Option<[ExteriorField; 2]>,
    vertical: VerticalAdvection,
) {
    let (nn, nl) = (ops.n_nodes, transport.n_levels);
    let context = LayerContext {
        transport,
        mesh,
        ops,
        geom,
        boundaries,
    };
    let [exterior_u, exterior_v] = exterior.map_or([None, None], |[u, v]| [Some(u), Some(v)]);
    let n = mesh.n_elements * nn * nl;
    for_each_block(
        mesh.n_elements,
        [&mut rhs_u[..n], &mut rhs_v[..n]],
        || TransportScratch::take(ops, nl),
        |scratch, k, [rhs_u, rhs_v]| {
            for (rhs_k, field, exterior) in [(rhs_u, u, exterior_u), (rhs_v, v, exterior_v)] {
                for l in 0..nl {
                    let inflow = |_, _, node, tag, interior, _| {
                        exterior
                            .and_then(|e| e.at(tag, node, l))
                            .unwrap_or(interior)
                    };
                    let div = context.flux_divergence(k, l, field, false, inflow, scratch);
                    for (i, &d) in div.iter().enumerate() {
                        rhs_k[i * nl + l] -= d;
                    }
                }

                subtract_vertical_flux(rhs_k, field, transport, k, nn, vertical, scratch);
            }
        },
    );
}

/// What the horizontal flux divergence of a layer needs.
struct LayerContext<'a> {
    transport: &'a LayerTransport,
    mesh: &'a Mesh2D,
    ops: &'a DGOperators2D,
    geom: &'a GeometricFactors2D,
    boundaries: &'a Boundaries3D,
}

impl LayerContext<'_> {
    /// `∇·(Q_l φ)` on element `k`, layer `l`, of the field `φ` (layout of
    /// [`Solution3D`]): in the element in split form
    /// ([`advective_divergence_element`]) if `split`, else conservative
    /// ([`transport_divergence_element`] of `Q_l φ`); face flux `F_l φ↑`
    /// with `φ↑` upwind of `F_l`. On inflow through an open face `φ↑` is
    /// `inflow(face, face_node, node, tag, interior value, F_l)`, with `node`
    /// the face node's `[element][node]` index. Written to (and returned
    /// from) `scratch.div`.
    fn flux_divergence<'s>(
        &self,
        k: usize,
        l: usize,
        field: &[f64],
        split: bool,
        inflow: impl Fn(usize, usize, usize, BoundaryTag, f64, f64) -> f64,
        scratch: &'s mut TransportScratch,
    ) -> &'s [f64] {
        let (ops, transport) = (self.ops, self.transport);
        let (nn, nfn, nl) = (ops.n_nodes, ops.n_face_nodes, transport.n_levels);
        let TransportScratch {
            hu,
            hv,
            phi,
            fr,
            fs,
            face,
            div,
            ..
        } = scratch;
        for i in 0..nn {
            let idx = (k * nn + i) * nl + l;
            hu[i] = transport.hu[idx];
            hv[i] = transport.hv[idx];
            phi[i] = field[idx];
        }
        let el = ElementIndex::new(k);
        for f in 0..4 {
            let exterior = self.boundaries.exterior(self.mesh, el, f);
            for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
                let slot = (k * 4 + f) * nfn + fi;
                let flux = transport.face[slot * nl + l];
                let interior = field[(k * nn + node) * nl + l];
                let upwind = match exterior {
                    _ if flux >= 0.0 => interior,
                    FaceExterior::Element(nb) => {
                        let nb_node = ops.face_nodes[nb.face][nfn - 1 - fi];
                        field[(nb.element * nn + nb_node) * nl + l]
                    }
                    FaceExterior::Open(tag) => inflow(f, fi, k * nn + node, tag, interior, flux),
                    // A wall passes no volume: only a round-off flux
                    FaceExterior::Wall => interior,
                };
                face[f * nfn + fi] = flux * upwind;
            }
        }
        if split {
            advective_divergence_element(ops, self.geom, k, hu, hv, phi, face, fr, fs, div);
        } else {
            for ((hu, hv), &phi) in hu.iter_mut().zip(hv.iter_mut()).zip(phi.iter()) {
                *hu *= phi;
                *hv *= phi;
            }
            transport_divergence_element(ops, self.geom, k, hu, hv, face, div);
        }
        div
    }
}

/// Buffers of the transport kernels ([`apply_tracer_transport_3d`],
/// [`apply_momentum_transport_3d`]), sized for one element.
struct TransportScratch {
    hu: Vec<f64>,
    hv: Vec<f64>,
    phi: Vec<f64>,
    fr: Vec<f64>,
    fs: Vec<f64>,
    face: Vec<f64>,
    div: Vec<f64>,
    vertical: Vec<f64>,
    slope: Vec<f64>,
}

impl TransportScratch {
    /// Buffers for elements of `ops` and `n_levels` layers, from this
    /// thread's cache.
    fn take(ops: &DGOperators2D, n_levels: usize) -> Pooled<Self> {
        Pooled::take(
            |s: &Self| {
                s.hu.len() == ops.n_nodes
                    && s.face.len() == 4 * ops.n_face_nodes
                    && s.slope.len() == n_levels
            },
            || Self::new(ops, n_levels),
        )
    }

    /// Buffers for elements of `ops` and `n_levels` layers.
    fn new(ops: &DGOperators2D, n_levels: usize) -> Self {
        Self {
            hu: vec![0.0; ops.n_nodes],
            hv: vec![0.0; ops.n_nodes],
            phi: vec![0.0; ops.n_nodes],
            fr: vec![0.0; ops.n_nodes],
            fs: vec![0.0; ops.n_nodes],
            face: vec![0.0; 4 * ops.n_face_nodes],
            div: vec![0.0; ops.n_nodes],
            vertical: vec![0.0; n_levels + 1],
            slope: vec![0.0; n_levels],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::data::BoundaryTag;
    use crate::solver::rhs::advection_3d::{
        ExtrapolationTracerBC3D, FixedTracerBC3D, UpwindTracerBC3D,
    };
    use std::f64::consts::PI;

    const LX: f64 = 2000.0;
    const LY: f64 = 1000.0;

    /// A closed, periodic or open basin over a sloping, uneven bed with a
    /// sheared, non-uniform flow, a barotropic transport that is not the depth
    /// integral of the 3D velocities, and the free-surface rate that transport
    /// implies.
    struct Case {
        mesh: Mesh2D,
        ops: DGOperators2D,
        geom: GeometricFactors2D,
        sigma: SigmaGrid,
        bathymetry: Bathymetry2D,
        state: Solution3D,
        du_hu: Vec<f64>,
        du_hv: Vec<f64>,
        du_face: Vec<f64>,
        eta_rate: Vec<f64>,
        boundaries: Boundaries3D,
    }

    fn wave(x: f64, y: f64) -> f64 {
        (2.0 * PI * x / LX).sin() * (PI * y / LY).cos()
    }

    impl Case {
        fn new(mesh: Mesh2D, closed: bool) -> Self {
            let ops = DGOperators2D::new(2);
            let geom = GeometricFactors2D::compute(&mesh, &ops);
            let sigma = SigmaGrid::uniform(4);
            let (nn, nfn, nl) = (ops.n_nodes, ops.n_face_nodes, sigma.n_levels());
            let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
                -20.0 - 0.01 * x + 2.0 * (PI * y / LY).cos()
            });
            let mut state = Solution3D::new(mesh.n_elements, nn, nl);
            let mut du_hu = vec![0.0; mesh.n_elements * nn];
            let mut du_hv = vec![0.0; mesh.n_elements * nn];
            for k in 0..mesh.n_elements {
                let el = ElementIndex::new(k);
                for i in 0..nn {
                    let [x, y] = mesh.reference_to_physical(el, ops.nodes_r[i], ops.nodes_s[i]);
                    let idx = k * nn + i;
                    state.eta.data[idx] = 0.3 * wave(x, y);
                    for l in 0..nl {
                        let s = sigma.sigma_rho()[l];
                        state.u[idx * nl + l] = 0.4 * wave(x, y) + 0.2 * (s + 0.5);
                        state.v[idx * nl + l] = -0.2 + 0.1 * wave(y, x) - 0.1 * s;
                        state.temp[idx * nl + l] = 8.0 + 1e-3 * x - 2e-3 * y + l as f64;
                    }
                    // A transport with no normal component at the walls
                    let taper = if closed {
                        (PI * x / LX).sin() * (PI * y / LY).sin()
                    } else {
                        1.0
                    };
                    du_hu[idx] = 3.0 * taper * (1.0 + 0.5 * wave(x, y));
                    du_hv[idx] = -1.5 * taper * wave(y, x);
                }
            }
            // The barotropic face flux as a 2D kernel gives it: central average
            // of the nodal transport plus a jump term, single-valued on each
            // face, zero through walls; at open faces the interior's normal
            // transport plus an offset (an open-boundary condition's flux
            // differs from the interior's)
            let boundaries = Boundaries3D::default();
            let mut du_face = vec![0.0; mesh.n_elements * 4 * nfn];
            for k in 0..mesh.n_elements {
                let el = ElementIndex::new(k);
                for f in 0..4 {
                    let exterior = boundaries.exterior(&mesh, el, f);
                    for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
                        let (nx, ny) = geom.normal(k, f, fi);
                        let a = k * nn + node;
                        let qa = nx * du_hu[a] + ny * du_hv[a];
                        du_face[(k * 4 + f) * nfn + fi] = match exterior {
                            FaceExterior::Element(nb) => {
                                let b = nb.element * nn + ops.face_nodes[nb.face][nfn - 1 - fi];
                                let qb = nx * du_hu[b] + ny * du_hv[b];
                                0.5 * (qa + qb) - 0.3 * (state.eta.data[b] - state.eta.data[a])
                            }
                            FaceExterior::Open(_) => qa + 0.4 * state.eta.data[a],
                            FaceExterior::Wall => 0.0,
                        };
                    }
                }
            }
            // ∂η/∂t = −∇·DU_avg2, the nodal identity of the barotropic pass
            let mut eta_rate = vec![0.0; du_hu.len()];
            for k in 0..mesh.n_elements {
                transport_divergence_element(
                    &ops,
                    &geom,
                    k,
                    &du_hu[k * nn..(k + 1) * nn],
                    &du_hv[k * nn..(k + 1) * nn],
                    &du_face[k * 4 * nfn..(k + 1) * 4 * nfn],
                    &mut eta_rate[k * nn..(k + 1) * nn],
                );
            }
            eta_rate.iter_mut().for_each(|r| *r = -*r);
            Self {
                mesh,
                ops,
                geom,
                sigma,
                bathymetry,
                state,
                du_hu,
                du_hv,
                du_face,
                eta_rate,
                boundaries,
            }
        }

        fn closed() -> Self {
            Self::new(Mesh2D::uniform_rectangle(0.0, LX, 0.0, LY, 4, 3), true)
        }

        /// Open on all four sides, with flow through them.
        fn open() -> Self {
            let mesh = Mesh2D::uniform_rectangle_with_bc(0.0, LX, 0.0, LY, 4, 3, BoundaryTag::Open);
            Self::new(mesh, false)
        }

        fn periodic() -> Self {
            Self::new(Mesh2D::uniform_periodic(0.0, LX, 0.0, LY, 4, 3), false)
        }

        fn transport(&self) -> LayerTransport {
            self.transport_with(&Exterior3D::default())
        }

        fn transport_with(&self, exterior: &Exterior3D) -> LayerTransport {
            let mut transport =
                LayerTransport::new(self.mesh.n_elements, &self.ops, self.sigma.n_levels());
            let barotropic = BarotropicFlux {
                hu: &self.du_hu,
                hv: &self.du_hv,
                face: &self.du_face,
                eta_rate: &self.eta_rate,
                rivers: None,
            };
            transport.compute(
                &self.state,
                Some(barotropic),
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.sigma,
                &self.bathymetry,
                &self.boundaries,
                exterior,
            );
            transport
        }

        fn tracer_rhs(
            &self,
            transport: &LayerTransport,
            tracer: &[f64],
            bc: &dyn TracerBoundaryCondition3D,
        ) -> Vec<f64> {
            let mut rhs = vec![f64::NAN; tracer.len()];
            apply_tracer_transport_3d(
                &mut rhs,
                tracer,
                transport,
                &self.mesh,
                &self.ops,
                &self.geom,
                bc,
                None,
                &self.boundaries,
                VerticalAdvection::default(),
            );
            rhs
        }

        /// `∫ Σ_l φ_l dA` (GLL quadrature) of a field in the 3D layout.
        fn integral(&self, field: &[f64]) -> f64 {
            let (nn, nl) = (self.ops.n_nodes, self.sigma.n_levels());
            (0..self.mesh.n_elements)
                .map(|k| {
                    let sums: Vec<f64> = (0..nn)
                        .map(|i| field[(k * nn + i) * nl..][..nl].iter().sum())
                        .collect();
                    self.geom.integrate_element(k, &sums)
                })
                .sum()
        }
    }

    fn max_abs(values: impl IntoIterator<Item = f64>) -> f64 {
        values.into_iter().fold(0.0, |m, x| m.max(x.abs()))
    }

    /// The layer transports add up to the barotropic transport, at the nodes
    /// and on every face, and walls carry nothing.
    #[test]
    fn layer_transports_add_up_to_the_barotropic_transport() {
        for case in [Case::closed(), Case::open()] {
            layer_transports_add_up(&case);
        }
    }

    fn layer_transports_add_up(case: &Case) {
        let transport = case.transport();
        let (nn, nfn, nl) = (
            case.ops.n_nodes,
            case.ops.n_face_nodes,
            case.sigma.n_levels(),
        );
        let scale = max_abs(case.du_hu.iter().chain(&case.du_hv).copied());
        for idx in 0..case.mesh.n_elements * nn {
            let column = idx * nl..(idx + 1) * nl;
            let sum_u: f64 = transport.hu[column.clone()].iter().sum();
            let sum_v: f64 = transport.hv[column].iter().sum();
            assert!((sum_u - case.du_hu[idx]).abs() < 1e-14 * scale);
            assert!((sum_v - case.du_hv[idx]).abs() < 1e-14 * scale);
        }
        for (slot, &du) in case.du_face.iter().enumerate() {
            let sum: f64 = transport.face[slot * nl..(slot + 1) * nl].iter().sum();
            assert!(
                (sum - du).abs() < 1e-14 * scale,
                "face slot {slot}: Σ F_l = {sum}, DU_avg2 = {du}"
            );
        }
        for k in 0..case.mesh.n_elements {
            for f in 0..4 {
                let exterior = case
                    .boundaries
                    .exterior(&case.mesh, ElementIndex::new(k), f);
                if exterior == FaceExterior::Wall {
                    let slots = (k * 4 + f) * nfn * nl..(k * 4 + f + 1) * nfn * nl;
                    assert_eq!(max_abs(transport.face[slots].iter().copied()), 0.0);
                }
            }
        }
    }

    /// At an open face each layer carries the 2D flux in the interior's
    /// vertical profile: `F_l = Q_l·n + Δσ_l (F_2D − Σ_m Q_m·n)`, so the
    /// shear leaves (and enters) with the flow. Before, every layer took the
    /// uniform share `Δσ_l F_2D`, as at a wall.
    #[test]
    fn open_faces_carry_the_interior_layer_profile() {
        let case = Case::open();
        let transport = case.transport();
        let (nn, nfn, nl) = (
            case.ops.n_nodes,
            case.ops.n_face_nodes,
            case.sigma.n_levels(),
        );
        let d_sigma = case.sigma.d_sigma();
        let scale = max_abs(transport.face.iter().copied());
        let (mut max_err, mut max_shear) = (0.0_f64, 0.0_f64);
        for k in 0..case.mesh.n_elements {
            for f in 0..4 {
                let el = ElementIndex::new(k);
                if !matches!(
                    case.boundaries.exterior(&case.mesh, el, f),
                    FaceExterior::Open(_)
                ) {
                    continue;
                }
                for (fi, &node) in case.ops.face_nodes[f].iter().enumerate() {
                    let (nx, ny) = case.geom.normal(k, f, fi);
                    let column = (k * nn + node) * nl;
                    let q: Vec<f64> = (0..nl)
                        .map(|l| nx * transport.hu[column + l] + ny * transport.hv[column + l])
                        .collect();
                    let q_sum: f64 = q.iter().sum();
                    let slot = (k * 4 + f) * nfn + fi;
                    for l in 0..nl {
                        let expected = q[l] + d_sigma[l] * (case.du_face[slot] - q_sum);
                        max_err = max_err.max((transport.face[slot * nl + l] - expected).abs());
                        max_shear = max_shear.max((q[l] - d_sigma[l] * q_sum).abs());
                    }
                }
            }
        }
        assert!(
            max_shear > 0.05 * scale,
            "test regime: no shear at the open faces"
        );
        assert!(
            max_err < 1e-14 * scale,
            "open-face layer fluxes off the interior profile by {max_err:.3e}"
        );
    }

    /// With `∂η/∂t = −∇·DU_avg2` (the barotropic pass's nodal identity), Ω
    /// integrated from the bed closes at the surface to round-off: nothing is
    /// left for the linear correction.
    #[test]
    fn omega_vanishes_at_the_surface_without_correction() {
        for case in [Case::closed(), Case::periodic(), Case::open()] {
            let transport = case.transport();
            let scale = max_abs(transport.omega.iter().copied());
            assert!(scale > 1e-4, "test flow should drive a non-trivial Ω");
            assert!(
                transport.surface_residual < 1e-12 * scale,
                "surface residual {:.2e} (Ω scale {scale:.2e})",
                transport.surface_residual
            );
        }
    }

    /// Constancy: a uniform tracer changes its inventory exactly as its layer
    /// thickness changes, `∂(H_z C)/∂t = C Δσ_l ∂η/∂t`, so `C` stays put. The
    /// boundary value must not matter (walls carry no volume).
    #[test]
    fn uniform_tracer_follows_the_layer_thickness() {
        let c = 34.7;
        let fixed = FixedTracerBC3D::new(5.0);
        let same = FixedTracerBC3D::new(c);
        let cases: [(Case, [&dyn TracerBoundaryCondition3D; 2]); 2] = [
            (Case::closed(), [&ExtrapolationTracerBC3D, &fixed]),
            // Open: the inflow must be the same water
            (Case::open(), [&ExtrapolationTracerBC3D, &same]),
        ];
        for (case, bcs) in cases {
            uniform_tracer_follows(&case, c, bcs);
        }
    }

    fn uniform_tracer_follows(case: &Case, c: f64, bcs: [&dyn TracerBoundaryCondition3D; 2]) {
        let transport = case.transport();
        let tracer = vec![c; case.state.temp.len()];
        let (nn, nl) = (case.ops.n_nodes, case.sigma.n_levels());
        let scale = c * max_abs(case.eta_rate.iter().copied());
        for bc in bcs {
            let rhs = case.tracer_rhs(&transport, &tracer, bc);
            for idx in 0..case.mesh.n_elements * nn {
                for l in 0..nl {
                    let expected = c * case.sigma.d_sigma()[l] * case.eta_rate[idx];
                    let got = rhs[idx * nl + l];
                    assert!(
                        (got - expected).abs() < 1e-12 * scale,
                        "node {idx}, layer {l}: {got:.6e} vs C·Δσ·∂η/∂t = {expected:.6e}"
                    );
                }
            }
        }
    }

    /// Conservation: the inventory tendency integrates to zero in a closed
    /// basin, whatever the tracer's boundary value (the P0.22 regression: heat
    /// and salt must not cross a coastline that water cannot), and on a
    /// periodic mesh.
    #[test]
    fn tracer_inventory_is_conserved() {
        let bcs: [&dyn TracerBoundaryCondition3D; 3] = [
            &ExtrapolationTracerBC3D,
            &FixedTracerBC3D::new(5.0),
            &UpwindTracerBC3D::new(5.0),
        ];
        for case in [Case::closed(), Case::periodic()] {
            let transport = case.transport();
            let hu_scale = max_abs(transport.hu.iter().copied());
            let scale = case.integral(&vec![hu_scale / 1000.0; case.state.temp.len()])
                * max_abs(case.state.temp.iter().copied());
            for bc in bcs {
                let rhs = case.tracer_rhs(&transport, &case.state.temp, bc);
                let tendency = case.integral(&rhs);
                assert!(
                    tendency.abs() < 1e-12 * scale,
                    "inventory tendency {tendency:.3e} (advective scale {scale:.3e})"
                );
            }
        }
    }

    /// Open faces: the inventory changes by exactly the tracer carried through
    /// them, `d/dt ∫ Σ_l H_z C = −∮ Σ_l F_l C_up` (the interior's value on
    /// outflow, the boundary condition's on inflow): nothing is created or
    /// lost at the boundary.
    #[test]
    fn open_boundaries_change_the_inventory_by_the_boundary_flux() {
        let case = Case::open();
        let transport = case.transport();
        let (nn, nfn, nl) = (
            case.ops.n_nodes,
            case.ops.n_face_nodes,
            case.sigma.n_levels(),
        );
        let tracer = &case.state.temp;
        let bcs: [&dyn TracerBoundaryCondition3D; 2] =
            [&ExtrapolationTracerBC3D, &UpwindTracerBC3D::new(5.0)];
        for bc in bcs {
            let tendency = case.integral(&case.tracer_rhs(&transport, tracer, bc));
            let (mut outflow, mut scale) = (0.0, 0.0);
            for k in 0..case.mesh.n_elements {
                let el = ElementIndex::new(k);
                for f in 0..4 {
                    let FaceExterior::Open(tag) = case.boundaries.exterior(&case.mesh, el, f)
                    else {
                        continue;
                    };
                    for (fi, &node) in case.ops.face_nodes[f].iter().enumerate() {
                        let slot = (k * 4 + f) * nfn + fi;
                        let weight = case.ops.weights_1d[fi] * case.geom.surface_jacobian(k, f, fi);
                        for l in 0..nl {
                            let flux = transport.face[slot * nl + l];
                            let interior = tracer[(k * nn + node) * nl + l];
                            let c = if flux >= 0.0 {
                                interior
                            } else {
                                bc.exterior_value(&TracerBCContext3D {
                                    element: k,
                                    face: f,
                                    level: l,
                                    face_node: fi,
                                    boundary_tag: Some(tag),
                                    interior_value: interior,
                                    normal_velocity: flux,
                                })
                            };
                            outflow += weight * flux * c;
                            scale += (weight * flux * c).abs();
                        }
                    }
                }
            }
            assert!(
                (tendency + outflow).abs() < 1e-12 * scale,
                "inventory tendency {tendency:.6e} vs boundary outflow {outflow:.6e}"
            );
        }
    }

    /// Vertically the tracer is upwinded on Ω: upward flux takes the lower
    /// layer's value, downward the upper layer's. Inventory form: the flux
    /// difference is not divided by `H_z`.
    #[test]
    fn vertical_tracer_flux_is_upwind() {
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 1, 1);
        let ops = DGOperators2D::new(1);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let nl = 3;
        let mut transport = LayerTransport::new(1, &ops, nl);
        let tracer: Vec<f64> = (0..ops.n_nodes).flat_map(|_| [1.0, 2.0, 4.0]).collect();
        for (w, expected) in [(1.0, [-1.0, -1.0, 2.0]), (-1.0, [2.0, 2.0, -4.0])] {
            for column in transport.omega.chunks_exact_mut(nl + 1) {
                column.copy_from_slice(&[0.0, w, w, 0.0]);
            }
            let mut rhs = vec![0.0; tracer.len()];
            apply_tracer_transport_3d(
                &mut rhs,
                &tracer,
                &transport,
                &mesh,
                &ops,
                &geom,
                &ExtrapolationTracerBC3D,
                None,
                &Boundaries3D::default(),
                VerticalAdvection::Upwind,
            );
            for column in rhs.chunks_exact(nl) {
                for (got, want) in column.iter().zip(expected) {
                    assert!((got - want).abs() < 1e-14, "Ω = {w}: {column:?}");
                }
            }
        }
    }

    /// Surface values of `scheme` on `column` with layers `d_sigma` and a
    /// uniform `Ω = w` through the interior surfaces.
    fn surfaces(scheme: VerticalAdvection, column: &[f64], d_sigma: &[f64], w: f64) -> Vec<f64> {
        let nl = column.len();
        let mut omega = vec![w; nl + 1];
        (omega[0], omega[nl]) = (0.0, 0.0);
        let (mut slope, mut surface) = (vec![0.0; nl], vec![f64::NAN; nl + 1]);
        scheme.surface_values(column, &omega, d_sigma, &mut slope, &mut surface);
        surface
    }

    /// Layers stretched by up to 2.5× between neighbours, and a linear
    /// profile `C = 3 + 2z` sampled as layer means (their centre values).
    fn stretched_linear() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let d_sigma = vec![0.05, 0.125, 0.2, 0.25, 0.375];
        let mut faces = vec![-1.0];
        for ds in &d_sigma {
            faces.push(faces.last().unwrap() + ds);
        }
        let column = (0..5).map(|l| 3.0 + (faces[l] + faces[l + 1])).collect();
        (column, d_sigma, faces)
    }

    /// The Akima surface values: exact for a linear profile (on stretched
    /// levels too), the plain average next to an extremum (zero slope
    /// there), and independent of the sign of Ω (centred).
    #[test]
    fn vertical_akima_values_are_centred_and_exact_for_linear_profiles() {
        let akima = VerticalAdvection::Akima;
        let linear = [1.0, 3.0, 5.0, 7.0, 9.0];
        for w in [1.0, -1.0] {
            let surface = surfaces(akima, &linear, &[1.0; 5], w);
            for l in 1..linear.len() {
                assert!((surface[l] - (2.0 * l as f64)).abs() < 1e-14, "{surface:?}");
            }
        }
        let (column, d_sigma, faces) = stretched_linear();
        for w in [1.0, -1.0] {
            let surface = surfaces(akima, &column, &d_sigma, w);
            for l in 1..5 {
                let want = 3.0 + 2.0 * faces[l];
                assert!((surface[l] - want).abs() < 1e-13, "{surface:?}");
            }
        }
        // Harmonic-mean slopes d = [1, 4/3, 2] (one-sided at the ends):
        // C_{1/2} = ½(1 + 2 − (4/3 − 1)/3), C_{3/2} = ½(2 + 4 − (2 − 4/3)/3),
        // ROMS's index-space formula on uniform levels
        let surface = surfaces(akima, &[1.0, 2.0, 4.0], &[1.0; 3], 1.0);
        assert!((surface[1] - 13.0 / 9.0).abs() < 1e-14, "{surface:?}");
        assert!((surface[2] - 26.0 / 9.0).abs() < 1e-14, "{surface:?}");
        // The same on uniform levels of any thickness
        let thin = surfaces(akima, &[1.0, 2.0, 4.0], &[0.1; 3], 1.0);
        assert!((thin[1] - 13.0 / 9.0).abs() < 1e-13, "{thin:?}");
        // Level 1 is a maximum: its slope is zero
        let peaked = [1.0, 3.0, 2.0, 1.0];
        let surface = surfaces(akima, &peaked, &[1.0; 4], -1.0);
        let d = [2.0, 0.0, -1.0, -1.0];
        for l in 1..4 {
            let want = 0.5 * (peaked[l - 1] + peaked[l] - (d[l] - d[l - 1]) / 3.0);
            assert!((surface[l] - want).abs() < 1e-14, "{surface:?}");
        }
        // One level: nothing to reconstruct
        assert_eq!(surfaces(akima, &[4.0], &[1.0], 1.0), vec![0.0, 0.0]);
    }

    /// The TVD surface values: third-order upwind `(5C_up + 2C_down −
    /// C_far)/6` where the limiter allows it, linear out of an end layer,
    /// exact for a linear profile (on stretched levels too), first order at
    /// an extremum, and never outside the two neighbouring layers' values.
    #[test]
    fn vertical_tvd_values_are_third_order_upwind_and_bounded() {
        let tvd = VerticalAdvection::Tvd;
        // Smooth and monotone: 1, 2, 4, 7 (unlimited increments)
        let column = [1.0, 2.0, 4.0, 7.0];
        let up = surfaces(tvd, &column, &[1.0; 4], 1.0);
        assert_eq!(up[1], 1.5, "linear out of the bed layer");
        assert!((up[2] - (5.0 * 2.0 + 2.0 * 4.0 - 1.0) / 6.0).abs() < 1e-14);
        assert!((up[3] - (5.0 * 4.0 + 2.0 * 7.0 - 2.0) / 6.0).abs() < 1e-14);
        let down = surfaces(tvd, &column, &[1.0; 4], -1.0);
        assert!((down[1] - (5.0 * 2.0 + 2.0 * 1.0 - 4.0) / 6.0).abs() < 1e-14);
        assert!((down[2] - (5.0 * 4.0 + 2.0 * 2.0 - 7.0) / 6.0).abs() < 1e-14);
        assert_eq!(down[3], 5.5, "linear out of the surface layer");

        let (column, d_sigma, faces) = stretched_linear();
        for w in [1.0, -1.0] {
            let surface = surfaces(tvd, &column, &d_sigma, w);
            for l in 1..5 {
                let want = 3.0 + 2.0 * faces[l];
                assert!((surface[l] - want).abs() < 1e-13, "Ω = {w}: {surface:?}");
            }
        }

        // A step and a spike: no value outside its neighbours
        for column in [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ] {
            for w in [1.0, -1.0] {
                let surface = surfaces(tvd, &column, &[1.0; 6], w);
                for l in 1..6 {
                    let (lo, hi) = (column[l - 1].min(column[l]), column[l - 1].max(column[l]));
                    assert!(
                        (lo..=hi).contains(&surface[l]),
                        "Ω = {w}: {surface:?} for {column:?}"
                    );
                }
            }
        }
        // Upwind of the spike's peak (an extremum): first order
        let surface = surfaces(tvd, &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[1.0; 6], 1.0);
        assert_eq!(surface[3], 1.0);
    }

    /// The limited Akima surface values: Akima's where they lie within the
    /// TVD bounds (a smooth monotone profile, a linear one on stretched
    /// levels), never outside the two neighbouring layers' values, and the
    /// upwind layer's value next to an extremum. At the foot of a front
    /// Akima's value is clipped to the upwind gradient.
    #[test]
    fn vertical_limited_akima_values_are_akimas_within_the_tvd_bounds() {
        let (akima, limited) = (VerticalAdvection::Akima, VerticalAdvection::LimitedAkima);
        for w in [1.0, -1.0] {
            let column = [1.0, 2.0, 4.0, 7.0, 11.0];
            assert_eq!(
                surfaces(limited, &column, &[1.0; 5], w),
                surfaces(akima, &column, &[1.0; 5], w),
                "Ω = {w}"
            );
            let (column, d_sigma, faces) = stretched_linear();
            let surface = surfaces(limited, &column, &d_sigma, w);
            for l in 1..5 {
                let want = 3.0 + 2.0 * faces[l];
                assert!((surface[l] - want).abs() < 1e-13, "Ω = {w}: {surface:?}");
            }
        }

        // The foot of a front: harmonic-mean slopes 0, 0.18, 0 at layers
        // 1–3, so Akima's value at surface 3 is ½(0.1 + 1 + 0.18/3) = 0.58,
        // 0.48 above the upwind layer's 0.1 when the gradient behind it is
        // 0.1: clipped to 0.2
        let front = [0.0, 0.0, 0.1, 1.0, 1.0, 1.0];
        let unlimited = surfaces(akima, &front, &[1.0; 6], 1.0);
        assert!((unlimited[3] - 0.58).abs() < 1e-14, "{unlimited:?}");
        let clipped = surfaces(limited, &front, &[1.0; 6], 1.0);
        assert!((clipped[3] - 0.2).abs() < 1e-14, "{clipped:?}");
        for column in [front, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] {
            for w in [1.0, -1.0] {
                let surface = surfaces(limited, &column, &[1.0; 6], w);
                for l in 1..6 {
                    let (lo, hi) = (column[l - 1].min(column[l]), column[l - 1].max(column[l]));
                    assert!(
                        (lo..=hi).contains(&surface[l]),
                        "Ω = {w}: {surface:?} for {column:?}"
                    );
                }
            }
        }
        // Upwind of the spike's peak (an extremum): first order
        let surface = surfaces(limited, &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[1.0; 6], 1.0);
        assert_eq!(surface[3], 1.0);
    }

    /// `column` advected up a column of `d_sigma` layers by `Ω = w` for
    /// `steps` SSP-RK3 steps at the Courant number `courant` (on the thinnest
    /// layer), `scheme` at the interior surfaces, the same water flowing in
    /// at the bed and out at the surface.
    fn advect_column(
        scheme: VerticalAdvection,
        mut column: Vec<f64>,
        d_sigma: &[f64],
        courant: f64,
        steps: usize,
    ) -> Vec<f64> {
        let nl = column.len();
        let w = 1.0;
        let dt = courant * d_sigma.iter().copied().fold(f64::MAX, f64::min) / w;
        let omega = vec![w; nl + 1];
        let (mut slope, mut surface) = (vec![0.0; nl], vec![0.0; nl + 1]);
        let mut rate = |c: &[f64]| -> Vec<f64> {
            scheme.surface_values(c, &omega, d_sigma, &mut slope, &mut surface);
            (surface[0], surface[nl]) = (c[0], c[nl - 1]);
            (0..nl)
                .map(|l| -w * (surface[l + 1] - surface[l]) / d_sigma[l])
                .collect()
        };
        for _ in 0..steps {
            let q0 = column.clone();
            let k: Vec<f64> = rate(&column);
            let q1: Vec<f64> = q0.iter().zip(&k).map(|(q, k)| q + dt * k).collect();
            let k: Vec<f64> = rate(&q1);
            let q2: Vec<f64> = (0..nl)
                .map(|l| 0.75 * q0[l] + 0.25 * (q1[l] + dt * k[l]))
                .collect();
            let k: Vec<f64> = rate(&q2);
            column = (0..nl)
                .map(|l| q0[l] / 3.0 + 2.0 / 3.0 * (q2[l] + dt * k[l]))
                .collect();
        }
        column
    }

    /// A front advected 16 layers up a column: TVD and limited Akima stay
    /// within [0, 1] and spread it over 6 layers, upwind over 18; Akima
    /// overshoots by 16 % (and spreads it over 19). A halocline-like tanh
    /// profile: the error is 800× smaller than upwind's with Akima and with
    /// limited Akima (whose limiter does not act on it: the same error to
    /// the last digit), 60× with TVD (whose limiter acts in the profile's
    /// tails), and 450× and 40× on stretched layers.
    #[test]
    fn vertical_schemes_on_a_front_and_a_smooth_profile() {
        use VerticalAdvection::{Akima, LimitedAkima, Tvd, Upwind};
        let nl = 100;
        let uniform = vec![1.0 / nl as f64; nl];
        let front: Vec<f64> = (0..nl).map(|l| if l < 30 { 1.0 } else { 0.0 }).collect();
        // 40 layers at 0.4 per step: the front moves 16 layers
        let run = |scheme| advect_column(scheme, front.clone(), &uniform, 0.4, 40);
        let (upwind, akima, tvd) = (run(Upwind), run(Akima), run(Tvd));
        let limited = run(LimitedAkima);
        let range = |c: &[f64]| {
            c.iter()
                .fold((f64::MAX, f64::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)))
        };
        let smeared = |c: &[f64]| c.iter().filter(|&&x| x > 0.01 && x < 0.99).count();
        for (name, c) in [("TVD", &tvd), ("limited Akima", &limited)] {
            let (lo, hi) = range(c);
            assert!(
                lo >= -1e-12 && hi <= 1.0 + 1e-12,
                "{name} left [0, 1]: [{lo}, {hi}]"
            );
            assert!(
                smeared(c) * 2 < smeared(&upwind),
                "{name}: front over {} layers, upwind {}",
                smeared(c),
                smeared(&upwind)
            );
        }
        let (lo, hi) = range(&upwind);
        assert!(
            lo >= -1e-12 && hi <= 1.0 + 1e-12,
            "upwind left [0, 1]: [{lo}, {hi}]"
        );
        let (lo, hi) = range(&akima);
        assert!(
            hi > 1.1,
            "test regime: Akima should overshoot: [{lo}, {hi}]"
        );

        // A smooth profile on uniform and stretched layers (2.5 % thicker
        // from each layer to the next), advected 16 thin layers
        let stretched: Vec<f64> = {
            let raw: Vec<f64> = (0..nl).map(|l| 1.05_f64.powf(l as f64 / 2.0)).collect();
            let total: f64 = raw.iter().sum();
            raw.iter().map(|r| r / total).collect()
        };
        for d_sigma in [&uniform, &stretched] {
            let mut faces = vec![0.0];
            for ds in d_sigma {
                faces.push(faces.last().unwrap() + ds);
            }
            // Layer means of a halocline-like ½(1 + tanh((σ − 0.4)/0.08)), and
            // of it shifted by the travel w·t
            let mean = |shift: f64| -> Vec<f64> {
                let primitive = |x: f64| {
                    let y = (x - shift - 0.4) / 0.08;
                    0.5 * (x + 0.08 * y.cosh().ln())
                };
                (0..nl)
                    .map(|l| (primitive(faces[l + 1]) - primitive(faces[l])) / d_sigma[l])
                    .collect()
            };
            let steps = 40;
            let thinnest = d_sigma.iter().copied().fold(f64::MAX, f64::min);
            let travel = 0.4 * thinnest * steps as f64;
            let exact = mean(travel);
            let error = |scheme| {
                let c = advect_column(scheme, mean(0.0), d_sigma, 0.4, steps);
                // Away from the inflow at the bed and the outflow at the
                // surface (both first order here)
                max_abs((nl / 4..nl - 4).map(|l| c[l] - exact[l]))
            };
            let (upwind, akima, tvd) = (error(Upwind), error(Akima), error(Tvd));
            let limited = error(LimitedAkima);
            assert!(
                akima * 200.0 < upwind && tvd * 20.0 < upwind && limited < 1.01 * akima,
                "smooth profile: upwind {upwind:.3e}, Akima {akima:.3e}, TVD {tvd:.3e},                  limited Akima {limited:.3e}"
            );
        }
    }

    impl Case {
        /// Momentum inventory tendencies `(∂(H_z u)/∂t, ∂(H_z v)/∂t)` of the
        /// advection of `(u, v)`.
        fn momentum_rhs(
            &self,
            transport: &LayerTransport,
            u: &[f64],
            v: &[f64],
        ) -> (Vec<f64>, Vec<f64>) {
            let mut rhs_u = vec![0.0; u.len()];
            let mut rhs_v = vec![0.0; v.len()];
            apply_momentum_transport_3d(
                &mut rhs_u,
                &mut rhs_v,
                u,
                v,
                transport,
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.boundaries,
                None,
                VerticalAdvection::Centred,
            );
            (rhs_u, rhs_v)
        }
    }

    /// Momentum constancy: a velocity uniform in space changes its inventory
    /// exactly as the layer thickness changes, `∂(H_z u)/∂t = u Δσ_l ∂η/∂t`,
    /// so it stays uniform, through walls, open faces and periodic ones. (The
    /// old velocity-form advection, `∇·(u u)` with its own Rusanov flux and
    /// `Ω` only in the vertical term, was not built on the layer transports.)
    #[test]
    fn uniform_velocity_follows_the_layer_thickness() {
        let (u0, v0) = (0.7, -0.3);
        for case in [Case::closed(), Case::open(), Case::periodic()] {
            let transport = case.transport();
            let n = case.state.u.len();
            let (rhs_u, rhs_v) = case.momentum_rhs(&transport, &vec![u0; n], &vec![v0; n]);
            let (nn, nl) = (case.ops.n_nodes, case.sigma.n_levels());
            let scale = max_abs(case.eta_rate.iter().copied());
            for idx in 0..case.mesh.n_elements * nn {
                for l in 0..nl {
                    let thickness_rate = case.sigma.d_sigma()[l] * case.eta_rate[idx];
                    for (c, rhs) in [(u0, &rhs_u), (v0, &rhs_v)] {
                        let got = rhs[idx * nl + l];
                        assert!(
                            (got - c * thickness_rate).abs() < 1e-12 * scale,
                            "node {idx}, layer {l}: {got:.6e} vs u·Δσ·∂η/∂t = {:.6e}",
                            c * thickness_rate
                        );
                    }
                }
            }
        }
    }

    /// Conservation: the momentum advection moves momentum between layers and
    /// elements but, summed over the layers, creates none in a closed basin
    /// (walls carry no volume, so no momentum) or on a periodic mesh.
    #[test]
    fn momentum_advection_conserves_the_column_momentum() {
        for case in [Case::closed(), Case::periodic()] {
            let transport = case.transport();
            let (rhs_u, rhs_v) = case.momentum_rhs(&transport, &case.state.u, &case.state.v);
            let hu_scale = max_abs(transport.hu.iter().copied());
            let u_scale = max_abs(case.state.u.iter().chain(&case.state.v).copied());
            let scale = case.integral(&vec![hu_scale * u_scale / 1000.0; rhs_u.len()]);
            assert!(
                max_abs(rhs_u.iter().copied()) > 1e-3 * hu_scale * u_scale / 1000.0,
                "test regime: no advection"
            );
            for rhs in [rhs_u, rhs_v] {
                let tendency = case.integral(&rhs);
                assert!(
                    tendency.abs() < 1e-12 * scale,
                    "momentum tendency {tendency:.3e} (advective scale {scale:.3e})"
                );
            }
        }
    }

    /// Open faces: the column momentum changes by exactly what the layer
    /// fluxes carry out, `−∮ Σ_l F_l u_l`, with the interior's velocity both
    /// ways (zero gradient).
    #[test]
    fn open_boundaries_change_the_momentum_by_the_boundary_flux() {
        let case = Case::open();
        let transport = case.transport();
        let (nn, nfn, nl) = (
            case.ops.n_nodes,
            case.ops.n_face_nodes,
            case.sigma.n_levels(),
        );
        let (rhs_u, rhs_v) = case.momentum_rhs(&transport, &case.state.u, &case.state.v);
        for (rhs, field) in [(&rhs_u, &case.state.u), (&rhs_v, &case.state.v)] {
            let tendency = case.integral(rhs);
            let (mut outflow, mut scale) = (0.0, 0.0);
            for k in 0..case.mesh.n_elements {
                for f in 0..4 {
                    let exterior = case
                        .boundaries
                        .exterior(&case.mesh, ElementIndex::new(k), f);
                    if !matches!(exterior, FaceExterior::Open(_)) {
                        continue;
                    }
                    for (fi, &node) in case.ops.face_nodes[f].iter().enumerate() {
                        let slot = (k * 4 + f) * nfn + fi;
                        let weight = case.ops.weights_1d[fi] * case.geom.surface_jacobian(k, f, fi);
                        for l in 0..nl {
                            let carried = weight
                                * transport.face[slot * nl + l]
                                * field[(k * nn + node) * nl + l];
                            outflow += carried;
                            scale += carried.abs();
                        }
                    }
                }
            }
            assert!(
                (tendency + outflow).abs() < 1e-12 * scale,
                "momentum tendency {tendency:.6e} vs boundary outflow {outflow:.6e}"
            );
        }
    }

    /// Vertically the velocity is centred at the σ-surfaces, in inventory
    /// form: `−(Ω_{l+1/2} u_{l+1/2} − Ω_{l−1/2} u_{l−1/2})`, not divided by
    /// `H_z`. Linear `Ω(σ)` and `u(σ)` make the centred values exact.
    #[test]
    fn vertical_momentum_flux_is_centred() {
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 1, 1);
        let ops = DGOperators2D::new(1);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let sigma = SigmaGrid::uniform(4);
        let nl = sigma.n_levels();
        let omega = |s: f64| 0.03 * (s + 1.0) * s;
        let u = |s: f64| 0.4 - 0.5 * s;
        let mut transport = LayerTransport::new(1, &ops, nl);
        for column in transport.omega.chunks_exact_mut(nl + 1) {
            for (w, &s) in column.iter_mut().zip(sigma.sigma_w()) {
                *w = omega(s);
            }
        }
        let column: Vec<f64> = sigma.sigma_rho().iter().map(|&s| u(s)).collect();
        let field: Vec<f64> = (0..ops.n_nodes).flat_map(|_| column.clone()).collect();
        let (mut rhs_u, mut rhs_v) = (vec![0.0; field.len()], vec![0.0; field.len()]);
        apply_momentum_transport_3d(
            &mut rhs_u,
            &mut rhs_v,
            &field,
            &field,
            &transport,
            &mesh,
            &ops,
            &geom,
            &Boundaries3D::default(),
            None,
            VerticalAdvection::Centred,
        );
        let sw = sigma.sigma_w();
        // Ω vanishes at the bed and the surface
        let flux = |f: usize| omega(sw[f]) * u(sw[f]);
        for rhs in [&rhs_u, &rhs_v] {
            for (i, column) in rhs.chunks_exact(nl).enumerate() {
                for (l, got) in column.iter().enumerate() {
                    let expected = -(flux(l + 1) - flux(l));
                    assert!(
                        (got - expected).abs() < 1e-15,
                        "node {i}, layer {l}: {got} vs {expected}"
                    );
                }
            }
        }
    }

    /// `None`: the state's own layer transports, `H_z u` uncorrected, with `Ω`
    /// closed at the surface by the free-surface rate they imply.
    #[test]
    fn own_layer_transports_close_omega() {
        let case = Case::closed();
        let mut transport =
            LayerTransport::new(case.mesh.n_elements, &case.ops, case.sigma.n_levels());
        transport.compute(
            &case.state,
            None,
            &case.mesh,
            &case.ops,
            &case.geom,
            &case.sigma,
            &case.bathymetry,
            &case.boundaries,
            &Exterior3D::default(),
        );
        let (nn, nl) = (case.ops.n_nodes, case.sigma.n_levels());
        for idx in 0..case.mesh.n_elements * nn {
            let depth = case.state.eta.data[idx] - case.bathymetry.data[idx];
            for l in 0..nl {
                let expected = depth * case.sigma.d_sigma()[l] * case.state.u[idx * nl + l];
                assert!((transport.hu[idx * nl + l] - expected).abs() < 1e-14 * expected.abs());
            }
        }
        let scale = max_abs(transport.omega.iter().copied());
        assert!(scale > 1e-4, "test flow should drive a non-trivial Ω");
        assert!(
            transport.surface_residual < 1e-12 * scale,
            "surface residual {:.2e}",
            transport.surface_residual
        );
    }

    /// Ω of the own transports is stored at the w-points and satisfies layer
    /// continuity (the regression of the old `compute_vertical_velocity`,
    /// whose layer-centre storage mixed neighbouring layers). A zero-mean
    /// shear `u_l = U_l sin(2πx)` over a flat bed has no column divergence,
    /// so Ω closes at the surface by itself and each layer's increment
    /// `Ω_{l+1/2} − Ω_{l−1/2} = −Δσ_l ∇·(D u_l)` scales with `U_l`.
    #[test]
    fn own_omega_of_a_zero_mean_shear_has_layer_continuity() {
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 1);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let sigma = SigmaGrid::uniform(4);
        let (nn, nl) = (ops.n_nodes, sigma.n_levels());
        let bathymetry = Bathymetry2D::constant(mesh.n_elements, nn, -10.0);
        let shear: Vec<f64> = sigma.sigma_rho().iter().map(|&s| s + 0.5).collect();
        let mut state = Solution3D::new(mesh.n_elements, nn, nl);
        for k in 0..mesh.n_elements {
            let el = ElementIndex::new(k);
            for i in 0..nn {
                let [x, _] = mesh.reference_to_physical(el, ops.nodes_r[i], ops.nodes_s[i]);
                for (u, &shear_l) in state.u_column_mut(el, i).iter_mut().zip(&shear) {
                    *u = shear_l * (2.0 * PI * x).sin();
                }
            }
        }
        let mut transport = LayerTransport::new(mesh.n_elements, &ops, nl);
        transport.compute(
            &state,
            None,
            &mesh,
            &ops,
            &geom,
            &sigma,
            &bathymetry,
            &Boundaries3D::default(),
            &Exterior3D::default(),
        );
        let scale = max_abs(transport.omega.iter().copied());
        assert!(scale > 1e-3, "test flow should drive a non-trivial Ω");
        let tol = 1e-12 * scale;
        let d_sigma = sigma.d_sigma();
        for col in transport.omega.chunks_exact(nl + 1) {
            assert!(col[0].abs() < tol && col[nl].abs() < tol, "{col:?}");
            let rate = (col[1] - col[0]) / (shear[0] * d_sigma[0]);
            for l in 0..nl {
                let increment = col[l + 1] - col[l];
                let expected = rate * shear[l] * d_sigma[l];
                assert!(
                    (increment - expected).abs() < tol,
                    "layer {l}: Ω increment {increment}, continuity requires {expected}"
                );
            }
        }
    }

    /// A nesting parent at every node of the open case: a sheared velocity
    /// unlike the interior's, and `temp`, as `[u, v, temp]` per node and
    /// layer (slot = node).
    struct Parent {
        tags: Vec<BoundaryTag>,
        slot_of_node: Vec<u32>,
        fields: [Vec<f64>; 3],
        n_levels: usize,
    }

    impl Parent {
        fn new(case: &Case, temp: impl Fn(f64) -> f64) -> Self {
            let nl = case.sigma.n_levels();
            let n = case.state.u.len() / nl;
            let mut fields: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(n * nl));
            for _ in 0..n {
                for &s in case.sigma.sigma_rho() {
                    fields[0].push(0.3 - 0.4 * s);
                    fields[1].push(0.2 * s);
                    fields[2].push(temp(s));
                }
            }
            Self {
                tags: vec![BoundaryTag::Open],
                slot_of_node: (0..n as u32).collect(),
                fields,
                n_levels: nl,
            }
        }

        fn field(&self, i: usize) -> ExteriorField<'_> {
            ExteriorField {
                tags: &self.tags,
                slot_of_node: &self.slot_of_node,
                n_levels: self.n_levels,
                values: &self.fields[i],
            }
        }

        fn exterior(&self) -> Exterior3D<'_> {
            Exterior3D {
                velocity: Some([self.field(0), self.field(1)]),
                temp: Some(self.field(2)),
                salt: None,
            }
        }
    }

    /// With a nesting parent, an open face's layer fluxes are the central
    /// average of the interior's and the parent's `H_z u`, corrected to the
    /// 2D flux, which they still add up to.
    #[test]
    fn open_faces_average_the_parent_profile() {
        let case = Case::open();
        let parent = Parent::new(&case, |s| 5.0 + s);
        let transport = case.transport_with(&parent.exterior());
        let (nn, nfn, nl) = (
            case.ops.n_nodes,
            case.ops.n_face_nodes,
            case.sigma.n_levels(),
        );
        let d_sigma = case.sigma.d_sigma();
        let scale = max_abs(transport.face.iter().copied());
        let mut open_faces = 0;
        for k in 0..case.mesh.n_elements {
            for f in 0..4 {
                let el = ElementIndex::new(k);
                if case.boundaries.exterior(&case.mesh, el, f)
                    != FaceExterior::Open(BoundaryTag::Open)
                {
                    continue;
                }
                open_faces += 1;
                for (fi, &node) in case.ops.face_nodes[f].iter().enumerate() {
                    let (nx, ny) = case.geom.normal(k, f, fi);
                    let flat = k * nn + node;
                    let depth = case.state.eta.data[flat] - case.bathymetry.data[flat];
                    let central: Vec<f64> = (0..nl)
                        .map(|l| {
                            let (i, p) = (flat * nl + l, flat * nl + l);
                            let q_in = nx * transport.hu[i] + ny * transport.hv[i];
                            let q_ex = depth
                                * d_sigma[l]
                                * (nx * parent.fields[0][p] + ny * parent.fields[1][p]);
                            0.5 * (q_in + q_ex)
                        })
                        .collect();
                    let slot = (k * 4 + f) * nfn + fi;
                    let correction = case.du_face[slot] - central.iter().sum::<f64>();
                    let fluxes = &transport.face[slot * nl..(slot + 1) * nl];
                    for l in 0..nl {
                        let expected = central[l] + d_sigma[l] * correction;
                        assert!((fluxes[l] - expected).abs() < 1e-14 * scale);
                    }
                    let sum: f64 = fluxes.iter().sum();
                    assert!((sum - case.du_face[slot]).abs() < 1e-14 * scale);
                }
            }
        }
        assert!(open_faces > 0);
    }

    /// Water flowing in through a nested open face brings the parent's
    /// tracer and velocity (and the tracer boundary condition is not
    /// consulted): the inventories change by exactly `−∮ Σ_l F_l φ↑`, with
    /// `φ↑` the parent's on inflow and the interior's on outflow.
    #[test]
    fn nested_inflow_brings_the_parent_values() {
        let case = Case::open();
        let parent = Parent::new(&case, |s| 5.0 + s);
        let exterior = parent.exterior();
        let transport = case.transport_with(&exterior);
        let (nn, nfn, nl) = (
            case.ops.n_nodes,
            case.ops.n_face_nodes,
            case.sigma.n_levels(),
        );
        // Tracer: a boundary condition that would give nonsense
        let mut rhs_t = vec![0.0; case.state.temp.len()];
        apply_tracer_transport_3d(
            &mut rhs_t,
            &case.state.temp,
            &transport,
            &case.mesh,
            &case.ops,
            &case.geom,
            &FixedTracerBC3D::new(-1e6),
            exterior.temp,
            &case.boundaries,
            VerticalAdvection::default(),
        );
        let (mut rhs_u, mut rhs_v) = (vec![0.0; rhs_t.len()], vec![0.0; rhs_t.len()]);
        apply_momentum_transport_3d(
            &mut rhs_u,
            &mut rhs_v,
            &case.state.u,
            &case.state.v,
            &transport,
            &case.mesh,
            &case.ops,
            &case.geom,
            &case.boundaries,
            exterior.velocity,
            VerticalAdvection::Centred,
        );
        let fields = [
            (&rhs_t, &case.state.temp, 2),
            (&rhs_u, &case.state.u, 0),
            (&rhs_v, &case.state.v, 1),
        ];
        for (rhs, field, p) in fields {
            let tendency = case.integral(rhs);
            let (mut outflow, mut scale, mut inflows) = (0.0, 0.0, 0);
            for k in 0..case.mesh.n_elements {
                for f in 0..4 {
                    let el = ElementIndex::new(k);
                    if !matches!(
                        case.boundaries.exterior(&case.mesh, el, f),
                        FaceExterior::Open(_)
                    ) {
                        continue;
                    }
                    for (fi, &node) in case.ops.face_nodes[f].iter().enumerate() {
                        let slot = (k * 4 + f) * nfn + fi;
                        let weight = case.ops.weights_1d[fi] * case.geom.surface_jacobian(k, f, fi);
                        for l in 0..nl {
                            let flux = transport.face[slot * nl + l];
                            let idx = (k * nn + node) * nl + l;
                            let value = if flux >= 0.0 {
                                field[idx]
                            } else {
                                inflows += 1;
                                parent.fields[p][idx]
                            };
                            outflow += weight * flux * value;
                            scale += (weight * flux * value).abs();
                        }
                    }
                }
            }
            assert!(inflows > 0, "test regime: no inflow");
            assert!(
                (tendency + outflow).abs() < 1e-12 * scale,
                "tendency {tendency:.6e} vs boundary outflow {outflow:.6e}"
            );
        }
    }

    /// Constancy with a nesting parent: a uniform tracer, and a parent of the
    /// same tracer, stays uniform however the parent shears the boundary
    /// fluxes (the layers still carry exactly the water the free surface
    /// moved).
    #[test]
    fn uniform_tracer_stays_uniform_with_a_parent() {
        let case = Case::open();
        let c = 34.7;
        let parent = Parent::new(&case, |_| c);
        let exterior = parent.exterior();
        let transport = case.transport_with(&exterior);
        let nl = case.sigma.n_levels();
        let tracer = vec![c; case.state.temp.len()];
        let mut rhs = vec![0.0; tracer.len()];
        apply_tracer_transport_3d(
            &mut rhs,
            &tracer,
            &transport,
            &case.mesh,
            &case.ops,
            &case.geom,
            &ExtrapolationTracerBC3D,
            exterior.temp,
            &case.boundaries,
            VerticalAdvection::default(),
        );
        let scale = c * max_abs(case.eta_rate.iter().copied());
        for (idx, column) in rhs.chunks_exact(nl).enumerate() {
            for (l, got) in column.iter().enumerate() {
                let expected = c * case.sigma.d_sigma()[l] * case.eta_rate[idx];
                assert!(
                    (got - expected).abs() < 1e-12 * scale,
                    "{got} vs {expected}"
                );
            }
        }
    }

    /// Inventory ↔ concentration: nodal in unmarked elements; in marked
    /// elements the element's inventory over its volume per level, which
    /// converts back to the same element inventory; a dry node, or a dry
    /// marked element, keeps its last concentration.
    #[test]
    fn inventory_conversion_keeps_element_inventories() {
        let case = Case::closed();
        let (nn, nl) = (case.ops.n_nodes, case.sigma.n_levels());
        let n_el = case.mesh.n_elements;
        // Element 0 dry at one node, element 1 dry everywhere, element 2 marked
        let mut eta = case.state.eta.data.clone();
        eta[0] = case.bathymetry.data[0];
        eta[nn..2 * nn].copy_from_slice(&case.bathymetry.data[nn..2 * nn]);
        let mut marked = vec![false; n_el];
        marked[1] = true;
        marked[2] = true;
        let concentration = case.state.temp.clone();
        let mut inventory = concentration.clone();
        to_inventory(&mut inventory, &eta, case.sigma.d_sigma(), &case.bathymetry);
        let last = vec![-1.0; concentration.len()];
        let mut out = last.clone();
        from_inventory(
            &inventory,
            &eta,
            case.sigma.d_sigma(),
            &case.bathymetry,
            &case.geom,
            &marked,
            &mut out,
        );
        for k in 0..n_el {
            for i in 0..nn {
                let idx = k * nn + i;
                let dry = eta[idx] - case.bathymetry.data[idx] < DRY_DEPTH;
                for l in 0..nl {
                    let (c, got) = (concentration[idx * nl + l], out[idx * nl + l]);
                    match (k, dry) {
                        // Dry: the last value; the other nodes nodal
                        (0, true) | (1, _) => assert_eq!(got, -1.0, "element {k} node {i}"),
                        (2, _) => {}
                        _ => assert!((got - c).abs() < 1e-12 * c.abs(), "{got} vs {c}"),
                    }
                }
            }
        }
        // Element 2: one value per level, and its inventory per level kept
        let mut back = out.clone();
        to_inventory(&mut back, &eta, case.sigma.d_sigma(), &case.bathymetry);
        for l in 0..nl {
            let level = |field: &[f64]| -> f64 {
                (0..nn)
                    .map(|i| case.geom.node_mass(2, i) * field[(2 * nn + i) * nl + l])
                    .sum()
            };
            let value = out[(2 * nn) * nl + l];
            for i in 0..nn {
                assert_eq!(out[(2 * nn + i) * nl + l], value);
            }
            let (before, after) = (level(&inventory), level(&back));
            assert!(
                (before - after).abs() < 1e-13 * before.abs(),
                "level {l}: {before} vs {after}"
            );
        }
    }

    /// The w-cells' σ-thicknesses: half the end layers, the means between,
    /// summing to one on stretched levels.
    #[test]
    fn w_cells_span_the_column() {
        let sigma = SigmaGrid::new(
            6,
            crate::vertical::SongHaidvogelStretching::new(5.0, 0.4, 10.0),
        );
        let d = sigma.d_sigma();
        let mut w = vec![0.0; 7];
        w_cell_thicknesses(d, &mut w);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        assert_eq!(w[0], 0.5 * d[0]);
        assert_eq!(w[3], 0.5 * (d[2] + d[3]));
        assert_eq!(w[6], 0.5 * d[5]);
    }

    /// The w-cells of the layer transports ([`LayerTransport::stagger_from`])
    /// carry the layers' water: their fluxes add up to the layers', `Ω`
    /// vanishes at the bed and the surface, a uniform field at the w-points
    /// changes its inventory exactly as its w-cell's thickness,
    /// `C Δσ_w ∂η/∂t` (constancy), and a varying one keeps its inventory in
    /// a closed basin and on a periodic mesh (conservation).
    #[test]
    fn w_cells_keep_constancy_and_conservation() {
        for (case, closed) in [
            (Case::closed(), true),
            (Case::open(), false),
            (Case::periodic(), true),
        ] {
            let layers = case.transport();
            let (ne, nn, nl) = (
                case.mesh.n_elements,
                case.ops.n_nodes,
                case.sigma.n_levels(),
            );
            let nw = nl + 1;
            let mut w_cells = LayerTransport::new(ne, &case.ops, nw);
            w_cells.stagger_from(&layers);
            for (cells, layer) in [
                (&w_cells.hu, &layers.hu),
                (&w_cells.hv, &layers.hv),
                (&w_cells.face, &layers.face),
            ] {
                for (w, l) in cells.chunks_exact(nw).zip(layer.chunks_exact(nl)) {
                    let (sum_w, sum_l) = (w.iter().sum::<f64>(), l.iter().sum::<f64>());
                    assert!((sum_w - sum_l).abs() <= 1e-14 * sum_l.abs().max(1.0));
                }
            }
            for omega in w_cells.omega.chunks_exact(nw + 1) {
                assert_eq!((omega[0], omega[nw]), (0.0, 0.0));
            }

            let c = 0.37;
            let n = ne * nn * nw;
            let rhs = case.tracer_rhs(&w_cells, &vec![c; n], &ExtrapolationTracerBC3D);
            let scale = c * max_abs(case.eta_rate.iter().copied());
            for idx in 0..ne * nn {
                for j in 0..nw {
                    let expected = c * w_cells.d_sigma[j] * case.eta_rate[idx];
                    let got = rhs[idx * nw + j];
                    assert!(
                        (got - expected).abs() < 1e-12 * scale,
                        "node {idx}, w-point {j}: {got:.6e} vs C·Δσ_w·∂η/∂t = {expected:.6e}"
                    );
                }
            }

            if !closed {
                continue;
            }
            // A field that varies in x, y and z, upwind horizontally and
            // limited-Akima vertically: the inventory tendency integrates to
            // zero
            let field: Vec<f64> = (0..n)
                .map(|m| {
                    let (idx, j) = (m / nw, m % nw);
                    case.state.temp[idx * nl + j.min(nl - 1)] + 0.5 * j as f64
                })
                .collect();
            let rhs = case.tracer_rhs(&w_cells, &field, &ExtrapolationTracerBC3D);
            let integral = |values: &[f64]| -> f64 {
                (0..ne)
                    .map(|k| {
                        let sums: Vec<f64> = (0..nn)
                            .map(|i| values[(k * nn + i) * nw..][..nw].iter().sum())
                            .collect();
                        case.geom.integrate_element(k, &sums)
                    })
                    .sum()
            };
            let hu_scale = max_abs(w_cells.hu.iter().copied());
            let scale = integral(&vec![hu_scale / 1000.0; n]) * max_abs(field.iter().copied());
            let tendency = integral(&rhs);
            assert!(
                tendency.abs() < 1e-12 * scale,
                "inventory tendency {tendency:.3e} (advective scale {scale:.3e})"
            );
        }
    }
}
