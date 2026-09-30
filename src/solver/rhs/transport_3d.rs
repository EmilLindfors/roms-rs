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
//!    `{{Q_l}}·n` at the element faces;
//! 2. corrected by the layer's share of the difference to the barotropic
//!    transport, `Q_l += Δσ_l (DU_avg2 − Σ_m Q_m)` at the nodes and on the
//!    faces, so that `Σ_l Q_l = DU_avg2` exactly. The dissipative part of the
//!    2D face flux is shared out the same way;
//! 3. `Ω` integrated up from the bed, with the layer's share of the barotropic
//!    `∂η/∂t`: `Ω_{l+1/2} = Ω_{l−1/2} − ∇·Q_l − Δσ_l ∂η/∂t`.
//!
//! The DG divergence ([`transport_divergence_element`]) is linear in the nodal
//! transport and the face flux, so `Σ_l ∇·Q_l = ∇·DU_avg2 = −∂η/∂t` and `Ω` at
//! the surface vanishes to round-off. That holds wherever the barotropic pass
//! keeps the nodal identity; where it only keeps element balances (`WetDry`
//! elements with a dry node, positivity-limited elements, see
//! `BarotropicTransport`), the surface residual is spread linearly over the
//! column. It integrates to zero over the element (the element balance), so
//! every layer's continuity still holds for the element as a whole, and the
//! mode splitter carries the tracers of those elements as element means per
//! level ([`inventory_to_concentration`]): constant and conservative there
//! too. This is what makes 3D wetting and drying work.
//!
//! [`apply_tracer_transport_3d`] then advects a tracer with these fluxes in
//! inventory form: upwind in `C` on the face fluxes and on `Ω`.

use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::rhs::advection_3d::{TracerBCContext3D, TracerBoundaryCondition3D};
use crate::solver::state::Solution3D;
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
    /// Largest `|Ω|` at the surface before the residual was spread over the
    /// column (m/s): round-off where the barotropic pass keeps the nodal
    /// identity `η̄ − ηⁿ = −Δt∇·DU_avg2`.
    pub surface_residual: f64,
    // One element's layer: nodal transport, face fluxes, divergence
    layer_hu: Vec<f64>,
    layer_hv: Vec<f64>,
    layer_face: Vec<f64>,
    layer_div: Vec<f64>,
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
            surface_residual: 0.0,
            layer_hu: vec![0.0; nn],
            layer_hv: vec![0.0; nn],
            layer_face: vec![0.0; 4 * nfn],
            layer_div: vec![0.0; nn * n_levels],
        }
    }

    /// Number of layers.
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }

    /// Layer transports and `Ω` of `state` (its `η`, `u`, `v`), corrected to
    /// the barotropic transport `barotropic`.
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        state: &Solution3D,
        barotropic: BarotropicFlux,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        sigma: &SigmaGrid,
        bathymetry: &Bathymetry2D,
    ) {
        let (nn, nfn, nl) = (ops.n_nodes, ops.n_face_nodes, self.n_levels);
        assert_eq!(state.n_levels, nl, "layer count of the state");
        let d_sigma = sigma.d_sigma();

        // 1. Nodal layer transports, corrected to DU_avg2
        for k in 0..state.n_elements {
            let bed = bathymetry.element(ElementIndex::new(k));
            for (i, &b) in bed.iter().enumerate() {
                let idx = k * nn + i;
                let depth = state.eta.data[idx] - b;
                let column = idx * nl..(idx + 1) * nl;
                let (mut sum_u, mut sum_v) = (0.0, 0.0);
                for (((hu, hv), (&u, &v)), &ds) in self.hu[column.clone()]
                    .iter_mut()
                    .zip(&mut self.hv[column.clone()])
                    .zip(state.u[column.clone()].iter().zip(&state.v[column.clone()]))
                    .zip(d_sigma)
                {
                    *hu = depth * ds * u;
                    *hv = depth * ds * v;
                    sum_u += *hu;
                    sum_v += *hv;
                }
                let (corr_u, corr_v) = (barotropic.hu[idx] - sum_u, barotropic.hv[idx] - sum_v);
                for ((hu, hv), &ds) in self.hu[column.clone()]
                    .iter_mut()
                    .zip(&mut self.hv[column])
                    .zip(d_sigma)
                {
                    *hu += ds * corr_u;
                    *hv += ds * corr_v;
                }
            }
        }

        // 2. Face fluxes: central average of the nodal transports, corrected
        // to the barotropic face flux
        for k in 0..state.n_elements {
            let el = ElementIndex::new(k);
            for f in 0..4 {
                let neighbor = mesh.neighbor(el, f);
                for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
                    let (nx, ny) = geom.normal(k, f, fi);
                    let interior = (k * nn + node) * nl;
                    let exterior = neighbor.map(|nb| {
                        (nb.element * nn + ops.face_nodes[nb.face][nfn - 1 - fi]) * nl
                    });
                    let slot = (k * 4 + f) * nfn + fi;
                    let fluxes = &mut self.face[slot * nl..(slot + 1) * nl];
                    let mut sum = 0.0;
                    for (l, flux) in fluxes.iter_mut().enumerate() {
                        // A physical boundary carries no layer flux of its own:
                        // all of it comes from the barotropic flux (zero at
                        // walls, the 2D open-boundary flux elsewhere).
                        *flux = exterior.map_or(0.0, |e| {
                            let q_in = nx * self.hu[interior + l] + ny * self.hv[interior + l];
                            let q_ex = nx * self.hu[e + l] + ny * self.hv[e + l];
                            0.5 * (q_in + q_ex)
                        });
                        sum += *flux;
                    }
                    let corr = barotropic.face[slot] - sum;
                    for (flux, &ds) in fluxes.iter_mut().zip(d_sigma) {
                        *flux += ds * corr;
                    }
                }
            }
        }

        // 3. Ω from the bed up
        self.surface_residual = 0.0;
        let sigma_w = sigma.sigma_w();
        for k in 0..state.n_elements {
            for l in 0..nl {
                for i in 0..nn {
                    self.layer_hu[i] = self.hu[(k * nn + i) * nl + l];
                    self.layer_hv[i] = self.hv[(k * nn + i) * nl + l];
                }
                for (slot, flux) in self.layer_face.iter_mut().enumerate() {
                    *flux = self.face[((k * 4 * nfn) + slot) * nl + l];
                }
                transport_divergence_element(
                    ops,
                    geom,
                    k,
                    &self.layer_hu,
                    &self.layer_hv,
                    &self.layer_face,
                    &mut self.layer_div[l * nn..(l + 1) * nn],
                );
            }
            for i in 0..nn {
                let idx = k * nn + i;
                let rate = barotropic.eta_rate[idx];
                let omega = &mut self.omega[idx * (nl + 1)..(idx + 1) * (nl + 1)];
                omega[0] = 0.0;
                for l in 0..nl {
                    omega[l + 1] = omega[l] - self.layer_div[l * nn + i] - d_sigma[l] * rate;
                }
                let residual = omega[nl];
                self.surface_residual = self.surface_residual.max(residual.abs());
                for (w, &s) in omega.iter_mut().zip(sigma_w) {
                    *w -= (s + 1.0) * residual;
                }
            }
        }
    }
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

/// Multiply a tracer by the layer thickness of `η`: concentration `C` →
/// inventory `H_z C`.
pub fn tracer_to_inventory(
    tracer: &mut [f64],
    eta: &[f64],
    sigma: &SigmaGrid,
    bathymetry: &Bathymetry2D,
) {
    let nl = sigma.n_levels();
    let d_sigma = sigma.d_sigma();
    for ((column, &e), &b) in tracer.chunks_exact_mut(nl).zip(eta).zip(&bathymetry.data) {
        for (c, &ds) in column.iter_mut().zip(d_sigma) {
            *c *= layer_thickness_of(e - b, ds);
        }
    }
}

/// Inventory `q = H_z C` → concentration, written to `out`.
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
/// Converting back with [`tracer_to_inventory`] keeps each element's
/// inventory per level exactly.
#[allow(clippy::too_many_arguments)]
pub fn inventory_to_concentration(
    q: &[f64],
    eta: &[f64],
    sigma: &SigmaGrid,
    bathymetry: &Bathymetry2D,
    geom: &GeometricFactors2D,
    element_means: &[bool],
    out: &mut [f64],
) {
    let nl = sigma.n_levels();
    let nn = bathymetry.n_nodes;
    let d_sigma = sigma.d_sigma();
    for k in 0..bathymetry.n_elements {
        let bed = bathymetry.element(ElementIndex::new(k));
        let eta_k = &eta[k * nn..(k + 1) * nn];
        let block = k * nn * nl..(k + 1) * nn * nl;
        let (q_k, out_k) = (&q[block.clone()], &mut out[block]);
        if !element_means[k] {
            for (i, (&e, &b)) in eta_k.iter().zip(bed).enumerate() {
                if e - b < DRY_DEPTH {
                    continue;
                }
                for (l, &ds) in d_sigma.iter().enumerate() {
                    out_k[i * nl + l] = q_k[i * nl + l] / layer_thickness_of(e - b, ds);
                }
            }
            continue;
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
    }
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
/// inflow through a physical boundary. Vertically `C` is upwinded on `Ω`
/// (first order, TODO P4.5).
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
    scratch: &mut TracerTransportScratch,
) {
    let (nn, nfn, nl) = (ops.n_nodes, ops.n_face_nodes, transport.n_levels);
    let TracerTransportScratch {
        hu,
        hv,
        face,
        div,
        vertical,
    } = scratch;

    for k in 0..mesh.n_elements {
        let el = ElementIndex::new(k);
        for l in 0..nl {
            for i in 0..nn {
                let idx = (k * nn + i) * nl + l;
                hu[i] = transport.hu[idx] * tracer[idx];
                hv[i] = transport.hv[idx] * tracer[idx];
            }
            for f in 0..4 {
                let neighbor = mesh.neighbor(el, f);
                for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
                    let slot = (k * 4 + f) * nfn + fi;
                    let flux = transport.face[slot * nl + l];
                    let interior = tracer[(k * nn + node) * nl + l];
                    let upwind = if flux >= 0.0 {
                        interior
                    } else if let Some(nb) = neighbor {
                        let nb_node = ops.face_nodes[nb.face][nfn - 1 - fi];
                        tracer[(nb.element * nn + nb_node) * nl + l]
                    } else {
                        bc.exterior_value(&TracerBCContext3D {
                            element: k,
                            face: f,
                            level: l,
                            face_node: fi,
                            boundary_tag: mesh.boundary_tag(el, f),
                            interior_value: interior,
                            normal_velocity: flux,
                        })
                    };
                    face[f * nfn + fi] = flux * upwind;
                }
            }
            transport_divergence_element(ops, geom, k, hu, hv, face, div);
            for (i, &d) in div.iter().enumerate() {
                rhs[(k * nn + i) * nl + l] = -d;
            }
        }

        // Vertical upwind flux through the σ-surfaces
        for i in 0..nn {
            let idx = k * nn + i;
            let omega = &transport.omega[idx * (nl + 1)..(idx + 1) * (nl + 1)];
            let column = &tracer[idx * nl..(idx + 1) * nl];
            vertical[0] = 0.0;
            vertical[nl] = 0.0;
            for l in 1..nl {
                let w = omega[l];
                vertical[l] = w * if w >= 0.0 { column[l - 1] } else { column[l] };
            }
            for (l, r) in rhs[idx * nl..(idx + 1) * nl].iter_mut().enumerate() {
                *r -= vertical[l + 1] - vertical[l];
            }
        }
    }
}

/// Buffers of [`apply_tracer_transport_3d`], sized for one element.
pub struct TracerTransportScratch {
    hu: Vec<f64>,
    hv: Vec<f64>,
    face: Vec<f64>,
    div: Vec<f64>,
    vertical: Vec<f64>,
}

impl TracerTransportScratch {
    /// Buffers for elements of `ops` and `n_levels` layers.
    pub fn new(ops: &DGOperators2D, n_levels: usize) -> Self {
        Self {
            hu: vec![0.0; ops.n_nodes],
            hv: vec![0.0; ops.n_nodes],
            face: vec![0.0; 4 * ops.n_face_nodes],
            div: vec![0.0; ops.n_nodes],
            vertical: vec![0.0; n_levels + 1],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::rhs::advection_3d::{
        ExtrapolationTracerBC3D, FixedTracerBC3D, UpwindTracerBC3D,
    };
    use std::f64::consts::PI;

    const LX: f64 = 2000.0;
    const LY: f64 = 1000.0;

    /// A closed (or periodic) basin over a sloping, uneven bed with a sheared,
    /// non-uniform flow, a barotropic transport that is not the depth integral
    /// of the 3D velocities, and the free-surface rate that transport implies.
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
            // face, zero through walls
            let mut du_face = vec![0.0; mesh.n_elements * 4 * nfn];
            for k in 0..mesh.n_elements {
                let el = ElementIndex::new(k);
                for f in 0..4 {
                    let neighbor = mesh.neighbor(el, f);
                    for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
                        let (nx, ny) = geom.normal(k, f, fi);
                        let a = k * nn + node;
                        du_face[(k * 4 + f) * nfn + fi] = neighbor.map_or(0.0, |nb| {
                            let b = nb.element * nn + ops.face_nodes[nb.face][nfn - 1 - fi];
                            let qa = nx * du_hu[a] + ny * du_hv[a];
                            let qb = nx * du_hu[b] + ny * du_hv[b];
                            0.5 * (qa + qb) - 0.3 * (state.eta.data[b] - state.eta.data[a])
                        });
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
            }
        }

        fn closed() -> Self {
            Self::new(Mesh2D::uniform_rectangle(0.0, LX, 0.0, LY, 4, 3), true)
        }

        fn periodic() -> Self {
            Self::new(Mesh2D::uniform_periodic(0.0, LX, 0.0, LY, 4, 3), false)
        }

        fn transport(&self) -> LayerTransport {
            let mut transport =
                LayerTransport::new(self.mesh.n_elements, &self.ops, self.sigma.n_levels());
            let barotropic = BarotropicFlux {
                hu: &self.du_hu,
                hv: &self.du_hv,
                face: &self.du_face,
                eta_rate: &self.eta_rate,
            };
            transport.compute(
                &self.state,
                barotropic,
                &self.mesh,
                &self.ops,
                &self.geom,
                &self.sigma,
                &self.bathymetry,
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
            let mut scratch = TracerTransportScratch::new(&self.ops, self.sigma.n_levels());
            apply_tracer_transport_3d(
                &mut rhs,
                tracer,
                transport,
                &self.mesh,
                &self.ops,
                &self.geom,
                bc,
                &mut scratch,
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
        let case = Case::closed();
        let transport = case.transport();
        let (nn, nfn, nl) = (case.ops.n_nodes, case.ops.n_face_nodes, case.sigma.n_levels());
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
                if case.mesh.neighbor(ElementIndex::new(k), f).is_none() {
                    let slots = (k * 4 + f) * nfn * nl..(k * 4 + f + 1) * nfn * nl;
                    assert_eq!(max_abs(transport.face[slots].iter().copied()), 0.0);
                }
            }
        }
    }

    /// With `∂η/∂t = −∇·DU_avg2` (the barotropic pass's nodal identity), Ω
    /// integrated from the bed closes at the surface to round-off: nothing is
    /// left for the linear correction.
    #[test]
    fn omega_vanishes_at_the_surface_without_correction() {
        for case in [Case::closed(), Case::periodic()] {
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
        let case = Case::closed();
        let transport = case.transport();
        let c = 34.7;
        let tracer = vec![c; case.state.temp.len()];
        let (nn, nl) = (case.ops.n_nodes, case.sigma.n_levels());
        let scale = c * max_abs(case.eta_rate.iter().copied());
        let bcs: [&dyn TracerBoundaryCondition3D; 2] =
            [&ExtrapolationTracerBC3D, &FixedTracerBC3D::new(5.0)];
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
        let mut scratch = TracerTransportScratch::new(&ops, nl);
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
                &mut scratch,
            );
            for column in rhs.chunks_exact(nl) {
                for (got, want) in column.iter().zip(expected) {
                    assert!((got - want).abs() < 1e-14, "Ω = {w}: {column:?}");
                }
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
        tracer_to_inventory(&mut inventory, &eta, &case.sigma, &case.bathymetry);
        let last = vec![-1.0; concentration.len()];
        let mut out = last.clone();
        inventory_to_concentration(
            &inventory,
            &eta,
            &case.sigma,
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
        tracer_to_inventory(&mut back, &eta, &case.sigma, &case.bathymetry);
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
            assert!((before - after).abs() < 1e-13 * before.abs(), "level {l}: {before} vs {after}");
        }
    }
}
