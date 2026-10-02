//! Horizontal eddy viscosity of the 3D momentum, on the vertical shear.
//!
//! Along every σ-layer `l`, the inventory tendency is
//!
//! ```text
//!     ∂(H_z u_l)/∂t = ∇·(ν_l H_z ∇u′_l),    u′_l = u_l − ⟨u⟩,    H_z = Δσ_l D,
//! ```
//!
//! with `⟨u⟩ = Σ_l Δσ_l u_l` the column's depth mean and `D = η − B`: the
//! Laplacian along σ-surfaces (ROMS's `UV_VIS2` with `MIX_S_UV`), of the shear
//! only. As a velocity tendency, `Δσ_l` cancels: `∇·(ν_l D∇u′_l)/D`. It is
//! discretised with BR1 per layer (Bassi & Rebay 1997; Hesthaven & Warburton
//! 2008, §7.2), as the 2D module's viscosity
//! (`solver/rhs/swe_2d_viscosity.rs`), with the shared per-element kernels of
//! `diffusion_2d`.
//!
//! The viscosity ([`HorizontalViscosity3D`]) is a constant background plus,
//! optionally, Smagorinsky's (1963) `(C_s Δ)²|S_l|`, from the horizontal
//! strain rate `|S| = √(2S₁₁² + 2S₂₂² + 4S₁₂²)` of the layer's own velocity
//! `u_l` (its BR1 gradient: the shear's plus the depth mean's) and the node
//! spacing `Δ = √(area)/N`. It follows the shear: an interface that rolls up
//! at the grid scale gets it, a smooth flow little.
//!
//! The depth mean is the 2D module's (its own `HorizontalViscosity2D`): with a
//! constant `ν`, BR1 is linear and `D` the same on every layer, so the column
//! sum `Σ_l Δσ_l ∇·(νD∇u′_l) = ∇·(νD∇Σ_l Δσ_l u′_l)` vanishes, to round-off,
//! at every node. With Smagorinsky's `ν_l` it does not: what is left,
//! `∇·(D Σ_l Δσ_l ν_l∇u′_l)`, is the stress the shear's viscosity exerts on
//! the mean flow (zero for a ν uncorrelated with the shear), and the slow
//! forcing `G`, which takes the column integral of the 3D momentum tendency,
//! hands it to the depth mean. Neither way is anything counted twice: the 2D
//! module differentiates only `⟨u⟩`, this kernel only `u′`.
//!
//! The face flux is central and single-valued, so interior faces conserve
//! the layer momentum. Walls mirror the velocity (free slip on the normal
//! component, as the 2D module's reflective ghost state); open faces
//! extrapolate it (zero gradient). Boundary faces take the interior flux, as
//! in the 2D module. Thin columns (`min_column_depth`) carry no shear and,
//! for the strain, no mean flow (the films' velocity is the 2D module's, as
//! in the 3D advection): they enter their neighbours' gradients at rest and
//! get no tendency.

use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::core::blocks::{Pooled, for_each_block};
use crate::solver::state::Solution3D;
use crate::source::swe_2d::viscosity::strain_rate_magnitude;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

use super::boundary_3d::{Boundaries3D, FaceExterior};
use super::diffusion_2d::{
    DiffusionScratch, ScalarGradient2D, br1_diffusion_element, br1_gradient_element,
};

/// The horizontal eddy viscosity of the 3D shear: `ν = ν₀ + (C_s Δ)²|S|`
/// (m²/s; see the [module documentation](self)).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HorizontalViscosity3D {
    /// The constant background `ν₀` (m²/s).
    pub background: f64,
    /// Smagorinsky's coefficient `C_s` (typically 0.1–0.2; 0 for none).
    pub smagorinsky: f64,
}

impl HorizontalViscosity3D {
    /// A constant viscosity `nu` (m²/s).
    pub fn constant(nu: f64) -> Self {
        Self {
            background: nu,
            smagorinsky: 0.0,
        }
    }

    /// Smagorinsky's viscosity with coefficient `cs`, without a background.
    pub fn smagorinsky(cs: f64) -> Self {
        Self {
            background: 0.0,
            smagorinsky: cs,
        }
    }

    /// Whether the viscosity is zero everywhere.
    pub fn is_zero(&self) -> bool {
        self.background == 0.0 && self.smagorinsky == 0.0
    }
}

/// Buffers of [`apply_horizontal_viscosity_3d`], reused between calls.
pub struct ViscosityScratch3D {
    /// `D` of every column, zero in thin ones, `[element][node]`.
    depth: Vec<f64>,
    /// `(C_s Δ)²` of every element.
    smagorinsky_area: Vec<f64>,
    /// The depth mean `(⟨u⟩, ⟨v⟩)` of every column (zero in thin ones).
    mean: Vec<[f64; 2]>,
    /// Their BR1 gradients (with Smagorinsky only).
    mean_gradient: Vec<[ScalarGradient2D; 2]>,
    /// One layer's shear gradients and `ν_l D`, `[element][node]`.
    layer: Vec<LayerNode>,
}

/// The BR1 gradients of a layer's shear `(u′, v′)` at a node, and `ν_l D`
/// there.
#[derive(Clone, Copy, Default)]
struct LayerNode {
    gradient: [ScalarGradient2D; 2],
    coefficient: f64,
}

impl ViscosityScratch3D {
    /// Buffers for `n_elements` elements of `ops`.
    pub fn new(n_elements: usize, ops: &DGOperators2D) -> Self {
        let n_total = n_elements * ops.n_nodes;
        Self {
            depth: vec![0.0; n_total],
            smagorinsky_area: vec![0.0; n_elements],
            mean: vec![[0.0; 2]; n_total],
            mean_gradient: vec![[ScalarGradient2D::default(); 2]; n_total],
            layer: vec![LayerNode::default(); n_total],
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

/// The mesh, operators and column data the kernel reads.
struct Columns<'a> {
    state: &'a Solution3D,
    viscosity: HorizontalViscosity3D,
    mesh: &'a Mesh2D,
    ops: &'a DGOperators2D,
    geom: &'a GeometricFactors2D,
    boundaries: &'a Boundaries3D,
}

impl Columns<'_> {
    /// BR1 gradients on element `k` of the velocity field `value(idx)`
    /// (`[element][node]` index, both components), mirrored at walls, into
    /// `scratch.gradient`.
    fn element_gradients(
        &self,
        k: usize,
        value: impl Fn(usize) -> [f64; 2],
        scratch: &mut ElementScratch,
    ) {
        let (mesh, geom, nn) = (self.mesh, self.geom, self.ops.n_nodes);
        let ElementScratch { own, gradient, .. } = scratch;
        let [grad_u, grad_v] = gradient;
        br1_gradient_element(
            ElementIndex::new(k),
            mesh,
            self.ops,
            geom,
            |j, node| value(j.as_usize() * nn + node),
            |k, face, fi, _node, interior| match self.boundaries.exterior(mesh, k, face) {
                // Mirrored: no normal velocity at the wall
                FaceExterior::Wall => {
                    let (nx, ny) = geom.normal(k.as_usize(), face, fi);
                    let [u, v] = interior;
                    let normal = u * nx + v * ny;
                    [u - 2.0 * normal * nx, v - 2.0 * normal * ny]
                }
                _ => interior,
            },
            own,
            [grad_u, grad_v],
        );
    }

    /// Depths, depth means, the Smagorinsky areas and (with Smagorinsky)
    /// the means' gradients.
    fn prepare(
        &self,
        bathymetry: &Bathymetry2D,
        sigma: &SigmaGrid,
        min_column_depth: f64,
        scratch: &mut ViscosityScratch3D,
    ) {
        let (state, nl) = (self.state, self.state.n_levels);
        let ne = self.mesh.n_elements;
        let ViscosityScratch3D {
            depth,
            smagorinsky_area,
            mean,
            mean_gradient,
            ..
        } = scratch;
        for_each_block(
            ne,
            [&mut depth[..]],
            || (),
            |_, k, [depth_k]| {
                let nodes = k * depth_k.len()..(k + 1) * depth_k.len();
                for ((d, &eta), &b) in depth_k
                    .iter_mut()
                    .zip(&state.eta.data[nodes.clone()])
                    .zip(&bathymetry.data[nodes])
                {
                    let depth = eta - b;
                    *d = if depth < min_column_depth { 0.0 } else { depth };
                }
            },
        );
        let depth: &[f64] = depth;
        for_each_block(
            ne,
            [&mut mean[..]],
            || (),
            |_, k, [mean_k]| {
                let nn = mean_k.len();
                for (i, m) in mean_k.iter_mut().enumerate() {
                    let idx = k * nn + i;
                    let column = idx * nl..(idx + 1) * nl;
                    *m = if depth[idx] == 0.0 {
                        [0.0; 2]
                    } else {
                        [
                            sigma.depth_average(&state.u[column.clone()]),
                            sigma.depth_average(&state.v[column]),
                        ]
                    };
                }
            },
        );
        let cs = self.viscosity.smagorinsky;
        if cs > 0.0 {
            let width = 1.0 / self.ops.order.max(1) as f64;
            for (k, area) in smagorinsky_area.iter_mut().enumerate() {
                *area = (cs * width * self.geom.element_size(k)).powi(2);
            }
            let (mean, nn): (&[[f64; 2]], _) = (mean, self.ops.n_nodes);
            for_each_block(
                ne,
                [&mut mean_gradient[..]],
                || ElementScratch::take(nn),
                |scratch, k, [gradient_k]| {
                    self.element_gradients(k, |idx| mean[idx], scratch);
                    for (i, g) in gradient_k.iter_mut().enumerate() {
                        *g = [scratch.gradient[0][i], scratch.gradient[1][i]];
                    }
                },
            );
        }
    }

    /// Layer `level`'s shear gradients and `ν_l D` into `scratch.layer`.
    fn layer(&self, level: usize, scratch: &mut ViscosityScratch3D) {
        let (state, nn, nl) = (self.state, self.ops.n_nodes, self.state.n_levels);
        let ViscosityScratch3D {
            depth,
            smagorinsky_area,
            mean,
            mean_gradient,
            layer,
        } = scratch;
        let shear = |idx: usize| {
            if depth[idx] == 0.0 {
                [0.0; 2]
            } else {
                let [mu, mv] = mean[idx];
                [
                    state.u[idx * nl + level] - mu,
                    state.v[idx * nl + level] - mv,
                ]
            }
        };
        let background = self.viscosity.background;
        let smagorinsky = self.viscosity.smagorinsky > 0.0;
        for_each_block(
            self.mesh.n_elements,
            [&mut layer[..]],
            || ElementScratch::take(nn),
            |scratch, k, [layer_k]| {
                self.element_gradients(k, shear, scratch);
                for (i, node) in layer_k.iter_mut().enumerate() {
                    let idx = k * nn + i;
                    let (gu, gv) = (scratch.gradient[0][i], scratch.gradient[1][i]);
                    let nu = if smagorinsky {
                        let [mu, mv] = mean_gradient[idx];
                        let strain = strain_rate_magnitude(
                            gu.dx + mu.dx,
                            gu.dy + mu.dy,
                            gv.dx + mv.dx,
                            gv.dy + mv.dy,
                        );
                        background + smagorinsky_area[k] * strain
                    } else {
                        background
                    };
                    *node = LayerNode {
                        gradient: [gu, gv],
                        coefficient: nu * depth[idx],
                    };
                }
            },
        );
    }
}

/// Add the velocity tendency `∇·(ν_l D∇u′_l)/D` of the horizontal
/// `viscosity` on the vertical shear of `state` to `rhs_u`, `rhs_v`
/// (`[element][node][level]`; see the [module documentation](self)).
/// Columns shallower than `min_column_depth` get nothing.
#[allow(clippy::too_many_arguments)]
pub fn apply_horizontal_viscosity_3d(
    rhs_u: &mut [f64],
    rhs_v: &mut [f64],
    state: &Solution3D,
    viscosity: HorizontalViscosity3D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    bathymetry: &Bathymetry2D,
    sigma: &SigmaGrid,
    boundaries: &Boundaries3D,
    min_column_depth: f64,
    scratch: &mut ViscosityScratch3D,
) {
    if viscosity.is_zero() {
        return;
    }
    let columns = Columns {
        state,
        viscosity,
        mesh,
        ops,
        geom,
        boundaries,
    };
    let (nn, nl) = (ops.n_nodes, state.n_levels);
    let n = mesh.n_elements * nn * nl;
    columns.prepare(bathymetry, sigma, min_column_depth, scratch);

    for level in 0..nl {
        columns.layer(level, scratch);
        let (depth, layer) = (&scratch.depth, &scratch.layer);
        for_each_block(
            mesh.n_elements,
            [&mut rhs_u[..n], &mut rhs_v[..n]],
            || ElementScratch::take(nn),
            |scratch, k, rhs| {
                let ElementScratch { diffusion, out, .. } = &mut **scratch;
                let [out_u, out_v] = out;
                br1_diffusion_element(
                    ElementIndex::new(k),
                    mesh,
                    ops,
                    geom,
                    |j, node| {
                        let node = layer[j.as_usize() * nn + node];
                        node.gradient
                            .map(|g| (node.coefficient * g.dx, node.coefficient * g.dy))
                    },
                    diffusion,
                    [out_u, out_v],
                );
                for (rhs_k, out) in rhs.into_iter().zip(out.iter()) {
                    for (i, &d) in out.iter().enumerate() {
                        let idx = k * nn + i;
                        if depth[idx] > 0.0 {
                            rhs_k[i * nl + level] += d / depth[idx];
                        }
                    }
                }
            },
        );
    }
}

/// The largest viscosity `ν_l` (m²/s) of every element's nodes and layers
/// into `largest` (`[element]`), for the time step: the background, plus
/// Smagorinsky's from the strain of `state` (thin columns have the
/// background). Arguments as for [`apply_horizontal_viscosity_3d`].
#[allow(clippy::too_many_arguments)]
pub fn largest_horizontal_viscosity_3d(
    largest: &mut [f64],
    state: &Solution3D,
    viscosity: HorizontalViscosity3D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    bathymetry: &Bathymetry2D,
    sigma: &SigmaGrid,
    boundaries: &Boundaries3D,
    min_column_depth: f64,
    scratch: &mut ViscosityScratch3D,
) {
    largest.fill(viscosity.background);
    if viscosity.smagorinsky == 0.0 {
        return;
    }
    let columns = Columns {
        state,
        viscosity,
        mesh,
        ops,
        geom,
        boundaries,
    };
    let nn = ops.n_nodes;
    columns.prepare(bathymetry, sigma, min_column_depth, scratch);
    for level in 0..state.n_levels {
        columns.layer(level, scratch);
        for (idx, (node, &d)) in scratch.layer.iter().zip(&scratch.depth).enumerate() {
            if d > 0.0 {
                let nu = &mut largest[idx / nn];
                *nu = nu.max(node.coefficient / d);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vertical::UniformStretching;

    /// A periodic `n × n` square of side `length`, order `order`.
    fn periodic(
        n: usize,
        order: usize,
        length: f64,
    ) -> (Mesh2D, DGOperators2D, GeometricFactors2D) {
        let mesh = Mesh2D::uniform_periodic(0.0, length, 0.0, length, n, n);
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        (mesh, ops, geom)
    }

    fn node_xy(mesh: &Mesh2D, ops: &DGOperators2D, idx: usize) -> [f64; 2] {
        let (k, i) = (idx / ops.n_nodes, idx % ops.n_nodes);
        mesh.reference_to_physical(ElementIndex::new(k), ops.nodes_r[i], ops.nodes_s[i])
    }

    /// `state` with `u = ū + profile(σ)·shape(x, y)` (and `v` from
    /// `shape_v`), `η` from `eta`.
    fn sheared_state(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        sigma: &SigmaGrid,
        eta: impl Fn(f64, f64) -> f64,
        mean: f64,
        shape_u: impl Fn(f64, f64) -> f64,
        shape_v: impl Fn(f64, f64) -> f64,
    ) -> Solution3D {
        let (nn, nl) = (ops.n_nodes, sigma.n_levels());
        let mut state = Solution3D::new(mesh.n_elements, nn, nl);
        // A profile with zero depth mean on any spacing
        let sigma_rho = sigma.sigma_rho();
        let raw: Vec<f64> = sigma_rho.iter().map(|s| (2.0 * s + 1.0).powi(3)).collect();
        let offset = sigma.depth_average(&raw);
        for idx in 0..mesh.n_elements * nn {
            let [x, y] = node_xy(mesh, ops, idx);
            state.eta.data[idx] = eta(x, y);
            state.ubar.data[idx] = mean;
            for (l, r) in raw.iter().enumerate() {
                let p = r - offset;
                state.u[idx * nl + l] = mean + p * shape_u(x, y);
                state.v[idx * nl + l] = mean + p * shape_v(x, y);
            }
        }
        state
    }

    fn apply(
        state: &Solution3D,
        viscosity: HorizontalViscosity3D,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        bathymetry: &Bathymetry2D,
        sigma: &SigmaGrid,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut rhs_u = vec![0.0; state.u.len()];
        let mut rhs_v = vec![0.0; state.v.len()];
        let mut scratch = ViscosityScratch3D::new(mesh.n_elements, ops);
        apply_horizontal_viscosity_3d(
            &mut rhs_u,
            &mut rhs_v,
            state,
            viscosity,
            mesh,
            ops,
            geom,
            bathymetry,
            sigma,
            &Boundaries3D::default(),
            0.1,
            &mut scratch,
        );
        (rhs_u, rhs_v)
    }

    /// With a constant ν the column sum `Σ_l Δσ_l D·(tendency)` vanishes at
    /// every node (the depth mean stays the 2D module's); with Smagorinsky's
    /// it does not. Either way the layer momentum `∫ H_z·(tendency)` of every
    /// layer is conserved on a periodic mesh, with a varying free surface
    /// and stretched levels, and the shear's energy `½∫ H_z|u′|²` decays.
    #[test]
    fn shear_viscosity_conserves_layer_momentum_and_dissipates() {
        use crate::vertical::SongHaidvogelStretching;
        let length = 1e3;
        let (mesh, ops, geom) = periodic(4, 3, length);
        let sigma = SigmaGrid::new(8, SongHaidvogelStretching::new(3.0, 0.4, 5.0));
        let depth = 20.0;
        let bathymetry = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -depth);
        let k = 2.0 * std::f64::consts::PI / length;
        let state = sheared_state(
            &mesh,
            &ops,
            &sigma,
            |x, y| 2.0 * (k * x).sin() * (k * y).cos(),
            0.3,
            |x, y| (k * x).cos() + 0.5 * (2.0 * k * y).sin(),
            |x, y| (k * (x + y)).sin(),
        );
        let (nn, nl) = (ops.n_nodes, sigma.n_levels());
        let d_sigma = sigma.d_sigma();
        let smagorinsky = HorizontalViscosity3D {
            background: 1.0,
            smagorinsky: 0.2,
        };
        for viscosity in [HorizontalViscosity3D::constant(5.0), smagorinsky] {
            let (rhs_u, rhs_v) = apply(&state, viscosity, &mesh, &ops, &geom, &bathymetry, &sigma);
            let scale = rhs_u.iter().fold(0.0_f64, |m, r| m.max(r.abs()));
            assert!(scale > 1e-6, "test regime: no tendency ({scale:e})");
            let mut layer_momentum = vec![[0.0; 2]; nl];
            let (mut largest_sum, mut work) = (0.0_f64, 0.0);
            for idx in 0..mesh.n_elements * nn {
                let d = state.eta.data[idx] + depth;
                let mass = geom.node_mass(idx / nn, idx % nn);
                for (rhs, field, c) in [(&rhs_u, &state.u, 0), (&rhs_v, &state.v, 1)] {
                    let column = idx * nl..(idx + 1) * nl;
                    let (rhs, field) = (&rhs[column.clone()], &field[column]);
                    let mean = sigma.depth_average(field);
                    let sum: f64 = rhs.iter().zip(d_sigma).map(|(r, ds)| r * ds).sum();
                    largest_sum = largest_sum.max(sum.abs());
                    for (l, ((r, f), ds)) in rhs.iter().zip(field).zip(d_sigma).enumerate() {
                        layer_momentum[l][c] += mass * ds * d * r;
                        work += mass * ds * d * (f - mean) * r;
                    }
                }
            }
            if viscosity.smagorinsky == 0.0 {
                assert!(
                    largest_sum < 1e-12 * scale,
                    "column sum {largest_sum:e} (scale {scale:e})"
                );
            } else {
                assert!(
                    largest_sum > 1e-3 * scale,
                    "test regime: Smagorinsky's column sum {largest_sum:e} (scale {scale:e})"
                );
            }
            let volume_scale = scale * length * length * depth;
            for (l, m) in layer_momentum.iter().enumerate() {
                assert!(
                    m[0].abs().max(m[1].abs()) < 1e-12 * volume_scale,
                    "{viscosity:?}: layer {l} gains momentum {m:?} (scale {volume_scale:e})"
                );
            }
            assert!(
                work < 0.0,
                "{viscosity:?}: the shear gains energy ({work:e})"
            );
        }
    }

    /// Smagorinsky's ν at the nodes is `ν₀ + (C_s Δ)²|S|` of each layer's
    /// own velocity (its shear and the depth mean), `Δ` the node spacing: on
    /// a smooth field at P4 the largest of each element is the exact
    /// strain's to 1e-3 (BR1 gradients).
    #[test]
    fn smagorinsky_viscosity_follows_the_layer_strain() {
        let length = 1e3;
        let (mesh, ops, geom) = periodic(8, 4, length);
        let sigma = SigmaGrid::new(4, UniformStretching);
        let (nn, nl) = (ops.n_nodes, sigma.n_levels());
        let bathymetry = Bathymetry2D::constant(mesh.n_elements, nn, -10.0);
        let k = 2.0 * std::f64::consts::PI / length;
        let (background, cs) = (0.5, 0.15);
        // A profile with zero depth mean
        let profile: Vec<f64> = sigma.sigma_rho().iter().map(|s| 2.0 * s + 1.0).collect();
        // u_l = 0.2 + 0.1 cos(ky) + p_l sin(kx), v_l = ½ p_l cos(ky)
        let mut state = Solution3D::new(mesh.n_elements, nn, nl);
        let mut exact = vec![0.0_f64; mesh.n_elements];
        for idx in 0..mesh.n_elements * nn {
            let [x, y] = node_xy(&mesh, &ops, idx);
            let width = cs * geom.element_size(idx / nn) / 4.0;
            for (l, p) in profile.iter().enumerate() {
                state.u[idx * nl + l] = 0.2 + 0.1 * (k * y).cos() + p * (k * x).sin();
                state.v[idx * nl + l] = 0.5 * p * (k * y).cos();
                let strain = strain_rate_magnitude(
                    p * k * (k * x).cos(),
                    -0.1 * k * (k * y).sin(),
                    0.0,
                    -0.5 * p * k * (k * y).sin(),
                );
                let nu = &mut exact[idx / nn];
                *nu = nu.max(background + width * width * strain);
            }
        }
        let mut largest = vec![0.0; mesh.n_elements];
        largest_horizontal_viscosity_3d(
            &mut largest,
            &state,
            HorizontalViscosity3D {
                background,
                smagorinsky: cs,
            },
            &mesh,
            &ops,
            &geom,
            &bathymetry,
            &sigma,
            &Boundaries3D::default(),
            0.1,
            &mut ViscosityScratch3D::new(mesh.n_elements, &ops),
        );
        let scale = exact.iter().fold(0.0_f64, |m, &x| m.max(x - background));
        assert!(scale > 0.1, "test regime: Smagorinsky's ν only {scale:e}");
        for (k, (nu, want)) in largest.iter().zip(&exact).enumerate() {
            assert!(
                (nu - want).abs() < 1e-3 * scale,
                "element {k}: ν = {nu:.6}, exact {want:.6}"
            );
        }
    }

    /// Over a flat bed and a flat surface the tendency is `ν∇²u′`, so a
    /// shear `p(σ)·sin(kx)` decays at the rate `νk²`. The operator's nodal
    /// residual converges only at order N − 1 (as any DG Laplacian's), but
    /// the decay rate, its mass-weighted Rayleigh quotient
    /// `−∫ H_z u′·L(u′) / ∫ H_z u′²`, converges at 2N: relative errors
    /// 1.3e-2, 1.6e-5, 1.2e-8 at P1–P3 on 16² elements (rates 1.98, 3.99,
    /// 5.99).
    #[test]
    fn shear_viscosity_decays_a_shear_mode_at_the_viscous_rate() {
        let length = 1e3;
        let depth = 10.0;
        let nu = 2.0;
        let sigma = SigmaGrid::new(4, UniformStretching);
        let k = 2.0 * std::f64::consts::PI / length;
        for order in [1, 2, 3] {
            let mut errors = Vec::new();
            for n in [4, 8, 16] {
                let (mesh, ops, geom) = periodic(n, order, length);
                let bathymetry = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -depth);
                let state = sheared_state(
                    &mesh,
                    &ops,
                    &sigma,
                    |_, _| 0.0,
                    0.1,
                    |x, _| (k * x).sin(),
                    |_, y| (k * y).cos(),
                );
                let constant = HorizontalViscosity3D::constant(nu);
                let (rhs_u, rhs_v) =
                    apply(&state, constant, &mesh, &ops, &geom, &bathymetry, &sigma);
                let (nn, nl) = (ops.n_nodes, sigma.n_levels());
                let (mut work, mut energy) = (0.0, 0.0);
                for idx in 0..mesh.n_elements * nn {
                    let mass = geom.node_mass(idx / nn, idx % nn);
                    for (l, ds) in sigma.d_sigma().iter().enumerate() {
                        let i = idx * nl + l;
                        let (su, sv) = (state.u[i] - 0.1, state.v[i] - 0.1);
                        work += mass * ds * (su * rhs_u[i] + sv * rhs_v[i]);
                        energy += mass * ds * (su * su + sv * sv);
                    }
                }
                let rate = -work / energy;
                errors.push((rate / (nu * k * k) - 1.0).abs());
            }
            let rates: Vec<f64> = errors.windows(2).map(|e| (e[0] / e[1]).log2()).collect();
            assert!(
                rates.iter().all(|&r| r > 2.0 * order as f64 - 0.3),
                "P{order}: decay-rate errors {errors:?}, rates {rates:?}"
            );
        }
    }

    /// A uniform column (no shear) gets no tendency, also next to walls and
    /// thin columns.
    #[test]
    fn unsheared_columns_feel_no_viscosity() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1e3, 0.0, 500.0, 6, 3);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let sigma = SigmaGrid::new(5, UniformStretching);
        let nn = ops.n_nodes;
        let mut bathymetry = Bathymetry2D::constant(mesh.n_elements, nn, -10.0);
        let mut state = Solution3D::new(mesh.n_elements, nn, sigma.n_levels());
        for idx in 0..mesh.n_elements * nn {
            let [x, y] = node_xy(&mesh, &ops, idx);
            // A beach at the right end: the last column is nearly dry
            bathymetry.data[idx] = -10.0 + 9.99 * (x / 1e3).powi(4);
            let (u, v) = (0.1 + 1e-4 * x, -0.2 + 1e-4 * y);
            let column = idx * sigma.n_levels()..(idx + 1) * sigma.n_levels();
            state.u[column.clone()].fill(u);
            state.v[column].fill(v);
            state.ubar.data[idx] = u;
            state.vbar.data[idx] = v;
        }
        // Smagorinsky's ν follows the (here uniform) strain, but without
        // shear there is no tendency
        for viscosity in [
            HorizontalViscosity3D::constant(10.0),
            HorizontalViscosity3D::smagorinsky(0.2),
        ] {
            let (rhs_u, rhs_v) = apply(&state, viscosity, &mesh, &ops, &geom, &bathymetry, &sigma);
            let largest = rhs_u
                .iter()
                .chain(&rhs_v)
                .fold(0.0_f64, |m, r| m.max(r.abs()));
            // Round-off of u − ⟨u⟩ (Σ Δσ_l is 1 to round-off)
            assert!(
                largest < 1e-14,
                "{viscosity:?}: tendency {largest:e} without shear"
            );
        }
    }
}
