//! Horizontal eddy viscosity of the 3D momentum, on the vertical shear.
//!
//! Along every σ-layer `l`, the inventory tendency is
//!
//! ```text
//!     ∂(H_z u_l)/∂t = ∇·(ν H_z ∇u′_l),    u′_l = u_l − ⟨u⟩,    H_z = Δσ_l D,
//! ```
//!
//! with `⟨u⟩ = Σ_l Δσ_l u_l` the column's depth mean and `D = η − B`: the
//! Laplacian along σ-surfaces (ROMS's `UV_VIS2` with `MIX_S_UV`), of the shear
//! only. As a velocity tendency, `Δσ_l` cancels: `∇·(νD∇u′_l)/D`. It is
//! discretised with BR1 per layer (Bassi & Rebay 1997; Hesthaven & Warburton
//! 2008, §7.2), as the 2D module's viscosity
//! (`solver/rhs/swe_2d_viscosity.rs`), with the shared per-element kernels of
//! `diffusion_2d`.
//!
//! The depth mean is the 2D module's (its own `HorizontalViscosity2D`): with a
//! constant `ν`, BR1 is linear and `D` the same on every layer, so the column
//! sum `Σ_l Δσ_l ∇·(νD∇u′_l) = ∇·(νD∇Σ_l Δσ_l u′_l)` vanishes, to round-off,
//! at every node. The slow forcing `G` takes the column integral of the 3D
//! momentum tendency, so nothing is counted twice.
//!
//! The face flux is central and single-valued, so interior faces conserve
//! the layer momentum. Walls mirror the shear (free slip on the normal
//! component, as the 2D module's reflective ghost state); open faces
//! extrapolate it (zero gradient). Boundary faces take the interior flux, as
//! in the 2D module. Thin columns (`min_column_depth`) carry no shear: they
//! enter their neighbours' gradients with `u′ = 0` and get no tendency.

use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::state::Solution3D;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

use super::boundary_3d::{Boundaries3D, FaceExterior};
use super::diffusion_2d::{
    DiffusionScratch, ScalarGradient2D, br1_diffusion_element, br1_gradient_element,
};

/// Buffers of [`apply_horizontal_viscosity_3d`], reused between calls.
pub struct ViscosityScratch3D {
    /// `νD` of every column, zero in thin ones, `[element][node]`.
    coefficient: Vec<f64>,
    /// The depth mean `⟨u⟩`, `⟨v⟩` of every column.
    mean: [Vec<f64>; 2],
    /// The shear of one layer, `[element][node]`.
    shear: [Vec<f64>; 2],
    /// Its BR1 gradients.
    gradient: [Vec<ScalarGradient2D>; 2],
    own: Vec<f64>,
    diffusion: DiffusionScratch,
    out: Vec<f64>,
}

impl ViscosityScratch3D {
    /// Buffers for `n_elements` elements of `ops`.
    pub fn new(n_elements: usize, ops: &DGOperators2D) -> Self {
        let (nn, n_total) = (ops.n_nodes, n_elements * ops.n_nodes);
        Self {
            coefficient: vec![0.0; n_total],
            mean: [vec![0.0; n_total], vec![0.0; n_total]],
            shear: [vec![0.0; n_total], vec![0.0; n_total]],
            gradient: [
                vec![ScalarGradient2D::default(); n_total],
                vec![ScalarGradient2D::default(); n_total],
            ],
            own: vec![0.0; nn],
            diffusion: DiffusionScratch::new(nn),
            out: vec![0.0; nn],
        }
    }
}

/// Add the velocity tendency `∇·(νD∇u′_l)/D` of a constant horizontal
/// viscosity `nu` (m²/s) on the vertical shear of `state` to `rhs_u`,
/// `rhs_v` (`[element][node][level]`; see the [module documentation](self)).
/// Columns shallower than `min_column_depth` get nothing.
#[allow(clippy::too_many_arguments)]
pub fn apply_horizontal_viscosity_3d(
    rhs_u: &mut [f64],
    rhs_v: &mut [f64],
    state: &Solution3D,
    nu: f64,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    bathymetry: &Bathymetry2D,
    sigma: &SigmaGrid,
    boundaries: &Boundaries3D,
    min_column_depth: f64,
    scratch: &mut ViscosityScratch3D,
) {
    if nu == 0.0 {
        return;
    }
    let (nn, nl) = (ops.n_nodes, state.n_levels);
    let ViscosityScratch3D {
        coefficient,
        mean,
        shear,
        gradient,
        own,
        diffusion,
        out,
    } = scratch;
    for ((c, &eta), &b) in coefficient
        .iter_mut()
        .zip(&state.eta.data)
        .zip(&bathymetry.data)
    {
        let depth = eta - b;
        *c = if depth < min_column_depth {
            0.0
        } else {
            nu * depth
        };
    }
    for (mean, field) in mean.iter_mut().zip([&state.u, &state.v]) {
        for (m, column) in mean.iter_mut().zip(field.chunks_exact(nl)) {
            *m = sigma.depth_average(column);
        }
    }

    for level in 0..nl {
        // The layer's shear; none in thin columns
        for (idx, &c) in coefficient.iter().enumerate() {
            let [su, sv] = &mut *shear;
            if c == 0.0 {
                su[idx] = 0.0;
                sv[idx] = 0.0;
            } else {
                su[idx] = state.u[idx * nl + level] - mean[0][idx];
                sv[idx] = state.v[idx * nl + level] - mean[1][idx];
            }
        }

        let [su, sv] = &*shear;
        for (component, gradient) in gradient.iter_mut().enumerate() {
            for (k, grad) in gradient.chunks_exact_mut(nn).enumerate() {
                br1_gradient_element(
                    ElementIndex::new(k),
                    mesh,
                    ops,
                    geom,
                    |j, node| shear[component][j.as_usize() * nn + node],
                    |k, face, fi, node, interior| match boundaries.exterior(mesh, k, face) {
                        // Mirrored: no normal shear at the wall
                        FaceExterior::Wall => {
                            let (nx, ny) = geom.normal(k.as_usize(), face, fi);
                            let idx = k.as_usize() * nn + node;
                            let normal = su[idx] * nx + sv[idx] * ny;
                            interior - 2.0 * normal * if component == 0 { nx } else { ny }
                        }
                        _ => interior,
                    },
                    own,
                    grad,
                );
            }
        }

        for (component, rhs) in [&mut *rhs_u, &mut *rhs_v].into_iter().enumerate() {
            let gradient = &gradient[component];
            for k in 0..mesh.n_elements {
                br1_diffusion_element(
                    ElementIndex::new(k),
                    mesh,
                    ops,
                    geom,
                    |j, node| {
                        let idx = j.as_usize() * nn + node;
                        let (c, g) = (coefficient[idx], gradient[idx]);
                        (c * g.dx, c * g.dy)
                    },
                    diffusion,
                    out,
                );
                for (i, &d) in out.iter().enumerate() {
                    let idx = k * nn + i;
                    if coefficient[idx] > 0.0 {
                        rhs[idx * nl + level] += d * nu / coefficient[idx];
                    }
                }
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
        nu: f64,
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
            nu,
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

    /// The column sum `Σ_l Δσ_l D·(tendency)` vanishes at every node (the
    /// depth mean stays the 2D module's), and the layer momentum
    /// `∫ H_z·(tendency)` of every layer is conserved on a periodic mesh,
    /// with a varying free surface and stretched levels.
    #[test]
    fn shear_viscosity_leaves_the_depth_mean_and_conserves_layer_momentum() {
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
        let (rhs_u, rhs_v) = apply(&state, 5.0, &mesh, &ops, &geom, &bathymetry, &sigma);
        let (nn, nl) = (ops.n_nodes, sigma.n_levels());
        let d_sigma = sigma.d_sigma();
        let scale = rhs_u.iter().fold(0.0_f64, |m, r| m.max(r.abs()));
        assert!(scale > 1e-6, "test regime: no tendency ({scale:e})");
        let mut layer_momentum = vec![[0.0; 2]; nl];
        for idx in 0..mesh.n_elements * nn {
            let d = state.eta.data[idx] + depth;
            let mass = geom.node_mass(idx / nn, idx % nn);
            for (rhs, c) in [(&rhs_u, 0), (&rhs_v, 1)] {
                let column = &rhs[idx * nl..(idx + 1) * nl];
                let sum: f64 = column.iter().zip(d_sigma).map(|(r, ds)| r * ds).sum();
                assert!(
                    sum.abs() < 1e-12 * scale,
                    "column sum {sum:e} at node {idx} (scale {scale:e})"
                );
                for (l, (r, ds)) in column.iter().zip(d_sigma).enumerate() {
                    layer_momentum[l][c] += mass * ds * d * r;
                }
            }
        }
        let volume_scale = scale * length * length * depth;
        for (l, m) in layer_momentum.iter().enumerate() {
            assert!(
                m[0].abs().max(m[1].abs()) < 1e-12 * volume_scale,
                "layer {l} gains momentum {m:?} (scale {volume_scale:e})"
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
                let (rhs_u, rhs_v) = apply(&state, nu, &mesh, &ops, &geom, &bathymetry, &sigma);
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
        let (rhs_u, rhs_v) = apply(&state, 10.0, &mesh, &ops, &geom, &bathymetry, &sigma);
        let largest = rhs_u
            .iter()
            .chain(&rhs_v)
            .fold(0.0_f64, |m, r| m.max(r.abs()));
        // Round-off of u − ⟨u⟩ (Σ Δσ_l is 1 to round-off)
        assert!(largest < 1e-14, "tendency {largest:e} without shear");
    }
}
