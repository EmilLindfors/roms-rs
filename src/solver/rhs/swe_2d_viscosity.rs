//! Horizontal eddy viscosity of the 2D SWE momentum, `∇·(νh∇u)` and
//! `∇·(νh∇v)`, discretised with BR1 (Bassi & Rebay 1997; Hesthaven &
//! Warburton 2008, §7.2).
//!
//! Two passes, both element by element:
//! 1. [`ViscousTerm::gradients`]: the BR1 gradients of u = hu/h and v = hv/h
//!    of an element (central face values; the boundary condition's ghost
//!    velocity on boundary faces), into a [`ViscousWorkspace`].
//! 2. [`ViscousTerm::add`], in the RHS kernel's element loop: the divergence
//!    of νh∇u from the element's own and its face neighbours' gradients, with
//!    the central face flux.
//!
//! So the term of element k reads the state beyond its face neighbours j:
//! j's gradient at the shared face nodes reads j's neighbours. With the
//! diagonal GLL LIFT (a face's lift touches only its own nodes) it reads them
//! only at the corner nodes, i.e. the elements across the two faces of j that
//! meet the shared face. The whole-mesh RHS runs pass 1 over every element
//! first; the element-subset RHS of local time stepping runs it over the
//! listed elements and their face neighbours, each at its own time, and the
//! multirate stepper follows the wider stencil
//! ([`RhsStencil::FacesAndCorners`](crate::time::RhsStencil::FacesAndCorners)).
//!
//! The face flux is the same on both sides of every interior face, so the
//! viscous term conserves momentum; it adds nothing to the mass equation.

use crate::boundary::SWEBoundaryCondition2D;
use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::SWESolution2D;
use crate::solver::core::blocks::{Pooled, for_each_block};
use crate::source::swe_2d::viscosity::HorizontalViscosity2D;
use crate::types::ElementIndex;

use super::diffusion_2d::{
    DiffusionScratch, ScalarGradient2D, br1_diffusion_element, br1_gradient_element,
};
use super::swe_2d::{SWE2DRhsConfig, boundary_state};

/// Per-element scratch of the viscous term (part of the RHS kernel's
/// per-thread element workspace).
pub(super) struct ViscousScratch {
    own: Vec<f64>,
    diffusion: DiffusionScratch,
    diff: [Vec<f64>; 2],
}

impl ViscousScratch {
    pub(super) fn new(n_nodes: usize) -> Self {
        Self {
            own: vec![0.0; n_nodes],
            diffusion: DiffusionScratch::new(n_nodes),
            diff: [vec![0.0; n_nodes], vec![0.0; n_nodes]],
        }
    }
}

/// Velocity gradients of every mesh node (valid where pass 1 ran), and the
/// element-neighbourhood scratch of the subset RHS. Reused between RHS
/// evaluations.
#[derive(Default)]
pub(super) struct ViscousWorkspace {
    pub(super) grad_u: Vec<ScalarGradient2D>,
    pub(super) grad_v: Vec<ScalarGradient2D>,
    /// The listed elements and their face neighbours, distinct
    pub(super) neighbourhood: Vec<u32>,
    stamp: Vec<u32>,
    pass: u32,
}

impl ViscousWorkspace {
    /// Gradient storage for `n_total` nodes.
    pub(super) fn resize(&mut self, n_total: usize) {
        self.grad_u.resize(n_total, ScalarGradient2D::default());
        self.grad_v.resize(n_total, ScalarGradient2D::default());
    }

    /// `elements` and their face neighbours into `self.neighbourhood`, each
    /// once, without sorting (an element is taken when its stamp is not the
    /// current pass).
    pub(super) fn collect_neighbourhood(&mut self, mesh: &Mesh2D, elements: &[u32]) {
        if self.stamp.len() != mesh.n_elements || self.pass == u32::MAX {
            self.stamp.clear();
            self.stamp.resize(mesh.n_elements, 0);
            self.pass = 0;
        }
        self.pass += 1;
        self.neighbourhood.clear();
        let (stamp, pass, list) = (&mut self.stamp, self.pass, &mut self.neighbourhood);
        let mut take = |k: usize| {
            if stamp[k] != pass {
                stamp[k] = pass;
                list.push(k as u32);
            }
        };
        for &k in elements {
            take(k as usize);
            let k_idx = ElementIndex::new(k as usize);
            for face in 0..4 {
                if let Some(nb) = mesh.neighbor(k_idx, face) {
                    take(nb.element);
                }
            }
        }
    }
}

/// The viscous term of one RHS evaluation (see the [module documentation](self)).
pub(super) struct ViscousTerm<'a, 'c, BC: SWEBoundaryCondition2D> {
    q: &'a SWESolution2D,
    mesh: &'a Mesh2D,
    ops: &'a DGOperators2D,
    geom: &'a GeometricFactors2D,
    config: &'a SWE2DRhsConfig<'c, BC>,
    visc: &'c HorizontalViscosity2D,
}

impl<'a, 'c, BC: SWEBoundaryCondition2D> ViscousTerm<'a, 'c, BC> {
    /// `None` when `config` has no viscosity.
    pub(super) fn new(
        q: &'a SWESolution2D,
        mesh: &'a Mesh2D,
        ops: &'a DGOperators2D,
        geom: &'a GeometricFactors2D,
        config: &'a SWE2DRhsConfig<'c, BC>,
    ) -> Option<Self> {
        let visc = config.viscosity?;
        Some(Self {
            q,
            mesh,
            ops,
            geom,
            config,
            visc,
        })
    }

    /// Velocity at node `i` of element `k`; zero where the depth is at or
    /// below the viscosity's `h_min`.
    #[inline]
    fn velocity(&self, k: ElementIndex, i: usize) -> (f64, f64) {
        node_velocity(self.q, self.visc, k, i)
    }

    /// Velocity component of the boundary condition's exterior state at
    /// boundary node `node` (face node `fi` of `face`) of element `k`, at
    /// `time`.
    #[allow(clippy::too_many_arguments)]
    fn boundary_velocity(
        &self,
        component: usize,
        time: f64,
        k: ElementIndex,
        face: usize,
        fi: usize,
        node: usize,
    ) -> f64 {
        let normal = self.geom.normal(k.as_usize(), face, fi);
        let ghost = boundary_state(
            self.q,
            self.mesh,
            self.ops,
            self.config,
            time,
            k,
            face,
            node,
            normal,
        )
        .state();
        let h_min = self.visc.h_min;
        if ghost.h <= h_min {
            return 0.0;
        }
        let h_safe = ghost.h.max(h_min);
        match component {
            0 => ghost.hu / h_safe,
            1 => ghost.hv / h_safe,
            _ => unreachable!("invalid velocity component"),
        }
    }

    /// Pass 1: the BR1 gradients of u and v of element `k` into its rows
    /// `grad_u`, `grad_v` (`n_nodes` each), with the boundary condition
    /// evaluated at `time`.
    pub(super) fn gradients(
        &self,
        k: usize,
        time: f64,
        scratch: &mut ViscousScratch,
        grad_u: &mut [ScalarGradient2D],
        grad_v: &mut [ScalarGradient2D],
    ) {
        let k = ElementIndex::new(k);
        for (component, out) in [grad_u, grad_v].into_iter().enumerate() {
            br1_gradient_element(
                k,
                self.mesh,
                self.ops,
                self.geom,
                |j, node| {
                    let (u, v) = self.velocity(j, node);
                    if component == 0 { u } else { v }
                },
                |k, face, fi, node, _interior| {
                    self.boundary_velocity(component, time, k, face, fi, node)
                },
                &mut scratch.own,
                out,
            );
        }
    }

    /// Pass 2: `∇·(νh∇u)`, `∇·(νh∇v)` of element `k` added to its rows `hu`,
    /// `hv`, from the gradients `grad` of pass 1 (whole-mesh arrays, valid for
    /// `k` and its face neighbours).
    pub(super) fn add(
        &self,
        k: usize,
        grad: [&[ScalarGradient2D]; 2],
        scratch: &mut ViscousScratch,
        hu: &mut [f64],
        hv: &mut [f64],
    ) {
        let n_nodes = self.ops.n_nodes;
        // νh at a node; zero where (nearly) dry
        let coefficient = |j: ElementIndex, node: usize| {
            let flat = j.as_usize() * n_nodes + node;
            let h = self.q.get_state(j, node).h;
            if h <= self.visc.h_min {
                return 0.0;
            }
            let delta =
                HorizontalViscosity2D::filter_width(self.geom.area[j.as_usize()], self.ops.order);
            let (gu, gv) = (grad[0][flat], grad[1][flat]);
            self.visc
                .compute_viscosity(gu.dx, gu.dy, gv.dx, gv.dy, delta)
                * h
        };
        let k = ElementIndex::new(k);
        let ViscousScratch {
            diffusion, diff, ..
        } = scratch;
        for (component, diff) in diff.iter_mut().enumerate() {
            let gradient = grad[component];
            br1_diffusion_element(
                k,
                self.mesh,
                self.ops,
                self.geom,
                |j, node| {
                    let coeff = coefficient(j, node).max(0.0);
                    let g = gradient[j.as_usize() * n_nodes + node];
                    (coeff * g.dx, coeff * g.dy)
                },
                diffusion,
                diff,
            );
        }
        for (row, diff) in [hu, hv].into_iter().zip(diff.iter()) {
            for (r, d) in row.iter_mut().zip(diff) {
                *r += d;
            }
        }
    }
}

/// Velocity at node `i` of element `k`; zero where the depth is at or below
/// the viscosity's `h_min`.
#[inline]
fn node_velocity(
    q: &SWESolution2D,
    visc: &HorizontalViscosity2D,
    k: ElementIndex,
    i: usize,
) -> (f64, f64) {
    let state = q.get_state(k, i);
    if state.h > visc.h_min {
        let h_safe = state.h.max(visc.h_min);
        (state.hu / h_safe, state.hv / h_safe)
    } else {
        (0.0, 0.0)
    }
}

/// The largest viscosity ν (m²/s) of every element's nodes into `largest`
/// (`[element]`), for the time step: the background, plus Smagorinsky's from
/// the strain of `q`'s BR1 velocity gradients at the nodes deeper than
/// `h_min` (the others carry no viscous flux).
///
/// The gradients are the RHS's except on boundary faces, which take the
/// interior value (no jump): the boundary condition's exterior state needs a
/// time, and a time step is chosen before the stages' times are known. In
/// parallel with the `parallel` feature, the same result either way.
pub fn largest_viscosity_swe_2d(
    largest: &mut [f64],
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    visc: &HorizontalViscosity2D,
) {
    assert_eq!(largest.len(), mesh.n_elements, "one value per element");
    if !visc.is_strain_dependent() {
        largest.fill(visc.background);
        return;
    }
    struct Scratch {
        own: Vec<f64>,
        gradient: [Vec<ScalarGradient2D>; 2],
    }
    let nn = ops.n_nodes;
    for_each_block(
        mesh.n_elements,
        [largest],
        || {
            Pooled::take(
                |s: &Scratch| s.own.len() == nn,
                || Scratch {
                    own: vec![0.0; nn],
                    gradient: std::array::from_fn(|_| vec![ScalarGradient2D::default(); nn]),
                },
            )
        },
        |scratch, k, [out]| {
            let k = ElementIndex::new(k);
            let Scratch { own, gradient } = &mut **scratch;
            for (component, out) in gradient.iter_mut().enumerate() {
                br1_gradient_element(
                    k,
                    mesh,
                    ops,
                    geom,
                    |j, node| {
                        let (u, v) = node_velocity(q, visc, j, node);
                        if component == 0 { u } else { v }
                    },
                    |_, _, _, _, interior| interior,
                    own,
                    out,
                );
            }
            let delta = HorizontalViscosity2D::filter_width(geom.area[k.as_usize()], ops.order);
            out[0] = (0..nn)
                .filter(|&i| q.get_state(k, i).h > visc.h_min)
                .map(|i| {
                    let (gu, gv) = (gradient[0][i], gradient[1][i]);
                    visc.compute_viscosity(gu.dx, gu.dy, gv.dx, gv.dy, delta)
                })
                .fold(visc.background, f64::max);
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::swe_2d::viscosity::strain_rate_magnitude;

    /// On a linear shear `u = a y + b x`, `v = c x` the BR1 gradients are
    /// exact (continuous across faces, and boundary faces have no jump), so
    /// the largest ν is the background plus `(C_s Δ)²|S|` with the constant
    /// strain, on every element and order. Dry nodes carry none.
    #[test]
    fn largest_viscosity_is_exact_on_a_linear_shear() {
        let (a, b, c) = (0.3, -0.1, 0.05);
        let visc = HorizontalViscosity2D::smagorinsky(0.4).with_background(0.25);
        let strain = strain_rate_magnitude(b, a, c, 0.0);
        for order in 1..=3 {
            let mesh = Mesh2D::uniform_rectangle(0.0, 300.0, 0.0, 200.0, 3, 2);
            let ops = DGOperators2D::new(order);
            let geom = GeometricFactors2D::compute(&mesh, &ops);
            let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
            q.set_from_functions(&mesh, &ops, |_, _| 2.0, |x, y| a * y + b * x, |x, _| c * x);
            let mut largest = vec![0.0; mesh.n_elements];
            largest_viscosity_swe_2d(&mut largest, &q, &mesh, &ops, &geom, &visc);
            // 100 m × 100 m elements: Δ = 100/N
            let delta = 100.0 / order as f64;
            let expected = 0.25 + (0.4 * delta).powi(2) * strain;
            for (k, nu) in largest.iter().enumerate() {
                assert!(
                    (nu - expected).abs() < 1e-11 * expected,
                    "P{order}, element {k}: {nu} against {expected}"
                );
            }

            // A dry element has the background only
            for var in 0..3 {
                q.data[var][..ops.n_nodes].fill(0.0);
            }
            largest_viscosity_swe_2d(&mut largest, &q, &mesh, &ops, &geom, &visc);
            assert_eq!(largest[0], 0.25, "P{order}: dry element");
        }
    }

    /// Without Smagorinsky's part the largest ν is the background, with no
    /// gradient pass.
    #[test]
    fn largest_viscosity_of_a_constant_is_the_constant() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        let mut largest = vec![0.0; mesh.n_elements];
        let visc = HorizontalViscosity2D::constant(3.0);
        largest_viscosity_swe_2d(&mut largest, &q, &mesh, &ops, &geom, &visc);
        assert_eq!(largest, vec![3.0; 4]);
    }
}
