//! Nodal fields from one mesh and order onto the nodes of another.
//!
//! Two models coupled on different meshes, e.g. the spectral waves on a
//! coarse P1 grid and the circulation on the coastline mesh at P2 (TODO
//! F.4), exchange nodal fields every coupling interval. [`MeshTransfer2D`]
//! fixes, for every node of the target mesh, the source element holding it
//! and the values of that element's nodal basis there
//! ([`DGOperators2D::interpolation_weights`]), so a transfer is one dot
//! product per target node: the source's DG polynomial evaluated at the
//! target node, exact for every field the source represents.
//!
//! A target node on a face of the source mesh, where a DG field has two
//! values, takes the one of the source element under its own element's
//! centre where that element holds it: on the same mesh a transfer is the
//! identity, and a target element inside one source element sees one
//! polynomial.
//!
//! A target node outside the source mesh (the meshes' coastlines differ)
//! takes the value at the nearest point of the source mesh
//! ([`PointLocator2D::nearest`]): a constant extension across the gap.
//!
//! The weights of a source element of order 2 or more are not all
//! positive, so a transfer can overshoot the source's range near a sharp
//! change (a field that must stay non-negative is clamped by its user). At
//! order 1 they are the bilinear weights, positive inside the element.
//!
//! A transfer samples the source at the target's nodes: from a fine mesh
//! onto a coarse one it picks values rather than averaging them, so
//! features smaller than the target's elements alias (an L2 projection
//! would average them; TODO F.4).

use crate::mesh::{Mesh2D, PointLocator2D};
use crate::types::ElementIndex;

use super::DGOperators2D;

/// Interpolation of nodal fields from a source mesh onto the nodes of a
/// target mesh (see the module docs).
#[derive(Clone, Debug)]
pub struct MeshTransfer2D {
    /// Nodes per source element
    source_nodes: usize,
    /// Nodal points of the source (`n_elements × source_nodes`)
    source_points: usize,
    /// Source element of each target node
    elements: Vec<u32>,
    /// Basis values of each target node, `[target node × source_nodes]`
    weights: Vec<f64>,
    /// Target nodes outside the source mesh (ascending), and the largest
    /// distance (m) from one to the nearest point of the source mesh
    outside: Vec<u32>,
    largest_gap: f64,
}

impl MeshTransfer2D {
    /// The transfer from nodal fields on `source` (order of `source_ops`)
    /// onto the nodes of `target` (order of `target_ops`), element by
    /// element as the solution fields are stored.
    pub fn new(
        source: &Mesh2D,
        source_ops: &DGOperators2D,
        target: &Mesh2D,
        target_ops: &DGOperators2D,
    ) -> Self {
        assert!(source.n_elements > 0, "an empty source mesh");
        assert!(
            source.n_elements <= u32::MAX as usize,
            "source elements beyond u32"
        );
        let locator = PointLocator2D::new(source);
        let (sn, tn) = (source_ops.n_nodes, target_ops.n_nodes);
        let n_target = target.n_elements * tn;
        let mut transfer = Self {
            source_nodes: sn,
            source_points: source.n_elements * sn,
            elements: Vec::with_capacity(n_target),
            weights: vec![0.0; n_target * sn],
            outside: Vec::new(),
            largest_gap: 0.0,
        };
        for k in ElementIndex::iter(target.n_elements) {
            // The source element under the target element's centre takes
            // every node of it that it holds, faces included
            let home = locator
                .locate(target.reference_to_physical(k, 0.0, 0.0))
                .map(|c| c.element);
            for i in 0..tn {
                let p =
                    target.reference_to_physical(k, target_ops.nodes_r[i], target_ops.nodes_s[i]);
                let (point, gap) = home
                    .and_then(|h| locator.in_element(h, p))
                    .map(|point| (point, 0.0))
                    .or_else(|| locator.nearest(p))
                    .unwrap_or_else(|| panic!("target node {p:?} is not finite"));
                if gap > 0.0 {
                    transfer.outside.push(transfer.elements.len() as u32);
                    transfer.largest_gap = transfer.largest_gap.max(gap);
                }
                let row = transfer.elements.len();
                source_ops.interpolation_weights_into(
                    point.r,
                    point.s,
                    &mut transfer.weights[row * sn..(row + 1) * sn],
                );
                transfer.elements.push(point.element.as_usize() as u32);
            }
        }
        transfer
    }

    /// Nodes of the target mesh (the length of a transferred field).
    pub fn n_target_points(&self) -> usize {
        self.elements.len()
    }

    /// Nodes of the source mesh (the length of a field to transfer).
    pub fn n_source_points(&self) -> usize {
        self.source_points
    }

    /// Target nodes outside the source mesh, which take the value at the
    /// nearest point of it.
    pub fn n_outside(&self) -> usize {
        self.outside.len()
    }

    /// Whether target node `p` is outside the source mesh.
    pub fn is_outside(&self, p: usize) -> bool {
        self.outside.binary_search(&(p as u32)).is_ok()
    }

    /// The source element whose polynomial target node `p` evaluates.
    pub fn source_element(&self, p: usize) -> usize {
        self.elements[p] as usize
    }

    /// The largest distance from a target node outside the source mesh to
    /// the nearest point of it (0 if every node is inside).
    pub fn largest_gap(&self) -> f64 {
        self.largest_gap
    }

    /// The source field `source` (one value per source node) at the target
    /// node `p`.
    #[inline]
    pub fn value_at(&self, p: usize, source: &[f64]) -> f64 {
        let sn = self.source_nodes;
        let base = self.elements[p] as usize * sn;
        self.weights[p * sn..(p + 1) * sn]
            .iter()
            .zip(&source[base..base + sn])
            .map(|(w, v)| w * v)
            .sum()
    }

    /// The source field `source` at every target node, into `target`.
    pub fn apply_into(&self, source: &[f64], target: &mut [f64]) {
        assert_eq!(
            source.len(),
            self.source_points,
            "one value per source node"
        );
        assert_eq!(
            target.len(),
            self.n_target_points(),
            "one value per target node"
        );
        fill(target, |p| self.value_at(p, source));
    }

    /// The source field `source` at every target node.
    pub fn apply(&self, source: &[f64]) -> Vec<f64> {
        let mut target = vec![0.0; self.n_target_points()];
        self.apply_into(source, &mut target);
        target
    }

    /// A field of `N` components per node (a vector, a symmetric tensor) at
    /// every target node, each component transferred alone.
    pub fn apply_components<const N: usize>(&self, source: &[[f64; N]]) -> Vec<[f64; N]> {
        assert_eq!(
            source.len(),
            self.source_points,
            "one value per source node"
        );
        let sn = self.source_nodes;
        let mut target = vec![[0.0; N]; self.n_target_points()];
        fill(&mut target, |p| {
            let base = self.elements[p] as usize * sn;
            let mut out = [0.0; N];
            for (&w, v) in self.weights[p * sn..(p + 1) * sn]
                .iter()
                .zip(&source[base..base + sn])
            {
                for (o, x) in out.iter_mut().zip(v) {
                    *o += w * x;
                }
            }
            out
        });
        target
    }
}

/// `target[p] = value(p)` for every `p`, in parallel blocks with the
/// `parallel` feature.
fn fill<T: Send>(target: &mut [T], value: impl Fn(usize) -> T + Sync) {
    const BLOCK: usize = 4096;
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        target
            .par_chunks_mut(BLOCK)
            .enumerate()
            .for_each(|(b, chunk)| {
                for (i, t) in chunk.iter_mut().enumerate() {
                    *t = value(b * BLOCK + i);
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    for (p, t) in target.iter_mut().enumerate() {
        *t = value(p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth field of total degree ≤ 2.
    fn quadratic(x: f64, y: f64) -> f64 {
        1.5 - 0.7 * x + 0.3 * y + 0.2 * x * x - 0.45 * x * y + 0.1 * y * y
    }

    fn nodal(mesh: &Mesh2D, ops: &DGOperators2D, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
        ElementIndex::iter(mesh.n_elements)
            .flat_map(|k| {
                (0..ops.n_nodes).map(move |i| {
                    let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                    (x, y)
                })
            })
            .map(|(x, y)| f(x, y))
            .collect()
    }

    /// A rectangle of parallelograms (affine elements, on which a quadratic is
    /// exactly a P2 field), sheared.
    fn sheared(nx: usize, ny: usize, x1: f64, y1: f64) -> Mesh2D {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, x1, 0.0, y1, nx, ny);
        for v in &mut mesh.vertices {
            v[0] += 0.3 * v[1];
        }
        mesh
    }

    /// A P2 source of a quadratic is that quadratic at every node of a target
    /// of another resolution and order, inside the source.
    #[test]
    fn a_field_the_source_represents_is_transferred_exactly() {
        let source = sheared(5, 4, 10.0, 8.0);
        let source_ops = DGOperators2D::new(2);
        // The target: finer, P3, inside the source
        let mut target = Mesh2D::uniform_rectangle(1.0, 9.0, 0.5, 7.5, 7, 9);
        for v in &mut target.vertices {
            v[0] += 0.3 * v[1] + 0.05 * (3.0 * v[1]).sin();
        }
        let target_ops = DGOperators2D::new(3);
        let transfer = MeshTransfer2D::new(&source, &source_ops, &target, &target_ops);
        assert_eq!(transfer.n_outside(), 0);
        assert_eq!(transfer.n_target_points(), 7 * 9 * 16);
        let got = transfer.apply(&nodal(&source, &source_ops, quadratic));
        let expected = nodal(&target, &target_ops, quadratic);
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-12, "{g} against {e}");
        }
        // Vectors and tensors component by component
        let pairs: Vec<[f64; 2]> = nodal(&source, &source_ops, quadratic)
            .iter()
            .map(|&q| [q, -2.0 * q])
            .collect();
        for (g, e) in transfer.apply_components(&pairs).iter().zip(&expected) {
            assert!((g[0] - e).abs() < 1e-12 && (g[1] + 2.0 * e).abs() < 1e-12);
        }
    }

    /// On the same mesh and order a transfer is the identity, also for a
    /// field that jumps at every face (the face nodes keep their own
    /// element's values, not the neighbour's).
    #[test]
    fn the_same_mesh_transfers_a_discontinuous_field_unchanged() {
        let mesh = sheared(6, 5, 12.0, 10.0);
        let ops = DGOperators2D::new(2);
        let transfer = MeshTransfer2D::new(&mesh, &ops, &mesh, &ops);
        let field: Vec<f64> = (0..mesh.n_elements * ops.n_nodes)
            .map(|p| (p / ops.n_nodes) as f64 * 10.0 + (p % ops.n_nodes) as f64)
            .collect();
        let got = transfer.apply(&field);
        for (g, f) in got.iter().zip(&field) {
            assert!((g - f).abs() < 1e-12, "{g} against {f}");
        }
    }

    /// Interpolation from a P1 mesh onto a P2 one and back keeps a bilinear
    /// field on matching rectangles, and a smooth field converges at second
    /// order from P1 sources.
    #[test]
    fn a_smooth_field_converges_at_the_sources_order() {
        let f = |x: f64, y: f64| (0.7 * x).sin() * (0.4 * y).cos();
        let target = Mesh2D::uniform_rectangle(0.0, 8.0, 0.0, 6.0, 11, 7);
        let target_ops = DGOperators2D::new(2);
        let expected = nodal(&target, &target_ops, f);
        let mut errors = Vec::new();
        for n in [4, 8, 16] {
            let source = Mesh2D::uniform_rectangle(0.0, 8.0, 0.0, 6.0, n, n);
            let ops = DGOperators2D::new(1);
            let transfer = MeshTransfer2D::new(&source, &ops, &target, &target_ops);
            let got = transfer.apply(&nodal(&source, &ops, f));
            errors.push(
                got.iter()
                    .zip(&expected)
                    .map(|(g, e)| (g - e).abs())
                    .fold(0.0, f64::max),
            );
        }
        let rates: Vec<f64> = errors.windows(2).map(|w| (w[0] / w[1]).log2()).collect();
        assert!(
            rates.iter().all(|&r| r > 1.8),
            "{errors:?}, rates {rates:?}"
        );
    }

    /// Target nodes outside the source take the value at the nearest point of
    /// the source mesh: a strip beyond its right end takes its right edge's
    /// values, a hole in the source its nearest face's.
    #[test]
    fn nodes_outside_the_source_take_the_nearest_value() {
        let full = Mesh2D::uniform_rectangle(0.0, 4.0, 0.0, 4.0, 4, 4);
        // Element (1, 1) removed: a hole over [1, 2] × [1, 2]
        let (source, _) = full.retain_elements(
            |k| {
                let [x, y] = full.reference_to_physical(k, 0.0, 0.0);
                !(1.0..2.0).contains(&x) || !(1.0..2.0).contains(&y)
            },
            crate::mesh::BoundaryTag::Wall,
        );
        let ops = DGOperators2D::new(1);
        let f = |x: f64, y: f64| 2.0 + x + 0.5 * y + 0.25 * x * y;
        let field = nodal(&source, &ops, f);
        // The target reaches 1 beyond the right end
        let target = Mesh2D::uniform_rectangle(0.0, 5.0, 0.0, 4.0, 12, 7);
        let target_ops = DGOperators2D::new(1);
        let transfer = MeshTransfer2D::new(&source, &ops, &target, &target_ops);
        assert!(transfer.n_outside() > 0);
        assert!((transfer.largest_gap() - 1.0).abs() < 1e-12);
        let got = transfer.apply(&field);
        let xs = nodal(&target, &target_ops, |x, _| x);
        let ys = nodal(&target, &target_ops, |_, y| y);
        let in_hole = |x: f64, y: f64| x > 1.0 && x < 2.0 && y > 1.0 && y < 2.0;
        for (p, (&x, &y)) in xs.iter().zip(&ys).enumerate() {
            assert_eq!(
                transfer.is_outside(p),
                x > 4.0 || in_hole(x, y),
                "({x}, {y})"
            );
        }
        let mut inside_hole = 0;
        for (p, (&x, &y)) in xs.iter().zip(&ys).enumerate() {
            let expected = if x > 4.0 {
                f(4.0, y)
            } else if in_hole(x, y) {
                inside_hole += 1;
                // The nearest face of the hole
                let d = [x - 1.0, 2.0 - x, y - 1.0, 2.0 - y];
                let m = d.iter().cloned().fold(f64::INFINITY, f64::min);
                match d.iter().position(|&v| v == m).unwrap() {
                    0 => f(1.0, y),
                    1 => f(2.0, y),
                    2 => f(x, 1.0),
                    _ => f(x, 2.0),
                }
            } else {
                f(x, y)
            };
            assert!(
                (got[p] - expected).abs() < 1e-12,
                "({x}, {y}): {} against {expected}",
                got[p]
            );
        }
        assert!(inside_hole > 0);
    }
}
