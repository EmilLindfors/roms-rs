//! Drag of fish-farm net cages as a porous momentum sink (2D form).
//!
//! A net cage is a porous region over its footprint, down to the bottom of
//! the net. Per unit volume the netting removes momentum at
//!
//! ```text
//! S = −½ C_d a |u| u,
//! ```
//!
//! with C_d the drag coefficient of the netting per net area and a the net
//! area per unit cage volume (review in Klebert et al. 2013, Ocean Eng. 58).
//! C_d depends on the solidity S_n of the netting (Løland 1991):
//!
//! ```text
//! C_d = 0.04 + (−0.04 + 0.33 S_n + 6.54 S_n² − 4.88 S_n³) cos θ,
//! ```
//!
//! θ the angle between the flow and the net normal. Fouling raises S_n, so
//! it is a per-cage parameter.
//!
//! # Depth-integrated form
//!
//! Integrating S over the net depth d (only the part of the column the net
//! reaches) gives the sink of the depth-integrated momentum,
//!
//! ```text
//! ∂(hu)/∂t = −½ C_d a min(d, h) |u| u = −Λ hu,   Λ = ½ C_d a |u| min(d, h)/h,
//! ```
//!
//! a linear damping like the bottom friction ([`super::BottomFriction2D`]),
//! applied point-implicitly in every RK stage
//! (`SWEPhysics2DBuilder::with_cage_drag`): Λ is large in shallow water and
//! must not limit Δt. It only touches the momentum, so mass conservation and
//! the lake at rest are unchanged.
//!
//! # Cages smaller than an element
//!
//! A cage (40–60 m across) is comparable to or smaller than an element. A
//! node is not switched on or off: its Λ is weighted by the share φᵢ of its
//! area inside the footprint. The area of GLL node i is its subcell in
//! reference space, `[ξᵢ, ξᵢ₊₁]²` with `ξᵢ₊₁ − ξᵢ = wᵢ` (the subcells of the
//! `WetDry` formulation), mapped to the element, and
//!
//! ```text
//! φᵢ = |subcellᵢ ∩ footprint| / (wᵢ Jᵢ),
//! ```
//!
//! so that `Σᵢ wᵢ Jᵢ φᵢ` is exactly the footprint area and the integrated drag
//! in a uniform current is that of the cage, wherever the cage sits in the
//! mesh. On a parallelogram wᵢJᵢ is the subcell area and φᵢ ∈ [0, 1]; on a
//! general quadrilateral φᵢ can exceed 1 by the (small) difference.
//!
//! The intersection area comes from adaptive bisection on the footprint's
//! signed distance: a rectangle whose image lies wholly inside or outside
//! counts with its exact area (J of the bilinear map is linear in r and s, so
//! the midpoint rule is exact), and only rectangles the footprint boundary
//! crosses are split, until they are smaller than 10⁻³ of the footprint,
//! where 4 × 4 samples decide.

use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, bilinear_jacobian};
use crate::types::ElementIndex;

/// Løland (1991) drag coefficient of a net panel per net area, at solidity
/// `solidity` ∈ [0, 1) and angle `angle` (rad) between the flow and the panel
/// normal.
pub fn loland_drag_coefficient(solidity: f64, angle: f64) -> f64 {
    let s = solidity;
    0.04 + (-0.04 + 0.33 * s + 6.54 * s * s - 4.88 * s * s * s) * angle.cos()
}

/// Horizontal footprint of a net cage.
#[derive(Clone, Debug, PartialEq)]
pub enum CageFootprint {
    /// Circular cage (the common plastic ring cage)
    Circle {
        /// Centre (m)
        center: [f64; 2],
        /// Radius (m)
        radius: f64,
    },
    /// Simple polygon (e.g. a square steel cage), vertices in either order
    Polygon(Vec<[f64; 2]>),
}

impl CageFootprint {
    /// Whether the point `p` lies inside the footprint.
    pub fn contains(&self, [x, y]: [f64; 2]) -> bool {
        match self {
            Self::Circle { center, radius } => {
                let (dx, dy) = (x - center[0], y - center[1]);
                dx * dx + dy * dy <= radius * radius
            }
            Self::Polygon(vertices) => {
                // Even-odd rule
                let mut inside = false;
                let mut j = vertices.len() - 1;
                for (i, &[xi, yi]) in vertices.iter().enumerate() {
                    let [xj, yj] = vertices[j];
                    if (yi > y) != (yj > y) && x < xi + (y - yi) * (xj - xi) / (yj - yi) {
                        inside = !inside;
                    }
                    j = i;
                }
                inside
            }
        }
    }

    /// Signed distance from `p` to the footprint boundary, negative inside.
    pub fn signed_distance(&self, p: [f64; 2]) -> f64 {
        match self {
            Self::Circle { center, radius } => (p[0] - center[0]).hypot(p[1] - center[1]) - radius,
            Self::Polygon(vertices) => {
                let n = vertices.len();
                let distance = (0..n)
                    .map(|i| segment_distance(p, vertices[i], vertices[(i + 1) % n]))
                    .fold(f64::INFINITY, f64::min);
                if self.contains(p) {
                    -distance
                } else {
                    distance
                }
            }
        }
    }

    /// Axis-aligned bounding box `(min, max)`.
    pub fn bounding_box(&self) -> ([f64; 2], [f64; 2]) {
        match self {
            Self::Circle { center, radius } => (
                [center[0] - radius, center[1] - radius],
                [center[0] + radius, center[1] + radius],
            ),
            Self::Polygon(vertices) => vertices.iter().fold(
                ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
                |(lo, hi), &[x, y]| ([lo[0].min(x), lo[1].min(y)], [hi[0].max(x), hi[1].max(y)]),
            ),
        }
    }

    /// Area (m²).
    pub fn area(&self) -> f64 {
        match self {
            Self::Circle { radius, .. } => std::f64::consts::PI * radius * radius,
            Self::Polygon(vertices) => {
                let n = vertices.len();
                let twice: f64 = (0..n)
                    .map(|i| {
                        let ([x0, y0], [x1, y1]) = (vertices[i], vertices[(i + 1) % n]);
                        x0 * y1 - x1 * y0
                    })
                    .sum();
                0.5 * twice.abs()
            }
        }
    }
}

/// One net cage: footprint, net depth and the netting's drag per length.
#[derive(Clone, Debug, PartialEq)]
pub struct NetCage {
    /// Horizontal footprint
    pub footprint: CageFootprint,
    /// Depth of the net bottom below the surface (m)
    pub net_depth: f64,
    /// C_d·a: drag coefficient times net area per cage volume (1/m)
    pub drag_per_length: f64,
}

impl NetCage {
    /// A cage with the given footprint, net depth (m) and C_d·a (1/m).
    ///
    /// # Panics
    /// If `net_depth` or `drag_per_length` is negative or not finite, or a
    /// polygon has fewer than three vertices.
    pub fn new(footprint: CageFootprint, net_depth: f64, drag_per_length: f64) -> Self {
        assert!(
            net_depth.is_finite() && net_depth >= 0.0,
            "net depth must be finite and non-negative, got {net_depth}"
        );
        assert!(
            drag_per_length.is_finite() && drag_per_length >= 0.0,
            "C_d·a must be finite and non-negative, got {drag_per_length}"
        );
        match &footprint {
            CageFootprint::Circle { radius, .. } => assert!(
                radius.is_finite() && *radius > 0.0,
                "cage radius must be positive, got {radius}"
            ),
            CageFootprint::Polygon(vertices) => assert!(
                vertices.len() >= 3,
                "a cage polygon needs at least three vertices"
            ),
        }
        Self {
            footprint,
            net_depth,
            drag_per_length,
        }
    }

    /// Circular cage of radius `radius` (m) and net depth `net_depth` (m)
    /// with netting of solidity `solidity`.
    ///
    /// The current crosses the net twice, each time through a projected area
    /// 2R·d, so a = 2·2R d/(πR² d) = 4/(πR); C_d is Løland's at normal
    /// incidence. This counts the rear net at the full cage-mean velocity; the
    /// velocity reduction inside the cage comes from the model's own flow.
    pub fn circular(center: [f64; 2], radius: f64, net_depth: f64, solidity: f64) -> Self {
        let a = 4.0 / (std::f64::consts::PI * radius);
        Self::new(
            CageFootprint::Circle { center, radius },
            net_depth,
            loland_drag_coefficient(solidity, 0.0) * a,
        )
    }
}

/// Cage drag at one node, in the global node numbering of
/// [`crate::solver::SWESolution2D`] (`k·n_nodes + i`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CageNode {
    /// Global node index
    pub node: usize,
    /// φ·½C_d·a (1/m), φ the node's share of the footprint (module docs)
    pub coefficient: f64,
    /// Net depth of the cage (m)
    pub net_depth: f64,
}

/// Cage drag on a mesh: the nodes whose quadrature area overlaps a cage, with
/// their weighted drag coefficients (see the module docs). Sparse, sorted by
/// node; a node inside two overlapping cages appears twice.
#[derive(Clone, Debug, Default)]
pub struct CageDrag2D {
    nodes: Vec<CageNode>,
    n_total: usize,
}

/// Boundary rectangles are split until their reach is below this fraction of
/// the footprint's smaller bounding-box side
const RELATIVE_TOLERANCE: f64 = 1e-3;
/// At most this many bisections below a GLL subcell
const MAX_LEVEL: u32 = 20;
/// Samples per direction in a boundary leaf
const LEAF_SAMPLES: usize = 4;

impl CageDrag2D {
    /// Weight the drag of `cages` onto the nodes of `mesh` at the order of
    /// `ops`.
    pub fn new(mesh: &Mesh2D, ops: &DGOperators2D, cages: &[NetCage]) -> Self {
        let n_1d = ops.n_1d;
        // Subcell interfaces ξ₀ = −1, ξᵢ₊₁ = ξᵢ + wᵢ
        let mut xi = vec![-1.0; n_1d + 1];
        for i in 0..n_1d {
            xi[i + 1] = xi[i] + ops.weights_1d[i];
        }
        let boxes: Vec<_> = cages.iter().map(|c| c.footprint.bounding_box()).collect();

        let mut nodes = Vec::new();
        for k in ElementIndex::iter(mesh.n_elements) {
            let verts = mesh.element_vertices(k);
            let (lo, hi) = verts.iter().fold(
                ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
                |(lo, hi), &[x, y]| ([lo[0].min(x), lo[1].min(y)], [hi[0].max(x), hi[1].max(y)]),
            );
            for (cage, (clo, chi)) in cages.iter().zip(&boxes) {
                let overlaps =
                    lo[0] <= chi[0] && clo[0] <= hi[0] && lo[1] <= chi[1] && clo[1] <= hi[1];
                if !overlaps || cage.drag_per_length == 0.0 {
                    continue;
                }
                for j in 0..n_1d {
                    for i in 0..n_1d {
                        let inside = footprint_area_in(
                            &verts,
                            [xi[i], xi[i + 1], xi[j], xi[j + 1]],
                            &cage.footprint,
                        );
                        let node_area = ops.weights_1d[i]
                            * ops.weights_1d[j]
                            * bilinear_jacobian(&verts, ops.nodes_1d[i], ops.nodes_1d[j]);
                        let phi = inside / node_area;
                        if phi > 0.0 {
                            nodes.push(CageNode {
                                node: k.as_usize() * ops.n_nodes + j * n_1d + i,
                                coefficient: phi * 0.5 * cage.drag_per_length,
                                net_depth: cage.net_depth,
                            });
                        }
                    }
                }
            }
        }
        // Stable: overlapping cages keep their input order at a node
        nodes.sort_by_key(|n| n.node);
        Self {
            nodes,
            n_total: mesh.n_elements * ops.n_nodes,
        }
    }

    /// The nodes with cage drag, sorted by node.
    pub fn nodes(&self) -> &[CageNode] {
        &self.nodes
    }

    /// Number of nodes of the mesh this was built for (`K·n_nodes`).
    pub fn n_total_nodes(&self) -> usize {
        self.n_total
    }

    /// Whether no node carries cage drag.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Index of the first entry at node `node` or after.
    pub(crate) fn first_at_or_after(&self, node: usize) -> usize {
        self.nodes.partition_point(|n| n.node < node)
    }

    /// Damping rate Λ = Σ φ·½C_d·a·|u|·min(d, h)/h (1/s) of the entries of one
    /// node, at depth `h > 0` and speed `speed`.
    #[inline]
    pub fn damping_rate(entries: &[CageNode], h: f64, speed: f64) -> f64 {
        entries
            .iter()
            .map(|n| n.coefficient * n.net_depth.min(h) / h)
            .sum::<f64>()
            * speed
    }
}

/// Distance from `p` to the segment `[a, b]`.
fn segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dy)
}

/// Bilinear map of the quadrilateral `verts` at `(r, s)`.
fn map(verts: &[[f64; 2]; 4], r: f64, s: f64) -> [f64; 2] {
    let n = [
        0.25 * (1.0 - r) * (1.0 - s),
        0.25 * (1.0 + r) * (1.0 - s),
        0.25 * (1.0 + r) * (1.0 + s),
        0.25 * (1.0 - r) * (1.0 + s),
    ];
    let x = (0..4).map(|v| n[v] * verts[v][0]).sum();
    let y = (0..4).map(|v| n[v] * verts[v][1]).sum();
    [x, y]
}

/// Area inside `footprint` of the image of the reference rectangle
/// `[r0, r1] × [s0, s1]` under the bilinear map of `verts`.
fn footprint_area_in(verts: &[[f64; 2]; 4], rect: [f64; 4], footprint: &CageFootprint) -> f64 {
    let (lo, hi) = footprint.bounding_box();
    let tolerance = RELATIVE_TOLERANCE * (hi[0] - lo[0]).min(hi[1] - lo[1]);
    inside_area(verts, rect, footprint, tolerance, MAX_LEVEL)
}

/// Area inside `footprint` of the image of `[r0, r1] × [s0, s1]`.
fn inside_area(
    verts: &[[f64; 2]; 4],
    [r0, r1, s0, s1]: [f64; 4],
    footprint: &CageFootprint,
    tolerance: f64,
    level: u32,
) -> f64 {
    let (rc, sc) = (0.5 * (r0 + r1), 0.5 * (s0 + s1));
    let center = map(verts, rc, sc);
    let distance = footprint.signed_distance(center);
    // The image lies in the convex hull of its mapped corners, so within the
    // largest corner distance of the mapped centre
    let reach = [(r0, s0), (r1, s0), (r1, s1), (r0, s1)]
        .into_iter()
        .map(|(r, s)| {
            let [x, y] = map(verts, r, s);
            (x - center[0]).hypot(y - center[1])
        })
        .fold(0.0, f64::max);
    // J is linear in (r, s): the midpoint rule gives the exact area
    let area = bilinear_jacobian(verts, rc, sc) * (r1 - r0) * (s1 - s0);
    if distance >= reach {
        0.0
    } else if distance <= -reach {
        area
    } else if level == 0 || reach < tolerance {
        // Boundary leaf: midpoint samples, weighted by J
        let (mut inside, mut total) = (0.0, 0.0);
        for b in 0..LEAF_SAMPLES {
            let s = s0 + (b as f64 + 0.5) / LEAF_SAMPLES as f64 * (s1 - s0);
            for a in 0..LEAF_SAMPLES {
                let r = r0 + (a as f64 + 0.5) / LEAF_SAMPLES as f64 * (r1 - r0);
                let jac = bilinear_jacobian(verts, r, s);
                total += jac;
                if footprint.contains(map(verts, r, s)) {
                    inside += jac;
                }
            }
        }
        area * inside / total
    } else {
        [(r0, rc), (rc, r1)]
            .into_iter()
            .flat_map(|(ra, rb)| [(s0, sc), (sc, s1)].map(|(sa, sb)| [ra, rb, sa, sb]))
            .map(|rect| inside_area(verts, rect, footprint, tolerance, level - 1))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loland_coefficient_values() {
        // Open water and a typical clean net (S_n = 0.25)
        assert!((loland_drag_coefficient(0.0, 0.0) - 0.0).abs() < 1e-15);
        let cd = loland_drag_coefficient(0.25, 0.0);
        assert!((cd - 0.415).abs() < 1e-12, "{cd}");
        // Parallel to the flow only the skin friction term is left
        let parallel = loland_drag_coefficient(0.25, std::f64::consts::FRAC_PI_2);
        assert!((parallel - 0.04).abs() < 1e-12);
        // Fouling (higher solidity) raises the drag
        assert!(loland_drag_coefficient(0.4, 0.0) > cd);
    }

    #[test]
    fn footprint_geometry() {
        let circle = CageFootprint::Circle {
            center: [10.0, 5.0],
            radius: 2.0,
        };
        assert!(circle.contains([11.0, 6.0]));
        assert!(!circle.contains([12.0, 7.0]));
        assert_eq!(circle.bounding_box(), ([8.0, 3.0], [12.0, 7.0]));

        // Clockwise square
        let square = CageFootprint::Polygon(vec![[0.0, 0.0], [0.0, 3.0], [3.0, 3.0], [3.0, 0.0]]);
        assert!(square.contains([1.0, 2.0]));
        assert!(!square.contains([4.0, 1.0]));
        assert!((square.area() - 9.0).abs() < 1e-14);
        assert_eq!(square.bounding_box(), ([0.0, 0.0], [3.0, 3.0]));

        // Signed distance: negative inside, to the nearest edge or vertex
        assert!((circle.signed_distance([10.0, 5.0]) + 2.0).abs() < 1e-15);
        assert!((circle.signed_distance([13.0, 5.0]) - 1.0).abs() < 1e-15);
        assert!((square.signed_distance([1.0, 2.0]) + 1.0).abs() < 1e-15);
        assert!((square.signed_distance([5.0, 1.0]) - 2.0).abs() < 1e-15);
        assert!((square.signed_distance([6.0, 7.0]) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn circular_cage_drag_per_length() {
        let cage = NetCage::circular([0.0, 0.0], 25.0, 20.0, 0.25);
        let expected = 0.415 * 4.0 / (std::f64::consts::PI * 25.0);
        assert!((cage.drag_per_length - expected).abs() < 1e-15);
    }

    #[test]
    fn damping_rate_scales_with_the_wet_net_depth() {
        let entry = CageNode {
            node: 0,
            coefficient: 0.01,
            net_depth: 20.0,
        };
        // Net shorter than the column: min(d, h)/h = 1/2
        assert!((CageDrag2D::damping_rate(&[entry], 40.0, 2.0) - 0.01).abs() < 1e-15);
        // Net longer than the column: the whole column is inside
        assert!((CageDrag2D::damping_rate(&[entry], 10.0, 2.0) - 0.02).abs() < 1e-15);
        // Two overlapping cages add
        assert!((CageDrag2D::damping_rate(&[entry, entry], 10.0, 2.0) - 0.04).abs() < 1e-15);
    }

    #[test]
    fn element_far_from_cages_gets_no_entries() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 100.0, 0.0, 100.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let cage = NetCage::circular([500.0, 500.0], 20.0, 10.0, 0.2);
        assert!(CageDrag2D::new(&mesh, &ops, &[cage]).is_empty());
    }
}
