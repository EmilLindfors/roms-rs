//! Moving a point along a straight segment through the mesh.
//!
//! [`walk`] follows the segment from a point in a known element to its end,
//! crossing one face at a time. Elements are convex quadrilaterals with
//! straight faces, so the segment leaves element `k` through the face it
//! reaches first: the face `f` with outward normal `n_f` (·d > 0) and the
//! smallest parameter
//!
//! ```text
//! t_f = n_f·(v_f − x₀) / n_f·d,        x(t) = x₀ + t d,   d = x₁ − x₀,
//! ```
//!
//! and the end point is in `k` when every such `t_f ≥ 1`. The walk then
//! continues in the neighbour across that face. At a boundary face the
//! segment either
//! - reflects: the rest of the segment is mirrored in the face line
//!   (specular reflection, as for a particle bouncing off a coastline), and
//!   the walk goes on in the same element; or
//! - exits: the walk stops at the crossing point (an open boundary).
//!
//! Periodic faces (an interior connection between faces that do not
//! coincide) translate the point by the offset between the two faces.
//!
//! The cost is proportional to the number of faces crossed, so a particle
//! that moves a fraction of an element per step is relocated in O(1),
//! without the bucket search of [`PointLocator2D::locate`](crate::mesh::PointLocator2D::locate).

use crate::mesh::{BoundaryTag, Mesh2D, MeshPoint, inverse_bilinear};
use crate::types::ElementIndex;

/// Upper bound on face crossings and reflections in one walk; a walk that
/// needs more (e.g. a segment many thousand elements long, or trapped in a
/// corner by round-off) stops where it is.
const MAX_CROSSINGS: usize = 100_000;

/// Where a [`walk`] ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WalkEnd {
    /// The end point (after any reflections) is inside the mesh.
    Inside {
        /// Physical position
        position: [f64; 2],
        /// Element and reference coordinates
        point: MeshPoint,
    },
    /// The segment left the mesh through a boundary face that does not
    /// reflect.
    Exited {
        /// Crossing point on the boundary face
        position: [f64; 2],
        /// Element and reference coordinates of the crossing point
        point: MeshPoint,
        /// Tag of the boundary face
        tag: Option<BoundaryTag>,
    },
}

impl WalkEnd {
    /// Physical position where the walk ended.
    pub fn position(&self) -> [f64; 2] {
        match *self {
            Self::Inside { position, .. } | Self::Exited { position, .. } => position,
        }
    }

    /// Element and reference coordinates where the walk ended.
    pub fn point(&self) -> MeshPoint {
        match *self {
            Self::Inside { point, .. } | Self::Exited { point, .. } => point,
        }
    }
}

/// Element and reference coordinates of `p`, which lies in (or, by
/// round-off, just outside) element `k`.
fn point_in(mesh: &Mesh2D, k: ElementIndex, p: [f64; 2]) -> MeshPoint {
    let [r, s] = inverse_bilinear(&mesh.element_vertices(k), p).unwrap_or([0.0, 0.0]);
    MeshPoint {
        element: k,
        r: r.clamp(-1.0, 1.0),
        s: s.clamp(-1.0, 1.0),
    }
}

/// Follow the segment from `from` (in element `start`) to `to` (see the
/// module docs). `reflects(tag)` decides, for each boundary face the
/// segment reaches, whether it reflects (`true`) or lets the point out.
pub fn walk(
    mesh: &Mesh2D,
    start: ElementIndex,
    from: [f64; 2],
    to: [f64; 2],
    reflects: impl Fn(Option<BoundaryTag>) -> bool,
) -> WalkEnd {
    let (mut k, mut from, mut to) = (start, from, to);
    // The face the walk entered through (or reflected from): never an exit
    let mut entry: Option<usize> = None;
    for _ in 0..MAX_CROSSINGS {
        let v = mesh.element_vertices(k);
        let d = [to[0] - from[0], to[1] - from[1]];
        // The first face the segment reaches, if before its end
        let mut exit: Option<(usize, f64)> = None;
        for f in 0..4 {
            if entry == Some(f) {
                continue;
            }
            let (a, b) = (v[f], v[(f + 1) % 4]);
            // Outward normal of a counter-clockwise face (not normalised)
            let n = [b[1] - a[1], a[0] - b[0]];
            let nd = n[0] * d[0] + n[1] * d[1];
            if nd <= 0.0 {
                continue;
            }
            let t = (n[0] * (a[0] - from[0]) + n[1] * (a[1] - from[1])) / nd;
            if t < 1.0 && exit.is_none_or(|(_, t_min)| t < t_min) {
                exit = Some((f, t.max(0.0)));
            }
        }
        let Some((f, t)) = exit else {
            return WalkEnd::Inside {
                position: to,
                point: point_in(mesh, k, to),
            };
        };
        let cross = [from[0] + t * d[0], from[1] + t * d[1]];
        match mesh.neighbor(k, f) {
            Some(next) => {
                let kn = ElementIndex::new(next.element);
                // A periodic connection joins faces in different places:
                // carry the point over by the offset between them
                let (a, b) = (v[f], v[(f + 1) % 4]);
                let vn = mesh.element_vertices(kn);
                let (an, bn) = (vn[next.face], vn[(next.face + 1) % 4]);
                let offset = [
                    0.5 * (an[0] + bn[0] - a[0] - b[0]),
                    0.5 * (an[1] + bn[1] - a[1] - b[1]),
                ];
                let length = (b[0] - a[0]).abs() + (b[1] - a[1]).abs();
                if offset[0].abs() + offset[1].abs() > 1e-9 * length {
                    from = [cross[0] + offset[0], cross[1] + offset[1]];
                    to = [to[0] + offset[0], to[1] + offset[1]];
                } else {
                    from = cross;
                }
                k = kn;
                entry = Some(next.face);
            }
            None => {
                let tag = mesh.boundary_tag(k, f);
                if !reflects(tag) {
                    return WalkEnd::Exited {
                        position: cross,
                        point: point_in(mesh, k, cross),
                        tag,
                    };
                }
                // Mirror the end point in the face line
                let (a, b) = (v[f], v[(f + 1) % 4]);
                let n = [b[1] - a[1], a[0] - b[0]];
                let nn = n[0] * n[0] + n[1] * n[1];
                let beyond = ((to[0] - a[0]) * n[0] + (to[1] - a[1]) * n[1]) / nn;
                to = [to[0] - 2.0 * beyond * n[0], to[1] - 2.0 * beyond * n[1]];
                from = cross;
                entry = Some(f);
            }
        }
    }
    // Out of crossings: stop at the last point reached
    WalkEnd::Inside {
        position: from,
        point: point_in(mesh, k, from),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::PointLocator2D;

    /// A structured mesh of [0, 3] × [−1, 1] with interior vertices
    /// displaced (general convex quadrilaterals).
    fn distorted_mesh(nx: usize, ny: usize) -> Mesh2D {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 3.0, -1.0, 1.0, nx, ny);
        let (hx, hy) = (3.0 / nx as f64, 2.0 / ny as f64);
        for v in &mut mesh.vertices {
            let [x, y] = *v;
            if x > 1e-12 && x < 3.0 - 1e-12 && y > -1.0 + 1e-12 && y < 1.0 - 1e-12 {
                v[0] += 0.2 * hx * (7.0 * x + 3.0 * y).sin();
                v[1] += 0.2 * hy * (5.0 * x - 4.0 * y).cos();
            }
        }
        mesh
    }

    fn close(a: [f64; 2], b: [f64; 2], tol: f64) -> bool {
        (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol
    }

    /// Walks between random interior points end in the element the bucket
    /// search finds, at the same reference coordinates.
    #[test]
    fn walk_ends_where_locate_finds_the_point() {
        let mesh = distorted_mesh(17, 9);
        let locator = PointLocator2D::new(&mesh);
        let point = |i: usize| {
            let t = i as f64 + 0.5;
            [
                0.02 + 2.96 * (0.618_033_988_7 * t).fract(),
                -0.98 + 1.96 * (0.414_213_562_4 * t).fract(),
            ]
        };
        for i in 0..200 {
            let (a, b) = (point(i), point(i + 7));
            let start = locator.locate(a).unwrap().element;
            let end = walk(&mesh, start, a, b, |_| true);
            let WalkEnd::Inside { position, point } = end else {
                panic!("left the mesh")
            };
            assert_eq!(position, b);
            let expected = locator.locate(b).unwrap();
            assert_eq!(point.element, expected.element, "{a:?} → {b:?}");
            assert!((point.r - expected.r).abs() < 1e-12 && (point.s - expected.s).abs() < 1e-12);
        }
    }

    /// A segment through a wall at 45° ends at its mirror image; through a
    /// corner it is reflected by both walls.
    #[test]
    fn walls_reflect_the_segment() {
        let mesh = distorted_mesh(12, 8);
        let locator = PointLocator2D::new(&mesh);
        let from = [2.5, 0.6];
        let start = locator.locate(from).unwrap().element;
        // Through the top wall y = 1
        let end = walk(&mesh, start, from, [2.7, 1.2], |_| true);
        assert!(close(end.position(), [2.7, 0.8], 1e-12), "{end:?}");
        // Through the corner (3, 1): both coordinates mirrored
        let end = walk(&mesh, start, from, [3.2, 1.3], |_| true);
        assert!(close(end.position(), [2.8, 0.7], 1e-12), "{end:?}");
        let found = locator.locate(end.position()).unwrap();
        assert_eq!(end.point().element, found.element);
        // Several bounces across the channel: y = −1 → 1 → −1 ...
        let end = walk(&mesh, start, [0.5, 0.0], [0.5, 4.5], |_| true);
        assert!(close(end.position(), [0.5, 0.5], 1e-12), "{end:?}");
    }

    /// Faces that do not reflect let the point out at the crossing.
    #[test]
    fn open_faces_let_the_point_out() {
        let mesh = Mesh2D::uniform_rectangle_with_sides(
            0.0,
            3.0,
            -1.0,
            1.0,
            6,
            4,
            [
                BoundaryTag::Wall,
                BoundaryTag::Open,
                BoundaryTag::Wall,
                BoundaryTag::Wall,
            ],
        );
        let locator = PointLocator2D::new(&mesh);
        let from = [2.0, 0.0];
        let start = locator.locate(from).unwrap().element;
        let end = walk(&mesh, start, from, [4.0, 0.5], |tag| {
            tag == Some(BoundaryTag::Wall)
        });
        let WalkEnd::Exited { position, tag, .. } = end else {
            panic!("{end:?}")
        };
        assert_eq!(tag, Some(BoundaryTag::Open));
        assert!(close(position, [3.0, 0.25], 1e-12), "{position:?}");
    }

    /// A periodic channel carries the point across the seam.
    #[test]
    fn periodic_faces_wrap_the_point() {
        let mesh = Mesh2D::channel_periodic_x(0.0, 4.0, 0.0, 1.0, 8, 2);
        let locator = PointLocator2D::new(&mesh);
        let from = [3.7, 0.4];
        let start = locator.locate(from).unwrap().element;
        let end = walk(&mesh, start, from, [3.7 + 4.0 * 2.0 + 0.6, 0.4], |_| true);
        assert!(close(end.position(), [0.3, 0.4], 1e-12), "{end:?}");
        assert_eq!(
            end.point().element,
            locator.locate([0.3, 0.4]).unwrap().element
        );
        // Backwards too
        let end = walk(&mesh, start, from, [-0.5, 0.4], |_| true);
        assert!(close(end.position(), [3.5, 0.4], 1e-12), "{end:?}");
    }
}
