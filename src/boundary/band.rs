//! Distances of mesh nodes to open boundaries, for relaxation bands.
//!
//! Shared by the 2D nesting ([`crate::boundary::OceanModelState`]) and the 3D
//! nesting ([`crate::boundary::Nesting3D`]), so that both relax over the same
//! band.

use crate::mesh::{BoundaryTag, Mesh2D};
use crate::operators::DGOperators2D;
use crate::types::ElementIndex;

/// Distance (m) of every node (`[element][node]`) to the nearest boundary
/// face tagged with one of `tags`: 0 at the nodes of those faces, the
/// distance to the polylines through their nodes within `band_width`, and
/// infinity beyond. `None` if no boundary face has one of the tags.
pub fn boundary_distances(
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    tags: &[BoundaryTag],
    band_width: f64,
) -> Option<Vec<f64>> {
    let n_nodes = ops.n_nodes;
    let position = |flat: usize| {
        let (k, i) = (flat / n_nodes, flat % n_nodes);
        let [x, y] =
            mesh.reference_to_physical(ElementIndex::new(k), ops.nodes_r[i], ops.nodes_s[i]);
        (x, y)
    };

    // Open-boundary faces as polylines through their nodes
    let mut boundary_nodes = Vec::new();
    let mut segments = Vec::new();
    for k in ElementIndex::iter(mesh.n_elements) {
        for face in 0..4 {
            if mesh.neighbor(k, face).is_some()
                || !mesh
                    .boundary_tag(k, face)
                    .is_some_and(|t| tags.contains(&t))
            {
                continue;
            }
            let nodes: Vec<usize> = ops.face_nodes[face]
                .iter()
                .map(|&i| k.as_usize() * n_nodes + i)
                .collect();
            for pair in nodes.windows(2) {
                segments.push((position(pair[0]), position(pair[1])));
            }
            boundary_nodes.extend(nodes);
        }
    }
    if boundary_nodes.is_empty() {
        return None;
    }

    let mut distance = vec![f64::INFINITY; mesh.n_elements * n_nodes];
    for &flat in &boundary_nodes {
        distance[flat] = 0.0;
    }
    if band_width > 0.0 {
        let index = SegmentIndex::new(segments, band_width);
        for (flat, d) in distance.iter_mut().enumerate() {
            if *d > 0.0 {
                *d = index.distance(position(flat)).unwrap_or(f64::INFINITY);
            }
        }
    }
    Some(distance)
}

/// Line segments binned on a uniform grid of cells as large as the search
/// radius, for distances up to that radius.
struct SegmentIndex {
    segments: Vec<((f64, f64), (f64, f64))>,
    cells: std::collections::HashMap<(i64, i64), Vec<u32>>,
    size: f64,
}

impl SegmentIndex {
    fn new(segments: Vec<((f64, f64), (f64, f64))>, radius: f64) -> Self {
        let mut cells: std::collections::HashMap<(i64, i64), Vec<u32>> =
            std::collections::HashMap::new();
        let cell = |v: f64| (v / radius).floor() as i64;
        for (s, &(a, b)) in segments.iter().enumerate() {
            for cx in cell(a.0.min(b.0))..=cell(a.0.max(b.0)) {
                for cy in cell(a.1.min(b.1))..=cell(a.1.max(b.1)) {
                    cells.entry((cx, cy)).or_default().push(s as u32);
                }
            }
        }
        Self {
            segments,
            cells,
            size: radius,
        }
    }

    /// Distance from `p` to the nearest segment, if one is within the radius.
    fn distance(&self, p: (f64, f64)) -> Option<f64> {
        let (cx, cy) = (
            (p.0 / self.size).floor() as i64,
            (p.1 / self.size).floor() as i64,
        );
        let mut best = f64::INFINITY;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &s in self.cells.get(&(cx + dx, cy + dy)).into_iter().flatten() {
                    let (a, b) = self.segments[s as usize];
                    best = best.min(point_segment_distance(p, a, b));
                }
            }
        }
        (best <= self.size).then_some(best)
    }
}

fn point_segment_distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 {
        (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p.0 - a.0 - t * dx).hypot(p.1 - a.1 - t * dy)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4 km × 1 km channel open at the west end: the distance of a node is
    /// its x, within the band, and only the tagged faces count.
    #[test]
    fn distances_are_to_the_tagged_faces() {
        let mesh = Mesh2D::uniform_rectangle_with_sides(
            0.0,
            4000.0,
            0.0,
            1000.0,
            4,
            1,
            [
                BoundaryTag::Wall,
                BoundaryTag::Wall,
                BoundaryTag::Wall,
                BoundaryTag::Open,
            ],
        );
        let ops = DGOperators2D::new(2);
        let distance = boundary_distances(&mesh, &ops, &[BoundaryTag::Open], 1500.0).unwrap();
        for k in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let [x, _] = mesh.reference_to_physical(
                    ElementIndex::new(k),
                    ops.nodes_r[i],
                    ops.nodes_s[i],
                );
                let d = distance[k * ops.n_nodes + i];
                if x <= 1500.0 {
                    assert!((d - x).abs() < 1e-9, "x = {x}: {d}");
                } else {
                    assert_eq!(d, f64::INFINITY);
                }
            }
        }
        assert!(boundary_distances(&mesh, &ops, &[BoundaryTag::River], 1500.0).is_none());
    }

    #[test]
    fn segment_distance() {
        let index = SegmentIndex::new(vec![((0.0, 0.0), (10.0, 0.0))], 5.0);
        assert_eq!(index.distance((5.0, 3.0)), Some(3.0));
        assert_eq!(index.distance((13.0, 4.0)), Some(5.0));
        assert_eq!(index.distance((5.0, 6.0)), None);
    }
}
