//! The mesh as the viewer sees it, and the field at the shown time.
//!
//! [`Nodes`] holds what never changes, per DG node in the solver's element-major order:
//! the node's place in the world, the bed, and the geometric factors and reference
//! derivative matrices that give the exact gradient of a nodal field (for the surface
//! normals). [`Field`] is η, u and v at [`crate::playback::Playback::t`], linear in
//! time between the two snapshots around it; in a 3D run (u, v) is the depth mean or
//! the σ-layer [`ShownLayer`] picks. A [`Probe`] evaluates a nodal field at a point by
//! the element's own polynomial (`dg_rs::solver::Probe2D`'s weights).
//!
//! World frame: Bevy is +Y up, so mesh (x, y, z) goes to (x − x₀, z·vz, −(y − y₀)), the
//! origin at the mesh's centre and heights exaggerated `vz` times.

use bevy::prelude::*;
use dg_rs::mesh::PointLocator2D;
use dg_rs::solver::Probe2D;
use dg_rs::types::ElementIndex;

use crate::playback::Playback;
use crate::scenario::Scenario;

/// Mesh coordinates to the world.
#[derive(Resource, Clone, Copy, Debug)]
pub struct Frame {
    /// Mesh point at the world origin (m)
    pub origin: [f64; 2],
    /// Vertical exaggeration
    pub vz: f32,
}

impl Frame {
    /// World position of the mesh point (x, y) at height z (m).
    pub fn world(&self, [x, y]: [f64; 2], z: f32) -> Vec3 {
        Vec3::new(
            (x - self.origin[0]) as f32,
            z * self.vz,
            -(y - self.origin[1]) as f32,
        )
    }
}

#[derive(Resource)]
pub struct Nodes {
    pub n_elements: usize,
    /// Nodes per element
    pub n_nodes: usize,
    /// Nodes per element edge
    pub n_1d: usize,
    /// World (X, Z) of every node
    pub xz: Vec<Vec2>,
    /// Bed elevation B (m, negative under water)
    pub bed: Vec<f32>,
    /// ∂r/∂x, ∂r/∂y, ∂s/∂x, ∂s/∂y at every node
    pub rx: Vec<f32>,
    pub ry: Vec<f32>,
    pub sx: Vec<f32>,
    pub sy: Vec<f32>,
    /// Reference differentiation matrices, row-major `[i * n_nodes + j]`
    pub dr: Vec<f32>,
    pub ds: Vec<f32>,
    /// Element nodes on each face, in order along it
    pub face_nodes: [Vec<usize>; 4],
    /// (element, face) of every face on the mesh boundary
    pub boundary: Vec<(usize, usize)>,
}

impl Nodes {
    pub fn new(scenario: &Scenario, frame: &Frame) -> Self {
        let (mesh, ops, geom) = (&scenario.mesh, &scenario.ops, &scenario.geom);
        let n_nodes = ops.n_nodes;
        let mut xz = Vec::with_capacity(mesh.n_elements * n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..n_nodes {
                let p = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let w = frame.world(p, 0.0);
                xz.push(Vec2::new(w.x, w.z));
            }
        }
        let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let boundary = ElementIndex::iter(mesh.n_elements)
            .flat_map(|k| {
                (0..4)
                    .filter(move |&f| mesh.is_boundary_face(k, f))
                    .map(move |f| (k.as_usize(), f))
            })
            .collect();
        Self {
            n_elements: mesh.n_elements,
            n_nodes,
            n_1d: ops.n_1d,
            xz,
            bed: f32s(&scenario.bathymetry.data),
            rx: f32s(&geom.rx),
            ry: f32s(&geom.ry),
            sx: f32s(&geom.sx),
            sy: f32s(&geom.sy),
            dr: f32s(&ops.dr_row_major),
            ds: f32s(&ops.ds_row_major),
            face_nodes: ops.face_nodes.clone(),
            boundary,
        }
    }

    pub fn len(&self) -> usize {
        self.n_elements * self.n_nodes
    }

    /// World normals of the surface z·vz for the nodal field `z` (m), written to `out`:
    /// the element polynomial's exact gradient at every node.
    pub fn normals(&self, z: &[f32], vz: f32, out: &mut [[f32; 3]]) {
        let n = self.n_nodes;
        for k in 0..self.n_elements {
            let zk = &z[k * n..(k + 1) * n];
            for i in 0..n {
                let (row_r, row_s) = (&self.dr[i * n..(i + 1) * n], &self.ds[i * n..(i + 1) * n]);
                let (mut dzr, mut dzs) = (0.0, 0.0);
                for j in 0..n {
                    dzr += row_r[j] * zk[j];
                    dzs += row_s[j] * zk[j];
                }
                let g = k * n + i;
                let dzx = vz * (self.rx[g] * dzr + self.sx[g] * dzs);
                let dzy = vz * (self.ry[g] * dzr + self.sy[g] * dzs);
                // World X = x, Z = −y: the surface Y = f(X, Z) has normal (−f_X, 1, −f_Z) = (−z_x, 1, z_y).
                out[g] = Vec3::new(-dzx, 1.0, dzy).normalize().to_array();
            }
        }
    }
}

/// Which current a 3D run's water and arrows show.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShownLayer {
    #[default]
    DepthMean,
    /// A σ-layer, 0 at the bed
    Level(usize),
}

/// η, u and v at the shown time.
#[derive(Resource, Default)]
pub struct Field {
    pub eta: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    /// Model time of the field (s); `None` before the first snapshot
    pub t: Option<f64>,
}

/// Interpolates the snapshots around the shown time.
pub fn interpolate(playback: Res<Playback>, shown: Res<ShownLayer>, mut field: ResMut<Field>) {
    if !(playback.changed || shown.is_changed()) {
        return;
    }
    let Some((a, b, w)) = playback.bracket() else {
        return;
    };
    let lerp = |out: &mut Vec<f32>, x: &[f32], y: &[f32]| {
        out.clear();
        out.extend(x.iter().zip(y).map(|(x, y)| x + w * (y - x)));
    };
    lerp(&mut field.eta, &a.eta, &b.eta);
    match (*shown, &a.layers, &b.layers) {
        (ShownLayer::Level(l), Some(la), Some(lb)) => {
            let nl = la.n_levels;
            let l = l.min(nl - 1);
            let level = |out: &mut Vec<f32>, x: &[f32], y: &[f32]| {
                out.clear();
                out.extend(
                    x.iter()
                        .skip(l)
                        .step_by(nl)
                        .zip(y.iter().skip(l).step_by(nl))
                        .map(|(x, y)| x + w * (y - x)),
                );
            };
            level(&mut field.u, &la.u, &lb.u);
            level(&mut field.v, &la.v, &lb.v);
        }
        _ => {
            lerp(&mut field.u, &a.u, &b.u);
            lerp(&mut field.v, &a.v, &b.v);
        }
    }
    field.t = Some(playback.t);
}

/// A fixed point of the mesh, sampled by the element polynomial.
#[derive(Clone, Debug)]
pub struct Probe {
    /// First node of the point's element
    start: usize,
    weights: Vec<f32>,
}

impl Probe {
    /// A probe at mesh point `p`, or `None` outside the mesh.
    pub fn at(locator: &PointLocator2D, scenario: &Scenario, p: [f64; 2]) -> Option<Self> {
        let probe = Probe2D::at(locator, &scenario.ops, p)?;
        Some(Self {
            start: probe.element().as_usize() * scenario.ops.n_nodes,
            weights: probe.weights().iter().map(|&w| w as f32).collect(),
        })
    }

    pub fn eval(&self, field: &[f32]) -> f32 {
        let values = &field[self.start..self.start + self.weights.len()];
        self.weights.iter().zip(values).map(|(w, v)| w * v).sum()
    }

    /// Level `level` of a layered field `[node][level]` with `n_levels` levels.
    pub fn eval_level(&self, field: &[f32], n_levels: usize, level: usize) -> f32 {
        self.weights
            .iter()
            .enumerate()
            .map(|(i, w)| w * field[(self.start + i) * n_levels + level])
            .sum()
    }
}
