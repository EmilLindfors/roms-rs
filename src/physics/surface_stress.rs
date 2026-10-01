//! Surface stress on the 3D columns, per mesh node.
//!
//! [`Forcing::surface_stress`](crate::physics::Forcing) is one stress for the
//! whole domain. Real wind varies along the coast, so
//! [`Hydrostatic3D::with_surface_stress`](crate::physics::Hydrostatic3D::with_surface_stress)
//! takes a field `τ(x, t)` (N/m², mesh axes) on every column instead, as a
//! [`SurfaceStress3D`]. It enters the model in three places, each per column:
//!
//! - the slow forcing `G` of the barotropic mode, `τ/ρ₀` at `tⁿ` (the mode
//!   splitter extrapolates `G` to the step average);
//! - the surface momentum flux `τ/ρ₀` of the implicit vertical diffusion, at
//!   the middle of the step, `tⁿ + Δt/2`;
//! - the surface friction velocity `u* = √(|τ|/ρ₀)` of the turbulence
//!   closure (the GLS surface boundary), at the same time.
//!
//! The depth mean of the 3D velocity is the barotropic mode's, so only `G`
//! moves it; the column's flux sets the vertical shear. Implementations:
//! [`AnalyticSurfaceStress`] for a field given as a function, and
//! [`crate::source::GriddedWindStress`] for a weather model's 10 m wind
//! ([`crate::source::GriddedAtmosphere2D::split_for_3d`]).

use crate::mesh::Mesh2D;
use crate::operators::DGOperators2D;
use crate::types::ElementIndex;

/// A surface stress field on the mesh nodes.
pub trait SurfaceStress3D: Send + Sync {
    /// Overwrite `tau_x` and `tau_y` (one entry per mesh node,
    /// `[element][node]`) with the surface stress at simulation time `t`
    /// (N/m², in the mesh axes).
    fn surface_stress_into(&self, t: f64, tau_x: &mut [f64], tau_y: &mut [f64]);
}

/// A surface stress given as a function `(x, y, t) → [τ_x, τ_y]` (N/m², mesh
/// coordinates and axes), sampled at the mesh nodes.
pub struct AnalyticSurfaceStress {
    positions: Vec<[f64; 2]>,
    stress: Box<dyn Fn(f64, f64, f64) -> [f64; 2] + Send + Sync>,
}

impl AnalyticSurfaceStress {
    /// `stress` at the nodes of `mesh` with the operators `ops`.
    pub fn new(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        stress: impl Fn(f64, f64, f64) -> [f64; 2] + Send + Sync + 'static,
    ) -> Self {
        let positions = ElementIndex::iter(mesh.n_elements)
            .flat_map(|k| {
                (0..ops.n_nodes)
                    .map(move |i| mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]))
            })
            .collect();
        Self {
            positions,
            stress: Box::new(stress),
        }
    }
}

impl std::fmt::Debug for AnalyticSurfaceStress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnalyticSurfaceStress")
            .field("nodes", &self.positions.len())
            .finish_non_exhaustive()
    }
}

impl SurfaceStress3D for AnalyticSurfaceStress {
    fn surface_stress_into(&self, t: f64, tau_x: &mut [f64], tau_y: &mut [f64]) {
        assert_eq!(tau_x.len(), self.positions.len(), "one stress per node");
        assert_eq!(tau_y.len(), self.positions.len(), "one stress per node");
        for ((&[x, y], tx), ty) in self.positions.iter().zip(tau_x).zip(tau_y) {
            [*tx, *ty] = (self.stress)(x, y, t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_stress_is_sampled_at_the_nodes() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 1.0, 2, 1);
        let ops = DGOperators2D::new(2);
        let stress = AnalyticSurfaceStress::new(&mesh, &ops, |x, y, t| [x + t, y - t]);
        let n = mesh.n_elements * ops.n_nodes;
        let (mut tx, mut ty) = (vec![0.0; n], vec![0.0; n]);
        stress.surface_stress_into(0.5, &mut tx, &mut ty);
        for k in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(
                    ElementIndex::new(k),
                    ops.nodes_r[i],
                    ops.nodes_s[i],
                );
                let idx = k * ops.n_nodes + i;
                assert_eq!([tx[idx], ty[idx]], [x + 0.5, y - 0.5]);
            }
        }
    }
}
