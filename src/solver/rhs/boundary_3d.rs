//! Physical boundaries of the 3D kernels.
//!
//! One classification of every element face, shared by the momentum
//! advection ([`crate::solver::rhs::apply_horizontal_advection_3d`]), the
//! layer transports and their `Ω` ([`crate::solver::rhs::LayerTransport`])
//! and the tracer transport ([`crate::solver::rhs::apply_tracer_transport_3d`]),
//! so that the three agree on what a boundary is. (Each kernel used to decide
//! for itself; the same leak through the coastline was found twice, TODO
//! P0.7/P0.22.)
//!
//! - **Walls** (the tags in [`Boundaries3D::wall_tags`], [`BoundaryTag::Wall`]
//!   by default, and untagged faces): no volume flux through any layer (the 2D
//!   wall passes none), the velocity mirrored (free slip), and so no tracer.
//! - **Open** faces (every other tag): each layer carries its share of the 2D
//!   open-boundary flux in the interior's vertical profile, so a sheared or
//!   exchange flow leaves (and enters) with its shear. Momentum is extrapolated
//!   from the interior (zero gradient, ROMS's "gradient" condition for the 3D
//!   velocity): the shear is advected out, and inflow brings the interior's
//!   profile. Tracers flowing in take the value of the
//!   [`crate::solver::rhs::TracerBoundaryCondition3D`].
//!
//! The depth mean at open faces is the 2D module's (its open-boundary
//! condition); the classification must match it: a tag the 2D boundary
//! condition treats as a wall must be a wall here too.
//!
//! Not yet: prescribed 3D velocity profiles at open faces (nesting of the
//! baroclinic velocity, TODO P4.2).

use crate::mesh::data::BoundaryTag;
use crate::mesh::{ElementFace, Mesh2D};
use crate::types::ElementIndex;

/// What lies beyond an element face (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FaceExterior {
    /// Another element, and its face.
    Element(ElementFace),
    /// A wall.
    Wall,
    /// An open boundary with this tag.
    Open(BoundaryTag),
}

/// Which boundary tags are walls for the 3D kernels (see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct Boundaries3D {
    /// Tags of the wall faces; faces without a tag are walls too.
    pub wall_tags: Vec<BoundaryTag>,
}

impl Default for Boundaries3D {
    /// Faces tagged [`BoundaryTag::Wall`] (and untagged faces) are walls,
    /// every other boundary is open.
    fn default() -> Self {
        Self::with_walls([BoundaryTag::Wall])
    }
}

impl Boundaries3D {
    /// Walls at the faces tagged `tags` (and untagged faces); every other
    /// boundary face is open.
    pub fn with_walls(tags: impl IntoIterator<Item = BoundaryTag>) -> Self {
        Self {
            wall_tags: tags.into_iter().collect(),
        }
    }

    /// What lies beyond face `face` of `element`.
    #[inline]
    pub fn exterior(&self, mesh: &Mesh2D, element: ElementIndex, face: usize) -> FaceExterior {
        match mesh.neighbor(element, face) {
            Some(neighbor) => FaceExterior::Element(neighbor),
            None => match mesh.boundary_tag(element, face) {
                Some(tag) if !self.wall_tags.contains(&tag) => FaceExterior::Open(tag),
                _ => FaceExterior::Wall,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faces_are_classified_by_neighbour_and_tag() {
        // South and north open, east and west walls
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 1.0, 2, 1);
        for edge in &mut mesh.edges {
            if edge.right.is_none() && edge.left.face % 2 == 0 {
                edge.boundary_tag = Some(BoundaryTag::Open);
            }
        }
        let el = ElementIndex::new(0);
        let default = Boundaries3D::default();
        assert_eq!(
            default.exterior(&mesh, el, 0),
            FaceExterior::Open(BoundaryTag::Open)
        );
        assert_eq!(
            default.exterior(&mesh, el, 1),
            FaceExterior::Element(ElementFace {
                element: 1,
                face: 3
            })
        );
        assert_eq!(default.exterior(&mesh, el, 3), FaceExterior::Wall);

        let closed = Boundaries3D::with_walls([BoundaryTag::Wall, BoundaryTag::Open]);
        assert_eq!(closed.exterior(&mesh, el, 0), FaceExterior::Wall);

        mesh.edges.iter_mut().for_each(|e| e.boundary_tag = None);
        assert_eq!(default.exterior(&mesh, el, 0), FaceExterior::Wall);
    }
}
