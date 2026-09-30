//! Physical boundaries of the 3D kernels.
//!
//! One classification of every element face, shared by the momentum
//! advection ([`crate::solver::rhs::apply_momentum_transport_3d`]), the
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
//! condition), so the classification must match it: a tag the 2D boundary
//! condition treats as a wall must be a wall here too.
//! [`Boundaries3D::matching`] derives it from the 2D boundary condition
//! (`SWEBoundaryCondition2D::is_wall`), which `Hydrostatic3D` does by
//! default.
//!
//! A nesting parent ([`crate::boundary::Nesting3D`]) replaces the interior's
//! values outside the open faces of its tags by its own profiles
//! ([`Exterior3D`]): the layer volume fluxes take the central average of the
//! interior's and the parent's `H_z u` (then corrected to the 2D flux, as
//! everywhere), and water flowing in brings the parent's velocity and
//! tracers.

use crate::boundary::SWEBoundaryCondition2D;
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
    /// Tags of the wall faces.
    pub wall_tags: Vec<BoundaryTag>,
    /// Whether faces without a tag are walls (the default).
    pub untagged_are_walls: bool,
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
            untagged_are_walls: true,
        }
    }

    /// The classification of the 2D boundary condition `bc` for the boundary
    /// faces of `mesh`: a tag is a wall where `bc.is_wall(tag)` says so, and
    /// as in [`Self::default`] where `bc` cannot tell.
    pub fn matching(mesh: &Mesh2D, bc: &dyn SWEBoundaryCondition2D) -> Self {
        let default = Self::default();
        let mut boundaries = Self {
            wall_tags: Vec::new(),
            untagged_are_walls: bc.is_wall(None).unwrap_or(default.untagged_are_walls),
        };
        for edge in mesh.edges.iter().filter(|edge| edge.right.is_none()) {
            let Some(tag) = edge.boundary_tag else {
                continue;
            };
            if boundaries.wall_tags.contains(&tag) {
                continue;
            }
            let wall = bc
                .is_wall(Some(tag))
                .unwrap_or_else(|| default.wall_tags.contains(&tag));
            if wall {
                boundaries.wall_tags.push(tag);
            }
        }
        boundaries
    }

    /// What lies beyond face `face` of `element`.
    #[inline]
    pub fn exterior(&self, mesh: &Mesh2D, element: ElementIndex, face: usize) -> FaceExterior {
        match mesh.neighbor(element, face) {
            Some(neighbor) => FaceExterior::Element(neighbor),
            None => match mesh.boundary_tag(element, face) {
                Some(tag) if !self.wall_tags.contains(&tag) => FaceExterior::Open(tag),
                Some(_) => FaceExterior::Wall,
                None if self.untagged_are_walls => FaceExterior::Wall,
                // An untagged open face: the tracer boundary conditions see
                // the default open tag
                None => FaceExterior::Open(BoundaryTag::Open),
            },
        }
    }
}

/// Values of one field outside the open faces of some tags, per node and
/// layer: a nesting parent's profiles (see [`crate::boundary::Nesting3D`]).
#[derive(Clone, Copy, Debug)]
pub struct ExteriorField<'a> {
    /// The open-boundary tags the values apply to.
    pub tags: &'a [BoundaryTag],
    /// Slot of every node (`[element][node]`), `u32::MAX` for none.
    pub slot_of_node: &'a [u32],
    /// Layers per slot.
    pub n_levels: usize,
    /// `[slot][level]`.
    pub values: &'a [f64],
}

impl ExteriorField<'_> {
    /// The value at `node` (`[element][node]`), layer `level`, outside a
    /// face tagged `tag`, if there is one.
    #[inline]
    pub fn at(&self, tag: BoundaryTag, node: usize, level: usize) -> Option<f64> {
        if !self.tags.contains(&tag) {
            return None;
        }
        let slot = *self.slot_of_node.get(node)?;
        (slot != u32::MAX).then(|| self.values[slot as usize * self.n_levels + level])
    }
}

/// The exterior values of the 3D kernels at open faces: the interior's
/// (extrapolation) unless a field is given.
#[derive(Clone, Copy, Debug, Default)]
pub struct Exterior3D<'a> {
    /// `(u, v)` in mesh axes (m/s).
    pub velocity: Option<[ExteriorField<'a>; 2]>,
    pub temp: Option<ExteriorField<'a>>,
    pub salt: Option<ExteriorField<'a>>,
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

    /// The classification follows the 2D boundary condition: tags it
    /// dispatches to a wall are walls, tags it keeps open are open, and tags
    /// it cannot tell about fall back to the default.
    #[test]
    fn walls_are_derived_from_the_2d_boundary_condition() {
        use crate::boundary::{BCContext2D, MultiBoundaryCondition2D, Reflective2D};
        use crate::boundary::{CharacteristicOBC, StillWater};
        use crate::solver::SWEState2D;

        // South: Custom(7), north: River, east: Open, west: Wall
        let mut mesh = Mesh2D::uniform_rectangle_with_sides(
            0.0,
            2.0,
            0.0,
            1.0,
            2,
            1,
            [
                BoundaryTag::Custom(7),
                BoundaryTag::Open,
                BoundaryTag::River,
                BoundaryTag::Wall,
            ],
        );
        let open = CharacteristicOBC::new(StillWater::default());
        let wall = Reflective2D::default();
        // Walls by default, the characteristic OBC on Open, and Custom(7)
        // also a wall
        let bc = MultiBoundaryCondition2D::new(&wall)
            .with_open(&open)
            .with_custom(7, &wall);
        let boundaries = Boundaries3D::matching(&mesh, &bc);
        let mut walls = boundaries.wall_tags.clone();
        walls.sort_by_key(|t| format!("{t:?}"));
        assert_eq!(
            walls,
            vec![
                BoundaryTag::Custom(7),
                BoundaryTag::River,
                BoundaryTag::Wall
            ]
        );
        assert!(boundaries.untagged_are_walls);

        // Open by default: River opens, untagged faces too
        let bc = MultiBoundaryCondition2D::new(&open).with_wall(&wall);
        let boundaries = Boundaries3D::matching(&mesh, &bc);
        assert_eq!(boundaries.wall_tags, vec![BoundaryTag::Wall]);
        assert!(!boundaries.untagged_are_walls);
        mesh.edges.iter_mut().for_each(|e| e.boundary_tag = None);
        let east = (0..4)
            .map(|f| boundaries.exterior(&mesh, ElementIndex::new(1), f))
            .find(|e| matches!(e, FaceExterior::Open(_)));
        assert_eq!(east, Some(FaceExterior::Open(BoundaryTag::Open)));

        // A condition that cannot tell: the default
        struct Unknown;
        impl crate::boundary::SWEBoundaryCondition2D for Unknown {
            fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
                ctx.interior_state
            }
            fn name(&self) -> &'static str {
                "unknown"
            }
        }
        let tagged = Mesh2D::uniform_rectangle_with_sides(
            0.0,
            2.0,
            0.0,
            1.0,
            2,
            1,
            [
                BoundaryTag::Custom(7),
                BoundaryTag::Open,
                BoundaryTag::River,
                BoundaryTag::Wall,
            ],
        );
        assert_eq!(
            Boundaries3D::matching(&tagged, &Unknown),
            Boundaries3D::default()
        );
    }
}
