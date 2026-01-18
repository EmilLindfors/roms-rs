//! Boundary tags for 2D mesh edges.
//!
//! Each boundary edge can be tagged with a type that determines
//! how boundary conditions are applied.

/// Tag identifying the type of a boundary edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoundaryTag {
    /// Solid wall (reflective/no-flux boundary)
    Wall,

    /// Open boundary (radiation/absorbing)
    Open,

    /// Tidal forcing boundary (prescribed elevation)
    TidalForcing,

    /// Periodic boundary paired with another edge
    /// The value is the ID of the paired boundary group
    Periodic(u32),

    /// River inflow boundary
    River,

    /// Dirichlet boundary (prescribed state)
    Dirichlet,

    /// Neumann boundary (prescribed flux)
    Neumann,

    /// Custom tag for user-defined boundary conditions
    Custom(u32),
}

impl BoundaryTag {
    /// Check if this is a periodic boundary.
    pub fn is_periodic(&self) -> bool {
        matches!(self, BoundaryTag::Periodic(_))
    }

    /// Check if this is a solid wall.
    pub fn is_wall(&self) -> bool {
        matches!(self, BoundaryTag::Wall)
    }

    /// Check if this is an open boundary.
    pub fn is_open(&self) -> bool {
        matches!(
            self,
            BoundaryTag::Open | BoundaryTag::TidalForcing | BoundaryTag::River
        )
    }
}

impl Default for BoundaryTag {
    fn default() -> Self {
        BoundaryTag::Wall
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_tag_equality() {
        assert_eq!(BoundaryTag::Wall, BoundaryTag::Wall);
        assert_ne!(BoundaryTag::Wall, BoundaryTag::Open);
        assert_eq!(BoundaryTag::Periodic(1), BoundaryTag::Periodic(1));
        assert_ne!(BoundaryTag::Periodic(1), BoundaryTag::Periodic(2));
    }

    #[test]
    fn test_is_periodic() {
        assert!(BoundaryTag::Periodic(0).is_periodic());
        assert!(!BoundaryTag::Wall.is_periodic());
        assert!(!BoundaryTag::Open.is_periodic());
    }

    #[test]
    fn test_is_wall() {
        assert!(BoundaryTag::Wall.is_wall());
        assert!(!BoundaryTag::Open.is_wall());
        assert!(!BoundaryTag::Periodic(0).is_wall());
    }

    #[test]
    fn test_is_open() {
        assert!(BoundaryTag::Open.is_open());
        assert!(BoundaryTag::TidalForcing.is_open());
        assert!(BoundaryTag::River.is_open());
        assert!(!BoundaryTag::Wall.is_open());
        assert!(!BoundaryTag::Periodic(0).is_open());
    }
}
