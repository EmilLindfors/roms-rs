//! 1D mesh representation.
//!
//! A 1D mesh is a partition of an interval [x_min, x_max] into elements.

use crate::mesh::traits::{
    FaceConnection, Mesh1DGeometry, MeshGeometry, MeshTopology, Neighbor,
};

/// Boundary face identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BoundaryFace {
    /// Left boundary (x = x_min)
    #[default]
    Left,
    /// Right boundary (x = x_max)
    Right,
}

/// 1D mesh of an interval.
#[derive(Clone)]
pub struct Mesh1D {
    /// Left endpoint of domain
    pub x_min: f64,
    /// Right endpoint of domain
    pub x_max: f64,
    /// Number of elements
    pub n_elements: usize,
    /// Element vertices: vertices[k] is left endpoint of element k
    /// vertices has length n_elements + 1
    pub vertices: Vec<f64>,
    /// Element sizes: h[k] = vertices[k+1] - vertices[k]
    pub element_sizes: Vec<f64>,
    /// Neighbor connectivity: neighbors[k] = (left_neighbor, right_neighbor)
    /// None if at boundary (non-periodic)
    pub neighbors: Vec<(Option<usize>, Option<usize>)>,
    /// Whether this mesh has periodic boundary conditions
    pub is_periodic: bool,
}

impl Mesh1D {
    /// Create a uniform mesh of [x_min, x_max] with n_elements elements.
    pub fn uniform(x_min: f64, x_max: f64, n_elements: usize) -> Self {
        assert!(n_elements > 0, "Need at least one element");
        assert!(x_max > x_min, "x_max must be greater than x_min");

        let h = (x_max - x_min) / n_elements as f64;

        let vertices: Vec<f64> = (0..=n_elements).map(|i| x_min + i as f64 * h).collect();

        let element_sizes = vec![h; n_elements];

        let neighbors: Vec<(Option<usize>, Option<usize>)> = (0..n_elements)
            .map(|k| {
                let left = if k > 0 { Some(k - 1) } else { None };
                let right = if k < n_elements - 1 {
                    Some(k + 1)
                } else {
                    None
                };
                (left, right)
            })
            .collect();

        Self {
            x_min,
            x_max,
            n_elements,
            vertices,
            element_sizes,
            neighbors,
            is_periodic: false,
        }
    }

    /// Create a uniform periodic mesh of [x_min, x_max] with n_elements elements.
    ///
    /// The left and right boundaries are connected, so there are no boundary faces.
    pub fn uniform_periodic(x_min: f64, x_max: f64, n_elements: usize) -> Self {
        assert!(n_elements > 0, "Need at least one element");
        assert!(x_max > x_min, "x_max must be greater than x_min");

        let h = (x_max - x_min) / n_elements as f64;

        let vertices: Vec<f64> = (0..=n_elements).map(|i| x_min + i as f64 * h).collect();

        let element_sizes = vec![h; n_elements];

        // Periodic neighbors: wrap around at boundaries
        let neighbors: Vec<(Option<usize>, Option<usize>)> = (0..n_elements)
            .map(|k| {
                let left = if k > 0 {
                    Some(k - 1)
                } else {
                    Some(n_elements - 1) // Wrap to last element
                };
                let right = if k < n_elements - 1 {
                    Some(k + 1)
                } else {
                    Some(0) // Wrap to first element
                };
                (left, right)
            })
            .collect();

        Self {
            x_min,
            x_max,
            n_elements,
            vertices,
            element_sizes,
            neighbors,
            is_periodic: true,
        }
    }

    /// Map reference coordinate r in [-1, 1] to physical coordinate x in element k.
    ///
    /// x = x_k + (1 + r) * h_k / 2
    /// where x_k is the left vertex and h_k is the element size.
    pub fn reference_to_physical(&self, k: usize, r: f64) -> f64 {
        let x_left = self.vertices[k];
        let h = self.element_sizes[k];
        x_left + (1.0 + r) * h / 2.0
    }

    /// Map physical coordinate x to reference coordinate r in element k.
    ///
    /// r = 2 * (x - x_k) / h_k - 1
    pub fn physical_to_reference(&self, k: usize, x: f64) -> f64 {
        let x_left = self.vertices[k];
        let h = self.element_sizes[k];
        2.0 * (x - x_left) / h - 1.0
    }

    /// Get the Jacobian dx/dr for element k.
    ///
    /// For 1D affine elements: dx/dr = h_k / 2
    pub fn jacobian(&self, k: usize) -> f64 {
        self.element_sizes[k] / 2.0
    }

    /// Get the inverse Jacobian dr/dx for element k.
    pub fn jacobian_inv(&self, k: usize) -> f64 {
        2.0 / self.element_sizes[k]
    }

    /// Check if a face is on the boundary.
    ///
    /// Returns `None` for periodic meshes (no boundaries) or interior faces.
    pub fn is_boundary(&self, element: usize, face: usize) -> Option<BoundaryFace> {
        // Periodic meshes have no boundaries
        if self.is_periodic {
            return None;
        }

        match face {
            0 => {
                if element == 0 {
                    Some(BoundaryFace::Left)
                } else {
                    None
                }
            }
            1 => {
                if element == self.n_elements - 1 {
                    Some(BoundaryFace::Right)
                } else {
                    None
                }
            }
            _ => panic!("Invalid face index for 1D element"),
        }
    }

    /// Get total domain length.
    pub fn length(&self) -> f64 {
        self.x_max - self.x_min
    }

    /// Get minimum element size.
    pub fn h_min(&self) -> f64 {
        self.element_sizes
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Get maximum element size.
    pub fn h_max(&self) -> f64 {
        self.element_sizes
            .iter()
            .copied()
            .fold(0.0, f64::max)
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl MeshTopology for Mesh1D {
    type Coord = f64;
    type RefCoord = f64;
    type BoundaryTag = BoundaryFace;

    const FACES_PER_ELEMENT: usize = 2;

    #[inline]
    fn n_elements(&self) -> usize {
        self.n_elements
    }

    fn n_faces(&self) -> usize {
        if self.is_periodic {
            self.n_elements
        } else {
            self.n_elements + 1
        }
    }

    fn n_boundary_faces(&self) -> usize {
        if self.is_periodic {
            0
        } else {
            2
        }
    }

    fn face_connection(&self, element: usize, local_face: usize) -> FaceConnection<Self::BoundaryTag> {
        let neighbor_opt = match local_face {
            0 => self.neighbors[element].0,
            1 => self.neighbors[element].1,
            _ => panic!("Invalid face index {local_face} for 1D element (expected 0 or 1)"),
        };

        match neighbor_opt {
            Some(neighbor_elem) => {
                // Neighbor's face is the opposite: if we're looking at our left (0),
                // neighbor is to our left, so their right face (1) connects to us
                let neighbor_face = 1 - local_face;
                FaceConnection::Interior(Neighbor {
                    element: neighbor_elem,
                    face: neighbor_face,
                })
            }
            None => {
                let tag = if local_face == 0 {
                    BoundaryFace::Left
                } else {
                    BoundaryFace::Right
                };
                FaceConnection::Boundary(tag)
            }
        }
    }

    fn is_periodic(&self) -> bool {
        self.is_periodic
    }
}

impl MeshGeometry for Mesh1D {
    #[inline]
    fn reference_to_physical(&self, element: usize, ref_coord: f64) -> f64 {
        Mesh1D::reference_to_physical(self, element, ref_coord)
    }

    #[inline]
    fn physical_to_reference(&self, element: usize, coord: f64) -> f64 {
        Mesh1D::physical_to_reference(self, element, coord)
    }

    #[inline]
    fn jacobian_det(&self, element: usize) -> f64 {
        self.jacobian(element)
    }

    #[inline]
    fn h_min(&self) -> f64 {
        Mesh1D::h_min(self)
    }

    #[inline]
    fn h_max(&self) -> f64 {
        Mesh1D::h_max(self)
    }

    #[inline]
    fn element_diameter(&self, element: usize) -> f64 {
        self.element_sizes[element]
    }
}

impl Mesh1DGeometry for Mesh1D {
    #[inline]
    fn element_bounds(&self, element: usize) -> (f64, f64) {
        (self.vertices[element], self.vertices[element + 1])
    }

    #[inline]
    fn element_size(&self, element: usize) -> f64 {
        self.element_sizes[element]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_mesh() {
        let mesh = Mesh1D::uniform(0.0, 1.0, 4);

        assert_eq!(mesh.n_elements, 4);
        assert_eq!(mesh.vertices.len(), 5);
        assert!((mesh.h_min() - 0.25).abs() < 1e-14);
    }

    #[test]
    fn test_reference_to_physical() {
        let mesh = Mesh1D::uniform(0.0, 1.0, 4);

        // Element 0: [0, 0.25]
        assert!((mesh.reference_to_physical(0, -1.0) - 0.0).abs() < 1e-14);
        assert!((mesh.reference_to_physical(0, 1.0) - 0.25).abs() < 1e-14);
        assert!((mesh.reference_to_physical(0, 0.0) - 0.125).abs() < 1e-14);

        // Element 2: [0.5, 0.75]
        assert!((mesh.reference_to_physical(2, -1.0) - 0.5).abs() < 1e-14);
        assert!((mesh.reference_to_physical(2, 1.0) - 0.75).abs() < 1e-14);
    }

    #[test]
    fn test_physical_to_reference() {
        let mesh = Mesh1D::uniform(0.0, 1.0, 4);

        // Roundtrip test
        for k in 0..4 {
            for &r in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
                let x = mesh.reference_to_physical(k, r);
                let r_back = mesh.physical_to_reference(k, x);
                assert!((r - r_back).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_jacobian() {
        let mesh = Mesh1D::uniform(0.0, 2.0, 4);

        // h = 0.5, so J = dx/dr = 0.25
        for k in 0..4 {
            assert!((mesh.jacobian(k) - 0.25).abs() < 1e-14);
            assert!((mesh.jacobian_inv(k) - 4.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_neighbors() {
        let mesh = Mesh1D::uniform(0.0, 1.0, 4);

        assert_eq!(mesh.neighbors[0], (None, Some(1)));
        assert_eq!(mesh.neighbors[1], (Some(0), Some(2)));
        assert_eq!(mesh.neighbors[2], (Some(1), Some(3)));
        assert_eq!(mesh.neighbors[3], (Some(2), None));
    }

    #[test]
    fn test_boundary_detection() {
        let mesh = Mesh1D::uniform(0.0, 1.0, 4);

        // Element 0, face 0 (left) is at left boundary
        assert_eq!(mesh.is_boundary(0, 0), Some(BoundaryFace::Left));
        assert_eq!(mesh.is_boundary(0, 1), None);

        // Element 3, face 1 (right) is at right boundary
        assert_eq!(mesh.is_boundary(3, 0), None);
        assert_eq!(mesh.is_boundary(3, 1), Some(BoundaryFace::Right));

        // Interior elements have no boundary faces
        assert_eq!(mesh.is_boundary(1, 0), None);
        assert_eq!(mesh.is_boundary(1, 1), None);
    }

    #[test]
    fn test_periodic_mesh() {
        let mesh = Mesh1D::uniform_periodic(0.0, 1.0, 4);

        assert!(mesh.is_periodic);
        assert_eq!(mesh.n_elements, 4);

        // All neighbors should be Some (wrapped)
        assert_eq!(mesh.neighbors[0], (Some(3), Some(1))); // Left wraps to last
        assert_eq!(mesh.neighbors[1], (Some(0), Some(2)));
        assert_eq!(mesh.neighbors[2], (Some(1), Some(3)));
        assert_eq!(mesh.neighbors[3], (Some(2), Some(0))); // Right wraps to first
    }

    #[test]
    fn test_periodic_no_boundaries() {
        let mesh = Mesh1D::uniform_periodic(0.0, 1.0, 4);

        // Periodic mesh has no boundary faces
        for k in 0..4 {
            assert_eq!(mesh.is_boundary(k, 0), None);
            assert_eq!(mesh.is_boundary(k, 1), None);
        }
    }

    // =========================================================================
    // Trait Implementation Tests
    // =========================================================================

    use crate::mesh::traits::{MeshCFL, MeshGeometry, MeshIter, MeshTopology};

    #[test]
    fn test_mesh_topology_trait() {
        let mesh = Mesh1D::uniform(0.0, 1.0, 4);

        assert_eq!(MeshTopology::n_elements(&mesh), 4);
        assert_eq!(mesh.n_faces(), 5); // n_elements + 1 for non-periodic
        assert_eq!(mesh.n_boundary_faces(), 2);
        assert!(!<Mesh1D as MeshTopology>::is_periodic(&mesh));

        // Face connections
        let conn = mesh.face_connection(0, 0);
        assert!(conn.is_boundary());
        assert_eq!(*conn.boundary_tag().unwrap(), BoundaryFace::Left);

        let conn = mesh.face_connection(0, 1);
        assert!(conn.is_interior());
        let neighbor = conn.neighbor().unwrap();
        assert_eq!(neighbor.element, 1);
        assert_eq!(neighbor.face, 0); // Neighbor's left face
    }

    #[test]
    fn test_mesh_topology_trait_periodic() {
        let mesh = Mesh1D::uniform_periodic(0.0, 1.0, 4);

        assert_eq!(mesh.n_faces(), 4); // Same as n_elements for periodic
        assert_eq!(mesh.n_boundary_faces(), 0);
        assert!(<Mesh1D as MeshTopology>::is_periodic(&mesh));

        // All faces should be interior
        for k in 0..4 {
            for f in 0..2 {
                assert!(mesh.face_connection(k, f).is_interior());
            }
        }

        // Element 0's left face connects to element 3's right face
        let conn = mesh.face_connection(0, 0);
        let neighbor = conn.neighbor().unwrap();
        assert_eq!(neighbor.element, 3);
        assert_eq!(neighbor.face, 1);
    }

    #[test]
    fn test_mesh_geometry_trait() {
        let mesh = Mesh1D::uniform(0.0, 2.0, 4);

        // h = 0.5, so J = 0.25
        assert!((MeshGeometry::jacobian_det(&mesh, 0) - 0.25).abs() < 1e-14);
        assert!((mesh.jacobian_det_inv(0) - 4.0).abs() < 1e-14);

        assert!((MeshGeometry::h_min(&mesh) - 0.5).abs() < 1e-14);
        assert!((MeshGeometry::h_max(&mesh) - 0.5).abs() < 1e-14);
        assert!((mesh.element_diameter(0) - 0.5).abs() < 1e-14);

        // Coordinate mappings
        let x = MeshGeometry::reference_to_physical(&mesh, 0, 0.0);
        assert!((x - 0.25).abs() < 1e-14); // Center of element 0

        let r = MeshGeometry::physical_to_reference(&mesh, 0, 0.25);
        assert!((r - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_mesh_cfl_trait() {
        let mesh = Mesh1D::uniform(0.0, 1.0, 10);

        // dt = CFL * h_min / (wave_speed * (2*order + 1))
        // For order=2, factor = 5.0
        let dt = mesh.compute_dt(1.0, 2, 0.5);
        let expected = 0.5 * 0.1 / 5.0; // CFL=0.5, h_min=0.1, factor=5
        assert!((dt - expected).abs() < 1e-14);

        // Zero wave speed should give infinity
        let dt = mesh.compute_dt(0.0, 2, 0.5);
        assert!(dt.is_infinite());
    }

    #[test]
    fn test_mesh_iter_trait() {
        let mesh = Mesh1D::uniform(0.0, 1.0, 4);

        let elements: Vec<usize> = mesh.elements().collect();
        assert_eq!(elements, vec![0, 1, 2, 3]);
        assert_eq!(mesh.elements().len(), 4);
    }

    #[test]
    fn test_mesh1d_geometry_trait() {
        use crate::mesh::traits::Mesh1DGeometry;

        let mesh = Mesh1D::uniform(0.0, 1.0, 4);

        let (left, right) = mesh.element_bounds(1);
        assert!((left - 0.25).abs() < 1e-14);
        assert!((right - 0.5).abs() < 1e-14);

        assert!((Mesh1DGeometry::element_size(&mesh, 1) - 0.25).abs() < 1e-14);

        // 1D normals
        assert_eq!(mesh.face_normal_1d(0, 0), -1.0);
        assert_eq!(mesh.face_normal_1d(0, 1), 1.0);
    }
}
