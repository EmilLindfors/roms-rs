//! 1D mesh representation.
//!
//! A 1D mesh is a partition of an interval [x_min, x_max] into elements.

/// Boundary face identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryFace {
    /// Left boundary (x = x_min)
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
}
