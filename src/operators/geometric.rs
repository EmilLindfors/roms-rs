//! Geometric factors for 2D elements.
//!
//! For affine quadrilateral elements, the Jacobian is constant per element.
//! The geometric factors transform derivatives from reference to physical space:
//!
//! ∂u/∂x = rx * ∂u/∂r + sx * ∂u/∂s
//! ∂u/∂y = ry * ∂u/∂r + sy * ∂u/∂s
//!
//! where (rx, ry, sx, sy) are the entries of the inverse Jacobian matrix.

use crate::mesh::Mesh2D;

/// Geometric factors for 2D elements.
///
/// For each element, stores the Jacobian information needed for
/// DG operations in physical space.
#[derive(Clone)]
pub struct GeometricFactors2D {
    /// dr/dx for each element (inverse Jacobian entry)
    pub rx: Vec<f64>,

    /// dr/dy for each element
    pub ry: Vec<f64>,

    /// ds/dx for each element
    pub sx: Vec<f64>,

    /// ds/dy for each element
    pub sy: Vec<f64>,

    /// Jacobian determinant for each element: det(J) = x_r * y_s - x_s * y_r
    pub det_j: Vec<f64>,

    /// Inverse Jacobian determinant: 1/det(J)
    pub det_j_inv: Vec<f64>,

    /// Surface Jacobian for each face of each element: surface_j[k][f]
    /// This is the ratio of physical to reference edge length.
    pub surface_j: Vec<[f64; 4]>,

    /// Outward unit normal for each face of each element: normals[k][f] = (nx, ny)
    pub normals: Vec<[(f64, f64); 4]>,

    /// Number of elements
    pub n_elements: usize,
}

impl GeometricFactors2D {
    /// Compute geometric factors from a mesh.
    ///
    /// For affine quadrilaterals, the Jacobian is constant per element.
    /// The mapping from reference (r, s) to physical (x, y) is:
    ///
    /// x(r, s) = a0 + a1*r + a2*s + a3*r*s
    /// y(r, s) = b0 + b1*r + b2*s + b3*r*s
    ///
    /// For parallelograms (affine), a3 = b3 = 0 and the Jacobian is constant.
    pub fn compute(mesh: &Mesh2D) -> Self {
        let n_elements = mesh.n_elements;

        let mut rx = Vec::with_capacity(n_elements);
        let mut ry = Vec::with_capacity(n_elements);
        let mut sx = Vec::with_capacity(n_elements);
        let mut sy = Vec::with_capacity(n_elements);
        let mut det_j = Vec::with_capacity(n_elements);
        let mut det_j_inv = Vec::with_capacity(n_elements);
        let mut surface_j = Vec::with_capacity(n_elements);
        let mut normals = Vec::with_capacity(n_elements);

        for k in 0..n_elements {
            let verts = mesh.element_vertices(k);

            // Compute Jacobian for this element
            let (jac, s_jac, norm) = compute_element_jacobian(&verts);

            rx.push(jac.rx);
            ry.push(jac.ry);
            sx.push(jac.sx);
            sy.push(jac.sy);
            det_j.push(jac.det);
            det_j_inv.push(1.0 / jac.det);
            surface_j.push(s_jac);
            normals.push(norm);
        }

        Self {
            rx,
            ry,
            sx,
            sy,
            det_j,
            det_j_inv,
            surface_j,
            normals,
            n_elements,
        }
    }

    /// Get the Jacobian determinant for an element.
    #[inline]
    pub fn jacobian(&self, k: usize) -> f64 {
        self.det_j[k]
    }

    /// Get the inverse Jacobian determinant for an element.
    #[inline]
    pub fn jacobian_inv(&self, k: usize) -> f64 {
        self.det_j_inv[k]
    }

    /// Get the surface Jacobian for a face.
    #[inline]
    pub fn surface_jacobian(&self, k: usize, face: usize) -> f64 {
        self.surface_j[k][face]
    }

    /// Get the outward unit normal for a face.
    #[inline]
    pub fn normal(&self, k: usize, face: usize) -> (f64, f64) {
        self.normals[k][face]
    }

    /// Transform reference derivatives to physical derivatives.
    ///
    /// Given ∂u/∂r and ∂u/∂s, compute ∂u/∂x and ∂u/∂y.
    #[inline]
    pub fn transform_derivatives(&self, k: usize, du_dr: f64, du_ds: f64) -> (f64, f64) {
        let du_dx = self.rx[k] * du_dr + self.sx[k] * du_ds;
        let du_dy = self.ry[k] * du_dr + self.sy[k] * du_ds;
        (du_dx, du_dy)
    }
}

/// Internal struct for Jacobian entries.
struct ElementJacobian {
    rx: f64,
    ry: f64,
    sx: f64,
    sy: f64,
    det: f64,
}

/// Compute Jacobian for an affine quadrilateral element.
///
/// For a quadrilateral with vertices v0, v1, v2, v3 in counter-clockwise order:
/// - v0 at (r, s) = (-1, -1)
/// - v1 at (r, s) = (+1, -1)
/// - v2 at (r, s) = (+1, +1)
/// - v3 at (r, s) = (-1, +1)
///
/// The bilinear mapping is:
/// x(r, s) = (1-r)(1-s)/4 * x0 + (1+r)(1-s)/4 * x1 + (1+r)(1+s)/4 * x2 + (1-r)(1+s)/4 * x3
///
/// For affine quads (parallelograms), we use the linear approximation:
/// x_r = (x1 - x0 + x2 - x3) / 4
/// x_s = (x3 - x0 + x2 - x1) / 4
fn compute_element_jacobian(
    verts: &[(f64, f64); 4],
) -> (ElementJacobian, [f64; 4], [(f64, f64); 4]) {
    let (x0, y0) = verts[0];
    let (x1, y1) = verts[1];
    let (x2, y2) = verts[2];
    let (x3, y3) = verts[3];

    // Jacobian entries: J = [[x_r, x_s], [y_r, y_s]]
    // For affine elements (parallelograms), these are constant
    let x_r = (x1 - x0 + x2 - x3) / 4.0;
    let x_s = (x3 - x0 + x2 - x1) / 4.0;
    let y_r = (y1 - y0 + y2 - y3) / 4.0;
    let y_s = (y3 - y0 + y2 - y1) / 4.0;

    // Jacobian determinant
    let det = x_r * y_s - x_s * y_r;

    // Inverse Jacobian: J^{-1} = [[r_x, r_y], [s_x, s_y]] = (1/det) * [[y_s, -x_s], [-y_r, x_r]]
    let rx = y_s / det;
    let ry = -x_s / det;
    let sx = -y_r / det;
    let sy = x_r / det;

    // Compute surface Jacobians and normals for each face
    let mut surface_j = [0.0; 4];
    let mut normals = [(0.0, 0.0); 4];

    // Face 0 (bottom): from v0 to v1, s = -1
    {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let edge_len = (dx * dx + dy * dy).sqrt();
        surface_j[0] = edge_len / 2.0; // Reference edge has length 2

        // Outward normal (pointing down, -s direction)
        let nx = dy / edge_len;
        let ny = -dx / edge_len;
        normals[0] = (nx, ny);
    }

    // Face 1 (right): from v1 to v2, r = +1
    {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let edge_len = (dx * dx + dy * dy).sqrt();
        surface_j[1] = edge_len / 2.0;

        // Outward normal (pointing right, +r direction)
        let nx = dy / edge_len;
        let ny = -dx / edge_len;
        normals[1] = (nx, ny);
    }

    // Face 2 (top): from v2 to v3, s = +1
    {
        let dx = x3 - x2;
        let dy = y3 - y2;
        let edge_len = (dx * dx + dy * dy).sqrt();
        surface_j[2] = edge_len / 2.0;

        // Outward normal (pointing up, +s direction)
        let nx = dy / edge_len;
        let ny = -dx / edge_len;
        normals[2] = (nx, ny);
    }

    // Face 3 (left): from v3 to v0, r = -1
    {
        let dx = x0 - x3;
        let dy = y0 - y3;
        let edge_len = (dx * dx + dy * dy).sqrt();
        surface_j[3] = edge_len / 2.0;

        // Outward normal (pointing left, -r direction)
        let nx = dy / edge_len;
        let ny = -dx / edge_len;
        normals[3] = (nx, ny);
    }

    (
        ElementJacobian {
            rx,
            ry,
            sx,
            sy,
            det,
        },
        surface_j,
        normals,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_unit_square_mesh() -> Mesh2D {
        Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2)
    }

    #[test]
    fn test_geometric_factors_dimensions() {
        let mesh = create_unit_square_mesh();
        let geom = GeometricFactors2D::compute(&mesh);

        assert_eq!(geom.n_elements, mesh.n_elements);
        assert_eq!(geom.rx.len(), mesh.n_elements);
        assert_eq!(geom.det_j.len(), mesh.n_elements);
        assert_eq!(geom.surface_j.len(), mesh.n_elements);
        assert_eq!(geom.normals.len(), mesh.n_elements);
    }

    #[test]
    fn test_unit_square_jacobian() {
        // Single element from (0,0) to (1,1)
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        let geom = GeometricFactors2D::compute(&mesh);

        // For a unit square mapped from [-1,1]² to [0,1]²:
        // x = (1 + r) / 2, y = (1 + s) / 2
        // x_r = 1/2, x_s = 0, y_r = 0, y_s = 1/2
        // det = 1/4
        // rx = y_s / det = 2, ry = 0, sx = 0, sy = 2

        assert!((geom.det_j[0] - 0.25).abs() < 1e-14, "det_J should be 0.25");
        assert!((geom.rx[0] - 2.0).abs() < 1e-14, "rx should be 2");
        assert!(geom.ry[0].abs() < 1e-14, "ry should be 0");
        assert!(geom.sx[0].abs() < 1e-14, "sx should be 0");
        assert!((geom.sy[0] - 2.0).abs() < 1e-14, "sy should be 2");
    }

    #[test]
    fn test_scaled_rectangle_jacobian() {
        // Rectangle [0, 2] × [0, 1]
        let mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 1.0, 1, 1);
        let geom = GeometricFactors2D::compute(&mesh);

        // For a 2×1 rectangle:
        // x = (1 + r), y = (1 + s) / 2
        // x_r = 1, x_s = 0, y_r = 0, y_s = 1/2
        // det = 1/2

        assert!((geom.det_j[0] - 0.5).abs() < 1e-14, "det_J should be 0.5");
    }

    #[test]
    fn test_surface_jacobian_unit_square() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        let geom = GeometricFactors2D::compute(&mesh);

        // For a unit square, each edge has length 1
        // Surface Jacobian = physical_length / reference_length = 1 / 2 = 0.5
        for face in 0..4 {
            assert!(
                (geom.surface_j[0][face] - 0.5).abs() < 1e-14,
                "Surface Jacobian for face {} should be 0.5",
                face
            );
        }
    }

    #[test]
    fn test_normals_unit_square() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        let geom = GeometricFactors2D::compute(&mesh);

        // Check that normals are unit vectors
        for face in 0..4 {
            let (nx, ny) = geom.normals[0][face];
            let len = (nx * nx + ny * ny).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-14,
                "Normal for face {} should be unit vector",
                face
            );
        }

        // Check normal directions for axis-aligned square
        // Face 0 (bottom): should point in -y direction
        assert!((geom.normals[0][0].0).abs() < 1e-14);
        assert!((geom.normals[0][0].1 - (-1.0)).abs() < 1e-14);

        // Face 1 (right): should point in +x direction
        assert!((geom.normals[0][1].0 - 1.0).abs() < 1e-14);
        assert!((geom.normals[0][1].1).abs() < 1e-14);

        // Face 2 (top): should point in +y direction
        assert!((geom.normals[0][2].0).abs() < 1e-14);
        assert!((geom.normals[0][2].1 - 1.0).abs() < 1e-14);

        // Face 3 (left): should point in -x direction
        assert!((geom.normals[0][3].0 - (-1.0)).abs() < 1e-14);
        assert!((geom.normals[0][3].1).abs() < 1e-14);
    }

    #[test]
    fn test_transform_derivatives() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        let geom = GeometricFactors2D::compute(&mesh);

        // For unit square: rx = sy = 2, ry = sx = 0
        // So: du/dx = 2 * du/dr, du/dy = 2 * du/ds

        let du_dr = 1.0;
        let du_ds = 2.0;
        let (du_dx, du_dy) = geom.transform_derivatives(0, du_dr, du_ds);

        assert!((du_dx - 2.0).abs() < 1e-14);
        assert!((du_dy - 4.0).abs() < 1e-14);
    }

    #[test]
    fn test_positive_jacobian() {
        // All elements should have positive Jacobian (counter-clockwise orientation)
        let mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 1.0, 3, 2);
        let geom = GeometricFactors2D::compute(&mesh);

        for k in 0..mesh.n_elements {
            assert!(
                geom.det_j[k] > 0.0,
                "Element {} should have positive Jacobian",
                k
            );
        }
    }

    #[test]
    fn test_jacobian_inverse_consistency() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 1.0, 2, 2);
        let geom = GeometricFactors2D::compute(&mesh);

        for k in 0..mesh.n_elements {
            assert!(
                (geom.det_j[k] * geom.det_j_inv[k] - 1.0).abs() < 1e-14,
                "det_J * det_J_inv should be 1"
            );
        }
    }
}
