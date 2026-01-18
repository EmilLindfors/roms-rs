//! GPU-resident mesh connectivity for surface term computation.
//!
//! This module stores the face connectivity and geometric information
//! needed for Riemann solver computations on the GPU.

use burn::prelude::*;

use crate::mesh::{BoundaryTag, Mesh2D};
use crate::operators::GeometricFactors2D;

/// Convert a BoundaryTag to a numeric value for GPU storage.
fn boundary_tag_to_i64(tag: BoundaryTag) -> i64 {
    match tag {
        BoundaryTag::Wall => 0,
        BoundaryTag::Open => 1,
        BoundaryTag::TidalForcing => 2,
        BoundaryTag::Periodic(_) => 3,
        BoundaryTag::River => 4,
        BoundaryTag::Dirichlet => 5,
        BoundaryTag::Neumann => 6,
        BoundaryTag::Custom(_) => 7,
    }
}

/// GPU-resident connectivity information for surface term computation.
///
/// This structure holds all the pre-computed index tensors needed to
/// efficiently gather face states and compute surface integrals on the GPU.
#[derive(Clone, Debug)]
pub struct BurnConnectivity<B: Backend> {
    /// Interior face element pairs: [n_interior_faces, 2]
    /// Each row contains (elem_minus, elem_plus) - the two elements sharing the face
    pub interior_face_elements: Tensor<B, 2, Int>,

    /// Interior face local indices: [n_interior_faces, 2]
    /// Each row contains (face_idx_minus, face_idx_plus) - local face index in each element
    pub interior_face_local: Tensor<B, 2, Int>,

    /// Face node mapping: [n_elements, 4, n_face_nodes]
    /// Maps (element, face, local_face_node) -> volume node index
    pub face_node_map: Tensor<B, 3, Int>,

    /// Outward normals at face centers: [n_elements, 4, 2]
    /// For each element and face: (nx, ny) unit normal
    pub face_normals: Tensor<B, 3>,

    /// Face Jacobians (surface scaling): [n_elements, 4]
    /// Ratio of physical to reference face length
    pub face_jacobians: Tensor<B, 2>,

    /// Boundary face info: [n_boundary_faces, 3]
    /// Each row: (element, local_face, boundary_tag)
    pub boundary_faces: Tensor<B, 2, Int>,

    /// Number of interior faces
    pub n_interior_faces: usize,

    /// Number of boundary faces
    pub n_boundary_faces: usize,

    /// Number of face nodes
    pub n_face_nodes: usize,

    /// Device
    pub device: B::Device,
}

impl<B: Backend> BurnConnectivity<B>
where
    B::FloatElem: From<f64>,
    B::IntElem: From<i64>,
{
    /// Build GPU connectivity from CPU mesh and operators.
    pub fn from_mesh(
        mesh: &Mesh2D,
        geom: &GeometricFactors2D,
        face_nodes: &[Vec<usize>; 4],
        n_face_nodes: usize,
        device: &B::Device,
    ) -> Self {
        let n_elements = mesh.n_elements;

        // Build interior face pairs from mesh edges
        let mut interior_elements: Vec<i64> = Vec::new();
        let mut interior_local: Vec<i64> = Vec::new();
        let mut boundary_data: Vec<i64> = Vec::new();

        // Iterate through edges to build face connectivity
        for edge in &mesh.edges {
            let left = &edge.left;
            if let Some(right) = &edge.right {
                // Interior edge - record the face pair
                interior_elements.push(left.element as i64);
                interior_elements.push(right.element as i64);
                interior_local.push(left.face as i64);
                interior_local.push(right.face as i64);
            } else {
                // Boundary edge
                let tag = edge.boundary_tag.map(boundary_tag_to_i64).unwrap_or(0);
                boundary_data.push(left.element as i64);
                boundary_data.push(left.face as i64);
                boundary_data.push(tag);
            }
        }

        let n_interior_faces = interior_elements.len() / 2;
        let n_boundary_faces = boundary_data.len() / 3;

        // Build face node map: [n_elements, 4, n_face_nodes]
        let mut face_node_data: Vec<i64> = Vec::with_capacity(n_elements * 4 * n_face_nodes);
        for k in 0..n_elements {
            for face in 0..4 {
                for &node_idx in &face_nodes[face] {
                    face_node_data.push(node_idx as i64);
                }
            }
        }

        // Build face normals: [n_elements, 4, 2]
        // For now, use reference normals scaled by Jacobian
        let mut face_normal_data: Vec<f64> = Vec::with_capacity(n_elements * 4 * 2);
        let mut face_jacobian_data: Vec<f64> = Vec::with_capacity(n_elements * 4);

        // Reference face normals
        let ref_normals: [(f64, f64); 4] = [
            (0.0, -1.0),  // Face 0 (bottom)
            (1.0, 0.0),   // Face 1 (right)
            (0.0, 1.0),   // Face 2 (top)
            (-1.0, 0.0),  // Face 3 (left)
        ];

        for k in 0..n_elements {
            let j = geom.det_j[k];
            let rx = geom.rx[k];
            let ry = geom.ry[k];
            let sx = geom.sx[k];
            let sy = geom.sy[k];

            for face in 0..4 {
                let (nr, ns) = ref_normals[face];

                // Transform normal to physical coordinates
                // n_physical = J^{-T} * n_reference (unnormalized)
                let nx_unnorm = rx * nr + ry * ns;
                let ny_unnorm = sx * nr + sy * ns;

                // Normalize
                let len = (nx_unnorm * nx_unnorm + ny_unnorm * ny_unnorm).sqrt();
                let (nx, ny) = if len > 1e-14 {
                    (nx_unnorm / len, ny_unnorm / len)
                } else {
                    (nr, ns)
                };

                face_normal_data.push(nx);
                face_normal_data.push(ny);

                // Face Jacobian (length ratio)
                // For quadrilaterals: approximately sqrt(det_j) for each face
                // More accurate: use edge length computation
                face_jacobian_data.push(j.sqrt());
            }
        }

        // Create tensors
        let interior_face_elements = if n_interior_faces > 0 {
            let data: Vec<B::IntElem> = interior_elements.into_iter().map(|x| B::IntElem::from(x)).collect();
            Tensor::from_data(
                burn::tensor::TensorData::new(data, vec![n_interior_faces, 2]),
                device,
            )
        } else {
            Tensor::zeros([0, 2], device)
        };

        let interior_face_local = if n_interior_faces > 0 {
            let data: Vec<B::IntElem> = interior_local.into_iter().map(|x| B::IntElem::from(x)).collect();
            Tensor::from_data(
                burn::tensor::TensorData::new(data, vec![n_interior_faces, 2]),
                device,
            )
        } else {
            Tensor::zeros([0, 2], device)
        };

        let face_node_map = {
            let data: Vec<B::IntElem> = face_node_data.into_iter().map(|x| B::IntElem::from(x)).collect();
            Tensor::from_data(
                burn::tensor::TensorData::new(data, vec![n_elements, 4, n_face_nodes]),
                device,
            )
        };

        let face_normals = {
            let data: Vec<B::FloatElem> = face_normal_data.into_iter().map(|x| B::FloatElem::from(x)).collect();
            Tensor::from_data(
                burn::tensor::TensorData::new(data, vec![n_elements, 4, 2]),
                device,
            )
        };

        let face_jacobians = {
            let data: Vec<B::FloatElem> = face_jacobian_data.into_iter().map(|x| B::FloatElem::from(x)).collect();
            Tensor::from_data(
                burn::tensor::TensorData::new(data, vec![n_elements, 4]),
                device,
            )
        };

        let boundary_faces = if n_boundary_faces > 0 {
            let data: Vec<B::IntElem> = boundary_data.into_iter().map(|x| B::IntElem::from(x)).collect();
            Tensor::from_data(
                burn::tensor::TensorData::new(data, vec![n_boundary_faces, 3]),
                device,
            )
        } else {
            Tensor::zeros([0, 3], device)
        };

        Self {
            interior_face_elements,
            interior_face_local,
            face_node_map,
            face_normals,
            face_jacobians,
            boundary_faces,
            n_interior_faces,
            n_boundary_faces,
            n_face_nodes,
            device: device.clone(),
        }
    }
}

#[cfg(test)]
#[cfg(feature = "burn-ndarray")]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use crate::mesh::Mesh2DBuilder;
    use crate::operators::DGOperators2D;

    #[test]
    fn test_connectivity_creation() {
        // Create a simple 2x2 mesh
        let mesh = Mesh2DBuilder::unit_square()
            .with_resolution(2, 2)
            .build();

        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let device = burn_ndarray::NdArrayDevice::Cpu;

        let conn = BurnConnectivity::<NdArray<f64>>::from_mesh(
            &mesh,
            &geom,
            &ops.face_nodes,
            ops.n_face_nodes,
            &device,
        );

        // Check dimensions
        assert_eq!(conn.n_face_nodes, ops.n_face_nodes);
        assert!(conn.n_interior_faces > 0); // 2x2 mesh has interior faces

        // Check face node map shape
        let map_shape = conn.face_node_map.shape();
        assert_eq!(map_shape.dims[0], mesh.n_elements);
        assert_eq!(map_shape.dims[1], 4);
        assert_eq!(map_shape.dims[2], ops.n_face_nodes);

        // Check face normals shape
        let normals_shape = conn.face_normals.shape();
        assert_eq!(normals_shape.dims[0], mesh.n_elements);
        assert_eq!(normals_shape.dims[1], 4);
        assert_eq!(normals_shape.dims[2], 2);
    }
}
