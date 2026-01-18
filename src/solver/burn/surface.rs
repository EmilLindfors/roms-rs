//! GPU surface term computation for DG methods.
//!
//! This module handles the extraction of face states and application
//! of LIFT matrices for surface integral contributions.

use burn::prelude::*;

use super::connectivity::BurnConnectivity;
use super::operators::BurnOperators2D;
use super::solution::BurnSWESolution2D;

/// Face states extracted from the solution.
///
/// Contains the state values at all face nodes for interior faces.
#[derive(Clone, Debug)]
pub struct BurnFaceStates<B: Backend> {
    /// h values at face nodes [n_interior_faces, n_face_nodes]
    pub h: Tensor<B, 2>,
    /// hu values at face nodes
    pub hu: Tensor<B, 2>,
    /// hv values at face nodes
    pub hv: Tensor<B, 2>,
}

/// Gather face states from the solution using connectivity indices.
///
/// This extracts the interior (-) and exterior (+) states at all
/// interior faces for Riemann solver computation.
///
/// # Arguments
/// * `solution` - GPU solution [n_elements, n_nodes]
/// * `connectivity` - Pre-computed connectivity tensors
///
/// # Returns
/// (interior_states, exterior_states) for all interior faces
pub fn gather_face_states<B: Backend>(
    solution: &BurnSWESolution2D<B>,
    connectivity: &BurnConnectivity<B>,
) -> (BurnFaceStates<B>, BurnFaceStates<B>)
where
    B::FloatElem: From<f64>,
    B::IntElem: From<i64>,
    f64: From<B::FloatElem>,
    i64: From<B::IntElem>,
{
    let n_interior_faces = connectivity.n_interior_faces;
    let n_face_nodes = connectivity.n_face_nodes;
    let device = &connectivity.device;

    if n_interior_faces == 0 {
        // No interior faces - return empty tensors
        let empty = Tensor::zeros([0, n_face_nodes], device);
        return (
            BurnFaceStates {
                h: empty.clone(),
                hu: empty.clone(),
                hv: empty.clone(),
            },
            BurnFaceStates {
                h: empty.clone(),
                hu: empty.clone(),
                hv: empty,
            },
        );
    }

    // For now, implement a simple gather by downloading indices and
    // reconstructing on device. A more efficient implementation would
    // use custom GPU gather kernels.

    // Download connectivity data
    let face_elements: Vec<i64> = connectivity
        .interior_face_elements
        .to_data()
        .to_vec::<B::IntElem>()
        .unwrap()
        .into_iter()
        .map(|x| i64::from(x))
        .collect();

    let face_local: Vec<i64> = connectivity
        .interior_face_local
        .to_data()
        .to_vec::<B::IntElem>()
        .unwrap()
        .into_iter()
        .map(|x| i64::from(x))
        .collect();

    let face_node_map: Vec<i64> = connectivity
        .face_node_map
        .to_data()
        .to_vec::<B::IntElem>()
        .unwrap()
        .into_iter()
        .map(|x| i64::from(x))
        .collect();

    // Download solution
    let h_data: Vec<f64> = solution
        .h
        .to_data()
        .to_vec::<B::FloatElem>()
        .unwrap()
        .into_iter()
        .map(|x| f64::from(x))
        .collect();
    let hu_data: Vec<f64> = solution
        .hu
        .to_data()
        .to_vec::<B::FloatElem>()
        .unwrap()
        .into_iter()
        .map(|x| f64::from(x))
        .collect();
    let hv_data: Vec<f64> = solution
        .hv
        .to_data()
        .to_vec::<B::FloatElem>()
        .unwrap()
        .into_iter()
        .map(|x| f64::from(x))
        .collect();

    // Extract face states
    let n_nodes = solution.n_nodes;
    let n_elements = solution.n_elements;

    let mut h_minus = Vec::with_capacity(n_interior_faces * n_face_nodes);
    let mut hu_minus = Vec::with_capacity(n_interior_faces * n_face_nodes);
    let mut hv_minus = Vec::with_capacity(n_interior_faces * n_face_nodes);
    let mut h_plus = Vec::with_capacity(n_interior_faces * n_face_nodes);
    let mut hu_plus = Vec::with_capacity(n_interior_faces * n_face_nodes);
    let mut hv_plus = Vec::with_capacity(n_interior_faces * n_face_nodes);

    for f in 0..n_interior_faces {
        let elem_minus = face_elements[f * 2] as usize;
        let elem_plus = face_elements[f * 2 + 1] as usize;
        let face_minus = face_local[f * 2] as usize;
        let face_plus = face_local[f * 2 + 1] as usize;

        for fn_idx in 0..n_face_nodes {
            // Get volume node index for this face node
            let vol_idx_minus = face_node_map
                [(elem_minus * 4 + face_minus) * n_face_nodes + fn_idx] as usize;
            let vol_idx_plus = face_node_map
                [(elem_plus * 4 + face_plus) * n_face_nodes + fn_idx] as usize;

            // Get global index in solution array
            let idx_minus = elem_minus * n_nodes + vol_idx_minus;
            let idx_plus = elem_plus * n_nodes + vol_idx_plus;

            h_minus.push(h_data[idx_minus]);
            hu_minus.push(hu_data[idx_minus]);
            hv_minus.push(hv_data[idx_minus]);

            h_plus.push(h_data[idx_plus]);
            hu_plus.push(hu_data[idx_plus]);
            hv_plus.push(hv_data[idx_plus]);
        }
    }

    // Upload to device
    let shape: Vec<usize> = vec![n_interior_faces, n_face_nodes];

    let minus_states = BurnFaceStates {
        h: Tensor::from_data(
            burn::tensor::TensorData::new(
                h_minus.into_iter().map(|x| B::FloatElem::from(x)).collect::<Vec<_>>(),
                shape.clone(),
            ),
            device,
        ),
        hu: Tensor::from_data(
            burn::tensor::TensorData::new(
                hu_minus.into_iter().map(|x| B::FloatElem::from(x)).collect::<Vec<_>>(),
                shape.clone(),
            ),
            device,
        ),
        hv: Tensor::from_data(
            burn::tensor::TensorData::new(
                hv_minus.into_iter().map(|x| B::FloatElem::from(x)).collect::<Vec<_>>(),
                shape.clone(),
            ),
            device,
        ),
    };

    let plus_states = BurnFaceStates {
        h: Tensor::from_data(
            burn::tensor::TensorData::new(
                h_plus.into_iter().map(|x| B::FloatElem::from(x)).collect::<Vec<_>>(),
                shape.clone(),
            ),
            device,
        ),
        hu: Tensor::from_data(
            burn::tensor::TensorData::new(
                hu_plus.into_iter().map(|x| B::FloatElem::from(x)).collect::<Vec<_>>(),
                shape.clone(),
            ),
            device,
        ),
        hv: Tensor::from_data(
            burn::tensor::TensorData::new(
                hv_plus.into_iter().map(|x| B::FloatElem::from(x)).collect::<Vec<_>>(),
                shape,
            ),
            device,
        ),
    };

    (minus_states, plus_states)
}

/// Get face normals for interior faces.
///
/// Returns (nx, ny) tensors for all interior face nodes.
pub fn get_interior_face_normals<B: Backend>(
    connectivity: &BurnConnectivity<B>,
) -> (Tensor<B, 2>, Tensor<B, 2>)
where
    B::FloatElem: From<f64>,
    B::IntElem: From<i64>,
    f64: From<B::FloatElem>,
    i64: From<B::IntElem>,
{
    let n_interior_faces = connectivity.n_interior_faces;
    let n_face_nodes = connectivity.n_face_nodes;
    let device = &connectivity.device;

    if n_interior_faces == 0 {
        let empty = Tensor::zeros([0, n_face_nodes], device);
        return (empty.clone(), empty);
    }

    // Download connectivity
    let face_elements: Vec<i64> = connectivity
        .interior_face_elements
        .to_data()
        .to_vec::<B::IntElem>()
        .unwrap()
        .into_iter()
        .map(|x| i64::from(x))
        .collect();

    let face_local: Vec<i64> = connectivity
        .interior_face_local
        .to_data()
        .to_vec::<B::IntElem>()
        .unwrap()
        .into_iter()
        .map(|x| i64::from(x))
        .collect();

    let normals: Vec<f64> = connectivity
        .face_normals
        .to_data()
        .to_vec::<B::FloatElem>()
        .unwrap()
        .into_iter()
        .map(|x| f64::from(x))
        .collect();

    // Extract normals for each interior face (from minus side)
    let n_elements = connectivity.face_normals.shape().dims[0];
    let mut nx_data = Vec::with_capacity(n_interior_faces * n_face_nodes);
    let mut ny_data = Vec::with_capacity(n_interior_faces * n_face_nodes);

    for f in 0..n_interior_faces {
        let elem = face_elements[f * 2] as usize;
        let face = face_local[f * 2] as usize;

        // Get normal for this face
        let base = (elem * 4 + face) * 2;
        let nx = normals[base];
        let ny = normals[base + 1];

        // Replicate for all face nodes
        for _ in 0..n_face_nodes {
            nx_data.push(nx);
            ny_data.push(ny);
        }
    }

    // Upload to device
    let shape: Vec<usize> = vec![n_interior_faces, n_face_nodes];

    let nx = Tensor::from_data(
        burn::tensor::TensorData::new(
            nx_data.into_iter().map(|x| B::FloatElem::from(x)).collect::<Vec<_>>(),
            shape.clone(),
        ),
        device,
    );

    let ny = Tensor::from_data(
        burn::tensor::TensorData::new(
            ny_data.into_iter().map(|x| B::FloatElem::from(x)).collect::<Vec<_>>(),
            shape,
        ),
        device,
    );

    (nx, ny)
}

/// Apply LIFT to flux differences and accumulate into RHS.
///
/// This computes the surface integral contribution:
/// rhs += M^{-1} * ∫ (F* - F) * φ dS
///
/// For all faces of all elements.
///
/// # Arguments
/// * `ops` - GPU operators with LIFT matrices
/// * `flux_diff` - Flux differences at all face nodes
/// * `connectivity` - Pre-computed connectivity
/// * `rhs` - Output RHS tensor to accumulate into [n_elements, n_nodes]
pub fn apply_lift_all_faces<B: Backend>(
    ops: &BurnOperators2D<B>,
    flux_diff_h: &Tensor<B, 2>,   // [n_elements * 4, n_face_nodes]
    flux_diff_hu: &Tensor<B, 2>,
    flux_diff_hv: &Tensor<B, 2>,
    connectivity: &BurnConnectivity<B>,
    rhs_h: &mut Tensor<B, 2>,     // [n_elements, n_nodes]
    rhs_hu: &mut Tensor<B, 2>,
    rhs_hv: &mut Tensor<B, 2>,
)
where
    B::FloatElem: From<f64>,
    B::IntElem: From<i64>,
    f64: From<B::FloatElem>,
    i64: From<B::IntElem>,
{
    let n_elements = rhs_h.shape().dims[0];
    let n_nodes = ops.n_nodes;
    let n_face_nodes = ops.n_face_nodes;
    let device = &ops.device;

    // Download flux differences
    let flux_h_data: Vec<f64> = flux_diff_h
        .to_data()
        .to_vec::<B::FloatElem>()
        .unwrap()
        .into_iter()
        .map(|x| f64::from(x))
        .collect();
    let flux_hu_data: Vec<f64> = flux_diff_hu
        .to_data()
        .to_vec::<B::FloatElem>()
        .unwrap()
        .into_iter()
        .map(|x| f64::from(x))
        .collect();
    let flux_hv_data: Vec<f64> = flux_diff_hv
        .to_data()
        .to_vec::<B::FloatElem>()
        .unwrap()
        .into_iter()
        .map(|x| f64::from(x))
        .collect();

    // Download face Jacobians
    let face_jac: Vec<f64> = connectivity
        .face_jacobians
        .to_data()
        .to_vec::<B::FloatElem>()
        .unwrap()
        .into_iter()
        .map(|x| f64::from(x))
        .collect();

    // Download LIFT matrices
    let mut lift_data: [Vec<f64>; 4] = Default::default();
    for face in 0..4 {
        lift_data[face] = ops.lift[face]
            .to_data()
            .to_vec::<B::FloatElem>()
            .unwrap()
            .into_iter()
            .map(|x| f64::from(x))
            .collect();
    }

    // Download current RHS
    let mut rhs_h_data: Vec<f64> = rhs_h
        .to_data()
        .to_vec::<B::FloatElem>()
        .unwrap()
        .into_iter()
        .map(|x| f64::from(x))
        .collect();
    let mut rhs_hu_data: Vec<f64> = rhs_hu
        .to_data()
        .to_vec::<B::FloatElem>()
        .unwrap()
        .into_iter()
        .map(|x| f64::from(x))
        .collect();
    let mut rhs_hv_data: Vec<f64> = rhs_hv
        .to_data()
        .to_vec::<B::FloatElem>()
        .unwrap()
        .into_iter()
        .map(|x| f64::from(x))
        .collect();

    // Apply LIFT for each element and face
    for k in 0..n_elements {
        for face in 0..4 {
            let flux_base = (k * 4 + face) * n_face_nodes;
            let face_jac_val = face_jac[k * 4 + face];

            // Apply LIFT: rhs += scale * LIFT * flux_diff
            for i in 0..n_nodes {
                let mut sum_h = 0.0;
                let mut sum_hu = 0.0;
                let mut sum_hv = 0.0;

                for fi in 0..n_face_nodes {
                    let lift_val = lift_data[face][i * n_face_nodes + fi];
                    sum_h += lift_val * flux_h_data[flux_base + fi];
                    sum_hu += lift_val * flux_hu_data[flux_base + fi];
                    sum_hv += lift_val * flux_hv_data[flux_base + fi];
                }

                let idx = k * n_nodes + i;
                rhs_h_data[idx] += face_jac_val * sum_h;
                rhs_hu_data[idx] += face_jac_val * sum_hu;
                rhs_hv_data[idx] += face_jac_val * sum_hv;
            }
        }
    }

    // Upload back to device
    let shape: Vec<usize> = vec![n_elements, n_nodes];
    *rhs_h = Tensor::from_data(
        burn::tensor::TensorData::new(
            rhs_h_data.into_iter().map(|x| B::FloatElem::from(x)).collect::<Vec<_>>(),
            shape.clone(),
        ),
        device,
    );
    *rhs_hu = Tensor::from_data(
        burn::tensor::TensorData::new(
            rhs_hu_data.into_iter().map(|x| B::FloatElem::from(x)).collect::<Vec<_>>(),
            shape.clone(),
        ),
        device,
    );
    *rhs_hv = Tensor::from_data(
        burn::tensor::TensorData::new(
            rhs_hv_data.into_iter().map(|x| B::FloatElem::from(x)).collect::<Vec<_>>(),
            shape,
        ),
        device,
    );
}

#[cfg(test)]
#[cfg(feature = "burn-ndarray")]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use crate::mesh::Mesh2DBuilder;
    use crate::operators::{DGOperators2D, GeometricFactors2D};
    use crate::types::ElementIndex;

    #[test]
    fn test_gather_face_states() {
        // Create a simple mesh
        let mesh = Mesh2DBuilder::unit_square()
            .with_resolution(2, 2)
            .build();

        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let device = burn_ndarray::NdArrayDevice::Cpu;

        // Create connectivity
        let conn = BurnConnectivity::<NdArray<f64>>::from_mesh(
            &mesh,
            &geom,
            &ops.face_nodes,
            ops.n_face_nodes,
            &device,
        );

        // Create a solution with known values
        let mut cpu_sol = crate::solver::state::SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let val = (k.as_usize() * ops.n_nodes + i) as f64;
                cpu_sol.set_state(k, i, crate::solver::state::SWEState2D::new(val, val, val));
            }
        }

        let gpu_sol = BurnSWESolution2D::<NdArray<f64>>::from_cpu(&cpu_sol, &device);

        // Gather face states
        let (minus, plus) = gather_face_states(&gpu_sol, &conn);

        // Check shapes
        if conn.n_interior_faces > 0 {
            let minus_shape = minus.h.shape();
            assert_eq!(minus_shape.dims[0], conn.n_interior_faces);
            assert_eq!(minus_shape.dims[1], conn.n_face_nodes);

            let plus_shape = plus.h.shape();
            assert_eq!(plus_shape.dims[0], conn.n_interior_faces);
            assert_eq!(plus_shape.dims[1], conn.n_face_nodes);
        }
    }
}
