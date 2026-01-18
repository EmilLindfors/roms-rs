//! GPU-resident DG operators for 2D elements.
//!
//! This module stores the differentiation and LIFT matrices on the GPU
//! for efficient batched tensor operations.

use burn::prelude::*;

use crate::operators::DGOperators2D;

/// GPU-resident DG operators for 2D quadrilateral elements.
///
/// All matrices are stored as 2D tensors on the GPU device.
/// These are uploaded once at initialization and reused for all RHS computations.
///
/// **Performance note:** Transpose matrices are pre-computed and cached to avoid
/// repeated transpose operations during RHS computation.
#[derive(Clone, Debug)]
pub struct BurnOperators2D<B: Backend> {
    /// Differentiation matrix w.r.t. r: [n_nodes, n_nodes]
    pub dr: Tensor<B, 2>,

    /// Differentiation matrix w.r.t. s: [n_nodes, n_nodes]
    pub ds: Tensor<B, 2>,

    /// Pre-computed transpose of Dr for efficient batched matmul
    pub dr_t: Tensor<B, 2>,

    /// Pre-computed transpose of Ds for efficient batched matmul
    pub ds_t: Tensor<B, 2>,

    /// LIFT matrices for each face: [n_nodes, n_face_nodes]
    pub lift: [Tensor<B, 2>; 4],

    /// Number of volume nodes per element
    pub n_nodes: usize,

    /// Number of nodes per face
    pub n_face_nodes: usize,

    /// Number of 1D nodes (order + 1)
    pub n_1d: usize,

    /// Polynomial order
    pub order: usize,

    /// Device these operators reside on
    pub device: B::Device,
}

impl<B: Backend> BurnOperators2D<B>
where
    B::FloatElem: From<f64>,
{
    /// Upload CPU operators to GPU.
    ///
    /// This performs a one-time data transfer that should be done
    /// at initialization, not during time stepping.
    ///
    /// **Performance note:** Transpose matrices are pre-computed here to avoid
    /// repeated transpose operations during RHS computation.
    pub fn from_cpu(ops: &DGOperators2D, device: &B::Device) -> Self {
        let n_nodes = ops.n_nodes;
        let n_face_nodes = ops.n_face_nodes;

        // Upload differentiation matrices (row-major format already prepared)
        let dr = Self::tensor_from_row_major(&ops.dr_row_major, n_nodes, n_nodes, device);
        let ds = Self::tensor_from_row_major(&ops.ds_row_major, n_nodes, n_nodes, device);

        // Pre-compute transposes (avoid repeated transpose in hot path)
        let dr_t = dr.clone().transpose();
        let ds_t = ds.clone().transpose();

        // Upload LIFT matrices
        let lift = [
            Self::tensor_from_row_major(&ops.lift_row_major[0], n_nodes, n_face_nodes, device),
            Self::tensor_from_row_major(&ops.lift_row_major[1], n_nodes, n_face_nodes, device),
            Self::tensor_from_row_major(&ops.lift_row_major[2], n_nodes, n_face_nodes, device),
            Self::tensor_from_row_major(&ops.lift_row_major[3], n_nodes, n_face_nodes, device),
        ];

        Self {
            dr,
            ds,
            dr_t,
            ds_t,
            lift,
            n_nodes,
            n_face_nodes,
            n_1d: ops.n_1d,
            order: ops.order,
            device: device.clone(),
        }
    }

    /// Helper to create a tensor from row-major data.
    fn tensor_from_row_major(
        data: &[f64],
        nrows: usize,
        ncols: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let data_converted: Vec<B::FloatElem> =
            data.iter().map(|&x| B::FloatElem::from(x)).collect();
        Tensor::from_data(
            burn::tensor::TensorData::new(data_converted, vec![nrows, ncols]),
            device,
        )
    }

    /// Get the pre-computed transpose of Dr for batched matmul.
    ///
    /// For computing `d/dr(flux)`, we compute `flux @ Dr^T` where flux is [E, N].
    ///
    /// **Performance note:** Returns a clone of the pre-computed transpose,
    /// avoiding repeated transpose operations in the hot path.
    #[inline]
    pub fn dr_transpose(&self) -> Tensor<B, 2> {
        self.dr_t.clone()
    }

    /// Get the pre-computed transpose of Ds for batched matmul.
    ///
    /// **Performance note:** Returns a clone of the pre-computed transpose,
    /// avoiding repeated transpose operations in the hot path.
    #[inline]
    pub fn ds_transpose(&self) -> Tensor<B, 2> {
        self.ds_t.clone()
    }
}

#[cfg(test)]
#[cfg(feature = "burn-ndarray")]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn test_operators_upload() {
        let cpu_ops = DGOperators2D::new(2);
        let device = burn_ndarray::NdArrayDevice::Cpu;

        let gpu_ops = BurnOperators2D::<NdArray<f64>>::from_cpu(&cpu_ops, &device);

        assert_eq!(gpu_ops.n_nodes, cpu_ops.n_nodes);
        assert_eq!(gpu_ops.n_face_nodes, cpu_ops.n_face_nodes);
        assert_eq!(gpu_ops.order, cpu_ops.order);

        // Check dimensions
        let dr_shape = gpu_ops.dr.shape();
        assert_eq!(dr_shape.dims[0], cpu_ops.n_nodes);
        assert_eq!(dr_shape.dims[1], cpu_ops.n_nodes);

        let lift0_shape = gpu_ops.lift[0].shape();
        assert_eq!(lift0_shape.dims[0], cpu_ops.n_nodes);
        assert_eq!(lift0_shape.dims[1], cpu_ops.n_face_nodes);
    }

    #[test]
    fn test_dr_transpose() {
        let cpu_ops = DGOperators2D::new(2);
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let gpu_ops = BurnOperators2D::<NdArray<f64>>::from_cpu(&cpu_ops, &device);

        let dr_t = gpu_ops.dr_transpose();
        let dr_t_shape = dr_t.shape();
        assert_eq!(dr_t_shape.dims[0], cpu_ops.n_nodes);
        assert_eq!(dr_t_shape.dims[1], cpu_ops.n_nodes);
    }
}
