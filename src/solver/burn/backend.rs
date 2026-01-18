//! Backend abstraction for DG GPU computations.
//!
//! This module provides the `DGBackend` trait that abstracts over
//! different Burn backends (CUDA, WGPU, NdArray).

use burn::prelude::*;

/// Trait for backends suitable for DG computations.
///
/// This trait bounds Burn backends to ensure they support:
/// - All necessary tensor operations
/// - Float elements that can be converted to/from f64
pub trait DGBackend: Backend {
    /// Get the default device for this backend.
    fn default_device() -> Self::Device;
}

// Implement DGBackend for all standard Burn backends

#[cfg(feature = "burn-ndarray")]
impl DGBackend for burn_ndarray::NdArray {
    fn default_device() -> Self::Device {
        burn_ndarray::NdArrayDevice::Cpu
    }
}

#[cfg(feature = "burn-wgpu")]
impl DGBackend for burn_wgpu::Wgpu {
    fn default_device() -> Self::Device {
        burn_wgpu::WgpuDevice::default()
    }
}

#[cfg(feature = "burn-cuda")]
impl DGBackend for burn_cuda::Cuda {
    fn default_device() -> Self::Device {
        burn_cuda::CudaDevice::default()
    }
}

/// Helper to create a tensor from a Vec<f64> on the specified device.
#[inline]
pub fn tensor_from_vec<B: Backend>(data: Vec<f64>, shape: [usize; 2], device: &B::Device) -> Tensor<B, 2>
where
    B::FloatElem: From<f64>,
{
    let data_converted: Vec<B::FloatElem> = data.into_iter().map(|x| B::FloatElem::from(x)).collect();
    Tensor::from_data(
        burn::tensor::TensorData::new(data_converted, shape.to_vec()),
        device,
    )
}

/// Helper to create a 1D tensor from a Vec<f64> on the specified device.
#[inline]
pub fn tensor_from_vec_1d<B: Backend>(data: Vec<f64>, device: &B::Device) -> Tensor<B, 1>
where
    B::FloatElem: From<f64>,
{
    let len = data.len();
    let data_converted: Vec<B::FloatElem> = data.into_iter().map(|x| B::FloatElem::from(x)).collect();
    Tensor::from_data(
        burn::tensor::TensorData::new(data_converted, vec![len]),
        device,
    )
}

/// Helper to download a 2D tensor to a Vec<f64>.
#[inline]
pub fn tensor_to_vec<B: Backend>(tensor: &Tensor<B, 2>) -> Vec<f64>
where
    f64: From<B::FloatElem>,
{
    tensor
        .to_data()
        .to_vec::<B::FloatElem>()
        .unwrap()
        .into_iter()
        .map(|x| f64::from(x))
        .collect()
}
