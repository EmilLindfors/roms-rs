//! GPU-resident SWE solution storage.
//!
//! This module provides `BurnSWESolution2D` which stores the shallow water
//! state (h, hu, hv) on the GPU in a batched format suitable for tensor operations.

use burn::prelude::*;

use crate::solver::state::SWESolution2D;
use crate::types::ElementIndex;

/// GPU-resident 2D SWE solution.
///
/// Stores the conserved variables (h, hu, hv) as separate 2D tensors
/// with shape [n_elements, n_nodes]. This layout enables efficient
/// batched matrix operations across all elements.
#[derive(Clone, Debug)]
pub struct BurnSWESolution2D<B: Backend> {
    /// Water depth: [n_elements, n_nodes]
    pub h: Tensor<B, 2>,

    /// x-momentum (h*u): [n_elements, n_nodes]
    pub hu: Tensor<B, 2>,

    /// y-momentum (h*v): [n_elements, n_nodes]
    pub hv: Tensor<B, 2>,

    /// Number of elements
    pub n_elements: usize,

    /// Number of nodes per element
    pub n_nodes: usize,

    /// Device this solution resides on
    pub device: B::Device,
}

impl<B: Backend> BurnSWESolution2D<B>
where
    B::FloatElem: From<f64>,
    f64: From<B::FloatElem>,
{
    /// Create a new GPU solution initialized to zero.
    pub fn zeros(n_elements: usize, n_nodes: usize, device: &B::Device) -> Self {
        let shape = [n_elements, n_nodes];
        Self {
            h: Tensor::zeros(shape, device),
            hu: Tensor::zeros(shape, device),
            hv: Tensor::zeros(shape, device),
            n_elements,
            n_nodes,
            device: device.clone(),
        }
    }

    /// Upload CPU solution to GPU.
    ///
    /// Converts from AoS (Array of Structs) layout to batched SoA layout:
    /// - CPU: data[elem][node] = [h, hu, hv]
    /// - GPU: h[elem, node], hu[elem, node], hv[elem, node]
    pub fn from_cpu(sol: &SWESolution2D, device: &B::Device) -> Self {
        let n_elements = sol.n_elements;
        let n_nodes = sol.n_nodes;
        let total = n_elements * n_nodes;

        // Extract each variable into a contiguous array
        let mut h_data = Vec::with_capacity(total);
        let mut hu_data = Vec::with_capacity(total);
        let mut hv_data = Vec::with_capacity(total);

        for k in ElementIndex::iter(n_elements) {
            for i in 0..n_nodes {
                let state = sol.get_state(k, i);
                h_data.push(B::FloatElem::from(state.h));
                hu_data.push(B::FloatElem::from(state.hu));
                hv_data.push(B::FloatElem::from(state.hv));
            }
        }

        let shape: Vec<usize> = vec![n_elements, n_nodes];
        Self {
            h: Tensor::from_data(
                burn::tensor::TensorData::new(h_data, shape.clone()),
                device,
            ),
            hu: Tensor::from_data(
                burn::tensor::TensorData::new(hu_data, shape.clone()),
                device,
            ),
            hv: Tensor::from_data(
                burn::tensor::TensorData::new(hv_data, shape),
                device,
            ),
            n_elements,
            n_nodes,
            device: device.clone(),
        }
    }

    /// Download GPU solution to CPU.
    ///
    /// Converts from batched SoA layout back to AoS layout.
    pub fn to_cpu(&self) -> SWESolution2D {
        use crate::solver::state::SWEState2D;

        let h_data: Vec<f64> = self
            .h
            .to_data()
            .to_vec::<B::FloatElem>()
            .unwrap()
            .into_iter()
            .map(|x| f64::from(x))
            .collect();
        let hu_data: Vec<f64> = self
            .hu
            .to_data()
            .to_vec::<B::FloatElem>()
            .unwrap()
            .into_iter()
            .map(|x| f64::from(x))
            .collect();
        let hv_data: Vec<f64> = self
            .hv
            .to_data()
            .to_vec::<B::FloatElem>()
            .unwrap()
            .into_iter()
            .map(|x| f64::from(x))
            .collect();

        let mut sol = SWESolution2D::new(self.n_elements, self.n_nodes);

        for k in ElementIndex::iter(self.n_elements) {
            for i in 0..self.n_nodes {
                let idx = k.as_usize() * self.n_nodes + i;
                sol.set_state(
                    k,
                    i,
                    SWEState2D::new(h_data[idx], hu_data[idx], hv_data[idx]),
                );
            }
        }

        sol
    }

    /// Write GPU solution into an existing CPU solution buffer.
    ///
    /// More efficient than `to_cpu()` when reusing existing storage.
    pub fn write_to_cpu(&self, sol: &mut SWESolution2D) {
        use crate::solver::state::SWEState2D;

        debug_assert_eq!(sol.n_elements, self.n_elements);
        debug_assert_eq!(sol.n_nodes, self.n_nodes);

        let h_data: Vec<f64> = self
            .h
            .to_data()
            .to_vec::<B::FloatElem>()
            .unwrap()
            .into_iter()
            .map(|x| f64::from(x))
            .collect();
        let hu_data: Vec<f64> = self
            .hu
            .to_data()
            .to_vec::<B::FloatElem>()
            .unwrap()
            .into_iter()
            .map(|x| f64::from(x))
            .collect();
        let hv_data: Vec<f64> = self
            .hv
            .to_data()
            .to_vec::<B::FloatElem>()
            .unwrap()
            .into_iter()
            .map(|x| f64::from(x))
            .collect();

        for k in ElementIndex::iter(self.n_elements) {
            for i in 0..self.n_nodes {
                let idx = k.as_usize() * self.n_nodes + i;
                sol.set_state(
                    k,
                    i,
                    SWEState2D::new(h_data[idx], hu_data[idx], hv_data[idx]),
                );
            }
        }
    }

    /// Scale all values by a constant: self *= c
    ///
    /// Used in RK time stepping.
    pub fn scale(&mut self, c: f64) {
        self.h = self.h.clone().mul_scalar(c);
        self.hu = self.hu.clone().mul_scalar(c);
        self.hv = self.hv.clone().mul_scalar(c);
    }

    /// AXPY operation: self += c * other
    ///
    /// Used in RK time stepping.
    pub fn axpy(&mut self, c: f64, other: &Self) {
        debug_assert_eq!(self.n_elements, other.n_elements);
        debug_assert_eq!(self.n_nodes, other.n_nodes);

        self.h = self.h.clone().add(other.h.clone().mul_scalar(c));
        self.hu = self.hu.clone().add(other.hu.clone().mul_scalar(c));
        self.hv = self.hv.clone().add(other.hv.clone().mul_scalar(c));
    }

    /// Copy from another GPU solution.
    pub fn copy_from(&mut self, other: &Self) {
        debug_assert_eq!(self.n_elements, other.n_elements);
        debug_assert_eq!(self.n_nodes, other.n_nodes);

        self.h = other.h.clone();
        self.hu = other.hu.clone();
        self.hv = other.hv.clone();
    }

    /// Clone this solution to a new GPU tensor.
    pub fn clone_solution(&self) -> Self {
        Self {
            h: self.h.clone(),
            hu: self.hu.clone(),
            hv: self.hv.clone(),
            n_elements: self.n_elements,
            n_nodes: self.n_nodes,
            device: self.device.clone(),
        }
    }

    /// Get minimum h value (downloads to CPU for checking).
    pub fn min_h(&self) -> f64 {
        let min_tensor = self.h.clone().min();
        let min_data = min_tensor.to_data();
        f64::from(min_data.to_vec::<B::FloatElem>().unwrap()[0])
    }

    /// Get maximum h value (downloads to CPU for checking).
    pub fn max_h(&self) -> f64 {
        let max_tensor = self.h.clone().max();
        let max_data = max_tensor.to_data();
        f64::from(max_data.to_vec::<B::FloatElem>().unwrap()[0])
    }

    /// Check if solution contains NaN or Inf values.
    pub fn is_valid(&self) -> bool {
        // Check h for NaN/Inf
        let h_finite = self.h.clone().is_nan().bool_not().all();
        let hu_finite = self.hu.clone().is_nan().bool_not().all();
        let hv_finite = self.hv.clone().is_nan().bool_not().all();

        let h_ok = h_finite.to_data().to_vec::<bool>().unwrap()[0];
        let hu_ok = hu_finite.to_data().to_vec::<bool>().unwrap()[0];
        let hv_ok = hv_finite.to_data().to_vec::<bool>().unwrap()[0];

        h_ok && hu_ok && hv_ok
    }
}

#[cfg(test)]
#[cfg(feature = "burn-ndarray")]
mod tests {
    use super::*;
    use crate::solver::state::SWEState2D;
    use burn_ndarray::NdArray;

    #[test]
    fn test_solution_roundtrip() {
        let n_elements = 4;
        let n_nodes = 9;
        let device = burn_ndarray::NdArrayDevice::Cpu;

        // Create CPU solution with known values
        let mut cpu_sol = SWESolution2D::new(n_elements, n_nodes);
        for k in ElementIndex::iter(n_elements) {
            for i in 0..n_nodes {
                let val = (k.as_usize() * n_nodes + i) as f64;
                cpu_sol.set_state(k, i, SWEState2D::new(val, val * 2.0, val * 3.0));
            }
        }

        // Upload to GPU
        let gpu_sol = BurnSWESolution2D::<NdArray<f64>>::from_cpu(&cpu_sol, &device);

        // Download back to CPU
        let roundtrip = gpu_sol.to_cpu();

        // Verify values match
        for k in ElementIndex::iter(n_elements) {
            for i in 0..n_nodes {
                let orig = cpu_sol.get_state(k, i);
                let rt = roundtrip.get_state(k, i);
                assert!((orig.h - rt.h).abs() < 1e-12);
                assert!((orig.hu - rt.hu).abs() < 1e-12);
                assert!((orig.hv - rt.hv).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_solution_axpy() {
        let n_elements = 2;
        let n_nodes = 4;
        let device = burn_ndarray::NdArrayDevice::Cpu;

        let mut a = BurnSWESolution2D::<NdArray<f64>>::zeros(n_elements, n_nodes, &device);
        let b = BurnSWESolution2D::<NdArray<f64>>::zeros(n_elements, n_nodes, &device);

        // Set a to all ones
        let ones = Tensor::ones([n_elements, n_nodes], &device);
        a.h = ones.clone();
        a.hu = ones.clone().mul_scalar(2.0);
        a.hv = ones.clone().mul_scalar(3.0);

        // Set b to all halves
        let b_h = Tensor::ones([n_elements, n_nodes], &device).mul_scalar(0.5);
        let b_hu = Tensor::ones([n_elements, n_nodes], &device).mul_scalar(1.0);
        let b_hv = Tensor::ones([n_elements, n_nodes], &device).mul_scalar(1.5);

        let mut b = b;
        b.h = b_h;
        b.hu = b_hu;
        b.hv = b_hv;

        // a += 2.0 * b
        a.axpy(2.0, &b);

        let cpu_a = a.to_cpu();
        let state = cpu_a.get_state(ElementIndex::new(0), 0);
        assert!((state.h - 2.0).abs() < 1e-12); // 1.0 + 2.0 * 0.5
        assert!((state.hu - 4.0).abs() < 1e-12); // 2.0 + 2.0 * 1.0
        assert!((state.hv - 6.0).abs() < 1e-12); // 3.0 + 2.0 * 1.5
    }

    #[test]
    fn test_solution_min_max() {
        let n_elements = 2;
        let n_nodes = 4;
        let device = burn_ndarray::NdArrayDevice::Cpu;

        let mut sol = BurnSWESolution2D::<NdArray<f64>>::zeros(n_elements, n_nodes, &device);

        // Create h with values 1.0, 2.0, 3.0, ..., 8.0
        let h_data: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        sol.h = Tensor::from_data(
            burn::tensor::TensorData::new(h_data, vec![n_elements, n_nodes]),
            &device,
        );

        assert!((sol.min_h() - 1.0).abs() < 1e-12);
        assert!((sol.max_h() - 8.0).abs() < 1e-12);
    }
}
