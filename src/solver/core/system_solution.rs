//! Generic solution storage for systems of conservation laws.
//!
//! Provides multi-variable solution containers with interleaved layout for
//! good cache locality during flux computations.

use crate::types::ElementIndex;

/// Solution storage for systems of N conservation laws (1D).
///
/// Stores nodal values in interleaved layout:
/// `data[k * n_nodes * N + i * N + var]` for element k, node i, variable var.
///
/// This layout provides good cache locality when accessing all variables at a node,
/// which is the common pattern in flux computations.
#[derive(Clone)]
pub struct SystemSolution<const N: usize> {
    /// Nodal values in interleaved layout
    pub data: Vec<f64>,
    /// Number of elements
    pub n_elements: usize,
    /// Number of nodes per element
    pub n_nodes: usize,
}

impl<const N: usize> SystemSolution<N> {
    /// Create a new solution storage initialized to zero.
    pub fn new(n_elements: usize, n_nodes: usize) -> Self {
        Self {
            data: vec![0.0; n_elements * n_nodes * N],
            n_elements,
            n_nodes,
        }
    }

    /// Create a new solution storage initialized to zero (alias for `new`).
    pub fn zeros(n_elements: usize, n_nodes: usize) -> Self {
        Self::new(n_elements, n_nodes)
    }

    /// Get the state at node i in element k.
    pub fn get(&self, k: ElementIndex, i: usize) -> [f64; N] {
        let base = (k.as_usize() * self.n_nodes + i) * N;
        let mut result = [0.0; N];
        result[..N].copy_from_slice(&self.data[base..base + N]);
        result
    }

    /// Set the state at node i in element k.
    pub fn set(&mut self, k: ElementIndex, i: usize, values: [f64; N]) {
        let base = (k.as_usize() * self.n_nodes + i) * N;
        self.data[base..base + N].copy_from_slice(&values[..N]);
    }

    /// Get a single variable at node i in element k.
    pub fn get_var(&self, k: ElementIndex, i: usize, var: usize) -> f64 {
        let base = (k.as_usize() * self.n_nodes + i) * N;
        self.data[base + var]
    }

    /// Set a single variable at node i in element k.
    pub fn set_var(&mut self, k: ElementIndex, i: usize, var: usize, value: f64) {
        let base = (k.as_usize() * self.n_nodes + i) * N;
        self.data[base + var] = value;
    }

    /// Get all nodal values for a single variable in element k.
    ///
    /// Returns a vector since we can't return a strided slice.
    pub fn element_var(&self, k: ElementIndex, var: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.n_nodes);
        for i in 0..self.n_nodes {
            result.push(self.get_var(k, i, var));
        }
        result
    }

    /// Scale all values by a constant.
    pub fn scale(&mut self, c: f64) {
        for v in &mut self.data {
            *v *= c;
        }
    }

    /// Add c * other to self (axpy operation).
    pub fn axpy(&mut self, c: f64, other: &Self) {
        assert_eq!(self.data.len(), other.data.len());
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += c * *b;
        }
    }

    /// Add other to self.
    pub fn add(&mut self, other: &Self) {
        self.axpy(1.0, other);
    }

    /// Get maximum absolute value across all variables.
    pub fn max_abs(&self) -> f64 {
        self.data.iter().map(|&x| x.abs()).fold(0.0, f64::max)
    }

    /// Get maximum absolute value for a specific variable.
    pub fn max_abs_var(&self, var: usize) -> f64 {
        let mut max_val: f64 = 0.0;
        for k in ElementIndex::iter(self.n_elements) {
            for i in 0..self.n_nodes {
                max_val = max_val.max(self.get_var(k, i, var).abs());
            }
        }
        max_val
    }
}

/// Solution storage for 2D systems of N conservation laws.
///
/// Stores nodal values in interleaved layout:
/// `data[k * n_nodes * N + i * N + var]` for element k, node i, variable var.
///
/// This layout provides good cache locality when accessing all variables at a node,
/// which is the common pattern in flux computations.
#[derive(Clone)]
pub struct SystemSolution2D<const N: usize> {
    /// Nodal values in interleaved layout
    pub data: Vec<f64>,
    /// Number of elements
    pub n_elements: usize,
    /// Number of nodes per element
    pub n_nodes: usize,
}

impl<const N: usize> SystemSolution2D<N> {
    /// Create a new solution storage initialized to zero.
    pub fn new(n_elements: usize, n_nodes: usize) -> Self {
        Self {
            data: vec![0.0; n_elements * n_nodes * N],
            n_elements,
            n_nodes,
        }
    }

    /// Create a new solution storage initialized to zero (alias for `new`).
    pub fn zeros(n_elements: usize, n_nodes: usize) -> Self {
        Self::new(n_elements, n_nodes)
    }

    /// Create a solution from raw interleaved data.
    ///
    /// The data must be in the format: `[var0_e0_n0, var1_e0_n0, ..., varN_e0_n0, var0_e0_n1, ...]`
    /// i.e., variables interleaved at each node, nodes sequential within elements.
    pub fn from_data(data: Vec<f64>, n_elements: usize, n_nodes: usize) -> Self {
        debug_assert_eq!(
            data.len(),
            n_elements * n_nodes * N,
            "Data size mismatch: expected {}, got {}",
            n_elements * n_nodes * N,
            data.len()
        );
        Self {
            data,
            n_elements,
            n_nodes,
        }
    }

    /// Get the state at node i in element k.
    #[inline(always)]
    pub fn get(&self, k: ElementIndex, i: usize) -> [f64; N] {
        let base = (k.as_usize() * self.n_nodes + i) * N;
        let mut result = [0.0; N];
        result[..N].copy_from_slice(&self.data[base..base + N]);
        result
    }

    /// Set the state at node i in element k.
    #[inline(always)]
    pub fn set(&mut self, k: ElementIndex, i: usize, values: [f64; N]) {
        let base = (k.as_usize() * self.n_nodes + i) * N;
        self.data[base..base + N].copy_from_slice(&values[..N]);
    }

    /// Get a single variable at node i in element k.
    #[inline(always)]
    pub fn get_var(&self, k: ElementIndex, i: usize, var: usize) -> f64 {
        let base = (k.as_usize() * self.n_nodes + i) * N;
        self.data[base + var]
    }

    /// Set a single variable at node i in element k.
    #[inline(always)]
    pub fn set_var(&mut self, k: ElementIndex, i: usize, var: usize, value: f64) {
        let base = (k.as_usize() * self.n_nodes + i) * N;
        self.data[base + var] = value;
    }

    /// Get direct slice access to element data for batch operations.
    #[inline(always)]
    pub fn element_data(&self, k: ElementIndex) -> &[f64] {
        let start = k.as_usize() * self.n_nodes * N;
        let end = start + self.n_nodes * N;
        &self.data[start..end]
    }

    /// Get mutable direct slice access to element data for batch operations.
    #[inline(always)]
    pub fn element_data_mut(&mut self, k: ElementIndex) -> &mut [f64] {
        let start = k.as_usize() * self.n_nodes * N;
        let end = start + self.n_nodes * N;
        &mut self.data[start..end]
    }

    /// Get all nodal values for a single variable in element k.
    ///
    /// Returns a vector since we can't return a strided slice.
    pub fn element_var(&self, k: ElementIndex, var: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.n_nodes);
        for i in 0..self.n_nodes {
            result.push(self.get_var(k, i, var));
        }
        result
    }

    /// Scale all values by a constant.
    pub fn scale(&mut self, c: f64) {
        for v in &mut self.data {
            *v *= c;
        }
    }

    /// Add c * other to self (axpy operation).
    pub fn axpy(&mut self, c: f64, other: &Self) {
        assert_eq!(self.data.len(), other.data.len());
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += c * *b;
        }
    }

    /// Add other to self.
    pub fn add(&mut self, other: &Self) {
        self.axpy(1.0, other);
    }

    /// Get maximum absolute value across all variables.
    pub fn max_abs(&self) -> f64 {
        self.data.iter().map(|&x| x.abs()).fold(0.0, f64::max)
    }

    /// Get maximum absolute value for a specific variable.
    pub fn max_abs_var(&self, var: usize) -> f64 {
        let mut max_val: f64 = 0.0;
        for k in ElementIndex::iter(self.n_elements) {
            for i in 0..self.n_nodes {
                max_val = max_val.max(self.get_var(k, i, var).abs());
            }
        }
        max_val
    }

    /// Copy from another solution.
    pub fn copy_from(&mut self, other: &Self) {
        assert_eq!(self.data.len(), other.data.len());
        self.data.copy_from_slice(&other.data);
    }

    /// Fill all values with a constant.
    pub fn fill(&mut self, value: f64) {
        for v in &mut self.data {
            *v = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_solution_basic() {
        let n_elem = 4;
        let n_nodes = 3;
        let mut sol: SystemSolution<2> = SystemSolution::new(n_elem, n_nodes);

        assert_eq!(sol.data.len(), n_elem * n_nodes * 2);

        // Set and get values
        sol.set(ElementIndex::new(0), 0, [1.0, 2.0]);
        let vals = sol.get(ElementIndex::new(0), 0);
        assert!((vals[0] - 1.0).abs() < 1e-14);
        assert!((vals[1] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_system_solution_axpy() {
        let n_elem = 2;
        let n_nodes = 3;

        let mut a: SystemSolution<2> = SystemSolution::new(n_elem, n_nodes);
        let mut b: SystemSolution<2> = SystemSolution::new(n_elem, n_nodes);

        for i in 0..a.data.len() {
            a.data[i] = 1.0;
            b.data[i] = 2.0;
        }

        a.axpy(0.5, &b); // a = a + 0.5 * b = 1 + 1 = 2

        for &v in &a.data {
            assert!((v - 2.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_system_solution_element_var() {
        let n_elem = 2;
        let n_nodes = 3;
        let mut sol: SystemSolution<2> = SystemSolution::new(n_elem, n_nodes);

        let k0 = ElementIndex::new(0);

        // Set some values
        for i in 0..n_nodes {
            sol.set(k0, i, [i as f64, (i + 10) as f64]);
        }

        // Get first variable values for element 0
        let var0_vals = sol.element_var(k0, 0);
        assert_eq!(var0_vals.len(), n_nodes);
        for i in 0..n_nodes {
            assert!((var0_vals[i] - i as f64).abs() < 1e-14);
        }

        // Get second variable values for element 0
        let var1_vals = sol.element_var(k0, 1);
        for i in 0..n_nodes {
            assert!((var1_vals[i] - (i + 10) as f64).abs() < 1e-14);
        }
    }

    #[test]
    fn test_system_solution_2d_basic() {
        let n_elem = 4;
        let n_nodes = 9; // (2+1)²
        let mut sol: SystemSolution2D<3> = SystemSolution2D::new(n_elem, n_nodes);

        assert_eq!(sol.data.len(), n_elem * n_nodes * 3);

        // Set and get values
        let k0 = ElementIndex::new(0);
        sol.set(k0, 0, [1.0, 2.0, 3.0]);
        let vals = sol.get(k0, 0);
        assert!((vals[0] - 1.0).abs() < 1e-14);
        assert!((vals[1] - 2.0).abs() < 1e-14);
        assert!((vals[2] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_system_solution_2d_axpy() {
        let n_elem = 2;
        let n_nodes = 4;

        let mut a: SystemSolution2D<3> = SystemSolution2D::new(n_elem, n_nodes);
        let mut b: SystemSolution2D<3> = SystemSolution2D::new(n_elem, n_nodes);

        for i in 0..a.data.len() {
            a.data[i] = 1.0;
            b.data[i] = 2.0;
        }

        a.axpy(0.5, &b); // a = a + 0.5 * b = 1 + 1 = 2

        for &v in &a.data {
            assert!((v - 2.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_system_solution_2d_element_var() {
        let n_elem = 2;
        let n_nodes = 3;
        let mut sol: SystemSolution2D<3> = SystemSolution2D::new(n_elem, n_nodes);

        let k0 = ElementIndex::new(0);

        // Set some values
        for i in 0..n_nodes {
            sol.set(k0, i, [i as f64, (i + 10) as f64, (i + 20) as f64]);
        }

        // Get first variable values for element 0
        let var0_vals = sol.element_var(k0, 0);
        assert_eq!(var0_vals.len(), n_nodes);
        for i in 0..n_nodes {
            assert!((var0_vals[i] - i as f64).abs() < 1e-14);
        }

        // Get second variable values for element 0
        let var1_vals = sol.element_var(k0, 1);
        for i in 0..n_nodes {
            assert!((var1_vals[i] - (i + 10) as f64).abs() < 1e-14);
        }

        // Get third variable values for element 0
        let var2_vals = sol.element_var(k0, 2);
        for i in 0..n_nodes {
            assert!((var2_vals[i] - (i + 20) as f64).abs() < 1e-14);
        }
    }
}
