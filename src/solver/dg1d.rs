//! DG solution storage for 1D problems.

use crate::mesh::Mesh1D;
use crate::operators::DGOperators1D;

/// Solution storage for 1D DG discretization.
///
/// Stores nodal values in a contiguous array with layout [n_elements, n_nodes].
/// Access via `element(k)` to get a slice of nodal values for element k.
#[derive(Clone)]
pub struct DGSolution1D {
    /// Nodal values, stored as data[k * n_nodes + i] for element k, node i
    pub data: Vec<f64>,
    /// Number of elements
    pub n_elements: usize,
    /// Number of nodes per element
    pub n_nodes: usize,
}

impl DGSolution1D {
    /// Create a new solution storage initialized to zero.
    pub fn new(n_elements: usize, n_nodes: usize) -> Self {
        Self {
            data: vec![0.0; n_elements * n_nodes],
            n_elements,
            n_nodes,
        }
    }

    /// Get the nodal values for element k.
    pub fn element(&self, k: usize) -> &[f64] {
        let start = k * self.n_nodes;
        &self.data[start..start + self.n_nodes]
    }

    /// Get mutable nodal values for element k.
    pub fn element_mut(&mut self, k: usize) -> &mut [f64] {
        let start = k * self.n_nodes;
        &mut self.data[start..start + self.n_nodes]
    }

    /// Set the solution from a function.
    ///
    /// Evaluates f(x) at each physical node location.
    pub fn set_from_function<F>(&mut self, mesh: &Mesh1D, ops: &DGOperators1D, f: F)
    where
        F: Fn(f64) -> f64,
    {
        for k in 0..mesh.n_elements {
            for (i, &r) in ops.nodes.iter().enumerate() {
                let x = mesh.reference_to_physical(k, r);
                self.element_mut(k)[i] = f(x);
            }
        }
    }

    /// Compute the L2 error against an exact solution.
    ///
    /// Uses the quadrature weights and Jacobian for accurate integration.
    pub fn l2_error<F>(&self, mesh: &Mesh1D, ops: &DGOperators1D, exact: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let mut error_sq = 0.0;

        for k in 0..mesh.n_elements {
            let j = mesh.jacobian(k);
            let u_k = self.element(k);

            for (i, &r) in ops.nodes.iter().enumerate() {
                let x = mesh.reference_to_physical(k, r);
                let u_exact = exact(x);
                let diff = u_k[i] - u_exact;
                error_sq += ops.weights[i] * diff * diff * j;
            }
        }

        error_sq.sqrt()
    }

    /// Compute the L∞ (maximum) error against an exact solution.
    pub fn linf_error<F>(&self, mesh: &Mesh1D, ops: &DGOperators1D, exact: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let mut max_error: f64 = 0.0;

        for k in 0..mesh.n_elements {
            let u_k = self.element(k);

            for (i, &r) in ops.nodes.iter().enumerate() {
                let x = mesh.reference_to_physical(k, r);
                let u_exact = exact(x);
                let error = (u_k[i] - u_exact).abs();
                max_error = max_error.max(error);
            }
        }

        max_error
    }

    /// Scale all values by a constant.
    pub fn scale(&mut self, c: f64) {
        for v in &mut self.data {
            *v *= c;
        }
    }

    /// Add c * other to self (axpy operation).
    pub fn axpy(&mut self, c: f64, other: &DGSolution1D) {
        assert_eq!(self.data.len(), other.data.len());
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += c * *b;
        }
    }

    /// Add other to self.
    pub fn add(&mut self, other: &DGSolution1D) {
        self.axpy(1.0, other);
    }

    /// Get maximum absolute value.
    pub fn max_abs(&self) -> f64 {
        self.data.iter().map(|&x| x.abs()).fold(0.0, f64::max)
    }

    /// Compute the integral of the solution over the domain.
    ///
    /// Uses quadrature weights and Jacobian for accurate integration:
    /// ∫ u dx = Σ_k Σ_i w_i * u_{k,i} * J_k
    pub fn integrate(&self, mesh: &Mesh1D, ops: &DGOperators1D) -> f64 {
        let mut integral = 0.0;

        for k in 0..mesh.n_elements {
            let j = mesh.jacobian(k);
            let u_k = self.element(k);

            for (i, &w) in ops.weights.iter().enumerate() {
                integral += w * u_k[i] * j;
            }
        }

        integral
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solution_storage() {
        let n_elem = 4;
        let n_nodes = 3;
        let mut sol = DGSolution1D::new(n_elem, n_nodes);

        assert_eq!(sol.data.len(), n_elem * n_nodes);

        // Set values
        sol.element_mut(0)[0] = 1.0;
        sol.element_mut(0)[1] = 2.0;
        sol.element_mut(1)[0] = 3.0;

        // Read back
        assert!((sol.element(0)[0] - 1.0).abs() < 1e-14);
        assert!((sol.element(0)[1] - 2.0).abs() < 1e-14);
        assert!((sol.element(1)[0] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_axpy() {
        let n_elem = 2;
        let n_nodes = 3;

        let mut a = DGSolution1D::new(n_elem, n_nodes);
        let mut b = DGSolution1D::new(n_elem, n_nodes);

        for i in 0..a.data.len() {
            a.data[i] = 1.0;
            b.data[i] = 2.0;
        }

        a.axpy(0.5, &b); // a = a + 0.5 * b = 1 + 1 = 2

        for &v in &a.data {
            assert!((v - 2.0).abs() < 1e-14);
        }
    }
}
