//! Bathymetry (bottom topography) storage for shallow water simulations.
//!
//! Bathymetry B(x) represents the bed elevation above some reference datum.
//! The water surface elevation is η = h + B, where h is the water depth.
//!
//! For well-balanced schemes, we need both B and its gradient ∂B/∂x.

use crate::mesh::Mesh1D;
use crate::operators::DGOperators1D;

/// Bathymetry data for 1D shallow water simulations.
///
/// Stores nodal bathymetry values and their spatial gradients.
/// Same layout as DGSolution1D: data[k * n_nodes + i] for element k, node i.
#[derive(Clone)]
pub struct Bathymetry1D {
    /// Nodal bathymetry values B(x)
    pub data: Vec<f64>,
    /// Pre-computed gradients dB/dx at each node
    pub gradient: Vec<f64>,
    /// Number of elements
    pub n_elements: usize,
    /// Number of nodes per element
    pub n_nodes: usize,
}

impl Bathymetry1D {
    /// Create flat bathymetry (B = 0 everywhere).
    pub fn flat(n_elements: usize, n_nodes: usize) -> Self {
        Self {
            data: vec![0.0; n_elements * n_nodes],
            gradient: vec![0.0; n_elements * n_nodes],
            n_elements,
            n_nodes,
        }
    }

    /// Create constant bathymetry (B = constant everywhere).
    pub fn constant(n_elements: usize, n_nodes: usize, value: f64) -> Self {
        Self {
            data: vec![value; n_elements * n_nodes],
            gradient: vec![0.0; n_elements * n_nodes], // Zero gradient for constant
            n_elements,
            n_nodes,
        }
    }

    /// Initialize bathymetry from a function B(x).
    ///
    /// Evaluates B(x) at each physical node location and computes gradients.
    pub fn from_function<F>(mesh: &Mesh1D, ops: &DGOperators1D, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        let n_elements = mesh.n_elements;
        let n_nodes = ops.n_nodes;
        let mut bathy = Self::flat(n_elements, n_nodes);

        // Set bathymetry values
        for k in 0..n_elements {
            for (i, &r) in ops.nodes.iter().enumerate() {
                let x = mesh.reference_to_physical(k, r);
                bathy.set(k, i, f(x));
            }
        }

        // Compute gradients
        bathy.compute_gradients(mesh, ops);

        bathy
    }

    /// Get bathymetry at node i in element k.
    pub fn get(&self, k: usize, i: usize) -> f64 {
        self.data[k * self.n_nodes + i]
    }

    /// Set bathymetry at node i in element k.
    pub fn set(&mut self, k: usize, i: usize, value: f64) {
        self.data[k * self.n_nodes + i] = value;
    }

    /// Get bathymetry gradient at node i in element k.
    pub fn get_gradient(&self, k: usize, i: usize) -> f64 {
        self.gradient[k * self.n_nodes + i]
    }

    /// Get all bathymetry values for element k.
    pub fn element(&self, k: usize) -> &[f64] {
        let start = k * self.n_nodes;
        &self.data[start..start + self.n_nodes]
    }

    /// Get all gradient values for element k.
    pub fn element_gradient(&self, k: usize) -> &[f64] {
        let start = k * self.n_nodes;
        &self.gradient[start..start + self.n_nodes]
    }

    /// Compute gradients dB/dx using the differentiation matrix.
    ///
    /// dB/dx = (dr/dx) * Dr * B = jacobian_inv * Dr * B
    #[allow(clippy::needless_range_loop)]
    pub fn compute_gradients(&mut self, mesh: &Mesh1D, ops: &DGOperators1D) {
        let n = self.n_nodes;

        for k in 0..self.n_elements {
            let j_inv = mesh.jacobian_inv(k);
            let b_k = self.element(k);

            // dB/dr = Dr * B_k
            let mut db_dr = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    db_dr[i] += ops.dr[(i, j)] * b_k[j];
                }
            }

            // dB/dx = j_inv * dB/dr
            let start = k * n;
            for i in 0..n {
                self.gradient[start + i] = j_inv * db_dr[i];
            }
        }
    }

    /// Get bathymetry at left face of element k.
    pub fn left_face(&self, k: usize) -> f64 {
        self.get(k, 0)
    }

    /// Get bathymetry at right face of element k.
    pub fn right_face(&self, k: usize) -> f64 {
        self.get(k, self.n_nodes - 1)
    }

    /// Get maximum bathymetry value in the domain.
    pub fn max(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get minimum bathymetry value in the domain.
    pub fn min(&self) -> f64 {
        self.data.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Get maximum absolute gradient in the domain.
    pub fn max_gradient(&self) -> f64 {
        self.gradient.iter().map(|&x| x.abs()).fold(0.0, f64::max)
    }
}

/// Common bathymetry profiles for testing.
pub mod profiles {
    /// Gaussian bump centered at x_c with height A and width σ.
    ///
    /// B(x) = A * exp(-(x - x_c)² / (2σ²))
    pub fn gaussian_bump(x: f64, x_c: f64, amplitude: f64, sigma: f64) -> f64 {
        let dx = x - x_c;
        amplitude * (-dx * dx / (2.0 * sigma * sigma)).exp()
    }

    /// Linear slope from B_left to B_right.
    ///
    /// B(x) = B_left + (B_right - B_left) * (x - x_left) / (x_right - x_left)
    pub fn linear_slope(x: f64, x_left: f64, x_right: f64, b_left: f64, b_right: f64) -> f64 {
        let t = (x - x_left) / (x_right - x_left);
        b_left + (b_right - b_left) * t
    }

    /// Step function (discontinuous bathymetry).
    ///
    /// B(x) = B_left if x < x_step, B_right otherwise
    pub fn step(x: f64, x_step: f64, b_left: f64, b_right: f64) -> f64 {
        if x < x_step { b_left } else { b_right }
    }

    /// Parabolic bowl for wetting/drying tests.
    ///
    /// B(x) = h_0 * (x / a)² where a is the half-width
    pub fn parabolic_bowl(x: f64, x_center: f64, a: f64, h_0: f64) -> f64 {
        let dx = x - x_center;
        h_0 * (dx / a).powi(2)
    }

    /// Sinusoidal bed.
    ///
    /// B(x) = B_mean + A * sin(k * x + φ)
    pub fn sinusoidal(x: f64, b_mean: f64, amplitude: f64, wavenumber: f64, phase: f64) -> f64 {
        b_mean + amplitude * (wavenumber * x + phase).sin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn make_mesh_and_ops() -> (Mesh1D, DGOperators1D) {
        let mesh = Mesh1D::uniform(0.0, 10.0, 10);
        let ops = DGOperators1D::new(3);
        (mesh, ops)
    }

    #[test]
    fn test_flat_bathymetry() {
        let bathy = Bathymetry1D::flat(4, 3);

        assert_eq!(bathy.data.len(), 12);
        assert_eq!(bathy.gradient.len(), 12);

        for &b in &bathy.data {
            assert!(b.abs() < TOL);
        }
        for &g in &bathy.gradient {
            assert!(g.abs() < TOL);
        }
    }

    #[test]
    fn test_constant_bathymetry() {
        let bathy = Bathymetry1D::constant(4, 3, 5.0);

        for &b in &bathy.data {
            assert!((b - 5.0).abs() < TOL);
        }
        for &g in &bathy.gradient {
            assert!(g.abs() < TOL); // Gradient of constant is zero
        }
    }

    #[test]
    fn test_from_function_linear() {
        let (mesh, ops) = make_mesh_and_ops();

        // Linear bathymetry: B(x) = x
        let bathy = Bathymetry1D::from_function(&mesh, &ops, |x| x);

        // Check some values
        for k in 0..mesh.n_elements {
            for (i, &r) in ops.nodes.iter().enumerate() {
                let x = mesh.reference_to_physical(k, r);
                let b = bathy.get(k, i);
                assert!((b - x).abs() < TOL, "B({}) = {}, expected {}", x, b, x);
            }
        }

        // Gradient should be 1 everywhere
        for k in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let grad = bathy.get_gradient(k, i);
                assert!((grad - 1.0).abs() < 1e-8, "dB/dx = {}, expected 1.0", grad);
            }
        }
    }

    #[test]
    fn test_from_function_quadratic() {
        let (mesh, ops) = make_mesh_and_ops();

        // Quadratic bathymetry: B(x) = x²
        let bathy = Bathymetry1D::from_function(&mesh, &ops, |x| x * x);

        // Gradient should be 2x
        for k in 0..mesh.n_elements {
            for (i, &r) in ops.nodes.iter().enumerate() {
                let x = mesh.reference_to_physical(k, r);
                let expected_grad = 2.0 * x;
                let grad = bathy.get_gradient(k, i);
                assert!(
                    (grad - expected_grad).abs() < 1e-6,
                    "dB/dx at x={} is {}, expected {}",
                    x,
                    grad,
                    expected_grad
                );
            }
        }
    }

    #[test]
    fn test_element_access() {
        let (mesh, ops) = make_mesh_and_ops();
        let bathy = Bathymetry1D::from_function(&mesh, &ops, |x| x);

        let elem = bathy.element(2);
        assert_eq!(elem.len(), ops.n_nodes);

        let grad_elem = bathy.element_gradient(2);
        assert_eq!(grad_elem.len(), ops.n_nodes);
    }

    #[test]
    fn test_face_values() {
        let (mesh, ops) = make_mesh_and_ops();
        let bathy = Bathymetry1D::from_function(&mesh, &ops, |x| x);

        for k in 0..mesh.n_elements {
            let left = bathy.left_face(k);
            let right = bathy.right_face(k);

            // Right face should be greater than left for increasing B
            assert!(right > left);
        }
    }

    #[test]
    fn test_min_max() {
        let (mesh, ops) = make_mesh_and_ops();
        let bathy = Bathymetry1D::from_function(&mesh, &ops, |x| x);

        let min = bathy.min();
        let max = bathy.max();

        assert!(min < 0.1); // Near x = 0
        assert!(max > 9.9); // Near x = 10
    }

    #[test]
    fn test_gaussian_bump_profile() {
        let b = profiles::gaussian_bump(5.0, 5.0, 1.0, 1.0);
        assert!((b - 1.0).abs() < TOL); // At center, B = amplitude

        let b_off = profiles::gaussian_bump(0.0, 5.0, 1.0, 1.0);
        assert!(b_off < 1e-5); // Far from center, B ≈ 0
    }

    #[test]
    fn test_linear_slope_profile() {
        let b = profiles::linear_slope(5.0, 0.0, 10.0, 0.0, 1.0);
        assert!((b - 0.5).abs() < TOL); // Midpoint

        let b_left = profiles::linear_slope(0.0, 0.0, 10.0, 0.0, 1.0);
        assert!(b_left.abs() < TOL);

        let b_right = profiles::linear_slope(10.0, 0.0, 10.0, 0.0, 1.0);
        assert!((b_right - 1.0).abs() < TOL);
    }

    #[test]
    fn test_parabolic_bowl_profile() {
        let h_0 = 1.0;
        let a = 5.0;

        let b_center = profiles::parabolic_bowl(5.0, 5.0, a, h_0);
        assert!(b_center.abs() < TOL); // At center, B = 0

        let b_edge = profiles::parabolic_bowl(10.0, 5.0, a, h_0);
        assert!((b_edge - h_0).abs() < TOL); // At edge, B = h_0
    }
}
