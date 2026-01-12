//! State vector abstractions for 2D systems of equations.
//!
//! For 2D shallow water equations, we have state (h, hu, hv) where:
//! - h = water depth
//! - hu = x-momentum (h * u)
//! - hv = y-momentum (h * v)

use std::ops::{Add, Mul, Sub};

use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};

/// 2D Shallow water state: (h, hu, hv).
///
/// Stores the conserved variables for 2D shallow water equations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SWEState2D {
    /// Water depth h (must be non-negative)
    pub h: f64,
    /// x-momentum hu = h * u
    pub hu: f64,
    /// y-momentum hv = h * v
    pub hv: f64,
}

impl SWEState2D {
    /// Create a new 2D SWE state.
    #[inline(always)]
    pub fn new(h: f64, hu: f64, hv: f64) -> Self {
        Self { h, hu, hv }
    }

    /// Create a state from primitive variables (h, u, v).
    #[inline(always)]
    pub fn from_primitives(h: f64, u: f64, v: f64) -> Self {
        Self {
            h,
            hu: h * u,
            hv: h * v,
        }
    }

    /// Compute velocity components (u, v) with dry cell protection.
    ///
    /// Uses desingularization to avoid division by zero:
    /// u = 2 * h * hu / (h² + max(h, h_min)²)
    #[inline(always)]
    pub fn velocity(&self, h_min: f64) -> (f64, f64) {
        let h_reg = self.h.max(h_min);
        let denom = self.h * self.h + h_reg * h_reg;
        let inv_denom = 1.0 / denom;
        let factor = 2.0 * self.h * inv_denom;
        (self.hu * factor, self.hv * factor)
    }

    /// Compute velocity without desingularization (for wet cells).
    ///
    /// Returns (0, 0) if h <= h_min to avoid NaN.
    #[inline(always)]
    pub fn velocity_simple(&self, h_min: f64) -> (f64, f64) {
        if self.h > h_min {
            let h_inv = 1.0 / self.h;
            (self.hu * h_inv, self.hv * h_inv)
        } else {
            (0.0, 0.0)
        }
    }

    /// Compute velocity magnitude |u| = sqrt(u² + v²).
    pub fn velocity_magnitude(&self, h_min: f64) -> f64 {
        let (u, v) = self.velocity(h_min);
        (u * u + v * v).sqrt()
    }

    /// Total water surface elevation eta = h + B.
    pub fn surface_elevation(&self, bathymetry: f64) -> f64 {
        self.h + bathymetry
    }

    /// Check if this cell is "dry" (h < h_min).
    pub fn is_dry(&self, h_min: f64) -> bool {
        self.h < h_min
    }

    /// Create a zero state.
    #[inline(always)]
    pub fn zero() -> Self {
        Self {
            h: 0.0,
            hu: 0.0,
            hv: 0.0,
        }
    }

    /// Convert to array representation [h, hu, hv].
    #[inline(always)]
    pub fn to_array(&self) -> [f64; 3] {
        [self.h, self.hu, self.hv]
    }

    /// Create from array representation [h, hu, hv].
    #[inline(always)]
    pub fn from_array(arr: [f64; 3]) -> Self {
        Self {
            h: arr[0],
            hu: arr[1],
            hv: arr[2],
        }
    }

    /// Compute the wave celerity c = sqrt(g * h).
    pub fn celerity(&self, g: f64) -> f64 {
        (g * self.h.max(0.0)).sqrt()
    }

    /// Compute the maximum wave speed |u| + c.
    pub fn max_wave_speed(&self, g: f64, h_min: f64) -> f64 {
        let (u, v) = self.velocity(h_min);
        let speed = (u * u + v * v).sqrt();
        let c = self.celerity(g);
        speed + c
    }

    /// Compute the Froude number Fr = |u| / c.
    pub fn froude_number(&self, g: f64, h_min: f64) -> f64 {
        let c = self.celerity(g);
        if c > h_min {
            self.velocity_magnitude(h_min) / c
        } else {
            0.0
        }
    }
}

impl Add for SWEState2D {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            h: self.h + other.h,
            hu: self.hu + other.hu,
            hv: self.hv + other.hv,
        }
    }
}

impl Sub for SWEState2D {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            h: self.h - other.h,
            hu: self.hu - other.hu,
            hv: self.hv - other.hv,
        }
    }
}

impl Mul<f64> for SWEState2D {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            h: self.h * scalar,
            hu: self.hu * scalar,
            hv: self.hv * scalar,
        }
    }
}

impl Mul<SWEState2D> for f64 {
    type Output = SWEState2D;

    fn mul(self, state: SWEState2D) -> SWEState2D {
        SWEState2D {
            h: self * state.h,
            hu: self * state.hu,
            hv: self * state.hv,
        }
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
    pub fn get(&self, k: usize, i: usize) -> [f64; N] {
        let base = (k * self.n_nodes + i) * N;
        let mut result = [0.0; N];
        result[..N].copy_from_slice(&self.data[base..base + N]);
        result
    }

    /// Set the state at node i in element k.
    #[inline(always)]
    pub fn set(&mut self, k: usize, i: usize, values: [f64; N]) {
        let base = (k * self.n_nodes + i) * N;
        self.data[base..base + N].copy_from_slice(&values[..N]);
    }

    /// Get a single variable at node i in element k.
    #[inline(always)]
    pub fn get_var(&self, k: usize, i: usize, var: usize) -> f64 {
        let base = (k * self.n_nodes + i) * N;
        self.data[base + var]
    }

    /// Set a single variable at node i in element k.
    #[inline(always)]
    pub fn set_var(&mut self, k: usize, i: usize, var: usize, value: f64) {
        let base = (k * self.n_nodes + i) * N;
        self.data[base + var] = value;
    }

    /// Get direct slice access to element data for batch operations.
    #[inline(always)]
    pub fn element_data(&self, k: usize) -> &[f64] {
        let start = k * self.n_nodes * N;
        let end = start + self.n_nodes * N;
        &self.data[start..end]
    }

    /// Get mutable direct slice access to element data for batch operations.
    #[inline(always)]
    pub fn element_data_mut(&mut self, k: usize) -> &mut [f64] {
        let start = k * self.n_nodes * N;
        let end = start + self.n_nodes * N;
        &mut self.data[start..end]
    }

    /// Get all nodal values for a single variable in element k.
    ///
    /// Returns a vector since we can't return a strided slice.
    pub fn element_var(&self, k: usize, var: usize) -> Vec<f64> {
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
        for k in 0..self.n_elements {
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

/// Specialized solution for 2D SWE (3-variable system: h, hu, hv).
pub type SWESolution2D = SystemSolution2D<3>;

impl SWESolution2D {
    /// Get SWE state at node i in element k.
    #[inline(always)]
    pub fn get_state(&self, k: usize, i: usize) -> SWEState2D {
        let base = (k * self.n_nodes + i) * 3;
        SWEState2D {
            h: self.data[base],
            hu: self.data[base + 1],
            hv: self.data[base + 2],
        }
    }

    /// Set SWE state at node i in element k.
    #[inline(always)]
    pub fn set_state(&mut self, k: usize, i: usize, state: SWEState2D) {
        let base = (k * self.n_nodes + i) * 3;
        self.data[base] = state.h;
        self.data[base + 1] = state.hu;
        self.data[base + 2] = state.hv;
    }

    /// Initialize from functions for h, u, and v.
    ///
    /// Evaluates h(x, y), u(x, y), v(x, y) at each physical node location
    /// and stores (h, h*u, h*v).
    pub fn set_from_functions<H, U, V>(
        &mut self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        h_fn: H,
        u_fn: U,
        v_fn: V,
    ) where
        H: Fn(f64, f64) -> f64,
        U: Fn(f64, f64) -> f64,
        V: Fn(f64, f64) -> f64,
    {
        for k in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let (x, y) = mesh.reference_to_physical(k, r, s);
                let h = h_fn(x, y);
                let u = u_fn(x, y);
                let v = v_fn(x, y);
                self.set_state(k, i, SWEState2D::from_primitives(h, u, v));
            }
        }
    }

    /// Initialize from a single state function.
    pub fn set_from_state_function<F>(&mut self, mesh: &Mesh2D, ops: &DGOperators2D, f: F)
    where
        F: Fn(f64, f64) -> SWEState2D,
    {
        for k in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let (x, y) = mesh.reference_to_physical(k, r, s);
                self.set_state(k, i, f(x, y));
            }
        }
    }

    /// Compute the integral of water depth h over the domain.
    ///
    /// Uses quadrature weights and Jacobian for accurate integration:
    /// ∫∫ h dA = Σ_k Σ_i w_i * h_{k,i} * J_k
    pub fn integrate_depth(&self, ops: &DGOperators2D, geom: &GeometricFactors2D) -> f64 {
        let mut integral = 0.0;
        for k in 0..self.n_elements {
            let j = geom.det_j[k];
            for (i, &w) in ops.weights.iter().enumerate() {
                let h = self.get_var(k, i, 0);
                integral += w * h * j;
            }
        }
        integral
    }

    /// Compute the integral of x-momentum hu over the domain.
    pub fn integrate_x_momentum(&self, ops: &DGOperators2D, geom: &GeometricFactors2D) -> f64 {
        let mut integral = 0.0;
        for k in 0..self.n_elements {
            let j = geom.det_j[k];
            for (i, &w) in ops.weights.iter().enumerate() {
                let hu = self.get_var(k, i, 1);
                integral += w * hu * j;
            }
        }
        integral
    }

    /// Compute the integral of y-momentum hv over the domain.
    pub fn integrate_y_momentum(&self, ops: &DGOperators2D, geom: &GeometricFactors2D) -> f64 {
        let mut integral = 0.0;
        for k in 0..self.n_elements {
            let j = geom.det_j[k];
            for (i, &w) in ops.weights.iter().enumerate() {
                let hv = self.get_var(k, i, 2);
                integral += w * hv * j;
            }
        }
        integral
    }

    /// Compute L2 error for water depth against exact solution.
    pub fn l2_error_depth<F>(
        &self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        exact: F,
    ) -> f64
    where
        F: Fn(f64, f64) -> f64,
    {
        let mut error_sq = 0.0;
        for k in 0..mesh.n_elements {
            let j = geom.det_j[k];
            for i in 0..self.n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let (x, y) = mesh.reference_to_physical(k, r, s);
                let h = self.get_var(k, i, 0);
                let h_exact = exact(x, y);
                let diff = h - h_exact;
                error_sq += ops.weights[i] * diff * diff * j;
            }
        }
        error_sq.sqrt()
    }

    /// Compute L-infinity error for water depth.
    pub fn linf_error_depth<F>(&self, mesh: &Mesh2D, ops: &DGOperators2D, exact: F) -> f64
    where
        F: Fn(f64, f64) -> f64,
    {
        let mut max_error: f64 = 0.0;
        for k in 0..mesh.n_elements {
            for i in 0..self.n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let (x, y) = mesh.reference_to_physical(k, r, s);
                let h = self.get_var(k, i, 0);
                let h_exact = exact(x, y);
                max_error = max_error.max((h - h_exact).abs());
            }
        }
        max_error
    }

    /// Check if any cell has negative depth.
    pub fn has_negative_depth(&self) -> bool {
        for k in 0..self.n_elements {
            for i in 0..self.n_nodes {
                if self.get_var(k, i, 0) < 0.0 {
                    return true;
                }
            }
        }
        false
    }

    /// Get minimum depth across the domain.
    pub fn min_depth(&self) -> f64 {
        let mut min_h = f64::INFINITY;
        for k in 0..self.n_elements {
            for i in 0..self.n_nodes {
                min_h = min_h.min(self.get_var(k, i, 0));
            }
        }
        min_h
    }

    /// Get maximum depth across the domain.
    pub fn max_depth(&self) -> f64 {
        let mut max_h = f64::NEG_INFINITY;
        for k in 0..self.n_elements {
            for i in 0..self.n_nodes {
                max_h = max_h.max(self.get_var(k, i, 0));
            }
        }
        max_h
    }

    /// Extract face values for a specific element and face.
    ///
    /// Returns a vector of SWEState2D for each node on the face.
    pub fn extract_face_states(
        &self,
        k: usize,
        face: usize,
        ops: &DGOperators2D,
    ) -> Vec<SWEState2D> {
        ops.face_nodes[face]
            .iter()
            .map(|&i| self.get_state(k, i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swe_state_2d_basic() {
        let state = SWEState2D::new(2.0, 3.0, 4.0);
        assert!((state.h - 2.0).abs() < 1e-14);
        assert!((state.hu - 3.0).abs() < 1e-14);
        assert!((state.hv - 4.0).abs() < 1e-14);
    }

    #[test]
    fn test_swe_state_2d_from_primitives() {
        let state = SWEState2D::from_primitives(2.0, 1.5, 2.0);
        assert!((state.h - 2.0).abs() < 1e-14);
        assert!((state.hu - 3.0).abs() < 1e-14); // hu = h * u = 2 * 1.5
        assert!((state.hv - 4.0).abs() < 1e-14); // hv = h * v = 2 * 2.0
    }

    #[test]
    fn test_swe_state_2d_velocity() {
        let state = SWEState2D::new(2.0, 3.0, 4.0);
        let h_min = 1e-6;

        // Simple velocity
        let (u, v) = state.velocity_simple(h_min);
        assert!((u - 1.5).abs() < 1e-14);
        assert!((v - 2.0).abs() < 1e-14);

        // Regularized velocity should be very close for wet cells
        let (u_reg, v_reg) = state.velocity(h_min);
        assert!((u_reg - 1.5).abs() < 1e-10);
        assert!((v_reg - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_swe_state_2d_velocity_dry() {
        let state = SWEState2D::new(1e-10, 1e-10, 1e-10);
        let h_min = 1e-6;

        // Simple velocity returns (0, 0) for dry cells
        let (u, v) = state.velocity_simple(h_min);
        assert!((u - 0.0).abs() < 1e-14);
        assert!((v - 0.0).abs() < 1e-14);

        // Regularized velocity should be finite
        let (u_reg, v_reg) = state.velocity(h_min);
        assert!(u_reg.is_finite());
        assert!(v_reg.is_finite());
    }

    #[test]
    fn test_swe_state_2d_wave_speed() {
        let state = SWEState2D::new(1.0, 1.0, 0.0); // h=1, u=1, v=0
        let g = 9.81;
        let h_min = 1e-6;

        let c = state.celerity(g);
        assert!((c - g.sqrt()).abs() < 1e-14);

        let max_speed = state.max_wave_speed(g, h_min);
        assert!((max_speed - (1.0 + g.sqrt())).abs() < 1e-10);
    }

    #[test]
    fn test_swe_state_2d_arithmetic() {
        let a = SWEState2D::new(1.0, 2.0, 3.0);
        let b = SWEState2D::new(4.0, 5.0, 6.0);

        let sum = a + b;
        assert!((sum.h - 5.0).abs() < 1e-14);
        assert!((sum.hu - 7.0).abs() < 1e-14);
        assert!((sum.hv - 9.0).abs() < 1e-14);

        let diff = a - b;
        assert!((diff.h - (-3.0)).abs() < 1e-14);
        assert!((diff.hu - (-3.0)).abs() < 1e-14);
        assert!((diff.hv - (-3.0)).abs() < 1e-14);

        let scaled = a * 2.0;
        assert!((scaled.h - 2.0).abs() < 1e-14);
        assert!((scaled.hu - 4.0).abs() < 1e-14);
        assert!((scaled.hv - 6.0).abs() < 1e-14);

        let scaled2 = 3.0 * a;
        assert!((scaled2.h - 3.0).abs() < 1e-14);
        assert!((scaled2.hu - 6.0).abs() < 1e-14);
        assert!((scaled2.hv - 9.0).abs() < 1e-14);
    }

    #[test]
    fn test_swe_state_2d_array_conversion() {
        let state = SWEState2D::new(1.5, 2.5, 3.5);
        let arr = state.to_array();
        assert!((arr[0] - 1.5).abs() < 1e-14);
        assert!((arr[1] - 2.5).abs() < 1e-14);
        assert!((arr[2] - 3.5).abs() < 1e-14);

        let state2 = SWEState2D::from_array(arr);
        assert_eq!(state, state2);
    }

    #[test]
    fn test_system_solution_2d_basic() {
        let n_elem = 4;
        let n_nodes = 9; // (2+1)²
        let mut sol = SWESolution2D::new(n_elem, n_nodes);

        assert_eq!(sol.data.len(), n_elem * n_nodes * 3);

        // Set and get values
        sol.set(0, 0, [1.0, 2.0, 3.0]);
        let vals = sol.get(0, 0);
        assert!((vals[0] - 1.0).abs() < 1e-14);
        assert!((vals[1] - 2.0).abs() < 1e-14);
        assert!((vals[2] - 3.0).abs() < 1e-14);

        // Set and get state
        let state = SWEState2D::new(4.0, 5.0, 6.0);
        sol.set_state(1, 2, state);
        let state2 = sol.get_state(1, 2);
        assert_eq!(state, state2);
    }

    #[test]
    fn test_system_solution_2d_axpy() {
        let n_elem = 2;
        let n_nodes = 4;

        let mut a = SWESolution2D::new(n_elem, n_nodes);
        let mut b = SWESolution2D::new(n_elem, n_nodes);

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
        let mut sol = SWESolution2D::new(n_elem, n_nodes);

        // Set some values
        for i in 0..n_nodes {
            sol.set(0, i, [i as f64, (i + 10) as f64, (i + 20) as f64]);
        }

        // Get depth values for element 0
        let h_vals = sol.element_var(0, 0);
        assert_eq!(h_vals.len(), n_nodes);
        for i in 0..n_nodes {
            assert!((h_vals[i] - i as f64).abs() < 1e-14);
        }

        // Get x-momentum values for element 0
        let hu_vals = sol.element_var(0, 1);
        for i in 0..n_nodes {
            assert!((hu_vals[i] - (i + 10) as f64).abs() < 1e-14);
        }

        // Get y-momentum values for element 0
        let hv_vals = sol.element_var(0, 2);
        for i in 0..n_nodes {
            assert!((hv_vals[i] - (i + 20) as f64).abs() < 1e-14);
        }
    }

    #[test]
    fn test_swe_solution_2d_negative_depth() {
        let mut sol = SWESolution2D::new(2, 3);
        assert!(!sol.has_negative_depth());

        sol.set_var(0, 0, 0, -0.1);
        assert!(sol.has_negative_depth());
    }

    #[test]
    fn test_swe_solution_2d_min_max_depth() {
        let mut sol = SWESolution2D::new(2, 3);
        sol.set_var(0, 0, 0, 1.0);
        sol.set_var(0, 1, 0, 2.0);
        sol.set_var(1, 0, 0, 0.5);
        sol.set_var(1, 1, 0, 3.0);

        assert!((sol.min_depth() - 0.0).abs() < 1e-14); // Other nodes are still 0
        assert!((sol.max_depth() - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_swe_state_2d_froude_number() {
        // Subcritical flow: Fr < 1
        let state = SWEState2D::from_primitives(1.0, 1.0, 0.0); // h=1, u=1, v=0
        let g = 9.81;
        let h_min = 1e-6;
        let fr = state.froude_number(g, h_min);
        assert!(fr < 1.0);
        assert!((fr - 1.0 / g.sqrt()).abs() < 1e-10);

        // Supercritical flow: Fr > 1
        let state_super = SWEState2D::from_primitives(0.1, 5.0, 0.0); // shallow, fast
        let fr_super = state_super.froude_number(g, h_min);
        assert!(fr_super > 1.0);
    }
}
