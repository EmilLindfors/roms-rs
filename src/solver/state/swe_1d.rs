//! 1D Shallow water state types.
//!
//! For shallow water equations, we have state (h, hu) where:
//! - h = water depth
//! - hu = momentum (h * velocity)

use std::ops::{Add, Mul, Sub};

use crate::mesh::Mesh1D;
use crate::operators::DGOperators1D;
use crate::types::ElementIndex;

use super::super::core::SystemSolution;

/// Shallow water state: (h, hu).
///
/// Stores the conserved variables for 1D shallow water equations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SWEState {
    /// Water depth h (must be non-negative)
    pub h: f64,
    /// Momentum hu = h * u
    pub hu: f64,
}

impl SWEState {
    /// Create a new SWE state.
    pub fn new(h: f64, hu: f64) -> Self {
        Self { h, hu }
    }

    /// Create a state from primitive variables (h, u).
    pub fn from_primitives(h: f64, u: f64) -> Self {
        Self { h, hu: h * u }
    }

    /// Compute velocity u = hu / h with dry cell protection.
    ///
    /// Uses desingularization to avoid division by zero:
    /// u = 2 * h * hu / (h^2 + max(h, h_min)^2)
    pub fn velocity(&self, h_min: f64) -> f64 {
        let h_reg = self.h.max(h_min);
        2.0 * self.h * self.hu / (self.h * self.h + h_reg * h_reg)
    }

    /// Compute velocity without desingularization (for wet cells).
    ///
    /// Returns 0.0 if h <= h_min to avoid NaN.
    pub fn velocity_simple(&self, h_min: f64) -> f64 {
        if self.h > h_min {
            self.hu / self.h
        } else {
            0.0
        }
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
    pub fn zero() -> Self {
        Self { h: 0.0, hu: 0.0 }
    }

    /// Convert to array representation [h, hu].
    pub fn to_array(&self) -> [f64; 2] {
        [self.h, self.hu]
    }

    /// Create from array representation [h, hu].
    pub fn from_array(arr: [f64; 2]) -> Self {
        Self {
            h: arr[0],
            hu: arr[1],
        }
    }
}

impl Add for SWEState {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            h: self.h + other.h,
            hu: self.hu + other.hu,
        }
    }
}

impl Sub for SWEState {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            h: self.h - other.h,
            hu: self.hu - other.hu,
        }
    }
}

impl Mul<f64> for SWEState {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            h: self.h * scalar,
            hu: self.hu * scalar,
        }
    }
}

impl Mul<SWEState> for f64 {
    type Output = SWEState;

    fn mul(self, state: SWEState) -> SWEState {
        SWEState {
            h: self * state.h,
            hu: self * state.hu,
        }
    }
}

/// Specialized solution for 2-variable systems (like shallow water).
pub type SWESolution = SystemSolution<2>;

impl SWESolution {
    /// Get SWE state at node i in element k.
    pub fn get_state(&self, k: ElementIndex, i: usize) -> SWEState {
        let arr = self.get(k, i);
        SWEState::from_array(arr)
    }

    /// Set SWE state at node i in element k.
    pub fn set_state(&mut self, k: ElementIndex, i: usize, state: SWEState) {
        self.set(k, i, state.to_array());
    }

    /// Initialize from functions for h and u.
    ///
    /// Evaluates h(x) and u(x) at each physical node location
    /// and stores (h, h*u).
    pub fn set_from_functions<H, U>(&mut self, mesh: &Mesh1D, ops: &DGOperators1D, h_fn: H, u_fn: U)
    where
        H: Fn(f64) -> f64,
        U: Fn(f64) -> f64,
    {
        for k in ElementIndex::iter(mesh.n_elements) {
            for (i, &r) in ops.nodes.iter().enumerate() {
                let x = mesh.reference_to_physical(k.as_usize(), r);
                let h = h_fn(x);
                let u = u_fn(x);
                self.set_state(k, i, SWEState::from_primitives(h, u));
            }
        }
    }

    /// Initialize from a single state function.
    pub fn set_from_state_function<F>(&mut self, mesh: &Mesh1D, ops: &DGOperators1D, f: F)
    where
        F: Fn(f64) -> SWEState,
    {
        for k in ElementIndex::iter(mesh.n_elements) {
            for (i, &r) in ops.nodes.iter().enumerate() {
                let x = mesh.reference_to_physical(k.as_usize(), r);
                self.set_state(k, i, f(x));
            }
        }
    }

    /// Compute the integral of water depth h over the domain.
    ///
    /// Uses quadrature weights and Jacobian for accurate integration:
    /// integral h dx = sum_k sum_i w_i * h_{k,i} * J_k
    pub fn integrate_depth(&self, mesh: &Mesh1D, ops: &DGOperators1D) -> f64 {
        let mut integral = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            let j = mesh.jacobian(k.as_usize());
            for (i, &w) in ops.weights.iter().enumerate() {
                let h = self.get_var(k, i, 0);
                integral += w * h * j;
            }
        }
        integral
    }

    /// Compute the integral of momentum hu over the domain.
    pub fn integrate_momentum(&self, mesh: &Mesh1D, ops: &DGOperators1D) -> f64 {
        let mut integral = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            let j = mesh.jacobian(k.as_usize());
            for (i, &w) in ops.weights.iter().enumerate() {
                let hu = self.get_var(k, i, 1);
                integral += w * hu * j;
            }
        }
        integral
    }

    /// Compute L2 error for water depth against exact solution.
    pub fn l2_error_depth<F>(&self, mesh: &Mesh1D, ops: &DGOperators1D, exact: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let mut error_sq = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            let j = mesh.jacobian(k.as_usize());
            for (i, &r) in ops.nodes.iter().enumerate() {
                let x = mesh.reference_to_physical(k.as_usize(), r);
                let h = self.get_var(k, i, 0);
                let h_exact = exact(x);
                let diff = h - h_exact;
                error_sq += ops.weights[i] * diff * diff * j;
            }
        }
        error_sq.sqrt()
    }

    /// Compute L-infinity error for water depth.
    pub fn linf_error_depth<F>(&self, mesh: &Mesh1D, ops: &DGOperators1D, exact: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let mut max_error: f64 = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            for (i, &r) in ops.nodes.iter().enumerate() {
                let x = mesh.reference_to_physical(k.as_usize(), r);
                let h = self.get_var(k, i, 0);
                let h_exact = exact(x);
                max_error = max_error.max((h - h_exact).abs());
            }
        }
        max_error
    }

    /// Check if any cell has negative depth.
    pub fn has_negative_depth(&self) -> bool {
        for k in ElementIndex::iter(self.n_elements) {
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
        for k in ElementIndex::iter(self.n_elements) {
            for i in 0..self.n_nodes {
                min_h = min_h.min(self.get_var(k, i, 0));
            }
        }
        min_h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ElementIndex;

    fn k(idx: usize) -> ElementIndex {
        ElementIndex::new(idx)
    }

    #[test]
    fn test_swe_state_basic() {
        let state = SWEState::new(2.0, 3.0);
        assert!((state.h - 2.0).abs() < 1e-14);
        assert!((state.hu - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_swe_state_from_primitives() {
        let state = SWEState::from_primitives(2.0, 1.5);
        assert!((state.h - 2.0).abs() < 1e-14);
        assert!((state.hu - 3.0).abs() < 1e-14); // hu = h * u = 2 * 1.5
    }

    #[test]
    fn test_swe_state_velocity() {
        let state = SWEState::new(2.0, 3.0);
        let h_min = 1e-6;

        // Simple velocity
        let u = state.velocity_simple(h_min);
        assert!((u - 1.5).abs() < 1e-14);

        // Regularized velocity should be very close for wet cells
        let u_reg = state.velocity(h_min);
        assert!((u_reg - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_swe_state_velocity_dry() {
        let state = SWEState::new(1e-10, 1e-10);
        let h_min = 1e-6;

        // Simple velocity returns 0 for dry cells
        let u = state.velocity_simple(h_min);
        assert!((u - 0.0).abs() < 1e-14);

        // Regularized velocity should be finite
        let u_reg = state.velocity(h_min);
        assert!(u_reg.is_finite());
    }

    #[test]
    fn test_swe_state_arithmetic() {
        let a = SWEState::new(1.0, 2.0);
        let b = SWEState::new(3.0, 4.0);

        let sum = a + b;
        assert!((sum.h - 4.0).abs() < 1e-14);
        assert!((sum.hu - 6.0).abs() < 1e-14);

        let diff = a - b;
        assert!((diff.h - (-2.0)).abs() < 1e-14);
        assert!((diff.hu - (-2.0)).abs() < 1e-14);

        let scaled = a * 2.0;
        assert!((scaled.h - 2.0).abs() < 1e-14);
        assert!((scaled.hu - 4.0).abs() < 1e-14);

        let scaled2 = 3.0 * a;
        assert!((scaled2.h - 3.0).abs() < 1e-14);
        assert!((scaled2.hu - 6.0).abs() < 1e-14);
    }

    #[test]
    fn test_swe_state_array_conversion() {
        let state = SWEState::new(1.5, 2.5);
        let arr = state.to_array();
        assert!((arr[0] - 1.5).abs() < 1e-14);
        assert!((arr[1] - 2.5).abs() < 1e-14);

        let state2 = SWEState::from_array(arr);
        assert_eq!(state, state2);
    }

    #[test]
    fn test_system_solution_basic() {
        let n_elem = 4;
        let n_nodes = 3;
        let mut sol = SWESolution::new(n_elem, n_nodes);

        assert_eq!(sol.data.len(), n_elem * n_nodes * 2);

        // Set and get values
        sol.set(k(0), 0, [1.0, 2.0]);
        let vals = sol.get(k(0), 0);
        assert!((vals[0] - 1.0).abs() < 1e-14);
        assert!((vals[1] - 2.0).abs() < 1e-14);

        // Set and get state
        let state = SWEState::new(3.0, 4.0);
        sol.set_state(k(1), 2, state);
        let state2 = sol.get_state(k(1), 2);
        assert_eq!(state, state2);
    }

    #[test]
    fn test_system_solution_axpy() {
        let n_elem = 2;
        let n_nodes = 3;

        let mut a = SWESolution::new(n_elem, n_nodes);
        let mut b = SWESolution::new(n_elem, n_nodes);

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
        let mut sol = SWESolution::new(n_elem, n_nodes);

        // Set some values
        for i in 0..n_nodes {
            sol.set(k(0), i, [i as f64, (i + 10) as f64]);
        }

        // Get depth values for element 0
        let h_vals = sol.element_var(k(0), 0);
        assert_eq!(h_vals.len(), n_nodes);
        for i in 0..n_nodes {
            assert!((h_vals[i] - i as f64).abs() < 1e-14);
        }

        // Get momentum values for element 0
        let hu_vals = sol.element_var(k(0), 1);
        for i in 0..n_nodes {
            assert!((hu_vals[i] - (i + 10) as f64).abs() < 1e-14);
        }
    }

    #[test]
    fn test_swe_solution_negative_depth() {
        let mut sol = SWESolution::new(2, 3);
        assert!(!sol.has_negative_depth());

        sol.set_var(k(0), 0, 0, -0.1);
        assert!(sol.has_negative_depth());
    }

    #[test]
    fn test_swe_solution_min_depth() {
        let mut sol = SWESolution::new(2, 3);
        sol.set_var(k(0), 0, 0, 1.0);
        sol.set_var(k(0), 1, 0, 2.0);
        sol.set_var(k(1), 0, 0, 0.5);

        assert!((sol.min_depth() - 0.0).abs() < 1e-14); // Others are still 0
    }
}
