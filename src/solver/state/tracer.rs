//! Tracer state abstractions for temperature and salinity.
//!
//! Tracers are transported by the flow field but stored separately from
//! the hydrodynamic state. This allows clean separation of concerns and
//! easy extension to additional tracers.
//!
//! The tracers are stored as concentrations (not h*C) for easier interpretation
//! and boundary condition specification.

use std::ops::{Add, Mul, Sub};

use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::types::ElementIndex;

use super::swe_2d::SWESolution2D;

/// Tracer state containing temperature and salinity.
///
/// # Units
/// - Temperature: degrees Celsius (°C)
/// - Salinity: practical salinity units (PSU), approximately g/kg
///
/// # Norwegian coast typical values
/// - Temperature: 4-18°C (seasonal variation)
/// - Salinity: 25-35 PSU (fresher near river mouths)
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TracerState {
    /// Temperature in degrees Celsius
    pub temperature: f64,
    /// Salinity in PSU (practical salinity units)
    pub salinity: f64,
}

impl TracerState {
    /// Create a new tracer state.
    pub fn new(temperature: f64, salinity: f64) -> Self {
        Self {
            temperature,
            salinity,
        }
    }

    /// Create a state with typical Norwegian coastal water values.
    ///
    /// Default: T = 8°C, S = 34 PSU (Atlantic water influence)
    pub fn norwegian_coastal() -> Self {
        Self {
            temperature: 8.0,
            salinity: 34.0,
        }
    }

    /// Create a state for fresh river water.
    ///
    /// Default: T = 6°C, S = 0 PSU
    pub fn freshwater(temperature: f64) -> Self {
        Self {
            temperature,
            salinity: 0.0,
        }
    }

    /// Create a state for Atlantic water.
    ///
    /// T = 7-8°C, S = 35 PSU
    pub fn atlantic_water() -> Self {
        Self {
            temperature: 7.5,
            salinity: 35.0,
        }
    }

    /// Create a zero state (used for computations).
    pub fn zero() -> Self {
        Self {
            temperature: 0.0,
            salinity: 0.0,
        }
    }

    /// Convert to array representation [T, S].
    pub fn to_array(&self) -> [f64; 2] {
        [self.temperature, self.salinity]
    }

    /// Create from array representation [T, S].
    pub fn from_array(arr: [f64; 2]) -> Self {
        Self {
            temperature: arr[0],
            salinity: arr[1],
        }
    }

    /// Check if values are physically reasonable.
    ///
    /// Temperature should be > -2°C (seawater freezing) and < 40°C.
    /// Salinity should be >= 0 and <= 42 PSU.
    pub fn is_valid(&self) -> bool {
        self.temperature > -2.0
            && self.temperature < 40.0
            && self.salinity >= 0.0
            && self.salinity <= 42.0
    }

    /// Clamp values to physically reasonable ranges.
    pub fn clamp(&mut self) {
        self.temperature = self.temperature.clamp(-2.0, 40.0);
        self.salinity = self.salinity.clamp(0.0, 42.0);
    }
}

impl Add for TracerState {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            temperature: self.temperature + other.temperature,
            salinity: self.salinity + other.salinity,
        }
    }
}

impl Sub for TracerState {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            temperature: self.temperature - other.temperature,
            salinity: self.salinity - other.salinity,
        }
    }
}

impl Mul<f64> for TracerState {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            temperature: self.temperature * scalar,
            salinity: self.salinity * scalar,
        }
    }
}

impl Mul<TracerState> for f64 {
    type Output = TracerState;

    fn mul(self, state: TracerState) -> TracerState {
        TracerState {
            temperature: self * state.temperature,
            salinity: self * state.salinity,
        }
    }
}

/// Conservative tracer state: (hT, hS).
///
/// Used for the actual transport equations where we solve:
/// ∂(hT)/∂t + ∇·(hT**u**) = diffusion + sources
/// ∂(hS)/∂t + ∇·(hS**u**) = diffusion + sources
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConservativeTracerState {
    /// h * Temperature
    pub h_t: f64,
    /// h * Salinity
    pub h_s: f64,
}

impl ConservativeTracerState {
    /// Create a new conservative tracer state.
    pub fn new(h_t: f64, h_s: f64) -> Self {
        Self { h_t, h_s }
    }

    /// Create from water depth and tracer concentrations.
    pub fn from_depth_and_tracers(h: f64, tracers: TracerState) -> Self {
        Self {
            h_t: h * tracers.temperature,
            h_s: h * tracers.salinity,
        }
    }

    /// Extract tracer concentrations given water depth.
    ///
    /// Uses desingularization to avoid division by zero.
    pub fn to_concentrations(&self, h: f64, h_min: f64) -> TracerState {
        let h_safe = h.max(h_min);
        if h > h_min {
            TracerState {
                temperature: self.h_t / h_safe,
                salinity: self.h_s / h_safe,
            }
        } else {
            // Dry cell - return background values
            TracerState::zero()
        }
    }

    /// Create a zero state.
    pub fn zero() -> Self {
        Self { h_t: 0.0, h_s: 0.0 }
    }

    /// Convert to array representation [hT, hS].
    pub fn to_array(&self) -> [f64; 2] {
        [self.h_t, self.h_s]
    }

    /// Create from array representation [hT, hS].
    pub fn from_array(arr: [f64; 2]) -> Self {
        Self {
            h_t: arr[0],
            h_s: arr[1],
        }
    }
}

impl Add for ConservativeTracerState {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            h_t: self.h_t + other.h_t,
            h_s: self.h_s + other.h_s,
        }
    }
}

impl Sub for ConservativeTracerState {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            h_t: self.h_t - other.h_t,
            h_s: self.h_s - other.h_s,
        }
    }
}

impl Mul<f64> for ConservativeTracerState {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            h_t: self.h_t * scalar,
            h_s: self.h_s * scalar,
        }
    }
}

impl Mul<ConservativeTracerState> for f64 {
    type Output = ConservativeTracerState;

    fn mul(self, state: ConservativeTracerState) -> ConservativeTracerState {
        ConservativeTracerState {
            h_t: self * state.h_t,
            h_s: self * state.h_s,
        }
    }
}

/// Solution storage for 2D tracer fields.
///
/// Stores nodal values in interleaved layout for cache efficiency.
/// Uses conservative variables (hT, hS) internally.
#[derive(Clone)]
pub struct TracerSolution2D {
    /// Nodal values in interleaved layout: [hT, hS] per node
    pub data: Vec<f64>,
    /// Number of elements
    pub n_elements: usize,
    /// Number of nodes per element
    pub n_nodes: usize,
}

impl TracerSolution2D {
    /// Number of tracer variables (T, S).
    pub const N_VARS: usize = 2;

    /// Create a new solution storage initialized to zero.
    pub fn new(n_elements: usize, n_nodes: usize) -> Self {
        Self {
            data: vec![0.0; n_elements * n_nodes * Self::N_VARS],
            n_elements,
            n_nodes,
        }
    }

    /// Create from pre-allocated data vector.
    ///
    /// Used by parallel implementations to construct from collected results.
    pub fn from_data(data: Vec<f64>, n_elements: usize, n_nodes: usize) -> Self {
        debug_assert_eq!(data.len(), n_elements * n_nodes * Self::N_VARS);
        Self {
            data,
            n_elements,
            n_nodes,
        }
    }

    /// Create storage initialized with uniform tracer values.
    pub fn uniform(n_elements: usize, n_nodes: usize, h: f64, tracers: TracerState) -> Self {
        let mut sol = Self::new(n_elements, n_nodes);
        let cons = ConservativeTracerState::from_depth_and_tracers(h, tracers);
        for k in ElementIndex::iter(n_elements) {
            for i in 0..n_nodes {
                sol.set_conservative(k, i, cons);
            }
        }
        sol
    }

    /// Get the conservative state at node i in element k.
    pub fn get_conservative(&self, k: ElementIndex, i: usize) -> ConservativeTracerState {
        let base = (k.as_usize() * self.n_nodes + i) * Self::N_VARS;
        ConservativeTracerState::from_array([self.data[base], self.data[base + 1]])
    }

    /// Set the conservative state at node i in element k.
    pub fn set_conservative(&mut self, k: ElementIndex, i: usize, state: ConservativeTracerState) {
        let base = (k.as_usize() * self.n_nodes + i) * Self::N_VARS;
        let arr = state.to_array();
        self.data[base] = arr[0];
        self.data[base + 1] = arr[1];
    }

    /// Get tracer concentrations at node i in element k.
    ///
    /// Requires the water depth h at this node.
    pub fn get_concentrations(&self, k: ElementIndex, i: usize, h: f64, h_min: f64) -> TracerState {
        self.get_conservative(k, i).to_concentrations(h, h_min)
    }

    /// Set from concentrations and water depth at node i in element k.
    pub fn set_from_concentrations(&mut self, k: ElementIndex, i: usize, h: f64, tracers: TracerState) {
        let cons = ConservativeTracerState::from_depth_and_tracers(h, tracers);
        self.set_conservative(k, i, cons);
    }

    /// Get hT (h * Temperature) at a node.
    pub fn get_h_t(&self, k: ElementIndex, i: usize) -> f64 {
        let base = (k.as_usize() * self.n_nodes + i) * Self::N_VARS;
        self.data[base]
    }

    /// Get hS (h * Salinity) at a node.
    pub fn get_h_s(&self, k: ElementIndex, i: usize) -> f64 {
        let base = (k.as_usize() * self.n_nodes + i) * Self::N_VARS;
        self.data[base + 1]
    }

    /// Set hT at a node.
    pub fn set_h_t(&mut self, k: ElementIndex, i: usize, value: f64) {
        let base = (k.as_usize() * self.n_nodes + i) * Self::N_VARS;
        self.data[base] = value;
    }

    /// Set hS at a node.
    pub fn set_h_s(&mut self, k: ElementIndex, i: usize, value: f64) {
        let base = (k.as_usize() * self.n_nodes + i) * Self::N_VARS;
        self.data[base + 1] = value;
    }

    /// Initialize from functions for T(x,y) and S(x,y).
    ///
    /// Also needs h(x,y) or a depth solution to compute conservative variables.
    pub fn set_from_functions<T, S, H>(
        &mut self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        t_fn: T,
        s_fn: S,
        h_fn: H,
    ) where
        T: Fn(f64, f64) -> f64,
        S: Fn(f64, f64) -> f64,
        H: Fn(f64, f64) -> f64,
    {
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [x, y] = mesh.reference_to_physical(k, r, s);
                let h = h_fn(x, y);
                let temp = t_fn(x, y);
                let sal = s_fn(x, y);
                self.set_from_concentrations(k, i, h, TracerState::new(temp, sal));
            }
        }
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

    /// Copy from another solution.
    pub fn copy_from(&mut self, other: &Self) {
        assert_eq!(self.data.len(), other.data.len());
        self.data.copy_from_slice(&other.data);
    }

    /// Integrate hT over the domain (total heat content proxy).
    pub fn integrate_h_t(&self, ops: &DGOperators2D, geom: &GeometricFactors2D) -> f64 {
        let mut integral = 0.0;
        for k in ElementIndex::iter(self.n_elements) {
            let j = geom.det_j[k];
            for (i, &w) in ops.weights.iter().enumerate() {
                integral += w * self.get_h_t(k, i) * j;
            }
        }
        integral
    }

    /// Integrate hS over the domain (total salt content).
    pub fn integrate_h_s(&self, ops: &DGOperators2D, geom: &GeometricFactors2D) -> f64 {
        let mut integral = 0.0;
        for k in ElementIndex::iter(self.n_elements) {
            let j = geom.det_j[k];
            for (i, &w) in ops.weights.iter().enumerate() {
                integral += w * self.get_h_s(k, i) * j;
            }
        }
        integral
    }

    /// Get minimum and maximum temperature in the domain.
    ///
    /// Returns (min_T, max_T). Requires depth for conversion.
    pub fn temperature_range(&self, swe_solution: &SWESolution2D, h_min: f64) -> (f64, f64) {
        let mut min_t = f64::INFINITY;
        let mut max_t = f64::NEG_INFINITY;
        for k in ElementIndex::iter(self.n_elements) {
            for i in 0..self.n_nodes {
                let h = swe_solution.get_var(k, i, 0);
                let tracers = self.get_concentrations(k, i, h, h_min);
                if h > h_min {
                    min_t = min_t.min(tracers.temperature);
                    max_t = max_t.max(tracers.temperature);
                }
            }
        }
        (min_t, max_t)
    }

    /// Get minimum and maximum salinity in the domain.
    pub fn salinity_range(&self, swe_solution: &SWESolution2D, h_min: f64) -> (f64, f64) {
        let mut min_s = f64::INFINITY;
        let mut max_s = f64::NEG_INFINITY;
        for k in ElementIndex::iter(self.n_elements) {
            for i in 0..self.n_nodes {
                let h = swe_solution.get_var(k, i, 0);
                let tracers = self.get_concentrations(k, i, h, h_min);
                if h > h_min {
                    min_s = min_s.min(tracers.salinity);
                    max_s = max_s.max(tracers.salinity);
                }
            }
        }
        (min_s, max_s)
    }

    /// Extract face values for a specific element and face.
    pub fn extract_face_states(
        &self,
        k: ElementIndex,
        face: usize,
        ops: &DGOperators2D,
    ) -> Vec<ConservativeTracerState> {
        ops.face_nodes[face]
            .iter()
            .map(|&i| self.get_conservative(k, i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn test_tracer_state_basic() {
        let state = TracerState::new(10.0, 35.0);
        assert!((state.temperature - 10.0).abs() < TOL);
        assert!((state.salinity - 35.0).abs() < TOL);
    }

    #[test]
    fn test_tracer_state_presets() {
        let coastal = TracerState::norwegian_coastal();
        assert!(coastal.is_valid());

        let fresh = TracerState::freshwater(6.0);
        assert!((fresh.salinity - 0.0).abs() < TOL);
        assert!(fresh.is_valid());

        let atlantic = TracerState::atlantic_water();
        assert!((atlantic.salinity - 35.0).abs() < TOL);
        assert!(atlantic.is_valid());
    }

    #[test]
    fn test_tracer_state_validation() {
        let valid = TracerState::new(10.0, 34.0);
        assert!(valid.is_valid());

        let cold = TracerState::new(-5.0, 34.0);
        assert!(!cold.is_valid());

        let negative_sal = TracerState::new(10.0, -1.0);
        assert!(!negative_sal.is_valid());
    }

    #[test]
    fn test_tracer_state_arithmetic() {
        let a = TracerState::new(10.0, 30.0);
        let b = TracerState::new(5.0, 5.0);

        let sum = a + b;
        assert!((sum.temperature - 15.0).abs() < TOL);
        assert!((sum.salinity - 35.0).abs() < TOL);

        let diff = a - b;
        assert!((diff.temperature - 5.0).abs() < TOL);
        assert!((diff.salinity - 25.0).abs() < TOL);

        let scaled = a * 0.5;
        assert!((scaled.temperature - 5.0).abs() < TOL);
        assert!((scaled.salinity - 15.0).abs() < TOL);
    }

    #[test]
    fn test_conservative_tracer_state() {
        let h = 10.0;
        let tracers = TracerState::new(8.0, 34.0);
        let cons = ConservativeTracerState::from_depth_and_tracers(h, tracers);

        assert!((cons.h_t - 80.0).abs() < TOL);
        assert!((cons.h_s - 340.0).abs() < TOL);

        // Convert back
        let h_min = 1e-6;
        let recovered = cons.to_concentrations(h, h_min);
        assert!((recovered.temperature - 8.0).abs() < TOL);
        assert!((recovered.salinity - 34.0).abs() < TOL);
    }

    #[test]
    fn test_conservative_dry_cell() {
        let cons = ConservativeTracerState::new(0.0, 0.0);
        let h_min = 1e-6;
        let tracers = cons.to_concentrations(0.0, h_min);

        // Should return zero for dry cells
        assert!((tracers.temperature - 0.0).abs() < TOL);
        assert!((tracers.salinity - 0.0).abs() < TOL);
    }

    // Helper to create ElementIndex in tests
    fn k(idx: usize) -> ElementIndex {
        ElementIndex::new(idx)
    }

    #[test]
    fn test_tracer_solution_basic() {
        let n_elem = 4;
        let n_nodes = 9;
        let mut sol = TracerSolution2D::new(n_elem, n_nodes);

        assert_eq!(sol.data.len(), n_elem * n_nodes * 2);

        // Set and get
        let h = 5.0;
        let tracers = TracerState::new(12.0, 33.0);
        sol.set_from_concentrations(k(0), 0, h, tracers);

        let cons = sol.get_conservative(k(0), 0);
        assert!((cons.h_t - 60.0).abs() < TOL);
        assert!((cons.h_s - 165.0).abs() < TOL);

        let recovered = sol.get_concentrations(k(0), 0, h, 1e-6);
        assert!((recovered.temperature - 12.0).abs() < TOL);
        assert!((recovered.salinity - 33.0).abs() < TOL);
    }

    #[test]
    fn test_tracer_solution_uniform() {
        let n_elem = 2;
        let n_nodes = 4;
        let h = 10.0;
        let tracers = TracerState::norwegian_coastal();

        let sol = TracerSolution2D::uniform(n_elem, n_nodes, h, tracers);

        for k in ElementIndex::iter(n_elem) {
            for i in 0..n_nodes {
                let recovered = sol.get_concentrations(k, i, h, 1e-6);
                assert!((recovered.temperature - tracers.temperature).abs() < TOL);
                assert!((recovered.salinity - tracers.salinity).abs() < TOL);
            }
        }
    }

    #[test]
    fn test_tracer_solution_axpy() {
        let n_elem = 2;
        let n_nodes = 3;

        let mut a = TracerSolution2D::new(n_elem, n_nodes);
        let mut b = TracerSolution2D::new(n_elem, n_nodes);

        for i in 0..a.data.len() {
            a.data[i] = 1.0;
            b.data[i] = 2.0;
        }

        a.axpy(0.5, &b); // a = a + 0.5 * b = 1 + 1 = 2

        for &v in &a.data {
            assert!((v - 2.0).abs() < TOL);
        }
    }
}
