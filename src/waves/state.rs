//! The wave model's state: action density per spectral component and DG node.

use crate::time::Integrable;

/// Action density `N(σ, θ)` (m²·s/rad², i.e. variance density over σ) at every DG
/// node for every spectral component, `data[c · n_points + p]` with component
/// `c = i n_θ + j` and node `p = k n_nodes + i` (element-major, as the 2D
/// solutions). Component-major, so that each component is one contiguous nodal
/// field for the DG propagation, and its spectral neighbours are other slices.
#[derive(Clone, Debug)]
pub struct WaveSolution {
    pub n_components: usize,
    pub n_points: usize,
    pub data: Vec<f64>,
}

impl WaveSolution {
    /// No energy anywhere.
    pub fn new(n_components: usize, n_points: usize) -> Self {
        Self {
            n_components,
            n_points,
            data: vec![0.0; n_components * n_points],
        }
    }

    /// The nodal field of component `c`.
    pub fn component(&self, c: usize) -> &[f64] {
        &self.data[c * self.n_points..(c + 1) * self.n_points]
    }

    pub fn component_mut(&mut self, c: usize) -> &mut [f64] {
        &mut self.data[c * self.n_points..(c + 1) * self.n_points]
    }

    /// The spectrum at node `p`, one value per component, into `out`.
    pub fn spectrum_into(&self, p: usize, out: &mut [f64]) {
        for (c, x) in out.iter_mut().enumerate() {
            *x = self.data[c * self.n_points + p];
        }
    }

    /// The same action spectrum `n[c]` at every node.
    pub fn fill_uniform(&mut self, n: &[f64]) {
        assert_eq!(n.len(), self.n_components);
        for (c, &value) in n.iter().enumerate() {
            self.component_mut(c).fill(value);
        }
    }
}

impl Integrable for WaveSolution {
    fn scale(&mut self, c: f64) {
        self.data.iter_mut().for_each(|x| *x *= c);
    }

    fn axpy(&mut self, c: f64, other: &Self) {
        self.data
            .iter_mut()
            .zip(&other.data)
            .for_each(|(x, y)| *x += c * y);
    }
}
