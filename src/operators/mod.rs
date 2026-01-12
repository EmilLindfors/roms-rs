//! DG operators: differentiation, mass, LIFT matrices.
//!
//! This module provides:
//! - 1D operators for interval elements (`DGOperators1D`)
//! - 2D operators for tensor-product quadrilateral elements (`DGOperators2D`)
//! - Geometric factors for coordinate transformations (`GeometricFactors2D`)

mod differentiation;
mod geometric;
mod lift;
mod mass;
mod operators_2d;

pub use differentiation::{differentiation_matrix, strong_differentiation_matrix};
pub use geometric::GeometricFactors2D;
pub use lift::lift_matrix;
pub use mass::{mass_matrix, mass_matrix_inv};
pub use operators_2d::{DGOperators2D, FACE_NORMALS};

use crate::basis::Vandermonde;
use crate::polynomial::{gauss_lobatto_nodes, gauss_lobatto_weights};
use faer::Mat;

/// All DG operators for 1D elements bundled together.
#[derive(Clone)]
pub struct DGOperators1D {
    /// Polynomial order
    pub order: usize,
    /// Number of nodes per element (order + 1)
    pub n_nodes: usize,
    /// Reference nodes in [-1, 1]
    pub nodes: Vec<f64>,
    /// Quadrature weights
    pub weights: Vec<f64>,
    /// Differentiation matrix: Dr[i,j] = dφ_j/dr at node i
    pub dr: Mat<f64>,
    /// Strong form differentiation: D = M^{-1} * Dr^T * M
    pub d_strong: Mat<f64>,
    /// Mass matrix (diagonal for Gauss-Lobatto)
    pub mass: Mat<f64>,
    /// Inverse mass matrix
    pub mass_inv: Mat<f64>,
    /// LIFT matrix: maps face values to volume
    pub lift: Mat<f64>,
    /// Vandermonde matrix and its inverse
    pub vandermonde: Vandermonde,
}

impl DGOperators1D {
    /// Create DG operators for a given polynomial order.
    pub fn new(order: usize) -> Self {
        let n_nodes = order + 1;

        // Generate Gauss-Lobatto nodes and weights
        let nodes = gauss_lobatto_nodes(order);
        let weights = gauss_lobatto_weights(order, &nodes);

        // Build Vandermonde matrix
        let vandermonde = Vandermonde::new(order, &nodes);

        // Build operators
        let dr = differentiation_matrix(&vandermonde);
        let d_strong = strong_differentiation_matrix(&dr, &weights);
        let mass = mass_matrix(&weights);
        let mass_inv = mass_matrix_inv(&weights);
        let lift = lift_matrix(&mass_inv);

        Self {
            order,
            n_nodes,
            nodes,
            weights,
            dr,
            d_strong,
            mass,
            mass_inv,
            lift,
            vandermonde,
        }
    }
}
