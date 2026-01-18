//! GPU RHS computation for 2D SWE.
//!
//! This module orchestrates the full RHS computation on the GPU,
//! combining volume terms, surface terms, and source terms.

use burn::prelude::*;

use super::connectivity::BurnConnectivity;
use super::flux::hll_flux_batched;
use super::kernels::{
    apply_diff_matrix_batched, combine_derivatives_batched, compute_swe_fluxes,
    coriolis_source_batched, friction_source_batched,
};
use super::operators::BurnOperators2D;
use super::solution::BurnSWESolution2D;
use super::surface::{gather_face_states, get_interior_face_normals};

/// GPU geometric factors for 2D elements.
#[derive(Clone, Debug)]
pub struct BurnGeometricFactors2D<B: Backend> {
    /// rx: dr/dx for each element [n_elements]
    pub rx: Tensor<B, 1>,
    /// sx: ds/dx for each element
    pub sx: Tensor<B, 1>,
    /// ry: dr/dy for each element
    pub ry: Tensor<B, 1>,
    /// sy: ds/dy for each element
    pub sy: Tensor<B, 1>,
    /// Jacobian determinant (det_j) for each element
    pub det_j: Tensor<B, 1>,
    /// Device
    pub device: B::Device,
}

impl<B: Backend> BurnGeometricFactors2D<B>
where
    B::FloatElem: From<f64>,
{
    /// Upload CPU geometric factors to GPU.
    pub fn from_cpu(
        geom: &crate::operators::GeometricFactors2D,
        device: &B::Device,
    ) -> Self {
        let n = geom.rx.len();

        Self {
            rx: Self::tensor_1d(&geom.rx, device),
            sx: Self::tensor_1d(&geom.sx, device),
            ry: Self::tensor_1d(&geom.ry, device),
            sy: Self::tensor_1d(&geom.sy, device),
            det_j: Self::tensor_1d(&geom.det_j, device),
            device: device.clone(),
        }
    }

    fn tensor_1d(data: &[f64], device: &B::Device) -> Tensor<B, 1> {
        let n = data.len();
        let data_converted: Vec<B::FloatElem> =
            data.iter().map(|&x| B::FloatElem::from(x)).collect();
        Tensor::from_data(
            burn::tensor::TensorData::new(data_converted, vec![n]),
            device,
        )
    }
}

/// Configuration for GPU RHS computation.
#[derive(Clone, Debug)]
pub struct BurnRhsConfig {
    /// Gravitational acceleration
    pub g: f64,
    /// Minimum depth for dry cell treatment
    pub h_min: f64,
    /// Coriolis parameter (f-plane, or None for no Coriolis)
    pub coriolis_f: Option<f64>,
    /// Manning friction coefficient squared * g (or None for no friction)
    pub manning_g_n2: Option<f64>,
}

impl Default for BurnRhsConfig {
    fn default() -> Self {
        Self {
            g: 9.81,
            h_min: 1e-4,
            coriolis_f: None,
            manning_g_n2: None,
        }
    }
}

/// Compute the RHS of the 2D SWE on GPU.
///
/// This is the main entry point for GPU-accelerated RHS computation.
/// It computes:
///
/// 1. Volume terms: -∇·F using differentiation matrices
/// 2. Surface terms: ∮ (F* - F)·n dS using Riemann solver + LIFT
/// 3. Source terms: Coriolis, friction, etc.
///
/// # Arguments
/// * `solution` - Current state on GPU
/// * `ops` - GPU operators
/// * `geom` - GPU geometric factors
/// * `connectivity` - Pre-computed connectivity
/// * `config` - Physical parameters
///
/// # Returns
/// RHS solution on GPU
pub fn compute_rhs_swe_2d_burn<B: Backend>(
    solution: &BurnSWESolution2D<B>,
    ops: &BurnOperators2D<B>,
    geom: &BurnGeometricFactors2D<B>,
    connectivity: &BurnConnectivity<B>,
    config: &BurnRhsConfig,
) -> BurnSWESolution2D<B>
where
    B::FloatElem: From<f64>,
    B::IntElem: From<i64>,
    f64: From<B::FloatElem>,
    i64: From<B::IntElem>,
{
    let device = &ops.device;
    let n_elements = solution.n_elements;
    let n_nodes = solution.n_nodes;

    // =========================================================================
    // Step 1: Compute physical fluxes at all nodes
    // =========================================================================
    let (flux_x_h, flux_x_hu, flux_x_hv, flux_y_h, flux_y_hu, flux_y_hv) =
        compute_swe_fluxes(&solution.h, &solution.hu, &solution.hv, config.g, config.h_min);

    // =========================================================================
    // Step 2: Compute volume term: d(flux)/dr and d(flux)/ds
    // =========================================================================
    let dr_t = ops.dr_transpose();
    let ds_t = ops.ds_transpose();

    // Derivatives of x-flux
    let dfx_dr_h = apply_diff_matrix_batched(&dr_t, &flux_x_h);
    let dfx_dr_hu = apply_diff_matrix_batched(&dr_t, &flux_x_hu);
    let dfx_dr_hv = apply_diff_matrix_batched(&dr_t, &flux_x_hv);

    let dfx_ds_h = apply_diff_matrix_batched(&ds_t, &flux_x_h);
    let dfx_ds_hu = apply_diff_matrix_batched(&ds_t, &flux_x_hu);
    let dfx_ds_hv = apply_diff_matrix_batched(&ds_t, &flux_x_hv);

    // Derivatives of y-flux
    let dfy_dr_h = apply_diff_matrix_batched(&dr_t, &flux_y_h);
    let dfy_dr_hu = apply_diff_matrix_batched(&dr_t, &flux_y_hu);
    let dfy_dr_hv = apply_diff_matrix_batched(&dr_t, &flux_y_hv);

    let dfy_ds_h = apply_diff_matrix_batched(&ds_t, &flux_y_h);
    let dfy_ds_hu = apply_diff_matrix_batched(&ds_t, &flux_y_hu);
    let dfy_ds_hv = apply_diff_matrix_batched(&ds_t, &flux_y_hv);

    // Combine derivatives with geometric factors
    let mut rhs_h = combine_derivatives_batched(
        &dfx_dr_h, &dfx_ds_h, &dfy_dr_h, &dfy_ds_h,
        &geom.rx, &geom.sx, &geom.ry, &geom.sy,
    );
    let mut rhs_hu = combine_derivatives_batched(
        &dfx_dr_hu, &dfx_ds_hu, &dfy_dr_hu, &dfy_ds_hu,
        &geom.rx, &geom.sx, &geom.ry, &geom.sy,
    );
    let mut rhs_hv = combine_derivatives_batched(
        &dfx_dr_hv, &dfx_ds_hv, &dfy_dr_hv, &dfy_ds_hv,
        &geom.rx, &geom.sx, &geom.ry, &geom.sy,
    );

    // =========================================================================
    // Step 3: Compute surface terms (Riemann solver + LIFT)
    // =========================================================================
    if connectivity.n_interior_faces > 0 {
        // Gather face states
        let (states_minus, states_plus) = gather_face_states(solution, connectivity);

        // Get face normals
        let (nx, ny) = get_interior_face_normals(connectivity);

        // Compute numerical flux using HLL
        let (f_star_h, f_star_hu, f_star_hv) = hll_flux_batched(
            &states_minus.h, &states_minus.hu, &states_minus.hv,
            &states_plus.h, &states_plus.hu, &states_plus.hv,
            &nx, &ny,
            config.g, config.h_min,
        );

        // Compute physical flux at face for minus side
        // (We need to compute F·n for the minus side to get flux difference)
        // For now, simplified: use half the numerical flux contribution
        // A full implementation would compute F_minus·n and F_plus·n

        // The flux difference F* - F·n needs to be scattered back to elements
        // and multiplied by LIFT. For the initial implementation, we accumulate
        // using CPU code in apply_lift_all_faces.

        // TODO: Implement full surface term computation
        // This requires computing flux differences and applying LIFT for all faces
    }

    // =========================================================================
    // Step 4: Add source terms
    // =========================================================================

    // Coriolis force
    if let Some(f) = config.coriolis_f {
        let (s_hu, s_hv) = coriolis_source_batched(&solution.hu, &solution.hv, f);
        rhs_hu = rhs_hu.add(s_hu);
        rhs_hv = rhs_hv.add(s_hv);
    }

    // Manning friction
    if let Some(g_n2) = config.manning_g_n2 {
        let (s_hu, s_hv) = friction_source_batched(
            &solution.h, &solution.hu, &solution.hv, g_n2, config.h_min,
        );
        rhs_hu = rhs_hu.add(s_hu);
        rhs_hv = rhs_hv.add(s_hv);
    }

    // =========================================================================
    // Step 5: Scale by inverse Jacobian
    // =========================================================================
    // Each element has its own Jacobian determinant
    let inv_j = geom.det_j.clone().recip().unsqueeze_dim(1); // [E, 1]

    rhs_h = rhs_h.mul(inv_j.clone());
    rhs_hu = rhs_hu.mul(inv_j.clone());
    rhs_hv = rhs_hv.mul(inv_j);

    // Return RHS as a solution structure
    BurnSWESolution2D {
        h: rhs_h,
        hu: rhs_hu,
        hv: rhs_hv,
        n_elements,
        n_nodes,
        device: device.clone(),
    }
}

/// Simplified RHS computation for testing (volume terms only, no surface terms).
///
/// This is useful for verifying volume term computation before adding
/// the complexity of surface terms.
pub fn compute_rhs_volume_only<B: Backend>(
    solution: &BurnSWESolution2D<B>,
    ops: &BurnOperators2D<B>,
    geom: &BurnGeometricFactors2D<B>,
    config: &BurnRhsConfig,
) -> BurnSWESolution2D<B>
where
    B::FloatElem: From<f64>,
{
    let device = &ops.device;
    let n_elements = solution.n_elements;
    let n_nodes = solution.n_nodes;

    // Compute physical fluxes
    let (flux_x_h, flux_x_hu, flux_x_hv, flux_y_h, flux_y_hu, flux_y_hv) =
        compute_swe_fluxes(&solution.h, &solution.hu, &solution.hv, config.g, config.h_min);

    // Compute volume term
    let dr_t = ops.dr_transpose();
    let ds_t = ops.ds_transpose();

    let dfx_dr_h = apply_diff_matrix_batched(&dr_t, &flux_x_h);
    let dfx_ds_h = apply_diff_matrix_batched(&ds_t, &flux_x_h);
    let dfy_dr_h = apply_diff_matrix_batched(&dr_t, &flux_y_h);
    let dfy_ds_h = apply_diff_matrix_batched(&ds_t, &flux_y_h);

    let dfx_dr_hu = apply_diff_matrix_batched(&dr_t, &flux_x_hu);
    let dfx_ds_hu = apply_diff_matrix_batched(&ds_t, &flux_x_hu);
    let dfy_dr_hu = apply_diff_matrix_batched(&dr_t, &flux_y_hu);
    let dfy_ds_hu = apply_diff_matrix_batched(&ds_t, &flux_y_hu);

    let dfx_dr_hv = apply_diff_matrix_batched(&dr_t, &flux_x_hv);
    let dfx_ds_hv = apply_diff_matrix_batched(&ds_t, &flux_x_hv);
    let dfy_dr_hv = apply_diff_matrix_batched(&dr_t, &flux_y_hv);
    let dfy_ds_hv = apply_diff_matrix_batched(&ds_t, &flux_y_hv);

    let rhs_h = combine_derivatives_batched(
        &dfx_dr_h, &dfx_ds_h, &dfy_dr_h, &dfy_ds_h,
        &geom.rx, &geom.sx, &geom.ry, &geom.sy,
    );
    let rhs_hu = combine_derivatives_batched(
        &dfx_dr_hu, &dfx_ds_hu, &dfy_dr_hu, &dfy_ds_hu,
        &geom.rx, &geom.sx, &geom.ry, &geom.sy,
    );
    let rhs_hv = combine_derivatives_batched(
        &dfx_dr_hv, &dfx_ds_hv, &dfy_dr_hv, &dfy_ds_hv,
        &geom.rx, &geom.sx, &geom.ry, &geom.sy,
    );

    // Scale by inverse Jacobian
    let inv_j = geom.det_j.clone().recip().unsqueeze_dim(1);

    BurnSWESolution2D {
        h: rhs_h.mul(inv_j.clone()),
        hu: rhs_hu.mul(inv_j.clone()),
        hv: rhs_hv.mul(inv_j),
        n_elements,
        n_nodes,
        device: device.clone(),
    }
}

#[cfg(test)]
#[cfg(feature = "burn-ndarray")]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use crate::mesh::Mesh2DBuilder;
    use crate::operators::DGOperators2D;
    use crate::operators::GeometricFactors2D;
    use crate::types::ElementIndex;

    #[test]
    fn test_rhs_volume_only_constant_state() {
        // A constant state should have zero volume term (no gradients)
        let mesh = Mesh2DBuilder::unit_square()
            .with_resolution(2, 2)
            .build();

        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let device = burn_ndarray::NdArrayDevice::Cpu;

        // Create GPU structures
        let burn_ops = BurnOperators2D::<NdArray<f64>>::from_cpu(&ops, &device);
        let burn_geom = BurnGeometricFactors2D::<NdArray<f64>>::from_cpu(&geom, &device);

        // Constant state: h=1, hu=0, hv=0 (lake at rest)
        let mut cpu_sol = crate::solver::state::SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                cpu_sol.set_state(k, i, crate::solver::state::SWEState2D::new(1.0, 0.0, 0.0));
            }
        }

        let burn_sol = BurnSWESolution2D::<NdArray<f64>>::from_cpu(&cpu_sol, &device);

        let config = BurnRhsConfig::default();
        let rhs = compute_rhs_volume_only(&burn_sol, &burn_ops, &burn_geom, &config);

        // For constant state with zero velocity, volume RHS should be zero
        let rhs_cpu = rhs.to_cpu();
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let state = rhs_cpu.get_state(k, i);
                assert!(state.h.abs() < 1e-10, "rhs_h should be ~0, got {}", state.h);
                assert!(state.hu.abs() < 1e-10, "rhs_hu should be ~0, got {}", state.hu);
                assert!(state.hv.abs() < 1e-10, "rhs_hv should be ~0, got {}", state.hv);
            }
        }
    }

    #[test]
    fn test_geometric_factors_upload() {
        let mesh = Mesh2DBuilder::unit_square()
            .with_resolution(2, 2)
            .build();

        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let device = burn_ndarray::NdArrayDevice::Cpu;

        let burn_geom = BurnGeometricFactors2D::<NdArray<f64>>::from_cpu(&geom, &device);

        // Check dimensions
        assert_eq!(burn_geom.rx.shape().dims[0], mesh.n_elements);
        assert_eq!(burn_geom.det_j.shape().dims[0], mesh.n_elements);
    }
}
