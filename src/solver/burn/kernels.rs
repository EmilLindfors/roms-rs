//! GPU computational kernels for 2D SWE solver.
//!
//! This module provides batched tensor operations for:
//! - Volume terms: Differentiation matrix-vector products
//! - Flux computation: SWE physical fluxes
//! - Derivative combination: Chain rule with geometric factors
//! - Source terms: Coriolis and friction
//!
//! All operations process all elements in parallel on the GPU.

use burn::prelude::*;

/// Apply differentiation matrix to all elements in batch.
///
/// Computes `d/dr(flux) = flux @ D^T` for all elements.
///
/// # Arguments
/// * `d_t` - Transposed differentiation matrix [n_nodes, n_nodes]
/// * `flux` - Flux values [n_elements, n_nodes]
///
/// # Returns
/// Derivatives [n_elements, n_nodes]
pub fn apply_diff_matrix_batched<B: Backend>(d_t: &Tensor<B, 2>, flux: &Tensor<B, 2>) -> Tensor<B, 2> {
    // flux @ D^T: [E, N] @ [N, N] -> [E, N]
    flux.clone().matmul(d_t.clone())
}

/// Compute SWE physical fluxes for all elements.
///
/// For 2D shallow water equations, the flux in x-direction is:
/// - F_h = hu
/// - F_hu = hu² / h + 0.5 * g * h²
/// - F_hv = hu * hv / h
///
/// The flux in y-direction is:
/// - G_h = hv
/// - G_hu = hu * hv / h
/// - G_hv = hv² / h + 0.5 * g * h²
///
/// # Arguments
/// * `h` - Water depth [n_elements, n_nodes]
/// * `hu` - x-momentum [n_elements, n_nodes]
/// * `hv` - y-momentum [n_elements, n_nodes]
/// * `g` - Gravitational acceleration
/// * `h_min` - Minimum depth for dry cell protection
///
/// # Returns
/// (flux_x_h, flux_x_hu, flux_x_hv, flux_y_h, flux_y_hu, flux_y_hv)
pub fn compute_swe_fluxes<B: Backend>(
    h: &Tensor<B, 2>,
    hu: &Tensor<B, 2>,
    hv: &Tensor<B, 2>,
    g: f64,
    h_min: f64,
) -> (
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 2>,
)
where
    B::FloatElem: From<f64>,
{
    // Regularized depth for velocity computation
    let h_reg = h.clone().clamp_min(h_min);

    // Velocity components: u = hu/h, v = hv/h
    let u = hu.clone().div(h_reg.clone());
    let v = hv.clone().div(h_reg.clone());

    // Pressure term: 0.5 * g * h²
    let half_g = 0.5 * g;
    let pressure = h.clone().powf_scalar(2.0).mul_scalar(half_g);

    // X-direction fluxes
    let flux_x_h = hu.clone();
    let flux_x_hu = hu.clone().mul(u.clone()).add(pressure.clone());
    let flux_x_hv = hu.clone().mul(v.clone());

    // Y-direction fluxes
    let flux_y_h = hv.clone();
    let flux_y_hu = hv.clone().mul(u);
    let flux_y_hv = hv.clone().mul(v).add(pressure);

    (flux_x_h, flux_x_hu, flux_x_hv, flux_y_h, flux_y_hu, flux_y_hv)
}

/// Combine reference derivatives with geometric factors to get physical derivatives.
///
/// Computes the divergence in physical coordinates:
/// div(F) = dF/dx + dG/dy = (dF/dr * rx + dF/ds * sx) + (dG/dr * ry + dG/ds * sy)
///
/// The RHS is: rhs = -div(F)
///
/// # Arguments
/// * `dfx_dr` - d(flux_x)/dr for all variables [n_elements, n_nodes]
/// * `dfx_ds` - d(flux_x)/ds for all variables
/// * `dfy_dr` - d(flux_y)/dr for all variables
/// * `dfy_ds` - d(flux_y)/ds for all variables
/// * `rx`, `sx`, `ry`, `sy` - Geometric factors [n_elements]
///
/// # Returns
/// Negative divergence: -div(F) [n_elements, n_nodes]
pub fn combine_derivatives_batched<B: Backend>(
    dfx_dr: &Tensor<B, 2>,
    dfx_ds: &Tensor<B, 2>,
    dfy_dr: &Tensor<B, 2>,
    dfy_ds: &Tensor<B, 2>,
    rx: &Tensor<B, 1>,
    sx: &Tensor<B, 1>,
    ry: &Tensor<B, 1>,
    sy: &Tensor<B, 1>,
) -> Tensor<B, 2> {
    // Broadcast geometric factors from [E] to [E, N]
    let rx_2d = rx.clone().unsqueeze_dim(1); // [E, 1]
    let sx_2d = sx.clone().unsqueeze_dim(1);
    let ry_2d = ry.clone().unsqueeze_dim(1);
    let sy_2d = sy.clone().unsqueeze_dim(1);

    // Physical derivatives via chain rule:
    // dF/dx = dF/dr * rx + dF/ds * sx
    // dG/dy = dG/dr * ry + dG/ds * sy
    let df_dx = dfx_dr.clone().mul(rx_2d).add(dfx_ds.clone().mul(sx_2d));
    let dg_dy = dfy_dr.clone().mul(ry_2d).add(dfy_ds.clone().mul(sy_2d));

    // Divergence = dF/dx + dG/dy
    let divergence = df_dx.add(dg_dy);

    // RHS = -divergence
    divergence.neg()
}

/// Compute Coriolis source term for all elements.
///
/// The Coriolis force in 2D SWE is:
/// S_hu = +f * hv
/// S_hv = -f * hu
///
/// # Arguments
/// * `hu` - x-momentum [n_elements, n_nodes]
/// * `hv` - y-momentum [n_elements, n_nodes]
/// * `f` - Coriolis parameter (scalar for f-plane)
///
/// # Returns
/// (S_hu, S_hv) source terms
pub fn coriolis_source_batched<B: Backend>(
    hu: &Tensor<B, 2>,
    hv: &Tensor<B, 2>,
    f: f64,
) -> (Tensor<B, 2>, Tensor<B, 2>)
where
    B::FloatElem: From<f64>,
{
    // S_hu = +f * hv
    let s_hu = hv.clone().mul_scalar(f);

    // S_hv = -f * hu
    let s_hv = hu.clone().mul_scalar(-f);

    (s_hu, s_hv)
}

/// Compute Manning friction source term for all elements.
///
/// The friction force is:
/// S_hu = -C_f * |u| * u
/// S_hv = -C_f * |u| * v
///
/// where C_f = g * n² / h^(1/3), u = hu/h, v = hv/h
///
/// # Arguments
/// * `h` - Water depth [n_elements, n_nodes]
/// * `hu` - x-momentum
/// * `hv` - y-momentum
/// * `g_n2` - Precomputed g * manning_n²
/// * `h_min` - Minimum depth for dry cell protection
///
/// # Returns
/// (S_hu, S_hv) friction source terms
pub fn friction_source_batched<B: Backend>(
    h: &Tensor<B, 2>,
    hu: &Tensor<B, 2>,
    hv: &Tensor<B, 2>,
    g_n2: f64,
    h_min: f64,
) -> (Tensor<B, 2>, Tensor<B, 2>)
where
    B::FloatElem: From<f64>,
{
    // Regularized depth
    let h_reg = h.clone().clamp_min(h_min);

    // Velocity components
    let u = hu.clone().div(h_reg.clone());
    let v = hv.clone().div(h_reg.clone());

    // Velocity magnitude: |u| = sqrt(u² + v²)
    let u_mag = u.clone().powf_scalar(2.0).add(v.clone().powf_scalar(2.0)).sqrt();

    // Friction coefficient: C_f = g * n² / h^(1/3)
    let c_f = h_reg.powf_scalar(-1.0 / 3.0).mul_scalar(g_n2);

    // Source: S = -C_f * |u| * velocity
    let factor = c_f.mul(u_mag).neg();

    let s_hu = factor.clone().mul(u);
    let s_hv = factor.mul(v);

    (s_hu, s_hv)
}

/// Compute bathymetry gradient source term.
///
/// For well-balanced schemes, the bathymetry source is:
/// S_hu = -g * h * dB/dx
/// S_hv = -g * h * dB/dy
///
/// # Arguments
/// * `h` - Water depth [n_elements, n_nodes]
/// * `db_dx` - Bathymetry gradient in x [n_elements, n_nodes]
/// * `db_dy` - Bathymetry gradient in y
/// * `g` - Gravitational acceleration
///
/// # Returns
/// (S_hu, S_hv) bathymetry source terms
pub fn bathymetry_source_batched<B: Backend>(
    h: &Tensor<B, 2>,
    db_dx: &Tensor<B, 2>,
    db_dy: &Tensor<B, 2>,
    g: f64,
) -> (Tensor<B, 2>, Tensor<B, 2>)
where
    B::FloatElem: From<f64>,
{
    // S_hu = -g * h * dB/dx
    let s_hu = h.clone().mul(db_dx.clone()).mul_scalar(-g);

    // S_hv = -g * h * dB/dy
    let s_hv = h.clone().mul(db_dy.clone()).mul_scalar(-g);

    (s_hu, s_hv)
}

#[cfg(test)]
#[cfg(feature = "burn-ndarray")]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn test_apply_diff_matrix_batched() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let n_elements = 2;
        let n_nodes = 3;

        // Simple differentiation matrix (identity for testing)
        let d_t: Tensor<NdArray<f64>, 2> = Tensor::eye(n_nodes, &device);

        // Flux values
        let flux_data: Vec<f64> = (1..=6).map(|i| i as f64).collect();
        let flux: Tensor<NdArray<f64>, 2> = Tensor::from_data(
            burn::tensor::TensorData::new(flux_data.clone(), vec![n_elements, n_nodes]),
            &device,
        );

        let result = apply_diff_matrix_batched(&d_t, &flux);

        // With identity D^T, result should equal input
        let result_data = result.to_data().to_vec::<f64>().unwrap();
        for (a, b) in result_data.iter().zip(flux_data.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_compute_swe_fluxes() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let n_elements = 1;
        let n_nodes = 2;
        let g = 9.81;
        let h_min = 1e-6;

        // Constant state: h=1, u=1, v=0 => hu=1, hv=0
        let h: Tensor<NdArray<f64>, 2> =
            Tensor::ones([n_elements, n_nodes], &device);
        let hu: Tensor<NdArray<f64>, 2> =
            Tensor::ones([n_elements, n_nodes], &device);
        let hv: Tensor<NdArray<f64>, 2> =
            Tensor::zeros([n_elements, n_nodes], &device);

        let (fx_h, fx_hu, fx_hv, fy_h, fy_hu, fy_hv) =
            compute_swe_fluxes(&h, &hu, &hv, g, h_min);

        // Check x-direction fluxes
        let fx_h_data = fx_h.to_data().to_vec::<f64>().unwrap();
        let fx_hu_data = fx_hu.to_data().to_vec::<f64>().unwrap();
        let fx_hv_data = fx_hv.to_data().to_vec::<f64>().unwrap();

        // F_h = hu = 1
        assert!((fx_h_data[0] - 1.0).abs() < 1e-12);
        // F_hu = hu²/h + 0.5*g*h² = 1 + 0.5*9.81*1 = 5.905
        assert!((fx_hu_data[0] - (1.0 + 0.5 * g)).abs() < 1e-10);
        // F_hv = hu*hv/h = 0
        assert!(fx_hv_data[0].abs() < 1e-12);

        // Check y-direction fluxes (should be zero since hv=0)
        let fy_h_data = fy_h.to_data().to_vec::<f64>().unwrap();
        assert!(fy_h_data[0].abs() < 1e-12);
    }

    #[test]
    fn test_coriolis_source_batched() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let n_elements = 1;
        let n_nodes = 4;
        let f = 1.2e-4; // Norwegian coast

        let hu: Tensor<NdArray<f64>, 2> =
            Tensor::ones([n_elements, n_nodes], &device);
        let hv: Tensor<NdArray<f64>, 2> =
            Tensor::ones([n_elements, n_nodes], &device).mul_scalar(2.0);

        let (s_hu, s_hv) = coriolis_source_batched(&hu, &hv, f);

        let s_hu_data = s_hu.to_data().to_vec::<f64>().unwrap();
        let s_hv_data = s_hv.to_data().to_vec::<f64>().unwrap();

        // S_hu = f * hv = 1.2e-4 * 2.0 = 2.4e-4
        assert!((s_hu_data[0] - 2.4e-4).abs() < 1e-10);
        // S_hv = -f * hu = -1.2e-4 * 1.0 = -1.2e-4
        assert!((s_hv_data[0] - (-1.2e-4)).abs() < 1e-10);
    }

    #[test]
    fn test_combine_derivatives_batched() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let n_elements = 2;
        let n_nodes = 3;

        // Simple case: all derivatives = 1
        let ones_2d: Tensor<NdArray<f64>, 2> =
            Tensor::ones([n_elements, n_nodes], &device);

        // Geometric factors: rx=1, sx=0, ry=0, sy=1 (identity transformation)
        let rx: Tensor<NdArray<f64>, 1> = Tensor::ones([n_elements], &device);
        let sx: Tensor<NdArray<f64>, 1> = Tensor::zeros([n_elements], &device);
        let ry: Tensor<NdArray<f64>, 1> = Tensor::zeros([n_elements], &device);
        let sy: Tensor<NdArray<f64>, 1> = Tensor::ones([n_elements], &device);

        let result = combine_derivatives_batched(
            &ones_2d, &ones_2d, &ones_2d, &ones_2d, &rx, &sx, &ry, &sy,
        );

        let result_data = result.to_data().to_vec::<f64>().unwrap();

        // div = dfx_dr*rx + dfx_ds*sx + dfy_dr*ry + dfy_ds*sy
        //     = 1*1 + 1*0 + 1*0 + 1*1 = 2
        // RHS = -div = -2
        for &val in &result_data {
            assert!((val - (-2.0)).abs() < 1e-12);
        }
    }
}
