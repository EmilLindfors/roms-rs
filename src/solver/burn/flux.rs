//! GPU Riemann solvers for SWE surface terms.
//!
//! This module provides batched implementations of numerical fluxes
//! for computing surface integrals on the GPU.

use burn::prelude::*;

/// HLL numerical flux for batched face computations.
///
/// The HLL flux uses wave speed estimates to blend left and right fluxes:
///
/// ```text
///           | F_L                           if s_L >= 0
/// F_HLL =   | (s_R*F_L - s_L*F_R + s_L*s_R*(q_R - q_L)) / (s_R - s_L)  if s_L < 0 < s_R
///           | F_R                           if s_R <= 0
/// ```
///
/// # Arguments
/// * `h_l`, `hu_l`, `hv_l` - Left state [n_faces, n_face_nodes]
/// * `h_r`, `hu_r`, `hv_r` - Right state
/// * `nx`, `ny` - Outward normals [n_faces, n_face_nodes]
/// * `g` - Gravitational acceleration
/// * `h_min` - Minimum depth for dry cell protection
///
/// # Returns
/// (F_h, F_hu, F_hv) numerical fluxes in normal direction
pub fn hll_flux_batched<B: Backend>(
    h_l: &Tensor<B, 2>,
    hu_l: &Tensor<B, 2>,
    hv_l: &Tensor<B, 2>,
    h_r: &Tensor<B, 2>,
    hu_r: &Tensor<B, 2>,
    hv_r: &Tensor<B, 2>,
    nx: &Tensor<B, 2>,
    ny: &Tensor<B, 2>,
    g: f64,
    h_min: f64,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)
where
    B::FloatElem: From<f64>,
{
    // Regularize depths
    let h_l_reg = h_l.clone().clamp_min(h_min);
    let h_r_reg = h_r.clone().clamp_min(h_min);

    // Velocity components
    let u_l = hu_l.clone().div(h_l_reg.clone());
    let v_l = hv_l.clone().div(h_l_reg.clone());
    let u_r = hu_r.clone().div(h_r_reg.clone());
    let v_r = hv_r.clone().div(h_r_reg.clone());

    // Normal velocities: un = u*nx + v*ny
    let un_l = u_l.clone().mul(nx.clone()).add(v_l.clone().mul(ny.clone()));
    let un_r = u_r.clone().mul(nx.clone()).add(v_r.clone().mul(ny.clone()));

    // Wave celerities: c = sqrt(g*h)
    let c_l = h_l_reg.clone().mul_scalar(g).sqrt();
    let c_r = h_r_reg.clone().mul_scalar(g).sqrt();

    // HLL wave speed estimates (Einfeldt)
    // s_L = min(un_L - c_L, un_R - c_R)
    // s_R = max(un_L + c_L, un_R + c_R)
    let s_l_l = un_l.clone().sub(c_l.clone());
    let s_l_r = un_r.clone().sub(c_r.clone());
    let s_l = s_l_l.clone().min_pair(s_l_r);

    let s_r_l = un_l.clone().add(c_l);
    let s_r_r = un_r.clone().add(c_r);
    let s_r = s_r_l.clone().max_pair(s_r_r);

    // Physical fluxes in normal direction
    // F_h = h * un
    // F_hu = hu * un + 0.5 * g * h² * nx
    // F_hv = hv * un + 0.5 * g * h² * ny
    let half_g = 0.5 * g;

    let pressure_l = h_l.clone().powf_scalar(2.0).mul_scalar(half_g);
    let pressure_r = h_r.clone().powf_scalar(2.0).mul_scalar(half_g);

    let f_h_l = h_l.clone().mul(un_l.clone());
    let f_hu_l = hu_l.clone().mul(un_l.clone()).add(pressure_l.clone().mul(nx.clone()));
    let f_hv_l = hv_l.clone().mul(un_l.clone()).add(pressure_l.mul(ny.clone()));

    let f_h_r = h_r.clone().mul(un_r.clone());
    let f_hu_r = hu_r.clone().mul(un_r.clone()).add(pressure_r.clone().mul(nx.clone()));
    let f_hv_r = hv_r.clone().mul(un_r.clone()).add(pressure_r.mul(ny.clone()));

    // HLL flux computation with conditional blending
    // For numerical stability, we compute the general formula and use masks

    // Denominator: s_R - s_L (with protection against zero)
    let denom = s_r.clone().sub(s_l.clone()).clamp_min(1e-10);
    let inv_denom = denom.recip();

    // General HLL formula:
    // F = (s_R * F_L - s_L * F_R + s_L * s_R * (q_R - q_L)) / (s_R - s_L)
    let s_l_s_r = s_l.clone().mul(s_r.clone());

    let f_h_hll = s_r.clone().mul(f_h_l.clone())
        .sub(s_l.clone().mul(f_h_r.clone()))
        .add(s_l_s_r.clone().mul(h_r.clone().sub(h_l.clone())))
        .mul(inv_denom.clone());

    let f_hu_hll = s_r.clone().mul(f_hu_l.clone())
        .sub(s_l.clone().mul(f_hu_r.clone()))
        .add(s_l_s_r.clone().mul(hu_r.clone().sub(hu_l.clone())))
        .mul(inv_denom.clone());

    let f_hv_hll = s_r.clone().mul(f_hv_l.clone())
        .sub(s_l.clone().mul(f_hv_r.clone()))
        .add(s_l_s_r.mul(hv_r.clone().sub(hv_l.clone())))
        .mul(inv_denom);

    // Apply masks for the three regions
    let zero = Tensor::zeros_like(&s_l);

    // Region 1: s_L >= 0 -> use F_L
    let mask_left = s_l.clone().greater_equal(zero.clone());

    // Region 3: s_R <= 0 -> use F_R
    let mask_right = s_r.clone().lower_equal(zero);

    // Region 2: s_L < 0 < s_R -> use F_HLL (default)
    // mask_hll = !mask_left && !mask_right

    // Select fluxes based on masks
    let f_h = f_h_l.clone().mask_where(mask_left.clone(), f_h_hll.clone());
    let f_h = f_h_r.mask_where(mask_right.clone(), f_h);

    let f_hu = f_hu_l.clone().mask_where(mask_left.clone(), f_hu_hll.clone());
    let f_hu = f_hu_r.mask_where(mask_right.clone(), f_hu);

    let f_hv = f_hv_l.clone().mask_where(mask_left, f_hv_hll.clone());
    let f_hv = f_hv_r.mask_where(mask_right, f_hv);

    (f_h, f_hu, f_hv)
}

/// Roe numerical flux for batched face computations.
///
/// The Roe flux uses Roe-averaged states for wave decomposition:
/// F_Roe = 0.5 * (F_L + F_R) - 0.5 * |A| * (q_R - q_L)
///
/// where |A| is the absolute value of the Roe matrix.
///
/// # Arguments
/// Same as `hll_flux_batched`
///
/// # Returns
/// (F_h, F_hu, F_hv) numerical fluxes in normal direction
pub fn roe_flux_batched<B: Backend>(
    h_l: &Tensor<B, 2>,
    hu_l: &Tensor<B, 2>,
    hv_l: &Tensor<B, 2>,
    h_r: &Tensor<B, 2>,
    hu_r: &Tensor<B, 2>,
    hv_r: &Tensor<B, 2>,
    nx: &Tensor<B, 2>,
    ny: &Tensor<B, 2>,
    g: f64,
    h_min: f64,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)
where
    B::FloatElem: From<f64>,
{
    // Regularize depths
    let h_l_reg = h_l.clone().clamp_min(h_min);
    let h_r_reg = h_r.clone().clamp_min(h_min);

    // Velocity components
    let u_l = hu_l.clone().div(h_l_reg.clone());
    let v_l = hv_l.clone().div(h_l_reg.clone());
    let u_r = hu_r.clone().div(h_r_reg.clone());
    let v_r = hv_r.clone().div(h_r_reg.clone());

    // Roe-averaged states (using sqrt weighting)
    let sqrt_h_l = h_l_reg.clone().sqrt();
    let sqrt_h_r = h_r_reg.clone().sqrt();
    let sqrt_sum = sqrt_h_l.clone().add(sqrt_h_r.clone());
    let inv_sqrt_sum = sqrt_sum.recip();

    let u_roe = sqrt_h_l.clone().mul(u_l.clone())
        .add(sqrt_h_r.clone().mul(u_r.clone()))
        .mul(inv_sqrt_sum.clone());
    let v_roe = sqrt_h_l.clone().mul(v_l.clone())
        .add(sqrt_h_r.clone().mul(v_r.clone()))
        .mul(inv_sqrt_sum.clone());
    let h_roe = sqrt_h_l.mul(sqrt_h_r);
    let c_roe = h_roe.clone().mul_scalar(g).sqrt();

    // Normal and tangential Roe velocities
    let un_roe = u_roe.clone().mul(nx.clone()).add(v_roe.clone().mul(ny.clone()));

    // Wave strengths
    let dh = h_r.clone().sub(h_l.clone());
    let dhu = hu_r.clone().sub(hu_l.clone());
    let dhv = hv_r.clone().sub(hv_l.clone());

    // Normal momentum jump
    let dun = dhu.clone().mul(nx.clone()).add(dhv.clone().mul(ny.clone()));

    // Wave strength coefficients for SWE
    let inv_2c = c_roe.clone().mul_scalar(2.0).recip();

    // alpha_1 = (dh - dun/c) / 2
    let alpha_1 = dh.clone().sub(dun.clone().mul(inv_2c.clone())).mul_scalar(0.5);

    // alpha_2 = dh - alpha_1 - alpha_3 (computed via continuity)
    // alpha_3 = (dh + dun/c) / 2
    let alpha_3 = dh.clone().add(dun.mul(inv_2c)).mul_scalar(0.5);

    // Eigenvalues
    let lambda_1 = un_roe.clone().sub(c_roe.clone());
    let lambda_2 = un_roe.clone();
    let lambda_3 = un_roe.add(c_roe);

    // Absolute eigenvalues (entropy fix via smoothing)
    let eps = h_roe.mul_scalar(0.1).add_scalar(1e-6);
    let abs_lambda_1 = lambda_1.clone().abs().max_pair(eps.clone());
    let abs_lambda_2 = lambda_2.abs().max_pair(eps.clone());
    let abs_lambda_3 = lambda_3.abs().max_pair(eps);

    // Physical fluxes
    let un_l = u_l.clone().mul(nx.clone()).add(v_l.clone().mul(ny.clone()));
    let un_r = u_r.clone().mul(nx.clone()).add(v_r.clone().mul(ny.clone()));

    let half_g = 0.5 * g;
    let pressure_l = h_l.clone().powf_scalar(2.0).mul_scalar(half_g);
    let pressure_r = h_r.clone().powf_scalar(2.0).mul_scalar(half_g);

    let f_h_l = h_l.clone().mul(un_l.clone());
    let f_hu_l = hu_l.clone().mul(un_l.clone()).add(pressure_l.clone().mul(nx.clone()));
    let f_hv_l = hv_l.clone().mul(un_l).add(pressure_l.mul(ny.clone()));

    let f_h_r = h_r.clone().mul(un_r.clone());
    let f_hu_r = hu_r.clone().mul(un_r.clone()).add(pressure_r.clone().mul(nx.clone()));
    let f_hv_r = hv_r.clone().mul(un_r).add(pressure_r.mul(ny.clone()));

    // Average flux
    let f_h_avg = f_h_l.add(f_h_r).mul_scalar(0.5);
    let f_hu_avg = f_hu_l.add(f_hu_r).mul_scalar(0.5);
    let f_hv_avg = f_hv_l.add(f_hv_r).mul_scalar(0.5);

    // Dissipation terms (simplified for SWE)
    // For h: sum of |lambda_i| * alpha_i
    let diss_h = abs_lambda_1.clone().mul(alpha_1.clone())
        .add(abs_lambda_3.clone().mul(alpha_3.clone()));

    // For hu and hv: more complex eigenvector contributions
    // Simplified: use max wave speed dissipation
    let max_wave = abs_lambda_1.clone().max_pair(abs_lambda_3.clone());
    let diss_hu = max_wave.clone().mul(dhu);
    let diss_hv = max_wave.mul(dhv);

    // Roe flux = average - 0.5 * dissipation
    let f_h = f_h_avg.sub(diss_h.mul_scalar(0.5));
    let f_hu = f_hu_avg.sub(diss_hu.mul_scalar(0.5));
    let f_hv = f_hv_avg.sub(diss_hv.mul_scalar(0.5));

    (f_h, f_hu, f_hv)
}

/// Rusanov (Local Lax-Friedrichs) flux for batched computations.
///
/// The simplest upwind flux: F = 0.5 * (F_L + F_R) - 0.5 * alpha * (q_R - q_L)
/// where alpha = max(|u_L| + c_L, |u_R| + c_R)
///
/// More diffusive than HLL/Roe but very stable.
pub fn rusanov_flux_batched<B: Backend>(
    h_l: &Tensor<B, 2>,
    hu_l: &Tensor<B, 2>,
    hv_l: &Tensor<B, 2>,
    h_r: &Tensor<B, 2>,
    hu_r: &Tensor<B, 2>,
    hv_r: &Tensor<B, 2>,
    nx: &Tensor<B, 2>,
    ny: &Tensor<B, 2>,
    g: f64,
    h_min: f64,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)
where
    B::FloatElem: From<f64>,
{
    // Regularize depths
    let h_l_reg = h_l.clone().clamp_min(h_min);
    let h_r_reg = h_r.clone().clamp_min(h_min);

    // Velocity components
    let u_l = hu_l.clone().div(h_l_reg.clone());
    let v_l = hv_l.clone().div(h_l_reg.clone());
    let u_r = hu_r.clone().div(h_r_reg.clone());
    let v_r = hv_r.clone().div(h_r_reg.clone());

    // Normal velocities
    let un_l = u_l.clone().mul(nx.clone()).add(v_l.clone().mul(ny.clone()));
    let un_r = u_r.clone().mul(nx.clone()).add(v_r.clone().mul(ny.clone()));

    // Wave celerities
    let c_l = h_l_reg.mul_scalar(g).sqrt();
    let c_r = h_r_reg.mul_scalar(g).sqrt();

    // Maximum wave speed
    let alpha_l = un_l.clone().abs().add(c_l);
    let alpha_r = un_r.clone().abs().add(c_r);
    let alpha = alpha_l.max_pair(alpha_r);

    // Physical fluxes
    let half_g = 0.5 * g;
    let pressure_l = h_l.clone().powf_scalar(2.0).mul_scalar(half_g);
    let pressure_r = h_r.clone().powf_scalar(2.0).mul_scalar(half_g);

    let f_h_l = h_l.clone().mul(un_l.clone());
    let f_hu_l = hu_l.clone().mul(un_l.clone()).add(pressure_l.clone().mul(nx.clone()));
    let f_hv_l = hv_l.clone().mul(un_l).add(pressure_l.mul(ny.clone()));

    let f_h_r = h_r.clone().mul(un_r.clone());
    let f_hu_r = hu_r.clone().mul(un_r.clone()).add(pressure_r.clone().mul(nx.clone()));
    let f_hv_r = hv_r.clone().mul(un_r).add(pressure_r.mul(ny.clone()));

    // Rusanov flux
    let dh = h_r.clone().sub(h_l.clone());
    let dhu = hu_r.clone().sub(hu_l.clone());
    let dhv = hv_r.clone().sub(hv_l.clone());

    let f_h = f_h_l.add(f_h_r).mul_scalar(0.5)
        .sub(alpha.clone().mul(dh).mul_scalar(0.5));
    let f_hu = f_hu_l.add(f_hu_r).mul_scalar(0.5)
        .sub(alpha.clone().mul(dhu).mul_scalar(0.5));
    let f_hv = f_hv_l.add(f_hv_r).mul_scalar(0.5)
        .sub(alpha.mul(dhv).mul_scalar(0.5));

    (f_h, f_hu, f_hv)
}

#[cfg(test)]
#[cfg(feature = "burn-ndarray")]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn test_hll_flux_symmetric() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let n_faces = 1;
        let n_nodes = 2;
        let g = 9.81;
        let h_min = 1e-6;

        // Symmetric state: should give zero flux difference
        let h: Tensor<NdArray<f64>, 2> = Tensor::ones([n_faces, n_nodes], &device);
        let hu: Tensor<NdArray<f64>, 2> = Tensor::ones([n_faces, n_nodes], &device);
        let hv: Tensor<NdArray<f64>, 2> = Tensor::zeros([n_faces, n_nodes], &device);
        let nx: Tensor<NdArray<f64>, 2> = Tensor::ones([n_faces, n_nodes], &device);
        let ny: Tensor<NdArray<f64>, 2> = Tensor::zeros([n_faces, n_nodes], &device);

        let (f_h, f_hu, f_hv) = hll_flux_batched(
            &h, &hu, &hv, &h, &hu, &hv, &nx, &ny, g, h_min,
        );

        // For symmetric states, flux should be the physical flux
        let f_h_data = f_h.to_data().to_vec::<f64>().unwrap();
        let f_hu_data = f_hu.to_data().to_vec::<f64>().unwrap();

        // F_h = h * un = 1 * 1 = 1
        assert!((f_h_data[0] - 1.0).abs() < 1e-10);

        // F_hu = hu * un + 0.5 * g * h² = 1 + 0.5 * 9.81 = 5.905
        let expected_f_hu = 1.0 + 0.5 * g;
        assert!((f_hu_data[0] - expected_f_hu).abs() < 1e-10);
    }

    #[test]
    fn test_rusanov_flux_conservative() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let n_faces = 1;
        let n_nodes = 2;
        let g = 9.81;
        let h_min = 1e-6;

        // Left and right states
        let h_l: Tensor<NdArray<f64>, 2> = Tensor::ones([n_faces, n_nodes], &device).mul_scalar(2.0);
        let h_r: Tensor<NdArray<f64>, 2> = Tensor::ones([n_faces, n_nodes], &device);
        let hu: Tensor<NdArray<f64>, 2> = Tensor::zeros([n_faces, n_nodes], &device);
        let hv: Tensor<NdArray<f64>, 2> = Tensor::zeros([n_faces, n_nodes], &device);
        let nx: Tensor<NdArray<f64>, 2> = Tensor::ones([n_faces, n_nodes], &device);
        let ny: Tensor<NdArray<f64>, 2> = Tensor::zeros([n_faces, n_nodes], &device);

        let (f_h, _f_hu, _f_hv) = rusanov_flux_batched(
            &h_l, &hu, &hv, &h_r, &hu, &hv, &nx, &ny, g, h_min,
        );

        // With zero velocity and depth difference, flux should be non-zero
        let f_h_data = f_h.to_data().to_vec::<f64>().unwrap();

        // Physical flux is zero (no velocity), but dissipation is non-zero
        // F = 0.5 * (0 + 0) - 0.5 * alpha * (1 - 2) = 0.5 * alpha
        // alpha = max(c_L, c_R) = max(sqrt(g*2), sqrt(g*1)) = sqrt(g*2)
        assert!(f_h_data[0] > 0.0); // Flux should be positive (L->R flow)
    }
}
