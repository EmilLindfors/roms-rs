//! Equation of State (EOS) for seawater.
//!
//! Computes density from potential temperature, salinity, and pressure (depth).
//!
//! Based on the standard Jackett and McDougall (1995) or UNESCO 1980 EOS.
//!
//! # Equations
//!
//! $\rho(S, T, z)$
//!
//! In Boussinesq models, we often care about $\rho_{pot}$ referenced to surface
//! or the in-situ density perturbation.
//!
//! For this implementation, we use a linearized or simplified nonlinear EOS suitable
//! for coastal applications.

use crate::solver::state::Solution3D;
use crate::types::ElementIndex;

/// Equation of State trait.
pub trait EquationOfState: Send + Sync {
    /// Compute density from T, S, and depth.
    ///
    /// # Arguments
    /// * `t`: Potential temperature [deg C]
    /// * `s`: Salinity [PSU]
    /// * `z`: Depth [m] (negative downwards)
    ///
    /// Returns density [kg/m^3].
    fn compute_density(&self, t: f64, s: f64, z: f64) -> f64;
    
    /// Update the density field in the 3D solution.
    fn update_density(&self, state: &mut Solution3D);
}

/// Linear Equation of State.
///
/// $\rho = \rho_0 (1 - \alpha (T - T_0) + \beta (S - S_0))$
#[derive(Debug, Clone, Copy)]
pub struct LinearEOS {
    pub rho0: f64,
    pub t0: f64,
    pub s0: f64,
    pub alpha: f64, // Thermal expansion [1/C]
    pub beta: f64,  // Haline contraction [1/PSU]
}

impl Default for LinearEOS {
    fn default() -> Self {
        Self {
            rho0: 1025.0,
            t0: 10.0,
            s0: 35.0,
            alpha: 1.7e-4,
            beta: 7.6e-4,
        }
    }
}

impl EquationOfState for LinearEOS {
    fn compute_density(&self, t: f64, s: f64, _z: f64) -> f64 {
        self.rho0 * (1.0 - self.alpha * (t - self.t0) + self.beta * (s - self.s0))
    }

    fn update_density(&self, state: &mut Solution3D) {
        let n_elements = state.n_elements;
        let n_nodes = state.n_nodes;
        let n_levels = state.n_levels;
        
        for k in ElementIndex::iter(n_elements) {
            let _ki = k.as_usize();
            for i in 0..n_nodes {
                // Get T, S columns (read-only)
                // We extract them first to avoid borrow conflicts, or access element-wise?
                // Slice splitting is hard with `state`.
                // However, we can just iterate by index since everything is contiguous.
                // But `update_density` takes &mut Solution3D.
                // We can't get &temp and &mut rho simultaneously if they are fields of same struct
                // unless we split the borrow.
                // Since `Solution3D` fields are public, we can borrow fields individually.
                
                // This is the borrow checker friendly way:
                // We cannot access `state.temp_column` and `state.rho_column_mut` simultaneously
                // if those methods take `&self` and `&mut self`.
                // But we CAN access `state.temp` and `state.rho` vectors directly.
                
                let start_idx = (k.as_usize() * n_nodes + i) * n_levels;
                let end_idx = start_idx + n_levels;
                
                let t_col = &state.temp[start_idx..end_idx];
                let s_col = &state.salt[start_idx..end_idx];
                let rho_col = &mut state.rho[start_idx..end_idx];
                
                for l in 0..n_levels {
                    rho_col[l] = self.compute_density(t_col[l], s_col[l], 0.0);
                }
            }
        }
    }
}

/// UNESCO 1980 Equation of State (simplified).
///
/// Ignores pressure effects (potential density at surface pressure).
#[derive(Debug, Clone, Copy)]
pub struct UnescoEOS {
    pub rho0: f64,
}

impl EquationOfState for UnescoEOS {
    fn compute_density(&self, t: f64, s: f64, _z: f64) -> f64 {
        // Simple polynomial fit (Jackett & McDougall 1995 or similar)
        // Values for coefficients...
        // This is a placeholder for the full nonlinear implementation.
        let t2 = t * t;
        let s32 = s.powf(1.5);
        
        // Approx coeff for sigma-t (kg/m^3 - 1000)
        let sigma = 999.842594 + 6.793952e-2 * t - 9.095290e-3 * t2 + 1.001685e-4 * t2*t
            + (0.824493 - 4.0899e-3 * t + 7.6438e-5 * t2) * s
            - 5.72466e-3 * s32 + 4.8314e-4 * s*s;
            
        sigma
    }
    
    fn update_density(&self, state: &mut Solution3D) {
        let n_elements = state.n_elements;
        let n_nodes = state.n_nodes;
        let n_levels = state.n_levels;
        
        // Parallel iteration could be done here if we had par_iter
        // For now, sequential loop with split borrows.
        
        // We can iterate over the full flat vectors directly!
        // No need to do element/node loops since it's pointwise.
        
        let total_size = n_elements * n_nodes * n_levels;
        for idx in 0..total_size {
             let t = state.temp[idx];
             let s = state.salt[idx];
             // z dependence ignored in this simplified version
             state.rho[idx] = self.compute_density(t, s, 0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_eos() {
        let eos = LinearEOS::default();
        // T=T0, S=S0 -> rho=rho0
        let rho = eos.compute_density(10.0, 35.0, 0.0);
        assert!((rho - 1025.0).abs() < 1e-10);
        
        // Warmer -> lighter
        let rho_warm = eos.compute_density(20.0, 35.0, 0.0);
        assert!(rho_warm < 1025.0);
        
        // Saltier -> heavier
        let rho_salty = eos.compute_density(10.0, 36.0, 0.0);
        assert!(rho_salty > 1025.0);
    }
    
    #[test]
    fn test_update_density() {
        let n_elem = 1;
        let n_nodes = 1;
        let n_levels = 2;
        let mut state = Solution3D::new(n_elem, n_nodes, n_levels);
        
        state.temp.fill(20.0);
        state.salt.fill(35.0);
        
        let eos = LinearEOS::default();
        eos.update_density(&mut state);
        
        let expected = eos.compute_density(20.0, 35.0, 0.0);
        assert!((state.rho[0] - expected).abs() < 1e-10);
        assert!((state.rho[1] - expected).abs() < 1e-10);
    }
}
