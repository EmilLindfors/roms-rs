//! Horizontal viscosity models for 2D shallow water equations.
//!
//! Adds turbulent viscosity diffusion to the momentum equations:
//!   ∂(hu)/∂t + ... = ... + ∇·(ν h ∇u)
//!   ∂(hv)/∂t + ... = ... + ∇·(ν h ∇v)
//!
//! Two models are provided:
//! - **Constant**: Uniform eddy viscosity ν = ν₀
//! - **Smagorinsky**: Strain-dependent ν = (C_s · Δ)² · |S|,
//!   where |S| is the magnitude of the strain rate tensor
//!
//! These terms are injected directly in the RHS (not via `SourceTerm2D`)
//! because they require access to differentiation operators and geometric
//! factors for gradient computation.

/// Viscosity model selection.
#[derive(Debug, Clone, Copy)]
pub enum ViscosityModel {
    /// Constant eddy viscosity: ν = ν₀
    Constant(f64),
    /// Smagorinsky model: ν = (C_s · Δ)² · |S|
    ///
    /// where C_s is the Smagorinsky coefficient (typically 0.1–0.2)
    /// and Δ is the local grid spacing (estimated from √det_j).
    Smagorinsky { cs: f64 },
}

/// Horizontal viscosity configuration for 2D SWE momentum diffusion.
#[derive(Debug, Clone)]
pub struct HorizontalViscosity2D {
    /// The viscosity model.
    pub model: ViscosityModel,
    /// Minimum depth below which viscosity is disabled (avoids division by tiny h).
    pub h_min: f64,
}

impl HorizontalViscosity2D {
    /// Create a constant eddy viscosity model.
    ///
    /// # Arguments
    /// * `nu` - Constant viscosity coefficient [m²/s]
    pub fn constant(nu: f64) -> Self {
        Self {
            model: ViscosityModel::Constant(nu),
            h_min: 1e-3,
        }
    }

    /// Create a Smagorinsky viscosity model.
    ///
    /// # Arguments
    /// * `cs` - Smagorinsky coefficient (typically 0.1–0.2)
    pub fn smagorinsky(cs: f64) -> Self {
        Self {
            model: ViscosityModel::Smagorinsky { cs },
            h_min: 1e-3,
        }
    }

    /// Compute the viscosity at a point given velocity gradients and element size.
    ///
    /// # Arguments
    /// * `du_dx`, `du_dy` - Velocity gradients of u
    /// * `dv_dx`, `dv_dy` - Velocity gradients of v
    /// * `delta` - Local grid spacing (e.g., √det_j)
    pub fn compute_viscosity(
        &self,
        du_dx: f64,
        du_dy: f64,
        dv_dx: f64,
        dv_dy: f64,
        delta: f64,
    ) -> f64 {
        match self.model {
            ViscosityModel::Constant(nu) => nu,
            ViscosityModel::Smagorinsky { cs } => {
                // Strain rate tensor components:
                //   S₁₁ = ∂u/∂x,  S₂₂ = ∂v/∂y,  S₁₂ = 0.5·(∂u/∂y + ∂v/∂x)
                let s11 = du_dx;
                let s22 = dv_dy;
                let s12 = 0.5 * (du_dy + dv_dx);

                // |S| = √(2·(S₁₁² + S₂₂² + 2·S₁₂²))
                let strain_mag = (2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12)).sqrt();

                // ν = (C_s · Δ)² · |S|
                let cs_delta = cs * delta;
                cs_delta * cs_delta * strain_mag
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_viscosity() {
        let visc = HorizontalViscosity2D::constant(1.5);
        // Should return constant regardless of strain
        assert_eq!(visc.compute_viscosity(1.0, 2.0, 3.0, 4.0, 100.0), 1.5);
        assert_eq!(visc.compute_viscosity(0.0, 0.0, 0.0, 0.0, 100.0), 1.5);
    }

    #[test]
    fn test_smagorinsky_zero_strain() {
        let visc = HorizontalViscosity2D::smagorinsky(0.1);
        // Zero velocity gradients → zero viscosity
        let nu = visc.compute_viscosity(0.0, 0.0, 0.0, 0.0, 100.0);
        assert!(
            nu.abs() < 1e-15,
            "Zero strain should give zero viscosity, got {nu}"
        );
    }

    #[test]
    fn test_smagorinsky_pure_shear() {
        let cs = 0.1;
        let delta = 100.0;
        let visc = HorizontalViscosity2D::smagorinsky(cs);

        // Pure shear: du/dy = 1.0, dv/dx = 1.0, all others zero
        // S₁₁ = 0, S₂₂ = 0, S₁₂ = 0.5·(1+1) = 1.0
        // |S| = √(2·(0 + 0 + 2·1²)) = √4 = 2
        // ν = (0.1·100)² · 2 = 100 · 2 = 200
        let nu = visc.compute_viscosity(0.0, 1.0, 1.0, 0.0, delta);
        assert!(
            (nu - 200.0).abs() < 1e-10,
            "Pure shear: expected 200.0, got {nu}"
        );
    }

    #[test]
    fn test_smagorinsky_scaling() {
        let delta = 50.0;
        let visc1 = HorizontalViscosity2D::smagorinsky(0.1);
        let visc2 = HorizontalViscosity2D::smagorinsky(0.2);

        // Same strain field
        let nu1 = visc1.compute_viscosity(1.0, 0.5, -0.5, -1.0, delta);
        let nu2 = visc2.compute_viscosity(1.0, 0.5, -0.5, -1.0, delta);

        // ν ∝ cs², so doubling cs should give 4× viscosity
        let ratio = nu2 / nu1;
        assert!(
            (ratio - 4.0).abs() < 1e-10,
            "Doubling cs should give 4x viscosity, got ratio {ratio}"
        );
    }
}
