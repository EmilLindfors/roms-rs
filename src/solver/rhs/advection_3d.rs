//! Boundary conditions of the 3D tracers, and the strong-form divergence of
//! the diagnostic vertical velocity.
//!
//! The advection of the 3D momentum and tracers, in inventory form with the
//! layer transports, is in [`crate::solver::rhs::transport_3d`].

use crate::mesh::data::BoundaryTag;
use crate::operators::DGOperators2D;

pub(crate) const MIN_LAYER_THICKNESS: f64 = 1.0e-12;

/// Boundary context for scalar 3D tracer concentrations.
pub struct TracerBCContext3D {
    pub element: usize,
    pub face: usize,
    pub level: usize,
    pub face_node: usize,
    pub boundary_tag: Option<BoundaryTag>,
    pub interior_value: f64,
    /// Outward volume flux of the layer through the face node (m²/s; the sign
    /// is what matters: negative is inflow).
    pub normal_velocity: f64,
}

/// Boundary condition for scalar 3D tracer concentrations.
///
/// [`crate::solver::rhs::apply_tracer_transport_3d`] consults it for the
/// concentration of water flowing in through a physical boundary. The tracer
/// flux is the layer's volume flux times that concentration, so a wall (no
/// volume flux; the 2D wall condition) passes no tracer, whatever the value,
/// and an open boundary passes exactly the tracer its volume flux carries.
/// The layer volume fluxes at open boundaries carry the 2D open-boundary flux
/// in the interior's vertical profile (see
/// [`crate::solver::rhs::boundary_3d`]).
pub trait TracerBoundaryCondition3D: Send + Sync {
    fn exterior_value(&self, ctx: &TracerBCContext3D) -> f64;
}

/// Zero-gradient tracer boundary condition.
pub struct ExtrapolationTracerBC3D;

impl TracerBoundaryCondition3D for ExtrapolationTracerBC3D {
    fn exterior_value(&self, ctx: &TracerBCContext3D) -> f64 {
        ctx.interior_value
    }
}

/// Fixed tracer concentration on all boundary faces.
pub struct FixedTracerBC3D {
    pub value: f64,
}

impl FixedTracerBC3D {
    pub fn new(value: f64) -> Self {
        Self { value }
    }
}

impl TracerBoundaryCondition3D for FixedTracerBC3D {
    fn exterior_value(&self, _ctx: &TracerBCContext3D) -> f64 {
        self.value
    }
}

/// Fixed concentration for inflow, zero-gradient for outflow.
pub struct UpwindTracerBC3D {
    pub inflow_value: f64,
}

impl UpwindTracerBC3D {
    pub fn new(inflow_value: f64) -> Self {
        Self { inflow_value }
    }
}

impl TracerBoundaryCondition3D for UpwindTracerBC3D {
    fn exterior_value(&self, ctx: &TracerBCContext3D) -> f64 {
        if ctx.normal_velocity < 0.0 {
            self.inflow_value
        } else {
            ctx.interior_value
        }
    }
}

/// Compute volume divergence term assuming constant geometric factors.
///
/// Output:
/// out_dr = Dr * (rx*Fx + ry*Fy)
/// out_ds = Ds * (sx*Fx + sy*Fy)
///
/// So Div = out_dr + out_ds
pub fn compute_strong_divergence(
    fx: &[f64],
    fy: &[f64],
    ops: &DGOperators2D,
    rx: f64,
    ry: f64,
    sx: f64,
    sy: f64,
    out_dr: &mut [f64],
    out_ds: &mut [f64],
) {
    let n = ops.n_nodes;

    // Temp buffers for transformed flux components
    // Fr = rx*Fx + ry*Fy
    // Fs = sx*Fx + sy*Fy
    let mut fr = vec![0.0; n];
    let mut fs = vec![0.0; n];

    for i in 0..n {
        fr[i] = rx * fx[i] + ry * fy[i];
        fs[i] = sx * fx[i] + sy * fy[i];
    }

    // Matmul: out_dr = Dr * fr
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += ops.dr[(i, j)] * fr[j];
        }
        out_dr[i] = sum;
    }

    // Matmul: out_ds = Ds * fs
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += ops.ds[(i, j)] * fs[j];
        }
        out_ds[i] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-12,
            "expected {b}, got {a}, error {}",
            (a - b).abs()
        );
    }

    #[test]
    fn upwind_tracer_bc_uses_fixed_value_only_on_inflow() {
        let bc = UpwindTracerBC3D::new(3.5);
        let mut ctx = TracerBCContext3D {
            element: 0,
            face: 0,
            level: 0,
            face_node: 0,
            boundary_tag: None,
            interior_value: 8.0,
            normal_velocity: -0.2,
        };

        assert_close(bc.exterior_value(&ctx), 3.5);

        ctx.normal_velocity = 0.2;
        assert_close(bc.exterior_value(&ctx), 8.0);
    }
}
