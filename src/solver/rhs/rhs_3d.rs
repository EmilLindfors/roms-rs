//! 3D RHS computation.
//!
//! Assembles the full semi-discrete right-hand side for the 3D primitive equations.
//!
//! RHS = Advection + Coriolis + Pressure Gradient + (Horizontal Viscosity)
//!
//! Vertical mixing is handled implicitly in the time stepper.

use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::physics::eos::EquationOfState;
use crate::solver::rhs::advection_3d::{
    TracerBoundaryCondition3D, apply_horizontal_advection_3d, apply_tracer_advection_3d,
    apply_vertical_advection_3d,
};
use crate::solver::rhs::baroclinic::compute_pressure_gradient;
use crate::solver::rhs::coriolis_3d::apply_coriolis_3d;
use crate::solver::state::Solution3D;
use crate::source::CoriolisSource2D;
use crate::vertical::SigmaGrid;

/// Configuration for 3D RHS.
pub struct Rhs3DConfig<'a> {
    pub mesh: &'a Mesh2D,
    pub ops: &'a DGOperators2D,
    pub geom: &'a GeometricFactors2D,
    pub bathymetry: &'a Bathymetry2D,
    pub sigma: &'a SigmaGrid,
    pub coriolis: &'a CoriolisSource2D,
    pub eos: &'a dyn EquationOfState,
    pub temp_bc: &'a dyn TracerBoundaryCondition3D,
    pub salt_bc: &'a dyn TracerBoundaryCondition3D,
    pub g: f64,
    pub rho0: f64,
}

/// Compute the explicit RHS for the 3D momentum equations.
///
/// Updates `rhs.u` and `rhs.v` with tendencies from advection, Coriolis, and pressure gradient.
///
/// # Arguments
/// * `rhs` - Accumulator for RHS tendencies (will be overwritten/added to)
/// * `state` - Current state
/// * `w_vel` - Vertical velocity field (diagnostic)
/// * `config` - Static configuration and auxiliary data
pub fn compute_rhs_3d(
    rhs: &mut Solution3D,
    state: &Solution3D,
    w_vel: &[f64],
    config: &Rhs3DConfig,
) {
    // 1. Update density from T, S
    // Note: Density update is handled externally (e.g. at start of step)
    // to allow state to be immutable during RHS evaluation.
    // config.eos.update_density(state);

    // 2. Compute Pressure Gradient Force (PGF)
    // We compute this FIRST and overwrite rhs.u/rhs.v because compute_pressure_gradient
    // takes mutable slices for output and overwrites them.
    // This effectively clears the RHS accumulator for momentum.
    //
    // We request the BAROCLINIC-ONLY PGF (rho_ref = ρ₀). Under mode splitting the
    // barotropic term −g∇η is supplied by the 2D sub-model's ½gh² flux; including
    // it here too would double-count it in the depth-averaged G-term coupling
    // (see ModeSplitIntegrator). The ρ₀ part of the full PGF integrates to exactly
    // −g∇η, so subtracting ρ₀ leaves only the density-driven baroclinic force.
    compute_pressure_gradient(
        state,
        config.bathymetry,
        config.sigma,
        config.ops,
        config.geom,
        config.g,
        config.rho0,
        config.rho0, // baroclinic-only PGF
        &mut rhs.u,
        &mut rhs.v,
    );

    // 3. Horizontal Advection (Momentum)
    // Adds to RHS
    apply_horizontal_advection_3d(rhs, state, config.mesh, config.ops, config.geom);

    // 4. Coriolis
    // Adds to RHS
    apply_coriolis_3d(rhs, state, config.mesh, config.ops, config.coriolis);

    // 5. Horizontal Viscosity/Diffusion
    // TODO: Add horizontal mixing terms here

    // 6. Tracer Horizontal Advection
    // RHS_T = -Adv(T)
    rhs.temp.fill(0.0);
    rhs.salt.fill(0.0);

    apply_tracer_advection_3d(
        &mut rhs.temp,
        &state.temp,
        state,
        config.mesh,
        config.ops,
        config.geom,
        config.bathymetry,
        config.sigma,
        config.temp_bc,
    );
    apply_tracer_advection_3d(
        &mut rhs.salt,
        &state.salt,
        state,
        config.mesh,
        config.ops,
        config.geom,
        config.bathymetry,
        config.sigma,
        config.salt_bc,
    );

    // 7. Vertical Advection (All Fields)
    // Must be called after horizontal terms are accumulated, as it adds to them.
    apply_vertical_advection_3d(rhs, state, w_vel, config.sigma, config.bathymetry);
}
