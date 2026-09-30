//! 3D RHS computation.
//!
//! Assembles the semi-discrete right-hand side of the 3D primitive equations
//! in two parts, because under mode splitting they need different inputs:
//!
//! - [`compute_momentum_rhs_3d`]: the horizontal momentum terms (baroclinic
//!   pressure gradient, horizontal advection, Coriolis). The slow forcing `G`
//!   of the barotropic mode is built from their depth mean, before the
//!   barotropic pass of the step.
//! - [`compute_transport_rhs_3d`]: everything that moves with the layer volume
//!   fluxes (vertical momentum advection, tracer advection). The fluxes are
//!   corrected to the barotropic transport of the step
//!   ([`crate::solver::rhs::LayerTransport`]), so they exist only after the
//!   barotropic pass. Vertical momentum advection has zero depth mean, so `G`
//!   does not depend on it.
//!
//! Vertical mixing is handled implicitly in the time stepper.

use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::rhs::advection_3d::{
    TracerBoundaryCondition3D, apply_horizontal_advection_3d, apply_vertical_advection_3d,
};
use crate::solver::rhs::baroclinic::compute_pressure_gradient;
use crate::solver::rhs::boundary_3d::Boundaries3D;
use crate::solver::rhs::coriolis_3d::apply_coriolis_3d;
use crate::solver::rhs::transport_3d::{
    LayerTransport, TracerTransportScratch, apply_tracer_transport_3d,
};
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
    pub temp_bc: &'a dyn TracerBoundaryCondition3D,
    pub salt_bc: &'a dyn TracerBoundaryCondition3D,
    /// Walls and open faces of the domain.
    pub boundaries: &'a Boundaries3D,
    pub g: f64,
    pub rho0: f64,
    /// Columns shallower than this (m) are thin (3D wetting and drying).
    pub min_column_depth: f64,
}

/// Overwrite `rhs.u` and `rhs.v` with the horizontal momentum tendency:
/// baroclinic pressure gradient, horizontal advection and Coriolis. The other
/// fields of `rhs` are left alone.
///
/// `state.rho` must be current (density is refreshed by the caller, so that
/// `state` stays immutable here).
pub fn compute_momentum_rhs_3d(rhs: &mut Solution3D, state: &Solution3D, config: &Rhs3DConfig) {
    // The BAROCLINIC-ONLY PGF (rho_ref = ρ₀), which overwrites rhs.u/rhs.v.
    // Under mode splitting the barotropic term −g∇η is supplied by the 2D
    // sub-model's ½gh² flux; including it here too would double-count it in
    // the depth-averaged G-term coupling (see ModeSplitIntegrator). The ρ₀
    // part of the full PGF integrates to exactly −g∇η, so subtracting ρ₀
    // leaves only the density-driven baroclinic force.
    compute_pressure_gradient(
        state,
        config.mesh,
        config.bathymetry,
        config.sigma,
        config.ops,
        config.geom,
        config.g,
        config.rho0,
        config.rho0, // baroclinic-only PGF
        config.min_column_depth,
        &mut rhs.u,
        &mut rhs.v,
    );
    apply_horizontal_advection_3d(
        rhs,
        state,
        config.mesh,
        config.ops,
        config.geom,
        config.boundaries,
    );
    apply_coriolis_3d(rhs, state, config.mesh, config.ops, config.coriolis);
    // TODO: horizontal viscosity/diffusion (P4.5)
}

/// Add the vertical momentum advection by `transport.omega` to `rhs.u` and
/// `rhs.v`, and overwrite `rhs.temp` and `rhs.salt` with the tracers'
/// **inventory** tendencies `∂(H_z C)/∂t` under the layer transports
/// `transport` (computed from `state` for this stage).
pub fn compute_transport_rhs_3d(
    rhs: &mut Solution3D,
    state: &Solution3D,
    transport: &LayerTransport,
    config: &Rhs3DConfig,
    scratch: &mut TracerTransportScratch,
) {
    apply_vertical_advection_3d(
        rhs,
        state,
        &transport.omega,
        config.sigma,
        config.bathymetry,
    );
    let tracers = [
        (&mut rhs.temp, &state.temp, config.temp_bc),
        (&mut rhs.salt, &state.salt, config.salt_bc),
    ];
    for (out, tracer, bc) in tracers {
        apply_tracer_transport_3d(
            out,
            tracer,
            transport,
            config.mesh,
            config.ops,
            config.geom,
            bc,
            config.boundaries,
            scratch,
        );
    }
}
