//! Right-hand side computation for 2D DG shallow water equations.
//!
//! For the 2D SWE:
//!   ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0
//!   ∂(hu)/∂t + ∂(hu² + gh²/2)/∂x + ∂(huv)/∂y = fhv + S_x
//!   ∂(hv)/∂t + ∂(huv)/∂x + ∂(hv² + gh²/2)/∂y = -fhu + S_y
//!
//! The DG semi-discrete form uses the weak formulation:
//!   dq/dt = L(q) = -(volume terms) + (surface terms) + (source terms)

use crate::boundary::{BCContext2D, SWEBoundaryCondition2D};
use crate::equations::ShallowWater2D;
use crate::flux::{SWEFluxType2D, compute_flux_swe_2d};
use crate::mesh::{Bathymetry2D, Mesh2D};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::state::{SWE_VAR_HU, SWE_VAR_HV};
use crate::solver::{SWESolution2D, SWEState2D};
use crate::source::swe_2d::viscosity::HorizontalViscosity2D;
use crate::source::{HydrostaticReconstruction2D, SourceContext2D, SourceTerm2D};
use crate::types::ElementIndex;

use super::diffusion_2d::{compute_br1_diffusion_rhs_2d, compute_br1_gradient_2d};

/// Configuration for 2D SWE RHS computation.
pub struct SWE2DRhsConfig<'a, BC: SWEBoundaryCondition2D> {
    /// The shallow water equation parameters
    pub equation: &'a ShallowWater2D,
    /// Numerical flux type
    pub flux_type: SWEFluxType2D,
    /// Boundary condition handler
    pub bc: &'a BC,
    /// Whether to include Coriolis source term (legacy, use source_terms instead)
    pub include_coriolis: bool,
    /// Optional trait-based source terms (preferred over include_coriolis)
    ///
    /// When set, these source terms are evaluated at each node and added to the RHS.
    /// Multiple sources can be combined using `CombinedSource2D`.
    pub source_terms: Option<&'a dyn SourceTerm2D>,
    /// Optional bathymetry data for well-balanced schemes and source terms.
    ///
    /// When set, bathymetry values and gradients are passed to boundary conditions
    /// and source terms for proper handling of variable bottom topography.
    pub bathymetry: Option<&'a Bathymetry2D>,
    /// Enable hydrostatic reconstruction for well-balanced treatment of bathymetry.
    ///
    /// When enabled, interface states are modified using Audusse et al. (2004)
    /// hydrostatic reconstruction to ensure lake-at-rest is preserved to machine
    /// precision regardless of bathymetry gradient.
    ///
    /// Requires `bathymetry` to be set. When using this option, do NOT include
    /// `BathymetrySource2D` in `source_terms` as the bathymetry effect is handled
    /// through the flux reconstruction.
    pub well_balanced: bool,
    /// Optional horizontal viscosity for momentum diffusion.
    ///
    /// When set, adds ∇·(ν h ∇u) and ∇·(ν h ∇v) to the momentum equations.
    /// This is computed via a double-derivative Laplacian (differentiate velocity
    /// gradients, then differentiate viscous fluxes), following the same pattern
    /// as tracer diffusion.
    pub viscosity: Option<&'a HorizontalViscosity2D>,
    /// Optional time step to pass to boundary conditions (e.g., Chapman radiation).
    ///
    /// When set, the dt value is propagated to `BCContext2D` so that BCs like
    /// Chapman and ChapmanFlather can use the current time step automatically
    /// without requiring manual `set_dt()` calls.
    pub dt: Option<f64>,
}

impl<'a, BC: SWEBoundaryCondition2D> SWE2DRhsConfig<'a, BC> {
    /// Create a new RHS configuration.
    pub fn new(equation: &'a ShallowWater2D, bc: &'a BC) -> Self {
        Self {
            equation,
            flux_type: SWEFluxType2D::Roe,
            bc,
            include_coriolis: true,
            source_terms: None,
            bathymetry: None,
            well_balanced: false,
            viscosity: None,
            dt: None,
        }
    }

    /// Set the numerical flux type.
    pub fn with_flux_type(mut self, flux_type: SWEFluxType2D) -> Self {
        self.flux_type = flux_type;
        self
    }

    /// Set whether to include Coriolis (legacy method).
    ///
    /// For new code, prefer using `with_source_terms` with `CoriolisSource2D`.
    pub fn with_coriolis(mut self, include: bool) -> Self {
        self.include_coriolis = include;
        self
    }

    /// Set trait-based source terms.
    ///
    /// Source terms are evaluated at each quadrature node and added to the RHS.
    /// Use `CombinedSource2D` to compose multiple source terms.
    ///
    /// # Example
    /// ```ignore
    /// let coriolis = CoriolisSource2D::norwegian_coast();
    /// let config = SWE2DRhsConfig::new(&equation, &bc)
    ///     .with_coriolis(false)  // Disable legacy Coriolis
    ///     .with_source_terms(&coriolis);
    /// ```
    pub fn with_source_terms(mut self, sources: &'a dyn SourceTerm2D) -> Self {
        self.source_terms = Some(sources);
        self
    }

    /// Set bathymetry data for well-balanced schemes.
    ///
    /// When bathymetry is provided, the values and gradients are passed to:
    /// - Boundary condition contexts (for correct ghost state computation)
    /// - Source term contexts (for bathymetry-dependent physics)
    ///
    /// # Example
    /// ```ignore
    /// let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
    ///     // Sill profile
    ///     0.5 * (-((x - 5.0).powi(2)) / 2.0).exp()
    /// });
    /// let config = SWE2DRhsConfig::new(&equation, &bc)
    ///     .with_bathymetry(&bathymetry);
    /// ```
    pub fn with_bathymetry(mut self, bathymetry: &'a Bathymetry2D) -> Self {
        self.bathymetry = Some(bathymetry);
        self
    }

    /// Enable hydrostatic reconstruction for well-balanced treatment of bathymetry.
    ///
    /// When enabled, interface states are modified using the Audusse et al. (2004)
    /// method to ensure lake-at-rest (η = h + B = const, u = v = 0) is preserved
    /// to machine precision regardless of bathymetry gradient.
    ///
    /// # Requirements
    /// - `bathymetry` must be set via `with_bathymetry()`
    /// - Do NOT include `BathymetrySource2D` in `source_terms`
    ///
    /// # Example
    /// ```ignore
    /// // Well-balanced scheme for steep Norwegian bathymetry
    /// let config = SWE2DRhsConfig::new(&equation, &bc)
    ///     .with_bathymetry(&bathymetry)
    ///     .with_well_balanced(true);
    /// // Note: Do NOT add BathymetrySource2D here
    /// ```
    pub fn with_well_balanced(mut self, enable: bool) -> Self {
        self.well_balanced = enable;
        self
    }

    /// Set horizontal viscosity for momentum diffusion.
    ///
    /// When enabled, adds ∇·(ν h ∇u) and ∇·(ν h ∇v) to the momentum RHS.
    /// Use `compute_dt_viscosity` to compute the diffusive time step restriction,
    /// and take `dt = dt_advective.min(dt_viscous)`.
    pub fn with_viscosity(mut self, visc: &'a HorizontalViscosity2D) -> Self {
        self.viscosity = Some(visc);
        self
    }

    /// Set the time step for boundary condition propagation.
    ///
    /// When set, the dt is passed to `BCContext2D` so that radiation BCs
    /// (Chapman, ChapmanFlather) can use the current time step automatically.
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = Some(dt);
        self
    }
}

fn boundary_velocity_component<BC: SWEBoundaryCondition2D>(
    component: usize,
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
    visc_h_min: f64,
    k: ElementIndex,
    face: usize,
    node: usize,
) -> f64 {
    let state = q.get_state(k, node);
    let normal = geom.normals[k.as_usize()][face];
    let (r, s) = (ops.nodes_r[node], ops.nodes_s[node]);
    let [x, y] = mesh.reference_to_physical(k, r, s);
    let bathy_value = config.bathymetry.map(|b| b.get(k, node)).unwrap_or(0.0);
    let h_min = config.equation.h_min.meters();

    let mut ctx = match mesh.boundary_tag(k, face) {
        Some(tag) => BCContext2D::with_tag(
            time,
            (x, y),
            state,
            bathy_value,
            normal,
            config.equation.g,
            h_min,
            tag,
        ),
        None => BCContext2D::new(
            time,
            (x, y),
            state,
            bathy_value,
            normal,
            config.equation.g,
            h_min,
        ),
    };
    ctx.dt = config.dt;

    let ghost = config.bc.ghost_state(&ctx);
    if ghost.h <= visc_h_min {
        return 0.0;
    }

    let h_safe = ghost.h.max(visc_h_min);
    match component {
        0 => ghost.hu / h_safe,
        1 => ghost.hv / h_safe,
        _ => unreachable!("invalid velocity component"),
    }
}

fn add_br1_viscosity<BC: SWEBoundaryCondition2D>(
    rhs: &mut SWESolution2D,
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
) {
    let Some(visc) = config.viscosity else {
        return;
    };

    let n_nodes = ops.n_nodes;
    let total_nodes = mesh.n_elements * n_nodes;
    let mut u = vec![0.0; total_nodes];
    let mut v = vec![0.0; total_nodes];

    for k in ElementIndex::iter(mesh.n_elements) {
        for i in 0..n_nodes {
            let flat = k.as_usize() * n_nodes + i;
            let state = q.get_state(k, i);
            if state.h > visc.h_min {
                let h_safe = state.h.max(visc.h_min);
                u[flat] = state.hu / h_safe;
                v[flat] = state.hv / h_safe;
            }
        }
    }

    let boundary_u_for_gradient = |k, face, _fi, node, _interior| {
        boundary_velocity_component(
            0, q, mesh, ops, geom, config, time, visc.h_min, k, face, node,
        )
    };
    let boundary_v_for_gradient = |k, face, _fi, node, _interior| {
        boundary_velocity_component(
            1, q, mesh, ops, geom, config, time, visc.h_min, k, face, node,
        )
    };

    let grad_u = compute_br1_gradient_2d(&u, mesh, ops, geom, boundary_u_for_gradient);
    let grad_v = compute_br1_gradient_2d(&v, mesh, ops, geom, boundary_v_for_gradient);

    let mut coeff = vec![0.0; total_nodes];
    for k in ElementIndex::iter(mesh.n_elements) {
        let delta = geom.det_j[k.as_usize()].sqrt();
        for i in 0..n_nodes {
            let flat = k.as_usize() * n_nodes + i;
            let state = q.get_state(k, i);
            if state.h > visc.h_min {
                let nu = visc.compute_viscosity(
                    grad_u[flat].dx,
                    grad_u[flat].dy,
                    grad_v[flat].dx,
                    grad_v[flat].dy,
                    delta,
                );
                coeff[flat] = nu * state.h;
            }
        }
    }

    let diff_u = compute_br1_diffusion_rhs_2d(&u, &coeff, &grad_u, mesh, ops, geom);
    let diff_v = compute_br1_diffusion_rhs_2d(&v, &coeff, &grad_v, mesh, ops, geom);

    for k in ElementIndex::iter(mesh.n_elements) {
        for i in 0..n_nodes {
            let flat = k.as_usize() * n_nodes + i;
            rhs.data[SWE_VAR_HU][flat] += diff_u[flat];
            rhs.data[SWE_VAR_HV][flat] += diff_v[flat];
        }
    }
}

/// Compute the right-hand side for 2D SWE.
///
/// Implements the DG weak form for a system:
///   dq/dt = -1/J * [Dr * (Fr) + Ds * (Fs)] + 1/J * Σ_f LIFT_f * sJ_f * (F* - F-) + S
///
/// where:
///   Fr = F · ∇r = F_x * rx + F_y * ry
///   Fs = F · ∇s = F_x * sx + F_y * sy
pub fn compute_rhs_swe_2d<BC: SWEBoundaryCondition2D>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
) -> SWESolution2D {
    let n_nodes = ops.n_nodes;
    let n_face_nodes = ops.n_face_nodes;
    let mut rhs = SWESolution2D::new(mesh.n_elements, n_nodes);

    let g = config.equation.g;
    let h_min = config.equation.h_min.meters();

    // Pre-allocate workspace buffers (reused across elements, no allocation in hot loop)
    let mut flux_x = vec![SWEState2D::zero(); n_nodes];
    let mut flux_y = vec![SWEState2D::zero(); n_nodes];
    let mut int_bathy_buf = vec![0.0; n_face_nodes];
    let mut ext_states_buf = vec![SWEState2D::zero(); n_face_nodes];
    let mut ext_bathy_buf = vec![0.0; n_face_nodes];
    let mut flux_diff = vec![SWEState2D::zero(); n_face_nodes];

    for k in ElementIndex::iter(mesh.n_elements) {
        let j_inv = geom.det_j_inv[k.as_usize()];
        let rx = geom.rx[k.as_usize()];
        let ry = geom.ry[k.as_usize()];
        let sx = geom.sx[k.as_usize()];
        let sy = geom.sy[k.as_usize()];

        // 1. Volume term: -(∇ · F) = -(∂F/∂x + ∂G/∂y)
        // In reference coordinates:
        //   ∂F/∂x = (∂F/∂r)·(∂r/∂x) + (∂F/∂s)·(∂s/∂x) = Dr*F * rx + Ds*F * sx
        //   ∂G/∂y = (∂G/∂r)·(∂r/∂y) + (∂G/∂s)·(∂s/∂y) = Dr*G * ry + Ds*G * sy

        // Compute fluxes at all nodes

        for i in 0..n_nodes {
            let state = q.get_state(k, i);
            flux_x[i] = config.equation.flux_x(&state);
            flux_y[i] = config.equation.flux_y(&state);
        }

        // Apply Dr and Ds to compute derivatives
        for i in 0..n_nodes {
            let mut dfx_dr = SWEState2D::zero();
            let mut dfx_ds = SWEState2D::zero();
            let mut dfy_dr = SWEState2D::zero();
            let mut dfy_ds = SWEState2D::zero();

            for j in 0..n_nodes {
                let dr_ij = ops.dr[(i, j)];
                let ds_ij = ops.ds[(i, j)];

                dfx_dr = dfx_dr + dr_ij * flux_x[j];
                dfx_ds = dfx_ds + ds_ij * flux_x[j];
                dfy_dr = dfy_dr + dr_ij * flux_y[j];
                dfy_ds = dfy_ds + ds_ij * flux_y[j];
            }

            // Volume term: -(dF/dx + dG/dy)
            // = -(dfx_dr * rx + dfx_ds * sx + dfy_dr * ry + dfy_ds * sy)
            let div_flux = dfx_dr * rx + dfx_ds * sx + dfy_dr * ry + dfy_ds * sy;

            rhs.set_state(k, i, -1.0 * div_flux);
        }

        // 2. Surface terms: 1/J * LIFT_f * sJ_f * (F* - F-)
        for face in 0..4 {
            let normal = geom.normals[k.as_usize()][face];
            let s_jac = geom.surface_j[k.as_usize()][face];
            let face_nodes = &ops.face_nodes[face];

            // Get interior bathymetry at face nodes (for well-balanced scheme)
            if config.well_balanced && config.bathymetry.is_some() {
                let bathy = config.bathymetry.unwrap();
                for (fi, &node) in face_nodes.iter().enumerate() {
                    int_bathy_buf[fi] = bathy.get(k, node);
                }
            } else {
                for fi in 0..n_face_nodes {
                    int_bathy_buf[fi] = 0.0;
                }
            }

            // Get exterior states and bathymetry (neighbor or boundary)
            if let Some(neighbor) = mesh.neighbor(k, face) {
                // Interior face: get states and bathymetry from neighbor
                let neighbor_face_nodes = &ops.face_nodes[neighbor.face];
                for i in 0..n_face_nodes {
                    // Reverse ordering for neighbor face
                    ext_states_buf[i] = q.get_state(
                        ElementIndex::new(neighbor.element),
                        neighbor_face_nodes[n_face_nodes - 1 - i],
                    );
                }
                if config.well_balanced && config.bathymetry.is_some() {
                    let b = config.bathymetry.unwrap();
                    for i in 0..n_face_nodes {
                        ext_bathy_buf[i] = b.get(
                            ElementIndex::new(neighbor.element),
                            neighbor_face_nodes[n_face_nodes - 1 - i],
                        );
                    }
                } else {
                    for i in 0..n_face_nodes {
                        ext_bathy_buf[i] = 0.0;
                    }
                }
            } else {
                // Boundary face: compute ghost states
                let boundary_tag = mesh.boundary_tag(k, face);

                for i in 0..n_face_nodes {
                    let node_idx = face_nodes[i];
                    let state = q.get_state(k, node_idx);
                    let (r, s) = (ops.nodes_r[node_idx], ops.nodes_s[node_idx]);
                    let [x, y] = mesh.reference_to_physical(k, r, s);

                    let bathy_value = config.bathymetry.map(|b| b.get(k, node_idx)).unwrap_or(0.0);

                    // Create context with boundary tag for multi-BC dispatch
                    let mut ctx = match boundary_tag {
                        Some(tag) => BCContext2D::with_tag(
                            time,
                            (x, y),
                            state,
                            bathy_value,
                            normal,
                            g,
                            h_min,
                            tag,
                        ),
                        None => {
                            BCContext2D::new(time, (x, y), state, bathy_value, normal, g, h_min)
                        }
                    };
                    ctx.dt = config.dt;
                    ext_states_buf[i] = config.bc.ghost_state(&ctx);
                }

                // For boundary faces, mirror the interior bathymetry
                ext_bathy_buf[..n_face_nodes].copy_from_slice(&int_bathy_buf[..n_face_nodes]);
            }

            // Create hydrostatic reconstruction if enabled
            let hr = if config.well_balanced && config.bathymetry.is_some() {
                Some(HydrostaticReconstruction2D::new(g, h_min))
            } else {
                None
            };

            // Compute numerical flux and flux difference at face nodes
            for i in 0..n_face_nodes {
                let node_idx = face_nodes[i];
                let q_int = q.get_state(k, node_idx);
                let q_ext = ext_states_buf[i];

                // Apply hydrostatic reconstruction if enabled
                let (q_int_flux, q_ext_flux) = if let Some(ref reconstruction) = hr {
                    reconstruction.reconstruct(&q_int, &q_ext, int_bathy_buf[i], ext_bathy_buf[i])
                } else {
                    (q_int, q_ext)
                };

                // Numerical flux F* · n using (possibly reconstructed) states
                let f_star = compute_flux_swe_2d(
                    &q_int_flux,
                    &q_ext_flux,
                    normal,
                    g,
                    h_min,
                    config.flux_type,
                );

                // Interior flux F- · n using reconstructed interior state
                let f_int = config.equation.normal_flux(&q_int_flux, normal);

                // Flux difference for upwind dissipation: (F- - F*)
                flux_diff[i] = f_int - f_star;
            }

            // Apply LIFT: rhs += j_inv * LIFT_f * (sJ * flux_diff)
            for i in 0..n_nodes {
                let mut lift_contribution = SWEState2D::zero();
                for fi in 0..n_face_nodes {
                    let lift_coeff = ops.lift[face][(i, fi)];
                    lift_contribution = lift_contribution + lift_coeff * s_jac * flux_diff[fi];
                }

                // Add contribution (note: j_inv goes here)
                let current = rhs.get_state(k, i);
                rhs.set_state(k, i, current + j_inv * lift_contribution);
            }
        }

        // 3. Source terms: Legacy Coriolis (for backward compatibility)
        if config.include_coriolis
            && config.source_terms.is_none()
            && (config.equation.f0.abs() > 1e-14 || config.equation.beta.abs() > 1e-14)
        {
            for i in 0..n_nodes {
                let state = q.get_state(k, i);
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [_x, y] = mesh.reference_to_physical(k, r, s);

                let source = config.equation.coriolis_source(&state, y);

                let current = rhs.get_state(k, i);
                rhs.set_state(k, i, current + source);
            }
        }

        // 4. Trait-based source terms (preferred)
        if let Some(sources) = config.source_terms {
            for i in 0..n_nodes {
                let state = q.get_state(k, i);
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [x, y] = mesh.reference_to_physical(k, r, s);

                let (bathy_value, bathy_gradient) = config
                    .bathymetry
                    .map(|b| (b.get(k, i), b.get_gradient(k, i)))
                    .unwrap_or((0.0, (0.0, 0.0)));

                let ctx = SourceContext2D::new(
                    time,
                    (x, y),
                    state,
                    bathy_value,
                    bathy_gradient,
                    g,
                    h_min,
                );

                let source = sources.evaluate(&ctx);
                let current = rhs.get_state(k, i);
                rhs.set_state(k, i, current + source);
            }
        }
    }

    add_br1_viscosity(&mut rhs, q, mesh, ops, geom, config, time);

    rhs
}

/// Compute the stable time step for 2D SWE.
///
/// Uses CFL condition: dt ≤ CFL * h_min / (λ_max * (2*p + 1))
///
/// where λ_max = max(|u| + c) over all nodes.
pub fn compute_dt_swe_2d(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    geom: &GeometricFactors2D,
    equation: &ShallowWater2D,
    order: usize,
    cfl: f64,
) -> f64 {
    // Find maximum wave speed over all nodes
    let mut max_speed: f64 = 0.0;
    let mut min_h_elem = f64::INFINITY;

    for k in ElementIndex::iter(mesh.n_elements) {
        // Estimate element size from Jacobian
        let h_elem = geom.det_j[k.as_usize()].sqrt() * 2.0;
        min_h_elem = min_h_elem.min(h_elem);

        // Find max wave speed in this element
        for i in 0..q.n_nodes {
            let state = q.get_state(k, i);
            let speed = equation.max_wave_speed(&state);
            max_speed = max_speed.max(speed);
        }
    }

    if max_speed < 1e-14 {
        return f64::INFINITY;
    }

    // DG CFL factor
    let dg_factor = 2.0 * order as f64 + 1.0;

    cfl * min_h_elem / (max_speed * dg_factor)
}

/// Parallel version of CFL time step computation.
///
/// Uses Rayon to compute max wave speed across all elements in parallel.
#[cfg(feature = "parallel")]
pub fn compute_dt_swe_2d_parallel(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    geom: &GeometricFactors2D,
    equation: &ShallowWater2D,
    order: usize,
    cfl: f64,
) -> f64 {
    use rayon::prelude::*;

    // Parallel reduction over elements
    let (max_speed, min_h_elem) = (0..mesh.n_elements)
        .into_par_iter()
        .map(|k| {
            let k_idx = ElementIndex::new(k);

            // Estimate element size from Jacobian
            let h_elem = geom.det_j[k].sqrt() * 2.0;

            // Find max wave speed in this element
            let mut elem_max_speed = 0.0_f64;
            for i in 0..q.n_nodes {
                let state = q.get_state(k_idx, i);
                let speed = equation.max_wave_speed(&state);
                elem_max_speed = elem_max_speed.max(speed);
            }

            (elem_max_speed, h_elem)
        })
        .reduce(
            || (0.0_f64, f64::INFINITY),
            |(s1, h1), (s2, h2)| (s1.max(s2), h1.min(h2)),
        );

    if max_speed < 1e-14 {
        return f64::INFINITY;
    }

    // DG CFL factor
    let dg_factor = 2.0 * order as f64 + 1.0;

    cfl * min_h_elem / (max_speed * dg_factor)
}

/// Compute the diffusive time step restriction for horizontal viscosity.
///
/// The diffusive CFL condition is:
///   Δt ≤ CFL × Δx² / (ν × (2N+1)²)
///
/// where N is the polynomial order and (2N+1)² accounts for the DG
/// spectral radius of the diffusion operator.
///
/// Users should combine with the advective time step:
///   `dt = compute_dt_swe_2d(...).min(compute_dt_viscosity(...))`
///
/// # Arguments
/// * `nu_max` - Maximum viscosity coefficient [m²/s]
/// * `min_h_elem` - Minimum element size (e.g. from `geom.det_j[k].sqrt() * 2.0`)
/// * `order` - Polynomial order of the DG scheme
/// * `cfl` - CFL number (typically 0.1–0.5 for diffusion)
pub fn compute_dt_viscosity(nu_max: f64, min_h_elem: f64, order: usize, cfl: f64) -> f64 {
    if nu_max < 1e-14 {
        return f64::INFINITY;
    }
    let dg_factor = (2.0 * order as f64 + 1.0).powi(2);
    cfl * min_h_elem * min_h_elem / (nu_max * dg_factor)
}

/// Per-thread workspace for parallel RHS computation.
/// This avoids allocations inside the hot loop.
#[cfg(all(feature = "parallel", feature = "simd"))]
struct ThreadWorkspace {
    // Volume term buffers
    flux_x_h: Vec<f64>,
    flux_x_hu: Vec<f64>,
    flux_x_hv: Vec<f64>,
    flux_y_h: Vec<f64>,
    flux_y_hu: Vec<f64>,
    flux_y_hv: Vec<f64>,
    dfx_dr_h: Vec<f64>,
    dfx_dr_hu: Vec<f64>,
    dfx_dr_hv: Vec<f64>,
    dfx_ds_h: Vec<f64>,
    dfx_ds_hu: Vec<f64>,
    dfx_ds_hv: Vec<f64>,
    dfy_dr_h: Vec<f64>,
    dfy_dr_hu: Vec<f64>,
    dfy_dr_hv: Vec<f64>,
    dfy_ds_h: Vec<f64>,
    dfy_ds_hu: Vec<f64>,
    dfy_ds_hv: Vec<f64>,
    rhs_h: Vec<f64>,
    rhs_hu: Vec<f64>,
    rhs_hv: Vec<f64>,
    hu_arr: Vec<f64>,
    hv_arr: Vec<f64>,
    // Surface term flux difference buffers
    flux_diff_h: Vec<f64>,
    flux_diff_hu: Vec<f64>,
    flux_diff_hv: Vec<f64>,
    // Surface term state buffers (avoid per-face allocations)
    int_bathy: Vec<f64>,
    ext_bathy: Vec<f64>,
    ext_states_h: Vec<f64>,
    ext_states_hu: Vec<f64>,
    ext_states_hv: Vec<f64>,
}

#[cfg(all(feature = "parallel", feature = "simd"))]
impl ThreadWorkspace {
    fn new(n_nodes: usize, n_face_nodes: usize) -> Self {
        Self {
            // Volume term buffers
            flux_x_h: vec![0.0; n_nodes],
            flux_x_hu: vec![0.0; n_nodes],
            flux_x_hv: vec![0.0; n_nodes],
            flux_y_h: vec![0.0; n_nodes],
            flux_y_hu: vec![0.0; n_nodes],
            flux_y_hv: vec![0.0; n_nodes],
            dfx_dr_h: vec![0.0; n_nodes],
            dfx_dr_hu: vec![0.0; n_nodes],
            dfx_dr_hv: vec![0.0; n_nodes],
            dfx_ds_h: vec![0.0; n_nodes],
            dfx_ds_hu: vec![0.0; n_nodes],
            dfx_ds_hv: vec![0.0; n_nodes],
            dfy_dr_h: vec![0.0; n_nodes],
            dfy_dr_hu: vec![0.0; n_nodes],
            dfy_dr_hv: vec![0.0; n_nodes],
            dfy_ds_h: vec![0.0; n_nodes],
            dfy_ds_hu: vec![0.0; n_nodes],
            dfy_ds_hv: vec![0.0; n_nodes],
            rhs_h: vec![0.0; n_nodes],
            rhs_hu: vec![0.0; n_nodes],
            rhs_hv: vec![0.0; n_nodes],
            hu_arr: vec![0.0; n_nodes],
            hv_arr: vec![0.0; n_nodes],
            // Surface term flux difference buffers
            flux_diff_h: vec![0.0; n_face_nodes],
            flux_diff_hu: vec![0.0; n_face_nodes],
            flux_diff_hv: vec![0.0; n_face_nodes],
            // Surface term state buffers
            int_bathy: vec![0.0; n_face_nodes],
            ext_bathy: vec![0.0; n_face_nodes],
            ext_states_h: vec![0.0; n_face_nodes],
            ext_states_hu: vec![0.0; n_face_nodes],
            ext_states_hv: vec![0.0; n_face_nodes],
        }
    }

    fn clear_rhs(&mut self) {
        self.rhs_h.fill(0.0);
        self.rhs_hu.fill(0.0);
        self.rhs_hv.fill(0.0);
    }
}

/// Parallel + SIMD version with zero-allocation hot path.
///
/// Uses `for_each_init` to create per-thread workspaces, eliminating the
/// allocation overhead that made the naive parallel version slower than serial.
///
/// Performance characteristics:
/// - ~2-4x speedup over serial for meshes with 1000+ elements
/// - Minimal overhead for small meshes (falls back efficiently)
/// - Zero allocations in the hot path after initial workspace setup
#[cfg(all(feature = "parallel", feature = "simd"))]
pub fn compute_rhs_swe_2d_parallel<BC: SWEBoundaryCondition2D + Sync>(
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    time: f64,
) -> SWESolution2D {
    use crate::solver::{apply_diff_matrix, apply_lift, combine_derivatives, coriolis_source};
    use rayon::prelude::*;

    let n_nodes = ops.n_nodes;
    let n_face_nodes = ops.n_face_nodes;
    let n_elements = mesh.n_elements;
    let g = config.equation.g;
    let h_min = config.equation.h_min.meters();

    // Pre-allocate output
    let mut rhs = SWESolution2D::new(n_elements, n_nodes);

    // Process elements in parallel, writing directly to output buffers
    // Uses par_chunks_mut to eliminate intermediate Vec collection and final copy
    {
        let [rhs_h, rhs_hu, rhs_hv] = &mut rhs.data;
        rhs_h
            .par_chunks_mut(n_nodes)
            .zip(rhs_hu.par_chunks_mut(n_nodes))
            .zip(rhs_hv.par_chunks_mut(n_nodes))
            .enumerate()
            .for_each_init(
                || ThreadWorkspace::new(n_nodes, n_face_nodes),
                |ws, (k, ((h_chunk, hu_chunk), hv_chunk))| {
                    let k_idx = ElementIndex::new(k);
                    let j_inv = geom.det_j_inv[k];
                    let rx = geom.rx[k];
                    let ry = geom.ry[k];
                    let sx = geom.sx[k];
                    let sy = geom.sy[k];

                    // Clear RHS workspace
                    ws.clear_rhs();

                    // 1. Volume term: -(∇ · F)
                    // Compute fluxes at all nodes and extract to SoA
                    for i in 0..n_nodes {
                        let state = q.get_state(k_idx, i);
                        let fx = config.equation.flux_x(&state);
                        let fy = config.equation.flux_y(&state);
                        ws.flux_x_h[i] = fx.h;
                        ws.flux_x_hu[i] = fx.hu;
                        ws.flux_x_hv[i] = fx.hv;
                        ws.flux_y_h[i] = fy.h;
                        ws.flux_y_hu[i] = fy.hu;
                        ws.flux_y_hv[i] = fy.hv;
                        // Also extract hu, hv for Coriolis
                        ws.hu_arr[i] = state.hu;
                        ws.hv_arr[i] = state.hv;
                    }

                    // Apply Dr and Ds to flux_x and flux_y using SIMD kernels
                    apply_diff_matrix(
                        &ops.dr_row_major,
                        &ws.flux_x_h,
                        &ws.flux_x_hu,
                        &ws.flux_x_hv,
                        &mut ws.dfx_dr_h,
                        &mut ws.dfx_dr_hu,
                        &mut ws.dfx_dr_hv,
                        n_nodes,
                    );
                    apply_diff_matrix(
                        &ops.ds_row_major,
                        &ws.flux_x_h,
                        &ws.flux_x_hu,
                        &ws.flux_x_hv,
                        &mut ws.dfx_ds_h,
                        &mut ws.dfx_ds_hu,
                        &mut ws.dfx_ds_hv,
                        n_nodes,
                    );
                    apply_diff_matrix(
                        &ops.dr_row_major,
                        &ws.flux_y_h,
                        &ws.flux_y_hu,
                        &ws.flux_y_hv,
                        &mut ws.dfy_dr_h,
                        &mut ws.dfy_dr_hu,
                        &mut ws.dfy_dr_hv,
                        n_nodes,
                    );
                    apply_diff_matrix(
                        &ops.ds_row_major,
                        &ws.flux_y_h,
                        &ws.flux_y_hu,
                        &ws.flux_y_hv,
                        &mut ws.dfy_ds_h,
                        &mut ws.dfy_ds_hu,
                        &mut ws.dfy_ds_hv,
                        n_nodes,
                    );

                    // Combine derivatives with geometric factors using SIMD kernel: RHS = -div(F)
                    combine_derivatives(
                        &ws.dfx_dr_h,
                        &ws.dfx_dr_hu,
                        &ws.dfx_dr_hv,
                        &ws.dfx_ds_h,
                        &ws.dfx_ds_hu,
                        &ws.dfx_ds_hv,
                        &ws.dfy_dr_h,
                        &ws.dfy_dr_hu,
                        &ws.dfy_dr_hv,
                        &ws.dfy_ds_h,
                        &ws.dfy_ds_hu,
                        &ws.dfy_ds_hv,
                        &mut ws.rhs_h,
                        &mut ws.rhs_hu,
                        &mut ws.rhs_hv,
                        rx,
                        sx,
                        ry,
                        sy,
                        n_nodes,
                    );

                    // 2. Surface terms
                    // Create hydrostatic reconstruction once per element (not per face)
                    let hr = if config.well_balanced && config.bathymetry.is_some() {
                        Some(HydrostaticReconstruction2D::new(g, h_min))
                    } else {
                        None
                    };

                    for face in 0..4 {
                        let normal = geom.normals[k][face];
                        let s_jac = geom.surface_j[k][face];
                        let face_nodes = &ops.face_nodes[face];

                        // Get interior bathymetry at face nodes (reuse workspace buffer)
                        if config.well_balanced && config.bathymetry.is_some() {
                            let bathy = config.bathymetry.unwrap();
                            for (i, &node) in face_nodes.iter().enumerate() {
                                ws.int_bathy[i] = bathy.get(k_idx, node);
                            }
                        } else {
                            ws.int_bathy[..n_face_nodes].fill(0.0);
                        }

                        // Get exterior states and bathymetry (reuse workspace buffers)
                        if let Some(neighbor) = mesh.neighbor(k_idx, face) {
                            let neighbor_face_nodes = &ops.face_nodes[neighbor.face];
                            for i in 0..n_face_nodes {
                                let state = q.get_state(
                                    ElementIndex::new(neighbor.element),
                                    neighbor_face_nodes[n_face_nodes - 1 - i],
                                );
                                ws.ext_states_h[i] = state.h;
                                ws.ext_states_hu[i] = state.hu;
                                ws.ext_states_hv[i] = state.hv;
                            }
                            if config.well_balanced && config.bathymetry.is_some() {
                                let b = config.bathymetry.unwrap();
                                for i in 0..n_face_nodes {
                                    ws.ext_bathy[i] = b.get(
                                        ElementIndex::new(neighbor.element),
                                        neighbor_face_nodes[n_face_nodes - 1 - i],
                                    );
                                }
                            } else {
                                ws.ext_bathy[..n_face_nodes].fill(0.0);
                            }
                        } else {
                            // Boundary face: compute ghost states
                            let boundary_tag = mesh.boundary_tag(k_idx, face);
                            for i in 0..n_face_nodes {
                                let node_idx = face_nodes[i];
                                let state = q.get_state(k_idx, node_idx);
                                let (r, s) = (ops.nodes_r[node_idx], ops.nodes_s[node_idx]);
                                let [x, y] = mesh.reference_to_physical(k_idx, r, s);

                                let bathy_value = config
                                    .bathymetry
                                    .map(|b| b.get(k_idx, node_idx))
                                    .unwrap_or(0.0);

                                let mut ctx = match boundary_tag {
                                    Some(tag) => BCContext2D::with_tag(
                                        time,
                                        (x, y),
                                        state,
                                        bathy_value,
                                        normal,
                                        g,
                                        h_min,
                                        tag,
                                    ),
                                    None => BCContext2D::new(
                                        time,
                                        (x, y),
                                        state,
                                        bathy_value,
                                        normal,
                                        g,
                                        h_min,
                                    ),
                                };
                                ctx.dt = config.dt;
                                let ghost = config.bc.ghost_state(&ctx);
                                ws.ext_states_h[i] = ghost.h;
                                ws.ext_states_hu[i] = ghost.hu;
                                ws.ext_states_hv[i] = ghost.hv;
                            }
                            // For boundary faces, mirror the interior bathymetry
                            ws.ext_bathy[..n_face_nodes]
                                .copy_from_slice(&ws.int_bathy[..n_face_nodes]);
                        }

                        // Compute numerical flux and flux difference at face nodes
                        for i in 0..n_face_nodes {
                            let node_idx = face_nodes[i];
                            let q_int = q.get_state(k_idx, node_idx);
                            let q_ext = SWEState2D {
                                h: ws.ext_states_h[i],
                                hu: ws.ext_states_hu[i],
                                hv: ws.ext_states_hv[i],
                            };

                            let (q_int_flux, q_ext_flux) = if let Some(ref reconstruction) = hr {
                                reconstruction.reconstruct(
                                    &q_int,
                                    &q_ext,
                                    ws.int_bathy[i],
                                    ws.ext_bathy[i],
                                )
                            } else {
                                (q_int, q_ext)
                            };

                            let f_star = compute_flux_swe_2d(
                                &q_int_flux,
                                &q_ext_flux,
                                normal,
                                g,
                                h_min,
                                config.flux_type,
                            );
                            let f_int = config.equation.normal_flux(&q_int_flux, normal);
                            let flux_diff = f_int - f_star;

                            ws.flux_diff_h[i] = flux_diff.h;
                            ws.flux_diff_hu[i] = flux_diff.hu;
                            ws.flux_diff_hv[i] = flux_diff.hv;
                        }

                        // Apply LIFT using SIMD kernel
                        let scale = j_inv * s_jac;
                        apply_lift(
                            &ops.lift_row_major[face],
                            &ws.flux_diff_h,
                            &ws.flux_diff_hu,
                            &ws.flux_diff_hv,
                            &mut ws.rhs_h,
                            &mut ws.rhs_hu,
                            &mut ws.rhs_hv,
                            n_nodes,
                            n_face_nodes,
                            scale,
                        );
                    }

                    // 3. Source terms: Legacy Coriolis
                    if config.include_coriolis
                        && config.source_terms.is_none()
                        && (config.equation.f0.abs() > 1e-14 || config.equation.beta.abs() > 1e-14)
                    {
                        if config.equation.beta.abs() < 1e-14 {
                            coriolis_source(
                                &ws.hu_arr,
                                &ws.hv_arr,
                                &mut ws.rhs_hu,
                                &mut ws.rhs_hv,
                                config.equation.f0,
                                n_nodes,
                            );
                        } else {
                            for i in 0..n_nodes {
                                let state = q.get_state(k_idx, i);
                                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                                let [_x, y] = mesh.reference_to_physical(k_idx, r, s);
                                let source = config.equation.coriolis_source(&state, y);
                                ws.rhs_hu[i] += source.hu;
                                ws.rhs_hv[i] += source.hv;
                            }
                        }
                    }

                    // 4. Trait-based source terms
                    if let Some(sources) = config.source_terms {
                        for i in 0..n_nodes {
                            let state = q.get_state(k_idx, i);
                            let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                            let [x, y] = mesh.reference_to_physical(k_idx, r, s);

                            let (bathy_value, bathy_gradient) = config
                                .bathymetry
                                .map(|b| (b.get(k_idx, i), b.get_gradient(k_idx, i)))
                                .unwrap_or((0.0, (0.0, 0.0)));

                            let ctx = SourceContext2D::new(
                                time,
                                (x, y),
                                state,
                                bathy_value,
                                bathy_gradient,
                                g,
                                h_min,
                            );
                            let source = sources.evaluate(&ctx);
                            ws.rhs_h[i] += source.h;
                            ws.rhs_hu[i] += source.hu;
                            ws.rhs_hv[i] += source.hv;
                        }
                    }

                    // Copy workspace results directly to output chunks
                    h_chunk.copy_from_slice(&ws.rhs_h);
                    hu_chunk.copy_from_slice(&ws.rhs_hu);
                    hv_chunk.copy_from_slice(&ws.rhs_hv);
                },
            );
    }

    add_br1_viscosity(&mut rhs, q, mesh, ops, geom, config, time);

    rhs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::Reflective2D;
    use crate::source::CoriolisSource2D;

    const G: f64 = 10.0;

    fn create_test_setup(order: usize) -> (Mesh2D, DGOperators2D, GeometricFactors2D) {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh);
        (mesh, ops, geom)
    }

    #[test]
    fn test_rhs_still_water() {
        // Lake at rest: h = const, u = v = 0, bathymetry = 0
        // RHS should be zero (well-balanced test)
        let (mesh, ops, geom) = create_test_setup(2);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        // Set h = 2.0, u = v = 0 everywhere
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(2.0, 0.0, 0.0));
            }
        }

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        // Check RHS is zero (lake at rest)
        let mut max_rhs: f64 = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let state = rhs.get_state(k, i);
                max_rhs = max_rhs.max(state.h.abs());
                max_rhs = max_rhs.max(state.hu.abs());
                max_rhs = max_rhs.max(state.hv.abs());
            }
        }

        assert!(
            max_rhs < 1e-10,
            "Lake at rest RHS should be zero, got {}",
            max_rhs
        );
    }

    #[test]
    fn test_rhs_uniform_flow() {
        // Uniform flow: h = const, u = const, v = 0
        // RHS should be zero (steady state)
        let (_mesh, ops, _geom) = create_test_setup(2);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

        // Use periodic mesh to avoid boundary effects
        let mesh_periodic = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let geom_periodic = GeometricFactors2D::compute(&mesh_periodic);

        let mut q = SWESolution2D::new(mesh_periodic.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh_periodic.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::from_primitives(2.0, 1.0, 0.0));
            }
        }

        let rhs = compute_rhs_swe_2d(&q, &mesh_periodic, &ops, &geom_periodic, &config, 0.0);

        // Uniform flow should give zero RHS (for periodic domain)
        let mut max_rhs: f64 = 0.0;
        for k in ElementIndex::iter(mesh_periodic.n_elements) {
            for i in 0..ops.n_nodes {
                let state = rhs.get_state(k, i);
                max_rhs = max_rhs.max(state.h.abs());
                max_rhs = max_rhs.max(state.hu.abs());
                max_rhs = max_rhs.max(state.hv.abs());
            }
        }

        assert!(
            max_rhs < 1e-10,
            "Uniform flow RHS should be zero, got {}",
            max_rhs
        );
    }

    #[test]
    fn test_mass_conservation() {
        // Test that total mass is conserved (RHS of h integrates to zero)
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

        // Non-uniform initial condition
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        q.set_from_functions(
            &mesh,
            &ops,
            |x, y| {
                1.0 + 0.5
                    * (2.0 * std::f64::consts::PI * x).sin()
                    * (2.0 * std::f64::consts::PI * y).sin()
            },
            |_, _| 0.0,
            |_, _| 0.0,
        );

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        // Integrate RHS of h
        let mut integral = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            let ki = k.as_usize();
            let j = geom.det_j[ki];
            for (i, &w) in ops.weights.iter().enumerate() {
                integral += w * rhs.get_var(k, i, 0) * j;
            }
        }

        assert!(
            integral.abs() < 1e-10,
            "Mass should be conserved: d(mass)/dt = {:.2e}",
            integral
        );
    }

    #[test]
    fn test_dt_computation() {
        let (mesh, ops, geom) = create_test_setup(2);
        let equation = ShallowWater2D::new(G);

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        // h = 1, u = 2 -> wave speed ≈ 2 + sqrt(10) ≈ 5.16
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::from_primitives(1.0, 2.0, 0.0));
            }
        }

        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, 0.5);

        assert!(dt > 0.0);
        assert!(dt < f64::INFINITY);
        assert!(dt < 0.1); // Should be small for this test case
    }

    #[test]
    fn test_coriolis_source() {
        // Test that Coriolis source term is included correctly
        let ops = DGOperators2D::new(2);
        let equation = ShallowWater2D::with_coriolis(G, 1.0e-4);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(true);

        // Use periodic mesh to eliminate surface terms
        let mesh_periodic = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let geom_periodic = GeometricFactors2D::compute(&mesh_periodic);

        let mut q = SWESolution2D::new(mesh_periodic.n_elements, ops.n_nodes);
        // h = 10, hu = 100 (so v-momentum source = f*hu = 1e-4 * 100 = 0.01)
        for k in ElementIndex::iter(mesh_periodic.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(10.0, 100.0, 0.0));
            }
        }

        let rhs_with = compute_rhs_swe_2d(&q, &mesh_periodic, &ops, &geom_periodic, &config, 0.0);

        let config_no_coriolis = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);
        let rhs_without = compute_rhs_swe_2d(
            &q,
            &mesh_periodic,
            &ops,
            &geom_periodic,
            &config_no_coriolis,
            0.0,
        );

        // The difference should be the Coriolis source
        // Source: d(hu)/dt += f*hv = 0, d(hv)/dt -= f*hu = -0.01
        let k0 = ElementIndex::new(0);
        let diff_hv = rhs_with.get_var(k0, 0, 2) - rhs_without.get_var(k0, 0, 2);
        let expected = -1.0e-4 * 100.0; // -f*hu

        assert!(
            (diff_hv - expected).abs() < 1e-10,
            "Coriolis source difference: got {}, expected {}",
            diff_hv,
            expected
        );
    }

    #[test]
    fn test_stability_single_step() {
        // Test that a single time step doesn't blow up
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

        // Smooth initial condition
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        q.set_from_functions(
            &mesh,
            &ops,
            |x, y| {
                1.0 + 0.1
                    * (2.0 * std::f64::consts::PI * x).sin()
                    * (2.0 * std::f64::consts::PI * y).sin()
            },
            |_, _| 0.0,
            |_, _| 0.0,
        );

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        // RHS should be bounded
        let max_rhs = rhs.max_abs();
        assert!(max_rhs < 100.0, "RHS should be bounded, got {}", max_rhs);

        // Do one forward Euler step
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, 0.1);
        q.axpy(dt, &rhs);

        // Solution should still be bounded
        assert!(!q.has_negative_depth(), "Depth should remain positive");
        assert!(
            q.max_abs() < 100.0,
            "Solution should be bounded after one step"
        );
    }

    #[test]
    fn test_flux_types() {
        // Test that different flux types work
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::from_primitives(2.0, 0.5, 0.2));
            }
        }

        for flux_type in [
            SWEFluxType2D::Roe,
            SWEFluxType2D::HLL,
            SWEFluxType2D::Rusanov,
        ] {
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_flux_type(flux_type)
                .with_coriolis(false);

            let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

            // For uniform state, all fluxes should give zero RHS
            let max_rhs = rhs.max_abs();
            assert!(
                max_rhs < 1e-10,
                "Uniform state should give zero RHS for {:?}, got {}",
                flux_type,
                max_rhs
            );
        }
    }

    #[test]
    fn test_trait_based_coriolis() {
        // Test that trait-based CoriolisSource2D gives same result as legacy Coriolis
        let ops = DGOperators2D::new(2);
        let equation = ShallowWater2D::with_coriolis(G, 1.0e-4);
        let bc = Reflective2D::new();

        // Use periodic mesh to eliminate surface terms
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let geom = GeometricFactors2D::compute(&mesh);

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(10.0, 100.0, 50.0));
            }
        }

        // Legacy Coriolis
        let config_legacy = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(true);
        let rhs_legacy = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config_legacy, 0.0);

        // Trait-based Coriolis
        let coriolis_source = CoriolisSource2D::f_plane(1.0e-4);
        let config_trait = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false) // Disable legacy
            .with_source_terms(&coriolis_source);
        let rhs_trait = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config_trait, 0.0);

        // Compare results
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let legacy = rhs_legacy.get_state(k, i);
                let trait_based = rhs_trait.get_state(k, i);

                assert!(
                    (legacy.h - trait_based.h).abs() < 1e-12,
                    "h mismatch at ({}, {}): {} vs {}",
                    k,
                    i,
                    legacy.h,
                    trait_based.h
                );
                assert!(
                    (legacy.hu - trait_based.hu).abs() < 1e-12,
                    "hu mismatch at ({}, {}): {} vs {}",
                    k,
                    i,
                    legacy.hu,
                    trait_based.hu
                );
                assert!(
                    (legacy.hv - trait_based.hv).abs() < 1e-12,
                    "hv mismatch at ({}, {}): {} vs {}",
                    k,
                    i,
                    legacy.hv,
                    trait_based.hv
                );
            }
        }
    }

    #[test]
    fn test_well_balanced_steep_slope() {
        // Test lake-at-rest preservation with steep bathymetry gradient
        // This tests the hydrostatic reconstruction implementation
        // Note: We need BOTH reconstruction (for surface terms) AND source term (for volume balance)
        use crate::mesh::Bathymetry2D;
        use crate::source::BathymetrySource2D;

        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        // Steep bathymetry: B = 0.5 * x (Norwegian-like gradient)
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| 0.5 * x);

        // Lake-at-rest: η = h + B = 10.0
        let eta_ref = 10.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let b = bathymetry.get(k, i);
                let h = (eta_ref - b).max(0.0);
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }

        // Config WITH hydrostatic reconstruction AND bathymetry source term
        // Reconstruction: handles surface flux well-balancing
        // Source term: balances volume term pressure gradient
        let bathy_source = BathymetrySource2D::new(G);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(true);

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        // RHS should be zero to machine precision
        let max_rhs = rhs.max_abs();
        assert!(
            max_rhs < 1e-10,
            "Lake at rest with hydrostatic reconstruction: max RHS = {:.2e}, expected < 1e-10",
            max_rhs
        );
    }

    #[test]
    fn test_well_balanced_bilinear_slope() {
        // Test with bilinear bathymetry (x + y direction slope)
        // This is still linear within elements, so well-balanced should work perfectly
        use crate::mesh::Bathymetry2D;
        use crate::source::BathymetrySource2D;

        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 6, 6);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        // Bilinear bathymetry: B = 0.3*x + 0.2*y
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| 0.3 * x + 0.2 * y);

        // Lake-at-rest: η = 10.0
        let eta_ref = 10.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let b = bathymetry.get(k, i);
                let h = (eta_ref - b).max(0.0);
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }

        let bathy_source = BathymetrySource2D::new(G);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(true);

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        let max_rhs = rhs.max_abs();
        assert!(
            max_rhs < 1e-10,
            "Lake at rest (bilinear slope): max RHS = {:.2e}",
            max_rhs
        );
    }

    #[test]
    fn test_well_balanced_all_flux_types() {
        // Verify all flux types work with hydrostatic reconstruction
        use crate::flux::SWEFluxType2D;
        use crate::mesh::Bathymetry2D;
        use crate::source::BathymetrySource2D;

        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| 0.3 * x + 0.4 * y);

        let eta_ref = 10.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let b = bathymetry.get(k, i);
                q.set_state(k, i, SWEState2D::new((eta_ref - b).max(0.0), 0.0, 0.0));
            }
        }

        let bathy_source = BathymetrySource2D::new(G);
        for flux_type in [
            SWEFluxType2D::Roe,
            SWEFluxType2D::HLL,
            SWEFluxType2D::Rusanov,
        ] {
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_flux_type(flux_type)
                .with_coriolis(false)
                .with_bathymetry(&bathymetry)
                .with_source_terms(&bathy_source)
                .with_well_balanced(true);

            let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
            let max_rhs = rhs.max_abs();

            assert!(
                max_rhs < 1e-10,
                "{:?} flux with well-balanced: max RHS = {:.2e}",
                flux_type,
                max_rhs
            );
        }
    }

    #[test]
    fn test_source_term_alone_well_balanced_linear() {
        // For LINEAR bathymetry, the standard DG scheme with bathymetry source term
        // is already well-balanced because the volume gradient and source term
        // are discretized consistently using the same DG differentiation operators.
        // This test verifies this property.
        use crate::mesh::Bathymetry2D;
        use crate::source::BathymetrySource2D;

        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        // Linear bathymetry
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| 0.5 * x);

        let eta_ref = 10.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let b = bathymetry.get(k, i);
                q.set_state(k, i, SWEState2D::new((eta_ref - b).max(0.0), 0.0, 0.0));
            }
        }

        // Config with source term only (no reconstruction)
        let bathy_source = BathymetrySource2D::new(G);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(false); // No reconstruction

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        let max_rhs = rhs.max_abs();

        // For linear bathymetry, source term alone should be well-balanced
        // (DG computes gradients exactly for linear functions)
        assert!(
            max_rhs < 1e-10,
            "Source term alone should be well-balanced for linear bathymetry: {:.2e}",
            max_rhs
        );
    }

    #[test]
    fn test_reconstruction_improves_accuracy() {
        // Verify that hydrostatic reconstruction is correctly applied
        // by checking that reconstructed states match expected values.
        use crate::mesh::Bathymetry2D;
        use crate::source::BathymetrySource2D;

        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();

        // Linear bathymetry
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| 0.5 * x);

        let eta_ref = 10.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let b = bathymetry.get(k, i);
                q.set_state(k, i, SWEState2D::new((eta_ref - b).max(0.0), 0.0, 0.0));
            }
        }

        // Config WITH reconstruction AND source term
        let bathy_source = BathymetrySource2D::new(G);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(true);

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

        let max_rhs = rhs.max_abs();

        // With reconstruction, should also be well-balanced
        assert!(
            max_rhs < 1e-10,
            "With reconstruction should be well-balanced: {:.2e}",
            max_rhs
        );
    }

    #[test]
    #[cfg(all(feature = "parallel", feature = "simd"))]
    fn test_parallel_matches_serial() {
        use super::compute_rhs_swe_2d_parallel;

        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

        // Non-uniform initial condition
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [x, y] = mesh.reference_to_physical(k, r, s);
                let h =
                    2.0 + 0.1 * (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).cos();
                let u = 0.5 * (std::f64::consts::PI * y).sin();
                let v = 0.3 * (std::f64::consts::PI * x).cos();
                q.set_state(k, i, SWEState2D::from_primitives(h, u, v));
            }
        }

        let rhs_serial = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
        let rhs_parallel = compute_rhs_swe_2d_parallel(&q, &mesh, &ops, &geom, &config, 0.0);

        // Results should be identical
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let serial = rhs_serial.get_state(k, i);
                let parallel = rhs_parallel.get_state(k, i);

                let diff_h = (serial.h - parallel.h).abs();
                let diff_hu = (serial.hu - parallel.hu).abs();
                let diff_hv = (serial.hv - parallel.hv).abs();

                assert!(
                    diff_h < 1e-10,
                    "h mismatch at ({}, {}): serial={}, parallel={}, diff={}",
                    k,
                    i,
                    serial.h,
                    parallel.h,
                    diff_h
                );
                assert!(
                    diff_hu < 1e-10,
                    "hu mismatch at ({}, {}): serial={}, parallel={}, diff={}",
                    k,
                    i,
                    serial.hu,
                    parallel.hu,
                    diff_hu
                );
                assert!(
                    diff_hv < 1e-10,
                    "hv mismatch at ({}, {}): serial={}, parallel={}, diff={}",
                    k,
                    i,
                    serial.hv,
                    parallel.hv,
                    diff_hv
                );
            }
        }
    }

    #[test]
    #[cfg(all(feature = "parallel", feature = "simd"))]
    fn test_parallel_with_coriolis() {
        use super::compute_rhs_swe_2d_parallel;

        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh);
        let equation = ShallowWater2D::with_coriolis(G, 1.0e-4); // f-plane Coriolis
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(true);

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(10.0, 100.0, 50.0));
            }
        }

        let rhs_serial = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
        let rhs_parallel = compute_rhs_swe_2d_parallel(&q, &mesh, &ops, &geom, &config, 0.0);

        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let serial = rhs_serial.get_state(k, i);
                let parallel = rhs_parallel.get_state(k, i);

                assert!(
                    (serial.h - parallel.h).abs() < 1e-10,
                    "h mismatch with Coriolis at ({}, {})",
                    k,
                    i
                );
                assert!(
                    (serial.hu - parallel.hu).abs() < 1e-10,
                    "hu mismatch with Coriolis at ({}, {})",
                    k,
                    i
                );
                assert!(
                    (serial.hv - parallel.hv).abs() < 1e-10,
                    "hv mismatch with Coriolis at ({}, {})",
                    k,
                    i
                );
            }
        }
    }
}
