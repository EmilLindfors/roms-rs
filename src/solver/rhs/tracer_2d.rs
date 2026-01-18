//! Right-hand side computation for 2D tracer transport.
//!
//! For tracers (temperature T, salinity S) advected by the SWE velocity field:
//!   ∂(hT)/∂t + ∇·(hT**u**) = κ_T ∇²T + S_T
//!   ∂(hS)/∂t + ∇·(hS**u**) = κ_S ∇²S + S_S
//!
//! The DG semi-discrete form:
//!   d(hC)/dt = L(hC) = -(volume flux) + (surface flux) + (diffusion) + (sources)

use crate::flux::{TracerFluxType, tracer_numerical_flux};
use crate::mesh::{BoundaryTag, Mesh2D};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::{
    ConservativeTracerState, SWESolution2D, SWEState2D, TracerSolution2D, TracerState,
};
use crate::types::{Depth, ElementIndex};

/// Context for tracer boundary condition evaluation.
#[derive(Clone, Copy, Debug)]
pub struct TracerBCContext2D {
    /// Current simulation time
    pub time: f64,
    /// Physical position (x, y)
    pub position: (f64, f64),
    /// Interior tracer state (conservative: hT, hS)
    pub interior_tracer: ConservativeTracerState,
    /// Interior SWE state (h, hu, hv)
    pub interior_swe: SWEState2D,
    /// Outward unit normal (nx, ny)
    pub normal: (f64, f64),
    /// Minimum depth threshold
    pub h_min: f64,
    /// Gravitational acceleration
    pub g: f64,
    /// Optional boundary tag
    pub boundary_tag: Option<BoundaryTag>,
}

impl TracerBCContext2D {
    /// Create a new tracer BC context.
    pub fn new(
        time: f64,
        position: (f64, f64),
        interior_tracer: ConservativeTracerState,
        interior_swe: SWEState2D,
        normal: (f64, f64),
        h_min: f64,
        g: f64,
    ) -> Self {
        Self {
            time,
            position,
            interior_tracer,
            interior_swe,
            normal,
            h_min,
            g,
            boundary_tag: None,
        }
    }

    /// Create a context with boundary tag.
    pub fn with_tag(
        time: f64,
        position: (f64, f64),
        interior_tracer: ConservativeTracerState,
        interior_swe: SWEState2D,
        normal: (f64, f64),
        h_min: f64,
        g: f64,
        tag: BoundaryTag,
    ) -> Self {
        Self {
            time,
            position,
            interior_tracer,
            interior_swe,
            normal,
            h_min,
            g,
            boundary_tag: Some(tag),
        }
    }

    /// Get interior tracer concentrations.
    pub fn interior_concentrations(&self) -> TracerState {
        self.interior_tracer
            .to_concentrations(self.interior_swe.h, self.h_min)
    }

    /// Get interior normal velocity.
    pub fn interior_normal_velocity(&self) -> f64 {
        let (u, v) = self.interior_swe.velocity_simple(Depth::new(self.h_min));
        let (nx, ny) = self.normal;
        u * nx + v * ny
    }

    /// Check if flow is inward (into domain).
    pub fn is_inflow(&self) -> bool {
        self.interior_normal_velocity() < 0.0
    }
}

/// Trait for tracer boundary conditions.
///
/// For advection-dominated transport, tracers typically use:
/// - Inflow: prescribed tracer values
/// - Outflow: zero-gradient (extrapolation)
pub trait TracerBoundaryCondition2D: Send + Sync {
    /// Compute the ghost state for flux evaluation.
    fn ghost_state(&self, ctx: &TracerBCContext2D) -> (ConservativeTracerState, SWEState2D);

    /// Name of this boundary condition.
    fn name(&self) -> &'static str;
}

/// Extrapolation (zero-gradient) boundary condition for tracers.
///
/// Copies interior state to exterior. Good for outflow boundaries.
#[derive(Clone, Debug, Default)]
pub struct ExtrapolationTracerBC;

impl TracerBoundaryCondition2D for ExtrapolationTracerBC {
    fn ghost_state(&self, ctx: &TracerBCContext2D) -> (ConservativeTracerState, SWEState2D) {
        (ctx.interior_tracer, ctx.interior_swe)
    }

    fn name(&self) -> &'static str {
        "extrapolation"
    }
}

/// Fixed tracer values at boundary (Dirichlet).
///
/// Used for inflow boundaries where tracer values are known.
#[derive(Clone, Debug)]
pub struct FixedTracerBC {
    /// Fixed temperature (°C)
    pub temperature: f64,
    /// Fixed salinity (PSU)
    pub salinity: f64,
}

impl FixedTracerBC {
    /// Create a new fixed tracer BC.
    pub fn new(temperature: f64, salinity: f64) -> Self {
        Self {
            temperature,
            salinity,
        }
    }

    /// Atlantic water inflow.
    pub fn atlantic() -> Self {
        Self {
            temperature: 7.5,
            salinity: 35.0,
        }
    }

    /// Norwegian coastal water.
    pub fn coastal() -> Self {
        Self {
            temperature: 8.0,
            salinity: 34.0,
        }
    }

    /// Freshwater river inflow.
    pub fn freshwater(temperature: f64) -> Self {
        Self {
            temperature,
            salinity: 0.0,
        }
    }
}

impl TracerBoundaryCondition2D for FixedTracerBC {
    fn ghost_state(&self, ctx: &TracerBCContext2D) -> (ConservativeTracerState, SWEState2D) {
        let h = ctx.interior_swe.h;
        let tracers = TracerState::new(self.temperature, self.salinity);
        let cons = ConservativeTracerState::from_depth_and_tracers(h, tracers);
        (cons, ctx.interior_swe)
    }

    fn name(&self) -> &'static str {
        "fixed_tracer"
    }
}

/// Upwind-based boundary condition (recommended for most cases).
///
/// Uses fixed values for inflow, extrapolation for outflow.
#[derive(Clone, Debug)]
pub struct UpwindTracerBC {
    /// Temperature for inflow
    pub inflow_temperature: f64,
    /// Salinity for inflow
    pub inflow_salinity: f64,
}

impl UpwindTracerBC {
    /// Create a new upwind tracer BC.
    pub fn new(inflow_temperature: f64, inflow_salinity: f64) -> Self {
        Self {
            inflow_temperature,
            inflow_salinity,
        }
    }

    /// Atlantic water for open boundaries.
    pub fn atlantic() -> Self {
        Self::new(7.5, 35.0)
    }
}

impl TracerBoundaryCondition2D for UpwindTracerBC {
    fn ghost_state(&self, ctx: &TracerBCContext2D) -> (ConservativeTracerState, SWEState2D) {
        if ctx.is_inflow() {
            // Inflow: use prescribed values
            let h = ctx.interior_swe.h;
            let tracers = TracerState::new(self.inflow_temperature, self.inflow_salinity);
            let cons = ConservativeTracerState::from_depth_and_tracers(h, tracers);
            (cons, ctx.interior_swe)
        } else {
            // Outflow: extrapolate
            (ctx.interior_tracer, ctx.interior_swe)
        }
    }

    fn name(&self) -> &'static str {
        "upwind_tracer"
    }
}

/// Tracer source term trait.
///
/// Source terms can include:
/// - Surface heat flux (heating/cooling)
/// - River discharge (freshwater input)
/// - Mixing with other water masses
pub trait TracerSourceTerm2D: Send + Sync {
    /// Evaluate the source term at a point.
    ///
    /// Returns (S_hT, S_hS) - sources for h*T and h*S equations.
    fn evaluate(
        &self,
        time: f64,
        position: (f64, f64),
        swe: &SWEState2D,
        tracers: &ConservativeTracerState,
    ) -> ConservativeTracerState;

    /// Name of this source term.
    fn name(&self) -> &'static str;
}

/// Configuration for tracer RHS computation.
pub struct Tracer2DRhsConfig<'a, BC: TracerBoundaryCondition2D> {
    /// Numerical flux type
    pub flux_type: TracerFluxType,
    /// Boundary condition handler
    pub bc: &'a BC,
    /// Gravitational acceleration
    pub g: f64,
    /// Minimum depth threshold
    pub h_min: f64,
    /// Horizontal diffusivity for temperature (m²/s)
    pub kappa_t: f64,
    /// Horizontal diffusivity for salinity (m²/s)
    pub kappa_s: f64,
    /// Optional source terms
    pub source_terms: Option<&'a dyn TracerSourceTerm2D>,
}

impl<'a, BC: TracerBoundaryCondition2D> Tracer2DRhsConfig<'a, BC> {
    /// Create a new tracer RHS configuration.
    pub fn new(bc: &'a BC, g: f64, h_min: f64) -> Self {
        Self {
            flux_type: TracerFluxType::Upwind,
            bc,
            g,
            h_min,
            kappa_t: 0.0, // No diffusion by default
            kappa_s: 0.0,
            source_terms: None,
        }
    }

    /// Set numerical flux type.
    pub fn with_flux_type(mut self, flux_type: TracerFluxType) -> Self {
        self.flux_type = flux_type;
        self
    }

    /// Set horizontal diffusivity for both tracers.
    ///
    /// Typical values for coastal oceans: 1-100 m²/s.
    pub fn with_diffusivity(mut self, kappa: f64) -> Self {
        self.kappa_t = kappa;
        self.kappa_s = kappa;
        self
    }

    /// Set different diffusivities for T and S.
    pub fn with_diffusivities(mut self, kappa_t: f64, kappa_s: f64) -> Self {
        self.kappa_t = kappa_t;
        self.kappa_s = kappa_s;
        self
    }

    /// Set tracer source terms.
    pub fn with_source_terms(mut self, sources: &'a dyn TracerSourceTerm2D) -> Self {
        self.source_terms = Some(sources);
        self
    }
}

/// Compute the right-hand side for 2D tracer transport.
///
/// Implements the DG weak form for the tracer equations:
///   d(hC)/dt = -∇·(hC**u**) + κ∇²C + S
///
/// # Arguments
/// * `tracers` - Current tracer solution (hT, hS)
/// * `swe` - Current SWE solution (provides velocity field)
/// * `mesh` - Computational mesh
/// * `ops` - DG operators
/// * `geom` - Geometric factors
/// * `config` - RHS configuration
/// * `time` - Current time
pub fn compute_rhs_tracer_2d<BC: TracerBoundaryCondition2D>(
    tracers: &TracerSolution2D,
    swe: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &Tracer2DRhsConfig<BC>,
    time: f64,
) -> TracerSolution2D {
    let n_nodes = ops.n_nodes;
    let n_face_nodes = ops.n_face_nodes;
    let mut rhs = TracerSolution2D::new(mesh.n_elements, n_nodes);

    let h_min = Depth::new(config.h_min);
    let g = config.g;

    for k in ElementIndex::iter(mesh.n_elements) {
        let j_inv = geom.det_j_inv[k.as_usize()];
        let rx = geom.rx[k.as_usize()];
        let ry = geom.ry[k.as_usize()];
        let sx = geom.sx[k.as_usize()];
        let sy = geom.sy[k.as_usize()];

        // 1. Volume term: -∇·(hC **u**)
        // Flux F = hC * u, G = hC * v
        let mut flux_x = vec![ConservativeTracerState::zero(); n_nodes];
        let mut flux_y = vec![ConservativeTracerState::zero(); n_nodes];

        for i in 0..n_nodes {
            let tracer = tracers.get_conservative(k, i);
            let swe_state = swe.get_state(k, i);
            let (u, v) = swe_state.velocity_simple(h_min);

            flux_x[i] = ConservativeTracerState {
                h_t: tracer.h_t * u,
                h_s: tracer.h_s * u,
            };
            flux_y[i] = ConservativeTracerState {
                h_t: tracer.h_t * v,
                h_s: tracer.h_s * v,
            };
        }

        // Apply Dr and Ds to compute derivatives
        for i in 0..n_nodes {
            let mut dfx_dr = ConservativeTracerState::zero();
            let mut dfx_ds = ConservativeTracerState::zero();
            let mut dfy_dr = ConservativeTracerState::zero();
            let mut dfy_ds = ConservativeTracerState::zero();

            for j in 0..n_nodes {
                let dr_ij = ops.dr[(i, j)];
                let ds_ij = ops.ds[(i, j)];

                dfx_dr = dfx_dr + dr_ij * flux_x[j];
                dfx_ds = dfx_ds + ds_ij * flux_x[j];
                dfy_dr = dfy_dr + dr_ij * flux_y[j];
                dfy_ds = dfy_ds + ds_ij * flux_y[j];
            }

            // Volume term: -(dF/dx + dG/dy)
            let div_flux = dfx_dr * rx + dfx_ds * sx + dfy_dr * ry + dfy_ds * sy;
            rhs.set_conservative(k, i, -1.0 * div_flux);
        }

        // 2. Surface terms: LIFT_f * sJ_f * (F- - F*)
        for face in 0..4 {
            let normal = geom.normals[k.as_usize()][face];
            let s_jac = geom.surface_j[k.as_usize()][face];
            let face_nodes = &ops.face_nodes[face];

            // Get exterior states
            let ext_states: Vec<(ConservativeTracerState, SWEState2D)> =
                if let Some(neighbor) = mesh.neighbor(k, face) {
                    // Interior face: get from neighbor
                    let neighbor_face_nodes = &ops.face_nodes[neighbor.face];
                    let neighbor_k = ElementIndex::new(neighbor.element);
                    (0..n_face_nodes)
                        .map(|i| {
                            let ni = neighbor_face_nodes[n_face_nodes - 1 - i];
                            (
                                tracers.get_conservative(neighbor_k, ni),
                                swe.get_state(neighbor_k, ni),
                            )
                        })
                        .collect()
                } else {
                    // Boundary face: compute ghost states
                    let boundary_tag = mesh.boundary_tag(k, face);

                    (0..n_face_nodes)
                        .map(|i| {
                            let node_idx = face_nodes[i];
                            let tracer = tracers.get_conservative(k, node_idx);
                            let swe_state = swe.get_state(k, node_idx);
                            let (r, s) = (ops.nodes_r[node_idx], ops.nodes_s[node_idx]);
                            let [x, y] = mesh.reference_to_physical(k, r, s);

                            let ctx = match boundary_tag {
                                Some(tag) => TracerBCContext2D::with_tag(
                                    time,
                                    (x, y),
                                    tracer,
                                    swe_state,
                                    normal,
                                    h_min.meters(),
                                    g,
                                    tag,
                                ),
                                None => TracerBCContext2D::new(
                                    time,
                                    (x, y),
                                    tracer,
                                    swe_state,
                                    normal,
                                    h_min.meters(),
                                    g,
                                ),
                            };
                            config.bc.ghost_state(&ctx)
                        })
                        .collect()
                };

            // Compute flux difference at face nodes
            let mut flux_diff = vec![ConservativeTracerState::zero(); n_face_nodes];

            for i in 0..n_face_nodes {
                let node_idx = face_nodes[i];
                let tracer_int = tracers.get_conservative(k, node_idx);
                let swe_int = swe.get_state(k, node_idx);
                let (tracer_ext, swe_ext) = ext_states[i];

                // Numerical flux F* · n
                let f_star = tracer_numerical_flux(
                    config.flux_type,
                    &tracer_int,
                    &tracer_ext,
                    &swe_int,
                    &swe_ext,
                    normal,
                    h_min.meters(),
                    g,
                );

                // Interior flux F- · n
                let (u, v) = swe_int.velocity_simple(h_min);
                let un = u * normal.0 + v * normal.1;
                let f_int = ConservativeTracerState {
                    h_t: tracer_int.h_t * un,
                    h_s: tracer_int.h_s * un,
                };

                // Flux difference for surface term
                flux_diff[i] = f_int - f_star;
            }

            // Apply LIFT
            for i in 0..n_nodes {
                let mut lift_contrib = ConservativeTracerState::zero();
                for fi in 0..n_face_nodes {
                    let lift_coeff = ops.lift[face][(i, fi)];
                    lift_contrib = lift_contrib + lift_coeff * s_jac * flux_diff[fi];
                }

                let current = rhs.get_conservative(k, i);
                rhs.set_conservative(k, i, current + j_inv * lift_contrib);
            }
        }

        // 3. Diffusion terms: κ∇²C (if enabled)
        // Use BR1 (Bassi-Rebay 1) scheme for simplicity:
        // First compute ∇C, then apply κ∇·(∇C)
        if config.kappa_t > 0.0 || config.kappa_s > 0.0 {
            // Compute tracer gradients
            let mut d_t_dx = vec![0.0; n_nodes];
            let mut d_t_dy = vec![0.0; n_nodes];
            let mut d_s_dx = vec![0.0; n_nodes];
            let mut d_s_dy = vec![0.0; n_nodes];

            // Get concentrations at all nodes
            let concentrations: Vec<TracerState> = (0..n_nodes)
                .map(|i| {
                    let h = swe.get_state(k, i).h;
                    tracers.get_concentrations(k, i, h, h_min.meters())
                })
                .collect();

            // Compute gradients using differentiation matrices
            for i in 0..n_nodes {
                let mut dt_dr = 0.0;
                let mut dt_ds = 0.0;
                let mut ds_dr = 0.0;
                let mut ds_ds = 0.0;

                for j in 0..n_nodes {
                    let dr_ij = ops.dr[(i, j)];
                    let ds_ij = ops.ds[(i, j)];

                    dt_dr += dr_ij * concentrations[j].temperature;
                    dt_ds += ds_ij * concentrations[j].temperature;
                    ds_dr += dr_ij * concentrations[j].salinity;
                    ds_ds += ds_ij * concentrations[j].salinity;
                }

                // Transform to physical coordinates
                d_t_dx[i] = dt_dr * rx + dt_ds * sx;
                d_t_dy[i] = dt_dr * ry + dt_ds * sy;
                d_s_dx[i] = ds_dr * rx + ds_ds * sx;
                d_s_dy[i] = ds_dr * ry + ds_ds * sy;
            }

            // Add diffusion term: κ h ∇²C (keep in conservative form)
            // ∇·(h κ ∇C) = h κ ∇²C + κ ∇h · ∇C
            // For simplicity, use: h κ ∇²C (ignoring depth gradient term)
            //
            // The Laplacian ∇²T = ∂²T/∂x² + ∂²T/∂y² is computed by
            // differentiating the physical gradients:
            //   ∂²T/∂x² = ∂/∂x(∂T/∂x) = (rx·∂/∂r + sx·∂/∂s)(∂T/∂x)
            //   ∂²T/∂y² = ∂/∂y(∂T/∂y) = (ry·∂/∂r + sy·∂/∂s)(∂T/∂y)
            for i in 0..n_nodes {
                let h = swe.get_state(k, i).h.max(h_min.meters());

                // Compute derivatives of physical gradients in reference space
                // ∂/∂r(∂T/∂x), ∂/∂s(∂T/∂x), ∂/∂r(∂T/∂y), ∂/∂s(∂T/∂y)
                let mut d_dtdx_dr = 0.0;
                let mut d_dtdx_ds = 0.0;
                let mut d_dtdy_dr = 0.0;
                let mut d_dtdy_ds = 0.0;
                let mut d_dsdx_dr = 0.0;
                let mut d_dsdx_ds = 0.0;
                let mut d_dsdy_dr = 0.0;
                let mut d_dsdy_ds = 0.0;

                for j in 0..n_nodes {
                    let dr_ij = ops.dr[(i, j)];
                    let ds_ij = ops.ds[(i, j)];

                    // Differentiate physical gradients
                    d_dtdx_dr += dr_ij * d_t_dx[j];
                    d_dtdx_ds += ds_ij * d_t_dx[j];
                    d_dtdy_dr += dr_ij * d_t_dy[j];
                    d_dtdy_ds += ds_ij * d_t_dy[j];

                    d_dsdx_dr += dr_ij * d_s_dx[j];
                    d_dsdx_ds += ds_ij * d_s_dx[j];
                    d_dsdy_dr += dr_ij * d_s_dy[j];
                    d_dsdy_ds += ds_ij * d_s_dy[j];
                }

                // Transform to physical second derivatives using chain rule:
                // ∂²T/∂x² = rx·∂/∂r(∂T/∂x) + sx·∂/∂s(∂T/∂x)
                // ∂²T/∂y² = ry·∂/∂r(∂T/∂y) + sy·∂/∂s(∂T/∂y)
                let d2t_dx2 = rx * d_dtdx_dr + sx * d_dtdx_ds;
                let d2t_dy2 = ry * d_dtdy_dr + sy * d_dtdy_ds;
                let laplacian_t = d2t_dx2 + d2t_dy2;

                let d2s_dx2 = rx * d_dsdx_dr + sx * d_dsdx_ds;
                let d2s_dy2 = ry * d_dsdy_dr + sy * d_dsdy_ds;
                let laplacian_s = d2s_dx2 + d2s_dy2;

                // Add diffusion source
                let current = rhs.get_conservative(k, i);
                let diffusion = ConservativeTracerState {
                    h_t: config.kappa_t * h * laplacian_t,
                    h_s: config.kappa_s * h * laplacian_s,
                };
                rhs.set_conservative(k, i, current + diffusion);
            }
        }

        // 4. Source terms
        if let Some(sources) = config.source_terms {
            for i in 0..n_nodes {
                let tracer = tracers.get_conservative(k, i);
                let swe_state = swe.get_state(k, i);
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [x, y] = mesh.reference_to_physical(k, r, s);

                let source = sources.evaluate(time, (x, y), &swe_state, &tracer);
                let current = rhs.get_conservative(k, i);
                rhs.set_conservative(k, i, current + source);
            }
        }
    }

    rhs
}

/// Compute the right-hand side for 2D tracer transport (parallel version).
///
/// Uses Rayon for parallel element processing. Each element computes
/// independently, with thread-safe reads from neighboring elements.
///
/// # Arguments
/// * `tracers` - Current tracer solution (hT, hS)
/// * `swe` - Current SWE solution (provides velocity field)
/// * `mesh` - Computational mesh
/// * `ops` - DG operators
/// * `geom` - Geometric factors
/// * `config` - RHS configuration
/// * `time` - Current time
#[cfg(feature = "parallel")]
pub fn compute_rhs_tracer_2d_parallel<BC: TracerBoundaryCondition2D>(
    tracers: &TracerSolution2D,
    swe: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &Tracer2DRhsConfig<BC>,
    time: f64,
) -> TracerSolution2D {
    use rayon::prelude::*;

    let n_nodes = ops.n_nodes;
    let n_face_nodes = ops.n_face_nodes;
    let h_min = Depth::new(config.h_min);
    let g = config.g;

    // Pre-allocate output
    let mut rhs = TracerSolution2D::new(mesh.n_elements, n_nodes);

    // Process elements in parallel using par_chunks_mut
    let chunk_size = n_nodes * 2; // 2 vars per node (hT, hS)
    rhs.data
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(k, rhs_chunk)| {
            let k_idx = ElementIndex::new(k);
            let j_inv = geom.det_j_inv[k];
            let rx = geom.rx[k];
            let ry = geom.ry[k];
            let sx = geom.sx[k];
            let sy = geom.sy[k];

            // Element-local RHS storage
            let mut rhs_k = vec![ConservativeTracerState::zero(); n_nodes];

            // 1. Volume term: -∇·(hC **u**)
            let mut flux_x = vec![ConservativeTracerState::zero(); n_nodes];
            let mut flux_y = vec![ConservativeTracerState::zero(); n_nodes];

            for i in 0..n_nodes {
                let tracer = tracers.get_conservative(k_idx, i);
                let swe_state = swe.get_state(k_idx, i);
                let (u, v) = swe_state.velocity_simple(h_min);

                flux_x[i] = ConservativeTracerState {
                    h_t: tracer.h_t * u,
                    h_s: tracer.h_s * u,
                };
                flux_y[i] = ConservativeTracerState {
                    h_t: tracer.h_t * v,
                    h_s: tracer.h_s * v,
                };
            }

            // Apply Dr and Ds to compute derivatives
            for i in 0..n_nodes {
                let mut dfx_dr = ConservativeTracerState::zero();
                let mut dfx_ds = ConservativeTracerState::zero();
                let mut dfy_dr = ConservativeTracerState::zero();
                let mut dfy_ds = ConservativeTracerState::zero();

                for j in 0..n_nodes {
                    let dr_ij = ops.dr[(i, j)];
                    let ds_ij = ops.ds[(i, j)];

                    dfx_dr = dfx_dr + dr_ij * flux_x[j];
                    dfx_ds = dfx_ds + ds_ij * flux_x[j];
                    dfy_dr = dfy_dr + dr_ij * flux_y[j];
                    dfy_ds = dfy_ds + ds_ij * flux_y[j];
                }

                // Volume term: -(dF/dx + dG/dy)
                let div_flux = dfx_dr * rx + dfx_ds * sx + dfy_dr * ry + dfy_ds * sy;
                rhs_k[i] = -1.0 * div_flux;
            }

            // 2. Surface terms: LIFT_f * sJ_f * (F- - F*)
            for face in 0..4 {
                let normal = geom.normals[k][face];
                let s_jac = geom.surface_j[k][face];
                let face_nodes = &ops.face_nodes[face];

                // Get exterior states
                let ext_states: Vec<(ConservativeTracerState, SWEState2D)> =
                    if let Some(neighbor) = mesh.neighbor(k_idx, face) {
                        let neighbor_face_nodes = &ops.face_nodes[neighbor.face];
                        (0..n_face_nodes)
                            .map(|i| {
                                let ni = neighbor_face_nodes[n_face_nodes - 1 - i];
                                (
                                    tracers.get_conservative(ElementIndex::new(neighbor.element), ni),
                                    swe.get_state(ElementIndex::new(neighbor.element), ni),
                                )
                            })
                            .collect()
                    } else {
                        let boundary_tag = mesh.boundary_tag(k_idx, face);
                        (0..n_face_nodes)
                            .map(|i| {
                                let node_idx = face_nodes[i];
                                let tracer = tracers.get_conservative(k_idx, node_idx);
                                let swe_state = swe.get_state(k_idx, node_idx);
                                let (r, s) = (ops.nodes_r[node_idx], ops.nodes_s[node_idx]);
                                let [x, y] = mesh.reference_to_physical(k_idx, r, s);

                                let ctx = match boundary_tag {
                                    Some(tag) => TracerBCContext2D::with_tag(
                                        time,
                                        (x, y),
                                        tracer,
                                        swe_state,
                                        normal,
                                        h_min.meters(),
                                        g,
                                        tag,
                                    ),
                                    None => TracerBCContext2D::new(
                                        time,
                                        (x, y),
                                        tracer,
                                        swe_state,
                                        normal,
                                        h_min.meters(),
                                        g,
                                    ),
                                };
                                config.bc.ghost_state(&ctx)
                            })
                            .collect()
                    };

                // Compute flux difference at face nodes
                let mut flux_diff = vec![ConservativeTracerState::zero(); n_face_nodes];
                for i in 0..n_face_nodes {
                    let node_idx = face_nodes[i];
                    let tracer_int = tracers.get_conservative(k_idx, node_idx);
                    let swe_int = swe.get_state(k_idx, node_idx);
                    let (tracer_ext, swe_ext) = ext_states[i];

                    // Numerical flux F* · n
                    let f_star = tracer_numerical_flux(
                        config.flux_type,
                        &tracer_int,
                        &tracer_ext,
                        &swe_int,
                        &swe_ext,
                        normal,
                        h_min.meters(),
                        g,
                    );

                    // Interior flux F- · n
                    let (u, v) = swe_int.velocity_simple(h_min);
                    let un = u * normal.0 + v * normal.1;
                    let f_int = ConservativeTracerState {
                        h_t: tracer_int.h_t * un,
                        h_s: tracer_int.h_s * un,
                    };

                    flux_diff[i] = f_int - f_star;
                }

                // Apply LIFT
                for i in 0..n_nodes {
                    let mut lift_contrib = ConservativeTracerState::zero();
                    for fi in 0..n_face_nodes {
                        let lift_coeff = ops.lift[face][(i, fi)];
                        lift_contrib = lift_contrib + lift_coeff * s_jac * flux_diff[fi];
                    }
                    rhs_k[i] = rhs_k[i] + j_inv * lift_contrib;
                }
            }

            // 3. Diffusion terms: κ∇²C (if enabled)
            if config.kappa_t > 0.0 || config.kappa_s > 0.0 {
                let mut d_t_dx = vec![0.0; n_nodes];
                let mut d_t_dy = vec![0.0; n_nodes];
                let mut d_s_dx = vec![0.0; n_nodes];
                let mut d_s_dy = vec![0.0; n_nodes];

                let concentrations: Vec<TracerState> = (0..n_nodes)
                    .map(|i| {
                        let h = swe.get_state(k_idx, i).h;
                        tracers.get_concentrations(k_idx, i, h, h_min.meters())
                    })
                    .collect();

                for i in 0..n_nodes {
                    let mut dt_dr = 0.0;
                    let mut dt_ds = 0.0;
                    let mut ds_dr = 0.0;
                    let mut ds_ds = 0.0;

                    for j in 0..n_nodes {
                        let dr_ij = ops.dr[(i, j)];
                        let ds_ij = ops.ds[(i, j)];

                        dt_dr += dr_ij * concentrations[j].temperature;
                        dt_ds += ds_ij * concentrations[j].temperature;
                        ds_dr += dr_ij * concentrations[j].salinity;
                        ds_ds += ds_ij * concentrations[j].salinity;
                    }

                    d_t_dx[i] = dt_dr * rx + dt_ds * sx;
                    d_t_dy[i] = dt_dr * ry + dt_ds * sy;
                    d_s_dx[i] = ds_dr * rx + ds_ds * sx;
                    d_s_dy[i] = ds_dr * ry + ds_ds * sy;
                }

                for i in 0..n_nodes {
                    let h = swe.get_state(k_idx, i).h.max(h_min.meters());

                    let mut d_dtdx_dr = 0.0;
                    let mut d_dtdx_ds = 0.0;
                    let mut d_dtdy_dr = 0.0;
                    let mut d_dtdy_ds = 0.0;
                    let mut d_dsdx_dr = 0.0;
                    let mut d_dsdx_ds = 0.0;
                    let mut d_dsdy_dr = 0.0;
                    let mut d_dsdy_ds = 0.0;

                    for j in 0..n_nodes {
                        let dr_ij = ops.dr[(i, j)];
                        let ds_ij = ops.ds[(i, j)];

                        d_dtdx_dr += dr_ij * d_t_dx[j];
                        d_dtdx_ds += ds_ij * d_t_dx[j];
                        d_dtdy_dr += dr_ij * d_t_dy[j];
                        d_dtdy_ds += ds_ij * d_t_dy[j];

                        d_dsdx_dr += dr_ij * d_s_dx[j];
                        d_dsdx_ds += ds_ij * d_s_dx[j];
                        d_dsdy_dr += dr_ij * d_s_dy[j];
                        d_dsdy_ds += ds_ij * d_s_dy[j];
                    }

                    let d2t_dx2 = rx * d_dtdx_dr + sx * d_dtdx_ds;
                    let d2t_dy2 = ry * d_dtdy_dr + sy * d_dtdy_ds;
                    let laplacian_t = d2t_dx2 + d2t_dy2;

                    let d2s_dx2 = rx * d_dsdx_dr + sx * d_dsdx_ds;
                    let d2s_dy2 = ry * d_dsdy_dr + sy * d_dsdy_ds;
                    let laplacian_s = d2s_dx2 + d2s_dy2;

                    let diffusion = ConservativeTracerState {
                        h_t: config.kappa_t * h * laplacian_t,
                        h_s: config.kappa_s * h * laplacian_s,
                    };
                    rhs_k[i] = rhs_k[i] + diffusion;
                }
            }

            // 4. Source terms
            if let Some(sources) = config.source_terms {
                for i in 0..n_nodes {
                    let tracer = tracers.get_conservative(k_idx, i);
                    let swe_state = swe.get_state(k_idx, i);
                    let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                    let [x, y] = mesh.reference_to_physical(k_idx, r, s);

                    let source = sources.evaluate(time, (x, y), &swe_state, &tracer);
                    rhs_k[i] = rhs_k[i] + source;
                }
            }

            // Write to output chunk: [hT0, hS0, hT1, hS1, ...]
            for (i, state) in rhs_k.into_iter().enumerate() {
                rhs_chunk[i * 2] = state.h_t;
                rhs_chunk[i * 2 + 1] = state.h_s;
            }
        });

    rhs
}

/// Compute stable time step for tracer advection.
///
/// The CFL condition for pure advection is the same as for SWE
/// (dominated by wave speed). For diffusion, there's an additional
/// constraint: dt ≤ dx² / (2κ).
pub fn compute_dt_tracer_2d(
    swe: &SWESolution2D,
    mesh: &Mesh2D,
    geom: &GeometricFactors2D,
    g: f64,
    h_min: f64,
    kappa: f64,
    order: usize,
    cfl: f64,
) -> f64 {
    let mut max_speed: f64 = 0.0;
    let mut min_h_elem = f64::INFINITY;

    for k in ElementIndex::iter(mesh.n_elements) {
        let h_elem = geom.det_j[k.as_usize()].sqrt() * 2.0;
        min_h_elem = min_h_elem.min(h_elem);

        for i in 0..swe.n_nodes {
            let state = swe.get_state(k, i);
            let (u, v) = state.velocity_simple(Depth::new(h_min));
            let c = (g * state.h.max(0.0)).sqrt();
            let speed = (u * u + v * v).sqrt() + c;
            max_speed = max_speed.max(speed);
        }
    }

    if max_speed < 1e-14 && kappa < 1e-14 {
        return f64::INFINITY;
    }

    let dg_factor = 2.0 * order as f64 + 1.0;

    // Advection CFL
    let dt_advection = if max_speed > 1e-14 {
        cfl * min_h_elem / (max_speed * dg_factor)
    } else {
        f64::INFINITY
    };

    // Diffusion stability (more restrictive for high-order)
    let dt_diffusion = if kappa > 1e-14 {
        let diff_factor = dg_factor * dg_factor;
        cfl * min_h_elem * min_h_elem / (2.0 * kappa * diff_factor)
    } else {
        f64::INFINITY
    };

    dt_advection.min(dt_diffusion)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh2D;
    use crate::operators::{DGOperators2D, GeometricFactors2D};

    const G: f64 = 9.81;
    const H_MIN: f64 = 1e-6;
    const TOL: f64 = 1e-10;

    fn create_test_setup(order: usize) -> (Mesh2D, DGOperators2D, GeometricFactors2D) {
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh);
        (mesh, ops, geom)
    }

    #[test]
    fn test_rhs_uniform_tracers() {
        // Uniform tracers with uniform velocity should give zero RHS
        let (mesh, ops, geom) = create_test_setup(2);
        let bc = ExtrapolationTracerBC;
        let config = Tracer2DRhsConfig::new(&bc, G, H_MIN);

        // Uniform SWE state: h=10, u=1, v=0
        let mut swe = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                swe.set_state(k, i, SWEState2D::from_primitives(10.0, 1.0, 0.0));
            }
        }

        // Uniform tracers
        let tracers = TracerSolution2D::uniform(
            mesh.n_elements,
            ops.n_nodes,
            10.0,
            TracerState::new(8.0, 34.0),
        );

        let rhs = compute_rhs_tracer_2d(&tracers, &swe, &mesh, &ops, &geom, &config, 0.0);

        // Check RHS is zero
        let mut max_rhs: f64 = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let state = rhs.get_conservative(k, i);
                max_rhs = max_rhs.max(state.h_t.abs());
                max_rhs = max_rhs.max(state.h_s.abs());
            }
        }

        assert!(
            max_rhs < TOL,
            "Uniform tracers should give zero RHS, got {}",
            max_rhs
        );
    }

    #[test]
    fn test_rhs_zero_velocity() {
        // Zero velocity should give zero advection
        let (mesh, ops, geom) = create_test_setup(2);
        let bc = ExtrapolationTracerBC;
        let config = Tracer2DRhsConfig::new(&bc, G, H_MIN);

        // Still water
        let mut swe = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                swe.set_state(k, i, SWEState2D::from_primitives(10.0, 0.0, 0.0));
            }
        }

        // Non-uniform tracers
        let mut tracers = TracerSolution2D::new(mesh.n_elements, ops.n_nodes);
        tracers.set_from_functions(
            &mesh,
            &ops,
            |x, _| 8.0 + x,  // T varies with x
            |_, y| 34.0 + y, // S varies with y
            |_, _| 10.0,
        );

        let rhs = compute_rhs_tracer_2d(&tracers, &swe, &mesh, &ops, &geom, &config, 0.0);

        // RHS should be zero (no advection without velocity)
        let mut max_rhs: f64 = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let state = rhs.get_conservative(k, i);
                max_rhs = max_rhs.max(state.h_t.abs());
                max_rhs = max_rhs.max(state.h_s.abs());
            }
        }

        assert!(
            max_rhs < TOL,
            "Zero velocity should give zero RHS, got {}",
            max_rhs
        );
    }

    #[test]
    fn test_tracer_conservation() {
        // Total tracer content should be conserved (periodic domain)
        let (mesh, ops, geom) = create_test_setup(2);
        let bc = ExtrapolationTracerBC;
        let config = Tracer2DRhsConfig::new(&bc, G, H_MIN);

        // Uniform velocity
        let mut swe = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                swe.set_state(k, i, SWEState2D::from_primitives(10.0, 1.0, 0.5));
            }
        }

        // Non-uniform tracers
        let mut tracers = TracerSolution2D::new(mesh.n_elements, ops.n_nodes);
        tracers.set_from_functions(
            &mesh,
            &ops,
            |x, y| {
                8.0 + (2.0 * std::f64::consts::PI * x).sin()
                    * (2.0 * std::f64::consts::PI * y).cos()
            },
            |_, _| 34.0,
            |_, _| 10.0,
        );

        let rhs = compute_rhs_tracer_2d(&tracers, &swe, &mesh, &ops, &geom, &config, 0.0);

        // Integrate RHS (should be zero for conservation)
        let mut integral_h_t = 0.0;
        let mut integral_h_s = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            let j = geom.det_j[k.as_usize()];
            for (i, &w) in ops.weights.iter().enumerate() {
                let state = rhs.get_conservative(k, i);
                integral_h_t += w * state.h_t * j;
                integral_h_s += w * state.h_s * j;
            }
        }

        assert!(
            integral_h_t.abs() < 1e-10,
            "hT should be conserved: d(total)/dt = {:.2e}",
            integral_h_t
        );
        assert!(
            integral_h_s.abs() < 1e-10,
            "hS should be conserved: d(total)/dt = {:.2e}",
            integral_h_s
        );
    }

    #[test]
    fn test_upwind_bc() {
        // Test upwind BC behavior
        let bc = UpwindTracerBC::atlantic();

        // Inflow case
        let ctx_in = TracerBCContext2D::new(
            0.0,
            (0.0, 0.0),
            ConservativeTracerState::new(80.0, 340.0), // Interior: T=8, S=34
            SWEState2D::from_primitives(10.0, -1.0, 0.0), // Flow into domain
            (1.0, 0.0),                                // Normal pointing out
            H_MIN,
            G,
        );

        let (ghost_in, _) = bc.ghost_state(&ctx_in);
        // Should use Atlantic values
        let tracers_in = ghost_in.to_concentrations(10.0, H_MIN);
        assert!((tracers_in.temperature - 7.5).abs() < TOL);
        assert!((tracers_in.salinity - 35.0).abs() < TOL);

        // Outflow case
        let ctx_out = TracerBCContext2D::new(
            0.0,
            (0.0, 0.0),
            ConservativeTracerState::new(80.0, 340.0),
            SWEState2D::from_primitives(10.0, 1.0, 0.0), // Flow out of domain
            (1.0, 0.0),
            H_MIN,
            G,
        );

        let (ghost_out, _) = bc.ghost_state(&ctx_out);
        // Should extrapolate interior
        let tracers_out = ghost_out.to_concentrations(10.0, H_MIN);
        assert!((tracers_out.temperature - 8.0).abs() < TOL);
        assert!((tracers_out.salinity - 34.0).abs() < TOL);
    }

    #[test]
    fn test_dt_computation() {
        let (mesh, ops, geom) = create_test_setup(2);

        let mut swe = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                swe.set_state(k, i, SWEState2D::from_primitives(10.0, 2.0, 1.0));
            }
        }

        // Pure advection
        let dt_adv = compute_dt_tracer_2d(&swe, &mesh, &geom, G, H_MIN, 0.0, ops.order, 0.5);
        assert!(dt_adv > 0.0);
        assert!(dt_adv < f64::INFINITY);

        // With diffusion - should be more restrictive
        let dt_diff = compute_dt_tracer_2d(&swe, &mesh, &geom, G, H_MIN, 10.0, ops.order, 0.5);
        assert!(dt_diff > 0.0);
        assert!(dt_diff < dt_adv, "Diffusion should reduce dt");
    }

    #[test]
    fn test_flux_types() {
        let (mesh, ops, geom) = create_test_setup(2);
        let bc = ExtrapolationTracerBC;

        let mut swe = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                swe.set_state(k, i, SWEState2D::from_primitives(10.0, 1.0, 0.0));
            }
        }

        let tracers = TracerSolution2D::uniform(
            mesh.n_elements,
            ops.n_nodes,
            10.0,
            TracerState::new(8.0, 34.0),
        );

        // Test all flux types produce zero RHS for uniform state
        for flux_type in [
            TracerFluxType::Upwind,
            TracerFluxType::Roe,
            TracerFluxType::LaxFriedrichs,
        ] {
            let config = Tracer2DRhsConfig::new(&bc, G, H_MIN).with_flux_type(flux_type);
            let rhs = compute_rhs_tracer_2d(&tracers, &swe, &mesh, &ops, &geom, &config, 0.0);

            let mut max_rhs: f64 = 0.0;
            for k in ElementIndex::iter(mesh.n_elements) {
                for i in 0..ops.n_nodes {
                    let state = rhs.get_conservative(k, i);
                    max_rhs = max_rhs.max(state.h_t.abs());
                    max_rhs = max_rhs.max(state.h_s.abs());
                }
            }

            assert!(
                max_rhs < TOL,
                "{:?} flux should give zero RHS for uniform, got {}",
                flux_type,
                max_rhs
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_tracer_matches_serial() {
        use super::compute_rhs_tracer_2d_parallel;

        // Test with non-uniform flow and tracers for thorough comparison
        let (mesh, ops, geom) = create_test_setup(3); // Higher order
        let bc = ExtrapolationTracerBC;
        let config = Tracer2DRhsConfig::new(&bc, G, H_MIN);

        // Non-uniform SWE state
        let mut swe = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [x, y] = mesh.reference_to_physical(k, r, s);
                let h = 10.0 + 0.5 * (2.0 * std::f64::consts::PI * x).sin();
                let u = 1.0 + 0.2 * y;
                let v = 0.5 - 0.1 * x;
                swe.set_state(k, i, SWEState2D::from_primitives(h, u, v));
            }
        }

        // Non-uniform tracers
        let mut tracers = TracerSolution2D::new(mesh.n_elements, ops.n_nodes);
        tracers.set_from_functions(
            &mesh,
            &ops,
            |x, y| 8.0 + (2.0 * std::f64::consts::PI * x).cos() * (std::f64::consts::PI * y).sin(),
            |x, y| 34.0 + 0.5 * x * y,
            |x, _| 10.0 + 0.5 * (2.0 * std::f64::consts::PI * x).sin(),
        );

        // Compute serial and parallel RHS
        let rhs_serial = compute_rhs_tracer_2d(&tracers, &swe, &mesh, &ops, &geom, &config, 0.5);
        let rhs_parallel =
            compute_rhs_tracer_2d_parallel(&tracers, &swe, &mesh, &ops, &geom, &config, 0.5);

        // Compare results
        let mut max_diff: f64 = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let serial = rhs_serial.get_conservative(k, i);
                let parallel = rhs_parallel.get_conservative(k, i);

                let diff_h_t = (serial.h_t - parallel.h_t).abs();
                let diff_h_s = (serial.h_s - parallel.h_s).abs();

                max_diff = max_diff.max(diff_h_t).max(diff_h_s);
            }
        }

        assert!(
            max_diff < 1e-12,
            "Parallel tracer RHS should match serial: max diff = {:.2e}",
            max_diff
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_tracer_with_diffusion() {
        use super::compute_rhs_tracer_2d_parallel;

        // Test with diffusion enabled
        let (mesh, ops, geom) = create_test_setup(2);
        let bc = ExtrapolationTracerBC;
        let config = Tracer2DRhsConfig::new(&bc, G, H_MIN).with_diffusivity(1.0);

        let mut swe = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                swe.set_state(k, i, SWEState2D::from_primitives(10.0, 1.0, 0.5));
            }
        }

        let mut tracers = TracerSolution2D::new(mesh.n_elements, ops.n_nodes);
        tracers.set_from_functions(
            &mesh,
            &ops,
            |x, y| 8.0 + (2.0 * std::f64::consts::PI * x).sin() * y,
            |x, _| 34.0 + x,
            |_, _| 10.0,
        );

        let rhs_serial = compute_rhs_tracer_2d(&tracers, &swe, &mesh, &ops, &geom, &config, 0.0);
        let rhs_parallel =
            compute_rhs_tracer_2d_parallel(&tracers, &swe, &mesh, &ops, &geom, &config, 0.0);

        let mut max_diff: f64 = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let serial = rhs_serial.get_conservative(k, i);
                let parallel = rhs_parallel.get_conservative(k, i);
                max_diff = max_diff
                    .max((serial.h_t - parallel.h_t).abs())
                    .max((serial.h_s - parallel.h_s).abs());
            }
        }

        assert!(
            max_diff < 1e-12,
            "Parallel tracer RHS with diffusion should match serial: max diff = {:.2e}",
            max_diff
        );
    }
}
