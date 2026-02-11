# ROMS vs dg-rs: Comprehensive Comparison Report

## Executive Summary

This report compares the operational ROMS (Regional Ocean Modeling System) -- specifically the NorKyst-800 configuration used by MET Norway -- with this Rust-based Discontinuous Galerkin (DG) solver (`dg-rs`). The analysis covers architecture, numerical methods, feature completeness, and a prioritized roadmap for closing the gap with ROMS for Norwegian coastal modeling.

**Bottom line**: `dg-rs` has a remarkably complete 2D shallow water solver with many ROMS-like features (sigma coordinates, tidal forcing, wetting/drying, tracer transport, equation of state, GPU acceleration). The critical gaps are: **full 3D baroclinic mode**, **vertical mixing parameterizations**, **split-explicit time stepping**, and **operational data assimilation**. The DG approach offers fundamental advantages in geometric flexibility, conservation, and high-order accuracy that could make it superior to ROMS for Norwegian fjord modeling -- if the remaining features are implemented.

---

## 1. How Real ROMS Works

### 1.1 Grid and Spatial Discretization

ROMS uses a **structured curvilinear orthogonal grid** on an **Arakawa C-grid** staggering, where:
- Free-surface elevation, density, and tracers are located at cell centers (rho-points)
- Horizontal velocities u, v are staggered to cell edges (u-points, v-points)

Vertically, ROMS uses **terrain-following (sigma/s) coordinates** with the Song-Haidvogel stretching function. The coordinate sigma ranges from -1 (bottom) to 0 (surface), with configurable stretching parameters (theta_s, theta_b, hc) that control resolution distribution near the surface and bottom.

**Key references**: Shchepetkin & McWilliams (2005), Song & Haidvogel (1994)

### 1.2 Governing Equations

ROMS solves the **3D hydrostatic primitive equations** under the Boussinesq approximation:

- **Momentum**: Horizontal momentum equations with pressure gradient, Coriolis, advection, and turbulent mixing
- **Continuity**: 3D incompressibility constraint
- **Free surface**: Vertically integrated continuity equation
- **Tracers**: Temperature and salinity transport with diffusion
- **Equation of state**: Density from T, S (and optionally pressure)

### 1.3 Split-Explicit Time Stepping

The most distinctive numerical feature of ROMS is its **split-explicit time-stepping scheme**:

- **Baroclinic (slow) mode**: 3D momentum and tracer equations are advanced with a large time step (typically 10-300 seconds for NorKyst-800)
- **Barotropic (fast) mode**: 2D depth-averaged free surface and transport equations are subcycled with many small time steps within each baroclinic step

The barotropic mode uses a specially designed cosine-shape time filter to average the fast mode, guaranteeing exact conservation and preventing aliasing. This is critical for stability because surface gravity waves propagate much faster than baroclinic processes.

ROMS uses a **3rd-order Adams-Bashforth-like predictor** and a **trapezoidal corrector** for the baroclinic mode, coupled with forward-backward subcycling for the barotropic mode.

### 1.4 Advection Schemes

ROMS supports multiple advection options:
- **3rd-order upstream-biased** (model default): Provides implicit diffusion, reducing need for explicit horizontal viscosity
- **4th-order centered**: Higher accuracy but requires explicit diffusion
- **MPDATA**: Multidimensional positive-definite advection for tracers
- **Akima spline** advection for vertical transport

### 1.5 Vertical Mixing: KPP and GLS

ROMS provides several vertical mixing parameterizations:

**KPP (K-Profile Parameterization)**:
- Determines boundary layer depth based on bulk Richardson number
- Non-local transport in convective boundary layers
- Matches interior mixing in the thermocline
- Standard choice for most applications

**GLS (Generic Length Scale)**:
- Two-equation turbulence closure (TKE + length scale)
- Can recover k-epsilon, k-omega, k-kl, and Mellor-Yamada formulations through parameter choices
- More physically based than KPP for energetic mixing regimes (tides, strong currents)
- Recommended for coastal/fjord applications with strong tidal mixing

**MY2.5 (Mellor-Yamada 2.5)**:
- Two-equation model based on turbulent kinetic energy and turbulence length scale
- Still widely used but GLS is generally preferred

### 1.6 Data Assimilation

ROMS has a unique 4D-Var data assimilation capability supporting three variants:
- **I4D-Var**: Primal formulation in model space
- **4D-PSAS**: Dual formulation using physical-space statistical analysis
- **R4D-Var**: Representer-based dual formulation

These support both strong constraint (model is perfect) and weak constraint (model has errors) modes, and work with nested grids.

### 1.7 Grid Nesting

ROMS supports:
- **One-way nesting**: Parent model provides boundary conditions to child
- **Two-way nesting**: Child feeds back to parent (refinement of parent solution)
- **Multiple nesting levels**: E.g., 2.4km -> 800m -> 160m

---

## 2. NorKyst-800: Norway's Operational Configuration

NorKyst-800 is the operational coastal ocean forecasting system for mainland Norway, run daily by MET Norway and the Institute of Marine Research (IMR).

### 2.1 Model Domain and Resolution

| Parameter | NorKyst v2 | NorKyst v3 |
|-----------|-----------|-----------|
| Horizontal resolution | 800 m | 800 m |
| Grid points | 2747 x 902 | 2747 x 1148 |
| Vertical levels | 35 | 40 |
| Total cells | ~87 million | ~126 million |
| Forecast length | 66 hours | 120 hours (5 days) |

### 2.2 Vertical Configuration

- **Stretching**: Song-Haidvogel with strong surface refinement
- **theta_s**: 5-7 (strong surface layer resolution)
- **theta_b**: 0-1 (weak to moderate bottom refinement)
- **hc**: 200-300 m (critical depth for deep fjords)
- 40 levels with high resolution prioritized in uppermost layers

### 2.3 Forcing

- **Atmospheric**: Hourly fields from AROME-MetCoOP (wind, pressure, heat flux, precipitation)
- **Tidal**: 8 primary harmonic constituents from TPXO global tidal model
- **Lateral boundaries**: Temperature and salinity from TOPAZ (Arctic) and CMEMS Baltic model
- **Rivers**: 1760 rivers with daily discharge from NVE, distributed to 69 coastal regions
- **Freshwater runoff**: Froude-number-dependent vertical distribution

### 2.4 Key Physical Parameterizations Used

- **Vertical mixing**: GLS (Generic Length Scale) -- k-epsilon or k-omega formulation
- **Bottom friction**: Quadratic bottom drag with log-layer formulation
- **Horizontal mixing**: Smagorinsky-type
- **Advection**: 3rd-order upstream-biased (ROMS default)
- **Wetting/drying**: Not standard in NorKyst but available in ROMS

### 2.5 Nested Models

NorKyst v3 includes two-way nested 160m resolution models for:
- **Oslofjorden**: Complex urban fjord with shipping, sewage
- **Sulafjorden**: Active aquaculture region

---

## 3. Feature Gap Analysis: dg-rs vs ROMS

### 3.1 What dg-rs Already Has

The `dg-rs` codebase is impressively comprehensive for a 2D solver. Here is what exists:

#### Spatial Discretization
- [x] Nodal DG with Legendre polynomial basis (arbitrary order p)
- [x] Gauss-Lobatto-Legendre (GLL) quadrature nodes
- [x] Vandermonde matrices for nodal-modal transforms (1D and 2D)
- [x] Differentiation, mass, LIFT operators (1D and 2D)
- [x] **2D quadrilateral mesh** with edge-based connectivity
- [x] Gmsh file reader for unstructured meshes
- [x] GeoTIFF bathymetry reader
- [x] Coastline shapefile reader
- [x] Land mask handling
- [x] UTM and local coordinate projections

#### Equations and Fluxes
- [x] 1D and 2D Shallow Water Equations (conservative form)
- [x] 1D and 2D scalar advection
- [x] Multiple numerical fluxes: Roe, HLL, HLLC, Rusanov/Lax-Friedrichs, upwind
- [x] Eigenvalue-based flux evaluation
- [x] Tracer transport (temperature, salinity) with upwind/Roe/Lax-Friedrichs fluxes
- [x] **UNESCO EOS-80 equation of state** (full and linear)
- [x] **Baroclinic pressure source term** (depth-averaged)

#### Source Terms
- [x] Coriolis effect (f-plane and beta-plane)
- [x] Bottom friction (Manning, Chezy, spatially varying)
- [x] Wind stress with drag coefficient parameterization
- [x] Atmospheric pressure gradients
- [x] Tidal potential forcing
- [x] Bathymetry gradients (well-balanced hydrostatic reconstruction)
- [x] Sponge layers with multiple profiles

#### Boundary Conditions
- [x] Reflective (wall, no-flux)
- [x] Radiation (Sommerfeld)
- [x] Chapman (SSH radiation)
- [x] Flather (characteristic-based velocity)
- [x] Chapman-Flather combined
- [x] Harmonic tidal (multiple constituents)
- [x] Harmonic Flather (tidal + velocity radiation)
- [x] Discharge (prescribed flow rate)
- [x] One-way nesting from parent model
- [x] Ocean nesting from NetCDF (with `netcdf` feature)
- [x] TST (Time Series Template) open boundaries
- [x] Multi-boundary condition support (different BCs per boundary tag)

#### Time Integration
- [x] SSP-RK3 (Strong Stability Preserving Runge-Kutta, 3rd order)
- [x] Forward Euler (for testing)
- [x] Coupled SWE + tracer time stepping
- [x] Adaptive time stepping (CFL-based)
- [x] Time-dependent boundary conditions (evaluated at correct RK stage times)

#### Limiters and Stability
- [x] TVB (Total Variation Bounded) slope limiter
- [x] Kuzmin vertex-based limiter
- [x] Positivity-preserving limiter
- [x] Limiter chaining (composable)
- [x] **Wetting/drying algorithm**

#### Vertical Coordinates (for future 3D)
- [x] Sigma coordinate grid with ROMS convention
- [x] Song-Haidvogel stretching
- [x] Double tanh stretching
- [x] Uniform stretching
- [x] Batch operations (SIMD-friendly)
- [x] Physical depth conversion

#### Performance
- [x] SIMD-optimized kernels (via `pulp` crate)
- [x] Parallel RHS computation (via `rayon`)
- [x] GPU acceleration via Burn framework (CUDA, WGPU, NdArray backends)
- [x] Structure-of-arrays (SoA) memory layout for hot paths
- [x] Pre-allocated workspaces to avoid allocation in RHS

#### Analysis and Validation
- [x] Harmonic analysis (tidal constituent extraction)
- [x] Tide gauge station validation
- [x] ADCP current validation
- [x] Comparison metrics (RMSE, skill score, etc.)
- [x] Stability monitoring
- [x] Diagnostics tracker and progress reporter

#### I/O
- [x] NetCDF output (CF-conventions compliant)
- [x] NetCDF forcing reader (ERA5, ECMWF compatible)
- [x] Ocean model reader (for nesting data)
- [x] VTK output for visualization
- [x] GeoTIFF bathymetry reader
- [x] Coastline shapefile reader
- [x] Tide gauge file reader/writer
- [x] Constituent data reader

### 3.2 Critical Gaps (Blocking for ROMS-equivalent capability)

| Feature | ROMS | dg-rs | Priority |
|---------|------|-------|----------|
| **3D baroclinic mode** | Full 3D primitive equations | 2D depth-averaged only | CRITICAL |
| **Vertical mixing** | KPP, GLS, MY2.5 | None (sigma grid exists but no mixing) | CRITICAL |
| **Split-explicit time stepping** | Barotropic/baroclinic splitting | Single-mode SSP-RK3 only | HIGH |
| **3D tracer transport** | Full 3D T, S with vertical mixing | 2D depth-averaged tracers | HIGH |
| **Horizontal mixing** | Smagorinsky, Laplacian, biharmonic | None | MEDIUM |
| **Data assimilation** | 4D-Var (3 variants) | None | LOW (for research) |
| **Two-way nesting** | Full implementation | One-way nesting only | LOW |

### 3.3 Moderate Gaps (Important for production use)

| Feature | ROMS | dg-rs | Priority |
|---------|------|-------|----------|
| **River forcing** | 1760 rivers with vertical shape | Discharge BC exists but no vertical distribution | HIGH |
| **Heat/moisture fluxes** | Full surface heat/moisture budget | Wind stress and atmospheric pressure only | MEDIUM |
| **Ice model coupling** | ROMS-CICE coupling | None | LOW (Norway-specific) |
| **Biological models** | NPZD, Fennel, etc. | None | LOW |
| **Sediment transport** | ROMS-SED | None | LOW |

### 3.4 Features Where dg-rs is Ahead of or Equal to ROMS

| Feature | ROMS | dg-rs |
|---------|------|-------|
| **Spatial accuracy** | 2nd-3rd order FD | Arbitrary high-order DG (spectral accuracy) |
| **Local conservation** | Global conservation (FD) | Element-wise exact conservation |
| **Geometric flexibility** | Structured curvilinear grid | Unstructured quadrilateral mesh |
| **hp-adaptivity potential** | Fixed order everywhere | Can vary polynomial order per element |
| **Wetting/drying** | Available but tricky on FD grids | Natural in DG framework |
| **GPU acceleration** | Limited (some CUDA ports exist) | Built-in Burn framework (CUDA/WGPU) |
| **Language safety** | Fortran (no memory safety) | Rust (memory safe, no data races) |
| **Parallelism model** | MPI (message passing) | Rayon (shared memory) + GPU |

---

## 4. Advantages of the DG Approach for Norwegian Coastal Modeling

### 4.1 Fjord Geometry

Norwegian fjords present extreme geometric challenges:
- Narrow straits (100-500m wide)
- Deep basins (up to 1300m) adjacent to shallow sills (20-50m)
- Complex branching geometry
- Thousands of islands and skerries

ROMS uses a **structured grid**, meaning it must waste resolution on open ocean areas to maintain resolution in narrow fjord passages. A 800m structured grid cannot resolve a 200m-wide strait without local refinement, which requires nesting.

DG on **unstructured meshes** can place fine elements exactly where needed: in narrow straits, around sills, and along complex coastlines -- without wasting resolution in open water. This is a fundamental advantage.

### 4.2 Conservation Properties

DG methods are **locally conservative by construction**: mass, momentum, and tracer content are conserved element-by-element. This is critical for:
- Long-term simulations where conservation errors accumulate
- Tracer transport (salinity, temperature, pollutants, fish larvae)
- Wetting/drying fronts where FD methods can lose mass

ROMS achieves global conservation through careful numerical design, but local conservation requires the split-explicit averaging filter -- a complex mechanism that DG avoids entirely.

### 4.3 High-Order Accuracy

DG with polynomial order p achieves (p+1)-th order accuracy, and for smooth solutions, approaches spectral accuracy. This means:
- Better representation of tidal propagation over long distances
- Sharper density fronts (less numerical diffusion)
- Better resolution of eddies and vortices per degree of freedom

ROMS is limited to 2nd-3rd order accuracy with its finite difference stencils.

### 4.4 Shock-Capturing for Hydraulic Transitions

Norwegian coastal waters feature hydraulic jumps and bores at sills and in narrow straits. DG methods with limiters (TVB, Kuzmin, positivity-preserving -- all implemented in dg-rs) handle these discontinuities naturally through their element-interface framework.

### 4.5 GPU Suitability

DG methods are inherently more GPU-friendly than FD methods because:
- Element-local operations are independent (embarrassingly parallel)
- Matrix-vector operations (differentiation, lifting) map well to GPU BLAS
- No global linear system solve required (unlike continuous Galerkin)

The existing Burn integration in dg-rs positions it well for GPU acceleration.

---

## 5. Prioritized Development Roadmap

### Phase 1: Complete the 2D Solver (1-2 months)

These items complete the 2D capability to operational quality:

1. **Horizontal viscosity/diffusion**: Implement Smagorinsky-type turbulent viscosity for the 2D solver. This stabilizes flow around complex geometry and is needed for realistic eddy patterns.

2. **Surface heat flux budget**: Extend forcing to include shortwave radiation, longwave radiation, sensible/latent heat fluxes. These are needed for multi-day forecasts where SST evolution matters.

3. **River forcing enhancement**: Add vertical shape function to discharge BCs (Froude-number-dependent distribution as in NorKyst). Currently the discharge BC is 2D-only, but this prepares for 3D.

4. **Operational validation**: Run comparison against NorKyst-800 output for a test period at known tide gauge and ADCP stations. The validation infrastructure already exists.

### Phase 2: 3D Extension (3-6 months)

This is the critical path to ROMS-equivalent capability:

5. **3D solution storage**: Extend `SWESolution2D` to store 3D fields on the sigma grid. Use the existing `SigmaGrid` module. Layout: `[n_elements x n_nodes x n_levels]` for each variable.

6. **3D tracer transport**: Extend the tracer solver to 3D with vertical advection and diffusion. The 2D tracer framework and equation of state already exist.

7. **Vertical mixing parameterization**: Implement at least one closure scheme:
   - **Start with GLS** (k-epsilon): This is what NorKyst uses, and it is the most physically based for coastal applications
   - GLS is a 1D vertical problem at each horizontal point, making it relatively straightforward to implement

8. **Baroclinic pressure gradient**: Extend the existing baroclinic source from depth-averaged to full 3D. This is the internal pressure gradient that drives estuarine circulation.

9. **Split-explicit time stepping**: Implement barotropic/baroclinic mode splitting:
   - Barotropic: 2D depth-averaged SWE (already working)
   - Baroclinic: 3D momentum and tracers with larger time step
   - Coupling via time-averaged barotropic fields

   Alternatively, consider whether SSP-RK3 without splitting is viable for the target resolutions. For 800m resolution with depths of 100-1000m, the barotropic CFL would require dt ~ 1-3s, while baroclinic dt could be 30-100s. Splitting gives a factor of 10-30x speedup.

### Phase 3: Production Features (6-12 months)

10. **Two-way nesting**: Extend the one-way nesting BC to support feedback from child to parent grids. This enables the multi-scale approach NorKyst uses.

11. **MPI parallelism**: For truly large domains (millions of elements), shared-memory parallelism (rayon) is insufficient. Add domain decomposition with MPI for distributed memory. (Alternatively, if GPU performance is sufficient, a single-GPU approach may cover many use cases.)

12. **Data assimilation**: Start with simple ensemble methods (EnKF) or optimal interpolation before tackling 4D-Var. Assimilation of SST and SSH observations would significantly improve forecast skill.

13. **Biological coupling**: Interface for biogeochemical models (NPZD as a starting point). Important for salmon lice dispersion -- a major application of NorKyst.

---

## 6. Comparison with Other DG Ocean Models

### 6.1 Thetis (Firedrake/Python)

[Thetis](https://gmd.copernicus.org/articles/11/4359/2018/) is the closest existing DG coastal ocean model:
- Full 3D hydrostatic with DG discretization on triangular prisms
- Mode splitting for barotropic/baroclinic coupling
- Built on Firedrake (automated FEM code generation)
- Second-order accuracy in space and time
- Python-based (slower than compiled code)

**dg-rs advantage**: Rust performance, GPU acceleration, operational focus
**Thetis advantage**: Already has 3D mode, automated weak form derivation

### 6.2 DGSWE (various academic codes)

Several academic DG shallow water codes exist but none target operational Norwegian coastal modeling specifically.

---

## 7. Risk Assessment

### 7.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Split-explicit coupling instability in DG | Medium | High | Fallback to non-split SSP-RK3 for moderate domains |
| Sigma coordinate pressure gradient errors | Medium | High | Well-documented problem; use density Jacobian method |
| GPU memory limitations for 3D | Low | Medium | Hybrid CPU-GPU approach; only accelerate element-local ops |
| Numerical diffusion at high order | Low | Low | DG inherently low-diffusion; limiters well-implemented |

### 7.2 Scope Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Feature creep toward full ROMS clone | High | High | Focus on Norwegian coast applications only |
| Insufficient validation data | Medium | Medium | NorKyst output freely available on THREDDS |
| Performance insufficient for operational use | Medium | High | GPU acceleration already in place; profile and optimize |

---

## 8. Conclusion

`dg-rs` represents a compelling alternative approach to Norwegian coastal ocean modeling. Its 2D SWE solver is feature-complete and well-engineered, with particularly strong implementations of:
- Boundary conditions (Chapman, Flather, harmonic tidal -- matching ROMS capabilities)
- Source terms (Coriolis, bottom friction, wind, atmospheric pressure, tidal potential)
- Validation infrastructure (tide gauge, ADCP, harmonic analysis)
- Performance infrastructure (SIMD, parallel, GPU)

The path to ROMS-equivalent capability is clear but substantial: the 3D extension with vertical mixing is the critical milestone. The DG approach offers genuine advantages for Norwegian fjord modeling that justify this investment -- particularly in geometric flexibility for complex fjord topography and high-order accuracy for tidal propagation.

The recommended strategy is to validate the 2D solver against NorKyst-800 output for barotropic tides, then extend to 3D incrementally, using the existing sigma coordinate and tracer infrastructure as a foundation.

---

## References

### ROMS Core
- Shchepetkin, A.F. & McWilliams, J.C. (2005). [The Regional Oceanic Modeling System (ROMS): a split-explicit, free-surface, topography-following-coordinate oceanic model](https://people.atmos.ucla.edu/alex/ROMS/ROMSArticle2005.pdf). Ocean Modelling, 9(4), 347-404.
- [WikiROMS: Numerical Solution Technique](https://www.myroms.org/wiki/Numerical_Solution_Technique)
- [WikiROMS: Vertical Mixing Parameterizations](https://www.myroms.org/wiki/Vertical_Mixing_Parameterizations)
- [ROMS GitHub Repository](https://github.com/myroms/roms)

### NorKyst-800
- [MET Norway Ocean Models](https://ocean.met.no/models)
- [NorKyst version 3 preprint (2025)](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-3986/)
- [IMR Ocean Models](https://www.hi.no/en/hi/forskning/research-data-1/models/ocean-models)
- Albretsen, J. et al. (2011). [The hydrodynamic foundation for salmon lice dispersion modeling along the Norwegian coast](https://link.springer.com/article/10.1007/s10236-020-01378-0).

### DG Ocean Modeling
- K\"arna, T. et al. (2018). [Thetis coastal ocean model: discontinuous Galerkin discretization for the three-dimensional hydrostatic equations](https://gmd.copernicus.org/articles/11/4359/2018/). Geoscientific Model Development, 11, 4359-4382.
- Hesthaven, J.S. & Warburton, T. (2008). Nodal Discontinuous Galerkin Methods. Springer.

### Vertical Mixing
- Warner, J.C. et al. (2005). [Performance of four turbulence closure models implemented using a generic length scale method](https://www.sciencedirect.com/science/article/abs/pii/S1463500303000702). Ocean Modelling, 8(1-2), 81-113.
- [WikiROMS: GLS Mixing](https://www.myroms.org/wiki/GLS_MIXING)

### Data Assimilation
- Moore, A.M. et al. (2011). [The Regional Ocean Modeling System (ROMS) 4-dimensional variational data assimilation systems](https://www.sciencedirect.com/science/article/abs/pii/S0079661111000516). Progress in Oceanography, 91(1), 34-49.
