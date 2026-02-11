# TODO - Norwegian Coastal Model Development

Roadmap for developing a DG solver for simulating currents along the Norwegian coast.

---

## RESOLVED: Tracer Transport Instability (Fixed 2026-01-09)

**Status: FIXED** ✓

### Original Issue (Discovered 2026-01-08)

Running `examples/norwegian_fjord.rs` revealed severe numerical instability with blow-up.

### Root Causes Identified and Fixed

1. **[FIXED] Incorrect Laplacian in diffusion term** (`src/solver/rhs_tracer_2d.rs:537-587`)
   - Bug: Mixed reference/physical coordinate derivatives in chain rule
   - Fix: Proper second-derivative transformation: `∂²T/∂x² = rx·∂/∂r(∂T/∂x) + sx·∂/∂s(∂T/∂x)`

2. **[FIXED] Missing tracer limiters** (`src/solver/limiters_tracer_2d.rs` - NEW FILE)
   - Added TVB slope limiter for oscillation control
   - Added Zhang-Shu positivity limiter for physical bounds
   - Integrated into SSP-RK3 time stepping (applied after each stage)

3. **[FIXED] TVB threshold scaling bug** (`src/solver/limiters_tracer_2d.rs:68-118`)
   - Bug: TVB threshold = M * h_elem² with h_elem in meters (~1000m) made threshold ~10⁸
   - Fix: Normalized threshold = M * (h_elem / L_ref)² with domain size reference

### Results After Fix

```
t =   1.0 h | T: [7.4, 7.9] C | S: [34.4, 35.9] PSU  <- STABLE
t =   6.0 h | T: [7.4, 8.0] C | S: [34.1, 36.3] PSU  <- PHYSICAL
t =  12.4 h | T: [7.4, 8.2] C | S: [33.8, 36.2] PSU  <- CORRECT
```

### Files Modified
- `src/solver/rhs_tracer_2d.rs` - Fixed Laplacian computation
- `src/solver/limiters_tracer_2d.rs` - NEW: 2D tracer limiters with 9 unit tests
- `src/solver/mod.rs` - Export new limiter module
- `src/mesh/mesh2d.rs` - Added `element_diameter()` method
- `src/time/coupled_swe_tracer.rs` - Limiter integration, `with_tracer_limiters(m, domain_size)`
- `examples/norwegian_fjord.rs` - Enable limiters with proper domain scaling

### Remaining Items (Lower Priority)
- [x] Create minimal test case for regression testing (`tests/tracer_2d_test.rs`)
- [x] Verify tracer conservation with periodic BC test (`test_tracer_conservation_periodic`)
- [x] Kuzmin vertex-based limiter for unstructured meshes (implemented 2026-01-09)

---

## Phase 1: 1D Shallow Water Equations ✓

Foundation work before extending to 2D. Validate numerics on simpler 1D system.

### 1.1 System of Equations Support ✓
- [x] Trait-based equation interface (`ConservationLaw` trait)
- [x] State vector abstraction (h, hu) with `SWEState` and `SWESolution`
- [x] Implement 1D shallow water equations: ∂h/∂t + ∂(hu)/∂x = 0, ∂(hu)/∂t + ∂(hu² + gh²/2)/∂x = -ghB_x
- [x] Roe solver for shallow water (eigenvalue decomposition with entropy fix)
- [x] HLL/HLLC flux as alternative
- [x] Lax-Friedrichs flux for comparison

### 1.2 Source Terms ✓
- [x] Bottom topography source term (hydrostatic reconstruction, well-balanced)
- [x] Lake-at-rest preservation verified to machine precision
- [x] Bottom friction (Manning and Chézy formulations)
- [x] Semi-implicit friction for stiff cases
- [x] Bathymetry storage with gradient computation

### 1.3 Boundary Conditions ✓
- [x] Radiation/absorbing boundary conditions (characteristic-based)
- [x] Flather radiation condition
- [x] Sponge layer for gradual damping
- [x] Tidal forcing at open boundaries (harmonic constituents M2, S2, K1, O1, N2, P1)
- [x] Interpolated tidal BC for external forcing data
- [x] Wall (reflective) boundary conditions
- [x] Partial slip wall BC
- [x] Extrapolation and fixed state BCs
- [x] Discharge (prescribed flow rate) BC

### 1.4 Robustness ✓
- [x] Positivity-preserving limiter for water depth (Zhang-Shu type)
- [x] Wetting/drying treatment (velocity desingularization)
- [x] TVB limiter for shocks
- [x] Characteristic-based limiting for systems
- [x] Kuzmin vertex-based slope limiter (`tracer_kuzmin_limiter_2d`, `KuzminParameter2D`)

### 1.5 Advanced Schemes — *deferred to after Phase 2*
These are research-level enhancements that benefit more from 2D infrastructure:
- [ ] Entropy-stable DG (SBP operators + entropy-conservative fluxes) — *most papers target 2D curvilinear meshes*
- [ ] IMEX time stepping via diffsol — *primarily for 3D implicit vertical diffusion*
- [ ] Subcell positivity preservation with convex limiting (Wu et al. 2024) — *requires entropy-stable framework*

---

## Phase 2: 2D Implementation ✓

Core capability for coastal modeling.

### 2.1 Reference Elements ✓
- [x] 2D Legendre polynomials on reference square [-1,1]²
- [x] Tensor-product GLL nodes for quadrilaterals
- [ ] Fekete or Warp-Blend nodes for triangles — *deferred (quads sufficient for coastal modeling)*
- [x] 2D Vandermonde matrices

### 2.2 Operators ✓
- [x] 2D differentiation matrices (Dr, Ds)
- [x] 2D mass matrix
- [x] Surface LIFT operator for edges
- [x] Face interpolation matrices (volume → edge nodes)
- [x] Surface mass matrices (per face)
- [x] GeometricFactors struct (Jacobian, inverse Jacobian, det_J, normals, surface_J)
- [ ] Sum-factorization for tensor-product elements — *deferred (premature optimization)*
- [ ] Filter matrix for dealiasing/stabilization — *deferred to Phase 3*

### 2.3 Mesh ✓
- [x] 2D mesh data structure (elements, edges, vertices)
- [x] Edge-based connectivity for flux computation
- [ ] Support for mixed quad/triangle meshes — *deferred (quads sufficient)*
- [x] Mesh file I/O (Gmsh format 2.2)
- [ ] Curved elements for high-order geometry — *deferred to Phase 3*
- [x] BoundaryTag enum (Dirichlet, Neumann, Periodic, Open, TidalForcing, Wall, River, Custom)

### 2.4 Solver ✓
- [x] 2D advection equation (validation with convergence tests)
- [x] 2D shallow water equations with flux F(q), G(q)
- [x] Coriolis force (f-plane and β-plane, Norwegian coast config)
- [x] 2D Riemann solvers: Roe, HLL, Rusanov (rotation approach)
- [x] 2D boundary conditions: Reflective, Radiation, Flather, Tidal, Discharge
- [x] Lake-at-rest verification (well-balanced test)
- [x] Mass/momentum conservation verified

---

## Phase 3: Norwegian Coast Features (In Progress)

Physics and numerics specific to fjord/coastal modeling.

### 3.1 Tidal Dynamics ✓
- [x] Tidal constituent forcing (M2, S2, K1, O1) — *existing from Phase 1*
- [x] Tidal potential body force (`TidalPotential`, `TidalPotentialConstituent`)
- [x] Harmonic analysis of results (`HarmonicAnalysis`, `TimeSeries`, `ComparisonMetrics`)

### 3.2 Fjord-Specific ✓
- [x] Narrow strait treatment (`StraitFriction2D` with width-dependent enhancement)
- [x] Sill overflow dynamics (`SillOverflow2D`, `SillWithFriction`, gradient-based form drag)
- [x] River discharge boundaries (`MultiBoundaryCondition2D` dispatch, `Discharge2D` + `BoundaryTag::River`)
- [x] `Bathymetry2D` with nodal values and gradients for well-balanced schemes
- [x] `BathymetrySource2D` source term (S = -gh∇B)
- [x] `ManningFriction2D`, `ChezyFriction2D`, `SpatiallyVaryingManning2D`
- [x] Bathymetry profiles: `gaussian_bump`, `sill`, `strait`, `parabolic_bowl`

### 3.3 Open Boundaries ✓
- [x] Flather radiation condition — *existing from Phase 2*
- [x] Chapman condition for sea surface height (`Chapman2D`, `ChapmanFlather2D`)
- [x] TST-OBC (Tidal and Subtidal) decomposition (`TSTOBC2D`, `TSTConfig`, `TSTConstituent`)
- [x] Sponge layers / relaxation zones (`SpongeLayer2D`, `SpongeProfile`)
- [x] Nesting capability for regional models (`NestingBC2D`, Dirichlet and Flather modes)
- [x] I/O utilities (`read_constituent_file`, `read_timeseries_file`, `BoundaryTimeSeries`)
- [ ] TPXO tidal forcing integration — *deferred (using generic constituent files instead)*

### 3.4 Validation
- [x] Analytical test cases (dam break, geostrophic balance, standing wave, sponge absorption)
- [x] Comparison with tide gauge data (NEW: 2026-01-10)
  - `TideGaugeStation` with metadata (name, location, datum offset)
  - `StationValidationResult` with metrics (RMSE, bias, correlation, skill score)
  - `ValidationSummary` for multi-station assessment
  - `HarmonicAnalysis` integration for constituent comparison
  - Norwegian stations predefined: Bergen, Stavanger, Trondheim, Kristiansund, etc.
  - File I/O: `read_tide_gauge_file()`, `write_tide_gauge_file()`
  - Files: `src/analysis/tide_gauge.rs`, `src/io/tide_gauge_reader.rs`
  - Example: `examples/tide_gauge_validation.rs` - 24-hour simulation with validation
  - 15 new unit tests + 9 integration tests
- [x] ADCP current validation (NEW: 2026-01-12)
  - `ADCPStation` with metadata (name, location, depth, water_depth)
  - `CurrentTimeSeries` for (time, u, v) velocity data
  - `CurrentPoint` for individual (u, v) measurements
  - `CurrentValidationMetrics` with u/v/speed RMSE, bias, correlation, vector correlation, direction RMSE
  - `ADCPValidationResult` for single station validation
  - `ADCPValidationSummary` for multi-station assessment
  - Norwegian stations predefined: Froya, Hitra, Smola, Bergen
  - File I/O: `read_adcp_file()`, `write_adcp_file()`, supports text and CSV formats
  - Files: `src/analysis/adcp.rs`, `src/io/adcp_reader.rs`
  - 13 new unit tests

### 3.5 Source Term Infrastructure ✓
- [x] `SourceTerm2D` trait for extensible physics
- [x] `SourceContext2D` with position, state, bathymetry, gradients
- [x] `CombinedSource2D` for composing multiple sources
- [x] `CoriolisSource2D` (f-plane, β-plane, Norwegian coast config)
- [x] Integration into `SWE2DRhsConfig` via `source_terms` field

---

## RESOLVED: Tidal Boundary Condition Issues (Fixed 2026-01-12)

**Status: FIXED** ✓

### Original Issue (Discovered 2026-01-10)

Testing tide gauge validation revealed fundamental issues with tidal boundary conditions:

1. **Bathymetry convention mismatch** - Flather BCs expect surface elevation η = h + B, but without explicit bathymetry B defaults to 0, causing η = h instead of η ≈ 0
2. **Closed basin instability** - `HarmonicFlather2D` with reflective walls creates resonance that amplifies to blow-up
3. **Phase errors in simple domains** - Even stable configurations show large phase mismatches due to wave propagation in idealized geometry

### Root Causes Identified

1. **Bathymetry Convention**: Without explicit bathymetry (B = 0), Flather computes spurious ~22 m/s velocities
2. **Velocity Feedback Resonance**: Flather BC amplifies reflected waves in closed basins
3. **Missing Wave Absorption**: Dirichlet BCs don't absorb outgoing waves → phase errors

### Fixes Implemented

1. **[FIXED] Bathymetry validation for all Flather BCs**
   - Created `src/boundary/bathymetry_validation.rs` with shared validation logic
   - `validate_bathymetry_convention()` detects misconfigured bathymetry
   - `warn_once_if_misconfigured()` emits one-time warning per BC type
   - Added to: `Flather2D`, `HarmonicFlather2D`, `ChapmanFlather2D`, `NestingBC2D`
   - 4 unit tests

2. **[FIXED] Stability monitoring infrastructure**
   - Created `src/analysis/stability.rs` with `StabilityMonitor`
   - `StabilityThresholds` with presets: `tidal_default()`, `strict()`, `relaxed()`
   - `StabilityWarning` enum for different issue types
   - `should_stop()` and `suggest_remediation()` for actionable guidance
   - 5 unit tests

3. **[FIXED] Tidal simulation builder with presets**
   - Created `src/boundary/tidal_config.rs` with `TidalSimulationBuilder`
   - Presets: `closed_basin_stable()`, `open_ocean()`, `fjord_mouth()`
   - `SpongeConfig` for easy sponge layer configuration
   - `build_bc()` and `build_sponge()` methods
   - 7 unit tests

4. **[FIXED] BC selection documentation**
   - Added comprehensive guide to `src/boundary/mod.rs` module docs
   - BC selection table by scenario
   - Bathymetry convention requirements with examples
   - Sponge layer recommendations for closed basins

### Files Added
- `src/boundary/bathymetry_validation.rs` - Shared validation logic (4 tests)
- `src/analysis/stability.rs` - Stability monitoring (5 tests)
- `src/boundary/tidal_config.rs` - Tidal simulation builder (7 tests)

### Files Modified
- `src/boundary/boundary_2d.rs` - Validation in `Flather2D`, `HarmonicFlather2D`, added `with_h_min()` to `HarmonicTidal2D`
- `src/boundary/chapman.rs` - Validation in `ChapmanFlather2D`
- `src/boundary/nesting_bc.rs` - Validation in `NestingBC2D` (Flather mode)
- `src/boundary/mod.rs` - BC selection guide, exports
- `src/analysis/mod.rs` - Export stability types
- `src/lib.rs` - Re-export new public types

### Recommended Usage

```rust
// For closed/semi-closed basins (fjords, bays):
let builder = TidalSimulationBuilder::closed_basin_stable(0.5, 50.0, 5000.0);
let bc = builder.build_bc();  // Uses HarmonicTidal2D (no velocity feedback)
let sponge = builder.build_sponge((x_min, x_max), (y_min, y_max));

// For open ocean boundaries:
let builder = TidalSimulationBuilder::open_ocean(0.5, 50.0);
let bc = builder.build_bc();  // Uses HarmonicFlather2D (best absorption)

// Stability monitoring:
let mut monitor = StabilityMonitor::new(StabilityThresholds::tidal_default());
let status = monitor.check(&q, dt);
if monitor.should_stop() {
    for suggestion in monitor.suggest_remediation() {
        eprintln!("  * {}", suggestion);
    }
}
```

---

## RESOLVED: Real Data Simulation Stability (Fixed 2026-01-12)

**Status: FIXED** ✓

### Issue Summary

Testing with real Norwegian bathymetry (Froya-Smola-Hitra region) revealed that:
1. **Data loading works correctly** - GeoTIFF bathymetry, GSHHS coastline, land masking all functional
2. **DG without limiters is unstable** - Steep bathymetry gradients (up to 1.5 slope) cause oscillations → blow-up
3. **Coupled simulation framework works** - `norwegian_fjord.rs` with Kuzmin limiters is stable
4. **Pure SWE needs limiter infrastructure** - No easy way to apply limiters without tracer coupling

### Files Added (Working)
- `src/io/projection.rs` - LocalProjection, UtmProjection for coordinate transforms
- `src/io/geotiff.rs` - GeoTIFF bathymetry reader (ported from aqc-h3o)
- `src/io/coastline.rs` - GSHHS shapefile reader (ported from aqc-h3o)
- `src/mesh/land_mask.rs` - LandMask2D for wet/dry classification
- `examples/froya_real_data.rs` - Real data example (minimal config stable)

### Required Fixes

1. **[FIXED] Unrealistic velocities caused by non-well-balanced scheme** ✓
   - **Root Cause Identified**: DG flux and source term use different discretizations
   - Pressure gradient `∇(gh²/2)` and bathymetry source `-gh∇B` don't cancel numerically
   - With FLAT bathymetry: velocities are realistic (0.1-0.3 m/s) ✓
   - With REAL bathymetry (0.66 gradient): velocities explode to 38 m/s ✗
   - **Solution Implemented**: 2D Hydrostatic reconstruction (Audusse et al. 2004)
   - Implementation:
     - [x] Ported `HydrostaticReconstruction` from 1D to 2D (`src/source/hydrostatic_reconstruction_2d.rs`)
     - [x] Modified `compute_rhs_swe_2d` with `well_balanced` option for reconstructed interface states
     - [x] Lake-at-rest preserved to machine precision (4 new tests)
   - **Usage**: `SWE2DRhsConfig::new(&eq, &bc).with_bathymetry(&bathy).with_well_balanced(true).with_source_terms(&BathymetrySource2D::new(G))`

2. **[DONE] Add SWE-only limiter infrastructure** ✓
   - Created `ssp_rk3_swe_2d_step_limited()` with TVB/Kuzmin/positivity limiters
   - Ported limiter application from coupled framework to standalone SWE
   - Added `SWE2DTimeConfig` with `SWELimiterType` enum (None, Tvb, Kuzmin, PositivityOnly)
   - Files: `src/solver/limiters_swe_2d.rs`, `src/time/ssp_rk3_swe_2d.rs`
   - Result: **Full tidal cycle (12.4 hours, 72,124 steps) stable with real bathymetry!**
   - Fixed Flather BC `h_ref` parameter for correct water depth matching

2. **[DONE] Well-balanced scheme for steep bathymetry** ✓
   - Implemented `HydrostaticReconstruction2D` (Audusse et al. 2004)
   - `SWE2DRhsConfig::with_well_balanced(true)` enables reconstruction at interfaces
   - Lake-at-rest verified with steep (0.66) and bilinear bathymetry gradients
   - Works with all flux types (Roe, HLL, Rusanov)
   - **CRITICAL**: For real bathymetry data, use `bathymetry.to_cell_average()` to ensure
     constant bathymetry within elements. Non-linear bathymetry breaks well-balanced property.
   - Without bathymetry source term when using cell-average (gradients are zero)
   - Result: Realistic velocities (~2.8 m/s) instead of spurious 38+ m/s

3. **[DONE] Cross-element bathymetry smoothing** ✓
   - Added `smooth_cross_element()` method with neighbor-averaging
   - Uses mesh edge connectivity to find face-adjacent elements
   - Configurable smoothing strength (alpha) and iterations
   - Optional gradient threshold to only smooth steep regions
   - Recommended max gradient: 0.3-0.5 for stability with well-balanced schemes
   - 6 new unit tests for smoothing correctness
   - Files: `src/mesh/bathymetry_2d.rs`

4. **[DONE] Improved wetting/drying** ✓
   - Implemented `WetDryConfig` with thin-layer blending (smooth Hermite interpolation)
   - Velocity capping (default 20 m/s) prevents unrealistic velocities
   - Smooth momentum damping in shallow areas
   - Integrated into `SWE2DTimeConfig` via `.with_wet_dry_treatment()`
   - Result: Max velocity reduced from 2.79 m/s to 1.12 m/s with real bathymetry

5. **[DONE] Tidal BC ramp-up period** ✓
   - Added `with_ramp_up(duration)` to `TidalBC`, `HarmonicFlather2D`, `HarmonicTidal2D`
   - Smooth Hermite interpolation: 3t² - 2t³ for continuous entry/exit
   - Mean elevation NOT ramped (starts at full value)
   - Typical ramp-up: 1-3 tidal periods (1 hour default in Froya example)
   - Prevents initial impulse from tidal forcing
   - 5 new unit tests for ramp functionality

### Workarounds (Current)
- Use `norwegian_fjord.rs` pattern with dummy tracers for limiter access
- Reduce polynomial order (N=2 instead of N=3)
- Use Rusanov flux (more diffusive than Roe)
- Increase H_MIN to 5m for stability
- Disable source terms for minimal stable runs

---

## Phase 4: Performance & Scalability

Production-ready performance for operational use.

### 4.1 CPU Optimization
- [x] Rayon parallelization for 2D RHS (`compute_rhs_swe_2d_parallel`, `compute_rhs_tracer_2d_parallel`)
- [x] `par_chunks_mut` pattern for efficient parallel writes (avoids allocation overhead)
- [x] Automatic serial/parallel selection based on mesh size (threshold: 1000 elements)
- [x] Profile hot paths with flamegraph (see `scripts/flamegraph.sh`)
- [x] Profiling infrastructure: `scripts/flamegraph.sh`, `scripts/samply.sh`, `scripts/perf-stat.sh`
- [x] `#[inline(always)]` on critical accessors (`get_state`, `set_state`, `SWEState2D` methods)
- [x] Replaced `powf(1.0/3.0)` with `cbrt()` in Manning friction (-22% samples)
- [x] Fused wetting/drying passes with early-exit for wet/dry elements (-46% to -100%)
- [x] Optimized hydrostatic reconstruction with ratio-based velocity preservation (-70%)
- [x] Direct data access in cell averages and limiters (-60% state accessor overhead)
- [ ] Cache-friendly data layout (SoA vs AoS)
- [ ] Load balancing for parallel mesh

### 4.2 SIMD Optimization ✓
- [x] Inline attributes enable auto-vectorization for simple loops
- [x] Precomputed inverse values avoid repeated divisions
- [x] **Element kernel vectorization** - DG operators (Dr, Ds multiplication) via faer GEMV
- [x] SIMD kernels with `pulp` crate for portable SIMD (runtime feature detection)
- [x] `compute_rhs_swe_2d_simd` - SIMD-optimized RHS function with SoA data layout
- [x] Volume term: `apply_diff_matrix` using faer GEMV (1.16-1.81x speedup)
- [x] Chain rule: `combine_derivatives` with pulp SIMD intrinsics
- [x] Source terms: `coriolis_source` vectorized (1.4x speedup)
- [x] LIFT application: row-major cache for efficient access
- [x] Row-major matrix caches in `DGOperators2D` (dr_row_major, ds_row_major, lift_row_major)
- [x] Benchmark suite in `benches/rhs_swe_2d_bench.rs` with criterion
- [ ] Vectorize flux computation (4-8 edges at a time) - *deferred (complex branching)*
- [ ] Further SIMD for Manning friction (cbrt is scalar)

### 4.3 GPU Support
- [ ] Port reference operators to CUDA (cudarc)
- [ ] GPU-resident solution storage
- [ ] Batched element operations
- [ ] Async CPU-GPU data transfer
- [ ] Multi-GPU support

### 4.4 I/O & Monitoring
- [x] NetCDF output (CF-conventions for oceanography) - `NetCDFWriter`, `ForcingReader`
- [ ] HDF5 checkpointing via hdf5-metno (Norwegian Met Institute fork)
- [ ] Restart file capability
- [x] Runtime diagnostics (CFL, conservation, energy) — *`SWEDiagnostics2D`, `DiagnosticsTracker`*
- [x] Progress reporting for long runs — *`ProgressReporter` with ETA, steps/sec*
- [x] VTK output for visualization (`write_vtk_swe`, `write_vtk_coupled`, `write_vtk_series`)

---

## Phase 5: Future Extensions

Beyond MVP, for advanced applications.

### 5.1 3D Capability
- [ ] Generalized σ/s-coordinates (35-40 levels, clustered at surface/bottom)
- [ ] Baroclinic pressure gradient (balanced methods for steep bathymetry)
- [ ] Mode splitting for barotropic-baroclinic separation
- [ ] Vertical mixing (turbulence closure, k-ε or GLS)
- [ ] Hybridized DG (HDG) for implicit vertical solves (Kang et al. 2019)

### 5.2 Additional Physics
- [x] Wind stress forcing (`WindStress2D`, `DragCoefficient`)
- [x] Atmospheric pressure forcing (`AtmosphericPressure2D`, `P_STANDARD`)
- [ ] Wave-current interaction
- [ ] Sediment transport

### 5.3 Data Assimilation
- [ ] Adjoint model capability
- [ ] EnKF integration hooks

---

## Code Quality (Ongoing)

### Testing
- [x] Add criterion benchmarks — *6 benchmark suites: SIMD kernels, RHS, flux, limiters, time stepping, sources*
- [ ] Property-based tests with proptest
- [ ] Manufactured solution tests
- [ ] CI/CD pipeline

### Documentation
- [ ] Rustdoc examples for public APIs
- [ ] Mathematical formulation document
- [ ] User guide with examples

### Refactoring
- [x] Error handling (thiserror instead of panics) — *8 error types migrated*
- [ ] Abstract mesh trait for 1D/2D/3D
- [ ] Plugin architecture for physics modules

### API Usability Issues (Discovered 2026-01-12)

While writing criterion benchmarks, several API inconsistencies were found:

1. **Bathymetry constructors inconsistent with mesh API**
   - `Bathymetry2D::flat(n_elements, n_nodes)` takes raw counts
   - `Bathymetry2D::from_function(mesh, ops, geom, f)` takes structured types
   - Consider: `Bathymetry2D::flat(mesh, ops)` or `Bathymetry2D::constant(mesh, ops, value)`

2. **Source term API requires manual context construction**
   - Users must construct `SourceContext2D` manually with 7 parameters
   - Consider: Add convenience methods that take `(q, mesh, ops, bathy, k, i)` directly

3. **KuzminParameter2D lacks `::new()` constructor**
   - Only `::strict()` and `::relaxed(f64)` available
   - Fine if intentional (forcing explicit choice), but inconsistent with `TVBParameter2D::new()`

4. **Flux functions require tuple for normal vector**
   - `roe_flux_swe_2d(left, right, (nx, ny), g, h_min)` uses tuple
   - Reasonable, but parameter order `(q_l, q_r, normal, g, h_min)` differs from signature readability

5. **Multiple time stepping APIs**
   - `ssp_rk3_step_2d` (generic, takes closure) - for DGSolution2D
   - `ssp_rk3_swe_2d_step_limited` (SWE-specific, takes config) - for SWESolution2D
   - Tests define their own `ssp_rk3_swe_step` helper - suggests missing mid-level API

**Benchmark Files Added** (2026-01-12):
- `benches/rhs_computation_bench.rs` - RHS at various mesh sizes/orders
- `benches/flux_bench.rs` - Numerical flux comparisons (Roe/HLL/Rusanov)
- `benches/limiter_bench.rs` - TVB/Kuzmin/positivity limiters
- `benches/time_stepping_bench.rs` - SSP-RK3 time integration
- `benches/source_term_bench.rs` - Coriolis/friction/bathymetry sources

---

## Completed ✓

### SIMD Optimization for 2D SWE RHS (2026-01-10)

**Problem**: The RHS computation for 2D shallow water equations was CPU-bound, with the volume term (Dr/Ds matrix multiplication) being the primary bottleneck.

**Solution**: Implemented SIMD-optimized kernels using `pulp` crate for portable SIMD and `faer` for optimized GEMV operations.

**Files Added/Modified**:
- `src/solver/simd_kernels.rs` (NEW) - SIMD kernel implementations
- `src/solver/simd_swe_2d.rs` (NEW) - SoA data structures and workspace
- `src/solver/rhs_swe_2d.rs` - Added `compute_rhs_swe_2d_simd` function
- `src/operators/operators_2d.rs` - Added row-major matrix caches
- `benches/rhs_swe_2d_bench.rs` (NEW) - Criterion benchmarks
- `Cargo.toml` - Added `pulp` dependency, `simd` feature

**Key Components**:
- **`apply_diff_matrix`**: Uses faer GEMV for matrices ≥20 nodes, scalar for smaller
- **`combine_derivatives`**: Pulp SIMD for chain rule combination with geometric factors
- **`coriolis_source`**: Vectorized Coriolis source term evaluation
- **`apply_lift`**: Scalar (LIFT matrices too small for SIMD overhead)
- **Row-major caches**: `dr_row_major`, `ds_row_major`, `lift_row_major` in DGOperators2D

**Benchmark Results** (criterion, release build):

| Kernel | P2 (9 nodes) | P4 (25 nodes) | P5 (36 nodes) |
|--------|--------------|---------------|---------------|
| diff_matrix scalar | 54 ns | 296 ns | 646 ns |
| diff_matrix SIMD | 54 ns | 254 ns | 357 ns |
| **Speedup** | 1.0x | **1.16x** | **1.81x** |
| coriolis scalar | 4 ns | 10 ns | 13 ns |
| coriolis SIMD | 4 ns | 7 ns | 9.3 ns |
| **Speedup** | 1.0x | **1.4x** | **1.4x** |

**Design Decisions**:
- Size-based dispatch: faer GEMV has ~150ns overhead, only beneficial for P4+ (≥20 nodes)
- SoA (Structure of Arrays) layout for SIMD-friendly memory access
- Row-major matrix caches computed once at DGOperators2D construction
- LIFT stays scalar (n_face_nodes = 3-6, too small for SIMD benefit)

**Usage**:
```rust
// Enable with: cargo run --release --features simd

#[cfg(feature = "simd")]
use dg_rs::compute_rhs_swe_2d_simd;

// Automatically used in examples when simd feature is enabled
let rhs = compute_rhs_swe_2d_simd(&q, &mesh, &ops, &geom, &config, time);
```

**Tests Added**:
- `test_simd_matches_scalar` - Verifies SIMD output matches scalar to 1e-10
- `test_simd_with_coriolis` - Tests SIMD Coriolis source term
- 9 kernel unit tests comparing scalar and SIMD implementations

---

### Performance Profiling & Optimization (2026-01-09)

**Problem**: Real-data simulations (Froya-Smola-Hitra) were CPU-bound with no visibility into hotspots.

**Solution**: Added profiling infrastructure and optimized key hotspots identified via flamegraph analysis.

**Profiling Tools Added** (`scripts/`):
- `flamegraph.sh` - Generate SVG flamegraphs using cargo-flamegraph (WSL2 perf workaround included)
- `samply.sh` - Interactive profiler with Firefox Profiler UI
- `perf-stat.sh` - Quick CPU statistics (IPC, cache, branches)

**Cargo.toml Changes**:
- Added `[profile.profiling]` inheriting from release with `debug = true`
- Release profile now includes `debug = 1` (line tables for profiling)

**Optimizations Implemented** (measured improvements):

| Function | Change | Improvement |
|----------|--------|-------------|
| `ManningFriction2D::friction_coefficient` | `powf(1.0/3.0)` → `cbrt()` | -22% |
| `ManningFriction2D::explicit_source` | Precompute `h_inv`, avoid redundant ops | -19% |
| `apply_wet_dry_correction_all` | Fused passes, early-exit for wet/dry | -46% |
| `WetDryConfig::damp_momentum` | Inlined into fused loop | -100% |
| `WetDryConfig::apply_velocity_cap` | Fast path for fully-wet elements | -79% |
| `WetDryConfig::compute_velocity` | Inlined into fused loop | -100% |
| `HydrostaticReconstruction2D::reconstruct` | Ratio-based velocity, `#[inline]` | -70% |
| `SystemSolution2D::get/set_state` | `#[inline(always)]`, direct array access | -60% |
| `swe_cell_averages_2d` | Direct `element_data()` access, precomputed inverse | ~0% |

**Files Modified**:
- `src/source/friction_2d.rs` - `cbrt()`, `#[inline]` on hot functions
- `src/solver/state_2d.rs` - `#[inline(always)]` on all accessors, added `element_data()` batch accessor
- `src/solver/wetting_drying.rs` - Fused `apply_wet_dry_correction_all`, added `blending_factor_fast()`
- `src/source/hydrostatic_reconstruction_2d.rs` - `#[inline]`, ratio-based reconstruction, `reconstruct_wet()` fast path
- `src/solver/limiters_swe_2d.rs` - Direct data access in `swe_cell_averages_2d`
- `Cargo.toml` - Profiling profile configuration

**Usage**:
```bash
# Generate flamegraph
./scripts/flamegraph.sh froya_real_data

# Interactive profiler
./scripts/samply.sh froya_real_data

# Quick CPU stats
./scripts/perf-stat.sh froya_real_data
```

**Output**: `output/flamegraph/*.svg` - Before/after comparison available

---

### 2D Hydrostatic Reconstruction for Well-Balanced SWE (2026-01-09)

**Problem**: Steep Norwegian bathymetry (gradient ~0.66) caused spurious 38 m/s velocities for lake-at-rest conditions because DG pressure gradient `∇(gh²/2)` and bathymetry source term `-gh∇B` use different discretizations.

**Solution**: Two-part approach:

1. **Hydrostatic reconstruction** (Audusse et al. 2004) for 2D:
   - At each face, reconstruct states relative to `B* = max(B_L, B_R)`
   - `h_L* = max(0, η_L - B*)`, `h_R* = max(0, η_R - B*)` where `η = h + B`
   - For lake-at-rest (η = const), both sides get same depth → zero flux

2. **Cell-average bathymetry** for real data:
   - For non-linear bathymetry from GeoTIFF, use `bathymetry.to_cell_average()` to make
     bathymetry constant within each element (zero gradients)
   - All bathymetry effects handled by interface reconstruction
   - No `BathymetrySource2D` needed when gradients are zero

**Files Added/Modified**:
- `src/source/hydrostatic_reconstruction_2d.rs` (NEW) - `HydrostaticReconstruction2D` struct
- `src/mesh/bathymetry_2d.rs` - Added `linearize()` and `to_cell_average()` methods
- `src/solver/rhs_swe_2d.rs` - Added `well_balanced` field to `SWE2DRhsConfig`
- `src/source/mod.rs`, `src/lib.rs` - Export new module

**Tests Added** (4 new):
- `test_well_balanced_steep_slope` - 0.66 gradient, RHS < 1e-11
- `test_well_balanced_bilinear_slope` - Non-axis-aligned gradient
- `test_well_balanced_all_flux_types` - Roe, HLL, Rusanov all give zero RHS
- `test_source_term_alone_well_balanced_linear` - Standard DG is well-balanced for linear bathymetry

**Usage with real bathymetry**:
```rust
// Load and convert to cell-average
let mut bathymetry = Bathymetry2D::from_geotiff(&mesh, &ops, &geom, &geotiff, &proj);
bathymetry.to_cell_average();  // CRITICAL for well-balanced

// RHS config with reconstruction, no BathymetrySource2D
let config = SWE2DRhsConfig::new(&eq, &bc)
    .with_bathymetry(&bathy)
    .with_well_balanced(true)
    .with_source_terms(&coriolis_friction);  // No bathymetry source
```

**Result**: Froya real data simulation now produces realistic velocities (~2.8 m/s) instead of spurious 38+ m/s.

### Improved Wetting/Drying Treatment (2026-01-09)

**Problem**: Basic H_MIN cutoff caused sharp transitions and unrealistic velocities at wet/dry fronts.

**Solution**: Implemented `WetDryConfig` in `src/solver/wetting_drying.rs` with:
- **Thin-layer blending**: Smooth Hermite interpolation from dry (h=h_min) to wet (h=h_thin=10×h_min)
- **Velocity capping**: Maximum velocity limit (default 20 m/s) based on shallow water physics
- **Momentum damping**: Gradual momentum reduction in shallow areas, zero at dry
- **Desingularized velocity**: u = 2h·hu / (h² + h_reg²) for numerical stability

**Files Added**:
- `src/solver/wetting_drying.rs` (NEW) - WetDryConfig, apply_wet_dry_correction(), apply_wet_dry_correction_all()

**Integration**:
- Added `.with_wet_dry_treatment()` to `SWE2DTimeConfig`
- Applied automatically after each RK stage in `apply_configured_limiter()`
- 6 unit tests for blending, velocity cap, thin-layer damping, flux factors

**Result**: Max velocity reduced from 2.79 m/s to 1.12 m/s with real bathymetry data.

### Tidal BC Ramp-Up Period (2026-01-09)

**Problem**: Sudden activation of tidal forcing at t=0 caused initial impulse and numerical oscillations.

**Solution**: Added smooth ramp-up to all tidal boundary conditions:
- `TidalBC.with_ramp_up(duration)` for 1D BCs
- `HarmonicFlather2D.with_ramp_up(duration)` for 2D Flather BCs
- `HarmonicTidal2D.with_ramp_up(duration)` for 2D Dirichlet BCs

**Mathematical formulation**:
- Ramp factor R(t) = 3τ² - 2τ³ where τ = t/duration (smooth Hermite interpolation)
- R(0) = 0, R(duration) = 1, R'(0) = R'(duration) = 0 (smooth entry/exit)
- Elevation: η(t) = η₀ + R(t) × Σᵢ Aᵢ cos(ωᵢ t + φᵢ)
- Mean elevation η₀ NOT ramped (physical initial state preserved)

**Files Modified**:
- `src/boundary/tidal.rs` - Added ramp_duration, ramp_factor(), ramp_derivative()
- `src/boundary/boundary_2d.rs` - Added ramp to HarmonicFlather2D, HarmonicTidal2D
- `examples/froya_real_data.rs` - Updated to use 1-hour ramp-up

**Tests Added**: 5 new tests for ramp_factor, ramp_factor_no_ramp, elevation_with_ramp, ramp_smooth_derivative

**Usage**:
```rust
let tidal_bc = HarmonicFlather2D::m2_only(0.8, 0.0, 0.0)
    .with_ramp_up(3600.0);  // 1-hour ramp-up
```

### Atmospheric Pressure Forcing (2026-01-09)

**Problem**: Storm surge modeling requires atmospheric pressure gradient forcing to drive currents and account for the inverse barometer effect.

**Solution**: Implemented `AtmosphericPressure2D` source term with multiple pressure field types:
- Constant pressure (no forcing)
- Uniform gradient (steady gradient in x/y directions)
- Time-varying gradient (rotating/oscillating gradients)
- Spatio-temporal field (arbitrary P(x,y,t) with numerical gradient)
- Moving storm (Holland 1980 model)

**Physics**:
```text
S_h  = 0
S_hu = -h/ρ * ∂P/∂x
S_hv = -h/ρ * ∂P/∂y
```

Inverse barometer effect: Δη ≈ -ΔP / (ρg) ≈ -1 cm per hPa

**Features**:
- Multiple drag coefficient formulations (Large-Pond, Wu, Smith, Yelland-Taylor)
- Presets: `norwegian_winter_storm()`, `severe_storm()`
- Direction from meteorological convention (0°=N, 90°=E)
- Inverse barometer calculation for initial/boundary conditions

**Files Added/Modified**:
- `src/source/atmospheric_pressure.rs` (NEW) - AtmosphericPressure2D, P_STANDARD, RHO_WATER_PRESSURE
- `src/source/mod.rs` - Export new module
- `src/lib.rs` - Re-export types
- `examples/froya_real_data.rs` - Added pressure forcing option

**Tests Added**: 14 unit tests covering all pressure field types, Holland storm model, inverse barometer

**Usage**:
```rust
// Uniform gradient from southwest
let pressure = AtmosphericPressure2D::from_direction(1.5e-3, 225.0);

// Moving storm
let storm = AtmosphericPressure2D::norwegian_winter_storm(x0, y0);

// Add to combined sources
let sources = CombinedSource2D::new(vec![&coriolis, &friction, &wind, &pressure]);
```

### NetCDF I/O for Oceanographic Data (2026-01-09)

**Problem**: Need CF-conventions compliant output for integration with oceanographic tools and ERA5/ECMWF forcing data input.

**Solution**: Implemented optional `netcdf` feature with:
- `NetCDFWriter` - CF-1.8 compliant output with standard names (sea_surface_height, eastward_sea_water_velocity, etc.)
- `NetCDFWriterConfig` - Builder pattern for configuration (title, institution, compression)
- `NetCDFMeshInfo` - Mesh coordinate storage with optional lat/lon
- `ForcingReader` - ERA5/ECMWF forcing data reader with scale/offset unpacking
- Fill value handling and bilinear interpolation

**Files Added**:
- `src/io/netcdf_io.rs` (NEW) - NetCDF writer and forcing reader
- Modified `src/io/mod.rs` - Conditional exports
- Modified `src/lib.rs` - Re-exports with feature gate
- Modified `Cargo.toml` - Optional netcdf and chrono dependencies
- Modified `examples/froya_real_data.rs` - Optional NetCDF output

**Usage**:
```rust
// Enable with: cargo run --features netcdf

#[cfg(feature = "netcdf")]
{
    let config = NetCDFWriterConfig::new("output.nc")
        .with_title("Simulation")
        .with_institution("NTNU");
    let mut writer = NetCDFWriter::create(config, &mesh_info)?;
    writer.write_timestep(time, &h, &eta, Some(&u), Some(&v))?;
}
```

**Tests**: 701 tests passing with `--features netcdf`

### VTK Output & Parallel RHS (2026-01-09)

**VTK Output for ParaView Visualization**
- [x] `src/io/vtk.rs` - VTU (XML UnstructuredGrid) writer
- [x] Sub-cell decomposition for high-order DG (P3 → 9 sub-quads per element)
- [x] `write_vtk_swe()` - SWE solution with h, velocity magnitude, Froude number
- [x] `write_vtk_coupled()` - Coupled SWE + tracer with temperature, salinity
- [x] `write_vtk_series()` - Generate `.pvd` file for time series animation
- [x] Bathymetry field included when provided
- [x] TimeValue in FieldData for ParaView time handling

**Parallel 2D RHS Computation**
- [x] `compute_rhs_swe_2d_parallel()` - Rayon-parallelized SWE RHS
- [x] `compute_rhs_tracer_2d_parallel()` - Rayon-parallelized tracer RHS
- [x] `TracerSolution2D::from_data()` - Constructor for parallel results
- [x] `par_chunks_mut` pattern for efficient parallel writes (avoids `flat_map` allocation overhead)
- [x] Automatic serial/parallel selection in examples (threshold: 1000 elements)
- [x] Speedup: ~2.4x SWE, ~1.9x tracer at 5000 elements
- [x] Tests: `test_parallel_matches_serial`, `test_parallel_tracer_matches_serial`

**Files Added/Modified**
- `src/io/vtk.rs` (NEW) - VTK writer with sub-cell decomposition
- `src/io/mod.rs` - Export VTK functions
- `src/lib.rs` - Re-export VTK and parallel functions
- `src/solver/rhs_swe_2d.rs` - Added parallel version
- `src/solver/rhs_tracer_2d.rs` - Added parallel version
- `src/solver/tracer_state.rs` - Added `from_data()`
- `src/solver/mod.rs` - Export parallel functions
- `examples/norwegian_fjord.rs` - VTK output, auto serial/parallel selection

### Tracer Transport Stability (2026-01-09)
- [x] Fixed Laplacian computation in `rhs_tracer_2d.rs` (correct chain rule for second derivatives)
- [x] `limiters_tracer_2d.rs` module with TVB and Zhang-Shu positivity limiters
- [x] `TracerBounds` for physical tracer limits (T: [-2, 40]°C, S: [0, 42] PSU)
- [x] `TVBParameter2D` with domain-size normalization for consistent threshold scaling
- [x] `tracer_cell_averages_2d()` mass-weighted averaging
- [x] `tracer_positivity_limiter_2d()` θ-scaling for bounds enforcement
- [x] `tracer_tvb_limiter_2d()` slope limiting with neighbor comparison
- [x] `apply_tracer_limiters_2d()` combined limiter application
- [x] `ssp_rk3_coupled_step_limited()` RK3 with limiter after each stage
- [x] `run_coupled_simulation_limited()` simulation driver with limiters
- [x] `element_diameter()` method in Mesh2D for TVB threshold
- [x] 9 unit tests for limiter correctness
- [x] Norwegian fjord example updated with proper limiter configuration
- [x] Integration test suite `tests/tracer_2d_test.rs` with 5 tests:
  - `test_tracer_conservation_periodic` - Conservation with periodic BC
  - `test_limiter_prevents_oscillations` - Regression test for limiter fix
  - `test_limiter_preserves_cell_average` - Cell average preservation
  - `test_coupled_simulation_stability_with_limiters` - Full simulation stability
  - `test_no_limiter_produces_oscillations` - Verify limiters are necessary
- [x] Kuzmin vertex-based slope limiter (`tracer_kuzmin_limiter_2d`, `KuzminParameter2D`)
- [x] `apply_tracer_limiters_kuzmin_2d()` combined Kuzmin + positivity limiter
- [x] 680+ total tests passing

### Phase 3.2: Fjord-Specific Features
- [x] `Bathymetry2D` with nodal values and gradient computation (Dr, Ds transform)
- [x] `BathymetrySource2D` well-balanced source term (S = -gh∇B)
- [x] `ManningFriction2D` 2D bottom friction with semi-implicit update
- [x] `ChezyFriction2D` and `SpatiallyVaryingManning2D` variants
- [x] `StraitFriction2D` enhanced friction for narrow passages (width-dependent)
- [x] `SillOverflow2D` form drag parameterization (gradient threshold-based)
- [x] `SillWithFriction` combined sill + Manning friction
- [x] `SillDetector` utility for sill region identification
- [x] `MultiBoundaryCondition2D` dispatcher for tag-based BC routing
- [x] `BCContext2D` extended with `boundary_tag` field
- [x] RHS integration: bathymetry values/gradients and boundary tags wired through
- [x] Norwegian fjord presets: `norwegian_strait()`, `norwegian_sill()`, `fjord()`
- [x] Bathymetry profiles: `gaussian_bump`, `sill`, `strait`, `parabolic_bowl`
- [x] 475+ total tests passing

### Phase 3: Norwegian Coast Features (Partial)
- [x] `SourceTerm2D` trait system for extensible 2D physics
- [x] `SourceContext2D` with full state context for source evaluation
- [x] `CombinedSource2D` for composing multiple source terms
- [x] `CoriolisSource2D` (f-plane, β-plane, Norwegian coast config)
- [x] `TidalPotential` body force with Love number correction
- [x] `TidalPotentialConstituent` for M2, S2, K1, O1, N2, P1
- [x] `Chapman2D` radiation BC for sea surface height
- [x] `ChapmanFlather2D` combined elevation/velocity BC
- [x] `SpongeLayer2D` with profiles (Linear, Quadratic, Cosine, Exponential)
- [x] `rectangular_sponge_fn` helper for rectangular domains
- [x] Integration of `source_terms` into `SWE2DRhsConfig`
- [x] Validation tests: circular dam break, standing wave, geostrophic balance, sponge absorption
- [x] `HarmonicAnalysis` for tidal time series decomposition (least-squares fitting)
- [x] `TimeSeries` type with location/name metadata
- [x] `ComparisonMetrics` for model-observation validation (RMSE, correlation, skill score)
- [x] `ConstituentComparison` for amplitude/phase error analysis
- [x] Standard and Norwegian coast constituent sets (M2, S2, N2, K1, O1, P1)
- [x] Rayleigh criterion for minimum record length calculation
- [x] 430+ total tests passing

### Phase 3.3: Open Boundaries
- [x] `TSTOBC2D` Tidal-Subtidal open boundary condition
- [x] `TSTConfig` configuration from constituent files
- [x] `TSTConstituent` for harmonic decomposition
- [x] `NestingBC2D` one-way nesting from parent models (Dirichlet + Flather modes)
- [x] `BoundaryTimeSeries` with linear interpolation for nesting
- [x] `read_constituent_file` parser for tidal harmonic data
- [x] `read_timeseries_file` parser for nesting time series
- [x] `constituent_period` lookup for M2, S2, K1, O1, N2, P1, K2, Q1, MF, MM, M4
- [x] I/O module (`src/io/`) with constituent and timeseries readers
- [x] 536+ total tests passing

### Phase 2: 2D Implementation
- [x] 2D tensor-product Legendre polynomials on [-1,1]²
- [x] 2D Vandermonde matrix with gradient matrices (Vr, Vs)
- [x] `DGOperators2D` bundle (Dr, Ds, mass, LIFT, face interpolation)
- [x] `GeometricFactors2D` (rx, ry, sx, sy, det_J, normals, surface_J)
- [x] `Mesh2D` for quadrilateral elements with edge-based connectivity
- [x] Uniform, periodic, and channel mesh generators
- [x] Gmsh mesh file I/O (format 2.2)
- [x] `BoundaryTag` enum (Wall, Open, TidalForcing, River, Periodic, Dirichlet, Neumann, Custom)
- [x] `Advection2D` equation with upwind flux
- [x] 2D advection RHS with volume + surface terms
- [x] 2D convergence tests (P2→2.98, P3→4.00 order)
- [x] `SWEState2D` and `SWESolution2D` (interleaved storage)
- [x] `ShallowWater2D` equation with Coriolis (f-plane, β-plane)
- [x] Norwegian coast configuration (f ≈ 1.2×10⁻⁴ s⁻¹)
- [x] 2D Riemann solvers: Roe, HLL, Rusanov (rotation approach)
- [x] 2D boundary conditions: Reflective, Radiation, Flather, Tidal, Discharge
- [x] 2D SWE RHS with volume, surface, and source terms
- [x] Lake-at-rest verification (well-balanced)
- [x] Mass/momentum conservation verified
- [x] SSP-RK3 time integration for 2D
- [x] 350 total tests (329 unit + 11 convergence + 8 SWE 2D + 2 doc)

### Phase 1: 1D Shallow Water Equations
- [x] `ConservationLaw` trait for equation abstraction
- [x] `ShallowWater1D` equation implementation
- [x] `SWEState` and `SWESolution` state vector types
- [x] Roe approximate Riemann solver with Harten-Hyman entropy fix
- [x] HLL two-wave solver with Einfeldt wave speed estimates
- [x] Hydrostatic reconstruction for well-balanced schemes (Audusse et al. 2004)
- [x] Manning and Chézy bottom friction with semi-implicit update
- [x] Bathymetry storage with DG gradient computation
- [x] Boundary conditions: Reflective, Radiation, Flather, Tidal, Sponge layer
- [x] Tidal constituents: M2, S2, K1, O1, N2, P1 with harmonic forcing
- [x] TVB slope limiter for oscillation control
- [x] Zhang-Shu positivity limiter for water depth
- [x] Characteristic-based limiting for systems
- [x] SSP-RK3 time integration for SWE with limiter application
- [x] Conservation diagnostics (mass, momentum, energy)
- [x] 176 unit tests covering all components

### Core 1D DG (Phase 0)
- [x] Legendre polynomials and GLL nodes
- [x] Vandermonde matrix and differentiation operators
- [x] Mass and LIFT matrices
- [x] 1D mesh with neighbor connectivity
- [x] Upwind and Lax-Friedrichs numerical fluxes
- [x] SSP-RK3 time integration
- [x] Time-aware RK for proper BC handling
- [x] Periodic boundary conditions
- [x] Rayon parallelism (`--features parallel`)
- [x] Conservation verification tests
- [x] Convergence tests (P1→2.0, P3→4.0, P5→5.9 order)
- [x] Accuracy documentation (ACCURACY.md)
- [x] Performance baseline (PERFORMANCE.md)
- [x] Claude guidelines (CLAUDE.md)

---

## Dependencies Roadmap

| Phase | Add | Purpose | Status |
|-------|-----|---------|--------|
| Current | `faer` | Linear algebra | ✓ |
| Current | `rayon` (optional) | CPU parallelism | ✓ |
| Current | `tempfile` (dev) | Testing | ✓ |
| Phase 2 | custom Gmsh reader | Mesh I/O | ✓ |
| Phase 1 | `thiserror` | Error handling | ✓ |
| Phase 1 | `diffsol` | IMEX time integration | pending |
| Phase 2 | `tritet` | Mesh generation (Triangle/TetGen) | deferred |
| Phase 3 | `netcdf` (optional) | CF-conventions output + forcing input | ✓ |
| Phase 3 | `hdf5-metno` | Checkpointing (Norwegian Met fork) | pending |
| Phase 4 | `cudarc` | GPU compute (f64 support) | pending |
| Phase 4 | `criterion` | Benchmarking | pending |

---

## Key References

Essential papers for implementation:
- Hesthaven & Warburton: "Nodal Discontinuous Galerkin Methods" (foundational algorithms)
- Kronbichler & Kormann 2019: Matrix-free implementation with sum-factorization
- Kang, Giraldo & Bui-Thanh 2019: IMEX HDG-DG for shallow water
- Kärnä et al. 2018, 2020: Thetis ocean-specific DG recipes
- Wu et al. 2024: Entropy stability + subcell positivity for shallow water
- Berntsen: Balanced pressure gradient methods for steep bathymetry
