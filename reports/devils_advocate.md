# Devil's Advocate Analysis: dg-rs for Norwegian Coastal Modeling

**Date:** 2026-02-11
**Role:** Constructive Critic
**Codebase:** ~52,000 LOC Rust, ~2,400 LOC integration tests, ~1,500 LOC benchmarks, 948 unit tests

---

## Executive Summary

This project is a serious, well-engineered attempt to build a DG shallow water solver in Rust. The mathematical foundation is solid, the test coverage is above average for scientific code, and the architecture is clean. However, it faces fundamental viability challenges for its stated goal of operational Norwegian coastal modeling. The gap between "working 2D SWE solver" and "operational 3D ocean model competitive with NorKyst800" is enormous -- roughly the difference between a working engine and a production automobile.

---

## TOP 5 RISKS THAT COULD MAKE THIS PROJECT FAIL

### Risk 1: The 3D Gap Is Insurmountable for a Small Team

**Severity: CRITICAL**

The codebase is fundamentally a 2D shallow water solver. The `src/vertical/` module contains only sigma coordinate infrastructure (~600 LOC) -- coordinate transforms, stretching functions, and nothing else. There is no:

- 3D state type or solution storage
- Mode-split time stepping (barotropic/baroclinic splitting)
- Vertical mixing parameterization (KPP, GLS, or anything)
- Baroclinic pressure gradient computation
- Equation of state for density from T/S
- Vertical advection or diffusion
- 3D boundary conditions

The project's own `3D_TODO.md` estimates ~6,300 LOC for the 3D extension across 7 phases. This is almost certainly an underestimate by a factor of 2-3x. For reference:

- ROMS has ~200,000+ lines of Fortran for its 3D solver
- Thetis (the closest DG comparison) is built on Firedrake, which itself has ~500,000+ LOC of infrastructure

Norwegian coastal modeling **requires** 3D for stratification (fjord estuarine circulation), river plumes, and internal waves. A 2D SWE solver cannot replicate what NorKyst800 does, period.

**Evidence:** `src/vertical/sigma.rs` is purely coordinate transforms. `src/vertical/stretching.rs` implements stretching functions. Neither contains any physics, advection, diffusion, or mixing code. The 3D_TODO.md itself lists every Phase 1-7 item as unchecked `[ ]`.

### Risk 2: Performance Claims Are Unvalidated at Scale

**Severity: HIGH**

The largest benchmark run documented is 5,000 elements at P3 (~80,000 DOFs for 3 variables) -- see `GPU_EVALUATION.md`. NorKyst800 operates with:

- 800m resolution covering the entire Norwegian coast
- ~2.4 million horizontal grid points
- 40 vertical levels
- = ~96 million DOFs per time step

The current solver's documented throughput is ~9ms per RK3 step for 5,000 elements. Linear extrapolation to 2.4M grid points (roughly 150,000+ elements at P3) suggests ~270ms per step -- **before** adding 3D physics, vertical operations, I/O, and mode splitting. A typical NorKyst800 simulation runs 24-hour forecasts requiring millions of time steps.

More critically, the `compute_rhs_swe_2d` function at `src/solver/rhs/swe_2d.rs:149-398` **allocates a new `SWESolution2D` every single call** (line 159). The SIMD version pre-allocates workspace buffers outside the element loop (good), but the parallel version at line 806 allocates per-element `Vec`s inside the Rayon closure:

```rust
// src/solver/rhs/swe_2d.rs:806-810 (parallel version)
let mut rhs_k = vec![SWEState2D::zero(); n_nodes];
let mut flux_x = vec![SWEState2D::zero(); n_nodes];
let mut flux_y = vec![SWEState2D::zero(); n_nodes];
```

And the parallel+SIMD version at lines 1063-1093 allocates **28 Vec buffers per element per Rayon task**. For 150,000 elements with 3 RHS evaluations per RK3 step, that's 12.6 million small allocations per time step. The allocator will become a bottleneck long before the math does.

**The CLAUDE.md itself says: "Don't allocate in RHS: The RHS function is called thousands of times per simulation."** The code violates its own rule.

### Risk 3: GPU Strategy Is Dead on Arrival

**Severity: HIGH**

The `GPU_EVALUATION.md` is admirably honest: Burn's CUDA backend does not support f64 matrix multiplication. The GPU module at `src/solver/burn/rhs.rs:193-198` contains this damning comment:

```rust
// TODO: Implement full surface term computation
// This requires computing flux differences and applying LIFT for all faces
```

The surface term is **the defining feature of DG methods** -- without it, you have a broken solver. The GPU implementation computes volume terms only. The `compute_rhs_swe_2d_burn` function is not a functional RHS computation -- it is incomplete.

Furthermore, even if Burn added f64 support tomorrow:
- Consumer GPUs deliver 1/32 the f64 throughput of f32 (RTX 4060: 0.5 vs 15 TFLOPS)
- DG element-local operations are small matrices (10x10 at P3), which are a terrible fit for GPU warp-level parallelism
- The data transfer overhead between CPU and GPU is non-trivial for the required mesh connectivity access patterns

The GPU code (~850 LOC across 10 files in `src/solver/burn/`) is dead weight. It compiles, but it cannot run a real simulation.

### Risk 4: No Validation Against Observations or Operational Models

**Severity: HIGH**

The test suite includes:
- Advection convergence tests (1D and 2D) -- **good**
- Mass conservation tests -- **good**
- Lake-at-rest tests -- **good**
- Dam break, standing wave, geostrophic balance -- **good**
- Harmonic analysis infrastructure -- **exists**

What is completely absent:
- **No comparison with ROMS/NorKyst800 output** for any test case
- **No comparison with tide gauge observations** (the `tide_gauge_validation_test.rs` exists but tests infrastructure, not physics)
- **No SWE convergence rate test** -- convergence is only tested for scalar advection, not for the nonlinear shallow water system
- **No multi-day simulation stability test** -- can the solver run for 30 days without blowing up?
- **No real-data benchmark with known answer** -- the `froya_real_data.rs` example runs but has no reference solution to compare against

For operational oceanography, the Norwegian Meteorological Institute (MET Norway) requires validated skill scores against tide gauges, ADCP data, and satellite altimetry. None of this validation pipeline exists.

### Risk 5: Ecosystem and Community Disadvantage

**Severity: MEDIUM-HIGH**

ROMS has:
- 30+ years of development
- Thousands of validated regional setups worldwide
- Active community (myroms.org) with forums, workshops, and documentation
- Extensive pre-processing tools (grid generation, forcing preparation)
- Post-processing tools (NCO, CDO, xarray/xroms)
- Data assimilation (4D-Var)
- Coupled models (atmosphere, waves, sediment, biogeochemistry)
- Operational track record at dozens of agencies worldwide

This project has:
- 1 developer (based on git history)
- 0 external users
- 0 published papers
- 0 validated regional setups
- No data assimilation
- No atmospheric forcing reader beyond basic wind stress
- No wave coupling
- No biogeochemistry
- No grid generation tool (relies on programmatic mesh building)

Building community trust for a new ocean model takes 5-10 years of published validation studies. Agencies will not switch from ROMS to an unvalidated alternative, regardless of language advantages.

---

## TOP 5 STRENGTHS THAT MAKE IT WORTH PURSUING

### Strength 1: Exceptional Code Quality and Architecture

The codebase demonstrates genuine Rust expertise applied to numerical methods:

- **Clean module separation**: `polynomial/`, `basis/`, `operators/`, `mesh/`, `flux/`, `solver/`, `time/` -- each module has a clear responsibility
- **Strong typing**: `ElementIndex`, `FaceIndex`, `Depth`, `Elevation`, `Sigma`, `PhysicalZ` newtypes prevent unit confusion errors
- **Trait-based extensibility**: `SourceTerm2D`, `SWEBoundaryCondition2D`, `Limiter2D` traits allow clean composition
- **948 unit tests** across 117 files is excellent for scientific computing code
- **Both AoS and SoA** data layouts are available, with SIMD kernels for the hot path

The code reads like it was written by someone who understands both Rust idioms and numerical methods. The boundary condition architecture (`src/boundary/`) with `MultiBoundaryCondition2D`, tag-based dispatch, and the comprehensive set of BC types (Reflective, Flather, Chapman, Radiation, Tidal, Nesting, Sponge) is particularly well-designed.

### Strength 2: Correct DG Implementation with Verified Convergence

The numerical foundation is sound:

- **P1 through P5 convergence verified** for advection (`tests/convergence_test.rs`) with proper error-doubling checks
- **Conservation verified** to machine precision for periodic domains
- **2D convergence verified** (P2, P3) on quadrilateral meshes
- **Multiple Riemann solvers**: Roe, HLL, Rusanov, Lax-Friedrichs -- all with unit tests (`src/flux/`)
- **Well-balanced hydrostatic reconstruction** (Audusse et al. 2004) implemented and tested
- **Kuzmin vertex-based limiter** for oscillation control -- a modern choice
- **Wetting/drying** treatment present

The mathematical formulation matches Hesthaven-Warburton (2008) correctly. The DG operators (Vandermonde, differentiation, LIFT matrices) are tested for polynomial exactness. This is not a toy implementation.

### Strength 3: Rust's Safety Guarantees Are Real for Scientific Computing

Specific advantages visible in this codebase:

- **No segfaults from array bounds**: `ElementIndex` wrapping prevents off-by-one in mesh access
- **No data races in parallel code**: Rayon's ownership model guarantees correct parallelization
- **Compile-time feature management**: `#[cfg(feature = "parallel")]`, `#[cfg(feature = "simd")]` cleanly separate code paths
- **Dependency safety**: `faer` (pure Rust BLAS) eliminates Fortran compiler/ABI compatibility issues
- **Reproducible builds**: `Cargo.lock` pins every dependency

For long-running operational simulations, eliminating entire classes of bugs (use-after-free, data races, buffer overflows) is a genuine operational advantage. ROMS segfaults are a real problem in production.

### Strength 4: Comprehensive Boundary Condition System

The boundary condition module (`src/boundary/`) is remarkably complete for a project this young:

- **Tidal forcing**: `HarmonicTidal2D` with 8 constituents, `HarmonicFlather2D` for radiation
- **Nesting**: `NestingBC2D` for one-way coupling to parent models, `OceanNestingBC2D` for NetCDF-based nesting
- **TST OBC**: Tide/Subtidal/Trend separation for open boundaries -- a sophisticated approach
- **Chapman-Flather**: Combined elevation/velocity radiation -- standard for ocean modeling
- **Sponge layers**: Configurable damping zones with multiple profile options
- **Discharge**: River input with `ConstantDischarge2D` and `Discharge2D`
- **Multi-BC dispatch**: Tag-based boundary condition selection per mesh segment
- **Bathymetry validation**: Automated checking of depth conventions at boundaries

This infrastructure is directly targeted at the NorKyst800 nesting use case and shows deep domain understanding.

### Strength 5: Honest Self-Assessment and Good Engineering Practices

The project includes:

- `GPU_EVALUATION.md` -- unflinchingly honest about GPU failure
- `PERFORMANCE.md` -- measured benchmarks with actual numbers, not aspirational claims
- `3D_TODO.md` -- realistic roadmap with effort estimates
- `CLAUDE.md` -- clear coding standards and numerical requirements
- `ACCURACY.md` -- convergence rate documentation

This level of documentation and self-awareness is rare in scientific computing. The developer knows what works, what does not, and what is missing. This is the foundation for a project that can actually improve over time rather than accumulating technical debt.

---

## HONEST ASSESSMENT OF TIME-TO-OPERATIONAL-READINESS

### What "Operational Norwegian Coastal Modeling" Actually Requires

Based on NorKyst800's documented configuration:

1. **3D hydrodynamics**: Barotropic/baroclinic mode splitting, sigma coordinates, 40 vertical levels
2. **Turbulence closure**: GLS (k-epsilon variant) vertical mixing
3. **Atmospheric forcing**: ERA5 or AROME-MetCoOP surface fields (hourly)
4. **Tidal forcing**: TPXO constituents at open boundaries
5. **River input**: NVE discharge for 1,760 rivers distributed across 69 coastal regions
6. **Nesting**: Lateral BCs from TOPAZ (Arctic) and CMEMS Baltic model
7. **Data assimilation**: At minimum, SST nudging; ideally 4D-Var
8. **Validation**: Skill scores against tide gauges, ADCPs, CTD profiles
9. **Operational pipeline**: Automated daily runs, monitoring, restart capability
10. **Output**: CF-compliant NetCDF, THREDDS/OPeNDAP serving

### Current Status vs Requirements

| Requirement | Status | Gap |
|---|---|---|
| 2D SWE solver | COMPLETE | - |
| 3D hydrodynamics | NOT STARTED | ~15,000 LOC (realistic) |
| Turbulence closure | NOT STARTED | ~3,000 LOC |
| Atmospheric forcing | PARTIAL (wind, pressure) | ~2,000 LOC |
| Tidal forcing | MOSTLY COMPLETE | Minor gaps |
| River input | BASIC | ~1,000 LOC for NVE integration |
| Nesting framework | GOOD FOUNDATION | ~2,000 LOC for full 3D |
| Data assimilation | NOT STARTED | ~10,000-20,000 LOC |
| Validation pipeline | INFRASTRUCTURE ONLY | ~5,000 LOC + months of analysis |
| Operational pipeline | NOT STARTED | ~3,000 LOC + DevOps |
| CF-compliant output | PARTIAL | ~1,000 LOC |

### Timeline Estimate

**Optimistic (full-time team of 2-3):**
- 3D solver core: 6-12 months
- Physics and forcing: 3-6 months
- Validation: 6-12 months (cannot be parallelized -- requires running and analyzing)
- Operational readiness: 3-6 months
- **Total: 2-3 years minimum**

**Realistic (single developer, part-time):**
- 3D solver core: 2-3 years
- Everything else: 3-5 years additional
- **Total: 5-8 years**

**For comparison:** ROMS development started in ~1995 and reached operational maturity around 2005-2010. Thetis started ~2015 and is still not widely used operationally (primarily a research tool). NorKyst800 took several years to set up and validate even using the mature ROMS framework.

### Alternative Viable Paths

1. **2D storm surge / flood model**: The current 2D SWE solver could be operationally useful for coastal flood forecasting (cf. BRGM's operational DG model in France). This requires far less development than full 3D.

2. **Research tool for DG method development**: Use this as a testbed for numerical methods (new limiters, adaptive mesh refinement, novel BCs) rather than an operational model. Publish papers on the methods, not the application.

3. **DG pre/post-processor for ROMS**: Use the DG solver for high-resolution coastal inserts nested within ROMS, leveraging DG's superior geometry handling for complex fjord boundaries while ROMS handles the 3D physics.

4. **Barotropic tidal model**: A 2D tidal model with excellent boundary conditions could be operationally useful for tidal prediction without needing 3D. The existing tidal infrastructure (harmonic analysis, Flather BCs, sponge layers) supports this.

---

## SPECIFIC CODE ISSUES FOUND

### 1. Allocations in Hot Path (CRITICAL for performance)

**File:** `src/solver/rhs/swe_2d.rs:159`
```rust
let mut rhs = SWESolution2D::new(mesh.n_elements, n_nodes);
```
This allocates `n_elements * n_nodes * 3 * 8` bytes every RHS call. For 5,000 elements at P3 (10 nodes): 1.2 MB allocation, ~2,500 times per simulated minute at CFL=0.5.

**File:** `src/solver/rhs/swe_2d.rs:177-178` (scalar version)
```rust
let mut flux_x = vec![SWEState2D::zero(); n_nodes];
let mut flux_y = vec![SWEState2D::zero(); n_nodes];
```
Allocated per element inside the loop. For 5,000 elements: 10,000 allocations per RHS call.

The SIMD version (line 429+) correctly pre-allocates outside the loop. But the scalar and parallel versions do not.

### 2. Incomplete GPU Surface Terms

**File:** `src/solver/burn/rhs.rs:193-198`
```rust
// TODO: Implement full surface term computation
// This requires computing flux differences and applying LIFT for all faces
```
The GPU RHS function is non-functional as a DG solver. Without surface terms, it is just a finite-difference-like volume computation.

### 3. Missing SWE Convergence Tests

The `tests/convergence_test.rs` only tests scalar advection convergence. There is no convergence rate verification for the shallow water equations. This is a significant gap because:
- SWE is a nonlinear system with wave coupling
- Limiters can degrade convergence order
- Flux choices affect convergence differently for systems vs scalars
- Well-balanced treatment affects convergence near bathymetry features

### 4. Geometric Factor Storage Is Suboptimal

**File:** `src/solver/rhs/swe_2d.rs:166-169`
```rust
let rx = geom.rx[k.as_usize()];
let ry = geom.ry[k.as_usize()];
let sx = geom.sx[k.as_usize()];
let sy = geom.sy[k.as_usize()];
```
The geometric factors are stored as **per-element scalars** (`Vec<f64>`), which is correct only for affine (straight-sided) elements. For curved elements (needed for accurate fjord boundary representation), geometric factors must be per-node. This is not currently a bug, but it is a design limitation that will require restructuring for realistic coastline modeling.

### 5. Benchmarks Test Micro-Kernels, Not End-to-End

All 6 benchmark files test individual kernels (diff_matrix, lift, coriolis, flux, source_terms, time_stepping). There is no benchmark for:
- Full RHS evaluation time as a function of mesh size
- End-to-end DOFs/second throughput
- Parallel scaling efficiency
- Memory bandwidth utilization

The `PERFORMANCE.md` documents a single end-to-end timing (Froya at 600 elements) but not as a criterion benchmark with statistical rigor.

---

## CONCLUSION

This is one of the best-architected scientific computing projects I have reviewed. The code quality, test coverage, and honest documentation put it ahead of many published research codes. The 2D SWE solver is functionally correct and demonstrates genuine understanding of both Rust engineering and numerical methods.

However, the project's stated goal -- operational Norwegian coastal modeling to replace or complement NorKyst800 -- is roughly 5-8 years away for a single developer. The 3D gap is not just missing code; it represents missing physics, missing validation, and missing operational infrastructure.

**The most impactful near-term pivot would be:** Position this as a high-accuracy 2D barotropic/storm surge model for coastal flood forecasting, where the DG method's superior geometry handling provides genuine advantages over finite-difference models, and where 3D physics are not required. The boundary condition infrastructure is already well-suited for this use case.
