# TODO - DG Coastal Ocean Model (dg-rs)

Roadmap informed by multi-agent review (2026-02-11) comparing against ROMS/NorKyst-800.
See `reports/` for full analysis: ROMS comparison, math evaluation, BC assessment, devil's advocate.

**Current state**: Feature-complete 2D SWE solver with 30/30 math components correct, ~968 tests, SIMD/parallel/GPU scaffolding. The 2D solver is research-ready; production use requires addressing the items below.

**Strategic direction**: Position as a **high-accuracy 2D barotropic/storm surge model** first, then extend to 3D incrementally. DG's geometric flexibility for complex fjord coastlines is the key differentiator over ROMS.

---

## Priority 0: Bugs and Correctness Issues ~~(Fix Now)~~ RESOLVED

All P0 issues identified by the review team have been fixed (2026-02-11).

### P0.1 Depth Formula Inconsistency — FIXED
- [x] Removed `h_ref` from depth formula in `HarmonicFlather2D` and `TSTOBC2D`
- [x] Both now use correct `h = η_tidal - bathymetry`
- [x] `h_ref` field retained for API compat but deprecated and ignored
- [x] Added `tidal_depth_with_bathy(t, bathymetry)` to `TSTOBC2D`; old `tidal_depth()` deprecated
- [x] Updated all affected tests to use `bathymetry = -50.0` (physically correct bed elevation)

### P0.2 RHS Heap Allocations — FIXED
- [x] Sequential: hoisted `flux_x`, `flux_y`, and face buffers before the element loop
- [x] Parallel: converted `for_each` → `for_each_init` with per-thread workspace tuple
- [x] Parallel+SIMD: converted `for_each` → `for_each_init` with `SIMDParallelWorkspace` struct (29 buffers)
- [x] SIMD: added face buffer pre-allocation before element loop
- [x] Zero heap allocations inside element loops across all 4 RHS variants

### P0.3 Non-Standard Song-Haidvogel Stretching — FIXED
- [x] Added `ROMSVstretching4` implementing standard Shchepetkin & McWilliams (2005) formula
- [x] Documented existing `SongHaidvogelStretching::cs_function` as non-standard blend
- [x] Existing stretching kept as-is (no breaking change); users can choose `ROMSVstretching4` for standard ROMS behavior
- [x] 5 new tests for bounds, monotonicity, zero-param uniform, surface refinement, sigma_rho ordering

### P0.4 Cell Average Jacobian Weighting — DOCUMENTED
- [x] Added "Affine element assumption" doc to `swe_cell_averages_2d`, `tracer_cell_averages_2d`, and parallel variant
- [x] Documented `GeometricFactors2D` as assuming constant Jacobian per element
- [x] Current implementation is exact for parallelogram meshes; doc notes per-node J needed for curved elements

---

## Priority 1: Complete the 2D Solver for Production

These items bring the 2D barotropic solver to operational quality.

### P1.1 Missing Tests (Math Evaluator + BC Assessor) — DONE
- [x] **SWE convergence test**: P1 achieves order 1.67, P2 achieves order 3.02 (linearized SWE on periodic domain)
- [x] **Radiation2D unit tests**: 5 tests covering still water, outgoing/incoming waves, velocity decomposition, deep water
- [x] **Multi-day stability test**: 100 wave-crossing times (~316s sim time) on 4x4 P1 periodic mesh; checks no blow-up, no negative depth, mass conservation < 1e-10
- [x] **Dam-break exact solution**: Implemented exact Riemann solver (Toro 2009, Ch. 10) with Newton-Raphson iteration, replacing rough `h_m=0.5*(h_l+h_r)` approximation. 4 tests: symmetric, classic, continuity, dry-bed limit.
- [x] **Long-time conservation test**: 50 oscillation periods on periodic domain; verifies mass, x-momentum, and y-momentum conservation all < 1e-10

### P1.2 Horizontal Viscosity/Diffusion — DONE
- [x] Implemented `HorizontalViscosity2D` with constant and Smagorinsky models (`src/source/swe_2d/viscosity.rs`)
- [x] Viscous terms ∇·(ν h ∇u), ∇·(ν h ∇v) injected in sequential and parallel RHS via double-derivative Laplacian
- [x] `compute_dt_viscosity()` diffusive CFL helper
- [x] 4 unit tests (constant, zero-strain, pure-shear, cs-scaling) + 5 integration tests (exponential decay, Smagorinsky smoke, mass conservation, CFL, parallel consistency)
- [ ] Surface viscous flux (penalty terms at element interfaces) — improves convergence for high-order but not essential for stability

### P1.3 Boundary Condition Improvements (BC Assessor)
- [x] **Nudging/relaxation zones**: Implemented as `SpongeLayer2D` in `src/source/boundary/sponge.rs` with spatially varying relaxation (16 tests)
- [x] **2D sponge layer**: `SpongeLayer2D` source term is production-ready (exponential/polynomial profiles, multi-field relaxation)
- [x] **Auto-update Chapman dt**: `BCContext2D` now carries `dt: Option<f64>`, propagated from `SWE2DRhsConfig`; Chapman/ChapmanFlather prefer `ctx.dt` over stored `self.dt`
- [x] **Fix InterpolatedTidalBC search**: Replaced O(n) linear search with O(log n) binary search (matches `BoundaryTimeSeries` pattern)
- [ ] **Orlanski adaptive radiation**: Estimate phase speed from solution (used in ROMS, less critical for 2D barotropic) — deferred

### P1.4 NetCDF I/O Performance (BC Assessor) — DONE
- [x] **Curvilinear grid search**: Replaced O(ny×nx) brute-force with `CurvilinearIndex` bucket grid (O(1) amortized lookup)
- [x] Bucket grid built at initialization in `from_file()`; ~√(n_cells) buckets per dimension, capped at 1024
- [x] Unit test verifying all cell centers are found correctly via the index

### P1.5 Validation Against Observations
- [ ] Run comparison against NorKyst-800 barotropic tides for a test period
- [ ] Compare with Norwegian tide gauge observations (infrastructure exists in `analysis/tide_gauge.rs`)
- [ ] Compare with ADCP current data (infrastructure exists in `analysis/adcp.rs`)
- [ ] Document skill scores (RMSE, bias, correlation) for key stations
- [ ] Target: Bergen, Stavanger, Trondheim, Kristiansund tide gauges

### P1.6 End-to-End Benchmarks (Devil's Advocate)
- [ ] Full RHS evaluation time as function of mesh size (not just micro-kernels)
- [ ] End-to-end DOFs/second throughput metric
- [ ] Parallel scaling efficiency (1, 2, 4, 8, 16 cores)
- [ ] Memory bandwidth utilization measurement

---

## Priority 2: Clean Up Dead Code and Tech Debt

### P2.1 GPU Module Assessment
- [ ] `src/solver/burn/rhs.rs:193-198` has `TODO: Implement full surface term computation` -- surface terms are the defining DG feature
- [ ] Burn lacks f64 matmul -- the GPU module (~850 LOC across 10 files) cannot run a valid simulation
- [ ] **Decision needed**: Remove dead GPU code, or gate behind `#[cfg(feature = "burn-experimental")]` with clear docs
- [ ] `GPU_EVALUATION.md` already documents this honestly

### P2.2 Mass Matrix Comment (Math Evaluator: LOW) — FIXED
- [x] `src/operators/mass.rs` comment corrected: now states (N+1)-point GLL is exact for degree 2N-1, notes degree-2N aliasing is harmless

### P2.3 Geometric Factors Documentation — DONE
- [x] `GeometricFactors2D` documented with affine element assumption (done in P0.4)
- [x] Cell average functions documented with same assumption

### P2.4 Characteristic-Based Limiting
- [ ] 1D characteristic limiter exists (`limiters_1d.rs`) but marked `#[allow(dead_code)]`
- [ ] 2D TVB/Kuzmin limiters are component-wise (more dissipative than characteristic-based)
- [ ] Consider integrating characteristic limiting for 2D SWE system, especially near shocks

---

## Priority 3: 2D Operational Features

Features needed for a production 2D barotropic/storm surge model.

### P3.1 Surface Heat Flux Budget
- [ ] Shortwave radiation, longwave radiation, sensible/latent heat fluxes
- [ ] Needed for multi-day forecasts where SST evolution matters
- [ ] Currently only wind stress and atmospheric pressure are implemented

### P3.2 River Forcing Enhancement
- [ ] Add vertical shape function to `Discharge2D` (Froude-number-dependent distribution)
- [ ] Integrate with NVE river discharge database (1,760 rivers in NorKyst-800)
- [ ] Currently basic: `ConstantDischarge2D` and `Discharge2D` exist

### P3.3 ROMS-Compatible Output
- [ ] `NetCDFWriter` produces CF-1.8 but NOT ROMS-compatible output
- [ ] For validation against NorKyst, need reconcilable formats
- [ ] Add ROMS variable name mapping option

### P3.4 Restart / Checkpointing
- [ ] HDF5 or NetCDF restart file capability
- [ ] Essential for operational runs (daily forecasts must hot-start)

### P3.5 Operational Pipeline
- [ ] Automated run scripts
- [ ] Monitoring and alerting
- [ ] THREDDS/OPeNDAP output serving

---

## Priority 4: 3D Extension

The critical path to ROMS-equivalent capability. See `3D_TODO.md` for detailed implementation plan.

**Realistic estimate**: 2-3 years (team of 2-3) or 5-8 years (solo part-time).
**The devil's advocate notes the 3D_TODO.md estimate of ~6,300 LOC is likely 2-3x too low.**

### Phase 4.1: Vertical Infrastructure
- [x] ~~Fix Song-Haidvogel stretching to match ROMS Vstretching=4~~ (done in P0.3 — `ROMSVstretching4` added)
- [ ] 3D state types: `Solution3D` with `[n_elements x n_nodes x n_levels]` layout
- [ ] EOS upgrade: TEOS-10 (current UNESCO EOS-80 is adequate but older)
- [ ] **PGE handling**: Implement pressure gradient error correction (Shchepetkin-McWilliams 2003) -- CRITICAL for steep fjord bathymetry

### Phase 4.2: Mode-Split Time Stepping
- [ ] Barotropic/baroclinic splitting (existing 2D SWE = barotropic mode)
- [ ] Barotropic subcycling with cosine time filter
- [ ] G-term coupling between modes
- [ ] Vertical velocity diagnosis from continuity

### Phase 4.3: Vertical Mixing
- [ ] GLS (Generic Length Scale) -- k-epsilon variant (NorKyst-800 uses this)
- [ ] GLS is a 1D vertical problem per horizontal point -- relatively straightforward
- [ ] Background diffusivity for numerical stability
- [ ] Convective adjustment for unstable stratification
- [ ] KPP as alternative (recommended for open ocean)

### Phase 4.4: 3D Physics
- [ ] 3D baroclinic pressure gradient (extend existing depth-averaged version)
- [ ] Vertical advection/diffusion (FD in sigma space)
- [ ] 3D tracer transport (extend existing 2D tracer framework)
- [ ] Implicit vertical diffusion (tridiagonal solver)
- [ ] 3D boundary conditions (extend Flather/Chapman for velocity profiles)

### Phase 4.5: 3D Validation
- [ ] Internal wave mode-1 propagation test
- [ ] Lock exchange (density-driven flow)
- [ ] Idealized fjord estuarine circulation
- [ ] Comparison with NorKyst-800 output (3D fields)

---

## Priority 5: Advanced Features (Future)

Beyond operational 2D, for research and long-term capability.

### Research Numerics
- [ ] Entropy-stable DG (SBP operators + entropy-conservative fluxes)
- [ ] IMEX time stepping via diffsol (implicit vertical diffusion)
- [ ] Subcell positivity preservation with convex limiting (Wu et al. 2024)
- [ ] Sum-factorization for tensor-product elements
- [ ] Curved elements for high-order coastal geometry representation
- [ ] hp-adaptive mesh refinement

### Operational
- [ ] Data assimilation (start with EnKF, then 4D-Var)
- [ ] Two-way nesting (child feeds back to parent)
- [ ] MPI for distributed memory (>millions of elements)
- [ ] Wave-current interaction
- [ ] Biological coupling (NPZD for salmon lice dispersion)
- [ ] Sediment transport

---

## Completed (Summary)

See `TODO_old.md` for detailed history of resolved issues.

- Phase 0: Core 1D DG (Legendre, Vandermonde, operators, SSP-RK3)
- Phase 1: 1D SWE (Roe/HLL/HLLC fluxes, well-balanced, limiters, BCs)
- Phase 2: 2D implementation (tensor-product quads, 2D SWE, convergence verified)
- Phase 3: Norwegian coast features (tidal forcing, Chapman/Flather/TST-OBC, nesting, wind, Coriolis, friction, atmospheric pressure, wetting/drying, harmonic analysis, tide gauge + ADCP validation infrastructure)
- Performance: SIMD kernels, Rayon parallelism, profiling infrastructure, optimization passes
- I/O: NetCDF (CF-1.8), VTK, GeoTIFF bathymetry, GSHHS coastline, Gmsh mesh
- P0 fixes (2026-02-11): depth formula fix, RHS zero-allocation, ROMS Vstretching=4, affine element docs
- P1.2 (2026-02-11): Smagorinsky horizontal viscosity (constant + strain-dependent models, sequential + parallel RHS, diffusive CFL)

---

## Key Metrics

| Metric | Current | Target (2D operational) |
|--------|---------|------------------------|
| Unit tests | 968 | 1,100+ |
| Math correctness | 30/30 | 30/30 ✓ |
| RHS allocations in hot path | Zero ✓ | Zero ✓ |
| Depth formula consistency | Fixed ✓ | Fixed ✓ |
| SWE convergence verified | P1: 1.67, P2: 3.02 ✓ | Yes (P1-P5) |
| Validated against observations | No | Yes (5+ tide gauges) |
| Multi-day stability | Untested | 30+ days stable |
| GPU functional | No (incomplete) | Documented/removed |

---

## References

- Hesthaven & Warburton (2008) - Nodal DG Methods
- Toro (2009) - Riemann Solvers for Fluid Dynamics
- Shchepetkin & McWilliams (2005) - ROMS split-explicit method
- Audusse et al. (2004) - Hydrostatic reconstruction
- Zhang & Shu (2010) - Positivity-preserving limiters
- Kuzmin (2010) - Vertex-based slope limiting
- Chapman (1985) - Free surface radiation BC
- Flather (1976) - Barotropic velocity BC
- Warner et al. (2005) - GLS vertical mixing
- NorKyst v3 preprint (2025) - egusphere-2025-3986
