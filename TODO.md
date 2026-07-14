# TODO - DG Coastal Ocean Model (dg-rs)

Roadmap informed by multi-agent review (2026-02-11) comparing against ROMS/NorKyst-800,
updated by the full-crate review (2026-07-08) in `REVIEW.md`.
See `reports/` for the 2026-02-11 analysis: ROMS comparison, math evaluation, BC assessment, devil's advocate.

**Current state**: Feature-complete 2D SWE solver, ~1,091 tests passing, SIMD/parallel scaffolding, early 3D layer. The 2D solver is research-ready; production use requires addressing the items below.

**Strategic direction**: Position as a **high-accuracy 2D barotropic/storm surge model** first, then extend to 3D incrementally. DG's geometric flexibility for complex fjord coastlines is the key differentiator over ROMS.

---

## Priority 0: Correctness Bugs from the 2026-07-08 Review (Fix Now)

The seven "top bugs" from `REVIEW.md`. Each fix landed with a regression test that would
have caught it. Together they are the difference between "demo" and "trustworthy first
realistic run".

**Status (2026-07-09):** P0.5, P0.6, P0.7, P0.8, P0.9 (phase sign), P0.10 all FIXED with
tests; P0.11 (delete burn) DEFERRED per user decision (burn-cuda is the primary GPU target).
Follow-ups remain: nodal tidal corrections (P0.9), cosine barotropic time filter (P0.6),
`ubar/vbar` nesting (P0.8), a `--no-default-features` CI step (P0.10), and the burn surface
term (P0.11). All 986 default-feature lib tests pass; `--no-default-features` compiles.

### P0.5 Hardcoded 100 m depth in vertical diffusion (BLOCKER, 3D) — FIXED
- [x] `src/physics/vertical_diffusion.rs` — replaced `let h = 100.0` with `h = -bathymetry.get(...)` (still-water depth; `eta + h = eta - B = water_depth`, matching the PGF/advection paths)
- [x] Threaded `&Bathymetry2D` through `apply_vertical_diffusion` and `ModeSplitIntegrator::{step, step_with_stage_hook}`; caller passes `physics.bathymetry`
- [x] Regression test `diffusion_layer_thickness_is_depth_dependent`: 50 m vs 500 m column give differing profiles under identical surface stress (would be identical with the old hardcoded depth)

### P0.6 Mode splitting double-counts the barotropic pressure gradient (BLOCKER, 3D) — FIXED
- [x] `compute_pressure_gradient` gained a `rho_ref` parameter: `rho_ref = 0` → full PGF, `rho_ref = ρ₀` → baroclinic-only (the ρ₀ contribution integrates to exactly `−g∇η`). `compute_rhs_3d` now requests the baroclinic-only PGF, so the depth-averaged G-term no longer carries `−g∇η` (the 2D `½gh²` flux supplies it)
- [x] Regression test `baroclinic_only_pgf_excludes_surface_slope`: constant-density tilted surface → full PGF = `−g∇η`, baroclinic-only PGF = 0 (deterministic)
- [x] **Second bug found while validating the seiche:** the barotropic subcycle used a flat time-average (centred at t+dt/2) as the t+dt state, halving the evolution rate and *doubling* the seiche period. Switched `ModeSplitIntegrator::{step, step_with_stage_hook}` to the subcycle **endpoint**. Confirmed: full PGF + endpoint → 142 s ≈ 2L/√(2gH) (the double-count signature); baroclinic-only + endpoint → within 15% of the analytic 2L/√(gH) ≈ 202 s
- [x] Regression test `seiche_period_matches_analytic` (closed basin, fundamental mode) — now passing
- [ ] Follow-up (REVIEW.md §1.5): replace the raw endpoint with a properly centred cosine time filter to suppress barotropic aliasing over long baroclinic steps

### P0.7 Coastal walls transmissive in the 3D path (BLOCKER, 3D) — FIXED
- [x] Omega diagnostic (`vertical_velocity.rs`) and 3D momentum advection (`advection_3d.rs`) now build a mirrored-normal reflective ghost state at physical boundaries instead of copying the interior
- [x] Added shared `boundary::reflect_velocity(u, v, nx, ny)` (single source of truth for the free-slip wall: `u_ext·n = −u_int·n`, tangential preserved); refactored `Reflective2D` and both 3D sites to use it
- [x] Regression test `reflect_velocity_zero_normal_flux_at_wall` asserts `un_ext = −un_int` for arbitrary velocities/normals — i.e. zero Rusanov mass flux through the wall (the leak). Note: domain-integrated volume `∫η` is governed by the 2D barotropic BC, so a full-sim volume check would not isolate this 3D-diagnostic bug; the flux-level property is the targeted guard.

### P0.8 NorKyst nesting reads the wrong ROMS vertical layer (BLOCKER, nesting) — FIXED
- [x] `OceanModelReader::reshape_to_3d` now samples the 4D `[time][s_rho][y][x]` field at `s_rho = n_depth-1` (free surface) instead of index 0 (seabed); ROMS/NorKyst s-coordinates are bottom-up
- [x] Refactored the function to take `(n_dims, n_depth)` instead of `&[netcdf::Dimension]` so it is pure and unit-testable
- [x] Regression test `reshape_to_3d_takes_surface_layer_not_seabed` (distinct per-layer values; asserts surface value chosen) — netcdf-gated, verified passing under `--features netcdf`
- [ ] Follow-up (may split out): read `ubar`/`vbar` directly for a barotropic child instead of the surface s-layer (REVIEW.md §3.4)

### P0.9 Tidal phase sign + missing astronomy (MAJOR, tides) — phase sign FIXED
- [x] Greenwich phase lag was ADDED, not subtracted — now negated on ingest so elevation is `A·cos(ωt − G)`:
  - `TSTConstituent::from_degrees` and `TSTConfig::from_constituent_data` store internal `φ = −G`
  - `ConstituentData::evaluate` subtracts `phase_radians()` instead of adding
- [x] Documented the convention (internal phase φ = −G) on all three paths
- [x] Fixed `test_from_constituent_data` (encoded the old `+G` bug); strengthened `test_tst_constituent_from_degrees` to be sign-sensitive; added `test_greenwich_phase_lag_is_subtracted` (peaks at t = G/ω, guards against the mirrored tide)
- [x] `tidal.rs` reviewed: uses the low-level internal-phase convention (caller passes φ = −G directly), not a Greenwich-lag ingest — correct as-is
- [ ] Follow-up (may split out): nodal corrections (f, u) and equilibrium argument V₀ from Doodson args at a configured epoch (K1/O1 amplitude errors are 11–19% without them)

### P0.10 `--no-default-features` build broken (BLOCKER, build) — FIXED
- [x] `src/solver/simd/batched.rs` — replaced the unconditional `rayon::current_num_threads()` call with a `faer_par()` helper that returns `Par::Rayon` under the `parallel` feature and `Par::Seq` otherwise (keeps batched GEMM usable single-threaded)
- [x] Verified `cargo check --no-default-features` and `cargo check --no-default-features --features parallel,simd` both compile
- [ ] Add a CI/check step: `cargo check --no-default-features` (no CI infra exists yet; follow-up)

### P0.11 Burn GPU RHS is physically wrong (BLOCKER, GPU) — DEFERRED (keep module for now)
Decision (2026-07-09): keep `src/solver/burn/` for now — burn-cuda is the stated primary GPU
target, so deletion is on hold. The module remains behind the `burn` feature and does not affect
default builds. The underlying defect still stands and MUST be fixed before any GPU use:
- [ ] `src/solver/burn/rhs.rs` computes the HLL flux then discards it (no surface term → no inter-element coupling → not the SWE); masked by a lake-at-rest-only test at tol 0.1
- [ ] Implement the surface term and add a burn-vs-CPU equivalence test before trusting any GPU result
- [ ] Until then, treat the GPU path as non-functional / experimental

---

## Priority 0 (2026-02-11): Bugs and Correctness Issues ~~(Fix Now)~~ RESOLVED

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

### P2.1 GPU Module Assessment — superseded by P0.11 (delete the module)
- [ ] `src/solver/burn/rhs.rs:193-198` has `TODO: Implement full surface term computation` -- surface terms are the defining DG feature
- [ ] Burn lacks f64 matmul -- the GPU module (~850 LOC across 10 files) cannot run a valid simulation
- [ ] **Decision needed**: Remove dead GPU code, or gate behind `#[cfg(feature = "burn-experimental")]` with clear docs
- [ ] `REVIEW.md` §4.2 recommends deletion (also documents that the "GPU slower" result was caused by per-stage GPU→CPU→GPU round-trips, not only f64 support; supersedes the removed `GPU_EVALUATION.md`)

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
| Unit tests | 986 | 1,100+ |
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
