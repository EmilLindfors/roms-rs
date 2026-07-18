# TODO - DG Coastal Ocean Model (dg-rs)

Roadmap informed by multi-agent review (2026-02-11) comparing against ROMS/NorKyst-800,
updated by the full-crate review (2026-07-08) in `REVIEW.md`.
See `reports/` for the 2026-02-11 analysis: ROMS comparison, math evaluation, BC assessment, devil's advocate.

**Current state**: Feature-complete 2D SWE solver (986 default-feature lib tests passing), SIMD/parallel scaffolding, early 3D layer. The 2D solver is research-ready; production use requires addressing the items below. The seven P0 correctness blockers are now fixed (P0.11 deferred) — see Priority 0.

**Strategic direction**: Position as a **high-accuracy 2D barotropic/storm surge model** first, then extend to 3D incrementally. DG's geometric flexibility for complex fjord coastlines is the key differentiator over ROMS.

---

## ▶ Next session — start here

Last worked: 2026-07-15 (P1.5 NorKyst text-ingest reader — item 1 below.
`io::read_norkyst_text_file`/`parse_norkyst_text_str` turn `norkyst-client --format text`
point/site output into `NorKystTextData`, exposing `sea_surface_height_series()` (zeta, for the
tidal fit) and `surface_current_series()` (shallowest-level u/v, for ADCP). `std`-only, no
`netcdf` feature; 10 unit tests. Needed a one-line companion fix in the `norkyst-client` repo
(`print_ocean_data`) to emit a `surface …` line — the text format had been dropping the surface
elevation, writing it only to parquet/arrow. NOTE: met.no THREDDS was returning HTTP 503 this
session, so no real NorKyst series was fetched — the reader is ready; the remaining P1.5 blocker
is still live data.) Earlier the same day: harmonic-analysis nodal-correction inference —
`HarmonicResult::reference_constants(&epoch)` / `ConstituentResult::to_reference` invert the
nodal correction so *apparent* constants fitted from a finite record become catalogue-comparable
*reference* constants; end-to-end round-trip test against `tides::correct_amplitude_phase`. And
earlier still: the P0.9 Doodson/Schureman
`src/tides/` astronomy module (V₀, f/u) and epoch-aware BC builders. **Every P0 correctness item
is fixed except P0.11** (burn GPU surface term, deferred with the module). Pick up in roughly
this order:

1. **P1.5 Validation Against Observations** — the main milestone toward operational quality
   (NorKyst-800 tides, Bergen/Stavanger/Trondheim/Kristiansund tide gauges, ADCP; infra exists).
   The tidal-comparison code path is now complete end-to-end: fit a record with
   `HarmonicAnalysis`, call `HarmonicResult::reference_constants(&epoch)` to strip the nodal
   modulation, then compare to catalogue `(H, G)` pairs. **The remaining blocker is data, not
   code.** NorKyst-800 data can be pulled with the global `norkyst-client` CLI (v0.1.0, `~/.cargo/bin`)
   — point/sites/grid extraction over OPeNDAP; see the P1.5 section for invocations. Ingest is
   now solved: fetch with `--format text` and read it with `io::read_norkyst_text_file` (no
   parquet dependency, no convert step). Real Kartverket gauge records still need to be dropped
   into `data/tide_gauges/` (only synthetic `heimsjo.txt` ships today), and met.no THREDDS was
   returning 503 this session — retry the fetch when the server is back.
2. **P1.6 End-to-end benchmarks** — full-RHS throughput and parallel scaling.
3. **Optional P0.9 extensions** (not blocking): ~~hook `nodal_correction` into
   harmonic-analysis inference (`analysis/harmonic.rs`)~~ DONE 2026-07-15
   (`HarmonicResult::reference_constants(&epoch)` / `ConstituentResult::to_reference`
   invert the nodal correction so fitted apparent constants become catalogue-comparable
   reference constants). Remaining: add the full satellite-modulation (t_vuf-style)
   apparatus only if the ~0.1% closed-form f/u proves insufficient during P1.5 validation.
   (Epoch-aware constructors for `HarmonicTidal2D`/`HarmonicFlather2D` are also done —
   `with_nodal_corrections(epoch_jd)`.)

Build note: default features need the `roms-rs` conda env (`conda activate roms-rs`) for HDF5/netCDF;
otherwise use `cargo test --no-default-features --features parallel,simd`.

---

## Priority 0: Correctness Bugs from the 2026-07-08 Review — RESOLVED (P0.11 deferred)

The seven "top bugs" from `REVIEW.md`. Each fix landed with a regression test that would
have caught it. Together they are the difference between "demo" and "trustworthy first
realistic run".

**Status (2026-07-09):** P0.5, P0.6, P0.7, P0.8, P0.9 (phase sign), P0.10 all FIXED with
tests; P0.11 (delete burn) DEFERRED per user decision (burn-cuda is the primary GPU target).
The cosine barotropic time filter (P0.6), `ubar/vbar` barotropic nesting (P0.8), the
`--no-default-features` CI step (P0.10), and the P0.9 nodal-correction / equilibrium-argument
astronomy module were completed 2026-07-15. The only remaining P0 follow-up is the burn surface
term (P0.11, deferred with the module). All 997 `--no-default-features` lib tests pass (the new
`tides` astronomy suite included); the default-feature build compiles.

### P0.5 Hardcoded 100 m depth in vertical diffusion (BLOCKER, 3D) — FIXED
- [x] `src/physics/vertical_diffusion.rs` — replaced `let h = 100.0` with `h = -bathymetry.get(...)` (still-water depth; `eta + h = eta - B = water_depth`, matching the PGF/advection paths)
- [x] Threaded `&Bathymetry2D` through `apply_vertical_diffusion` and `ModeSplitIntegrator::{step, step_with_stage_hook}`; caller passes `physics.bathymetry`
- [x] Regression test `diffusion_layer_thickness_is_depth_dependent`: 50 m vs 500 m column give differing profiles under identical surface stress (would be identical with the old hardcoded depth)

### P0.6 Mode splitting double-counts the barotropic pressure gradient (BLOCKER, 3D) — FIXED
- [x] `compute_pressure_gradient` gained a `rho_ref` parameter: `rho_ref = 0` → full PGF, `rho_ref = ρ₀` → baroclinic-only (the ρ₀ contribution integrates to exactly `−g∇η`). `compute_rhs_3d` now requests the baroclinic-only PGF, so the depth-averaged G-term no longer carries `−g∇η` (the 2D `½gh²` flux supplies it)
- [x] Regression test `baroclinic_only_pgf_excludes_surface_slope`: constant-density tilted surface → full PGF = `−g∇η`, baroclinic-only PGF = 0 (deterministic)
- [x] **Second bug found while validating the seiche:** the barotropic subcycle used a flat time-average (centred at t+dt/2) as the t+dt state, halving the evolution rate and *doubling* the seiche period. Switched `ModeSplitIntegrator::{step, step_with_stage_hook}` to the subcycle **endpoint**. Confirmed: full PGF + endpoint → 142 s ≈ 2L/√(2gH) (the double-count signature); baroclinic-only + endpoint → within 15% of the analytic 2L/√(gH) ≈ 202 s
- [x] Regression test `seiche_period_matches_analytic` (closed basin, fundamental mode) — now passing
- [x] Follow-up (REVIEW.md §1.5): replaced the raw endpoint with a centred cosine (Hann) time filter — the barotropic subcycle now overshoots to `t+2·dt` (`dt_bt` unchanged, CFL preserved) and the fields fed back to the slow mode are a raised-cosine weighted average `w_m = 1 − cos(π·m/n_bt)` centred at `t+dt`, suppressing barotropic aliasing while staying consistent for the slow trend. Shared `subcycle_barotropic_filtered` helper (both `step` and `step_with_stage_hook`); regression test `cosine_filter_is_centred_at_baroclinic_endpoint` proves centring (constant tendency → exact `t+dt` value); `seiche_period_matches_analytic` still passes

### P0.7 Coastal walls transmissive in the 3D path (BLOCKER, 3D) — FIXED
- [x] Omega diagnostic (`vertical_velocity.rs`) and 3D momentum advection (`advection_3d.rs`) now build a mirrored-normal reflective ghost state at physical boundaries instead of copying the interior
- [x] Added shared `boundary::reflect_velocity(u, v, nx, ny)` (single source of truth for the free-slip wall: `u_ext·n = −u_int·n`, tangential preserved); refactored `Reflective2D` and both 3D sites to use it
- [x] Regression test `reflect_velocity_zero_normal_flux_at_wall` asserts `un_ext = −un_int` for arbitrary velocities/normals — i.e. zero Rusanov mass flux through the wall (the leak). Note: domain-integrated volume `∫η` is governed by the 2D barotropic BC, so a full-sim volume check would not isolate this 3D-diagnostic bug; the flux-level property is the targeted guard.

### P0.8 NorKyst nesting reads the wrong ROMS vertical layer (BLOCKER, nesting) — FIXED
- [x] `OceanModelReader::reshape_to_3d` now samples the 4D `[time][s_rho][y][x]` field at `s_rho = n_depth-1` (free surface) instead of index 0 (seabed); ROMS/NorKyst s-coordinates are bottom-up
- [x] Refactored the function to take `(n_dims, n_depth)` instead of `&[netcdf::Dimension]` so it is pure and unit-testable
- [x] Regression test `reshape_to_3d_takes_surface_layer_not_seabed` (distinct per-layer values; asserts surface value chosen) — netcdf-gated, verified passing under `--features netcdf`
- [x] Follow-up (REVIEW.md §3.4): `OceanModelReader` now prefers the depth-averaged `ubar_eastward`/`vbar_northward` (NorKyst rho-point) / `ubar`/`vbar` (ROMS) fields over the surface s-layer of 3D `u`/`v` — the 2D barotropic child needs `hu = h·ubar`, not the surface current. Added a size-consistency guard in `read_variable` (skips staggered u-/v-point variables whose element count doesn't match the rho grid, so they can't silently corrupt the read). Regression test `nesting_prefers_depth_averaged_ubar_over_surface_layer` (netcdf-gated) builds a file with both fields and asserts the depth-averaged one wins

### P0.9 Tidal phase sign + missing astronomy (MAJOR, tides) — phase sign FIXED
- [x] Greenwich phase lag was ADDED, not subtracted — now negated on ingest so elevation is `A·cos(ωt − G)`:
  - `TSTConstituent::from_degrees` and `TSTConfig::from_constituent_data` store internal `φ = −G`
  - `ConstituentData::evaluate` subtracts `phase_radians()` instead of adding
- [x] Documented the convention (internal phase φ = −G) on all three paths
- [x] Fixed `test_from_constituent_data` (encoded the old `+G` bug); strengthened `test_tst_constituent_from_degrees` to be sign-sensitive; added `test_greenwich_phase_lag_is_subtracted` (peaks at t = G/ω, guards against the mirrored tide)
- [x] `tidal.rs` reviewed: uses the low-level internal-phase convention (caller passes φ = −G directly), not a Greenwich-lag ingest — correct as-is
- [x] Follow-up (2026-07-15): nodal corrections (f, u) and equilibrium argument V₀ from Doodson args at a configured epoch (K1/O1 amplitude errors are 11–19% without them). New `src/tides/` module: `AstronomicalArguments` (Meeus J2000 mean longitudes s/h/p/N/p₁ + mean lunar time τ), a Gregorian→Julian-Date converter, per-constituent Doodson equilibrium arguments, and Schureman closed-form `f`/`u` for M2, S2, N2, K2, K1, O1, P1, Q1, M4, MS4, MN4, M6, Mf, Mm, Ssa. Exposed as `tides::nodal_correction(name, &astro)` and wired into prediction via `TSTConfig::from_constituent_data_at_epoch(data, h_ref, dx, epoch_jd)`, the epoch-aware builders `HarmonicFlather2D::with_nodal_corrections(epoch_jd)` and `HarmonicTidal2D::with_nodal_corrections(epoch_jd)`, the `TidalConstituent::with_nodal_correction` primitive, and the `tides::correct_amplitude_phase` helper (all apply `A → f·A`, internal phase `φ → φ + (V₀+u)`). Validated by: mean longitudes at J2000 vs Meeus constants; Meeus Julian-date worked examples; an internal-consistency test recovering each constituent's tabulated period from analytic dV₀/dt (also checks the Doodson coefficients + longitude rates); S2 V₀ = 30°·H_UT (solar time); nodal-factor ranges (M2 0.963–1.038, K1/O1 diurnal 11–19% modulation); node-cycle periodicity; plus per-BC tests that the corrections are baked into amplitude/phase and that unknown constituents pass through unchanged. Not (yet) done: full satellite-modulation apparatus (t_vuf-style) and hooking corrections into harmonic-analysis inference — the closed-form `f`/`u` used here are accurate to ~0.1%/~0.1°, far below the omitted-correction error.

### P0.10 `--no-default-features` build broken (BLOCKER, build) — FIXED
- [x] `src/solver/simd/batched.rs` — replaced the unconditional `rayon::current_num_threads()` call with a `faer_par()` helper that returns `Par::Rayon` under the `parallel` feature and `Par::Seq` otherwise (keeps batched GEMM usable single-threaded)
- [x] Verified `cargo check --no-default-features` and `cargo check --no-default-features --features parallel,simd` both compile
- [x] Added CI (`.github/workflows/ci.yml`, 2026-07-15): a `check` matrix over `--no-default-features` and the `parallel`/`simd` combinations, a `test` job (`cargo nextest run --no-default-features --features parallel,simd`), a `clippy` job (`--lib --tests`; benches excluded — they've drifted from the API), and a `fmt --check` job. Native-HDF5/netCDF default build is intentionally left out of CI (awkward to provision on hosted runners). Ran `cargo fmt --all` to make the fmt gate green and fixed a pre-existing `approx_constant` clippy error in `vandermonde_2d` tests.

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
**Tidal-comparison code path is complete** (fit with `HarmonicAnalysis` → `HarmonicResult::reference_constants(&epoch)` → compare to catalogue `(H,G)`). Remaining work is fetching real data and wiring it through.

- [ ] **Fetch NorKyst-800 data via the global `norkyst-client` CLI** (v0.1.0 at `~/.cargo/bin/norkyst-client`). Extracts NorKyst historical/forecast over OPeNDAP:
  - Point time series at a gauge: `norkyst-client --source historical --lat <lat> --lon <lon> --start-date YYYY-MM-DD --end-date YYYY-MM-DD --time-grain hourly --format parquet -o <dir>`
  - Multiple gauges at once: `--sites sites.csv` (CSV columns `id,lat,lon`) — build one row per target station.
  - Regional grid (for a bbox run): `--bbox min_lat min_lon max_lat max_lon` or `--area <1-13> --geojson PO.geojson`; `--partition-grain day|month`.
  - **Caveat:** output formats are `text|arrow|parquet|vortex` — NOT NetCDF. The existing NetCDF nesting reader (`OceanModelReader`) will not ingest these directly.
  - **Ingest reader DONE (2026-07-15):** `io::read_norkyst_text_file` / `parse_norkyst_text_str` parse `--format text` point/site output into `NorKystTextData` (`std`-only, no `netcdf` feature). `sea_surface_height_series()` → `TimeSeries` of `zeta` for the tidal fit; `surface_current_series()` → shallowest-level `(u,v)` for ADCP. This required a one-line companion fix in `norkyst-client` (`print_ocean_data`) to emit a `surface sea_surface_height=… bottom_depth=…` line — the text format previously dropped the surface elevation (it was written only to parquet/arrow). Fetch with `--format text -o <file>` (or stdout) and read it directly; no parquet dependency added.
  - **Native parquet reader + real Bergen result DONE (2026-07-18):** point/NCSS text fetch is chronically 503; the reliable path is `--bbox … --format parquet` over OPeNDAP. Extended norkyst-client's grid writer to emit `sea_surface_height`/`bottom_depth` (client PR #2), then added `io::read_norkyst_parquet_glob` behind the off-by-default `parquet` feature (`sea_surface_height_series_nearest(lon,lat)` → zeta `TimeSeries`). The example reads a parquet dataset directly with `--features parquet` — no DuckDB/Python step. Ran on a real 2-month Bergen fetch: **M2 H≈0.427 m G≈278°, S2 H≈0.117 m, fit R²≈0.97** over 61 days (P1 aliases into K1 at 61 days — needs ~6 months to separate). Remaining for a *scored* result: real Kartverket `(H,G)` for Bergen to diff against, and a longer fetch for the diurnal band.
  - **End-to-end example DONE (2026-07-16):** `examples/norkyst_tidal_validation.rs` runs reader → `HarmonicAnalysis::fit` → `reference_constants` → catalogue compare. `cargo run --example norkyst_tidal_validation --no-default-features --features parallel,simd` does a synthetic round-trip (recovers a known catalogue to ~0 RMS); pass a real text file as an arg to fit/print inferred reference constants. Note the absolute-time→epoch recipe it encodes: shift the read series to `t = 0` and take `AstronomicalArguments` at the first sample before calling `reference_constants`. Once real Kartverket `(H, G)` values are on hand, drop them in as the catalogue to score a station.
- [ ] Run comparison against NorKyst-800 barotropic tides for a test period (use the fetched series as the reference)
- [ ] Compare with Norwegian tide gauge observations (infrastructure exists in `analysis/tide_gauge.rs`; real Kartverket records still need to be dropped into `data/tide_gauges/` — only synthetic `heimsjo.txt` ships today)
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
| Unit tests | 1,004 (no-default lib) | 1,100+ |
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
