# Changelog

All notable changes to this project should be documented in this file.

## [Unreleased]

### Added

- Added `examples/norkyst_tidal_validation.rs` wiring the full P1.5 path end-to-end: NorKyst text file → reader → `HarmonicAnalysis` fit → `reference_constants` (nodal correction removed) → catalogue `(H, G)` comparison. Runs a self-contained synthetic round-trip by default (synthesize a tide from a known catalogue with the forward Schureman correction, write it as NorKyst text, read it back through the real reader, and confirm the inference recovers the catalogue to ~0 RMS) and accepts a real `norkyst-client --format text` file as an argument. Documents the absolute-time → epoch recipe (shift the series to `t = 0`, take `AstronomicalArguments` at the first sample) that `reference_constants` requires.
- Added a NorKyst text-ingest reader (`io/norkyst_reader.rs`, `std`-only, no `netcdf` feature) that parses `norkyst-client --format text` point/site output into `analysis::TimeSeries` for the P1.5 validation path. `NorKystTextData::sea_surface_height_series()` yields the `zeta` series to fit with `HarmonicAnalysis` and compare against catalogue constants; `surface_current_series()` yields shallowest-level `(u, v)` for a first ADCP touchpoint. Times are parsed to exact Unix seconds via a proleptic-Gregorian conversion (chrono `… UTC`, RFC3339 `…Z`, or bare seconds), and `Option<f64>` `Debug` fields (`Some(..)`/`None`/`NaN`) are handled. Companion one-line fix in `norkyst-client` (`print_ocean_data`) so the text format now emits a `surface sea_surface_height=… bottom_depth=…` line — previously the surface elevation was written only to parquet/arrow, so a text reader could not recover tidal elevation. 10 unit tests.
- Added `REVIEW.md`: full-crate code and numerics review (2026-07-08) covering the DG core, 3D/vertical physics, mesh/boundary conditions, performance/GPU, and architecture, with a prioritized fix plan. Supersedes `NUMERICAL_REVIEW.md` and `GPU_EVALUATION.md` (both removed); corrects the allocation conclusion in `PROFILE.md`.
- Rewrote `OUTLINE.md` as a current-state architecture document (module tree, layer status, evolution path) — it previously described the 1D-only crate. Updated the stale architecture tree, key types, and 2D-extension guidance in `CLAUDE.md` to match.
- Added a shared 2D DG diffusion helper for conservative element and face coupling.
- Added regression coverage for tracer diffusion conservation across element jumps.
- Added regression coverage for wet/dry positivity and mass-preservation behavior.
- Added Windows native dependency documentation for HDF5 and netCDF-C setup with Miniforge/conda-forge.
- Added regression tests for the P0 review bugs: depth-dependent vertical-diffusion layer thickness; baroclinic-only vs full PGF; wall zero-normal-flux (`reflect_velocity`); ROMS surface-layer selection in nesting; Greenwich phase-lag sign; and a closed-basin seiche whose period matches the analytic `2L/√(gH)`.
- Added `cosine_filter_is_centred_at_baroclinic_endpoint` regression test proving the mode-split barotropic time filter is centred at `t+dt` (constant tendency reproduces the exact `t+dt` value).
- Added GitHub Actions CI (`.github/workflows/ci.yml`): a check matrix over `--no-default-features` and the `parallel`/`simd` feature combinations (guards TODO P0.10), a `cargo nextest` test job, a `clippy` job (`--lib --tests`), and a `cargo fmt --check` job. The native-HDF5/netCDF default build is left out of CI.
- Added nodal-correction inference to harmonic analysis (`analysis/harmonic.rs`): `ConstituentResult::{remove_nodal_correction, to_reference}` and `HarmonicResult::reference_constants(&epoch)` convert *apparent* constituents fitted from a finite record into nodal-corrected *reference* constants (`H_ref = H_app/f`, `φ_ref = φ_app − (V₀+u)`) at the record's `t = 0` epoch — the exact inverse of `tides::correct_amplitude_phase`, making short-record fits directly comparable to a published catalogue (Kartverket/ROMS). Covered by an end-to-end round-trip test (reference → forward-correct → synthesize → fit → infer → reference) and a pass-through test for unsupported constituents.
- Added `tides` module (`src/tides/`): a from-scratch Doodson/Schureman tidal astronomy layer supplying equilibrium arguments `V₀` and 18.61-year nodal corrections (`f`, `u`) for the standard constituents (M2, S2, N2, K2, K1, O1, P1, Q1, M4, MS4, MN4, M6, Mf, Mm, Ssa). Includes `AstronomicalArguments` (Meeus J2000 mean longitudes s/h/p/N/p₁ and mean lunar time τ), a Gregorian→Julian-Date converter, and `nodal_correction(name, &astro)`. Wired into tidal prediction via `TSTConfig::from_constituent_data_at_epoch`, epoch-aware builders `HarmonicFlather2D::with_nodal_corrections(epoch_jd)` / `HarmonicTidal2D::with_nodal_corrections(epoch_jd)`, the `TidalConstituent::with_nodal_correction` primitive, and the `correct_amplitude_phase` helper. Validated by reference-value tests (mean longitudes at J2000, Meeus Julian-date examples), an internal-consistency test that recovers each constituent's tabulated period from the analytic dV₀/dt, and nodal-factor range checks (M2 0.963–1.038; K1/O1 the 11–19% diurnal modulation).

### Changed

- Replaced element-local tracer diffusion with a conservative BR1-style DG diffusion path using lifted gradients and central face fluxes.
- Replaced element-local SWE momentum viscosity with the shared conservative DG diffusion path.
- Updated element-level wet/dry correction APIs to receive DG quadrature data, allowing corrected states to preserve weighted element mass.
- Updated the README and Claude guidance with the Windows HDF5/netCDF-C build workflow.
- API (3D, breaking): `apply_vertical_diffusion` and `ModeSplitIntegrator::{step, step_with_stage_hook}` now take `&Bathymetry2D`; `compute_pressure_gradient` takes an extra `rho_ref` argument. All in-workspace call sites updated.

### Fixed

- **P0.5 (3D):** Vertical diffusion no longer uses a hardcoded 100 m depth. `apply_vertical_diffusion` now takes `&Bathymetry2D` and builds layer thicknesses from the true still-water depth `−B` (so `eta + h = eta − B`), matching the PGF/advection paths.
- **P0.6 (3D):** Mode splitting no longer double-counts the barotropic pressure gradient. `compute_pressure_gradient` gained a `rho_ref` argument; the 3D RHS requests the baroclinic-only PGF (`rho_ref = ρ₀`), leaving `−g∇η` to the 2D `½gh²` flux. Also fixed the barotropic subcycle to advance to the subcycle endpoint instead of a flat time-average (which doubled the seiche period). Follow-up (2026-07-15): the raw endpoint is now replaced by a centred cosine (Hann) time filter — the subcycle overshoots to `t+2·dt` (`dt_bt` unchanged) and feeds the slow mode a raised-cosine weighted average centred at `t+dt`, suppressing barotropic aliasing over long baroclinic steps.
- **P0.7 (3D):** Coastal/land walls in the 3D continuity (omega) and momentum-advection diagnostics now use a reflective free-slip ghost state (`u_ext·n = −u_int·n`) instead of copying the interior, so mass/momentum no longer leak through the coastline. Added a shared `boundary::reflect_velocity` helper (also now used by `Reflective2D`).
- **P0.8 (nesting):** `OceanModelReader` now reads the ROMS/NorKyst surface s-layer (`n_depth − 1`, bottom-up s-coordinates) rather than index 0 (the seabed) when reducing 4D fields. Follow-up (2026-07-15): the reader now prefers the depth-averaged `ubar_eastward`/`vbar_northward` (or ROMS `ubar`/`vbar`) barotropic velocity over the surface s-layer of 3D `u`/`v` — the 2D barotropic child needs `hu = h·ubar` — and `read_variable` gained a size-consistency guard that skips staggered (u-/v-point) variables whose element count does not match the rho grid.
- **P0.9 (tides):** The Greenwich phase lag is now subtracted, not added, so tidal elevation is `A·cos(ωt − G)` (previously a time-reversed tide). Fixed in `TSTConstituent`, `TSTConfig::from_constituent_data`, and `ConstituentData::evaluate`.
- **P0.10 (build):** `cargo check --no-default-features` compiles again — the batched SIMD GEMM path selects `Par::Seq` when the `parallel` feature is off instead of calling `rayon::current_num_threads()` unconditionally.
- Fixed horizontal tracer diffusion and SWE viscosity so discontinuities across element interfaces are handled by the DG surface coupling instead of being invisible to local Laplacians.
- Fixed dry-element positivity limiting so elements with nonnegative dry averages do not inject water by filling all nodes to `h_min`.
- Fixed element-level wet/dry corrections to rescale corrected depths and momentum back to the pre-correction nonnegative element mass.
- Fixed the default Windows build path by documenting the required HDF5 `1.14.x` pin; current conda-forge HDF5 `2.x` packages are rejected by `hdf5-metno-sys 0.10.1`.

### Verified

- `cargo check`
- `cargo check --no-default-features --features parallel,simd`
- `cargo test --no-default-features --features parallel,simd --test tracer_2d_test`
- `cargo test --no-default-features --features parallel,simd --test viscosity_test`
- `cargo test --no-default-features --features parallel,simd --test wet_dry_test`
- `cargo test --no-default-features --features parallel,simd --lib` (1002 lib tests, incl. the new `tides` astronomy suite and epoch-aware BC builder tests) + tides doctests; `cargo clippy --no-default-features --features parallel,simd --lib --tests` (no new warnings); `cargo fmt --all --check`; default-feature `cargo check --lib` (conda `roms-rs` env)
