# Changelog

All notable changes to this project should be documented in this file.

## [Unreleased]

### Added

- Added `REVIEW.md`: full-crate code and numerics review (2026-07-08) covering the DG core, 3D/vertical physics, mesh/boundary conditions, performance/GPU, and architecture, with a prioritized fix plan. Supersedes `NUMERICAL_REVIEW.md` and `GPU_EVALUATION.md` (both removed); corrects the allocation conclusion in `PROFILE.md`.
- Rewrote `OUTLINE.md` as a current-state architecture document (module tree, layer status, evolution path) — it previously described the 1D-only crate. Updated the stale architecture tree, key types, and 2D-extension guidance in `CLAUDE.md` to match.
- Added a shared 2D DG diffusion helper for conservative element and face coupling.
- Added regression coverage for tracer diffusion conservation across element jumps.
- Added regression coverage for wet/dry positivity and mass-preservation behavior.
- Added Windows native dependency documentation for HDF5 and netCDF-C setup with Miniforge/conda-forge.
- Added regression tests for the P0 review bugs: depth-dependent vertical-diffusion layer thickness; baroclinic-only vs full PGF; wall zero-normal-flux (`reflect_velocity`); ROMS surface-layer selection in nesting; Greenwich phase-lag sign; and a closed-basin seiche whose period matches the analytic `2L/√(gH)`.

### Changed

- Replaced element-local tracer diffusion with a conservative BR1-style DG diffusion path using lifted gradients and central face fluxes.
- Replaced element-local SWE momentum viscosity with the shared conservative DG diffusion path.
- Updated element-level wet/dry correction APIs to receive DG quadrature data, allowing corrected states to preserve weighted element mass.
- Updated the README and Claude guidance with the Windows HDF5/netCDF-C build workflow.
- API (3D, breaking): `apply_vertical_diffusion` and `ModeSplitIntegrator::{step, step_with_stage_hook}` now take `&Bathymetry2D`; `compute_pressure_gradient` takes an extra `rho_ref` argument. All in-workspace call sites updated.

### Fixed

- **P0.5 (3D):** Vertical diffusion no longer uses a hardcoded 100 m depth. `apply_vertical_diffusion` now takes `&Bathymetry2D` and builds layer thicknesses from the true still-water depth `−B` (so `eta + h = eta − B`), matching the PGF/advection paths.
- **P0.6 (3D):** Mode splitting no longer double-counts the barotropic pressure gradient. `compute_pressure_gradient` gained a `rho_ref` argument; the 3D RHS requests the baroclinic-only PGF (`rho_ref = ρ₀`), leaving `−g∇η` to the 2D `½gh²` flux. Also fixed the barotropic subcycle to advance to the subcycle endpoint instead of a flat time-average (which doubled the seiche period).
- **P0.7 (3D):** Coastal/land walls in the 3D continuity (omega) and momentum-advection diagnostics now use a reflective free-slip ghost state (`u_ext·n = −u_int·n`) instead of copying the interior, so mass/momentum no longer leak through the coastline. Added a shared `boundary::reflect_velocity` helper (also now used by `Reflective2D`).
- **P0.8 (nesting):** `OceanModelReader` now reads the ROMS/NorKyst surface s-layer (`n_depth − 1`, bottom-up s-coordinates) rather than index 0 (the seabed) when reducing 4D fields.
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
