# Tech debt and structure (P6)

## Priority 6: Tech Debt and Structure (`review-2026-09-25.md` §6.2)

- [ ] Move the remaining examples (`profile_cpu`, `quick_profile`, `high_res_benchmark`; Frøya is done) from the legacy `ssp_rk3_swe_2d` stepper and its `SWE2DTimeConfig` onto `Simulation` + `SWEPhysics2D`. This is the precondition for deleting the legacy integrators below. The two paths currently duplicate limiter and wet/dry dispatch.
- [ ] Delete the legacy integrators (`ssp_rk3*.rs`, three `coupled_swe_tracer` variants, `burn_ssp_rk3.rs`; ~2.4k lines). Make `CoupledState2D` implement `Integrable`; one time loop (fold `Simulation3D::run_with_callback`). `Simulation3D` still fires callbacks at the first step past each interval (drift); `Simulation` now lands on them exactly.
- [ ] Collapse the duplicate config enums (`SWEFluxType2D` vs `StandardFlux2D`; three limiter enums).
- [ ] Prune ~200 root re-exports to a `prelude`; export `Simulation3D`.
- [ ] Error handling:
  - `SimulationResult` → `Result<SimulationStats, SimulationError>`;
  - reachable panics (`timeseries_reader.rs:198`, `geotiff.rs:239-245`);
  - library `eprintln!` in `tide_gauge.rs`.
- [ ] Feature-gate `tiff`/`shapefile`/`geo` behind `geodata`.
- [ ] `solver/simd/` is mostly dead (2026-10-08): `compute_volume_terms_batched`/`BatchedVolumeWorkspace` (faer GEMM) has no caller, the SIMD `combine_derivatives` has none, and the rest serves only the `Standard` kernel slated for retirement. Retire the module with `Standard` (P1.2). (The `simd` feature's `pulp 0.18` is gone since 2026-10-08: its two kernels are on fearless_simd.)
- [ ] faer 0.23's own pulp paths (everything except matmul) stop at AVX2 on stable; its matmul already uses AVX-512 through `private-gemm-x86`. faer 0.24's `nightly` feature is now just `pulp/x86-v4` and builds on stable. It only matters for setup-time solves (harmonic analysis, bathymetry fits, the test spectra), so upgrade when convenient, not for speed.
- [x] ~~`cargo clippy` reports 103 warnings (2026-10-07, Rust 1.98) and CI runs it without `-D warnings`.~~ Done 2026-10-09 (branch `chore/clippy-pass`): none in any feature set (none, `parallel`, `simd`, `parallel,simd`, default with NetCDF; `--all-targets`, 157 warnings and one error before). `needless_range_loop` and `too_many_arguments` are allowed crate-wide in `Cargo.toml` `[lints.clippy]` (the 79 per-site allows removed). CI runs `--all-targets -- -D warnings`. Two lessons from `cargo clippy --fix`: it sees one feature set, so it removed an import that other feature sets needed and left a `#[cfg]` hanging on the next item (`examples/high_res_benchmark.rs`); and it rewrote five `for v in &mut x { *v = c }` loops as `&mut x.fill(c);`. Check every feature set after it.
- [ ] Build the `netcdf` feature in CI. It is excluded because it needs native HDF5/netCDF-C, so the NetCDF reader and nesting tests (P0.20) only run locally. `cargo check --features netcdf/static` builds HDF5 and netCDF-C from source with CMake and no conda (verified on Windows 2026-09-25: ~5.5 min cold, cached afterwards); use it for a CI job, and consider it as the documented local alternative to conda.
- [ ] Characteristic-based limiting for the 2D SWE system (1D version exists but is dead code; was P2.4). Likely subsumed by P1.2.
- [x] ~~Consolidate the root markdown files into `docs/`.~~ Done 2026-10-10: `OUTLINE.md` → `docs/architecture.md`, `ACCURACY.md` → `docs/accuracy.md`, `REVIEW.md` → `notes/review-2026-09-25.md`; the outdated ones deleted (see CHANGELOG). Decide crate vs repo name; `dg-core`/`roms-rs` workspace split.
