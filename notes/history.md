# Resolved items (history)

## Resolved (history)

### Priority 0 from the 2026-07-08 review — resolved (P0.11 deferred, above)
Details are in `CHANGELOG.md` and git history. Caveats found on 2026-09-25 are noted in brackets.

- **P0.5** Hardcoded 100 m depth in vertical diffusion: FIXED. Layer thicknesses come from η − B via `&Bathymetry2D`. Regression: `diffusion_layer_thickness_is_depth_dependent` (weak: it only checks that the profiles differ).
- **P0.6** Barotropic PGF double count: FIXED. `rho_ref` selects the baroclinic-only PGF. The endpoint/flat-average bug was fixed; Hann time filter centred at t+dt with correct moments. Regressions: `baroclinic_only_pgf_excludes_surface_slope`, `seiche_period_matches_analytic`, `cosine_filter_is_centred_at_baroclinic_endpoint`.
  - [Still open: the subcycle-in-every-stage structure, Forward Euler substeps, and the advection/Coriolis double count in G. See P4.1.]
- **P0.7** 3D walls transmissive: FIXED for Ω and momentum via the shared `boundary::reflect_velocity`.
  - [Tracer flux still leaks: P0.22.]
- **P0.8** NorKyst read the seabed layer: FIXED for s_rho files. `ubar/vbar` preferred; size guard skips staggered variables.
  - [Wrong for z-level files, and no rotation/transport conservation: P1.5.]
- **P0.9** Tidal phase sign (A·cos(ωt − G)): FIXED. Plus the `src/tides/` Doodson/Schureman astronomy (V₀, f, u for 15 constituents), epoch-aware BC builders, and harmonic nodal-correction inference (`reference_constants`). Verified correct in the 2026-09-25 review.
  - [Constituent periods are truncated: P0.15.]
- **P0.10** `--no-default-features` build: FIXED. `faer_par()` helper; CI (check matrix, nextest, clippy, fmt).

### Priority 0 from 2026-02-11 — resolved
- **P0.1** Depth-formula inconsistency (`h = η_tidal − B` in `HarmonicFlather2D`/`TSTOBC2D`): FIXED.
- **P0.2** RHS heap allocations inside element loops: FIXED for the element loops only.
  - [Per-step allocations remain: P1.1, P2.3.]
- **P0.3** Added `ROMSVstretching4`.
  - [It is **not** the standard Shchepetkin & McWilliams (2005) Vstretching 4, contrary to the original note: P4.3.]
- **P0.4** Affine-element assumption documented for cell averages and `GeometricFactors2D`.

### Priority 1–2 items from 2026-02-11 — done
- **P1.1** Tests: SWE convergence (linearised: P1 1.67, P2 3.02; see P1.7), Radiation2D unit tests, exact dam-break Riemann solver, 50-period periodic conservation, "multi-day" stability run (316 s).
- **P1.2** Horizontal viscosity: constant + Smagorinsky; conservative BR1 face coupling now shared with tracer diffusion; diffusive CFL helper.
- **P1.3** `SpongeLayer2D`; `BCContext2D::dt` plumbing (but `with_dt` has no callers: P1.4); binary search in `InterpolatedTidalBC`.
  - Orlanski radiation deferred (P1.4).
- **P1.4** `CurvilinearIndex` bucket grid for NetCDF curvilinear lookup.
- **P2.2** Mass-matrix comment corrected. **P2.3** Geometric-factor docs.

### Completed phases
- Phase 0: core 1D DG (Legendre, Vandermonde, operators, SSP-RK3).
- Phase 1: 1D SWE (Roe/HLL/HLLC, well-balanced, limiters, BCs).
- Phase 2: 2D tensor-product quads, 2D SWE, convergence verified.
- Phase 3: Norwegian-coast features:
  - tidal forcing, Chapman/Flather/TST-OBC, nesting;
  - wind, Coriolis, friction, atmospheric pressure;
  - wetting/drying, harmonic analysis;
  - tide-gauge and ADCP infrastructure.
- Performance scaffolding: SIMD kernels, rayon parallelism, profiling scripts. I/O: NetCDF (CF-1.8), VTK, GeoTIFF, GSHHS, Gmsh 2.2, NorKyst text/parquet readers.
- 3D scaffolding: sigma grid, stretching, `Solution3D`, mode splitter, Ω diagnostic, implicit vertical diffusion, baroclinic PGF (standard form), EOS.
