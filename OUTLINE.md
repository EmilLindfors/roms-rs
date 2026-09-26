# Architecture

A DG (Discontinuous Galerkin) solver for Norwegian coastal ocean modeling: a
feature-rich 2D barotropic SWE solver (not yet correct on realistic bathymetry
and open boundaries; see `REVIEW.md`) plus an early-stage ROMS-style 3D
layer (2D DG horizontal × finite-difference vertical, sigma coordinates, mode
splitting).

For the current health assessment and prioritized fix plan, see `REVIEW.md`
(2026-09-25). For the 3D roadmap, see `3D_TODO.md`. This document describes
what exists and how it fits together.

## Design Philosophy

**Evolve, don't over-engineer.** Start concrete, abstract when patterns emerge.

| Principle | Approach |
|-----------|----------|
| Single crate (for now) | 17 modules, ~75k LOC; a `dg-core` / `roms-rs` workspace split is the likely end state (REVIEW.md §6.2) |
| Concrete first | `Mesh1D`, `Mesh2D`; generic traits (`Integrable`, `SourceTerm2D`) added once 2+ implementations existed |
| faer for LA | Dense element-local operators; QR/SVD available for least-squares (harmonic analysis) |
| f64 everywhere | Coastal stability requires double precision |
| Norwegian focus | Fjords, tides, wetting/drying — not generality |

## Module Structure (current)

```
src/
├── lib.rs            # Public API, re-exports (oversized — REVIEW.md §6.2)
├── types/            # Newtypes: indices, bounds, physical (Depth, Sigma), sides
│
│  # ---- DG core (dimension/equation-agnostic) ----
├── polynomial/       # Legendre P_n, GLL nodes/weights (1D + tensor-product 2D)
├── basis/            # Vandermonde matrices, nodal<->modal (1D + 2D)
├── operators/        # Dr/Ds, mass, LIFT, GeometricFactors2D (affine-only — REVIEW.md §4.1)
├── mesh/
│   ├── core/         # Mesh1D, Mesh2D (quads), Mesh2DBuilder
│   ├── data/         # Bathymetry, land mask, boundary tags
│   ├── io/           # Gmsh MSH 2.2 reader
│   └── traits/       # Mesh traits, Point; MeshGPUData (unimplemented)
├── flux/             # Upwind, Lax-Friedrichs, Roe, HLL; 2D SWE + tracer fluxes;
│                     # entropy-conservative/-stable SWE fluxes (Wintermeyer)
│
│  # ---- Equations & physics ----
├── equations/        # ConservationLaw trait; advection 1D/2D, SWE 1D/2D, UNESCO EOS
├── source/           # SourceTerm1D/2D traits; Coriolis, friction, wind, tidal
│                     # potential, viscosity, bathymetry, sponge; well-balanced
│                     # hydrostatic reconstruction
├── boundary/         # CharacteristicOBC + external-state providers (still
│                     # water, harmonic tide, tidal atlas, NorKyst parent,
│                     # inverse barometer), reflective, clamped tidal,
│                     # discharge, multi-BC dispatch
│
│  # ---- Solver ----
├── solver/
│   ├── state/        # DGSolution1D/2D, SWESolution2D (SoA), Solution3D
│   ├── rhs/          # RHS kernels: scalar/SWE 1D/2D (collocated or entropy-stable
│   │                 # split form), tracer, diffusion (BR1), 3D advection,
│   │                 # baroclinic PGF, 3D RHS assembly
│   ├── limiters/     # Zhang-Shu positivity, Kuzmin/TVB slope limiters
│   ├── algorithms/   # Wetting/drying, tridiagonal (Thomas) solve
│   ├── simd/         # pulp kernels + batched faer paths (parallel feature)
│   ├── diagnostics/  # Runtime diagnostics
│   └── burn/         # Burn GPU prototype — incomplete, deletion recommended
│                     # (REVIEW.md §5; TODO P0.11/P2.7)
├── time/             # Integrable + TimeIntegrator traits, generic SSPRK3,
│                     # mode_split (barotropic subcycling); legacy per-type
│                     # ssp_rk3_* variants slated for deletion (REVIEW.md §6.2)
│
│  # ---- 3D vertical layer ----
├── vertical/         # SigmaGrid, Song-Haidvogel / ROMS Vstretching=4
├── physics/          # PhysicsModule trait, SWEPhysics2D(+builder),
│                     # Hydrostatic3D, vertical mixing/diffusion/velocity, EOS
│
│  # ---- Application layer ----
├── simulation/       # Simulation runner (2D), Simulation3D
├── io/               # NetCDF (nesting/output), VTK, GeoTIFF bathymetry,
│                     # GSHHS coastline, projections, observation readers
└── analysis/         # Harmonic (tidal) analysis, skill metrics, tide gauge,
                      # ADCP, stability monitoring
```

Layering intent (dependencies point up this list): `types` → DG core →
equations/physics → solver → simulation/io/analysis. The io/boundary modules
are the adapter edge; the DG core must stay free of I/O and feature flags
(`netcdf`, `parallel`, `simd` gate adapters and kernels, not math).

## Status by Layer

| Layer | State |
|-------|-------|
| 1D/2D DG core | Solid operators/fluxes/SSP-RK3: verified convergence (P1–P5 1D, P1–P3 2D) on flat/linear-bottom periodic problems, machine-precision conservation for continuous bathymetry, correct Roe/HLL (REVIEW.md §1.10) |
| Well-balancing | `SWEFormulation2D::EntropyStable` (Wintermeyer split form) is well-balanced for any nodal bathymetry, including face jumps, with exact mass conservation and an entropy inequality; the default collocated form stays balanced only for deg B ≤ p/2. The reconstruction mass leak is fixed (1D and 2D). Open: η-based limiting (REVIEW.md §1.4), wetting/drying for the split form (REVIEW.md §1.5) |
| Geometry | General straight-sided quadrilaterals: per-node isoparametric (bilinear) factors, conservative/curvilinear forms in the 2D SWE, tracer, advection and diffusion kernels, free-stream preserving and well-balanced (TODO P1.3). The 3D kernels and the Burn/batched prototypes still require parallelograms (asserted). Open: triangles, curved faces, robust Gmsh MSH 4.1, CSR (REVIEW.md §4.1) |
| Time integration | Generic `SSPRK3` over `Integrable` is the real path; five legacy copies pending deletion |
| Mode splitting | Prototype: Hann filter correct, but the FE subcycle runs inside every RK stage (first-order, damps M2 3–14 %/period), G double-counts advection, stresses never reach the depth mean. Needs a ROMS-style once-per-step forward-backward pass (REVIEW.md §2) |
| 3D physics | Scaffolding: sigma grid, tridiagonal and implicit diffusion solid; tracers not constancy-preserving, vertical advection ×D (fix in review), non-balanced PGF, no GLS; essentially untested (REVIEW.md §3, §6.1) |
| Boundaries/nesting | One characteristic OBC (flux F(q_b)·n, independent of the Riemann solver; reflection ~1e-6), NorKyst boundary tides from a harmonic atlas, `ModelClock`; NorKyst nesting still lacks rotation/transport conservation and a relaxation band (REVIEW.md §4.3–§4.5, TODO P1.5) |
| Performance | SoA + rayon workspace pattern correct; ~47× ROMS core-hours as configured (model estimate): dt estimator, dense volume term, land elements, per-step allocations (REVIEW.md §5) |
| GPU | Burn prototype physically wrong and slower-by-design; fix or replace with custom fused f64 kernels over CSR (REVIEW.md §5; TODO P0.11/P2.7) |

## Evolution Path

Historical phases 0–3 (1D advection → 1D SWE → 2D quads → Norwegian coast
features) are complete; see `TODO.md` for the ledger. The 2026-07-08
correctness batch is done (Burn deferred). Current direction, from
REVIEW.md §7 (2026-09-25):

1. **Correctness batch** — the confirmed small bugs (tidal periods, 3D
   vertical advection, EOS/Chezy units, parallel limiter dry branch, nesting
   time base, dt estimator). Done: Flather applied twice, reconstruction mass
   leak.
2. **Correct 2D barotropic tides** — non-allocating unified RHS first, then
   well-balanced entropy-stable DGSEM (split form done, opt-in
   `SWEFormulation2D::EntropyStable`) with positivity/wet-dry (done: h ≥ 0,
   desingularization, point-implicit friction, HLL default, and the
   shoreline-balanced split form `SWEFormulation2D::WetDry`), η-limiting;
   isoparametric quads +
   triangles; proper NorKyst nesting. Done: one characteristic OBC,
   spatially varying NorKyst boundary tides, model clock.
3. **Cheap enough to matter** — per-element dt, sum factorisation, one Riemann
   solve per face, water-only meshes, then local time stepping or an implicit
   free surface.
4. **Validation and the speed claim** — tide gauges/NorKyst skill, and a
   cost-vs-error benchmark against ROMS 2D on the same hardware.
5. **3D rebuilt on the ROMS recipe** — once-per-step forward-backward
   barotropic pass with consistent averaged transport, Hz-weighted fluxes,
   inventory-form tracers, balanced PGF, GLS mixing, 3D test suite.
6. **Structure** — delete superseded layers, prune API surface, then the
   `dg-core` / `roms-rs` workspace split.

## What NOT to Do

1. **Don't break convergence or conservation** — any spatial/temporal change
   must keep the convergence and conservation tests passing.
2. **Don't allocate in the RHS** — and don't add new APIs shaped
   `fn rhs(&state) -> State`; use write-into signatures.
3. **Don't extend the Burn module** — it is scheduled for deletion; a future
   GPU port goes through flat SoA/CSR data + cudarc-style batched kernels.
4. **Don't add another hand-rolled integrator or run loop** — implement
   `Integrable` and use `SSPRK3`/`Simulation`.
5. **Don't preserve backwards compatibility** — pre-1.0, no external
   consumers: rename/delete cleanly and fix call sites in the same change.

## Testing Strategy

Four tiers, all present in the 1D/2D core and mandatory for new
discretizations (the 3D layer currently fails this bar — REVIEW.md §6.1):

1. **Unit**: operator exactness on polynomials, flux consistency.
2. **Convergence**: observed order ≥ N+1 on smooth solutions, multiple
   resolutions (`tests/convergence_test.rs`).
3. **Conservation**: mass/momentum to machine precision on periodic domains,
   including long-time (50+ period) runs.
4. **Physics regression**: lake-at-rest (must use non-trivial bathymetry),
   dam-break vs exact Riemann, dynamic wet/dry (Ritter dry dam break and
   Thacker bowl in `tests/wet_dry_2d_test.rs`),
   stratified lake-at-rest for 3D PGF (missing).
