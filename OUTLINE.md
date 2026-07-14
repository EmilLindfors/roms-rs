# Architecture

A DG (Discontinuous Galerkin) solver for Norwegian coastal ocean modeling: a
feature-complete 2D barotropic SWE solver plus an early-stage ROMS-style 3D
layer (2D DG horizontal × finite-difference vertical, sigma coordinates, mode
splitting).

For the current health assessment and prioritized fix plan, see `REVIEW.md`
(2026-07-08). For the 3D roadmap, see `3D_TODO.md`. This document describes
what exists and how it fits together.

## Design Philosophy

**Evolve, don't over-engineer.** Start concrete, abstract when patterns emerge.

| Principle | Approach |
|-----------|----------|
| Single crate (for now) | 17 modules, ~75k LOC; a `dg-core` / `roms-rs` workspace split is the likely end state (REVIEW.md §5.7) |
| Concrete first | `Mesh1D`, `Mesh2D`; generic traits (`Integrable`, `SourceTerm2D`) added once 2+ implementations existed |
| faer for LA | Dense element-local operators; QR/SVD available for least-squares (harmonic analysis) |
| f64 everywhere | Coastal stability requires double precision |
| Norwegian focus | Fjords, tides, wetting/drying — not generality |

## Module Structure (current)

```
src/
├── lib.rs            # Public API, re-exports (oversized — REVIEW.md §5.6)
├── types/            # Newtypes: indices, bounds, physical (Depth, Sigma), sides
│
│  # ---- DG core (dimension/equation-agnostic) ----
├── polynomial/       # Legendre P_n, GLL nodes/weights (1D + tensor-product 2D)
├── basis/            # Vandermonde matrices, nodal<->modal (1D + 2D)
├── operators/        # Dr/Ds, mass, LIFT, GeometricFactors2D (affine-only — REVIEW.md §3.1)
├── mesh/
│   ├── core/         # Mesh1D, Mesh2D (quads), Mesh2DBuilder
│   ├── data/         # Bathymetry, land mask, boundary tags
│   ├── io/           # Gmsh MSH 2.2 reader
│   └── traits/       # Mesh traits, Point; MeshGPUData (unimplemented)
├── flux/             # Upwind, Lax-Friedrichs, Roe, HLL; 2D SWE + tracer fluxes
│
│  # ---- Equations & physics ----
├── equations/        # ConservationLaw trait; advection 1D/2D, SWE 1D/2D, UNESCO EOS
├── source/           # SourceTerm1D/2D traits; Coriolis, friction, wind, tidal
│                     # potential, viscosity, bathymetry, sponge; well-balanced
│                     # hydrostatic reconstruction
├── boundary/         # Chapman/Flather, TST-OBC, radiation, reflective, tidal
│                     # forcing, multi-BC dispatch, NorKyst nesting
│
│  # ---- Solver ----
├── solver/
│   ├── state/        # DGSolution1D/2D, SWESolution2D (SoA), Solution3D
│   ├── rhs/          # RHS kernels: scalar/SWE 1D/2D, tracer, diffusion (BR1),
│   │                 # 3D advection, baroclinic PGF, 3D RHS assembly
│   ├── limiters/     # Zhang-Shu positivity, Kuzmin/TVB slope limiters
│   ├── algorithms/   # Wetting/drying, tridiagonal (Thomas) solve
│   ├── simd/         # pulp kernels + batched faer paths (parallel feature)
│   ├── diagnostics/  # Runtime diagnostics
│   └── burn/         # Burn GPU prototype — incomplete, deletion recommended
│                     # (REVIEW.md §4.2)
├── time/             # Integrable + TimeIntegrator traits, generic SSPRK3,
│                     # mode_split (barotropic subcycling); legacy per-type
│                     # ssp_rk3_* variants slated for deletion (REVIEW.md §5.3)
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
| 1D/2D DG core | Solid: verified convergence (P1–P5 1D, P1–P3 2D), machine-precision conservation, correct Roe/HLL, tested normals/connectivity |
| Well-balancing | Works only for low-degree bathymetry; needs Audusse/entropy-stable DGSEM rework (REVIEW.md §1.1–§1.4) |
| Geometry | Affine parallelogram quads only; per-node isoparametric factors + triangles are the strategic unlock (REVIEW.md §3.1) |
| Time integration | Generic `SSPRK3` over `Integrable` is the real path; five legacy copies pending deletion |
| Mode splitting | Prototype: FE subcycle, flat averaging — needs forward-backward stepper + shaped filter (REVIEW.md §1.5) |
| 3D physics | Early: sigma grid/stretching/tridiagonal solid; PGF, mixing, wall BCs and coupling have known bugs; essentially untested (REVIEW.md §2, §5.2) |
| Boundaries/nesting | Coherent Chapman-Flather set; tidal astronomy incomplete, NorKyst ingestion has blocker-level bugs (REVIEW.md §3.3–§3.4) |
| Performance | SoA layout + rayon workspace pattern correct; RHS-returns-by-value API is the allocation bottleneck (REVIEW.md §4.1) |
| GPU | Burn prototype physically wrong and slower-by-design; delete, keep SoA/CSR readiness (REVIEW.md §4.2) |

## Evolution Path

Historical phases 0–3 (1D advection → 1D SWE → 2D quads → Norwegian coast
features) are complete; see `TODO.md` for the ledger. Current direction, from
REVIEW.md §7:

1. **Correctness batch** — the seven top bugs (hardcoded diffusion depth,
   barotropic PGF double-count, 3D wall BCs, NorKyst layer index, tidal phase
   sign, featureless build, Burn RHS).
2. **Non-allocating RHS** — `compute_rhs_into` + workspace integrator; unify
   serial/parallel RHS into one per-element kernel.
3. **Numerics upgrades** — well-balanced entropy-stable DGSEM + η-based
   limiting; forward-backward barotropic mode splitting.
4. **Geometry generalization** — per-node geometric factors, triangles, CSR
   connectivity (also the GPU-readiness work).
5. **3D hardening** — density-Jacobian PGF, GLS mixing, 3D test suite
   (stratified lake-at-rest, seiche, Thacker bowl, lock exchange).
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
discretizations (the 3D layer currently fails this bar — REVIEW.md §5.2):

1. **Unit**: operator exactness on polynomials, flux consistency.
2. **Convergence**: observed order ≥ N+1 on smooth solutions, multiple
   resolutions (`tests/convergence_test.rs`).
3. **Conservation**: mass/momentum to machine precision on periodic domains,
   including long-time (50+ period) runs.
4. **Physics regression**: lake-at-rest (must use non-trivial bathymetry),
   dam-break vs exact Riemann, dynamic wet/dry (Thacker bowl — missing),
   stratified lake-at-rest for 3D PGF (missing).
