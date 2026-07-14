# Full Code & Numerics Review

**Date:** 2026-07-08
**Scope:** Whole-crate review — DG numerical core, 3D/vertical physics, mesh & boundary conditions, solver/performance/GPU, architecture/API/testing — with the Norwegian-coast 3D end goal as the yardstick.
**Method:** Five parallel source-level sub-reviews plus direct verification of key paths (`mode_split.rs`, `baroclinic.rs`), a clean compile check, and a full test run (1,091 passed, 0 failed, 43 ignored, `--no-default-features --features parallel,simd`).

This document **supersedes** `NUMERICAL_REVIEW.md` (all of its findings re-verified below; see §1.0) and `GPU_EVALUATION.md` (its conclusion is corrected in §4.2). It also revisits the allocation conclusion in `PROFILE.md` (§4.1).

---

## Verdict

The 1D/2D DG core is genuinely good — correct weak form, correct Roe/HLL solvers with entropy fix and dry-state handling, correct SSP-RK3, real convergence and conservation tests, a properly structured rayon parallel path, and the right SoA memory layout.

The debt is concentrated in exactly the layers that matter for the end goal: **the 3D path has three outright correctness bugs and essentially zero tests**, the well-balancing story quietly fails for realistic bathymetry, the mode-splitting scheme is unconditionally unstable in theory, and the real-data pipeline (NorKyst nesting, tides, coastline) has bugs that would make a first realistic run silently wrong. Several superseded APIs were never deleted after each abstraction upgrade, which is now the main structural drag.

Strategic framing: the TODO's call — "high-accuracy 2D barotropic model first, 3D incrementally" — is right, but the current well-balancing + affine-geometry limitations mean the 2D model is not yet the differentiator it is positioned as. The well-balanced entropy-stable DGSEM + isoparametric geometry pair is what makes "DG handles fjord geometry better than ROMS" actually true rather than aspirational.

---

## Top correctness bugs (fix first — all cheap)

1. **Vertical diffusion uses a hardcoded 100 m depth for every column** — `src/physics/vertical_diffusion.rs:49` (`let h = 100.0; // TODO`). All layer thicknesses, mixing-closure depths, and implicit-solve coefficients are wrong except in exactly-100 m water. Fjords are 50–1300 m. Actual depth is available via `state.eta` + `bathymetry.water_depth`, exactly as the PGF code does. One-line fix.

2. **Mode splitting double-counts the barotropic pressure gradient.** The 3D RHS produces the *full* PGF (`src/solver/rhs/rhs_3d.rs:62`); the splitter depth-averages it into the G-term forcing (`src/time/mode_split.rs:113`), but the 2D sub-model's `½gh²` flux already contains `−g∇η`. The barotropic mode is driven by the surface gradient twice, pushing wave speeds toward √(2gH). Fix: build G from advection + Coriolis + *baroclinic-only* PGF. Validate: compare the `minimal_3d` seiche period against `2L/√(gH)`.

3. **Coastal walls are transmissive in the 3D path.** Both the omega diagnostic (`src/physics/vertical_velocity.rs:142-158`) and 3D momentum advection (`src/solver/rhs/advection_3d.rs:285-289`) set exterior = interior at physical boundaries, so continuity and momentum leak through land. In a fjord domain nearly every boundary is a wall. Needs the mirrored-normal reflective state (`u_ext·n = −u_int·n`, tangential preserved).

4. **NorKyst nesting reads the wrong ROMS vertical layer** — `src/io/netcdf_io.rs:1337-1348` takes index 0 as "surface", but ROMS s-coordinates are bottom-up: the child is forced with *seabed* velocities. Related nesting bugs in §3.4.

5. **The Burn GPU RHS is physically wrong** — `src/solver/burn/rhs.rs:172-204` computes the HLL flux and then discards it (`// TODO: Implement full surface term computation`). No surface term means no inter-element coupling: it does not solve the SWE. Masked by a lake-at-rest-only test at tol 0.1 (`src/time/burn_ssp_rk3.rs:289`) where the missing term vanishes. See §4.2.

6. **Tidal phases are likely time-reversed**: the Greenwich phase lag is *added* rather than subtracted (`src/boundary/tst_obc.rs:93,146`, `src/io/constituent_reader.rs:115-121`), and there are no nodal corrections (f, u) or equilibrium argument V₀ — absolute phase is arbitrary and K1/O1 amplitudes can be off by 11–19% depending on year. Directly blocks tide-gauge validation (TODO P1.5).

7. **`--no-default-features` does not compile** — `src/solver/simd/batched.rs:213` calls `rayon::current_num_threads()` but `mod batched` is compiled unconditionally (`src/solver/simd/mod.rs:8`) and `rayon` is optional. Gate `batched` on `all(feature = "simd", feature = "parallel")` or make `simd` pull rayon.

---

## 1. DG numerical core

### 1.0 Status of prior NUMERICAL_REVIEW.md findings

| Prior finding | Status |
|---|---|
| 3D bathymetry sign convention opposite to 2D (BLOCKER) | **FIXED** — `water_depth = eta - B` consistent at `src/mesh/data/bathymetry_2d.rs:168-170`, `src/physics/hydrostatic_3d.rs:212-213`, `src/solver/rhs/baroclinic.rs:173-179`, `src/physics/vertical_velocity.rs:64,115,137` |
| Horizontal diffusion/viscosity not conservative DG (BLOCKER) | **FIXED** — shared BR1 lifted-gradient operator with central face fluxes |
| Wet/dry positivity not mass-conservative (BLOCKER) | **FIXED** — `wetting_drying.rs:294,443,494` rescale pre-correction element mass; dry-element limiter preserves nonnegative average (`limiters/swe_2d.rs:123`); locked by regression tests |
| Mode split mislabeled order-3 SSP (HIGH) | **FIXED (labeling only)** — `mode_split.rs:399` reports order 1, non-SSP. The FE subcycle itself is unchanged (§1.5) |
| General quads accepted, geometry affine-only (HIGH) | **FIXED (guarded)** — `geometric.rs:160` rejects non-parallelogram quads. Isoparametric metrics still missing (§3.1) |
| 3D tracer advection non-conservative (HIGH) | Partially fixed per prior doc; momentum still not thickness-weighted (§2.5) |
| Nonlinear GLL aliasing unaddressed (MEDIUM) | **STILL PRESENT** (§1.6) |

### 1.1 [MAJOR] Well-balanced only for bathymetry degree ≤ p/2 per element

`src/solver/rhs/swe_2d.rs:355-383` (volume) + `src/source/swe_2d/bathymetry.rs:82,89`.
The volume term differentiates nodal `½gh²` with collocation Dr/Ds; the source is pointwise `−gh∂B/∂x`. At lake-at-rest these cancel nodally only if Dr differentiates `½gh²` exactly, i.e. `2·deg(h) ≤ p`. Since deg(h) = deg(B) in-element: p=3 with linear B → balanced (what the tests use); p=1 with linear B → **not** balanced (untested); realistic per-node bathymetry at deg = p → never balanced for p ≥ 1. Steep-fjord runs get spurious lake-at-rest currents.
**Fix:** Audusse interface hydrostatic-source formulation, or well-balanced entropy-stable flux-differencing DGSEM (Wintermeyer et al. 2017) — the latter also fixes §1.6.

### 1.2 [MAJOR] `with_well_balanced` doc tells users to drop the source term the volume balance needs

`src/solver/rhs/swe_2d.rs:50-53,142-153`. Doc says "do NOT include BathymetrySource2D… handled through flux reconstruction." Wrong: reconstruction (`:477`) only zeroes the surface flux difference; the volume pressure term (`:380`) still leaves `rhs_hu = +gh∂B/∂x`. Every passing well-balanced test (`:1487,1532,1573`) in fact ADDS `BathymetrySource2D` ("we need BOTH"). A doc-following user silently gets non-well-balanced physics. Fix the doc; ideally auto-install or assert the source.

### 1.3 [MAJOR] Hydrostatic reconstruction feeds reconstructed state into the interior flux

`src/solver/rhs/swe_2d.rs:493-497` (serial), `:1021-1022` (parallel). Strong form is `LIFT·(F_int − F*)` but `F_int` is evaluated on the reconstructed state. SBP telescoping (discrete mass conservation) requires `F_int` to be the actual nodal flux. When neighbouring elements carry different bathymetry at a shared face, element mass balance ≠ −∮F*·n. Currently **latent**: all tests use globally continuous B, so reconstruction is a no-op. Production GeoTIFF/Gmsh per-element bathymetry is face-discontinuous and activates it. No test exercises discontinuous B. The Audusse interface-source form (§1.1) fixes this too.

### 1.4 [MAJOR] Kuzmin slope limiter operates on depth h, not free surface η

`src/solver/limiters/swe_2d.rs:256-319` (averages `:41`). At lake-at-rest over a slope, η is constant but h varies element-to-element, so `compute_kuzmin_alpha` (`:224`) activates, flattening h and generating spurious momentum. Must limit η = h + B (plus velocities) and reconstruct h. The Zhang–Shu positivity limiter (`:110`) is fine (only shrinks toward the average).

### 1.5 [MAJOR] Barotropic subcycle is Forward Euler — unconditionally unstable for gravity waves

`src/time/mode_split.rs:133-140,285-291`. Barotropic SWE eigenvalues are purely imaginary; FE has |1+iθ| > 1 for all θ > 0 — it amplifies at *any* substep size and survives only on dissipation elsewhere. This is exactly why ROMS uses forward–backward stepping. Additional issues:
- The averaging is a flat right-endpoint arithmetic mean (`:144-153`), not a shaped (cosine/power-law) filter with moment constraints — under-damps barotropic→baroclinic aliasing (Shchepetkin & McWilliams 2005).
- The subcycle runs inside *every* RK3 stage (3× ROMS's once-per-step cost).
- `step` and `step_with_stage_hook` are ~150 duplicated lines — collapse before rewriting.
- 14 `clone()`s per outer step (see §4.1).

**Fix:** forward–backward (or generalized FB) barotropic stepper, subcycled once per baroclinic step, shaped averaging window, G-term without the barotropic PGF (top bug #2).

### 1.6 [MEDIUM] Nonlinear GLL aliasing unaddressed

`src/operators/operators_2d.rs:110-149` + `src/solver/rhs/swe_2d.rs:355-383`. Nonlinear fluxes (hu², ½gh², huv) evaluated at GLL nodes and differentiated directly; no overintegration, split form, entropy-stable flux differencing, or modal filter anywhere. Strong straits currents can lose order or go energy-unstable, masked by the limiter. Fix: split-form/entropy-stable volume operator (Gassner 2013; Wintermeyer et al. 2017) — composes with the §1.1 fix.

### 1.7 [MINOR]

- Zhang–Shu positivity uses the GLL nodal min only (`limiters/swe_2d.rs:131-143`); the full guarantee needs the min over the complete quadrature set including interior Gauss points. Common DGSEM practice, but not the full proof.
- CFL element size uses `√det_J·2` (`swe_2d.rs:584,626`) — anisotropy-blind; over-estimates stable dt for long-narrow fjord elements. Prefer min-edge or `det_J ÷ max surface_J`.

### 1.8 Verified correct (don't touch)

- Roe 2D (`flux/swe_2d.rs:58`): wave strengths, shear wave, dissipation, Harten–Hyman entropy fix (`:175`) all correct.
- HLL 2D (`:198`): Einfeldt speeds with correct dry-state estimates (`:295,301`).
- Strong-form DG surface `+J⁻¹·LIFT·sJ·(F_int−F*)` (`:500-511`) and volume metric contraction (`:380`).
- Geometric factors: inverse-Jacobian, outward CCW normals, surface Jacobians (unit-tested).
- SSP-RK3 Shu–Osher coefficients and stage times (`ssp_rk3.rs:60`); convergence confirmed P1/P3/P5 (1D), P2/P3 (2D advection), P1/P2 (2D SWE).
- Coriolis sign `[0, +fhv, −fhu]` (`shallow_water_2d.rs:245`).

---

## 2. 3D / vertical physics

### 2.1 [BLOCKER] Hardcoded 100 m depth in vertical diffusion

Top bug #1 above. `src/physics/vertical_diffusion.rs:49-53,171,199-204`.

### 2.2 [MAJOR] Baroclinic PGF is the standard (non-balanced) sigma form

`src/solver/rhs/baroclinic.rs` uses `∇p|_z = ∇p|_σ + gρ∇z|_σ` (doc `:23-27`, code `:183-185`). The two large cancelling terms are discretized inconsistently — pressure by analytic midpoint integration then DG-differentiated (`:124-141,152-161`) vs analytic `gρ∇z|_σ` (`:178-179`) — the classic sigma-coordinate PGF error producing spurious along-slope currents over steep topography. `3D_TODO.md:144-150` itself flags the balanced method as "CRITICAL for fjords". The only test (`:227-278`) uses **constant density**, where the error cannot appear.
**Fix:** Shchepetkin–McWilliams (2003) density-Jacobian or Berntsen balanced form, plus a **stratified lake-at-rest** regression test.

Additional note: the PGF's η-gradient is computed with element-local (strong-form) derivatives and no face terms — inconsistent across element boundaries for a DG field; fold into the balanced-PGF rework.

### 2.3 [MAJOR] Mode-split coupling double-counts barotropic PGF

Top bug #2 above. `src/solver/rhs/rhs_3d.rs:62,71` + `src/time/mode_split.rs:113-140,174-190`. The 3D side of the correction (`rhs_total = rhs_slow − G + d_bar_eff`) is correct; only the 2D forcing is over-driven.

### 2.4 [MAJOR] Closed coastal boundaries transmissive in omega and 3D momentum advection

Top bug #3 above. `src/physics/vertical_velocity.rs:142-158`, `src/solver/rhs/advection_3d.rs:285-289`.

### 2.5 [MEDIUM]

- **`hc` silently ignored**: `SongHaidvogelStretching` stores/prints `hc` (`src/vertical/stretching.rs:146,244`) but `cs_function` (`:177-217`) never uses it; the transform `z = η + (η+H)σ` (`src/vertical/sigma.rs:199-201`) is pure Vtransform=1-style terrain-following. Implement hc modulation or remove the field.
- **Momentum advection not layer-thickness weighted** (`advection_3d.rs:127-130,606-643`), unlike tracers (correctly on `Hz·C` inventory form, `vertical_velocity.rs:405-406,425`) and unlike the Hz-weighted omega it pairs with — discrete consistency/conservation gap. Advect `Hz·u`/`Hz·v` with the same inventory-then-÷Hz structure.
- **Surface temperature flux driven by buoyancy flux** (`vertical_diffusion.rs:117`) — units m²/s³, should be `Q_net/(ρ₀·c_p)`. **Bottom stress is a single domain-wide constant** (`vertical_mixing.rs:22,88,100`) — not quadratic-drag, not velocity-dependent.
- **Vertical mixing thin for stratified fjords**: only `ConstantMixing` + `PacanowskiPhilander`; no KPP/GLS, no convective adjustment (unstable stratification clamps to Ri=0 → ν₀≈10⁻² m²/s, too weak). GLS k-ε (NorKyst's choice) is the right target — a per-column 1D problem.

### 2.6 [MINOR / SUGGESTION]

- **Two divergent EOS**: `src/equations/equation_of_state.rs` is complete UNESCO EOS-80 (secant bulk modulus), but the 3D path wires the truncated placeholder `UnescoEOS` in `src/physics/eos.rs:110-126` (drops higher-order S-T terms, ignores pressure, `:95,144`). Delegate the physics trait to `equations::EquationOfState` with a `LinearEOS` fast path.
- `Hydrostatic3D::compute_dt` hardcodes internal-wave speed 2.0 m/s (`src/physics/hydrostatic_3d.rs:310-315`); should be ~NH/π from stratification.
- `post_process` labels grid-relative Ω as vertical velocity w (`hydrostatic_3d.rs:329-348`) — relabel or convert to physical w for output.
- **Tridiagonal solver verified correct** (`src/solver/algorithms/tridiagonal.rs`) — standard Thomas on a diagonally-dominant backward-Euler system; BCs wired and sign-consistent. Only issue is §2.1 feeding it wrong Δz.
- **Omega diagnostic verified correct** apart from §2.4: Hz-weighted divergence integrated from bottom with the linear correction enforcing Ω(surface)=0 (`vertical_velocity.rs:196-233`) — standard ROMS omega, both kinematic BCs satisfied.

---

## 3. Mesh & boundary conditions

### 3.1 [BLOCKER] Geometry pipeline is affine-parallelogram-quad only

`src/operators/geometric.rs:160-186` panics on any non-parallelogram quad; Jacobian stored one-constant-per-element (`:210-213`; `mesh2d.rs:1024-1044` evaluates det at center only). A coastline-fitted curvilinear mesh — the entire point of DG for fjords — cannot run at all. Also quad-only: `elements: Vec<[usize;4]>` (`mesh2d.rs:73`), no triangle type exists, and the Gmsh reader silently drops triangles.
**Fix (strategic):** per-node geometric factors (isoparametric bilinear quads) — arrays go `[K]` → `[K × n_nodes]`, mechanical — plus triangle support (recommended for realistic coastline meshing). This also removes the affine caveat from cell averages and limiters.

### 3.2 [MAJOR] Connectivity is AoS/nested-Vec; `MeshGPUData` implemented by nothing

`edges: Vec<Edge>` with `Option<ElementFace>`/`Option<BoundaryTag>` (`mesh2d.rs:37-47`); `vertex_to_elements: Vec<Vec<usize>>` (`:105`) = one heap alloc per vertex. Won't scale to millions of elements or upload to a GPU. `MeshGPUData` (`src/mesh/traits/mesh_traits.rs:342`) is defined and re-exported but has no impl, while the burn module keeps its own duplicate SoA connectivity (`src/solver/burn/connectivity.rs:49`). Flatten to CSR; make `MeshGPUData` the single representation.

### 3.3 Open boundary conditions

The intended coherent set exists (`ChapmanFlather2D` dispatched per tag by `MultiBoundaryCondition2D`, weak ghost-state imposition, correct rotation math and `√(g/h)` Flather factor, no allocation in ghost paths). But labels overstate the physics:

- **[MAJOR]** "Chapman" (`src/boundary/chapman.rs:132-137`) is not Chapman (1985) — it blends *external* η with interior at α=1/(1+cfl) (≈0.8 external at cfl 0.25), so interior waves don't radiate. Acceptable as a forced-elevation BC but mislabeled. Rename or implement the real update.
- **[MAJOR]** `tst_obc.rs:255-279` "tidal/subtidal" split double-counts: both Flather and "subtidal" terms are driven by the identical `(η_int − η_tidal)` with no frequency separation; not Blayo–Debreu (2005) — no characteristic classification exists anywhere. Implement a running-mean subtidal filter or relabel honestly.
- **[MAJOR]** Tidal forcing astronomy incomplete + phase sign (top bug #6). `constituent_reader.rs` reads a custom CSV of harmonic constants, not TPXO/OTPS; no spatially varying tidal grid.
- **[MINOR]** `Radiation2D` (`boundary_2d.rs:283-285`) is bathymetry-blind Flather-in-depth driven by `(h_int − h_ext)` — over sloping bathymetry it responds to bed slope. Drive from η. No real Orlanski exists despite docs.
- **[MINOR]** Chapman CFL silently defaults when `ctx.dt` unset (`chapman.rs:132`) and no reviewed BC populates `ctx.dt` — confirm the RK driver sets `BCContext2D::dt` per stage, else every Chapman boundary runs an arbitrary 50/50 blend.

### 3.4 [BLOCKER/MAJOR] NorKyst nesting

- **[BLOCKER]** Wrong vertical layer (top bug #4): `netcdf_io.rs:1337-1348` reads index 0 = seabed as "surface". Fix: `n_depth−1`, or better read ubar/vbar.
- **[MAJOR]** Barotropic child forced with a single-layer velocity (`ocean_nesting.rs:143-144`); reader never seeks ubar/vbar (`netcdf_io.rs:1073-1086`). Use depth-averaged transport.
- **[MAJOR]** No C-grid destaggering or grid-angle rotation (`netcdf_io.rs:1075,1082`) — NorKyst is a rotated grid; raw u/v are grid-relative on staggered points. Accept only `*_eastward`/`*_northward` or destagger + rotate by `angle`.
- **[MAJOR]** Bilinear weights assume axis-aligned cells (`netcdf_io.rs:1547-1548`); on the rotated grid denominators collapse to the 1e-10 clamp near N-S edges → silent nearest-corner. Solve the inverse bilinear map (2-D Newton).
- **[MAJOR]** Parent time base never aligned: CF units/epoch unparsed, `find_bracket` clamps out-of-range → forcing silently freezes at the first snapshot (`netcdf_io.rs:1206-1216,1721-1727`). Parse CF units; error on out-of-range.
- **[MAJOR (design)]** No relaxation/sponge zone at all — both nesting BCs force only the single ghost layer, tangential vel and η pure Dirichlet (`nesting_bc.rs:196`, `ocean_nesting.rs:223`) → obliquely incident disturbances reflect. Add a Davies (1976) flow-relaxation band over N cells. (`SpongeLayer2D` exists as a source term — wire it to nesting.)
- **[MAJOR (perf)]** Parent fields are `Vec<Vec<Vec<f32>>>` with interpolation stencils recomputed every RK stage behind an RwLock HashMap (`netcdf_io.rs:985-993,1576-1582`). Flatten to `[time,y,x]` strided arrays; precompute per-boundary-node stencils at setup.
- **[MINOR]** `nesting_bc.rs:173-179` passes bathymetry, not parent η, as `expected_elevation`, defeating the diagnostic.

### 3.5 [MAJOR] Coastline / land mask / projection

- GSHHS holes discarded: every ring flattened to a solid exterior polygon (`coastline.rs:82-95`) — inner basins/lakes become land. Match `PolygonRing::{Outer,Inner}`.
- Point-in-polygon is O(polygons·vertices) per query with no spatial index (`coastline.rs:128`) — intractable for millions of nodes vs full-res GSHHS Norway. Add an R-tree/bbox pre-filter.
- **Coriolis f is hardcoded, never computed from latitude** — `standard_norwegian()` = 1.2e-4 f-plane; true f varies ~12% over 57.5–71.5°N. Add `coriolis_from_latitude`/beta helpers.
- Production uses `LocalProjection` (frozen-cosine equirectangular), not the test-only `UtmProjection`. Fine under ~50 km; ~3% E-W scale spread for Frøya-scale domains. Document the limit; use UTM-33N (EPSG:32633 — mainland standard; the doc wrongly suggests 32N) or Lambert conformal for coast-scale.

### 3.6 [MAJOR] Gmsh reader robustness

Advertises MSH 4.x but only parses 2.2 (`gmsh.rs:133-135`); assumes contiguous node tags 1..=N (`:161-184,254-257` — sparse tags → silently wrong coords, id 0 → underflow panic); `.parse().unwrap()` on untrusted input (`:254-257`); triangles silently dropped with a misleading error (`:273-278`).

### 3.7 Verified correct

Outward-normal convention consistent between on-the-fly and precomputed paths (unit-tested all four faces); edge orientation ±1 and neighbor left/right logic correct (`mesh2d.rs:797-806,941-967`); periodic wrap correct and tested (`:619-735`). Minor: `BoundaryTag::Periodic(u32)` is dead — remove or wire up; `physical_to_reference` Newton lacks max-iter/det guards (`mesh2d.rs:990-1019`).

---

## 4. Solver core, performance, GPU

### 4.1 [MAJOR] The real allocation bottleneck is the API shape, not the kernels

`PhysicsModule::compute_rhs(&self, state, t) -> S` (`src/physics/traits.rs:73`) and the integrator's `F: FnMut(&S, f64) -> S` (`src/time/integrator.rs:139-141`) force a full solution allocation per RHS evaluation; `SSPRK3::step` adds two state clones (`integrator.rs:235,241`). Net ≈ 5 full `SWESolution2D` allocations per step ≈ 70 MB alloc/free per step at 65k-element P2 scale. This — not the tiny per-face Vecs `PROFILE.md:214-235` investigated — is almost certainly the "17% system time" in the profile. `Integrable`'s doc even says "RHS should not allocate" while the signature forces it. `mode_split.rs` has 14 clones per step plus a fresh `Solution3D` per stage.
**Fix:** `compute_rhs_into(&self, state, t, out: &mut S)` + workspace-owning integrator with preallocated stage buffers. Highest-leverage CPU change; do it before the numerics rewrites so the new formulations are written once in the non-allocating shape.

### 4.2 Burn GPU path: recommend deletion

This section supersedes `GPU_EVALUATION.md` (2026-01-18: native CPU 2.0 ms RHS / 9.4 ms RK3 step at 5k P3 elements vs burn-ndarray 18/60 ms; burn-cuda and burn-wgpu failed on f64 matmul). That document's conclusion — "f64 DG on GPU is impractical through Burn" — misattributes the failure:

- **[BLOCKER]** The RHS omits surface terms entirely (top bug #5) — the GPU path never computed valid physics to benchmark.
- **[MAJOR]** Face gather and LIFT do full GPU→CPU→GPU round-trips every stage (`src/solver/burn/surface.rs:99-161,256-303,366-458` — the latter dead code). This alone explains "GPU slower", independent of f64 support.
- The module is a third hand-copy of the SWE physics (`kernels.rs` re-implements fluxes/HLL/Coriolis/friction) — pure drift liability.

**Recommendation:** delete `src/solver/burn/` + `src/time/burn_ssp_rk3.rs`. Keep GPU-readiness where it actually lives: flat SoA storage and CSR connectivity (§3.2). A future real port (cudarc per CLAUDE.md's original intent) wants batched-GEMM volume kernels + a fused surface kernel over CSR faces — burn's tensor granularity is the wrong tool.

### 4.3 [MAJOR] High-level `Simulation` API always runs the serial scalar RHS

`src/physics/builder.rs:79-100` unconditionally calls `compute_rhs_swe_2d` even with `parallel`+`simd` on; the fast parallel path is only reachable by examples each hand-rolling `#[cfg]` selection (`froya_real_data.rs:576-583` etc.). Route the selection through one feature-gated point.

### 4.4 [MAJOR] Serial and parallel RHS are two independent ~600-line implementations

Serial uses scalar `SWEState2D` loops (`swe_2d.rs:362-383,501-511`); parallel uses SIMD kernels (`:835-898,1031-1042`). The equivalence test is the right guard, but serial is the test reference while production runs parallel — drift risk. Factor one per-element kernel; `iter` vs `par_iter` should be the only difference.

### 4.5 [MINOR]

- SIMD `apply_diff_matrix` uses faer only for n_nodes ≥ 20 (`simd/kernels.rs:441,455-473`) — P2 (9) / P3 (16) production runs execute the dominant O(N²) volume term as scalar code; the pulp layer has no diff-matrix kernel. Batch as one `D × X(n×3)` GEMM per element, ideally across elements (`D · [n_nodes, 3K]`).
- `add_br1_viscosity` allocates large buffers in the RHS hot path (`swe_2d.rs:250-251,279,298-299`) when viscosity is on.
- Manning friction has no SIMD path (`simd/kernels.rs:158-190`, scalar cbrt) despite being profiled as a bottleneck.
- Benchmarks measure single-element micro-kernels only; no full-RHS-over-mesh, per-step, parallel-scaling, or allocation benchmark — §4.1 is invisible to CI. Add full-RHS + full-step benches.
- 1D `SystemSolution<N>::element_var` allocates a Vec per call (`core/system_solution.rs:82-88`); the 2D SoA version correctly returns a slice.

### 4.6 Verified good (don't lose)

- `SystemSolution2D`/`DGSolution2D` SoA layout: per-variable contiguous, element-major, inline slice accessors.
- Parallel RHS: `par_chunks_mut` + `for_each_init` + per-thread workspace (`swe_2d.rs:797-1104`) — disjoint outputs, no per-element alloc, no false sharing.
- `combine_derivatives`/`coriolis` pulp kernels: real width-agnostic SIMD with correct scalar tails, tested vs scalar.

---

## 5. Architecture, API, errors, tests

### 5.1 [BLOCKER] Featureless build broken

Top bug #7 (`src/solver/simd/batched.rs:213`).

### 5.2 [BLOCKER] The 3D solver is essentially untested

No `#[cfg(test)]` in `rhs_3d.rs`, `hydrostatic_3d.rs`, `vertical_velocity.rs`; nothing in `tests/`; `simulation_3d.rs:218-232` contains a fake test whose body is comments ("better to just check if it compiles"). CLAUDE.md makes convergence tests mandatory. Delete the fake test; add the §7 test list.

### 5.3 [MAJOR] Delete the superseded time-integration and run-loop layers

`src/time/integrator.rs` already defines `Integrable` + `TimeIntegrator` + generic `SSPRK3` (with `step_with_stage_hook`), and all six solution types implement `Integrable` (`integrator.rs:391-496`). Yet five hand-rolled SSP-RK3 copies persist and are exported (`ssp_rk3.rs`, `ssp_rk3_2d.rs`, `ssp_rk3_swe.rs`, `ssp_rk3_swe_2d.rs`, three more variants in `coupled_swe_tracer.rs:359-543`, plus `burn_ssp_rk3.rs`), and five ad-hoc run loops duplicate `Simulation::run` (`runner.rs:196`). `Simulation3D` (`simulation_3d.rs:79-201`) is a near-verbatim copy of `Simulation::run`; its `where` clause (`:30`) leaks implementation details. Pre-1.0 verdict: delete the legacy free functions, make `CoupledState2D` implement `Integrable`, extract one time loop. Removes ~1,200 lines and a class of variant drift. The `_timed`/untimed split is vestigial — the trait's `step` already takes `t`.

`ConservationLaw` (`src/equations/mod.rs:44`) has the right idea (`const N_VARS`) but allocates `Vec<f64>` per flux/eigenvalue call and the 2D path ignores it — if unified, use `type State = [f64; N]`/`Copy` state, not the trait as-is.

Duplicate config enums: `SWEFluxType2D` (`flux/swe_2d.rs:381`) vs `StandardFlux2D` (`flux/traits.rs:196`); three limiter-selection enums (`solver/limiters/standard.rs:168`, `time/ssp_rk3_swe_2d.rs:36`, `time/coupled_swe_tracer.rs:139`). Collapse each family.

### 5.4 [MAJOR] Error handling gaps

Sound where it exists (per-adapter thiserror enums in `io/`, `#[from]` conversions, no `Box<dyn Error>` in signatures). But:
- `src/analysis/` has no error type — failures on external observation data are asserts/panics (`analysis/mod.rs:106`, `metrics.rs:36-41`, `harmonic.rs:199-205`, `tide_gauge.rs:235-239`, `adcp.rs:354-359`). Expected failures need an `AnalysisError`.
- Reachable panics on user input: NaN time panics (`timeseries_reader.rs:198`); geotiff OOB at bbox edge (`geotiff.rs:236-248`).
- Silent data corruption: NetCDF `read_variable` tries i16 before f32 (`netcdf_io.rs:1238`) — f32 variables truncated; `get_state` silently zeroes missing ssh/u/v (`:1398-1414`); CF time units claim epoch seconds but raw sim time written (`:241` vs `:397`).
- Hand-rolled date math with 30-day months, no leap years (`tide_gauge_reader.rs:308-313`) — corrupts tidal-phase comparison.
- `SimulationResult` is stringly-typed (`runner.rs:60-62` — `success: bool` + `Option<String>`). Make it `Result<SimulationStats, SimulationError>`.
- Library `eprintln!` swallowing errors (`tide_gauge_reader.rs:336`, `tide_gauge.rs:612,624`).

### 5.5 [MAJOR] Analysis numerics

Harmonic fit uses unregularized normal equations with no record-length check (`harmonic.rs:222-247`); K1/P1 need ~182 days (Rayleigh) — short records silently return garbage with R²≈1. `minimum_record_length()` (`:164`) exists but `fit()` never consults it. Use QR/SVD on A directly (faer has both); error/warn on short records.

### 5.6 Feature flags, deps, API surface

- `tiff`, `shapefile`, `geo` are unconditional deps used only by two adapters — gate behind a `geodata` feature.
- ~200 items re-exported flat at the crate root (`lib.rs:48-267`), including deep internals; meanwhile `Simulation3D` is *not* exported. Prune to ~20 workflow types or add a `prelude`.
- ~15 redundant inner `#[cfg(feature = "netcdf")]` in an already-gated module; dead error variants (`NetCDFError::FeatureDisabled`, `VtkError::InvalidMesh`); doc examples use wrong crate name `use dg::...` (all `ignore`d).
- `Depth::new`/`Sigma::new` document "# Panics" but use `debug_assert!` (`types/physical.rs:38-45,199-207`) — release builds accept `Depth(-5.0)`. Make the check real (or fix docs); `new_unchecked` has a "# Safety" section but isn't `unsafe fn`.
- Builder: solid consuming builder, but `SWEPhysics2D` exposes all fields `pub` (builder enforces nothing); single `Option<Arc<dyn SourceTerm2D>>` slot — default to `CombinedSource2D`/`Vec` for composition; unused `PhysicsConfig` trait — delete.

### 5.7 Identity & docs

- Crate `dg-rs`, repo/README `roms-rs`. Decide. Natural end state: a workspace — `dg-core` (polynomial/basis/operators/mesh/flux/time) + `roms-rs` (equations/physics/boundary/io/analysis/simulation). The split also structurally fixes feature-gating and API-surface problems.
- 13 loose markdown files at repo root — consolidate into `docs/`.
- CLAUDE.md's architecture diagram still describes the 1D-only crate — stale.
- Three io text readers duplicate ~150 lines of CSV heuristics with divergent bugs (unreachable header-skip `adcp_reader.rs:126-136`; sentinel `0.0` lat/lon instead of `Option`); `write_vtk_swe`/`write_vtk_coupled` 90% identical (`vtk.rs:344-559`).

---

## 6. Test-coverage gaps (ranked by value)

1. **Stratified lake-at-rest** (3D) — catches the sigma PGF error (§2.2) and future regressions.
2. **Lake-at-rest with p=1 linear B, and with face-discontinuous per-element B** — catches both well-balancing MAJORs (§1.1, §1.3).
3. **`minimal_3d` seiche period vs `2L/√(gH)`** — catches the mode-split PGF double-count (§2.3).
4. **Thacker parabolic bowl** — dynamic wetting/drying (a stated requirement; only static positivity is tested).
5. 3D convergence + tracer/mass conservation tests; barotropic–baroclinic consistency test.
6. Mode-split stability/convergence test.
7. Fix vacuous tests: `test_standing_wave_period` passes with <2 peaks (`tests/validation_2d.rs:182-194`); `test_lake_at_rest` uses flat bathymetry (`tests/swe_2d_test.rs:57-84`); the fake 3D init test (§5.2).

Counterweight: the 1D/2D core is genuinely well tested — real convergence-order assertions at multiple resolutions (`tests/convergence_test.rs:94,132,172,487,766`), machine-precision conservation including 50-period long-time tests, a negative-control limiter test, ~130 inline test modules.

---

## 7. Suggested order of work

**Week-scale (correctness, small diffs):** the seven top bugs — hardcoded depth, PGF double-count, 3D wall BCs, NorKyst layer index, tidal phase sign, featureless build, plus delete-or-quarantine the Burn RHS. Each with a regression test. This is the difference between "demo" and "trustworthy first realistic run".

**Month-scale (numerics that gate the fjord mission):**
1. `compute_rhs_into` + workspace integrator; unify serial/parallel RHS (§4.1, §4.4) — *first*, so the rewrites below are written once in the non-allocating shape.
2. Well-balanced entropy-stable DGSEM volume/source formulation + η-based limiting (§1.1–§1.4, §1.6) — regression tests written first.
3. Forward–backward barotropic stepper, shaped averaging, once-per-step subcycling, baroclinic-only G-term (§1.5, §2.3).

**Quarter-scale (strategic):**
- Per-node geometric factors + triangle support + CSR connectivity (§3.1, §3.2).
- Density-Jacobian baroclinic PGF + stratified lake-at-rest test (§2.2).
- Delete burn, legacy integrators, duplicate run loops (§4.2, §5.3).
- GLS k-ε vertical mixing (§2.5).
- NorKyst nesting done properly: destagger + rotate, CF time base, ubar/vbar, relaxation zone (§3.4).
- Latitude-dependent Coriolis; GSHHS holes + spatial index (§3.5).
- dg-core / roms-rs workspace split; prune the root API surface (§5.6, §5.7).

---

## References

- Audusse et al. (2004), well-balanced hydrostatic reconstruction.
- Gassner (2013), split-form DGSEM.
- Wintermeyer et al. (2017), entropy-stable well-balanced DGSEM for SWE.
- Shchepetkin & McWilliams (2003), density-Jacobian PGF; (2005), ROMS mode splitting and barotropic averaging filters.
- Berntsen (2002), balanced internal pressure gradients in sigma coordinates.
- Davies (1976), flow-relaxation open boundaries.
- Blayo & Debreu (2005), characteristic open boundary conditions.
- Zhang & Shu (2010), positivity-preserving limiters.
