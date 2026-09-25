# TODO - DG Coastal Ocean Model (dg-rs)

This roadmap is driven by the full-crate review of **2026-09-25** in `REVIEW.md`, which supersedes the 2026-07-08 review (in git history). See `reports/` for the 2026-02-11 ROMS/NorKyst comparison, math evaluation, BC assessment and devil's-advocate reports.

**Current state.**
- **What is solid:** the DG core (GLL operators, strong-form DGSEM, Roe/HLL, SSP-RK3, SoA/rayon), the tidal astronomy module, and the NorKyst ingest readers. 1,024 lib + 61 integration + 59 doctests pass (`--no-default-features --features parallel,simd`).
- **What the tests don't cover:** the suite runs in under 2 s, and the physics tests only exercise easy configurations.
- **What the review found broken:**
  - Realistic bathymetry is not well-balanced: metre-per-second spurious currents within an hour. (Since fixed opt-in by the split-form DGSEM, P1.2.)
  - The cell-average workaround leaks mass. (Since fixed, P0.13.)
  - Flather-type open boundaries reflect about 1/3 of outgoing waves. (Since fixed, P0.12.)
  - Coastline-fitted meshes cannot be loaded.
  - Mode splitting is first-order and damps the barotropic mode.
  - The 3D layer is scaffolding with three blocker-level math errors.
- **Cost:** as configured, roughly ~50× ROMS core-hours for a 2D tidal run (cost-model estimate, `REVIEW.md` §5).

**Strategic direction.**
1. **Make the 2D barotropic tide model correct, then cheap, then validated.** This is where DG's fjord-geometry and accuracy-per-DOF advantages can actually beat ROMS.
2. **Then rebuild 3D on the ROMS recipe.**

"Faster than ROMS" is a claim to be *proven* with a cost-vs-error benchmark (P3.2), plausible for 2D tides at P3+ with local time stepping or an implicit free surface. It should not be assumed.

---

## ▶ Next session — start here

Last reviewed: 2026-09-25 (`REVIEW.md`). Pick up in this order:

1. **Priority 0 (2026-09-25): confirmed small bugs.**
   - Each needs a regression test that fails before the fix.
   - P0.15–P0.18 (tidal periods, 3D vertical advection ÷Hz with Ω at w-points, EOS units, Chezy units) are fixed in [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2).
   - P0.12 (Flather), P0.13 (reconstruction mass leak), P0.14 (docs), P0.19 (limiter plumbing), P0.20 (NorKyst time base), P0.21 (time step) and P0.22 (3D tracer wall flux) are fixed. Priority 0 is done apart from the deferred P0.11 (GPU); next is P1.1.
2. **P1.1 non-allocating RHS + unified serial/parallel kernel.** Do this before any numerics rewrite, so the new formulation is written once in the right shape.
3. **P1.2 well-balanced entropy-stable DGSEM** (Wintermeyer et al. 2017/2018). This is the foundational numerics change. Write the P1.7 gating tests first.
   - Split form done ([PR #5](https://github.com/EmilLindfors/roms-rs/pull/5)). Wet/dry robustness is done for the collocated form: h ≥ 0, desingularization, implicit friction, HLL default and the CFL cap.
   - `SWEFormulation2D::WetDry` is done: split-form wet/dry that keeps a shoreline lake at rest exactly.
   - Next: its first-order shoreline elements, η-based limiting, and moving Frøya onto it.
4. **P3.1 validation.** The data pipeline works, but re-run the Bergen NorKyst fit now that P0.15 ([PR #2](https://github.com/EmilLindfors/roms-rs/pull/2)) is merged: it used the truncated periods.

Build note: default features need the `roms-rs` conda env (`conda activate roms-rs`) for HDF5/netCDF. Otherwise use `cargo test --no-default-features --features parallel,simd`.

---

## Priority 0 (2026-09-25 review): Confirmed Correctness Bugs — OPEN

All confirmed in code or by measurement (`REVIEW.md` "Top correctness bugs"). Each is small. Each needs a regression test that would have caught it.

### P0.12 Flather characteristic relation applied twice (BLOCKER, open boundaries) — FIXED: [PR #4](https://github.com/EmilLindfors/roms-rs/pull/4)
- [x] **The bug.** Ghost normal velocity is set to `un_ext + √(g/h)(η_int − η_ext)`, and the Riemann solver then applies the characteristic relation again.
  - Result: reflection coefficient ≈ −1/3; a forced tide arrives at ≈ 0.65 amplitude; −0.57 reflection for the `ChapmanFlather2D` defaults.
  - Affected: `Flather2D`, `HarmonicFlather2D` (`boundary_2d.rs:381,821`), `TSTOBC2D` (`tst_obc.rs:347`), `NestingBC2D` (`nesting_bc.rs:189`), `OceanNestingBC2D` (`ocean_nesting.rs:218`), `ChapmanFlather2D` (`chapman.rs:278`).
- [x] **Fix.** Ghost = external state (η_ext, u_ext) and let the upwind flux impose Flather. `OceanNestingBC2D::with_flather(false)` is already correct.
- [x] **Tests.** Outgoing-pulse reflection < 1 %; delivered progressive-tide amplitude ≥ 0.95. (`tests/open_boundary_flather_test.rs`: reflection < 1e-5, amplitude 0.997–1.001.)

### P0.13 Hydrostatic reconstruction leaks mass when B jumps across faces (BLOCKER, 2D) — FIXED (2D and 1D, [PR #6](https://github.com/EmilLindfors/roms-rs/pull/6))
- [x] **The bug.** `f_int = normal_flux(&q_int_flux)` uses the *reconstructed* state (`swe_2d.rs:494`, parallel `:1021`). Measured: relative mass change −5.5e-5 per second with cell-averaged B and slope-correlated flow; ~1e-20 with continuous B.
  - Active in `froya_real_data` (`to_cell_average` + `with_well_balanced(true)`).
- [x] **Fix.** Use `normal_flux(&q_int)`, keep F* from the reconstructed states, and add ½g(h*⁻² − h⁻²)n to the momentum components only (DG Audusse / Xing–Shu). Apply identically in serial and parallel.
- [x] **Tests.**
  - A periodic mass-rate test with cell-averaged B and u ∝ cos·cos (a uniform u hides the bug by symmetry).
  - Lake-at-rest with cell-averaged B, serial and parallel.

### P0.14 Documentation directs users to the unbalanced configuration (MAJOR, 2D) — FIXED: [PR #11](https://github.com/EmilLindfors/roms-rs/pull/11)
- [x] The `SWE2DRhsConfig::well_balanced` / `with_well_balanced` docs (`swe_2d.rs:50-53,142-153`) say to omit `BathymetrySource2D`. Doing so gives ≈ 0.6 m/s² lake-at-rest residuals with nodal B. The advice holds only for cell-constant B.
- [x] The `Bathymetry2D::linearize()` doc says "linear B is well-balanced", which is false at p = 1 (0.07 m/s² measured).
  - Doc corrected (p ≥ 2 only, face jumps need `with_well_balanced`, split form balanced for any B); `test_lake_at_rest_linearized_bathymetry`.

### P0.15 Truncated tidal constituent periods (MAJOR, tides/analysis) — FIXED: [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2)
- [x] **The bug.** `TidalConstituent::{m2,k1,o1,n2,p1}` (`boundary/tidal.rs:76-113`) use 12.42 / 23.93 / 25.82 / 12.66 / 24.07 h. They feed `HarmonicAnalysis::standard()`/`norwegian_coast()` and the harmonic BCs.
  - Phase error on a 60-day fit: ~1° (M2) to ~7° (P1); >10° over a year.
- [x] **Fix.** Use `constituent_period()` (`constituent_reader.rs:156`) or Doodson speeds.
- [x] **Test.** Fit a *true-frequency* synthetic signal; the phase error must be < 0.1°. The current round-trips synthesise with the same truncated periods.

### P0.16 3D vertical momentum advection too large by a factor of D (BLOCKER, 3D) — FIXED: [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2)
- [x] `apply_vertical_advection_field` divides the Ω·u flux difference by Δσ (`advection_3d.rs:641`). Ω is in m/s, so the divisor must be Hz = D·Δσ; the tracer version (`:688`) already does this.
- [x] Test: a column with known linear Ω and u(σ) at D = 200 m, compared analytically.
- [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2) also moves Ω to w-points (`n_levels + 1` interface values), which covers the w-point item in P4.2.

### P0.17 UNESCO EOS pressure units (MEDIUM, latent) — FIXED: [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2)
- [x] `EquationOfState::density`/`secant_bulk_modulus` (`equation_of_state.rs:144-160,281-307`) feed dbar into the bar-based formula. ρ(5 °C, 35, 1000 dbar) ≈ 1069.5 instead of ≈ 1032.3. Convert p_bar = p_dbar / 10.
- [x] Test against the UNESCO check values (ρ(35, 25 °C, 1000 bar) = 1062.53817; ρ(35, 5 °C, 0) = 1027.67547).

### P0.18 Chezy friction units (MEDIUM, 2D + 1D) — FIXED: [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2)
- [x] `ChezyFriction2D` (`friction.rs:200-239`) and the 1D Chezy compute −C_D|u|u/h with dimensionless C_D. The momentum source is −C_D|u|u (currently 50× too weak in 50 m of water).
- [x] Test the magnitude, not just the sign.

### P0.19 Limiter plumbing (MAJOR, 2D) — FIXED: [PR #11](https://github.com/EmilLindfors/roms-rs/pull/11)
- [x] The parallel fused Kuzmin+positivity limiter (`limiters/swe_2d.rs:686-703`) lacks the serial dry-element branch (`:118-129`): dry cells keep momentum and negative means survive. Add the branch plus a serial-vs-parallel equivalence test with dry cells.
  - Done: serial and parallel limiters now run the same per-element kernels (`positivity_limit_element`, `kuzmin_limit_element`), in place (no `to_vec`/copy-back), and agree bitwise. `StandardLimiter2D` uses the fused kernel and the parallel versions under `parallel`.
- [x] The high-level `Simulation` API limits once per step, not per RK stage (`runner.rs:286-287`, `builder.rs:115-125`). That voids the Zhang–Shu guarantee; move limiting into the stage hook.
  - Done: `step_with_stage_hook` is now a `TimeIntegrator` method (`step` = no-op hook) and `Simulation` passes `post_process` as the hook.

### P0.20 NorKyst nesting time base and ingest (BLOCKER, nesting) — FIXED
- [x] `read_time` returns raw values and `find_bracket` clamps out-of-range times (`netcdf_io.rs:1226-1236`), so the forcing freezes at the first snapshot. Parse CF `units` and add an epoch/offset to `OceanNestingBC2D`. Error on out-of-range times.
  - Done: `OceanModelReader::time` is Unix seconds from CF `units`/`calendar` (missing units, non-Gregorian calendars and non-increasing times are errors). Time interpolation no longer clamps. `OceanNestingBC2D::with_epoch` (default: first snapshot), `check_time_coverage`, and a panic in `ghost_state` outside the file.
- [x] Remove `"h"` (bathymetry) from the SSH candidate names (`netcdf_io.rs:1068`).
- [x] `read_variable` tries i16 before f32 (`:1258`), so unpacked float fields are truncated to integers. Read by declared type.
  - Done: branches on `vartype()`, reads numerics as f64, CF unpacking with `_FillValue`/`missing_value` (netCDF default integer fills when absent).

### P0.21 Time step (CRITICAL for cost, 2D) — FIXED: [PR #11](https://github.com/EmilLindfors/roms-rs/pull/11)
- [x] `compute_dt_swe_2d` (`swe_2d.rs:582-602`, parallel `:620-650`) pairs the global minimum element size with the global maximum wave speed. Use min over elements of hₖ/λₖ with a direction-aware (anisotropic) length.
  - Done: per node, Δt = CFL/(2N+1)·4/(λ_r + λ_s) with λ_r = |u·∇r| + c|∇r|; unchanged on squares at rest. Serial and parallel share one kernel.
- [x] `froya_real_data.rs:163` uses CFL = 0.1 where ~0.5 is stable (5× cost). Also respect the DGSEM positivity bound for wet/dry runs (CFL ≤ 0.75/0.42/0.29/0.23 for N = 1–4 in current units).
  - Done: `positivity_cfl_swe_2d(N)`; with the directional dt the bound holds for any element shape. Froya now runs at it (5/12 at P2). Not yet enforced automatically by the steppers.

### P0.22 3D tracer flux leaks through coastal walls (MAJOR, 3D; P0.7 follow-up) — FIXED: [PR #12](https://github.com/EmilLindfors/roms-rs/pull/12)
- [x] The tracer kernel sets `u_ext = u_int` at physical boundaries (`advection_3d.rs:489-491`), so heat and salt cross the coastline while the mass flux is zero. Use `boundary::reflect_velocity` as for Ω and momentum.
  - Done: reflected velocity **and** the interior tracer/Hz at the ghost, so the Rusanov tracer flux is exactly zero like the volume flux (a ghost value from the BC would still leak through the dissipation term). Test: `horizontal_tracer_advection_closed_basin_conserves_inventory` (was −0.39 of the advective scale).
  - Consequence: `TracerBoundaryCondition3D` (`Hydrostatic3D::temp_bc/salt_bc`) is no longer consulted. It comes back with 3D open boundaries (P4.2).

### P0.11 Burn GPU RHS is physically wrong (BLOCKER, GPU) — DEFERRED
Decision (2026-07-09): keep `src/solver/burn/` behind the `burn` feature for now (burn-cuda is the stated GPU target).
- [ ] `src/solver/burn/rhs.rs` computes the HLL flux, then discards it. There is no surface term, so it is not solving the SWE.
- [ ] Implement the surface term plus a burn-vs-CPU equivalence test before trusting any GPU result.
- See P2.7: the review recommends custom fused f64 kernels over CSR faces rather than Burn tensor ops with fusion disabled (`REVIEW.md` §5.2).

---

## Priority 1: A Correct 2D Barotropic Tide Model (`REVIEW.md` §1, §4, §7 Phase A)

### P1.1 Non-allocating RHS and one kernel — DONE for the `Simulation` path
- [x] `compute_rhs_into(&self, state, t, out: &mut S)` plus a workspace-owning integrator. Today the production stepper allocates ~33 full-field arrays per step (`REVIEW.md` §5.3).
  - Done: `PhysicsModule::compute_rhs_into`, `TimeIntegrator::step_with_workspace` + `StageWorkspace` (the only method an integrator implements), buffer-reusing `clone_from` for the SoA solutions. A warm `Simulation` step allocates 0 B (`tests/allocation_test.rs`; was 1.3 MB per step at 24² P3).
- [x] Unify the serial (`swe_2d.rs:318-563`) and parallel (`:775-1110`) RHS into one per-element kernel. `iter` vs `par_iter` should be the only difference.
  - Done: `SWE2DRhsKernel`; bitwise serial = parallel; per-thread cached, padded scratch. Parallel RHS 1.2–1.7× faster with `_into` (false sharing between cached workspaces cost ~25 % until padded).
- [x] Route the `Simulation`/`SWEPhysics2D` path through the parallel kernel (`builder.rs:99`).
- [ ] Remaining allocations: the limiters allocate the cell-average and vertex-bound `Vec`s every stage (P2.3); horizontal viscosity allocates ~10 field arrays per RHS (`add_br1_viscosity`); the legacy `ssp_rk3_swe_2d` stepper used by the examples still clones stages (P6: move the examples to `Simulation`); `DGSolution1D`/`DGSolution2D`/`TracerSolution2D`/`Solution3D` use the allocating default `clone_from`, and `Simulation3D`/mode splitting do not use the stage workspace (P4.1).
- [ ] Use the new split to specialise hot paths without duplicating them again: e.g. a `const N` (order) generic kernel for the fixed-size matrix products, and skipping the `reference_to_physical` + `SourceContext2D` build for sources that need neither (P2.2).

### P1.2 Well-balanced, entropy-stable, positivity-preserving DGSEM
- [x] Flux-differencing volume term with the Wintermeyer et al. (2017) two-point flux and the g·hᵢ(DB)ᵢ term. It is exactly well-balanced for any nodal B (including face-discontinuous B), conservative, and fixes GLL aliasing.
  - Done ([PR #5](https://github.com/EmilLindfors/roms-rs/pull/5)): opt-in via `SWEFormulation2D::EntropyStable` (`EntropyConservative` for verification). Exact mass conservation, entropy conserved/dissipated to round-off, P2/P3 convergence 3.06/4.00 on a nonlinear manufactured solution with bathymetry.
  - Follow-up: switch `examples/froya_real_data.rs` from cell-averaged B to nodal B + split form.
- [x] Wet/dry and positivity: Zhang–Shu towards h ≥ 0 (not h_min), Kurganov–Petrova velocity desingularization below `h_dry` (1 mm). Done for the collocated (`Standard`) formulation (`solver/algorithms/wetting_drying.rs`, `limiters/swe_2d.rs`). Gates in `tests/wet_dry_2d_test.rs`: Ritter dry dam break (L1 1.5 %, first order), Thacker's paraboloid (L1 7.4 % after one period, was 9.2 %), shoreline lake at rest (|u| < 0.06 m/s where h > 1 cm, was up to the 20 m/s cap). All conserve mass to round-off with h ≥ 0.
- [x] **Shoreline well-balancing and wet/dry for the split form: `SWEFormulation2D::WetDry`.** HLL on Audusse-reconstructed states at faces, flux differencing in wet elements, and subcell finite volumes with the same flux in partially dry elements.
  - A shoreline lake at rest is kept to round-off (P1–P4, smooth/rough beds; the P1.7 gate `lake_at_rest_with_shoreline_is_exact` passes).
  - Fully wet it keeps N+1 (P2 3.06, P3 4.02) and dissipates energy.
  - `Standard` is still not balanced in partially dry elements: on a 2 % beach it settles into a stationary circulation of ≈ 5 cm/s where h > 1 cm, and 1–4 m/s in the film.
- [ ] **`WetDry` accuracy on moving shorelines.** Every element the shoreline crosses drops to first-order subcell FV. On Thacker's paraboloid (P2) this gives 4.8 % at 40², against 1.3 % for `Standard`; both converge at ≈ 2nd order. Options:
  - a second-order (MUSCL) subcell reconstruction that stays positivity preserving and well-balanced;
  - convex DG/FV blending (Hennemann–Gassner / Rueda-Ramírez subcell limiting) with a balanced DG part;
  - using FV only in elements with a dry node that is not at rest.
- [ ] Guide the choice of `h_dry`. It should be about 1e-4 of the characteristic depth: the 1 mm default suits field scale, but it dominates the error of the 0.1 m Thacker case (7.4 % vs 4.5 %). Consider deriving it from the bathymetry range, or making it relative.
- [ ] The Chen–Noelle (2017) reconstruction instead of Audusse. It is more accurate for bed steps larger than the depth (thin films on steep subcell beds, e.g. fjord walls); it changes only B* and h* (`b* = min(max B, min η)`, `h* = min(η − b*, h)`).
- [ ] Dry fronts lag: in the Ritter test the h = 1 mm front is at 71.5 m instead of 79.8 m (P2, 1 m elements; the same on `main`). It moves only to 73 m at 0.5 m resolution, or 75 m with h_dry = 1e-5. The thin-layer relaxation plays no role. Suspects: strict Kuzmin at the front, and zeroing the momentum of elements with mean < h_dry. Revisit with η-based limiting.
- [ ] Positivity alone does not control a dry front: without a slope limiter, P1/P2 dam breaks shed a thin film that runs ahead at the 20 m/s velocity cap. Partially dry cells need limiting (next item).
- [ ] Limit η (and velocities or characteristic variables), not h, with a troubled-cell indicator (TVB-M/KXRCF). Limit h only in partially dry cells (Vater et al. 2019).
  - Today, strict Kuzmin on h with cell-average B collapses sloping elements to P0 on every flood/ebb half-cycle.
- [x] The positivity CFL is enforced: `PhysicsModule::max_cfl`, applied by `Simulation`; `SWEPhysics2D` reports `positivity_cfl_swe_2d(N)` when it has wetting/drying. Roe with wetting/drying prints a warning at build time.
- [x] Negative-mean elements are counted, not hidden: the positivity limiters and wet/dry correction return the count; `SWEPhysics2D::negative_depth_clips`.
- [x] Point-implicit friction in every RK stage (`TimeIntegrator::step_with_relaxation`, `BottomFriction2D`, `SWEPhysics2DBuilder::with_implicit_friction`). It keeps friction balances exact for any dt, is L-stable, and is first order in the friction term (13–32 % above the exact decay at dt·Λ ≈ 3; explicit flips the sign there).
  - [ ] `SpatiallyVaryingManning2D` does not implement `BottomFriction2D`: the rate needs the node position. Precompute a per-node n field (also avoids the closure call per node).
  - [ ] A second-order IMEX treatment (e.g. an SSP IMEX-RK) if the friction term's first-order error matters in shallow tidal flats.
- [x] The dt-dependent wet/dry momentum damping is gone. It is replaced by the point-implicit thin-layer relaxation r(h) = (h_dry/h − 1)²/τ, which is zero at h ≥ h_dry.
- [x] HLL is the default flux for wet/dry runs in `SWEPhysics2DBuilder`. `SWE2DRhsConfig::new` still defaults to Roe (low-level API).
- [ ] Retire the cell-average/`linearize` workarounds: `WetDry` now covers wet/dry with nodal B.
- [ ] Move `examples/froya_real_data.rs` off the legacy `SWE2DTimeConfig`, which sets `H_MIN` = 5 m and makes land a 5 m film. Use `Simulation` + `WetDry` (nodal B) + `with_wet_dry` + `with_implicit_friction`, and mesh water only (P2.4).
- [ ] `SWEPhysics2D::post_process` builds a `LimiterContext2D` every stage, whose `new` computes `mesh.h_min()` (a sqrt per edge over all elements). No limiter reads `LimiterContext2D::h_min`: drop the field or cache it.

### P1.3 Geometry for real coastlines
- [ ] Per-node isoparametric geometric factors (`[K]` → `[K × n_nodes]`). Remove the parallelogram-only panic (`geometric.rs:160-186`).
- [ ] Triangles (or quad-dominant meshes), and remove the hardcoded 4 faces (`swe_2d.rs:386,908`).
- [ ] Gmsh: real MSH 4.1 parsing, sparse node tags, no `.unwrap()` on input, error (not drop) on unsupported element types.
- [ ] CSR connectivity. Make `MeshGPUData` the single representation.

### P1.4 Open boundaries and tidal forcing
- [ ] Spatially varying tidal forcing: a TPXO/FES reader, or boundary harmonics from NorKyst. Uniform-phase forcing is wrong by ~30° of M2 phase along 200 km.
- [ ] A `ModelClock { epoch_unix }` threaded through BCs (`OceanNestingBC2D::with_epoch` is the first per-BC epoch; fold it in), forcing, NetCDF output (writer time units, `netcdf_io.rs:241` vs `:397`) and validation (`tide_gauge.rs:234-239` compares by length only).
- [ ] Nodal f/u at the run or record midpoint, not frozen at the epoch; guard against double application.
- [ ] **One characteristic OBC.** Flather, Chapman, radiation, TST and nesting differ only in the external data that sets the incoming invariant `w− = u_n − 2√(gh)`; the outgoing `w+` always comes from the interior. Replace the per-type ghost logic with:
  - a single `CharacteristicOBC` that builds the boundary state from the invariants (`u_b = ½(w+_int + w−_ext)`, `√(g h_b) = ¼(w+_int − w−_ext)`, u_t from the upwind side) and evaluates the flux as `F(q_b)·n`, so the result does not depend on the Riemann solver (ghost = external state is exact only for Roe; Lax–Friedrichs matches only at u = 0). The 1D `RadiationBC` (`radiation.rs`) already builds this state;
  - external-state providers (constant/still water, harmonic tide, time series, nesting interpolation) in place of the BC zoo. Drop the finite-difference remnants: Chapman's elevation blend (a reflecting clamp in DG), TST's extra "subtidal" velocity, `NestingBC2D::flather_weight`, and `SWE2DRhsConfig::with_dt` (no callers);
  - nesting relaxation in a sponge/FRS band (`source/`), not in the ghost; 1D characteristic OBCs reflect oblique incidence (~17 % at 45°);
  - elevation-only forcing: take transports from NorKyst/TPXO, or assume an incoming progressive wave (`u_n,ext = −√(g/h) η_ext`); `u_ext = 0` delivers a progressive tide at half amplitude.
  - Gate: extend `tests/open_boundary_flather_test.rs` to all three fluxes, oblique incidence and elevation-only forcing.
  - Done: `Radiation2D` is now η-referenced still water (ghost = external state) instead of zero-gradient extrapolation.
- [ ] Inverse-barometer consistency at open boundaries; cap Large–Pond Cd.

### P1.5 Nesting done properly (NorKyst/ROMS parent)
- [ ] Conserve parent transport: ubar_child = ubar_parent·h_parent/h_child. Read parent `h`, and blend child bathymetry to the parent's in the relaxation band (`ocean_nesting.rs:141-145`).
- [ ] Rotate grid-relative velocities by `angle`, or accept only `*_eastward/northward`. Solve the inverse bilinear map on the curvilinear grid (`netcdf_io.rs:1596-1597`).
- [ ] z-level files: branch on the vertical coordinate and its `positive` attribute. Index 0 is the surface in NorKyst ZDEPTHS files (`netcdf_io.rs:1389`).
- [ ] A Davies / Martinsen–Engedahl flow-relaxation band over N cells (`SpongeLayer2D` needs non-rectangular distance functions).
- [ ] Fix the off-by-one for descending coordinates (`find_bracket` `:1762-1766` + `get_state` `:1443`) and the silent zeros in `get_state`.
- [ ] Flatten parent fields to strided `[time, y, x]` arrays and precompute per-boundary-node stencils at setup (no per-stage RwLock HashMap).

### P1.6 Real-domain inputs
- [ ] GeoTIFF: check the CRS (UTM33 grids silently give B = 0 today), use pixel-centre interpolation, and do L2/area projection instead of point sampling (`geotiff.rs:99-113`).
- [ ] Land/nodata must not become B = 0 (`bathymetry_2d.rs:133`). Use the land mask in the solver or mesh water only.
- [ ] GSHHS inner rings (holes) and a spatial index for point-in-polygon (`coastline.rs:81-95`).
- [ ] f(latitude) and β helpers. Fix `norwegian_coast_beta()` (β = 1.6e-11 is the 45°N value; 60°N ≈ 1.14e-11). UTM33 / Lambert for coast-scale domains.
- [ ] Atmospheric forcing reader for MET Nordic / MEPS / AROME-Arctic (Lambert grid, 2D lat/lon, grid-relative winds, CF time), wired to `WindStress2D`/`AtmosphericPressure2D`.
- [ ] Rivers as volume sources (`RiverTracerSource` adds no volume; `Discharge2D` is weak). NVE / ROMS river-file reader.
- [ ] Fix the tidal-potential sign vs its docs (`source/swe_2d/tidal.rs:299-305`) and the diurnal Love factor (low priority, < 5 mm effect).

### P1.7 Gating tests (write first; most fail today)
- [ ] Lake-at-rest with steep nodal B (e.g. 30→400 m), p = 1–4, serial and parallel: residual < 1e-10.
- [x] Lake-at-rest with face-discontinuous B, and with wet/dry shorelines.
  - Face-discontinuous B: split forms, `test_lake_at_rest_any_nodal_bathymetry` (cell-averaged and rough beds, P1–P4).
  - Shorelines: `WetDry`, `test_wet_dry_lake_at_rest_with_shorelines` (RHS) and `tests/wet_dry_2d_test.rs::lake_at_rest_with_shoreline_is_exact` (100 s runs). `Standard` is bounded by `…_stays_near_rest`.
- [x] Mass conservation with discontinuous B and slope-correlated flow: split forms `test_mass_conservation`, `WetDry` with dry regions `test_wet_dry_mass_conservation_with_dry_regions`; `Standard` covered by the P0.13 tests.
- [x] Thacker parabolic bowl (dynamic wet/dry), planar SWASHES case: `thacker_planar_oscillation_converges_{standard,wet_dry}`. Both converge at ≈ 2nd order. Also the Ritter dry dam break, for both formulations.
- [ ] Outgoing-pulse reflection < 1 %; delivered tidal amplitude.
- [ ] Nonlinear SWE convergence with bathymetry on curved meshes. Assert N+1: current thresholds 1.5/2.5 are below it (`tests/convergence_test.rs`).
- [ ] Fix vacuous tests:
  - `test_geostrophic_balance` passes with Coriolis removed (tolerance 0.49 vs ~1e-3 signal).
  - `tests/swe_2d_test.rs::test_lake_at_rest` uses a flat bottom.
  - The well-balanced tests use only linear B at p ≥ 2.
  - "Multi-day stability" runs 316 s.
  - `test_parallel_matches_serial` excludes bathymetry, boundaries and limiters.

---

## Priority 2: Cheap Enough to Matter (`REVIEW.md` §5, §7 Phase B)

### P2.1 Time step
- [ ] Per-element directional dt (P0.21) run near the stability limit. Evaluate SSPRK(4,3) or low-storage RK (+1.3–1.5×).
- [ ] The other dt helpers still pair the global minimum √detJ size with the global maximum speed: `compute_dt_advection_2d`, `compute_dt_tracer_2d`, `compute_dt_viscosity` (takes a global size), `Mesh`-trait `compute_dt`, `time/ssp_rk3.rs:110`, `compute_dt_swe` (1D). Port them to the per-node metric form of `compute_dt_swe_2d` (one shared helper over ∇r, ∇s).

### P2.2 Kernels
- [ ] Sum-factorised contravariant volume term. Currently 4 dense (p+1)⁴ mat-vecs per element (`swe_2d.rs:835-874`): 2.6× fewer flops at P2, 3.5× at P3.
- [ ] Diagonal LIFT (GLL mass is diagonal; `kernels.rs:490-517` applies it densely).
- [ ] One Riemann solve per interior face (each is currently solved twice).
- [ ] SIMD Manning; drop per-node `reference_to_physical` and `dyn` dispatch in the RHS.

### P2.3 Fused in-place stepper
- [ ] Limiter, positivity and wet/dry in one parallel in-place pass. Today the "parallel" paths `to_vec` every field, allocate per element, and copy back serially (`limiters/swe_2d.rs:626-725`, `wetting_drying.rs:526-541`): 1.5–2.5× of wall time.
  - Limiters done (P0.19: in-place `par_chunks_exact_mut` over the SoA fields). The `wetting_drying.rs` parallel path is done too (P1.2: in place, plus a node-parallel implicit damping pass). Still open: the per-stage `Vec` of cell averages and vertex bounds (move to a workspace with P1.1). Then fuse the limiter, the wet/dry correction and the implicit damping into one pass.

### P2.4 Water-only work
- [ ] Mesh water only, or keep an active-element set (41 % of Frøya's elements are land). Limit only troubled cells.

### P2.5 Local time stepping or implicit free surface
- [ ] One 250 m element in 1300 m water forces ~22× more global steps. Multirate/LTS SSP-RK or semi-implicit free surface (SLIM/Thetis practice): 2–20×.

### P2.6 Benchmarks (was P1.6)
- [ ] Full-RHS and full-step throughput vs mesh size at realistic size (≥ 100k elements, bathymetry, limiter, open BCs), not cache-resident micro-cases.
- [ ] DOFs/second; parallel scaling 1–16 cores; memory-bandwidth utilisation; allocation counting in CI.
- [ ] Reconcile PERFORMANCE.md vs PROFILE.md: the same 65k run is recorded as 70.5 s and 217 s, and some cited RHS variants no longer exist.
- [ ] The criterion benches do not compile (`benches/source_term_bench.rs`, `time_stepping_bench.rs`: `usize` where `ElementIndex` is expected, tuple access on `[f64; 2]` vertices). CI never builds them (clippy runs `--lib --tests`); fix them and add `--benches` to the CI check.

### P2.7 GPU and distributed memory (decision)
- [ ] Either fix Burn (P0.11) or replace it with custom fused f64 kernels (CubeCL/cudarc) over CSR faces. Burn with fusion disabled (`Cargo.toml:38`) moves ~13× more bytes per node-stage than fused kernels.
- [ ] MPI/domain decomposition plan (METIS partitioning, halo exchange of face traces).

---

## Priority 3: Validation and the Speed Claim (`REVIEW.md` §7 Phase C)

### P3.1 Validation against observations (was P1.5)
The tidal-comparison code path is complete: fit with `HarmonicAnalysis` → `HarmonicResult::reference_constants(&epoch)` → compare to catalogue (H, G). **Re-run after P0.15**: the fits used truncated constituent periods.

- [ ] **Fetch NorKyst-800 data** via the global `norkyst-client` CLI (v0.1.0 at `~/.cargo/bin/norkyst-client`), which extracts NorKyst historical/forecast over OPeNDAP.
  - **Commands:**
    - Point time series at a gauge: `norkyst-client --source historical --lat <lat> --lon <lon> --start-date YYYY-MM-DD --end-date YYYY-MM-DD --time-grain hourly --format parquet -o <dir>`
    - Multiple gauges: `--sites sites.csv` (CSV columns `id,lat,lon`).
    - Regional grid: `--bbox min_lat min_lon max_lat max_lon` or `--area <1-13> --geojson PO.geojson`; `--partition-grain day|month`.
  - **Formats:** `text|arrow|parquet|vortex`, not NetCDF (`OceanModelReader` will not ingest these).
  - **Text reader (DONE 2026-07-15):** `io::read_norkyst_text_file` / `parse_norkyst_text_str` → `NorKystTextData` (`sea_surface_height_series()`, `surface_current_series()`). Needed the companion `norkyst-client` fix that emits the `surface …` line.
  - **Parquet reader + real Bergen result (DONE 2026-07-18):**
    - The point/NCSS text fetch is chronically 503; the reliable path is `--bbox … --format parquet`. `io::read_norkyst_parquet_glob` sits behind the `parquet` feature.
    - Real 2-month Bergen fetch: M2 H ≈ 0.427 m, G ≈ 278°; S2 H ≈ 0.117 m; R² ≈ 0.97 over 61 days. P1 aliases into K1 at 61 days; ~6 months are needed to separate them.
    - **Phases carry the P0.15 period error (~1° for M2 at 61 days).**
  - **End-to-end example (DONE 2026-07-16):** `examples/norkyst_tidal_validation.rs` (synthetic round-trip by default, or a real file as argument). Recipe: shift the series to t = 0 and take `AstronomicalArguments` at the first sample before `reference_constants`.
- [ ] Real Kartverket (H, G) for Bergen, Stavanger, Trondheim, Kristiansund. Drop records into `data/tide_gauges/` (only a synthetic `heimsjo.txt` ships). Longer fetch for the diurnal band.
- [ ] Run the model against NorKyst-800 barotropic tides for a test period. This needs P0.12, P0.20, P1.3–P1.5.
- [ ] Compare with ADCP currents (`analysis/adcp.rs`). Add a tidal-ellipse fit and a complex-difference skill metric.
- [ ] Document skill scores (RMSE, bias, correlation) per station.
- [ ] Harmonic analysis: a record-length/Rayleigh guard in `fit()`; P1-from-K1 inference for short records; an `AnalysisError` type instead of asserts.

### P3.2 Cost-vs-accuracy benchmark against ROMS
- [ ] Pareto benchmark on the same hardware: M2/S2/K1/O1 error at tide gauges vs core-hours, ROMS 2D at 800/400/200 m vs DG P1–P4. This is what settles "faster than ROMS" (`REVIEW.md` §5.5).

### P3.3 Long-run stability
- [ ] 30+ day real-domain tidal run: stable, mass-conserving, no spurious residual currents at rest.

---

## Priority 4: 3D Rebuilt on the ROMS Recipe (`REVIEW.md` §2, §3, §7 Phase D)

The 2026-02-11 plan (vertical infrastructure → mode splitting → mixing → physics → validation) is superseded. The scaffolding exists, but the coupling must be restructured before new physics is added. See `3D_TODO.md` for background (its checklists predate the current code).

### P4.1 Mode splitting
- [ ] Replace "subcycle inside every SSP-RK3 stage" with one generalized forward-backward (AB3-AM4) barotropic pass per baroclinic step. Take η/ū out of the RK combination.
  - Today: first-order; M2 loses 3–14 % per period; 6·n_bt Forward Euler 2D RHS per step (Shchepetkin & McWilliams 2005).
- [ ] Accumulate the secondary-weighted barotropic transport (DU_avg2) alongside the Hann-filtered η/ū.
- [ ] G-term = ⟨RHS₃D⟩ − RHS₂D(ū) (the `rufrc` construction). Today it double-counts advection and Coriolis.
- [ ] Surface and bottom stress into the barotropic forcing. Today the 3D `Forcing` never reaches the depth mean, so there is no wind setup or Ekman transport.
- [ ] Barotropic CFL control (n_bt = ⌈dt/dt_bt,max⌉); positivity/limiter in the fast mode (unguarded 1/h at `hydrostatic_3d.rs:238`); step transport (Dū), not ū.
- [ ] Recompute density at every RK stage (frozen today, `simulation_3d.rs:150`).
- [ ] Remove the `step`/`step_with_stage_hook` duplication and the allocations (~64 N₃D arrays per step).

### P4.2 Consistent continuity, tracers and momentum
- [ ] Hz-weighted per-level face fluxes corrected so their vertical sum equals DU_avg2; η advanced by the divergence of the same flux.
- [x] Ω stored at w-points (not layer centres re-averaged to faces) — done in [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2).
- [ ] Assert Ω(0) = 0 instead of forcing it with the linear correction.
- [ ] One 3D face-state helper for physical boundaries shared by the Ω, momentum and tracer kernels (today each has its own copy of the wall logic; P0.7/P0.22 were the same bug found twice). It should take the boundary tag, so 3D open boundaries (volume flux from the 2D OBC / nesting, tracer from `TracerBoundaryCondition3D` on inflow) are added in one place for all three.
- [ ] Step Hz·C and Hz·u (inventory form), then divide by the new Hz. Today a 1 m tide over 20 m of water pumps salinity by about ±1.7 psu.

### P4.3 Pressure gradient and vertical grid
- [ ] Balanced PGF: Shchepetkin & McWilliams (2003) density-Jacobian, or z-level interpolation. Lift pressure/η jumps at element faces.
  - Today: ~1e-4 m/s² spurious forcing in a stratified fjord at rest (P3/250 m), the size of real estuarine forcing.
- [ ] rx0/rx1 diagnostics and ROMS-style bathymetry smoothing.
- [ ] Vtransform 2 + the true Vstretching 4. `ROMSVstretching4` is not ROMS Vs4, and `hc` is unused; Ω must use ∇·(Hz·u) once Hz varies horizontally.

### P4.4 Vertical mixing and boundary layers
- [ ] GLS k-ε (NorKyst's closure) as a per-column solve reusing the tridiagonal solver (Umlauf & Burchard 2003; Warner et al. 2005). KPP as an alternative.
- [ ] Convective adjustment (low-shear unstable columns currently get background mixing).
- [ ] Implicit quadratic bottom drag with a log-layer Cd; consistent with 2D friction.
- [ ] Surface heat flux Q_net/(ρ₀c_p) (currently fed a buoyancy flux); remove the hardcoded ρ₀ = 1025.
- [ ] Surface heat-flux budget (shortwave, longwave, sensible/latent) for multi-day SST evolution (formerly "P3.1 Surface Heat Flux Budget").

### P4.5 Remaining 3D physics and numerics
- [ ] Higher-order vertical advection (4th-order centred/Akima or HSIMT). First-order upwind today over-diffuses the halocline.
- [ ] Rotated/geopotential horizontal diffusion via BR1 (no 3D horizontal diffusion exists).
- [ ] 3D wetting/drying masks (any dry node gives NaN today).
- [ ] EOS: delegate the physics trait to the fixed UNESCO EOS (P0.17) with a linear fast path; TEOS-10 later.
- [ ] `compute_dt`: internal-wave speed from stratification (not a hardcoded 2 m/s) and the vertical CFL. Relabel or convert Ω vs w in output.
- [ ] Parallelise the 3D kernels; no `Vec` allocation in inner loops (~186M malloc/free per 3D RHS at 50k × 30).
  - E.g. `apply_tracer_advection_3d` allocates five `Vec`s per face per level and recomputes `layer_thickness` for every node in the lift loop of every face; hoist both (Hz per element-level once, face buffers in a workspace).

### P4.6 3D validation (before any NorKyst 3D comparison)
- [ ] Stratified lake-at-rest: linear N² ≤ 1e-12; tanh pycnocline over a seamount (Beckmann & Haidvogel 1993).
- [ ] Constant-T preservation under tide over a sloping bed; Hz·T inventory conservation.
- [ ] Wind setup τ/(ρgD); Ekman transport τ/(ρ₀f); column momentum after one implicit diffusion step = Δt·τ/ρ₀.
- [ ] Temporal convergence of the coupled scheme; 100-period barotropic energy decay.
- [ ] Lock exchange; mode-1 internal-wave speed; idealised fjord estuarine circulation (Sognefjord-like).
- [ ] Comparison with NorKyst-800 3D fields.

---

## Priority 5: Operational Features

### P5.1 River forcing
- [ ] Vertical shape function for `Discharge2D`; NVE database (1,760 rivers in NorKyst-800). See P1.6 for volume sources.

### P5.2 ROMS-compatible output
- [ ] `NetCDFWriter` produces CF-1.8 but not ROMS-compatible output; add a ROMS variable-name mapping.

### P5.3 Restart / checkpointing
- [ ] HDF5 or NetCDF restart files (operational daily forecasts must hot-start).

### P5.4 Operational pipeline
- [ ] Automated run scripts, monitoring and alerting, THREDDS/OPeNDAP serving.

---

## Priority 6: Tech Debt and Structure (`REVIEW.md` §6.2)

- [ ] Move `examples/froya_real_data.rs` (and the other examples) from the legacy `ssp_rk3_swe_2d` stepper and its `SWE2DTimeConfig` onto `Simulation` + `SWEPhysics2D`; this is the precondition for deleting the legacy integrators below. The two paths currently duplicate limiter and wet/dry dispatch.
- [ ] Delete the legacy integrators (`ssp_rk3*.rs`, three `coupled_swe_tracer` variants, `burn_ssp_rk3.rs`; ~2.4k lines). Make `CoupledState2D` implement `Integrable`; one time loop (fold `Simulation3D::run_with_callback`).
- [ ] Collapse the duplicate config enums (`SWEFluxType2D` vs `StandardFlux2D`; three limiter enums).
- [ ] Prune ~200 root re-exports to a `prelude`; export `Simulation3D`.
- [ ] Error handling:
  - `SimulationResult` → `Result<SimulationStats, SimulationError>`;
  - reachable panics (`timeseries_reader.rs:198`, `geotiff.rs:239-245`);
  - library `eprintln!` in `tide_gauge.rs`.
- [ ] Feature-gate `tiff`/`shapefile`/`geo` behind `geodata`.
- [ ] Build the `netcdf` feature in CI. It is excluded because it needs native HDF5/netCDF-C, so the NetCDF reader and nesting tests (P0.20) only run locally. `cargo check --features netcdf/static` builds HDF5 and netCDF-C from source with CMake and no conda (verified on Windows 2026-09-25: ~5.5 min cold, cached afterwards); use it for a CI job, and consider it as the documented local alternative to conda.
- [ ] Characteristic-based limiting for the 2D SWE system (1D version exists but is dead code; was P2.4). Likely subsumed by P1.2.
- [ ] Consolidate the root markdown files into `docs/`. Decide crate vs repo name; `dg-core`/`roms-rs` workspace split.

---

## Priority 7: Advanced Features (Future)

### Research numerics
- [ ] IMEX time stepping via diffsol (implicit vertical diffusion / free surface; see P2.5).
- [ ] Subcell positivity preservation with convex limiting (Wu et al. 2024).
- [ ] hp-adaptive mesh refinement.
- Entropy-stable DG (split form done, see P1.2), sum factorisation and curved elements are now in P1.2, P2.2 and P1.3.

### Operational
- [ ] Data assimilation (start with EnKF, then 4D-Var).
- [ ] Two-way nesting (child feeds back to parent).
- [ ] MPI for distributed memory (see P2.7).
- [ ] Wave–current interaction.
- [ ] Biological coupling (NPZD for salmon-lice dispersion).
- [ ] Sediment transport.

---

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

---

## Key Metrics

| Metric | Current (2026-09-25) | Target (2D operational) |
|---|---|---|
| Tests | 1,024 lib + 61 integration + 59 doc (no-default); suite < 2 s | + the P1.7 gating physics tests |
| Lake-at-rest, steep nodal B (30→400 m, 1.25 km, 1 h) | **Fails**: 0.9 m/s (P3), 2.3 m/s (P2), blow-up (P1) | residual < 1e-10, p = 1–4 |
| Mass conservation, face-discontinuous B | **Fails**: −5.5e-5 relative per second | machine precision |
| Open-boundary reflection (Flather) | ≈ 33 % | < 1 % |
| SWE convergence | P1 1.67, P2 3.02 (linearised, flat, periodic; thresholds 1.5/2.5) | N+1, nonlinear, bathymetry, curved |
| Allocations per 2D step (production stepper) | `Simulation` path: 0 B warm (without limiter/viscosity); legacy `ssp_rk3_swe_2d`: ~33 full-field arrays | 0 |
| 2D tidal cost vs ROMS (model estimate) | ~47× core-hours | ≤ 1× per unit accuracy (P3.2) |
| Validated against observations | No (NorKyst Bergen harmonic fit only) | 5+ tide gauges, ADCP |
| Multi-day stability | 316 s tested | 30+ days, real domain |
| 3D | Scaffolding; 3 blockers (tracer constancy, vertical advection ×D, PGF) | stratified lake-at-rest, constancy, lock exchange |
| GPU | Non-functional (P0.11) | fixed or replaced (P2.7) |

---

## References

- Hesthaven & Warburton (2008), Nodal DG Methods.
- Toro (2009), Riemann Solvers for Fluid Dynamics.
- Wintermeyer, Winters, Gassner & Kopriva (2017); Wintermeyer et al. (2018), entropy-stable well-balanced DGSEM for SWE; wet/dry extension.
- Audusse et al. (2004), hydrostatic reconstruction; Xing & Shu (2006); Xing, Zhang & Shu (2010).
- Zhang & Shu (2010), positivity-preserving limiters; Kuzmin (2010), vertex-based slope limiting; Vater, Beisiegel & Behrens (2019).
- Flather (1976); Chapman (1985); Kärnä et al. (2011), weak Flather in DG; Davies (1976), flow relaxation.
- Shchepetkin & McWilliams (2003), density-Jacobian PGF; (2005), ROMS split-explicit stepping.
- Umlauf & Burchard (2003); Warner et al. (2005), GLS vertical mixing.
- Kärnä et al. (2018), Thetis 3D.
- NorKyst v3 preprint (2025), egusphere-2025-3986.
