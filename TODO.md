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
  - The 3D layer is scaffolding with three blocker-level math errors. (Vertical advection ×D since fixed, P0.16; tracer constancy and the σ pressure gradient remain, P4.2/P4.3.)
- **Cost:** as configured, roughly ~50× ROMS core-hours for a 2D tidal run (cost-model estimate, `REVIEW.md` §5).

**Strategic direction.**
1. **Make the 2D barotropic tide model correct, then cheap, then validated.** This is where DG's fjord-geometry and accuracy-per-DOF advantages can actually beat ROMS.
2. **Then rebuild 3D on the ROMS recipe.**

"Faster than ROMS" is a claim to be *proven* with a cost-vs-error benchmark (P3.2), plausible for 2D tides at P3+ with local time stepping or an implicit free surface. It should not be assumed.

---

## ▶ Next session — start here

Last reviewed: 2026-09-25 (`REVIEW.md`). Done so far: Priority 0 (except the deferred P0.11 GPU), P1.1 for the `Simulation` path, and the P1.2 core (split form; `WetDry` with second-order shoreline subcells as the wet/dry default). Frøya now runs on `Simulation` + `WetDry` on a water-only mesh, with correctly georeferenced bathymetry: the GeoTIFF reader had ignored its georeferencing. Lake at rest over the real bathymetry holds to 1.5e-10 m/s for 1 h. P1.4 is done: one characteristic OBC (`CharacteristicOBC` + external-state providers), NorKyst-800 boundary tides from a harmonic atlas, and a `ModelClock`. A full M2 cycle with NorKyst boundary tides at 9,653 P2 elements is stable, with 0 clips, in 6 min. Pick up in this order:

1. **P3.1 validation: the tooling is done; the month-long run is still to do.** Heimsjø and Kristiansund lie outside the Frøya domain. Mausund (Kartverket MSU) is the gauge inside it, with a year of observations in `data/tide_gauges/mausund_obs.txt` and NorKyst at the gauge in `data/froya_station_tides.txt`. Both are untracked, like all of `data/`. Regenerate them with `./scripts/kartverket_gauge.sh Mausund 63.869331 8.665231 2024-07-01 2025-07-01` and `norkyst_boundary_tides -- spacing_km=0 points=8.665231,63.869331 out=/dev/null` (≈ 20 min). The run takes ≈ 9 h at 8.2 ms/step, so it waits until P2 makes it cheaper, or is started overnight:
   ```
   cargo run --release --no-default-features --features parallel,simd --example froya_real_data -- \
       start=2025-05-31T00:00:00Z hours=720 ramp_hours=3 spinup_hours=24 output_minutes=1440
   ```
   It prints the per-constituent comparison against the gauge and NorKyst. 29 days after spin-up resolve N2 and Q1; 15 days (`hours=384`) resolve M2/S2/K1/O1.
   - **First results (2026-09-26), from 2025-06-01, 1 day spin-up:**
     - Run cost (plugged in, 24 threads): 1 km (`nx=60 ny=45`) 2.0 ms/step, so 16 days take 35 min; 500 m (default) 4.7 ms/step, so 3 days take 31 min and 30 days ≈ 5 h. Progress lines in a redirected log appear in bursts, so judge progress from the process, not the log.
     - **1 km, 15 days analysed.** Model against the gauge:
       - M2 1.033×, **+5.4°** (7.4 cm); S2 1.095×, +7.2°; K1 1.35×, +11.8°; O1 0.92×, −19.6°; M4 1.23×.
       - Complex-difference RSS 9.5 cm (NorKyst 16.6 cm, mostly its N2).
       - Against the gauge's tidal prediction: RMSE 9.8 cm (centred 7.0 cm), correlation 0.990.
       - NorKyst's own M2 matches the gauge at −0.5°, so the phase lag builds up inside our model between the boundary and Mausund.
     - **Resolution explains most of it.** The 500 m run (3 days) is 6.4 min earlier (−3.1° of M2) and 1.3 % lower than 1 km at Mausund, so ≈ 1.02×, +2.3° at 500 m. Against the gauge's tidal prediction over hours 24–72, centred RMSE is 4.1 cm at 1 km and 2.9 cm at 500 m. The station nodes differ too: at 1 km the node is 325 m away and 16.4 m deep; at 500 m, 199 m away and 7.6 m deep.
     - `land_elevation=1.5` against 5 m: 0.3 mm RMS at Mausund. The shoreline cliffs are local and do not reach the gauge.
     - **P3 at 1 km against P2 at 500 m (3 days, hours 24–72 compared):**
       - Setups:
         - 1 km P2: 22k nodes, dt 1.31 s, 1.9 ms/step (≈ 6 min).
         - 1 km P3: 40k nodes, dt 0.66 s (positivity CFL), 3.0 ms/step (20 min).
         - 500 m P2: 87k nodes, dt 0.65 s, 4.7 ms/step (31 min).
       - Centred RMSE against the gauge's tidal prediction: 4.1 / 3.3 / 2.9 cm. Shift against 1 km P2: — / −4.3 / −6.4 min.
       - P3 at 1 km gets about two thirds of the gain of halving the mesh for about two thirds of its cost: roughly break-even here. The cheapest setup by far is 1 km P2 (≈ 5× less than 500 m P2 per simulated hour).
       - Confounded: each setup samples a different node (325 m / 16 m deep, 259 m / 27 m, 199 m / 7.6 m). Interpolate at the gauge position before comparing further.
       - The positivity CFL (0.29 at P3 against 0.42 at P2) costs P3 a factor of ≈ 1.45 in dt. Relaxing it (a subcell bound, or limiting only near dry fronts) would favour higher order.
     - Next steps:
       - The 500 m, 15-day run (≈ 2.5 h) for the harmonic numbers at resolution.
       - Station interpolation at the gauge position, instead of the nearest node ≥ 3 m deep.
       - Friction (Manning n = 0.025 everywhere) is the other phase suspect. The bed builder is not: the library `BedRaster` bed (P1.6, 2026-09-26) gives 3.3 cm centred RMSE (hours 24–72) sampled at the nodes and 3.6 cm projected, against 4.1 cm with the old builder, all at 1 km.
   - **Finding: NorKyst-800 has almost no N2 here.** At Mausund over June 2025, N2 is 0.027 m in NorKyst against 0.156 m observed (both 30-day fits; the year fit gives 0.154 m). Q1 is 1.9× the observed, K1 and S2 are 12–21 % high, M4 is 5× too weak, and M2 agrees (1.026, −0.5°). NorKyst's own complex-difference RSS against the gauge is 0.14 m, 0.13 m of it from N2. The Frøya run therefore re-infers the boundary atlas's N2 and Q1 from M2/O1 with Mausund's ratios (`gauge_ratios=N2,Q1`, the default). Report the N2 deficiency to MET, and check other NorKyst gauges (Bergen) for it.
   - Re-run the Bergen NorKyst fit too: it used the truncated periods fixed in P0.15 ([PR #2](https://github.com/EmilLindfors/roms-rs/pull/2)).
2. **Shoreline cliffs (P1.6).** Lone wet nodes next to land nodes at `LAND_ELEVATION` = +5 m carry η jumps of ~1 m and 1–2.8 m/s through the tide (details under P1.4). They dominate the Frøya extremes; a foreshore DEM, or a gentler land elevation, is the likely fix. The bed builder is now in the library (`BedRaster`, with an L2 projection, 2026-09-26). The remaining piece without a DEM is to drop lone wet nodes to land when building the water-only mesh.
3. **Then let the error budget choose:** geometry (P1.3) if the coastline dominates the error, cost (P2, profile first) if run time does.
4. **η-based limiting is on hold.** The subcells limit the partially dry elements, and tides are smooth. Revisit only if a real run shows oscillations.

Build note: default features need the `roms-rs` conda env (`conda activate roms-rs`) for HDF5/netCDF. Otherwise use `cargo test --no-default-features --features parallel,simd`.

---

## Application target: currents around a salmon farm

Goal (added 2026-09-26): resolve currents at a farm site (cages ≈ 40–60 m across, nets 20–40 m deep) inside a NorKyst-driven coastal domain, and track sea-lice larvae, feed and faeces from the cages. A 3D visualisation (Bevy, separate crate) reads the model output; it is not part of `dg-rs`.

This section only orders the existing items for that goal and adds the two farm-specific ones (F.1, F.2). Everything else lives in its own priority section.

**Stage A: 2D farm-scale currents.** Depth-averaged, so it answers tidal exposure and the wake of the farm, not the surface layer.
1. P1.3 geometry (blocker): coastline-fitted meshes with elements of tens of metres at the farm and kilometres offshore. General quadrilaterals are supported in 2D since 2026-09-26, and all-quad Gmsh meshes (MSH 4.1/2.2) load since 2026-09-26 (see `docs/gmsh-meshes.md`); the rest of P1.3 (triangles or quad-dominant meshes, CSR) is still open.
2. P1.6 shoreline cliffs and bathymetry: the 1–2.8 m/s spurious currents at wet nodes next to land sit where farms are. The bed + coastline builder is in the library and the L2 projection exists (2026-09-26: `BedRaster`, `Bathymetry2D::project`); the foreshore DEM is still missing (`BedRaster::new` takes one).
3. ~~F.1 cage drag (2D form)~~ done 2026-09-26 (`CageDrag2D`, `with_cage_drag`).
4. P2.5 local time stepping or implicit free surface: a single 20 m element sets the global dt, so farm refinement makes this the dominant cost.
5. P1.5 nesting of NorKyst ū, v̄, η (residual and coastal-current transport, not only the tidal atlas) and the P1.6 atmospheric forcing reader (wind stress).
6. P3.1 current validation: depth-averaged velocity in station series, sampling at the station position by interpolation inside the element (F.2 needs the same evaluation), ADCP comparison and tidal-ellipse fit. Farm-site current surveys, where available, are the natural data.
7. Minor: `SpatiallyVaryingManning2D` as `BottomFriction2D` (P1.2); the vacuous tests in P1.7.

**Stage B: 3D.** Lice larvae live in the upper few metres and respond to salinity, so dispersion needs the stratified surface layer.
- P4.2 (tracer constancy, 3D open boundaries for T/S), P4.3 (balanced PGF, vertical grid), P4.4 (GLS, quadratic bottom drag), P4.5 (3D wet/dry, which gives NaN today; vertical advection order; parallel, non-allocating 3D kernels), rivers (P1.6 volume sources, P5.1), then P4.6 validation.
- F.1 cage drag (3D form) and F.2 in 3D.

Not needed for this goal: P0.11/P2.7 (GPU, MPI), P3.2, P5.2–P5.4, most of P6 and P7. Until Stage B is validated, the particle tracker and the visualisation can be developed against NorKyst-800 3D fields.

### F.1 Cage drag
- [x] Momentum sink from the net as a porous region: S = −½·C_d·a·|u|u per unit volume, with a (net area per volume) and C_d from the net solidity (Løland 1991; review in Klebert et al. 2013, Ocean Eng. 58). Fouling raises the solidity, so it is a per-cage parameter (`NetCage`, `NetCage::circular` with Løland's C_d and a = 4/(πR)).
  - [x] 2D (2026-09-26, `source/swe_2d/cage.rs`): integrated over the net depth only, Λ = ½C_d a |u| min(d_net, h)/h, over the cage footprint.
  - [ ] 3D: apply per level, only to levels above the net bottom (Stage B).
  - [x] Cages are comparable to or smaller than an element: each node is weighted by the area of its GLL subcell inside the footprint over w·J (adaptive bisection on the signed distance), so Σ wJφ is the footprint area on any quadrilateral.
- [x] Point-implicit with the bottom friction (`ImplicitDamping2D::cages`, `SWEPhysics2DBuilder::with_cage_drag`).
- [x] Gate tests (`tests/cage_drag_test.rs`):
  - Porous band across a periodic channel driven by a body force: momentum-flux and pressure loss across the band = drag − forcing (1e-3); total drag = total forcing (1e-4).
  - Mass conserved; lake at rest over a rough bed with cages exact (1e-10).
  - Magnitude: domain-wide cage against a body force reaches the analytic u² = 2Gh/(C_d a min(d, h)) (1e-6); decay of uniform flow converges at first order.
- [ ] Follow-ups:
  - The rear net sees the cage-mean velocity; Løland's velocity reduction behind a panel (r = 1 − 0.46 C_d) is left to the resolved flow. Compare the wake with a farm survey or published flume data before trusting the default a = 4/(πR).
  - Angle dependence (C_d(θ)) and the lift term: irrelevant for a circular cage averaged over its perimeter, not for square cages in oblique flow.
  - Net deformation in strong currents (the net lifts and its projected depth shrinks, reducing d_net) is not modelled.
  - A cage-layout reader (positions, radii, net depths per site), e.g. from the Fiskeridirektoratet site register, for real farms.

### F.2 Lagrangian particle tracking
- [ ] Point location: find the element holding a point and its reference coordinates (Newton on the isoparametric map once P1.3 lands; a spatial index over elements).
- [ ] Velocity evaluation from the element polynomial (the same evaluation P3.1 station sampling needs), linear in time between output snapshots or online from the solver.
- [ ] RK4 advection; horizontal random walk; vertical random walk with the ∂K/∂z drift correction (Visser 1997, MEPS 158) once 3D exists.
- [ ] Coastline reflection, open-boundary exit, beaching of particles that reach a drying node.
- [ ] Behaviour hooks: sinking speed (feed, faeces), lice larvae depth preference and salinity avoidance (as in the IMR salmon-lice model; Sandvik et al. 2020, Aquac. Environ. Interact. 12).
- [ ] Gate tests:
  - Solid-body rotation: trajectories exact to the time-stepping error when the velocity is in the polynomial space.
  - No particle crosses a wall.
  - The vertical random walk keeps an initially uniform distribution uniform under variable K (Visser's well-mixed condition).

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
  - Follow-up done: `examples/froya_real_data.rs` uses nodal B and `WetDry`.
- [x] Wet/dry and positivity: Zhang–Shu towards h ≥ 0 (not h_min), Kurganov–Petrova velocity desingularization below `h_dry` (1 mm). Done for the collocated (`Standard`) formulation (`solver/algorithms/wetting_drying.rs`, `limiters/swe_2d.rs`). Gates in `tests/wet_dry_2d_test.rs`: Ritter dry dam break (L1 1.5 %, first order), Thacker's paraboloid (L1 7.4 % after one period, was 9.2 %), shoreline lake at rest (|u| < 0.06 m/s where h > 1 cm, was up to the 20 m/s cap). All conserve mass to round-off with h ≥ 0.
- [x] **Shoreline well-balancing and wet/dry for the split form: `SWEFormulation2D::WetDry`.** HLL on Audusse-reconstructed states at faces, flux differencing in wet elements, and subcell finite volumes with the same flux in partially dry elements.
  - A shoreline lake at rest is kept to round-off (P1–P4, smooth/rough beds; the P1.7 gate `lake_at_rest_with_shoreline_is_exact` passes).
  - Fully wet it keeps N+1 (P2 3.06, P3 4.02) and dissipates energy.
  - `Standard` is still not balanced in partially dry elements: on a 2 % beach it settles into a stationary circulation of ≈ 5 cm/s where h > 1 cm, and 1–4 m/s in the film.
- [x] **`WetDry` accuracy on moving shorelines.** Every element the shoreline crosses uses subcell FV, which was first order: 4.8 % on Thacker (P2, 40²) against 1.3 % for `Standard`.
  - Done: a second-order subcell reconstruction of h, η, u, v. It uses a monotonized-central limiter on the non-uniform GLL subcells, end-subcell slopes from the adjacent element, and the Audusse second-order bed term.
    - It stays well-balanced at shorelines and keeps h ≥ 0 at faces.
    - With subcells forced everywhere it converges at second order (P1–P3 ≈ 1.9).
    - Thacker at 40²: P1 3.2 %, P2 1.2 %, P3 0.9 %, P4 0.7 %, against `Standard` 10 / 1.3 / 1.8 / 2.3 %.
  - Also fixed: HLL returns zero, pressure included, when both reconstructed depths are below `h_min`; the hydrostatic correction now supplies all of ½g h².
  - [ ] End subcells on physical boundaries (no neighbour element) stay first order. Take the outer slope neighbour from the boundary condition (mirror the interior node for walls).
  - [ ] `outer_nodes` scales the neighbour's node distance by the affine element heights normal to the face. Revisit with per-node isoparametric factors (P1.3).
  - [ ] Any node below `h_dry` switches the whole element to subcells. Convex DG/FV blending (Hennemann–Gassner) could keep DG in the wet part, but it needs a DG part that is balanced with dry nodes present. Lower priority now that the subcells are second order.
  - [x] `WetDry` is now the default for wet/dry runs in `SWEPhysics2DBuilder`. `Standard` stays available via `with_formulation`, and `SWE2DRhsConfig` still defaults to it.
  - [ ] Retire the `Standard` wet/dry path (hydrostatic reconstruction + `BathymetrySource2D` in partially dry elements) together with the cell-average workarounds (below), once the legacy `ssp_rk3_swe_2d` users (Frøya, P6) are on `Simulation`.
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
- [x] Move `examples/froya_real_data.rs` off the legacy `SWE2DTimeConfig` (`H_MIN` = 5 m, land as a 5 m film, cell-averaged B). It now uses `Simulation` + `WetDry` (nodal B) + `with_wet_dry` + `with_implicit_friction` on a water-only mesh, with all four sides open.
- [x] `SWEPhysics2D::post_process` built a `LimiterContext2D` every stage, whose `new` computed `mesh.h_min()` serially (a sqrt per edge). No limiter read it: the field is gone (2026-09-26).

### P1.3 Geometry for real coastlines
- [x] Per-node isoparametric geometric factors (`[K]` → `[K × n_nodes]`), parallelogram-only panic removed (2026-09-26). `GeometricFactors2D::compute(mesh, ops)` evaluates the bilinear map at every node: metric, J, mass weights w·J, and per face node the normal and sJ from the same contravariant vectors (`sJ n = ±J∇r, ±J∇s`), so the discrete metric identities hold to round-off.
  - Kernels: the collocated SWE, tracer, advection and BR1 diffusion volume terms are in conservative form `J⁻¹[Dr(J∇r·F) + Ds(J∇s·F)]` with the per-node lift scale sJ/J; the split forms use the curvilinear Wintermeyer et al. (2017) form (`{{Ja}}` metric averages, bed term included); the `WetDry` subcells use the telescoping interface metrics (Fisher et al. 2013; Hennemann et al. 2021) plus a metric balance term in the bed source (zero on parallelograms and flat beds). Limiter, wet/dry and diagnostic means are mass-weighted (`LimiterContext2D` and the limiter / wet-dry functions take `&GeometricFactors2D`).
  - Gates on distorted periodic meshes (`tests/curvilinear_2d_test.rs`): free stream, lake at rest over rough and shoreline beds, mass/momentum/tracer conservation, entropy; convergence N+1 for advection and the split forms, 2 for the subcells (`convergence_test.rs`, ACCURACY.md).
  - Parallelograms take a fast path from a dense per-element copy of the node-0 geometry (`ElementGeometry`): the per-node fields are strided and cost ≈ 9 cache lines per element. Frøya (all parallelograms, `profile=80`, mains power, CPUs 0–7, five alternating runs, min/median): RHS 1 thread 4.44/4.51 ms against 4.36/4.38 on `main` (+2–3 %), 24 threads 1.22/1.26 against 1.17/1.20 (+4–5 %); `dt` 0.246/0.251 against 0.259/0.262 (−5 %); step at 24 threads +4–5 %. Before the fast paths the RHS was ≈ 10–20 % slower. [ ] Find the remaining few percent with a profiler (AMD uProf is installed).
  - [ ] The 3D kernels (`advection_3d`, `vertical_velocity`, `baroclinic`, `limiters/tracer_3d`), `batched` SIMD and the Burn prototype still use one metric per element (`affine_metric`); `Hydrostatic3D::new`, `compute_volume_terms_batched` and `BurnGeometricFactors2D::from_cpu` assert parallelograms. Convert the 3D horizontal kernels with P4.2 (Hz-weighted conservative fluxes).
  - [ ] Curved (high-order) faces: compute the same fields from the derivatives of an interpolated high-order map (Kopriva 2006: in 2D the metric identities hold for any map of degree ≤ N); fitted coastlines then need a boundary projection of the face nodes.
  - [ ] `Bathymetry2D::to_cell_average`/`linearize` use unweighted node means (as before); weight by the element mass.
- [ ] Triangles (or quad-dominant meshes), and remove the hardcoded 4 faces (`swe_2d.rs:386,908`).
- [x] Gmsh: real MSH 4.1 parsing, sparse node tags, no `.unwrap()` on input, error (not drop) on unsupported element types (2026-09-26). `read_gmsh_mesh`/`parse_gmsh_mesh` read MSH 4.1 ASCII and binary (both byte orders, parametric nodes) and MSH 2.2 ASCII into one validating builder: unused nodes dropped, clockwise quads reordered, degenerate/non-convex quads, non-manifold edges, overlapping neighbours, lines off the mesh and conflicting tags are errors with the Gmsh tags in the message. Boundary tags from physical group names (`coast` → wall, `open`, `tidal`, `river`, else `Custom(n)`), unnamed groups by number. Edge numbering is deterministic (was `HashMap` order). Gates: `tests/gmsh_mesh_test.rs` on Gmsh 4.15 output (`scripts/gmsh_fixtures.py`, a bay with a coastline and an island): the three formats give the same mesh, tags on the right curves, lake at rest over rough and shoreline beds ≤ 1.3e-12 (P1–P3), mass to 1e-13 over a run, a raised sea fills the bay through the tagged open boundaries only.
  - [ ] One `Mesh2D::from_quads(vertices, elements, boundary tags)` constructor: the edge building is written out again in `uniform_rectangle*`, `channel_periodic_x`, `uniform_periodic`, `retain_elements` and the Gmsh builder (`build_mesh` is the general one).
  - [ ] `Mesh2D::edge_orientation` is never read: the kernels rely on neighbours listing their shared face nodes in reverse (consistent counter-clockwise order, now checked by the Gmsh builder). Drop it or use it in a `Mesh2D::validate`.
  - [ ] Nodal fields from Gmsh (`$NodeData`, e.g. a bathymetry view interpolated by Gmsh) are skipped; read them if meshing workflows start carrying the bed in the mesh file.
  - [ ] Periodic Gmsh meshes (`$Periodic`) are rejected: pair the faces if a periodic real-geometry case appears.
- [ ] CSR connectivity. Make `MeshGPUData` the single representation.

### P1.4 Open boundaries and tidal forcing — DONE (except the relaxation band, moved to P1.5)
- [x] **One characteristic OBC.** `CharacteristicOBC` builds the boundary state from the invariants (`u_b = ½(w+_int + w−_ext)`, `√(g h_b) = ¼(w+_int − w−_ext)`, u_t from the upwind side; supercritical sides whole; sonic Riemann states against a dry side) and the kernels take its physical flux `F(q_b)·n` (`BoundaryState::Exact`), independent of the Riemann solver.
  - External data from providers: `StillWater`, `HarmonicTide`, `ParentTimeSeries`, `OceanModelState`, `BoundaryTides`, `InverseBarometer`, closures. Removed `Flather2D`, `HarmonicFlather2D`, `Radiation2D`, `Chapman2D`, `ChapmanFlather2D`, `TSTOBC2D`, `NestingBC2D`, `OceanNestingBC2D`, `SWE2DRhsConfig::with_dt`.
  - Elevation-only forcing: `ElevationOnly::IncomingWave` (simple wave from still water) or `AtRest` (half amplitude for a progressive wave, as before).
  - Gate (`tests/open_boundary_flather_test.rs`, Roe/HLL/Rusanov/EntropyStable/WetDry): reflection ~1e-6; delivered amplitude 0.997–1.001; 45° reflection 0.180 vs theory 0.172 (1D characteristic OBCs reflect oblique waves; see the module docs); flux identical across Riemann solvers; lake at rest 4e-14.
- [x] Spatially varying tidal forcing from NorKyst harmonics: `TidalAtlas` (text format) → `BoundaryTides` (complex IDW onto boundary nodes, coverage check, velocity rotated to mesh axes); `examples/norkyst_boundary_tides.rs` fits reference constants of η, ū, v̄ from 30 days of NorKyst-800 over OPeNDAP (`analysis::fit_reference_constants`, P1 and K2 inferred). Frøya: 187 points, η R² 0.985–0.991, M2 0.70–0.81 m, G 297.6–303.8°.
- [x] `ModelClock` (`time::ModelClock`): BCs (`HarmonicTide`, `BoundaryTides`, `OceanModelState`, which replaces the per-BC epoch), NetCDF output units (the writer claimed `since 1970-01-01` for simulation seconds), validation (`StationValidationResult::compute` now pairs by time).
- [x] Nodal f/u at the run or record midpoint, V₀ at the epoch; corrections stored apart from the constants, so they cannot be applied twice.
- [x] Inverse-barometer level at open boundaries (`InverseBarometer`); Large–Pond Cd held at its 25 m/s value.
- **Frøya result** (`examples/froya_real_data.rs`, 9,653 P2 elements, NorKyst boundary tides, 2025-06-15): stable, 0 clips, 6 min per M2 cycle. The 1 h ramp overshoots (η 1.6 m, a 2.9 m/s jet at the southern boundary at t = 1 h) because it squeezes a spring-tide rise into an hour; `ramp_hours=3` removes it (0.48 m at 1 h, then the tide, η −1.14…+1.06 m). The "basins at +1.04 m through low tide" and the 2.8 m/s "draining flats" of the old evidence are **lone wet nodes in shoreline elements next to land nodes at `LAND_ELEVATION` = +5 m**: η jumps by 0.8–1.0 m between elements at the same vertex, with 1–2.8 m/s. A shoreline-geometry artifact (P1.6 foreshore DEM, P1.3), not the forcing.
- [ ] Nesting relaxation in a sponge/FRS band (`source/`), not in the ghost: moved to P1.5 (same item there).
- [ ] Transport consistency at the boundary: the atlas velocity is NorKyst's depth mean over NorKyst's depth; scale by `h_parent/h_child` (depth is in the atlas) where the beds differ (P1.5).
- [ ] Velocity from z-levels stops at 300 m (NorKyst has no `ubar`/`vbar` on THREDDS); deeper columns extend the 300 m value.

### P1.5 Nesting done properly (NorKyst/ROMS parent)
- [ ] Conserve parent transport: ubar_child = ubar_parent·h_parent/h_child. Read parent `h`, and blend child bathymetry to the parent's in the relaxation band (`ocean_nesting.rs:141-145`).
- [ ] Rotate grid-relative velocities by `angle`, or accept only `*_eastward/northward`. Solve the inverse bilinear map on the curvilinear grid (`netcdf_io.rs:1596-1597`).
- [ ] z-level files: branch on the vertical coordinate and its `positive` attribute. Index 0 is the surface in NorKyst ZDEPTHS files (`netcdf_io.rs:1389`).
- [ ] A Davies / Martinsen–Engedahl flow-relaxation band over N cells (`SpongeLayer2D` needs non-rectangular distance functions).
- [ ] Fix the off-by-one for descending coordinates (`find_bracket` `:1762-1766` + `get_state` `:1443`) and the silent zeros in `get_state`.
- [ ] Flatten parent fields to strided `[time, y, x]` arrays and precompute per-boundary-node stencils at setup (no per-stage RwLock HashMap).

### P1.6 Real-domain inputs
- [x] GeoTIFF georeferencing: the tags were looked up as `Tag::Unknown(n)`, which never matches the `tiff` crate's named variants, so every file silently used the bbox hint. `ModelTransformation` was not supported at all. Frøya's bathymetry was misplaced by up to ~20 km. Fixed, with a CRS check (projected/rotated rasters are rejected), `PixelIsPoint`, GDAL no-data and pixel-centre sampling.
- [x] GeoTIFF: L2 projection onto the element nodes instead of point sampling (2026-09-26). The Frøya GeoTIFF has 0.002° pixels (≈ 100 × 230 m), so elements coarser than that alias the bathymetry.
  - `Bathymetry2D::project`: per-element L2 projection onto Q_N (composite Gauss–Legendre quadrature on sub-cells ≤ half the data resolution). It is limited to the element's data range by Zhang–Shu scaling towards the mean, then made continuous by mass-weighted averaging of coincident nodes (`make_continuous`). Exact for Q_N, volume-preserving, bounded, N + 1 for smooth data, and it does not alias (unit tests in `bathymetry_2d.rs`).
  - Continuity is essential. The element projections alone put −18 m and +1…+5 m on one coastal vertex, and that lone deep node jetted at 8–12 m/s.
  - At 1 km it is not better than sampling at Mausund: centred RMSE 3.6 cm against 3.3 cm, with different station nodes (15.1 m and 16.4 m deep). The example samples by default (`bed=projected` to project). Re-compare once stations are interpolated (P3.1), and at 2 km+ where the aliasing is worse.
  - [ ] The projection shrinks a narrow inlet to one deep node among land nodes (Frøya projected at 1 km: B = −9.7 m among +0.5…+3.8 m, η stuck at +1.2 m). This is the shoreline-cliff artefact below in another place. Drop such nodes to land, or require a wet neighbour, when building the water-only mesh.
- [x] Land/nodata no longer become B = 0: `Bathymetry2D::from_geotiff` is removed. `io::BedRaster` holds the sea bed (dry/no-data pixels at 0) and a land mask rasterised from the coastline on a finer grid (land at `land_elevation`), or a full elevation model (`BedRaster::new`, for a merged DEM). `Bathymetry2D::from_raster`/`select_elements` build the water-only bed.
  - Baking `land_elevation` into the pixels instead (tried first) ramps the bed up to it across a pixel at every coast. At 1 km that closed sounds and cut M2 at Mausund to 0.61 of the gauge's (centred RMSE 15 cm). The coastline must decide land and water point by point, with the sea bed going to 0 at the shore.
  - [ ] The land mask costs ≈ 20 s of setup at Frøya (1.8 M point-in-polygon tests against GSHHS polygons, 4 × 4 cells per pixel): the spatial index below would remove most of it.
- [ ] `LandMask2D::from_coastline_and_bathymetry` counts water shallower than `min_depth` as land (the old Frøya run used 5 m, dropping every tidal flat) and uses nearest sampling. It is unused now; fix or delete it.
- [ ] The Frøya GeoTIFF stores land as 0 and has no land elevations, so there are no intertidal flats: land is a wall at mean sea level (given `LAND_ELEVATION` = 5 m in the example). Real wetting/drying needs a merged DEM (e.g. Kartverket's) for the foreshore.
  - Measured cost (P1.4 Frøya run): a wet node whose element neighbours are land at +5 m (a 5–10 m step within one P2 element) carries an η discontinuity of 0.8–1.0 m against the adjacent element at the same vertex, and 1–2.8 m/s, through the whole tide. These nodes set the run's η maximum and fastest current. Until a DEM exists, try a land elevation just above the highest tide (≈ +1.5 m) or a bed smoothed across the coastline. `froya_real_data land_elevation=1.5` tries the former.
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
- [x] Outgoing-pulse reflection < 1 %; delivered tidal amplitude: `tests/open_boundary_flather_test.rs` for all three fluxes and both split forms, plus oblique incidence and elevation-only forcing (P1.4).
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
- [x] One Riemann solve per interior face, for the split forms (2026-09-26). A face pass (`SplitFormSWE2D::edge_fluxes`, parallel over edges) evaluates F* once per interior face for both sides into a thread-cached buffer, and the element loop reads it. The RHS matches the old one to round-off (21 of 52,920 values differ, by ≤ 4e-19), and the mass fluxes of neighbours are now exactly opposite (`test_face_pass_exchanges_opposite_mass_fluxes`). Frøya, pinned single thread on mains power: RHS 5.3 → 4.85 ms (−9 %); at 24 threads it is neutral (0.62 ms, the extra fork/join costs what it saves).
  - [ ] The collocated (`Standard`) kernel still solves every face twice.
  - [ ] The face pass rebuilds `SWENodeState2D` (two divisions) for both face nodes, and the element pass rebuilds them again for its own nodes. Fusing the pass into the element loop (each element computing the faces it owns, e.g. its right and top edges, with a coloured or two-phase schedule) would save the second pass over memory, which is what cancels the gain at 24 threads.
- [x] Sources per element (2026-09-26): `SourceTerm2D::add_element(&ElementSources, h, hu, hv)`, called once per element. Its default builds the per-node context as before; `SourceTerms2D`/`CombinedSource2D` forward, and Coriolis overrides it with a plain loop (positions only on a β-plane). Frøya, pinned single thread: RHS 4.91 → 4.26–4.53 ms (≈ 11 %).
  - [ ] Override `add_element` in the other hot sources where it pays (wind stress, atmospheric pressure, tidal potential: position- or time-only fields could be precomputed per node).
- [x] 2026-09-26: `hypot` → `sqrt(x² + y²)` in the hot paths (MSVC `hypot` is a C runtime call); a Halley `cbrt` for Manning (2× the CRT's, ≤ 2 ulp); the WetDry HLL takes the node velocities through a face-aligned core (`hll_flux_face_aligned`) instead of dividing momenta back out. Implicit damping 3.5 → 1.3 ms, post-processing 0.84 → 0.41 ms (single thread).
- [x] The face-aligned HLL core, measured on mains power pinned to a Zen 5 core: RHS 5.8–6.7 → 5.4–5.7 ms (≈ 7 %). Benchmark this laptop only on mains power, pinned (logical CPUs 0–7 are Zen 5, 8–23 the slower Zen 5c); on battery everything runs ≈ 2× slower and noisier.
- [ ] Explicit SIMD (`pulp`, or fearless_simd) pays only once kernels vectorise across nodes or elements: the node passes (damping, positivity, wet/dry) first, then batched face fluxes. faer is not on the hot path (element-local 9-node operators).

### P2.3 Fused in-place stepper
- [ ] Limiter, positivity and wet/dry in one parallel in-place pass. Today the "parallel" paths `to_vec` every field, allocate per element, and copy back serially (`limiters/swe_2d.rs:626-725`, `wetting_drying.rs:526-541`): 1.5–2.5× of wall time.
  - Limiters done (P0.19: in-place `par_chunks_exact_mut` over the SoA fields). The `wetting_drying.rs` parallel path is done too (P1.2: in place, plus a node-parallel implicit damping pass). Still open: the per-stage `Vec` of cell averages and vertex bounds (move to a workspace with P1.1). Then fuse the limiter, the wet/dry correction and the implicit damping into one pass.
  - 2026-09-26: the positivity limiter computes each element mean in its element kernel (no separate pass or `Vec`), and `post_process` skips a `Positivity(h ≤ h_dry)` limiter when wet/dry is on, since the wet/dry correction applies the same limiter to every element it would change (bitwise identical; `redundant_positivity_pass_is_skipped_exactly`). The Kuzmin paths still allocate averages and vertex bounds per stage.

### P2.4 Water-only work
- [x] Mesh water only: `Mesh2D::retain_elements` (faces towards dropped elements become coastline walls). Frøya keeps only elements with a water node.
- [ ] Limit only troubled cells.

### P2.5 Local time stepping or implicit free surface
- [ ] One 250 m element in 1300 m water forces ~22× more global steps. Multirate/LTS SSP-RK or semi-implicit free surface (SLIM/Thetis practice): 2–20×.

### P2.6 Benchmarks (was P1.6)
- Measured 2026-09-26: Frøya, 9,653 P2 elements (87k nodes), dt ≈ 0.65 s: 8.2 ms/step on 24 threads, so a 30-day validation run takes ≈ 9 h. Scaling stops at ≈ 8 threads (44.6 → 9 ms from 1 to 8–24 threads): the laptop's 4 Zen 5 + 8 Zen 5c cores and its power limit, not serial code. `froya_real_data profile=N` times each phase (RHS, post-processing, damping, dt) at 1 and all threads; single-thread RHS ≈ 85 % of the step.
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
- [x] **Gauge validation path (2026-09-26).**
  - `scripts/kartverket_gauge.sh` fetches Kartverket observations (the tide API serves a year of hourly data in one request; it has no harmonic constants, so fit the record).
  - `examples/tide_gauge_fit.rs` fits a gauge record and compares it with an atlas point.
  - `froya_real_data` samples stations and reports constituent and time-series skill.
  - `norkyst_boundary_tides points=` fits NorKyst at the gauges.
- [ ] Real Kartverket (H, G) for Bergen, Stavanger, Trondheim, Kristiansund: `./scripts/kartverket_gauge.sh <name> <lat> <lon> 2024-07-01 2025-07-01` (codes and positions from `tide_request=stationlist`). Only Mausund and a synthetic `heimsjo.txt` ship. `norwegian_stations` positions are approximate (Heimsjø is at 63.425, 9.102).
- [ ] Run the model against NorKyst-800 barotropic tides for a test period. This needs P0.12, P0.20, P1.3–P1.5.
- [ ] Compare with ADCP currents (`analysis/adcp.rs`). Add a tidal-ellipse fit and a complex-difference skill metric.
- [ ] Document skill scores (RMSE, bias, correlation) per station.
- [ ] Harmonic analysis: `fit_reference_constants` now has a Rayleigh check, inference (equilibrium or from a reference fit), `resolvable_constituents` and `predict`; `HarmonicAnalysis::fit` still has none of them. Port its callers (`StationValidationResult::compute_with_harmonics`, `validate_stations`) or give it the same guard; an `AnalysisError` type instead of asserts and `String` errors.
- [ ] Station series are η only. Add depth-averaged velocity, which the ADCP comparison and the tidal-ellipse fit need. Sample at the gauge position by interpolating in the element, not at the nearest node ≥ 3 m deep (Mausund's node is 199 m away, and 7.6 m deep).

### P3.2 Cost-vs-accuracy benchmark against ROMS
- [ ] Pareto benchmark on the same hardware: M2/S2/K1/O1 error at tide gauges vs core-hours, ROMS 2D at 800/400/200 m vs DG P1–P4. This is what settles "faster than ROMS" (`REVIEW.md` §5.5).

### P3.3 Long-run stability
- [ ] 30+ day real-domain tidal run: stable, mass-conserving, no spurious residual currents at rest.

---

## Priority 4: 3D Rebuilt on the ROMS Recipe (`REVIEW.md` §2, §3, §7 Phase D)

The 2026-02-11 plan (vertical infrastructure → mode splitting → mixing → physics → validation) is superseded. The scaffolding exists, but the coupling must be restructured before new physics is added. See `3D_TODO.md` for background (its checklists predate the current code).

### P4.1 Mode splitting

**Before PR 1:** SSP-RK3 wrapped a Forward Euler barotropic subcycle that ran in every stage. It was first-order, M2 lost 3–14 % per period, and it cost 6·n_bt 2D RHS per step (Shchepetkin & McWilliams 2005; `REVIEW.md` §2).
**After PR 1:** one filtered pass per step, 0.02–0.1 % barotropic amplitude loss per period, ≈ 3.9·n_bt 2D RHS per step.
**After PR 2:** G holds only what the 2D module cannot compute: the baroclinic PGF, the shear dispersion and the 3D stresses. Its step average is AB3-extrapolated, and the slow coupling is second order.
**After PR 3:** each step also records DU_avg2, the transport that moved η (η̄ − ηⁿ = −Δt∇·DU_avg2), for the 3D continuity of P4.2.

**Target step n → n+1 (plan of 2026-09-26):**
1. One 3D RHS at tⁿ, after refreshing density. It is reused as the first 3D RK stage.
2. Slow forcing G = D·(⟨R₃D(u)⟩ − R_adv+Cor(ū)) + (τ_s − τ_b)/ρ₀ (see the design decisions). Its step average comes from Gⁿ, Gⁿ⁻¹, Gⁿ⁻² by AB3 with variable steps (lower order on the first two steps).
3. One barotropic pass over [t, ≈ t + 1.5·dt] with n_bt = ⌈dt/dt_bt,max⌉ from the 2D CFL. Accumulate:
   - the primary-weighted η̄ and transport (DU_avg1), which become the new barotropic state;
   - the secondary-weighted face-flux transport (DU_avg2), for P4.2.
4. 3D RK stages with Hz at each stage time from the pass, and density recomputed every stage.
5. Implicit vertical diffusion, then reset the depth mean of u to DU_avg1/D. The existing reconciliation does this.

**Design decisions (deviations from the ROMS recipe):**
- **Fast-mode sub-steps are SSP-RK3, not generalised forward-backward (AB3-AM4).**
  - FB needs continuity and momentum evaluated separately at staggered times. The DG upwind fluxes couple h and hu in both equations.
  - The Zhang–Shu positivity guarantee that wet/dry relies on needs an SSP scheme.
  - Cost is about 4.5·n_bt RHS per step (vs 6·n_bt today) at the DG CFL. It reuses `SWEPhysics2D` (wet/dry, limiter, `compute_rhs_into`).
  - FB can come later as an optimisation once the coupling is verified.
- **G is Thetis-style, not the ROMS `rufrc`** (revised 2026-09-26, PR 2). The plan was the full PGF in the 3D RHS with G = ∫R₃D dz − R₂D(q̄ⁿ) + stresses. It was dropped as structurally unsound here:
  - In that construction each step's fast pass adds only R₂D(q) − R₂D(qⁿ). The bulk of the pressure force on the slow barotropic mode (tides, seiches) would come from ∫R₃D.
  - The 3D PGF (`compute_pressure_gradient`) differentiates η within each element and never lifts the jumps at element faces (P4.3). So the slow mode would lose the DG pressure coupling between elements.
  - Instead, the 2D module owns the depth-mean flow: −g∇η with its DG face coupling, advection of ū, Coriolis on ū, and its friction on ū.
  - G = D·(⟨R₃D(u)⟩ − R_adv+Cor(ū)) + (τ_s − τ_b)/ρ₀: the same 3D advection + Coriolis operator applied to columns of uniform ū removes the mean-flow part. What is left is the baroclinic PGF, the shear dispersion −∇·⟨u′u′⟩ and the stresses. For unsheared flow G is exactly the stress.
  - The 3D PGF stays baroclinic-only (P0.6).
  - Rule: configure the 2D module with the same Coriolis parameter, and without wind/friction sources that duplicate `Forcing` (`Hydrostatic3D` module docs).
  - Revisit the ROMS form once P4.3 lifts the 3D pressure/η jumps at faces.

**PR 1: one barotropic pass** — done on `feat/p4-1-barotropic-pass`
- [x] The fast pass steps transport (h, hu, hv) in a `SWESolution2D` through `SWEPhysics2D`: positivity, limiter, wet/dry and implicit damping at every stage. The unguarded 1/h (`Hydrostatic3D::compute_rhs_2d`) is gone.
  - `Solution3D` still stores η/ū/v̄ as the slow barotropic state, with ū = D̄ū/D̄ after the pass (guarded for h̄ ≤ 0). The 3D kernels read η; moving them to h is P4.2 work.
- [x] One SSP-RK3 barotropic pass per baroclinic step. η/ū/v̄ leave the 3D RK combination: the 3D stages get constant rates, i.e. the barotropic state linearly interpolated to each stage time.
- [x] Shchepetkin & McWilliams (2005) power-law filter (p = 2, q = 4, r = 0.284) over ≈ 1.3·dt, replacing the Hann window over 2·dt.
  - The r term makes the second moment about t + dt almost zero. Measured per-period amplitude loss at 50 steps per period: Hann 5.0 %, power law with r = 0: 3.2 %, r = 0.284: 0.02–0.1 % (n_bt 40–7). r = 0.3 amplifies.
  - The centring iteration needs n_bt ≥ 4 (`MIN_BAROTROPIC_SUBSTEPS`).
- [x] Barotropic CFL control: n_bt = max(4, ⌈dt/dt_bt,max⌉) every step, at `min(barotropic_cfl, max_cfl)`.
- [x] Density refreshed at every 3D RK stage (`Simulation3D` stage hook).
- [x] One `step`. The 3D and 2D stage buffers and the field buffers are reused across steps, and `Solution3D::copy_from` copies in place. Still allocating: `compute_rhs_3d` internals (P4.5).
- [x] Gate: `seiche_amplitude_is_kept_by_the_mode_split`. It loses 0.035 % per period more than the 2D model at n_bt = 23 (was ≈ 23 % per period), and conserves volume to round-off.
  - The 100-period form was cut to 10 periods: the loss per period is constant, and the test must stay cheap in debug CI (16 s).

**PR 2: consistent slow forcing** — done on `feat/p4-1-slow-forcing`
- [x] G without the double count: D·(⟨R₃D(u)⟩ − R_adv+Cor(ū)) instead of D·⟨R₃D(u)⟩, with AB3 step averages (`step_average_weights`, variable steps; the history restarts when a run does not continue).
  - In a nonlinear unsheared seiche the old G drifted from the 2D model by 0.14 of the amplitude in two periods; the new G drifts by 1.8e-3, falling with dt.
- [x] Surface and bottom stress into G, as (τ_s − τ_b)/ρ₀ from `Forcing`. The vertical diffusion now also divides by the model's ρ₀, not 1025, so G and the columns agree.
  - τ_b is still the user's constant; a quadratic drag from the bottom-layer velocity, consistent with the 2D friction, is P4.4.
- [x] ~~Full PGF in the 3D RHS~~: dropped, see the design decisions.
- [x] `ModeSplitPhysics` trait: the splitter's view of the 3D model (2D module, 3D RHS, slow forcing, implicit vertical terms, stage hook). `Hydrostatic3D` implements it; `ModeSplitIntegrator::step(state, physics, dt, t)`.

**PR 3: transport for 3D continuity** — done on `feat/p4-1-du-avg2`
- [x] DU_avg2 (`BarotropicTransport`, `ModeSplitIntegrator::barotropic_transport`): the nodal (hu, hv) and the face mass flux F*_h of every RK stage of the pass, weighted by W_j·b_s/n_bt (secondary filter weights W_j = Σ_{m≥j} w_m, SSP-RK3 weights b = (1/6, 1/6, 2/3)).
  - The 2D kernel writes F*_h per element face node on request (`compute_rhs_swe_2d_face_mass_into`, serial and parallel; `BarotropicPhysics`). The plain RHS is unchanged bit for bit.
  - `BarotropicTransport::divergence_into` is the kernel's strong-form divergence. η̄ − ηⁿ = −Δt·∇·DU_avg2 holds to 4e-13 of the η change for `Standard` and `EntropyStable`: the mass part of the two-point flux is {{hu}}, so the flux-differencing volume term is the nodal D·(hu, hv).
  - `WetDry` elements with a dry node use subcell finite volumes, and the positivity limiter and wet/dry correction change nodal depths. There only the element balance ∫(η̄ − ηⁿ) = −Δt∮F*_h holds, to round-off. P4.2's per-level correction has to work with that: in those elements, correct at the element (or subcell) level, not the nodal one.
- [x] Fixed on the way (PR 1 bug): η at the end of a step came from the RK combination of constant rates, which can leave a dry node a hair below its bed. The next pass then started from h ≈ −1e-16 and the positivity limiter "emptied" the element (15 clips in 10 steps on a beach). Now η = h̄ + B and ū = D̄ū/D̄ are set exactly from the filtered state; for h̄ ≥ 0, (h̄ + B) − B ≥ 0 in floating point.
- Gate tests: `barotropic_transport_moves_the_free_surface` (nodal identity, both formulations; with primary weights or without the face flux it fails by O(1)), `barotropic_transport_balances_every_element_with_wetting_and_drying` (beach, 0 clips), `face_mass_flux_balances_each_element` (kernel: RHS unchanged, serial = parallel, ∫dh/dt = −∮F*_h for all three formulations).

**Gate tests:**
- [x] 2D–3D equivalence: `unsheared_flow_matches_the_2d_model`. A nonlinear seiche (η/H = 0.05) over two periods: 1.8e-3 of the amplitude at 50 steps per period (old G: 0.14).
- [x] Seiche amplitude: `seiche_amplitude_is_kept_by_the_mode_split`. The filtered splitting loses 0.035 % per period more than the 2D model, measured over 10 periods (was ≈ 23 % per period).
  - The "< 1 % over 100 periods" target was too strict for any filtered-reset splitting. The loss is ≈ μ₂ω²/2 per step, first order in Δt with a constant ≈ 1e-3 of a plain average's. For M2 at Δt = 300 s it is ≈ 0.008 % per period.
- [x] Temporal convergence: `slow_forcing_is_integrated_to_second_order`. The AB3 steps are third order; the two starting steps set second order globally (measured ratios 4.5 → 4.1). The filter's μ₂ term is separate: see the previous item.
- [x] Wind setup: `wind_setup_balances_the_surface_stress`, slope 1.0001 × τ/(ρ₀gD).
- [x] Ekman: `wind_drives_the_ekman_inertial_transport`, 3.9e-4 of τ/(ρ₀f) from the inertial-Ekman solution.
- [x] Column momentum after one implicit diffusion step = Δt·(τ_s − τ_b)/ρ₀: `column_momentum_changes_by_the_stress_impulse`.

### P4.2 Consistent continuity, tracers and momentum
- [ ] Hz-weighted per-level face fluxes corrected so their vertical sum equals DU_avg2 (available since P4.1 PR 3: `ModeSplitIntegrator::barotropic_transport`, nodal + face parts, see its docs for the `WetDry` caveat); η advanced by the divergence of the same flux.
- [x] Ω stored at w-points (not layer centres re-averaged to faces) — done in [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2).
- [ ] Assert Ω(0) = 0 instead of forcing it with the linear correction.
- [ ] One 3D face-state helper for physical boundaries shared by the Ω, momentum and tracer kernels (today each has its own copy of the wall logic; P0.7/P0.22 were the same bug found twice). It should take the boundary tag, so 3D open boundaries (volume flux from the 2D OBC / nesting, tracer from `TracerBoundaryCondition3D` on inflow) are added in one place for all three.
- [ ] Step Hz·C and Hz·u (inventory form), then divide by the new Hz. Today a 1 m tide over 20 m of water pumps salinity by about ±1.7 psu.
  - Measured side effect (2026-09-26): with a T/S-dependent EOS, the drift of T and S with η feeds a spurious baroclinic PGF into the mode-split G. It damps a barotropic seiche by 0.6 % per period (P1), or amplifies it by 0.4 % per period (P2), independent of amplitude and dt. The seiche tests use a density independent of T and S until this is fixed.

### P4.3 Pressure gradient and vertical grid
- [ ] Balanced PGF: Shchepetkin & McWilliams (2003) density-Jacobian, or z-level interpolation. Lift pressure/η jumps at element faces.
  - Today: ~1e-4 m/s² spurious forcing in a stratified fjord at rest (P3/250 m), the size of real estuarine forcing.
- [ ] rx0/rx1 diagnostics and ROMS-style bathymetry smoothing.
- [ ] Vtransform 2 + the true Vstretching 4. `ROMSVstretching4` is not ROMS Vs4, and `hc` is unused; Ω must use ∇·(Hz·u) once Hz varies horizontally.

### P4.4 Vertical mixing and boundary layers
- [ ] GLS k-ε (NorKyst's closure) as a per-column solve reusing the tridiagonal solver (Umlauf & Burchard 2003; Warner et al. 2005). KPP as an alternative.
- [ ] Convective adjustment (low-shear unstable columns currently get background mixing).
- [ ] Implicit quadratic bottom drag with a log-layer Cd; consistent with 2D friction.
- [ ] Surface heat flux Q_net/(ρ₀c_p) (currently fed a buoyancy flux). The momentum stresses use the model's ρ₀ since P4.1 PR 2; the heat flux still needs it.
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
- [x] Wind setup τ/(ρgD); Ekman transport τ/(ρ₀f); column momentum after one implicit diffusion step = Δt·τ/ρ₀ (P4.1 gate tests).
- [x] Temporal convergence of the coupled scheme; barotropic energy decay (P4.1 gate tests).
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

- [ ] Move the remaining examples (`profile_cpu`, `quick_profile`, `high_res_benchmark`; Frøya is done) from the legacy `ssp_rk3_swe_2d` stepper and its `SWE2DTimeConfig` onto `Simulation` + `SWEPhysics2D`. This is the precondition for deleting the legacy integrators below. The two paths currently duplicate limiter and wet/dry dispatch.
- [ ] Delete the legacy integrators (`ssp_rk3*.rs`, three `coupled_swe_tracer` variants, `burn_ssp_rk3.rs`; ~2.4k lines). Make `CoupledState2D` implement `Integrable`; one time loop (fold `Simulation3D::run_with_callback`). `Simulation3D` still fires callbacks at the first step past each interval (drift); `Simulation` now lands on them exactly.
- [ ] Collapse the duplicate config enums (`SWEFluxType2D` vs `StandardFlux2D`; three limiter enums).
- [ ] Prune ~200 root re-exports to a `prelude`; export `Simulation3D`.
- [ ] Error handling:
  - `SimulationResult` → `Result<SimulationStats, SimulationError>`;
  - reachable panics (`timeseries_reader.rs:198`, `geotiff.rs:239-245`);
  - library `eprintln!` in `tide_gauge.rs`.
- [ ] Feature-gate `tiff`/`shapefile`/`geo` behind `geodata`.
- [ ] `cargo clippy --no-default-features --features parallel,simd --lib` reports 130 warnings (2026-09-25), but the CLAUDE.md checklist says none. CI runs clippy without `-D warnings`, so nothing stops new ones. Clear them, then add `-D warnings` to the CI job.
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
- [ ] Biological coupling (NPZD). Salmon-lice dispersion is particle tracking, now F.2 under "Application target".
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
| Tests | 1,090 lib + 88 integration + 59 doc (no-default) | + the P1.7 gating physics tests |
| Lake-at-rest, steep nodal B | Split forms: RHS ≤ 4e-12 for any nodal B, P1–P4. Real Frøya bathymetry (0–404 m, `WetDry`, P2): max \|u\| 1.5e-10 m/s after 1 h (the collocated scheme: 0.9–2.3 m/s or blow-up) | residual < 1e-10, p = 1–4 |
| Mass conservation, face-discontinuous B | Machine precision (P0.13; split forms by construction) | machine precision |
| Open-boundary reflection (Flather) | < 1e-5 (P0.12, `tests/open_boundary_flather_test.rs`) | < 1 % |
| SWE convergence | P1 1.67, P2 3.02 (linearised, flat, periodic; thresholds 1.5/2.5) | N+1, nonlinear, bathymetry, curved |
| Allocations per 2D step (production stepper) | `Simulation` path: 0 B warm (without limiter/viscosity); legacy `ssp_rk3_swe_2d`: ~33 full-field arrays | 0 |
| 2D tidal cost vs ROMS (model estimate) | ~47× core-hours | ≤ 1× per unit accuracy (P3.2) |
| Validated against observations | Mausund (Kartverket MSU), 15 days at 1 km: M2 1.03×, +5.4° (≈ +2.3° at 500 m); tidal-prediction RMSE 9.8 cm | 5+ tide gauges, ADCP |
| Multi-day stability | One M2 cycle on Frøya (9,653 P2 elements, 11.4 min wall, 0 clips) | 30+ days, real domain |
| 3D | Scaffolding; 2 blockers (tracer constancy P4.2, σ PGF P4.3); vertical advection ×D (P0.16) and the tracer wall leak (P0.22) fixed; mode splitting: one filtered barotropic pass (0.035 % seiche damping per period), second-order slow coupling, wind setup/Ekman to < 0.1 % (P4.1 PR 1–2; DU_avg2 is PR 3) | stratified lake-at-rest, constancy, lock exchange |
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
