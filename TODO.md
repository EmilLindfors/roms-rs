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
  - The 3D layer is scaffolding with three blocker-level math errors. (Vertical advection ×D since fixed, P0.16; tracer constancy and the σ pressure gradient since fixed, P4.2/P4.3, 2026-09-30.)
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
       - The positivity CFL (0.29 at P3 against 0.42 at P2) costs P3 a factor of ≈ 1.45 in dt. Relaxing it (a subcell bound, or limiting only near dry fronts) would favour higher order. Since 2026-09-29 it is relaxed in fully wet elements up to 0.9 of the linear limit (P2.1), which is 0.66/1.04 at P3 against 0.94/1.36 at P2 (SSP-RK3/SSP-RK(4,3)): the ratio in dt is now ≈ 1.3–1.4 where the linear limit binds; re-run the P2/P3 comparison.
     - Next steps:
       - **Coastline mesh (2026-09-28): the best gauge skill so far**, 2.3 cm centred against the prediction and 4.7 cm against the observations over hours 24–72, at ≈ 2 min per model hour (P1.3). The 15-day run for the harmonic numbers should use it (`mesh=data/froya_coast.msh lts=8 hours=384`, ≈ 13 h on an idle machine at 12 threads with SSP-RK3; with SSP-RK(4,3), the example's default since 2026-09-29 (P2.1), now ≈ 38 s per model hour at 24 threads with the relaxed positivity bound, shorelines included (P2.1, measured 2026-09-29 over 2 model hours), so ≈ 4 h; run nothing else alongside, since local time stepping degrades badly under contention, P2.5).
       - The 500 m, 15-day run (≈ 2.5 h) for the harmonic numbers at resolution.
       - ~~Station interpolation at the gauge position, instead of the nearest node ≥ 3 m deep.~~ Done 2026-09-27 (P3.1), with a finding: at 1 km Mausund cannot be sampled at the gauge. See "Station sampling at 1 km" below.
       - Friction (Manning n = 0.025 everywhere) is the other phase suspect. The bed builder is not: the library `BedRaster` bed (P1.6, 2026-09-26) gives 3.3 cm centred RMSE (hours 24–72) sampled at the nodes and 3.6 cm projected, against 4.1 cm with the old builder, all at 1 km.
   - **Finding: NorKyst-800 has almost no N2 here.** At Mausund over June 2025, N2 is 0.027 m in NorKyst against 0.156 m observed (both 30-day fits; the year fit gives 0.154 m). Q1 is 1.9× the observed, K1 and S2 are 12–21 % high, M4 is 5× too weak, and M2 agrees (1.026, −0.5°). NorKyst's own complex-difference RSS against the gauge is 0.14 m, 0.13 m of it from N2. The Frøya run therefore re-infers the boundary atlas's N2 and Q1 from M2/O1 with Mausund's ratios (`gauge_ratios=N2,Q1`, the default). Report the N2 deficiency to MET, and check other NorKyst gauges (Bergen) for it.
   - **Station sampling at 1 km (2026-09-27, P3.1).** The gauge's own position lies in a shoreline element (the harbour is not resolved). Sampled there at the nearest point ≥ 3 m deep (51 m away, 3.3 m deep), the model sat in a perched pocket: mean +1.38 m, M2 0.30× the gauge's, no current. The example now samples only in submerged elements (every node ≥ 3 m deep), which at 1 km is 1.29 km from the gauge in 55 m of water. There, over hours 24–72 from 2025-06-15: M2 0.997×, −8.2° (2-day fit, N2 inferred), centred RMSE 4.4 cm against the gauge's prediction (3.2 cm at the old node, 325 m away and 16 m deep). The depth-averaged M2 current there is 3.6 mm/s against NorKyst's 0.25 m/s at its point 606 m from the gauge. Each gauge now also gets a station at its nearest atlas point, but that point lies in a shoreline element too.
     - 1 km: sampled 1.2 km from NorKyst's point (21 m deep). M2 ellipse 0.056 m/s at 160° against NorKyst's 0.246 m/s at 149°.
     - 500 m (4.5–5.8 ms/step, 30–40 min for 3 days): the gauge is sampled 379 m away, 9 m deep, with M2 1.000×, −9.4° (2-day fit) and centred RMSE 3.5 cm against the prediction, 5.9 cm against the observations. NorKyst's point is sampled 657 m away, 11 m deep: M2 current 4.7 mm/s, 2.4 mm/s at the gauge.
     - **Answered 2026-09-27: mostly the point-sampled bed** (`scripts/norkyst_current_comparison.py`: M2 ellipses of the run and of NorKyst's hourly ū, v̄ fitted identically over hours 24–72 at every NorKyst point in the domain).
       - Peak speeds are not comparable: NorKyst's ū is the total flow (coastal current, wind), and the model's peak speed was 0.45 of NorKyst's, tides only against everything.
       - Tide against tide, the model's M2 current is 0.89 of NorKyst's where both are over 120 m deep (depth ratio 1.00), but only 0.41 within 3 km of Mausund. The 50 m DEM there has 20–30 % land (skerries), with water of median 17–20 m between them; point samples at nodes 500 m apart land on skerries (1.5–3 m nodes where NorKyst has 45–55 m) and close the sounds.
       - With the L2-projected bed (`bed=projected`, now the example's default) the ratio near Mausund is 0.63, 0.88 domain-wide (0.83 point). At the NorKyst-point station (sampled 410 m away instead of 1.2 km) the M2 ellipse is at 157° against NorKyst's 149°, lag 65° against 62°, 0.071 against 0.246 m/s. The gauge improves too: centred RMSE 4.0 against 4.4 cm (prediction), 6.3 against 6.6 cm (observations).
       - In shallow water the comparison is limited by NorKyst itself: its smoothed bed is 2–3× deeper than the DEM where NorKyst has 3–60 m, so its ū is not the child's for the same transport. The η M2 ratio of 1.11 is aliasing (a 2-day fit lumps N2 and S2 into M2, and the model's N2 is gauge-corrected).
       - 500 m, projected (fit over hours 24–50): 0.86 domain-wide, 0.88 over 120 m, 0.68 within 3 km of Mausund (1 km: 0.61 over the same hours). The model is at ≈ 0.9 of NorKyst wherever the beds agree and approaches it near Mausund as the mesh refines. But at the NorKyst-point station (237 m away, 41 m deep) the M2 current drops to 0.010 m/s, in NorKyst's direction (148° against 149°). That is a local lee of islands the 500 m mesh resolves and NorKyst's 800 m grid (45–55 m deep there) does not have. The gauge at 500 m: centred RMSE 3.5 cm (prediction), 5.8 cm (observations), as with point samples.
       - [ ] Which is right near the skerries, the model's lee or NorKyst's open flow, needs observed currents (farm surveys, see above) and a coastline-fitted mesh (P1.3).
   - Re-run the Bergen NorKyst fit too: it used the truncated periods fixed in P0.15 ([PR #2](https://github.com/EmilLindfors/roms-rs/pull/2)).
2. **Shoreline cliffs (P1.6): fixed with Kartverket's topobathy model (2026-09-26).** Lone wet nodes next to land nodes at `LAND_ELEVATION` = +5 m had carried η jumps of ~1 m and 1–2.8 m/s through the tide. The topobathy model has land heights and depths in one grid (`scripts/kartverket_topobathy.sh`, `data/froya_topobathy.tif`). With it, plus land capped at 5 m and isolated wet nodes raised, the Frøya 1 km run has no cliff nodes: max |u| 0.9–1.4 m/s, and η extremes are only mm films. Centred RMSE at Mausund is 3.2 cm (point) / 3.1 cm (projected) against 4.1 cm on the old builder. Open: the sea part's vertical datum (below, P1.6).
3. **Quick checks on the Mausund sub-domain (2026-09-29).** A 22 × 19 km box around the gauge (`bbox=8.45,63.78,8.90,63.95`, `data/mausund_coast.msh`, its own atlas `data/mausund_boundary_tides.txt`; recipe in `docs/gmsh-meshes.md`) costs a quarter of Frøya: ≈ 30 min for 3 days alone on the machine. Always with `tide_transport=3` (see P1.4). Its absolute skill is below the full mesh's (3.1–4.2 cm centred against 2.3 cm), so use it for relative comparisons.
4. **Then let the error budget choose:** geometry (P1.3) if the coastline dominates the error, cost (P2, profile first) if run time does.
5. **η-based limiting is on hold.** The subcells limit the partially dry elements, and tides are smooth. Revisit only if a real run shows oscillations.

Build note: default features need the `roms-rs` conda env (`conda activate roms-rs`) for HDF5/netCDF. Otherwise use `cargo test --no-default-features --features parallel,simd`.

---

## Application target: currents around a salmon farm

Goal (added 2026-09-26): resolve currents at a farm site (cages ≈ 40–60 m across, nets 20–40 m deep) inside a NorKyst-driven coastal domain, and track sea-lice larvae, feed and faeces from the cages. A 3D visualisation (Bevy, separate crate) reads the model output; it is not part of `dg-rs`.

This section only orders the existing items for that goal and adds the two farm-specific ones (F.1, F.2). Everything else lives in its own priority section.

**Stage A: 2D farm-scale currents.** Depth-averaged, so it answers tidal exposure and the wake of the farm, not the surface layer.
1. P1.3 geometry (blocker): coastline-fitted meshes with elements of tens of metres at the farm and kilometres offshore. General quadrilaterals are supported in 2D since 2026-09-26, and all-quad Gmsh meshes (MSH 4.1/2.2) load since 2026-09-26 (see `docs/gmsh-meshes.md`). A coastline-fitted Frøya mesh is built from the topobathy model since 2026-09-27 (`scripts/gmsh_coastline_mesh.py`, `froya_real_data mesh=`); since 2026-09-28 without the degenerate quads that set its time step (1.84× faster), it costs ≈ 4× the 500 m grid per model hour, mostly in the local time stepping's stage dependency (see P1.3). Triangles or quad-dominant meshes and CSR are still open.
2. P1.6 shoreline cliffs and bathymetry: done at 50 m (2026-09-26: `BedRaster`, `Bathymetry2D::project`, Kartverket's topobathy model, `raise_isolated_wet_nodes`). Farm-scale meshes (tens of metres) need the service's 1 m level, which has depths only inside surveyed projects: merge it over the 50 m model where it has data (see P1.6).
3. ~~F.1 cage drag (2D form)~~ done 2026-09-26 (`CageDrag2D`, `with_cage_drag`).
4. ~~P2.5 local time stepping~~ done 2026-09-27 (`MultirateSSPRK3`: 2.4–2.8× on a 20 m farm mesh; with horizontal viscosity since 2026-09-28; see P2.5 for the follow-ups).
5. ~~P1.5 nesting of NorKyst ū, v̄, η and the P1.6 atmospheric forcing reader~~ done 2026-09-27 (`OceanModelState` with transport scaling, relaxation band and bed blending; `AtmosphereReader` + `GriddedAtmosphere2D`; gauge-corrected tides with NorKyst's residual, `with_tidal_correction`; see P1.5 for the follow-ups).
6. ~~P3.1 current validation~~ tooling done 2026-09-27: station series of η, ū, v̄ sampled at the station by evaluating the DG polynomial there (`PointLocator2D`, `Probe2D`; F.2 reuses them), tidal-ellipse fit and complex difference (`fit_tidal_ellipses`), ADCP comparison paired by time. Still to do: real current data (farm-site surveys, see P3.1).
7. ~~F.2 particle tracking in 2D~~ done 2026-09-28 (`particles::ParticleTracker2D`: RK4 on the DG velocity, random walk, walls, open exits, stranding; online or between snapshots; see F.2 for the follow-ups).
8. ~~Minor: `SpatiallyVaryingManning2D` as `BottomFriction2D` (P1.2); the vacuous tests in P1.7~~ done 2026-09-28.

**Stage B: 3D.** Lice larvae live in the upper few metres and respond to salinity, so dispersion needs the stratified surface layer.
- P4.2 (tracer constancy: done 2026-09-30; still 3D open boundaries for momentum and T/S profiles, momentum in inventory form), P4.3 (balanced PGF: done 2026-09-30; vertical grid), P4.4 (GLS, quadratic bottom drag), P4.5 (3D wet/dry: done 2026-09-30; vertical advection order; parallel, non-allocating 3D kernels), rivers (P1.6 volume sources, P5.1), then P4.6 validation.
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
  - [ ] Farm runs need a spun-up tide and horizontal viscosity (found with `viz/`, 2026-09-27/28). In the fjord-farm scenario the fastest water was a jet through the cages. Drag minus no-drag at the west cage centre, 50 model minutes, P2, `examples/local_time_stepping_farm.rs` setup:
    - Abrupt start, inviscid (the example as it is): the tide switched on at full amplitude from rest sends a ≈ 110–130 mm/s surge past the farm at t ≈ 5–10 min. The drag opposes it (−5.5 mm/s at 10 min), and the deficit then *stays*: −6.9 mm/s at 50 min and −6.7 at 2 h, while the tidal current there is 2–5 mm/s (the fjord is short and closed, so the tide is nearly standing). Local and global stepping agree to 0.01 mm/s, so the scheme is not at fault.
    - The persistence comes from the model, not the ocean: without horizontal viscosity nothing removes a 50–100 m eddy but bottom friction (≈ 1e-7 /s at 90 m) and weak upwind dissipation (the largest anomaly within 300 m: 9.0 → 7.3 mm/s over 2 h). With ν = 1 m²/s (`with_viscosity`, global stepping) it decays with e-folding ≈ 1000 s ≈ L²/ν: −3.5 → −1.5 → −1.1 mm/s at 10/20/30 min (largest 4.5 → 1.9 → 1.3), and the background flow is unchanged (no-drag velocities equal to 0.01 mm/s).
    - With a 1 h ramp (`HarmonicTide::with_ramp_up`) there is no surge: flood peaks at 19 mm/s at the farm and the drag takes 0.35–0.65 mm/s (3–4 %) off it. Still inviscid, that deficit then does not decay after the flow reverses (−0.67 mm/s at 2 h).
    - Done 2026-09-28: the tide ramps up over 1 h in `examples/local_time_stepping_farm.rs` (`ramp=`) and in the viewer (`--ramp`). At 15 min, local stepping still does 2.60× less RHS work, and local and global agree to 1.6e-6 m/s.
    - To do: run farm cases with viscosity (affordable under LTS since 2026-09-28: `local_time_stepping_farm nu=1`, 2.1× the inviscid LTS cost), and choose ν (constant 1–10 m²/s or Smagorinsky with a background: at these shears Smagorinsky alone gives ν ≈ (0.1·20 m)²·|S| ≈ 5e-4 m²/s, i.e. nothing). A farm-wake study also wants a site with real tidal currents.

### F.2 Lagrangian particle tracking
- [x] Point location: find the element holding a point and its reference coordinates (2026-09-27, P3.1: `mesh::PointLocator2D`, a bucket grid over element bounding boxes and Newton on the bilinear map). Curved (isoparametric) elements, once P1.3 has them, need the same Newton on the higher-order map.
- [x] 2D tracking (2026-09-28, `src/particles/`): `ParticleTracker2D` moves `Particle2D`s through a `ParticleVelocity2D`, sampled as the element polynomial at the particle (`interpolation_weights_into` on the stack, no allocation).
  - [x] Velocity: `SWEVelocity2D` is (hu, hv)/h of a solution, steady or linear in time between two snapshots (output files, or online: the previous and current state a `Simulation` callback hands over); `NodalVelocity2D` is any nodal field.
  - [x] Locating: every move walks the straight segment face by face from the particle's element (`particles::walk`), O(faces crossed), no bucket search after release.
  - [x] RK4 advection; horizontal random walk √(2KΔt)ξ with constant K. Each particle has its own SplitMix64 stream (tracker seed, particle id), so a run does not depend on thread count or particle order. Particles are stepped in parallel with `parallel`.
  - [x] Boundaries: walls reflect specularly (`with_reflecting` picks the tags; untagged faces reflect), other faces let the particle out at the crossing (`ParticleStatus::Exited(tag)`), periodic faces carry it across. Particles in water shallower than `with_stranding_depth` strand and float again when it deepens.
  - [x] Gates (`tests/particle_tracking_test.rs`):
    - Solid-body rotation on distorted quads (exact in Q1): one revolution returns to 7e-7 at 100 steps, rate 4.00.
    - A walk of 0.45 per step (several elements) with a flow into a corner never leaves the basin.
    - Well-mixed: a uniform distribution stays uniform in a closed basin (χ² over 64 bins 72 and 57 for steps below and above the element size); in open water the variance is 2Kt to 5 %.
    - Open exit at the crossing point, periodic wrap, stranding and refloating, reproducibility per particle.
    - Online in a running `Simulation` (uniform oscillating flow): second order in the snapshot interval (rate 2.00).
  - [x] `local_time_stepping_farm particles=N [particle_seconds=60] [kh=0.1]` releases N particles in each cage and tracks them online. 1 h from rest with 500 per cage (4 threads): the tracking costs 0.08 s against 198 s for the solver. The clouds drift 42 m with the flood and spread 38 m RMS, which is the random walk's √(4Kt) (the tidal current at the farm is mm/s, so the site says little about advection; a farm-wake study needs a site with real currents, see F.1).
- [ ] Follow-ups:
  - Variable K (drift ∇K) and, for a depth-integrated tracer (∂(hc)/∂t = ∇·(hK∇c)), the drift K∇h/h (Dimou & Adams 1993). The constant-K walk is right for a surface or depth-uniform concentration only.
  - Curved (isoparametric) elements, once P1.3 has them: the walk assumes straight faces, and `inverse_bilinear` the bilinear map.
  - Particle output (positions per snapshot) for the viewer (F.3) and for connectivity statistics between farms.
  - A higher-order time interpolation between snapshots (cubic, as for the nesting fields in P1.5) when the output interval is long against the tidal period.
- [ ] Vertical random walk with the ∂K/∂z drift correction (Visser 1997, MEPS 158) once 3D exists, gated by Visser's well-mixed test under variable K.
- [ ] Behaviour hooks: sinking speed (feed, faeces), lice larvae depth preference and salinity avoidance (as in the IMR salmon-lice model; Sandvik et al. 2020, Aquac. Environ. Interact. 12). They need a vertical position, so they come with Stage B.

### F.3 3D visualisation (`viz/`, Bevy 0.20)
- [x] Crate `dg-viz` in `viz/` (its own workspace, so dg-rs builds and tests never compile Bevy; dg-rs without NetCDF, so no native HDF5). Runs the fjord-farm scenario of `examples/local_time_stepping_farm.rs` on a solver thread (`run_with_callback`), plays the snapshots back (rate, pause, seek; oldest dropped beyond a memory budget) as the water surface over the bed, each element as p² quads of its own nodes with the polynomial's exact normals, coloured by speed or η; walls along the mesh boundary; current arrows sampled by the element polynomial (`Probe2D` weights) on a domain grid and a fine farm grid; net cages (collar, textured net) riding η; orbit camera; `--screenshot` for headless checks (2026-09-27).
- [ ] Salmon in the cages (pfish-bevy in `../blender-procedural-fish-addon/rust`, once the add-on has a salmon).
- [x] Particles (2026-09-28, `viz/src/particles.rs`): released continuously from the cages (`--particles N` per cage every `--release S`, random walk `--kh`), tracked on the solver thread between the callback snapshots, drawn as small octahedra riding η, coloured by age (grey when stranded); P toggles them, the HUD counts them. At the fjord farm the cloud is mostly random walk (currents < 1 cm/s).
  - [ ] Particle types with their own behaviour (lice larvae, sinking feed and faeces) once F.2 has behaviour hooks; trails; a density field instead of points for 10⁵+ particles.
- [x] Bathymetry through the water (2026-09-28): the water is 40 % opaque by default (`--water-alpha`, - and = step it, T still cycles translucent/opaque/hidden), the HUD has a bed-depth colour scale, and the bed carries depth contours (`viz/src/contours.rs`: marching triangles over the bed's own triangles, a round interval of about ten lines, every fifth bright, drawn as ribbons a few pixels wide that follow the camera distance; B toggles them).
- [x] Frøya in the viewer (2026-09-28): `--scenario froya` runs Frøya–Smøla–Hitra as `froya_real_data mesh=data/froya_coast.msh` does (the topobathy bed projected onto the coastline mesh, NorKyst-800 atlas tides from 2025-06-15, open faces beyond the atlas walled off, a 3 h ramp); F frames the Mausund gauge. ≈ 18× faster than real time at 10 solver threads, so play it back at `--rate 15`.
  - [ ] Land: the islands are holes with walls down to the base. Draw the elevation model's land (a raster surface, or a coarse land mesh) around the water.
  - [ ] Bevy's default font has no ø, æ, å: bundle a font for Norwegian names (the title is ASCII for now).
- [ ] A snapshot file (element-major f32 η, u, v plus the mesh and bed) so runs can be replayed without re-running, and other scenarios (~~Frøya coastline mesh~~ done, nested runs) selectable from the command line.
- [ ] 3D fields (Stage B): vertical sections and a surface layer from `Solution3D`.

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
- [ ] Remaining allocations: the limiters allocate the cell-average and vertex-bound `Vec`s every stage (P2.3); the 2D tracer's BR1 diffusion allocates its whole-mesh arrays per RHS and runs serially (`tracer_2d.rs`; port it onto the per-element `br1_gradient_element`/`br1_diffusion_element` as the SWE viscosity was, 2026-09-28); the legacy `ssp_rk3_swe_2d` stepper used by the examples still clones stages (P6: move the examples to `Simulation`); `DGSolution1D`/`DGSolution2D`/`TracerSolution2D`/`Solution3D` use the allocating default `clone_from`, and `Simulation3D`/mode splitting do not use the stage workspace (P4.1).
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
  - [ ] End subcells on physical boundaries (no neighbour element) stay first order. Take the outer slope neighbour from the boundary condition (mirror the interior node for walls). (Note 2026-09-28: a mirrored node makes h, η and the tangential velocity even across the wall, so the MC limiter gives them zero slope and only the normal velocity gains a slope; η stays first order, and with Coriolis ∂η/∂n = −f u_t/g ≠ 0 at a wall. One-sided slopes (node 0 to node 1, bounded because the subcell face lies between them) would be second order but break the balance argument at a dry end node, whose face depth must stay zero: use them only where the end node and its neighbour are wet.)
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
  - [x] `SpatiallyVaryingManning2D` did not implement `BottomFriction2D`: the rate needs the node position. Done 2026-09-28: it holds g·n² per node (`new` from n(x, y) at the nodes, `from_nodal` from values), `damping_rate` takes the node index, and the damping checks the field against the mesh. Its explicit source goes through `add_element` only (`evaluate` has no node index and panics).
    - [ ] A roughness map from data (sediment type or a calibrated n per region, e.g. lower n over deep rock, higher over tidal flats), projected onto the nodes like the bed (`Bathymetry2D::project`). Frøya runs n = 0.025 everywhere, and friction is one of the phase suspects at Mausund (P3.1).
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
- [x] Coastline-fitted Frøya mesh from the topobathy model (2026-09-27): `scripts/gmsh_coastline_mesh.py` (0 m contour, land and water narrower than 2× the coastal size removed, resampled coastline, Gmsh recombination + subdivision, `open`/`coast` groups, convexity check) and `froya_real_data mesh=` (see `docs/gmsh-meshes.md`). Frøya at a 200 m coast: 12,206 quads, 33 islands, 479 km of coastline; lake at rest 1.9e-12 m/s; the tide runs with 0 clips.
  - [x] Cost, part 1 (2026-09-28): the step was set by nearly degenerate quads, not small ones. Gmsh's default `Mesh.RecombineMinimumQuality` (0.01) pairs triangles along straight coastline into quads with two coastline edges, and after subdivision they have corners up to 178° (110 corners over 160°, 48 of them at coastline nodes that belong to one quad only). There the bilinear Jacobian nearly vanishes: dt 0.0096 s in 50–110 m of water, from ≈ 80 m quads. `scripts/gmsh_coastline_mesh.py` now recombines at quality ≥ 0.3 (worse pairs stay triangles, which subdivide into good quads) and runs 5 Laplace passes: 12,490 quads (+2 %), largest corner 153°, smallest dt 0.112 s (11.6×). Measured over 30 model minutes at 24 threads (`lts=8`): 108 → 59 s (1.84×); the new mesh needs 5 levels (coarse step 3.5 s, 115 ms) instead of 8 (2.3 s, 139 ms). Global steps on the new mesh: 220 s per model hour, against 118 s with LTS and ≈ 30 s for the 500 m grid.
  - [ ] Cost, what is left (≈ 4× the 500 m grid per model hour). Estimated per element from the mesh (`CFL/(2N+1)·4/(c(|∇r|+|∇s|))` at the corners, DEM depths), the ideal multirate work Σ 1/Δt_k is 1.6e4 element-steps per model second, about the 500 m grid's 1.5e4 with global steps. Power-of-two levels make it 2.3e4, and the stepper does 4.4e4 (2.46× less than global at the finest step): the three-hop stage dependency costs ≈ 1.9× here (P2.5 follow-ups, level coherence). Of the ideal work, quads of subdivided triangles (26 % of the elements, median 98 m in 9 m of water) take 38 %, coast-sized elements in water over 50 m deep 19 %. So depth-based sizing near steep coasts (h ∝ √(gH)) would save at most ≈ 15 %; the bigger levers are the stepper's level coherence and a quad mesher that does not subdivide (Gmsh's quasi-structured quads took over 20 min here).
  - [x] Validate against the gauge and NorKyst (`scripts/norkyst_current_comparison.py`) once a 3-day run is affordable; at 200 m the skerries narrower than 400 m are shoals in the bed, not islands.
    - Done 2026-09-28 (`froya_real_data mesh=data/froya_coast.msh lts=8 hours=72`, from 2025-06-15, 24 h spin-up; 2.8 h wall at 12 threads, partly contended; 0 clips). Mausund, sampled 198 m from the gauge in 5.3 m of water, hours 24–72: centred RMSE 2.3 cm against the gauge's tidal prediction (1 km grid 3.1–4.4 cm, 500 m 2.9–3.5 cm in earlier runs), 4.7 cm against the observations (500 m 5.8 cm, 1 km 6.3–6.6 cm), bias −2.7 cm, correlation 0.9994. M2 1.013×, −12.0° from a 2-day fit (which lumps N2 and S2 into M2; the 500 m grid gave −9.4° the same way). The best gauge skill so far, at ≈ 2 min per model hour on an idle machine.
    - Currents: at NorKyst's point near Mausund (sampled 37 m away, 46 m deep) the M2 current is 0.021 m/s against NorKyst's 0.246 m/s: the island lee found at 500 m (0.010 m/s) persists with a fitted coastline. Deciding between the model's lee and NorKyst's open flow still needs observed currents (P3.1).
    - [x] Pinnacles: the run's largest current, 2.2–2.8 m/s through all three days at (10.6, −13.8) km, is one P2 node on a 200 m wide, 6 m shoal whose nodes 200–300 m away are 15–45 m deep. hu stays smooth across the element (≈ 16 against 22 m²/s), so u = hu/h spikes where h dips; real flow over so narrow a shoal would speed up far less. Stable and local, but it sets the max |u| statistics. Smooth the bed to the mesh the way ROMS does (a bound on the slope factor rx0 = |ΔB|/(h₁ + h₂) between neighbouring nodes, e.g. 0.2, by local volume-preserving diffusion), or report the speed statistics on elements' mean velocities.
      - Done 2026-09-29: `Bathymetry2D::max_rx0` (diagnostic) and `smooth_rx0` (Sikirić et al. 2009 "PlusMinus": each pair over the bound moves the volume that brings it to r_max, mass-weighted, swept until the largest r_x0 is within 1e-6; volume exact, continuous, nodes shallower than `min_depth` never change, so the coastline stays). `froya_real_data rx0=0.3 [rx0_min_depth=3]` (off by default) prints the largest r_x0 either way, and the step table has the largest speed in open water (still-water depth ≥ 3 m).
      - Frøya coastline mesh: r_x0 up to 0.94 (5 m beside 162 m at the steep coasts); r_max 0.2/0.3/0.5 changes 23.7k/16.3k/6.2k of ≈ 50k global nodes, by up to 92/81/54 m. That is a lot of bed for a 2D model that is well-balanced for any bed; the full-mesh effect on the pinnacle and at the gauge is not measured yet (a 3-day run).
      - Mausund sub-domain (hours 24–72, `tide_transport=3`): rx0=0.3 changes 4,010 nodes (up to 64 m) and improves everything measured: centred RMSE against the prediction 4.2 → 3.1 cm, against the observations 6.5 → 5.6 cm; the M2 current against NorKyst's (`scripts/norkyst_current_comparison.py`) quartiles 0.64–1.83 → 0.68–1.69, within 3 km of Mausund 1.32 → 1.11, transport where NorKyst is 30–60 m deep 0.55 → 0.71. Max open-water speed 2.15 → 2.01 m/s: the fastest open water is now 4–5 m deep near the shore, not shoals.
      - Full coastline mesh, 3 days from 2025-06-15 with `rx0=0.3` (2026-09-29; 2.25 h wall at 24 threads, idle, 0 clips): neutral at the gauge. Centred RMSE against the prediction 2.3 cm (baseline 2.3), against the observations 4.5 cm (4.7), bias −2.9 cm (−2.7), M2 1.013×, −12.4° (1.013×, −12.0°). The station node moved from 5.3 to 14.9 m deep (the smoothing deepened the shallows it sat in). The pinnacle at (10.6, −13.8) km is gone: that node is now 23 m deep and carries 1.22–1.25 m/s, against 2.2–2.8 m/s on the 6 m shoal.
      - The largest current is now 2.5 m/s at (9.6, −15.6) km, in 2.3–3.0 m of water: a shoal shallower than `rx0_min_depth`, which the smoothing leaves alone by design. A lower `min_depth` (1 m) would reach it, at the cost of more shore.
      - [ ] Decide the default. Gauge-neutral on the full mesh, better on the sub-domain, and it removes the pinnacle; against it, 16k of ≈ 50k nodes change by up to 81 m. Still missing: the map-wide NorKyst current comparison on the full mesh, which needs a baseline run with hourly output (`output_minutes=60`, ≈ 2.3 h), then `scripts/norkyst_current_comparison.py output/<run> lat0=63.8 lon0=8.6`. Consider a looser bound (0.5) or a `min_depth` that spares steep coasts.
      - [ ] `raise_isolated_wet_nodes` walks coincident-node members and their grid neighbours itself; it could use the same `NodeGraph` adjacency as the rx0 functions (check the raise order stays deterministic). `max_rx0` rebuilds the graph on every call (setup-only, cheap).
- [ ] Triangles (or quad-dominant meshes), and remove the hardcoded 4 faces (`swe_2d.rs:386,908`).
- [x] Gmsh: real MSH 4.1 parsing, sparse node tags, no `.unwrap()` on input, error (not drop) on unsupported element types (2026-09-26). `read_gmsh_mesh`/`parse_gmsh_mesh` read MSH 4.1 ASCII and binary (both byte orders, parametric nodes) and MSH 2.2 ASCII into one validating builder: unused nodes dropped, clockwise quads reordered, degenerate/non-convex quads, non-manifold edges, overlapping neighbours, lines off the mesh and conflicting tags are errors with the Gmsh tags in the message. Boundary tags from physical group names (`coast` → wall, `open`, `tidal`, `river`, else `Custom(n)`), unnamed groups by number. Edge numbering is deterministic (was `HashMap` order). Gates: `tests/gmsh_mesh_test.rs` on Gmsh 4.15 output (`scripts/gmsh_fixtures.py`, a bay with a coastline and an island): the three formats give the same mesh, tags on the right curves, lake at rest over rough and shoreline beds ≤ 1.3e-12 (P1–P3), mass to 1e-13 over a run, a raised sea fills the bay through the tagged open boundaries only.
  - [x] One `Mesh2D::from_quads(vertices, elements, boundary tags)` constructor: the edge building is written out again in `uniform_rectangle*`, `channel_periodic_x`, `uniform_periodic`, `retain_elements` and the Gmsh builder (`build_mesh` is the general one). Done 2026-09-28: `Mesh2D::from_quads(vertices, elements, periodic face pairs, tag per boundary face) -> Result<_, QuadMeshError>` (non-manifold edges, overlapping or clockwise neighbours, bad periodic pairs), used by all of them. The structured meshes' connectivity is unchanged face for face (checked against a dump of the old builders); only the wrap-around edges of fully periodic meshes now carry their left face's vertices. `retain_elements` on a periodic mesh keeps periodic edges between kept elements, where it could panic before.
  - [x] `Mesh2D::edge_orientation` is never read: the kernels rely on neighbours listing their shared face nodes in reverse (consistent counter-clockwise order, now checked by the Gmsh builder). Drop it or use it in a `Mesh2D::validate`. Done 2026-09-28: dropped; `from_quads` checks the opposite traversal for every mesh.
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
- [x] Transport consistency at the boundary: the atlas velocity is NorKyst's depth mean over NorKyst's depth; scale by `h_parent/h_child` (depth is in the atlas) where the beds differ. Nesting does this since P1.5 (`OceanModelState`); `BoundaryTides` does not yet (it has `source_depths`), and `blend_bathymetry` needs an `OceanModelState` for the parent depth.
  - Done 2026-09-28: `BoundaryTides::with_transport_scaling(max_ratio)`, `froya_real_data tide_transport=3`; off by default. At Frøya it hardly matters (1 km, hours 24–72 from 2025-06-15): centred RMSE at Mausund 4.0 → 3.8 cm against the tidal prediction and 6.3 → 6.2 cm against the observations, M2 1.001× → 1.006×, the M2 current at NorKyst's point unchanged (0.071 m/s). The characteristic boundary takes the velocity only through the incoming invariant, and the open boundary lies mostly in deep water where the beds agree. Revisit for boundaries across shallow banks.
    - Found 2026-09-29: it matters a lot for the Mausund sub-domain, whose sides cross skerries where the child's bed is ≈ 0.6 of NorKyst's: centred RMSE at Mausund 8.1 → 4.2 cm against the prediction, M2 0.96× → 1.01× the gauge's. Consider making it the default for atlas tides (and check the full mesh with it on).
  - [ ] Bed blending towards the atlas depths near the open boundary (as `OceanModelState::blend_bathymetry`), so that the ratio is 1 at the boundary itself.
- [ ] Velocity from z-levels stops at 300 m (NorKyst has no `ubar`/`vbar` on THREDDS); deeper columns extend the 300 m value.

### P1.5 Nesting done properly (NorKyst/ROMS parent) — DONE 2026-09-27
- [x] Conserve parent transport: ū_child = ū_parent·D_parent/D_child (total depths, limited to a factor 3 either way) from the parent `h`; `OceanModelState::blend_bathymetry` blends the child bed to the parent's across the relaxation band, so the ratio is 1 at the boundary. Gate: a 10 m child under a 20 m parent carries the parent's 10 m²/s (u = 1.000 m/s; 0.500 without scaling).
- [x] Rotation: grid-relative `ubar`/`u` are averaged from u-/v-points to ρ-points and rotated to east/north by `angle` (or by the grid angle from the coordinates); east/north is rotated into the mesh axes per node with the projection's convergence (`io::east_axis`). The inverse bilinear map on curvilinear grids is solved by Newton in each candidate cell (`io::GeoGrid`), so the interpolation weights are exact in index space.
- [x] z-level files: the vertical axis is classified by name (`s_rho`: ROMS s-levels, averaged with `s_level_weights` from `s_w`, `Cs_w`, `hc`, `Vtransform`, `h`) or by its coordinate's `positive` attribute (NorKyst `depth`, positive down, level 0 the surface: trapezoid rule to the bed, `depth_average_z`). Surface tracers take the right level for either.
- [x] A flow-relaxation band over a width inside the open boundary (`OceanModelState::relaxation`, `NestingRelaxation2D`): γ(d) = profile(1 − d/w)/τ towards (h, hu, hv) of the parent, distances to the tagged open faces (any shape, not only rectangles), skipped where child or parent is dry.
- [x] Descending coordinates bracket the right interval (`GeoGrid`), and `get_state` returns `None`/`Option` fields instead of silent zeros; missing corners are dropped from the bilinear weights.
- [x] Parent fields are `FieldSeries` strided `[time][point]`; stencils (spatial, rotation, parent depth) are precomputed per boundary and band node at setup; no RwLock/HashMap at run time. Time interpolation is cubic by default: hourly output of M2 is 3 % off linearly, 0.15 % cubically (gate: nested progressive wave, 1.8 % vs 7.0 % of the amplitude with output every T/8).
- [x] Data: `examples/norkyst_nesting_subset.rs` cuts a NorKyst-800 window out of THREDDS (ζ, h, ū/v̄ from the z-levels, optionally the forcing wind) into a small NetCDF; `froya_real_data norkyst=<file>` nests in it (`band_km`, `band_minutes`, `blend`, `ib`).
- [x] Frøya check (1 km, 72 h from 2025-06-15, Mausund hours 24–72): the nested child follows NorKyst (4.1 cm centred at the gauge), costs the same as atlas tides (1.8–2.0 ms/step at 24 threads, band and weather included), but is worse against the gauge: 10.7 cm centred against 5.6 cm, because NorKyst's own tide there is 8.3 cm off and lags 10–20 min. The band changes Mausund by 4 mm.
- [x] **Tides + residual** (2026-09-27). NorKyst's N2 is 5× too small at Mausund (P1.4 re-infers it in the atlas from a gauge), and nesting took NorKyst's tides as they are. The raw atlas is NorKyst's own harmonic fit, so parent + (corrected atlas − raw atlas) keeps NorKyst's residual (coastal current, surge) with the corrected tides: `TidalAtlas::difference`, `TidalAtlas::tides_at` (the atlas at arbitrary points), `OceanModelState::with_tidal_correction` (added to the precomputed per-slot series, band nodes included); `froya_real_data nest_tides=corrected` (default) or `raw`.
  - Frøya, 1 km, hours 24–72 from 2025-06-15, Mausund sampled 1.3 km from the gauge (P3.1): centred RMSE against the gauge's prediction / the observations 10.6 / 12.1 cm nested raw → 5.4 / 7.6 cm nested corrected, against 4.4 / 6.6 cm with atlas tides. Cost unchanged (1.9 ms/step).
  - [ ] What is left against atlas tides is NorKyst's own M2 (0.3–1 cm) and its phase lag of 10–20 min at the boundary; correcting M2/S2 too needs a reference other than NorKyst (gauges along the boundary, or TPXO/FES) in the corrected atlas.
- [ ] Parent ζ datum and pressure: NorKyst's ζ is 0.28 m below MSL along the Frøya boundary (30-day mean, atlas `Z0`; −0.26 m at Mausund, where it made the nested run 0.26 m low). `froya_real_data` shifts by −mean Z0 (`NestingOptions::with_reference_level`); a library default needs a datum source (e.g. a long parent mean, or NN2000 → MSL). Check whether NorKyst v3 is forced with sea-level pressure (then its ζ carries the inverse barometer and `ib=1` would count it twice).
- [ ] The relaxation band is explicit: τ must stay well above the time step (30 min against 0.25–10 s). A point-implicit form in the relaxation hook would allow FRS-like τ ≈ Δt.
- [ ] `OceanModelReader` loads the whole file (a 3-day, 100 × 100 subset is 22 MB); months of forcing, or a direct OPeNDAP read of the aggregation, need snapshots read lazily around the current time.
- [ ] Nesting needs a projected mesh with an open-boundary tag; a nesting of 3D fields (T/S, baroclinic velocity) belongs to Stage B (P4.2).

### P1.6 Real-domain inputs
- [x] GeoTIFF georeferencing: the tags were looked up as `Tag::Unknown(n)`, which never matches the `tiff` crate's named variants, so every file silently used the bbox hint. `ModelTransformation` was not supported at all. Frøya's bathymetry was misplaced by up to ~20 km. Fixed, with a CRS check (projected/rotated rasters are rejected), `PixelIsPoint`, GDAL no-data and pixel-centre sampling.
- [x] GeoTIFF: L2 projection onto the element nodes instead of point sampling (2026-09-26). The Frøya GeoTIFF has 0.002° pixels (≈ 100 × 230 m), so elements coarser than that alias the bathymetry.
  - `Bathymetry2D::project`: per-element L2 projection onto Q_N (composite Gauss–Legendre quadrature on sub-cells ≤ half the data resolution). It is limited to the element's data range by Zhang–Shu scaling towards the mean, then made continuous by mass-weighted averaging of coincident nodes (`make_continuous`). Exact for Q_N, volume-preserving, bounded, N + 1 for smooth data, and it does not alias (unit tests in `bathymetry_2d.rs`).
  - Continuity is essential. The element projections alone put −18 m and +1…+5 m on one coastal vertex, and that lone deep node jetted at 8–12 m/s.
  - At 1 km it is not better than sampling at Mausund: centred RMSE 3.6 cm against 3.3 cm, with different station nodes (15.1 m and 16.4 m deep). Re-compared 2026-09-27 with interpolated stations: projected is better at 1 km, both at the gauge (4.0 against 4.4 cm centred) and for currents (point samples close sounds between skerries, see P3.1), so the example projects by default (`bed=point` to sample).
  - [x] The projection shrinks a narrow inlet to one deep node among land nodes (Frøya projected at 1 km: B = −9.7 m among +0.5…+3.8 m, η stuck at +1.2 m). `Bathymetry2D::raise_isolated_wet_nodes` raises a wet node whose grid-line neighbours are all dry to the lowest of them (14–21 nodes at Frøya 1 km). It does not help against a cliff (a wet node beside +5 m land, e.g. (−7.9, 4.5) km on the GSHHS-mask bed, 2.16 m/s): only land heights fix those.
- [x] Land/nodata no longer become B = 0: `Bathymetry2D::from_geotiff` is removed. `io::BedRaster` holds the sea bed (dry/no-data pixels at 0) and a land mask rasterised from the coastline on a finer grid (land at `land_elevation`), or a full elevation model (`BedRaster::new`, for a merged DEM). `Bathymetry2D::from_raster`/`select_elements` build the water-only bed.
  - Baking `land_elevation` into the pixels instead (tried first) ramps the bed up to it across a pixel at every coast. At 1 km that closed sounds and cut M2 at Mausund to 0.61 of the gauge's (centred RMSE 15 cm). The coastline must decide land and water point by point, with the sea bed going to 0 at the shore.
  - [ ] The land mask costs ≈ 20 s of setup at Frøya (1.8 M point-in-polygon tests against GSHHS polygons, 4 × 4 cells per pixel): the spatial index below would remove most of it.
- [ ] `LandMask2D::from_coastline_and_bathymetry` counts water shallower than `min_depth` as land (the old Frøya run used 5 m, dropping every tidal flat) and uses nearest sampling. It is unused now; fix or delete it.
- [x] The Frøya GeoTIFF stores land as 0 and has no land elevations, so there are no intertidal flats: land is a wall at mean sea level (given `LAND_ELEVATION` = 5 m in the example). Real wetting/drying needs a merged DEM for the foreshore.
  - Done 2026-09-26 with Kartverket's national topobathy model: a free WCS (hoydedata.no), fetched by `scripts/kartverket_topobathy.sh` as an EPSG:4326 GeoTIFF and read by `BedRaster::elevation_model`.
    - The service has two levels. Square cells of 42 m and coarser get the 50 m model (land and sea interpolated together); 40 m and finer get the 1 m level, where unsurveyed sea is a flat 0. The script refuses cells under 50 m, and the example rejects a raster that is more than 5 % exact zeros.
    - The 50 m model also has holes marked by an exact 0: a 30 × 30 pixel patch in 320 m of water in the north-west corner, which piled water up to +1.2 m. `BedRaster::fill_holes` interpolates them harmonically.
    - Land above the highest water only makes cliffs in shoreline elements: +30 m beside a −7 m corner drew a film up to η = +12.7 m in the `WetDry` subcells. `BedRaster::clamp_land` caps it at `land_elevation`.
    - With the DEM, a narrow fjord arm through Hitra crosses the southern boundary, 6.4 km from the nearest atlas point. The example turns open faces beyond `ATLAS_COVERAGE` into walls.
    - Frøya, 1 km, 3 days (hours 24–72 at Mausund against the gauge's prediction): centred RMSE 4.1 cm (old builder) → 3.2 cm (DEM, point) / 3.1 cm (DEM, projected). Max |u| at 24/48/72 h: 2.08/1.40/2.16 → 1.43/0.89/1.33 m/s. The cliff node at (−7.9, 4.5) km is gone.
  - [ ] Vertical datum of the sea part. Land heights are NN2000, which is 8 cm above mean sea level at Mausund (tide API: chart datum 0, mean sea level +1.433 m, NN2000 +1.515 m). A seamless model should hold depths in NN2000 too, but the service does not document it. A comparison with the chart-datum `bathymetry50m` WCS failed: that layer is empty over most of the domain. If the sea part were in chart datum, depths would be 1.43 m too shallow at Mausund. Check against the chart-datum depth points (WFS `wfs.dybdedata`) in a surveyed shallow area, and add the 8 cm NN2000 → mean sea level offset if it matters.
  - [ ] Farm scale: the 1 m level (land lidar plus surveyed depth projects) over the 50 m model where it has data, telling the flat-0 fill apart from real 0 m land (e.g. by the surveyed-project coverage layer, "Dybdedata - dekning"). One request is at most 3840 × 2160 pixels, so tile it.
  - [ ] η statistics in the example count mm films on the foreshore as wet (h > 1 mm), so "η max" is often a film left by high tide. Use h > 1 cm for the range.
  - Measured cost (P1.4 Frøya run, before the DEM): a wet node whose element neighbours are land at +5 m (a 5–10 m step within one P2 element) carries an η discontinuity of 0.8–1.0 m against the adjacent element at the same vertex, and 1–2.8 m/s, through the whole tide. These nodes set the run's η maximum and fastest current.
- [ ] GSHHS inner rings (holes) and a spatial index for point-in-polygon (`coastline.rs:81-95`).
- [ ] f(latitude) and β helpers. Fix `norwegian_coast_beta()` (β = 1.6e-11 is the 45°N value; 60°N ≈ 1.14e-11). UTM33 / Lambert for coast-scale domains.
- [x] Atmospheric forcing reader for MET Nordic / MEPS / AROME-Arctic (Lambert grid, 2D lat/lon, grid-relative winds, CF time) (2026-09-27): `io::AtmosphereReader::from_file(s)` reads `x_wind_10m`/`y_wind_10m` (rotated with the local grid angle), `u10`/`v10`, NorKyst's `Uwind_eastward`, or MET Nordic's `wind_speed_10m` + `wind_direction_10m`, and sea-level pressure in Pa or hPa; singleton dimensions dropped, consecutive forecast files joined (later wins). `source::GriddedAtmosphere2D` applies wind stress (Large & Pond) and −(h/ρ)∇p with per-node stencils, the exact gradient of the bilinear interpolant, rotation into the mesh axes and a ramp; it is a `BoundaryLevel` for `InverseBarometer`. `examples/met_forcing_subset.rs` fetches MET Nordic analysis windows. Gates (`tests/nesting_test.rs`): inverse-barometer equilibrium held to 2e-10 m/s in a closed and an open basin (1.9e-3 m/s at the open boundaries without the level), wind set-up slope within 0.6 % of τ/(ρgH), on a mesh rotated 30° from east.
  - [x] Cost: each weather snapshot is regridded once onto the nodes (a shared cache of 4, looked up through a per-thread cache; a shared `RwLock` per element cost 4× the RHS at 24 threads). Frøya 1 km: +1–3 % per step at 24 threads, +20 % RHS single-threaded.
  - [ ] Memory: 128 B per mesh node for the stencils plus 20 B per node and cached snapshot; a farm mesh of 10⁶ nodes needs ≈ 210 MB. Share stencils per element, or drop them after regridding when the snapshots are few.
  - [ ] Wind stress from the wind relative to the surface current (τ ∝ |U₁₀ − u|(U₁₀ − u)) and a stability-dependent drag (COARE), if the wind-driven residual matters at farm sites.
- [ ] Rivers as volume sources (`RiverTracerSource` adds no volume; `Discharge2D` is weak). NVE / ROMS river-file reader.
- [ ] Fix the tidal-potential sign vs its docs (`source/swe_2d/tidal.rs:299-305`) and the diurnal Love factor (low priority, < 5 mm effect).

### P1.7 Gating tests (write first; most fail today)
- [x] Lake-at-rest with steep nodal B (e.g. 30→400 m), p = 1–4, serial and parallel: residual < 1e-10. Done 2026-09-28 (`tests/swe_2d_test.rs::test_lake_at_rest`: a slope from 30 to 400 m plus bumps, walls; `EntropyStable` and `WetDry` ≤ 1e-11 m²/s², serial = parallel bitwise; `Standard` with reconstruction 2.5–115 as the negative control).
- [x] Lake-at-rest with face-discontinuous B, and with wet/dry shorelines.
  - Face-discontinuous B: split forms, `test_lake_at_rest_any_nodal_bathymetry` (cell-averaged and rough beds, P1–P4).
  - Shorelines: `WetDry`, `test_wet_dry_lake_at_rest_with_shorelines` (RHS) and `tests/wet_dry_2d_test.rs::lake_at_rest_with_shoreline_is_exact` (100 s runs). `Standard` is bounded by `…_stays_near_rest`.
- [x] Mass conservation with discontinuous B and slope-correlated flow: split forms `test_mass_conservation`, `WetDry` with dry regions `test_wet_dry_mass_conservation_with_dry_regions`; `Standard` covered by the P0.13 tests.
- [x] Thacker parabolic bowl (dynamic wet/dry), planar SWASHES case: `thacker_planar_oscillation_converges_{standard,wet_dry}`. Both converge at ≈ 2nd order. Also the Ritter dry dam break, for both formulations.
- [x] Outgoing-pulse reflection < 1 %; delivered tidal amplitude: `tests/open_boundary_flather_test.rs` for all three fluxes and both split forms, plus oblique incidence and elevation-only forcing (P1.4).
- [ ] Nonlinear SWE convergence with bathymetry on curved meshes. Assert N+1: current thresholds 1.5/2.5 are below it (`tests/convergence_test.rs`).
- [x] Fix vacuous tests (2026-09-28):
  - `test_geostrophic_balance` passed with Coriolis removed: it put a linear η on a periodic mesh, whose 0.1 m seam swamped the 1e-3 Coriolis term. Now a periodic geostrophic jet (an exact nonlinear steady state): residual 3e-6 against the Coriolis term 6e-3, converging at 2.9 (P3); 6 h drift 1.1e-4 of the jet speed. Both checks have a no-Coriolis negative control (residual = the Coriolis term, drift 5e-2).
  - `tests/swe_2d_test.rs::test_lake_at_rest` used a flat bottom: now the steep-bed gate above.
  - The well-balanced tests use only linear B at p ≥ 2: they test the collocated `Standard` form, which is balanced only for such beds. Arbitrary nodal beds are covered for the split forms (`test_lake_at_rest_any_nodal_bathymetry`, the steep-bed gate), with `Standard` as the negative control.
  - "Multi-day stability" ran 316 s: renamed `test_long_run_mass_conservation`. A real one is `test_multi_day_tidal_run`: 3 days of M2 through an open boundary into a shoaling basin with Coriolis and implicit friction on `Simulation`; the last two cycles repeat to 5e-7 m (volume) and 1.4e-6 m (head η), and the head amplitude is the forcing's to 0.4 %. 13 s in debug.
  - `test_parallel_matches_serial` excluded bathymetry, boundaries and limiters: bathymetry and walls were already in (cases 2–3), limiters and wet/dry have their own (`test_parallel_limiters_match_serial`, `test_parallel_correction_matches_serial`); an open tidal boundary with all three formulations is added.
  - Also found: `test_standing_wave_period` timed the peaks of η at one node, 0.9 % off, under a 5 % tolerance, and skipped the check when it found fewer than two peaks. It now times the zero crossings of the cos(kx) mode amplitude: 3e-5, asserted to 1e-4.
  - [ ] `tests/swe_2d_test.rs` has three overlapping long-run conservation tests on the legacy stepper (`test_long_term_stability`, `test_long_time_conservation`, `test_long_run_mass_conservation`); merge them into one on `Simulation` when the examples and tests move off the legacy steppers (P6).

---

## Priority 2: Cheap Enough to Matter (`REVIEW.md` §5, §7 Phase B)

### P2.1 Time step
- [x] Per-element directional dt (P0.21) run near the stability limit. Evaluate SSPRK(4,3) or low-storage RK (+1.3–1.5×).
  - Done 2026-09-29: `SSPRK43` (Spiteri–Ruuth SSP-RK(4,3), SSP coefficient 2) and `MultirateSSPRK43`. In every wet/dry run the step is set by the Zhang–Shu positivity CFL (0.75/0.42/0.29/0.23 at N = 1–4), which is 2–4× below DGSEM's linear limit with SSP-RK3 (1D upwind advection eigenvalues: 3.2/2.2/1.8/1.5 in these units, about half that for diagonal 2D flow). An SSP method keeps a forward-Euler bound up to its SSP coefficient, so `Simulation` now caps the CFL at `ssp_coefficient × max_cfl`; SSP-RK(4,3) doubles the step for 4/3 of the RHS work and stays inside its own linear region at the doubled bound for N = 1–4 (negative real axis 5.15 against 2.51, so the BR1 viscous bound, expressed in SSP-RK3's real-axis limit, stays safe too).
  - The fused and multirate steppers run any two-register Shu–Osher scheme (`SspScheme`: `Rk3`, `Rk43`), replacing the hard-coded SSP-RK3 (`IntegratorInfo::ssp_scheme` replaces `is_one_level_multirate`; `Multirate<B>` with the aliases `MultirateSSPRK3`/`MultirateSSPRK43`, `Multirate::with_base` for a run-time `StandardIntegrator`). One level is the base scheme bit for bit; every LTS gate runs on both bases (multirate SSP-RK(4,3) converges to the global solution at 2.4–2.6).
  - Mausund sub-domain, 6 model hours at 24 threads, 0 clips in all: global 190.3 → 130.2 s (1.46×), LTS (`lts=8`) 101.2 → 64.1 s (1.58×). Frøya coastline mesh, `lts=8`, 2 model hours: 140.5 → 106.0 s (1.33×; the fourth stage hop widens the interface band, 3.06× → 2.86× less work than global steps), η at Mausund within 2.7e-5 m of SSP-RK3. Max η error at Mausund against SSP-RK3 at CFL 0.1: global 0.48 → 1.13 mm, LTS 2.50 → 2.51 mm (the level coupling dominates). `froya_real_data rk=43` is the default (`rk=3`, and `cfl=` to lower the CFL).
  - Where only linear stability limits the step (no wetting/drying) SSP-RK(4,3) gains ≈ 1.38× in stable CFL for 4/3 the stages: a wash.
  - [x] Those runs may simply be stepping too cautiously: measure the actual linear limit before raising any CFL. Done 2026-09-29, with the positivity bound relaxed per element (`PositivityBound`, see CHANGELOG).
    - Linear limits from the eigenvalues of the linearised DGSEM operator (`tests/linear_stability_test.rs`): `linear_cfl_swe_2d` = 1.6/1.04/0.73/0.56 (SSP-RK3) and 2.2/1.51/1.15/0.93 (SSP-RK(4,3)) for N = 1–4. Elongated elements at rest set SSP-RK3's (towards the 1D limit), squares SSP-RK(4,3)'s. Distorted quads, a bed and a mean flow raise it: the per-node metric of the time step is conservative there. The old estimate (1.1 at P2 for diagonal flow) was right for SSP-RK3 at P2 and too high above it.
    - So `cfl = 1` (the examples' and many tests' value) is linearly unstable for SSP-RK3 at N ≥ 2 and SSP-RK(4,3) at N = 4. The positivity cap had kept every wet/dry run inside. `SimulationConfig::cfl`'s default of 0.5 is inside for N ≤ 4 with either scheme.
    - Positivity: at ρ times the Zhang–Shu bound the mean is at least M − ρM_∂ (the water off and on the boundary nodes), so elements with every node wet step at ρ = 0.9·min(M/M_∂, W/W_∂), up to 0.9 of the linear limit. Mausund sub-domain: 1.14× fewer global steps and 1.20× fewer LTS coarse steps (1.25× wall), η at Mausund unchanged against a CFL-0.1 reference, 0 clips.
    - [x] The global step (and the finest LTS level) is still set by elements with a node at or below h_dry, which keep the plain bound. Find which and whether they can be relaxed safely. Done 2026-09-29 (see CHANGELOG).
      - On the Mausund mesh the binding elements are steep shorelines: a dry corner node beside 25–90 m of water (84–120 m elements). Their own deep nodes set the rate; the neighbours' face nodes add nothing. Relaxing them with the same ρ would have raised the global step 1.40× (0.42 → 0.59 s), close to the linear-only bound (0.61 s).
      - They cannot use the DGSEM linear cap: a dry node puts them on the `WetDry` subcells, whose stability limit (`linear_cfl_subcells_swe_2d`, measured for the first time) is 0.6–0.85 of it. At P2 under SSP-RK(4,3) it is 1.1, below the relaxed 1.35.
      - Now capped at 0.9 of that limit: 1.19× fewer global steps and 1.16× fewer LTS coarse steps on Mausund, 1.23× on the Frøya coastline mesh (LTS), 0 clips.
      - [x] A fully wet element that gets a node below h_dry during a step runs the subcells at the DGSEM cap for the rest of it (1.35 against 1.1 at P2/SSP-RK(4,3)). Done 2026-09-30: elements with a node at or below `SUBCELL_CAP_DEPTH_FACTOR` · h_dry (100 × 1 mm = 10 cm) take the subcell cap. On Mausund the global step does not change for bands up to 10 m; the ideal LTS work rises by 0.1 % (10 cm), 1.3 % (1 m) and 7 % (10 m).
      - [x] η at the Mausund station moved by up to 2.5 mm (against a CFL-0.1 reference) during one drying event near the station, against 1.3 mm before; LTS went the other way (3.0 → 2.2 mm). Point samples beside a drying element are this sensitive to the step. Watch the gauge skill in the next 3-day run. Done 2026-09-30: the 3-day Mausund run (`lts=8`, hours 24–72) has the same skill as before, to the printed digits (centred RMSE 4.2 cm against the prediction and 6.5 cm against the observations, M2 1.013×, −8.7°). η differs by 0.5 mm rms (4.1 mm at most), and the run takes 560 s against 689 s.
      - [x] What sets the step now (Mausund, P2, SSP-RK(4,3)): the same 84 m cliff element at the subcell cap (0.50 s), then fully wet elements at ρ ≈ 1.6 (0.59 s), against 0.61 s if only the linear limit applied. Analysed 2026-09-30:
        - The directional ρ below gains at most ≈ 2 % on those wet elements (their cap, 1.36, is just above 1.33).
        - Per-subcell rates (each subcell's own width instead of the narrowest) do not help cliffs: their deep nodes lie on the offshore face, in the narrowest subcells.
        - The remaining ≈ 1.2× is in the mesh: an 84 m element in 67 m of water. See depth-based sizing near steep coasts (P1.3, "Cost, what is left").
    - [ ] ρ uses all boundary nodes (M_∂). The Zhang–Shu split into r- and s-lines charges each direction's end nodes only, weighted by that direction's share of the rate: θ_r M_r-ends + θ_s M_s-ends ≤ max of the two, giving 3 instead of 1.8 at P2 (6 against 3.3 at P3) for uniform depth. Only worth it where the linear cap does not bind first (SSP-RK3 at P2: 0.42·1.62 = 0.68 against 0.94).
    - [x] Re-measure the full Frøya coastline mesh cost (the 15-day run estimate) on mains power with the relaxed bound and the new default CFL. Done 2026-09-29, with the shoreline relaxation: `lts=8`, 2 model hours at 24 threads, 76.3 s (93.8 s without it), so the 15-day run takes ≈ 4 h.
    - [ ] The linear limits (DGSEM and subcells) are measured for N ≤ 4 on straight-sided quadrilaterals. Curved (isoparametric) elements (P1.3) and N ≥ 5 need their own entries; `PositivityBound` does not relax where the table has none.
  - [ ] More stages: SSP-RK(9,3) (C = 6, efficiency 2/3) or SSP-RK(10,4) (C = 6, 0.6) would raise the positivity CFL above the linear limit, so the linear limit would bind (≈ 1.55 at P2 with SSP-RK(4,3) already, against the capped 0.83); their stages also combine earlier stage values, which the two-register `SspScheme` does not express, and each stage widens the LTS interface band by a stencil width. On the tiny runup gate (20 columns, 8 levels) the fourth hop already costs more than the doubled step saves (49k against 42k element evaluations); on the Mausund mesh it does not.
- [ ] The other dt helpers still pair the global minimum √detJ size with the global maximum speed (note 2026-09-28: for advection the SWE form's constant does not carry over: with no isotropic wave speed, `4/(λ_r + λ_s)` doubles the step of grid-aligned flow against the classic `|a|Δt/Δx = CFL/(2N+1)`, beyond the DG–RK3 limit at P3 (≈ 0.91/(2N+1)); use `2/(λ_r + λ_s)` and check the stability limits): `compute_dt_advection_2d`, `compute_dt_tracer_2d`, `compute_dt_viscosity` (takes a global size, and its (2N+1)² underestimates the BR1 spectral radius 2–5× at N = 2–4; `SWEPhysics2D` uses the measured per-element `element_dt_viscous_swe_2d` since 2026-09-28), `Mesh`-trait `compute_dt`, `time/ssp_rk3.rs:110`, `compute_dt_swe` (1D). Port them to the per-node metric form of `compute_dt_swe_2d` (one shared helper over ∇r, ∇s).

### P2.2 Kernels
- Note 2026-09-29: the two items below concern the collocated `Standard` kernel (`swe_2d.rs`) only. The production `WetDry`/split-form kernel already works line by line (`line_volume`, O(N) per node) and applies the GLL LIFT as its single diagonal entry, so they pay only if `Standard` stays in use (it is slated for retirement, P1.2).
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
- [ ] Build flags (2026-09-28, expected 10–20 %, unmeasured): `-C target-cpu=native` for the SIMD paths, and `mimalloc` where the allocating paths still run (the viscous RHS, the Kuzmin limiters). `viz/` builds dg-rs without it. Note for any crate that drives the solver: `Simulation<P, I>` and the physics builder are generic, so the hot loops are compiled in the *calling* crate at its opt-level. A `[profile.dev.package."*"] opt-level = 3` override does not reach them (a scratch driver at opt-level 0 ran several times slower).

### P2.3 Fused in-place stepper
- [x] Global SSP-RK3 as one fused pass per stage (measured 2026-09-29, Mausund sub-domain, 2,799 P2 elements, 24 threads): the stage combination (up to four whole-field passes), the implicit damping and the wet/dry correction are separate fork/joins, and the last two do not speed up at all at this size (post_process 0.117 ms on 1 thread, 0.126 ms on 24; damping 0.38 → 0.17 ms): 44 % of a parallel step. `MultirateSSPRK3::new(0)` already runs SSP-RK3 through the fused per-element `stage_where` and is ≈ 10 % faster per step (18.7–21.1 s against 22.3 s for 30 model minutes). Route `Simulation` + `SSPRK3` through it when the physics module supports local time stepping (check the bit-for-bit equality with global SSP-RK3 and the dt, which now matches).
  - Done 2026-09-29: `Simulation` runs `SSPRK3` (`IntegratorInfo::ssp_scheme`, first `is_one_level_multirate`) as one-level `MultirateStepper` steps (`assign_one_level`, set once; the step is the global `compute_dt`) when the physics module supports local time stepping. Bit for bit the whole-state stages (`one_level_is_ssp_rk3_bit_for_bit` now also checks the default path; `with_fused_stages(false)` keeps the old one). Mausund sub-domain, 30 model minutes of global steps at 24 threads: 21.3/24.6 → 16.0/19.4 s (1.27–1.33×), station output byte-identical.
  - [ ] The third stage still writes its value to a scratch field and then sums it into `acc`, and the finish pass copies `acc` back into the state and post-processes: with one level the stage could write the state directly (a pass fewer). Measure first; the remaining passes are the RHS-bound ones.
- [ ] Limiter, positivity and wet/dry in one parallel in-place pass. Today the "parallel" paths `to_vec` every field, allocate per element, and copy back serially (`limiters/swe_2d.rs:626-725`, `wetting_drying.rs:526-541`): 1.5–2.5× of wall time.
  - Limiters done (P0.19: in-place `par_chunks_exact_mut` over the SoA fields). The `wetting_drying.rs` parallel path is done too (P1.2: in place, plus a node-parallel implicit damping pass). Still open: the per-stage `Vec` of cell averages and vertex bounds (move to a workspace with P1.1). Then fuse the limiter, the wet/dry correction and the implicit damping into one pass.
  - 2026-09-26: the positivity limiter computes each element mean in its element kernel (no separate pass or `Vec`), and `post_process` skips a `Positivity(h ≤ h_dry)` limiter when wet/dry is on, since the wet/dry correction applies the same limiter to every element it would change (bitwise identical; `redundant_positivity_pass_is_skipped_exactly`). The Kuzmin paths still allocate averages and vertex bounds per stage.

### P2.4 Water-only work
- [x] Mesh water only: `Mesh2D::retain_elements` (faces towards dropped elements become coastline walls). Frøya keeps only elements with a water node.
- [ ] Limit only troubled cells.

### P2.5 Local time stepping or implicit free surface
- [x] One 250 m element in 1300 m water forces ~22× more global steps. Multirate/LTS SSP-RK or semi-implicit free surface (SLIM/Thetis practice): 2–20×.
  - Done 2026-09-27: local time stepping, `MultirateSSPRK3` (see CHANGELOG). A conservative, SSP multirate SSP-RK3 in Constantinescu & Sandu's construction: power-of-two levels, reassigned every coarse step. One level is SSP-RK3 bit for bit. Gates in `tests/local_time_stepping_test.rs`.
  - Farm mesh (`examples/local_time_stepping_farm.rs`, 12–455 m quads, P2): levels 0–6, 2.6× less RHS work (4.6× ideal), 2.4–2.8× less wall time.
  - Frøya at 1 km: little to gain, 1.53× at best from depth. Fine and coarse elements interleave, so ≈ 1.05× less RHS work and 1.1–1.3× less wall time.
- [ ] Follow-ups:
  - Choose the coarse step to minimise work. It is anchored at 2^L·Δt_min, so an element with Δt_k just under 2·Δt_min runs at Δt_min, wasting up to 2× (≈ 1.4× on average) on the bulk. A histogram of log₂(Δt_k/Δt_min) gives the cost of each anchor in O(bins). Modelled 2026-09-28 on the Frøya coastline mesh (the stepper's level logic in Python on the mesh's element steps; it reproduces the measured work ratios to 1–2 %): the best anchor between ½ and 1 × Δt_min saves ≤ 0.5 %. The broad Δt distribution averages the rounding out; it can pay only where most elements share one Δt.
  - The three-hop dependency (stage s reads neighbours' stage values s hops out) leaves 2.6× of the ideal 4.6× on the farm mesh, and almost nothing where levels interleave (Frøya). Levels could be made coherent (raise an element to the finest level around it when that costs less than its buffers); a mesh grading ≤ 1.2 per element also helps. Frøya coastline mesh, modelled (2026-09-28): Σ 1/Δt_k = 2.1e4 element-steps per model second; power-of-two levels with jumps ≤ 1 make it 3.0e4, and the spreading 4.6e4 (×1.51). Raising a face neighbour of a fine element costs nothing for it (its stage-1 rate is already the fine one) but pushes the fine rate one hop further out, so coherence alone does not remove the spreading; that needs a scheme whose coarse stages do not follow the fine ones (e.g. interpolated neighbour stages, at the price of exact conservation or the SSP property).
  - [x] Element time steps were too pessimistic (2026-09-29): `element_dt_swe_2d` priced every node of each face neighbour at the element's largest metric in both directions, so a coarse element beside deep water stepped as if that water sat at its worst node. It now takes the neighbours' face nodes (the states the interface flux and the Zhang–Shu bound of the element mean see) with the metric at the coincident node. Frøya coastline mesh, `lts=8`, 2 model hours at 24 threads: 235.7 → 156.7 s (1.50×), coarse step 3.5 → 9.7 s, 3.06× less RHS work than global steps (2.46×); Mausund η agrees to 1e-5 m. Mausund sub-domain, 6 h: 140.9 → 95.5 s (1.48×), η within 3 mm (LTS coupling). With one level the step now equals the global one (37 % more steps before). Regression test `element_dt_sees_the_neighbours_face_nodes_only`.
  - [x] Pass overhead (2026-09-27): the passes now run over element lists, with work proportional to the list. Elements are ordered by rate once per coarse step, so each active set is a prefix. The subset RHS runs only the faces of its elements (distinct edges by a per-edge stamp, not a sort), serially below 64 elements. `LocalTimeStepping::stage_where`/`finish_where` take the lists. Frøya coastline mesh (12k quads, 2^8 levels): 254 → 162–177 ms per coarse step; farm mesh unchanged (26.7 against 26.9–28.2 s for 15 min). What is left is the RHS work itself (≈ 0.4 µs per element evaluation with post-processing).
  - [ ] Oversubscription (measured 2026-09-28): with other processes taking ≈ 40 % of the CPU, 72 model seconds of the farm under LTS took 358 s at the default 24 threads against 3.9 s at `RAYON_NUM_THREADS=4` (90×). LTS runs many small fork/joins per coarse step, and a preempted worker stalls each one. Raise the serial cut-off of the subset passes (64 elements) by work rather than count, cap the pool for small meshes, or document running with a thread count below the free cores.
  - Element-local limiting only: the Kuzmin limiters (vertex bounds from neighbour means) panic under LTS. (BR1 horizontal viscosity runs under LTS since 2026-09-28, with a wider stencil.)
  - The interface coupling error is ≈ 50× SSP-RK3's time error at the same CFL for a gravity wave crossing four levels (1.5e-3 relative at CFL 0.8). It scales as (λΔt)², so tides are unaffected, but check eddies shed by cages as they leave the farm level.
  - Validate on Frøya: `froya_real_data lts=6` reproduces the global run at Mausund (M2 to 1e-4) over 6 h; repeat over the 15-day harmonic run.
  - [x] Horizontal viscosity under LTS (measured 2026-09-28). Farm runs need it (the drag wake otherwise never decays, see F.1), but BR1 couples each element to its neighbours' gradients, so `with_viscosity` was global-stepping only, and the viscous RHS itself cost 4.5× per step (serial, ≈ 10 arrays allocated per RHS).
    - Done 2026-09-28. The BR1 term runs element by element in parallel, with its gradients in a cached workspace (`solver/rhs/swe_2d_viscosity.rs`), and gives the old result bit for bit. The stepper follows the wider stencil (`RhsStencil::FacesAndCorners`: the face neighbours' gradients read the elements at their far corners). `SWEPhysics2D` bounds the step by a measured BR1 spectral radius (`element_dt_viscous_swe_2d`).
    - Farm mesh, ν = 1 m²/s, 15 model minutes at 24 threads: global 474 s with the old term → 130 s now → 64 s with LTS (7.4× in all; 2.44× less RHS work than global; 31 s inviscid). Local and global agree to 5e-8 m/s at the farm.
    - [ ] What is left of the viscous cost (≈ 2× the inviscid RHS). Compute u and v gradients in one pass: today each component loads the velocities again and evaluates the boundary condition again, and at open boundaries that is a tidal evaluation per node. For the subset RHS, the gradients of an element's neighbours are recomputed in every pass that lists a neighbour of theirs; a per-stage gradient cache, filled at the rate of the gradient's own inputs, would avoid it.
    - [ ] The wider stencil makes the coarse side pay the finer rate over 3 stencil widths (2.60× → 2.44× on the farm). Where ν is small (farm: 1 m²/s on 20 m elements), the viscous flux could be lagged across level interfaces (frozen at the coarse substep), at the price of exact momentum conservation there.
    - [ ] Smagorinsky ν is not in the time-step bound: it follows the strain. Bound it from the previous step's maximum ν.
    - [ ] Viscosity on the 3D path (P4) and a background ν plus Smagorinsky combination (`ViscosityModel` has one or the other).
- [ ] Semi-implicit free surface: the alternative for 3D mode splitting (P4), where the barotropic step is the stiff part, and (2026-09-28) the main cost lever for farm-scale Stage A too. On the farm mesh the global step is ≈ 0.024 s (124k steps per 50 model minutes), set by the gravity wave (√(gh) ≈ 30 m/s at 90 m) on the 20 m farm elements, while the currents there are mm/s–cm/s. A free surface implicit in the gravity-wave terms (θ-method as in Casulli's TRIM/UnTRIM; SLIM and Thetis for DG) steps at the advective CFL, up to 10–100× fewer steps where the flow is slow; LTS gets 2.4–2.8× on the same mesh. Cost: a global linear solve per step (SPD for the θ-scheme) and care with wetting/drying.

### P2.6 Benchmarks (was P1.6)
- Measured 2026-09-26: Frøya, 9,653 P2 elements (87k nodes), dt ≈ 0.65 s: 8.2 ms/step on 24 threads, so a 30-day validation run takes ≈ 9 h. Scaling stops at ≈ 8 threads (44.6 → 9 ms from 1 to 8–24 threads): the laptop's 4 Zen 5 + 8 Zen 5c cores and its power limit, not serial code. `froya_real_data profile=N` times each phase (RHS, post-processing, damping, dt) at 1 and all threads; single-thread RHS ≈ 85 % of the step.
- [ ] Full-RHS and full-step throughput vs mesh size at realistic size (≥ 100k elements, bathymetry, limiter, open BCs), not cache-resident micro-cases.
- [ ] DOFs/second; parallel scaling 1–16 cores; memory-bandwidth utilisation; allocation counting in CI.
- [ ] Reconcile PERFORMANCE.md vs PROFILE.md: the same 65k run is recorded as 70.5 s and 217 s, and some cited RHS variants no longer exist.
- [x] The criterion benches do not compile (`benches/source_term_bench.rs`, `time_stepping_bench.rs`: `usize` where `ElementIndex` is expected, tuple access on `[f64; 2]` vertices). CI never builds them (clippy runs `--lib --tests`); fix them and add `--benches` to the CI check. Done 2026-09-28: `limiter_bench` (limiters take `&GeometricFactors2D`), `source_term_bench` (index newtypes, array vertices) and `time_stepping_bench`, whose step benches now run the production path (`Simulation` with `SSPRK3` and `SWEPhysics2D`) instead of the legacy `ssp_rk3_swe_2d_step_limited`; CI clippy runs `--benches`. All 119 benches pass `cargo test --benches`.
  - Found with them: `Simulation` took an extra sliver step when round-off left t just short of a target (10 steps of 0.005 end at 0.049999999999999996; the 11th step was 7e-18 s, a full three-stage step). Fixed (`LANDING_SLACK`, also in `Simulation3D`); regression test `round_off_leaves_no_sliver_step`.
  - [ ] `Simulation3D` still fires callbacks at the first step past each interval (drifting by up to a step), as the 2D runner did before its callbacks landed exactly; port the 2D logic with P4.1.

### P2.7 GPU and distributed memory (decision)
- [ ] Either fix Burn (P0.11) or replace it with custom fused f64 kernels (CubeCL/cudarc) over CSR faces. Burn with fusion disabled (`Cargo.toml:38`) moves ~13× more bytes per node-stage than fused kernels.
- [ ] MPI/domain decomposition plan (METIS partitioning, halo exchange of face traces).
- Sizing (2026-09-28): neither more cores nor a GPU helps the current farm mesh (3,584 P2 elements, 32k nodes). LTS went from ≈ 10× real time on 4 threads to ≈ 17× on 10 (1.6× for 2.5× the threads): the per-step work is too small to spread. On a GPU, 124k steps × 3 stages × several kernels per 50 model minutes would be dominated by launch overhead. Consumer GPUs also run f64 at about 1/64 of f32 (e.g. the RTX 4060 Laptop here), and f32 is ruled out. GPU and many-core pay off from ≈ 10⁵ elements (real farm-scale coastal meshes), on data-centre GPUs with full-rate FP64 (A100/H100 class). Fewer steps (semi-implicit free surface, P2.5) and cheaper viscosity come first.

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
- [x] Compare with ADCP currents (`analysis/adcp.rs`). Add a tidal-ellipse fit and a complex-difference skill metric. Done 2026-09-27: `analysis::fit_tidal_ellipses` / `TidalEllipse` (complex difference √(|ΔŨ|² + |ΔṼ|²)); `ADCPValidationResult::compute` pairs by time; `froya_real_data currents=<ADCP files>` compares ellipses and time series, and compares the model's ellipses with NorKyst's at every station.
  - [ ] Real current data: farm-site current surveys (the site reports to Fiskeridirektoratet carry 1-month current measurements, usually at 5 m, 15 m and the net bottom, not depth means), and any IMR/NIVA moorings near Frøya. Depth-averaged comparisons need the depth mean of a profile; single-depth records wait for Stage B.
- [ ] Document skill scores (RMSE, bias, correlation) per station.
- [ ] Harmonic analysis: `fit_reference_constants` now has a Rayleigh check, inference (equilibrium or from a reference fit), `resolvable_constituents` and `predict`; `HarmonicAnalysis::fit` still has none of them. Port its callers (`StationValidationResult::compute_with_harmonics`, `validate_stations`) or give it the same guard; an `AnalysisError` type instead of asserts and `String` errors.
- [x] Station series are η only. Add depth-averaged velocity, which the ADCP comparison and the tidal-ellipse fit need. Sample at the gauge position by interpolating in the element, not at the nearest node ≥ 3 m deep (Mausund's node is 199 m away, and 7.6 m deep). Done 2026-09-27 (`Probe2D`): η, ū, v̄ at the station itself, or at the nearest point ≥ 3 m deep where the model is too shallow or dry there; velocities rotated to east/north.

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
  - Revisit the ROMS form once P4.3 lifts the 3D pressure/η jumps at faces. (P4.3 does since 2026-09-30: the 3D PGF with `rho_ref = 0` has the DG face coupling of η.)

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
- [x] Hz-weighted per-level face fluxes corrected so their vertical sum equals DU_avg2; η advanced by the divergence of the same flux. Done 2026-09-30 (`solver::rhs::transport_3d`, `LayerTransport`): nodal `H_z u` and the central face average, plus each layer's `Δσ_l` share of the difference to DU_avg2, nodally and on every face; Ω from the bed with each layer's share of the pass's ∂η/∂t.
- [x] Ω stored at w-points (not layer centres re-averaged to faces) — done in [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2).
- [x] ~~Assert Ω(0) = 0 instead of forcing it with the linear correction.~~ Measured and gated instead of asserted (2026-09-30): `LayerTransport::surface_residual` / `Hydrostatic3D::last_surface_residual` is 3e-15 against Ω of 7e-3 m/s under a tide over a slope. The linear correction stays, because `WetDry` elements with a dry node and positivity-limited elements only keep the pass's element balance, not the nodal identity, so there the residual is O(1) and must go somewhere.
  - [x] In those elements, correct per element so that constancy holds there too: done 2026-09-30 with 3D wet/dry (P4.5), element means per level (a subcell version is noted there).
- [ ] One 3D face-state helper for physical boundaries shared by the Ω, momentum and tracer kernels (P0.7/P0.22 were the same bug found twice). It should take the boundary tag, so 3D open boundaries (volume flux from the 2D OBC / nesting, tracer from `TracerBoundaryCondition3D` on inflow) are added in one place for all three.
  - Partly done 2026-09-30: Ω and the tracers now share one volume flux (`LayerTransport`), which at a physical boundary is the layer share of the 2D boundary flux (zero at walls), and the tracer consults `TracerBoundaryCondition3D` on inflow. Momentum still has its own wall reflection and no open-boundary condition; the 3D velocity profile at open boundaries (nesting of baroclinic u, v) is still to do.
- [x] Step Hz·C (inventory form), then divide by the new Hz. Done 2026-09-30: `ModeSplitIntegrator` carries the tracers as `H_z C` through the 3D stages. Gates `uniform_tracers_stay_uniform_under_a_tide_over_a_sloping_bed` (drift 9.3e-13 in one period; before 12.8) and `tracer_inventories_are_conserved_under_a_tide` (3.1e-15; before 1.1e-2). The seiche/wind/Ekman gates run with the T/S-dependent EOS again (the side effect below is gone).
  - Measured side effect (2026-09-26, fixed by the above): with a T/S-dependent EOS, the drift of T and S with η fed a spurious baroclinic PGF into the mode-split G. It damped a barotropic seiche by 0.6 % per period (P1), or amplified it by 0.4 % per period (P2), independent of amplitude and dt.
- [ ] Step Hz·u as well. The 3D momentum is still in velocity form, and its horizontal advection is `∇·(u u)` with a Rusanov speed, not `(1/H_z)∇·(Q u)` with the layer transports; only its vertical advection uses the new Ω. The depth mean is reset to ū every step, so this affects the shear only, but momentum is neither conservative nor consistent with continuity over a sloping bed. Moving it onto `LayerTransport` needs the mean-flow part of G (`R_adv+Cor(ū)`, a one-level run of the same operator) moved with it.
- [ ] Cost of the inventory form (noted 2026-09-30): each 3D stage copies the whole stage state to convert the tracers to concentrations (`Buffers::concentrations`), and `post_stage` converts back and forth. Cheap next to the 3D RHS today; with the P4.5 kernel work, have the RHS read `H_z C` and divide on the fly instead.
- [ ] The `w` output of `Hydrostatic3D::post_process` is still Ω of the 3D velocities alone (`compute_vertical_velocity`, closed with their own ∂η/∂t): there is no barotropic transport between steps. Output the last step's layer-transport Ω instead (e.g. keep the stage-3 Ω or recompute at tⁿ⁺¹ with DU_avg2), and then drop `compute_vertical_velocity`.

### P4.3 Pressure gradient and vertical grid
- [x] Balanced PGF. Lift pressure/η jumps at element faces. Done 2026-09-30 (`solver::rhs::baroclinic`): ∇p|_z directly, as the DG derivative of pairwise pressure differences at a common depth (Stelling & van Kester 1994 in a flux-differencing form), each column's ρ a monotone Hermite cubic in z integrated exactly (the Shchepetkin & McWilliams 2003 ingredient), one-sided pairs where a node lies below the other column's bed, and the pressure jump at every face lifted with a central flux.
  - Constant N² at rest over a 30→400 m slope: round-off (≤ 1.5e-14 m/s²) at P1–P4, uniform and stretched levels; the σ form gave 1.3e-5 – 2.5e-3.
  - The review's pycnocline fjord on 30 stretched levels (θs 5): 2.0e-7 / 1.5e-6 / 3.3e-6 m/s² at P1 500 m / P3 250 m / P3 100 m, against 1.6e-3 / 1.1e-4 / 2.2e-5 for the σ form and ≈ 5e-5 of real estuarine forcing.
  - Mode-split fjord at rest (10→150 m over ≈ 400 m, linear N²): 2.8e-11 m/s after 1 h; on `main` it blew up within 16 steps.
  - [ ] On 30 *uniform* levels the pycnocline case stays at 2–4e-4 m/s² (σ form 4e-4 – 1.6e-3): 13 m apart at 400 m, the deep columns do not sample a 3 m pycnocline. Resolve it in every column (stretched levels, as NorKyst), or subtract a finely tabulated horizontal-mean reference profile ρ̄(z) before integrating (Mellor et al. 1998) — an option to add if real stratification needs it.
  - [ ] Second order for smooth fields at P2 (measured 1.7–1.8 in the max norm, pre-asymptotic); first order in the one-sided pairs at the bottom of steep slopes. Higher-order vertical reconstruction (unlimited slopes where monotone) is the lever if needed.
  - [ ] Cost: n_nodes²/2 pair evaluations per level per element, each a binary search and a cubic integral in two columns (81/2 pairs at P2). Not profiled; parallelise with the rest of the 3D kernels (P4.5).
  - [ ] Curvilinear elements: the pairs use the affine metric (`Hydrostatic3D` asserts affine); the collocated chain rule `rx_i Dr + sx_i Ds` extends it.
- [ ] rx0/rx1 diagnostics and ROMS-style bathymetry smoothing. r_x0 is done in 2D (`Bathymetry2D::max_rx0`, volume-preserving `smooth_rx0`, 2026-09-29, P1.3). Still open: r_x1 (Haney number), which needs the σ levels, and an r_x1 bound for the 3D grid.
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
- [x] 3D wetting/drying (any dry node gave NaN). Done 2026-09-30:
  - Thin columns (D < `Hydrostatic3D::with_min_column_depth`, default 0.1 m, ROMS's Dcrit) carry no shear (u = ū, zero 3D momentum tendency, no vertical diffusion), get no stress through G, and exert no baroclinic pressure. The 3D momentum advection and the mean-flow part of G see them with zero velocity, so films at the 2D velocity cap (20 m/s) stay the 2D module's.
  - Tracers are element means per level for the step in elements where the 2D pass broke the nodal identity (`WetDry` subcells, positivity limiter): constant and conservative. Nodes (or marked elements) with essentially no water keep their last concentration.
  - Gates (beach from −4 to +2 m, 0.3 m slosh, wind, stratified, ≈ 4 periods): no NaN, 0 clips, T within its initial range; uniform T/S drift 5.3e-9 (round-off amplified by nearly dry elements); inventories 7.4e-12; a stratified lake with dry land at rest 2.0e-13 m/s. Before: NaN in the first step.
  - [ ] The shoreline elements' element means flatten the tracer along σ within them while the pass is inconsistent there (a band one element wide at moving shorelines). A subcell (GLL-cell) finite-volume tracer update, matching the 2D module's subcells, would keep the nodal structure; it needs the 2D subcell mass fluxes in DU_avg2.
  - [ ] Films: ū = hu/h in columns of a few mm reaches the 2D cap (20 m/s). Harmless to the 3D fields now, but it sets output statistics; consider reporting speeds only where D ≥ the minimum column depth (as `froya_real_data` does with its open-water column).
- [ ] EOS: delegate the physics trait to the fixed UNESCO EOS (P0.17) with a linear fast path; TEOS-10 later.
- [ ] `compute_dt`: internal-wave speed from stratification (not a hardcoded 2 m/s) and the vertical CFL. Relabel or convert Ω vs w in output.
- [ ] Parallelise the 3D kernels; no `Vec` allocation in inner loops (~186M malloc/free per 3D RHS at 50k × 30).
  - E.g. `apply_horizontal_surface_terms` (3D momentum advection) and `compute_vertical_velocity` allocate `Vec`s per face per level, and `compute_strong_divergence` two per call; move them to workspaces as `LayerTransport` and `TracerTransportScratch` do (2026-09-30), and parallelise those over elements.

### P4.6 3D validation (before any NorKyst 3D comparison)
- [x] Stratified lake-at-rest, linear N² ≤ 1e-12: the PGF to ≤ 1.5e-14 m/s² over a 30→400 m slope; the mode-split model to 2.8e-11 m/s after 1 h (P4.3 gates, 2026-09-30).
- [ ] Tanh pycnocline over a seamount (Beckmann & Haidvogel 1993): the spin-up of spurious currents over days. The PGF-level fjord pycnocline gate is done (P4.3).
- [x] Constant-T preservation under tide over a sloping bed; Hz·T inventory conservation (P4.2 gates, 2026-09-30).
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
| 3D | Scaffolding; tracer constancy (P4.2) and the balanced PGF (P4.3) fixed 2026-09-30; vertical advection ×D (P0.16) and the tracer wall leak (P0.22) fixed; mode splitting: one filtered barotropic pass (0.035 % seiche damping per period), second-order slow coupling, wind setup/Ekman to < 0.1 % (P4.1 PR 1–2; DU_avg2 is PR 3) | lock exchange, mode-1 internal wave, seamount spin-up |
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
