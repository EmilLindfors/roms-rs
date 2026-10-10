# TODO

Open items only, one line each. The context (measurements, decisions, what was tried, finished work) is in [`notes/`](notes/README.md): each section heading links its notes file, and searching that file for the item's ID finds the details. When an item is done, delete its line here and record the outcome in the notes file (and `CHANGELOG.md` if notable).

Direction (from the 2026-09-25 review, `notes/review-2026-09-25.md`): the salmon-farm application (currents, waves and particles around cages inside a NorKyst-driven coastal domain) on a 2D model that is correct, cheap and validated, and a 3D model rebuilt on the ROMS recipe. "Faster than ROMS" is to be proven by P3.2, not assumed.

## Next up

1. **P3.1** The Mausund error budget: M2's +3 % from NorKyst (`gauge_gains=M2,…`) together with the overtides that are too weak (M4 0.35×). See Validation.
2. **P1.3** A day of the full Frøya coastline mesh in 3D. See Geometry.
3. **P4.4** Per-column surface heat and freshwater fluxes, the next step for a real 3D run; then **P1.5** a real NorKyst 3D nested run.
4. **F.4** The wave model's RK propagation cost (≈ 80 of 141 ms per step), and choosing WAM's or SWAN's source scheme from observations. See Waves.

## Validation (P3) — [notes/validation.md](notes/validation.md)

- [ ] **P3.1** Remove M2's +3 % at Mausund (2.6 cm |ΔZ|, the largest error, NorKyst's own) with `froya_real_data gauge_gains=M2,…` so N2 follows; run it with the overtide work, since M4 scales with M2².
- [ ] **P3.1** Find why the overtides are too weak (M4 0.35×, MS4 0.34×, M6 0.28×): suspects are friction (Manning 0.025 everywhere) and the 5.3 m station node.
- [ ] **P3.1 / P3.3** The month-long real-domain run (`froya_real_data … hours=720`, or 30+ days on the coastline mesh): resolves N2/Q1, checks long-run stability, mass and no residual currents at rest.
- [ ] **P3.1** Settle whether the model's island lee near the skerries (M2 current 0.01–0.07 m/s against NorKyst's 0.246) is right, from observed currents (farm-site surveys, IMR/NIVA moorings near Frøya).
- [ ] **P3.1** Re-run P3 at 1 km against P2 at 500 m with the relaxed positivity CFL (dt ratio now ≈ 1.3–1.4) and stations interpolated at the gauge.
- [ ] **P3.1** The 500 m, 15-day harmonic run (≈ 2.5 h) for the constituents at resolution.
- [ ] **P3.1** Fetch real Kartverket (H, G) for Bergen, Stavanger, Trondheim, Kristiansund (`scripts/kartverket_gauge.sh`); fix the approximate `norwegian_stations` positions (Heimsjø is at 63.425, 9.102).
- [ ] **P3.1** Report NorKyst-800's missing N2 to MET (0.027 m against 0.156 m observed at Mausund); re-run the Bergen NorKyst fit with the P0.15 period fix and check it for the same deficit.
- [ ] **P3.1** Run the model against NorKyst-800 barotropic tides for a test period.
- [ ] **P3.1** Document skill scores (RMSE, bias, correlation) per station.
- [ ] **P3.1** Give `HarmonicAnalysis::fit` the guards of `fit_reference_constants` (Rayleigh check, inference, `resolvable_constituents`) or port its callers; an `AnalysisError` in place of asserts and `String` errors.
- [ ] **P3.2** The Pareto benchmark on the same hardware: M2/S2/K1/O1 gauge error against core-hours, ROMS 2D at 800/400/200 m against DG P1–P4.
- [ ] **P3.1** Low priority (≤ 4 mm left): whether the interior amplifies the diurnals (shelf waves).
- [ ] **P3.1** On hold: η-based limiting; revisit only if a real run shows oscillations.

## Salmon farm and cage drag (F.1) — [notes/farm.md](notes/farm.md)

- [ ] **F.1** Run farm cases with horizontal viscosity (`local_time_stepping_farm nu=1`, 2.1× the inviscid LTS cost) and choose ν: constant 1–10 m²/s or `HorizontalViscosity2D::smagorinsky(cs).with_background(nu)` (Smagorinsky alone ≈ 5e-4 m²/s).
- [ ] **F.1** A farm-wake case at a real fjord site with real tidal currents through an open boundary (the test fjord has mm/s currents; `farm_3d` is a periodic channel).
- [ ] **F.1** Net-made wake TKE (`P = Σ λ_l |u_l|²`) as a source in GLS's `k` and `ψ` per caged layer (Beudin et al. 2017, `c_ψ4`).
- [ ] **F.1** Compare the cage wake with a farm survey or flume data before trusting the default a = 4/(πR) (Løland's r = 1 − 0.46 C_d is left to the resolved flow).
- [ ] **F.1** Check a site's guessed cage count and size (`FarmSite::cage_layout`, Kattholmen 2 × 4) against an aerial photo or the operator.
- [ ] **F.1** Angle dependence C_d(θ) and the lift term, for square cages in oblique flow.
- [ ] **F.1** Net deformation in strong currents (the net lifts, d_net shrinks).
- [ ] **F.1** Low priority: a sparse layer-drag buffer over `CageDrag2D::nodes` (`ModeSplitIntegrator` `layer_drag_rate` is dense and zeroed every step by `Hydrostatic3D::layer_drag_into`).
- [ ] **F.1** Low priority: drag on the feed raft and mooring lines (read by `io::farm_site`, not modelled).
- [ ] **F.1** Not worth it at farm steps: make the explicit `G` shear part of the 3D cage (and bottom) drag consistent with the implicit pass; at λΔt ≥ 10 it reverses stopped flow by < 1 % (keep λΔt ≲ 5).

## Particles (F.2) — [notes/particles.md](notes/particles.md)

- [ ] **F.2** The walk's drift K∇h/h for depth-integrated tracers, H_z-weighted for σ-layers (Dimou & Adams 1993); the walk is right only for uniform depth.
- [ ] **F.2** A physical basis for the horizontal K from a site's drifter or dye spreading (connectivity shares move by 9–73 over 0.01–1 m²/s).
- [ ] **F.2** Check connectivity at the pens converges with mesh refinement under Smagorinsky K.
- [ ] **F.2** Log K along the larvae's paths, to explain why `kh=smag` keeps more larvae at home (82.5 %) than a constant 0.01 (73.3 %).
- [ ] **F.2** Draw release points per seed and per batch, so the seed spread measures release uncertainty, not only walk noise.
- [ ] **F.2** Log the larvae's largest displacement in `farm_3d` before longer runs (the 6 km periodic channel is shorter than the 7.1 km tidal excursion).
- [ ] **F.2** Measure the Euler–Maruyama error against Δt where K varies over a step; consider Visser's half-drift step or backward Itô (Spivakovskaya et al. 2007).
- [ ] **F.2** Copepodid infectivity(age, T) (Skern-Mauritzen et al. 2020) next to `SalmonLice::survival`, as the `ConnectivityRecorder::record` weight.
- [ ] **F.2** Light for lice from `AtmosphereReader` shortwave or cloud cover, salinity/CDOM-linked attenuation (Aksnes et al. 2009), a per-position sun for wide domains.
- [ ] **F.2** Keep Ω at the w-points in `Solution3D`, so the tracker's dσ/dt matches continuity exactly (a layer-centre average today).
- [ ] **F.2** Cubic time interpolation between snapshots for long output intervals.
- [ ] **F.2** Resuspension of settled particles above a critical bed stress (3D stranding also uses column depth only).
- [ ] **F.2** Locate lice cue crossings within the step, or keep √(2KΔt) ≪ K/w_s near haloclines (the light gate's χ² is 13/23/41 at Δt = 5/10/20 s).
- [ ] **F.2** Second-order contact exposure with the exact passage time (segment–circle intersection) if UΔt cannot be kept small against the cage.
- [ ] **F.2** Polygon contact zones (point-in-polygon behind `ContactZone::covers`) for site polygons and whole farms.
- [ ] **F.2** A bucket grid over contact zones in `ConnectivityRecorder::record` (O(particles × zones)) for region-wide registers.
- [ ] **F.2** Parallel connectivity recording per particle (`FarmParticles::steps`); matters at 10⁵+ particles.
- [ ] **F.2** Low priority: split `Solution3DVelocity` sampling by what the caller needs (RK stages sample K they discard; 0.6 s of 163 s).
- [ ] **F.2** Curved isoparametric elements in particle walking, `inverse_bilinear` and `PointLocator2D`'s Newton, once P1.3 has them.
- [ ] **F.2** Only if needed: pass `&Particle3D` to behaviours for per-particle traits; per-tidal-phase exposure bins (needs contact times beyond `Contact::first`).

## Waves (F.4) — [notes/waves.md](notes/waves.md)

#### Sources
- [ ] **F.4** Choose WAM's or SWAN's source scheme from observations (a buoy or ADCP wave record at a sheltered farm), then decide whether `wave_sources=wam wave_substeps=4 wave_cfl=1.0` becomes the default (WAM's is 1.5× cheaper, 6–9 % lower H_s in the lees).
- [ ] **F.4** Sweep `GrowthLimiter::Rate`'s C (3e-7 and 1e-6 against today's 5e-7) at the lee and Frohavet points and on the fetch/duration gates: is the lee's −9 % the cap or real?
- [ ] **F.4** Retune whitecapping (Komen δ=1: E* 0.60× KC92, 1.5× Pierson–Moskowitz at 1000 h): try van der Westhuysen 2007 with Yan input or ST6, then tighten the gates.
- [ ] **F.4** Triads (LTA, Eldeberky 1996) for the surf zone.
- [ ] **F.4** `c_σ` from a changing depth (∂σ/∂d ∂d/∂t): tides shift frequencies over flats.
- [ ] **F.4** Diffraction behind islands (phase-decoupled, Holthuijsen et al. 2003) for farms sheltered by skerries.
- [ ] **F.4** With substeps the boundary spectra and wind are held over the outer step; check against hourly forcing before ~10 min steps.
- [ ] **F.4** `Quadruplets::source_and_diagonal` leaves out a component's share in its own gather on grids with frequency ratio ≥ 1+λ: assert or handle it.
- [ ] **F.4** Drop `SourceIntegration::Midpoint` and `with_time_integrator` unless the midpoint helps with `Implicit`.
- [ ] **F.4** Low priority (not farm water): under 1 m depth the 56 s step has +8 % H_s bias; try moving the DIA's `quadruplet_scale` into the substeps.
- [ ] **F.4** Low priority: SWAN's tail beyond σ_max in the whitecapping means (`Means::of`, `src/waves/sources.rs`).

#### Depth limit
- [ ] **F.4** Re-run the storm with `wave_depth_limit=0` on current code, for like-for-like and the limit's uncontended cost.
- [ ] **F.4** Read the coupled storm's limiter counts by depth band (`depth_limit_bands`); if it binds in metres of water, count the removed variance as breaking in `dissipation()`.
- [ ] **F.4** `the_depth_limit_leaves_a_resolved_surf_zone_to_breaking` takes 10.8 s: trim it or move it to the slow tier (`.config/nextest.toml`).
- [ ] **F.4** Check the depth limit's γ against SWAN's very-shallow limiter before trusting the surf zone or a coarse grid's coastal nodes.

#### Validation
- [ ] **F.4** H_s, T_p and direction skill against the nearest wave buoys (MET/Kystverket, E39 Sulafjorden and Halsafjorden) and MyWave WAM hindcasts, focusing on the lees.
- [ ] **F.4** A 500 m P2 wave grid (`wave_mesh=120,90,2`): are Frohavet (17 % low) and the lee (19 % low) the 1 km grid's or WAM's error?
- [ ] **F.4** A reflection coefficient for walls (cliffs, quays); waves dying on the coast drive spurious coastal currents.
- [ ] **F.4** Check which wave nodes along 63.75°N (8.22–8.28°E skerries) are dry at mean and low water (the reef point loses 7.5 cm more with absorbing land under SWAN's sources).
- [ ] **F.4** Nest NorKyst for the storm days (`norkyst=`) for the remote surge (+0.29 m observed against +0.08 m), then compare the wave setup at Mausund with the gauge's residual.

#### Coupling
- [ ] **F.4** H_s, T_p and direction at the stations (gauges, NorKyst's point, farms) in the station files.
- [ ] **F.4** Measure the coupling lag's error at Δ = 10 / 5 / 2.5 min; if it matters, extrapolate (`2q(t) − q(t−Δ)`) or a second pass.
- [ ] **F.4** Make `WaveForceForm::Dissipation` the `CoupledWaves2D` default once a surf-zone case with a measured set-down is checked.
- [ ] **F.4** The coupling passes `compute_dt(cfl)` to the dissipation, not the step taken (`span/steps`): keep the exchange's last `dt`.
- [ ] **F.4** A wave-dependent breaking flux into GLS: ρg × (whitecapping + breaking) per column from `WaveModel2D::dissipation` (COAWST `TKE_WAVEDISS`).
- [ ] **F.4** 3D vertical structure (a): the Stokes transport in continuity (Eulerian return flow `D ū = −M_s`); changes the 2D model too.
- [ ] **F.4** 3D vertical structure (b): the breaking force near the surface (Uchiyama 2010 `B(z)` or a roller), gated against the constant-ν undertow (Svendsen 1984, Stive & Wind 1986).
- [ ] **F.4** 3D vertical structure (c): the vortex force and Bernoulli head (McWilliams et al. 2004; ROMS `WEC_VF`).
- [ ] **F.4** Waves on the 3D bed: `BottomDrag3D` with Soulsby's enhancement or Grant–Madsen roughness (ROMS `SSW_BBL`).
- [ ] **F.4** The maximum bed stress τ_max with the wave–current angle, for sediment and mooring loads.
- [ ] **F.4** `farm_3d` with `wind=`: run the wave model on the farm mesh, couple the Stokes drift each output interval and use `surface_roughness` instead of the constant `waves=`.
- [ ] **F.4** Lateral mixing for the longshore current, and a runup case with `WetDry` at a real shoreline (the force on partly dry elements is untested).
- [ ] **F.4** Cheaper Stokes drift sampling for 10⁵ particles: tabulate per node, or Breivik 2016's two-parameter profile.
- [ ] **F.4** Low priority: the vertical Stokes velocity (left out, as in OpenDrift/LADiM). Later: cage drag from the orbital velocity.

#### Cost
- [ ] **F.4** Profile the RK propagation's volume terms, face terms and per-stage positivity scaling (≈ 80 of the 141 ms step), and `compute_dt`'s O(points × components).
- [ ] **F.4** Sources without the DIA (102 ms on 1 thread): pass the cached `cg` into `advance`, cache the friction's `sinh(kd)` in `set_water_level`, try a vector `exp`.
- [ ] **F.4** Node-major `WaveSolution` to remove the last two transposes per step (≈ 15 ms per half-step at 24 threads).
- [ ] **F.4** Fuse adjacent implicit half-passes across steps and between substeps (a pair is 9.6 ms of 141); measure first.
- [ ] **F.4** Parallelise an exchange's per-node diagnostics (`parameters`, `bed_wave_stress`, `radiation_stress`, `StokesDriftField::new`, wavenumber tables) in one pass per spectrum (an exchange takes 176 ms).
- [ ] **F.4** L2 projection from circulation to waves in `WaveCoupling2D` (`M_c⁻¹ Pᵀ M_f`), replacing the cap in `WaveCoupling2D::currents`.
- [ ] **F.4** Measure a 2 km P2 wave grid (`wave_mesh=30,23,2`) against 1 km P1 at Frøya, for cost and lee H_s.
- [ ] **F.4** The 0.04 Hz bin in 150 m sets the 7 s step: check `f_min` 0.05 Hz (≈ 25 % longer step) against a 0.04 Hz run, or LTS by frequency group.
- [ ] **F.4** Mask the wave grid to the coastline mesh's footprint (638 of 10,040 nodes lie outside it).
- [ ] **F.4** Lower priority: implicit geographic stepping with Gauss–Seidel sweeps (SWAN), fewer components (24 × 25).

#### Wet/dry and forcing
- [ ] **F.4** Put the coast where the water ends in partly dry elements, with `WetDry`'s shoreline subcells (dry nodes are zeroed today).
- [ ] **F.4** Then decide whether `WaveModel2D`'s default becomes absorbing land (changes the bits of every gate with land).
- [ ] **F.4** Boundary spectra for past dates (the June 2025 runs) from MyWave 800 m's hourly partitions (wind sea + three swells, JONSWAP with cos^m), gated on days where both exist.
- [ ] **F.4** Low priority: mask sources at dry nodes in mixed DIA chunks (≤ 2–3 ms of 105); a dry-node list for `absorb_on_land`; a wind-only atmosphere snapshot on the wave mesh.

#### Viewer
- [ ] **F.4** H_s and the force per node in the coupled run's snapshots; H_s/direction colouring and arrows, a breaking map, Gerstner waves from each node's spectrum.

## Viewer (F.3) — [notes/viz.md](notes/viz.md)

- [ ] **F.3** Separate physics from σ-truncation in the fjord3d rest-state flow under GLS (3.6e-5 m/s at 1 h, growing): halve level spacing and κ in turn, compare with Phillips's boundary-layer velocity.
- [ ] **F.3** Compare the fjord3d cage wake at P1 on 40 m quads with the 2D 20 m run, a 3D P2 run and the integrated `CageDrag2D` drag (3D LTS would allow P2, see P1.3).
- [ ] **F.3** Write `froya_real_data`'s `Clock:` line to `<output>/run.log` or a `clock.txt` the viewer reads (VTU replays lack date and gauge).
- [ ] **F.3** Find which material's shader defs cause Bevy's `pbr_fragment` WGSL error (`cannot find declaration of uv`), likely `viz/src/shaders/water.wesl`, before Bevy 0.20 ships.
- [ ] **F.3** An app-level test (`MinimalPlugins`, `switch` twice) that toggling the photo view restores the chart view's materials, light, clear colour and HDR.
- [ ] **F.3** Switch runs in one window: build scenario, terrain and resources `OnEnter` a run state; keep terrain between replays of one domain.
- [ ] **F.3** More menu options: threads, order, LTS levels, particles, sites (`data/sites/*.txt`), places, terrain patches.
- [ ] **F.3** Draw the run's own `station_mausund.txt` (10-min samples) in the gauge trace for hourly VTU runs (hourly frames clip peaks ≈ 3 %).
- [ ] **F.3** The 15-day Frøya run with 10-minute frames (≈ 4 h, 2.3k frames) for on-demand replay.
- [ ] **F.3** A close-up of tidal flats wetting and drying: find a resolved flat, add a `--place`.
- [ ] **F.3** Offline connectivity from `.dgsnap`/`.dgpart` (needs each particle's source cage and interpolation between frames), or record online.
- [ ] **F.3** Save the solver state on live saves, not the viewer's f32 snapshot; add S to the viewer's `Layers`.
- [ ] **F.3** Drive the photo view's sea from the F.4 wave model (H_s, T_p, direction per node) and its wind from `AtmosphereReader`; ride the waves on the top σ-layer's current in 3D.
- [ ] **F.3** A movable, see-through 3D section (cross-channel behind the cages, or placed with the mouse); a temperature scale that follows the field.
- [ ] **F.3** Builders in `viz/` for other live scenarios (nested runs, the Mausund sub-domain).
- [ ] **F.3** Cut terrain patches out of coarser patches too, so overlapping patches work; a world-space label per farm site.
- [ ] **F.3** Measure the photo view's GPU cost (32 waves × 2 phases × 3 noises per pixel); bake into a height/slope texture if too slow. Make the wave-group noise periodic over the hourly `misc.z` wrap.
- [ ] **F.3** Reflections of land, cages and frame (SSR or a planar reflection) and shadows for low sun; shallow-water optics (caustics, bed refraction, swash foam).
- [ ] **F.3** Particle trails, and a density field instead of points for 10⁵+ particles.
- [ ] **F.3** Low priority: a bucket grid for terrain boundary snapping (0.2 s at 7k segments); `bracket` returning nothing on a far seek until both frames arrive.
- [ ] **F.3** Blocked: salmon in the cages (pfish-bevy, once the add-on has a salmon).

## 2D numerics (P1.1, P1.2, P1.7) — [notes/2d-numerics.md](notes/2d-numerics.md)

- [ ] **P1.2** Limit η (and velocities or characteristic variables), not h, with a troubled-cell indicator (TVB-M/KXRCF); limit h only in partially dry cells (Vater et al. 2019). Strict Kuzmin on h collapses sloping elements to P0 every half-cycle.
- [ ] **P1.7** Nonlinear SWE convergence with bathymetry on curved meshes, asserting N+1 in `tests/convergence_test.rs` (thresholds 1.5/2.5 today).
- [ ] **P1.1** Port the 2D tracer's BR1 diffusion (`tracer_2d.rs`) onto `br1_gradient_element`/`br1_diffusion_element` (it allocates whole-mesh arrays every RHS and runs serially). Its walls take the interior flux, non-dissipative with a zero-gradient ghost: no flux through walls, plus the eigenvalue test (P1.3).
- [ ] **P1.1** Remaining stage allocations: a buffer-reusing `clone_from` for `DGSolution1D`/`DGSolution2D`/`TracerSolution2D`/`Solution3D`; limiter `Vec`s (P2.3); `Simulation3D` off the stage workspace (P4.1).
- [ ] **P1.2** End subcells on physical boundaries stay first order: the outer slope neighbour from the BC, or one-sided slopes where both nodes are wet.
- [ ] **P1.2** Dry fronts lag (Ritter 1 mm front at 71.5 m against 79.8 m): suspects strict Kuzmin at the front and zeroed momentum below h_dry; revisit with η-based limiting.
- [ ] **P1.2** Retire the `Standard` wet/dry path and the cell-average/`linearize` workarounds once the legacy `ssp_rk3_swe_2d` users are on `Simulation` (P6).
- [ ] **P1.2** A roughness map from data (sediment type or calibrated n per region), projected like `Bathymetry2D::project` (Frøya uses n = 0.025 everywhere; a Mausund phase suspect).
- [ ] **P1.2** Guidance for `h_dry` (≈ 1e-4 of the characteristic depth), derived from the bathymetry or relative (1 mm dominates the 0.1 m Thacker error, 7.4 % against 4.5 %).
- [ ] **P1.2** Chen–Noelle (2017) reconstruction instead of Audusse, for bed steps larger than the depth.
- [ ] **P1.2** `SWE2DRhsConfig::new` defaults to `Standard` unchecked: warn on an unbalanced sloping bed in `with_bathymetry`, as the builder does.
- [ ] **P1.2** `outer_nodes` scales by affine element heights: per-node isoparametric factors (P1.3).
- [ ] **P1.1** Specialise hot paths through the kernel split: a `const N` kernel for fixed-size products; skip `reference_to_physical` + `SourceContext2D` for sources that need neither (P2.2).
- [ ] **P1.7** Merge the three legacy long-run conservation tests in `tests/swe_2d_test.rs` into one on `Simulation` when tests move off the legacy steppers.
- [ ] **P1.2** Only if friction's first-order error matters on flats: second-order IMEX for point-implicit friction (13–32 % above exact decay at dt·Λ ≈ 3).
- [ ] **P1.2** Lower priority: convex DG/FV blending (Hennemann–Gassner) so one node below `h_dry` doesn't switch a whole element to subcells.

## Geometry and meshes (P1.3) — [notes/geometry-mesh.md](notes/geometry-mesh.md)

- [ ] **P1.3** `froya_real_data tide3d=1`'s progress line reports T only in columns ≥ the thin depth, where the thin-column drift never showed until it spread: add the thin and shore columns' range.
- [ ] **P1.3** A day of the full Frøya coastline mesh in 3D (≈ 1.06 s per step at 24 threads): M2 cycle and gauge, η and ū at Mausund against the 2D run.
- [ ] **P1.3** Mass-weight the 2D tracer limiter's element means (`solver/limiters/tracer_2d.rs:238`, `:708`, `ops.weights[i] * h` without J): use `geom.node_mass`.
- [ ] **P1.3** A jittered-mesh case for `EntropyStable`/`WetDry` in `convergence_test.rs` (the averaged metric loses an order pointwise on such meshes).
- [ ] **P1.3** Rerun the unsmoothed full-mesh 3-day baseline on current code (without `slopes3d`, ≈ 1–2.5 h) for a like-for-like comparison with the 19 m free depth.
- [ ] **P1.3** Decide the 2D `rx0` default, after the map-wide NorKyst current comparison (`output_minutes=60`, `scripts/norkyst_current_comparison.py`).
- [ ] **P1.3** Make implicit vertical advection (`Hydrostatic3D::with_implicit_vertical_advection`) the library default once the full Frøya mesh has run with it.
- [ ] **P1.3 / P2.5 / P4.5** Local time stepping in 3D: the step is bound by 2.5–3 m sound elements (0.79 s), and the farm's 40 m quads set the fjord's. Run the barotropic pass on `MultirateStepper` with hooks for the filter/DU_avg2 sums (per-element stage weights). Blockers: the filter's uniform substep grid, DU_avg2, the nodal identity, AB3 history.
- [ ] **P1.3** Cap or ramp the first 3D steps from rest (`compute_dt` sees no Ω at rest; thin layers need |Ω|Δt/H_z ≤ 1).
- [ ] **P1.3** The 3D viscous step bound (`element_dt_viscous_swe_2d`) is harsh (ν = 50 m²/s: 0.54 s against 5 s): implicit shear viscosity, or the bound from the operator's spectrum.
- [ ] **P1.3** What dropping baroclinic pressure in thin-column elements (a one-element coastal band) does to estuarine and wind-driven shore currents, once tides and rivers run in 3D.
- [ ] **P1.3** Watch how often the splitter falls back to element-mean tracers (`CONSTANCY_TOLERANCE`) in a tidal run.
- [ ] **P1.3** Is the GLS creep (1 cm/s after a day at the 1 km grid's shallow slopes) physical slope flow (Phillips 1970; Wunsch 1970)? Compare with K/(slope·L) or run a slice.
- [ ] **P1.3** Lighten the 3D slope bound now that the vertical reference exists (does r_x0 0.25–0.3 hold Frøya at rest? At 0.2, element 11163 grows, e-folding ≈ 55 min).
- [ ] **P1.3** Mausund's pinnacle element 291 (a 1.2 m node among 42–55 m, unsmoothed) cools its bed layer from 8.6 to 6.0 °C within 30 min at 2 h of the 3D tide, below NorKyst's 7.50 minimum: the straddle below, seen in T.
- [ ] **P1.3** The straddle instability over unsmoothed cliffs (smoothing works around it): a per-element Haney rx1 bound, a capped or implicit pair density, or geopotential diffusion; run the spectrum harness (≥ 1800 s, check the residuals) on the slices.
- [ ] **P1.3** The shipped 3D rest state still grows at terrace vertices beside shore elements: an oscillatory mode, e-folding ≈ 2–2.4 h, period ≈ 3 h, in the bed layers across the pycnocline's foot (LimitedAkima about the reference; any Akima weight). Find its mechanism (a fully wet slice of the same terrace is stable) and fix it; gate `a_terrace_beside_a_shore_element_stays_at_rest_for_twelve_hours`.
- [ ] **P1.3** Rerun the full coastline mesh at rest with today's defaults for 24–48 h, seeded (`debug_3d=…,perturb=1e-6`): the 12 h round-off runs cannot see a 2 h mode.
- [ ] **P1.3** Coastline-mesh cost (≈ 4× the 500 m grid; 4.4e4 element-steps against 2.3e4 ideal): multirate level coherence (three-hop stage dependency ≈ 1.9×) and a non-subdividing quad mesher.
- [ ] **P1.3** `batched` SIMD and the Burn prototype assume parallelograms (`affine_metric`, `compute_volume_terms_batched`, `BurnGeometricFactors2D::from_cpu`).
- [ ] **P1.3** Triangles or quad-dominant meshes; remove the hardcoded 4 faces (`swe_2d.rs:386,908`).
- [ ] **P1.3** CSR connectivity: `MeshGPUData` as the single mesh representation.
- [ ] **P1.3** Curved high-order faces from an interpolated high-order map (Kopriva 2006), with a boundary projection of face nodes.
- [ ] **P1.3** Mass-weight `Bathymetry2D::to_cell_average`/`linearize` node means, unless retired first (P1.2).
- [ ] **P1.3** Profile the remaining curvilinear RHS overhead with AMD uProf (+2–3 % at 1 thread, +4–5 % at 24 on Frøya).
- [ ] **P1.3** If suite time matters: shorten `a_terrace_beside_a_shore_element_stays_at_rest_about_its_reference` (1–4 min, slow tier).
- [ ] **P1.3** Only if needed: implicit vertical advection's upwind flux about the vertical reference (`apply_tracer_transport_3d_about`); decaying the parent's bottom velocity below its bed in `OceanModelColumns`.
- [ ] **P1.3** Low priority: `raise_isolated_wet_nodes` on the rx0 functions' `NodeGraph`; `max_rx0` rebuilds the graph each call.
- [ ] **P1.3** When needed: Gmsh `$NodeData` (a bed in the mesh file) and `$Periodic` faces.

## Boundaries and nesting (P1.4, P1.5) — [notes/boundaries-nesting.md](notes/boundaries-nesting.md)

- [ ] **P1.5** A real NorKyst 3D nested run (T/S, baroclinic velocity): `Nesting3D` and `norkyst_nesting_subset profiles=1` exist.
- [ ] **P1.5** A datum for the parent ζ in the library default (NorKyst sits 0.28 m below MSL at Frøya); check whether NorKyst v3 is forced with sea-level pressure (`ib=1` would count the barometer twice).
- [ ] **P1.4** `BoundaryTides::with_transport_scaling` (`tide_transport=3`) as the default for atlas tides (Mausund sub-domain 8.1 → 4.2 cm); check the full mesh with it.
- [ ] **P1.4** Blend the bed towards the atlas depths near the open boundary for `BoundaryTides`, as `OceanModelState::blend_bathymetry` does.
- [ ] **P1.5 / P4.2** A point-implicit nesting relaxation band, 2D and 3D, so FRS-like τ ≈ Δt works (explicit today: τ 30 min against 0.25–10 s steps).
- [ ] **P1.5 / P4.2** Read parent snapshots lazily around the current time, 2D (`OceanModelReader`) and 3D (`from_file_with_profiles`, 46 MB per field for 3 days), for months of forcing or OPeNDAP.
- [ ] **P1.5** Correct NorKyst's own M2/S2 (0.3–1 cm, 10–20 min lag) in the corrected atlas from another reference (gauges along the boundary, TPXO/FES).
- [ ] **P1.4** Velocity from z-levels stops at 300 m: extend to full depth.
- [ ] **P1.5** `NestingRelaxation2D` relaxes the level at every band node ≥ 5 cm deep, puddles cut off by land included (Mausund element 58: a 5 cm puddle filled at 3e-7 m per step towards NorKyst's level): relax only water connected to the open faces, or momentum only in shore elements.
- [ ] **P1.5** A depth range in `norkyst-client`'s grid mode (`depth = "0:1:0"` today) as a second source of 3D profiles.

## Inputs and forcing (P1.6) — [notes/inputs-forcing.md](notes/inputs-forcing.md)

- [ ] **P1.6** Settle the topobathy sea part's vertical datum (NN2000 or chart datum, 1.43 m too shallow at Mausund): check against WFS `wfs.dybdedata` points; add the 8 cm NN2000 → MSL offset if it matters.
- [ ] **P1.6** Measure `lower_wall_land` in 3D (stratified rest, and a tide with `levels=20 slopes3d=on`) before making it the default, then in `viz/src/scenario.rs`.
- [ ] **P1.6** Put the coastline mesh's walls in the water in `scripts/gmsh_coastline_mesh.py` (dilate the opened land back, or the projected bed's 0 contour), never closing a sound.
- [ ] **P1.6** A farm-scale bed: tile the 1 m topobathy level over the 50 m model using the "Dybdedata - dekning" coverage layer; Kattholmen needs another source (dybdedata WFS or a survey).
- [ ] **P1.6** Pass the last step's `RiverInflow` to `Hydrostatic3D::update_vertical_velocity` (output Ω spreads river volume over the column).
- [ ] **P1.6** Fix `norwegian_coast_beta()` (1.6e-11 is 45°N; 60°N ≈ 1.14e-11); f(latitude)/β helpers, UTM33/Lambert for coast-scale domains.
- [ ] **P1.6** A spatial index for point-in-polygon and GSHHS inner rings in `coastline.rs:81-95` (the land mask costs ≈ 20 s of setup at Frøya).
- [ ] **P1.6** Fix or delete the unused `LandMask2D::from_coastline_and_bathymetry`.
- [ ] **P1.6** Count wet as h > 1 cm in the example's η statistics, so mm films don't set "η max".
- [ ] **P1.6** Cut the atmosphere forcing's memory (≈ 210 MB at 10⁶ nodes): share stencils per element, or drop them after regridding.
- [ ] **P1.6** River details: a fixed-depth profile (top d metres) for deep fjords; river momentum `Q/(width·depth)` for jets at narrow mouths; spreading over elements within a radius if one-element plumes show; the river T/S source in the legacy `CoupledState2D` path, or drop it with P6.
- [ ] **P1.6** If the wind-driven residual matters at farms: stress from the wind relative to the current, with a stability-dependent COARE drag.
- [ ] **P1.6** Low priority (< 5 mm): the tidal-potential sign against its docs (`source/swe_2d/tidal.rs:299-305`) and the diurnal Love factor.
- [ ] **P1.6** Not worth it now: a constrained bed projection at walls (median −0.2 m, 1st percentile −12 m).

## Performance (P2) — [notes/performance.md](notes/performance.md)

- [ ] **P2.5** A semi-implicit free surface (θ-method, as TRIM/UnTRIM, SLIM, Thetis), the main cost lever: farm meshes step at ≈ 0.024 s on gravity waves. Needs a global SPD solve and wet/dry care.
- [ ] **P2.1** The 2D `Simulation` has no check for a blown-up state (as `Simulation3D` gained 2026-10-10): stop at the first non-finite `h`, `hu`, `hv`.
- [ ] **P2.1** Port `compute_dt_advection_2d`, `compute_dt_tracer_2d`, `compute_dt_viscosity`, `Mesh::compute_dt`, `time/ssp_rk3.rs:110` and 1D `compute_dt_swe` to `compute_dt_swe_2d`'s per-node metric form in one helper (`2/(λ_r+λ_s)` for advection).
- [ ] **P2.5** LTS oversubscription: under ≈ 40 % external load the farm took 358 s at 24 threads against 3.9 s at 4. Serial cut-off by work, a capped pool for small meshes, or documented thread counts.
- [ ] **P2.5** Viscous cost (≈ 2× inviscid): a per-stage BR1 gradient cache for subset-RHS neighbours; store `νh` with the gradients in pass 1.
- [ ] **P2.5** Smagorinsky `compute_dt` (≈ 13 % of a step at 24 threads): record each element's max ν in the last stage's gradient pass (lagged), or bound every few steps.
- [ ] **P2.2** Line-by-line 3D horizontal kernels (`transport_divergence_element`, `advective_divergence_element`, the PGF volume loop) with contravariant vectors loaded once: ≈ 3× fewer pairs at P2.
- [ ] **P2.2** Finish SIMD batching of `element_rhs`: one node-state pass, a batched surface loop/LIFT, no copy of the batched volume.
- [ ] **P2.2** Shorten the face-pass gather with per-edge node-index lists, or an AoSoA layout (3.8× against 2.5× in the microbenchmark).
- [ ] **P2.2** Precompute the subcell batch's outer-node distances per face node with the geometric factors.
- [ ] **P2.2** Fuse the face pass into the element loop (coloured or two-phase schedule), removing the second memory pass.
- [ ] **P2.3** Move the Kuzmin limiters' per-stage `Vec`s to a workspace; fuse limiter, wet/dry correction and implicit damping into one in-place pass.
- [ ] **P2.5** Make the Kuzmin limiters work under LTS (they panic today).
- [ ] **P2.5** Validate LTS on Frøya over the 15-day harmonic run (`lts=6` matched global M2 to 1e-4 over 6 h).
- [ ] **P2.5** Check that cage-shed eddies survive leaving the farm's LTS level (interface error ≈ 50× SSP-RK3's).
- [ ] **P2.2** Measure `-C target-cpu=native` for the SIMD paths and `mimalloc` for the allocating paths.
- [ ] **P2.2** Override `SourceTerm2D::add_element` for wind stress, pressure and tidal potential with precomputed per-node fields (Coriolis gave ≈ 11 %).
- [ ] **P2.3** One-level fused stepper: the third stage writes the state directly; measure first.
- [ ] **P2.5** Reduce LTS interface spreading (2.6× of an ideal 4.6× on the farm): coherent levels, grading ≤ 1.2, or decoupled coarse stages.
- [ ] **P2.5** Lag the viscous flux across LTS interfaces where ν is small (2.60× → 2.44× on the farm).
- [ ] **P2.1** Measure linear stability limits for curved elements and N ≥ 5 (`PositivityBound` does not relax without a table entry).
- [ ] **P2.6** A wave-step bench before the F.4 SIMD port.
- [ ] **P2.6** Full-RHS and full-step throughput at realistic size (≥ 100k elements, bathymetry, limiter, open BCs); DOFs/s, scaling over 1–16 cores, bandwidth; allocation counting in CI.
- [ ] **P2.6** Audit the slow test tier (`.config/nextest.toml`, 25 gates, 28 test-minutes): spread-sea ladders, 3D rest gates' model hours, particle counts.
- [ ] **P2.4** Limit only troubled cells.
- [ ] **P2.7 / P0.11** Decide GPU: fix Burn (`src/solver/burn/rhs.rs` computes the HLL flux and discards it, so no surface term; fusion disabled, ~13× the bytes per node-stage) with a burn-vs-CPU test, or replace it with fused f64 CubeCL/cudarc kernels over CSR faces. Pays only from ≈ 10⁵ elements; fewer steps come first.
- [ ] **P2.7** An MPI/domain-decomposition plan (METIS, halo exchange of face traces).
- [ ] **P2.1** Low priority: more SSP stages (SSP-RK(9,3)/(10,4)) need a non-two-register `SspScheme`; the Zhang–Shu r/s split in ρ (≈ 2 % on Mausund).
- [ ] **P2.2** Only if `Standard` stays: sum-factorised volume term (`swe_2d.rs:835-874`), diagonal LIFT (`kernels.rs:490-517`), one Riemann solve per face.
- [ ] **P2.5** Low priority: the LTS coarse-step anchor from a log₂ histogram (≤ 0.5 % on Frøya); keep wet/dry films from setting a tiny Smagorinsky step, only if it shows.

## 3D: mode splitting (P4.1) — [notes/3d-mode-splitting.md](notes/3d-mode-splitting.md)

- [ ] **P4.1** Revisit the ROMS `rufrc`-style G (full PGF in the 3D RHS) now that the σ-pairs PGF has the DG face coupling of η.
- [ ] **P4.1 / P4.2** Check whether the 3D kernels still read η rather than h (PR 1 left moving them to h to P4.2).
- [ ] **P4.1** Low priority: the internal seiche's small dt-dependent phase residual (+0.011 / −0.018 / −0.039 % at 240 / 120 / 60 s).

## 3D: numerics (P4.2, P4.3, P4.5) — [notes/3d-numerics.md](notes/3d-numerics.md)

- [ ] **P4.5** The splitter's fixed per-stage data movement (≈ 60 field passes, 27 % of the step at dt 4 s, 12k elements): kernels read `H_z φ` and divide on load, the post-stage hook converts in place.
- [ ] **P4.5** The forward-backward (AB3-AM4) barotropic fast mode, fewer RHS per substep.
- [ ] **P4.5** Fewer parallel regions per barotropic stage: the fused `barotropic_stage` as the RHS's per-element continuation (≈ 4 of 85 ms at 1024 elements).
- [ ] **P4.2** Replace `with_thin_columns_at_rest`'s whole-`Solution3D` copy each stage with a thin-column mask read by the kernels.
- [ ] **P4.2** Skip `w` and `rho` in `Solution3D`'s stage combination, or a splitter-own stage state (changes the bits of `w`).
- [ ] **P4.2** A cheaper band parent-column evaluation (`OceanModelColumns` searches per node, stage and layer): precompute vertical weights, or time-interpolate once per baroclinic step.
- [ ] **P4.2** The parent's shear dispersion in G's depth mean at nested faces.
- [ ] **P4.3 / P1.3** ROMS `Vtransform=2` and the true Vstretching 4 (`ROMSVstretching4` is not ROMS Vs4; `SongHaidvogelStretching::hc` is unused in `vertical/sigma.rs` `z_at_levels_into`); Ω must use ∇·(H_z u) once H_z varies horizontally.
- [ ] **P4.3** An r_x1 (Haney) diagnostic on the σ levels and an r_x1 bound for the 3D grid.
- [ ] **P4.3** The σ-pairs PGF on curvilinear elements (affine metric today; `Hydrostatic3D` asserts affine): the chain rule `rx_i Dr + sx_i Ds`.
- [ ] **P4.5** The horizontal Kuzmin tracer limiter as the default (`farm_3d`, `Hydrostatic3D`; the P1 farm channel undershoots 0.07 °C without it), after a realistic run over slopes.
- [ ] **P4.5** Why P3 on 250 m without viscosity ends 3.3e-9 °C out of range (2–8e-12 at P1/P2): the vertical advection's element mean or the Kuzmin P3 bounds?
- [ ] **P4.5** 3D Smagorinsky on a fjord with a sill and tidal shear (may over-damp a tidal mixed layer; else Leith or Ri-dependent ν).
- [ ] **P4.5** Rotated/geopotential horizontal diffusion of 3D tracers via BR1 (none exists; `viscosity_3d` is momentum only).
- [ ] **P4.5** Subcell (GLL-cell) finite volumes for shoreline tracers instead of element means per level (needs 2D subcell fluxes in DU_avg2).
- [ ] **P4.5** Port the 3D advective dt bound to the 2D per-node metric form (`compute_dt_swe_2d`), with a linear stability measurement (changes what `cfl` means). Mausund holds `cfl_3d=2` and blows up at 3 in a 50 m element during the ramp (`froya_real_data` takes 1.5 now); find which term sets that limit (internal waves, the splitting, the implicit vertical advection at outflow Courant 10).
- [ ] **P4.5** `Simulation3D`'s default `cfl` (0.5, `SimulationConfig`'s, shared with 2D) is 0.17–0.25 of the measured limit: give 3D its own default once the bound is ported.
- [ ] **P4.5** EOS: delegate to the fixed UNESCO EOS with a linear fast path; TEOS-10 later. In `UnescoEOS::update_density`, `s * s.sqrt()` for `s.powf(1.5)` and a flat parallel loop.
- [ ] **P4.5** RPE (`PotentialEnergy3D`) in 3D run output (a `Simulation3D` callback or NetCDF attributes).
- [ ] **P4.5** One in-place Thomas solver for `tridiagonal::solve_tridiagonal`, `implicit_advection::ColumnSolve::solve` and the waves' `thomas`; in `solve_diffusion_column`, one elimination per pair and 8 columns as SIMD lanes.
- [ ] **P4.5** Keep the per-element viscosity in `ViscosityScratch3D` (allocated each step in `time_step_limits`/`compute_dt`).
- [ ] **P4.5** Move the 2D kernels' serial/parallel entry points and workspace guards onto `solver::core::blocks`; keep one of `DisjointChunks`/`blocks`.
- [ ] **P4.5** Bundle `apply_momentum_transport_3d`'s 12 arguments into `MomentumAdvection { horizontal, vertical }`, likewise for tracers.
- [ ] **P4.5** Remove `GriddedAtmosphere2D::split_for_3d`'s panic on a clone (split through the shared state, or `&self` cloning the `Arc`).
- [ ] **P4.5** Revisit or drop `TracerLimiter3DConfig::vertical_column_bounds`.
- [ ] **P4.5** Relabel or convert Ω and w in output (`Solution3D::w` is Ω at layer centres); report film speeds only where D ≥ the minimum column depth.
- [ ] **P4.2** Low priority: a baroclinic radiation OBC (per-mode Flather or Orlanski, ROMS RadNud), for narrow nesting bands (1 km: 18 % left).
- [ ] **P4.3** Low priority: the pycnocline PGF error on uniform levels (2–4e-4 m/s²): stretched levels or a tabulated ρ̄(z); higher-order vertical reconstruction in the PGF.
- [ ] **P4.5** Low priority, only if a profile shows it: G's advection from the previous step's transports; `Integrable::combine` overrides; PGF cubic integrals vectorised across levels; the remaining serial loops (≈ 3 ms/step at 8192 elements); a parallel-sorted or binned RPE; implicit vertical advection folded into the diffusion matrix.
- [ ] **P4.5** Low priority: the transient 1e-8 °C excess in the inviscid 125 m TVD run (Kuzmin bounds extrapolated past the end layers). Not worth it now: 2D passes on one thread below a size threshold.

## 3D: mixing (P4.4) — [notes/3d-mixing.md](notes/3d-mixing.md)

- [ ] **P4.4** Per-column surface heat and freshwater fluxes, Q_net/(ρ₀c_p), from bulk formulae on `AtmosphereReader` fields (one domain buoyancy flux today).
- [ ] **P4.4** Surface stress in thin columns (`with_min_column_depth`) through `G` as τ/ρ₀ (tidal flats get no wind in 3D today).
- [ ] **P4.4** Wave height for the surface roughness, z₀ₛ ≈ 0.6 H_s per column, from a MyWaveWAM reader or fetch-limited H_s(U₁₀, fetch).
- [ ] **P4.4** Positivity-safe advected `k`/`ψ` (Zhang–Shu scaling toward the element mean, or a length cap); check on a real run with fronts.
- [ ] **P4.4** k-ε with waves: σ_ε as a function of P/ε (Burchard 2001, GOTM `sig_peps`); until then k-ω or generic with waves.
- [ ] **P4.4** One bed roughness: `with_bottom_drag` hands its z₀ to `GlsMixing` (κ 0.41 in the drag, 0.416 in the closure); a spatially varying z₀ and a C_d consistent with the 2D Manning n (2D/3D undisturbed currents differ, 0.31 against 0.34 m/s at the farm).
- [ ] **P4.4** Thin columns take C_d,max = 0.1: check the 3D wet/dry front speed on a real beach against 2D `WetDry` with Manning.
- [ ] **P4.4** GLS cost (`vertical_implicit` 2.1 ms against 0.8 for Pacanowski–Philander, mostly `powf`): a k-ε fast path, one ε per w-point.
- [ ] **P4.4** Turbulence-advection cost: average layer transports on the fly in `turbulence_transport_rhs`, the end-point override as a reconstruction BC.
- [ ] **P4.4** Drop `PacanowskiPhilanderMixing`'s own `g`/`ρ₀` (API change).
- [ ] **P4.4** Low priority: reset thin columns' `k`/`ψ` on re-wetting; second-order bottom drag; reuse `GriddedWindStress`'s `tⁿ⁺¹` evaluation.

## 3D: validation (P4.6) — [notes/3d-validation.md](notes/3d-validation.md)

- [ ] **P4.6** A source for `TracerReferenceProfile` (a z-binned horizontal mean of the state or the parent), needed before the Kuzmin limiter over slopes in a realistic run.
- [ ] **P4.6** Move the smooth references to `from_smooth_fn` and measure (Frøya `tidal_run_3d`, `levels=N`, `seamount_3d.rs kuzmin=2`, `src/simulation/simulation_3d.rs:3359`).
- [ ] **P4.6** Tanh-pycnocline seamount at r_x0 0.08 spins up 0.38 → 1.5 cm/s over 5 days: 30 days, refine nx and levels, compare with Shchepetkin & McWilliams (2003, Fig. 10).
- [ ] **P4.6** Exact vertical PGF/advection energy exchange for curved profiles (the Hermite mean as the advection's face value); does it remove the residual seamount growth?
- [ ] **P4.6** The balanced reference sampled at the reference η gives an η/D error under a tide: refresh it or interpolate to current level depths; measure with a tide over the seamount.
- [ ] **P4.6** At r_x0 0.32 the Kuzmin limiter adds to the error flow: a detector for steep continuous fronts (Persson–Peraire), isopycnal bounds for spice.
- [ ] **P4.6** A Sognefjord-like estuary (sill, stretched levels, GLS, Coriolis) to steady state, against Stigebrandt's (1981) two-layer theory.
- [ ] **P4.6** Exact section budgets from the model's own face fluxes, RK-weighted (salt budget off 1–3 % from nodal `H_z u S`); needed for P5.2 farm exchange output.
- [ ] **P4.6** Periodic meshes: `Mesh2D::elements_at_vertex` has no periodic images, so Kuzmin bounds there are one-sided (also `limiters/swe_2d.rs:260`, `tracer_2d.rs:593`).
- [ ] **P4.6** The 3D limiter's remaining cost (1.6 ms/call): one buffer of inventory weights per call, T and S in one pass.
- [ ] **P4.6** Then compare with NorKyst-800 3D fields.

## Operational (P5) — [notes/operational.md](notes/operational.md)

- [ ] **P5.1** River forcing: NVE database (1,760 NorKyst rivers) and a ROMS river-file reader (`river_transport`, `river_temp`, `river_salt`) onto `source::RiverSources`; delete the superseded `Discharge2D`/`ConstantDischarge2D`/`RiverTracerSource` with P6.
- [ ] **P5.3** Refuse a restart resumed with different physics options (compare them, or a configuration hash).
- [ ] **P5.3** Numbered restart copies (`restart_keep=`) or per-time names, for spin-up branching.
- [ ] **P5.2** A ROMS variable-name mapping for `NetCDFWriter` (CF-1.8 today).
- [ ] **P5.3** 2D restarts (`Simulation`, `SWESolution2D`, the multirate stepper's state).
- [ ] **P5.3** A NetCDF export of restarts (`ocean_rst.nc`) once P5.2's conventions are settled.
- [ ] **P5.4** The operational pipeline: run scripts, monitoring/alerting, THREDDS/OPeNDAP serving.

## Tech debt (P6) — [notes/tech-debt.md](notes/tech-debt.md)

- [ ] **P6** Move `profile_cpu`, `quick_profile` and `high_res_benchmark` off `ssp_rk3_swe_2d`/`SWE2DTimeConfig` onto `Simulation` + `SWEPhysics2D` (precondition for the next item).
- [ ] **P6** Delete the legacy integrators (`ssp_rk3*.rs`, three `coupled_swe_tracer` variants, `burn_ssp_rk3.rs`; ~2.4k lines); `CoupledState2D` implements `Integrable`; one time loop with `Simulation3D::run_with_callback` folded in, landing callbacks exactly (they drift today: 300 s asked, 316 s apart).
- [ ] **P6** Build the `netcdf` feature in CI with `--features netcdf/static` (~5.5 min cold); consider it the documented alternative to conda.
- [ ] **P6** Collapse duplicate config types: `SWEFluxType2D`/`StandardFlux2D`, the three limiter enums, `HorizontalViscosity2D`/`3D` (differ only by `h_min`).
- [ ] **P6** Error handling: `SimulationResult` → `Result<SimulationStats, SimulationError>`; reachable panics (`timeseries_reader.rs:198`, `geotiff.rs:239-245`); library `eprintln!` in `tide_gauge.rs`.
- [ ] **P6** Prune ~200 root re-exports to a `prelude`; export `Simulation3D`.
- [ ] **P6** Feature-gate `tiff`/`shapefile`/`geo` behind `geodata`.
- [ ] **P6** Retire the mostly dead `solver/simd/` with the `Standard` kernel (P1.2).
- [ ] **P6** Decide crate vs repo name and a `dg-core`/`roms-rs` split.
- [ ] **P6** Characteristic-based limiting for 2D SWE (likely subsumed by P1.2).
- [ ] **P6** Low priority: faer 0.23 → 0.24 (setup-time solves only).

## Future (P7) — [notes/operational.md](notes/operational.md)

- [ ] **P7** IMEX time stepping via diffsol (implicit vertical diffusion).
- [ ] **P7** Subcell positivity preservation with convex limiting (Wu et al. 2024).
- [ ] **P7** hp-adaptive mesh refinement.
- [ ] **P7** Data assimilation (EnKF, then 4D-Var).
- [ ] **P7** Two-way nesting.
- [ ] **P7** Biological coupling (NPZD); sediment transport.
