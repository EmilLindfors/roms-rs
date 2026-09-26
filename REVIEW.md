# Code & Numerics Review

**Date:** 2026-09-25
**Scope:** The whole crate, measured against the project goal: ROMS-class modelling of currents along the Norwegian coast, in Rust, and faster than ROMS. Emphasis on the math.
**Supersedes:** the 2026-07-08 review, which is deleted and still available in git history (last present at commit `8679825`). Its findings are re-checked in §0.

**Method:**
- Full test run (`cargo test --release --no-default-features --features parallel,simd`): 1,024 lib, 61 integration and 59 doctests pass, with 43 doctests ignored. The whole suite finishes in under 2 s. Nothing realistic in scale or duration is exercised.
- Throwaway probes run against the real 2D RHS. They were not committed; their setups are in §1 so they can become regression tests.
- Five parallel source-level sub-reviews: 2D DG, mode splitting, 3D physics, boundaries/tides/nesting, performance.
- Direct verification of every headline claim.

**Evidence tags** used below:
- **[M]**: measured with a probe against the actual code.
- **[V]**: verified directly in the code or by derivation during this review.
- **[S]**: sub-review finding backed by a static derivation or a numpy replica of the code; not independently re-derived.

---

## Verdict

**What is sound:**
- The architecture: DG horizontal × finite-difference σ vertical with barotropic/baroclinic mode splitting, f64 throughout, SoA storage, rayon workspaces. This is the Thetis/SLIM design family.
- The DG core: GLL operators, strong-form surface/LIFT term, Roe and HLL fluxes, SSP-RK3.
- The tidal astronomy module.

**What does not hold yet:** "ROMS-like" and "faster". The layers that turn a DG solver into a coastal ocean model are where the math breaks:

1. **Bathymetry balance.** On realistic, steep, nodal bathymetry the 2D scheme is not well-balanced: a fjord at rest develops metre-per-second currents within an hour **[M]**. The only workaround (cell-averaged bathymetry with hydrostatic reconstruction, used by `froya_real_data`) does not conserve mass **[M]**.
2. **Open boundaries.** Every Flather-type BC applies the characteristic relation twice. About a third of each outgoing wave is reflected and the forced tide arrives at about 65 % amplitude **[V]**.
3. **Mesh and nesting.** Coastline-fitted meshes cannot be loaded. NorKyst nesting has no time base.
4. **Mode splitting.** The scheme is first-order and damps the barotropic mode (M2 loses 3–14 % per period) **[V]**. It double-counts advection and never passes 3D stresses to the depth mean.
5. **3D physics.** This layer is scaffolding: tracers are not constancy-preserving, vertical momentum advection is too large by a factor of the water depth D **[V]**, and the σ pressure-gradient error is as large as the real estuarine forcing **[S]**.
6. **Cost.** As configured, a NorKyst-sized 2D tidal run is roughly **~50× ROMS core-hours** (cost model, §5). Parity or better is plausible only per unit of tidal accuracy, at P3+, on water-only meshes, with local time stepping or an implicit free surface.

Every problem has a known fix in the literature, so the track is recoverable. But the roadmap must change: **make the 2D barotropic tide model correct and cheap first** (that is where DG can actually beat ROMS), and only then rebuild the 3D coupling on the ROMS recipe.

---

## Top correctness bugs (small, confirmed; fix first)

| # | Bug | Where | Evidence |
|---|---|---|---|
| 1 | Flather characteristic relation applied twice. Outgoing waves reflect with R ≈ −1/3; forced tide delivered at ≈ 0.65 | `boundary_2d.rs:381,821`, `tst_obc.rs:347`, `nesting_bc.rs:189`, `ocean_nesting.rs:218`, `chapman.rs:278` | [V] §4.2 |
| 2 | Hydrostatic-reconstruction interior flux evaluated on the reconstructed state. Mass leaks wherever B jumps across faces | `swe_2d.rs:494` (serial), `:1021` (parallel) | [M] §1.2 |
| 3 | `with_well_balanced` docs say to omit `BathymetrySource2D`. Doing so gives ≈ 0.6 m/s² lake-at-rest residuals | `swe_2d.rs:50-53,142-153` | [M] §1.3 |
| 4 | Tidal constituent periods truncated (M2 12.42 h, K1 23.93 h, …). Phase errors of degrees over month-long fits | `boundary/tidal.rs:76-113` | [V] §4.3 |
| 5 | 3D vertical momentum advection divided by Δσ instead of Hz, so it is D times too large | `advection_3d.rs:641` | [V] §3.2 |
| 6 | UNESCO EOS feeds dbar into a bar formula, making compressibility 10× too large (latent: 3D path does not use it yet) | `equation_of_state.rs:144-160,281-307` | [V] §3.8 |
| 7 | Chezy friction has an extra `/h`: 50× too weak in 50 m of water | `friction.rs:238-239` (+ 1D) | [V] §1.6 |
| 8 | Parallel fused Kuzmin+positivity limiter has no dry-element branch, so dry cells keep momentum | `limiters/swe_2d.rs:686-703` vs serial `:118-129` | [V] §1.7 |
| 9 | NorKyst nesting time read raw and clamped. The forcing freezes at the first snapshot | `netcdf_io.rs:1226-1236`, `find_bracket` | [V] §4.5 |
| 10 | `compute_dt_swe_2d` combines the global minimum element size with the global maximum wave speed; Frøya runs at CFL 0.1 | `swe_2d.rs:582-602`, `froya_real_data.rs:163` | [V] §5.3 |

**Status:** fixes for #4–#7 are merged in [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2). It also moves Ω to w-points (§3.1b). Line references in this document are to commit `8679825`, before that PR.

---

## 0. Status of the 2026-07-08 findings

| Prior item | Status (2026-09-25) |
|---|---|
| Top bugs 1–4, 6 (hardcoded 100 m, PGF double count, 3D walls, NorKyst layer, tidal phase sign) | **Fixed with regression tests.** Caveats below: the tracer wall leak remains (§3.4), and the layer fix is wrong for z-level files (§4.5) |
| Top bug 5, Burn GPU RHS drops the surface term | **Open** (deferred, P0.11) |
| Top bug 7, featureless build | **Fixed**, and CI guards it |
| §1.1 Well-balanced only when deg B ≤ p/2 | **Open, raised to BLOCKER** (§1.1, measured) |
| §1.2 Docs say to drop the bathymetry source | **Open** (§1.3) |
| §1.3 Reconstructed state in the interior flux, called "latent" | **Open and active**: `froya_real_data` uses cell-averaged B (§1.2) |
| §1.4 Kuzmin limits h rather than η | **Open**, and worse than stated (§1.4) |
| §1.5 Forward Euler barotropic subcycle, flat averaging, subcycle in every stage | FE **open**; averaging **fixed** (Hann filter, correct moments); subcycle-per-stage **open, 2× costlier** (§2.1–2.2) |
| §1.6 GLL aliasing | **Open** (§1.8) |
| §1.7 Zhang–Shu positivity uses only the GLL-node minimum | **Retracted.** For collocated DGSEM the GLL-node minimum is sufficient. The real gaps are §1.5 and §1.7 |
| §2.2 Non-balanced σ pressure gradient | **Open, quantified** (§3.3) |
| §2.5 `hc` ignored, momentum not Hz-weighted, flux units, constant bottom stress, thin mixing | **All open** (§3.2, §3.6, §3.7) |
| §2.6 Two EOS, 2 m/s internal-wave speed, Ω labelled as w | **Open.** New: the full EOS has a units bug (§3.8) |
| §3.1 Geometry is affine-parallelogram-only | **Open** (§4.1) |
| §3.3 OBC labels (Chapman, TST, Radiation), Chapman `dt` never set | **Open.** `SWE2DRhsConfig::with_dt` has no callers (§4.6) |
| §3.4 Nesting | Layer index and ubar/vbar **fixed**. CF time, rotation, bilinear weights, relaxation zone and stencil performance **open** (§4.5) |
| §3.5 Coastline holes / no spatial index / hardcoded f | **Open** (§4.8) |
| §3.6 Gmsh (MSH 4 claimed, 2.2 parsed; `unwrap`s; triangles dropped) | **Open** |
| §4.1 RHS-by-value API allocations | **Open, and undercounted.** About 33 full-field arrays per production step (§5.3) |
| §4.2–4.5 Burn, serial `Simulation` RHS, duplicated serial/parallel RHS, minor kernels | **All open** |
| §5.3 Legacy integrators and run loops | **Open.** About 2.4k lines still exported |
| §5.4 Error handling | Tide-gauge dates **fixed**. i16-before-f32 read, silent zeros in `get_state` and writer time units **open** (§4.5) |
| §5.5 Harmonic fit | LU on 13×13 normal equations is adequate. Missing record-length guard **open**; new truncated-period bug (§4.3) |

---

## 1. 2D horizontal DG core

### 1.1 [BLOCKER] Not well-balanced for realistic bathymetry [M][S]

**Mechanism.**
- The volume term collocates ∂(½gh²)/∂x with Dr/Ds (`swe_2d.rs:355-383`).
- `BathymetrySource2D` adds the pointwise term −gh ∂B/∂x.
- At lake-at-rest these cancel only if Dr differentiates ½gh² exactly, which needs 2·deg(h) ≤ p. Realistic nodal bathymetry has deg B = p, and linear B at p = 1 already fails.
- Hydrostatic reconstruction does nothing for continuous B.

**Probe (lake at rest)** [M]:
- Domain: 20 km × 20 km, 16 × 16 elements (1.25 km), B = −30 − 370·sin²(πx/L)·sin²(πy/L) (30→400 m).
- Configuration: reconstruction plus `BathymetrySource2D`, as all current well-balanced tests use.
- The 1 h runs use SSP-RK3 at CFL 0.3, reflective walls, no friction, no limiter.

| p | Residual acceleration at t = 0 | Equivalent surface slope | After 1 h |
|---|---|---|---|
| 1 | 0.17 m/s² | 1.7e-2 | blows up after ~30 min |
| 2 | 0.039 m/s² | 4e-3 | max \|u\| 2.3 m/s, η error 2.1 m |
| 3 | 4.9e-3 m/s² | 5e-4 | max \|u\| 0.91 m/s, η error 0.10 m |
| 4 | 2.3e-4 m/s² | 2.3e-5 | not run |

Same probe with linear B (30→400 m across the domain): p = 1 gives 0.07 m/s²; p ≥ 2 gives ~1e-14. **The linear-B, p ≥ 2 case is exactly and only what the well-balanced tests cover** (`swe_2d.rs:1457-1689`).

For scale, M2 surface slopes on the Norwegian shelf are about 1e-6 to 1e-5. Even p = 3 carries a spurious force 50–500× the tidal forcing.

An independent 1D replica [S] (200→50 m Gaussian sill, σ = 1 km, Δx = 500 m, 1 h) gave 2.1 m/s at p = 1, 29 cm/s at p = 2, and 2.7 cm/s at p = 3.

**Fix.** The entropy-stable, well-balanced flux-differencing DGSEM (Wintermeyer et al. 2017, JCP 340):
- Volume term 2Σⱼ Dᵢⱼ F#(qᵢ, qⱼ) + g hᵢ(DB)ᵢ, with a matching interface treatment.
- Exactly well-balanced for any nodal B, including B discontinuous across faces.
- Conservative, and fixes aliasing (§1.8).

The sub-review's replica gives residuals ≤ 3e-11 for p = 1–4. Extend with Wintermeyer et al. (2018, JCP 375) for positivity and wet/dry.

### 1.2 [BLOCKER] Hydrostatic reconstruction breaks mass conservation when B jumps across faces [M][V]

**Code.** The surface term uses `f_int = normal_flux(&q_int_flux, …)` with the *reconstructed* state (`swe_2d.rs:494`, parallel `:1021-1022`). The strong form needs the actual nodal flux F(q⁻) to telescope with the volume term. As written, each element's mass rate gains −∮(h⁻ − h*)u⁻·n.

**Where it is active.** Whenever B jumps across faces:
- `Bathymetry2D::to_cell_average()` (`bathymetry_2d.rs:404`);
- `linearize()` (`:348`);
- the Frøya example (`froya_real_data.rs:306` with `with_well_balanced(true)`).

**Probe** [M]:
- Periodic 20 km, 12 × 12 elements, B = −200 + 150·sin(2πx/L)·cos(2πy/L), η = 0.3 m, u = cos(2πx/L)·cos(2πy/L) m/s.
- Cell-averaged B with reconstruction: d(mass)/dt = −4.4e6 m³/s, relative −5.5e-5 per second, for p = 1, 2, 3.
- Continuous B, or no reconstruction: ~1e-20 relative.
- A spatially uniform u cancels by symmetry. That is why the leak hides in naive tests.

The sub-review [S] estimates a ~0.7 m/h mean-sea-level drift for ±20 m cell steps under a 0.5 m/s coastal current.

**Consequences.**
- Cell-averaged B is also first-order in bathymetry whatever p is.
- Combined with §1.4, it collapses sloping elements to P0.

**Fix (DG form of Audusse et al. 2004 / Xing & Shu 2006).** Use F(q⁻) for all components, F* from the reconstructed states, and add ½g(h*⁻² − h⁻²)n to the momentum components only. The replica conserves mass to 1e-15 and keeps lake-at-rest [S]. This is a stop-gap until §1.1.

### 1.3 [MAJOR] Docs direct users to the unbalanced configuration [M]

The `well_balanced` field and `with_well_balanced` (`swe_2d.rs:50-53,142-153`) say "do NOT include `BathymetrySource2D`". With nodal B this gives 0.57–0.69 m/s² residuals at every p [M]. Every passing test adds the source. The advice holds only for cell-constant B.

### 1.4 [MAJOR] Kuzmin limiter works on h; with cell-average B it drops slopes to P0 every half tide [S]

- **Code:** `limiters/swe_2d.rs:224-244,256-319`, fused parallel `:655-684`; default relaxation 1.0.
- **Mechanism:** with B constant per element, every sloping element is a patch extremum of h. The replica shows **100 % of sloping elements get α = 0 when ∇η·∇B > 0, and 0 % otherwise**. The same α is applied to hu and hv.
- **Consequence:** tidally asymmetric numerical damping, which produces spurious residual currents and overtides.
- **Fix:** limit η (plus velocities or characteristic variables) and rebuild h = η − B. Add a troubled-cell indicator (TVB-M/KXRCF). Limit h only in partially dry cells (Vater, Beisiegel & Behrens 2019).

### 1.5 [MAJOR] Lake-at-rest breaks at every shoreline [S]

- **Code:** the positivity limiter targets h ≥ h_min, not h ≥ 0 (`limiters/swe_2d.rs:81-96`); the wet/dry rescale scales every h in the element (`wetting_drying.rs:494-500`).
- **Replica on a 2 % beach:** momentum residual 0.02–0.15 m²/s², a 1 cm film on dry nodes, and wet nodes lowered by 1–5 cm in every RK stage.
- **Fix:** Zhang–Shu towards h ≥ 0, with a well-balanced wet/dry treatment (Xing, Zhang & Shu 2010; Wintermeyer et al. 2018). Validate with the Thacker bowl.

### 1.6 [MAJOR] Friction

- **Explicit inside every stage** [V]. `ManningFriction2D::explicit_source` is used, and the correct `semi_implicit_update` (`friction.rs:142`) is never called. With SSP-RK3, momentum changes sign once λΔt > 1.6. For n = 0.025 and u = 1 m/s that is h < 1.5 cm at Δt = 1 s, i.e. at every wet/dry front [S].
  - Fix: point-implicit update hu/(1 + Δt·C_f|u|/h) after each stage, or IMEX.
- **`ChezyFriction2D` units** [V]. Documented as −C_D|u|u/h with dimensionless C_D (`friction.rs:200-239`), but the momentum source is −C_D|u|u. The code is 1/h too weak; the same applies in 1D (`swe_1d/friction.rs`). Tests check only the sign.

### 1.7 [MAJOR] Limiter and wet/dry plumbing

- **Parallel fused limiter lacks the dry-element branch** [V] (`limiters/swe_2d.rs:686-703` vs serial `:118-129`). With h_avg < h_min, θ clamps to 0, dry cells keep hu_avg, and a negative mean depth survives. Serial and parallel runs diverge. This is the production path under `parallel` (`ssp_rk3_swe_2d.rs:140`).
- **The high-level `Simulation` API limits once per step, not per stage** [V] (`runner.rs:286-287`, `builder.rs:115-125`). That voids the Zhang–Shu guarantee.
- **Wet/dry momentum damping depends on dt** [S] (`wetting_drying.rs:44,469-473`). It multiplies hu by α(h) every stage, and h_thin = 10·h_min. With Frøya's `H_MIN = 5 m` (`froya_real_data.rs:74`) this erases currents in water shallower than 50 m.
  - Fix: implicit relaxation hu/(1 + Δt/τ(h)) with h_dry ≈ 1e-3 m.
- **The time step does not enforce the positivity bound** [S]. DGSEM Zhang–Shu needs CFL ≤ 0.75 / 0.42 / 0.29 / 0.23 for N = 1–4 in the code's CFL/(2N+1) units, and less on elongated elements. The default flux is Roe (`swe_2d.rs:74`), which is not positivity-preserving (Einfeldt et al. 1991). Use HLL by default for wet/dry runs.

### 1.8 [MEDIUM] Nonlinear aliasing

Nonlinear fluxes are collocated at GLL nodes with no split form, overintegration or filter. The only stabiliser is the limiter, which itself damages the solution (§1.4). Fixed by the same flux-differencing volume term as §1.1 (Gassner 2013; Wintermeyer 2017).

### 1.9 [MINOR]

- Near-dry velocity is not desingularized for h ≥ h_min (`state/swe_2d.rs:51-58`; fluxes use raw hu/h), so dt collapses at fronts. Use Kurganov–Petrova desingularization.
- Built-in Coriolis is silently dropped whenever any trait source term is supplied (`swe_2d.rs:515`, `:1046`).
- Kuzmin checks only the 4 vertex nodes, even at p ≥ 2.
- `norwegian_coast_beta()` uses β = 1.6e-11, the 45°N value; at 60°N it is 1.14e-11 (`coriolis.rs:97-98`).
- The dt formula scales as 1/(2N+1) rather than with the N² GLL spacing.
- The CFL length √detJ ignores anisotropy, so it overestimates dt for long, narrow fjord elements.

### 1.10 Verified correct (do not touch)

- GLL nodes and weights.
- LIFT = M⁻¹EᵀM_f, J⁻¹·sJ scaling, and the sign of the strong-form surface term.
- Affine metrics, outward normals, sJ = L/2, reversed neighbour face-node order.
- Discrete mass conservation via SBP when B is continuous or reconstruction is off [M].
- Roe (wave strengths, Harten–Hyman entropy fix), HLL-Einfeldt with dry-bed speeds, Rusanov.
- The reconstruction formula itself: F* is single-valued, and lake-at-rest is exact for cell-constant B [M].
- Pointwise Coriolis (energy-neutral semi-discretely).
- The Manning formula and its unused semi-implicit update.
- ∇B uses the same Dr/Ds as the volume term.
- SSP-RK3 stage times and per-stage limiting in `ssp_rk3_swe_2d_step_limited`.
- Zhang–Shu θ scaling conserves element mass.

---

## 2. Mode splitting & time integration

### 2.1 [MAJOR] SSP-RK3 wrapped around the barotropic subcycle: first-order and damped [V]

The subcycle runs inside every SSP-RK3 stage (`mode_split.rs:111-134`), and η/ū/v̄ are advanced through the RK combination of "effective rates" (filtered − old)/dt. With F the stage map (one filtered subcycle), one step is

  A = 1/3 + F/2 + F³/6.

Even for a perfect F = e^{iθ}, |A| ≈ 1 − θ²/2, so the scheme is first order in the barotropic mode. With the filter included [S]:

| Mode | dt = 60 s | dt = 300 s |
|---|---|---|
| M2, amplitude lost per period | ≈ 3 % | ≈ 14 % |
| 1-hour fjord seiche, amplitude lost per period | ≈ 30 % | ≈ 83 %, plus a 6 % phase-speed error |

For weakly damped modes the combination blows up once the substep phase θ_b ≳ 0.5–0.6 [S].

### 2.2 [MAJOR] Forward Euler substeps and cost [V]

- The substeps are Forward Euler (`mode_split.rs:377-387`): |1 + iθ_b| > 1 for all θ_b > 0.
- Each stage runs 2·n_bt substeps (the Hann window over [t, t+2dt]), so a baroclinic step costs **6·n_bt 2D RHS evaluations**.
- ROMS needs about (1.3–2)·n_bt generalized forward-backward steps, and FB stays neutral up to θ_b = 2, which allows ~4× larger dt_bt.
- Net: about 10–20× ROMS's barotropic work [S].
- **Fix:** one generalized FB pass per baroclinic step (AB3-AM4; Shchepetkin & McWilliams 2005). Take η/ū out of the RK combination, and give each 3D stage η at its stage time.

The Hann filter itself is correct [V]: Σw = 1, first moment centred at t + dt, symmetric.

### 2.3 [MAJOR] G-term double-counts barotropic advection (and Coriolis) [V]

G = ⟨RHS₃D⟩ (`mode_split.rs:116-117`) includes horizontal advection and Coriolis (`rhs_3d.rs:83,87`). The 2D sub-model flux already contains hu², huv (and a Coriolis source if configured). So ū·∇ū is counted twice in fast straits (Moskstraumen, Saltstraumen), which corrupts M4/M6 and the tidal residuals.

**Fix:** G = ⟨RHS₃D⟩ − RHS₂D(ū) for the terms both models compute (the ROMS `rufrc` construction).

### 2.4 [MAJOR] 3D stresses never reach the depth mean [V]

`Forcing` (surface/bottom stress) is passed only to vertical diffusion (`mode_split.rs:174`). The reconciliation then resets ⟨u⟩ to ū from the 2D sub-model (`:185-195`), which gets no `Forcing`. The comment "2D mode included stress" (`:179`) is false unless the user separately adds matching 2D sources, and nothing keeps the two consistent. In `minimal_3d` wind produces no depth-mean transport: no Ekman transport and no setup.

**Fix:** put (τ_s − τ_b)/(ρ₀D) into G, with τ_b from the bottom-layer 3D velocity.

### 2.5 [MAJOR] Fast mode has no CFL control, no positivity, no wet/dry [S][V]

- dt comes only from a 3D estimate with a hardcoded 2 m/s wave speed (`hydrostatic_3d.rs:311`), and dt_bt = dt/n with fixed n (`mode_split.rs:369`). √(gH) in Sognefjord is ≈ 113 m/s.
  - Fix: n_bt = ⌈dt/dt_bt,max⌉ every step.
- The 2D↔3D conversion divides by an unguarded h (`hydrostatic_3d.rs:238`), and `post_process` (limiter/positivity) is never called in the subcycle. The first dry node gives NaN.
- The subcycle steps the primitive ū rather than transport, and filters D and ū separately, so D_f·ū_f ≠ (Dū)_f.

### 2.6 [MEDIUM]

- **Density frozen across RK stages.** It is updated only before the step (`simulation_3d.rs:150`) and carried through axpy with zero tendency, so the baroclinic PGF is first-order in time.
- **Allocation:** about 64 N₃D-sized arrays per baroclinic step (three `Solution3D::new` + `rhs_slow.clone()` at `mode_split.rs:134`), plus ≥ 9 N₂D-sized arrays per substep × 6·n_bt. At 50k P2 elements × 30 levels × n_bt = 30, roughly 12 GB is allocated and initialised per step [S].
- `step` and `step_with_stage_hook` still duplicate about 100 lines.
- The module docs promise "slow terms once per 3D step" and a ROMS predictor-corrector; neither exists.

### 2.7 Verified correct

- The baroclinic-only PGF selection (P0.6) and its regression test.
- Subcycle and 3D stage times.
- ⟨u⟩ = ū reconciliation at every stage (vertically uniform correction, Δσ weights sum to 1).
- Hz built from the same η as the barotropic mode.
- Total ∫η conserved in closed basins without wet/dry.
- SSP-RK3 coefficients and stage-hook placement.
- The 2D coupled SWE–tracer path uses conservative hT.

---

## 3. 3D vertical physics

### 3.1 [BLOCKER] Tracers are neither constancy-preserving nor conservative [V][S]

- **(a) Missing thickness-change term.** T and S are stepped as concentrations. The RHS divides an inventory tendency by Hz (`advection_3d.rs:428,530,688`) and never includes −C·∂ₜHz/Hz. With the corrected Ω, a constant field obeys ∂ₜC = C·(∂ₜHz)/Hz, i.e. C ∝ D(t).
  - Consequence [S]: a 1 m tide over 20 m of water pumps salinity by ±1.7 psu, corrupting stratification, fronts and the PGF.
- **(b) Ω at the wrong points** [V]. *Fixed in [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2): Ω is stored at the w-points.* Ω is stored at layer centres (`vertical_velocity.rs:222-227`) and re-averaged to faces in the advection kernels (`advection_3d.rs:633,676`). The effective divergence becomes a 1-2-1 smoothing of the one continuity used.
- **(c) 2D/3D flux mismatch is hidden.** The linear correction forcing Ω(0) = 0 (`vertical_velocity.rs:208-220`) masks it:
  - Ω's face flux is Rusanov with |u| + √(gD) dissipation (`:170`);
  - the tracer flux uses |u|;
  - η comes from the 2D HLL/Roe flux via a filtered subcycle of *states*;
  - no time-averaged barotropic transport exists that Σₖ Hz·uₖ could be matched to.
- **Fix (Shchepetkin & McWilliams 2005; ROMS `set_massflux`/`omega.F`; Kärnä et al. 2018):**
  - Accumulate the secondary-weighted barotropic transport (DU_avg2), including DG face fluxes.
  - Correct the per-level Hz-weighted face fluxes so their vertical sum equals it.
  - Diagnose Ω at w-points from those fluxes.
  - Step Hz·C and divide by the new Hz.
  - Ω(0) = 0 then holds exactly and should be *asserted*, not forced.

### 3.2 [BLOCKER] Vertical momentum advection too large by a factor of D [V]

*Status: the vertical term is fixed in [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2) (TODO P0.16). The horizontal Hz-weighting below is still open (TODO P4.2).*

`apply_vertical_advection_field` computes `(flux[l+1] − flux[l]) / d_sigma[l]` with flux = Ω·u (`advection_3d.rs:641`). Ω is in m/s: it integrates div(D·u)·dσ (`vertical_velocity.rs:62-70,203-205`). The divisor must be Hz = D·Δσ, as the tracer version already does (`:688`). The term is 200× too large in 200 m of water.

The horizontal part −∇·(u⊗u) is not Hz-weighted either. That adds a spurious u(u·∇D)/D ≈ 1e-4 m/s² (u = 0.3 m/s, slope 0.25, D = 215 m), about 3× f·u [S].

**Fix:** advect Hz·u (ROMS `step3d_uv`) with Ω at w-points.

### 3.3 [BLOCKER] σ-coordinate pressure-gradient error is as large as the physical signal [S]

**Code:** `baroclinic.rs:137-156,160-201`.
- (a) The half-layer term is first-order. Its along-σ gradient does not vanish even for linear ρ(z): 2.6e-6 m/s² for N² = 1e-4 s⁻² over a 30→400 m slope, at every resolution.
- (b) The two large cancelling terms are discretised inconsistently.
- (c) Pressure and η jumps at element faces are never lifted. A density front sitting on a face exerts no force.

**Fjord case:** 30→400 m over 1.5 km, Δρ = 5 kg/m³ in the upper 20 m, 30 levels, fluid at rest. Spurious |F|:

| Resolution | Spurious \|F\| |
|---|---|
| P1, 500 m | 7.1e-4 m/s² |
| P3, 250 m | 1.0e-4 m/s² |
| P3, 100 m | 7.3e-5 m/s² |

Real estuarine forcing is about 5e-5. The rx1 at GLL spacing is 7.7–40, and there is no rx0/rx1 diagnostic or bathymetry smoothing. The only test uses constant density (`baroclinic.rs:244`).

**Fix:** Shchepetkin & McWilliams (2003) density-Jacobian, or z-level interpolation (Stelling & van Kester 1994; Berntsen 2002). At minimum, subtract a horizontal-mean ρ̄(z), integrate trapezoidally, and use the BR1 lifted gradient. Gate with a linear-N² lake at rest (≤ 1e-12) and a seamount test (Beckmann & Haidvogel 1993).

### 3.4 [MAJOR] Tracer flux leaks through coastal walls [V]

*Status: fixed in [PR #12](https://github.com/EmilLindfors/roms-rs/pull/12) (TODO P0.22).*

P0.7 fixed Ω and momentum. The tracer kernel still sets `u_ext = u_int` at physical boundaries (`advection_3d.rs:489-491`), so heat and salt cross the coastline while the mass flux is zero. The P0.7 test covers only the `reflect_velocity` helper.

### 3.5 [MAJOR] First-order upwind vertical tracer advection [V]

`advection_3d.rs:652-690` is first-order upwind. Numerical diffusivity ≈ ½|W|Hz ≈ 1e-4 to 1e-3 m²/s, above fjord-basin diffusivities of 1e-5 to 1e-4 (Stigebrandt & Aure 1989), so the halocline smears numerically.

**Fix:** 4th-order centred/Akima (ROMS) or TVD/HSIMT (Wu & Zhu 2010).

### 3.6 [MAJOR] Vertical mixing and boundary fluxes [V][S]

- No GLS (NorKyst's k-ε) or KPP.
- Unstable stratification with shear² < 1e-10 falls back to *background* mixing (`vertical_mixing.rs:139-145`), so there is no convective adjustment.
- Bottom stress is a user constant, not quadratic drag, and is divided by a hardcoded 1025 (`vertical_diffusion.rs:89`).
- The surface temperature flux is fed a buoyancy flux (`:122`); it should be Q_net/(ρ₀c_p).
- **Fix:**
  - GLS as a per-column solve reusing the tridiagonal solver (Umlauf & Burchard 2003; Warner et al. 2005).
  - Implicit quadratic drag Cd = [κ/ln(z_b/z₀)]².

### 3.7 [MEDIUM] Vertical grid [S]

- `SongHaidvogelStretching::hc` is stored but unused (`stretching.rs:177-217`), and the transform is Vtransform-1-style with hc = 0 (`sigma.rs:199-201`).
- **`ROMSVstretching4` is not ROMS Vstretching 4** (`stretching.rs:314-338`): it blends sinh/tanh with a weight in [−1, 0] (an extrapolation), and differs from the true Vs4 by 0.25–0.54 in σ.
- With (θs 7, θb 0.4, 30 levels), the bottom layer is 35 % of the column and the top layer is 13 mm at 30 m depth. TODO P0.3's "standard formula" claim is wrong.
- **Fix:** Vtransform 2 + true Vstretching 4. Hz then varies horizontally, so Ω must use ∇·(Hz·u).

### 3.8 [MEDIUM] EOS [V][S]

- `equations::EquationOfState` passes pressure in dbar to the UNESCO secant bulk modulus, which expects bar (`equation_of_state.rs:144-160,281-307`). ρ(5 °C, 35 psu, 1000 dbar) comes out ≈ 1069.5 instead of ≈ 1032.3 [V]. Latent: the 3D path does not use it. Its test only checks monotonicity.
- The `UnescoEOS` wired into the 3D path (`physics/eos.rs:111-126`) drops terms and ignores pressure: −0.14 kg/m³ offset, and ∂ρ/∂T off by −10 % to +24 % over 2–15 °C [S].
- **Fix:** convert p_bar = p_dbar/10; test against the UNESCO check values; delegate the physics trait to it (TEOS-10 later).

### 3.9 [MEDIUM] Missing 3D processes

- No horizontal viscosity or diffusion in 3D (`rhs_3d.rs:89-90`, TODO). The only horizontal dissipation is Rusanov plus an optional along-σ limiter, which mixes across isopycnals on slopes. Needs rotated/geopotential diffusion (Griffies 1998; Lemarié et al. 2012) via BR1.
- No 3D wetting/drying: 1e-12 thickness clamp (`advection_3d.rs:546`), Δt/dz → ∞ (`vertical_diffusion.rs:176`), unguarded 1/h. Any dry node gives NaN.
- `compute_dt` ignores the vertical CFL. The whole 3D path is serial and allocates Vecs in inner loops (§5.3).

### 3.10 Verified correct

- z_w = η + Dσ_w with D = η − B used consistently.
- ∇z|σ = ∇η + σ∇D.
- The baroclinic-only PGF recovers −g∇η for constant density.
- Ω is Hz-weighted and integrated from Ω(−1) = 0, with the linear correction proportional to cumulative thickness.
- Implicit diffusion has correct backward-Euler coefficients on uneven Hz, conserves ΣHz·φ, and applies surface stress with the right sign and units.
- The Thomas solver.
- The Pacanowski–Philander formulas.
- Horizontal tracer DG conserves Hz·C at fixed η.
- The 3D limiter is thickness-weighted.
- The Coriolis sign.
- UNESCO surface density matches the check value 1027.67547.

---

## 4. Mesh, boundaries, tides, nesting, forcing

### 4.1 [BLOCKER] Geometry: coastline-fitted meshes cannot be loaded

*Status (2026-09-26): the per-node isoparametric metrics are done for the 2D kernels (TODO P1.3; general straight-sided quadrilaterals, verified free-stream preserving, well-balanced and conservative). Triangles, MSH 4.1, CSR and the 3D kernels remain.*

- `geometric.rs:82,160-186` still panics on any non-parallelogram quad, and the Jacobian is one constant per element.
- There is no triangle type, and 4 faces are hardcoded in the RHS (`swe_2d.rs:386,908`).
- The Gmsh reader claims MSH 4 but parses 2.2 (`gmsh.rs:133`), has `.unwrap()` on untrusted input (`:254-257`), and silently drops triangles (`:273-275`).
- **Fix:** per-node isoparametric metrics (arrays go `[K]` → `[K × n_nodes]`), triangles (or quad-dominant Gmsh meshes), MSH 4.1, and CSR connectivity. The Wintermeyer formulation (§1.1) is already written for curvilinear meshes.

### 4.2 [BLOCKER] Flather applied twice [V]

**Code.** Each Flather-type BC sets `un_ghost = un_ext + √(g/h)(η_int − η_ext)`:
- `Flather2D` (`boundary_2d.rs:381`) and `HarmonicFlather2D` (`:821`);
- `TSTOBC2D` (`tst_obc.rs:347`);
- `NestingBC2D` (`nesting_bc.rs:189-193`) and `OceanNestingBC2D` (`ocean_nesting.rs:218-222`);
- `ChapmanFlather2D` (`chapman.rs:278`).

The Riemann solver then applies the characteristic relation again (`swe_2d.rs:470-497`).

**Linear analysis.** With w± = u ± √(g/H)η, the incoming invariant becomes w⁻_ext + √(g/H)(η_int − η_ext). Writing η_int = η_i + η_r gives η_r/η_i = −1/3.

**Measured in a 1D nonlinear FV replica** [S]:
- Reflection −0.33 at weight 1 and −0.29 at the nesting weight 0.8 used by Frøya.
- Reflection −0.57 for the `ChapmanFlather2D` defaults (`h_ref = 10`, `chapman.rs:210`) in 100 m of water.
- A progressive tide with the default `u_external = 0` (`boundary_2d.rs:682`) is delivered at **0.65** of its amplitude.

**Fix.** Ghost = external state (η_ext, u_ext); the upwind flux then imposes Flather exactly (Kärnä et al. 2011). `OceanNestingBC2D::with_flather(false)` is already the correct behaviour. Take u_ext from TPXO or NorKyst ubar. Add a pulse-exit reflection test (< 1 %) and a delivered-amplitude test.

### 4.3 [MAJOR] Tidal forcing content

- **Truncated periods** [V]. The `TidalConstituent` constructors (`boundary/tidal.rs:76-113`) use M2 12.42 h, K1 23.93 h, O1 25.82 h, N2 12.66 h, P1 24.07 h. They are used by `HarmonicAnalysis::standard()`/`norwegian_coast()` (`harmonic.rs:163-188`) and the harmonic BCs.
  - 60-day fit [S]: M2 +1.0°, N2 −2.5°, K1 +2.5° (−2 % amplitude), P1 −6.7°.
  - 1-year record: M2 +6°, N2 −16°, K1 +12°.
  - The round-trip tests synthesise with the same truncated periods, so they cannot see this.
  - Fix: use `constituent_period()` (`constituent_reader.rs:156`) or Doodson speeds.
- **Spatially uniform harmonic OBCs.** `HarmonicFlather2D`/`HarmonicTidal2D` ignore (x, y). The M2 phase varies ≈ 30° along a 200 km Norwegian boundary (Kelvin wave). There is no TPXO/FES reader.
- **Nodal f/u frozen at the epoch** rather than the run or record midpoint (≈ 6 % O1 error over a year). No guard against applying corrections twice.

### 4.4 [MAJOR] No single model clock

- Tides use `epoch_jd`.
- Nesting uses raw file time.
- The NetCDF writer stamps "seconds since 1970" on raw simulation time (`netcdf_io.rs:241` vs `:397`).
- `TideGaugeValidationResult::compute` only checks that series lengths match (`tide_gauge.rs:234-239`).

As a result, model and observation phases can be compared on different origins without any warning.

**Fix:** a `ModelClock { epoch_unix }` threaded through BCs, forcing, I/O and validation.

### 4.5 [BLOCKER/MAJOR] NorKyst/ROMS nesting

- **[BLOCKER] No CF time base** [V]. `read_time` returns raw values (`netcdf_io.rs:1226-1236`), and `find_bracket` clamps out-of-range times. Frøya's simulation time starts at 0 while NorKyst time is seconds since 1970, so the boundary freezes at the first snapshot.
- **[MAJOR] Transport not conserved** [V]. `hu = (ζ − B_child)·ubar_parent` (`ocean_nesting.rs:141-145`); the parent `h` is never read. The volume flux is wrong by h_child/h_parent (tens of % on the steep shelf).
  - Fix: ubar_c = ubar_p·h_p/h_c, and blend the child bathymetry to the parent's in the relaxation zone.
- **[MAJOR] z-level files** [S]. `reshape_to_3d` always takes `n_depth − 1` as the surface (`netcdf_io.rs:1389`); for NorKyst ZDEPTHS files index 0 is the surface.
- **[MAJOR] No rotation** [V]. Grid-relative `ubar` is accepted without `angle` rotation (never read).
- **[MAJOR] Bilinear weights** assume axis-aligned cells (`:1596-1597`).
- **[MAJOR] `"h"` (bathymetry) is in the SSH candidate list** [V] (`:1068`).
- **[MAJOR] i16 read before f32** [V] (`:1258`). Unpacked float fields are truncated to integers.
- **[MAJOR] No flow-relaxation zone** (Davies 1976 / Martinsen–Engedahl, which NorKyst uses). The sponge supports only rectangular distance functions.
- **[MEDIUM] Off-by-one for descending coordinates** [S] (`find_bracket` `:1762-1766` + `get_state` `:1443`).
- **[MEDIUM] Silent zeros and slow stencils.** `get_state` silently zeroes missing fields. Parent data is held as `Vec<Vec<Vec<f32>>>`, and stencils are recomputed every stage behind an RwLock.

### 4.6 [MEDIUM] OBC labels and wiring

- "Chapman" (`chapman.rs:129-137`) is relaxation toward external η, not radiation.
- `SWE2DRhsConfig::with_dt` has **no callers** [V], so Chapman always runs with cfl = 1 and α = 0.5.
- TST's "subtidal" term is only c·Δη/dx (≈ 6 % of Flather), with no frequency separation.
- `Radiation2D` is driven by h rather than η, so it is bathymetry-blind. Orlanski is named but not implemented. (Fixed since: `Radiation2D` is η-referenced, and its ghost is the external state rather than the interior, which it used to return for all subcritical flow; an outgoing pulse reflected ~10 % and the channel grew to ~50× the pulse amplitude.)

### 4.7 [MEDIUM] Atmospheric forcing

- `ForcingReader` handles only 1D lat/lon (ERA5-style), indexes time without CF parsing, and is not wired to `WindStress2D`/`AtmosphericPressure2D`. MET Nordic/MEPS/AROME-Arctic files (Lambert grid, 2D lat/lon, grid-relative winds) cannot be read.
- The inverse barometer is unused by open BCs (1 cm/hPa mismatch in storms).
- Large–Pond Cd is uncapped.

### 4.8 [MEDIUM] Bathymetry, coastline, projection, Coriolis

- **GeoTIFF:** assumes lon/lat tiepoints and never checks the CRS (`geotiff.rs:99-113`), so a UTM33 grid silently gives B = 0. It interpolates on pixel edges and point-samples nodes (no L2 projection).
- **Land/nodata** nodes get B = 0 (`bathymetry_2d.rs:133`). Frøya then initialises depth as max(−B, 5 m), so land starts with 5 m of water. The land mask is not used by any solver path.
- **Smoothing:** no ROMS rx0/rx1 bathymetry smoothing (needed for §3.3).
- **Coastline:** GSHHS holes are dropped, with no spatial index (`coastline.rs:81-95`).
- **Coriolis:** f is hardcoded; β is wrong (§1.9); there is no f(latitude) helper.

### 4.9 [MEDIUM] Rivers

- `RiverTracerSource` nudges tracers and adds no volume (`surface_flux.rs:258-320`).
- `Discharge2D` imposes flow only weakly (the Rusanov flux carries q/2 into still water).
- There is no NVE or ROMS river-file reader, so NorKyst's ~1760 rivers cannot be represented. Minor for barotropic tides, major for 3D.

### 4.10 [MINOR]

- The tidal-potential sign is reversed relative to its own documentation (`source/swe_2d/tidal.rs:15-16,299-305`); the magnitude is under 5 mm across 200 km. It uses a Love factor of 0.69 for diurnals, where K1 should be ≈ 0.736.
- The 1D `TidalBC` Flather term is missing a factor √g.
- `harmonic.rs::fit` has no record-length/Rayleigh guard and no P1-from-K1 inference.
- Reachable panics: `timeseries_reader.rs:198` (NaN time), `geotiff.rs:239-245`, analysis asserts.
- Library `eprintln!` in `tide_gauge.rs`.

### 4.11 Verified correct

- Schureman/Pugh f/u for M2/N2, K1, O1/Q1, K2, Mf, Mm.
- Doodson arguments, including the diurnal ±90° offsets.
- Meeus mean longitudes and the JD conversion; S2 V₀ = 30°·UT.
- f·A·cos(ωt + V₀ + u − G) applied consistently and inverted correctly in `reference_constants`.
- 2D Flather details apart from §4.2: sign relative to the outward normal, √(g/h), (un, ut) rotation.
- RK stage times passed to BCs.
- The UTM (Snyder) formulas and `LocalProjection` radii.
- Wind drag formulas and the −h∇p/ρ pressure source.
- `datetime.rs` (exact Gregorian, UTC offsets).
- The NorKyst parquet nearest-cell search.
- s_rho surface selection and ubar preference.
- The harmonic fit with exact periods.

---

## 5. Performance & "faster than ROMS"

### 5.1 Status of the prior §4 items

All open:
- the RHS-by-value API (`traits.rs:73`, `integrator.rs:139-141`);
- Burn (surface term missing, host round-trips; `Cargo.toml:38` disables fusion for f64);
- the `Simulation` API running the serial RHS (`builder.rs:99`);
- two independent RHS implementations (`swe_2d.rs:318-563` serial, `:775-1110` parallel);
- the faer threshold (`kernels.rs:441`);
- BR1 viscosity buffers;
- no SIMD Manning.

The benches do time a full RHS over a mesh, but not a representative one (§5.3).

### 5.2 Cost model (estimate, not a benchmark) [S]

**Assumptions:**
- Sustained rates: 2 GF/s and 5 GB/s per core for both codes.
- ROMS 2D: ~100 core-ns per point per fast step (±2×).
- Domain: NorKyst 2600 × 900, water fraction 0.55.
- SSP-RK3 CFL 0.209 (P2) / 0.130 (P3), run at 90 %; ROMS forward-backward Courant 0.9.
- To match ROMS accuracy, DG DOF spacing can be α = 1.25 (P2) or 1.75 (P3) × ROMS Δx. **This is an assumption**; cost scales as α⁻³.

**Current kernels:** 503 / 570 / 724 / 941 flops per node per RHS at P1–P4. Optimised: ~210–250. About 620 B per node per stage today vs ~130 B fused.

**Calibration:** the model gives 285 core-ns per node-stage at P2, against 362 (PROFILE.md) and 848 (PERFORMANCE.md) for the same Frøya configuration. The two docs disagree with each other.

2D tidal run, NorKyst-equivalent domain:

| Configuration | Core-hours per simulated day | vs ROMS 2D |
|---|---|---|
| ROMS 2D | 1.2 | 1× |
| DG P2 as configured (CFL 0.1, current kernels, land meshed) | 59 | **~47×** |
| P2, CFL near the stability limit, water-only mesh, current kernels | 6.9 | 5.5× |
| + optimised kernels | 2.6 | 2.1× |
| P3, all of the above | 1.1 | **~0.9×** (2.4× at α = 1.25, 0.6× at α = 2) |

- **Structural DG penalty:** at equal DOF spacing explicit DG needs a 2–2.5× smaller dt and 3 RHS evaluations per step, so about 6–7× more evaluations than ROMS. It is recovered only through accuracy per DOF (α³) and water-only meshing.
- **Fjord refinement:** a single 500 m (250 m) element in 1300 m water forces 11× (22×) more global steps without local time stepping.
- **3D:** best case ≈ 0.5–1.5× ROMS core-hours, and only with local time stepping or an implicit free surface. On fjord-refined meshes without them, 5–20× worse.
- **GPU:** fused f64 kernels on an H100 at ~126 B per node-stage would be worth roughly 10³ optimised cores at the bandwidth ceiling, realistically 3–5× less. Burn without fusion moves ~1.6 KB per node-stage (13× more).

### 5.3 Findings

1. **[CRITICAL] Time step 5–10× smaller than necessary** [V].
   - Frøya uses CFL 0.1 (`froya_real_data.rs:163`), where ~0.5 is stable in the code's units.
   - `compute_dt_swe_2d` pairs the global minimum element size with the global maximum wave speed (`swe_2d.rs:582-602`), up to ~11× too small on graded meshes.
   - The √detJ size overestimates anisotropic elements (a stability risk).
   - Frøya calls the serial dt every step (`:565`).
2. **[MAJOR] The "parallel" limiter and wet/dry paths allocate and finish serially.** `to_vec` of all fields, 3 Vecs per element, a collected `Vec<Vec>`, then a serial copy-back (`limiters/swe_2d.rs:626,637,710-725`, `wetting_drying.rs:526-541`). This runs after every stage. At 5k P3 elements, 36 % of step time is outside the RHS, and 17–28 % of time is system time. **1.5–2.5×.**
3. **[MAJOR] Dense, duplicated volume term.**
   - 4 dense (p+1)⁴ matrix-vector products per element (`swe_2d.rs:835-874`) instead of sum-factorised contractions on contravariant fluxes.
   - LIFT is applied densely (`kernels.rs:490-517`) although the GLL mass is diagonal.
   - Every interior Riemann problem is solved twice (per-element face loop, `:908`).
   - **2.6× fewer flops at P2, 3.5× at P3**, and removes the (p+1)⁴ growth that makes P3/P4 look unattractive.
4. **[MAJOR] Land elements computed.** 41 % of Frøya's elements are land (38,212 of 65,025 are wet), and the land mask is used only for diagnostics. **~1.7×.**
5. **[MAJOR] Allocation per step.** The production stepper `ssp_rk3_swe_2d_step_limited` (`ssp_rk3_swe_2d.rs:185-228`) clones and allocates state plus 3 fresh RHS outputs. With limiter and wet/dry, that is **33 full-field arrays (~155 MB at 65k P2) and ~1.2M small Vecs per step**. TODO's "RHS allocations: Zero ✓" covers only the element loop.
6. **[MAJOR] 3D path entirely serial, with Vecs allocated per face per level** (`advection_3d.rs:261-297,444-496`; `vertical_velocity.rs:109-157`): ~186M malloc/free per 3D RHS at 50k × 30.
7. **[MEDIUM] Benchmarks can't support a speed claim.**
   - The mesh bench stops at 10k P3 elements (fits in cache), with uniform flow, walls, no bathymetry and no limiter.
   - The step bench covers 64–256 elements, serial.
   - PERFORMANCE.md cites RHS variants that no longer exist.
   - The same 65k run is recorded as 70.5 s in one document and 217 s in the other.
   - There is no ROMS timing anywhere.
8. **[MINOR]**
   - `batched.rs` is unused.
   - `reference_to_physical` and `dyn` source dispatch run per node per RHS.
   - No mesh reordering (RCM/Hilbert).
   - No MPI or partitioning.

### 5.4 Top speedups (expected factor)

| # | Change | Expected factor |
|---|---|---|
| 1 | Per-element directional dt, run near the stability limit; optionally SSPRK(4,3) or low-storage RK (+1.3–1.5×) | **5–10×** |
| 2 | Local time stepping or implicit/semi-implicit free surface for tides (SLIM/Thetis practice) | **2–20×**, mesh-dependent |
| 3 | Sum-factorised contravariant volume term, diagonal LIFT, one Riemann solve per face | **2.5–3.5×** RHS |
| 4 | `compute_rhs_into` + fused in-place stepper with limiter/positivity/wet-dry in one parallel pass | **1.5–2.5×** |
| 5 | Water-only mesh or active-element set; limit only troubled cells | **~1.7×** |

Beyond these: custom fused f64 GPU kernels (CubeCL/cudarc over CSR faces, not Burn tensor ops with fusion off) and MPI are what would win on wall-clock time.

### 5.5 Judgement

- "Faster than ROMS" is plausible **per unit of tidal accuracy, for 2D barotropic tides, in fjord-rich domains**, at P3+. That requires optimised kernels, dt near the limit, a water-only unstructured mesh, and local time stepping or an implicit free surface.
- Even then the core-hour gain is 1–2.5× and rests on the unproven α ≈ 1.75–2.
- Wall-clock wins need GPU or MPI.
- For 3D the claim is far off.

**Settle it with a Pareto benchmark:** M2/S2/K1/O1 error at tide gauges vs core-hours on the same hardware, comparing ROMS 2D at 800/400/200 m with DG P1–P4.

---

## 6. Tests, structure, and documentation accuracy

### 6.1 Tests that would pass on broken physics

- **Well-balanced tests** (`swe_2d.rs:1457-1689`): all continuous *linear* B at p ≥ 2, the one configuration that balances (§1.1).
- **`tests/swe_2d_test.rs::test_lake_at_rest`:** flat bottom.
- **`test_geostrophic_balance`** (`tests/validation_2d.rs:205-283`): linear η on a periodic mesh (0.1 m jump at the wrap) with a tolerance of 0.49 m²/s² against balance terms of ~1e-3. It passes with Coriolis removed or sign-flipped [S].
- **SWE convergence** (`tests/convergence_test.rs:666-809`): linearised (ε = 1e-4), flat, periodic, Cartesian, with pass thresholds 1.5 (P1) and 2.5 (P2) [V], below the N+1 CLAUDE.md requires.
- **"Multi-day stability":** runs 316 s. `wet_dry_test.rs` has only static unit tests.
- **Parallel-equivalence test:** excludes bathymetry, reconstruction, boundaries and limiters.
- **Mode-split seiche** (`simulation_3d.rs:222-332`): P1, flat, unstratified, non-rotating, one zero crossing within 15 %, amplitude unchecked (it loses 23 % per period). The filter test proves only the weight moments.
- **3D:** the PGF tests use constant density; tracer tests use D = 1 m or fixed η; the EOS and stretching tests check only monotonicity and bounds; there is nothing in `tests/` for 3D.
- **Harmonic round-trips:** synthesise with the same truncated periods (§4.3).

**Missing tests (most would fail today):**
- **2D:**
  - lake-at-rest with steep nodal and face-discontinuous B at p = 1–4, serial and parallel;
  - mass conservation with discontinuous B and slope-correlated flow;
  - Thacker parabolic bowl;
  - outgoing-pulse reflection < 1 % and delivered tidal amplitude;
  - harmonic fit of a true-frequency signal.
- **Mode splitting and 3D:**
  - temporal convergence of the coupled scheme;
  - 100-period barotropic energy decay;
  - constant-T preservation under tide over a sloping bed;
  - Hz·T inventory conservation;
  - wind setup = τ/(ρgD) and Ekman transport τ/(ρ₀f);
  - stratified lake-at-rest (linear N² ≤ 1e-12; tanh pycnocline over a seamount);
  - lock exchange;
  - mode-1 internal-wave speed;
  - UNESCO check values.

### 6.2 Structural debt (still open from the prior review)

- About 2.4k lines of legacy integrators still exported: `ssp_rk3*.rs`, three `coupled_swe_tracer` variants, `burn_ssp_rk3.rs`. `Simulation3D` still has its own time loop.
- Serial and parallel 2D RHS are separate ~300-line implementations. Serial is the test reference while production runs parallel.
- Duplicate config enums (flux types; three limiter enums).
- ~200 root re-exports; `Simulation3D` is not exported.
- `analysis/` has no error type.
- `SimulationResult` is stringly-typed.
- Crate `dg-rs` vs repo `roms-rs`. Many root markdown files.
- Burn: either implement the surface term plus a CPU-equivalence test, or delete it. Its design cannot reach the §5.2 GPU numbers.

### 6.3 Documentation that overstates status

- **TODO "Key Metrics":**
  - "Math correctness 30/30" is not supported by the evidence here.
  - "RHS allocations in hot path: Zero ✓" is true only for the element loop (§5.3).
  - "SWE convergence verified" rests on thresholds below N+1.
- **TODO "2D solver is research-ready":** not on realistic bathymetry or open boundaries (§1, §4).
- **TODO P0.3:** `ROMSVstretching4` "implementing standard Shchepetkin & McWilliams (2005)" is wrong (§3.7).
- **`linearize()` doc:** "linear B is well-balanced" is false at p = 1 [M].
- **`with_well_balanced` docs:** §1.3.

---

## 7. Recommended order of work

**Week-scale: the confirmed small bugs** (top table, 1–10). Each needs a regression test that fails first.

**Phase A: a correct 2D barotropic tide model.** This is the product where DG can beat ROMS.
1. `compute_rhs_into` + workspace stepper; unify serial/parallel into one per-element kernel. Do this first, so the rewrites below are written once in the non-allocating shape.
2. Entropy-stable, well-balanced flux-differencing DGSEM (Wintermeyer 2017), positivity/wet-dry (2018), η-based limiting with a troubled-cell indicator, implicit friction. Gated by the §6.1 2D tests.
3. Per-node isoparametric metrics, triangles/MSH 4.1, CSR connectivity.
4. Boundaries: ghost = external state; spatially varying tidal forcing (TPXO/FES reader or NorKyst boundary harmonics); `ModelClock`; NorKyst nesting with CF time, rotation, transport conservation and a Davies/Martinsen–Engedahl relaxation zone.

**Phase B: cheap enough to matter.**
1. Directional per-element dt; sum factorisation; one Riemann solve per face; water-only meshes.
2. Local time stepping or implicit free surface.
3. Full-RHS / full-step / scaling benchmarks at realistic size in CI.

**Phase C: validation and the speed claim.**
1. M2/S2/K1/O1 against Kartverket gauges and NorKyst-800 (TODO P1.5 infrastructure exists).
2. Cost-vs-error Pareto benchmark against ROMS 2D on the same hardware (§5.5).

**Phase D: 3D rebuilt on the ROMS recipe.**
1. One generalized-FB barotropic pass per step, with secondary-weighted transport (DU_avg2).
2. Hz-weighted fluxes corrected to the barotropic transport, Ω at w-points, inventory-form tracers and momentum.
3. A `rufrc`-style G-term including stresses; quadratic bottom drag.
4. Balanced PGF (Shchepetkin & McWilliams 2003) gated by a stratified lake-at-rest test; rx0/rx1 smoothing; Vtransform 2 + true Vstretching 4.
5. GLS k-ε; higher-order vertical advection; rotated horizontal diffusion; 3D wet/dry; parallel 3D kernels.
6. Idealised Sognefjord, lock exchange and internal-wave tests before any NorKyst 3D comparison.

**Deferred:** GPU (custom fused f64 kernels over CSR, not Burn), MPI, and the `dg-core`/`roms-rs` workspace split.

---

## References

- Audusse, Bouchut, Bristeau, Klein & Perthame (2004), hydrostatic reconstruction.
- Beckmann & Haidvogel (1993), seamount PGF test.
- Berntsen (2002), internal pressure errors in σ-models.
- Davies (1976), flow-relaxation open boundaries.
- Delestre et al. (2012), limitations of hydrostatic reconstruction on steep slopes.
- Einfeldt, Munz, Roe & Sjögreen (1991), positivity of Godunov-type schemes.
- Flather (1976); Kärnä et al. (2011), weak Flather in DG.
- Gassner (2013), split-form DGSEM.
- Griffies (1998); Lemarié et al. (2012), isoneutral/rotated diffusion.
- Haney (1991), hydrostatic consistency.
- Kärnä et al. (2018), Thetis 3D (DG + mode splitting).
- Shchepetkin & McWilliams (2003), density-Jacobian PGF; (2005), ROMS split-explicit stepping and barotropic filters.
- Stelling & van Kester (1994), z-interpolated PGF.
- Stigebrandt & Aure (1989), fjord basin mixing.
- Umlauf & Burchard (2003); Warner et al. (2005, 2008), GLS and ROMS bottom boundary layer.
- Vater, Beisiegel & Behrens (2019), wet/dry limiting for DG SWE.
- Wintermeyer, Winters, Gassner & Kopriva (2017), entropy-stable well-balanced DGSEM for SWE; Wintermeyer et al. (2018), positivity and wet/dry extension.
- Wu & Zhu (2010), HSIMT vertical advection.
- Xing & Shu (2006); Xing, Zhang & Shu (2010), well-balanced positivity-preserving DG for SWE.
- Zhang & Shu (2010), positivity-preserving limiters.
