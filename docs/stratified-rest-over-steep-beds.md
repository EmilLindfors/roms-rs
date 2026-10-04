# A stratified rest state over steep beds in the 3D model

Literature and analysis for TODO P1.3's open blocker: a stratified fluid at
rest does not stay at rest where σ-levels cross the pycnocline inside an
element (cliffs, steep shores). The experiments and their numbers are in
TODO.md P1.3; this note collects what the literature says, why each cure
tried so far fails, and what a research-grade cure would have to do.
Written 2026-10-04.

## The problem in one paragraph

The 3D PGF is a sum over node pairs (σ-pairs, `solver::rhs::baroclinic`).
It is the exact discrete adjoint of the split-form tracer advection, so the
semi-discrete scheme conserves `KE + Σ gρz H_z` exactly. With the reference
profile subtracted, the force at rest is round-off. Yet round-off grows:
e-folding 15–17 min on the Frøya bed and ≈ 38–45 min in the x–z slice of
`a_pycnocline_over_a_cliff_stays_at_rest` (15 → 300 m inside one 1 km P2
element). A linear profile is stable.

## What the literature says

**The symptom is known, and it is an instability, not only an error.**

- Beckmann & Haidvogel (1993, JPO 23, doi:10.1175/1520-0485(1993)023<1736:NSOFAA>2.0.CO;2) found σ-models unstable over a wider range of parameters once forced. Their z-interpolated "corrected gradient" had far smaller errors but "a much more restrictive range of stable behavior". The constant-depth form here behaves the same way: 1e-2 m/s at 2 h, then NaN.
- Zängl (2012, MWR 140, doi:10.1175/MWR-D-12-00049.1) shows terrain-following PGFs are "prone to numerical instability if the height difference between adjacent grid points is much larger than the vertical layer spacing". That is the slice exactly: the mid levels sit at 2–3 m in the 15 m column and 40–130 m in the 300 m one.
- Haney (1991, JPO 21) states the consistency condition (rx1): a level at one point must lie between the neighbouring levels at the adjacent point. Mellor, Oey & Ezer (1998, JAOT 15) find the errors decay in prognostic runs with zero diffusivity. That held for their finite-difference schemes, and the slices here show it does not hold for an energy-exact non-dissipative one.
- Lemarié et al. (2012, Ocean Modelling 42, doi:10.1016/j.ocemod.2011.11.007) describe the feedback in which mixing along σ turns PGF-driven flow into real horizontal density gradients.

**Exact energy is not enough; a Casimir is needed.**

- Shepherd (1993, Atmosphere-Ocean 31, doi:10.1080/07055900.1993.9649460): "energy conservation alone is insufficient". Rest is stable when energy plus Casimirs (the pseudo-energy) has a minimum there. The non-kinetic part is the available potential energy.
- Holliday & McIntyre (1981, JFM 107, doi:10.1017/S0022112081001742) give the exact APE density. Its small-amplitude form, `½ g ρ′²/|∂_z ρ_r|`, has a weight `1/N²` that blows up in weakly stratified water.
- Ricardo, Duru & Lee (2024, SISC 46, doi:10.1137/24M1638938) report the same failure in thermal shallow water. Energy-bounded schemes "tend to exhibit numerical instability over relatively short time frames". Their cure is to also conserve, or upwind-dissipate, the buoyancy variance (a Casimir).
- Bell, Peixoto & Thuburn (2017, QJRMS 143, doi:10.1002/qj.2950) found energy- and enstrophy-conserving Boussinesq schemes linearly unstable. They prove the fixed versions stable by showing the linearised matrix is Hermitian under a positive weighting. That is the analysis tool for this problem.
- Tadmor (1987, 2003): for a scalar with convex entropy `C`, the entropy-conservative two-point flux is unique. With `C′(ρ) = −g Z_r(ρ)` (the APE Casimir) it is exactly the prototype's pair density `ρ*`. Upwind fluxes are entropy-stable for every convex `C` at once.

**What operational and DG models do.**

- NorKyst-800 (Albretsen et al. 2011, Fisken og Havet 2/2011): bed smoothed to rx0 ≈ 0.34 with a Haney number ≈ 6. They use θs = 8 and Tcline = 10 m, keeping levels near the surface "so less smoothing is necessary", and quote Shchepetkin: Haney < 3 is safe, > 8–10 is "insane".
- Norkyst v3 (Christensen et al. 2026, GMD 19, doi:10.5194/gmd-19-2785-2026): Laplacian smoothing and a 10 m minimum depth. "Volume-conserving smoothing methods … made it more difficult to control pressure gradient errors."
- Thetis (Kärnä et al. 2018, GMD 11, doi:10.5194/gmd-11-4359-2018): a DG σ-like model. A Lax–Friedrichs velocity penalty "is required to stabilize the internal pressure gradient", the baroclinic head is P2, and there is no 3D wetting and drying.
- SCHISM LSC² (Zhang et al. 2015, 2016, Ocean Modelling 85/102): per-node local σ grids with shaved cells, a z-based baroclinic gradient (cubic spline, constant extrapolation at the bed), and no bed smoothing. That claim comes from the SCHISM docs, not checked in the papers.
- Multi-envelope coordinates (Bruciaferri, Shapiro & Wobus 2018, Ocean Dynamics 68; Bruciaferri et al. 2024, JAMES 16; Wise et al. 2022, Ocean Modelling 170): levels follow smoothed "virtual bottoms", and the bed itself is not smoothed.
- ROMS wetting and drying (Warner et al. 2013, C&G 58): fluxes into dry cells are blocked, and the baroclinic PGF is masked on the same faces (`umask_wet` in `prsgrd3*.h`). The mass flux and the force are switched off together.

## Analysis: why each cure tried fails

**1. With pairs by level index, a stable rest state forces the pair density's stiffness.** Linearise about rest. The perturbation transport `δQ` advects the background, and the force responds to `δρ` through the pair's density `ρ̂`. The energy identity pairs the force with the advection for any symmetric `ρ̂`. A positive quadratic pseudo-energy `KE + ½ Σ m_i w_i δρ_i²` is then conserved only if each node's weight times the pair's background advection matches the pair's `∂ρ̂/∂ρ_i`. The vertical pairs of a resolved column pin `w_i` to `g/G_i`, the APE weight (`G = |∂_z ρ_r|`). For a straddling pair this forces `∂ρ̂/∂ρ_i ≈ (ρ_i − ρ̂)/(G_i Δz) ≈ chord/(2 G_i)`, which is what `ρ*` does.

Making the flux entropy-stable instead of entropy-conservative does not avoid this. At rest `Q = 0`, a flux continuous in `Q` must equal `ρ*` for both flow directions, so its linearisation about rest is `ρ*`'s. Only a flux discontinuous in `Q` (upwind) escapes, and at the PGF it acts as Coulomb friction of magnitude `g Δρ Δz/(2ρ₀Δx)`, ≈ 3e-3 m/s² on the slice's cliff.

The stiffness is large in the slice. For each σ-level, the ratio of the pair's chord to the stratification at the two nodes (`z` at the 15 m and 300 m columns):

| level | z (15 m) | z (300 m) | chord/G shallow | chord/G deep |
|---|---|---|---|---|
| 0 (bed) | −13.4 | −268 | 0.03 | 14.5 |
| 4 | −5.3 | −105 | 2.5 | 51 |
| 9 | −1.7 | −34 | 37 | 131 |
| 12 | −0.84 | −16.9 | 72 | 0.4 |
| 16 | −0.26 | −5.2 | 3.0 | 0.4 |
| 19 (top) | −0.03 | −0.56 | 1.1 | 0.9 |

Half the levels have a ratio above 10 at one node. The regularised reference (a floor on `G`) caps the ratio by making the Casimir inexact where the real water is weaker stratified than the floor. That is why 0.05 kg/m⁴ brought back the old growth at element 335.

**2. Diffusion of the anomaly about the reference can feed the APE.** A tracer tendency `T` changes the pseudo-energy at the rate `g Σ m (ρ′/G) T`. For `T = κ L ρ′` with `L` a conservative Laplacian this is not sign-definite when `G` differs across a pair. With two nodes at `G` = 0.25 and 4e-4 kg/m⁴, `sym(G⁻¹L)` has eigenvalues −3020 and +516. Hence "slows the growth, does not stop it". Diffusing the displacement `Z_r(ρ) − z` instead is provably dissipative, conservative and zero at rest, but it carries the same `1/G` weight and so the same stiffness.

**3. Stretching cannot remove the straddles at a shore.** Every level of a column shallower than the pycnocline pairs with levels of a deeper neighbour in the same element, under any σ-like map with the same level count per column. ROMS-style `hc` hybrids (unused in `SigmaGrid` today) and multi-envelope levels reduce the interior straddles only. Only pairing by depth (z-levels, shaved cells, LSC²) avoids them. Under mode splitting that cannot block the deep layers' flow toward the shallow node either: the layer transports must sum to the 2D pair flux, which knows no cliff.

**4. Shoreline elements leak through the wet–dry pairs.** See the experiments below. A dry node holds no volume, so the wet node's split-form exchange with it ends up in the wet column's own Ω. Its pressure work is exact only for linear profiles, and no PGF term on the horizontal pair can be its partner. ROMS's cure is to block the flux and the force together, which here would reach into the 2D `WetDry` mass flux as well.

## Experiments, 2026-10-04

The pair density (`exp/pair-density-wet-dry`, `DGRS_EC=1`) at time steps small enough for its stiffness, with GLS on the Frøya 1 km grid (20 levels):

- `thin=2`, 8 s step, GLS and the limiter: 3.4e-3 m/s at 0.26 h, 0.32 m/s at 0.5 h.
- The same without the limiter: 0.07 m/s at 0.38 h, 0.70 m/s at 0.63 h.
- Both grow in shoreline elements with dry nodes beside 80–135 m ones (553, 393: depths 3.8, 2.5, 0.2, 16.7, 0.5, 0.0, 80, 135, 82 m). The step was not the problem: constant mixing at 36 s held 6 h before the same elements grew.
- Even at the start, the limiter changes the rest state's force from 1e-9 to 3e-6 m/s².
- `thin=0.1`, 4 s step, GLS and the limiter: 4.2e-3 m/s at 0.19 h, 0.16 m/s at 0.31 h, again in a shoreline element with dry nodes (77: depths 0.0, 0.2, 0.0, 3.3, 0.0, 0.0, 13.4, 43, 1.3 m).
- So with GLS the pair density makes the shorelines far worse than the unchanged scheme, which reaches 6e-4 m/s at 0.7 h and grows first at a fully wet cliff (element 335). It cures fully wet cliffs, but not the bed Frøya has.

A pressure-work partner for wet–dry pairs (`DGRS_DRYBANK=1`: the σ-pairs difference with the wet column's density, matching the wet-side rule) makes the cliff-to-land slice worse. The e-folding drops from ≈ 5.8 h to ≈ 3.1 h (7.4e-11 → 1.7e-9 m/s over 6–24 h against 1.8e-10 → 5.8e-8).

## What a research-grade cure needs

Three independent pieces, none sufficient alone:

1. **A stability proof tool.** Linearise the slice's step map about rest (finite differences, ≈ 6000 unknowns) and take its spectrum, as Bell, Peixoto & Thuburn do. This gives every unstable mode, its growth and its location, and the stiffness (largest frequency) of a candidate, in seconds instead of 6 h runs.
2. **Wet–dry pairs consistent in 2D and 3D.** Block the wet node's exchange with a dry node in the layer continuity, the tracer advection and the PGF together (ROMS `umask_wet`). The 2D `WetDry` mass flux must block the same pairs, or the 3D layers cannot sum to it. This is the blocker on Frøya, at any time step.
3. **The pair density with its stiff part implicit.** The stiff coupling is local to one element and one level (the pair term of the force and of the advection): for P2, 9 nodes × (u, v, ρ) per element and level. That makes a linearly implicit (Rosenbrock/W-method or IMEX) stage solve of a 27 × 27 system per element-level feasible. It needs a reference profile, though, so it does not carry over to evolving stratifications (rivers, nesting, GLS mixing layers) without updating the Casimir.

The alternatives that avoid the problem do so by geometry: z-pairs or shaved cells as in SCHISM's LSC², or envelopes. They are a larger change here because the 2D/3D mode splitting pairs layer transports with the barotropic pair flux.

## Follow-up 2026-10-04: wetting and drying made consistent, and smoothing

Piece 2 is done, and it did not need the 2D kernel to change. Its wet/dry subcells already block a dry bank: Audusse et al.'s hydrostatic reconstruction gives `h* = 0` there. The 3D layers of those elements ignored that and moved their volume with the DG pairs. Now:

- The 2D kernel reports the mass flux through every subcell interface, and the mode splitter averages it like the face fluxes.
- Elements that took the subcells throughout the pass move their layers through the same interfaces (`Δσ_l F̄` plus the central baroclinic part between columns that are not thin), so the layers sum to the 2D update node by node.
- The pressure gradient's volume term in such an element is the subcell interfaces' central difference, the adjoint of that divergence.
- A thin column exchanges no baroclinic transport across faces either.
- 2D takes its subcells in exactly the elements where 3D has a thin column.

Which form the advection takes on the subcell interfaces matters. Upwind moved a steep shore's linear stratification by 8e-3 °C, the diapycnal mixing of diffusion along steep σ-levels (Marchesiello et al. 2009). Central keeps it to 5e-4 °C and is the PGF's energy partner.

With the shores consistent, the bed smoothing needs no shore rule. Pairs of columns that are not thin are bounded per element (r_x0 ≤ 0.2 below 1.5 × the pycnocline's bottom), volume-preserving, and the coastline does not move. Slices hold for a day at ≈ 2e-10 m/s: the 15 → 300 m cliff, 0.5 m beside 150 m, and land beside 300 m. Before, land beside 300 m reached 1.3e-3 m/s in 48 h unsmoothed.

Frøya at rest (20 levels, GLS and the limiter): on the 1 km grid, constant mixing without the limiter holds 5.8e-10 m/s for 24 h. With GLS and the limiter the speed creeps up, to 1.1e-3 m/s at 8 h and 1.1e-2 at 24 h, at the bed of shallow elements: mixing at the slopes, as the control shows. The coastline mesh with GLS: about 1e-3 m/s per hour for 1.5 h, then 8e-3 m/s at 2 h in a shore element at a wetting front. Before, it reached 0.16 m/s within 15 min.

The smoothing is still heavy on the coastline mesh: 23k of ≈ 50k nodes change, by up to 210 m. The ratio bound chains outward from every steep coast element, about a kilometre at 200 m elements. Pieces 1 and 3 (a spectrum harness, and an implicit pair density) remain the route to a lighter bound.

### Follow-up: baroclinic exchange inside shore elements

Longer runs on the coastline mesh (constant mixing, no limiter) showed a slower instability. Shore element 8722 grew with an e-folding of 38 min from 3.7 h on. It holds 28.5 and 16.6 m columns beside 2.9 and 0.3 m ones across a line of dry nodes. The baroclinic exchange through its subcells was energy-exact, yet it grew; the Casimir argument above has no reason to hold on this stencil either. A smaller thin depth or free depth, or upwind tracers, did not stop it. Shore elements now carry only their layers' share of the barotropic flux and feel no pressure gradient within themselves, which is the ROMS wet–dry mask in DG form. With that, a patch around the element holds 9e-10 m/s for 8.8 h. Their faces still couple them fully to wet neighbours.
