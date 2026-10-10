# Operational features and future work (P5, P7)

## Priority 5: Operational Features

### P5.1 River forcing
- [ ] NVE database (1,760 rivers in NorKyst-800) and a ROMS river-file reader onto `source::RiverSources` (vertical shapes: `RiverProfile`, done 2026-10-01; see P1.6). `Discharge2D`/`ConstantDischarge2D` (weak boundary inflow) and `RiverTracerSource` (tracer nudging, no volume) are superseded by `RiverSources`: delete them with the P6 cleanup.

### P5.2 ROMS-compatible output
- [ ] `NetCDFWriter` produces CF-1.8 but not ROMS-compatible output; add a ROMS variable-name mapping.

### P5.3 Restart / checkpointing
- [ ] HDF5 or NetCDF restart files (operational daily forecasts must hot-start).
  - [x] 3D restarts, for the multi-day 3D runs (15 days of the Frøya coastline mesh in 3D take ≈ 3–18 days of wall time, P1.3). Done 2026-10-06 (`io::Restart3D`, see CHANGELOG). A restart holds the whole `Solution3D`, the splitter's AB3 history of `G` (`SlowForcingRecord`) and the vertical reference, plus a domain fingerprint checked on resume. The barotropic pass starts from the state, and the forcing and clock are functions of time, so nothing else carries over. `froya_real_data tide3d=1 restart_hours=H` writes restarts, and `resume=<file>` continues one, with the station records and the snapshot file. Gate: `a_restarted_run_ends_bit_for_bit_where_the_uninterrupted_run_does`. Mausund, resumed in a new process: outputs byte-identical.
    - [ ] Only one restart file is kept (each replaces the last, atomically). For spin-up branching keep numbered copies (`restart_keep=`), or let the driver name the file per time.
    - [ ] The physics' configuration is not in the file: a resume with other options (mixing, limiter, nesting, wind) runs on silently from the restart's state. The example records the command line (`command` metadata) and prints it on resume. Compare the options that change the physics and refuse a mismatch, or store a configuration hash.
    - [ ] 2D runs (`Simulation`, `SWESolution2D` plus the multirate stepper's state) have no restart yet; the 15-day 2D runs take ≈ 4 h, so less pressing.
    - [ ] A NetCDF export of the restart (ROMS-like `ocean_rst.nc`) for other tools, once P5.2's output conventions are settled. The binary format stays the restart itself (exact `f64`, no native dependency, runs in CI).

### P5.4 Operational pipeline
- [ ] Automated run scripts, monitoring and alerting, THREDDS/OPeNDAP serving.

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
- [ ] Wave–current interaction: now F.4 (the spectral wave model, stage 1 done 2026-10-03; coupling is a stage there).
- [ ] Biological coupling (NPZD). Salmon-lice dispersion is particle tracking, now F.2 under "Application target".
- [ ] Sediment transport.
