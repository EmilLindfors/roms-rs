# Claude Guidelines for dg-rs

This is a high-performance Discontinuous Galerkin (DG) solver for coastal ocean modeling, targeting simulation of currents along the Norwegian coast. Numerical accuracy and computational efficiency are critical.

## Project Context

### Primary Application
- **Norwegian coastal current simulation**: Complex fjord geometry, strong tidal forcing, steep bathymetry
- **Shallow water equations**: Mass and momentum conservation with Coriolis, bottom friction, wetting/drying
- **Operational oceanography**: Results must be accurate enough for real-world decision support

### Design Philosophy
- **Correctness first**: Numerical methods must converge at theoretical rates
- **Performance matters**: Will scale to millions of elements; every unnecessary allocation counts
- **GPU-ready architecture**: Design data structures for future GPU port (cudarc)
- **Composable**: Clean separation between operators, mesh, flux, and time integration

## Code Standards

### Mathematical Rigor
- All numerical methods must have theoretical backing (cite Hesthaven-Warburton for DG)
- Convergence tests are mandatory for any spatial/temporal discretization
- Conservation properties must be verified for hyperbolic systems
- Document the mathematical formulation in module-level doc comments

### Performance Requirements
- Prefer stack allocation and slices over heap allocation in hot paths
- Use `faer` for linear algebra (already a dependency)
- Element-local operations should be vectorization-friendly
- Profile before optimizing; use `cargo bench` with criterion when benchmarks exist

### Testing Requirements
- **Unit tests**: Every operator must be tested for polynomial exactness
- **Convergence tests**: Verify (N+1) order accuracy for smooth solutions
- **Conservation tests**: Total mass/momentum must be preserved (periodic BCs)
- **Regression tests**: Any bug fix must include a test that would have caught it

## Architecture Overview

```
src/
├── polynomial/     # Legendre polynomials, GLL nodes/weights
├── basis/          # Vandermonde matrices for nodal-modal transforms
├── operators/      # Dr, Mass, LIFT matrices (reference element)
├── mesh/           # Physical mesh, connectivity, Jacobians
├── flux/           # Numerical fluxes (upwind, Lax-Friedrichs)
├── solver/         # Solution storage, RHS computation
└── time/           # SSP-RK3, CFL computation
```

### Key Types
- `DGOperators1D`: All reference element operators bundled together
- `Mesh1D`: Physical mesh with neighbor connectivity
- `DGSolution1D`: Nodal values stored contiguously `[n_elements × n_nodes]`
- `BoundaryCondition`: Dirichlet values at inflow boundaries

## Common Tasks

### Adding a New Flux Function
1. Add function to `src/flux/upwind.rs` (or new file)
2. Follow signature: `fn flux(u_minus, u_plus, a, normal) -> f64`
3. Add unit tests verifying continuous solution gives physical flux
4. Export from `src/flux/mod.rs` and `src/lib.rs`

### Adding a New Time Integrator
1. Add to `src/time/` module
2. For time-dependent BCs, pass time to RHS function (see `ssp_rk3_step_timed`)
3. Verify order of accuracy with ODE test (exponential growth)
4. Document stability region if relevant

### Extending to 2D
When implementing 2D elements:
- Use tensor-product GLL nodes for quadrilaterals
- Use Fekete or Warp-Blend nodes for triangles
- Face-based connectivity for inter-element coupling
- Surface integrals become edge integrals

## Norwegian Coast Specifics

### Physical Considerations
- **Fjords**: Long, narrow, deep - need anisotropic mesh refinement
- **Tides**: M2 dominant, strong currents in narrow straits
- **Fresh water**: River runoff creates stratification (future 3D)
- **Coriolis**: f ≈ 1.2×10⁻⁴ s⁻¹ at 60°N, important for circulation

### Numerical Considerations
- **Wetting/drying**: Tidal flats require robust treatment
- **Steep bathymetry**: Well-balanced schemes for lake-at-rest
- **Open boundaries**: Radiation conditions, tidal forcing
- **Bottom friction**: Manning/Chézy formulation

## What NOT to Do

- **Don't break convergence**: Any change to spatial discretization must pass convergence tests
- **Don't ignore conservation**: DG should conserve mass exactly (up to machine precision)
- **Don't allocate in RHS**: The RHS function is called thousands of times per simulation
- **Don't use f32**: Coastal models need f64 precision for stability
- **Don't skip BC timing**: Time-dependent BCs must be evaluated at correct RK stage times

## Verification Checklist

Before any PR:
- [ ] `cargo test` passes (all tests)
- [ ] `cargo test --features parallel` passes
- [ ] `cargo clippy` has no warnings
- [ ] Convergence rates match theory (check ACCURACY.md)
- [ ] No new allocations in hot paths (check with profiler if unsure)

## References

- Hesthaven & Warburton, "Nodal Discontinuous Galerkin Methods" (2008) - DG bible
- Karniadakis & Sherwin, "Spectral/hp Element Methods" (2005) - Polynomial bases
- LeVeque, "Finite Volume Methods for Hyperbolic Problems" (2002) - Conservation laws
- Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics" (2009) - Numerical fluxes

## Quick Commands

```bash
# Run all tests
cargo test

# Run with parallel feature
cargo test --features parallel

# Run convergence tests with output
cargo test --test convergence_test -- --nocapture

# Run example
cargo run --release --example advection_1d

# Check for issues
cargo clippy

# Profiling (see scripts/ for details)
./scripts/flamegraph.sh froya_real_data       # Generate flamegraph SVG
./scripts/flamegraph.sh froya_real_data netcdf  # With features
./scripts/perf-stat.sh froya_real_data        # Quick CPU stats
./scripts/samply.sh froya_real_data           # Interactive profiler UI
```
