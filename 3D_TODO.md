# 3D Implementation Roadmap

This document outlines the implementation plan for extending the DG solver to 3D for Norwegian coastal ocean modeling. The approach follows ROMS-style hybrid discretization: **DG horizontal + finite-difference vertical** with sigma (terrain-following) coordinates and mode-split time stepping.

## Architecture Decision

**Chosen approach: ROMS-style hybrid** (not full 3D DG hexahedra)

Rationale:
- Ocean is geometrically thin (~1km depth vs ~100km horizontal)
- Mode splitting handles fast barotropic waves efficiently (Δt_bt ≈ 1s vs Δt_bc ≈ 60s)
- Matches operational practice (ROMS, SCHISM, Thetis, NorKyst800)
- Sigma coordinates essential for fjord bathymetry
- Well-understood vertical mixing parameterizations (KPP, GLS)

```
┌─────────────────────────────────────────────────────────────┐
│              3D Solution Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   HORIZONTAL: 2D DG         VERTICAL: Finite Difference    │
│   ┌──────────────────┐      ┌──────────────────┐           │
│   │ Quadrilateral    │  ×   │ 35-40 σ-levels   │           │
│   │ elements (P1-P3) │      │ terrain-following │           │
│   │ Tensor-product   │      │ stretched grid    │           │
│   └──────────────────┘      └──────────────────┘           │
│                                                             │
│   MODE SPLITTING                                            │
│   ┌─────────────────┐       ┌─────────────────┐            │
│   │ Barotropic (2D) │       │ Baroclinic (3D) │            │
│   │ ~30-60× faster  │       │ Internal modes  │            │
│   │ Surface waves   │       │ Stratification  │            │
│   └─────────────────┘       └─────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Vertical Infrastructure

### 1.1 Sigma Coordinate System
- [ ] Create `src/vertical/mod.rs` module
- [ ] Implement `SigmaGrid` struct with:
  - [ ] Uniform spacing
  - [ ] Song-Haidvogel stretching (θ_s, θ_b, h_c parameters)
  - [ ] Surface/bottom clustering for boundary layers
- [ ] Sigma-to-z coordinate transform: `z = η + (η + H) × σ`
- [ ] Metric terms: `∂z/∂σ`, `∂z/∂x|_σ`, `∂z/∂y|_σ`
- [ ] Unit tests for coordinate transforms

```rust
// Target API
pub struct SigmaGrid {
    pub n_levels: usize,
    pub sigma: Vec<f64>,      // σ ∈ [-1, 0], -1 = bottom, 0 = surface
    pub sigma_w: Vec<f64>,    // σ at vertical cell faces
    pub stretching: StretchingParams,
}

pub enum StretchingParams {
    Uniform,
    SongHaidvogel { theta_s: f64, theta_b: f64, hc: f64 },
}
```

### 1.2 3D State Types
- [ ] Create `src/solver/state_3d.rs`
- [ ] Implement `State3D` (point-wise state)
- [ ] Implement `Solution3D` (full domain state)
- [ ] Implement `Integrable` trait for `Solution3D`
- [ ] Memory layout: `[n_elements × n_nodes × n_levels]` (level-major for cache)

```rust
pub struct Solution3D {
    // Barotropic (2D) - drives fast surface waves
    pub eta: DGSolution2D,        // Free surface elevation
    pub ubar: DGSolution2D,       // Depth-averaged u
    pub vbar: DGSolution2D,       // Depth-averaged v

    // Baroclinic (3D) - slow internal dynamics
    pub u: Vec<f64>,              // 3D u-velocity
    pub v: Vec<f64>,              // 3D v-velocity
    pub w: Vec<f64>,              // Vertical velocity (diagnosed)
    pub temp: Vec<f64>,           // Temperature
    pub salt: Vec<f64>,           // Salinity
    pub rho: Vec<f64>,            // Density (from EOS)

    // Metadata
    pub n_elements: usize,
    pub n_nodes: usize,
    pub n_levels: usize,
}
```

### 1.3 Equation of State
- [ ] Extend existing `src/equations/equation_of_state.rs`
- [ ] UNESCO EOS-80 or TEOS-10 implementation
- [ ] Compute ρ(T, S, p) with pressure dependence
- [ ] Thermal expansion coefficient α = -∂ρ/∂T / ρ
- [ ] Haline contraction coefficient β = ∂ρ/∂S / ρ

---

## Phase 2: Mode-Split Time Stepping

### 2.1 Mode Split Integrator
- [ ] Create `src/time/mode_split.rs`
- [ ] Implement `ModeSplitIntegrator` struct
- [ ] Barotropic subcycling (30-60 fast steps per slow step)
- [ ] G-term coupling (Thetis/ROMS approach)
- [ ] Velocity reconciliation after barotropic steps

```rust
pub struct ModeSplitIntegrator {
    pub bt_integrator: SSPRK3,    // Fast 2D
    pub bc_integrator: SSPRK3,    // Slow 3D
    pub n_bt_steps: usize,        // Barotropic steps per baroclinic
    pub split_method: SplitMethod,
}

pub enum SplitMethod {
    /// ROMS-style with predictor-corrector
    ROMSPredictor,
    /// Thetis G-term coupling
    GTerm,
}
```

### 2.2 Barotropic Solver Adaptation
- [ ] Extract depth-averaged forcing from 3D fields
- [ ] G-term: `G = ∫(advection + diffusion + baroclinic_pg) dz`
- [ ] Apply G-term as source in 2D SWE
- [ ] Average barotropic solution over subcycle for stability

### 2.3 Vertical Velocity Diagnosis
- [ ] Compute w from 3D continuity equation
- [ ] `∂w/∂z = -∂u/∂x - ∂v/∂y` integrated from bottom
- [ ] Kinematic boundary conditions: w = 0 at bottom, w = ∂η/∂t at surface

---

## Phase 3: 3D Physics

### 3.1 Baroclinic Pressure Gradient
- [ ] Extend `src/source/baroclinic.rs` for 3D
- [ ] Standard sigma-coordinate formulation
- [ ] **Balanced method** (Berntsen) for steep bathymetry - CRITICAL for fjords
- [ ] Fourth-order pressure gradient approximation
- [ ] Verify lake-at-rest preservation

```rust
/// Baroclinic pressure gradient with balanced method for steep topography
pub fn baroclinic_pg_balanced(
    rho: &[f64],           // Density profile at column
    sigma: &SigmaGrid,
    eta: f64,              // Surface elevation
    h: f64,                // Water depth
    dh_dx: f64,            // Bathymetry gradient
    g: f64,
) -> Vec<f64> {
    // Returns pressure gradient at each sigma level
    // Uses Berntsen's method to reduce spurious currents
}
```

### 3.2 Vertical Mixing
- [ ] Create `src/source/vertical_mixing.rs`
- [ ] Trait `VerticalMixing` for different closures
- [ ] Implement KPP (K-Profile Parameterization) - recommended for open ocean
- [ ] Implement GLS (Generic Length Scale) - k-ε variant for estuaries
- [ ] Background diffusivity for numerical stability
- [ ] Convective adjustment for unstable stratification

```rust
pub trait VerticalMixing: Send + Sync {
    /// Compute eddy viscosity profile Kv(z)
    fn compute_viscosity(&self, column: &Column3D, forcing: &SurfaceForcing) -> Vec<f64>;

    /// Compute eddy diffusivity profile Kt(z)
    fn compute_diffusivity(&self, column: &Column3D, forcing: &SurfaceForcing) -> Vec<f64>;
}
```

### 3.3 Vertical Diffusion (Implicit)
- [ ] Tridiagonal solver for implicit vertical diffusion
- [ ] Apply to momentum: `∂u/∂t = ... + ∂/∂z(Kv ∂u/∂z)`
- [ ] Apply to tracers: `∂T/∂t = ... + ∂/∂z(Kt ∂T/∂z)`
- [ ] Surface/bottom boundary conditions for diffusion

### 3.4 3D Advection
- [ ] Horizontal: Reuse DG operators per sigma level
- [ ] Vertical: Upwind or QUICK scheme in sigma space
- [ ] Tracer advection in conservative form
- [ ] Slope limiters for tracers (extend Kuzmin to 3D columns)

---

## Phase 4: 3D RHS Computation

### 4.1 Main RHS Function
- [ ] Create `src/solver/rhs_3d.rs`
- [ ] Combine horizontal DG + vertical FD
- [ ] Order of operations:
  1. Horizontal advection (DG, per level)
  2. Vertical advection (FD)
  3. Baroclinic pressure gradient
  4. Coriolis (extend existing)
  5. Vertical diffusion (implicit)
  6. Surface/bottom stresses

```rust
pub fn compute_rhs_3d(
    state: &Solution3D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    sigma: &SigmaGrid,
    mixing: &dyn VerticalMixing,
    params: &Physics3DParams,
    t: f64,
) -> Solution3D {
    // Returns time derivatives for all 3D fields
}
```

### 4.2 SIMD Optimization
- [ ] Extend `src/solver/simd_kernels.rs` for 3D operations
- [ ] Vectorize over sigma levels (inner loop)
- [ ] Memory layout optimized for vertical operations

### 4.3 Parallel Execution
- [ ] Rayon parallelization over elements (horizontal)
- [ ] Vertical operations sequential per element (small, cache-friendly)

---

## Phase 5: Boundary Conditions

### 5.1 Open Boundary Conditions (3D)
- [ ] Extend Flather BC to 3D velocity profiles
- [ ] Chapman BC for tracers
- [ ] Relaxation zones (sponge layers) - already exists for 2D
- [ ] One-way nesting from parent model (extend existing `NestingBC2D`)

### 5.2 River Inflow
- [ ] Point source / distributed source options
- [ ] Vertical distribution of river input (surface-weighted)
- [ ] Temperature and salinity of river water
- [ ] NVE river discharge data integration

### 5.3 Surface Forcing
- [ ] Wind stress: τ_x, τ_y → surface momentum flux
- [ ] Heat flux: Q_net → surface temperature BC
- [ ] Freshwater flux: E - P → surface salinity BC
- [ ] Atmospheric pressure (already exists, verify 3D usage)

### 5.4 Bottom Boundary
- [ ] Bottom stress from quadratic drag (extend existing Manning/Chézy)
- [ ] No-flux BC for tracers at bottom
- [ ] Geothermal heat flux (optional, usually negligible)

---

## Phase 6: I/O and Initialization

### 6.1 NetCDF Output
- [ ] Extend `src/io/netcdf_*.rs` for 3D fields
- [ ] CF-conventions compliant output
- [ ] Sigma coordinate metadata
- [ ] Compressed output for large domains

### 6.2 Initialization
- [ ] Initialize from ROMS/NEMO restart files
- [ ] Initialize from climatology (WOA, GLORYS)
- [ ] Spin-up strategy for stratification
- [ ] Hot-start capability

### 6.3 Diagnostics
- [ ] Mixed layer depth computation
- [ ] Potential energy anomaly
- [ ] Overturning streamfunction
- [ ] Harmonic analysis for 3D currents

---

## Phase 7: Validation

### 7.1 Analytical Tests
- [ ] Internal wave propagation (mode-1)
- [ ] Lock exchange (density-driven flow)
- [ ] Wind-driven upwelling
- [ ] Kelvin wave coastal propagation

### 7.2 Norwegian Fjord Tests
- [ ] Idealized fjord geometry
- [ ] Realistic Hardangerfjord or Sognefjord case
- [ ] Compare with mooring observations
- [ ] Compare with NorKyst800 results

---

## Implementation Priority

| Priority | Component | Rationale |
|----------|-----------|-----------|
| 1 | Sigma grid + 3D state | Foundation for everything |
| 2 | Mode splitting | Required for efficiency |
| 3 | Baroclinic PG (balanced) | Critical for fjords |
| 4 | Vertical mixing (KPP) | Physical realism |
| 5 | 3D RHS integration | Tie everything together |
| 6 | Boundary conditions | Real-world forcing |
| 7 | Validation | Ensure correctness |

---

## Key References

1. **DG Foundation**: Hesthaven & Warburton, "Nodal DG Methods" (2008) - Ch. 10 for 3D
2. **ROMS**: Shchepetkin & McWilliams (2005) - Mode splitting, sigma coordinates
3. **Thetis**: Kärnä et al. (2018) - DG coastal ocean, mode splitting
4. **Balanced PG**: Berntsen et al. - Pressure gradient errors in sigma coordinates
5. **KPP**: Large et al. (1994) - Vertical mixing parameterization
6. **Norwegian Fjords**: Asplin et al. (NorKyst800 documentation)

---

## Estimated Effort

| Phase | New Code (LOC) | Complexity |
|-------|----------------|------------|
| 1. Vertical Infrastructure | ~1,200 | Medium |
| 2. Mode Splitting | ~800 | High |
| 3. 3D Physics | ~1,500 | High |
| 4. 3D RHS | ~1,000 | High |
| 5. Boundary Conditions | ~800 | Medium |
| 6. I/O | ~500 | Low |
| 7. Validation | ~500 | Medium |
| **Total** | **~6,300** | |

---

## Dependencies on Existing Code

| Existing Module | 3D Usage |
|-----------------|----------|
| `mesh/mesh2d.rs` | Horizontal mesh (unchanged) |
| `operators/operators_2d.rs` | DG operators per level |
| `solver/state_2d.rs` | Barotropic state |
| `solver/rhs_swe_2d.rs` | Barotropic RHS template |
| `source/coriolis_2d.rs` | Apply per level |
| `source/friction_2d.rs` | Bottom stress |
| `source/baroclinic.rs` | Extend to 3D |
| `boundary/*.rs` | Extend for 3D profiles |
| `time/integrator.rs` | Wrap in mode-split |
| `equations/equation_of_state.rs` | Density computation |

---

## Open Questions

1. **Wetting/drying in 3D**: How to handle vertical structure when depth → 0?
2. **Implicit vertical diffusion**: Use existing faer, or tridiagonal solver?
3. **GPU acceleration**: cudarc for 3D kernels vs. level-by-level 2D kernels?
4. **Hybrid vertical**: Support z-levels in deep ocean, sigma in coastal?

---

## Success Criteria

- [ ] Internal wave mode-1 propagates at correct phase speed
- [ ] Lake-at-rest preserved to machine precision with stratification
- [ ] Stable with CFL > 0.5 for explicit horizontal, implicit vertical
- [ ] Reproduces basic fjord circulation (estuarine exchange)
- [ ] Performance: >10x faster than Python/Firedrake equivalent
