# dg-rs Architecture

A practical DG library for Norwegian coastal ocean modeling.

## Design Philosophy

**Evolve, don't over-engineer.** Start concrete, abstract when patterns emerge.

| Principle | Approach |
|-----------|----------|
| Single crate | Modules mirror future concerns; split only when needed |
| Concrete first | `Mesh1D`, `Mesh2D` before generic `Mesh<D>` trait |
| faer for LA | Faster than nalgebra, recommended by RESEARCH.md |
| GPU later | cudarc when needed; rayon sufficient for ~100k elements |
| Norwegian focus | Optimize for fjords, not generality |

---

## Current Module Structure

```
src/
├── lib.rs              # Public API, re-exports
├── polynomial/
│   ├── mod.rs
│   ├── legendre.rs     # Legendre P_n(x), P'_n(x)
│   └── nodes.rs        # Gauss-Lobatto nodes/weights
├── basis/
│   ├── mod.rs
│   └── vandermonde.rs  # V, V^{-1} matrices
├── operators/
│   ├── mod.rs
│   ├── differentiation.rs  # Dr = Vr * V^{-1}
│   ├── mass.rs             # M = diag(weights)
│   └── lift.rs             # LIFT = M^{-1} * E^T
├── mesh/
│   ├── mod.rs
│   └── mesh1d.rs       # Uniform/periodic 1D mesh
├── flux/
│   ├── mod.rs
│   └── upwind.rs       # Upwind, Lax-Friedrichs
├── solver/
│   ├── mod.rs
│   ├── dg1d.rs         # DGSolution1D storage
│   └── rhs.rs          # RHS computation (serial + parallel)
└── time/
    ├── mod.rs
    └── ssp_rk3.rs      # SSP-RK3, time-aware variant
```

---

## Evolution Path

### Phase 1: 1D Shallow Water (Current → Next)

Add system of equations support without heavy abstraction:

```rust
// Simple trait for conservation laws
pub trait ConservationLaw {
    const NUM_VARS: usize;

    fn flux(state: &[f64], a: f64) -> Vec<f64>;
    fn max_wavespeed(state: &[f64]) -> f64;
    fn roe_flux(u_l: &[f64], u_r: &[f64], normal: f64) -> Vec<f64>;
}

// Concrete implementation
pub struct ShallowWater1D {
    pub g: f64,  // gravity
}

impl ConservationLaw for ShallowWater1D {
    const NUM_VARS: usize = 2;  // [h, hu]
    // ...
}
```

**New files:**
- `src/equations/mod.rs` - ConservationLaw trait
- `src/equations/shallow_water.rs` - 1D SWE implementation
- `src/flux/roe.rs` - Roe solver for systems
- `src/solver/dg1d_system.rs` - System-aware solution storage

### Phase 2: 2D Triangular Mesh

Add 2D without generic dimension trait:

```rust
// Concrete 2D mesh for triangles
pub struct Mesh2D {
    pub vertices: Vec<[f64; 2]>,
    pub elements: Vec<[usize; 3]>,      // Triangle vertex indices
    pub neighbors: Vec<[Option<(usize, usize)>; 3]>,  // Per-edge neighbors
    pub boundary_tags: Vec<[Option<BoundaryTag>; 3]>,
}

pub enum BoundaryTag {
    Wall,
    Open,
    TidalForcing { constituent: TidalConstituent },
    River { discharge: f64 },
}

// Geometric factors computed once per mesh
pub struct GeometricFactors2D {
    pub jacobian: Vec<[[f64; 2]; 2]>,  // Per-element 2x2 Jacobian
    pub det_j: Vec<f64>,               // Jacobian determinant
    pub normals: Vec<[[f64; 2]; 3]>,   // Outward normal per edge
    pub edge_lengths: Vec<[f64; 3]>,   // Edge lengths for surface integrals
}
```

**New files:**
- `src/mesh/mesh2d.rs` - Triangle mesh
- `src/mesh/gmsh.rs` - Gmsh MSH reader (custom, not crate)
- `src/operators/operators2d.rs` - 2D differentiation, mass, LIFT
- `src/polynomial/warp_blend.rs` - Triangle node placement
- `src/solver/dg2d.rs` - 2D solution storage and RHS

### Phase 3: Norwegian Coast Features

Physics-specific additions:

```rust
// Tidal forcing
pub struct TidalConstituent {
    pub name: &'static str,  // "M2", "S2", "K1", "O1"
    pub amplitude: f64,
    pub phase: f64,
    pub frequency: f64,
}

// Boundary conditions for coastal modeling
pub struct CoastalBoundary {
    pub tidal: Vec<TidalConstituent>,
    pub subtidal: Option<SubtidalForcing>,  // From larger model
}

// Coriolis
pub fn coriolis_parameter(latitude: f64) -> f64 {
    2.0 * 7.2921e-5 * latitude.to_radians().sin()  // f = 2Ω sin(φ)
}
```

**New files:**
- `src/physics/tides.rs` - Tidal constituents and forcing
- `src/physics/coriolis.rs` - f-plane and β-plane
- `src/physics/friction.rs` - Manning bottom friction
- `src/boundary/flather.rs` - Radiation boundary conditions

### Phase 4: Performance

Only add complexity when profiling shows need:

```rust
// GPU kernel structure (cudarc)
#[cfg(feature = "cuda")]
pub struct DGSolverGPU {
    device: CudaDevice,
    operators: DeviceBuffer<f64>,  // Dr, Ds, LIFT on GPU
    solution: DeviceBuffer<f64>,   // [n_elements, n_nodes, n_vars]
}

// Batched element operations
#[cfg(feature = "cuda")]
impl DGSolverGPU {
    pub fn compute_rhs(&self) -> DeviceBuffer<f64> {
        // cuBLAS batched GEMM for element-local matrix products
        // Custom kernels for flux computation
    }
}
```

---

## Key Abstractions (Add When Needed)

### NumericalFlux trait (Phase 1)

```rust
pub trait NumericalFlux {
    fn compute(
        &self,
        u_left: &[f64],
        u_right: &[f64],
        normal: f64,
    ) -> Vec<f64>;
}

pub struct RoeFlux<E: ConservationLaw> { _marker: PhantomData<E> }
pub struct HLLFlux<E: ConservationLaw> { _marker: PhantomData<E> }
pub struct LaxFriedrichsFlux { pub max_speed: f64 }
```

### TimeIntegrator trait (Phase 1, if using diffsol)

```rust
// Or just use diffsol directly without abstraction
pub trait TimeIntegrator {
    fn step<F>(&self, u: &mut DGSolution, rhs: F, dt: f64)
    where F: Fn(&DGSolution) -> DGSolution;
}
```

### Mesh trait (Phase 2, only if needed)

```rust
// Only add if you genuinely need 1D/2D polymorphism
pub trait Mesh {
    type Element;
    type Face;

    fn n_elements(&self) -> usize;
    fn element(&self, k: usize) -> &Self::Element;
    fn neighbors(&self, k: usize) -> impl Iterator<Item = Option<usize>>;
}
```

---

## What NOT to Do

1. **Don't create multi-crate workspace** until >10k LOC
2. **Don't add ReferenceElement trait** - just implement Triangle directly
3. **Don't use Burn/wgpu** - wgpu lacks f64, Burn is ML-focused
4. **Don't abstract dimension** with const generics until 3D is actually needed
5. **Don't optimize prematurely** - profile first, rayon may be enough

---

## File Size Guidelines

| Size | Action |
|------|--------|
| <500 lines | Keep in single file |
| 500-1000 lines | Consider splitting |
| >1000 lines | Split into submodules |

Current codebase: ~1500 LOC total, well within single-crate territory.

---

## Testing Strategy

```rust
// 1. Unit tests: Mathematical correctness
#[test]
fn test_legendre_orthogonality() { /* ... */ }

#[test]
fn test_differentiation_exact_for_polynomials() { /* ... */ }

// 2. Convergence tests: Order of accuracy
#[test]
fn test_convergence_p3_advection() {
    // Verify 4th order convergence (N+1 for smooth solutions)
}

// 3. Conservation tests: Physical invariants
#[test]
fn test_mass_conservation_periodic() { /* ... */ }

// 4. Manufactured solutions: Full solver verification
#[test]
fn test_manufactured_solution_swe() {
    // u(x,t) = known function, compute source term analytically
    // Verify solver reproduces exact solution
}
```

---

## Summary

The architecture evolves with the code:

```
Phase 0 (done): 1D scalar advection
    └─> concrete types, working solver, verified convergence

Phase 1 (next): 1D shallow water
    └─> ConservationLaw trait, Roe solver, source terms

Phase 2: 2D triangles
    └─> Mesh2D, Operators2D, keep it concrete

Phase 3: Norwegian coast
    └─> Tides, Coriolis, boundary conditions

Phase 4: Performance
    └─> Profile → cudarc GPU only if needed
```

Add abstraction when you have 2+ concrete implementations that share patterns, not before.
