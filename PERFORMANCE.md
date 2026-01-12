# Performance

This document tracks computational performance of the DG solver.

## Computational Complexity

### Per Element Operations

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Volume term (Dr * u) | O(N²) | Dense matrix-vector multiply |
| Surface flux | O(1) | 2 face values per element |
| LIFT application | O(N) | Sparse (2 non-zeros per row) |
| **Total RHS** | **O(N²)** | Per element |

Where N = polynomial order + 1 (number of nodes per element).

### Per Timestep

| Operation | Complexity | Notes |
|-----------|------------|-------|
| RHS evaluation | O(K × N²) | K = number of elements |
| SSP-RK3 step | 3 × RHS | 3 stages |
| **Total** | **O(K × N²)** | Per timestep |

### Memory Usage

| Storage | Size | Notes |
|---------|------|-------|
| Solution | K × N | Nodal values |
| Operators | N² | Dr, Mass, LIFT (shared) |
| RHS workspace | K × N | Temporary for RK stages |

## Benchmarks

### SIMD Kernel Benchmarks (2026-01-10)

Measured with criterion on WSL2 Linux (Intel CPU), release build.

#### Differentiation Matrix (Dr/Ds × flux)

| Poly Order | Nodes | Scalar | SIMD (faer) | Speedup |
|------------|-------|--------|-------------|---------|
| P2 | 9 | 54 ns | 54 ns | 1.0x |
| P3 | 16 | 133 ns | 133 ns | 1.0x |
| P4 | 25 | 296 ns | 254 ns | **1.16x** |
| P5 | 36 | 646 ns | 357 ns | **1.81x** |

**Notes**:
- SIMD version uses faer GEMV for ≥20 nodes, scalar below (overhead threshold)
- Larger matrices benefit more from faer's optimized BLAS kernels
- P2/P3 use scalar path (faer overhead > computation time)

#### Coriolis Source Term

| Poly Order | Nodes | Scalar | SIMD (pulp) | Speedup |
|------------|-------|--------|-------------|---------|
| P2 | 9 | 4.2 ns | 4.2 ns | 1.0x |
| P3 | 16 | 6.5 ns | 5.8 ns | 1.1x |
| P4 | 25 | 10.1 ns | 7.2 ns | **1.4x** |
| P5 | 36 | 13.0 ns | 9.3 ns | **1.4x** |

**Notes**:
- Uses pulp SIMD intrinsics with automatic AVX2/SSE4 detection
- Simple FMA operations vectorize well
- Scalar tail handling for non-aligned sizes

#### LIFT Matrix Application

| Poly Order | Nodes × Face Nodes | Scalar | SIMD |
|------------|-------------------|--------|------|
| P2 | 9 × 3 | 31 ns | 31 ns (scalar) |
| P5 | 36 × 6 | 87 ns | 87 ns (scalar) |

**Notes**:
- LIFT matrices are small (face_nodes = 3-6), faer overhead dominates
- SIMD version falls back to scalar implementation

#### Combine Derivatives (Chain Rule)

| Poly Order | Nodes | Scalar | SIMD (pulp) | Speedup |
|------------|-------|--------|-------------|---------|
| P2 | 9 | 18 ns | 14 ns | 1.3x |
| P5 | 36 | 62 ns | 34 ns | **1.8x** |

**Notes**:
- Combines 12 input arrays with 4 geometric factors
- FMA-heavy computation benefits from SIMD

### Running Benchmarks

```bash
# Run all SIMD benchmarks
cargo bench --features simd

# Run specific benchmark group
cargo bench --features simd -- diff_matrix
cargo bench --features simd -- coriolis
cargo bench --features simd -- lift_matrix
cargo bench --features simd -- combine_derivatives
```

### Planned Benchmarks

- [ ] Full RHS evaluation time vs. element count
- [ ] End-to-end timestep throughput (DOFs/second)
- [ ] Memory bandwidth utilization
- [ ] Parallel + SIMD combined scaling

## Optimization Status

### Implemented ✓

1. **Parallelism** ✓
   - Element loop parallelization with rayon (`compute_rhs_swe_2d_parallel`)
   - ~2.4x speedup on 4 cores for large meshes
   - Automatic serial/parallel selection (threshold: 1000 elements)

2. **SIMD** ✓
   - `pulp` crate for portable SIMD with runtime detection (AVX-512/AVX2/SSE4)
   - `faer` GEMV for optimized matrix-vector products
   - SoA (Structure of Arrays) data layout for SIMD kernels
   - 1.2-1.8x speedup on volume term (P4+ polynomial orders)
   - Row-major matrix caches in DGOperators2D

3. **Scalar Optimizations** ✓
   - `cbrt()` instead of `powf(1.0/3.0)` in Manning friction
   - `#[inline(always)]` on hot accessors
   - Precomputed inverse values
   - Fused wetting/drying passes

### Future Optimizations

1. **Memory Layout**
   - Full SoA storage for solution (currently AoS with SoA workspace)
   - Blocked element processing for cache efficiency

2. **SIMD Extensions**
   - Vectorized flux computation (complex branching limits current benefit)
   - SIMD Manning friction (cbrt is scalar bottleneck)

3. **GPU**
   - Port kernels to CUDA via cudarc
   - Batched element operations
   - Expected 10-100x speedup for large problems

## Reference Timings

Measured on WSL2 Linux (Intel CPU), release build.

| Test Case | Elements | Order | DOFs | Steps | Time | DOF-steps/s |
|-----------|----------|-------|------|-------|------|-------------|
| advection_1d | 20 | P3 | 80 | 234 | 5 ms | 3.7 M |

### Derived Metrics

For the P3, 20 element case:
- **Time per timestep**: ~21 µs
- **RHS evaluations**: 702 (3 per SSP-RK3 step)
- **Time per RHS**: ~7 µs
- **RHS throughput**: ~11 M DOF/s

### Scaling Estimate

Based on O(K × N²) complexity:
- Doubling elements → ~2x time
- Doubling order → ~4x time (due to N² scaling)

### 2D SWE Reference Timings (2026-01-10)

Froya-Smola-Hitra real data simulation:

| Config | Elements | Order | DOFs | Duration | Steps | Time | Notes |
|--------|----------|-------|------|----------|-------|------|-------|
| Scalar | 600 | P2 | 5,400 | 30 min | 2,522 | ~84s | Without SIMD |
| SIMD | 600 | P2 | 5,400 | 30 min | 2,522 | ~84s | With `--features simd` |

**Notes**:
- At P2 (9 nodes), SIMD overhead equals scalar cost (see benchmark tables)
- SIMD benefits appear at P4+ where faer GEMV outperforms scalar
- Full physics: Coriolis, Manning friction, wind stress, pressure, well-balanced bathymetry
- Kuzmin limiters applied after each RK stage
