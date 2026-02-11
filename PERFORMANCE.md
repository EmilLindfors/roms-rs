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

### 2D SWE RHS Benchmarks (2026-01-18)

After SoA refactoring and par_chunks_mut optimization:

#### RHS Computation Time by Mesh Size

| Mesh | Elements | serial | parallel_opt | adaptive | batched |
|------|----------|--------|--------------|----------|---------|
| 16x16 | 256 | 0.3ms | 1.3ms | 0.3ms | 1.3ms |
| 32x32 | 1,024 | 1.2ms | 2.1ms | 1.3ms | 2.2ms |
| 64x64 | 4,096 | 4.8ms | 2.2ms | 2.2ms | 3.2ms |
| 100x100 | 10,000 | 11.9ms | 3.9ms | 3.9ms | 16.9ms |

**Notes**:
- `parallel_opt` = parallel + SIMD + per-thread workspace + direct output writes
- `adaptive` dispatches to serial for <2000 elements, parallel_opt otherwise
- `batched` uses faer GEMM for volume terms (slower due to surface term overhead)
- Parallel overhead dominates at small mesh sizes (<2000 elements)

#### Speedup vs Serial Baseline

| Mesh | Elements | parallel_opt | Notes |
|------|----------|--------------|-------|
| 64x64 | 4,096 | 2.2x | Parallel starts to win |
| 100x100 | 10,000 | 3.1x | Good parallel efficiency |

#### Source Term Overhead (256 elements, P3)

| Configuration | Time | Overhead vs baseline |
|---------------|------|---------------------|
| No sources | 112 µs | baseline |
| + Coriolis | 126 µs | +13% |
| + Bathymetry | 126 µs | +13% |
| + Friction | 143 µs | +28% |
| All sources | 157 µs | +40% |

### Real-World Performance (2026-01-18)

Frøya 65k element simulation with full physics:

| Version | Wall Time | Improvement |
|---------|-----------|-------------|
| Before SoA refactor | ~2m 13s | baseline |
| After SoA + par_chunks_mut | 1m 55s | **15% faster** |

**Configuration**:
- 65,536 elements, P3 order (15 nodes/element)
- ~1M DOFs per variable
- Kuzmin limiters, well-balanced bathymetry
- Manning friction, Coriolis, wetting/drying
- Ocean model nesting BC with interpolation

### Large-Scale Benchmark (2026-01-20)

Frøya 65k element simulation with full physics after TVB limiter consolidation:

| Metric | Value |
|--------|-------|
| Elements | 65,025 (255×255) |
| Polynomial order | P2 (9 nodes/element) |
| Total DOFs | 1,755,675 |
| Simulation time | 60 seconds |
| Time steps | 919 |
| Average dt | 0.07 s |
| **Wall clock time** | **3:37 (217 s)** |
| User time | 988.84 s |
| System time | 380.16 s |
| Peak memory | 257 MB |

**Performance Metrics:**
- **Steps/second**: 4.23
- **DOF-updates/second**: 7.43 M
- **Realtime ratio**: 0.28× (slower than realtime)
- **Parallel efficiency**: ~4.5× (988s user / 217s wall)

**Configuration:**
- Features: `parallel`, `simd`, `netcdf`
- Limiter: Kuzmin (vertex-based) + positivity
- Physics: Coriolis, Manning friction, wind stress, pressure gradient
- Well-balanced: Yes (cell-average bathymetry)
- Wetting/drying: Yes (thin-layer blending)
- BC: Ocean model nesting (NorKyst)

**Analysis:**
- At 4.23 steps/sec with dt=0.07s, each step takes ~237ms
- SSP-RK3 = 3 RHS evaluations per step → ~79ms per RHS
- Per-element RHS time: 79ms / 65k = ~1.2µs/element
- Main costs: volume terms (Dr/Ds multiply), flux computation, limiters

### Optimization History

| Date | Change | Impact |
|------|--------|--------|
| 2026-01-20 | TVB limiter removal | Code cleanup, no perf change |
| 2026-01-18 | SoA storage for SWESolution2D | Eliminated AoS↔SoA conversion |
| 2026-01-18 | par_chunks_mut for parallel_opt | -24% at 100x100 (eliminated Vec collection) |
| 2026-01-18 | Workspace buffers in batched | -25% to -39% for batched version |
| 2026-01-18 | Overall real-world | **15% faster** (2m13s → 1m55s) |
