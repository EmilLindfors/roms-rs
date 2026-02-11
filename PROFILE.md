# Performance Profiling Results

This document tracks profiling results and optimizations for the dg-rs solver.

---

## Latest Profile: 65k Elements (2026-01-18)

**Configuration:**
- Mesh: 65,025 elements (255×255), P2 (9 nodes/element)
- Wet elements: 38,212 (58.8%)
- Domain: 59.1 km × 44.6 km (Froya-Smola-Hitra region)
- Resolution: 232m × 175m (~185m effective)
- Features: parallel, simd, netcdf, mimalloc
- Duration: 60 seconds simulated, 919 timesteps

### Performance Summary

| Metric | Value |
|--------|-------|
| Total time | 70.5s (real) |
| CPU time | 7m 43s (user) + 2m 1s (sys) |
| Parallel efficiency | ~6.6× CPU utilization |
| dt average | 0.07s |
| Timesteps/second | 13.0 |
| DOFs | 1,755,675 (3 vars × 9 nodes × 65,025 elements) |
| System time ratio | 17% (indicates allocation overhead) |

### Flamegraph Location
```
output/flamegraph/froya_real_data_20260118_131947.svg
```

### Identified Bottlenecks at 65k Scale

| Issue | Impact | Allocations/RHS | Notes |
|-------|--------|-----------------|-------|
| **Face state Vecs** | HIGH | 780,300 | `int_bathy`, `ext_states`, `ext_bathy` |
| **HydrostaticReconstruction2D::new** | MEDIUM | 260,100 | Created per face |
| System time (17%) | HIGH | - | Memory allocation overhead visible |
| Boundary face allocations | MEDIUM | Variable | Ghost state collection |

### Analysis

1. **Face State Allocations**: Inside the parallel loop for each element, we allocate 3 Vecs per face (×4 faces):
   - `int_bathy: Vec<f64>` (n_face_nodes = 3 for P2)
   - `ext_states: Vec<SWEState2D>` (n_face_nodes)
   - `ext_bathy: Vec<f64>` (n_face_nodes)

   Total: 65,025 elements × 4 faces × 3 Vecs = **780,300 allocations per RHS call**

2. **Hydrostatic Reconstruction**: Creating `HydrostaticReconstruction2D::new(g, h_min)` inside each face loop. While lightweight, this is 260,100 constructions per RHS call.

3. **17% System Time**: The high system time (2m out of 10m total CPU) strongly indicates allocation overhead. With mimalloc enabled, this is still significant.

---

## Previous Profile: 600 Elements (2026-01-18)

**Configuration:**
- Mesh: 600 elements (30×20), P2 (9 nodes/element)
- Domain: 59 km × 45 km (Froya-Smola-Hitra region)
- Features: parallel, simd, netcdf, mimalloc
- Duration: 2 minutes (168 steps)

### Top Functions (After Optimization)

| Function | Time % | Notes |
|----------|--------|-------|
| `CoastlineData::load` | 3.26% | Startup only |
| `CombinedSource2D::evaluate` | 1.50% | Source term dispatch |
| `apply_diff_matrix` (SIMD) | 0.85% | Volume term |
| `compute_flux_swe_2d` | 0.69% | Numerical flux |
| `apply_lift` (SIMD) | 0.59% | Surface integral |
| `ManningFriction2D::evaluate` | 0.56% | Bottom friction |
| `MultiBoundaryCondition2D::ghost_state` | 0.44% | BC dispatch |
| **`OceanNestingBC2D::ghost_state`** | **0.38%** | Ocean BC (was 42.9%) |
| `SWEState2D::Add` | 0.26% | RK stage addition |
| `WindStress2D::evaluate` | 0.25% | Wind forcing |

### Critical Optimization: Ocean Model Cell Cache

**Problem:** `OceanModelReader::find_cell` was taking 42.9% of total time due to O(n×m) brute-force search over 26,085 curvilinear grid cells (NorKyst 141×185) for every boundary query.

**Solution:** Added memoization cache to `OceanModelReader`:
```rust
cell_cache: RwLock<HashMap<(i32, i32), Option<CellLookup>>>
```

- Quantized (lon, lat) keys at ~1m resolution (1e-5 degrees)
- Thread-safe with read-biased RwLock
- First query computes, subsequent queries hit cache

**Results:**

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| `find_cell` | 42.8% | 0.14% | **306x** |
| `ghost_state` (total BC) | 42.9% | 0.38% | **113x** |

**Files Modified:**
- `src/io/netcdf_io.rs` - Added `cell_cache`, `find_cell_cached()`, `quantize_coords()`

---

## Parallel Performance

### Adaptive Dispatch Threshold

The `compute_rhs_swe_2d_adaptive` function automatically selects:
- **Serial SIMD** for < 2000 elements (lower overhead)
- **Parallel SIMD** for ≥ 2000 elements (multi-core scaling)

### Benchmark Results (10,000 elements, P3)

| Variant | Time | vs Serial |
|---------|------|-----------|
| Serial | 13.0 ms | 1.0x |
| Parallel | 3.4 ms | **3.8x** |
| Parallel Optimized | 3.9 ms | **3.3x** |
| Adaptive | 4.1 ms | **3.2x** |

### ThreadWorkspace Optimization

Parallel RHS was initially **2-122x slower** than serial due to ~22 Vec allocations per element inside the parallel closure.

**Fix:** Created `ThreadWorkspace` struct with pre-allocated buffers, used with `for_each_init` for per-thread allocation:

```rust
pub struct ThreadWorkspace {
    pub flux_h: Vec<f64>,
    pub flux_hu: Vec<f64>,
    pub flux_hv: Vec<f64>,
    // ... 22 pre-allocated buffers
}
```

---

## Build Configuration

### Cargo.toml Release Profile
```toml
[profile.release]
opt-level = 3           # Maximum optimization
lto = "fat"             # Full link-time optimization
codegen-units = 1       # Single codegen unit for better optimization
panic = "abort"         # Smaller binary, slightly faster
debug = 1               # Line tables for profiling
```

### .cargo/config.toml
```toml
[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=native"]
```

### Default Features
```toml
default = ["parallel", "simd", "netcdf", "mimalloc"]
```

---

## Profiling Commands

```bash
# Generate flamegraph
./scripts/flamegraph.sh froya_real_data

# Interactive profiler (Firefox Profiler UI)
./scripts/samply.sh froya_real_data

# Quick CPU statistics
./scripts/perf-stat.sh froya_real_data

# Run benchmarks
cargo bench --bench rhs_computation_bench --features "parallel,simd"
```

---

## Historical Optimizations

### 2026-01-18: Ocean Model Cell Cache
- **Impact:** 42.9% → 0.38% for ocean BC (**113x speedup**)
- **Cause:** O(n×m) cell search for curvilinear NorKyst grid
- **Fix:** Memoization cache with quantized coordinate keys

### 2026-01-17: Parallel Allocation Fix
- **Impact:** Parallel 2-122x slower → 3.8x faster than serial
- **Cause:** Vec allocations inside parallel closure
- **Fix:** `ThreadWorkspace` with `for_each_init`

### 2026-01-09: Manning Friction
- **Impact:** -22% samples in friction calculation
- **Cause:** `powf(1.0/3.0)` slow
- **Fix:** Use `cbrt()` instead

### 2026-01-09: Wetting/Drying
- **Impact:** -46% to -100% in wet/dry functions
- **Cause:** Multiple passes, no early exit
- **Fix:** Fused passes with early-exit for wet/dry elements

### 2026-01-09: Hydrostatic Reconstruction
- **Impact:** -70% in reconstruction
- **Cause:** Complex velocity computation
- **Fix:** Ratio-based velocity preservation, `#[inline]`

---

## Optimization Attempts at 65k Scale (2026-01-18)

### Attempted: Face State Pre-allocation - REVERTED

**Hypothesis:** Eliminating ~780k small Vec allocations per RHS would reduce system time.

**Implementation:** Added face state buffers to ThreadWorkspace:
- SoA format (separate h, hu, hv arrays) - **FAILED** (12% slower due to cache locality loss)
- AoS format (Vec<SWEState2D>) - **FAILED** (16% slower due to extra copy overhead)

**Finding:** The small Vec allocations (72 bytes for 3×SWEState2D) are **NOT a bottleneck**.
mimalloc handles them efficiently, and pre-allocating into workspace buffers hurts cache
locality because:
1. Original: `collect()` creates contiguous Vec, then iterate once
2. Workspace: Write to buffer, then read back from buffer (2× memory traffic)

**Conclusion:** Keep original Vec allocation pattern. The 17% system time is NOT from
face state allocations.

### Kept: Hydrostatic Reconstruction Outside Face Loop

Moved `HydrostaticReconstruction2D::new(g, h_min)` from inside face loop to once per element.
- Impact: Neutral (struct is only 16 bytes, compiler likely optimized this anyway)
- Code clarity: Slightly better

### Actual Bottleneck Analysis

With mimalloc and modern allocators, the **actual bottleneck at 65k scale** is likely:
1. **Memory bandwidth** - 70MB+ working set doesn't fit in L3 cache
2. **Neighbor access pattern** - Scattered reads from neighbor elements (poor spatial locality)
3. **Parallel synchronization** - RwLock in ocean BC cell cache

Future optimization directions:
- **Morton/Hilbert ordering** - Reorder elements for better spatial locality
- **Prefetching** - Software prefetch for neighbor states
- **GPU offload** - Move RHS computation to CUDA/Metal

### Performance at 65k Scale

| Metric | Value |
|--------|-------|
| Elements | 65,025 (255×255) |
| Wet elements | 38,212 (58.8%) |
| Timesteps/second | 12-13 |
| Time per RHS call | ~2.5-3 ms |
| Variance | ±15% (thermal/load dependent) |
