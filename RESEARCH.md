# Implementing discontinuous Galerkin methods in Rust for Norwegian coastal ocean modeling

Building a production-quality DG solver for Norwegian fjords in Rust is achievable today, though it requires assembling components rather than using an existing framework. The Rust ecosystem provides **excellent foundational infrastructure** (faer for linear algebra, rayon for parallelism, netcdf/hdf5 for I/O) but lacks a DG-specific framework comparable to Trixi.jl or deal.II. For GPU acceleration requiring the double precision critical for ocean modeling, **cudarc with direct CUDA access** is the most capable option, as wgpu/WGSL lacks f64 support. The architectural patterns from Trixi.jl (Julia) and Thetis (Firedrake/Python) are most transferable to a Rust implementation, emphasizing traits for physics abstraction, sum-factorization for tensor-product elements, and entropy-stable schemes with positivity-preserving limiters for robust tracer transport.

## The Rust numerical computing ecosystem has strong foundations but no DG framework

The Rust ecosystem for numerical PDEs is **nascent but maturing rapidly**. No production DG crate exists, representing the most significant gap compared to Julia's Trixi.jl or C++'s deal.II. However, the building blocks are substantial.

**faer** emerges as the recommended linear algebra foundation with **1 million+ downloads** and benchmarks showing it outperforms both nalgebra and ndarray for medium-to-large matrices. For 1024×1024 matrix multiplication, faer achieves 33.9ms single-threaded and 6.6ms parallel, compared to nalgebra's 39.1ms. It supports both dense and sparse operations, parallel execution specified per-call, and SIMD optimizations including AVX-512 on nightly. For sparse matrices essential to DG global systems, **sprs** provides CSR/CSC formats with ~125,000 monthly downloads.

Mesh handling presents more challenges. The gmsh-sys bindings are **over 5 years outdated**, but **tritet** (wrapping Triangle and TetGen) offers active Delaunay triangulation support. The **spade** crate provides mature constrained Delaunay with 8+ million downloads. For scientific I/O, both **netcdf** and **hdf5-metno** (the actively-maintained fork from the Norwegian Meteorological Institute) offer thread-safe bindings with ndarray integration.

Time integration through **diffsol** deserves special mention—it provides explicit RK, BDF for stiff problems, and SDIRK/ESDIRK methods with both nalgebra and faer backends, making it well-suited for IMEX schemes needed in ocean modeling.

## GPU backends require careful selection based on precision requirements

Double precision is non-negotiable for coastal ocean modeling due to the sensitivity of baroclinic pressure gradients to numerical errors. This requirement dramatically narrows GPU options.

**cudarc provides the most capable path** for f64 computations, offering safe wrappers around the CUDA driver API, nvrtc runtime compilation, and critically, **cuBLAS batched GEMM operations** ideal for DG's element-local matrix structure (typically 10-100 DOF per element). The cuSPARSE library handles sparse assembly patterns, while cuSOLVER provides direct solvers. The drawback is NVIDIA-only deployment.

**rust-gpu** offers an intriguing alternative: writing GPU kernels in Rust that compile to SPIR-V. It has reached production readiness for many use cases, with f64 support via SPIR-V extensions (dependent on Vulkan driver support). This enables full Rust expressiveness—traits for different element types, compile-time dimension checking—while maintaining GPU performance. The tradeoff is Vulkan-only targeting and requiring nightly Rust.

**wgpu and WGSL are unsuitable for the core solver** because WGSL compute shaders do not support f64. This limitation is fundamental to the WebGPU specification. While wgpu provides excellent portability across Windows/Linux/macOS/Web, it cannot deliver the precision required for baroclinic computations. The only workaround—emulating f64 with two f32s—incurs 2-4x performance penalties.

**Burn**, despite its excellent CubeCL kernel architecture with hierarchical Batch→Global→Stage→Tile matrix multiplication, focuses on ML workloads with f32/f16 optimization. Its CUDA backend could potentially support f64, but this is not the primary use case.

The recommended hybrid approach: use **cudarc for critical f64 DG kernels** (element integration, flux computation, pressure gradients) while potentially leveraging Burn's autodiff capabilities for adjoint methods if f32 precision suffices for sensitivity analysis.

## Production codes reveal essential architectural patterns for DG implementation

Analyzing deal.II, Firedrake, DUNE, and Trixi.jl reveals converging architectural wisdom that translates well to Rust's type system.

**Dimension-independent programming** through templates (C++) or parametric types translates directly to Rust's const generics. A `struct Point<const DIM: usize>` pattern enables compile-time dimension specialization with zero runtime overhead, allowing the same DG code to handle 2D and 3D without performance penalties.

**Physics abstraction through traits** mirrors Trixi.jl's multiple dispatch approach. Defining `ConservationLaw<const NVARS: usize, const DIM: usize>` with methods for `flux`, `max_wavespeed`, and `source_terms` separates numerical machinery from physical equations. This enables implementing shallow water, Navier-Stokes, or tracer transport by simply providing new trait implementations.

**Matrix-free operator evaluation** is essential for high-order DG (p≥3). Rather than storing the global sparse matrix (memory-bandwidth limited), matrix-free approaches compute operator actions on-the-fly using sum-factorization. This exploits tensor-product structure: evaluating a 3D operator costs **O(p⁴) rather than O(p⁶)**. deal.II's implementation achieves 60% of arithmetic peak on Intel CPUs. For Rust, this means storing only geometry (Jacobians, metrics) and precomputed 1D differentiation/interpolation matrices.

**Trixi.jl's solver architecture** provides the most directly transferable pattern:
```
SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
→ semidiscretize(semi, tspan)
→ solve(ode, time_integrator)
```
This separates spatial discretization from time integration, enabling integration with external ODE libraries (Julia's DifferentialEquations.jl, Rust's diffsol).

GPU parallelization differs fundamentally from CPU: where CPU code batches cells together for SIMD vectorization, GPU code parallelizes over DOFs within elements. deal.II's CUDA implementation assigns each thread to a different DOF rather than cell batch. This suggests designing abstractions (similar to deal.II's `VectorizedArray`) that hide architecture-specific parallelization strategies.

## Ocean-specific DG codes provide practical implementation recipes

**Thetis** (built on Firedrake) represents the most mature true-DG coastal ocean model and offers the clearest implementation template. It uses linear discontinuous elements (P1DG) for tracers with **Kuzmin slope limiters** and Strong Stability Preserving (SSP) Runge-Kutta time integration. This combination achieves second-order accuracy in smooth regions while preventing overshoots—critical for maintaining physical bounds on salinity and temperature.

For the baroclinic pressure gradient—the notorious challenge in sigma-coordinate ocean modeling—Thetis computes baroclinic head r = (1/ρ₀)∫ρ'dz and evaluates the internal pressure gradient F_pg = g∇r using separate function spaces for baroclinic head (P1DG × P2) versus pressure gradient. This careful treatment is essential for Norwegian fjords where steep topography creates large pressure gradient errors in naive implementations.

**Mode splitting** for barotropic-baroclinic separation remains standard practice. Barotropic surface gravity waves propagate at ~200 m/s while baroclinic internal modes move at ~2 m/s—two orders of magnitude difference. Thetis uses G-term coupling: solve the 3D system with G=0, compute depth-average correction, then enforce that the vertical integral of perturbation velocity vanishes.

Comparing ADCIRC (continuous Galerkin with GWCE), FVCOM (finite volume), and true DG approaches reveals why DG is preferable for fjord modeling: research by Kubatko et al. showed **DG solution errors are typically an order of magnitude smaller** than CG approaches with better convergence rates. DG also provides automatic local conservation that CG-GWCE cannot guarantee.

## Norwegian fjords present specific numerical challenges requiring targeted solutions

The defining characteristics of Norwegian fjords—steep walls, strong stratification, narrow passages with sills, episodic deep water renewal—demand particular numerical attention.

**Baroclinic pressure gradient errors** are amplified by steep bathymetry. Studies of Hardangerfjord found standard methods produce significant spurious currents even at high resolution. The **balanced methods** developed by Berntsen explicitly balance density gradients and reduce errors by orders of magnitude for slope parameters exceeding 0.2. For a Rust implementation, this means implementing fourth-order or higher pressure gradient approximations with explicit balancing terms.

**Vertical coordinate choice** favors sigma (terrain-following) for fjords despite pressure gradient issues because it provides smooth bottom representation and natural boundary layer resolution. The NorKyst800 operational model uses 35 sigma levels. Hybrid approaches like SCHISM's LSC² (Localized Sigma Coordinates with shaved cells) or generalized s-coordinates allow optimization of vertical distribution while reducing pressure gradient errors. Clustering levels near surface and bottom captures the freshwater lens from river input and bottom boundary layer.

**Wetting and drying** matters for tidal flats at fjord heads. DG-specific algorithms classify elements as fully wet, fully dry, or partially wet, applying slope modification in transitional elements. The SLIM model's implicit approach uses a threshold depth with blending parameter, preserving both local mass conservation and well-balanced properties at wet/dry interfaces. Positivity-preserving schemes monitor water depth at each Runge-Kutta stage and adjust surface elevations to redistribute mass locally.

**Boundary conditions** are critical for fjord-coastal interaction. The **TST-OBC (Tidal and Subtidal)** approach resolves both components simultaneously by decomposing solutions into global/local and tidal/subtidal parts. NorKyst800 uses TPXO global inverse tide solutions for tidal harmonics combined with subtidal forcing from larger-scale models (NEMO). River inputs from NVE (Norwegian Water Resources and Energy Directorate) drive the estuarine circulation fundamental to fjord dynamics.

## Key research advances enable robust, efficient DG implementations

Recent algorithmic developments significantly improve DG solver robustness and efficiency.

**Entropy-stable DG schemes** (Chen & Shu 2017, Chan 2018) provide nonlinear stability without excessive artificial viscosity. For shallow water equations, Wu et al. (2024) combined entropy stability with subcell positivity preservation using convex limiting—directly applicable to coastal modeling where both robustness and physical bounds matter. Implementation uses summation-by-parts (SBP) operators with entropy-conservative two-point fluxes and interface dissipation.

**Hybridized DG (HDG)** reduces global coupling for implicit solves. Kang, Giraldo, and Bui-Thanh (2019) developed IMEX HDG-DG specifically for shallow water: gravity waves treated implicitly via HDG, advection explicitly via DG. This achieves high-order accuracy in space/time, allows larger timesteps, requires only one linear solve per stage, and produces smaller/sparser systems than pure DG. For ocean models, this is ideal for handling stiff vertical diffusion.

**GPU acceleration** has matured considerably. Abdi et al. (2019) achieved **15x speedup** using K20X GPUs versus 16-core CPUs with 90% weak scaling efficiency on 16,384 GPUs. Souza et al. (2023) report 20-50% of peak GPU floating-point performance for DG ocean models, with single GPUs comparable to 500+ CPU cores. The key optimization strategies: memory coalescing for unstructured meshes, vectorization over elements/faces, kernel fusion to reduce memory traffic, and communication-computation overlap for multi-GPU.

**Matrix-free implementations** (Kronbichler & Kormann 2019) achieve up to 60% of arithmetic peak through sum-factorization with even-odd decomposition. This paper, implemented in deal.II, provides the algorithmic blueprint for efficient high-order DG.

## Concrete recommendations for building a Rust DG coastal ocean solver

Based on this research, here is the recommended technology stack and architectural approach:

**Linear algebra and parallelism:** Use faer for dense operations and sprs for sparse matrices. rayon provides shared-memory parallelism for element-local computations; rsmpi enables distributed computing. diffsol with faer backend handles time integration including IMEX schemes.

**GPU computation:** cudarc for core f64 DG kernels on NVIDIA hardware. Structure kernels around cuBLAS batched GEMM for element matrices and cuSPARSE for global assembly. Consider rust-gpu for portable Vulkan/SPIR-V kernels if broader hardware support is needed and driver f64 support is available.

**Mesh and I/O:** tritet for mesh generation, custom readers for Gmsh MSH format, hdf5-metno for checkpointing and restart (actively maintained by the Norwegian Meteorological Institute), netcdf for standard oceanographic data formats.

**Core abstractions to implement:**
- `Mesh<const DIM: usize>` trait with cell/face iterators and topology
- `Basis<const DIM: usize>` trait for polynomial evaluation with tensor-product specialization
- `ConservationLaw<const NVARS: usize, const DIM: usize>` trait for physics
- `NumericalFlux` trait for Rusanov, HLL, entropy-stable fluxes
- `SemiDiscretization` interface connecting to diffsol time integrators

**Numerical choices for fjord modeling:** P1DG elements with Kuzmin slope limiters and SSP-RK time integration. Generalized sigma coordinates with 35-40 levels clustered near surface/bottom. Balanced pressure gradient methods essential for steep bathymetry. Mode-split time stepping with implicit vertical diffusion. TST-OBC or Flather/Chapman boundary conditions with relaxation zones.

**Essential papers for implementation:** Hesthaven & Warburton "Nodal Discontinuous Galerkin Methods" provides the foundational algorithms. Kronbichler & Kormann (2019) details matrix-free implementation. Kang, Giraldo & Bui-Thanh (2019) covers IMEX HDG-DG for shallow water. The Thetis papers (Kärnä et al. 2018, 2020) provide ocean-specific DG recipes including turbulence closure.

## Conclusion

A Rust-based DG solver for Norwegian coastal waters is both feasible and well-motivated. The language's ownership system guarantees data-race freedom crucial for parallel scientific computing, while zero-cost abstractions enable high-level code without performance penalties. The ecosystem gap—no existing DG framework—is substantial but not prohibitive given the strong foundational crates.

The critical path forward involves: (1) implementing a core DG framework with tensor-product elements and sum-factorization, taking architectural patterns from Trixi.jl; (2) developing CUDA kernels via cudarc for GPU acceleration, prioritizing batched element operations; (3) implementing entropy-stable schemes with positivity-preserving limiters for robustness; and (4) careful treatment of baroclinic pressure gradients using balanced methods essential for steep fjord topography.

Thetis on Firedrake demonstrates what production DG coastal ocean modeling looks like—the goal would be achieving similar capabilities with Rust's performance and safety benefits. The Norwegian Meteorological Institute's maintenance of hdf5-metno suggests institutional interest in Rust for scientific computing that could support such development.