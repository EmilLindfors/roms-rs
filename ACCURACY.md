# Accuracy Verification

This document records the verified numerical accuracy of the DG solver.

## Convergence Rates

The DG method achieves (N+1) order accuracy for polynomial order N with smooth solutions.

### Advection Equation: `du/dt + a*du/dx = 0`

Test configuration:
- Domain: [0, 2]
- Initial condition: `sin(πx)`
- Boundary: Dirichlet inflow (exact solution)
- Time integration: SSP-RK3 with time-aware BC evaluation

| Order | Expected | Observed | Test Parameters |
|-------|----------|----------|-----------------|
| P1    | 2.0      | 1.99     | n=[10,20,40,80], t=0.5, CFL=0.3 |
| P3    | 4.0      | 4.00     | n=[5,10,20,40], t=0.5, CFL=0.3 |
| P5    | 6.0      | 5.89     | n=[3,6,12,24], t=0.1, CFL=0.05 |

### Detailed Results

**P1 (2nd order)**
```
n= 10: error=1.22e-1
n= 20: error=3.39e-2, ratio=3.59, order=1.84
n= 40: error=8.76e-3, ratio=3.87, order=1.95
n= 80: error=2.21e-3, ratio=3.97, order=1.99
```

**P3 (4th order)**
```
n=  5: error=2.35e-3
n= 10: error=1.58e-4, ratio=14.82, order=3.89
n= 20: error=9.70e-6, ratio=16.32, order=4.03
n= 40: error=6.08e-7, ratio=15.96, order=4.00
```

**P5 (6th order)**
```
n=  3: error=1.32e-4
n=  6: error=2.79e-6, ratio=47.15, order=5.56
n= 12: error=3.86e-8, ratio=72.28, order=6.18
n= 24: error=6.52e-10, ratio=59.18, order=5.89
```

## Operator Accuracy

### Differentiation Matrix
- Exactly differentiates polynomials up to degree N
- Verified for orders 1-5

### Quadrature (Gauss-Lobatto)
- Exact for polynomials up to degree 2N-1
- Verified for orders 1-10

### Vandermonde Matrix
- Condition number acceptable for orders ≤ 10
- Nodal-modal roundtrip error < 1e-12

## Known Limitations

### Time Integration
- SSP-RK3 is 3rd order in time
- For high spatial orders (P5+), temporal error can dominate on fine meshes
- Mitigation: Use smaller CFL or shorter integration times for P5+ tests

### Boundary Conditions
- Time-dependent BCs require `ssp_rk3_step_timed` for high-order accuracy
- Using `ssp_rk3_step` (time-unaware) degrades convergence to ~1st order

## Running Convergence Tests

```bash
# Run all convergence tests with output
cargo test --test convergence_test -- --nocapture

# Run specific order test
cargo test test_convergence_p3 -- --nocapture
```

## Example Accuracy

Running the advection example (`cargo run --example advection_1d`):

| Configuration | Final L2 Error | Final L∞ Error |
|--------------|----------------|----------------|
| P3, 20 elem, t=1.0 | 9.88e-6 | 2.72e-5 |

---

## 2D Advection Equation

Test configuration:
- Domain: [0, 1] × [0, 1] (periodic)
- Velocity: (1, 1) (diagonal advection)
- Initial condition: `sin(2πx) * sin(2πy)`
- Time integration: SSP-RK3
- Mesh: uniform quadrilaterals

### Convergence Rates

| Order | Expected | Observed | Test Parameters |
|-------|----------|----------|-----------------|
| P2    | 3.0      | 2.98     | n=[4,8,16,32]×[4,8,16,32], t=0.1, CFL=0.3 |
| P3    | 4.0      | 4.00     | n=[4,8,16,32]×[4,8,16,32], t=0.1, CFL=0.3 |

### Detailed Results

**P2 (3rd order)**
```
4×4:   error=6.45e-3
8×8:   error=8.88e-4, ratio=7.27, order=2.86
16×16: error=1.14e-4, ratio=7.79, order=2.96
32×32: error=1.44e-5, ratio=7.92, order=2.98
```

**P3 (4th order)**
```
4×4:   error=5.43e-4
8×8:   error=3.53e-5, ratio=15.4, order=3.94
16×16: error=2.21e-6, ratio=16.0, order=4.00
32×32: error=1.38e-7, ratio=16.0, order=4.00
```

---

## 2D Shallow Water Equations

### Well-Balanced Property (Lake-at-Rest)

Test configuration:
- Domain: [0, 10] × [0, 10]
- Initial condition: h = 5.0, u = v = 0
- Boundary: Reflective
- Bathymetry: flat (B = 0)

| Order | Max RHS Residual | Mass Conservation |
|-------|------------------|-------------------|
| P2    | < 1e-10          | exact (periodic)  |
| P3    | < 1e-10          | exact (periodic)  |

The lake-at-rest state is preserved to machine precision.

### Conservation Properties

For periodic domains without source terms:

| Property | Conservation Error |
|----------|-------------------|
| Mass (h) | < 1e-11 (relative) |
| X-momentum (hu) | < 1e-10 |
| Y-momentum (hv) | < 1e-10 |

### Well-Balancing with Variable Bathymetry

Periodic 20 km domain (12×12), B = −200 + 150·sin(2πx/L)·cos(2πy/L) m, η = 0.3 m.
Beds: smooth nodal (degree ≥ p), cell-averaged (jumps at every face), and
rough (smooth + ±15 m per-node noise). Also on a walled 8×5 domain.

| Formulation | Bathymetry | Max \|d(hu)/dt\| at lake-at-rest (m²/s²) |
|-------------|------------|--------------------------------------|
| Standard + `BathymetrySource2D` + reconstruction | smooth nodal, P2 | 2.5 (not balanced: needs deg B ≤ p/2) |
| Standard + reconstruction | cell-averaged, P1–P3 | ≤ 3e-12 |
| Entropy-stable split form | smooth / cell-averaged / rough, P1–P4 | ≤ 4e-12 |

Mass conservation (slope-correlated flow, cell-averaged B with hydrostatic
reconstruction): relative d(mass)/dt ≈ 1e-20 (was −5.2e-5 per second before the
interior-flux fix). Split form, all beds: ≤ 6e-20.

### Entropy-Stable Split Form (Wintermeyer et al. 2017)

`SWEFormulation2D::EntropyStable` / `EntropyConservative` (flux differencing
with the Wintermeyer two-point flux, collocated bed source, interface bed term
½g h⁻[[B]]). Discrete entropy rate ∫ w·dq/dt on periodic and walled domains:

| Formulation | Relative entropy rate |
|-------------|-----------------------|
| EntropyConservative | ≤ 3e-14 (round-off) |
| EntropyStable | < 0 whenever (η, u, v) jump across faces |

Nonlinear manufactured solution over variable bathymetry on [0, √2]²
(H = 7 + cos(2π√2 x)·sin(2π√2 y)·cos(2πt), b = 2 + ½ sin(√2πx) + ½ sin(√2πy),
(u, v) = (½, 3/2)), t = 0.1, L2 error in h:

| Order | 4×4 | 8×8 | 16×16 | Observed order |
|-------|-----|-----|-------|----------------|
| P2 | 2.64e-1 | 3.94e-2 | 4.74e-3 | 3.06 |
| P3 | 3.98e-2 | 3.17e-3 | 1.98e-4 | 4.00 |

### Numerical Flux Comparison

For smooth solutions, all flux types produce similar results:

| Flux Type | Max Difference (relative) | Notes |
|-----------|---------------------------|-------|
| Roe       | reference                 | Most accurate for linear problems |
| HLL       | < 0.1                    | More diffusive, robust |
| Rusanov   | < 0.1                    | Most diffusive, most robust |

### Coriolis Verification

Norwegian coast configuration (f = 1.2×10⁻⁴ s⁻¹):
- Source term correctly deflects x-momentum to negative v
- d(hv)/dt = -f × h × u verified to < 1e-8 error

---

## 2D Operator Accuracy

### Differentiation Matrices (Dr, Ds)
- Exactly differentiate polynomials up to degree N
- Verified for orders 1-5
- Tensor-product structure: Dr = I ⊗ Dr_1d, Ds = Ds_1d ⊗ I

### 2D Quadrature (Tensor-Product GLL)
- Weight sum = 4 (reference element area)
- Exact for polynomials up to degree 2N-1 in each direction
- Verified for orders 1-5

### 2D Vandermonde Matrix
- Condition number acceptable for orders ≤ 5
- Nodal-modal roundtrip error < 1e-12
- Gradient matrices Vr, Vs computed analytically

### Geometric Factors
- Jacobian determinant positive for all valid meshes
- Surface Jacobians and normals consistent with element orientation
- Verified on uniform, stretched, and channel meshes

---

## Running 2D Tests

```bash
# Run 2D convergence tests
cargo test --test convergence_test test_convergence_2d -- --nocapture

# Run 2D SWE tests
cargo test --test swe_2d_test -- --nocapture

# Split-form (entropy-stable) SWE convergence with bathymetry
cargo test --test convergence_test split_form -- --nocapture

# Run all 2D-related tests
cargo test 2d -- --nocapture
```
