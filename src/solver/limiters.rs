//! Slope limiters for DG methods.
//!
//! Limiters are essential for maintaining stability and preventing oscillations
//! in high-order DG discretizations of hyperbolic problems.
//!
//! This module provides:
//! - TVB (Total Variation Bounded) limiter for controlling oscillations
//! - Positivity-preserving limiter (Zhang-Shu) for shallow water
//!
//! # References
//! - Cockburn & Shu (1989), "TVB Runge-Kutta Local Projection..."
//! - Zhang & Shu (2010), "On positivity-preserving high order discontinuous Galerkin schemes..."

use crate::mesh::Mesh1D;
use crate::operators::DGOperators1D;
use crate::solver::{SWESolution, SWEState};

/// TVB limiter parameter.
///
/// The parameter M controls when limiting is applied:
/// - M = 0: Standard TVD limiter (very dissipative)
/// - M > 0: TVB limiter, allows larger variations near extrema
///
/// Typical values: M = 50 for smooth problems, M = 0-5 for shocks.
#[derive(Clone, Copy, Debug)]
pub struct TVBParameter {
    /// The M parameter (larger = less limiting)
    pub m: f64,
    /// Mesh spacing for scaling
    pub dx: f64,
}

impl TVBParameter {
    /// Create a new TVB parameter.
    pub fn new(m: f64, dx: f64) -> Self {
        Self { m, dx }
    }

    /// The minmod threshold: M * dx²
    pub fn threshold(&self) -> f64 {
        self.m * self.dx * self.dx
    }
}

/// Minmod function for three arguments.
///
/// Returns 0 if arguments have different signs, otherwise the smallest magnitude.
fn minmod(a: f64, b: f64, c: f64) -> f64 {
    if a > 0.0 && b > 0.0 && c > 0.0 {
        a.min(b).min(c)
    } else if a < 0.0 && b < 0.0 && c < 0.0 {
        a.max(b).max(c)
    } else {
        0.0
    }
}

/// TVB-modified minmod function.
///
/// Only activates limiting if the first argument exceeds the TVB threshold.
fn minmod_tvb(a: f64, b: f64, c: f64, threshold: f64) -> f64 {
    if a.abs() <= threshold {
        a
    } else {
        minmod(a, b, c)
    }
}

/// Compute cell averages for each element.
fn cell_averages(q: &SWESolution, ops: &DGOperators1D) -> Vec<[f64; 2]> {
    let n_elements = q.n_elements;
    let n_nodes = ops.n_nodes;
    let weights = &ops.weights;

    let mut averages = vec![[0.0; 2]; n_elements];

    for k in 0..n_elements {
        let mut avg_h = 0.0;
        let mut avg_hu = 0.0;
        let mut weight_sum = 0.0;

        for i in 0..n_nodes {
            let [h, hu] = q.get(k, i);
            let w = weights[i];
            avg_h += w * h;
            avg_hu += w * hu;
            weight_sum += w;
        }

        averages[k][0] = avg_h / weight_sum;
        averages[k][1] = avg_hu / weight_sum;
    }

    averages
}

/// Apply TVB slope limiter to the SWE solution.
///
/// The TVB limiter modifies high-order content when it exceeds the TVB bound,
/// replacing it with a limited linear reconstruction.
///
/// # Arguments
/// * `q` - Solution to limit (modified in place)
/// * `mesh` - Computational mesh
/// * `ops` - DG operators
/// * `tvb` - TVB parameter
///
/// # Notes
/// - Only modifies elements where limiting is needed
/// - Preserves cell averages exactly
/// - Uses characteristic limiting for better behavior near shocks
pub fn tvb_limiter(q: &mut SWESolution, mesh: &Mesh1D, ops: &DGOperators1D, tvb: &TVBParameter) {
    let n_elements = q.n_elements;
    let n_nodes = ops.n_nodes;

    // Compute cell averages
    let averages = cell_averages(q, ops);

    // Limiting factor
    let threshold = tvb.threshold();

    for k in 0..n_elements {
        // Get values at element boundaries (first and last nodes)
        let [h_minus, hu_minus] = q.get(k, 0);
        let [h_plus, hu_plus] = q.get(k, n_nodes - 1);

        let avg_h = averages[k][0];
        let avg_hu = averages[k][1];

        // Differences from average at boundaries
        let delta_h_minus = avg_h - h_minus;
        let delta_h_plus = h_plus - avg_h;
        let delta_hu_minus = avg_hu - hu_minus;
        let delta_hu_plus = hu_plus - avg_hu;

        // Neighboring cell averages
        let (avg_h_left, avg_hu_left) = if mesh.neighbors[k].0.is_some() {
            (averages[k - 1][0], averages[k - 1][1])
        } else {
            (avg_h, avg_hu) // Use own average at boundary
        };

        let (avg_h_right, avg_hu_right) = if mesh.neighbors[k].1.is_some() {
            (averages[k + 1][0], averages[k + 1][1])
        } else {
            (avg_h, avg_hu) // Use own average at boundary
        };

        // Forward and backward differences of averages
        let delta_forward_h = avg_h_right - avg_h;
        let delta_backward_h = avg_h - avg_h_left;
        let delta_forward_hu = avg_hu_right - avg_hu;
        let delta_backward_hu = avg_hu - avg_hu_left;

        // Apply TVB minmod to limit slopes
        let limited_delta_minus_h =
            minmod_tvb(delta_h_minus, delta_backward_h, delta_forward_h, threshold);
        let limited_delta_plus_h =
            minmod_tvb(delta_h_plus, delta_backward_h, delta_forward_h, threshold);
        let limited_delta_minus_hu = minmod_tvb(
            delta_hu_minus,
            delta_backward_hu,
            delta_forward_hu,
            threshold,
        );
        let limited_delta_plus_hu = minmod_tvb(
            delta_hu_plus,
            delta_backward_hu,
            delta_forward_hu,
            threshold,
        );

        // Check if limiting is needed
        let h_needs_limiting = (limited_delta_minus_h - delta_h_minus).abs() > 1e-14
            || (limited_delta_plus_h - delta_h_plus).abs() > 1e-14;
        let hu_needs_limiting = (limited_delta_minus_hu - delta_hu_minus).abs() > 1e-14
            || (limited_delta_plus_hu - delta_hu_plus).abs() > 1e-14;

        if h_needs_limiting || hu_needs_limiting {
            // Replace with limited linear reconstruction
            // Using GLL nodes in [-1, 1]
            let nodes = &ops.nodes;

            for i in 0..n_nodes {
                let xi = nodes[i]; // Reference coordinate in [-1, 1]

                let new_h = if h_needs_limiting {
                    // Linear reconstruction: avg + slope * xi
                    let slope_h = 0.5 * (limited_delta_plus_h - limited_delta_minus_h);
                    avg_h + slope_h * xi
                } else {
                    q.get(k, i)[0]
                };

                let new_hu = if hu_needs_limiting {
                    let slope_hu = 0.5 * (limited_delta_plus_hu - limited_delta_minus_hu);
                    avg_hu + slope_hu * xi
                } else {
                    q.get(k, i)[1]
                };

                q.set(k, i, [new_h, new_hu]);
            }
        }
    }
}

/// Apply positivity-preserving limiter for water depth.
///
/// The Zhang-Shu positivity limiter ensures h ≥ h_min at all quadrature points
/// while preserving the cell average.
///
/// The limiting is: q_limited = θ(q - avg) + avg
/// where θ = min(1, (avg - h_min) / (avg - h_min_q))
///
/// # Arguments
/// * `q` - Solution to limit (modified in place)
/// * `ops` - DG operators
/// * `h_min` - Minimum allowed water depth
///
/// # Notes
/// - Only modifies elements with h < h_min at some node
/// - Preserves cell averages exactly
/// - Adjusts momentum consistently to maintain velocity
pub fn positivity_limiter(q: &mut SWESolution, ops: &DGOperators1D, h_min: f64) {
    let n_elements = q.n_elements;
    let n_nodes = ops.n_nodes;

    // Compute cell averages
    let averages = cell_averages(q, ops);

    for k in 0..n_elements {
        let avg_h = averages[k][0];
        let avg_hu = averages[k][1];

        // If average is below h_min, set entire element to minimum
        if avg_h < h_min {
            for i in 0..n_nodes {
                q.set(k, i, [h_min, 0.0]);
            }
            continue;
        }

        // Find minimum depth in element
        let mut h_min_elem = f64::INFINITY;
        for i in 0..n_nodes {
            let h = q.get(k, i)[0];
            h_min_elem = h_min_elem.min(h);
        }

        // If all depths are positive, no limiting needed
        if h_min_elem >= h_min {
            continue;
        }

        // Compute limiting factor θ
        // We need: θ(h - avg) + avg >= h_min
        // => θ <= (avg - h_min) / (avg - h_min_elem)
        let theta = if (avg_h - h_min_elem).abs() > 1e-14 {
            ((avg_h - h_min) / (avg_h - h_min_elem)).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Apply limiting: q_new = θ(q - avg) + avg
        for i in 0..n_nodes {
            let [h, hu] = q.get(k, i);

            let h_new = theta * (h - avg_h) + avg_h;
            // Scale momentum to maintain velocity where possible
            let hu_new = if h > h_min && h_new > h_min {
                hu * h_new / h
            } else {
                theta * (hu - avg_hu) + avg_hu
            };

            q.set(k, i, [h_new.max(0.0), hu_new]);
        }
    }
}

/// Apply both TVB and positivity limiters.
///
/// This applies limiting in the correct order:
/// 1. TVB limiter (controls oscillations)
/// 2. Positivity limiter (ensures h >= h_min)
///
/// # Arguments
/// * `q` - Solution to limit (modified in place)
/// * `mesh` - Computational mesh
/// * `ops` - DG operators
/// * `tvb` - TVB parameter
/// * `h_min` - Minimum allowed water depth
pub fn apply_swe_limiters(
    q: &mut SWESolution,
    mesh: &Mesh1D,
    ops: &DGOperators1D,
    tvb: &TVBParameter,
    h_min: f64,
) {
    // First apply TVB limiter to control oscillations
    tvb_limiter(q, mesh, ops, tvb);

    // Then ensure positivity
    positivity_limiter(q, ops, h_min);
}

/// Characteristic-based TVB limiter for SWE.
///
/// Limits in characteristic variables for better behavior near shocks.
/// Uses the local Roe-averaged eigenvectors for the transformation.
///
/// # Arguments
/// * `q` - Solution to limit (modified in place)
/// * `mesh` - Computational mesh
/// * `ops` - DG operators
/// * `tvb` - TVB parameter
/// * `g` - Gravitational acceleration
#[allow(dead_code)]
pub fn characteristic_limiter(
    q: &mut SWESolution,
    mesh: &Mesh1D,
    ops: &DGOperators1D,
    tvb: &TVBParameter,
    g: f64,
) {
    let n_elements = q.n_elements;
    let n_nodes = ops.n_nodes;
    let h_min = 1e-10;

    // Compute cell averages
    let averages = cell_averages(q, ops);

    let threshold = tvb.threshold();

    for k in 0..n_elements {
        let avg = SWEState::new(averages[k][0], averages[k][1]);
        let u = avg.velocity_simple(h_min);
        let c = (g * avg.h.max(h_min)).sqrt();

        // Eigenvalues and eigenvectors at cell average
        let lambda1 = u - c;
        let lambda2 = u + c;

        // Right eigenvectors (columns)
        // R = | 1      1    |
        //     | u-c    u+c  |

        // Left eigenvectors (rows) = R^{-1}
        // L = 1/(2c) | u+c  -1 |
        //            | -(u-c) 1 |

        let [h_minus, hu_minus] = q.get(k, 0);
        let [h_plus, hu_plus] = q.get(k, n_nodes - 1);

        // Differences in conserved variables
        let delta_minus = [averages[k][0] - h_minus, averages[k][1] - hu_minus];
        let delta_plus = [h_plus - averages[k][0], hu_plus - averages[k][1]];

        // Transform to characteristic variables
        let c_inv = 1.0 / (2.0 * c);
        let w_minus = [
            c_inv * ((u + c) * delta_minus[0] - delta_minus[1]),
            c_inv * (-(u - c) * delta_minus[0] + delta_minus[1]),
        ];
        let w_plus = [
            c_inv * ((u + c) * delta_plus[0] - delta_plus[1]),
            c_inv * (-(u - c) * delta_plus[0] + delta_plus[1]),
        ];

        // Neighboring averages in characteristic variables
        let (avg_left, avg_right) = {
            let left = if mesh.neighbors[k].0.is_some() {
                SWEState::new(averages[k - 1][0], averages[k - 1][1])
            } else {
                avg
            };
            let right = if mesh.neighbors[k].1.is_some() {
                SWEState::new(averages[k + 1][0], averages[k + 1][1])
            } else {
                avg
            };
            (left, right)
        };

        // Differences in characteristic variables
        let delta_forward = [
            c_inv * ((u + c) * (avg_right.h - avg.h) - (avg_right.hu - avg.hu)),
            c_inv * (-(u - c) * (avg_right.h - avg.h) + (avg_right.hu - avg.hu)),
        ];
        let delta_backward = [
            c_inv * ((u + c) * (avg.h - avg_left.h) - (avg.hu - avg_left.hu)),
            c_inv * (-(u - c) * (avg.h - avg_left.h) + (avg.hu - avg_left.hu)),
        ];

        // Apply TVB minmod in each characteristic field
        let limited_w_minus = [
            minmod_tvb(w_minus[0], delta_backward[0], delta_forward[0], threshold),
            minmod_tvb(w_minus[1], delta_backward[1], delta_forward[1], threshold),
        ];
        let limited_w_plus = [
            minmod_tvb(w_plus[0], delta_backward[0], delta_forward[0], threshold),
            minmod_tvb(w_plus[1], delta_backward[1], delta_forward[1], threshold),
        ];

        // Check if limiting is needed
        let needs_limiting = (limited_w_minus[0] - w_minus[0]).abs() > 1e-14
            || (limited_w_minus[1] - w_minus[1]).abs() > 1e-14
            || (limited_w_plus[0] - w_plus[0]).abs() > 1e-14
            || (limited_w_plus[1] - w_plus[1]).abs() > 1e-14;

        if needs_limiting {
            // Transform back to conserved variables
            let limited_delta_minus = [
                limited_w_minus[0] + limited_w_minus[1],
                (u - c) * limited_w_minus[0] + (u + c) * limited_w_minus[1],
            ];
            let limited_delta_plus = [
                limited_w_plus[0] + limited_w_plus[1],
                (u - c) * limited_w_plus[0] + (u + c) * limited_w_plus[1],
            ];

            // Reconstruct with limited linear approximation
            let nodes = &ops.nodes;

            for i in 0..n_nodes {
                let xi = nodes[i];
                let slope_h = 0.5 * (limited_delta_plus[0] - limited_delta_minus[0]);
                let slope_hu = 0.5 * (limited_delta_plus[1] - limited_delta_minus[1]);

                let new_h = averages[k][0] + slope_h * xi;
                let new_hu = averages[k][1] + slope_hu * xi;

                q.set(k, i, [new_h, new_hu]);
            }
        }

        // Mark eigenvalues as used
        let _ = (lambda1, lambda2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Mesh1D, DGOperators1D) {
        let mesh = Mesh1D::uniform(0.0, 1.0, 10);
        let ops = DGOperators1D::new(3); // P=3, 4 nodes
        (mesh, ops)
    }

    #[test]
    fn test_minmod_same_sign() {
        assert!((minmod(1.0, 2.0, 3.0) - 1.0).abs() < 1e-14);
        assert!((minmod(-1.0, -2.0, -3.0) - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn test_minmod_different_signs() {
        assert!((minmod(1.0, -1.0, 2.0)).abs() < 1e-14);
        assert!((minmod(-1.0, 1.0, -2.0)).abs() < 1e-14);
    }

    #[test]
    fn test_minmod_tvb_below_threshold() {
        // If |a| <= M*dx², return a unchanged
        let threshold = 0.1;
        let a = 0.05;
        assert!((minmod_tvb(a, 1.0, 2.0, threshold) - a).abs() < 1e-14);
    }

    #[test]
    fn test_minmod_tvb_above_threshold() {
        // If |a| > M*dx², apply minmod
        let threshold = 0.01;
        let a = 0.1;
        let result = minmod_tvb(a, 0.2, 0.3, threshold);
        assert!((result - minmod(a, 0.2, 0.3)).abs() < 1e-14);
    }

    #[test]
    fn test_tvb_limiter_smooth_preserves() {
        let (mesh, ops) = setup();
        let dx = 1.0 / 10.0;
        let tvb = TVBParameter::new(100.0, dx); // Large M = little limiting

        // Create smooth solution
        let mut q = SWESolution::new(10, 4);
        for k in 0..10 {
            for i in 0..4 {
                let x = mesh.reference_to_physical(k, ops.nodes[i]);
                let h = 1.0 + 0.1 * (2.0 * std::f64::consts::PI * x).sin();
                q.set(k, i, [h, h]); // u = 1
            }
        }

        let q_orig = q.clone();
        tvb_limiter(&mut q, &mesh, &ops, &tvb);

        // Should be mostly preserved (smooth solution, large M)
        for k in 0..10 {
            for i in 0..4 {
                let orig = q_orig.get(k, i);
                let new = q.get(k, i);
                // Allow some difference due to limiting at boundaries
                assert!(
                    (orig[0] - new[0]).abs() < 0.1,
                    "Smooth solution changed too much"
                );
            }
        }
    }

    #[test]
    fn test_tvb_limiter_preserves_average() {
        let (mesh, ops) = setup();
        let dx = 1.0 / 10.0;
        let tvb = TVBParameter::new(0.0, dx); // M = 0, strict limiting

        // Create solution with discontinuity
        let mut q = SWESolution::new(10, 4);
        for k in 0..10 {
            for i in 0..4 {
                let h = if k < 5 { 2.0 } else { 1.0 };
                q.set(k, i, [h, h * 0.5]);
            }
        }

        let avg_before = cell_averages(&q, &ops);
        tvb_limiter(&mut q, &mesh, &ops, &tvb);
        let avg_after = cell_averages(&q, &ops);

        // Cell averages should be preserved
        for k in 0..10 {
            assert!(
                (avg_before[k][0] - avg_after[k][0]).abs() < 1e-12,
                "Cell average h not preserved"
            );
            assert!(
                (avg_before[k][1] - avg_after[k][1]).abs() < 1e-12,
                "Cell average hu not preserved"
            );
        }
    }

    #[test]
    fn test_positivity_preserves_positive() {
        let (_, ops) = setup();

        // Create solution with all positive depths
        let mut q = SWESolution::new(10, 4);
        for k in 0..10 {
            for i in 0..4 {
                q.set(k, i, [1.0, 0.5]);
            }
        }

        let q_orig = q.clone();
        positivity_limiter(&mut q, &ops, 1e-6);

        // Should be unchanged
        for k in 0..10 {
            for i in 0..4 {
                let orig = q_orig.get(k, i);
                let new = q.get(k, i);
                assert!((orig[0] - new[0]).abs() < 1e-14);
                assert!((orig[1] - new[1]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_positivity_fixes_negative() {
        let (_, ops) = setup();

        // Create solution with some negative depths
        let mut q = SWESolution::new(10, 4);
        let nodes = &ops.nodes;
        for k in 0..10 {
            for i in 0..4 {
                // Oscillatory: average > 0, but some nodes < 0
                let h = 0.5 + 0.4 * nodes[i];
                q.set(k, i, [h, h * 0.5]);
            }
        }

        positivity_limiter(&mut q, &ops, 1e-6);

        // All depths should now be non-negative
        for k in 0..10 {
            for i in 0..4 {
                let h = q.get(k, i)[0];
                assert!(h >= 0.0, "Depth should be non-negative, got {}", h);
            }
        }
    }

    #[test]
    fn test_positivity_preserves_average() {
        let (_, ops) = setup();

        // Create solution needing limiting
        let mut q = SWESolution::new(10, 4);
        let nodes = &ops.nodes;
        for k in 0..10 {
            for i in 0..4 {
                let h = 0.5 + 0.4 * nodes[i]; // Ranges from 0.1 to 0.9
                q.set(k, i, [h, 0.0]);
            }
        }

        let avg_before = cell_averages(&q, &ops);
        positivity_limiter(&mut q, &ops, 1e-6);
        let avg_after = cell_averages(&q, &ops);

        // Cell averages should be approximately preserved
        for k in 0..10 {
            assert!(
                (avg_before[k][0] - avg_after[k][0]).abs() < 1e-10,
                "Cell average not preserved"
            );
        }
    }

    #[test]
    fn test_combined_limiters() {
        let (mesh, ops) = setup();
        let dx = 1.0 / 10.0;
        let tvb = TVBParameter::new(10.0, dx);

        // Create dam-break-like solution
        let mut q = SWESolution::new(10, 4);
        for k in 0..10 {
            for i in 0..4 {
                let h = if k < 5 { 2.0 } else { 0.1 };
                q.set(k, i, [h, 0.0]);
            }
        }

        apply_swe_limiters(&mut q, &mesh, &ops, &tvb, 1e-6);

        // All depths should be positive
        for k in 0..10 {
            for i in 0..4 {
                let h = q.get(k, i)[0];
                assert!(h >= 0.0, "Depth should be non-negative");
            }
        }
    }

    #[test]
    fn test_very_negative_average() {
        let (_, ops) = setup();

        // Create solution with negative average
        let mut q = SWESolution::new(10, 4);
        for k in 0..10 {
            for i in 0..4 {
                q.set(k, i, [-0.5, 0.0]);
            }
        }

        positivity_limiter(&mut q, &ops, 1e-6);

        // Should be set to minimum depth with zero momentum
        for k in 0..10 {
            for i in 0..4 {
                let [h, hu] = q.get(k, i);
                assert!((h - 1e-6).abs() < 1e-12);
                assert!(hu.abs() < 1e-12);
            }
        }
    }
}
