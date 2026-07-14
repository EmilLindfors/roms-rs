//! Bounds and slope limiters for 3D tracer concentrations.
//!
//! The 3D state stores temperature and salinity as concentrations, while the
//! conservative advection kernels transport layer inventory `Hz * C`. These
//! limiters therefore compute `Hz`-weighted averages before applying
//! Zhang-Shu/Kuzmin scaling.

use crate::mesh::{Bathymetry2D, Mesh2D};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::DGSolution2D;
use crate::solver::state::Solution3D;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

use super::tracer_2d::TracerBounds;

const MIN_LAYER_THICKNESS: f64 = 1.0e-12;
const MIN_INTEGRAL_WEIGHT: f64 = 1.0e-14;
const LIMITER_EPS: f64 = 1.0e-12;

/// Policy used when an element/layer average is already outside tracer bounds.
///
/// No limiter can both preserve inventory and put every nodal concentration
/// inside bounds if the inventory-weighted average is outside bounds. The
/// default keeps conservation and collapses the element/layer to its average;
/// callers that require hard bounds can opt into bounded average correction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TracerAveragePolicy3D {
    /// Preserve tracer inventory even if the average is outside bounds.
    PreserveConservation,
    /// Clamp out-of-bounds averages, reporting the resulting inventory change.
    EnforceBounds,
}

impl Default for TracerAveragePolicy3D {
    fn default() -> Self {
        Self::PreserveConservation
    }
}

/// 3D tracer limiter selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TracerLimiterType3D {
    /// Do not limit 3D tracers.
    None,
    /// Apply only physical bounds limiting.
    Bounds,
    /// Apply horizontal vertex-patch Kuzmin limiting, then physical bounds.
    HorizontalKuzmin { relaxation: f64 },
}

impl Default for TracerLimiterType3D {
    fn default() -> Self {
        Self::None
    }
}

/// Configuration for 3D tracer limiting.
#[derive(Clone, Copy, Debug)]
pub struct TracerLimiter3DConfig {
    /// Selected limiter.
    pub limiter_type: TracerLimiterType3D,
    /// Physical bounds for temperature and salinity.
    pub bounds: TracerBounds,
    /// Policy for element/layer averages already outside bounds.
    pub average_policy: TracerAveragePolicy3D,
    /// Also apply a vertical column bounds projection after horizontal limiting.
    pub vertical_column_bounds: bool,
}

impl Default for TracerLimiter3DConfig {
    fn default() -> Self {
        Self {
            limiter_type: TracerLimiterType3D::None,
            bounds: TracerBounds::default(),
            average_policy: TracerAveragePolicy3D::default(),
            vertical_column_bounds: false,
        }
    }
}

impl TracerLimiter3DConfig {
    /// Disable 3D tracer limiting.
    pub fn none() -> Self {
        Self::default()
    }

    /// Enable only physical bounds limiting.
    pub fn bounds(bounds: TracerBounds) -> Self {
        Self {
            limiter_type: TracerLimiterType3D::Bounds,
            bounds,
            ..Self::default()
        }
    }

    /// Enable horizontal Kuzmin limiting followed by physical bounds limiting.
    pub fn horizontal_kuzmin(bounds: TracerBounds, relaxation: f64) -> Self {
        Self {
            limiter_type: TracerLimiterType3D::HorizontalKuzmin {
                relaxation: relaxation.max(1.0),
            },
            bounds,
            ..Self::default()
        }
    }

    /// Set the policy used when an inventory-weighted average is outside bounds.
    pub fn with_average_policy(mut self, policy: TracerAveragePolicy3D) -> Self {
        self.average_policy = policy;
        self
    }

    /// Enable or disable the optional vertical column bounds projection.
    pub fn with_vertical_column_bounds(mut self, enabled: bool) -> Self {
        self.vertical_column_bounds = enabled;
        self
    }
}

/// Diagnostics from applying 3D tracer limiters.
#[derive(Clone, Copy, Debug, Default)]
pub struct TracerLimiter3DStats {
    /// Number of element/layer or column projections that changed temperature.
    pub limited_temperature_cells: usize,
    /// Number of element/layer or column projections that changed salinity.
    pub limited_salinity_cells: usize,
    /// Number of temperature averages that were already outside bounds.
    pub temperature_average_violations: usize,
    /// Number of salinity averages that were already outside bounds.
    pub salinity_average_violations: usize,
    /// Domain-integrated temperature inventory correction from enforced averages.
    pub temperature_inventory_correction: f64,
    /// Domain-integrated salinity inventory correction from enforced averages.
    pub salinity_inventory_correction: f64,
}

impl TracerLimiter3DStats {
    /// Returns true if the limiter changed any tracer values.
    pub fn changed(&self) -> bool {
        self.limited_temperature_cells > 0
            || self.limited_salinity_cells > 0
            || self.temperature_inventory_correction.abs() > 0.0
            || self.salinity_inventory_correction.abs() > 0.0
    }

    fn record_average_violation(&mut self, component: TracerComponent) {
        match component {
            TracerComponent::Temperature => self.temperature_average_violations += 1,
            TracerComponent::Salinity => self.salinity_average_violations += 1,
        }
    }

    fn record_limited(&mut self, component: TracerComponent) {
        match component {
            TracerComponent::Temperature => self.limited_temperature_cells += 1,
            TracerComponent::Salinity => self.limited_salinity_cells += 1,
        }
    }

    fn add_inventory_correction(&mut self, component: TracerComponent, correction: f64) {
        match component {
            TracerComponent::Temperature => self.temperature_inventory_correction += correction,
            TracerComponent::Salinity => self.salinity_inventory_correction += correction,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum TracerComponent {
    Temperature,
    Salinity,
}

/// Apply configured 3D tracer limiters to temperature and salinity.
pub fn apply_tracer_limiters_3d(
    state: &mut Solution3D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    bathymetry: &Bathymetry2D,
    sigma: &SigmaGrid,
    config: &TracerLimiter3DConfig,
) -> TracerLimiter3DStats {
    let mut stats = TracerLimiter3DStats::default();

    match config.limiter_type {
        TracerLimiterType3D::None => return stats,
        TracerLimiterType3D::Bounds => {}
        TracerLimiterType3D::HorizontalKuzmin { relaxation } => {
            apply_horizontal_kuzmin_field(
                &mut state.temp,
                &state.eta,
                state.n_elements,
                state.n_nodes,
                state.n_levels,
                mesh,
                ops,
                geom,
                bathymetry,
                sigma,
                relaxation,
                TracerComponent::Temperature,
                &mut stats,
            );
            apply_horizontal_kuzmin_field(
                &mut state.salt,
                &state.eta,
                state.n_elements,
                state.n_nodes,
                state.n_levels,
                mesh,
                ops,
                geom,
                bathymetry,
                sigma,
                relaxation,
                TracerComponent::Salinity,
                &mut stats,
            );
        }
    }

    apply_horizontal_bounds_field(
        &mut state.temp,
        &state.eta,
        state.n_elements,
        state.n_nodes,
        state.n_levels,
        ops,
        geom,
        bathymetry,
        sigma,
        config.bounds.t_min,
        config.bounds.t_max,
        config.average_policy,
        TracerComponent::Temperature,
        &mut stats,
    );
    apply_horizontal_bounds_field(
        &mut state.salt,
        &state.eta,
        state.n_elements,
        state.n_nodes,
        state.n_levels,
        ops,
        geom,
        bathymetry,
        sigma,
        config.bounds.s_min,
        config.bounds.s_max,
        config.average_policy,
        TracerComponent::Salinity,
        &mut stats,
    );

    if config.vertical_column_bounds {
        apply_vertical_column_bounds_field(
            &mut state.temp,
            &state.eta,
            state.n_elements,
            state.n_nodes,
            state.n_levels,
            ops,
            geom,
            bathymetry,
            sigma,
            config.bounds.t_min,
            config.bounds.t_max,
            config.average_policy,
            TracerComponent::Temperature,
            &mut stats,
        );
        apply_vertical_column_bounds_field(
            &mut state.salt,
            &state.eta,
            state.n_elements,
            state.n_nodes,
            state.n_levels,
            ops,
            geom,
            bathymetry,
            sigma,
            config.bounds.s_min,
            config.bounds.s_max,
            config.average_policy,
            TracerComponent::Salinity,
            &mut stats,
        );
    }

    stats
}

#[allow(clippy::too_many_arguments)]
fn apply_horizontal_bounds_field(
    field: &mut [f64],
    eta: &DGSolution2D,
    n_elements: usize,
    n_nodes: usize,
    n_levels: usize,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    bathymetry: &Bathymetry2D,
    sigma: &SigmaGrid,
    bound_min: f64,
    bound_max: f64,
    average_policy: TracerAveragePolicy3D,
    component: TracerComponent,
    stats: &mut TracerLimiter3DStats,
) {
    let d_sigma = sigma.d_sigma();

    for k in 0..n_elements {
        let element = ElementIndex::new(k);
        let jac = geom.det_j[k];
        for (level, &ds) in d_sigma.iter().enumerate().take(n_levels) {
            let mut weight_sum = 0.0;
            let mut inventory = 0.0;
            let mut min_value = f64::INFINITY;
            let mut max_value = f64::NEG_INFINITY;

            for (i, &w) in ops.weights.iter().enumerate().take(n_nodes) {
                let value = field[index(k, i, level, n_nodes, n_levels)];
                let weight = w * jac * layer_thickness(eta, bathymetry, element, i, ds);
                weight_sum += weight;
                inventory += weight * value;
                min_value = min_value.min(value);
                max_value = max_value.max(value);
            }

            if weight_sum <= MIN_INTEGRAL_WEIGHT {
                continue;
            }

            let avg = inventory / weight_sum;

            if avg < bound_min - LIMITER_EPS || avg > bound_max + LIMITER_EPS {
                stats.record_average_violation(component);
                let replacement = match average_policy {
                    TracerAveragePolicy3D::PreserveConservation => avg,
                    TracerAveragePolicy3D::EnforceBounds => avg.clamp(bound_min, bound_max),
                };

                for i in 0..n_nodes {
                    field[index(k, i, level, n_nodes, n_levels)] = replacement;
                }

                stats.record_limited(component);
                stats.add_inventory_correction(component, weight_sum * (replacement - avg));
                continue;
            }

            if min_value >= bound_min - LIMITER_EPS && max_value <= bound_max + LIMITER_EPS {
                continue;
            }

            let theta = compute_theta(avg, min_value, max_value, bound_min, bound_max);
            if theta >= 1.0 - LIMITER_EPS {
                continue;
            }

            for i in 0..n_nodes {
                let idx = index(k, i, level, n_nodes, n_levels);
                field[idx] = avg + theta * (field[idx] - avg);
            }
            stats.record_limited(component);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_vertical_column_bounds_field(
    field: &mut [f64],
    eta: &DGSolution2D,
    n_elements: usize,
    n_nodes: usize,
    n_levels: usize,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    bathymetry: &Bathymetry2D,
    sigma: &SigmaGrid,
    bound_min: f64,
    bound_max: f64,
    average_policy: TracerAveragePolicy3D,
    component: TracerComponent,
    stats: &mut TracerLimiter3DStats,
) {
    let d_sigma = sigma.d_sigma();

    for k in 0..n_elements {
        let element = ElementIndex::new(k);
        for i in 0..n_nodes {
            let horizontal_weight = ops.weights[i] * geom.det_j[k];
            let mut weight_sum = 0.0;
            let mut inventory = 0.0;
            let mut min_value = f64::INFINITY;
            let mut max_value = f64::NEG_INFINITY;

            for (level, &ds) in d_sigma.iter().enumerate().take(n_levels) {
                let idx = index(k, i, level, n_nodes, n_levels);
                let value = field[idx];
                let weight = layer_thickness(eta, bathymetry, element, i, ds);
                weight_sum += weight;
                inventory += weight * value;
                min_value = min_value.min(value);
                max_value = max_value.max(value);
            }

            if weight_sum <= MIN_INTEGRAL_WEIGHT {
                continue;
            }

            let avg = inventory / weight_sum;
            if avg < bound_min - LIMITER_EPS || avg > bound_max + LIMITER_EPS {
                stats.record_average_violation(component);
                let replacement = match average_policy {
                    TracerAveragePolicy3D::PreserveConservation => avg,
                    TracerAveragePolicy3D::EnforceBounds => avg.clamp(bound_min, bound_max),
                };

                for level in 0..n_levels {
                    field[index(k, i, level, n_nodes, n_levels)] = replacement;
                }

                stats.record_limited(component);
                stats.add_inventory_correction(
                    component,
                    horizontal_weight * weight_sum * (replacement - avg),
                );
                continue;
            }

            if min_value >= bound_min - LIMITER_EPS && max_value <= bound_max + LIMITER_EPS {
                continue;
            }

            let theta = compute_theta(avg, min_value, max_value, bound_min, bound_max);
            if theta >= 1.0 - LIMITER_EPS {
                continue;
            }

            for level in 0..n_levels {
                let idx = index(k, i, level, n_nodes, n_levels);
                field[idx] = avg + theta * (field[idx] - avg);
            }
            stats.record_limited(component);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_horizontal_kuzmin_field(
    field: &mut [f64],
    eta: &DGSolution2D,
    n_elements: usize,
    n_nodes: usize,
    n_levels: usize,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    bathymetry: &Bathymetry2D,
    sigma: &SigmaGrid,
    relaxation: f64,
    component: TracerComponent,
    stats: &mut TracerLimiter3DStats,
) {
    let averages = horizontal_layer_averages(
        field, eta, n_elements, n_nodes, n_levels, ops, geom, bathymetry, sigma,
    );

    for k in 0..n_elements {
        let element = ElementIndex::new(k);
        let vertices = mesh.element_vertex_indices(element);

        for level in 0..n_levels {
            let avg = averages[k * n_levels + level];
            if !avg.is_finite() {
                continue;
            }

            let mut alpha = 1.0_f64;

            for (local_vertex, &global_vertex) in vertices.iter().enumerate() {
                let (bound_min, bound_max) = vertex_patch_bounds(
                    global_vertex,
                    level,
                    mesh,
                    &averages,
                    n_levels,
                    relaxation,
                );
                let node_idx = vertex_to_node_index(local_vertex, ops.n_1d);
                let value = field[index(k, node_idx, level, n_nodes, n_levels)];
                alpha = alpha.min(compute_kuzmin_alpha(avg, value, bound_min, bound_max));
            }

            if alpha >= 1.0 - LIMITER_EPS {
                continue;
            }

            for i in 0..n_nodes {
                let idx = index(k, i, level, n_nodes, n_levels);
                field[idx] = avg + alpha * (field[idx] - avg);
            }
            stats.record_limited(component);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn horizontal_layer_averages(
    field: &[f64],
    eta: &DGSolution2D,
    n_elements: usize,
    n_nodes: usize,
    n_levels: usize,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    bathymetry: &Bathymetry2D,
    sigma: &SigmaGrid,
) -> Vec<f64> {
    let d_sigma = sigma.d_sigma();
    let mut averages = vec![f64::NAN; n_elements * n_levels];

    for k in 0..n_elements {
        let element = ElementIndex::new(k);
        let jac = geom.det_j[k];
        for (level, &ds) in d_sigma.iter().enumerate().take(n_levels) {
            let mut weight_sum = 0.0;
            let mut inventory = 0.0;

            for (i, &w) in ops.weights.iter().enumerate().take(n_nodes) {
                let weight = w * jac * layer_thickness(eta, bathymetry, element, i, ds);
                weight_sum += weight;
                inventory += weight * field[index(k, i, level, n_nodes, n_levels)];
            }

            if weight_sum > MIN_INTEGRAL_WEIGHT {
                averages[k * n_levels + level] = inventory / weight_sum;
            }
        }
    }

    averages
}

fn vertex_patch_bounds(
    vertex: usize,
    level: usize,
    mesh: &Mesh2D,
    averages: &[f64],
    n_levels: usize,
    relaxation: f64,
) -> (f64, f64) {
    let mut bound_min = f64::INFINITY;
    let mut bound_max = f64::NEG_INFINITY;

    for &elem in mesh.elements_at_vertex(vertex) {
        let value = averages[elem * n_levels + level];
        if value.is_finite() {
            bound_min = bound_min.min(value);
            bound_max = bound_max.max(value);
        }
    }

    if !bound_min.is_finite() || !bound_max.is_finite() {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }

    if relaxation > 1.0 {
        let range = bound_max - bound_min;
        let expand = 0.5 * range * (relaxation - 1.0);
        bound_min -= expand;
        bound_max += expand;
    }

    (bound_min, bound_max)
}

fn compute_theta(avg: f64, min_value: f64, max_value: f64, bound_min: f64, bound_max: f64) -> f64 {
    let mut theta: f64 = 1.0;

    if min_value < bound_min && (avg - min_value).abs() > LIMITER_EPS {
        theta = theta.min((avg - bound_min) / (avg - min_value));
    }

    if max_value > bound_max && (max_value - avg).abs() > LIMITER_EPS {
        theta = theta.min((bound_max - avg) / (max_value - avg));
    }

    theta.clamp(0.0, 1.0)
}

fn compute_kuzmin_alpha(avg: f64, value: f64, bound_min: f64, bound_max: f64) -> f64 {
    let deviation = value - avg;
    if deviation.abs() < LIMITER_EPS {
        return 1.0;
    }

    let mut alpha: f64 = 1.0;

    if value < bound_min && deviation < 0.0 {
        alpha = alpha.min((avg - bound_min) / (avg - value));
    }

    if value > bound_max && deviation > 0.0 {
        alpha = alpha.min((bound_max - avg) / (value - avg));
    }

    alpha.clamp(0.0, 1.0)
}

fn vertex_to_node_index(local_vertex: usize, n_1d: usize) -> usize {
    match local_vertex {
        0 => 0,
        1 => n_1d - 1,
        2 => n_1d * n_1d - 1,
        3 => n_1d * (n_1d - 1),
        _ => panic!("Invalid local vertex index: {}", local_vertex),
    }
}

fn layer_thickness(
    eta: &DGSolution2D,
    bathymetry: &Bathymetry2D,
    element: ElementIndex,
    node: usize,
    d_sigma: f64,
) -> f64 {
    let eta_value = eta.get(element.as_usize(), node);
    (bathymetry.water_depth(element, node, eta_value) * d_sigma).max(MIN_LAYER_THICKNESS)
}

#[inline]
fn index(k: usize, node: usize, level: usize, n_nodes: usize, n_levels: usize) -> usize {
    (k * n_nodes + node) * n_levels + level
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh2D;
    use crate::operators::{DGOperators2D, GeometricFactors2D};

    fn setup(
        nx: usize,
        ny: usize,
        order: usize,
        n_levels: usize,
    ) -> (
        Mesh2D,
        DGOperators2D,
        GeometricFactors2D,
        Bathymetry2D,
        SigmaGrid,
        Solution3D,
    ) {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, nx, ny);
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh);
        let bathymetry = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -10.0);
        let sigma = SigmaGrid::uniform(n_levels);
        let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, n_levels);
        state.eta.fill(0.0);
        state.temp.fill(5.0);
        state.salt.fill(30.0);
        (mesh, ops, geom, bathymetry, sigma, state)
    }

    fn total_inventory(
        field: &[f64],
        state: &Solution3D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        bathymetry: &Bathymetry2D,
        sigma: &SigmaGrid,
    ) -> f64 {
        let mut total = 0.0;
        for k in 0..state.n_elements {
            let element = ElementIndex::new(k);
            for i in 0..state.n_nodes {
                for (level, &ds) in sigma.d_sigma().iter().enumerate() {
                    let weight = ops.weights[i]
                        * geom.det_j[k]
                        * layer_thickness(&state.eta, bathymetry, element, i, ds);
                    total += weight * field[index(k, i, level, state.n_nodes, state.n_levels)];
                }
            }
        }
        total
    }

    #[test]
    fn bounds_limiter_enforces_bounds_and_preserves_inventory_when_average_is_bounded() {
        let (mesh, ops, geom, bathymetry, sigma, mut state) = setup(1, 1, 2, 2);
        let k = 0;
        let low = index(k, 0, 0, state.n_nodes, state.n_levels);
        let high = index(k, ops.n_nodes - 1, 0, state.n_nodes, state.n_levels);
        state.temp[low] = -5.0;
        state.temp[high] = 15.0;

        let before = total_inventory(&state.temp, &state, &ops, &geom, &bathymetry, &sigma);
        let config = TracerLimiter3DConfig::bounds(TracerBounds::new(0.0, 10.0, 0.0, 40.0));

        let stats =
            apply_tracer_limiters_3d(&mut state, &mesh, &ops, &geom, &bathymetry, &sigma, &config);

        let after = total_inventory(&state.temp, &state, &ops, &geom, &bathymetry, &sigma);
        assert!(stats.limited_temperature_cells > 0);
        assert!(
            (before - after).abs() < 1e-10,
            "inventory changed: before={before}, after={after}"
        );
        for &value in &state.temp {
            assert!(
                (-1e-12..=10.0 + 1e-12).contains(&value),
                "temperature out of bounds: {value}"
            );
        }
    }

    #[test]
    fn preserve_conservation_policy_keeps_out_of_bounds_average_inventory() {
        let (mesh, ops, geom, bathymetry, sigma, mut state) = setup(1, 1, 1, 1);
        state.temp.fill(-3.0);
        let before = total_inventory(&state.temp, &state, &ops, &geom, &bathymetry, &sigma);
        let config = TracerLimiter3DConfig::bounds(TracerBounds::new(0.0, 10.0, 0.0, 40.0));

        let stats =
            apply_tracer_limiters_3d(&mut state, &mesh, &ops, &geom, &bathymetry, &sigma, &config);

        let after = total_inventory(&state.temp, &state, &ops, &geom, &bathymetry, &sigma);
        assert_eq!(stats.temperature_average_violations, 1);
        assert!((before - after).abs() < 1e-12);
        assert!(state.temp.iter().all(|&value| (value + 3.0).abs() < 1e-12));
    }

    #[test]
    fn enforce_bounds_policy_reports_inventory_correction_for_bad_average() {
        let (mesh, ops, geom, bathymetry, sigma, mut state) = setup(1, 1, 1, 1);
        state.temp.fill(-3.0);
        let config = TracerLimiter3DConfig::bounds(TracerBounds::new(0.0, 10.0, 0.0, 40.0))
            .with_average_policy(TracerAveragePolicy3D::EnforceBounds);

        let stats =
            apply_tracer_limiters_3d(&mut state, &mesh, &ops, &geom, &bathymetry, &sigma, &config);

        assert_eq!(stats.temperature_average_violations, 1);
        assert!(stats.temperature_inventory_correction > 0.0);
        assert!(state.temp.iter().all(|&value| value.abs() < 1e-12));
    }

    #[test]
    fn horizontal_kuzmin_reduces_vertex_overshoot_and_preserves_inventory() {
        let (mesh, ops, geom, bathymetry, sigma, mut state) = setup(2, 1, 2, 1);

        for i in 0..ops.n_nodes {
            state.temp[index(0, i, 0, state.n_nodes, state.n_levels)] = 1.0;
            state.temp[index(1, i, 0, state.n_nodes, state.n_levels)] = 3.0;
        }

        let high = index(0, ops.n_1d - 1, 0, state.n_nodes, state.n_levels);
        let low = index(0, 0, 0, state.n_nodes, state.n_levels);
        state.temp[high] = 6.0;
        state.temp[low] = -4.0;

        let before = total_inventory(&state.temp, &state, &ops, &geom, &bathymetry, &sigma);
        let config = TracerLimiter3DConfig::horizontal_kuzmin(
            TracerBounds::new(-100.0, 100.0, 0.0, 40.0),
            1.0,
        );

        let stats =
            apply_tracer_limiters_3d(&mut state, &mesh, &ops, &geom, &bathymetry, &sigma, &config);

        let after = total_inventory(&state.temp, &state, &ops, &geom, &bathymetry, &sigma);
        assert!(stats.limited_temperature_cells > 0);
        assert!(
            state.temp[high] < 6.0,
            "Kuzmin limiter should reduce the vertex overshoot"
        );
        assert!(
            (before - after).abs() < 1e-10,
            "inventory changed: before={before}, after={after}"
        );
    }
}
