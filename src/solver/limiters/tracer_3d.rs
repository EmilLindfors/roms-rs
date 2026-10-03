//! Bounds and slope limiters for 3D tracer concentrations.
//!
//! The 3D state stores temperature and salinity as concentrations, while the
//! conservative advection kernels transport layer inventory `Hz * C`. These
//! limiters therefore compute `Hz`-weighted averages before applying
//! Zhang-Shu/Kuzmin scaling.

use std::sync::Arc;

use crate::mesh::{Bathymetry2D, Mesh2D};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::DGSolution2D;
use crate::solver::core::blocks::{Pooled, for_each_block, reduce_blocks};
use crate::solver::state::Solution3D;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

use super::tracer_2d::TracerBounds;

const MIN_LAYER_THICKNESS: f64 = 1.0e-12;
const MIN_INTEGRAL_WEIGHT: f64 = 1.0e-14;
const LIMITER_EPS: f64 = 1.0e-12;
/// The relative round-off margin of the Kuzmin bounds.
const ROUND_OFF: f64 = 256.0 * f64::EPSILON;

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
#[derive(Clone, Debug)]
pub struct TracerLimiter3DConfig {
    /// Selected limiter.
    pub limiter_type: TracerLimiterType3D,
    /// Physical bounds for temperature and salinity.
    pub bounds: TracerBounds,
    /// Policy for element/layer averages already outside bounds.
    pub average_policy: TracerAveragePolicy3D,
    /// Also apply a vertical column bounds projection after horizontal limiting.
    pub vertical_column_bounds: bool,
    /// A horizontally uniform stratification whose departure the horizontal
    /// Kuzmin limiter bounds, if any (see [`Self::with_reference_profile`]).
    pub reference: Option<Arc<TracerReferenceProfile>>,
}

impl Default for TracerLimiter3DConfig {
    fn default() -> Self {
        Self {
            limiter_type: TracerLimiterType3D::None,
            bounds: TracerBounds::default(),
            average_policy: TracerAveragePolicy3D::default(),
            vertical_column_bounds: false,
            reference: None,
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

    /// Limit horizontally the departure from `reference`, not the tracers
    /// themselves.
    ///
    /// The horizontal Kuzmin limiter bounds each node by its neighbours'
    /// layer means at the node's height, which is exact for a stratification
    /// linear in `z` but not for a curved one: where the σ-layers are thick
    /// against a pycnocline and cross it steeply within an element (over a
    /// seamount's flank), the layer means are not its values at their mean
    /// heights, and the limiter changes the pycnocline at rest (up to 1.8 °C of
    /// 17 °C over a 6 × 6 seamount). With the reference stratification taken
    /// out, a fluid at rest in it has nothing to limit; with a
    /// stratification that varies horizontally, the limiter sees only the
    /// variation. Use the domain's typical (or initial) profile.
    pub fn with_reference_profile(mut self, reference: TracerReferenceProfile) -> Self {
        self.reference = Some(Arc::new(reference));
        self
    }
}

/// A horizontally uniform stratification `T(z)`, `S(z)`, linear between
/// samples and constant beyond them: the reference of the horizontal Kuzmin
/// limiter ([`TracerLimiter3DConfig::with_reference_profile`]).
#[derive(Clone, Debug, PartialEq)]
pub struct TracerReferenceProfile {
    heights: Vec<f64>,
    temp: Vec<f64>,
    salt: Vec<f64>,
}

impl TracerReferenceProfile {
    /// Temperature and salinity at `heights` (m, increasing upward).
    ///
    /// # Panics
    ///
    /// If the lengths differ, there are no samples, or the heights do not
    /// increase.
    pub fn new(heights: Vec<f64>, temp: Vec<f64>, salt: Vec<f64>) -> Self {
        assert!(
            !heights.is_empty() && temp.len() == heights.len() && salt.len() == heights.len(),
            "a reference profile needs as many temperatures and salinities as heights"
        );
        assert!(
            heights.windows(2).all(|pair| pair[0] < pair[1]),
            "the heights of a reference profile must increase"
        );
        Self {
            heights,
            temp,
            salt,
        }
    }

    /// `profile(z) = (T, S)` sampled at `n` (≥ 2) heights evenly from
    /// `z_bottom` to `z_top`.
    pub fn from_fn(
        z_bottom: f64,
        z_top: f64,
        n: usize,
        profile: impl Fn(f64) -> (f64, f64),
    ) -> Self {
        assert!(
            n >= 2 && z_bottom < z_top,
            "a sampled profile needs n ≥ 2 and z_bottom < z_top"
        );
        let heights: Vec<f64> = (0..n)
            .map(|j| z_bottom + (z_top - z_bottom) * j as f64 / (n - 1) as f64)
            .collect();
        let (temp, salt) = heights.iter().map(|&z| profile(z)).unzip();
        Self::new(heights, temp, salt)
    }

    /// The temperature at height `z`.
    pub fn temperature(&self, z: f64) -> f64 {
        interpolate_profile(&self.heights, &self.temp, z)
    }

    /// The salinity at height `z`.
    pub fn salinity(&self, z: f64) -> f64 {
        interpolate_profile(&self.heights, &self.salt, z)
    }

    fn values(&self, component: TracerComponent) -> &[f64] {
        match component {
            TracerComponent::Temperature => &self.temp,
            TracerComponent::Salinity => &self.salt,
        }
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
    /// Domain-integrated temperature inventory correction from enforced
    /// averages. (Summed per element in an order that depends on the thread
    /// count: its last bits may differ between runs.)
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

    /// The sum of two diagnostics.
    fn merged(self, other: Self) -> Self {
        Self {
            limited_temperature_cells: self.limited_temperature_cells
                + other.limited_temperature_cells,
            limited_salinity_cells: self.limited_salinity_cells + other.limited_salinity_cells,
            temperature_average_violations: self.temperature_average_violations
                + other.temperature_average_violations,
            salinity_average_violations: self.salinity_average_violations
                + other.salinity_average_violations,
            temperature_inventory_correction: self.temperature_inventory_correction
                + other.temperature_inventory_correction,
            salinity_inventory_correction: self.salinity_inventory_correction
                + other.salinity_inventory_correction,
        }
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
            let mut centres = Pooled::take(
                |a: &Vec<f64>| a.len() == state.n_elements * state.n_nodes * state.n_levels,
                || vec![f64::NAN; state.n_elements * state.n_nodes * state.n_levels],
            );
            fill_centre_heights(
                &mut centres,
                &state.eta,
                state.n_nodes,
                state.n_levels,
                bathymetry,
                sigma,
            );
            let columns = LayerColumns {
                centres: &centres,
                eta: &state.eta,
                n_elements: state.n_elements,
                n_nodes: state.n_nodes,
                n_levels: state.n_levels,
                ops,
                geom,
                bathymetry,
                sigma,
            };
            let mut mean_heights = Pooled::take(
                |a: &Vec<f64>| a.len() == state.n_elements * state.n_levels,
                || vec![f64::NAN; state.n_elements * state.n_levels],
            );
            columns.layer_mean_heights(&mut mean_heights);
            let reference = |component| {
                config
                    .reference
                    .as_deref()
                    .map(|profile: &TracerReferenceProfile| {
                        (&profile.heights[..], profile.values(component))
                    })
            };
            apply_horizontal_kuzmin_field(
                &mut state.temp,
                reference(TracerComponent::Temperature),
                &columns,
                &mean_heights,
                mesh,
                relaxation,
                TracerComponent::Temperature,
                &mut stats,
            );
            apply_horizontal_kuzmin_field(
                &mut state.salt,
                reference(TracerComponent::Salinity),
                &columns,
                &mean_heights,
                mesh,
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
    let index = |i: usize, level: usize| i * n_levels + level;

    let block_stats = reduce_blocks(
        n_elements,
        [&mut field[..n_elements * n_nodes * n_levels]],
        || (),
        |_, k, [field]| {
            let mut stats = TracerLimiter3DStats::default();
            let element = ElementIndex::new(k);
            for (level, &ds) in d_sigma.iter().enumerate().take(n_levels) {
                let mut weight_sum = 0.0;
                let mut inventory = 0.0;
                let mut min_value = f64::INFINITY;
                let mut max_value = f64::NEG_INFINITY;

                for i in 0..n_nodes {
                    let value = field[index(i, level)];
                    let weight =
                        geom.node_mass(k, i) * layer_thickness(eta, bathymetry, element, i, ds);
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
                        field[index(i, level)] = replacement;
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
                    let idx = index(i, level);
                    field[idx] = avg + theta * (field[idx] - avg);
                }
                stats.record_limited(component);
            }
            stats
        },
        TracerLimiter3DStats::default,
        TracerLimiter3DStats::merged,
    );
    *stats = stats.merged(block_stats);
}

#[allow(clippy::too_many_arguments)]
fn apply_vertical_column_bounds_field(
    field: &mut [f64],
    eta: &DGSolution2D,
    n_elements: usize,
    n_nodes: usize,
    n_levels: usize,
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
    let index = |i: usize, level: usize| i * n_levels + level;

    let block_stats = reduce_blocks(
        n_elements,
        [&mut field[..n_elements * n_nodes * n_levels]],
        || (),
        |_, k, [field]| {
            let mut stats = TracerLimiter3DStats::default();
            let element = ElementIndex::new(k);
            for i in 0..n_nodes {
                let horizontal_weight = geom.node_mass(k, i);
                let mut weight_sum = 0.0;
                let mut inventory = 0.0;
                let mut min_value = f64::INFINITY;
                let mut max_value = f64::NEG_INFINITY;

                for (level, &ds) in d_sigma.iter().enumerate().take(n_levels) {
                    let value = field[index(i, level)];
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
                        field[index(i, level)] = replacement;
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
                    let idx = index(i, level);
                    field[idx] = avg + theta * (field[idx] - avg);
                }
                stats.record_limited(component);
            }
            stats
        },
        TracerLimiter3DStats::default,
        TracerLimiter3DStats::merged,
    );
    *stats = stats.merged(block_stats);
}

/// The layers' geometry and inventory weights, for the means and heights
/// the horizontal Kuzmin bounds compare.
struct LayerColumns<'a> {
    /// Every node's layer-centre height `z = η + D σ` (`[element][node][level]`).
    centres: &'a [f64],
    eta: &'a DGSolution2D,
    n_elements: usize,
    n_nodes: usize,
    n_levels: usize,
    ops: &'a DGOperators2D,
    geom: &'a GeometricFactors2D,
    bathymetry: &'a Bathymetry2D,
    sigma: &'a SigmaGrid,
}

/// Every node's layer-centre heights `z = η + D σ` into `centres`
/// (`[element][node][level]`).
fn fill_centre_heights(
    centres: &mut [f64],
    eta: &DGSolution2D,
    n_nodes: usize,
    n_levels: usize,
    bathymetry: &Bathymetry2D,
    sigma: &SigmaGrid,
) {
    let n_elements = centres.len() / (n_nodes * n_levels);
    let sigma_rho = sigma.sigma_rho();
    for_each_block(
        n_elements,
        [centres],
        || (),
        |_, k, [centres]| {
            let element = ElementIndex::new(k);
            for i in 0..n_nodes {
                let eta = eta.get(k, i);
                let depth = bathymetry.water_depth(element, i, eta).max(0.0);
                for (level, &s) in sigma_rho.iter().enumerate().take(n_levels) {
                    centres[i * n_levels + level] = eta + depth * s;
                }
            }
        },
    );
}

impl LayerColumns<'_> {
    /// The height `z = η + D σ` of a node's layer centre.
    fn centre_height(&self, element: ElementIndex, node: usize, level: usize) -> f64 {
        self.centres[index(element.as_usize(), node, level, self.n_nodes, self.n_levels)]
    }

    /// The inventory-weighted mean of `value(k, node, level)` over every
    /// element's layers into `means` (`[element][level]`; NaN for a layer
    /// without water).
    fn layer_means(&self, means: &mut [f64], value: impl Fn(usize, usize, usize) -> f64 + Sync) {
        let d_sigma = self.sigma.d_sigma();
        for_each_block(
            self.n_elements,
            [means],
            || (),
            |_, k, [means]| {
                let element = ElementIndex::new(k);
                for (level, &ds) in d_sigma.iter().enumerate().take(self.n_levels) {
                    let mut weight_sum = 0.0;
                    let mut inventory = 0.0;
                    for i in 0..self.n_nodes {
                        let weight = self.geom.node_mass(k, i)
                            * layer_thickness(self.eta, self.bathymetry, element, i, ds);
                        weight_sum += weight;
                        inventory += weight * value(k, i, level);
                    }
                    means[level] = if weight_sum > MIN_INTEGRAL_WEIGHT {
                        inventory / weight_sum
                    } else {
                        f64::NAN
                    };
                }
            },
        );
    }

    /// The inventory-weighted mean height of every element layer's centres:
    /// for a tracer linear in `z`, its layer mean is its value there.
    fn layer_mean_heights(&self, heights: &mut [f64]) {
        self.layer_means(heights, |k, i, level| {
            self.centre_height(ElementIndex::new(k), i, level)
        });
    }
}

/// A Kuzmin worker's scratch, per node of the element's current layer.
#[derive(Default)]
struct KuzminScratch {
    /// The union of the element's vertex patches.
    patches: Vec<usize>,
    /// The node's height.
    height: Vec<f64>,
    /// The reference stratification at the node's height (zero without one).
    reference: Vec<f64>,
    /// The departure from it, `T_i − T_ref(z_i)`.
    value: Vec<f64>,
    /// The limited state at `α = 0`.
    base: Vec<f64>,
    /// The inventory weight.
    weight: Vec<f64>,
    /// The bounds.
    bounds: Vec<(f64, f64)>,
    /// The tracer's magnitude, for the bounds' round-off margin.
    scale: Vec<f64>,
}

/// Horizontal Kuzmin limiting of `field`, layer by layer, against bounds
/// taken at constant height, of its departure from a `reference`
/// stratification (`(heights, values)`; none is zero).
///
/// On a sloping bed a σ-layer crosses the stratification, so a smooth `T(z)`
/// varies along it, extremal where the bed is (the top of a seamount). The
/// classic limiter took that for a front twice over: it bounded the nodes by
/// the neighbours' means on the same layer, which a smooth stratification
/// exceeds at every bed extremum, and it scaled the whole deviation from the
/// layer's mean, so that a round-off clip at a node near the mean's height
/// flattened the layer's stratification; a stratified seamount at rest
/// reached 0.15 m/s in 15 min. Here:
///
/// - each node is bounded by its patch's columns (a vertex by its own patch,
///   every node by the union of the element's vertex patches) *at the
///   node's height*: each column's layer means interpolated linearly in `z`,
///   and the means of its two layers bracketing that height (the cells above
///   and below of a 3D vertex patch; [`column_at_height`]). A `T(z)` linear
///   over the patch is within its bounds, and so is a smooth vertical
///   extremum between thick layers;
/// - what is scaled is the departure `d_i = T_i − T̂_i` from the element's
///   own column at the node's height, about its inventory-weighted mean `m`:
///   `T_i ← T̂_i + m + α (d_i − m)`, which keeps the layer's inventory (the
///   change `−(1 − α)(d_i − m)` has zero weighted mean). At `α = 0` the layer
///   is its own column's stratification, not a flat mean (`m` is spread
///   within the bounds as far as they allow, see below).
///
/// Layer means cannot represent a *curved* `T(z)` at the nodes' heights where
/// the layers are thick against the curvature and steep across an element
/// (a pycnocline over a seamount's flank: up to 1.8 °C of 17 changed at rest). A
/// `reference` stratification takes the curvature out: the limiter works on
/// `T − T_ref(z)`, zero at rest for any horizontally uniform `T_ref`.
///
/// On a flat bed with `η` flat and no reference every node sits at its
/// layer's mean height, `T̂_i` is the layer's mean and `m = 0`: the classic
/// limiter.
///
/// A layer whose nodes are all within their own column's bounds is left
/// before its patches are read: those bounds are within every patch's, so
/// `α = 1`. This about halves the Kuzmin pass's cost (`profile_3d`: 2.2 →
/// 1.2 ms for `T` and `S` on 1024 P2 elements × 20 levels).
#[allow(clippy::too_many_arguments)]
fn apply_horizontal_kuzmin_field(
    field: &mut [f64],
    reference: Option<(&[f64], &[f64])>,
    columns: &LayerColumns,
    heights: &[f64],
    mesh: &Mesh2D,
    relaxation: f64,
    component: TracerComponent,
    stats: &mut TracerLimiter3DStats,
) {
    let (n_elements, n_nodes, n_levels) = (columns.n_elements, columns.n_nodes, columns.n_levels);
    // The reference at a node's layer centre
    let reference_at = |k: usize, i: usize, level: usize| {
        reference.map_or(0.0, |(heights, values)| {
            interpolate_profile(
                heights,
                values,
                columns.centre_height(ElementIndex::new(k), i, level),
            )
        })
    };
    let mut averages = Pooled::take(
        |a: &Vec<f64>| a.len() == n_elements * n_levels,
        || vec![f64::NAN; n_elements * n_levels],
    );
    // The range of every column's nodal departures in its bottom and top
    // layers
    let mut end_ranges = Pooled::take(
        |a: &Vec<[f64; 4]>| a.len() == n_elements,
        || vec![[f64::NAN; 4]; n_elements],
    );
    {
        let field: &[f64] = field;
        let departure =
            |k, i, level| field[index(k, i, level, n_nodes, n_levels)] - reference_at(k, i, level);
        columns.layer_means(&mut averages, departure);
        for_each_block(
            n_elements,
            [&mut end_ranges[..]],
            || (),
            |_, k, [range]| {
                let (bottom, top) = (0..n_nodes).fold(
                    (
                        [f64::INFINITY, f64::NEG_INFINITY],
                        [f64::INFINITY, f64::NEG_INFINITY],
                    ),
                    |([b0, b1], [t0, t1]), i| {
                        let (b, t) = (departure(k, i, 0), departure(k, i, n_levels - 1));
                        ([b0.min(b), b1.max(b)], [t0.min(t), t1.max(t)])
                    },
                );
                range[0] = [bottom[0], bottom[1], top[0], top[1]];
            },
        );
    }
    let averages: &[f64] = &averages;
    let end_ranges: &[[f64; 4]] = &end_ranges;
    let index = |i: usize, level: usize| i * n_levels + level;
    // Element `e`'s column at height `z` (near `level`), if it has water
    // there: its value and the means of its two layers bracketing `z`
    let column_at = |e: usize, z: f64, level: usize| {
        let column = e * n_levels..(e + 1) * n_levels;
        averages[column.start + level].is_finite().then(|| {
            column_at_height(
                &averages[column.clone()],
                &heights[column],
                end_ranges[e],
                z,
                level,
            )
        })
    };
    // The range over `elements`' columns at height `z`: each column's value
    // there and the means of its two layers bracketing `z`, widened by
    // `relaxation` and by round-off relative to the tracer's `scale`
    let bounds_at = |elements: &[usize], z: f64, level: usize, scale: f64| {
        let (bound_min, bound_max) = elements
            .iter()
            .filter_map(|&e| {
                column_at(e, z, level)
                    .map(|(value, a, b)| (value.min(a).min(b), value.max(a).max(b)))
            })
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (a, b)| {
                (lo.min(a), hi.max(b))
            });
        relaxed_bounds(bound_min, bound_max, relaxation, scale)
    };

    let block_stats = reduce_blocks(
        n_elements,
        [&mut field[..n_elements * n_nodes * n_levels]],
        || Pooled::take(|_: &KuzminScratch| true, KuzminScratch::default),
        |scratch, k, [field]| {
            let KuzminScratch {
                patches,
                height,
                reference,
                value,
                base,
                weight,
                bounds,
                scale,
            } = &mut **scratch;
            let mut stats = TracerLimiter3DStats::default();
            let element = ElementIndex::new(k);
            let vertices = mesh.element_vertex_indices(element);
            // The union of the element's vertex patches
            patches.clear();
            for &vertex in &vertices {
                for &e in mesh.elements_at_vertex(vertex) {
                    if !patches.contains(&e) {
                        patches.push(e);
                    }
                }
            }

            for (level, &ds) in columns.sigma.d_sigma().iter().enumerate().take(n_levels) {
                if !averages[k * n_levels + level].is_finite() {
                    continue;
                }
                // The element's own column at each node's height, `T̂_i`
                for buffer in [
                    &mut *height,
                    &mut *reference,
                    &mut *value,
                    &mut *base,
                    &mut *weight,
                    &mut *scale,
                ] {
                    buffer.clear();
                }
                let (mut weight_sum, mut departure) = (0.0, 0.0);
                // Whether every node is within its own column's bounds alone,
                // which its patch's contain (the element is in every patch):
                // then nothing is limited, and the patches need not be read
                let mut within_own = true;
                for i in 0..n_nodes {
                    let z = columns.centre_height(element, i, level);
                    let Some((own, lower, upper)) = column_at(k, z, level) else {
                        break;
                    };
                    let node_reference = reference_at(k, i, level);
                    let node_value = field[index(i, level)] - node_reference;
                    let node_weight = columns.geom.node_mass(k, i)
                        * layer_thickness(columns.eta, columns.bathymetry, element, i, ds);
                    let node_scale = field[index(i, level)].abs().max(node_reference.abs());
                    let (lo, hi) = relaxed_bounds(
                        own.min(lower).min(upper),
                        own.max(lower).max(upper),
                        relaxation,
                        node_scale,
                    );
                    within_own &= (lo..=hi).contains(&node_value);
                    weight_sum += node_weight;
                    departure += node_weight * (node_value - own);
                    height.push(z);
                    reference.push(node_reference);
                    value.push(node_value);
                    base.push(own);
                    weight.push(node_weight);
                    scale.push(node_scale);
                }
                if base.len() < n_nodes || within_own {
                    continue;
                }
                // The bounds: every vertex within its own patch's, every other
                // node (P2 and up) within the union's
                bounds.clear();
                for i in 0..n_nodes {
                    let patch = match node_to_vertex(i, columns.ops.n_1d) {
                        Some(local_vertex) => mesh.elements_at_vertex(vertices[local_vertex]),
                        None => &patches[..],
                    };
                    bounds.push(bounds_at(patch, height[i], level, scale[i]));
                }

                // The limited state at `α = 0`: `T̂` shifted by the departure's
                // mean `m` to keep the inventory. The shift goes into each
                // node's room within its bounds (`T̂_i` is one of its
                // candidates) in proportion to it, then into the room within
                // the element's widest bounds, then (if not even those hold
                // the layer's mean) uniformly. A shift as a factor on the
                // column's structure, or a fallback to the layer's mean, would
                // be ill-conditioned: that structure vanishes at nodes near the
                // layer's mean height, where a round-off violation would set
                // the factor, and flattening it over a slope drives a flow.
                let m = departure / weight_sum;
                let mut need = m.abs() * weight_sum;
                let widest = bounds
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &(lo, hi)| {
                        (a.min(lo), b.max(hi))
                    });
                let room_of = |own: f64, (lo, hi): (f64, f64)| {
                    if m > 0.0 { hi - own } else { own - lo }.max(0.0)
                };
                for tier in 0..2 {
                    let node_bounds = |i: usize| if tier == 0 { bounds[i] } else { widest };
                    let room: f64 = (0..n_nodes)
                        .map(|i| weight[i] * room_of(base[i], node_bounds(i)))
                        .sum();
                    if need <= 0.0 || !room.is_finite() || room <= 0.0 {
                        continue;
                    }
                    let fraction = (need / room).min(1.0);
                    for (i, own) in base.iter_mut().enumerate() {
                        *own += m.signum() * fraction * room_of(*own, node_bounds(i));
                    }
                    need -= need.min(room);
                }
                for own in base.iter_mut() {
                    *own += m.signum() * need / weight_sum;
                }

                let alpha = (0..n_nodes)
                    .map(|i| {
                        let (lo, hi) = bounds[i];
                        compute_kuzmin_alpha(base[i], value[i], lo, hi)
                    })
                    .fold(1.0_f64, f64::min);
                if alpha >= 1.0 - LIMITER_EPS {
                    continue;
                }

                for i in 0..n_nodes {
                    field[index(i, level)] = reference[i] + base[i] + alpha * (value[i] - base[i]);
                }
                stats.record_limited(component);
            }
            stats
        },
        TracerLimiter3DStats::default,
        TracerLimiter3DStats::merged,
    );
    *stats = stats.merged(block_stats);
}

/// A column's layer `means` (at `heights`, bottom first) at height `z`,
/// searching from `level`: `(value, lower, upper)`, the means interpolated
/// linearly to `z` (exact for a tracer linear in `z`) and the means of the
/// two layers whose heights bracket `z` (the end pair beyond them).
///
/// Beyond its end layers' means the value extrapolates the end pair, clamped
/// to the range of the column's nodal values in that end layer
/// (`[bottom min, bottom max, top min, top max]`). A node of the column
/// itself is never beyond them, so at the deepest node of a pit the
/// extrapolation is still exact for a linear `T(z)`; and a thin column (a
/// shoreline film, its layers mm apart) does not extend its last step over
/// metres.
///
/// Kuzmin's bounds take, from each neighbour column, the bracketing means as
/// well as the value, as Thetis's vertex patches on prisms take the cells
/// above and below (Kärnä et al. 2018): a node may hold what its 3D
/// neighbourhood holds. Bounded by the interpolated values alone, the limiter
/// cut every smooth vertical extremum between thick layers (the departure of
/// a displaced pycnocline peaks where `∂T/∂z` does): on the seamount at
/// `r_x0` 0.32 it changed 55–75 % of the element layers every stage, by up
/// to 0.2 °C at depth, and every such change over a slope is a pressure
/// gradient the advection did not make. With the bracketing means it limits
/// about half as many. On a flat bed a node's own layer is one of the pair: a
/// sharp interface between layers is bounded by both water masses' means
/// (the lock exchanges stay in range to 5e-12 °C).
fn column_at_height(
    means: &[f64],
    heights: &[f64],
    end_ranges: [f64; 4],
    z: f64,
    level: usize,
) -> (f64, f64, f64) {
    let n = means.len();
    if n == 1 {
        return (means[0], means[0], means[0]);
    }
    let mut lo = level.min(n - 2);
    while lo > 0 && z < heights[lo] {
        lo -= 1;
    }
    while lo + 2 < n && z > heights[lo + 1] {
        lo += 1;
    }
    let (lower, upper) = (means[lo], means[lo + 1]);
    let spacing = heights[lo + 1] - heights[lo];
    if spacing <= 0.0 {
        return (0.5 * (lower + upper), lower, upper);
    }
    let value = lower + (z - heights[lo]) / spacing * (upper - lower);
    let value = if z < heights[0] {
        value.clamp(end_ranges[0], end_ranges[1])
    } else if z > heights[n - 1] {
        value.clamp(end_ranges[2], end_ranges[3])
    } else {
        value
    };
    (value, lower, upper)
}

/// `[bound_min, bound_max]` widened by `relaxation` (≥ 1) times its range,
/// and by a round-off margin relative to its values and the tracer's
/// `scale` (bounds interpolated at a node's height close on its value in a
/// smooth stratification); unbounded for no data.
fn relaxed_bounds(bound_min: f64, bound_max: f64, relaxation: f64, scale: f64) -> (f64, f64) {
    if !bound_min.is_finite() || !bound_max.is_finite() {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }

    let range = bound_max - bound_min;
    let expand = 0.5 * range * (relaxation - 1.0).max(0.0)
        + ROUND_OFF * bound_min.abs().max(bound_max.abs()).max(scale);
    (bound_min - expand, bound_max + expand)
}

/// `values` at `heights` (increasing) interpolated linearly to `z`, constant
/// beyond the ends.
fn interpolate_profile(heights: &[f64], values: &[f64], z: f64) -> f64 {
    let j = heights.partition_point(|&h| h < z);
    if j == 0 {
        values[0]
    } else if j == heights.len() {
        values[j - 1]
    } else {
        let t = (z - heights[j - 1]) / (heights[j] - heights[j - 1]);
        values[j - 1] + t * (values[j] - values[j - 1])
    }
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

/// The local vertex at `node`, if it is one.
fn node_to_vertex(node: usize, n_1d: usize) -> Option<usize> {
    (0..4).find(|&local_vertex| vertex_to_node_index(local_vertex, n_1d) == node)
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
        let geom = GeometricFactors2D::compute(&mesh, &ops);
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
        geom: &GeometricFactors2D,
        bathymetry: &Bathymetry2D,
        sigma: &SigmaGrid,
    ) -> f64 {
        let mut total = 0.0;
        for k in 0..state.n_elements {
            let element = ElementIndex::new(k);
            for i in 0..state.n_nodes {
                for (level, &ds) in sigma.d_sigma().iter().enumerate() {
                    let weight = geom.node_mass(k, i)
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

        let before = total_inventory(&state.temp, &state, &geom, &bathymetry, &sigma);
        let config = TracerLimiter3DConfig::bounds(TracerBounds::new(0.0, 10.0, 0.0, 40.0));

        let stats =
            apply_tracer_limiters_3d(&mut state, &mesh, &ops, &geom, &bathymetry, &sigma, &config);

        let after = total_inventory(&state.temp, &state, &geom, &bathymetry, &sigma);
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
        let before = total_inventory(&state.temp, &state, &geom, &bathymetry, &sigma);
        let config = TracerLimiter3DConfig::bounds(TracerBounds::new(0.0, 10.0, 0.0, 40.0));

        let stats =
            apply_tracer_limiters_3d(&mut state, &mesh, &ops, &geom, &bathymetry, &sigma, &config);

        let after = total_inventory(&state.temp, &state, &geom, &bathymetry, &sigma);
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

    /// At P2 the vertices can sit at the element's mean while an edge or
    /// the centre overshoots; those nodes are bounded by the element's
    /// vertex patches too. Before, only the vertices were checked, and a
    /// P2 lock exchange overshot by 0.46 °C.
    #[test]
    fn horizontal_kuzmin_bounds_every_node_at_p2() {
        let (mesh, ops, geom, bathymetry, sigma, mut state) = setup(2, 1, 2, 1);
        let n_nodes = state.n_nodes;
        // Element 0: vertices 2 (its mean), edges −2, centre 6; element 1: 3
        for i in 0..ops.n_nodes {
            let (a, b) = (i % 3, i / 3);
            let value = match (a == 1) as u8 + (b == 1) as u8 {
                0 => 2.0,
                1 => -2.0,
                _ => 6.0,
            };
            state.temp[index(0, i, 0, n_nodes, 1)] = value;
            state.temp[index(1, i, 0, n_nodes, 1)] = 3.0;
        }
        let before = total_inventory(&state.temp, &state, &geom, &bathymetry, &sigma);
        let config = TracerLimiter3DConfig::horizontal_kuzmin(
            TracerBounds::new(-100.0, 100.0, 0.0, 40.0),
            1.0,
        );
        let stats =
            apply_tracer_limiters_3d(&mut state, &mesh, &ops, &geom, &bathymetry, &sigma, &config);
        let after = total_inventory(&state.temp, &state, &geom, &bathymetry, &sigma);
        assert!(stats.limited_temperature_cells > 0);
        for &value in &state.temp[..n_nodes] {
            assert!(
                (2.0 - 1e-12..=3.0 + 1e-12).contains(&value),
                "a node left the patch bounds [2, 3]: {value}"
            );
        }
        assert!(
            (before - after).abs() < 1e-10,
            "inventory changed: before={before}, after={after}"
        );
    }

    /// TODO P4.6 (regression): on a sloping bed a σ-layer crosses the
    /// stratification, so a `T(z)` varies along it, extremal at the top of a
    /// bump. Bounded by the neighbours' means on the same layer, the limiter
    /// clipped it as a front (a stratified seamount at rest reached 0.15 m/s
    /// in 15 min); bounded at constant height, it leaves it.
    #[test]
    fn horizontal_kuzmin_leaves_a_linear_stratification_over_a_bump() {
        use crate::vertical::SongHaidvogelStretching;
        let (mesh, ops, geom, _, _, mut state) = setup(5, 4, 2, 8);
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
            -100.0 + 70.0 * (-((x - 0.5).powi(2) + (y - 0.45).powi(2)) / 0.08).exp()
        });
        let sigma = SigmaGrid::new(8, SongHaidvogelStretching::new(5.0, 0.4, 20.0));
        let n_levels = sigma.n_levels();
        for k in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let bed = bathymetry.get(ElementIndex::new(k), i);
                let eta = 0.3 + 2e-3 * bed;
                state.eta.set(k, i, eta);
                for (level, &s) in sigma.sigma_rho().iter().enumerate() {
                    let z = eta + (eta - bed) * s;
                    state.temp[index(k, i, level, ops.n_nodes, n_levels)] = 10.0 + 0.05 * z;
                }
            }
        }
        let before = state.temp.clone();
        let config = TracerLimiter3DConfig::horizontal_kuzmin(
            TracerBounds::new(-100.0, 100.0, 0.0, 40.0),
            1.0,
        );
        let stats =
            apply_tracer_limiters_3d(&mut state, &mesh, &ops, &geom, &bathymetry, &sigma, &config);
        let change = before
            .iter()
            .zip(&state.temp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            stats.limited_temperature_cells == 0 && change == 0.0,
            "limited {} layers, by up to {change:.3e} °C",
            stats.limited_temperature_cells
        );
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

        let before = total_inventory(&state.temp, &state, &geom, &bathymetry, &sigma);
        let config = TracerLimiter3DConfig::horizontal_kuzmin(
            TracerBounds::new(-100.0, 100.0, 0.0, 40.0),
            1.0,
        );

        let stats =
            apply_tracer_limiters_3d(&mut state, &mesh, &ops, &geom, &bathymetry, &sigma, &config);

        let after = total_inventory(&state.temp, &state, &geom, &bathymetry, &sigma);
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
