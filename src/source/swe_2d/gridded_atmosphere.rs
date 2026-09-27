//! Wind stress and atmospheric pressure from a weather model's grid.
//!
//! [`GriddedAtmosphere2D`] forces the 2D SWE with the 10 m wind and the
//! sea-level pressure of an [`AtmosphereReader`] (MET Nordic, MEPS,
//! AROME-Arctic, ERA5):
//!
//! ```text
//! S_hu = τ_x/ρ − (h/ρ) ∂p/∂x,    S_hv = τ_y/ρ − (h/ρ) ∂p/∂y,
//! τ = ρ_air C_d(|U₁₀|) |U₁₀| U₁₀
//! ```
//!
//! as [`WindStress2D`](super::WindStress2D) and
//! [`AtmosphericPressure2D`](super::AtmosphericPressure2D) do for analytic
//! fields. Everything spatial is precomputed per mesh node: the bilinear
//! stencil in the weather-model grid, the weights of the *exact* gradient of
//! that bilinear interpolant in mesh coordinates (for ∇p), and the rotation
//! from east/north into the mesh axes. At run time a node costs one time
//! stencil (linear: weather fields change on fronts, where cubic
//! interpolation would overshoot) and a few dozen multiply-adds.
//!
//! The sea settles under a pressure field at the inverse-barometer level
//! `η = −(p − p_ref)/(ρ g)`, `p_ref` = 101 325 Pa. Open boundaries whose
//! external data lack that response (tides, a parent model forced without
//! pressure) must add it, or they pin the boundary to the wrong level and
//! drive a current through it: [`GriddedAtmosphere2D`] implements
//! [`BoundaryLevel`], for
//! [`InverseBarometer`](crate::boundary::InverseBarometer).
//!
//! Both forcings can be ramped up from zero over a spin-up time
//! ([`GriddedAtmosphere2D::with_ramp_up`]); the inverse-barometer level
//! ramps with them.

use std::sync::{Arc, RwLock};

use crate::boundary::{BCContext2D, BoundaryLevel, tidal_ramp};
use crate::io::{
    AtmosphereReader, CoordinateProjection, P_REFERENCE, Stencil, TimeInterpolation, TimeStencil,
    east_axis,
};
use crate::mesh::Mesh2D;
use crate::operators::DGOperators2D;
use crate::solver::SWEState2D;
use crate::source::{ElementSources, SourceContext2D, SourceTerm2D};
use crate::time::ModelClock;
use crate::types::ElementIndex;

use super::wind::{DragCoefficient, RHO_AIR, RHO_WATER};

const G: f64 = 9.81;

/// Precomputed sampling of the weather grid at one mesh node.
#[derive(Clone, Copy, Debug)]
struct AtmosphereNode {
    stencil: Stencil,
    /// `(∂w/∂x, ∂w/∂y)` of the stencil's corners
    gradient: [(f64, f64); 4],
    /// East in the mesh plane, `(cos θ, sin θ)`
    east: (f64, f64),
}

impl AtmosphereNode {
    /// Sampling of `grid` at mesh position `(x, y)`, `None` outside it.
    fn at(
        grid: &crate::io::GeoGrid,
        projection: &(dyn CoordinateProjection + Send + Sync),
        (x, y): (f64, f64),
    ) -> Option<Self> {
        let (lat, lon) = projection.xy_to_geo(x, y);
        let p = grid.locate(lon, lat)?;
        Some(Self {
            stencil: Stencil::bilinear(grid, &p, |_| true)?,
            gradient: grid
                .plane_frame(&p, projection)
                .gradient_weights(p.fx, p.fy),
            east: east_axis(projection, lat, lon),
        })
    }
}

struct Inner {
    reader: Arc<AtmosphereReader>,
    projection: Arc<dyn CoordinateProjection + Send + Sync>,
    clock: ModelClock,
    nodes: Vec<AtmosphereNode>,
    n_nodes: usize,
    drag: DragCoefficient,
    rho_air: f64,
    rho_water: f64,
    h_min: f64,
    ramp: Option<f64>,
    wind: bool,
    pressure: bool,
    /// Recently used snapshots regridded onto the nodes
    cache: RwLock<Vec<(usize, Arc<Snapshot>)>>,
}

/// One weather snapshot on the mesh nodes: `[u, v, ∂p/∂x, ∂p/∂y, p − p_ref]`
/// per node, the wind in the mesh axes (unramped).
type Snapshot = Vec<[f32; 5]>;

/// Snapshots kept regridded: the bracketing pair and the neighbours a
/// multirate step may reach.
const CACHED_SNAPSHOTS: usize = 4;

thread_local! {
    /// Per-thread references to regridded snapshots, `(source, snapshot,
    /// values)`: the RHS kernels look snapshots up once per element on
    /// every thread, and a shared lock (or a shared reference count) there
    /// made the forcing cost 4× the RHS at 24 threads.
    static LOCAL_SNAPSHOTS: std::cell::RefCell<Vec<(usize, usize, Arc<Snapshot>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Wind stress and pressure-gradient force from gridded weather-model
/// fields; see the module docs. Cheap to clone.
#[derive(Clone)]
pub struct GriddedAtmosphere2D {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for GriddedAtmosphere2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GriddedAtmosphere2D")
            .field("reader", &self.inner.reader.summary())
            .field("nodes", &self.inner.nodes.len())
            .field("drag", &self.inner.drag)
            .field("ramp", &self.inner.ramp)
            .field("wind", &self.inner.wind)
            .field("pressure", &self.inner.pressure)
            .finish()
    }
}

impl GriddedAtmosphere2D {
    /// Sample `reader` at every node of `mesh`. Mesh coordinates map to
    /// longitude/latitude with `projection`, simulation time to UTC with
    /// `clock`. Every node must lie inside the weather grid.
    ///
    /// Wind stress uses Large & Pond (1981) drag, ρ_air = 1.225 kg/m³,
    /// ρ = 1025 kg/m³; nodes shallower than 1 cm are not forced.
    pub fn new<P: CoordinateProjection + Send + Sync + 'static>(
        reader: Arc<AtmosphereReader>,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        projection: P,
        clock: ModelClock,
    ) -> Result<Self, String> {
        let projection: Arc<dyn CoordinateProjection + Send + Sync> = Arc::new(projection);
        let n_nodes = ops.n_nodes;
        let mut nodes = Vec::with_capacity(mesh.n_elements * n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..n_nodes {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let node = AtmosphereNode::at(&reader.grid, projection.as_ref(), (x, y))
                    .ok_or_else(|| {
                        let (lat, lon) = projection.xy_to_geo(x, y);
                        format!(
                            "GriddedAtmosphere2D: mesh node at (x, y) = ({x:.0}, {y:.0}) m \
                             (lat {lat:.4}, lon {lon:.4}) is outside the weather grid"
                        )
                    })?;
                nodes.push(node);
            }
        }
        let (wind, pressure) = (reader.has_wind(), reader.has_pressure());
        Ok(Self {
            inner: Arc::new(Inner {
                reader,
                projection,
                clock,
                nodes,
                n_nodes,
                drag: DragCoefficient::LargePond,
                rho_air: RHO_AIR,
                rho_water: RHO_WATER,
                h_min: 0.01,
                ramp: None,
                wind,
                pressure,
                cache: RwLock::new(Vec::new()),
            }),
        })
    }

    fn inner_mut(&mut self) -> &mut Inner {
        Arc::get_mut(&mut self.inner).expect("configure GriddedAtmosphere2D before cloning it")
    }

    /// Wind drag coefficient (default Large & Pond).
    pub fn with_drag(mut self, drag: DragCoefficient) -> Self {
        self.inner_mut().drag = drag;
        self
    }

    /// Ramp both forcings (and the inverse-barometer level) up from zero
    /// over `duration` seconds of simulation time.
    pub fn with_ramp_up(mut self, duration: f64) -> Self {
        self.inner_mut().ramp = Some(duration);
        self
    }

    /// Switch the wind stress off (pressure only).
    pub fn without_wind(mut self) -> Self {
        self.inner_mut().wind = false;
        self
    }

    /// Switch the pressure gradient off (wind only).
    pub fn without_pressure(mut self) -> Self {
        self.inner_mut().pressure = false;
        self
    }

    /// The weather fields.
    pub fn reader(&self) -> &AtmosphereReader {
        &self.inner.reader
    }

    /// Check that simulation times `[t_start, t_end]` lie within the weather
    /// file.
    pub fn check_time_coverage(&self, t_start: f64, t_end: f64) -> Result<(), String> {
        for t in [t_start, t_end] {
            if !self.inner.reader.covers_time(self.inner.clock.unix(t)) {
                return Err(self.coverage_message(t));
            }
        }
        Ok(())
    }

    fn coverage_message(&self, t: f64) -> String {
        let (t0, t1) = self.inner.reader.time_range();
        let clock = &self.inner.clock;
        format!(
            "GriddedAtmosphere2D: simulation time {t} s is outside the weather file, which covers \
             simulation times [{}, {}] s (epoch {})",
            clock.model_time(t0),
            clock.model_time(t1),
            clock.format(0.0)
        )
    }

    fn time_stencil(&self, t: f64) -> TimeStencil {
        TimeStencil::new(
            &self.inner.reader.time,
            self.inner.clock.unix(t),
            TimeInterpolation::Linear,
        )
        // Out-of-range times are a configuration error, not a reason to
        // freeze the forcing
        .unwrap_or_else(|| panic!("{}", self.coverage_message(t)))
    }

    /// Wind (mesh axes) and pressure gradient at a node, ramped.
    #[inline]
    fn fields(
        &self,
        node: &AtmosphereNode,
        time: &TimeStencil,
        ramp: f64,
    ) -> ((f64, f64), (f64, f64)) {
        let inner = &self.inner;
        let reader = &inner.reader;
        let space = node.stencil.idx.iter().map(|&k| k as usize);
        let wind = match (&reader.u10, &reader.v10) {
            (Some(u), Some(v)) if inner.wind => {
                let w = space.clone().zip(node.stencil.w);
                let (e, n) = (u.interpolate(time, w.clone()), v.interpolate(time, w));
                let (c, s) = node.east;
                (ramp * (e * c - n * s), ramp * (e * s + n * c))
            }
            _ => (0.0, 0.0),
        };
        let gradient = match &reader.pressure {
            Some(p) if inner.pressure => (
                ramp * p.interpolate(time, space.clone().zip(node.gradient.map(|g| g.0))),
                ramp * p.interpolate(time, space.zip(node.gradient.map(|g| g.1))),
            ),
            _ => (0.0, 0.0),
        };
        (wind, gradient)
    }

    /// Momentum source at depth `h` from wind `(u, v)` and pressure gradient.
    #[inline]
    fn source(&self, h: f64, (u, v): (f64, f64), (px, py): (f64, f64)) -> (f64, f64) {
        let inner = &self.inner;
        if h < inner.h_min {
            return (0.0, 0.0);
        }
        let speed = u.hypot(v);
        let stress = inner.rho_air * inner.drag.compute(speed) * speed / inner.rho_water;
        (
            stress * u - h * px / inner.rho_water,
            stress * v - h * py / inner.rho_water,
        )
    }

    /// Node sampling at a mesh position, computed on the fly.
    fn node_at(&self, position: (f64, f64)) -> Option<AtmosphereNode> {
        AtmosphereNode::at(
            &self.inner.reader.grid,
            self.inner.projection.as_ref(),
            position,
        )
    }

    /// Inverse-barometer level (m) at a node.
    fn inverse_barometer(&self, node: &AtmosphereNode, t: f64) -> f64 {
        let inner = &self.inner;
        let Some(p) = inner.reader.pressure.as_ref().filter(|_| inner.pressure) else {
            return 0.0;
        };
        let time = self.time_stencil(t);
        let space = node
            .stencil
            .idx
            .iter()
            .map(|&k| k as usize)
            .zip(node.stencil.w);
        let ramp = tidal_ramp(t, inner.ramp);
        -ramp * (p.interpolate(&time, space) - P_REFERENCE) / (inner.rho_water * G)
    }
}

impl GriddedAtmosphere2D {
    /// Snapshot `t` on the nodes, from the shared cache or regridded now.
    fn shared_snapshot(&self, t: usize) -> Arc<Snapshot> {
        let inner = &self.inner;
        if let Some((_, s)) = inner.cache.read().unwrap().iter().find(|(k, _)| *k == t) {
            return Arc::clone(s);
        }
        let single = TimeStencil {
            idx: [t, 0, 0, 0],
            w: [1.0, 0.0, 0.0, 0.0],
            len: 1,
        };
        let pressure = inner.reader.pressure.as_ref().filter(|_| inner.pressure);
        let values: Snapshot = inner
            .nodes
            .iter()
            .map(|node| {
                let ((u, v), (px, py)) = self.fields(node, &single, 1.0);
                let p = pressure.map_or(0.0, |p| {
                    let space = node
                        .stencil
                        .idx
                        .iter()
                        .map(|&k| k as usize)
                        .zip(node.stencil.w);
                    p.interpolate(&single, space) - P_REFERENCE
                });
                [u, v, px, py, p].map(|x| x as f32)
            })
            .collect();
        let values = Arc::new(values);
        let mut cache = inner.cache.write().unwrap();
        if let Some((_, s)) = cache.iter().find(|(k, _)| *k == t) {
            return Arc::clone(s);
        }
        if cache.len() == CACHED_SNAPSHOTS {
            cache.remove(0);
        }
        cache.push((t, Arc::clone(&values)));
        values
    }

    /// Call `f` with the regridded snapshots of `time` and their weights,
    /// from this thread's cache.
    fn with_snapshots<R>(
        &self,
        time: &TimeStencil,
        f: impl FnOnce(&[(&[[f32; 5]], f64)]) -> R,
    ) -> R {
        let id = Arc::as_ptr(&self.inner) as usize;
        for (t, _) in time.terms() {
            let hit = LOCAL_SNAPSHOTS.with_borrow(|l| l.iter().any(|e| e.0 == id && e.1 == t));
            if !hit {
                let snapshot = self.shared_snapshot(t);
                LOCAL_SNAPSHOTS.with_borrow_mut(|l| {
                    if l.len() >= 2 * CACHED_SNAPSHOTS {
                        l.remove(0);
                    }
                    l.push((id, t, snapshot));
                });
            }
        }
        LOCAL_SNAPSHOTS.with_borrow(|l| {
            let mut refs: [(&[[f32; 5]], f64); 4] = [(&[], 0.0); 4];
            for (slot, (t, w)) in refs.iter_mut().zip(time.terms()) {
                let entry = l
                    .iter()
                    .find(|e| e.0 == id && e.1 == t)
                    .expect("inserted above");
                *slot = (&entry.2[..], w);
            }
            f(&refs[..time.len])
        })
    }

    /// `[u, v, ∂p/∂x, ∂p/∂y, p − p_ref]` at flat node `node`, unramped.
    #[inline]
    fn cached(snapshots: &[(&[[f32; 5]], f64)], node: usize) -> [f64; 5] {
        let mut sum = [0.0; 5];
        for (s, w) in snapshots {
            for (x, &v) in sum.iter_mut().zip(&s[node]) {
                *x += w * v as f64;
            }
        }
        sum
    }
}

impl SourceTerm2D for GriddedAtmosphere2D {
    /// Per-node evaluation by position, sampling the grid on the fly (slow;
    /// the RHS kernels use [`SourceTerm2D::add_element`]).
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        let Some(node) = self.node_at(ctx.position) else {
            return SWEState2D::zero();
        };
        let ramp = tidal_ramp(ctx.time, self.inner.ramp);
        let (wind, gradient) = self.fields(&node, &self.time_stencil(ctx.time), ramp);
        let (su, sv) = self.source(ctx.state.h, wind, gradient);
        SWEState2D::new(0.0, su, sv)
    }

    fn add_element(
        &self,
        element: &ElementSources<'_>,
        _h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
    ) {
        let inner = &self.inner;
        let ramp = tidal_ramp(element.time, inner.ramp);
        let base = element.element.as_usize() * inner.n_nodes;
        let depths = element.solution.element_h(element.element);
        self.with_snapshots(&self.time_stencil(element.time), |snapshots| {
            for (i, &h) in depths.iter().enumerate() {
                if h < inner.h_min {
                    continue;
                }
                let [u, v, px, py, _] = Self::cached(snapshots, base + i).map(|x| ramp * x);
                let (su, sv) = self.source(h, (u, v), (px, py));
                hu[i] += su;
                hv[i] += sv;
            }
        });
    }

    fn name(&self) -> &'static str {
        "gridded_atmosphere_2d"
    }
}

impl BoundaryLevel for GriddedAtmosphere2D {
    /// The inverse-barometer level at the boundary node.
    fn level(&self, ctx: &BCContext2D) -> f64 {
        match ctx.node_index.filter(|&i| i < self.inner.nodes.len()) {
            Some(i) => {
                let inner = &self.inner;
                if inner.reader.pressure.is_none() || !inner.pressure {
                    return 0.0;
                }
                let p = self.with_snapshots(&self.time_stencil(ctx.time), |snapshots| {
                    Self::cached(snapshots, i)[4]
                });
                -tidal_ramp(ctx.time, inner.ramp) * p / (inner.rho_water * G)
            }
            None => self
                .node_at(ctx.position)
                .map_or(0.0, |n| self.inverse_barometer(&n, ctx.time)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{FieldSeries, GeoGrid, LocalProjection};
    use crate::mesh::BoundaryTag;
    use crate::solver::SWESolution2D;

    const T0: f64 = 1_750_000_000.0;

    /// A weather grid over ±30 km around the projection origin with a wind
    /// of (east, north) = (10, 0) m/s and pressure rising 1 hPa per 10 km to
    /// the east, then 2 hPa after an hour.
    fn reader(projection: &LocalProjection) -> Arc<AtmosphereReader> {
        let (lat0, lon0) = projection.xy_to_geo(-30_000.0, -30_000.0);
        let (lat1, lon1) = projection.xy_to_geo(30_000.0, 30_000.0);
        let n = 7;
        let lon: Vec<f64> = (0..n)
            .map(|i| lon0 + (lon1 - lon0) * i as f64 / 6.0)
            .collect();
        let lat: Vec<f64> = (0..n)
            .map(|j| lat0 + (lat1 - lat0) * j as f64 / 6.0)
            .collect();
        let grid = GeoGrid::regular(lon.clone(), lat.clone()).unwrap();
        let m = grid.len();
        let pressure: Vec<f64> = [1.0, 2.0]
            .iter()
            .flat_map(|&scale| {
                let lat = lat.clone();
                let lon = lon.clone();
                (0..m).map(move |k| {
                    let (x, _) = projection.geo_to_xy(lat[k / n], lon[k % n]);
                    P_REFERENCE + scale * 0.01 * x
                })
            })
            .collect();
        Arc::new(
            AtmosphereReader::new(grid, vec![T0, T0 + 3600.0])
                .unwrap()
                .with_wind(
                    FieldSeries::new(m, vec![10.0; 2 * m]),
                    FieldSeries::new(m, vec![0.0; 2 * m]),
                )
                .with_pressure(FieldSeries::with_offset(m, &pressure, P_REFERENCE)),
        )
    }

    fn setup() -> (Mesh2D, DGOperators2D, GriddedAtmosphere2D) {
        let projection = LocalProjection::new(63.5, 8.5);
        let mesh = Mesh2D::uniform_rectangle_with_bc(
            -10_000.0,
            10_000.0,
            -10_000.0,
            10_000.0,
            4,
            4,
            BoundaryTag::Open,
        );
        let ops = DGOperators2D::new(2);
        let atmosphere = GriddedAtmosphere2D::new(
            reader(&projection),
            &mesh,
            &ops,
            projection,
            ModelClock::new(T0),
        )
        .unwrap();
        (mesh, ops, atmosphere)
    }

    /// Wind stress of Large & Pond at 10 m/s, and the force of a uniform
    /// pressure gradient, interpolated in time.
    #[test]
    fn element_source_is_wind_stress_and_pressure_gradient() {
        let (mesh, ops, atmosphere) = setup();
        let h = 20.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        q.h_data_mut().fill(h);
        let element = ElementSources {
            element: ElementIndex::new(5),
            time: 1800.0,
            solution: &q,
            mesh: &mesh,
            ops: &ops,
            bathymetry: None,
            g: G,
            h_min: 1e-6,
        };
        let (mut sh, mut su, mut sv) = (vec![0.0; 9], vec![0.0; 9], vec![0.0; 9]);
        atmosphere.add_element(&element, &mut sh, &mut su, &mut sv);
        let stress = RHO_AIR * 1.2e-3 * 100.0 / RHO_WATER;
        // ∂p/∂x = 1.5 · 0.01 Pa/m at half past
        let expected = stress - h * 0.015 / RHO_WATER;
        for i in 0..9 {
            assert!((su[i] - expected).abs() < 1e-9, "{} vs {expected}", su[i]);
            assert!(sv[i].abs() < 1e-12 && sh[i] == 0.0);
        }
        // Per-node evaluation by position agrees
        let s = atmosphere.evaluate(&element.context(4));
        assert!((s.hu - expected).abs() < 1e-7, "{} vs {expected}", s.hu);
    }

    /// The inverse-barometer level is −(p − p_ref)/(ρg), and ramps.
    #[test]
    fn inverse_barometer_level() {
        let (_, _, atmosphere) = setup();
        let ctx = |t: f64, x: f64| {
            BCContext2D::new(
                t,
                (x, 0.0),
                SWEState2D::new(10.0, 0.0, 0.0),
                -10.0,
                (1.0, 0.0),
                G,
                1e-6,
            )
        };
        let level = atmosphere.level(&ctx(0.0, 10_000.0));
        assert!((level + 100.0 / (RHO_WATER * G)).abs() < 1e-6, "{level}");
        // By node index: the level at that node's position
        let (mesh, ops, _) = setup();
        let [x, _] =
            mesh.reference_to_physical(ElementIndex::new(3), ops.nodes_r[2], ops.nodes_s[2]);
        let by_index = atmosphere.level(&ctx(0.0, 0.0).with_node_index(3 * 9 + 2));
        assert!(
            (by_index + 0.01 * x / (RHO_WATER * G)).abs() < 1e-9,
            "{by_index} at x = {x}"
        );
        let ramped = atmosphere.with_ramp_up(7200.0);
        assert_eq!(ramped.level(&ctx(0.0, 10_000.0)), 0.0);
    }

    #[test]
    fn mesh_outside_the_grid_is_an_error() {
        let projection = LocalProjection::new(63.5, 8.5);
        let mesh =
            Mesh2D::uniform_rectangle_with_bc(0.0, 50_000.0, 0.0, 1000.0, 5, 1, BoundaryTag::Wall);
        let ops = DGOperators2D::new(1);
        assert!(
            GriddedAtmosphere2D::new(
                reader(&projection),
                &mesh,
                &ops,
                projection,
                ModelClock::new(T0)
            )
            .is_err()
        );
    }
}
