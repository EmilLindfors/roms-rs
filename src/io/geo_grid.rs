//! Structured longitude/latitude grids of external models, and interpolation
//! from them onto arbitrary points.
//!
//! Parent ocean models (NorKyst-800, ROMS) and atmospheric models (MEPS,
//! MET Nordic, AROME-Arctic, ERA5) store fields on a structured grid of
//! `ny × nx` points with a longitude and latitude per point. The grid is
//! either *regular* (1D, monotone longitudes and latitudes, ascending or
//! descending) or *curvilinear* (2D arrays: a rotated polar-stereographic or
//! Lambert grid). [`GeoGrid`] locates a point in a grid cell and returns its
//! local cell coordinates `(fx, fy) ∈ [0, 1]²`, from which [`Stencil`] builds
//! bilinear interpolation weights.
//!
//! # Point location
//!
//! Regular grids are searched by bisection on each axis. On a curvilinear
//! grid, a bucket index over the cells' bounding boxes gives candidate
//! cells, and in each the inverse of the bilinear map
//!
//! ```text
//! P(fx, fy) = (1−fx)(1−fy) P₀₀ + fx(1−fy) P₀₁ + (1−fx) fy P₁₀ + fx fy P₁₁
//! ```
//!
//! (corners `P_ji` in longitude/latitude) is found by Newton's method; the
//! point belongs to the cell whose inverse lies in the unit square. The
//! bilinear weights are then exact in grid index space, whatever the shape
//! of the cells.
//!
//! # Axes in the model plane
//!
//! [`GeoGrid::plane_frame`] maps the cell around a point to the mesh plane
//! of a [`CoordinateProjection`]: the Jacobian `∂(x, y)/∂(fx, fy)`, for the
//! gradient of the bilinear interpolant in mesh coordinates, and the
//! direction of the grid's x axis, for rotating grid-relative vectors.
//! [`east_axis`] gives the direction of east, for rotating east/north
//! vectors into the mesh axes.

use crate::io::CoordinateProjection;

/// Location of a point in a grid: the cell with lower-left corner `(j, i)`
/// and local coordinates `(fx, fy) ∈ [0, 1]²` along `i` and `j`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridPoint {
    /// Row of the cell's first corner
    pub j: usize,
    /// Column of the cell's first corner
    pub i: usize,
    /// Local coordinate along `i` (0 at column `i`, 1 at `i + 1`)
    pub fx: f64,
    /// Local coordinate along `j` (0 at row `j`, 1 at `j + 1`)
    pub fy: f64,
}

/// Errors building a [`GeoGrid`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GeoGridError {
    /// Coordinate arrays of the wrong size.
    #[error("grid coordinates: expected {expected} values, got {got}")]
    Size {
        /// Expected number of values
        expected: usize,
        /// Actual number
        got: usize,
    },
    /// Fewer than two points along an axis.
    #[error("grid needs at least 2 × 2 points, got {ny} × {nx}")]
    TooSmall {
        /// Rows
        ny: usize,
        /// Columns
        nx: usize,
    },
    /// Regular-grid coordinates that are not strictly monotone.
    #[error("{axis} coordinates must be finite and strictly monotone")]
    NotMonotone {
        /// `longitude` or `latitude`
        axis: &'static str,
    },
}

/// A structured grid of `ny × nx` points with a longitude and latitude per
/// point, stored row-major (`j * nx + i`).
#[derive(Clone, Debug)]
pub struct GeoGrid {
    ny: usize,
    nx: usize,
    lon: Vec<f64>,
    lat: Vec<f64>,
    kind: GridKind,
}

#[derive(Clone, Debug)]
enum GridKind {
    /// Separable coordinates: `lon[i]`, `lat[j]`
    Regular {
        lon: Vec<f64>,
        lat: Vec<f64>,
    },
    Curvilinear(BucketIndex),
}

impl GeoGrid {
    /// A regular grid from 1D longitudes (`nx`) and latitudes (`ny`), each
    /// strictly ascending or strictly descending.
    pub fn regular(lon: Vec<f64>, lat: Vec<f64>) -> Result<Self, GeoGridError> {
        let (ny, nx) = (lat.len(), lon.len());
        if ny < 2 || nx < 2 {
            return Err(GeoGridError::TooSmall { ny, nx });
        }
        for (axis, values) in [("longitude", &lon), ("latitude", &lat)] {
            if !strictly_monotone(values) {
                return Err(GeoGridError::NotMonotone { axis });
            }
        }
        let lon_2d = (0..ny).flat_map(|_| lon.iter().copied()).collect();
        let lat_2d = lat
            .iter()
            .flat_map(|&v| std::iter::repeat_n(v, nx))
            .collect();
        Ok(Self {
            ny,
            nx,
            lon: lon_2d,
            lat: lat_2d,
            kind: GridKind::Regular { lon, lat },
        })
    }

    /// A grid from 2D longitudes and latitudes (`ny × nx`, row-major).
    ///
    /// A grid whose latitude is constant along rows and longitude along
    /// columns (common in files that store 2D coordinates for a regular grid)
    /// becomes [`Self::regular`].
    pub fn curvilinear(
        ny: usize,
        nx: usize,
        lon: Vec<f64>,
        lat: Vec<f64>,
    ) -> Result<Self, GeoGridError> {
        if ny < 2 || nx < 2 {
            return Err(GeoGridError::TooSmall { ny, nx });
        }
        for values in [&lon, &lat] {
            if values.len() != ny * nx {
                return Err(GeoGridError::Size {
                    expected: ny * nx,
                    got: values.len(),
                });
            }
        }
        const TOL: f64 = 1e-9;
        let separable = (0..ny)
            .all(|j| (0..nx).all(|i| (lat[j * nx + i] - lat[j * nx]).abs() < TOL))
            && (0..ny).all(|j| (0..nx).all(|i| (lon[j * nx + i] - lon[i]).abs() < TOL));
        if separable {
            let lon_1d: Vec<f64> = lon[..nx].to_vec();
            let lat_1d: Vec<f64> = (0..ny).map(|j| lat[j * nx]).collect();
            if strictly_monotone(&lon_1d) && strictly_monotone(&lat_1d) {
                return Self::regular(lon_1d, lat_1d);
            }
        }
        let index = BucketIndex::build(ny, nx, &lon, &lat);
        Ok(Self {
            ny,
            nx,
            lon,
            lat,
            kind: GridKind::Curvilinear(index),
        })
    }

    /// Rows and columns, `(ny, nx)`.
    pub fn dims(&self) -> (usize, usize) {
        (self.ny, self.nx)
    }

    /// Number of grid points, `ny · nx`.
    pub fn len(&self) -> usize {
        self.ny * self.nx
    }

    /// Whether the grid has no points (never true for a built grid).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the grid is regular (separable coordinates).
    pub fn is_regular(&self) -> bool {
        matches!(self.kind, GridKind::Regular { .. })
    }

    /// Flat index of point `(j, i)`.
    #[inline]
    pub fn index(&self, j: usize, i: usize) -> usize {
        j * self.nx + i
    }

    /// `(lon, lat)` of the point with flat index `index`.
    #[inline]
    pub fn position(&self, index: usize) -> (f64, f64) {
        (self.lon[index], self.lat[index])
    }

    /// Longitudes, row-major.
    pub fn lon(&self) -> &[f64] {
        &self.lon
    }

    /// Latitudes, row-major.
    pub fn lat(&self) -> &[f64] {
        &self.lat
    }

    /// Bounding box `(min_lon, min_lat, max_lon, max_lat)`.
    pub fn bbox(&self) -> (f64, f64, f64, f64) {
        let fold = |v: &[f64]| {
            v.iter()
                .filter(|x| x.is_finite())
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                    (lo.min(x), hi.max(x))
                })
        };
        let ((x0, x1), (y0, y1)) = (fold(&self.lon), fold(&self.lat));
        (x0, y0, x1, y1)
    }

    /// Flat indices of the corners of the cell of `p`: `[(j, i), (j, i+1),
    /// (j+1, i), (j+1, i+1)]`.
    #[inline]
    pub fn corners(&self, p: &GridPoint) -> [usize; 4] {
        let base = self.index(p.j, p.i);
        [base, base + 1, base + self.nx, base + self.nx + 1]
    }

    /// The cell holding `(lon, lat)` and the local coordinates in it, `None`
    /// outside the grid.
    pub fn locate(&self, lon: f64, lat: f64) -> Option<GridPoint> {
        if !(lon.is_finite() && lat.is_finite()) {
            return None;
        }
        match &self.kind {
            GridKind::Regular {
                lon: lons,
                lat: lats,
            } => {
                let (i, fx) = bracket(lons, lon)?;
                let (j, fy) = bracket(lats, lat)?;
                Some(GridPoint { j, i, fx, fy })
            }
            GridKind::Curvilinear(index) => {
                index.candidates(lon, lat).iter().find_map(|&(j, i)| {
                    let (j, i) = (j as usize, i as usize);
                    let (fx, fy) = inverse_bilinear(self.cell_corners(j, i), (lon, lat))?;
                    Some(GridPoint { j, i, fx, fy })
                })
            }
        }
    }

    /// Corner coordinates of cell `(j, i)`, in the order of [`Self::corners`].
    fn cell_corners(&self, j: usize, i: usize) -> [(f64, f64); 4] {
        let c = self.corners(&GridPoint {
            j,
            i,
            fx: 0.0,
            fy: 0.0,
        });
        c.map(|k| (self.lon[k], self.lat[k]))
    }

    /// The point among those with `accept(index)` nearest to `(lon, lat)`
    /// within `max_distance` metres, and its distance.
    ///
    /// A linear scan: meant for the few points bilinear interpolation cannot
    /// serve (e.g. a boundary node inside a parent-model land cell).
    pub fn nearest(
        &self,
        lon: f64,
        lat: f64,
        max_distance: f64,
        accept: impl Fn(usize) -> bool,
    ) -> Option<(usize, f64)> {
        (0..self.len())
            .filter(|&k| accept(k))
            .map(|k| (k, geodesic_distance((lon, lat), self.position(k))))
            .filter(|&(_, d)| d <= max_distance)
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// The cell around `p` in the mesh plane of `projection`.
    pub fn plane_frame<P: CoordinateProjection + ?Sized>(
        &self,
        p: &GridPoint,
        projection: &P,
    ) -> PlaneFrame {
        let [c00, c01, c10, c11] = self
            .corners(p)
            .map(|k| projection.geo_to_xy(self.lat[k], self.lon[k]));
        let (fx, fy) = (p.fx, p.fy);
        let d_dfx = (
            (1.0 - fy) * (c01.0 - c00.0) + fy * (c11.0 - c10.0),
            (1.0 - fy) * (c01.1 - c00.1) + fy * (c11.1 - c10.1),
        );
        let d_dfy = (
            (1.0 - fx) * (c10.0 - c00.0) + fx * (c11.0 - c01.0),
            (1.0 - fx) * (c10.1 - c00.1) + fx * (c11.1 - c01.1),
        );
        PlaneFrame { d_dfx, d_dfy }
    }

    /// Angle (radians, counter-clockwise) from east to the grid's x axis at
    /// every point, from centred differences of the coordinates (one-sided at
    /// the edges). Rotates grid-relative vector components `(a, b)` to east
    /// and north: `e = a cos α − b sin α`, `n = a sin α + b cos α`.
    pub fn x_axis_angles(&self) -> Vec<f64> {
        let (ny, nx) = (self.ny, self.nx);
        let mut angles = Vec::with_capacity(ny * nx);
        for j in 0..ny {
            for i in 0..nx {
                let (i0, i1) = (i.saturating_sub(1), (i + 1).min(nx - 1));
                let (a, b) = (self.index(j, i0), self.index(j, i1));
                let lat_mid = 0.5 * (self.lat[a] + self.lat[b]);
                let de = (self.lon[b] - self.lon[a]) * lat_mid.to_radians().cos();
                let dn = self.lat[b] - self.lat[a];
                angles.push(dn.atan2(de));
            }
        }
        angles
    }
}

/// A grid cell mapped to the mesh plane: the columns of the Jacobian
/// `∂(x, y)/∂(fx, fy)` at a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneFrame {
    /// `∂(x, y)/∂fx`: the grid's x axis in the plane (m per cell)
    pub d_dfx: (f64, f64),
    /// `∂(x, y)/∂fy`: the grid's y axis in the plane (m per cell)
    pub d_dfy: (f64, f64),
}

impl PlaneFrame {
    /// Weights `(∂w/∂x, ∂w/∂y)` of the four corners' values in the gradient
    /// of the bilinear interpolant at `(fx, fy)`, in the corner order of
    /// [`GeoGrid::corners`]: `∇f = Σ f_c (∂w_c/∂x, ∂w_c/∂y)`.
    pub fn gradient_weights(&self, fx: f64, fy: f64) -> [(f64, f64); 4] {
        // ∂w/∂(fx, fy) of the bilinear weights
        let dw = [
            (-(1.0 - fy), -(1.0 - fx)),
            (1.0 - fy, -fx),
            (-fy, 1.0 - fx),
            (fy, fx),
        ];
        // J = [[∂x/∂fx, ∂x/∂fy], [∂y/∂fx, ∂y/∂fy]]; ∇_xy w = J⁻ᵀ ∇_f w
        let (a, c) = self.d_dfx;
        let (b, d) = self.d_dfy;
        let det = a * d - b * c;
        dw.map(|(wx, wy)| ((d * wx - c * wy) / det, (-b * wx + a * wy) / det))
    }

    /// Unit vector of the grid's x axis in the plane, `(cos θ, sin θ)`.
    pub fn x_axis(&self) -> (f64, f64) {
        let (x, y) = self.d_dfx;
        let n = x.hypot(y);
        (x / n, y / n)
    }
}

/// Unit vector of local east in the mesh plane of `projection` at
/// `(lat, lon)`, `(cos θ, sin θ)`: east/north components `(e, n)` become
/// mesh components `u = e cos θ − n sin θ`, `v = e sin θ + n cos θ`.
///
/// Taken perpendicular (clockwise) to the direction of local north, i.e. for
/// a conformal projection.
pub fn east_axis<P: CoordinateProjection + ?Sized>(
    projection: &P,
    lat: f64,
    lon: f64,
) -> (f64, f64) {
    let (x0, y0) = projection.geo_to_xy(lat, lon);
    let (x1, y1) = projection.geo_to_xy(lat + 1e-4, lon);
    let (nx, ny) = (x1 - x0, y1 - y0);
    let n = nx.hypot(ny);
    (ny / n, -nx / n)
}

/// Bilinear weights of the four corners at local coordinates `(fx, fy)`, in
/// the order of [`GeoGrid::corners`].
#[inline]
pub fn bilinear_weights(fx: f64, fy: f64) -> [f64; 4] {
    [
        (1.0 - fx) * (1.0 - fy),
        fx * (1.0 - fy),
        (1.0 - fx) * fy,
        fx * fy,
    ]
}

/// Interpolation weights over at most four grid points: `Σ wₖ f[idx[k]]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stencil {
    /// Flat grid indices
    pub idx: [u32; 4],
    /// Weights, summing to 1 (unused entries 0)
    pub w: [f64; 4],
}

impl Stencil {
    /// A single grid point.
    pub fn point(index: usize) -> Self {
        Self {
            idx: [index as u32; 4],
            w: [1.0, 0.0, 0.0, 0.0],
        }
    }

    /// Bilinear weights at `p`, renormalised over the corners with
    /// `valid(index)` (e.g. wet points of a parent model); `None` when no
    /// corner is valid. Entries of weight 0 index a valid corner, so a
    /// field that is NaN only at invalid points interpolates to a number.
    pub fn bilinear(grid: &GeoGrid, p: &GridPoint, valid: impl Fn(usize) -> bool) -> Option<Self> {
        let idx = grid.corners(p);
        let mut w = bilinear_weights(p.fx, p.fy);
        for (wk, &k) in w.iter_mut().zip(&idx) {
            if !valid(k) {
                *wk = 0.0;
            }
        }
        let total: f64 = w.iter().sum();
        // Invalid corners keep weight 0 but point at a valid one, so that
        // their (NaN) values never enter a sum
        let fallback = *idx.iter().find(|&&k| valid(k))?;
        if total <= 1e-12 {
            // The point sits on invalid corners only (weights of the valid
            // ones vanish there): fall back to the valid corner nearest in
            // index space.
            let k = (0..4)
                .filter(|&c| valid(idx[c]))
                .min_by(|&a, &b| corner_distance(p, a).total_cmp(&corner_distance(p, b)))?;
            return Some(Self::point(idx[k]));
        }
        Some(Self {
            idx: std::array::from_fn(|c| if valid(idx[c]) { idx[c] } else { fallback } as u32),
            w: w.map(|wk| wk / total),
        })
    }

    /// `Σ wₖ f[idx[k]]`.
    #[inline]
    pub fn apply(&self, f: &[f32]) -> f64 {
        self.idx
            .iter()
            .zip(&self.w)
            .map(|(&k, &w)| w * f[k as usize] as f64)
            .sum()
    }
}

fn corner_distance(p: &GridPoint, corner: usize) -> f64 {
    let (cx, cy) = ((corner % 2) as f64, (corner / 2) as f64);
    (p.fx - cx).hypot(p.fy - cy)
}

/// Great-circle distance (m) between two `(lon, lat)` points (haversine,
/// spherical Earth).
pub fn geodesic_distance((lon1, lat1): (f64, f64), (lon2, lat2): (f64, f64)) -> f64 {
    const R: f64 = 6_371_000.0;
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dp = p2 - p1;
    let dl = (lon2 - lon1).to_radians();
    let a = (0.5 * dp).sin().powi(2) + p1.cos() * p2.cos() * (0.5 * dl).sin().powi(2);
    2.0 * R * a.sqrt().min(1.0).asin()
}

fn strictly_monotone(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
        && (values.windows(2).all(|w| w[1] > w[0]) || values.windows(2).all(|w| w[1] < w[0]))
}

/// Interval `k` of strictly monotone `coords` holding `value` and the local
/// coordinate in it (0 at `coords[k]`, 1 at `coords[k + 1]`); `None`
/// outside. Works for ascending and descending coordinates alike.
fn bracket(coords: &[f64], value: f64) -> Option<(usize, f64)> {
    let n = coords.len();
    let ascending = coords[1] > coords[0];
    let (lo, hi) = if ascending {
        (coords[0], coords[n - 1])
    } else {
        (coords[n - 1], coords[0])
    };
    if !(value >= lo && value <= hi) {
        return None;
    }
    // First index whose coordinate is past `value`, in the direction of the axis
    let past = if ascending {
        coords.partition_point(|&c| c <= value)
    } else {
        coords.partition_point(|&c| c >= value)
    };
    let k = past.clamp(1, n - 1) - 1;
    let f = (value - coords[k]) / (coords[k + 1] - coords[k]);
    Some((k, f.clamp(0.0, 1.0)))
}

/// Local coordinates of `target` in the bilinear cell with corners `c`
/// (order of [`GeoGrid::corners`]), if they lie in the unit square.
fn inverse_bilinear(c: [(f64, f64); 4], target: (f64, f64)) -> Option<(f64, f64)> {
    const EPS: f64 = 1e-9;
    if c.iter().any(|p| !(p.0.is_finite() && p.1.is_finite())) {
        return None;
    }
    let (mut fx, mut fy) = (0.5, 0.5);
    for _ in 0..20 {
        let w = bilinear_weights(fx, fy);
        let (x, y) = c
            .iter()
            .zip(&w)
            .fold((0.0, 0.0), |(x, y), (p, &wk)| (x + wk * p.0, y + wk * p.1));
        let (rx, ry) = (x - target.0, y - target.1);
        // Jacobian of the bilinear map
        let a = (1.0 - fy) * (c[1].0 - c[0].0) + fy * (c[3].0 - c[2].0);
        let b = (1.0 - fx) * (c[2].0 - c[0].0) + fx * (c[3].0 - c[1].0);
        let cc = (1.0 - fy) * (c[1].1 - c[0].1) + fy * (c[3].1 - c[2].1);
        let d = (1.0 - fx) * (c[2].1 - c[0].1) + fx * (c[3].1 - c[1].1);
        let det = a * d - b * cc;
        if det.abs() < 1e-300 {
            return None;
        }
        let dfx = (d * rx - b * ry) / det;
        let dfy = (-cc * rx + a * ry) / det;
        fx -= dfx;
        fy -= dfy;
        if !(fx.is_finite() && fy.is_finite()) {
            return None;
        }
        if dfx.abs() < 1e-13 && dfy.abs() < 1e-13 {
            break;
        }
    }
    ((-EPS..=1.0 + EPS).contains(&fx) && (-EPS..=1.0 + EPS).contains(&fy))
        .then(|| (fx.clamp(0.0, 1.0), fy.clamp(0.0, 1.0)))
}

/// Cells of a curvilinear grid binned by their bounding boxes on a regular
/// longitude/latitude grid of buckets.
#[derive(Clone, Debug)]
struct BucketIndex {
    /// Cells per bucket: `cells[start[b]..start[b + 1]]`
    start: Vec<u32>,
    cells: Vec<(u32, u32)>,
    n_bx: usize,
    n_by: usize,
    min_lon: f64,
    min_lat: f64,
    width: f64,
    height: f64,
}

impl BucketIndex {
    fn build(ny: usize, nx: usize, lon: &[f64], lat: &[f64]) -> Self {
        let n_cells = (ny - 1) * (nx - 1);
        let side = ((n_cells as f64).sqrt().ceil() as usize).clamp(1, 1024);
        let finite = |v: &[f64]| {
            v.iter()
                .filter(|x| x.is_finite())
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                    (lo.min(x), hi.max(x))
                })
        };
        let ((min_lon, max_lon), (min_lat, max_lat)) = (finite(lon), finite(lat));
        let width = ((max_lon - min_lon) / side as f64).max(1e-12);
        let height = ((max_lat - min_lat) / side as f64).max(1e-12);
        let bucket = |v: f64, min: f64, size: f64| {
            (((v - min) / size).floor().max(0.0) as usize).min(side - 1)
        };

        // Bucket ranges per cell, then a counting sort into CSR
        let mut ranges = Vec::with_capacity(n_cells);
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let k = [
                    j * nx + i,
                    j * nx + i + 1,
                    (j + 1) * nx + i,
                    (j + 1) * nx + i + 1,
                ];
                if k.iter()
                    .any(|&k| !(lon[k].is_finite() && lat[k].is_finite()))
                {
                    continue;
                }
                let (x0, x1) = k
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &k| {
                        (a.min(lon[k]), b.max(lon[k]))
                    });
                let (y0, y1) = k
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &k| {
                        (a.min(lat[k]), b.max(lat[k]))
                    });
                ranges.push((
                    (j as u32, i as u32),
                    bucket(x0, min_lon, width)..=bucket(x1, min_lon, width),
                    bucket(y0, min_lat, height)..=bucket(y1, min_lat, height),
                ));
            }
        }
        let mut count = vec![0u32; side * side + 1];
        for (_, bx, by) in &ranges {
            for y in by.clone() {
                for x in bx.clone() {
                    count[y * side + x + 1] += 1;
                }
            }
        }
        for b in 0..side * side {
            count[b + 1] += count[b];
        }
        let mut fill = count.clone();
        let mut cells = vec![(0, 0); count[side * side] as usize];
        for (cell, bx, by) in &ranges {
            for y in by.clone() {
                for x in bx.clone() {
                    let slot = &mut fill[y * side + x];
                    cells[*slot as usize] = *cell;
                    *slot += 1;
                }
            }
        }
        Self {
            start: count,
            cells,
            n_bx: side,
            n_by: side,
            min_lon,
            min_lat,
            width,
            height,
        }
    }

    fn candidates(&self, lon: f64, lat: f64) -> &[(u32, u32)] {
        // A point on the upper edge of the grid belongs to the last bucket
        let bucket = |v: f64, min: f64, size: f64, n: usize| {
            let b = ((v - min) / size).floor();
            (b >= 0.0 && b <= n as f64).then(|| (b as usize).min(n - 1))
        };
        let bx = bucket(lon, self.min_lon, self.width, self.n_bx);
        let by = bucket(lat, self.min_lat, self.height, self.n_by);
        match (bx, by) {
            (Some(x), Some(y)) => {
                let b = y * self.n_bx + x;
                &self.cells[self.start[b] as usize..self.start[b + 1] as usize]
            }
            _ => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::LocalProjection;

    /// A rotated, sheared curvilinear grid.
    fn rotated_grid(ny: usize, nx: usize) -> GeoGrid {
        let (mut lon, mut lat) = (Vec::new(), Vec::new());
        for j in 0..ny {
            for i in 0..nx {
                let (x, y) = (i as f64, j as f64);
                lon.push(8.0 + 0.01 * (x * 0.9 - y * 0.4) + 0.0005 * x * y);
                lat.push(63.0 + 0.005 * (x * 0.3 + y * 0.8));
            }
        }
        GeoGrid::curvilinear(ny, nx, lon, lat).unwrap()
    }

    /// Every point given by the forward bilinear map is found in its cell
    /// with the same local coordinates.
    #[test]
    fn curvilinear_location_inverts_the_bilinear_map() {
        let grid = rotated_grid(7, 9);
        assert!(!grid.is_regular());
        for (j, i, fx, fy) in [
            (0, 0, 0.3, 0.7),
            (3, 5, 0.9, 0.1),
            (5, 7, 0.5, 0.5),
            (2, 1, 0.0, 1.0),
        ] {
            let p = GridPoint { j, i, fx, fy };
            let w = bilinear_weights(fx, fy);
            let c = grid.corners(&p);
            let lon: f64 = c.iter().zip(&w).map(|(&k, w)| w * grid.lon()[k]).sum();
            let lat: f64 = c.iter().zip(&w).map(|(&k, w)| w * grid.lat()[k]).sum();
            let found = grid.locate(lon, lat).unwrap();
            // A point on a shared edge may be found in either cell
            let back: (f64, f64) = {
                let w = bilinear_weights(found.fx, found.fy);
                let c = grid.corners(&found);
                (
                    c.iter().zip(&w).map(|(&k, w)| w * grid.lon()[k]).sum(),
                    c.iter().zip(&w).map(|(&k, w)| w * grid.lat()[k]).sum(),
                )
            };
            assert!((back.0 - lon).abs() < 1e-12 && (back.1 - lat).abs() < 1e-12);
            if fy < 1.0 {
                assert_eq!((found.j, found.i), (j, i));
                assert!((found.fx - fx).abs() < 1e-10 && (found.fy - fy).abs() < 1e-10);
            }
        }
        assert!(grid.locate(0.0, 0.0).is_none());
        assert!(grid.locate(7.99, 63.0).is_none());
    }

    /// Descending coordinates (e.g. latitude from north to south, as in ERA5)
    /// bracket the right interval: the old `find_bracket` returned the upper
    /// index with the weight of the lower one.
    #[test]
    fn regular_grid_with_descending_latitude() {
        let grid = GeoGrid::regular(vec![5.0, 5.5, 6.0], vec![61.0, 60.5, 60.0]).unwrap();
        assert!(grid.is_regular());
        let p = grid.locate(5.2, 60.6).unwrap();
        assert_eq!((p.j, p.i), (0, 0));
        assert!((p.fx - 0.4).abs() < 1e-12 && (p.fy - 0.8).abs() < 1e-12);
        // Interpolating the latitude itself recovers the point
        let s = Stencil::bilinear(&grid, &p, |_| true).unwrap();
        let lat: Vec<f32> = grid.lat().iter().map(|&v| v as f32).collect();
        assert!((s.apply(&lat) - 60.6).abs() < 1e-5);
        // Edges belong to the grid, beyond is outside
        assert!(grid.locate(6.0, 60.0).is_some() && grid.locate(5.0, 61.0).is_some());
        assert!(grid.locate(6.01, 60.5).is_none() && grid.locate(5.5, 61.01).is_none());
        assert!(GeoGrid::regular(vec![5.0, 5.5, 5.2], vec![60.0, 61.0]).is_err());
    }

    /// 2D coordinates of a regular grid become a regular grid.
    #[test]
    fn separable_curvilinear_is_regular() {
        let (lon, lat) = ([4.0, 4.1, 4.2], [59.0, 59.1]);
        let lon2: Vec<f64> = (0..2).flat_map(|_| lon).collect();
        let lat2: Vec<f64> = lat.iter().flat_map(|&v| [v; 3]).collect();
        let grid = GeoGrid::curvilinear(2, 3, lon2, lat2).unwrap();
        assert!(grid.is_regular());
        assert_eq!(grid.dims(), (2, 3));
    }

    #[test]
    fn masked_stencil_renormalises_and_falls_back() {
        let grid = GeoGrid::regular(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
        let p = GridPoint {
            j: 0,
            i: 0,
            fx: 0.25,
            fy: 0.5,
        };
        // Corner 1 (fx = 1, fy = 0) invalid
        let s = Stencil::bilinear(&grid, &p, |k| k != 1).unwrap();
        assert!((s.w.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        assert_eq!(s.w[1], 0.0);
        let f = [1.0_f32, f32::NAN, 3.0, 4.0];
        assert!(s.apply(&f).is_finite());
        // On an invalid corner: the nearest valid one
        let p = GridPoint {
            j: 0,
            i: 0,
            fx: 1.0,
            fy: 0.0,
        };
        let s = Stencil::bilinear(&grid, &p, |k| k != 1).unwrap();
        assert_eq!(s.w[0], 1.0);
        assert!(Stencil::bilinear(&grid, &p, |_| false).is_none());
    }

    /// The gradient of the bilinear interpolant of a field linear in the
    /// plane is exact, on a rotated grid.
    #[test]
    fn gradient_weights_are_exact_for_linear_fields() {
        let grid = rotated_grid(5, 5);
        let proj = LocalProjection::new(63.01, 8.02);
        let field: Vec<f64> = (0..grid.len())
            .map(|k| {
                let (lon, lat) = grid.position(k);
                let (x, y) = proj.geo_to_xy(lat, lon);
                3.0 + 2e-3 * x - 5e-4 * y
            })
            .collect();
        // The local projection is affine in lon/lat, so the projected cell is
        // bilinear and a linear field is bilinear in (fx, fy)
        let p = GridPoint {
            j: 1,
            i: 2,
            fx: 0.3,
            fy: 0.6,
        };
        let frame = grid.plane_frame(&p, &proj);
        let g = frame.gradient_weights(p.fx, p.fy);
        let c = grid.corners(&p);
        let (gx, gy) = c.iter().zip(&g).fold((0.0, 0.0), |(x, y), (&k, w)| {
            (x + w.0 * field[k], y + w.1 * field[k])
        });
        assert!(
            (gx - 2e-3).abs() < 1e-12 && (gy + 5e-4).abs() < 1e-12,
            "({gx}, {gy})"
        );
    }

    #[test]
    fn east_axis_and_grid_angles() {
        let proj = LocalProjection::new(63.0, 8.0);
        let (c, s) = east_axis(&proj, 63.0, 8.0);
        assert!((c - 1.0).abs() < 1e-9 && s.abs() < 1e-9);
        // A grid rotated 30° from east (in metres), built in the local plane
        let angle = 30f64.to_radians();
        let (mut lon, mut lat) = (Vec::new(), Vec::new());
        for j in 0..3 {
            for i in 0..3 {
                let (x, y) = (1000.0 * i as f64, 1000.0 * j as f64);
                let (xr, yr) = (
                    x * angle.cos() - y * angle.sin(),
                    x * angle.sin() + y * angle.cos(),
                );
                let (la, lo) = proj.xy_to_geo(xr, yr);
                lon.push(lo);
                lat.push(la);
            }
        }
        let grid = GeoGrid::curvilinear(3, 3, lon, lat).unwrap();
        for a in grid.x_axis_angles() {
            assert!((a - angle).abs() < 1e-3, "{} vs {}", a.to_degrees(), 30.0);
        }
        let frame = grid.plane_frame(
            &GridPoint {
                j: 0,
                i: 0,
                fx: 0.5,
                fy: 0.5,
            },
            &proj,
        );
        let (c, s) = frame.x_axis();
        assert!((c - angle.cos()).abs() < 1e-9 && (s - angle.sin()).abs() < 1e-9);
    }

    #[test]
    fn geodesic_distance_of_a_tenth_degree() {
        let d = geodesic_distance((8.0, 63.0), (8.0, 63.1));
        assert!((d - 11_119.5).abs() < 1.0, "{d}");
    }
}
