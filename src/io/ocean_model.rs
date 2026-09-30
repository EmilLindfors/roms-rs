//! Parent ocean-model fields (NorKyst-800, ROMS) for nesting.
//!
//! [`OceanModelReader`] holds what a 2D barotropic child needs from its
//! parent, on the parent's own grid: the sea surface height `ζ`, the
//! depth-mean velocity `(ū, v̄)` as **east/north** components, and the
//! still-water depth `h`. The fields are [`FieldSeries`] strided
//! `[time][point]`, missing values (land) are NaN, and times are Unix
//! seconds.
//!
//! [`OceanModelReader::from_file`] (feature `netcdf`) reads NetCDF output:
//! it takes the barotropic `ubar`/`vbar` when present, else depth-averages
//! 3D velocities on z-levels or s-levels, and rotates grid-relative
//! components to east/north. [`OceanModelReader::new`] builds one from
//! arrays (tests, other formats).
//!
//! Point queries interpolate bilinearly over the wet corners of the grid
//! cell and return `None` where there is no data: nothing is silently zero.
//! Nesting on a mesh does not query points at run time; it precomputes
//! stencils once (`boundary::ParentNesting`).

use super::field_series::{FieldSeries, TimeInterpolation, TimeStencil};
use super::geo_grid::{GeoGrid, GridPoint, Stencil};
use super::profile_series::ProfileSeries;

/// Parent state at one point and time.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OceanState {
    /// Sea surface height ζ (m), if the parent has it
    pub ssh: Option<f64>,
    /// Depth-mean velocity (east, north) (m/s), if the parent has it
    pub velocity: Option<(f64, f64)>,
    /// Still-water depth h (m, positive), if the parent has it
    pub depth: Option<f64>,
    /// Surface temperature (°C), if available
    pub temperature: Option<f64>,
    /// Surface salinity, if available
    pub salinity: Option<f64>,
}

impl OceanState {
    /// Current speed (m/s).
    pub fn speed(&self) -> Option<f64> {
        self.velocity.map(|(u, v)| u.hypot(v))
    }

    /// Direction the current flows towards (degrees clockwise from north).
    pub fn direction(&self) -> Option<f64> {
        self.velocity
            .map(|(u, v)| (90.0 - v.atan2(u).to_degrees()).rem_euclid(360.0))
    }
}

/// Parent-model fields on the parent grid; see the module docs.
#[derive(Clone, Debug)]
pub struct OceanModelReader {
    /// Parent grid (ρ-points)
    pub grid: GeoGrid,
    /// Snapshot times, Unix seconds (UTC), strictly increasing. A single
    /// snapshot is time-invariant.
    pub time: Vec<f64>,
    /// Sea surface height ζ (m)
    pub ssh: Option<FieldSeries>,
    /// Depth-mean eastward velocity (m/s)
    pub u: Option<FieldSeries>,
    /// Depth-mean northward velocity (m/s)
    pub v: Option<FieldSeries>,
    /// Still-water depth h (m, positive down), NaN on land
    pub depth: Option<Vec<f64>>,
    /// Surface temperature (°C)
    pub temperature: Option<FieldSeries>,
    /// Surface salinity
    pub salinity: Option<FieldSeries>,
    /// How the velocity was obtained (e.g. `ubar_eastward/vbar_northward`,
    /// `u_eastward/v_northward averaged over z-levels`)
    pub velocity_source: Option<String>,
    /// Profiles of the eastward and northward velocity (m/s), for 3D nesting
    /// (see [`Self::with_profiles`]).
    pub u_profile: Option<ProfileSeries>,
    pub v_profile: Option<ProfileSeries>,
    /// Profiles of temperature (°C) and salinity.
    pub temperature_profile: Option<ProfileSeries>,
    pub salinity_profile: Option<ProfileSeries>,
}

impl OceanModelReader {
    /// Fields on `grid` at snapshot `time` (Unix seconds, strictly
    /// increasing); add fields with the `with_*` methods.
    pub fn new(grid: GeoGrid, time: Vec<f64>) -> Result<Self, String> {
        if time.is_empty() {
            return Err("OceanModelReader: no snapshots".into());
        }
        if time.iter().any(|t| !t.is_finite()) || time.windows(2).any(|w| w[1] <= w[0]) {
            return Err("OceanModelReader: times must be finite and strictly increasing".into());
        }
        Ok(Self {
            grid,
            time,
            ssh: None,
            u: None,
            v: None,
            depth: None,
            temperature: None,
            salinity: None,
            velocity_source: None,
            u_profile: None,
            v_profile: None,
            temperature_profile: None,
            salinity_profile: None,
        })
    }

    fn check(&self, name: &str, series: &FieldSeries) {
        assert!(
            series.n_points() == self.grid.len() && series.n_times() == self.time.len(),
            "OceanModelReader: {name} has {} × {} values, expected {} snapshots × {} points",
            series.n_times(),
            series.n_points(),
            self.time.len(),
            self.grid.len()
        );
    }

    /// Set the sea surface height (m).
    pub fn with_ssh(mut self, ssh: FieldSeries) -> Self {
        self.check("ssh", &ssh);
        self.ssh = Some(ssh);
        self
    }

    /// Set the depth-mean velocity (east, north) (m/s).
    pub fn with_velocity(mut self, u: FieldSeries, v: FieldSeries, source: &str) -> Self {
        self.check("u", &u);
        self.check("v", &v);
        self.u = Some(u);
        self.v = Some(v);
        self.velocity_source = Some(source.to_string());
        self
    }

    /// Set the still-water depth (m, positive; NaN on land).
    pub fn with_depth(mut self, depth: Vec<f64>) -> Self {
        assert_eq!(depth.len(), self.grid.len(), "OceanModelReader: depth size");
        self.depth = Some(depth);
        self
    }

    /// Set the surface temperature and salinity.
    pub fn with_tracers(
        mut self,
        temperature: Option<FieldSeries>,
        salinity: Option<FieldSeries>,
    ) -> Self {
        for (name, s) in [("temperature", &temperature), ("salinity", &salinity)] {
            if let Some(s) = s {
                self.check(name, s);
            }
        }
        self.temperature = temperature;
        self.salinity = salinity;
        self
    }

    /// Set the profiles for 3D nesting: velocity (east, north; m/s) and
    /// temperature and salinity, those present.
    pub fn with_profiles(
        mut self,
        velocity: Option<(ProfileSeries, ProfileSeries)>,
        temperature: Option<ProfileSeries>,
        salinity: Option<ProfileSeries>,
    ) -> Self {
        let (u, v) = velocity.unzip();
        for (name, p) in [
            ("u profile", &u),
            ("v profile", &v),
            ("temperature profile", &temperature),
            ("salinity profile", &salinity),
        ] {
            if let Some(p) = p {
                assert!(
                    p.n_points() == self.grid.len() && p.n_times() == self.time.len(),
                    "OceanModelReader: {name} has {} × {} profiles, expected {} snapshots × {} points",
                    p.n_times(),
                    p.n_points(),
                    self.time.len(),
                    self.grid.len()
                );
            }
        }
        self.u_profile = u;
        self.v_profile = v;
        self.temperature_profile = temperature;
        self.salinity_profile = salinity;
        self
    }

    /// Whether the parent has velocity profiles.
    pub fn has_current_profiles(&self) -> bool {
        self.u_profile.is_some() && self.v_profile.is_some()
    }

    /// Whether the parent has temperature and salinity profiles.
    pub fn has_tracer_profiles(&self) -> bool {
        self.temperature_profile.is_some() && self.salinity_profile.is_some()
    }

    /// Grid dimensions `(ny, nx)`.
    pub fn dims(&self) -> (usize, usize) {
        self.grid.dims()
    }

    /// Bounding box `(min_lon, min_lat, max_lon, max_lat)`.
    pub fn bbox(&self) -> (f64, f64, f64, f64) {
        self.grid.bbox()
    }

    /// Number of snapshots.
    pub fn n_times(&self) -> usize {
        self.time.len()
    }

    /// First and last snapshot time (Unix seconds).
    pub fn time_range(&self) -> Option<(f64, f64)> {
        Some((*self.time.first()?, *self.time.last()?))
    }

    /// Whether `time` (Unix seconds) lies within the snapshots (always true
    /// for a single, time-invariant snapshot).
    pub fn covers_time(&self, time: f64) -> bool {
        TimeStencil::new(&self.time, time, TimeInterpolation::Linear).is_some()
    }

    /// Weights of the snapshots at `time`, `None` outside the file.
    pub fn time_stencil(&self, time: f64, method: TimeInterpolation) -> Option<TimeStencil> {
        TimeStencil::new(&self.time, time, method)
    }

    /// Whether the parent has sea surface height.
    pub fn has_ssh(&self) -> bool {
        self.ssh.is_some()
    }

    /// Whether the parent has velocities.
    pub fn has_currents(&self) -> bool {
        self.u.is_some() && self.v.is_some()
    }

    /// Whether the parent has temperature.
    pub fn has_temperature(&self) -> bool {
        self.temperature.is_some()
    }

    /// Whether the parent has salinity.
    pub fn has_salinity(&self) -> bool {
        self.salinity.is_some()
    }

    /// Whether grid point `k` is wet: its depth (if known) is positive and
    /// its SSH and velocity (those present) have values at every snapshot.
    pub fn is_wet(&self, k: usize) -> bool {
        self.depth.as_ref().is_none_or(|d| d[k] > 0.0)
            && [&self.ssh, &self.u, &self.v]
                .into_iter()
                .flatten()
                .all(|s| s.is_valid(k))
            && (self.has_ssh() || self.has_currents() || self.depth.is_some())
    }

    /// Wet flag per grid point (see [`Self::is_wet`]).
    pub fn wet_mask(&self) -> Vec<bool> {
        (0..self.grid.len()).map(|k| self.is_wet(k)).collect()
    }

    /// Bilinear stencil over the wet corners of the cell holding
    /// `(lon, lat)`, `None` outside the grid or with no wet corner.
    pub fn stencil(&self, lon: f64, lat: f64) -> Option<(GridPoint, Stencil)> {
        let p = self.grid.locate(lon, lat)?;
        let s = Stencil::bilinear(&self.grid, &p, |k| self.is_wet(k))?;
        Some((p, s))
    }

    /// State at `(lon, lat)` at snapshot `time_idx`, interpolated over the
    /// wet corners; `None` outside the grid or on land.
    pub fn get_state(&self, lon: f64, lat: f64, time_idx: usize) -> Option<OceanState> {
        if time_idx >= self.time.len() {
            return None;
        }
        let single = TimeStencil {
            idx: [time_idx, 0, 0, 0],
            w: [1.0, 0.0, 0.0, 0.0],
            len: 1,
        };
        self.state_at(lon, lat, &single)
    }

    /// State at `(lon, lat)` and `time` (Unix seconds), linear in time;
    /// `None` outside the file's time range (no clamping), outside the grid
    /// or on land.
    pub fn get_state_interpolated(&self, lon: f64, lat: f64, time: f64) -> Option<OceanState> {
        let stencil = TimeStencil::new(&self.time, time, TimeInterpolation::Linear)?;
        self.state_at(lon, lat, &stencil)
    }

    fn state_at(&self, lon: f64, lat: f64, time: &TimeStencil) -> Option<OceanState> {
        let (_, s) = self.stencil(lon, lat)?;
        Some(self.evaluate(&s, time))
    }

    /// State from a precomputed spatial and time stencil.
    pub fn evaluate(&self, s: &Stencil, time: &TimeStencil) -> OceanState {
        let space = s.idx.iter().map(|&k| k as usize).zip(s.w);
        let field = |f: &Option<FieldSeries>| {
            f.as_ref()
                .map(|f| f.interpolate(time, space.clone()))
                .filter(|v| v.is_finite())
        };
        let depth = self
            .depth
            .as_ref()
            .map(|d| space.clone().map(|(k, w)| w * d[k]).sum::<f64>())
            .filter(|v| v.is_finite());
        OceanState {
            ssh: field(&self.ssh),
            velocity: field(&self.u).zip(field(&self.v)),
            depth,
            temperature: field(&self.temperature),
            salinity: field(&self.salinity),
        }
    }

    /// One-line description of the grid, times and fields.
    pub fn summary(&self) -> String {
        let (ny, nx) = self.dims();
        let (x0, y0, x1, y1) = self.bbox();
        let mut vars: Vec<String> = Vec::new();
        if self.has_ssh() {
            vars.push("SSH".into());
        }
        if let Some(source) = &self.velocity_source {
            vars.push(format!("currents ({source})"));
        }
        if self.depth.is_some() {
            vars.push("depth".into());
        }
        if self.has_temperature() {
            vars.push("temperature".into());
        }
        if self.has_salinity() {
            vars.push("salinity".into());
        }
        if let Some(p) = &self.u_profile {
            vars.push(format!("current profiles ({} levels)", p.n_levels()));
        }
        if let Some(p) = &self.temperature_profile {
            vars.push(format!("T/S profiles ({} levels)", p.n_levels()));
        }
        format!(
            "Ocean model: {ny}x{nx} {} grid, {} times, lon [{x0:.2}, {x1:.2}], lat [{y0:.2}, {y1:.2}], vars: {}",
            if self.grid.is_regular() {
                "regular"
            } else {
                "curvilinear"
            },
            self.time.len(),
            vars.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3 × 3 regular grid, two snapshots; the (2, 2) corner is land.
    fn reader() -> OceanModelReader {
        let grid = GeoGrid::regular(vec![8.0, 8.1, 8.2], vec![63.0, 63.1, 63.2]).unwrap();
        let n = grid.len();
        let land = 8;
        let field = |f: &dyn Fn(usize, usize) -> f64| {
            let data = (0..2)
                .flat_map(|t| (0..n).map(move |k| (t, k)))
                .map(|(t, k)| if k == land { f32::NAN } else { f(t, k) as f32 })
                .collect();
            FieldSeries::new(n, data)
        };
        let mut depth = vec![40.0; n];
        depth[land] = f64::NAN;
        OceanModelReader::new(grid, vec![0.0, 3600.0])
            .unwrap()
            .with_ssh(field(&|t, _| 0.1 + 0.2 * t as f64))
            .with_velocity(field(&|_, k| k as f64), field(&|_, _| -0.1), "test")
            .with_depth(depth)
    }

    #[test]
    fn states_are_interpolated_over_wet_corners() {
        let r = reader();
        assert!(!r.is_wet(8) && r.is_wet(4));
        // Centre of the cell (1, 1): corner (2, 2) is land, the other three
        // share the weights
        let s = r.get_state_interpolated(8.15, 63.15, 1800.0).unwrap();
        assert!((s.ssh.unwrap() - 0.2).abs() < 1e-6);
        let (u, v) = s.velocity.unwrap();
        assert!((u - (4.0 + 5.0 + 7.0) / 3.0).abs() < 1e-6, "{u}");
        assert!((v + 0.1).abs() < 1e-6);
        assert_eq!(s.depth, Some(40.0));
        // Outside the grid, time or on land: None, never zeros
        assert!(r.get_state_interpolated(7.9, 63.1, 0.0).is_none());
        assert!(r.get_state_interpolated(8.1, 63.1, 3601.0).is_none());
        assert!(r.get_state(8.2, 63.2, 0).is_some_and(|s| s.ssh.is_some()));
        assert!(r.get_state(8.1, 63.1, 2).is_none());
    }

    #[test]
    fn missing_fields_are_none() {
        let grid = GeoGrid::regular(vec![8.0, 8.1], vec![63.0, 63.1]).unwrap();
        let r = OceanModelReader::new(grid, vec![0.0])
            .unwrap()
            .with_ssh(FieldSeries::new(4, vec![0.3; 4]));
        let s = r.get_state(8.05, 63.05, 0).unwrap();
        assert!((s.ssh.unwrap() - 0.3).abs() < 1e-6);
        assert_eq!((s.velocity, s.depth), (None, None));
        assert!(r.covers_time(1e9));
        assert!(OceanModelReader::new(r.grid.clone(), vec![1.0, 1.0]).is_err());
    }

    #[test]
    fn direction_is_towards() {
        let s = OceanState {
            velocity: Some((0.0, -1.0)),
            ..Default::default()
        };
        assert_eq!(s.direction(), Some(180.0));
        assert_eq!(s.speed(), Some(1.0));
    }
}
