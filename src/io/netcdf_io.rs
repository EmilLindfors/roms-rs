//! NetCDF I/O for oceanographic simulations.
//!
//! This module provides CF-conventions compliant NetCDF output for simulation results
//! and readers for forcing data (wind, atmospheric pressure).
//!
//! # Features
//!
//! - **Writer**: Export simulation results (sea level, currents, tracers) to NetCDF4
//! - **Reader**: Load forcing data from ERA5, ECMWF, or similar sources
//!
//! # CF-Conventions
//!
//! Output files follow CF-1.8 conventions:
//! - Standard coordinate variables (time, lat, lon)
//! - Standard names for variables (sea_surface_height, eastward_sea_water_velocity, etc.)
//! - Time encoded as "seconds since 1970-01-01" (Unix epoch)
//!
//! # Example
//!
//! ```rust,ignore
//! use dg_rs::io::{NetCDFWriter, NetCDFWriterConfig};
//!
//! let config = NetCDFWriterConfig::new("output.nc")
//!     .with_title("Froya Simulation")
//!     .with_institution("NTNU");
//!
//! let mut writer = NetCDFWriter::create(config, &mesh)?;
//! writer.write_timestep(0.0, &solution)?;
//! ```

use std::path::Path;

#[cfg(feature = "netcdf")]
use chrono::Utc;
#[cfg(feature = "netcdf")]
use netcdf::create;
use thiserror::Error;

/// Error type for NetCDF operations.
#[derive(Debug, Error)]
pub enum NetCDFError {
    /// File I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// NetCDF library error
    #[cfg(feature = "netcdf")]
    #[error("NetCDF error: {0}")]
    NetCDF(#[from] netcdf::Error),

    /// Invalid data
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// Missing variable
    #[error("Missing variable: {0}")]
    MissingVariable(String),

    /// Feature not enabled
    #[error("NetCDF feature not enabled")]
    FeatureDisabled,
}

/// Fill value for missing data (CF-conventions standard).
pub const FILL_VALUE_F64: f64 = 9.96920996838687e+36;
pub const FILL_VALUE_F32: f32 = 9.96921e+36;

/// Check if a value is valid (not a fill value).
#[inline]
pub fn is_valid_f32(v: f32) -> bool {
    v.is_finite() && v.abs() < 1.0e+30
}

/// Check if a value is valid (not a fill value).
#[inline]
pub fn is_valid_f64(v: f64) -> bool {
    v.is_finite() && v.abs() < 1.0e+30
}

// ============================================================================
// NetCDF Writer
// ============================================================================

/// Configuration for NetCDF output.
#[derive(Debug, Clone)]
pub struct NetCDFWriterConfig {
    /// Output file path
    pub path: String,
    /// Title attribute (CF-conventions)
    pub title: Option<String>,
    /// Institution attribute
    pub institution: Option<String>,
    /// Source attribute (model name/version)
    pub source: Option<String>,
    /// History attribute (processing steps)
    pub history: Option<String>,
    /// References attribute
    pub references: Option<String>,
    /// Comment attribute
    pub comment: Option<String>,
    /// Whether to include velocity components (u, v)
    pub include_velocity: bool,
    /// Whether to include tracers (temperature, salinity)
    pub include_tracers: bool,
    /// Compression level (0-9, 0=none)
    pub compression_level: u8,
}

impl NetCDFWriterConfig {
    /// Create a new configuration with the given output path.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            title: None,
            institution: None,
            source: Some("dg-rs".to_string()),
            history: None,
            references: None,
            comment: None,
            include_velocity: true,
            include_tracers: false,
            compression_level: 4,
        }
    }

    /// Set the title attribute.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the institution attribute.
    pub fn with_institution(mut self, institution: impl Into<String>) -> Self {
        self.institution = Some(institution.into());
        self
    }

    /// Set the source attribute.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set the comment attribute.
    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }

    /// Enable/disable velocity output.
    pub fn with_velocity(mut self, include: bool) -> Self {
        self.include_velocity = include;
        self
    }

    /// Enable/disable tracer output.
    pub fn with_tracers(mut self, include: bool) -> Self {
        self.include_tracers = include;
        self
    }

    /// Set compression level (0-9).
    pub fn with_compression(mut self, level: u8) -> Self {
        self.compression_level = level.min(9);
        self
    }
}

/// Mesh information for NetCDF output.
#[derive(Debug, Clone)]
pub struct NetCDFMeshInfo {
    /// Number of nodes
    pub n_nodes: usize,
    /// Node x-coordinates
    pub x: Vec<f64>,
    /// Node y-coordinates
    pub y: Vec<f64>,
    /// Node latitudes (if available)
    pub lat: Option<Vec<f64>>,
    /// Node longitudes (if available)
    pub lon: Option<Vec<f64>>,
}

impl NetCDFMeshInfo {
    /// Create mesh info from coordinates.
    pub fn from_xy(x: Vec<f64>, y: Vec<f64>) -> Self {
        let n_nodes = x.len();
        Self {
            n_nodes,
            x,
            y,
            lat: None,
            lon: None,
        }
    }

    /// Create mesh info from lat/lon coordinates.
    pub fn from_latlon(lat: Vec<f64>, lon: Vec<f64>) -> Self {
        let n_nodes = lat.len();
        Self {
            n_nodes,
            x: lon.clone(), // Use lon as x
            y: lat.clone(), // Use lat as y
            lat: Some(lat),
            lon: Some(lon),
        }
    }

    /// Add lat/lon coordinates to existing mesh info.
    pub fn with_latlon(mut self, lat: Vec<f64>, lon: Vec<f64>) -> Self {
        self.lat = Some(lat);
        self.lon = Some(lon);
        self
    }
}

/// NetCDF writer for simulation output.
#[cfg(feature = "netcdf")]
pub struct NetCDFWriter {
    file: netcdf::FileMut,
    config: NetCDFWriterConfig,
    n_nodes: usize,
    time_index: usize,
}

#[cfg(feature = "netcdf")]
impl NetCDFWriter {
    /// Create a new NetCDF file for writing.
    pub fn create(config: NetCDFWriterConfig, mesh: &NetCDFMeshInfo) -> Result<Self, NetCDFError> {
        let mut file = create(&config.path)?;

        // Add dimensions
        file.add_unlimited_dimension("time")?;
        file.add_dimension("node", mesh.n_nodes)?;

        // Add coordinate variables
        {
            let mut time_var = file.add_variable::<f64>("time", &["time"])?;
            time_var.put_attribute("standard_name", "time")?;
            time_var.put_attribute("long_name", "simulation time")?;
            time_var.put_attribute("units", "seconds since 1970-01-01 00:00:00")?;
            time_var.put_attribute("calendar", "standard")?;
        }

        // Add x coordinate
        {
            let mut x_var = file.add_variable::<f64>("x", &["node"])?;
            x_var.put_attribute("standard_name", "projection_x_coordinate")?;
            x_var.put_attribute("long_name", "x coordinate")?;
            x_var.put_attribute("units", "m")?;
            x_var.put_values(&mesh.x, ..)?;
        }

        // Add y coordinate
        {
            let mut y_var = file.add_variable::<f64>("y", &["node"])?;
            y_var.put_attribute("standard_name", "projection_y_coordinate")?;
            y_var.put_attribute("long_name", "y coordinate")?;
            y_var.put_attribute("units", "m")?;
            y_var.put_values(&mesh.y, ..)?;
        }

        // Add lat/lon if available
        if let Some(ref lat) = mesh.lat {
            let mut lat_var = file.add_variable::<f64>("lat", &["node"])?;
            lat_var.put_attribute("standard_name", "latitude")?;
            lat_var.put_attribute("long_name", "latitude")?;
            lat_var.put_attribute("units", "degrees_north")?;
            lat_var.put_values(lat, ..)?;
        }

        if let Some(ref lon) = mesh.lon {
            let mut lon_var = file.add_variable::<f64>("lon", &["node"])?;
            lon_var.put_attribute("standard_name", "longitude")?;
            lon_var.put_attribute("long_name", "longitude")?;
            lon_var.put_attribute("units", "degrees_east")?;
            lon_var.put_values(lon, ..)?;
        }

        // Add data variables
        {
            let mut eta_var = file.add_variable::<f32>("eta", &["time", "node"])?;
            eta_var.put_attribute("standard_name", "sea_surface_height_above_geoid")?;
            eta_var.put_attribute("long_name", "sea surface elevation")?;
            eta_var.put_attribute("units", "m")?;
            eta_var.put_attribute("_FillValue", FILL_VALUE_F32)?;
        }

        {
            let mut h_var = file.add_variable::<f32>("h", &["time", "node"])?;
            h_var.put_attribute("standard_name", "sea_floor_depth_below_sea_surface")?;
            h_var.put_attribute("long_name", "water depth")?;
            h_var.put_attribute("units", "m")?;
            h_var.put_attribute("_FillValue", FILL_VALUE_F32)?;
        }

        if config.include_velocity {
            {
                let mut u_var = file.add_variable::<f32>("u", &["time", "node"])?;
                u_var.put_attribute("standard_name", "eastward_sea_water_velocity")?;
                u_var.put_attribute("long_name", "eastward velocity")?;
                u_var.put_attribute("units", "m s-1")?;
                u_var.put_attribute("_FillValue", FILL_VALUE_F32)?;
            }

            {
                let mut v_var = file.add_variable::<f32>("v", &["time", "node"])?;
                v_var.put_attribute("standard_name", "northward_sea_water_velocity")?;
                v_var.put_attribute("long_name", "northward velocity")?;
                v_var.put_attribute("units", "m s-1")?;
                v_var.put_attribute("_FillValue", FILL_VALUE_F32)?;
            }

            {
                let mut speed_var = file.add_variable::<f32>("speed", &["time", "node"])?;
                speed_var.put_attribute("standard_name", "sea_water_speed")?;
                speed_var.put_attribute("long_name", "current speed")?;
                speed_var.put_attribute("units", "m s-1")?;
                speed_var.put_attribute("_FillValue", FILL_VALUE_F32)?;
            }
        }

        if config.include_tracers {
            {
                let mut temp_var = file.add_variable::<f32>("temperature", &["time", "node"])?;
                temp_var.put_attribute("standard_name", "sea_water_temperature")?;
                temp_var.put_attribute("long_name", "sea water temperature")?;
                temp_var.put_attribute("units", "degC")?;
                temp_var.put_attribute("_FillValue", FILL_VALUE_F32)?;
            }

            {
                let mut sal_var = file.add_variable::<f32>("salinity", &["time", "node"])?;
                sal_var.put_attribute("standard_name", "sea_water_salinity")?;
                sal_var.put_attribute("long_name", "sea water salinity")?;
                sal_var.put_attribute("units", "1e-3")?; // PSU = g/kg = 1e-3
                sal_var.put_attribute("_FillValue", FILL_VALUE_F32)?;
            }
        }

        // Add global attributes
        file.add_attribute("Conventions", "CF-1.8")?;
        file.add_attribute("featureType", "point")?;

        if let Some(ref title) = config.title {
            file.add_attribute("title", title.as_str())?;
        }
        if let Some(ref institution) = config.institution {
            file.add_attribute("institution", institution.as_str())?;
        }
        if let Some(ref source) = config.source {
            file.add_attribute("source", source.as_str())?;
        }
        if let Some(ref comment) = config.comment {
            file.add_attribute("comment", comment.as_str())?;
        }

        // Add creation timestamp
        let now = Utc::now();
        file.add_attribute("history", format!("{}: Created by dg-rs", now.format("%Y-%m-%d %H:%M:%S UTC")).as_str())?;

        Ok(Self {
            file,
            config,
            n_nodes: mesh.n_nodes,
            time_index: 0,
        })
    }

    /// Write a timestep to the file.
    ///
    /// # Arguments
    /// * `time` - Simulation time in seconds
    /// * `h` - Water depth at each node
    /// * `eta` - Surface elevation at each node
    /// * `u` - Eastward velocity (optional)
    /// * `v` - Northward velocity (optional)
    pub fn write_timestep(
        &mut self,
        time: f64,
        h: &[f64],
        eta: &[f64],
        u: Option<&[f64]>,
        v: Option<&[f64]>,
    ) -> Result<(), NetCDFError> {
        let t_idx = self.time_index;

        // Write time
        {
            let mut time_var = self.file.variable_mut("time")
                .ok_or_else(|| NetCDFError::MissingVariable("time".to_string()))?;
            time_var.put_value(time, [t_idx])?;
        }

        // Write h
        {
            let h_f32: Vec<f32> = h.iter().map(|&x| x as f32).collect();
            let mut h_var = self.file.variable_mut("h")
                .ok_or_else(|| NetCDFError::MissingVariable("h".to_string()))?;
            h_var.put_values(&h_f32, (t_idx, ..))?;
        }

        // Write eta
        {
            let eta_f32: Vec<f32> = eta.iter().map(|&x| x as f32).collect();
            let mut eta_var = self.file.variable_mut("eta")
                .ok_or_else(|| NetCDFError::MissingVariable("eta".to_string()))?;
            eta_var.put_values(&eta_f32, (t_idx, ..))?;
        }

        // Write velocities if provided
        if self.config.include_velocity {
            if let (Some(u_data), Some(v_data)) = (u, v) {
                let u_f32: Vec<f32> = u_data.iter().map(|&x| x as f32).collect();
                let v_f32: Vec<f32> = v_data.iter().map(|&x| x as f32).collect();
                let speed_f32: Vec<f32> = u_data.iter().zip(v_data.iter())
                    .map(|(&u, &v)| (u * u + v * v).sqrt() as f32)
                    .collect();

                {
                    let mut u_var = self.file.variable_mut("u")
                        .ok_or_else(|| NetCDFError::MissingVariable("u".to_string()))?;
                    u_var.put_values(&u_f32, (t_idx, ..))?;
                }

                {
                    let mut v_var = self.file.variable_mut("v")
                        .ok_or_else(|| NetCDFError::MissingVariable("v".to_string()))?;
                    v_var.put_values(&v_f32, (t_idx, ..))?;
                }

                {
                    let mut speed_var = self.file.variable_mut("speed")
                        .ok_or_else(|| NetCDFError::MissingVariable("speed".to_string()))?;
                    speed_var.put_values(&speed_f32, (t_idx, ..))?;
                }
            }
        }

        self.time_index += 1;
        Ok(())
    }

    /// Write tracer data for a timestep.
    pub fn write_tracers(
        &mut self,
        temperature: Option<&[f64]>,
        salinity: Option<&[f64]>,
    ) -> Result<(), NetCDFError> {
        if !self.config.include_tracers {
            return Ok(());
        }

        let t_idx = self.time_index.saturating_sub(1); // Use previous time index

        if let Some(temp) = temperature {
            let temp_f32: Vec<f32> = temp.iter().map(|&x| x as f32).collect();
            let mut temp_var = self.file.variable_mut("temperature")
                .ok_or_else(|| NetCDFError::MissingVariable("temperature".to_string()))?;
            temp_var.put_values(&temp_f32, (t_idx, ..))?;
        }

        if let Some(sal) = salinity {
            let sal_f32: Vec<f32> = sal.iter().map(|&x| x as f32).collect();
            let mut sal_var = self.file.variable_mut("salinity")
                .ok_or_else(|| NetCDFError::MissingVariable("salinity".to_string()))?;
            sal_var.put_values(&sal_f32, (t_idx, ..))?;
        }

        Ok(())
    }

    /// Get the number of timesteps written.
    pub fn n_timesteps(&self) -> usize {
        self.time_index
    }
}

// ============================================================================
// NetCDF Reader for Forcing Data
// ============================================================================

/// Forcing data point (wind, pressure) at a specific location and time.
#[derive(Debug, Clone, Copy)]
pub struct ForcingDataPoint {
    /// Eastward wind component (m/s) at 10m height
    pub u10: f64,
    /// Northward wind component (m/s) at 10m height
    pub v10: f64,
    /// Mean sea level pressure (Pa)
    pub msl: f64,
}

impl ForcingDataPoint {
    /// Wind speed magnitude.
    pub fn wind_speed(&self) -> f64 {
        (self.u10 * self.u10 + self.v10 * self.v10).sqrt()
    }

    /// Wind direction (meteorological convention: direction wind is FROM, degrees).
    pub fn wind_direction(&self) -> f64 {
        let dir = 270.0 - self.v10.atan2(self.u10).to_degrees();
        if dir < 0.0 { dir + 360.0 } else if dir >= 360.0 { dir - 360.0 } else { dir }
    }
}

/// Reader for atmospheric forcing data (ERA5, ECMWF format).
#[cfg(feature = "netcdf")]
pub struct ForcingReader {
    /// Longitudes (1D)
    pub lon: Vec<f64>,
    /// Latitudes (1D)
    pub lat: Vec<f64>,
    /// Times (hours since epoch)
    pub time: Vec<f64>,
    /// U10 wind component [time][lat][lon]
    pub u10: Vec<Vec<Vec<f32>>>,
    /// V10 wind component [time][lat][lon]
    pub v10: Vec<Vec<Vec<f32>>>,
    /// Mean sea level pressure [time][lat][lon]
    pub msl: Vec<Vec<Vec<f32>>>,
}

#[cfg(feature = "netcdf")]
impl ForcingReader {
    /// Load forcing data from a NetCDF file.
    ///
    /// Expected variables:
    /// - `u10` or `10u`: 10m eastward wind (m/s)
    /// - `v10` or `10v`: 10m northward wind (m/s)
    /// - `msl` or `sp`: Mean sea level pressure (Pa)
    ///
    /// Coordinates:
    /// - `longitude` or `lon`: 1D longitude array
    /// - `latitude` or `lat`: 1D latitude array
    /// - `time`: Time coordinate (hours since reference)
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, NetCDFError> {
        let file = netcdf::open(path)?;

        // Read coordinates
        let lon = Self::read_coord(&file, &["longitude", "lon"])?;
        let lat = Self::read_coord(&file, &["latitude", "lat"])?;
        let time = Self::read_coord(&file, &["time"])?;

        let n_time = time.len();
        let n_lat = lat.len();
        let n_lon = lon.len();

        // Read u10
        let u10 = Self::read_3d_var(&file, &["u10", "10u"], n_time, n_lat, n_lon)?;

        // Read v10
        let v10 = Self::read_3d_var(&file, &["v10", "10v"], n_time, n_lat, n_lon)?;

        // Read msl (may need unit conversion from hPa to Pa)
        let mut msl = Self::read_3d_var(&file, &["msl", "sp", "prmsl"], n_time, n_lat, n_lon)?;

        // Check if pressure is in hPa and convert to Pa
        // ERA5 uses Pa, but some sources use hPa
        let max_pressure = msl.iter()
            .flat_map(|t| t.iter().flat_map(|r| r.iter()))
            .filter(|&&v| is_valid_f32(v))
            .fold(0.0f32, |a, &b| a.max(b));

        if max_pressure < 2000.0 {
            // Likely hPa, convert to Pa
            for time_slice in &mut msl {
                for lat_row in time_slice {
                    for val in lat_row {
                        if is_valid_f32(*val) {
                            *val *= 100.0;
                        }
                    }
                }
            }
        }

        Ok(Self { lon, lat, time, u10, v10, msl })
    }

    /// Read a coordinate variable.
    fn read_coord(file: &netcdf::File, names: &[&str]) -> Result<Vec<f64>, NetCDFError> {
        for name in names {
            if let Some(var) = file.variable(name) {
                let data: Vec<f64> = var.get_values(..)?;
                return Ok(data);
            }
        }
        Err(NetCDFError::MissingVariable(names.join(" or ")))
    }

    /// Read a 3D variable [time][lat][lon].
    fn read_3d_var(
        file: &netcdf::File,
        names: &[&str],
        n_time: usize,
        n_lat: usize,
        n_lon: usize,
    ) -> Result<Vec<Vec<Vec<f32>>>, NetCDFError> {
        for name in names {
            if let Some(var) = file.variable(name) {
                // Try to read scale_factor and add_offset
                let scale = var.attribute_value("scale_factor")
                    .and_then(|r| r.ok())
                    .and_then(|v| match v {
                        netcdf::AttributeValue::Double(d) => Some(d),
                        netcdf::AttributeValue::Float(f) => Some(f as f64),
                        _ => None,
                    })
                    .unwrap_or(1.0);

                let offset = var.attribute_value("add_offset")
                    .and_then(|r| r.ok())
                    .and_then(|v| match v {
                        netcdf::AttributeValue::Double(d) => Some(d),
                        netcdf::AttributeValue::Float(f) => Some(f as f64),
                        _ => None,
                    })
                    .unwrap_or(0.0);

                // Read raw data
                let raw: Vec<f32> = var.get_values(..)?;

                // Apply scaling and reshape
                let mut result = vec![vec![vec![0.0f32; n_lon]; n_lat]; n_time];
                for t in 0..n_time {
                    for j in 0..n_lat {
                        for i in 0..n_lon {
                            let idx = t * n_lat * n_lon + j * n_lon + i;
                            let val = raw[idx];
                            result[t][j][i] = if is_valid_f32(val) {
                                (val as f64 * scale + offset) as f32
                            } else {
                                FILL_VALUE_F32
                            };
                        }
                    }
                }

                return Ok(result);
            }
        }
        Err(NetCDFError::MissingVariable(names.join(" or ")))
    }

    /// Get forcing data at a specific location using bilinear interpolation.
    pub fn get_forcing(&self, lon: f64, lat: f64, time_idx: usize) -> Option<ForcingDataPoint> {
        if time_idx >= self.time.len() {
            return None;
        }

        // Find grid cell
        let (i0, i1, fx) = self.find_index(&self.lon, lon)?;
        let (j0, j1, fy) = self.find_index(&self.lat, lat)?;

        // Bilinear interpolation
        let u10 = self.interpolate_2d(&self.u10[time_idx], i0, i1, j0, j1, fx, fy)?;
        let v10 = self.interpolate_2d(&self.v10[time_idx], i0, i1, j0, j1, fx, fy)?;
        let msl = self.interpolate_2d(&self.msl[time_idx], i0, i1, j0, j1, fx, fy)?;

        Some(ForcingDataPoint {
            u10: u10 as f64,
            v10: v10 as f64,
            msl: msl as f64,
        })
    }

    /// Find index and interpolation factor for a coordinate.
    fn find_index(&self, coords: &[f64], value: f64) -> Option<(usize, usize, f64)> {
        if coords.len() < 2 {
            return None;
        }

        // Handle wrap-around for longitude
        let n = coords.len();

        for i in 0..n - 1 {
            let c0 = coords[i];
            let c1 = coords[i + 1];
            if (c0 <= value && value <= c1) || (c1 <= value && value <= c0) {
                let f = (value - c0) / (c1 - c0);
                return Some((i, i + 1, f));
            }
        }

        // Edge cases
        if value <= coords[0].min(coords[n - 1]) {
            Some((0, 0, 0.0))
        } else if value >= coords[0].max(coords[n - 1]) {
            Some((n - 1, n - 1, 0.0))
        } else {
            None
        }
    }

    /// Bilinear interpolation on a 2D grid.
    fn interpolate_2d(
        &self,
        data: &[Vec<f32>],
        i0: usize, i1: usize,
        j0: usize, j1: usize,
        fx: f64, fy: f64,
    ) -> Option<f32> {
        let v00 = data[j0][i0];
        let v01 = data[j0][i1];
        let v10 = data[j1][i0];
        let v11 = data[j1][i1];

        // Check for fill values
        if !is_valid_f32(v00) || !is_valid_f32(v01) || !is_valid_f32(v10) || !is_valid_f32(v11) {
            // Return nearest valid value
            let vals = [v00, v01, v10, v11];
            return vals.iter().find(|&&v| is_valid_f32(v)).copied();
        }

        let fx = fx as f32;
        let fy = fy as f32;

        let v0 = v00 * (1.0 - fx) + v01 * fx;
        let v1 = v10 * (1.0 - fx) + v11 * fx;
        let v = v0 * (1.0 - fy) + v1 * fy;

        Some(v)
    }

    /// Get the number of time steps.
    pub fn n_times(&self) -> usize {
        self.time.len()
    }

    /// Get data coverage summary.
    pub fn coverage_summary(&self) -> String {
        let lon_range = (self.lon.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                        self.lon.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        let lat_range = (self.lat.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                        self.lat.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

        format!(
            "Forcing data: {} times, lon [{:.2}, {:.2}], lat [{:.2}, {:.2}]",
            self.time.len(),
            lon_range.0, lon_range.1,
            lat_range.0, lat_range.1
        )
    }
}

// ============================================================================
// Ocean Model Reader (NorKyst, ROMS)
// ============================================================================

/// Ocean state at a single point (from parent model for nesting).
#[derive(Debug, Clone, Copy, Default)]
pub struct OceanState {
    /// Sea surface height (m)
    pub ssh: f64,
    /// Eastward velocity (m/s)
    pub u: f64,
    /// Northward velocity (m/s)
    pub v: f64,
    /// Temperature (°C), if available
    pub temperature: Option<f64>,
    /// Salinity (PSU), if available
    pub salinity: Option<f64>,
}

impl OceanState {
    /// Current speed magnitude.
    pub fn speed(&self) -> f64 {
        (self.u * self.u + self.v * self.v).sqrt()
    }

    /// Current direction (oceanographic: direction flow is TOWARD, degrees).
    pub fn direction(&self) -> f64 {
        let dir = 90.0 - self.v.atan2(self.u).to_degrees();
        if dir < 0.0 { dir + 360.0 } else if dir >= 360.0 { dir - 360.0 } else { dir }
    }
}

/// Ocean model grid type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OceanGridType {
    /// Regular lat/lon grid (1D coordinates)
    Regular,
    /// Curvilinear grid (2D lat/lon arrays) - NorKyst, ROMS
    Curvilinear,
}

/// Reader for ocean model output (NorKyst v3, ROMS format).
///
/// Supports:
/// - Curvilinear grids with 2D lat/lon
/// - Packed data (i16 with scale_factor/add_offset)
/// - Multiple time steps
/// - SSH, currents, temperature, salinity
///
/// # Example
///
/// ```rust,ignore
/// use dg_rs::io::OceanModelReader;
///
/// let reader = OceanModelReader::from_file("norkyst.nc")?;
/// println!("{}", reader.summary());
///
/// // Get state at a specific location and time
/// if let Some(state) = reader.get_state(8.5, 63.8, 0) {
///     println!("SSH: {:.3} m, Speed: {:.2} m/s", state.ssh, state.speed());
/// }
/// ```
#[cfg(feature = "netcdf")]
pub struct OceanModelReader {
    /// Grid type
    pub grid_type: OceanGridType,
    /// Latitude array [n_y][n_x] or [n_lat] for regular
    pub lat: Vec<Vec<f64>>,
    /// Longitude array [n_y][n_x] or [n_lon] for regular
    pub lon: Vec<Vec<f64>>,
    /// Time values (seconds since reference or hours)
    pub time: Vec<f64>,
    /// Sea surface height [time][y][x]
    pub ssh: Option<Vec<Vec<Vec<f32>>>>,
    /// Eastward velocity [time][y][x]
    pub u: Option<Vec<Vec<Vec<f32>>>>,
    /// Northward velocity [time][y][x]
    pub v: Option<Vec<Vec<Vec<f32>>>>,
    /// Temperature [time][y][x]
    pub temperature: Option<Vec<Vec<Vec<f32>>>>,
    /// Salinity [time][y][x]
    pub salinity: Option<Vec<Vec<Vec<f32>>>>,
    /// Grid dimensions (n_y, n_x)
    pub dims: (usize, usize),
    /// Bounding box (min_lon, min_lat, max_lon, max_lat)
    pub bbox: (f64, f64, f64, f64),
}

#[cfg(feature = "netcdf")]
impl OceanModelReader {
    /// Load ocean model data from a NetCDF file.
    ///
    /// Automatically detects:
    /// - NorKyst v3 format (u_eastward, v_northward, zeta)
    /// - ROMS format (u, v, zeta)
    /// - Generic format (eastward_sea_water_velocity, etc.)
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, NetCDFError> {
        let file = netcdf::open(path)?;

        // Detect lat/lon variable names and grid type
        let (lat_name, lon_name, grid_type) = Self::detect_grid_vars(&file)?;

        // Read coordinates
        let lat_var = file.variable(lat_name)
            .ok_or_else(|| NetCDFError::MissingVariable(lat_name.to_string()))?;
        let lon_var = file.variable(lon_name)
            .ok_or_else(|| NetCDFError::MissingVariable(lon_name.to_string()))?;

        let lat_dims = lat_var.dimensions();
        let (n_y, n_x, lat, lon) = if lat_dims.len() == 2 {
            // Curvilinear grid
            let n_y = lat_dims[0].len();
            let n_x = lat_dims[1].len();
            let lat_flat: Vec<f64> = lat_var.get_values(..)?;
            let lon_flat: Vec<f64> = lon_var.get_values(..)?;
            let lat = reshape_2d(&lat_flat, n_y, n_x);
            let lon = reshape_2d(&lon_flat, n_y, n_x);
            (n_y, n_x, lat, lon)
        } else {
            // Regular grid - create meshgrid
            let lat_1d: Vec<f64> = lat_var.get_values(..)?;
            let lon_1d: Vec<f64> = lon_var.get_values(..)?;
            let n_y = lat_1d.len();
            let n_x = lon_1d.len();
            let mut lat = vec![vec![0.0; n_x]; n_y];
            let mut lon = vec![vec![0.0; n_x]; n_y];
            for j in 0..n_y {
                for i in 0..n_x {
                    lat[j][i] = lat_1d[j];
                    lon[j][i] = lon_1d[i];
                }
            }
            (n_y, n_x, lat, lon)
        };

        // Calculate bounding box
        let bbox = Self::compute_bbox(&lat, &lon);

        // Read time
        let time = Self::read_time(&file)?;
        let n_time = time.len();

        // Read variables with automatic name detection
        let ssh = Self::read_variable(&file, &["zeta", "ssh", "sea_surface_height", "h"], n_time, n_y, n_x);
        let u = Self::read_variable(&file, &["u_eastward", "u", "eastward_sea_water_velocity"], n_time, n_y, n_x);
        let v = Self::read_variable(&file, &["v_northward", "v", "northward_sea_water_velocity"], n_time, n_y, n_x);
        let temperature = Self::read_variable(&file, &["temperature", "temp", "sea_water_temperature"], n_time, n_y, n_x);
        let salinity = Self::read_variable(&file, &["salinity", "salt", "sea_water_salinity"], n_time, n_y, n_x);

        Ok(Self {
            grid_type,
            lat,
            lon,
            time,
            ssh,
            u,
            v,
            temperature,
            salinity,
            dims: (n_y, n_x),
            bbox,
        })
    }

    /// Detect grid variable names and type.
    fn detect_grid_vars(file: &netcdf::File) -> Result<(&'static str, &'static str, OceanGridType), NetCDFError> {
        // Try common lat/lon names
        let lat_names = ["lat", "latitude", "nav_lat", "lat_rho"];
        let lon_names = ["lon", "longitude", "nav_lon", "lon_rho"];

        for &lat_name in &lat_names {
            if let Some(lat_var) = file.variable(lat_name) {
                let is_2d = lat_var.dimensions().len() == 2;
                for &lon_name in &lon_names {
                    if file.variable(lon_name).is_some() {
                        let grid_type = if is_2d { OceanGridType::Curvilinear } else { OceanGridType::Regular };
                        return Ok((lat_name, lon_name, grid_type));
                    }
                }
            }
        }

        Err(NetCDFError::MissingVariable("lat/lon coordinates".to_string()))
    }

    /// Read time coordinate.
    fn read_time(file: &netcdf::File) -> Result<Vec<f64>, NetCDFError> {
        let time_names = ["time", "ocean_time", "Time"];
        for name in time_names {
            if let Some(var) = file.variable(name) {
                let data: Vec<f64> = var.get_values(..)?;
                return Ok(data);
            }
        }
        // No time dimension - return single time step
        Ok(vec![0.0])
    }

    /// Read a variable with automatic packed data handling.
    fn read_variable(
        file: &netcdf::File,
        names: &[&str],
        n_time: usize,
        n_y: usize,
        n_x: usize,
    ) -> Option<Vec<Vec<Vec<f32>>>> {
        for name in names {
            if let Some(var) = file.variable(name) {
                // Get scale_factor and add_offset
                let scale = Self::get_attr_f64(&var, "scale_factor").unwrap_or(1.0);
                let offset = Self::get_attr_f64(&var, "add_offset").unwrap_or(0.0);
                let fill_i16 = Self::get_attr_i16(&var, "_FillValue").unwrap_or(i16::MAX);
                let fill_f32 = Self::get_attr_f32(&var, "_FillValue").unwrap_or(FILL_VALUE_F32);

                let dims = var.dimensions();
                let _total_size: usize = dims.iter().map(|d| d.len()).product();

                // Try reading as i16 (packed) first, then f32
                let flat: Vec<f32> = if let Ok(raw) = var.get_values::<i16, _>(..) {
                    raw.iter()
                        .map(|&v| {
                            if v == fill_i16 {
                                f32::NAN
                            } else {
                                (v as f64 * scale + offset) as f32
                            }
                        })
                        .collect()
                } else if let Ok(raw) = var.get_values::<f32, _>(..) {
                    raw.iter()
                        .map(|&v| {
                            if !v.is_finite() || v == fill_f32 || v.abs() > 1e30 {
                                f32::NAN
                            } else {
                                (v as f64 * scale + offset) as f32
                            }
                        })
                        .collect()
                } else {
                    continue;
                };

                // Reshape to [time][y][x], taking surface layer if 4D
                return Some(Self::reshape_to_3d(&flat, &dims, n_time, n_y, n_x));
            }
        }
        None
    }

    /// Get f64 attribute value.
    fn get_attr_f64(var: &netcdf::Variable, name: &str) -> Option<f64> {
        var.attribute_value(name)
            .and_then(|r| r.ok())
            .and_then(|v| match v {
                netcdf::AttributeValue::Double(d) => Some(d),
                netcdf::AttributeValue::Float(f) => Some(f as f64),
                _ => None,
            })
    }

    /// Get i16 attribute value.
    fn get_attr_i16(var: &netcdf::Variable, name: &str) -> Option<i16> {
        var.attribute_value(name)
            .and_then(|r| r.ok())
            .and_then(|v| match v {
                netcdf::AttributeValue::Short(s) => Some(s),
                netcdf::AttributeValue::Int(i) => Some(i as i16),
                _ => None,
            })
    }

    /// Get f32 attribute value.
    fn get_attr_f32(var: &netcdf::Variable, name: &str) -> Option<f32> {
        var.attribute_value(name)
            .and_then(|r| r.ok())
            .and_then(|v| match v {
                netcdf::AttributeValue::Float(f) => Some(f),
                netcdf::AttributeValue::Double(d) => Some(d as f32),
                _ => None,
            })
    }

    /// Reshape flat array to 3D [time][y][x], extracting surface from 4D if needed.
    fn reshape_to_3d(
        flat: &[f32],
        dims: &[netcdf::Dimension],
        n_time: usize,
        n_y: usize,
        n_x: usize,
    ) -> Vec<Vec<Vec<f32>>> {
        let mut result = vec![vec![vec![f32::NAN; n_x]; n_y]; n_time];

        match dims.len() {
            2 => {
                // [y][x] - single time step
                for j in 0..n_y {
                    for i in 0..n_x {
                        let idx = j * n_x + i;
                        if idx < flat.len() {
                            result[0][j][i] = flat[idx];
                        }
                    }
                }
            }
            3 => {
                // [time][y][x]
                for t in 0..n_time {
                    for j in 0..n_y {
                        for i in 0..n_x {
                            let idx = t * n_y * n_x + j * n_x + i;
                            if idx < flat.len() {
                                result[t][j][i] = flat[idx];
                            }
                        }
                    }
                }
            }
            4 => {
                // [time][depth][y][x] - take surface (depth=0)
                let n_depth = dims[1].len();
                for t in 0..n_time {
                    for j in 0..n_y {
                        for i in 0..n_x {
                            // depth=0 is surface
                            let idx = t * n_depth * n_y * n_x + 0 * n_y * n_x + j * n_x + i;
                            if idx < flat.len() {
                                result[t][j][i] = flat[idx];
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        result
    }

    /// Compute bounding box from lat/lon arrays.
    fn compute_bbox(lat: &[Vec<f64>], lon: &[Vec<f64>]) -> (f64, f64, f64, f64) {
        let mut min_lon = f64::INFINITY;
        let mut max_lon = f64::NEG_INFINITY;
        let mut min_lat = f64::INFINITY;
        let mut max_lat = f64::NEG_INFINITY;

        for row in lat {
            for &v in row {
                if v.is_finite() {
                    min_lat = min_lat.min(v);
                    max_lat = max_lat.max(v);
                }
            }
        }
        for row in lon {
            for &v in row {
                if v.is_finite() {
                    min_lon = min_lon.min(v);
                    max_lon = max_lon.max(v);
                }
            }
        }

        (min_lon, min_lat, max_lon, max_lat)
    }

    /// Get ocean state at a specific location using bilinear interpolation.
    pub fn get_state(&self, lon: f64, lat: f64, time_idx: usize) -> Option<OceanState> {
        if time_idx >= self.time.len() {
            return None;
        }

        // Find grid cell containing the point
        let (j0, i0, fy, fx) = self.find_cell(lon, lat)?;

        let j1 = (j0 + 1).min(self.dims.0 - 1);
        let i1 = (i0 + 1).min(self.dims.1 - 1);

        // Bilinear interpolation for each variable
        let ssh = self.ssh.as_ref()
            .and_then(|d| Self::interp_2d(&d[time_idx], j0, j1, i0, i1, fy, fx))
            .unwrap_or(0.0) as f64;

        let u = self.u.as_ref()
            .and_then(|d| Self::interp_2d(&d[time_idx], j0, j1, i0, i1, fy, fx))
            .unwrap_or(0.0) as f64;

        let v = self.v.as_ref()
            .and_then(|d| Self::interp_2d(&d[time_idx], j0, j1, i0, i1, fy, fx))
            .unwrap_or(0.0) as f64;

        let temperature = self.temperature.as_ref()
            .and_then(|d| Self::interp_2d(&d[time_idx], j0, j1, i0, i1, fy, fx))
            .map(|v| v as f64);

        let salinity = self.salinity.as_ref()
            .and_then(|d| Self::interp_2d(&d[time_idx], j0, j1, i0, i1, fy, fx))
            .map(|v| v as f64);

        Some(OceanState { ssh, u, v, temperature, salinity })
    }

    /// Get state with time interpolation.
    pub fn get_state_interpolated(&self, lon: f64, lat: f64, time: f64) -> Option<OceanState> {
        if self.time.is_empty() {
            return None;
        }

        // Find time indices
        let (t0, t1, ft) = self.find_time_index(time)?;

        let state0 = self.get_state(lon, lat, t0)?;
        if t0 == t1 || ft < 1e-10 {
            return Some(state0);
        }

        let state1 = self.get_state(lon, lat, t1)?;

        // Linear interpolation in time
        let ft = ft as f64;
        Some(OceanState {
            ssh: state0.ssh * (1.0 - ft) + state1.ssh * ft,
            u: state0.u * (1.0 - ft) + state1.u * ft,
            v: state0.v * (1.0 - ft) + state1.v * ft,
            temperature: match (state0.temperature, state1.temperature) {
                (Some(t0), Some(t1)) => Some(t0 * (1.0 - ft) + t1 * ft),
                (Some(t), None) | (None, Some(t)) => Some(t),
                _ => None,
            },
            salinity: match (state0.salinity, state1.salinity) {
                (Some(s0), Some(s1)) => Some(s0 * (1.0 - ft) + s1 * ft),
                (Some(s), None) | (None, Some(s)) => Some(s),
                _ => None,
            },
        })
    }

    /// Find grid cell containing a point (for curvilinear grids).
    fn find_cell(&self, target_lon: f64, target_lat: f64) -> Option<(usize, usize, f64, f64)> {
        let (n_y, n_x) = self.dims;

        // For regular grids, use fast index lookup
        if self.grid_type == OceanGridType::Regular && n_y > 0 && n_x > 0 {
            let lat_1d: Vec<f64> = (0..n_y).map(|j| self.lat[j][0]).collect();
            let lon_1d: Vec<f64> = (0..n_x).map(|i| self.lon[0][i]).collect();

            let (j0, _j1, fy) = find_bracket(&lat_1d, target_lat)?;
            let (i0, _i1, fx) = find_bracket(&lon_1d, target_lon)?;

            return Some((j0, i0, fy, fx));
        }

        // For curvilinear grids, search all cells
        let mut best_dist = f64::INFINITY;
        let mut best_cell = None;

        for j in 0..n_y.saturating_sub(1) {
            for i in 0..n_x.saturating_sub(1) {
                // Check if point is in this cell
                let lon00 = self.lon[j][i];
                let lon01 = self.lon[j][i + 1];
                let lon10 = self.lon[j + 1][i];
                let lon11 = self.lon[j + 1][i + 1];

                let lat00 = self.lat[j][i];
                let lat01 = self.lat[j][i + 1];
                let lat10 = self.lat[j + 1][i];
                let lat11 = self.lat[j + 1][i + 1];

                // Quick bounding box check
                let min_lon = lon00.min(lon01).min(lon10).min(lon11);
                let max_lon = lon00.max(lon01).max(lon10).max(lon11);
                let min_lat = lat00.min(lat01).min(lat10).min(lat11);
                let max_lat = lat00.max(lat01).max(lat10).max(lat11);

                if target_lon < min_lon - 0.1 || target_lon > max_lon + 0.1 ||
                   target_lat < min_lat - 0.1 || target_lat > max_lat + 0.1 {
                    continue;
                }

                // Compute bilinear coordinates
                let center_lon = (lon00 + lon01 + lon10 + lon11) / 4.0;
                let center_lat = (lat00 + lat01 + lat10 + lat11) / 4.0;
                let dist = (target_lon - center_lon).powi(2) + (target_lat - center_lat).powi(2);

                if dist < best_dist {
                    best_dist = dist;
                    // Approximate bilinear coordinates
                    let fx = (target_lon - lon00) / (lon01 - lon00).max(1e-10);
                    let fy = (target_lat - lat00) / (lat10 - lat00).max(1e-10);
                    best_cell = Some((j, i, fy.clamp(0.0, 1.0), fx.clamp(0.0, 1.0)));
                }
            }
        }

        best_cell
    }

    /// Find time index and interpolation factor.
    fn find_time_index(&self, time: f64) -> Option<(usize, usize, f64)> {
        find_bracket(&self.time, time)
    }

    /// Bilinear interpolation on 2D grid.
    fn interp_2d(
        data: &[Vec<f32>],
        j0: usize, j1: usize,
        i0: usize, i1: usize,
        fy: f64, fx: f64,
    ) -> Option<f32> {
        let v00 = data[j0][i0];
        let v01 = data[j0][i1];
        let v10 = data[j1][i0];
        let v11 = data[j1][i1];

        // Check for NaN
        if !v00.is_finite() || !v01.is_finite() || !v10.is_finite() || !v11.is_finite() {
            // Return first valid value
            for &v in &[v00, v01, v10, v11] {
                if v.is_finite() {
                    return Some(v);
                }
            }
            return None;
        }

        let fx = fx as f32;
        let fy = fy as f32;

        let v0 = v00 * (1.0 - fx) + v01 * fx;
        let v1 = v10 * (1.0 - fx) + v11 * fx;
        Some(v0 * (1.0 - fy) + v1 * fy)
    }

    /// Number of time steps.
    pub fn n_times(&self) -> usize {
        self.time.len()
    }

    /// Check if a variable is available.
    pub fn has_ssh(&self) -> bool { self.ssh.is_some() }
    pub fn has_currents(&self) -> bool { self.u.is_some() && self.v.is_some() }
    pub fn has_temperature(&self) -> bool { self.temperature.is_some() }
    pub fn has_salinity(&self) -> bool { self.salinity.is_some() }

    /// Get summary of data coverage.
    pub fn summary(&self) -> String {
        let vars: Vec<&str> = [
            self.has_ssh().then_some("SSH"),
            self.has_currents().then_some("currents"),
            self.has_temperature().then_some("temperature"),
            self.has_salinity().then_some("salinity"),
        ].into_iter().flatten().collect();

        format!(
            "Ocean model: {}x{} grid, {} times, lon [{:.2}, {:.2}], lat [{:.2}, {:.2}], vars: {}",
            self.dims.0, self.dims.1,
            self.time.len(),
            self.bbox.0, self.bbox.2,
            self.bbox.1, self.bbox.3,
            vars.join(", ")
        )
    }
}

/// Helper: reshape flat array to 2D.
fn reshape_2d(flat: &[f64], n_y: usize, n_x: usize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; n_x]; n_y];
    for j in 0..n_y {
        for i in 0..n_x {
            let idx = j * n_x + i;
            if idx < flat.len() {
                result[j][i] = flat[idx];
            }
        }
    }
    result
}

/// Helper: find bracket indices and interpolation factor.
fn find_bracket(coords: &[f64], value: f64) -> Option<(usize, usize, f64)> {
    if coords.is_empty() {
        return None;
    }
    if coords.len() == 1 {
        return Some((0, 0, 0.0));
    }

    // Check if ascending or descending
    let ascending = coords[1] > coords[0];

    for i in 0..coords.len() - 1 {
        let (c0, c1) = if ascending {
            (coords[i], coords[i + 1])
        } else {
            (coords[i + 1], coords[i])
        };

        if c0 <= value && value <= c1 {
            let f = (value - c0) / (c1 - c0).max(1e-10);
            return if ascending {
                Some((i, i + 1, f))
            } else {
                Some((i + 1, i, 1.0 - f))
            };
        }
    }

    // Edge cases - clamp to bounds
    if value <= coords[0].min(coords[coords.len() - 1]) {
        Some((0, 0, 0.0))
    } else {
        let n = coords.len() - 1;
        Some((n, n, 0.0))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_value_check() {
        assert!(is_valid_f32(10.0));
        assert!(is_valid_f32(-5.0));
        assert!(!is_valid_f32(f32::NAN));
        assert!(!is_valid_f32(f32::INFINITY));
        assert!(!is_valid_f32(FILL_VALUE_F32));
        assert!(!is_valid_f32(1.0e31));
    }

    #[test]
    fn test_forcing_data_point() {
        let forcing = ForcingDataPoint {
            u10: 5.0,
            v10: 5.0,
            msl: 101325.0,
        };

        let speed = forcing.wind_speed();
        assert!((speed - 7.071).abs() < 0.01);

        let dir = forcing.wind_direction();
        assert!((dir - 225.0).abs() < 1.0); // Wind from southwest
    }

    #[test]
    fn test_netcdf_config() {
        let config = NetCDFWriterConfig::new("test.nc")
            .with_title("Test Simulation")
            .with_institution("Test University")
            .with_compression(6);

        assert_eq!(config.path, "test.nc");
        assert_eq!(config.title, Some("Test Simulation".to_string()));
        assert_eq!(config.compression_level, 6);
    }

    #[test]
    fn test_mesh_info() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 2.0];
        let mesh = NetCDFMeshInfo::from_xy(x.clone(), y.clone());

        assert_eq!(mesh.n_nodes, 3);
        assert_eq!(mesh.x, x);
        assert_eq!(mesh.y, y);
        assert!(mesh.lat.is_none());
    }
}
