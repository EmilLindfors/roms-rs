//! I/O utilities for reading and writing data files.
//!
//! This module provides:
//! - **Tidal constituents**: Harmonic data (amplitude, phase) for open boundary forcing
//! - **Boundary time series**: Time-varying state data for nesting from parent models
//! - **VTK output**: Solution visualization in ParaView (VTU format)
//! - **NetCDF I/O**: CF-conventions output and forcing data input (requires `netcdf` feature)
//! - **GeoTIFF bathymetry**: Load depth data from GeoTIFF raster files
//! - **Coastline data**: Load coastline polygons from GSHHS shapefiles
//! - **Coordinate projections**: Transform between geographic and Cartesian coordinates
//!
//! # File Formats
//!
//! ## Tidal Constituent Files
//!
//! ```text
//! # Tidal constituents for Bergen
//! # location: 5.32 60.39
//! # reference_level: 0.0
//! # columns: name amplitude(m) phase(deg)
//! M2 0.45 125.3
//! S2 0.15 158.7
//! K1 0.08 45.2
//! ```
//!
//! ## Nesting Time Series Files
//!
//! ```text
//! # Nesting data from parent model
//! # location: 5.0 60.0
//! # columns: time(s) h(m) hu(m2/s) hv(m2/s)
//! 0.0 10.5 1.2 0.3
//! 3600.0 10.8 1.1 0.4
//! ```
//!
//! ## GeoTIFF Bathymetry
//!
//! GeoTIFF files with ModelPixelScale and ModelTiepoint tags for georeferencing.
//! Depth values should be negative (below sea level).
//!
//! ## GSHHS Shapefiles
//!
//! Global Self-consistent Hierarchical High-resolution Shorelines.
//! Use GSHHS_f_L1.shp for full resolution coastlines.
//!
//! # Example
//!
//! ```ignore
//! use std::path::Path;
//! use dg::io::{read_constituent_file, read_timeseries_file, GeoTiffBathymetry, CoastlineData, GeoBoundingBox};
//!
//! // Read tidal constituents
//! let constituents = read_constituent_file(Path::new("tides.txt")).unwrap();
//! for c in &constituents.constituents {
//!     println!("{}: A={:.3} m, φ={:.1}°", c.name, c.amplitude, c.phase_degrees);
//! }
//!
//! // Read nesting time series
//! let ts = read_timeseries_file(Path::new("nesting.txt")).unwrap();
//! let state = ts.interpolate(1800.0);  // Interpolate at t=1800s
//!
//! // Load bathymetry from GeoTIFF
//! let bathy = GeoTiffBathymetry::load(Path::new("data/bathymetry.tif")).unwrap();
//! if let Some(depth) = bathy.get_depth(63.8, 8.9) {
//!     println!("Depth: {} m", depth);
//! }
//!
//! // Load coastline data
//! let bbox = GeoBoundingBox::new(8.0, 63.5, 9.5, 64.0);
//! let coastline = CoastlineData::load(Path::new("data/GSHHS_f_L1.shp"), &bbox).unwrap();
//! println!("Point is water: {}", coastline.is_water(63.8, 8.9));
//! ```

mod adcp_reader;
mod coastline;
mod constituent_reader;
mod geotiff;
#[cfg(feature = "netcdf")]
mod netcdf_io;
mod projection;
mod tide_gauge_reader;
mod timeseries_reader;
mod vtk;

pub use coastline::{
    CoastlineData, CoastlineError, CoastlineStatistics, FROYA_BBOX, NORWAY_BBOX,
};
pub use constituent_reader::{
    ConstituentData, ConstituentEntry, ConstituentFileError, constituent_period,
    parse_constituents, read_constituent_file,
};
pub use geotiff::{BathymetryStatistics, GeoTiffBathymetry, GeoTiffError};
#[cfg(feature = "netcdf")]
pub use netcdf_io::{
    ForcingDataPoint, ForcingReader, NetCDFError, NetCDFMeshInfo, NetCDFWriter,
    NetCDFWriterConfig, OceanGridType, OceanModelReader, OceanState,
    FILL_VALUE_F32, FILL_VALUE_F64, is_valid_f32, is_valid_f64,
};
pub use projection::{CoordinateProjection, GeoBoundingBox, LocalProjection, UtmProjection};
pub use timeseries_reader::{
    BoundaryTimeSeries, TimeSeriesFileError, TimeSeriesRecord, parse_timeseries,
    read_timeseries_file,
};
pub use tide_gauge_reader::{
    TideGaugeFile, TideGaugeFileError, files_to_observation_map, read_tide_gauge_directory,
    read_tide_gauge_file, write_tide_gauge_file,
};
pub use vtk::{VtkError, write_vtk_coupled, write_vtk_series, write_vtk_swe};
pub use adcp_reader::{ADCPFile, ADCPFileError, read_adcp_file, write_adcp_file};
