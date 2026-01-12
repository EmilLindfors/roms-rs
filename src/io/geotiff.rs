//! GeoTIFF bathymetry reader.
//!
//! Loads bathymetry data from GeoTIFF files and provides depth interpolation.
//! Uses pure Rust `tiff` crate - no system dependencies required.
//!
//! # Example
//!
//! ```ignore
//! use std::path::Path;
//! use dg::io::GeoTiffBathymetry;
//!
//! let bathy = GeoTiffBathymetry::load(Path::new("data/bathymetry.tif"))?;
//!
//! // Get depth at a geographic coordinate
//! if let Some(depth) = bathy.get_depth(63.8, 8.9) {
//!     println!("Depth: {} m", depth);
//! }
//! ```

use std::fmt;
use std::fs::File;
use std::path::Path;

use thiserror::Error;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;

use super::projection::GeoBoundingBox;

/// Error type for GeoTIFF operations.
#[derive(Debug, Error)]
pub enum GeoTiffError {
    /// File I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// TIFF decoding error
    #[error("TIFF error: {0}")]
    Tiff(String),

    /// Missing or invalid geotransform tags
    #[error("Missing geotransform: {0}")]
    MissingGeotransform(String),

    /// Unsupported data type
    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(String),
}

impl From<tiff::TiffError> for GeoTiffError {
    fn from(e: tiff::TiffError) -> Self {
        GeoTiffError::Tiff(e.to_string())
    }
}

/// GeoTIFF-based bathymetry provider.
///
/// Loads depth data from a GeoTIFF file and provides interpolation methods.
/// Depth values are stored as negative (below sea level) following oceanographic convention.
pub struct GeoTiffBathymetry {
    /// Depth data (rows x cols), stored as f32 for memory efficiency
    depths: Vec<Vec<f32>>,
    /// Geographic bounding box
    bbox: GeoBoundingBox,
    /// Width in pixels
    width: usize,
    /// Height in pixels
    height: usize,
    /// No data value
    nodata: f32,
}

impl GeoTiffBathymetry {
    /// Load bathymetry from a GeoTIFF file.
    ///
    /// Extracts geotransform from ModelPixelScale (tag 33550) and
    /// ModelTiepoint (tag 33922) tags.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, GeoTiffError> {
        Self::load_with_bbox(path, None)
    }

    /// Load bathymetry with an optional bounding box hint.
    ///
    /// If the GeoTIFF lacks proper geotransform tags, the bbox hint is used.
    pub fn load_with_bbox<P: AsRef<Path>>(
        path: P,
        bbox_hint: Option<GeoBoundingBox>,
    ) -> Result<Self, GeoTiffError> {
        let file = File::open(&path)?;
        let mut decoder = Decoder::new(file)?;

        let (width, height) = decoder.dimensions()?;

        // Try to read GeoTIFF geotransform tags
        let pixel_scale = decoder.get_tag_f64_vec(Tag::Unknown(33550)).ok();
        let model_tiepoint = decoder.get_tag_f64_vec(Tag::Unknown(33922)).ok();

        // Calculate bounding box from GeoTIFF tags
        let bbox = if let (Some(scale), Some(tiepoint)) = (pixel_scale, model_tiepoint) {
            // ModelTiepoint format: [I, J, K, X, Y, Z]
            // ModelPixelScale format: [ScaleX, ScaleY, ScaleZ]
            if tiepoint.len() >= 6 && scale.len() >= 2 {
                let origin_x = tiepoint[3]; // X coordinate of origin (longitude)
                let origin_y = tiepoint[4]; // Y coordinate of origin (latitude)
                let pixel_width = scale[0];
                let pixel_height = scale[1];

                let min_lon = origin_x;
                let max_lon = origin_x + (width as f64 * pixel_width);
                let max_lat = origin_y;
                let min_lat = origin_y - (height as f64 * pixel_height);

                GeoBoundingBox::new(min_lon, min_lat, max_lon, max_lat)
            } else if let Some(hint) = bbox_hint {
                hint
            } else {
                return Err(GeoTiffError::MissingGeotransform(
                    "Invalid GeoTIFF tags and no bbox hint provided".to_string(),
                ));
            }
        } else if let Some(hint) = bbox_hint {
            hint
        } else {
            return Err(GeoTiffError::MissingGeotransform(
                "No GeoTIFF geotransform found and no bbox hint provided".to_string(),
            ));
        };

        // Decode the image
        let result = decoder.read_image()?;

        let depths_flat: Vec<f32> = match result {
            DecodingResult::U8(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::U16(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::U32(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::U64(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::F32(data) => data,
            DecodingResult::F64(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::I8(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::I16(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::I32(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::I64(data) => data.into_iter().map(|v| v as f32).collect(),
        };

        // Convert flat array to 2D array
        let mut depths = Vec::with_capacity(height as usize);
        for row in 0..height {
            let row_start = (row * width) as usize;
            let row_end = row_start + width as usize;
            depths.push(depths_flat[row_start..row_end].to_vec());
        }

        Ok(Self {
            depths,
            bbox,
            width: width as usize,
            height: height as usize,
            nodata: -9999.0,
        })
    }

    /// Set the no-data value.
    pub fn set_nodata(&mut self, nodata: f32) {
        self.nodata = nodata;
    }

    /// Get the bounding box of this bathymetry data.
    pub fn bbox(&self) -> &GeoBoundingBox {
        &self.bbox
    }

    /// Get the dimensions (width, height) in pixels.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Convert lat/lon to pixel coordinates.
    fn latlon_to_pixel(&self, lat: f64, lon: f64) -> Option<(usize, usize)> {
        if !self.bbox.contains(lat, lon) {
            return None;
        }

        let col =
            ((lon - self.bbox.min_lon) / (self.bbox.max_lon - self.bbox.min_lon) * self.width as f64)
                as usize;
        let row = ((self.bbox.max_lat - lat) / (self.bbox.max_lat - self.bbox.min_lat)
            * self.height as f64) as usize;

        if row >= self.height || col >= self.width {
            return None;
        }

        Some((row, col))
    }

    /// Convert lat/lon to fractional pixel coordinates for interpolation.
    fn latlon_to_pixel_frac(&self, lat: f64, lon: f64) -> Option<(f64, f64)> {
        if !self.bbox.contains(lat, lon) {
            return None;
        }

        let col_frac =
            (lon - self.bbox.min_lon) / (self.bbox.max_lon - self.bbox.min_lon) * self.width as f64;
        let row_frac = (self.bbox.max_lat - lat) / (self.bbox.max_lat - self.bbox.min_lat)
            * self.height as f64;

        Some((row_frac, col_frac))
    }

    /// Check if a depth value is valid (not nodata, not NaN, not land).
    fn is_valid_depth(&self, depth: f32) -> bool {
        !depth.is_nan()
            && !depth.is_infinite()
            && (depth - self.nodata).abs() > 0.01
            && depth <= 0.0 // Positive values indicate land elevation
    }

    /// Get depth at geographic coordinates using nearest neighbor.
    ///
    /// Returns None if the point is outside the data bounds, on land, or nodata.
    /// Depth is returned as f64 for DG precision requirements.
    pub fn get_depth(&self, lat: f64, lon: f64) -> Option<f64> {
        let (row, col) = self.latlon_to_pixel(lat, lon)?;
        let depth = self.depths[row][col];

        if self.is_valid_depth(depth) {
            Some(depth as f64)
        } else {
            None
        }
    }

    /// Get depth at geographic coordinates using bilinear interpolation.
    ///
    /// Provides smoother depth values for better gradient computation.
    /// Returns None if any of the interpolation points are invalid.
    pub fn get_depth_bilinear(&self, lat: f64, lon: f64) -> Option<f64> {
        let (row_frac, col_frac) = self.latlon_to_pixel_frac(lat, lon)?;

        let row0 = row_frac.floor() as usize;
        let col0 = col_frac.floor() as usize;
        let row1 = (row0 + 1).min(self.height - 1);
        let col1 = (col0 + 1).min(self.width - 1);

        // Get four corner depths
        let d00 = self.depths[row0][col0];
        let d01 = self.depths[row0][col1];
        let d10 = self.depths[row1][col0];
        let d11 = self.depths[row1][col1];

        // Check all corners are valid
        if !self.is_valid_depth(d00)
            || !self.is_valid_depth(d01)
            || !self.is_valid_depth(d10)
            || !self.is_valid_depth(d11)
        {
            // Fall back to nearest neighbor
            return self.get_depth(lat, lon);
        }

        // Bilinear interpolation
        let t = row_frac - row0 as f64;
        let s = col_frac - col0 as f64;

        let depth = (1.0 - t) * (1.0 - s) * d00 as f64
            + (1.0 - t) * s * d01 as f64
            + t * (1.0 - s) * d10 as f64
            + t * s * d11 as f64;

        Some(depth)
    }

    /// Check if a point is in water (valid depth data exists).
    pub fn is_water(&self, lat: f64, lon: f64) -> bool {
        self.get_depth(lat, lon).is_some()
    }

    /// Get the depth range in the data.
    pub fn depth_range(&self) -> (f64, f64) {
        let mut min_depth = f64::INFINITY;
        let mut max_depth = f64::NEG_INFINITY;

        for row in &self.depths {
            for &depth in row {
                if self.is_valid_depth(depth) {
                    let d = depth as f64;
                    min_depth = min_depth.min(d);
                    max_depth = max_depth.max(d);
                }
            }
        }

        (min_depth, max_depth)
    }

    /// Get statistics about the bathymetry data.
    pub fn statistics(&self) -> BathymetryStatistics {
        let mut valid_count = 0usize;
        let mut nodata_count = 0usize;
        let mut land_count = 0usize;
        let mut sum = 0.0f64;
        let mut min_depth = f64::INFINITY;
        let mut max_depth = f64::NEG_INFINITY;

        for row in &self.depths {
            for &depth in row {
                if (depth - self.nodata).abs() < 0.01 || depth.is_nan() || depth.is_infinite() {
                    nodata_count += 1;
                } else if depth > 0.0 {
                    land_count += 1;
                } else {
                    valid_count += 1;
                    let d = depth as f64;
                    sum += d;
                    min_depth = min_depth.min(d);
                    max_depth = max_depth.max(d);
                }
            }
        }

        let mean = if valid_count > 0 {
            sum / valid_count as f64
        } else {
            0.0
        };

        BathymetryStatistics {
            width: self.width,
            height: self.height,
            valid_count,
            nodata_count,
            land_count,
            min_depth: if min_depth.is_infinite() {
                0.0
            } else {
                min_depth
            },
            max_depth: if max_depth.is_infinite() {
                0.0
            } else {
                max_depth
            },
            mean_depth: mean,
            bbox: self.bbox,
        }
    }
}

/// Statistics about a bathymetry dataset.
#[derive(Debug, Clone)]
pub struct BathymetryStatistics {
    /// Width in pixels
    pub width: usize,
    /// Height in pixels
    pub height: usize,
    /// Number of valid water cells
    pub valid_count: usize,
    /// Number of nodata cells
    pub nodata_count: usize,
    /// Number of land cells (positive elevation)
    pub land_count: usize,
    /// Minimum depth (most negative, deepest)
    pub min_depth: f64,
    /// Maximum depth (least negative, shallowest water)
    pub max_depth: f64,
    /// Mean depth
    pub mean_depth: f64,
    /// Geographic bounding box
    pub bbox: GeoBoundingBox,
}

impl fmt::Display for BathymetryStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Bathymetry Statistics:")?;
        writeln!(f, "  Dimensions: {}x{} pixels", self.width, self.height)?;
        writeln!(
            f,
            "  Total cells: {}",
            self.width * self.height
        )?;
        writeln!(f, "  Valid water cells: {}", self.valid_count)?;
        writeln!(f, "  Land cells: {}", self.land_count)?;
        writeln!(f, "  NoData cells: {}", self.nodata_count)?;
        writeln!(
            f,
            "  Depth range: {:.1} to {:.1} m",
            self.min_depth, self.max_depth
        )?;
        writeln!(f, "  Mean depth: {:.1} m", self.mean_depth)?;
        writeln!(
            f,
            "  Bounding box: lon [{:.4}, {:.4}], lat [{:.4}, {:.4}]",
            self.bbox.min_lon, self.bbox.max_lon, self.bbox.min_lat, self.bbox.max_lat
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_bbox_contains() {
        let bbox = GeoBoundingBox::new(8.0, 63.5, 9.5, 64.0);
        assert!(bbox.contains(63.75, 8.75));
        assert!(!bbox.contains(65.0, 8.75));
    }

    // Integration tests would require actual GeoTIFF files
    // These are tested in examples/froya_real_data.rs
}
