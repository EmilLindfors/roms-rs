//! GSHHS coastline data reader.
//!
//! Loads coastline polygons from GSHHS (Global Self-consistent Hierarchical
//! High-resolution Geography) shapefiles and provides land/water classification.
//!
//! # Example
//!
//! ```ignore
//! use std::path::Path;
//! use dg::io::{CoastlineData, GeoBoundingBox};
//!
//! let bbox = GeoBoundingBox::new(8.0, 63.5, 9.5, 64.0);
//! let coastline = CoastlineData::load(Path::new("data/GSHHS_f_L1.shp"), &bbox)?;
//!
//! // Check if a point is in water
//! if coastline.is_water(63.8, 8.9) {
//!     println!("Point is in water");
//! }
//! ```

use std::fmt;
use std::path::Path;

use geo::{Contains, Coord, LineString, MultiPolygon, Point, Polygon};
use shapefile::{Reader, Shape};
use thiserror::Error;

use super::projection::GeoBoundingBox;

/// Error type for coastline operations.
#[derive(Debug, Error)]
pub enum CoastlineError {
    /// File I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Shapefile parsing error
    #[error("Shapefile error: {0}")]
    Shapefile(String),

    /// No polygons found in the data
    #[error("No polygons found in shapefile")]
    NoPolygons,
}

impl From<shapefile::Error> for CoastlineError {
    fn from(e: shapefile::Error) -> Self {
        CoastlineError::Shapefile(e.to_string())
    }
}

/// Coastline data from GSHHS shapefile.
///
/// Stores land polygons and provides efficient point-in-polygon tests
/// for land/water classification.
pub struct CoastlineData {
    /// Land mass polygons (GSHHS level 1 = ocean/land boundary)
    land_polygons: MultiPolygon<f64>,
    /// Bounding box of loaded data
    bbox: GeoBoundingBox,
    /// Number of polygons loaded
    polygon_count: usize,
}

impl CoastlineData {
    /// Load coastline data from a GSHHS shapefile.
    ///
    /// Only polygons intersecting the given bounding box are loaded.
    ///
    /// # Arguments
    /// * `path` - Path to GSHHS shapefile (.shp)
    /// * `bbox` - Geographic bounding box to filter polygons
    pub fn load<P: AsRef<Path>>(path: P, bbox: &GeoBoundingBox) -> Result<Self, CoastlineError> {
        let mut reader = Reader::from_path(path)?;
        let mut polygons = Vec::new();

        for result in reader.iter_shapes_and_records() {
            let (shape, _record) = result?;

            match shape {
                Shape::Polygon(polygon) => {
                    for ring in polygon.rings() {
                        let coords: Vec<Coord<f64>> = ring
                            .points()
                            .iter()
                            .map(|p| Coord { x: p.x, y: p.y })
                            .collect();

                        // Check if polygon intersects our bbox
                        if coords_intersect_bbox(&coords, bbox) {
                            // Filter coords to only those near our bbox (with margin)
                            let exterior = LineString::from(coords);
                            polygons.push(Polygon::new(exterior, vec![]));
                        }
                    }
                }
                _ => {} // Ignore non-polygon shapes
            }
        }

        if polygons.is_empty() {
            // This might be OK - could be all water
            // Return empty data rather than error
        }

        let polygon_count = polygons.len();

        Ok(Self {
            land_polygons: MultiPolygon(polygons),
            bbox: *bbox,
            polygon_count,
        })
    }

    /// Check if a point is in water (not inside any land polygon).
    ///
    /// Returns true if the point is outside all land polygons.
    /// Points outside the data bounding box are assumed to be water.
    pub fn is_water(&self, lat: f64, lon: f64) -> bool {
        // Points outside bbox are assumed water (open ocean)
        if !self.bbox.contains(lat, lon) {
            return true;
        }

        let point = Point::new(lon, lat);

        // Check if point is inside any land polygon
        !self.land_polygons.contains(&point)
    }

    /// Check if a point is on land.
    pub fn is_land(&self, lat: f64, lon: f64) -> bool {
        !self.is_water(lat, lon)
    }

    /// Get the bounding box of this coastline data.
    pub fn bbox(&self) -> &GeoBoundingBox {
        &self.bbox
    }

    /// Get the number of polygons loaded.
    pub fn polygon_count(&self) -> usize {
        self.polygon_count
    }

    /// Get statistics about the coastline data.
    pub fn statistics(&self) -> CoastlineStatistics {
        let mut total_vertices = 0;

        for polygon in self.land_polygons.0.iter() {
            total_vertices += polygon.exterior().0.len();
            for interior in polygon.interiors() {
                total_vertices += interior.0.len();
            }
        }

        CoastlineStatistics {
            polygon_count: self.polygon_count,
            total_vertices,
            bbox: self.bbox,
        }
    }
}

/// Statistics about coastline data.
#[derive(Debug, Clone)]
pub struct CoastlineStatistics {
    /// Number of land polygons
    pub polygon_count: usize,
    /// Total number of vertices
    pub total_vertices: usize,
    /// Geographic bounding box
    pub bbox: GeoBoundingBox,
}

impl fmt::Display for CoastlineStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Coastline Statistics:")?;
        writeln!(f, "  Polygons: {}", self.polygon_count)?;
        writeln!(f, "  Total vertices: {}", self.total_vertices)?;
        writeln!(
            f,
            "  Bounding box: lon [{:.4}, {:.4}], lat [{:.4}, {:.4}]",
            self.bbox.min_lon, self.bbox.max_lon, self.bbox.min_lat, self.bbox.max_lat
        )
    }
}

/// Check if any coordinates intersect the bounding box.
fn coords_intersect_bbox(coords: &[Coord<f64>], bbox: &GeoBoundingBox) -> bool {
    coords.iter().any(|c| {
        c.x >= bbox.min_lon && c.x <= bbox.max_lon && c.y >= bbox.min_lat && c.y <= bbox.max_lat
    })
}

/// Norway's coastal bounding box (useful preset).
pub const NORWAY_BBOX: GeoBoundingBox = GeoBoundingBox {
    min_lon: 4.5,
    min_lat: 57.5,
    max_lon: 31.0,
    max_lat: 71.5,
};

/// Froya-Smola-Hitra region bounding box.
pub const FROYA_BBOX: GeoBoundingBox = GeoBoundingBox {
    min_lon: 7.5,
    min_lat: 63.3,
    max_lon: 10.0,
    max_lat: 64.2,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_constants() {
        // Verify Norway bbox is reasonable
        assert!(NORWAY_BBOX.min_lon < NORWAY_BBOX.max_lon);
        assert!(NORWAY_BBOX.min_lat < NORWAY_BBOX.max_lat);

        // Verify Froya bbox is within Norway
        assert!(NORWAY_BBOX.contains(FROYA_BBOX.min_lat, FROYA_BBOX.min_lon));
        assert!(NORWAY_BBOX.contains(FROYA_BBOX.max_lat, FROYA_BBOX.max_lon));
    }

    #[test]
    fn test_coords_intersect() {
        let bbox = GeoBoundingBox::new(8.0, 63.5, 9.5, 64.0);

        // Coords inside bbox
        let inside = vec![
            Coord { x: 8.5, y: 63.75 },
            Coord { x: 9.0, y: 63.8 },
        ];
        assert!(coords_intersect_bbox(&inside, &bbox));

        // Coords outside bbox
        let outside = vec![
            Coord { x: 5.0, y: 60.0 },
            Coord { x: 6.0, y: 61.0 },
        ];
        assert!(!coords_intersect_bbox(&outside, &bbox));
    }
}
