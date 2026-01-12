//! Coordinate projection utilities for geographic data.
//!
//! Provides transformations between geographic coordinates (WGS84 lat/lon)
//! and projected Cartesian coordinates (meters).
//!
//! # Supported Projections
//!
//! - **LocalProjection**: Simple tangent plane projection, fast and accurate for small domains
//! - **UtmProjection**: UTM Zone 32N for Norwegian coast
//!
//! # Example
//!
//! ```ignore
//! use dg::io::{LocalProjection, CoordinateProjection};
//!
//! // Create projection centered on Froya
//! let proj = LocalProjection::new(63.75, 8.75);
//!
//! // Convert geographic to local
//! let (x, y) = proj.geo_to_xy(63.8, 8.9);
//!
//! // Convert back
//! let (lat, lon) = proj.xy_to_geo(x, y);
//! ```

use std::f64::consts::PI;

/// Geographic bounding box in WGS84 coordinates.
#[derive(Debug, Clone, Copy)]
pub struct GeoBoundingBox {
    /// Minimum longitude (western edge) in degrees
    pub min_lon: f64,
    /// Minimum latitude (southern edge) in degrees
    pub min_lat: f64,
    /// Maximum longitude (eastern edge) in degrees
    pub max_lon: f64,
    /// Maximum latitude (northern edge) in degrees
    pub max_lat: f64,
}

impl GeoBoundingBox {
    /// Create a new bounding box.
    pub fn new(min_lon: f64, min_lat: f64, max_lon: f64, max_lat: f64) -> Self {
        Self {
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        }
    }

    /// Check if a point is within this bounding box.
    pub fn contains(&self, lat: f64, lon: f64) -> bool {
        lon >= self.min_lon && lon <= self.max_lon && lat >= self.min_lat && lat <= self.max_lat
    }

    /// Get the center of the bounding box.
    pub fn center(&self) -> (f64, f64) {
        (
            (self.min_lat + self.max_lat) / 2.0,
            (self.min_lon + self.max_lon) / 2.0,
        )
    }

    /// Expand the bounding box by a factor (1.1 = 10% expansion).
    pub fn expand(&self, factor: f64) -> Self {
        let lat_range = self.max_lat - self.min_lat;
        let lon_range = self.max_lon - self.min_lon;
        let lat_margin = lat_range * (factor - 1.0) / 2.0;
        let lon_margin = lon_range * (factor - 1.0) / 2.0;

        Self {
            min_lon: self.min_lon - lon_margin,
            min_lat: self.min_lat - lat_margin,
            max_lon: self.max_lon + lon_margin,
            max_lat: self.max_lat + lat_margin,
        }
    }
}

/// Trait for coordinate projections.
pub trait CoordinateProjection {
    /// Convert geographic coordinates (lat, lon) to projected (x, y) in meters.
    fn geo_to_xy(&self, lat: f64, lon: f64) -> (f64, f64);

    /// Convert projected coordinates (x, y) to geographic (lat, lon).
    fn xy_to_geo(&self, x: f64, y: f64) -> (f64, f64);
}

/// Local tangent plane projection.
///
/// A simple and fast projection for small domains (< 100 km).
/// Uses a flat Earth approximation centered on a reference point.
///
/// Accuracy: ~0.1% at 50 km from reference, ~0.5% at 100 km.
#[derive(Debug, Clone, Copy)]
pub struct LocalProjection {
    /// Reference latitude in degrees
    ref_lat: f64,
    /// Reference longitude in degrees
    ref_lon: f64,
    /// Precomputed cos(ref_lat) for efficiency
    cos_lat: f64,
    /// Meters per degree latitude (~111,320 m)
    meters_per_deg_lat: f64,
    /// Meters per degree longitude at reference latitude
    meters_per_deg_lon: f64,
}

impl LocalProjection {
    /// WGS84 equatorial radius in meters
    const A: f64 = 6_378_137.0;
    /// WGS84 flattening
    const F: f64 = 1.0 / 298.257_223_563;

    /// Create a local projection centered at the given reference point.
    ///
    /// # Arguments
    /// * `ref_lat` - Reference latitude in degrees
    /// * `ref_lon` - Reference longitude in degrees
    pub fn new(ref_lat: f64, ref_lon: f64) -> Self {
        let lat_rad = ref_lat * PI / 180.0;
        let cos_lat = lat_rad.cos();

        // More accurate formula accounting for Earth's ellipsoidal shape
        let e2 = 2.0 * Self::F - Self::F * Self::F;
        let sin_lat = lat_rad.sin();
        let sin2 = sin_lat * sin_lat;

        // Radius of curvature in meridian
        let rho = Self::A * (1.0 - e2) / (1.0 - e2 * sin2).powf(1.5);
        // Radius of curvature in prime vertical
        let nu = Self::A / (1.0 - e2 * sin2).sqrt();

        let meters_per_deg_lat = rho * PI / 180.0;
        let meters_per_deg_lon = nu * cos_lat * PI / 180.0;

        Self {
            ref_lat,
            ref_lon,
            cos_lat,
            meters_per_deg_lat,
            meters_per_deg_lon,
        }
    }

    /// Get the reference latitude.
    pub fn ref_lat(&self) -> f64 {
        self.ref_lat
    }

    /// Get the reference longitude.
    pub fn ref_lon(&self) -> f64 {
        self.ref_lon
    }

    /// Get the scale factors (meters per degree).
    pub fn scale_factors(&self) -> (f64, f64) {
        (self.meters_per_deg_lat, self.meters_per_deg_lon)
    }
}

impl CoordinateProjection for LocalProjection {
    fn geo_to_xy(&self, lat: f64, lon: f64) -> (f64, f64) {
        let x = (lon - self.ref_lon) * self.meters_per_deg_lon;
        let y = (lat - self.ref_lat) * self.meters_per_deg_lat;
        (x, y)
    }

    fn xy_to_geo(&self, x: f64, y: f64) -> (f64, f64) {
        let lat = self.ref_lat + y / self.meters_per_deg_lat;
        let lon = self.ref_lon + x / self.meters_per_deg_lon;
        (lat, lon)
    }
}

/// UTM projection for a specific zone.
///
/// Universal Transverse Mercator projection. More accurate than LocalProjection
/// for larger domains, but slower to compute.
#[derive(Debug, Clone, Copy)]
pub struct UtmProjection {
    /// Central meridian in degrees
    central_meridian: f64,
    /// Scale factor at central meridian (0.9996 for UTM)
    scale_factor: f64,
    /// False easting in meters (500,000 for UTM)
    false_easting: f64,
    /// False northing in meters (0 for northern hemisphere, 10,000,000 for southern)
    false_northing: f64,
    /// Zone number (1-60)
    zone: u8,
    /// Northern hemisphere flag
    northern: bool,
}

impl UtmProjection {
    /// WGS84 equatorial radius in meters
    const A: f64 = 6_378_137.0;
    /// WGS84 flattening
    const F: f64 = 1.0 / 298.257_223_563;

    /// Create UTM Zone 32N projection (covers Norway 6°E - 12°E).
    pub fn zone_32n() -> Self {
        Self {
            central_meridian: 9.0,
            scale_factor: 0.9996,
            false_easting: 500_000.0,
            false_northing: 0.0,
            zone: 32,
            northern: true,
        }
    }

    /// Create UTM Zone 33N projection (covers Norway 12°E - 18°E).
    pub fn zone_33n() -> Self {
        Self {
            central_meridian: 15.0,
            scale_factor: 0.9996,
            false_easting: 500_000.0,
            false_northing: 0.0,
            zone: 33,
            northern: true,
        }
    }

    /// Create a UTM projection for a given zone and hemisphere.
    pub fn new(zone: u8, northern: bool) -> Self {
        assert!((1..=60).contains(&zone), "UTM zone must be 1-60");
        let central_meridian = (zone as f64 - 1.0) * 6.0 - 180.0 + 3.0;

        Self {
            central_meridian,
            scale_factor: 0.9996,
            false_easting: 500_000.0,
            false_northing: if northern { 0.0 } else { 10_000_000.0 },
            zone,
            northern,
        }
    }

    /// Get the zone number.
    pub fn zone(&self) -> u8 {
        self.zone
    }
}

impl CoordinateProjection for UtmProjection {
    fn geo_to_xy(&self, lat: f64, lon: f64) -> (f64, f64) {
        let lat_rad = lat * PI / 180.0;
        let lon_rad = lon * PI / 180.0;
        let lon0_rad = self.central_meridian * PI / 180.0;

        let e2 = 2.0 * Self::F - Self::F * Self::F;
        let e_prime2 = e2 / (1.0 - e2);

        let n = Self::A / (1.0 - e2 * lat_rad.sin().powi(2)).sqrt();
        let t = lat_rad.tan().powi(2);
        let c = e_prime2 * lat_rad.cos().powi(2);
        let a_coef = (lon_rad - lon0_rad) * lat_rad.cos();

        // Meridian arc length
        let e4 = e2 * e2;
        let e6 = e4 * e2;
        let m = Self::A
            * ((1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * lat_rad
                - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * (2.0 * lat_rad).sin()
                + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * (4.0 * lat_rad).sin()
                - (35.0 * e6 / 3072.0) * (6.0 * lat_rad).sin());

        let x = self.scale_factor * n
            * (a_coef
                + (1.0 - t + c) * a_coef.powi(3) / 6.0
                + (5.0 - 18.0 * t + t * t + 72.0 * c - 58.0 * e_prime2) * a_coef.powi(5) / 120.0)
            + self.false_easting;

        let y = self.scale_factor
            * (m
                + n * lat_rad.tan()
                    * (a_coef.powi(2) / 2.0
                        + (5.0 - t + 9.0 * c + 4.0 * c * c) * a_coef.powi(4) / 24.0
                        + (61.0 - 58.0 * t + t * t + 600.0 * c - 330.0 * e_prime2)
                            * a_coef.powi(6)
                            / 720.0))
            + self.false_northing;

        (x, y)
    }

    fn xy_to_geo(&self, x: f64, y: f64) -> (f64, f64) {
        let x = x - self.false_easting;
        let y = y - self.false_northing;

        let e2 = 2.0 * Self::F - Self::F * Self::F;
        let e_prime2 = e2 / (1.0 - e2);
        let e1 = (1.0 - (1.0 - e2).sqrt()) / (1.0 + (1.0 - e2).sqrt());

        let m = y / self.scale_factor;
        let mu = m
            / (Self::A
                * (1.0 - e2 / 4.0 - 3.0 * e2 * e2 / 64.0 - 5.0 * e2 * e2 * e2 / 256.0));

        let phi1 = mu
            + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
            + (21.0 * e1 * e1 / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
            + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin()
            + (1097.0 * e1.powi(4) / 512.0) * (8.0 * mu).sin();

        let n1 = Self::A / (1.0 - e2 * phi1.sin().powi(2)).sqrt();
        let t1 = phi1.tan().powi(2);
        let c1 = e_prime2 * phi1.cos().powi(2);
        let r1 = Self::A * (1.0 - e2) / (1.0 - e2 * phi1.sin().powi(2)).powf(1.5);
        let d = x / (n1 * self.scale_factor);

        let lat = phi1
            - (n1 * phi1.tan() / r1)
                * (d * d / 2.0
                    - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * e_prime2) * d.powi(4)
                        / 24.0
                    + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1 * t1 - 252.0 * e_prime2
                        - 3.0 * c1 * c1)
                        * d.powi(6)
                        / 720.0);

        let lon = self.central_meridian * PI / 180.0
            + (d
                - (1.0 + 2.0 * t1 + c1) * d.powi(3) / 6.0
                + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1 * c1 + 8.0 * e_prime2 + 24.0 * t1 * t1)
                    * d.powi(5)
                    / 120.0)
                / phi1.cos();

        (lat * 180.0 / PI, lon * 180.0 / PI)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    #[test]
    fn test_local_projection_roundtrip() {
        let proj = LocalProjection::new(63.75, 8.75);

        // Test roundtrip at several points
        let test_points = [
            (63.75, 8.75),   // Reference point
            (63.80, 8.90),   // Northeast
            (63.70, 8.60),   // Southwest
            (64.00, 9.00),   // Further north
            (63.50, 8.50),   // Further south
        ];

        for (lat, lon) in test_points {
            let (x, y) = proj.geo_to_xy(lat, lon);
            let (lat2, lon2) = proj.xy_to_geo(x, y);
            assert!(
                (lat - lat2).abs() < TOL,
                "Latitude roundtrip failed: {} -> {} -> {}",
                lat,
                y,
                lat2
            );
            assert!(
                (lon - lon2).abs() < TOL,
                "Longitude roundtrip failed: {} -> {} -> {}",
                lon,
                x,
                lon2
            );
        }
    }

    #[test]
    fn test_local_projection_scale() {
        let proj = LocalProjection::new(64.0, 9.0);

        // At 64°N, expect ~111 km per degree latitude
        let (meters_lat, meters_lon) = proj.scale_factors();
        assert!(
            (meters_lat - 111_000.0).abs() < 1000.0,
            "Latitude scale: {}",
            meters_lat
        );

        // At 64°N, expect ~49 km per degree longitude (cos(64°) ≈ 0.44)
        assert!(
            (meters_lon - 49_000.0).abs() < 1000.0,
            "Longitude scale: {}",
            meters_lon
        );
    }

    #[test]
    fn test_utm_zone_32n() {
        let proj = UtmProjection::zone_32n();

        // Test known point: Bergen (60.39°N, 5.32°E)
        // Expected UTM32N: approximately 297000 E, 6700000 N
        let (x, y) = proj.geo_to_xy(60.39, 5.32);
        assert!(
            (x - 297_000.0).abs() < 1000.0,
            "UTM easting for Bergen: {}",
            x
        );
        assert!(
            (y - 6_700_000.0).abs() < 10_000.0,
            "UTM northing for Bergen: {}",
            y
        );

        // Test roundtrip
        let (lat, lon) = proj.xy_to_geo(x, y);
        assert!((lat - 60.39).abs() < 0.001, "UTM lat roundtrip: {}", lat);
        assert!((lon - 5.32).abs() < 0.001, "UTM lon roundtrip: {}", lon);
    }

    #[test]
    fn test_geo_bbox() {
        let bbox = GeoBoundingBox::new(8.0, 63.5, 9.5, 64.0);

        assert!(bbox.contains(63.75, 8.75));
        assert!(!bbox.contains(65.0, 8.75));
        assert!(!bbox.contains(63.75, 10.0));

        let (center_lat, center_lon) = bbox.center();
        assert!((center_lat - 63.75).abs() < TOL);
        assert!((center_lon - 8.75).abs() < TOL);

        let expanded = bbox.expand(1.1);
        assert!(expanded.min_lon < bbox.min_lon);
        assert!(expanded.max_lon > bbox.max_lon);
    }
}
