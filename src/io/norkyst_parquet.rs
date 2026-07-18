//! Native reader for `norkyst-client` **grid** parquet output (feature `parquet`).
//!
//! Grid extraction (`--bbox`/`--selection`) runs over OPeNDAP and stays
//! available when the point/NCSS service is down, so it is the reliable path
//! for fetching NorKyst-800 tidal data — but it writes a partitioned parquet
//! dataset, not text. This reader ingests that parquet directly into the same
//! [`TimeSeries`] the validation harness consumes, so no external
//! convert-to-text step is needed.
//!
//! # Expected schema
//!
//! The columns written by `norkyst-client`'s `grid_writer` (superset used here):
//! `time` (Utf8 RFC3339), `grid_x`/`grid_y` (Int32), `latitude`/`longitude`
//! (Float64), and the nullable Float64 fields `sea_surface_height`,
//! `bottom_depth`, `temperature`, `salinity`, `u_current`, `v_current`.
//! `sea_surface_height`/`bottom_depth` are absent in pre-zeta client builds;
//! rows then carry `None` and the elevation series comes back empty.
//!
//! A grid file holds many cells × many times. Pick the cell nearest a station
//! with [`NorKystGridData::nearest_cell`] (or use the `*_nearest` helpers).

use std::path::Path;

use arrow::array::{Array, Float64Array, Int32Array, StringArray};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use thiserror::Error;

use super::norkyst_time::parse_datetime_seconds;
use crate::analysis::TimeSeries;

/// Error type for reading NorKyst grid parquet.
#[derive(Debug, Error)]
pub enum NorKystGridError {
    /// File I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Underlying parquet decode error.
    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    /// Underlying arrow decode error (record-batch iteration).
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    /// A required column is missing or has an unexpected type.
    #[error("schema error: {0}")]
    Schema(String),

    /// A `time` value could not be parsed.
    #[error("time parse error: {0}")]
    Time(String),

    /// No parquet files were found under the given directory.
    #[error("no parquet files found under {0}")]
    NoFiles(String),
}

/// One decoded grid row: a single cell at a single time.
#[derive(Clone, Copy, Debug)]
pub struct NorKystGridRow {
    /// Seconds since the Unix epoch (1970-01-01 00:00:00 UTC).
    pub time: f64,
    /// Model grid x index.
    pub grid_x: i32,
    /// Model grid y index.
    pub grid_y: i32,
    /// Latitude (degrees North).
    pub lat: f64,
    /// Longitude (degrees East).
    pub lon: f64,
    /// Sea-surface height `zeta` (m), if present and non-fill.
    pub sea_surface_height: Option<f64>,
    /// Still-water bottom depth `h` (m), if present and non-fill.
    pub bottom_depth: Option<f64>,
    /// Temperature (°C).
    pub temperature: Option<f64>,
    /// Salinity (PSU).
    pub salinity: Option<f64>,
    /// Eastward current (m/s).
    pub u_current: Option<f64>,
    /// Northward current (m/s).
    pub v_current: Option<f64>,
}

/// Decoded NorKyst grid parquet: all rows across all cells and times.
#[derive(Clone, Debug, Default)]
pub struct NorKystGridData {
    /// All decoded rows, in file order.
    pub rows: Vec<NorKystGridRow>,
}

impl NorKystGridData {
    /// Number of rows (cell × time samples).
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether there are no rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The `(grid_x, grid_y)` of the wet cell nearest `(lon, lat)`.
    ///
    /// "Wet" means the cell has at least one row with a sea-surface height, so
    /// land/fill cells are skipped. Distance uses an equirectangular
    /// approximation (adequate at a single fjord's scale). Returns `None` if no
    /// cell has sea-surface height.
    pub fn nearest_cell(&self, lon: f64, lat: f64) -> Option<(i32, i32)> {
        let cos_lat = lat.to_radians().cos();
        self.rows
            .iter()
            .filter(|r| r.sea_surface_height.is_some())
            .map(|r| {
                let dlat = r.lat - lat;
                let dlon = (r.lon - lon) * cos_lat;
                ((r.grid_x, r.grid_y), dlat * dlat + dlon * dlon)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(cell, _)| cell)
    }

    /// Time-ordered rows for one grid cell.
    fn cell_rows_sorted(&self, cell: (i32, i32)) -> Vec<&NorKystGridRow> {
        let mut rows: Vec<&NorKystGridRow> = self
            .rows
            .iter()
            .filter(|r| (r.grid_x, r.grid_y) == cell)
            .collect();
        rows.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rows
    }

    /// Sea-surface height (`zeta`) series at the wet cell nearest `(lon, lat)`.
    ///
    /// This is the series to feed the tidal validation path. Empty if no cell
    /// has sea-surface height (e.g. a pre-zeta client build).
    pub fn sea_surface_height_series_nearest(&self, lon: f64, lat: f64) -> TimeSeries {
        let Some(cell) = self.nearest_cell(lon, lat) else {
            return TimeSeries::new(&[], &[]);
        };
        let (times, values): (Vec<f64>, Vec<f64>) = self
            .cell_rows_sorted(cell)
            .iter()
            .filter_map(|r| r.sea_surface_height.map(|z| (r.time, z)))
            .unzip();
        TimeSeries::new(&times, &values)
    }

    /// Surface current series `(u, v)` at the wet cell nearest `(lon, lat)`.
    ///
    /// Rows missing either component are skipped. Useful for a first comparison
    /// against ADCP surface currents.
    pub fn surface_current_series_nearest(&self, lon: f64, lat: f64) -> (TimeSeries, TimeSeries) {
        let Some(cell) = self.nearest_cell(lon, lat) else {
            return (TimeSeries::new(&[], &[]), TimeSeries::new(&[], &[]));
        };
        let mut times = Vec::new();
        let mut us = Vec::new();
        let mut vs = Vec::new();
        for r in self.cell_rows_sorted(cell) {
            if let (Some(u), Some(v)) = (r.u_current, r.v_current) {
                times.push(r.time);
                us.push(u);
                vs.push(v);
            }
        }
        (TimeSeries::new(&times, &us), TimeSeries::new(&times, &vs))
    }
}

/// Read a single NorKyst grid parquet file.
pub fn read_norkyst_parquet(path: &Path) -> Result<NorKystGridData, NorKystGridError> {
    let mut data = NorKystGridData::default();
    read_one_into(path, &mut data.rows)?;
    Ok(data)
}

/// Read every `*.parquet` under `dir` (recursively), concatenating rows.
///
/// A `norkyst-client` grid dataset is a partitioned directory tree, so this is
/// the usual entry point.
pub fn read_norkyst_parquet_glob(dir: &Path) -> Result<NorKystGridData, NorKystGridError> {
    let mut files = Vec::new();
    collect_parquet_files(dir, &mut files)?;
    if files.is_empty() {
        return Err(NorKystGridError::NoFiles(dir.display().to_string()));
    }
    files.sort();
    let mut data = NorKystGridData::default();
    for f in &files {
        read_one_into(f, &mut data.rows)?;
    }
    Ok(data)
}

fn collect_parquet_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
    if dir.is_file() {
        if dir.extension().is_some_and(|e| e == "parquet") {
            out.push(dir.to_path_buf());
        }
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_parquet_files(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "parquet") {
            out.push(path);
        }
    }
    Ok(())
}

fn read_one_into(path: &Path, rows: &mut Vec<NorKystGridRow>) -> Result<(), NorKystGridError> {
    let file = std::fs::File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    for batch in reader {
        decode_batch(&batch?, rows)?;
    }
    Ok(())
}

fn decode_batch(
    batch: &RecordBatch,
    rows: &mut Vec<NorKystGridRow>,
) -> Result<(), NorKystGridError> {
    let time = str_col(batch, "time")?;
    let grid_x = i32_col(batch, "grid_x")?;
    let grid_y = i32_col(batch, "grid_y")?;
    let lat = f64_col(batch, "latitude")?;
    let lon = f64_col(batch, "longitude")?;
    // Optional columns (absent in older parquet).
    let ssh = f64_col_opt(batch, "sea_surface_height")?;
    let bottom = f64_col_opt(batch, "bottom_depth")?;
    let temp = f64_col_opt(batch, "temperature")?;
    let sal = f64_col_opt(batch, "salinity")?;
    let u = f64_col_opt(batch, "u_current")?;
    let v = f64_col_opt(batch, "v_current")?;

    rows.reserve(batch.num_rows());
    for i in 0..batch.num_rows() {
        let t = parse_datetime_seconds(time.value(i)).map_err(NorKystGridError::Time)?;
        rows.push(NorKystGridRow {
            time: t,
            grid_x: grid_x.value(i),
            grid_y: grid_y.value(i),
            lat: lat.value(i),
            lon: lon.value(i),
            sea_surface_height: opt_at(ssh, i),
            bottom_depth: opt_at(bottom, i),
            temperature: opt_at(temp, i),
            salinity: opt_at(sal, i),
            u_current: opt_at(u, i),
            v_current: opt_at(v, i),
        });
    }
    Ok(())
}

/// Value of a nullable Float64 column at row `i`, mapping null / NaN to `None`.
fn opt_at(col: Option<&Float64Array>, i: usize) -> Option<f64> {
    let arr = col?;
    if arr.is_null(i) {
        return None;
    }
    let v = arr.value(i);
    if v.is_nan() { None } else { Some(v) }
}

fn str_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray, NorKystGridError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| NorKystGridError::Schema(format!("missing column {name:?}")))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| NorKystGridError::Schema(format!("column {name:?} is not Utf8")))
}

fn i32_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int32Array, NorKystGridError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| NorKystGridError::Schema(format!("missing column {name:?}")))?
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| NorKystGridError::Schema(format!("column {name:?} is not Int32")))
}

fn f64_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float64Array, NorKystGridError> {
    f64_col_opt(batch, name)?
        .ok_or_else(|| NorKystGridError::Schema(format!("missing column {name:?}")))
}

/// A Float64 column if present; `Ok(None)` if the column is absent (optional
/// fields), `Err` if present with the wrong type.
fn f64_col_opt<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<Option<&'a Float64Array>, NorKystGridError> {
    match batch.column_by_name(name) {
        None => Ok(None),
        Some(col) => col
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(Some)
            .ok_or_else(|| NorKystGridError::Schema(format!("column {name:?} is not Float64"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    const TOL: f64 = 1e-9;

    fn row(time: f64, gx: i32, gy: i32, lat: f64, lon: f64, ssh: Option<f64>) -> NorKystGridRow {
        NorKystGridRow {
            time,
            grid_x: gx,
            grid_y: gy,
            lat,
            lon,
            sea_surface_height: ssh,
            bottom_depth: Some(100.0),
            temperature: None,
            salinity: None,
            u_current: Some(0.1),
            v_current: Some(-0.2),
        }
    }

    #[test]
    fn nearest_cell_picks_closest_wet_cell() {
        // Two cells; the far one is closer in index but the near one is closer
        // geographically to the target.
        let data = NorKystGridData {
            rows: vec![
                row(0.0, 10, 10, 60.30, 5.10, Some(0.1)),
                row(3600.0, 10, 10, 60.30, 5.10, Some(0.2)),
                row(0.0, 20, 20, 60.39, 5.32, Some(0.5)),
                row(3600.0, 20, 20, 60.39, 5.32, Some(0.6)),
            ],
        };
        assert_eq!(data.nearest_cell(5.32, 60.39), Some((20, 20)));

        let ts = data.sea_surface_height_series_nearest(5.32, 60.39);
        assert_eq!(ts.len(), 2);
        assert!((ts.values()[0] - 0.5).abs() < TOL);
        assert!((ts.values()[1] - 0.6).abs() < TOL);
        assert!((ts.duration() - 3600.0).abs() < TOL);
    }

    #[test]
    fn nearest_cell_ignores_dry_cells() {
        // A geographically-closer cell with no ssh must be skipped.
        let data = NorKystGridData {
            rows: vec![
                row(0.0, 5, 5, 60.39, 5.32, None), // dry, closest
                row(0.0, 6, 6, 60.50, 5.50, Some(0.3)),
            ],
        };
        assert_eq!(data.nearest_cell(5.32, 60.39), Some((6, 6)));
    }

    #[test]
    fn empty_when_no_wet_cell() {
        let data = NorKystGridData {
            rows: vec![row(0.0, 1, 1, 60.0, 5.0, None)],
        };
        assert!(data.nearest_cell(5.0, 60.0).is_none());
        assert!(data.sea_surface_height_series_nearest(5.0, 60.0).is_empty());
    }

    #[test]
    fn round_trip_through_a_real_parquet() {
        // Build a 2-cell × 2-time grid parquet with the real column layout and
        // read it back, exercising the arrow decode path.
        let schema = Arc::new(Schema::new(vec![
            Field::new("time", DataType::Utf8, false),
            Field::new("grid_x", DataType::Int32, false),
            Field::new("grid_y", DataType::Int32, false),
            Field::new("latitude", DataType::Float64, false),
            Field::new("longitude", DataType::Float64, false),
            Field::new("sea_surface_height", DataType::Float64, true),
            Field::new("bottom_depth", DataType::Float64, true),
            Field::new("u_current", DataType::Float64, true),
            Field::new("v_current", DataType::Float64, true),
        ]));
        let times = StringArray::from(vec![
            "2024-06-01T00:00:00+00:00",
            "2024-06-01T00:00:00+00:00",
            "2024-06-01T01:00:00+00:00",
            "2024-06-01T01:00:00+00:00",
        ]);
        let gx = Int32Array::from(vec![10, 20, 10, 20]);
        let gy = Int32Array::from(vec![10, 20, 10, 20]);
        let lat = Float64Array::from(vec![60.30, 60.39, 60.30, 60.39]);
        let lon = Float64Array::from(vec![5.10, 5.32, 5.10, 5.32]);
        let ssh = Float64Array::from(vec![Some(0.1), Some(0.5), Some(0.2), Some(0.6)]);
        let h = Float64Array::from(vec![Some(90.0), Some(120.0), Some(90.0), Some(120.0)]);
        let u = Float64Array::from(vec![Some(0.01), Some(0.02), Some(0.03), Some(0.04)]);
        let v = Float64Array::from(vec![Some(-0.01), Some(-0.02), Some(-0.03), Some(-0.04)]);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(times),
                Arc::new(gx),
                Arc::new(gy),
                Arc::new(lat),
                Arc::new(lon),
                Arc::new(ssh),
                Arc::new(h),
                Arc::new(u),
                Arc::new(v),
            ],
        )
        .unwrap();

        let path = std::env::temp_dir().join(format!(
            "dg_norkyst_parquet_test_{}.parquet",
            std::process::id()
        ));
        {
            let file = std::fs::File::create(&path).unwrap();
            let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        let data = read_norkyst_parquet(&path).unwrap();
        assert_eq!(data.len(), 4);

        // Nearest cell to Bergen is (20,20); zeta series 0.5 then 0.6, 1 h apart.
        let ts = data.sea_surface_height_series_nearest(5.32, 60.39);
        assert_eq!(ts.len(), 2);
        assert!((ts.values()[0] - 0.5).abs() < TOL);
        assert!((ts.duration() - 3600.0).abs() < TOL);

        let (u, v) = data.surface_current_series_nearest(5.32, 60.39);
        assert_eq!(u.len(), 2);
        assert!((u.values()[1] - 0.04).abs() < TOL);
        assert!((v.values()[0] - (-0.02)).abs() < TOL);

        let _ = std::fs::remove_file(&path);
    }
}
