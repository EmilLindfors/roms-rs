//! What is simulated: the mesh, the bed, the cages and the forcing.
//!
//! - [`Scenario::fjord_farm`] is the fjord arm of `examples/local_time_stepping_farm.rs`
//!   (`docs/gmsh-meshes.md`): 12 × 6 km, open to the sea in the west, quads refined
//!   from ≈ 450 m to ≈ 20 m at a fish farm in the middle. The bed deepens from 30 m at
//!   the head to 150 m at the mouth; an M2 tide enters through a characteristic open
//!   boundary, and Coriolis, Manning friction and two net cages act on the flow.
//! - [`Scenario::froya`] is Frøya–Smøla–Hitra as `examples/froya_real_data.rs` runs it
//!   on the coastline-fitted mesh (`mesh=`): the bed from Kartverket's topobathy model,
//!   projected onto the nodes, and NorKyst-800 boundary tides from the tidal atlas
//!   (`BoundaryTides`), on a clock starting 2025-06-15. The data files are those of the
//!   example (untracked, in `data/`; see `TODO.md` P1.6 and P3.1 for how to fetch them).
//! - [`Scenario::farm_channel`] is the stratified tidal channel of `examples/farm_3d.rs`,
//!   run by the 3D model ([`ThreeD`]): 6 km long (periodic along it), 600 m wide and
//!   40 m deep, 2 °C warmer over the top 10 m, an M2 current of ≈ 0.5 m/s driven by a
//!   depth-uniform body force, two 50 m cages with 20 m nets across the flow, GLS k-ε
//!   mixing and a log-layer bottom drag.

use std::error::Error;
use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;

use dg_rs::boundary::{BoundaryTides, TidalAtlas};
use dg_rs::io::{
    BedRaster, CoordinateProjection, GeoBoundingBox, GeoTiffBathymetry, LocalProjection,
};
use dg_rs::mesh::{Bathymetry2D, BoundaryTag, Mesh2D, read_gmsh_mesh};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::particles::ClearSkyLight;
use dg_rs::solver::{SWESolution2D, SWEState2D};
use dg_rs::source::NetCage;
use dg_rs::time::ModelClock;
use dg_rs::types::ElementIndex;
use dg_rs::vertical::{SigmaGrid, UniformStretching};

pub const G: f64 = 9.81;

/// The tide at the open boundary.
#[derive(Clone, Debug)]
pub enum Tide {
    /// M2 of this amplitude (m), in one phase along the whole boundary
    UniformM2(f64),
    /// Tides of a boundary atlas, varying along the boundary
    Atlas(BoundaryTides),
    /// A depth-uniform M2 body force along mesh x that drives a current of this
    /// amplitude (m/s) without friction: the tidal surface slope of a periodic channel
    BodyForceM2(f64),
}

/// Tidal forcing and friction of a scenario.
#[derive(Clone, Debug)]
pub struct Forcing {
    pub tide: Tide,
    /// Manning coefficient (s/m^(1/3))
    pub manning: f64,
    /// Coriolis parameter (1/s)
    pub coriolis: f64,
    /// Time over which the tide ramps up from rest (s); 0 switches it on at once.
    /// Switched on at once, the tide sends a start-up surge through the fjord
    /// (≈ 0.13 m/s at the farm) whose drag wake outlasts the tidal current there.
    pub ramp: f64,
}

/// The close-up view (F): its camera and the fine grid of current arrows around the
/// point of interest.
#[derive(Clone, Copy, Debug)]
pub struct CloseUp {
    /// Camera distance (world units)
    pub distance: f32,
    /// Camera direction: angle about the vertical from world +Z, and above the
    /// horizontal (rad)
    pub yaw: f32,
    pub pitch: f32,
    /// Spacing of the fine arrows (m)
    pub arrow_spacing: f64,
    /// Radius of the fine arrow grid (m)
    pub arrow_radius: f64,
}

/// The 3D model of a scenario (`Hydrostatic3D` with mode splitting).
#[derive(Clone, Debug)]
pub struct ThreeD {
    pub sigma: Arc<SigmaGrid>,
    /// Temperature (°C) at rest at height z (m, negative below the surface)
    pub temperature: fn(f64) -> f64,
    /// Salinity at rest
    pub salinity: f64,
    /// Bed roughness z₀ of the log-layer drag (m)
    pub roughness: f64,
    /// Horizontal viscosity: constant background (m²/s) and Smagorinsky coefficient
    pub viscosity: f64,
    pub smagorinsky: f64,
    /// The vertical section drawn through the water: its two ends (m, mesh coordinates)
    pub section: [[f64; 2]; 2],
    /// Surface light over the site, for the lice larvae
    pub light: ClearSkyLight,
}

pub struct Scenario {
    pub name: String,
    pub mesh: Arc<Mesh2D>,
    pub ops: Arc<DGOperators2D>,
    pub geom: Arc<GeometricFactors2D>,
    pub bathymetry: Arc<Bathymetry2D>,
    pub cages: Vec<NetCage>,
    /// Point of interest, where the close-up view looks: the farm, or a tide gauge
    /// (m, mesh coordinates)
    pub farm: [f64; 2],
    pub close_up: CloseUp,
    pub forcing: Forcing,
    /// The 3D model, for a scenario run in 3D
    pub three_d: Option<ThreeD>,
    /// The mesh's periods along x and y (m), for a periodic mesh
    pub periodic: Option<[f64; 2]>,
}

impl Scenario {
    pub fn fjord_farm(mesh_path: &Path, order: usize) -> Result<Self, Box<dyn Error>> {
        const LX: f64 = 12_000.0;
        const FARM: [f64; 2] = [6_000.0, 3_000.0];
        let bed =
            |x: f64, y: f64| -(30.0 + 120.0 * (1.0 - x / LX)) - 10.0 * (PI * y / 6_000.0).sin();

        let mesh = Arc::new(read_gmsh_mesh(mesh_path)?);
        let ops = Arc::new(DGOperators2D::new(order));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, bed));
        Ok(Self {
            name: "Fjord farm: M2 0.8 m, two 50 m cages".into(),
            cages: vec![
                NetCage::circular([FARM[0] - 40.0, FARM[1]], 25.0, 20.0, 0.25),
                NetCage::circular([FARM[0] + 40.0, FARM[1]], 25.0, 20.0, 0.25),
            ],
            farm: FARM,
            close_up: CloseUp {
                distance: 420.0,
                yaw: 0.7,
                pitch: 0.5,
                arrow_spacing: 25.0,
                arrow_radius: 500.0,
            },
            forcing: Forcing {
                tide: Tide::UniformM2(0.8),
                manning: 0.025,
                coriolis: 1.2e-4,
                ramp: 3600.0,
            },
            three_d: None,
            periodic: None,
            mesh,
            ops,
            geom,
            bathymetry,
        })
    }

    /// The stratified tidal channel of `examples/farm_3d.rs` (see the module docs) on
    /// `levels` σ-levels, with the larvae's light at Mausund on `clock`.
    pub fn farm_channel(order: usize, levels: usize, clock: ModelClock) -> Self {
        const LX: f64 = 6_000.0;
        const LY: f64 = 600.0;
        const DX: f64 = 60.0;
        const DEPTH: f64 = 40.0;
        const FARM: [f64; 2] = [3_000.0, 300.0];
        /// Mausund, off Frøya (longitude, latitude): where the sun is
        const SITE: [f64; 2] = [8.67, 63.87];

        let (nx, ny) = ((LX / DX).round() as usize, (LY / DX).round() as usize);
        let mesh = Arc::new(Mesh2D::channel_periodic_x(0.0, LX, 0.0, LY, nx, ny));
        let ops = Arc::new(DGOperators2D::new(order));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -DEPTH));
        let cages = [-60.0, 60.0]
            .map(|dy| NetCage::circular([FARM[0], FARM[1] + dy], 25.0, 20.0, 0.25))
            .to_vec();
        Self {
            name: "Farm channel, 3D: M2 0.5 m/s, 2 C thermocline at 10 m, two 50 m cages".into(),
            // The section runs along the flow through the first cage
            three_d: Some(ThreeD {
                sigma: Arc::new(SigmaGrid::new(levels, UniformStretching)),
                // 10 °C below, 12 °C above, a smooth step at 10 m
                temperature: |z| 11.0 + (0.5 * (z + 10.0)).tanh(),
                salinity: 35.0,
                roughness: 0.005,
                viscosity: 1.0,
                smagorinsky: 0.2,
                section: [
                    [FARM[0] - 600.0, FARM[1] - 60.0],
                    [FARM[0] + 600.0, FARM[1] - 60.0],
                ],
                light: ClearSkyLight::new(clock, SITE[0], SITE[1]),
            }),
            cages,
            // Periodic along the channel only
            periodic: Some([LX, f64::INFINITY]),
            farm: FARM,
            // Low, from the second cage's side, so the section stands behind the cages
            close_up: CloseUp {
                distance: 800.0,
                yaw: 2.6,
                pitch: 0.28,
                arrow_spacing: 25.0,
                arrow_radius: 600.0,
            },
            forcing: Forcing {
                tide: Tide::BodyForceM2(0.5),
                manning: 0.0,
                coriolis: 1.2e-4,
                ramp: 3600.0,
            },
            mesh,
            ops,
            geom,
            bathymetry,
        }
    }

    /// Frøya–Smøla–Hitra on the coastline mesh `mesh_path`, with the elevation model
    /// `dem_path` and the boundary tidal atlas `atlas_path` (see the module docs),
    /// run for `duration` seconds (for the tides' nodal corrections).
    pub fn froya(
        mesh_path: &Path,
        dem_path: &Path,
        atlas_path: &Path,
        order: usize,
        duration: f64,
    ) -> Result<Self, Box<dyn Error>> {
        /// Land above this height (m) is never wet: capped, so that shoreline cliffs
        /// do not drive films up the land
        const LAND_ELEVATION: f64 = 5.0;
        /// Open faces farther than this (m) from an atlas point become walls
        const ATLAS_COVERAGE: f64 = 5000.0;
        /// Mausund tide gauge (Kartverket MSU), the close-up view
        const MAUSUND: (f64, f64) = (63.869331, 8.665231);

        let bbox = GeoBoundingBox::new(8.0, 63.6, 9.2, 64.0);
        let (lat0, lon0) = bbox.center();
        let projection = LocalProjection::new(lat0, lon0);

        // Kartverket's topobathy model: land heights and depths in one grid; its
        // missing depths are an exact 0, filled from their surroundings
        let dem = GeoTiffBathymetry::load(dem_path)?;
        let mut raster = BedRaster::elevation_model(&dem, &bbox)?;
        raster.fill_holes(|b| b == 0.0);
        let raster = raster.clamp_land(LAND_ELEVATION);

        // The bed L2-projected onto the nodes; water one node wide is unresolved
        // and becomes shore; the elements without water are dropped
        let grid = read_gmsh_mesh(mesh_path)?;
        let ops = DGOperators2D::new(order);
        let grid_geom = GeometricFactors2D::compute(&grid, &ops);
        let mut grid_bed = Bathymetry2D::project(
            &grid,
            &ops,
            &grid_geom,
            raster.sampler(&projection),
            raster.pixel_size(),
        );
        grid_bed.raise_isolated_wet_nodes(&grid, &ops, &grid_geom, 0.0);
        let has_water = |k: ElementIndex| grid_bed.element(k).iter().any(|&b| b < 0.0);
        let (mut mesh, kept) = grid.retain_elements(has_water, BoundaryTag::Wall);
        let bathymetry = grid_bed.select_elements(&kept);

        // Water the atlas's parent model does not resolve (a fjord arm crossing the
        // domain edge far from any atlas point) has no tide to force: wall it off
        let atlas = TidalAtlas::read(atlas_path)?;
        for e in 0..mesh.edges.len() {
            let edge = &mesh.edges[e];
            if edge.right.is_some() || edge.boundary_tag != Some(BoundaryTag::Open) {
                continue;
            }
            let (k, face) = (ElementIndex::new(edge.left.element), edge.left.face);
            let uncovered = ops.face_nodes[face].iter().any(|&i| {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let (lat, lon) = projection.xy_to_geo(x, y);
                atlas
                    .nearest(lon, lat)
                    .is_none_or(|(_, d)| d > ATLAS_COVERAGE)
            });
            if uncovered {
                mesh.edges[e].boundary_tag = Some(BoundaryTag::Wall);
            }
        }

        let clock = ModelClock::parse("2025-06-15T00:00:00Z")?;
        let tides = atlas.boundary_tides(
            &mesh,
            &ops,
            &projection,
            BoundaryTag::Open,
            &clock,
            duration,
            ATLAS_COVERAGE,
        )?;
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let (x, y) = projection.geo_to_xy(MAUSUND.0, MAUSUND.1);
        Ok(Self {
            // ASCII: Bevy's default font has no ø
            name: "Froya-Smola-Hitra: NorKyst-800 tides from 2025-06-15".into(),
            cages: Vec::new(),
            farm: [x, y],
            close_up: CloseUp {
                distance: 6_000.0,
                yaw: 0.7,
                pitch: 0.5,
                arrow_spacing: 250.0,
                arrow_radius: 5_000.0,
            },
            forcing: Forcing {
                tide: Tide::Atlas(tides),
                manning: 0.025,
                coriolis: 1.31e-4,
                // A spring tide squeezed into one hour overshoots (TODO P1.4)
                ramp: 3.0 * 3600.0,
            },
            three_d: None,
            periodic: None,
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
            bathymetry: Arc::new(bathymetry),
        })
    }

    /// Still water at z = 0 over the bed.
    pub fn at_rest(&self) -> SWESolution2D {
        let mut q = SWESolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
        for k in ElementIndex::iter(self.mesh.n_elements) {
            for i in 0..self.ops.n_nodes {
                let h = (-self.bathymetry.get(k, i)).max(0.0);
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }
        q
    }
}
