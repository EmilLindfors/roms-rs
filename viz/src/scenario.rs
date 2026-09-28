//! What is simulated: the mesh, the bed, the cages and the forcing.
//!
//! [`Scenario::fjord_farm`] is the fjord arm of `examples/local_time_stepping_farm.rs`
//! (`docs/gmsh-meshes.md`): 12 × 6 km, open to the sea in the west, quads refined from
//! ≈ 450 m to ≈ 20 m at a fish farm in the middle. The bed deepens from 30 m at the head
//! to 150 m at the mouth; an M2 tide enters through a characteristic open boundary, and
//! Coriolis, Manning friction and two net cages act on the flow.

use std::error::Error;
use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;

use dg_rs::mesh::{Bathymetry2D, Mesh2D, read_gmsh_mesh};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::solver::{SWESolution2D, SWEState2D};
use dg_rs::source::NetCage;
use dg_rs::types::ElementIndex;

pub const G: f64 = 9.81;

/// Tidal forcing and friction of a scenario.
#[derive(Clone, Copy, Debug)]
pub struct Forcing {
    /// M2 amplitude at the open boundary (m)
    pub m2_amplitude: f64,
    /// Manning coefficient (s/m^(1/3))
    pub manning: f64,
    /// Coriolis parameter (1/s)
    pub coriolis: f64,
    /// Time over which the tide ramps up from rest (s); 0 switches it on at once.
    /// Switched on at once, the tide sends a start-up surge through the fjord
    /// (≈ 0.13 m/s at the farm) whose drag wake outlasts the tidal current there.
    pub ramp: f64,
}

pub struct Scenario {
    pub name: String,
    pub mesh: Arc<Mesh2D>,
    pub ops: Arc<DGOperators2D>,
    pub geom: Arc<GeometricFactors2D>,
    pub bathymetry: Arc<Bathymetry2D>,
    pub cages: Vec<NetCage>,
    /// Centre of the farm, where the camera starts (m, mesh coordinates)
    pub farm: [f64; 2],
    pub forcing: Forcing,
}

impl Scenario {
    pub fn fjord_farm(mesh_path: &Path, order: usize) -> Result<Self, Box<dyn Error>> {
        const LX: f64 = 12_000.0;
        const FARM: [f64; 2] = [6_000.0, 3_000.0];
        let bed = |x: f64, y: f64| -(30.0 + 120.0 * (1.0 - x / LX)) - 10.0 * (PI * y / 6_000.0).sin();

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
            forcing: Forcing {
                m2_amplitude: 0.8,
                manning: 0.025,
                coriolis: 1.2e-4,
                ramp: 3600.0,
            },
            mesh,
            ops,
            geom,
            bathymetry,
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
