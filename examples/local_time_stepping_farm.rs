//! Local time stepping on a farm-refined mesh (TODO P2.5).
//!
//! The fjord arm of `docs/gmsh-meshes.md`: 12 × 6 km, open to the sea in the
//! west, quads refined from ≈ 450 m to ≈ 20 m at a fish farm in the middle
//! (`tests/data/gmsh/fjord_farm.msh`, from `scripts/gmsh_farm_mesh.py`). The
//! bed deepens from 30 m at the head to 150 m at the mouth. An M2 tide of
//! 0.8 m enters through a characteristic open boundary; Coriolis, Manning
//! friction and two net cages act on the flow.
//!
//! The tide ramps up from rest over `ramp` seconds (default one hour). Switched
//! on at full amplitude (`ramp=0`), it sends a ≈ 0.13 m/s surge past the farm,
//! whose drag wake then outlasts the few mm/s of tidal current there: the model
//! has no horizontal viscosity on this path (TODO F.1). The step is set by
//! the gravity wave on the 20 m elements, so the ramp does not change the cost.
//!
//! The same run is taken twice from rest, with global SSP-RK3 and with
//! `MultirateSSPRK3`. It reports the wall time, the RHS work and the
//! difference between the two solutions: η and speed at the farm and over
//! the domain.
//!
//! ```bash
//! cargo run --release --no-default-features --features parallel,simd \
//!     --example local_time_stepping_farm -- [hours=1] [order=2] [levels=8] [ramp=3600]
//! ```

use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use dg_rs::boundary::{CharacteristicOBC, HarmonicTide, MultiBoundaryCondition2D, Reflective2D};
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::{Bathymetry2D, read_gmsh_mesh};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::PhysicsBuilder;
use dg_rs::simulation::{Simulation, SimulationResult};
use dg_rs::solver::{SWESolution2D, SWEState2D, StandardLimiter2D, WetDryConfig};
use dg_rs::source::{CageDrag2D, CoriolisSource2D, ManningFriction2D, NetCage};
use dg_rs::time::{MultirateSSPRK3, SSPRK3};
use dg_rs::types::ElementIndex;

const G: f64 = 9.81;
const LX: f64 = 12_000.0;
const FARM: [f64; 2] = [6_000.0, 3_000.0];

fn bed(x: f64, y: f64) -> f64 {
    -(30.0 + 120.0 * (1.0 - x / LX)) - 10.0 * (PI * y / 6_000.0).sin()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: HashMap<String, String> = std::env::args()
        .skip(1)
        .filter_map(|a| a.split_once('=').map(|(k, v)| (k.into(), v.into())))
        .collect();
    let get = |key: &str, default: f64| args.get(key).map_or(Ok(default), |v| v.parse());
    let hours: f64 = get("hours", 1.0)?;
    let order = get("order", 2.0)? as usize;
    let levels = get("levels", 8.0)? as usize;
    let ramp: f64 = get("ramp", 3600.0)?;

    let mesh = Arc::new(read_gmsh_mesh(Path::new("tests/data/gmsh/fjord_farm.msh"))?);
    let ops = Arc::new(DGOperators2D::new(order));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, bed));
    let sizes: Vec<f64> = ElementIndex::iter(mesh.n_elements)
        .map(|k| geom.area[k.as_usize()].sqrt())
        .collect();
    println!(
        "Mesh: {} quads (P{order}), element size {:.0}–{:.0} m",
        mesh.n_elements,
        sizes.iter().copied().fold(f64::INFINITY, f64::min),
        sizes.iter().copied().fold(0.0, f64::max)
    );

    let wall = Reflective2D::new();
    let tide = HarmonicTide::m2(0.8, 0.0);
    let sea = CharacteristicOBC::new(if ramp > 0.0 { tide.with_ramp_up(ramp) } else { tide });
    let cages = [
        NetCage::circular([FARM[0] - 40.0, FARM[1]], 25.0, 20.0, 0.25),
        NetCage::circular([FARM[0] + 40.0, FARM[1]], 25.0, 20.0, 0.25),
    ];
    let physics = || {
        PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            MultiBoundaryCondition2D::new(&wall).with_open(&sea),
        )
        .with_bathymetry(bathymetry.clone())
        .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
        .with_wet_dry(WetDryConfig::default())
        .with_implicit_friction(ManningFriction2D::new(G, 0.025))
        .with_cage_drag(CageDrag2D::new(&mesh, &ops, &cages))
        .with_source(CoriolisSource2D::f_plane(1.2e-4))
        .build()
    };
    let at_rest = || {
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let h = -bathymetry.get(k, i);
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }
        q
    };
    let t_end = hours * 3600.0;

    let run = |label: &str, q: &mut SWESolution2D, local: bool| -> SimulationResult {
        let start = Instant::now();
        let result = if local {
            Simulation::new(physics(), MultirateSSPRK3::new(levels))
                .with_cfl(1.0)
                .run(q, 0.0, t_end)
        } else {
            Simulation::new(physics(), SSPRK3)
                .with_cfl(1.0)
                .run(q, 0.0, t_end)
        };
        let wall = start.elapsed().as_secs_f64();
        println!(
            "{label:>9}: {:6} steps, dt {:.3}–{:.3} s, {wall:6.1} s wall{}",
            result.n_steps,
            result.dt_min,
            result.dt_max,
            result
                .local_time_stepping
                .map_or(String::new(), |s| format!(
                    ", levels 0–{}, {:.2}x less RHS work",
                    s.finest_level,
                    s.speedup()
                ))
        );
        assert!(result.success, "{label}: {:?}", result.error);
        result
    };

    println!("\nM2 from rest, {hours} h:");
    let mut global = at_rest();
    run("global", &mut global, false);
    let mut local = at_rest();
    run("local", &mut local, true);

    // Differences near the farm (within 500 m) and everywhere
    let mut near = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut all = (0.0_f64, 0.0_f64, 0.0_f64);
    for k in ElementIndex::iter(mesh.n_elements) {
        for i in 0..ops.n_nodes {
            let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
            let (a, b) = (global.get_state(k, i), local.get_state(k, i));
            let speed = |s: SWEState2D| (s.hu * s.hu + s.hv * s.hv).sqrt() / s.h;
            let (d_eta, d_speed) = ((a.h - b.h).abs(), (speed(a) - speed(b)).abs());
            let near_farm = (x - FARM[0]).hypot(y - FARM[1]) < 500.0;
            for d in [&mut all].into_iter().chain(near_farm.then_some(&mut near)) {
                d.0 = d.0.max(d_eta);
                d.1 = d.1.max(d_speed);
                d.2 = d.2.max(speed(a));
            }
        }
    }
    println!(
        "\nmax |local − global|: near the farm η {:.2e} m, speed {:.2e} m/s (max speed {:.2} m/s)",
        near.0, near.1, near.2
    );
    println!(
        "                      everywhere    η {:.2e} m, speed {:.2e} m/s (max speed {:.2} m/s)",
        all.0, all.1, all.2
    );
    Ok(())
}
