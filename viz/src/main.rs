//! dg-viz: a 3D view of a dg-rs simulation while it runs.
//!
//! The solver runs on its own thread ([`solver`]) and sends a snapshot of η and the
//! depth-averaged current every `--interval` seconds of model time; the viewer plays
//! them back ([`playback`]) as the water surface over the bed ([`surface`]), coloured by
//! current speed or elevation, with depth contours on the bed ([`contours`]), current
//! arrows ([`arrows`]), the farm's net cages riding the surface ([`cages`]) and
//! particles released from the cages, tracked with the flow ([`particles`]).
//!
//! Scenarios ([`scenario`]): the farm fjord of `examples/local_time_stepping_farm.rs`,
//! or Frøya–Smøla–Hitra on the coastline mesh with NorKyst-800 boundary tides, as
//! `examples/froya_real_data.rs` runs it (its data files in `../data/`).
//!
//! ```bash
//! cd viz && cargo run --release -- [options]
//! ```
//!
//! Options (defaults in brackets):
//! - `--scenario fjord|froya` [fjord]. Frøya reads `../data/froya_coast.msh` (from
//!   `scripts/gmsh_coastline_mesh.py`), `../data/froya_topobathy.tif` and
//!   `../data/froya_boundary_tides.txt`; the close-up (F) is the Mausund tide gauge,
//!   and it has no cages or particles. Its run is ≈ 30× faster than real time on 12
//!   threads, so play it back at `--rate 30` or below to keep up with the solver.
//! - `--mesh PATH` Gmsh mesh [`../tests/data/gmsh/fjord_farm.msh`, or Frøya's]
//! - `--order N` polynomial order [2]; `--levels N` local time stepping levels, 0 for
//!   global SSP-RK3 [8]
//! - `--hours H` model hours to run [25]; `--interval S` model seconds between
//!   snapshots [60]; `--threads N` solver threads [all but two cores]
//! - `--memory MB` snapshots kept, oldest dropped first [2048]
//! - `--rate R` model seconds shown per second [60]
//! - `--vz Z` vertical exaggeration [3]; `--speed-max V` fix the speed scale (m/s)
//!   instead of letting it follow the flow in round steps
//! - `--water-alpha A` opacity of the translucent water, 0.05–0.95 [0.4]; lower shows
//!   more of the bed (its depth scale is in the legend), higher more of the colouring
//! - `--ramp S` seconds over which the tide ramps up from rest; 0 switches it on at
//!   once, and the start-up surge then leaves a drag wake at the farm that outlasts
//!   the tidal current there [3600]
//! - `--drag on|off` whether the nets drag on the flow (drawn either way) [on]
//! - `--particles N` particles released in each cage every `--release S` model seconds
//!   [20, 60]; 0 for none. `--kh K` horizontal diffusivity of their random walk (m²/s)
//!   [0.1]
//! - `--view farm|domain` starting framing: the close-up (the farm, or the gauge) or
//!   the whole domain [farm]
//! - `--screenshot PATH` save a frame once the shown time reaches `--at S` (model
//!   seconds; default the end of the run) and quit: for checking a change headlessly.
//!
//! Keys: see [`hud`] (B: bed contours; - and =: water opacity).

mod arrows;
mod cages;
mod camera;
mod colormap;
mod contours;
mod field;
mod hud;
mod particles;
mod playback;
mod scenario;
mod solver;
mod surface;

use std::path::PathBuf;
use std::sync::Mutex;

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use dg_rs::mesh::PointLocator2D;

use arrows::{Arrows, ArrowsPlugin};
use cages::{CageLayout, CagesPlugin};
use camera::{CameraPlugin, OrbitCamera, Views};
use contours::ContoursPlugin;
use field::{Field, Frame, Nodes, Probe};
use hud::{HudPlugin, Title};
use particles::{ParticleConfig, ParticlesPlugin};
use playback::{Playback, PlaybackPlugin, SolverChannel, SolverState};
use scenario::Scenario;
use solver::SolverConfig;
use surface::{Colouring, SurfacePlugin, WaterOpacity};

struct Args {
    scenario: String,
    mesh: Option<PathBuf>,
    order: usize,
    levels: usize,
    hours: f64,
    interval: f64,
    threads: usize,
    memory_mb: usize,
    rate: f64,
    vz: f32,
    speed_max: Option<f32>,
    water_alpha: f32,
    drag: bool,
    ramp: Option<f64>,
    particles: ParticleConfig,
    view: String,
    screenshot: Option<String>,
    at: Option<f64>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            scenario: "fjord".into(),
            mesh: None,
            order: 2,
            levels: 8,
            hours: 25.0,
            interval: 60.0,
            threads: std::thread::available_parallelism()
                .map_or(1, |n| n.get().saturating_sub(2).max(1)),
            memory_mb: 2048,
            rate: 60.0,
            vz: 3.0,
            speed_max: None,
            water_alpha: WaterOpacity::default().0,
            drag: true,
            ramp: None,
            particles: ParticleConfig {
                per_release: 20,
                release_every: 60.0,
                kh: 0.1,
            },
            view: "farm".into(),
            screenshot: None,
            at: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
            match flag.as_str() {
                "--scenario" => args.scenario = value,
                "--mesh" => args.mesh = Some(value.into()),
                "--order" => args.order = num(&flag, &value)?,
                "--levels" => args.levels = num(&flag, &value)?,
                "--hours" => args.hours = num(&flag, &value)?,
                "--interval" => args.interval = num(&flag, &value)?,
                "--threads" => args.threads = num(&flag, &value)?,
                "--memory" => args.memory_mb = num(&flag, &value)?,
                "--rate" => args.rate = num(&flag, &value)?,
                "--vz" => args.vz = num(&flag, &value)?,
                "--speed-max" => args.speed_max = Some(num(&flag, &value)?),
                "--water-alpha" => {
                    args.water_alpha = num(&flag, &value)?;
                    if !(0.0..=1.0).contains(&args.water_alpha) {
                        return Err(format!("--water-alpha takes 0 to 1, not {value}"));
                    }
                }
                "--ramp" => args.ramp = Some(num(&flag, &value)?),
                "--drag" => {
                    args.drag = match value.as_str() {
                        "on" => true,
                        "off" => false,
                        _ => return Err(format!("--drag takes on or off, not {value}")),
                    }
                }
                "--particles" => args.particles.per_release = num(&flag, &value)?,
                "--release" => args.particles.release_every = num(&flag, &value)?,
                "--kh" => args.particles.kh = num(&flag, &value)?,
                "--view" => args.view = value,
                "--screenshot" => args.screenshot = Some(value),
                "--at" => args.at = Some(num(&flag, &value)?),
                _ => {
                    return Err(format!(
                        "unknown option {flag} (see the docs at the top of viz/src/main.rs)"
                    ));
                }
            }
        }
        Ok(args)
    }
}

fn num<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("bad value for {flag}: {value}"))
}

/// `--screenshot`: the frame to save, and how many frames since the shown time reached it.
#[derive(Resource)]
struct Capture {
    path: Option<String>,
    at: f64,
    frames: Option<u32>,
}

fn main() -> AppExit {
    let args = match Args::parse() {
        Ok(args) => args,
        Err(e) => {
            eprintln!("{e}");
            return AppExit::from_code(2);
        }
    };
    let repo = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/.."));
    let built = match args.scenario.as_str() {
        "fjord" => {
            let mesh = args
                .mesh
                .clone()
                .unwrap_or_else(|| repo.join("tests/data/gmsh/fjord_farm.msh"));
            Scenario::fjord_farm(&mesh, args.order)
        }
        "froya" => {
            let data = repo.join("data");
            let mesh = args
                .mesh
                .clone()
                .unwrap_or_else(|| data.join("froya_coast.msh"));
            println!(
                "Building Frøya from {} (the bed projection takes a while)...",
                data.display()
            );
            Scenario::froya(
                &mesh,
                &data.join("froya_topobathy.tif"),
                &data.join("froya_boundary_tides.txt"),
                args.order,
                args.hours * 3600.0,
            )
        }
        other => {
            eprintln!("unknown scenario {other}: fjord or froya");
            return AppExit::from_code(2);
        }
    };
    let mut scenario = match built {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot build the {} scenario: {e}", args.scenario);
            return AppExit::from_code(1);
        }
    };
    if let Some(ramp) = args.ramp {
        scenario.forcing.ramp = ramp;
    }

    let (lo, hi) = scenario.mesh.vertices.iter().fold(
        ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
        |(lo, hi), v| {
            (
                [lo[0].min(v[0]), lo[1].min(v[1])],
                [hi[0].max(v[0]), hi[1].max(v[1])],
            )
        },
    );
    let frame = Frame {
        origin: [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])],
        vz: args.vz,
    };
    let nodes = Nodes::new(&scenario, &frame);
    let locator = PointLocator2D::new(&scenario.mesh);
    let cages = CageLayout(
        scenario
            .cages
            .iter()
            .map(|cage| {
                let (c0, c1) = cage.footprint.bounding_box();
                let centre = [0.5 * (c0[0] + c1[0]), 0.5 * (c0[1] + c1[1])];
                (cage.clone(), Probe::at(&locator, &scenario, centre))
            })
            .collect(),
    );
    let arrows = Arrows::new(
        &locator,
        &scenario,
        48,
        scenario.close_up.arrow_spacing,
        scenario.close_up.arrow_radius,
    );
    println!(
        "{}: {} elements (P{}), {} nodes; {} + {} current arrows",
        scenario.name,
        scenario.mesh.n_elements,
        args.order,
        nodes.len(),
        arrows.count().0,
        arrows.count().1
    );

    let extent = (hi[0] - lo[0]).max(hi[1] - lo[1]) as f32;
    let views = Views {
        farm: OrbitCamera {
            focus: frame.world(scenario.farm, 0.0),
            yaw: 0.7,
            pitch: 0.5,
            distance: scenario.close_up.distance,
        },
        domain: OrbitCamera {
            focus: Vec3::ZERO,
            yaw: 0.35,
            pitch: 0.7,
            distance: 0.85 * extent,
        },
    };
    let start = if args.view == "domain" {
        views.domain
    } else {
        views.farm
    };

    let t_end = args.hours * 3600.0;
    let channel = solver::spawn(
        &scenario,
        SolverConfig {
            t_end,
            interval: args.interval,
            levels: args.levels,
            threads: args.threads,
            drag: args.drag,
            particles: args.particles,
        },
    );

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "dg-viz".into(),
                resolution: (1600, 900).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.05, 0.065, 0.08)))
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.75, 0.85, 1.0),
            brightness: 350.0,
            ..default()
        })
        .insert_resource(frame)
        .insert_resource(nodes)
        .insert_resource(cages)
        .insert_resource(arrows)
        .insert_resource(views)
        .insert_resource(Field::default())
        .insert_resource(Colouring::new(args.speed_max))
        .insert_resource(WaterOpacity(args.water_alpha))
        .insert_resource(Title(scenario.name.clone()))
        .insert_resource(Playback::new(
            args.rate,
            args.interval,
            args.memory_mb << 20,
        ))
        .insert_resource(SolverChannel(Mutex::new(channel)))
        .insert_resource(Capture {
            path: args.screenshot,
            at: args.at.unwrap_or(t_end),
            frames: None,
        })
        .add_plugins((
            PlaybackPlugin,
            SurfacePlugin,
            ContoursPlugin,
            CagesPlugin,
            ArrowsPlugin,
            ParticlesPlugin,
            CameraPlugin,
            HudPlugin,
        ))
        .add_systems(Startup, move |mut commands: Commands| {
            commands.spawn((
                Camera3d::default(),
                Projection::Perspective(PerspectiveProjection {
                    near: 1.0,
                    far: 400_000.0,
                    ..default()
                }),
                start.transform(),
                start,
            ));
            // Sun from the south-west, 35° up.
            commands.spawn((
                DirectionalLight {
                    illuminance: 9_000.0,
                    shadow_maps_enabled: false,
                    ..default()
                },
                Transform::from_xyz(-1.0, 1.0, 1.0).looking_at(Vec3::ZERO, Vec3::Y),
            ));
        })
        .add_systems(Update, field::interpolate)
        .add_systems(Update, capture.after(field::interpolate))
        .add_systems(Last, playback::settle)
        .run()
}

fn capture(
    mut commands: Commands,
    mut capture: ResMut<Capture>,
    mut playback: ResMut<Playback>,
    field: Res<Field>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(path) = capture.path.clone() else {
        return;
    };
    match capture.frames.as_mut() {
        None => {
            let ended = !matches!(playback.solver, SolverState::Running)
                && playback.newest().is_some_and(|t| playback.t >= t);
            if field.t.is_some_and(|t| t >= capture.at) || (field.t.is_some() && ended) {
                playback.paused = true;
                capture.frames = Some(0);
            }
        }
        Some(n) => {
            *n += 1;
            // A few frames for the meshes to reach the GPU, then some for the file to be written.
            if *n == 10 {
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(save_to_disk(path));
            } else if *n == 40 {
                exit.write(AppExit::Success);
            }
        }
    }
}
