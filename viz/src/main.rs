//! dg-viz: a 3D view of a dg-rs simulation while it runs.
//!
//! The solver runs on its own thread ([`solver`]) and sends a snapshot of η and the
//! depth-averaged current every `--interval` seconds of model time; the viewer plays
//! them back ([`playback`]) as the water surface over the bed ([`surface`]), coloured by
//! current speed or elevation, with current arrows ([`arrows`]) and the farm's net cages
//! riding the surface ([`cages`]).
//!
//! The scenario is the farm fjord of `examples/localtime_stepping_farm.rs`
//! ([`scenario::Scenario::fjord_farm`]).
//!
//! ```bash
//! cd viz && cargo run --release -- [options]
//! ```
//!
//! Options (defaults in brackets):
//! - `--mesh PATH` Gmsh mesh [`../tests/data/gmsh/fjord_farm.msh`]
//! - `--order N` polynomial order [2]; `--levels N` local time stepping levels, 0 for
//!   global SSP-RK3 [8]
//! - `--hours H` model hours to run [25]; `--interval S` model seconds between
//!   snapshots [60]; `--threads N` solver threads [all but two cores]
//! - `--memory MB` snapshots kept, oldest dropped first [2048]
//! - `--rate R` model seconds shown per second [60]
//! - `--vz Z` vertical exaggeration [3]; `--speed-max V` fix the speed scale (m/s)
//!   instead of letting it follow the flow in round steps
//! - `--ramp S` seconds over which the tide ramps up from rest; 0 switches it on at
//!   once, and the start-up surge then leaves a drag wake at the farm that outlasts
//!   the tidal current there [3600]
//! - `--drag on|off` whether the nets drag on the flow (drawn either way) [on]
//! - `--view farm|domain` starting framing [farm]
//! - `--screenshot PATH` save a frame once the shown time reaches `--at S` (model
//!   seconds; default the end of the run) and quit: for checking a change headlessly.
//!
//! Keys: see [`hud`].

mod arrows;
mod cages;
mod camera;
mod colormap;
mod field;
mod hud;
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
use field::{Field, Frame, Nodes, Probe};
use hud::{HudPlugin, Title};
use playback::{Playback, PlaybackPlugin, SolverChannel, SolverState};
use scenario::Scenario;
use solver::SolverConfig;
use surface::{Colouring, SurfacePlugin};

struct Args {
    mesh: PathBuf,
    order: usize,
    levels: usize,
    hours: f64,
    interval: f64,
    threads: usize,
    memory_mb: usize,
    rate: f64,
    vz: f32,
    speed_max: Option<f32>,
    drag: bool,
    ramp: Option<f64>,
    view: String,
    screenshot: Option<String>,
    at: Option<f64>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            mesh: PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/data/gmsh/fjord_farm.msh")),
            order: 2,
            levels: 8,
            hours: 25.0,
            interval: 60.0,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(2).max(1)),
            memory_mb: 2048,
            rate: 60.0,
            vz: 3.0,
            speed_max: None,
            drag: true,
            ramp: None,
            view: "farm".into(),
            screenshot: None,
            at: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
            match flag.as_str() {
                "--mesh" => args.mesh = value.into(),
                "--order" => args.order = num(&flag, &value)?,
                "--levels" => args.levels = num(&flag, &value)?,
                "--hours" => args.hours = num(&flag, &value)?,
                "--interval" => args.interval = num(&flag, &value)?,
                "--threads" => args.threads = num(&flag, &value)?,
                "--memory" => args.memory_mb = num(&flag, &value)?,
                "--rate" => args.rate = num(&flag, &value)?,
                "--vz" => args.vz = num(&flag, &value)?,
                "--speed-max" => args.speed_max = Some(num(&flag, &value)?),
                "--ramp" => args.ramp = Some(num(&flag, &value)?),
                "--drag" => args.drag = match value.as_str() {
                    "on" => true,
                    "off" => false,
                    _ => return Err(format!("--drag takes on or off, not {value}")),
                },
                "--view" => args.view = value,
                "--screenshot" => args.screenshot = Some(value),
                "--at" => args.at = Some(num(&flag, &value)?),
                _ => return Err(format!("unknown option {flag} (see the docs at the top of viz/src/main.rs)")),
            }
        }
        Ok(args)
    }
}

fn num<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T, String> {
    value.parse().map_err(|_| format!("bad value for {flag}: {value}"))
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
    let mut scenario = match Scenario::fjord_farm(&args.mesh, args.order) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot build the scenario from {}: {e}", args.mesh.display());
            return AppExit::from_code(1);
        }
    };
    if let Some(ramp) = args.ramp {
        scenario.forcing.ramp = ramp;
    }

    let (lo, hi) = scenario.mesh.vertices.iter().fold(([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]), |(lo, hi), v| {
        ([lo[0].min(v[0]), lo[1].min(v[1])], [hi[0].max(v[0]), hi[1].max(v[1])])
    });
    let frame = Frame { origin: [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])], vz: args.vz };
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
    let arrows = Arrows::new(&locator, &scenario, 48, 25.0, 500.0);
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
        farm: OrbitCamera { focus: frame.world(scenario.farm, 0.0), yaw: 0.7, pitch: 0.5, distance: 420.0 },
        domain: OrbitCamera { focus: Vec3::ZERO, yaw: 0.35, pitch: 0.7, distance: 0.85 * extent },
    };
    let start = if args.view == "domain" { views.domain } else { views.farm };

    let t_end = args.hours * 3600.0;
    let channel = solver::spawn(
        &scenario,
        SolverConfig { t_end, interval: args.interval, levels: args.levels, threads: args.threads, drag: args.drag },
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
        .insert_resource(GlobalAmbientLight { color: Color::srgb(0.75, 0.85, 1.0), brightness: 350.0, ..default() })
        .insert_resource(frame)
        .insert_resource(nodes)
        .insert_resource(cages)
        .insert_resource(arrows)
        .insert_resource(views)
        .insert_resource(Field::default())
        .insert_resource(Colouring::new(args.speed_max))
        .insert_resource(Title(scenario.name.clone()))
        .insert_resource(Playback::new(args.rate, args.interval, args.memory_mb << 20))
        .insert_resource(SolverChannel(Mutex::new(channel)))
        .insert_resource(Capture { path: args.screenshot, at: args.at.unwrap_or(t_end), frames: None })
        .add_plugins((PlaybackPlugin, SurfacePlugin, CagesPlugin, ArrowsPlugin, CameraPlugin, HudPlugin))
        .add_systems(Startup, move |mut commands: Commands| {
            commands.spawn((
                Camera3d::default(),
                Projection::Perspective(PerspectiveProjection { near: 1.0, far: 400_000.0, ..default() }),
                start.transform(),
                start,
            ));
            // Sun from the south-west, 35° up.
            commands.spawn((
                DirectionalLight { illuminance: 9_000.0, shadow_maps_enabled: false, ..default() },
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
    let Some(path) = capture.path.clone() else { return };
    match capture.frames.as_mut() {
        None => {
            let ended = !matches!(playback.solver, SolverState::Running) && playback.newest().is_some_and(|t| playback.t >= t);
            if field.t.is_some_and(|t| t >= capture.at) || (field.t.is_some() && ended) {
                playback.paused = true;
                capture.frames = Some(0);
            }
        }
        Some(n) => {
            *n += 1;
            // A few frames for the meshes to reach the GPU, then some for the file to be written.
            if *n == 10 {
                commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path));
            } else if *n == 40 {
                exit.write(AppExit::Success);
            }
        }
    }
}
