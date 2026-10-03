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
//! Frøya–Smøla–Hitra on the coastline mesh with NorKyst-800 boundary tides, as
//! `examples/froya_real_data.rs` runs it (its data files in `../data/`), or the
//! stratified farm channel of `examples/farm_3d.rs` in 3D: there the water shows the
//! depth-mean current or any σ-layer's, a vertical section along the flow through a
//! cage shows the speed or temperature in the water column ([`layers`]), and lice
//! larvae, faeces and feed are tracked in 3D ([`cloud_3d`]).
//!
//! ```bash
//! cd viz && cargo run --release -- [options]
//! ```
//!
//! Options (defaults in brackets):
//! - `--scenario fjord|froya|channel` [fjord]. Frøya reads `../data/froya_coast.msh` (from
//!   `scripts/gmsh_coastline_mesh.py`), `../data/froya_topobathy.tif` and
//!   `../data/froya_boundary_tides.txt`; the close-up (F) is the Mausund tide gauge,
//!   and it has no cages or particles. Its run is ≈ 30× faster than real time on 12
//!   threads, so play it back at `--rate 30` or below to keep up with the solver.
//!   The channel runs in 3D, from rest with the tide ramped up over an hour; its
//!   particles start at once, three kinds per cage.
//! - `--replay FILE|DIR` play back a run instead of running the solver ([`replay`]),
//!   shown linear in time between its frames; the rate defaults to an hour per
//!   second, and the status shows the run's UTC date when the run has a clock.
//!   - A snapshot file (`froya_real_data snapshot_minutes=N` writes
//!     `<output>/froya.dgsnap`) carries its mesh and bed: no scenario is built, and
//!     the close-up is its first station.
//!   - A directory of VTU frames, the `froya_NNNN.vtu` of `examples/froya_real_data.rs`
//!     with `mesh=data/froya_coast.msh`, e.g. `--replay ../output/froya_15d_k1o1`. The
//!     scenario (Frøya unless `--scenario` says otherwise) must be built as the run
//!     was: the frames' nodes and bed are checked against it. Frames are read in
//!     parallel on `--threads`; the clock comes from the run's `run.log`.
//! - `--save-snapshot FILE` with `--replay`: also write the frames read to a snapshot
//!   file, a tenth of the VTU frames' size, which later replays read without the
//!   scenario's data.
//! - `--sigma N` σ-levels of a 3D run [16]; `--start TIME` the UTC instant of model
//!   time 0, for the larvae's daylight [2025-06-15T00:00:00Z]; `--lice
//!   ladim|johnsen|passive` the larvae's behaviour (`dg_rs::particles::SalmonLice`)
//!   [ladim]; `--section speed|temperature|off` what the section shows at the start
//!   [speed]; `--layer mean|surface|bed|N` the current the water shows at the start
//!   (N: σ-layer from the bed, 0) [mean]
//! - `--mesh PATH` Gmsh mesh [`../tests/data/gmsh/fjord_farm.msh`, or Frøya's]
//! - `--order N` polynomial order [2]; `--levels N` local time stepping levels, 0 for
//!   global SSP-RK3 [8]
//! - `--hours H` model hours to run [25]; `--interval S` model seconds between
//!   snapshots [60]; `--threads N` solver threads [all but two cores]
//! - `--memory MB` snapshots kept, oldest dropped first [2048]
//! - `--rate R` model seconds shown per second [60; 3600 in a replay]
//! - `--vz Z` vertical exaggeration [3]; `--speed-max V` fix the speed scale (m/s)
//!   instead of letting it follow the flow in round steps
//! - `--water-alpha A` opacity of the translucent water, 0.05–0.95 [0.4]; lower shows
//!   more of the bed (its depth scale is in the legend), higher more of the colouring
//! - `--ramp S` seconds over which the tide ramps up from rest; 0 switches it on at
//!   once, and the start-up surge then leaves a drag wake at the farm that outlasts
//!   the tidal current there [3600]
//! - `--drag on|off` whether the nets drag on the flow (drawn either way) [on]
//! - `--particles N` particles released in each cage every `--release S` model seconds
//!   [20, 60] (in 3D, of each kind); 0 for none. `--kh K` horizontal diffusivity of
//!   their random walk (m²/s) [0.1]
//! - `--view farm|domain` starting framing: the close-up (the farm, or the gauge) or
//!   the whole domain [farm]
//! - `--screenshot PATH` save a frame once the shown time reaches `--at S` (model
//!   seconds; default the end of the run) and quit: for checking a change headlessly.
//!
//! Keys: see [`hud`] (B: bed contours; - and =: water opacity; in 3D , and . the
//! layer shown, V the section).

mod arrows;
mod cages;
mod camera;
mod cloud_3d;
mod colormap;
mod contours;
mod field;
mod hud;
mod layers;
mod particles;
mod playback;
mod replay;
mod scenario;
mod solver;
mod surface;

use std::path::PathBuf;
use std::sync::Mutex;

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use dg_rs::mesh::PointLocator2D;
use dg_rs::time::ModelClock;

use arrows::{Arrows, ArrowsPlugin};
use cages::{CageLayout, CagesPlugin};
use camera::{CameraPlugin, OrbitCamera, Views};
use contours::ContoursPlugin;
use field::{Field, Frame, Nodes, Probe, ShownLayer};
use hud::{HudPlugin, RunClock, Title};
use layers::{LayersPlugin, Levels, Section, SectionShows};
use particles::{Lice, ParticleConfig, ParticlesPlugin, Periodic};
use playback::{Playback, PlaybackPlugin, SolverChannel, SolverState, Source};
use replay::Replay;
use scenario::Scenario;
use solver::SolverConfig;
use surface::{Colouring, SurfacePlugin, WaterOpacity};

struct Args {
    scenario: Option<String>,
    replay: Option<PathBuf>,
    save_snapshot: Option<PathBuf>,
    mesh: Option<PathBuf>,
    order: usize,
    levels: usize,
    hours: f64,
    interval: f64,
    threads: usize,
    memory_mb: usize,
    rate: Option<f64>,
    vz: f32,
    speed_max: Option<f32>,
    water_alpha: f32,
    drag: bool,
    ramp: Option<f64>,
    particles: ParticleConfig,
    view: String,
    screenshot: Option<String>,
    at: Option<f64>,
    sigma: usize,
    start: String,
    section: SectionShows,
    layer: String,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            scenario: None,
            replay: None,
            save_snapshot: None,
            mesh: None,
            order: 2,
            levels: 8,
            hours: 25.0,
            interval: 60.0,
            threads: std::thread::available_parallelism()
                .map_or(1, |n| n.get().saturating_sub(2).max(1)),
            memory_mb: 2048,
            rate: None,
            vz: 3.0,
            speed_max: None,
            water_alpha: WaterOpacity::default().0,
            drag: true,
            ramp: None,
            particles: ParticleConfig {
                per_release: 20,
                release_every: 60.0,
                kh: 0.1,
                lice: Lice::Ladim,
            },
            view: "farm".into(),
            screenshot: None,
            at: None,
            sigma: 16,
            start: "2025-06-15T00:00:00Z".into(),
            section: SectionShows::Speed,
            layer: "mean".into(),
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
            match flag.as_str() {
                "--scenario" => args.scenario = Some(value),
                "--replay" => args.replay = Some(value.into()),
                "--save-snapshot" => args.save_snapshot = Some(value.into()),
                "--mesh" => args.mesh = Some(value.into()),
                "--order" => args.order = num(&flag, &value)?,
                "--levels" => args.levels = num(&flag, &value)?,
                "--hours" => args.hours = num(&flag, &value)?,
                "--interval" => args.interval = num(&flag, &value)?,
                "--threads" => args.threads = num(&flag, &value)?,
                "--memory" => args.memory_mb = num(&flag, &value)?,
                "--rate" => args.rate = Some(num(&flag, &value)?),
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
                "--lice" => {
                    args.particles.lice = match value.as_str() {
                        "ladim" => Lice::Ladim,
                        "johnsen" => Lice::Johnsen,
                        "passive" => Lice::Passive,
                        _ => {
                            return Err(format!(
                                "--lice takes ladim, johnsen or passive, not {value}"
                            ));
                        }
                    }
                }
                "--sigma" => args.sigma = num(&flag, &value)?,
                "--start" => args.start = value,
                "--section" => {
                    args.section = match value.as_str() {
                        "speed" => SectionShows::Speed,
                        "temperature" => SectionShows::Temperature,
                        "off" => SectionShows::Off,
                        _ => {
                            return Err(format!(
                                "--section takes speed, temperature or off, not {value}"
                            ));
                        }
                    }
                }
                "--layer" => args.layer = value,
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
    // A snapshot file holds its own domain; VTU frames are of a Frøya run unless
    // told otherwise
    let snapshot_file = args.replay.as_deref().filter(|p| p.is_file());
    let (mut scenario, replay) = if let Some(path) = snapshot_file {
        match Replay::snapshot(path) {
            Ok((replay, scenario)) => (scenario, Some(replay)),
            Err(e) => {
                eprintln!("cannot replay {}: {e}", path.display());
                return AppExit::from_code(1);
            }
        }
    } else {
        let scenario_name = args.scenario.clone().unwrap_or_else(|| {
            if args.replay.is_some() {
                "froya"
            } else {
                "fjord"
            }
            .into()
        });
        let built = match scenario_name.as_str() {
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
            "channel" => match ModelClock::parse(&args.start) {
                Ok(clock) => Ok(Scenario::farm_channel(args.order, args.sigma, clock)),
                Err(e) => Err(format!("--start {}: {e}", args.start).into()),
            },
            other => {
                eprintln!("unknown scenario {other}: fjord, froya or channel");
                return AppExit::from_code(2);
            }
        };
        let scenario = match built {
            Ok(s) => s,
            Err(e) => {
                eprintln!("cannot build the {scenario_name} scenario: {e}");
                return AppExit::from_code(1);
            }
        };
        let replay = match args.replay.as_deref().map(|d| Replay::vtu(d, &scenario)) {
            None => None,
            Some(Ok(replay)) => Some(replay),
            Some(Err(e)) => {
                eprintln!(
                    "cannot replay {}: {e}",
                    args.replay.as_ref().unwrap().display()
                );
                return AppExit::from_code(1);
            }
        };
        (scenario, replay)
    };
    let mut replay = replay;
    if let Some(replay) = &mut replay {
        println!(
            "Replaying {} frames of {}, {} s apart, t = {} to {} s",
            replay.frames(),
            args.replay.as_ref().unwrap().display(),
            replay.interval,
            replay.t_first,
            replay.t_last
        );
        let title = scenario
            .name
            .split(':')
            .next()
            .unwrap_or_default()
            .to_string();
        if let Some(path) = &args.save_snapshot {
            if let Err(e) = replay.save_to(path, &scenario) {
                eprintln!("cannot write {}: {e}", path.display());
                return AppExit::from_code(1);
            }
            println!("Saving the frames to {}", path.display());
        }
        scenario.name = match replay.clock {
            Some(clock) => format!("{title}: replay from {} UTC", &clock.format(0.0)[..16]),
            None => format!("{title}: replay"),
        };
    } else if args.save_snapshot.is_some() {
        eprintln!("--save-snapshot needs --replay");
        return AppExit::from_code(2);
    }
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
            yaw: scenario.close_up.yaw,
            pitch: scenario.close_up.pitch,
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

    // The water column of a 3D run: its levels and the section
    let column = scenario.three_d.as_ref().map(|three_d| {
        let farm = Probe::at(&locator, &scenario, scenario.farm);
        let bed: Vec<f32> = scenario.bathymetry.data.iter().map(|&b| b as f32).collect();
        let mut section = Section::new(&locator, &scenario, three_d.section);
        section.shows = args.section;
        (
            Levels {
                sigma: three_d
                    .sigma
                    .sigma_rho()
                    .iter()
                    .map(|&s| s as f32)
                    .collect(),
                depth: farm.map_or(0.0, |p| -p.eval(&bed)),
            },
            section,
        )
    });
    let top = scenario
        .three_d
        .as_ref()
        .map_or(0, |three_d| three_d.sigma.n_levels() - 1);
    let shown = match args.layer.as_str() {
        "mean" => ShownLayer::DepthMean,
        "surface" => ShownLayer::Level(top),
        "bed" => ShownLayer::Level(0),
        n => match n.parse::<usize>() {
            Ok(l) if l <= top => ShownLayer::Level(l),
            _ => {
                eprintln!("--layer takes mean, surface, bed or 0 to {top}, not {n}");
                return AppExit::from_code(2);
            }
        },
    };
    let periodic = Periodic(scenario.periodic.map(|p| p.map(|x| x as f32)));

    let (channel, source, t_end, interval, rate, run_clock) = match replay {
        Some(replay) => {
            let source = Source::Replay {
                name: args
                    .replay
                    .as_ref()
                    .and_then(|d| d.file_name())
                    .map_or(String::new(), |n| n.to_string_lossy().into_owned()),
                frames: replay.frames(),
                t_last: replay.t_last,
            };
            let (t_end, interval, clock) = (replay.t_last, replay.interval, replay.clock);
            (
                replay::spawn(replay, args.threads),
                source,
                t_end,
                interval,
                args.rate.unwrap_or(3600.0),
                clock,
            )
        }
        None => {
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
            let rate = args.rate.unwrap_or(60.0);
            (channel, Source::Solver, t_end, args.interval, rate, None)
        }
    };

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
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
    .insert_resource(shown)
    .insert_resource(periodic)
    .insert_resource(Colouring::new(args.speed_max))
    .insert_resource(WaterOpacity(args.water_alpha))
    .insert_resource(Title(scenario.name.clone()))
    .insert_resource(Playback::new(rate, interval, args.memory_mb << 20))
    .insert_resource(SolverChannel(Mutex::new(channel)))
    .insert_resource(source)
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
    .add_systems(Last, playback::settle);
    if let Some(clock) = run_clock {
        app.insert_resource(RunClock(clock));
    }
    if let Some((levels, section)) = column {
        app.insert_resource(levels)
            .insert_resource(section)
            .add_plugins(LayersPlugin);
    }
    app.run()
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
