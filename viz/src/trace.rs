//! A time series of the surface at the point of interest (top right; G hides it): the
//! model's η there over the whole run, the tide gauge's observations against it when
//! `--gauge` gives them, and a marker at the shown time. Over a 15-day replay it shows
//! the spring–neap cycle and where the model departs from the gauge.
//!
//! The model's η is sampled from every snapshot as it arrives, by the element
//! polynomial ([`Probe`]), so the trace grows while the solver runs or the frames are
//! read. A snapshot file read on demand holds only the frames around the shown time:
//! its reading thread samples every frame at the probe instead
//! ([`crate::playback::Playback::series`]), so the trace has all of them. A gauge often lies in a shoreline element, which the mesh may not resolve
//! (a perched pocket, or dry); as `examples/froya_real_data.rs` does, the trace then
//! samples the nearest node of an element whose every node is at least
//! [`MIN_DEPTH`] deep, and its title says how far away.
//!
//! The plot is drawn on the CPU into an image ([`WIDTH`] × [`HEIGHT`] pixels, twice
//! its size on screen) whenever a sample arrives, at most a few times a second.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::text::FontSize;
use dg_rs::mesh::PointLocator2D;
use dg_rs::time::ModelClock;
use dg_rs::types::ElementIndex;

use crate::field::Probe;
use crate::playback::Playback;
use crate::plot::Canvas;
use crate::scenario::Scenario;

/// Depth (m) of every node of an element the trace may sample in.
pub const MIN_DEPTH: f64 = 3.0;
/// The plot in pixels (drawn at half this size on screen).
const WIDTH: usize = 1200;
const HEIGHT: usize = 300;
/// Seconds between redraws while samples arrive.
const REDRAW: f32 = 0.25;

const MODEL: [u8; 3] = [255, 255, 255];
const GAUGE: [u8; 3] = [255, 150, 60];

#[derive(Resource)]
pub struct Trace {
    probe: Option<Probe>,
    /// Model time span of the plot (s)
    span: [f64; 2],
    /// (model time, η) of the model and of the gauge
    model: Vec<(f64, f32)>,
    gauge: Vec<(f64, f32)>,
    title: String,
    dirty: bool,
}

impl Trace {
    /// η at `at` (m, mesh coordinates: the scenario's point of interest, or the
    /// nearest submerged node) over the model times `span`, and the gauge's record
    /// `gauge` (Unix times, η) if any, placed on `clock`.
    pub fn new(
        scenario: &Scenario,
        locator: &PointLocator2D,
        at: [f64; 2],
        span: [f64; 2],
        gauge: Option<(String, Vec<(f64, f64)>)>,
        clock: Option<ModelClock>,
    ) -> Self {
        let (point, offset, depth) = submerged_point(scenario, at);
        let probe = Probe::at(locator, scenario, point);
        let gauge_name = gauge.as_ref().map(|(name, _)| name.clone());
        let gauge: Vec<(f64, f32)> = match (gauge, clock) {
            (Some((_, record)), Some(clock)) => record
                .into_iter()
                .map(|(unix, eta)| (clock.model_time(unix), eta as f32))
                .filter(|&(t, eta)| t >= span[0] && eta.is_finite())
                .collect(),
            _ => Vec::new(),
        };
        let place = gauge_name.as_deref().unwrap_or("the point of interest");
        let mut title = if offset > 1.0 {
            format!("surface at {place} (m), sampled {offset:.0} m away in {depth:.1} m")
        } else {
            format!("surface at {place} (m), {depth:.1} m deep")
        };
        title += match (gauge_name.is_some(), gauge.is_empty()) {
            (true, false) => ": model white, gauge orange (G)",
            (true, true) => ": model; no gauge record in the run (G)",
            _ => " (G)",
        };
        Self {
            probe,
            span,
            model: Vec::new(),
            gauge,
            title,
            dirty: true,
        }
    }

    /// Where the trace samples the model, if inside the mesh.
    pub fn probe(&self) -> Option<&Probe> {
        self.probe.as_ref()
    }
}

/// `p`, or the nearest node of an element with every node at least [`MIN_DEPTH`]
/// deep; with its distance from `p` and the still-water depth there.
fn submerged_point(scenario: &Scenario, p: [f64; 2]) -> ([f64; 2], f64, f64) {
    let (mesh, ops, bed) = (&scenario.mesh, &scenario.ops, &scenario.bathymetry);
    let n = ops.n_nodes;
    let submerged = |k: usize| {
        bed.data[k * n..(k + 1) * n]
            .iter()
            .all(|&b| b <= -MIN_DEPTH)
    };
    let locator = PointLocator2D::new(mesh);
    if let Some(probe) = dg_rs::solver::Probe2D::at(&locator, ops, p)
        && submerged(probe.element().as_usize())
    {
        let depth: f64 = -probe
            .weights()
            .iter()
            .zip(bed.element(probe.element()))
            .map(|(w, b)| w * b)
            .sum::<f64>();
        return (p, 0.0, depth);
    }
    let mut best = (p, f64::INFINITY, 0.0);
    for k in ElementIndex::iter(mesh.n_elements).filter(|k| submerged(k.as_usize())) {
        for i in 0..n {
            let q = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
            let d = (q[0] - p[0]).hypot(q[1] - p[1]);
            if d < best.1 {
                best = (q, d, -bed.get(k, i));
            }
        }
    }
    best
}

#[derive(Component)]
struct Plot;

#[derive(Component)]
struct Marker;

#[derive(Component)]
struct Panel;

#[derive(Component)]
enum Label {
    Title,
    Range,
    Start,
    End,
}

/// The plot's image.
#[derive(Resource)]
struct PlotImage(Handle<Image>);

pub struct TracePlugin;

impl Plugin for TracePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn)
            .add_systems(Update, (sample, draw, marker, toggle));
    }
}

fn font(size: f32) -> TextFont {
    TextFont {
        font_size: FontSize::Px(size),
        ..default()
    }
}

fn spawn(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image = images.add(Image::new_fill(
        Extent3d {
            width: WIDTH as u32,
            height: HEIGHT as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0; 4],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    ));
    commands.insert_resource(PlotImage(image.clone()));
    let shadow = TextShadow {
        offset: Vec2::splat(1.5),
        color: Color::srgba(0.0, 0.0, 0.0, 0.85),
    };
    commands
        .spawn((
            Panel,
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(14.0),
                top: Val::Px(12.0),
                width: Val::Px((WIDTH / 2) as f32),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(3.0),
                ..default()
            },
        ))
        .with_children(|panel| {
            panel.spawn((Label::Title, Text::new(""), font(13.0), shadow));
            panel
                .spawn((
                    Node {
                        width: Val::Px((WIDTH / 2) as f32),
                        height: Val::Px((HEIGHT / 2) as f32),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.35)),
                ))
                .with_children(|plot| {
                    plot.spawn((
                        Plot,
                        ImageNode::new(image),
                        Node {
                            width: Val::Percent(100.0),
                            height: Val::Percent(100.0),
                            ..default()
                        },
                    ));
                    plot.spawn((
                        Marker,
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Percent(0.0),
                            width: Val::Px(2.0),
                            height: Val::Percent(100.0),
                            ..default()
                        },
                        BackgroundColor(Color::srgba(1.0, 0.9, 0.3, 0.9)),
                    ));
                    plot.spawn((
                        Label::Range,
                        Text::new(""),
                        font(11.0),
                        shadow,
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(4.0),
                            top: Val::Px(2.0),
                            ..default()
                        },
                    ));
                });
            panel
                .spawn(Node {
                    justify_content: JustifyContent::SpaceBetween,
                    ..default()
                })
                .with_children(|ticks| {
                    ticks.spawn((Label::Start, Text::new(""), font(11.0), shadow));
                    ticks.spawn((Label::End, Text::new(""), font(11.0), shadow));
                });
        });
}

/// Samples the snapshots that arrived since the last one sampled.
fn sample(playback: Res<Playback>, mut trace: ResMut<Trace>) {
    if playback.on_demand.is_some() {
        if trace.model.len() < playback.series.len() {
            let new = trace.model.len();
            trace.model.extend_from_slice(&playback.series[new..]);
            trace.dirty = true;
        }
        // A file a run is still writing: the plot grows with it
        if let Some(newest) = playback.newest()
            && newest > trace.span[1]
        {
            trace.span[1] = newest;
            trace.dirty = true;
        }
        return;
    }
    let Some(probe) = trace.probe.clone() else {
        return;
    };
    let last = trace.model.last().map_or(f64::NEG_INFINITY, |s| s.0);
    let new = playback.frames.partition_point(|s| s.t <= last);
    if new == playback.frames.len() {
        return;
    }
    for s in playback.frames.range(new..) {
        trace.model.push((s.t, probe.eval(&s.eta)));
    }
    trace.dirty = true;
}

fn draw(
    time: Res<Time<Real>>,
    mut since: Local<f32>,
    mut trace: ResMut<Trace>,
    plot: Res<PlotImage>,
    mut images: ResMut<Assets<Image>>,
    clock: Option<Res<crate::hud::RunClock>>,
    mut labels: Query<(&Label, &mut Text)>,
) {
    *since += time.delta_secs();
    if !trace.dirty || *since < REDRAW {
        return;
    }
    *since = 0.0;
    trace.dirty = false;
    // The gauge within the plot's span (a file still being written widens it)
    let [t0, t1] = trace.span;
    let gauge = &trace.gauge[..trace.gauge.partition_point(|g| g.0 <= t1)];
    let (lo, hi) = trace
        .model
        .iter()
        .chain(gauge)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &(_, z)| {
            (lo.min(z), hi.max(z))
        });
    let (lo, hi) = if lo < hi { (lo, hi) } else { (-1.0, 1.0) };
    let pad = 0.08 * (hi - lo);
    let (lo, hi) = (lo - pad, hi + pad);
    let to_px = |&(t, z): &(f64, f32)| {
        (
            ((t - t0) / (t1 - t0).max(1e-9)) as f32 * (WIDTH - 1) as f32,
            (hi - z) / (hi - lo) * (HEIGHT - 1) as f32,
        )
    };
    let mut canvas = Canvas::new(WIDTH, HEIGHT);
    // Mean sea level
    if lo < 0.0 && hi > 0.0 {
        let y = (hi / (hi - lo) * (HEIGHT - 1) as f32) as usize;
        canvas.dashed_row(y, [0, WIDTH], [150, 150, 150], 0.6);
    }
    // A break in a record (missing observations) is not drawn across
    let gap = 3.0 * (t1 - t0) / gauge.len().max(2) as f64;
    polyline(&mut canvas, gauge, gap, &to_px, GAUGE, 4.0);
    polyline(&mut canvas, &trace.model, f64::INFINITY, &to_px, MODEL, 1.5);
    if let Some(mut image) = images.get_mut(&plot.0) {
        image.data = Some(canvas.rgba);
    }

    let date = |t: f64| match &clock {
        Some(c) => c.0.format(t)[..16].to_string(),
        None => format!("{:.1} h", t / 3600.0),
    };
    for (label, mut text) in &mut labels {
        let s = match label {
            Label::Title => trace.title.clone(),
            // Millimetres where the range is small (a channel, a lake)
            Label::Range if hi - lo < 0.1 => format!("{:+.1} to {:+.1} mm", 1e3 * hi, 1e3 * lo),
            Label::Range => format!("{hi:+.2} to {lo:+.2} m"),
            Label::Start => date(t0),
            Label::End => date(t1),
        };
        if text.0 != s {
            text.0 = s;
        }
    }
}

/// The series as straight segments `width` pixels wide, broken where samples are
/// more than `gap` seconds apart.
fn polyline(
    canvas: &mut Canvas,
    series: &[(f64, f32)],
    gap: f64,
    to_px: &impl Fn(&(f64, f32)) -> (f32, f32),
    colour: [u8; 3],
    width: f32,
) {
    for pair in series.windows(2) {
        if pair[1].0 - pair[0].0 > gap {
            continue;
        }
        canvas.segment(to_px(&pair[0]), to_px(&pair[1]), colour, width);
    }
    if let Some(last) = series.last() {
        let (x, y) = to_px(last);
        canvas.dot(x, y, colour, width);
    }
}

/// Keeps the marker at the shown time.
fn marker(playback: Res<Playback>, trace: Res<Trace>, mut node: Query<&mut Node, With<Marker>>) {
    if !playback.changed {
        return;
    }
    let [t0, t1] = trace.span;
    let f = ((playback.t - t0) / (t1 - t0).max(1e-9)).clamp(0.0, 1.0);
    for mut node in &mut node {
        node.left = Val::Percent(100.0 * f as f32);
    }
}

fn toggle(keys: Res<ButtonInput<KeyCode>>, mut panel: Query<&mut Visibility, With<Panel>>) {
    if keys.just_pressed(KeyCode::KeyG) {
        for mut v in &mut panel {
            *v = match *v {
                Visibility::Hidden => Visibility::Inherited,
                _ => Visibility::Hidden,
            };
        }
    }
}
