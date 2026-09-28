//! What is shown and how far the solver has got (top left), the colour scale (bottom
//! left) and the keys (bottom right; H hides them).

use bevy::prelude::*;
use bevy::text::FontSize;
use bevy::ui::{BackgroundGradient, ColorStop, LinearGradient};

use crate::colormap::srgb_at;
use crate::particles::Particles;
use crate::playback::{Playback, SolverState};
use crate::surface::{ColourBy, Colouring, SurfaceStyle};

// ASCII only: Bevy's default font has no arrows or middle dots.
const KEYS: &str = "Space pause   [ ] rate   Left/Right seek   Home/End\n\
C colour   T water   A arrows   P particles   F farm   O overview   H keys\n\
drag orbit   right-drag pan   wheel zoom";

/// The scenario's one-line description, heading the status.
#[derive(Resource)]
pub struct Title(pub String);

#[derive(Component)]
struct Status;

#[derive(Component)]
struct Help;

#[derive(Component)]
struct Bar;

/// The labels under the colour bar: low, middle, high.
#[derive(Component)]
struct Tick(usize);

#[derive(Component)]
struct ScaleTitle;

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn)
            .add_systems(Update, (status, scale, help));
    }
}

fn shadow() -> TextShadow {
    TextShadow {
        offset: Vec2::splat(1.5),
        color: Color::srgba(0.0, 0.0, 0.0, 0.85),
    }
}

fn font(size: f32) -> TextFont {
    TextFont {
        font_size: FontSize::Px(size),
        ..default()
    }
}

fn spawn(mut commands: Commands) {
    commands.spawn((
        Status,
        Text::new(""),
        font(15.0),
        shadow(),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(14.0),
            top: Val::Px(12.0),
            ..default()
        },
    ));
    commands.spawn((
        Help,
        Text::new(KEYS),
        font(13.0),
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.8)),
        shadow(),
        TextLayout {
            justify: Justify::Right,
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            right: Val::Px(14.0),
            bottom: Val::Px(12.0),
            ..default()
        },
    ));
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            left: Val::Px(14.0),
            bottom: Val::Px(14.0),
            width: Val::Px(260.0),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(4.0),
            ..default()
        })
        .with_children(|scale| {
            scale.spawn((ScaleTitle, Text::new(""), font(13.0), shadow()));
            scale.spawn((
                Bar,
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(12.0),
                    ..default()
                },
                BackgroundGradient::default(),
            ));
            scale
                .spawn(Node {
                    justify_content: JustifyContent::SpaceBetween,
                    ..default()
                })
                .with_children(|ticks| {
                    for i in 0..3 {
                        ticks.spawn((Tick(i), Text::new(""), font(12.0), shadow()));
                    }
                });
        });
}

/// `3 h 07 min`, from seconds.
fn clock(t: f64) -> String {
    let minutes = (t / 60.0).floor() as u64;
    format!("{} h {:02} min", minutes / 60, minutes % 60)
}

fn status(
    playback: Res<Playback>,
    title: Res<Title>,
    style: Res<SurfaceStyle>,
    particles: Res<Particles>,
    mut text: Query<&mut Text, With<Status>>,
) {
    let Ok(mut text) = text.single_mut() else {
        return;
    };
    let mut s = format!("{}\n", title.0);
    match playback.newest() {
        None => s += "starting the solver...\n",
        Some(_) => {
            let state = if playback.paused {
                "  paused"
            } else if playback.waiting() {
                "  waiting for the solver"
            } else {
                ""
            };
            s += &format!("t = {}   x{:.0}{state}\n", clock(playback.t), playback.rate);
        }
    }
    let newest = playback.newest().unwrap_or(0.0);
    s += &match &playback.solver {
        SolverState::Running => format!(
            "solver at {}, {:.0}x real time\n",
            clock(newest),
            playback.solver_speed()
        ),
        SolverState::Finished { steps, wall } => format!(
            "solver done: {} in {steps} steps, {wall:.0} s\n",
            clock(newest)
        ),
        SolverState::Failed(e) => format!("solver failed: {e}\n"),
    };
    if let (Some(lo), Some(hi)) = (playback.oldest(), playback.newest()) {
        s += &format!(
            "kept {} to {} ({} snapshots, {:.0} MB)",
            clock(lo),
            clock(hi),
            playback.frames.len(),
            playback.bytes() as f64 / 1e6
        );
    }
    let [active, stranded, out] = particles.counts;
    if active + stranded + out > 0 {
        s += &format!(
            "\nparticles: {active} in the water, {stranded} stranded, {out} out\n\
             colour: age, yellow new to dark blue {}; grey stranded",
            clock(particles.age_scale as f64)
        );
        if !particles.on {
            s += " (hidden, P)";
        }
    }
    if *style == SurfaceStyle::Hidden {
        s += "\nwater hidden (T)";
    }
    if text.0 != s {
        text.0 = s;
    }
}

fn scale(
    colouring: Res<Colouring>,
    mut bar: Query<&mut BackgroundGradient, With<Bar>>,
    mut ticks: Query<(&Tick, &mut Text), Without<ScaleTitle>>,
    mut title: Query<&mut Text, With<ScaleTitle>>,
    mut shown: Local<Option<(ColourBy, f32, f32)>>,
) {
    let key = (colouring.by, colouring.speed_max, colouring.eta_max);
    if *shown == Some(key) {
        return;
    }
    *shown = Some(key);
    let stops = colouring.stops();
    if let Ok(mut bar) = bar.single_mut() {
        let colours = (0..=16)
            .map(|i| ColorStop::from(Color::from(srgb_at(stops, i as f32 / 16.0))))
            .collect();
        *bar = LinearGradient::to_right(colours).into();
    }
    let (heading, lo, hi) = match colouring.by {
        ColourBy::Speed => ("current speed (m/s)", 0.0, colouring.speed_max),
        ColourBy::Elevation => (
            "surface elevation η (m)",
            -colouring.eta_max,
            colouring.eta_max,
        ),
    };
    if let Ok(mut title) = title.single_mut() {
        title.0 = heading.into();
    }
    for (tick, mut text) in &mut ticks {
        let v = lo + (hi - lo) * tick.0 as f32 / 2.0;
        text.0 = format!("{}", (v * 1000.0).round() / 1000.0);
    }
}

fn help(keys: Res<ButtonInput<KeyCode>>, mut help: Query<&mut Visibility, With<Help>>) {
    if keys.just_pressed(KeyCode::KeyH) {
        for mut visibility in &mut help {
            *visibility = if *visibility == Visibility::Hidden {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
        }
    }
}
