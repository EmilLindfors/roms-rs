//! The water column of a 3D run: which layer's current the water shows, and a
//! vertical section through the water.
//!
//! `,` and `.` step the current shown by the water's colouring, the arrows and the
//! close-up through the depth mean and the σ-layers, from the surface layer down
//! ([`ShownLayer`]).
//!
//! The section ([`Section`]) is a curtain hanging from the surface to the bed along a
//! line ([`crate::scenario::ThreeD::section`], through a cage along the flow). Its
//! columns sample every layer by the element polynomial at their point ([`Probe`]),
//! linear in time between the snapshots, at the layer centres `z = η + σ_l D` (the
//! top layer's value at the surface, the bottom layer's at the bed). V cycles it
//! through the horizontal speed, the temperature and off. The speed has a scale of its
//! own, fitted to the section in steps of 0.05 m/s (a cage's wake is a fifth of the
//! current, too little to read on the water's scale from zero); the temperature's is
//! fitted to the first snapshot.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::prelude::*;
use dg_rs::mesh::PointLocator2D;

use crate::colormap::{Lut, THERMAL, VIRIDIS};
use crate::field::{Frame, Probe, ShownLayer};
use crate::playback::Playback;
use crate::scenario::Scenario;

/// Columns along the section.
const COLUMNS: usize = 240;
/// Step of the section's speed scale (m/s).
const SPEED_STEP: f32 = 0.05;

/// What the section shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SectionShows {
    Speed,
    Temperature,
    Off,
}

/// The σ-levels of a 3D run: layer centres from the bed up, for the legend.
#[derive(Resource, Clone, Debug)]
pub struct Levels {
    pub sigma: Vec<f32>,
    /// Water depth at rest at the farm (m)
    pub depth: f32,
}

/// The section's columns and what it shows.
#[derive(Resource)]
pub struct Section {
    columns: Vec<([f64; 2], Probe)>,
    pub shows: SectionShows,
    /// Temperature at the ends of the colour map (°C), fitted to the first snapshot
    pub temperature: Option<[f32; 2]>,
    /// Speed at the ends of the colour map (m/s), following the section
    pub speed: Option<[f32; 2]>,
    mesh: Handle<Mesh>,
    bed: Vec<f32>,
    speed_lut: Lut,
    temp_lut: Lut,
}

/// The scale `[lo, hi]` in steps of [`SPEED_STEP`] around the speeds `min..max`: kept
/// while they fit and fill at least half of it, so it does not flicker.
fn follow(scale: Option<[f32; 2]>, min: f32, max: f32) -> [f32; 2] {
    if let Some([lo, hi]) = scale
        && lo <= min
        && max <= hi
        && hi - lo <= 2.0 * (max - min) + SPEED_STEP
    {
        return [lo, hi];
    }
    let lo = ((min / SPEED_STEP).floor() * SPEED_STEP).max(0.0);
    let hi = ((max / SPEED_STEP).ceil() * SPEED_STEP).max(lo + SPEED_STEP);
    [lo, hi]
}

impl Section {
    /// Columns every `length / COLUMNS` along the line from `a` to `b`, where the mesh
    /// is.
    pub fn new(locator: &PointLocator2D, scenario: &Scenario, [a, b]: [[f64; 2]; 2]) -> Self {
        let columns: Vec<([f64; 2], Probe)> = (0..=COLUMNS)
            .filter_map(|c| {
                let s = c as f64 / COLUMNS as f64;
                let at = [a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])];
                Probe::at(locator, scenario, at).map(|probe| (at, probe))
            })
            .collect();
        let bed_f32: Vec<f32> = scenario.bathymetry.data.iter().map(|&b| b as f32).collect();
        let bed = columns.iter().map(|(_, p)| p.eval(&bed_f32)).collect();
        Self {
            columns,
            shows: SectionShows::Speed,
            temperature: None,
            speed: None,
            mesh: Handle::default(),
            bed,
            speed_lut: Lut::new(VIRIDIS),
            temp_lut: Lut::new(THERMAL),
        }
    }
}

#[derive(Component)]
struct Curtain;

pub struct LayersPlugin;

impl Plugin for LayersPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn).add_systems(
            Update,
            (keys, draw.after(crate::field::interpolate)).chain(),
        );
    }
}

fn spawn(
    mut commands: Commands,
    mut section: ResMut<Section>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    section.mesh = meshes.add(
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new())
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new())
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, Vec::<[f32; 4]>::new())
        .with_inserted_indices(Indices::U32(Vec::new())),
    );
    commands.spawn((
        Name::new("Section"),
        Curtain,
        Mesh3d(section.mesh.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            unlit: true,
            double_sided: true,
            cull_mode: None,
            ..default()
        })),
        // It moves with the surface: a bounding box from its first frame would cull it.
        NoFrustumCulling,
        Visibility::Hidden,
    ));
}

fn keys(
    keys: Res<ButtonInput<KeyCode>>,
    levels: Res<Levels>,
    mut shown: ResMut<ShownLayer>,
    mut section: ResMut<Section>,
) {
    let top = levels.sigma.len() - 1;
    // Downwards: depth mean, surface layer, ..., bed layer
    if keys.just_pressed(KeyCode::Period) {
        *shown = match *shown {
            ShownLayer::DepthMean => ShownLayer::Level(top),
            ShownLayer::Level(0) => ShownLayer::Level(0),
            ShownLayer::Level(l) => ShownLayer::Level(l - 1),
        };
    }
    if keys.just_pressed(KeyCode::Comma) {
        *shown = match *shown {
            ShownLayer::Level(l) if l >= top => ShownLayer::DepthMean,
            ShownLayer::Level(l) => ShownLayer::Level(l + 1),
            ShownLayer::DepthMean => ShownLayer::DepthMean,
        };
    }
    if keys.just_pressed(KeyCode::KeyV) {
        section.shows = match section.shows {
            SectionShows::Speed => SectionShows::Temperature,
            SectionShows::Temperature => SectionShows::Off,
            SectionShows::Off => SectionShows::Speed,
        };
    }
}

fn draw(
    playback: Res<Playback>,
    frame: Res<Frame>,
    levels: Res<Levels>,
    mut section: ResMut<Section>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut visibility: Query<&mut Visibility, With<Curtain>>,
) {
    let Ok(mut visible) = visibility.single_mut() else {
        return;
    };
    if section.shows == SectionShows::Off {
        visible.set_if_neq(Visibility::Hidden);
        return;
    }
    if !(playback.changed || section.is_changed()) {
        return;
    }
    let Some((a, b, w)) = playback.bracket() else {
        return;
    };
    let (Some(la), Some(lb)) = (&a.layers, &b.layers) else {
        return;
    };
    let nl = la.n_levels;
    if section.temperature.is_none() {
        let (lo, hi) = la
            .temp
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &t| {
                (lo.min(t), hi.max(t))
            });
        // Whole half-degrees around the range, at least one degree wide
        let lo = (2.0 * lo).floor() / 2.0;
        let hi = ((2.0 * hi).ceil() / 2.0).max(lo + 1.0);
        section.bypass_change_detection().temperature = Some([lo, hi]);
    }
    let [t_lo, t_hi] = section.temperature.unwrap_or([0.0, 1.0]);
    visible.set_if_neq(Visibility::Inherited);

    let rows = nl + 2;
    let n = section.columns.len();
    let mut positions = Vec::with_capacity(n * rows);
    let mut normals = Vec::with_capacity(n * rows);
    let mut colours = Vec::with_capacity(n * rows);
    let lerp = |x: f32, y: f32| x + w * (y - x);
    // Across the line, horizontal: the curtain's normal
    let along = section.columns.last().map_or(Vec3::X, |(end, _)| {
        frame.world(*end, 0.0) - frame.world(section.columns[0].0, 0.0)
    });
    let normal = Vec3::new(-along.z, 0.0, along.x)
        .normalize_or_zero()
        .to_array();
    // The value of every column and layer, then the colours on its scale
    let temperature = section.shows == SectionShows::Temperature;
    let mut values = Vec::with_capacity(n * nl);
    for (c, (at, probe)) in section.columns.iter().enumerate() {
        let eta = lerp(probe.eval(&a.eta), probe.eval(&b.eta));
        let bed = section.bed[c];
        let depth = (eta - bed).max(0.0);
        for l in 0..nl {
            let at_level = |field_a: &[f32], field_b: &[f32]| {
                lerp(
                    probe.eval_level(field_a, nl, l),
                    probe.eval_level(field_b, nl, l),
                )
            };
            values.push(if temperature {
                at_level(&la.temp, &lb.temp)
            } else {
                at_level(&la.u, &lb.u).hypot(at_level(&la.v, &lb.v))
            });
        }
        // Bed, the layer centres, surface
        let heights = std::iter::once(bed)
            .chain(levels.sigma.iter().map(|&s| eta + s * depth))
            .chain(std::iter::once(eta));
        for z in heights {
            positions.push(frame.world(*at, z).to_array());
            normals.push(normal);
        }
    }
    let ([lo, hi], lut) = if temperature {
        ([t_lo, t_hi], &section.temp_lut)
    } else {
        let (min, max) = values
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        let scale = follow(section.speed, min, max);
        if section.speed != Some(scale) {
            section.bypass_change_detection().speed = Some(scale);
        }
        (scale, &section.speed_lut)
    };
    for column in values.chunks_exact(nl) {
        for row in 0..rows {
            let l = row.saturating_sub(1).min(nl - 1);
            colours.push(lut.at((column[l] - lo) / (hi - lo)));
        }
    }
    let mut indices = Vec::with_capacity(6 * (n.saturating_sub(1)) * (rows - 1));
    for c in 1..n as u32 {
        for r in 0..rows as u32 - 1 {
            let (p, q) = ((c - 1) * rows as u32 + r, c * rows as u32 + r);
            indices.extend([p, q, q + 1, p, q + 1, p + 1]);
        }
    }
    let Some(mut mesh) = meshes.get_mut(&section.mesh) else {
        return;
    };
    let same_size = matches!(
        mesh.attribute(Mesh::ATTRIBUTE_POSITION),
        Some(VertexAttributeValues::Float32x3(old)) if old.len() == positions.len()
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
    if !same_size {
        mesh.insert_indices(Indices::U32(indices));
    }
}
