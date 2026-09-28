//! Arrows of the depth-averaged current on two grids, one over the whole domain and a
//! finer one around the farm; the camera's distance picks which is drawn.
//!
//! Each arrow samples (u, v) by the element polynomial at its point ([`Probe`]), so it
//! shows the current there, not at the nearest node. Its length is the grid spacing at
//! the top of the speed scale; it is white over water coloured by speed, else coloured
//! by speed itself. A (key) toggles them.

use bevy::prelude::*;
use dg_rs::mesh::PointLocator2D;

use crate::camera::OrbitCamera;
use crate::colormap::{Lut, VIRIDIS};
use crate::field::{Field, Frame, Probe};
use crate::scenario::Scenario;
use crate::surface::{ColourBy, Colouring};

/// Metres of lift above the surface, as a share of the grid spacing.
const LIFT: f32 = 0.05;

struct Arrow {
    /// Mesh point
    at: [f64; 2],
    probe: Probe,
}

struct Grid {
    spacing: f32,
    arrows: Vec<Arrow>,
}

impl Grid {
    /// Arrows every `spacing` metres over [lo, hi], where the mesh is.
    fn new(locator: &PointLocator2D, scenario: &Scenario, lo: [f64; 2], hi: [f64; 2], spacing: f64) -> Self {
        let (nx, ny) = (((hi[0] - lo[0]) / spacing) as usize, ((hi[1] - lo[1]) / spacing) as usize);
        let arrows = (0..=ny)
            .flat_map(|j| (0..=nx).map(move |i| [lo[0] + i as f64 * spacing, lo[1] + j as f64 * spacing]))
            .filter_map(|at| Probe::at(locator, scenario, at).map(|probe| Arrow { at, probe }))
            .collect();
        Self { spacing: spacing as f32, arrows }
    }
}

#[derive(Resource)]
pub struct Arrows {
    coarse: Grid,
    fine: Grid,
    /// Camera distance below which the fine grid is drawn
    near: f32,
    pub on: bool,
    lut: Lut,
}

impl Arrows {
    /// A coarse grid of about `across` arrows over the mesh's longer side, and a fine
    /// one every `fine_spacing` metres within `fine_radius` of the farm.
    pub fn new(locator: &PointLocator2D, scenario: &Scenario, across: usize, fine_spacing: f64, fine_radius: f64) -> Self {
        let (lo, hi) = scenario.mesh.vertices.iter().fold(([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]), |(lo, hi), v| {
            ([lo[0].min(v[0]), lo[1].min(v[1])], [hi[0].max(v[0]), hi[1].max(v[1])])
        });
        let coarse_spacing = (hi[0] - lo[0]).max(hi[1] - lo[1]) / across as f64;
        let half = coarse_spacing / 2.0;
        let coarse = Grid::new(locator, scenario, [lo[0] + half, lo[1] + half], hi, coarse_spacing);
        let [fx, fy] = scenario.farm;
        let fine = Grid::new(
            locator,
            scenario,
            [fx - fine_radius, fy - fine_radius],
            [fx + fine_radius, fy + fine_radius],
            fine_spacing,
        );
        Self { coarse, fine, near: (4.0 * fine_radius) as f32, on: true, lut: Lut::new(VIRIDIS) }
    }

    pub fn count(&self) -> (usize, usize) {
        (self.coarse.arrows.len(), self.fine.arrows.len())
    }
}

pub struct ArrowsPlugin;

impl Plugin for ArrowsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (toggle, draw.after(crate::field::interpolate)));
    }
}

fn toggle(keys: Res<ButtonInput<KeyCode>>, mut arrows: ResMut<Arrows>) {
    if keys.just_pressed(KeyCode::KeyA) {
        arrows.on = !arrows.on;
    }
}

fn draw(
    arrows: Res<Arrows>,
    field: Res<Field>,
    frame: Res<Frame>,
    colouring: Res<Colouring>,
    camera: Query<&OrbitCamera>,
    mut gizmos: Gizmos,
) {
    if !arrows.on || field.t.is_none() {
        return;
    }
    let near = camera.single().is_ok_and(|c| c.distance < arrows.near);
    let grid = if near { &arrows.fine } else { &arrows.coarse };
    let scale = grid.spacing / colouring.speed_max.max(1e-6);
    for arrow in &grid.arrows {
        let (u, v) = (arrow.probe.eval(&field.u), arrow.probe.eval(&field.v));
        let speed = u.hypot(v);
        if speed * scale < 0.05 * grid.spacing {
            continue;
        }
        let eta = arrow.probe.eval(&field.eta);
        let start = frame.world(arrow.at, eta) + Vec3::Y * LIFT * grid.spacing;
        // Mesh (u, v) is world (u, −v) in X, Z; long arrows are capped at 1.5 spacings.
        let end = start + Vec3::new(u, 0.0, -v) * scale.min(1.5 * grid.spacing / speed);
        // White over water coloured by speed; the speed colour over elevation.
        let colour = match colouring.by {
            ColourBy::Speed => LinearRgba::rgb(0.9, 0.9, 0.9),
            ColourBy::Elevation => {
                let [r, g, b, _] = arrows.lut.at(speed / colouring.speed_max);
                LinearRgba::rgb(r, g, b)
            }
        };
        gizmos
            .arrow(start, end, colour)
            .with_tip_length(0.3 * start.distance(end));
    }
}
