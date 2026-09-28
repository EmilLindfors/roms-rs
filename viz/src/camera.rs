//! An orbit camera: left-drag turns about the focus, right- or middle-drag (or
//! Shift+left-drag) slides the focus over the water, the wheel zooms. F frames the
//! farm, O the whole domain.

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit};
use bevy::prelude::*;

#[derive(Component, Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub focus: Vec3,
    /// Angle about +Y from +Z (rad)
    pub yaw: f32,
    /// Angle above the horizontal (rad)
    pub pitch: f32,
    pub distance: f32,
}

impl OrbitCamera {
    pub fn transform(&self) -> Transform {
        let dir = Vec3::new(self.pitch.cos() * self.yaw.sin(), self.pitch.sin(), self.pitch.cos() * self.yaw.cos());
        Transform::from_translation(self.focus + self.distance * dir).looking_at(self.focus, Vec3::Y)
    }
}

/// The two framings F and O return to.
#[derive(Resource, Clone, Copy, Debug)]
pub struct Views {
    pub farm: OrbitCamera,
    pub domain: OrbitCamera,
}

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (input, place).chain());
    }
}

fn input(
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    views: Res<Views>,
    mut cameras: Query<&mut OrbitCamera>,
) {
    let Ok(mut orbit) = cameras.single_mut() else { return };
    if keys.just_pressed(KeyCode::KeyF) {
        *orbit = views.farm;
    }
    if keys.just_pressed(KeyCode::KeyO) {
        *orbit = views.domain;
    }
    let shift = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    let pan = mouse.pressed(MouseButton::Right) || mouse.pressed(MouseButton::Middle) || (shift && mouse.pressed(MouseButton::Left));
    let d = motion.delta;
    if pan && d != Vec2::ZERO {
        // Along the ground: the view's right and its forward flattened.
        let (sin, cos) = orbit.yaw.sin_cos();
        let right = Vec3::new(cos, 0.0, -sin);
        let forward = Vec3::new(-sin, 0.0, -cos);
        let k = 0.0012 * orbit.distance;
        orbit.focus += (-right * d.x + forward * d.y) * k;
    } else if mouse.pressed(MouseButton::Left) && d != Vec2::ZERO {
        orbit.yaw -= 0.005 * d.x;
        orbit.pitch = (orbit.pitch + 0.005 * d.y).clamp(0.02, 1.55);
    }
    let notches = match scroll.unit {
        MouseScrollUnit::Line => scroll.delta.y,
        MouseScrollUnit::Pixel => scroll.delta.y / 40.0,
    };
    if notches != 0.0 {
        orbit.distance = (orbit.distance * (-0.12 * notches).exp()).clamp(5.0, 200_000.0);
    }
}

fn place(mut cameras: Query<(&OrbitCamera, &mut Transform), Changed<OrbitCamera>>) {
    for (orbit, mut transform) in &mut cameras {
        *transform = orbit.transform();
    }
}
