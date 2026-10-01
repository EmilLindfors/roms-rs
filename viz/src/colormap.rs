//! Colour maps as lookup tables of linear RGBA (what vertex colours are).

use bevy::prelude::*;

/// Viridis (perceptually uniform): current speed.
pub const VIRIDIS: &[[u8; 3]] = &[
    [0x44, 0x01, 0x54],
    [0x48, 0x24, 0x75],
    [0x41, 0x44, 0x87],
    [0x35, 0x5F, 0x8D],
    [0x2A, 0x78, 0x8E],
    [0x21, 0x91, 0x8C],
    [0x22, 0xA8, 0x84],
    [0x44, 0xBF, 0x70],
    [0x7A, 0xD1, 0x51],
    [0xBD, 0xDF, 0x26],
    [0xFD, 0xE7, 0x25],
];

/// Blue–white–red (ColorBrewer RdBu, reversed): surface elevation about 0.
pub const DIVERGING: &[[u8; 3]] = &[
    [0x05, 0x30, 0x61],
    [0x21, 0x66, 0xAC],
    [0x43, 0x93, 0xC3],
    [0x92, 0xC5, 0xDE],
    [0xD1, 0xE5, 0xF0],
    [0xF7, 0xF7, 0xF7],
    [0xFD, 0xDB, 0xC7],
    [0xF4, 0xA5, 0x82],
    [0xD6, 0x60, 0x4D],
    [0xB2, 0x18, 0x2B],
    [0x67, 0x00, 0x1F],
];

/// Plasma (perceptually uniform, bright yellow to deep blue): particle age.
pub const PLASMA: &[[u8; 3]] = &[
    [0x0D, 0x08, 0x87],
    [0x5C, 0x01, 0xA6],
    [0x9C, 0x17, 0x9E],
    [0xCC, 0x47, 0x78],
    [0xED, 0x79, 0x53],
    [0xFD, 0xB4, 0x2F],
    [0xF0, 0xF9, 0x21],
];

/// Dark blue through purple and red to pale yellow (after cmocean's "thermal"):
/// temperature.
pub const THERMAL: &[[u8; 3]] = &[
    [0x04, 0x23, 0x33],
    [0x2A, 0x2D, 0x83],
    [0x67, 0x2E, 0x8E],
    [0x9D, 0x3A, 0x7D],
    [0xD2, 0x4F, 0x5C],
    [0xF4, 0x7E, 0x3A],
    [0xFB, 0xBA, 0x3D],
    [0xE8, 0xFA, 0x5B],
];

/// Sand in the shallows to dark slate at depth: the bed.
pub const SEABED: &[[u8; 3]] = &[
    [0xD9, 0xC9, 0xA3],
    [0xA9, 0xA5, 0x8A],
    [0x6F, 0x7F, 0x74],
    [0x3E, 0x55, 0x60],
    [0x22, 0x32, 0x3F],
];

/// 256 samples of a colour map, linear RGBA.
pub struct Lut([[f32; 4]; 256]);

impl Lut {
    pub fn new(stops: &[[u8; 3]]) -> Self {
        Self(std::array::from_fn(|i| {
            let c = LinearRgba::from(srgb_at(stops, i as f32 / 255.0));
            [c.red, c.green, c.blue, 1.0]
        }))
    }

    /// The colour at `x` ∈ [0, 1] (clamped).
    #[inline]
    pub fn at(&self, x: f32) -> [f32; 4] {
        self.0[(x.clamp(0.0, 1.0) * 255.0 + 0.5) as usize]
    }
}

/// The map at `x` ∈ [0, 1], interpolated in sRGB between its stops.
pub fn srgb_at(stops: &[[u8; 3]], x: f32) -> Srgba {
    let f = x.clamp(0.0, 1.0) * (stops.len() - 1) as f32;
    let i = (f as usize).min(stops.len() - 2);
    let w = f - i as f32;
    let [a, b] =
        [stops[i], stops[i + 1]].map(|c| Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32) / 255.0);
    let c = a.lerp(b, w);
    Srgba::new(c.x, c.y, c.z, 1.0)
}
