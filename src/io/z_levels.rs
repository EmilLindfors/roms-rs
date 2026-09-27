//! Depth means of vertical profiles: z-levels (fixed depths, as in
//! NorKyst-800's `*_zdepth` output) and ROMS s-levels (terrain-following).

/// Mean over the water column `[0, depth]` of a profile given at fixed
/// `levels` (m, positive down, increasing), `None` where the level lies below
/// the bed or has no data.
///
/// Trapezoid rule between the valid levels above the bed; the shallowest
/// value is extended up to the surface (z = 0) and the deepest down to the
/// bed. Returns `None` for an empty column or `depth ≤ 0`. The free-surface
/// elevation is ignored (a relative error of η/h).
pub fn depth_average_z(levels: &[f64], values: &[Option<f64>], depth: f64) -> Option<f64> {
    assert_eq!(
        levels.len(),
        values.len(),
        "levels and values differ in length"
    );
    if depth <= 0.0 {
        return None;
    }
    let mut profile = levels
        .iter()
        .zip(values)
        .filter_map(|(&z, v)| v.map(|v| (z, v)))
        .filter(|&(z, _)| z < depth);
    let (z_first, v_first) = profile.next()?;
    let (mut integral, mut last) = (v_first * z_first, (z_first, v_first));
    for (z, v) in profile {
        integral += 0.5 * (last.1 + v) * (z - last.0);
        last = (z, v);
    }
    integral += last.1 * (depth - last.0);
    Some(integral / depth)
}

/// Relative thickness of each ROMS s-level layer in a water column of
/// still-water depth `h`: the weights of the layer values in the depth mean
/// `ū = Σ wₖ uₖ`, bottom layer first.
///
/// `s_w` and `cs_w` are the stretched coordinate and stretching function at
/// the `N + 1` layer interfaces (−1 at the bed, 0 at the surface), `hc` the
/// critical depth. The interface depths are `z_w = S + ζ(1 + S/h)` with
/// `S = hc s + (h − hc) C` (`vtransform` 1) or `z_w = ζ + (ζ + h) S` with
/// `S = (hc s + h C)/(hc + h)` (`vtransform` 2); either way the layer
/// thicknesses are proportional to the differences of `S`, so the weights do
/// not depend on the free surface ζ. Returns `None` for other transforms,
/// mismatched lengths or `h ≤ 0`.
pub fn s_level_weights(
    s_w: &[f64],
    cs_w: &[f64],
    hc: f64,
    h: f64,
    vtransform: i32,
) -> Option<Vec<f64>> {
    if s_w.len() != cs_w.len() || s_w.len() < 2 || h.is_nan() || h <= 0.0 {
        return None;
    }
    let stretched: Vec<f64> = match vtransform {
        1 => s_w
            .iter()
            .zip(cs_w)
            .map(|(&s, &c)| hc * s + (h - hc) * c)
            .collect(),
        2 => s_w
            .iter()
            .zip(cs_w)
            .map(|(&s, &c)| (hc * s + h * c) / (hc + h))
            .collect(),
        _ => return None,
    };
    let total = stretched[stretched.len() - 1] - stretched[0];
    if total.is_nan() || total <= 0.0 {
        return None;
    }
    Some(
        stretched
            .windows(2)
            .map(|w| (w[1] - w[0]) / total)
            .collect(),
    )
}

/// Depth mean `Σ wₖ uₖ / Σ wₖ` over the layers with a value, `None` if none
/// has one.
pub fn weighted_mean(weights: &[f64], values: &[Option<f64>]) -> Option<f64> {
    let (sum, total) = weights
        .iter()
        .zip(values)
        .filter_map(|(&w, v)| v.map(|v| (w * v, w)))
        .fold((0.0, 0.0), |(s, t), (wv, w)| (s + wv, t + w));
    (total > 0.0).then(|| sum / total)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEVELS: [f64; 6] = [0.0, 5.0, 10.0, 25.0, 50.0, 100.0];

    #[test]
    fn uniform_profile_averages_to_itself() {
        let values = [Some(0.3); 6];
        for depth in [3.0, 17.0, 60.0, 400.0] {
            assert!((depth_average_z(&LEVELS, &values, depth).unwrap() - 0.3).abs() < 1e-14);
        }
    }

    /// A profile linear in z is integrated exactly between the levels; below
    /// the deepest valid level it is held constant.
    #[test]
    fn linear_profile_and_bed_extension() {
        let values: Vec<Option<f64>> = LEVELS.iter().map(|&z| Some(1.0 - 0.01 * z)).collect();
        // Bed at 50 m: levels 0..25 used, the 25 m value held to 50 m
        let avg = depth_average_z(&LEVELS, &values, 50.0).unwrap();
        let expected = ((1.0 - 0.01 * 12.5) * 25.0 + (1.0 - 0.25) * 25.0) / 50.0;
        assert!((avg - expected).abs() < 1e-14, "{avg} vs {expected}");
    }

    #[test]
    fn missing_levels_and_empty_columns() {
        // Only 5 m and 10 m valid, bed at 20 m: 0.5·5 + 0.5·5 (trapezoid) + 1.0·10
        let values = [None, Some(0.5), Some(0.5), None, None, None];
        let avg = depth_average_z(&LEVELS, &values, 20.0).unwrap();
        assert!((avg - 0.5).abs() < 1e-14);
        assert_eq!(depth_average_z(&LEVELS, &[None; 6], 20.0), None);
        assert_eq!(depth_average_z(&LEVELS, &[Some(1.0); 6], 0.0), None);
    }

    /// Uniform σ layers (`C = s`) have equal weights for both transforms;
    /// a stretched grid follows the thickness of its layers.
    #[test]
    fn s_level_weights_follow_layer_thickness() {
        let s_w = [-1.0, -0.75, -0.5, -0.25, 0.0];
        for vt in [1, 2] {
            let w = s_level_weights(&s_w, &s_w, 10.0, 80.0, vt).unwrap();
            for wk in &w {
                assert!((wk - 0.25).abs() < 1e-14, "Vtransform {vt}: {w:?}");
            }
        }
        // Surface-intensified stretching: thin top layers
        let cs_w = [-1.0, -0.5, -0.2, -0.05, 0.0];
        let w = s_level_weights(&s_w, &cs_w, 0.0, 100.0, 2).unwrap();
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        assert!((w[0] - 0.5).abs() < 1e-14 && (w[3] - 0.05).abs() < 1e-14);
        // A linear profile in z is averaged exactly (midpoint of each layer)
        let z_mid: Vec<f64> = cs_w.windows(2).map(|c| 50.0 * (c[0] + c[1])).collect();
        let u: Vec<Option<f64>> = z_mid.iter().map(|&z| Some(1.0 + 0.01 * z)).collect();
        let mean = weighted_mean(&w, &u).unwrap();
        assert!((mean - (1.0 - 0.01 * 50.0)).abs() < 1e-12, "{mean}");
        assert!(s_level_weights(&s_w, &cs_w, 10.0, 0.0, 2).is_none());
        assert!(s_level_weights(&s_w, &cs_w, 10.0, 50.0, 3).is_none());
        assert_eq!(weighted_mean(&w, &[None; 4]), None);
    }
}
