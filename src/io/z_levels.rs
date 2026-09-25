//! Z-level (fixed-depth) profiles, as in NorKyst-800's `*_zdepth` output.

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
}
