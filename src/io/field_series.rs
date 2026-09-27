//! Time series of gridded 2D fields, and interpolation in time.
//!
//! [`FieldSeries`] stores one value per grid point and snapshot, strided
//! `[time][point]` in `f32` (missing values are NaN), with an optional
//! offset for fields whose variation is small against their size
//! (atmospheric pressure: ±50 hPa around 1013 hPa, where `f32` alone would
//! resolve only ~0.01 Pa and spoil the gradients of the finest grids).
//!
//! [`TimeStencil`] gives the weights of the snapshots around a time:
//! linear, or cubic (4-point Lagrange). Hourly output of a tide is
//! interpolated linearly to 3 % of its amplitude at M2 (`(ωΔt)²/8`), cubically
//! to 0.15 % (`(ωΔt)⁴·9/384`).

/// Interpolation in time between snapshots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TimeInterpolation {
    /// Piecewise linear between the two bracketing snapshots.
    #[default]
    Linear,
    /// Cubic Lagrange through the four nearest snapshots (quadratic in the
    /// first and last interval). Continuous, exact for cubics; overshoots at
    /// jumps (e.g. a wind front), so meant for smooth signals such as tides.
    Cubic,
}

/// Weights of up to four snapshots: `Σ w[k] f[idx[k]]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeStencil {
    /// Snapshot indices
    pub idx: [usize; 4],
    /// Weights (unused entries 0)
    pub w: [f64; 4],
    /// Number of snapshots used
    pub len: usize,
}

impl TimeStencil {
    /// Weights at `time` for strictly increasing snapshot `times`; `None`
    /// outside `[times[0], times[last]]` (no extrapolation, no clamping). A
    /// single snapshot is time-invariant and covers every time.
    pub fn new(times: &[f64], time: f64, method: TimeInterpolation) -> Option<Self> {
        let single = |k: usize| Self {
            idx: [k, 0, 0, 0],
            w: [1.0, 0.0, 0.0, 0.0],
            len: 1,
        };
        let n = times.len();
        match n {
            0 => return None,
            1 => return Some(single(0)),
            _ => {}
        }
        if !(time >= times[0] && time <= times[n - 1]) {
            return None;
        }
        let i1 = times.partition_point(|&t| t < time).clamp(1, n - 1);
        let i0 = i1 - 1;
        if time == times[i0] {
            return Some(single(i0));
        }
        if time == times[i1] {
            return Some(single(i1));
        }
        let (lo, hi) = match method {
            TimeInterpolation::Linear => (i0, i1),
            TimeInterpolation::Cubic => (i0.saturating_sub(1), (i1 + 1).min(n - 1)),
        };
        let mut stencil = Self {
            idx: [0; 4],
            w: [0.0; 4],
            len: hi - lo + 1,
        };
        for (s, a) in (lo..=hi).enumerate() {
            let w: f64 = (lo..=hi)
                .filter(|&b| b != a)
                .map(|b| (time - times[b]) / (times[a] - times[b]))
                .product();
            stencil.idx[s] = a;
            stencil.w[s] = w;
        }
        Some(stencil)
    }

    /// The used `(index, weight)` pairs.
    #[inline]
    pub fn terms(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.idx[..self.len]
            .iter()
            .copied()
            .zip(self.w[..self.len].iter().copied())
    }
}

/// A 2D field per snapshot, strided `[time][point]`; NaN marks missing
/// values (land, fill values).
#[derive(Clone, Debug, PartialEq)]
pub struct FieldSeries {
    n_points: usize,
    offset: f64,
    data: Vec<f32>,
}

impl FieldSeries {
    /// Series from values strided `[time][point]` (`data.len()` a multiple
    /// of `n_points`).
    pub fn new(n_points: usize, data: Vec<f32>) -> Self {
        assert!(
            n_points > 0 && data.len().is_multiple_of(n_points),
            "FieldSeries: {} values for {n_points} points per snapshot",
            data.len()
        );
        Self {
            n_points,
            offset: 0.0,
            data,
        }
    }

    /// Series from `f64` values, stored as `offset + f32(value − offset)`.
    pub fn with_offset(n_points: usize, values: &[f64], offset: f64) -> Self {
        let data = values.iter().map(|&v| (v - offset) as f32).collect();
        Self {
            offset,
            ..Self::new(n_points, data)
        }
    }

    /// Points per snapshot.
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Number of snapshots.
    pub fn n_times(&self) -> usize {
        self.data.len() / self.n_points
    }

    /// The constant added to every stored value.
    pub fn offset(&self) -> f64 {
        self.offset
    }

    /// Stored values of snapshot `t` (without the offset).
    #[inline]
    pub fn snapshot(&self, t: usize) -> &[f32] {
        &self.data[t * self.n_points..(t + 1) * self.n_points]
    }

    /// Value at snapshot `t`, point `k` (NaN if missing).
    #[inline]
    pub fn get(&self, t: usize, k: usize) -> f64 {
        self.offset + self.data[t * self.n_points + k] as f64
    }

    /// Whether point `k` has a value at every snapshot.
    pub fn is_valid(&self, k: usize) -> bool {
        (0..self.n_times()).all(|t| self.data[t * self.n_points + k].is_finite())
    }

    /// `Σ_t w_t Σ_c w_c f[t][c]` for spatial `(index, weight)` pairs at the
    /// snapshots of `time`. The spatial weights need not sum to 1 (gradient
    /// weights sum to 0).
    #[inline]
    pub fn interpolate(
        &self,
        time: &TimeStencil,
        space: impl Iterator<Item = (usize, f64)> + Clone,
    ) -> f64 {
        let mut sum = 0.0;
        for (t, wt) in time.terms() {
            let snapshot = self.snapshot(t);
            let value: f64 = space.clone().map(|(k, w)| w * snapshot[k] as f64).sum();
            sum += wt * value;
        }
        if self.offset == 0.0 {
            sum
        } else {
            sum + self.offset * space.map(|(_, w)| w).sum::<f64>()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_stencil_brackets_without_clamping() {
        let times = [0.0, 10.0, 30.0];
        let s = TimeStencil::new(&times, 20.0, TimeInterpolation::Linear).unwrap();
        assert_eq!(s.terms().collect::<Vec<_>>(), vec![(1, 0.5), (2, 0.5)]);
        assert_eq!(
            TimeStencil::new(&times, 30.0, TimeInterpolation::Linear)
                .unwrap()
                .terms()
                .collect::<Vec<_>>(),
            vec![(2, 1.0)]
        );
        assert!(TimeStencil::new(&times, -1e-9, TimeInterpolation::Linear).is_none());
        assert!(TimeStencil::new(&times, 30.1, TimeInterpolation::Cubic).is_none());
        // One snapshot: constant
        assert!(TimeStencil::new(&[5.0], 1e9, TimeInterpolation::Cubic).is_some());
    }

    /// Cubic interpolation is exact for cubics inside, quadratics at the
    /// ends, on non-uniform times.
    #[test]
    fn cubic_stencil_is_exact_for_polynomials() {
        let times = [0.0, 1.0, 2.5, 3.0, 5.0, 6.0];
        let cubic = |t: f64| 1.0 - 2.0 * t + 0.3 * t * t - 0.05 * t * t * t;
        let quadratic = |t: f64| 1.0 - 2.0 * t + 0.3 * t * t;
        for t in [1.2, 2.0, 2.9, 4.1] {
            let s = TimeStencil::new(&times, t, TimeInterpolation::Cubic).unwrap();
            assert_eq!(s.len, 4);
            let v: f64 = s.terms().map(|(k, w)| w * cubic(times[k])).sum();
            assert!((v - cubic(t)).abs() < 1e-12, "t = {t}");
        }
        for t in [0.4, 5.5] {
            let s = TimeStencil::new(&times, t, TimeInterpolation::Cubic).unwrap();
            assert_eq!(s.len, 3);
            let v: f64 = s.terms().map(|(k, w)| w * quadratic(times[k])).sum();
            assert!((v - quadratic(t)).abs() < 1e-12, "t = {t}");
        }
    }

    /// An hourly M2 series: cubic keeps the error below 0.2 % of the
    /// amplitude, linear does not.
    #[test]
    fn hourly_m2_interpolation_error() {
        let omega = 2.0 * std::f64::consts::PI / (12.4206 * 3600.0);
        let times: Vec<f64> = (0..48).map(|h| 3600.0 * h as f64).collect();
        let values: Vec<f64> = times.iter().map(|&t| (omega * t).cos()).collect();
        let max_error = |method| {
            (0..2000)
                .map(|s| 5.0 * 3600.0 + 30.0 * 3600.0 * s as f64 / 2000.0)
                .map(|t| {
                    let s = TimeStencil::new(&times, t, method).unwrap();
                    let v: f64 = s.terms().map(|(k, w)| w * values[k]).sum();
                    (v - (omega * t).cos()).abs()
                })
                .fold(0.0, f64::max)
        };
        let (linear, cubic) = (
            max_error(TimeInterpolation::Linear),
            max_error(TimeInterpolation::Cubic),
        );
        assert!(linear > 0.02, "linear {linear}");
        assert!(cubic < 2e-3, "cubic {cubic}");
    }

    #[test]
    fn series_interpolates_in_space_and_time_with_offset() {
        let p = [101_300.0, 101_310.0, 101_320.0, 101_330.0];
        let series = FieldSeries::with_offset(2, &p, 101_325.0);
        assert_eq!(series.n_times(), 2);
        assert_eq!(series.get(1, 1), 101_330.0);
        let s = TimeStencil::new(&[0.0, 10.0], 5.0, TimeInterpolation::Linear).unwrap();
        let v = series.interpolate(&s, [(0, 0.5), (1, 0.5)].into_iter());
        assert!((v - 101_315.0).abs() < 1e-9);
        let difference = series.interpolate(&s, [(0, -1.0), (1, 1.0)].into_iter());
        assert!((difference - 10.0).abs() < 1e-9);
    }
}
