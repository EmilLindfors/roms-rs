//! Boundary spectra from a parent wave model (TODO F.4 forcing).
//!
//! A regional wave model publishes the 2D spectrum at a set of points every
//! hour or so: MET Norway's MyWave WAM 800 m, for one, at 32 points along
//! Midt-Norge (36 frequencies from 0.035 Hz in steps of 10 %, 36 directions
//! of 10°; read by [`crate::io::WaveSpectraFile`]). [`BoundarySpectra`]
//! carries them onto the open boundary of a [`WaveModel2D`]:
//!
//! - **On the model's spectral grid** ([`regrid_spectrum`]): the density is
//!   interpolated linearly in `ln f` and periodically in direction, converted
//!   from per hertz to per rad/s, and scaled so that its variance is the
//!   parent's over the model's frequency band (the parent's bins clipped to
//!   the band). The parent's H_s within the band is kept exactly, and the
//!   shape to the interpolation's accuracy.
//! - **In space**: each open-boundary node takes Shepard's inverse-distance
//!   mean (power 2) of its nearest parent points, the spectra as they are (no
//!   shift of the peak between points: a sum of the neighbours' seas).
//! - **In time**: linear between the parent's times, constant before the
//!   first and after the last.
//!
//! [`BoundarySpectra::apply`] sets the spectra at a time on the model
//! ([`WaveModel2D::set_boundary_spectra`]); a run calls it every step
//! ([`super::CoupledWaves2D::with_boundary`]).

use std::f64::consts::TAU;

use super::model::WaveModel2D;
use super::sources::Wind;
use super::spectrum::SpectralGrid;

/// 2D variance spectra of a parent wave model at points, over time.
#[derive(Clone, Debug)]
pub struct PointSpectra {
    /// Point positions in the model's mesh coordinates (m)
    pub positions: Vec<[f64; 2]>,
    /// Times (s of model time), ascending
    pub times: Vec<f64>,
    /// Frequencies (Hz), ascending
    pub frequencies: Vec<f64>,
    /// Directions the waves travel to (rad, counter-clockwise from the
    /// mesh's x), uniformly spaced around the circle in any order
    pub directions: Vec<f64>,
    /// Variance density (m²/(Hz·rad)), `[time][point][frequency][direction]`
    pub density: Vec<f64>,
}

impl PointSpectra {
    /// Components of one spectrum (frequencies × directions).
    pub fn n_components(&self) -> usize {
        self.frequencies.len() * self.directions.len()
    }

    /// The spectrum of point `point` at time index `time`.
    pub fn spectrum(&self, time: usize, point: usize) -> &[f64] {
        let nc = self.n_components();
        let start = (time * self.positions.len() + point) * nc;
        &self.density[start..start + nc]
    }

    fn check(&self) {
        let nc = self.n_components();
        assert!(!self.positions.is_empty(), "no points");
        assert!(!self.times.is_empty(), "no times");
        assert!(
            self.times.windows(2).all(|w| w[1] > w[0]),
            "times must ascend"
        );
        assert!(
            self.frequencies.windows(2).all(|w| w[1] > w[0]),
            "frequencies must ascend"
        );
        assert!(self.directions.len() >= 4, "too few directions");
        assert_eq!(
            self.density.len(),
            self.times.len() * self.positions.len() * nc,
            "one spectrum per time and point"
        );
    }
}

/// The spectrum `density` (m²/(Hz·rad), `[frequency][direction]` on
/// `frequencies` in Hz and `directions` in rad, the directions waves travel
/// to, uniformly spaced in any order) on `grid`, as variance density per
/// rad/s and radian (see the module docs). Non-finite or negative values
/// (fill values) count as 0.
pub fn regrid_spectrum(
    grid: &SpectralGrid,
    frequencies: &[f64],
    directions: &[f64],
    density: &[f64],
) -> Vec<f64> {
    let (nf_src, nd_src) = (frequencies.len(), directions.len());
    assert_eq!(density.len(), nf_src * nd_src);
    let value = |i: usize, j: usize| {
        let x = density[i * nd_src + j];
        if x.is_finite() && x > 0.0 { x } else { 0.0 }
    };
    // The source directions sorted on [0, 2π), for the periodic interpolation
    let mut order: Vec<(f64, usize)> = directions
        .iter()
        .enumerate()
        .map(|(j, &d)| (d.rem_euclid(TAU), j))
        .collect();
    order.sort_by(|a, b| a.0.total_cmp(&b.0));
    let d_theta_src = TAU / nd_src as f64;
    let bracket_direction = |theta: f64| {
        let t = theta.rem_euclid(TAU);
        let upper = order.partition_point(|&(d, _)| d <= t);
        let (lo, hi) = (order[(upper + nd_src - 1) % nd_src], order[upper % nd_src]);
        let span = (hi.0 - lo.0).rem_euclid(TAU);
        let span = if span == 0.0 { TAU } else { span };
        let w = (t - lo.0).rem_euclid(TAU) / span;
        (lo.1, hi.1, w)
    };
    let ln_f: Vec<f64> = frequencies.iter().map(|f| f.ln()).collect();
    let (nf, nd) = (grid.n_freq(), grid.n_dir());
    let mut e = vec![0.0; nf * nd];
    for i in 0..nf {
        let f = grid.sigma[i] / TAU;
        let lf = f.ln();
        if lf < ln_f[0] || lf > ln_f[nf_src - 1] {
            continue;
        }
        let upper = ln_f.partition_point(|&x| x <= lf).min(nf_src - 1).max(1);
        let (a, b) = (upper - 1, upper);
        let wf = ((lf - ln_f[a]) / (ln_f[b] - ln_f[a])).clamp(0.0, 1.0);
        for j in 0..nd {
            let (lo, hi, wd) = bracket_direction(grid.theta[j]);
            let at = |i: usize| (1.0 - wd) * value(i, lo) + wd * value(i, hi);
            // Per hertz to per rad/s
            e[i * nd + j] = ((1.0 - wf) * at(a) + wf * at(b)) / TAU;
        }
    }
    // The parent's variance over the model's band: its bins (edges at the
    // geometric means of neighbouring frequencies) clipped to the band
    let ratio = grid.frequency_ratio().sqrt();
    let (band_lo, band_hi) = (
        grid.sigma[0] / TAU / ratio,
        grid.sigma[nf - 1] / TAU * ratio,
    );
    let edge = |i: usize| -> f64 {
        if i == 0 {
            frequencies[0] * (frequencies[0] / frequencies[1]).sqrt()
        } else if i == nf_src {
            frequencies[nf_src - 1] * (frequencies[nf_src - 1] / frequencies[nf_src - 2]).sqrt()
        } else {
            (frequencies[i - 1] * frequencies[i]).sqrt()
        }
    };
    let mut parent = 0.0;
    for i in 0..nf_src {
        let width = (edge(i + 1).min(band_hi) - edge(i).max(band_lo)).max(0.0);
        if width > 0.0 {
            parent += width * d_theta_src * (0..nd_src).map(|j| value(i, j)).sum::<f64>();
        }
    }
    let ours = grid.moment(&e, 0);
    if ours > 0.0 {
        let scale = parent / ours;
        e.iter_mut().for_each(|x| *x *= scale);
    }
    e
}

/// A parent model's spectra on the open boundary of a wave model (see the
/// module docs).
#[derive(Clone, Debug)]
pub struct BoundarySpectra {
    times: Vec<f64>,
    n_points: usize,
    n_components: usize,
    /// The parent's spectra on the model's grid, `[time][point][component]`
    spectra: Vec<f64>,
    /// Per target, its parent points and weights (summing to 1)
    weights: Vec<Vec<(usize, f64)>>,
}

impl BoundarySpectra {
    /// The spectra of `parent` on `grid` at the points `targets` (mesh
    /// coordinates, m), each the inverse-distance mean of its `neighbours`
    /// nearest parent points.
    pub fn new(
        grid: &SpectralGrid,
        parent: &PointSpectra,
        targets: &[[f64; 2]],
        neighbours: usize,
    ) -> Self {
        parent.check();
        assert!(neighbours >= 1, "at least one neighbour");
        let (nt, np, nc) = (
            parent.times.len(),
            parent.positions.len(),
            grid.n_components(),
        );
        let mut spectra = Vec::with_capacity(nt * np * nc);
        for t in 0..nt {
            for p in 0..np {
                spectra.extend(regrid_spectrum(
                    grid,
                    &parent.frequencies,
                    &parent.directions,
                    parent.spectrum(t, p),
                ));
            }
        }
        let weights = targets
            .iter()
            .map(|&[x, y]| {
                let mut near: Vec<(usize, f64)> = parent
                    .positions
                    .iter()
                    .enumerate()
                    .map(|(p, &[px, py])| (p, (px - x).hypot(py - y)))
                    .collect();
                near.sort_by(|a, b| a.1.total_cmp(&b.1));
                near.truncate(neighbours);
                // On a parent point, that point alone
                if let Some(&(p, _)) = near.iter().find(|&&(_, d)| d < 1e-9) {
                    return vec![(p, 1.0)];
                }
                let total: f64 = near.iter().map(|&(_, d)| d.powi(-2)).sum();
                near.iter().map(|&(p, d)| (p, d.powi(-2) / total)).collect()
            })
            .collect();
        Self {
            times: parent.times.clone(),
            n_points: np,
            n_components: nc,
            spectra,
            weights,
        }
    }

    /// The spectra of `parent` on the open boundary of `model`
    /// ([`WaveModel2D::open_boundary_points`]), each node the mean of its
    /// `neighbours` nearest parent points.
    pub fn for_model(model: &WaveModel2D, parent: &PointSpectra, neighbours: usize) -> Self {
        let nn = model.ops.n_nodes;
        let targets: Vec<[f64; 2]> = model
            .open_boundary_points()
            .into_iter()
            .map(|p| {
                let k = crate::types::ElementIndex::new(p / nn);
                let i = p % nn;
                model
                    .mesh
                    .reference_to_physical(k, model.ops.nodes_r[i], model.ops.nodes_s[i])
            })
            .collect();
        Self::new(&model.grid, parent, &targets, neighbours)
    }

    /// Number of targets.
    pub fn n_targets(&self) -> usize {
        self.weights.len()
    }

    /// The parent's times (s of model time).
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// The parent's spectrum of point `point` at time `t`, on the model's
    /// grid (linear in time), into `out`.
    pub fn parent_at_into(&self, point: usize, t: f64, out: &mut [f64]) {
        let ((a, wa), (b, wb)) = self.bracket(t);
        let nc = self.n_components;
        let (sa, sb) = (
            &self.spectra[(a * self.n_points + point) * nc..][..nc],
            &self.spectra[(b * self.n_points + point) * nc..][..nc],
        );
        for ((o, x), y) in out.iter_mut().zip(sa).zip(sb) {
            *o = wa * x + wb * y;
        }
    }

    /// Every target's spectrum at time `t` into `out` (`[target][component]`).
    pub fn at_into(&self, t: f64, out: &mut [f64]) {
        let nc = self.n_components;
        assert_eq!(out.len(), self.n_targets() * nc);
        let ((a, wa), (b, wb)) = self.bracket(t);
        out.fill(0.0);
        for (target, weights) in out.chunks_exact_mut(nc).zip(&self.weights) {
            for &(p, w) in weights {
                let (sa, sb) = (
                    &self.spectra[(a * self.n_points + p) * nc..][..nc],
                    &self.spectra[(b * self.n_points + p) * nc..][..nc],
                );
                for ((o, x), y) in target.iter_mut().zip(sa).zip(sb) {
                    *o += w * (wa * x + wb * y);
                }
            }
        }
    }

    /// Every target's spectrum at time `t`.
    pub fn at(&self, t: f64) -> Vec<f64> {
        let mut out = vec![0.0; self.n_targets() * self.n_components];
        self.at_into(t, &mut out);
        out
    }

    /// Set the spectra at time `t` on `model`, whose open-boundary points
    /// these are ([`Self::for_model`]).
    pub fn apply(&self, model: &mut WaveModel2D, t: f64) {
        model.set_boundary_spectra(&self.at(t));
    }

    fn bracket(&self, t: f64) -> ((usize, f64), (usize, f64)) {
        bracket(&self.times, t)
    }
}

/// A parent model's wind over time, the same everywhere (e.g. the mean of
/// its points' wind), for [`WaveModel2D::set_wind`]: linear in time in the
/// vector's components, constant before the first time and after the last.
#[derive(Clone, Debug)]
pub struct WindSeries {
    times: Vec<f64>,
    /// The wind vector (m/s, mesh axes) at each time
    vectors: Vec<[f64; 2]>,
}

impl WindSeries {
    /// The wind `winds[k]` at time `times[k]` (s of model time, ascending).
    pub fn new(times: Vec<f64>, winds: &[Wind]) -> Self {
        assert_eq!(times.len(), winds.len(), "one wind per time");
        assert!(!times.is_empty(), "no times");
        assert!(times.windows(2).all(|w| w[1] > w[0]), "times must ascend");
        let vectors = winds
            .iter()
            .map(|w| [w.u10 * w.direction.cos(), w.u10 * w.direction.sin()])
            .collect();
        Self { times, vectors }
    }

    /// The wind at time `t`.
    pub fn at(&self, t: f64) -> Wind {
        let ((a, wa), (b, wb)) = bracket(&self.times, t);
        let [u, v] = [0, 1].map(|d| wa * self.vectors[a][d] + wb * self.vectors[b][d]);
        Wind {
            u10: u.hypot(v),
            direction: v.atan2(u),
        }
    }
}

/// The indices of `times` (ascending) around `t` and their weights: linear
/// between them, the end's beyond the ends.
fn bracket(times: &[f64], t: f64) -> ((usize, f64), (usize, f64)) {
    let upper = times.partition_point(|&x| x <= t);
    if upper == 0 {
        return ((0, 1.0), (0, 0.0));
    }
    if upper == times.len() {
        let last = times.len() - 1;
        return ((last, 1.0), (last, 0.0));
    }
    let (a, b) = (upper - 1, upper);
    let w = (t - times[a]) / (times[b] - times[a]);
    ((a, 1.0 - w), (b, w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// A JONSWAP-like spectrum (m²/(Hz·rad)) on MET's WAM 800 m grid: 36
    /// frequencies from 0.0345 Hz × 1.1ⁱ, 36 directions travelling to
    /// `90° − (5° + 10° j)` (their oceanographic "to" directions, as the
    /// reader converts them).
    fn parent_grid() -> (Vec<f64>, Vec<f64>) {
        let f = (0..36).map(|i| 0.034523 * 1.1f64.powi(i)).collect();
        let d = (0..36)
            .map(|j| (90.0 - (5.0 + 10.0 * j as f64)).to_radians())
            .collect();
        (f, d)
    }

    fn sea(f: &[f64], d: &[f64], fp: f64, mean: f64, m: i32) -> Vec<f64> {
        let mut e = Vec::with_capacity(f.len() * d.len());
        for &f in f {
            let width = if f <= fp { 0.07 } else { 0.09 };
            let r = (-(f - fp).powi(2) / (2.0 * width * width * fp * fp)).exp();
            let s = f.powi(-5) * (-1.25 * (f / fp).powi(-4)).exp() * 3.3f64.powf(r);
            for &t in d {
                e.push(s * (t - mean).cos().max(0.0).powi(m));
            }
        }
        e
    }

    /// A parent sea on MET's grid keeps its H_s within the model's band
    /// exactly, its mean direction to 0.1° and its peak to a bin.
    #[test]
    fn a_parent_spectrum_on_the_model_grid_keeps_its_sea() {
        let (f, d) = parent_grid();
        let grid = SpectralGrid::new(0.04, 0.5, 25, 36);
        for (fp, mean_deg, m) in [(0.08, 105.0f64, 10), (0.12, -30.0, 4), (0.2, 200.0, 2)] {
            let mean = mean_deg.to_radians();
            let src = sea(&f, &d, fp, mean, m);
            let e = regrid_spectrum(&grid, &f, &d, &src);
            // The parent's variance over the band, by its own bins
            let mut band = 0.0;
            let (lo, hi) = (
                grid.sigma[0] / TAU / grid.frequency_ratio().sqrt(),
                grid.sigma[24] / TAU * grid.frequency_ratio().sqrt(),
            );
            for i in 0..36 {
                let a = if i == 0 {
                    f[0] / 1.1f64.sqrt()
                } else {
                    (f[i - 1] * f[i]).sqrt()
                };
                let b = if i == 35 {
                    f[35] * 1.1f64.sqrt()
                } else {
                    (f[i] * f[i + 1]).sqrt()
                };
                let w = (b.min(hi) - a.max(lo)).max(0.0);
                band += w * (TAU / 36.0) * src[i * 36..(i + 1) * 36].iter().sum::<f64>();
            }
            let params = grid.parameters(&e);
            let hs = 4.0 * band.sqrt();
            assert!(
                (params.hs / hs - 1.0).abs() < 1e-12,
                "H_s {} against {hs}",
                params.hs
            );
            let off = (params.direction - mean + PI).rem_euclid(TAU) - PI;
            assert!(
                off.abs().to_degrees() < 0.1,
                "direction off by {}°",
                off.to_degrees()
            );
            assert!(
                (params.tp * fp - 1.0).abs() < grid.frequency_ratio() - 1.0,
                "T_p {} for f_p {fp}",
                params.tp
            );
        }
    }

    /// On its own grid a spectrum comes back as it is (per rad/s).
    #[test]
    fn on_the_same_grid_the_spectrum_is_kept() {
        let grid = SpectralGrid::new(0.05, 0.4, 12, 24);
        let f: Vec<f64> = grid.sigma.iter().map(|s| s / TAU).collect();
        let e = grid.jonswap(2.0, 9.0, 3.3, 0.7, 6.0);
        let per_hz: Vec<f64> = e.iter().map(|x| x * TAU).collect();
        let back = regrid_spectrum(&grid, &f, &grid.theta, &per_hz);
        let scale = e.iter().cloned().fold(0.0, f64::max);
        for (a, b) in back.iter().zip(&e) {
            assert!((a - b).abs() < 1e-12 * scale, "{a} against {b}");
        }
    }

    /// Between two points a target takes the inverse-distance mean, on a
    /// point the point's own spectrum; between two times the linear mean,
    /// before the first and after the last the end's.
    #[test]
    fn the_boundary_takes_the_nearest_parents_in_space_and_time() {
        let grid = SpectralGrid::new(0.05, 0.4, 8, 12);
        let f: Vec<f64> = grid.sigma.iter().map(|s| s / TAU).collect();
        let nc = grid.n_components();
        let a = grid.jonswap(1.0, 8.0, 3.3, 0.0, 4.0);
        let b = grid.jonswap(3.0, 12.0, 3.3, 1.5, 8.0);
        let per_hz = |e: &[f64], s: f64| e.iter().map(|x| s * x * TAU).collect::<Vec<_>>();
        // Point 0 has `a`, point 1 `b`, at t = 0; twice as much at t = 3600
        let mut density = per_hz(&a, 1.0);
        density.extend(per_hz(&b, 1.0));
        density.extend(per_hz(&a, 2.0));
        density.extend(per_hz(&b, 2.0));
        let parent = PointSpectra {
            positions: vec![[0.0, 0.0], [3000.0, 0.0]],
            times: vec![0.0, 3600.0],
            frequencies: f,
            directions: grid.theta.clone(),
            density,
        };
        let targets = [[0.0, 0.0], [1000.0, 0.0], [3000.0, 0.0]];
        let boundary = BoundarySpectra::new(&grid, &parent, &targets, 2);
        // 1 km from a, 2 km from b: weights 4/5 and 1/5
        for (t, scale) in [
            (-100.0, 1.0),
            (0.0, 1.0),
            (1800.0, 1.5),
            (3600.0, 2.0),
            (9e9, 2.0),
        ] {
            let out = boundary.at(t);
            for c in 0..nc {
                let close = |x: f64, y: f64| (x - y).abs() <= 1e-12 * (1.0 + y.abs());
                assert!(close(out[c], scale * a[c]));
                assert!(close(out[nc + c], scale * (0.8 * a[c] + 0.2 * b[c])));
                assert!(close(out[2 * nc + c], scale * b[c]));
            }
        }
    }
}
