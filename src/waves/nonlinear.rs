//! Four-wave (quadruplet) interactions by the discrete interaction approximation
//! (DIA; Hasselmann et al. 1985, "Computations and parameterizations of the
//! nonlinear energy transfer in a gravity-wave spectrum, Part II"), as in WAM and
//! SWAN (the SWAN Scientific Documentation, §2.3.3).
//!
//! The Boltzmann integral is replaced by one quadruplet per component and its
//! mirror image: two waves of the component's frequency σ and direction θ
//! interact with `σ₊ = (1 + λ)σ` at `θ + Δθ₊` and `σ₋ = (1 − λ)σ` at `θ + Δθ₋`,
//! λ = 0.25. The angles close the deep-water resonance `2k = k₊ + k₋`
//! (`Δθ₊ = −11.48°`, `Δθ₋ = 33.56°`, and the mirror `+11.48°`, `−33.56°`). Each
//! quadruplet transfers, in variance density `E(σ, θ)`,
//!
//! ```text
//! (δS, δS₊, δS₋) = (−2, 1, 1) C (2π)² g⁻⁴ (σ/2π)¹¹
//!     [E² (E₊/(1 + λ)⁴ + E₋/(1 − λ)⁴) − 2 E E₊ E₋/(1 − λ²)⁴],  C = 3·10⁷
//! ```
//!
//! (Komen et al. 1994, §3.6, with F(f, θ) = 2π E(σ, θ)). Losing two waves at σ and
//! gaining one at each of σ₊ and σ₋ keeps both energy and action: the volume
//! `Δσ` at σ maps to `(1 ± λ)Δσ` at σ±. Off-grid frequencies are interpolated
//! from (and the gains scattered back to) the two neighbouring bins with weights
//! linear in 1/σ, the only linear weights that keep energy and action both exact on
//! the discrete grid; directions linearly. Above the grid the spectrum is read as
//! the diagnostic tail `E(σ_max)(σ_max/σ)^p`, and transfers beyond either end of the
//! grid leave the prognostic spectrum.
//!
//! In finite depth the deep-water rate is scaled by
//! `R(k_p d) = 1 + (5.5/k_p d)(1 − 5/6 k_p d) exp(−5/4 k_p d)`, `k_p = 0.75 k̃`,
//! with `k_p d ≥ 0.5` (`R ≤ 4.43`; Hasselmann & Hasselmann 1981, as SWAN).

use std::f64::consts::TAU;

use super::spectrum::SpectralGrid;

/// Coefficients of the DIA.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quadruplets {
    /// Proportionality constant `C_nl4`
    pub c_nl4: f64,
    /// Frequency separation λ of the quadruplet
    pub lambda: f64,
}

impl Default for Quadruplets {
    /// WAM's and SWAN's `C_nl4 = 3·10⁷`, λ = 0.25.
    fn default() -> Self {
        Self {
            c_nl4: 3.0e7,
            lambda: 0.25,
        }
    }
}

/// Where one side of a quadruplet lands on the grid: the lower of the two bins it
/// falls between, as an offset, and the two bins' weights.
#[derive(Clone, Copy, Debug)]
struct Landing {
    offset: isize,
    weights: [f64; 2],
}

impl Landing {
    /// Frequency `σ γ^x` between bins `⌊x⌋` and `⌊x⌋ + 1`, weights linear in 1/σ.
    fn frequency(x: f64, gamma: f64) -> Self {
        let lo = x.floor();
        let inv = |y: f64| gamma.powf(-y);
        let w = (inv(x) - inv(lo + 1.0)) / (inv(lo) - inv(lo + 1.0));
        Self {
            offset: lo as isize,
            weights: [w, 1.0 - w],
        }
    }

    /// Direction `θ + x Δθ`, weights linear in θ.
    fn direction(x: f64) -> Self {
        let lo = x.floor();
        let frac = x - lo;
        Self {
            offset: lo as isize,
            weights: [1.0 - frac, frac],
        }
    }
}

/// The quadruplet's landings on a spectral grid.
#[derive(Clone, Copy, Debug)]
struct Stencil {
    plus: Landing,
    minus: Landing,
    /// Direction landings of σ₊ and σ₋ for the quadruplet and its mirror image
    dir_plus: [Landing; 2],
    dir_minus: [Landing; 2],
    /// `(1 ± λ) γ^(−offset)` for the bin σ_b: the volume ratio of the gain at σ±
    /// over the receiving bin's, `(1 ± λ) Δσ_i / Δσ_b`
    volume_plus: [f64; 2],
    volume_minus: [f64; 2],
}

impl Quadruplets {
    /// The deep-water resonance angles `(Δθ₊, Δθ₋)` (rad) of the quadruplet,
    /// `2k = k₊ + k₋` with `|k±| = (1 ± λ)² k`: (−11.48°, 33.56°) for λ = 0.25.
    pub fn angles(&self) -> (f64, f64) {
        let (a, b) = ((1.0 + self.lambda).powi(2), (1.0 - self.lambda).powi(2));
        let plus = ((4.0 + a * a - b * b) / (4.0 * a)).acos();
        let minus = ((4.0 + b * b - a * a) / (4.0 * b)).acos();
        (-plus, minus)
    }

    fn stencil(&self, grid: &SpectralGrid) -> Stencil {
        let gamma = grid.frequency_ratio();
        let ln_gamma = gamma.ln();
        let plus = Landing::frequency((1.0 + self.lambda).ln() / ln_gamma, gamma);
        let minus = Landing::frequency((1.0 - self.lambda).ln() / ln_gamma, gamma);
        let (theta_plus, theta_minus) = self.angles();
        let d = grid.d_theta;
        let volume = |side: Landing, factor: f64| {
            [0, 1].map(|b| factor * gamma.powi(-(side.offset as i32 + b)))
        };
        Stencil {
            plus,
            minus,
            dir_plus: [
                Landing::direction(theta_plus / d),
                Landing::direction(-theta_plus / d),
            ],
            dir_minus: [
                Landing::direction(theta_minus / d),
                Landing::direction(-theta_minus / d),
            ],
            volume_plus: volume(plus, 1.0 + self.lambda),
            volume_minus: volume(minus, 1.0 - self.lambda),
        }
    }

    /// The deep-water transfer `s[c]` (m²/(rad/s)/rad/s, overwritten) of the
    /// variance density `e[c]`, scaled by `scale` (the finite-depth factor), with
    /// the spectrum continued above the grid as `σ^(−tail)` (none: zero there).
    pub fn source(
        &self,
        grid: &SpectralGrid,
        e: &[f64],
        tail: Option<f64>,
        g: f64,
        scale: f64,
        s: &mut [f64],
    ) {
        let (nf, nd) = (grid.n_freq() as isize, grid.n_dir() as isize);
        s.fill(0.0);
        let st = self.stencil(grid);
        let gamma = grid.frequency_ratio();
        let lam = self.lambda;
        let (w_plus, w_minus, w_both) = (
            (1.0 + lam).powi(-4),
            (1.0 - lam).powi(-4),
            2.0 * (1.0 - lam * lam).powi(-4),
        );
        let constant = scale * self.c_nl4 * TAU * TAU / g.powi(4);
        DIA_SCRATCH.with_borrow_mut(|scratch| {
            // E with the rows the stencil reaches below the grid (zero) and
            // above it (the tail), and each direction landing's bins on the
            // circle: plain loads in the loops below (bounds, wrapping and the
            // tail at every access were most of the DIA's cost)
            let (lo, hi) = (
                st.plus.offset.min(st.minus.offset).min(0),
                (nf + st.plus.offset.max(st.minus.offset) + 1).max(nf),
            );
            let ext = &mut scratch.extended;
            ext.clear();
            for i in lo..hi {
                for j in 0..nd {
                    ext.push(if i < 0 {
                        0.0
                    } else if i < nf {
                        e[(i * nd + j) as usize]
                    } else {
                        match tail {
                            Some(p) => {
                                e[((nf - 1) * nd + j) as usize]
                                    * gamma.powf(-p * (i - nf + 1) as f64)
                            }
                            None => 0.0,
                        }
                    });
                }
            }
            let landings = [
                st.dir_plus[0],
                st.dir_plus[1],
                st.dir_minus[0],
                st.dir_minus[1],
            ];
            let bins = &mut scratch.bins;
            bins.clear();
            for d in landings {
                for b in 0..2 {
                    bins.extend((0..nd).map(|j| (j + d.offset + b).rem_euclid(nd) as usize));
                }
            }
            let (ext, bins) = (&*ext, &*bins);
            let nd_u = nd as usize;
            // The bins of landing `l` (0, 1: σ₊ and its mirror; 2, 3: σ₋) at
            // offset `b` from direction `j`
            let bin = |l: usize, b: usize, j: usize| bins[(2 * l + b) * nd_u + j];
            let gather = |i: isize, j: usize, f: Landing, l: usize, d: Landing| -> f64 {
                let mut sum = 0.0;
                for (a, wf) in f.weights.iter().enumerate() {
                    let row = (i + f.offset + a as isize - lo) as usize * nd_u;
                    for (b, wd) in d.weights.iter().enumerate() {
                        sum += wf * wd * ext[row + bin(l, b, j)];
                    }
                }
                sum
            };
            let scatter = |s: &mut [f64],
                           i: isize,
                           j: usize,
                           f: Landing,
                           l: usize,
                           d: Landing,
                           vol: [f64; 2],
                           r: f64| {
                for (a, (wf, vol)) in f.weights.iter().zip(vol).enumerate() {
                    let ib = i + f.offset + a as isize;
                    if !(0..nf).contains(&ib) {
                        continue;
                    }
                    let gain = r * wf * vol;
                    let row = ib as usize * nd_u;
                    for (b, wd) in d.weights.iter().enumerate() {
                        s[row + bin(l, b, j)] += gain * wd;
                    }
                }
            };
            for i in 0..nf {
                let factor = constant * (grid.sigma[i as usize] / TAU).powi(11);
                for j in 0..nd_u {
                    let ec = e[i as usize * nd_u + j];
                    if ec <= 0.0 {
                        continue;
                    }
                    for m in 0..2 {
                        let (dp, dm) = (st.dir_plus[m], st.dir_minus[m]);
                        let ep = gather(i, j, st.plus, m, dp);
                        let em = gather(i, j, st.minus, 2 + m, dm);
                        let r =
                            factor * ec * (ec * (ep * w_plus + em * w_minus) - w_both * ep * em);
                        s[i as usize * nd_u + j] -= 2.0 * r;
                        scatter(s, i, j, st.plus, m, dp, st.volume_plus, r);
                        scatter(s, i, j, st.minus, 2 + m, dm, st.volume_minus, r);
                    }
                }
            }
        });
    }
}

/// Per-thread storage of [`Quadruplets::source`]: the spectrum extended by
/// the rows its stencil reaches, and the direction landings' bins.
#[derive(Default)]
struct DiaScratch {
    extended: Vec<f64>,
    bins: Vec<usize>,
}

thread_local! {
    static DIA_SCRATCH: std::cell::RefCell<DiaScratch> =
        std::cell::RefCell::new(DiaScratch::default());
}

/// The finite-depth scaling `R(k_p d)` of the DIA, `k_p d` floored at 0.5.
pub fn shallow_water_factor(kp_d: f64) -> f64 {
    let x = kp_d.max(0.5);
    1.0 + 5.5 / x * (1.0 - 5.0 / 6.0 * x) * (-1.25 * x).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 9.81;

    /// A JONSWAP sea with the ends of the grid emptied, so that every quadruplet
    /// with energy lands inside the grid.
    fn confined(grid: &SpectralGrid, tp: f64) -> Vec<f64> {
        let mut e = grid.jonswap(2.0, tp, 3.3, 0.4, 2.0);
        let (nf, nd) = (grid.n_freq(), grid.n_dir());
        for i in (0..5).chain(nf - 5..nf) {
            e[i * nd..(i + 1) * nd].fill(0.0);
        }
        e
    }

    #[test]
    fn the_angles_close_the_deep_water_resonance() {
        let q = Quadruplets::default();
        let (tp, tm) = q.angles();
        assert!(
            (tp.to_degrees() + 11.48).abs() < 5e-3,
            "{}",
            tp.to_degrees()
        );
        assert!(
            (tm.to_degrees() - 33.56).abs() < 5e-3,
            "{}",
            tm.to_degrees()
        );
        // 2k = k₊ + k₋ in both components
        let (a, b) = (1.25f64.powi(2), 0.75f64.powi(2));
        assert!((a * tp.cos() + b * tm.cos() - 2.0).abs() < 1e-14);
        assert!((a * tp.sin() + b * tm.sin()).abs() < 1e-14);
    }

    /// The transfer as it was computed before the stencil's tables (every
    /// access bounds-checked, wrapped and continued into the tail on the
    /// spot): the reference of `the_tables_give_the_direct_transfer`.
    fn direct_source(
        q: &Quadruplets,
        grid: &SpectralGrid,
        e: &[f64],
        tail: Option<f64>,
        scale: f64,
        s: &mut [f64],
    ) {
        let (nf, nd) = (grid.n_freq() as isize, grid.n_dir() as isize);
        s.fill(0.0);
        let st = q.stencil(grid);
        let gamma = grid.frequency_ratio();
        let lam = q.lambda;
        let (w_plus, w_minus, w_both) = (
            (1.0 + lam).powi(-4),
            (1.0 - lam).powi(-4),
            2.0 * (1.0 - lam * lam).powi(-4),
        );
        let constant = scale * q.c_nl4 * TAU * TAU / G.powi(4);
        let at = |i: isize, j: isize| -> f64 {
            let j = j.rem_euclid(nd);
            if i < 0 {
                0.0
            } else if i < nf {
                e[(i * nd + j) as usize]
            } else {
                match tail {
                    Some(p) => {
                        e[((nf - 1) * nd + j) as usize] * gamma.powf(-p * (i - nf + 1) as f64)
                    }
                    None => 0.0,
                }
            }
        };
        let gather = |i: isize, j: isize, f: Landing, d: Landing| -> f64 {
            let mut sum = 0.0;
            for (a, wf) in f.weights.iter().enumerate() {
                for (b, wd) in d.weights.iter().enumerate() {
                    sum += wf * wd * at(i + f.offset + a as isize, j + d.offset + b as isize);
                }
            }
            sum
        };
        let scatter =
            |s: &mut [f64], i: isize, j: isize, f: Landing, d: Landing, vol: [f64; 2], r: f64| {
                for (a, (wf, vol)) in f.weights.iter().zip(vol).enumerate() {
                    let ib = i + f.offset + a as isize;
                    if !(0..nf).contains(&ib) {
                        continue;
                    }
                    let gain = r * wf * vol;
                    for (b, wd) in d.weights.iter().enumerate() {
                        let jb = (j + d.offset + b as isize).rem_euclid(nd);
                        s[(ib * nd + jb) as usize] += gain * wd;
                    }
                }
            };
        for i in 0..nf {
            let factor = constant * (grid.sigma[i as usize] / TAU).powi(11);
            for j in 0..nd {
                let ec = e[(i * nd + j) as usize];
                if ec <= 0.0 {
                    continue;
                }
                for m in 0..2 {
                    let (dp, dm) = (st.dir_plus[m], st.dir_minus[m]);
                    let ep = gather(i, j, st.plus, dp);
                    let em = gather(i, j, st.minus, dm);
                    let r = factor * ec * (ec * (ep * w_plus + em * w_minus) - w_both * ep * em);
                    s[(i * nd + j) as usize] -= 2.0 * r;
                    scatter(s, i, j, st.plus, dp, st.volume_plus, r);
                    scatter(s, i, j, st.minus, dm, st.volume_minus, r);
                }
            }
        }
    }

    /// The transfer through the stencil's tables (the extended spectrum, the
    /// landings' bins) is the direct one bit for bit: with and without the
    /// tail, energy at the grid's ends, and directions so few that the
    /// landings wrap around the circle by more than a bin.
    #[test]
    fn the_tables_give_the_direct_transfer() {
        for (n_freq, n_dir, tail) in [(25, 36, Some(4.0)), (30, 24, None), (12, 6, Some(5.0))] {
            let grid = SpectralGrid::new(0.04, 0.5, n_freq, n_dir);
            let e = grid.jonswap(3.0, 4.0, 3.3, 1.0, 2.0);
            // Energy in the top row, which the tail continues
            assert!(e[(n_freq - 1) * n_dir..].iter().any(|&x| x > 0.0));
            let q = Quadruplets::default();
            let (mut fast, mut direct) = (vec![0.0; e.len()], vec![0.0; e.len()]);
            q.source(&grid, &e, tail, G, 1.3, &mut fast);
            direct_source(&q, &grid, &e, tail, 1.3, &mut direct);
            assert!(direct.iter().any(|&x| x != 0.0));
            assert_eq!(fast, direct, "{n_freq}×{n_dir}, tail {tail:?}");
        }
    }

    #[test]
    fn the_transfer_keeps_energy_and_action() {
        for (n_freq, n_dir) in [(36, 36), (30, 24)] {
            let grid = SpectralGrid::new(0.04, 1.0, n_freq, n_dir);
            let e = confined(&grid, 6.0);
            let mut s = vec![0.0; e.len()];
            Quadruplets::default().source(&grid, &e, Some(4.0), G, 1.0, &mut s);
            let gross = grid.moment(&s.iter().map(|x| x.abs()).collect::<Vec<_>>(), 0);
            let energy = grid.moment(&s, 0) / gross;
            let action = grid.moment(&s, -1)
                / grid.moment(&s.iter().map(|x| x.abs()).collect::<Vec<_>>(), -1);
            assert!(energy.abs() < 1e-13, "{n_freq}×{n_dir}: energy {energy:e}");
            assert!(action.abs() < 1e-13, "{n_freq}×{n_dir}: action {action:e}");
        }
    }

    #[test]
    fn a_jonswap_sea_gains_below_the_peak_and_loses_above() {
        // The classic shape of the transfer (Hasselmann et al. 1985): a
        // positive lobe on the forward face, a negative one just above the peak,
        // and a positive one at high frequencies.
        let grid = SpectralGrid::new(0.04, 1.0, 36, 36);
        let tp = 8.0;
        let e = grid.jonswap(2.0, tp, 3.3, 0.0, 2.0);
        let mut s = vec![0.0; e.len()];
        Quadruplets::default().source(&grid, &e, Some(4.0), G, 1.0, &mut s);
        let nd = grid.n_dir();
        let row = |i: usize| s[i * nd..(i + 1) * nd].iter().sum::<f64>();
        let at = |f: f64| {
            let i = (0..grid.n_freq())
                .min_by(|&a, &b| {
                    let d = |i: usize| (grid.sigma[i] / TAU - f).abs();
                    d(a).total_cmp(&d(b))
                })
                .unwrap();
            row(i)
        };
        let fp = 1.0 / tp;
        // The negative lobe is at 1.3–1.5 f_p here (the 36-bin grid of
        // γ ≈ 1.096 and the DIA's single quadruplet place it)
        assert!(at(0.8 * fp) > 0.0, "forward face {}", at(0.8 * fp));
        assert!(at(1.4 * fp) < 0.0, "above the peak {}", at(1.4 * fp));
        assert!(at(3.0 * fp) > 0.0, "high frequencies {}", at(3.0 * fp));
    }

    #[test]
    fn the_finite_depth_factor_runs_from_one_to_its_bound() {
        assert!((shallow_water_factor(0.5) - 4.43).abs() < 5e-3);
        assert_eq!(shallow_water_factor(0.1), shallow_water_factor(0.5));
        assert!((shallow_water_factor(30.0) - 1.0).abs() < 1e-14);
        assert!(shallow_water_factor(1.0) > shallow_water_factor(2.0));
    }
}
