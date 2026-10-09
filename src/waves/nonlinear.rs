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

#[cfg(feature = "simd")]
#[path = "dia_lanes.rs"]
mod lanes;
#[cfg(feature = "simd")]
pub(crate) use lanes::LANES;

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
        self.source_into(grid, e, tail, g, scale, s, None);
    }

    /// [`Self::source`] and its diagonal derivative `diag[c] = ∂s[c]/∂e[c]`
    /// (1/s, overwritten): the rate at which each component's transfer
    /// changes with its own energy, from the quadruplets it starts and those
    /// it receives from (bins above the grid, read from the tail, count as
    /// constant). Exact for grids whose frequency ratio is below `1 + λ`;
    /// on coarser ones a landing reaches the component's own row, whose
    /// share in its own gather the diagonal leaves out.
    pub fn source_and_diagonal(
        &self,
        grid: &SpectralGrid,
        e: &[f64],
        tail: Option<f64>,
        g: f64,
        scale: f64,
        s: &mut [f64],
        diag: &mut [f64],
    ) {
        self.source_into(grid, e, tail, g, scale, s, Some(diag));
    }

    fn source_into(
        &self,
        grid: &SpectralGrid,
        e: &[f64],
        tail: Option<f64>,
        g: f64,
        scale: f64,
        s: &mut [f64],
        mut diag: Option<&mut [f64]>,
    ) {
        let (nf, nd) = (grid.n_freq(), grid.n_dir());
        s.fill(0.0);
        if let Some(diag) = diag.as_deref_mut() {
            diag.fill(0.0);
        }
        let (w_plus, w_minus, w_both) = self.weights();
        let constant = scale * self.c_nl4 * TAU * TAU / g.powi(4);
        DIA_SCRATCH.with_borrow_mut(|scratch| {
            let DiaScratch {
                tables, extended, ..
            } = scratch;
            let t = Tables::cached(tables, self, grid, tail);
            let st = &t.stencil;
            // E with the rows the stencil reaches below the grid (zero) and
            // above it (the tail): plain loads in the loops below (bounds,
            // wrapping and the tail at every access were most of the DIA's
            // cost)
            t.extend(e, 1, extended);
            let ext = &*extended;
            let gather = |i: usize, j: usize, f: Landing, l: usize, d: Landing| -> f64 {
                let mut sum = 0.0;
                for (a, wf) in f.weights.iter().enumerate() {
                    let row = t.row(i, f, a) * nd;
                    for (b, wd) in d.weights.iter().enumerate() {
                        sum += wf * wd * ext[row + t.bin(l, b, j)];
                    }
                }
                sum
            };
            let scatter = |s: &mut [f64],
                           i: usize,
                           j: usize,
                           f: Landing,
                           l: usize,
                           d: Landing,
                           vol: [f64; 2],
                           r: f64| {
                for (a, (wf, vol)) in f.weights.iter().zip(vol).enumerate() {
                    let Some(ib) = t.target(i, f, a) else {
                        continue;
                    };
                    let gain = r * wf * vol;
                    let row = ib * nd;
                    for (b, wd) in d.weights.iter().enumerate() {
                        s[row + t.bin(l, b, j)] += gain * wd;
                    }
                }
            };
            // The diagonal of a landing's gain, `vol ω² ∂r/∂e±` with `ω`
            // its gather weight
            let scatter_diagonal = |diag: &mut [f64],
                                    i: usize,
                                    j: usize,
                                    f: Landing,
                                    l: usize,
                                    d: Landing,
                                    vol: [f64; 2],
                                    dr: f64| {
                for (a, (wf, vol)) in f.weights.iter().zip(vol).enumerate() {
                    let Some(ib) = t.target(i, f, a) else {
                        continue;
                    };
                    let gain = dr * wf * wf * vol;
                    let row = ib * nd;
                    for (b, wd) in d.weights.iter().enumerate() {
                        diag[row + t.bin(l, b, j)] += gain * (wd * wd);
                    }
                }
            };
            for i in 0..nf {
                let factor = constant * t.sigma11[i];
                for j in 0..nd {
                    let ec = e[i * nd + j];
                    if ec <= 0.0 {
                        continue;
                    }
                    for m in 0..2 {
                        let (dp, dm) = (st.dir_plus[m], st.dir_minus[m]);
                        let ep = gather(i, j, st.plus, m, dp);
                        let em = gather(i, j, st.minus, 2 + m, dm);
                        let r =
                            factor * ec * (ec * (ep * w_plus + em * w_minus) - w_both * ep * em);
                        s[i * nd + j] -= 2.0 * r;
                        scatter(s, i, j, st.plus, m, dp, st.volume_plus, r);
                        scatter(s, i, j, st.minus, 2 + m, dm, st.volume_minus, r);
                        if let Some(diag) = diag.as_deref_mut() {
                            // ∂r/∂e, ∂r/∂e₊, ∂r/∂e₋
                            let (sum, both) = (ep * w_plus + em * w_minus, w_both * ep * em);
                            let dr = factor * (2.0 * ec * sum - both);
                            let dr_plus = factor * ec * (ec * w_plus - w_both * em);
                            let dr_minus = factor * ec * (ec * w_minus - w_both * ep);
                            diag[i * nd + j] -= 2.0 * dr;
                            let (vp, vm) = (st.volume_plus, st.volume_minus);
                            scatter_diagonal(diag, i, j, st.plus, m, dp, vp, dr_plus);
                            scatter_diagonal(diag, i, j, st.minus, 2 + m, dm, vm, dr_minus);
                        }
                    }
                }
            }
        });
    }

    /// The weights `((1 + λ)⁻⁴, (1 − λ)⁻⁴, 2 (1 − λ²)⁻⁴)` of the transfer's
    /// three products.
    fn weights(&self) -> (f64, f64, f64) {
        let lam = self.lambda;
        (
            (1.0 + lam).powi(-4),
            (1.0 - lam).powi(-4),
            2.0 * (1.0 - lam * lam).powi(-4),
        )
    }
}

/// What the transfer needs of the grid besides the spectrum, the same at
/// every node: the stencil, the rows of the extended spectrum, the tail's
/// factors above the grid, each direction landing's bins on the circle and
/// `(σ/2π)¹¹`. Built once per thread for the grid, the λ and the tail it
/// belongs to ([`Tables::cached`]).
struct Tables {
    /// What the tables were built for: λ, the tail, `Δθ`, `n_θ` and the
    /// frequencies
    lambda: f64,
    tail: Option<f64>,
    d_theta: f64,
    n_dir: usize,
    sigma: Vec<f64>,
    stencil: Stencil,
    /// The extended spectrum's rows are the frequencies `lo..hi`
    lo: isize,
    hi: isize,
    /// The tail's factor `γ^(−p (i − n_σ + 1))` of each row `i ≥ n_σ`, or
    /// none (zero above the grid)
    tail_factors: Option<Vec<f64>>,
    /// The bins of landing `l` (0, 1: σ₊ and its mirror; 2, 3: σ₋) at offset
    /// `b` from direction `j`: `bins[(2l + b) n_θ + j]`
    bins: Vec<usize>,
    /// `(σ_i/2π)¹¹`
    sigma11: Vec<f64>,
}

impl Tables {
    /// The tables in `slot`, rebuilt first unless they belong to this grid,
    /// λ and tail.
    fn cached<'a>(
        slot: &'a mut Option<Tables>,
        quadruplets: &Quadruplets,
        grid: &SpectralGrid,
        tail: Option<f64>,
    ) -> &'a Tables {
        let fits = |t: &Tables| {
            t.lambda == quadruplets.lambda
                && t.tail == tail
                && t.d_theta == grid.d_theta
                && t.n_dir == grid.n_dir()
                && t.sigma == grid.sigma
        };
        if !slot.as_ref().is_some_and(fits) {
            *slot = Some(Self::new(quadruplets, grid, tail));
        }
        slot.as_ref().expect("built above")
    }

    fn new(quadruplets: &Quadruplets, grid: &SpectralGrid, tail: Option<f64>) -> Self {
        let (nf, nd) = (grid.n_freq() as isize, grid.n_dir() as isize);
        let stencil = quadruplets.stencil(grid);
        let gamma = grid.frequency_ratio();
        let (lo, hi) = (
            stencil.plus.offset.min(stencil.minus.offset).min(0),
            (nf + stencil.plus.offset.max(stencil.minus.offset) + 1).max(nf),
        );
        let tail_factors = tail.map(|p| {
            (nf..hi)
                .map(|i| gamma.powf(-p * (i - nf + 1) as f64))
                .collect()
        });
        let landings = [
            stencil.dir_plus[0],
            stencil.dir_plus[1],
            stencil.dir_minus[0],
            stencil.dir_minus[1],
        ];
        let mut bins = Vec::with_capacity(8 * nd as usize);
        for d in landings {
            for b in 0..2 {
                bins.extend((0..nd).map(|j| (j + d.offset + b).rem_euclid(nd) as usize));
            }
        }
        Self {
            lambda: quadruplets.lambda,
            tail,
            d_theta: grid.d_theta,
            n_dir: grid.n_dir(),
            sigma: grid.sigma.clone(),
            stencil,
            lo,
            hi,
            tail_factors,
            bins,
            sigma11: grid.sigma.iter().map(|s| (s / TAU).powi(11)).collect(),
        }
    }

    /// `e` extended by the rows the stencil reaches (in `extended`, rows
    /// `lo..hi`): zero below the grid, the tail (or zero) above it. `e` holds
    /// `lanes` spectra interleaved (`e[c · lanes + l]`), and so does
    /// `extended`.
    fn extend(&self, e: &[f64], lanes: usize, extended: &mut Vec<f64>) {
        let width = self.n_dir * lanes;
        let nf = self.sigma.len() as isize;
        extended.clear();
        for i in self.lo..self.hi {
            if i < 0 {
                extended.extend(std::iter::repeat_n(0.0, width));
            } else if i < nf {
                let row = i as usize * width;
                extended.extend_from_slice(&e[row..row + width]);
            } else {
                let top = &e[(nf as usize - 1) * width..nf as usize * width];
                match &self.tail_factors {
                    Some(factors) => {
                        let factor = factors[(i - nf) as usize];
                        extended.extend(top.iter().map(|x| x * factor));
                    }
                    None => extended.extend(std::iter::repeat_n(0.0, width)),
                }
            }
        }
    }

    /// The extended spectrum's row of frequency `i`'s landing `f` at offset
    /// `a`.
    #[inline(always)]
    fn row(&self, i: usize, f: Landing, a: usize) -> usize {
        (i as isize + f.offset + a as isize - self.lo) as usize
    }

    /// The grid's row of frequency `i`'s landing `f` at offset `a`, or none
    /// off the grid.
    #[inline(always)]
    fn target(&self, i: usize, f: Landing, a: usize) -> Option<usize> {
        let ib = i as isize + f.offset + a as isize;
        (0..self.sigma.len() as isize)
            .contains(&ib)
            .then_some(ib as usize)
    }

    /// The bin of direction landing `l` at offset `b` from direction `j`.
    #[inline(always)]
    fn bin(&self, l: usize, b: usize, j: usize) -> usize {
        self.bins[(2 * l + b) * self.n_dir + j]
    }
}

/// Per-thread storage of [`Quadruplets::source`]: the tables, the spectrum
/// extended by the rows its stencil reaches, and the vector kernel's tiles.
#[derive(Default)]
struct DiaScratch {
    tables: Option<Tables>,
    extended: Vec<f64>,
    /// The extended spectra of [`Quadruplets::source_lanes`], interleaved
    #[cfg(feature = "simd")]
    extended_lanes: Vec<f64>,
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

    /// The transfer at [`LANES`] nodes at once is each node's own bit for
    /// bit: different seas and depths per lane, an empty lane, and the grids
    /// of `the_tables_give_the_direct_transfer`.
    #[cfg(feature = "simd")]
    #[test]
    fn the_lanes_give_each_nodes_transfer() {
        for (n_freq, n_dir, tail) in [(25, 36, Some(4.0)), (30, 24, None), (12, 6, Some(5.0))] {
            let grid = SpectralGrid::new(0.04, 0.5, n_freq, n_dir);
            let nc = grid.n_components();
            let q = Quadruplets::default();
            let seas: Vec<Vec<f64>> = (0..LANES)
                .map(|l| match l {
                    3 => vec![0.0; nc],
                    _ => {
                        let mut e = grid.jonswap(
                            1.0 + 0.4 * l as f64,
                            3.0 + l as f64,
                            3.3,
                            0.3 * l as f64,
                            2.0,
                        );
                        // Some empty components among full ones
                        for c in (l..nc).step_by(7) {
                            e[c] = 0.0;
                        }
                        e
                    }
                })
                .collect();
            let scales: [f64; LANES] = std::array::from_fn(|l| 1.0 + 0.37 * l as f64);
            let mut e = vec![0.0; nc * LANES];
            for (l, sea) in seas.iter().enumerate() {
                for (c, x) in sea.iter().enumerate() {
                    e[c * LANES + l] = *x;
                }
            }
            let mut lanes = vec![f64::NAN; nc * LANES];
            let mut diag_lanes = vec![f64::NAN; nc * LANES];
            q.source_lanes(
                &grid,
                &e,
                tail,
                G,
                scales,
                &mut lanes,
                Some(&mut diag_lanes),
            );
            for (l, sea) in seas.iter().enumerate() {
                let (mut alone, mut diag) = (vec![0.0; nc], vec![0.0; nc]);
                q.source_and_diagonal(&grid, sea, tail, G, scales[l], &mut alone, &mut diag);
                let bits = |x: &[f64]| x.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                let lane = |x: &[f64]| (0..nc).map(|c| x[c * LANES + l]).collect::<Vec<_>>();
                let at = format!("{n_freq}×{n_dir}, tail {tail:?}, lane {l}");
                assert_eq!(bits(&lane(&lanes)), bits(&alone), "{at}");
                assert_eq!(bits(&lane(&diag_lanes)), bits(&diag), "{at}: the diagonal");
            }
        }
    }

    /// The diagonal derivative is the transfer's own, by central differences
    /// of each component's energy (the transfer is a cubic in it, so the
    /// difference is exact but for round-off), with and without the tail; the
    /// transfer itself is `source`'s bit for bit.
    #[test]
    fn the_diagonal_is_the_transfers_derivative() {
        // Few directions, so that the landings wrap around the circle; the
        // frequencies closer than 1 + λ, so that no landing reaches the
        // component's own row
        for (n_freq, n_dir, tail) in [(25, 36, None), (24, 8, Some(5.0))] {
            let grid = SpectralGrid::new(0.04, 0.5, n_freq, n_dir);
            let q = Quadruplets::default();
            // The top rows emptied: the tail continues them, which the
            // diagonal leaves out
            let mut e = grid.jonswap(3.0, 4.0, 3.3, 0.6, 2.0);
            let nc = e.len();
            e[(n_freq - 2) * n_dir..].fill(0.0);
            let (mut s, mut diag, mut plain) = (vec![0.0; nc], vec![0.0; nc], vec![0.0; nc]);
            q.source_and_diagonal(&grid, &e, tail, G, 1.3, &mut s, &mut diag);
            q.source(&grid, &e, tail, G, 1.3, &mut plain);
            assert_eq!(s, plain);
            let scale = diag.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            assert!(scale > 0.0);
            let mut worst = 0.0f64;
            for c in 0..(n_freq - 2) * n_dir {
                let h = 1e-3 * e[c].max(1e-6 * e.iter().cloned().fold(0.0, f64::max));
                let mut shifted = e.clone();
                let (mut up, mut down) = (vec![0.0; nc], vec![0.0; nc]);
                shifted[c] = e[c] + h;
                q.source(&grid, &shifted, tail, G, 1.3, &mut up);
                shifted[c] = e[c] - h;
                q.source(&grid, &shifted, tail, G, 1.3, &mut down);
                let difference = (up[c] - down[c]) / (2.0 * h);
                worst = worst.max((difference - diag[c]).abs() / scale);
            }
            assert!(worst < 1e-6, "{n_freq}×{n_dir}: {worst:e} of the largest");
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
