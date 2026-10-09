//! The implicit spectral advection of [`WaveModel2D::node_pass`] for
//! [`LANES`] systems at once, one per SIMD lane (`fearless_simd`): the
//! refraction's cyclic systems over the directions take one frequency of a
//! node per lane, the frequency shift's systems over the frequencies one
//! direction per lane. The lanes' data lie in the node's spectrum (in L1):
//! the directions of a frequency are contiguous, the frequencies of a
//! direction `n_θ` apart.
//!
//! Every lane performs the scalar solver's operations in its order
//! ([`WaveModel2D::refract_implicitly`], [`WaveModel2D::shift_implicitly`],
//! `face_flux`, the Thomas algorithm and the Sherman–Morrison correction),
//! its branches as selects: a system whose rates are all zero keeps its
//! values (the scalar early return), and a cyclic system without corners
//! takes the plain Thomas solve. `f64::max`/`min`/`clamp` against a constant
//! become compare-and-select, which agree with them except for NaN and a tie
//! of +0 with −0.

use fearless_simd::{Level, dispatch, f64x8, mask64x8, prelude::*};

use super::{SpectralAdvection, WaveModel2D};

/// Systems per vector.
const LANES: usize = 8;

/// Largest number of bins (directions or frequencies) the vector solver
/// handles; larger grids take the scalar one.
const MAX_BINS: usize = 72;

impl WaveModel2D {
    /// Implicit refraction over `dt` of every frequency at node `p` on its
    /// spectrum (see [`Self::refract_implicitly`]; `turning` the node's
    /// turning terms at the direction faces). False (nothing done) for more
    /// than [`MAX_BINS`] directions.
    #[inline(never)] // keeps its large frame out of rayon's recursive split frames
    pub(super) fn refract_lanes(
        &self,
        p: usize,
        spectrum: &mut [f64],
        dt: f64,
        turning: &[[f64; 2]],
    ) -> bool {
        let (nf, nd) = (self.grid.n_freq(), self.grid.n_dir());
        if !(3..=MAX_BINS).contains(&nd) {
            return false;
        }
        let level = Level::new();
        for i0 in (0..nf).step_by(LANES) {
            let width = LANES.min(nf - i0);
            dispatch!(level, simd => self.refract_batch(simd, p, spectrum, i0, width, dt, turning));
        }
        true
    }

    /// Implicit frequency shifting over `dt` of every direction at node `p`
    /// on its spectrum (see [`Self::shift_implicitly`]; `strain` the node's
    /// strain term at each direction). False for more than [`MAX_BINS`]
    /// frequencies.
    #[inline(never)] // keeps its large frame out of rayon's recursive split frames
    pub(super) fn shift_lanes(
        &self,
        p: usize,
        spectrum: &mut [f64],
        dt: f64,
        advection: f64,
        strain: &[f64],
    ) -> bool {
        let (nf, nd) = (self.grid.n_freq(), self.grid.n_dir());
        if !(2..=MAX_BINS).contains(&nf) {
            return false;
        }
        let level = Level::new();
        for j0 in (0..nd).step_by(LANES) {
            let width = LANES.min(nd - j0);
            dispatch!(level, simd => self.shift_batch(simd, p, spectrum, j0, width, dt, advection, strain));
        }
        true
    }

    /// [`Self::refract_implicitly`] for the frequencies `i0..i0 + width`.
    #[inline(always)]
    fn refract_batch<S: Simd>(
        &self,
        simd: S,
        p: usize,
        spectrum: &mut [f64],
        i0: usize,
        width: usize,
        dt: f64,
        turning: &[[f64; 2]],
    ) {
        let nd = self.grid.n_dir();
        let np = self.n_points();
        let zero = f64x8::<S>::splat(simd, 0.0);
        let lambda = dt / self.grid.d_theta;
        let lambda_v = f64x8::<S>::splat(simd, lambda);
        let previous = |j: usize| if j == 0 { nd - 1 } else { j - 1 };
        let next = |j: usize| if j + 1 == nd { 0 } else { j + 1 };

        // The rows of the batch, `[direction][lane]`
        let mut row = [zero; MAX_BINS];
        for (j, row) in row[..nd].iter_mut().enumerate() {
            let mut t = [0.0; LANES];
            for (l, t) in t[..width].iter_mut().enumerate() {
                *t = spectrum[(i0 + l) * nd + j];
            }
            *row = f64x8::<S>::from_slice(simd, &t);
        }
        let original = row;

        // turning_rate: −(∂σ/∂d / k) depth − current, clamped to the limit
        let mut ratio = [0.0; LANES];
        for (l, ratio) in ratio[..width].iter_mut().enumerate() {
            let ip = (i0 + l) * np + p;
            *ratio = self.sigma_d[ip] / self.k[ip];
        }
        let ratio = f64x8::<S>::from_slice(simd, &ratio);
        let mut faces = [zero; MAX_BINS];
        let mut active = zero.simd_ne(zero);
        for (face, &[depth, current]) in faces[..nd].iter_mut().zip(turning) {
            let depth_term = ratio * f64x8::<S>::splat(simd, depth);
            let mut c = -depth_term - f64x8::<S>::splat(simd, current);
            if let Some(limit) = self.turning_limit {
                c = clamp(simd, c, -limit, limit);
            }
            active |= c.simd_ne(zero);
            *face = c;
        }

        let mut lower = [zero; MAX_BINS];
        let mut diagonal = [zero; MAX_BINS];
        let mut upper = [zero; MAX_BINS];
        let one = f64x8::<S>::splat(simd, 1.0);
        let minus_lambda = f64x8::<S>::splat(simd, -lambda);
        for j in 0..nd {
            let (above, below) = (faces[j], faces[previous(j)]);
            diagonal[j] = one + lambda_v * (max0(simd, above) - min0(simd, below));
            upper[j] = lambda_v * min0(simd, above);
            lower[j] = minus_lambda * max0(simd, below);
        }
        if self.spectral_advection == SpectralAdvection::VanLeer {
            let mut correction = [zero; MAX_BINS];
            for j in 0..nd {
                let (c, left, right) = (faces[j], row[j], row[next(j)]);
                let (far_left, far_right) = (row[previous(j)], row[next(next(j))]);
                let muscl = face_flux(simd, c, Some(far_left), left, right, Some(far_right));
                correction[j] = muscl - (max0(simd, c) * left + min0(simd, c) * right);
            }
            let mut divergence = [zero; MAX_BINS];
            for j in 0..nd {
                let below = correction[previous(j)];
                let taken = lambda_v * (max0(simd, correction[j]) + max0(simd, -below));
                divergence[j] = taken.simd_gt(row[j]).select(row[j] / taken, one);
            }
            for j in 0..nd {
                let c = correction[j];
                let donor = c.simd_ge(zero).select(divergence[j], divergence[next(j)]);
                correction[j] = c * donor;
            }
            for j in 0..nd {
                let d = lambda_v * (correction[j] - correction[previous(j)]);
                row[j] = max0(simd, row[j] - d);
            }
        }
        let solved = solve_cyclic(simd, &lower, &diagonal, &upper, &row, nd);
        for j in 0..nd {
            let x = active.select(solved[j], original[j]);
            let x = x.as_slice();
            for l in 0..width {
                spectrum[(i0 + l) * nd + j] = x[l];
            }
        }
    }

    /// [`Self::shift_implicitly`] for the directions `j0..j0 + width`.
    #[inline(always)]
    fn shift_batch<S: Simd>(
        &self,
        simd: S,
        p: usize,
        spectrum: &mut [f64],
        j0: usize,
        width: usize,
        dt: f64,
        advection: f64,
        strain: &[f64],
    ) {
        let (nf, nd) = (self.grid.n_freq(), self.grid.n_dir());
        let np = self.n_points();
        let zero = f64x8::<S>::splat(simd, 0.0);
        let one = f64x8::<S>::splat(simd, 1.0);
        let half = f64x8::<S>::splat(simd, 0.5);

        // The columns of the batch, `[frequency][lane]`
        let mut column = [zero; MAX_BINS];
        for (i, column) in column[..nf].iter_mut().enumerate() {
            *column = load(simd, &spectrum[i * nd + j0..i * nd + j0 + width]);
        }
        let original = column;
        let strain = load(simd, &strain[j0..j0 + width]);

        // shift_rate: ∂σ/∂d U·∇d − c_g k strain, per bin; the faces above
        let mut bins = [zero; MAX_BINS];
        let mut active = zero.simd_ne(zero);
        for (i, bin) in bins[..nf].iter_mut().enumerate() {
            let ip = i * np + p;
            let advected = f64x8::<S>::splat(simd, self.sigma_d[ip] * advection);
            let c = advected - f64x8::<S>::splat(simd, self.cg[ip] * self.k[ip]) * strain;
            active |= c.simd_ne(zero);
            *bin = c;
        }
        let mut faces = [zero; MAX_BINS];
        for i in 0..nf - 1 {
            faces[i] = half * (bins[i] + bins[i + 1]);
        }
        let (bottom, top) = (min0(simd, bins[0]), max0(simd, bins[nf - 1]));
        faces[nf - 1] = top;

        let mut lower = [zero; MAX_BINS];
        let mut diagonal = [zero; MAX_BINS];
        let mut upper = [zero; MAX_BINS];
        for i in 0..nf {
            let lambda = dt / self.grid.d_sigma[i];
            let lambda_v = f64x8::<S>::splat(simd, lambda);
            let above = faces[i];
            let below = if i > 0 { faces[i - 1] } else { bottom };
            diagonal[i] = one + lambda_v * (max0(simd, above) - min0(simd, below));
            upper[i] = if i + 1 < nf {
                lambda_v * min0(simd, above)
            } else {
                zero
            };
            lower[i] = if i > 0 {
                f64x8::<S>::splat(simd, -lambda) * max0(simd, below)
            } else {
                zero
            };
        }
        if self.spectral_advection == SpectralAdvection::VanLeer {
            let mut correction = [zero; MAX_BINS];
            for i in 0..nf - 1 {
                let (c, left, right) = (faces[i], column[i], column[i + 1]);
                let far_left = (i > 0).then(|| column[i - 1]);
                let far_right = (i + 2 < nf).then(|| column[i + 2]);
                let muscl = face_flux(simd, c, far_left, left, right, far_right);
                correction[i] = muscl - (max0(simd, c) * left + min0(simd, c) * right);
            }
            let mut divergence = [zero; MAX_BINS];
            for i in 0..nf {
                let below = if i > 0 { correction[i - 1] } else { zero };
                let rate = f64x8::<S>::splat(simd, dt / self.grid.d_sigma[i]);
                let taken = rate * (max0(simd, correction[i]) + max0(simd, -below));
                divergence[i] = taken.simd_gt(column[i]).select(column[i] / taken, one);
            }
            for i in 0..nf - 1 {
                let c = correction[i];
                let donor = c.simd_ge(zero).select(divergence[i], divergence[i + 1]);
                correction[i] = c * donor;
            }
            for i in 0..nf {
                let below = if i > 0 { correction[i - 1] } else { zero };
                let rate = f64x8::<S>::splat(simd, dt / self.grid.d_sigma[i]);
                let d = rate * (correction[i] - below);
                column[i] = max0(simd, column[i] - d);
            }
        }
        // No corners: solve_cyclic_tridiagonal takes the plain Thomas solve
        let mut solved = [zero; MAX_BINS];
        thomas(simd, &lower, &diagonal, &upper, &column, &mut solved, nf);
        for i in 0..nf {
            let x = active.select(solved[i], original[i]);
            let row = i * nd + j0;
            spectrum[row..row + width].copy_from_slice(&x.as_slice()[..width]);
        }
    }
}

/// The first `values.len()` (at most [`LANES`]) lanes from `values`, the
/// rest zero.
#[inline(always)]
fn load<S: Simd>(simd: S, values: &[f64]) -> f64x8<S> {
    if values.len() == LANES {
        f64x8::<S>::from_slice(simd, values)
    } else {
        let mut t = [0.0; LANES];
        t[..values.len()].copy_from_slice(values);
        f64x8::<S>::from_slice(simd, &t)
    }
}

/// `x.max(0.0)`.
#[inline(always)]
fn max0<S: Simd>(simd: S, x: f64x8<S>) -> f64x8<S> {
    let zero = f64x8::<S>::splat(simd, 0.0);
    x.simd_gt(zero).select(x, zero)
}

/// `x.min(0.0)`.
#[inline(always)]
fn min0<S: Simd>(simd: S, x: f64x8<S>) -> f64x8<S> {
    let zero = f64x8::<S>::splat(simd, 0.0);
    x.simd_lt(zero).select(x, zero)
}

/// `x.clamp(lo, hi)`.
#[inline(always)]
fn clamp<S: Simd>(simd: S, x: f64x8<S>, lo: f64, hi: f64) -> f64x8<S> {
    let (lo, hi) = (f64x8::<S>::splat(simd, lo), f64x8::<S>::splat(simd, hi));
    let x = x.simd_lt(lo).select(lo, x);
    x.simd_gt(hi).select(hi, x)
}

/// `face_flux`: the upwind flux with van Leer's limited reconstruction where
/// the bins beyond are given.
#[inline(always)]
fn face_flux<S: Simd>(
    simd: S,
    speed: f64x8<S>,
    far_left: Option<f64x8<S>>,
    left: f64x8<S>,
    right: f64x8<S>,
    far_right: Option<f64x8<S>>,
) -> f64x8<S> {
    let zero = f64x8::<S>::splat(simd, 0.0);
    let from_left = match far_left {
        Some(ll) => left + half_slope(simd, left - ll, right - left),
        None => left,
    };
    let from_right = match far_right {
        Some(rr) => right - half_slope(simd, right - left, rr - right),
        None => right,
    };
    speed
        .simd_ge(zero)
        .select(speed * from_left, speed * from_right)
}

/// Half of van Leer's slope, `ab/(a + b)` where `ab > 0`, else 0.
#[inline(always)]
fn half_slope<S: Simd>(simd: S, a: f64x8<S>, b: f64x8<S>) -> f64x8<S> {
    let zero = f64x8::<S>::splat(simd, 0.0);
    let ab = a * b;
    ab.simd_gt(zero).select(ab / (a + b), zero)
}

/// `thomas`, lane by lane: `lower[j] x[j−1] + diagonal[j] x[j] + upper[j]
/// x[j+1] = rhs[j]` for `j < n`.
#[inline(always)]
fn thomas<S: Simd>(
    simd: S,
    lower: &[f64x8<S>; MAX_BINS],
    diagonal: &[f64x8<S>; MAX_BINS],
    upper: &[f64x8<S>; MAX_BINS],
    rhs: &[f64x8<S>; MAX_BINS],
    x: &mut [f64x8<S>; MAX_BINS],
    n: usize,
) {
    let one = f64x8::<S>::splat(simd, 1.0);
    let mut c_prime = [f64x8::<S>::splat(simd, 0.0); MAX_BINS];
    c_prime[0] = upper[0] / diagonal[0];
    x[0] = rhs[0] / diagonal[0];
    for j in 1..n {
        let m = one / (diagonal[j] - lower[j] * c_prime[j - 1]);
        c_prime[j] = upper[j] * m;
        x[j] = (rhs[j] - lower[j] * x[j - 1]) * m;
    }
    for j in (0..n - 1).rev() {
        x[j] -= c_prime[j] * x[j + 1];
    }
}

/// `solve_cyclic_tridiagonal`, lane by lane: Sherman–Morrison on the system
/// without its corners, or the plain Thomas solve in the lanes without
/// corners.
#[inline(always)]
fn solve_cyclic<S: Simd>(
    simd: S,
    lower: &[f64x8<S>; MAX_BINS],
    diagonal: &[f64x8<S>; MAX_BINS],
    upper: &[f64x8<S>; MAX_BINS],
    rhs: &[f64x8<S>; MAX_BINS],
    n: usize,
) -> [f64x8<S>; MAX_BINS] {
    let zero = f64x8::<S>::splat(simd, 0.0);
    let one = f64x8::<S>::splat(simd, 1.0);
    let (alpha, beta) = (upper[n - 1], lower[0]);
    let plain: mask64x8<S> = alpha.simd_eq(zero) & beta.simd_eq(zero);

    let gamma = -diagonal[0];
    let mut modified = *diagonal;
    modified[0] -= gamma;
    modified[n - 1] -= alpha * beta / gamma;
    let mut u = [zero; MAX_BINS];
    u[0] = gamma;
    u[n - 1] = alpha;
    // thomas_pair: the same elimination for rhs and u
    let mut c_prime = [zero; MAX_BINS];
    let (mut x, mut z) = ([zero; MAX_BINS], [zero; MAX_BINS]);
    c_prime[0] = upper[0] / modified[0];
    x[0] = rhs[0] / modified[0];
    z[0] = u[0] / modified[0];
    for j in 1..n {
        let m = one / (modified[j] - lower[j] * c_prime[j - 1]);
        c_prime[j] = upper[j] * m;
        x[j] = (rhs[j] - lower[j] * x[j - 1]) * m;
        z[j] = (u[j] - lower[j] * z[j - 1]) * m;
    }
    for j in (0..n - 1).rev() {
        x[j] -= c_prime[j] * x[j + 1];
        z[j] -= c_prime[j] * z[j + 1];
    }
    let fact = (x[0] + beta * x[n - 1] / gamma) / (one + z[0] + beta * z[n - 1] / gamma);
    let mut out = [zero; MAX_BINS];
    for j in 0..n {
        out[j] = x[j] - fact * z[j];
    }
    // Lanes without corners (rare: the face between the last and the first
    // direction turns nothing)
    let any_plain =
        (alpha.as_slice().iter().zip(beta.as_slice())).any(|(&a, &b)| a == 0.0 && b == 0.0);
    if any_plain {
        let mut plain_x = [zero; MAX_BINS];
        thomas(simd, lower, diagonal, upper, rhs, &mut plain_x, n);
        for j in 0..n {
            out[j] = plain.select(plain_x[j], out[j]);
        }
    }
    out
}
