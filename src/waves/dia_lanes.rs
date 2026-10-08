//! The DIA of [`Quadruplets::source`] at [`LANES`] nodes at once, one node
//! per SIMD lane (`fearless_simd`). The stencil is the same at every node, so
//! with the spectra interleaved by node (`[component][lane]`) every gather
//! and scatter of the scalar transfer becomes one vector load or store.
//!
//! Every lane performs the scalar transfer's operations in its order. A
//! component without energy, which the scalar transfer skips, transfers zero
//! in its lane: adding a zero leaves every sum's bits as they were (the sums
//! start at +0 and never become −0), so the lanes give the scalar bits.

use std::f64::consts::TAU;

use fearless_simd::{Level, dispatch, f64x8, prelude::*};

use super::{DIA_SCRATCH, DiaScratch, Landing, Quadruplets, Tables};
use crate::waves::spectrum::SpectralGrid;

/// Nodes per vector.
pub(crate) const LANES: usize = 8;

impl Quadruplets {
    /// [`Self::source`] at [`LANES`] nodes at once: `e[c · LANES + l]` is
    /// node `l`'s variance density, `scales[l]` its finite-depth factor, and
    /// `s` (overwritten) its transfer in the same layout.
    #[inline(never)] // keeps its frame out of rayon's recursive split frames
    pub(crate) fn source_lanes(
        &self,
        grid: &SpectralGrid,
        e: &[f64],
        tail: Option<f64>,
        g: f64,
        scales: [f64; LANES],
        s: &mut [f64],
    ) {
        let nc = grid.n_components();
        assert_eq!(e.len(), nc * LANES);
        assert_eq!(s.len(), nc * LANES);
        s.fill(0.0);
        let constants = scales.map(|scale| scale * self.c_nl4 * TAU * TAU / g.powi(4));
        DIA_SCRATCH.with_borrow_mut(|scratch| {
            let DiaScratch {
                tables,
                extended_lanes,
                ..
            } = scratch;
            let t = Tables::cached(tables, self, grid, tail);
            t.extend(e, LANES, extended_lanes);
            let level = Level::new();
            // One bounds check per vector: the lanes as arrays
            let (e, ext) = (e.as_chunks().0, extended_lanes.as_chunks().0);
            let s = s.as_chunks_mut().0;
            dispatch!(level, simd => self.transfer_lanes(simd, t, e, ext, constants, s));
        });
    }

    /// The loops of [`Self::source`], lane by lane.
    #[inline(always)]
    fn transfer_lanes<S: Simd>(
        &self,
        simd: S,
        t: &Tables,
        e: &[[f64; LANES]],
        ext: &[[f64; LANES]],
        constants: [f64; LANES],
        s: &mut [[f64; LANES]],
    ) {
        let (nf, nd) = (t.sigma.len(), t.n_dir);
        let st = &t.stencil;
        let zero = f64x8::<S>::splat(simd, 0.0);
        let two = f64x8::<S>::splat(simd, 2.0);
        let (w_plus, w_minus, w_both) = self.weights();
        let (w_plus, w_minus, w_both) = (
            f64x8::<S>::splat(simd, w_plus),
            f64x8::<S>::splat(simd, w_minus),
            f64x8::<S>::splat(simd, w_both),
        );
        let constants = f64x8::<S>::from_slice(simd, &constants);
        let at = |x: &[[f64; LANES]], index: usize| f64x8::<S>::from_slice(simd, &x[index]);
        let gather = |i: usize, j: usize, f: Landing, l: usize, d: Landing| {
            let mut sum = zero;
            for (a, wf) in f.weights.iter().enumerate() {
                let row = t.row(i, f, a) * nd;
                for (b, wd) in d.weights.iter().enumerate() {
                    sum += f64x8::<S>::splat(simd, wf * wd) * at(ext, row + t.bin(l, b, j));
                }
            }
            sum
        };
        let add = |s: &mut [[f64; LANES]], index: usize, x: f64x8<S>| {
            let slot = &mut s[index];
            (f64x8::<S>::from_slice(simd, slot) + x).store_slice(slot);
        };
        let scatter = |s: &mut [[f64; LANES]],
                       i: usize,
                       j: usize,
                       f: Landing,
                       l: usize,
                       d: Landing,
                       vol: [f64; 2],
                       r: f64x8<S>| {
            for (a, (wf, vol)) in f.weights.iter().zip(vol).enumerate() {
                let Some(ib) = t.target(i, f, a) else {
                    continue;
                };
                let gain = r * f64x8::<S>::splat(simd, *wf) * f64x8::<S>::splat(simd, vol);
                let row = ib * nd;
                for (b, wd) in d.weights.iter().enumerate() {
                    add(s, row + t.bin(l, b, j), gain * f64x8::<S>::splat(simd, *wd));
                }
            }
        };
        for i in 0..nf {
            let factor = constants * f64x8::<S>::splat(simd, t.sigma11[i]);
            for j in 0..nd {
                let c = i * nd + j;
                let ec = at(e, c);
                let active = ec.simd_gt(zero);
                if !active.any_true() {
                    continue;
                }
                for m in 0..2 {
                    let (dp, dm) = (st.dir_plus[m], st.dir_minus[m]);
                    let ep = gather(i, j, st.plus, m, dp);
                    let em = gather(i, j, st.minus, 2 + m, dm);
                    let r = factor * ec * (ec * (ep * w_plus + em * w_minus) - w_both * ep * em);
                    // The scalar transfer skips a component without energy
                    let r = active.select(r, zero);
                    add(s, c, -(two * r));
                    scatter(s, i, j, st.plus, m, dp, st.volume_plus, r);
                    scatter(s, i, j, st.minus, 2 + m, dm, st.volume_minus, r);
                }
            }
        }
    }
}
