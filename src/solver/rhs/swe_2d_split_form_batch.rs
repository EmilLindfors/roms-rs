//! The split-form kernel's two hot loops for [`VOLUME_BATCH`] elements or
//! edges at once, one per SIMD lane (`fearless_simd`, `f64x8` at the CPU's
//! best level): the flux-differencing volume term of
//! [`SplitFormSWE2D::line_volume`], and the interior-face fluxes of
//! [`SplitFormSWE2D::edge_fluxes`] (the hydrostatic HLL of `WetDry`).
//!
//! A tensor-product element's lines are 2–5 nodes long at P1–P4, too short to
//! fill a vector, but every element runs the same arithmetic: lanes are
//! elements. The node states (and bed and contravariant vectors) of the batch
//! are transposed into `[node][lane]` vectors, the line loops of
//! `line_volume` run on them in the same order, and each lane's volume term
//! (times J) goes back to [`SplitFormWorkspace::batch`], where
//! [`SplitFormSWE2D::element_rhs`] continues with the surface terms. The face
//! pass likewise takes one edge per lane, its branches (dry sides, upwind
//! wave speeds) as selects.
//!
//! **Bit for bit.** Every lane performs the scalar kernel's operations in the
//! scalar kernel's order (multiplies and adds, no fused multiply-add; `sqrt`
//! and division are correctly rounded in both), so each element's RHS is
//! identical to the per-element evaluation. Parallelogram and curved elements
//! share a batch: a parallelogram's lanes carry its constant contravariant
//! vector at every node, whose pair average `½(m + m) = m` is exact, and the
//! bed term, the only place where the two forms associate differently, is
//! computed both ways and selected per lane. `f64::max`/`min` become
//! compare-and-select, which agrees with them except for NaN in the second
//! argument (never) and a tie of +0 with −0.
//!
//! Elements that take the wet/dry subcells stay on the scalar path; their
//! lanes, boundary edges' lanes and the unused lanes of a short batch are
//! filled with a copy of a batched one and discarded.

use fearless_simd::{Level, dispatch, f64x8, prelude::*};

use super::{SplitFormSWE2D, SplitFormWorkspace, SurfaceFlux, VOLUME_BATCH};
use crate::boundary::SWEBoundaryCondition2D;
use crate::operators::AffineMetric;
use crate::solver::SWEState2D;

const L: usize = VOLUME_BATCH;
const _: () = assert!(L == 8 && u8::BITS as usize == L);

#[cfg(test)]
thread_local! {
    /// Turns the batching off on this thread (the reference for the
    /// bit-for-bit gate)
    pub(super) static UNBATCHED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl<BC: SWEBoundaryCondition2D> SplitFormSWE2D<'_, '_, BC> {
    /// Volume terms (times J) of the elements `ks` (at most
    /// [`VOLUME_BATCH`]) into `ws.batch[l · n_nodes..]` for lane `l`. Bit `l`
    /// of the result is set if element `ks[l]` was computed there, i.e. it
    /// takes the flux-differencing volume term (no subcells). 0 for orders
    /// outside P1–P4.
    pub(in crate::solver::rhs) fn volume_batch(
        &self,
        ks: &[usize],
        ws: &mut SplitFormWorkspace,
    ) -> u8 {
        debug_assert!(ks.len() <= L);
        #[cfg(test)]
        if UNBATCHED.with(std::cell::Cell::get) {
            return 0;
        }
        let n = self.ops.n_nodes;
        let h = &self.q.data[0];
        let mut mask = 0u8;
        for (l, &k) in ks.iter().enumerate() {
            let subcells = self
                .subcells
                .is_some_and(|(depth, _)| h[k * n..(k + 1) * n].iter().any(|&h| h < depth));
            if !subcells {
                mask |= 1 << l;
            }
        }
        if mask == 0 {
            return 0;
        }
        // Lanes without a batched element repeat the first one
        let first = ks[mask.trailing_zeros() as usize];
        let mut lanes = [first; L];
        for (l, &k) in ks.iter().enumerate() {
            if mask & (1 << l) != 0 {
                lanes[l] = k;
            }
        }
        let level = Level::new();
        let done = match self.ops.n_1d {
            2 => dispatch!(level, simd => self.lines::<_, 2, 4>(simd, &lanes, &mut ws.batch)),
            3 => dispatch!(level, simd => self.lines::<_, 3, 9>(simd, &lanes, &mut ws.batch)),
            4 => dispatch!(level, simd => self.lines::<_, 4, 16>(simd, &lanes, &mut ws.batch)),
            5 => dispatch!(level, simd => self.lines::<_, 5, 25>(simd, &lanes, &mut ws.batch)),
            _ => false,
        };
        if done { mask } else { 0 }
    }

    /// [`Self::line_volume`] over every line of the elements `lanes`
    /// (`N1` nodes per line, `NN = N1²` per element), lane by lane in the
    /// scalar order, into `out[l · NN + i]`.
    #[inline(always)]
    fn lines<S: Simd, const N1: usize, const NN: usize>(
        &self,
        simd: S,
        lanes: &[usize; L],
        out: &mut [SWEState2D],
    ) -> bool {
        debug_assert_eq!(N1 * N1, NN);
        let g = self.config.equation.g;
        let h_min = self.config.equation.h_min.meters();
        let d1 = &self.ops.dr_1d_row_major;
        let bathymetry = self.config.bathymetry;
        let [q_h, q_hu, q_hv] = &self.q.data;

        // No closure below touches a vector: built without `target-cpu`
        // flags, a closure that is not inlined runs outside the vector
        // context `dispatch!` sets up (6 % slower than scalar, measured).
        let zero = f64x8::<S>::splat(simd, 0.0);
        let geom = self.geom;
        let elements = lanes.map(|k| geom.element_geometry(k));
        let nodal = [None; L];

        // Node states and bed of every node, one lane per element
        let h = transpose::<S, NN>(simd, lanes, q_h, nodal);
        let hu = transpose::<S, NN>(simd, lanes, q_hu, nodal);
        let hv = transpose::<S, NN>(simd, lanes, q_hv, nodal);
        let b = match bathymetry {
            Some(bathymetry) => transpose::<S, NN>(simd, lanes, &bathymetry.data, nodal),
            None => [zero; NN],
        };
        // SWENodeState2D::new: zero velocity at h ≤ h_min
        let h_min_v = f64x8::<S>::splat(simd, h_min);
        let mut u = [zero; NN];
        let mut v = [zero; NN];
        for i in 0..NN {
            let wet = h[i].simd_gt(h_min_v);
            u[i] = wet.select(hu[i] / h[i], zero);
            v[i] = wet.select(hv[i] / h[i], zero);
        }

        // Contravariant vectors J∇r (r-lines) and J∇s (s-lines) at every
        // node: `GeometricFactors2D::contravariant`, or a parallelogram's
        // constant `AffineMetric::contravariant`, products for products
        let constant =
            |f: fn(&AffineMetric) -> f64| elements.map(|e| e.affine.then(|| f(&e.metric)));
        let j = transpose::<S, NN>(simd, lanes, &geom.det_j, constant(|m| m.det_j));
        let rx = transpose::<S, NN>(simd, lanes, &geom.rx, constant(|m| m.rx));
        let ry = transpose::<S, NN>(simd, lanes, &geom.ry, constant(|m| m.ry));
        let sx = transpose::<S, NN>(simd, lanes, &geom.sx, constant(|m| m.sx));
        let sy = transpose::<S, NN>(simd, lanes, &geom.sy, constant(|m| m.sy));
        let (mut mrx, mut mry, mut msx, mut msy) = ([zero; NN], [zero; NN], [zero; NN], [zero; NN]);
        for i in 0..NN {
            mrx[i] = j[i] * rx[i];
            mry[i] = j[i] * ry[i];
            msx[i] = j[i] * sx[i];
            msy[i] = j[i] * sy[i];
        }
        let affine_lanes =
            f64x8::<S>::from_slice(simd, &elements.map(|e| f64::from(u8::from(e.affine))))
                .simd_gt(zero);

        let half = f64x8::<S>::splat(simd, 0.5);
        let g_v = f64x8::<S>::splat(simd, g);
        let nodes = Nodes {
            h: &h,
            hu: &hu,
            hv: &hv,
            u: &u,
            v: &v,
            half,
            half_g: f64x8::<S>::splat(simd, 0.5 * g),
        };
        let mut rh = [zero; NN];
        let mut rhu = [zero; NN];
        let mut rhv = [zero; NN];

        for line in 0..N1 {
            for direction in 0..2 {
                let idx = |a: usize| {
                    if direction == 0 {
                        line * N1 + a
                    } else {
                        a * N1 + line
                    }
                };
                let (mx, my) = if direction == 0 {
                    (&mrx, &mry)
                } else {
                    (&msx, &msy)
                };
                for a in 0..N1 {
                    let ia = idx(a);
                    let (mxa, mya) = (mx[ia], my[ia]);
                    let (fh, fu, fv) = nodes.flux(ia, ia, mxa, mya);
                    let s = f64x8::<S>::splat(simd, 2.0 * d1[a * N1 + a]);
                    let (mut ah, mut au, mut av) = (s * fh, s * fu, s * fv);
                    for c in (a + 1)..N1 {
                        let ic = idx(c);
                        let m = (half * (mxa + mx[ic]), half * (mya + my[ic]));
                        let (fh, fu, fv) = nodes.flux(ia, ic, m.0, m.1);
                        let s = f64x8::<S>::splat(simd, 2.0 * d1[a * N1 + c]);
                        ah += s * fh;
                        au += s * fu;
                        av += s * fv;
                        let s = f64x8::<S>::splat(simd, 2.0 * d1[c * N1 + a]);
                        rh[ic] -= s * fh;
                        rhu[ic] -= s * fu;
                        rhv[ic] -= s * fv;
                    }

                    if bathymetry.is_some() {
                        let (mut db, mut db_x, mut db_y) = (zero, zero, zero);
                        for c in 0..N1 {
                            let ic = idx(c);
                            let d_b = f64x8::<S>::splat(simd, d1[a * N1 + c]) * b[ic];
                            db += d_b;
                            db_x += d_b * mx[ic];
                            db_y += d_b * my[ic];
                        }
                        let db_x = affine_lanes.select(db * mxa, half * (db * mxa + db_x));
                        let db_y = affine_lanes.select(db * mya, half * (db * mya + db_y));
                        let force = g_v * h[ia];
                        // The scalar kernel adds SWEState2D::new(0.0, ..): −0 + 0 = +0
                        ah += zero;
                        au += force * db_x;
                        av += force * db_y;
                    }

                    rh[ia] -= ah;
                    rhu[ia] -= au;
                    rhv[ia] -= av;
                }
            }
        }

        for i in 0..NN {
            let (rh, rhu, rhv) = (rh[i].as_slice(), rhu[i].as_slice(), rhv[i].as_slice());
            for l in 0..L {
                out[l * NN + i] = SWEState2D::new(rh[l], rhu[l], rhv[l]);
            }
        }
        true
    }
}

/// `[node][lane]` vectors of the nodal `field` of the elements `lanes` (each
/// element's row read contiguously), or `constant[l]` at every node of lane
/// `l` where given.
#[inline(always)]
fn transpose<S: Simd, const NN: usize>(
    simd: S,
    lanes: &[usize; L],
    field: &[f64],
    constant: [Option<f64>; L],
) -> [f64x8<S>; NN] {
    let mut t = [[0.0; L]; NN];
    for (l, &k) in lanes.iter().enumerate() {
        match constant[l] {
            Some(c) => t.iter_mut().for_each(|t| t[l] = c),
            None => {
                for (t, &x) in t.iter_mut().zip(&field[k * NN..(k + 1) * NN]) {
                    t[l] = x;
                }
            }
        }
    }
    let mut out = [f64x8::<S>::splat(simd, 0.0); NN];
    for (out, t) in out.iter_mut().zip(&t) {
        *out = f64x8::<S>::from_slice(simd, t);
    }
    out
}

/// The node states of a batch of elements for the two-point flux.
struct Nodes<'a, S: Simd, const NN: usize> {
    h: &'a [f64x8<S>; NN],
    hu: &'a [f64x8<S>; NN],
    hv: &'a [f64x8<S>; NN],
    u: &'a [f64x8<S>; NN],
    v: &'a [f64x8<S>; NN],
    half: f64x8<S>,
    half_g: f64x8<S>,
}

impl<S: Simd, const NN: usize> Nodes<'_, S, NN> {
    /// `wintermeyer_flux_2d(q_l, q_r, m)`, operation for operation.
    #[inline(always)]
    fn flux(
        &self,
        l: usize,
        r: usize,
        mx: f64x8<S>,
        my: f64x8<S>,
    ) -> (f64x8<S>, f64x8<S>, f64x8<S>) {
        let (h, hu, hv, u, v) = (self.h, self.hu, self.hv, self.u, self.v);
        let mass = self.half * ((hu[l] + hu[r]) * mx + (hv[l] + hv[r]) * my);
        let pressure = self.half_g * (h[l] * h[r]);
        let u_avg = self.half * (u[l] + u[r]);
        let v_avg = self.half * (v[l] + v[r]);
        (
            mass,
            mass * u_avg + pressure * mx,
            mass * v_avg + pressure * my,
        )
    }
}

/// One side of a face for the batched hydrostatic HLL: the node states
/// (`SWENodeState2D`) of [`VOLUME_BATCH`] face nodes, one per lane.
#[derive(Clone, Copy)]
struct Side<S: Simd> {
    h: f64x8<S>,
    u: f64x8<S>,
    v: f64x8<S>,
    b: f64x8<S>,
}

impl<S: Simd> Side<S> {
    /// `SWENodeState2D::new` at the volume nodes `nodes[l]` (indices into
    /// the nodal fields `q` and `bed`).
    #[inline(always)]
    fn gather(simd: S, q: [&[f64]; 3], bed: Option<&[f64]>, nodes: [usize; L], h_min: f64) -> Self {
        let mut t = [[0.0; L]; 4];
        for (l, &j) in nodes.iter().enumerate() {
            (t[0][l], t[1][l], t[2][l]) = (q[0][j], q[1][j], q[2][j]);
            t[3][l] = bed.map_or(0.0, |b| b[j]);
        }
        let h = f64x8::<S>::from_slice(simd, &t[0]);
        let hu = f64x8::<S>::from_slice(simd, &t[1]);
        let hv = f64x8::<S>::from_slice(simd, &t[2]);
        let zero = f64x8::<S>::splat(simd, 0.0);
        let wet = h.simd_gt(f64x8::<S>::splat(simd, h_min));
        Self {
            h,
            u: wet.select(hu / h, zero),
            v: wet.select(hv / h, zero),
            b: f64x8::<S>::from_slice(simd, &t[3]),
        }
    }
}

/// A side of the hydrostatic HLL after the Audusse et al. (2004)
/// reconstruction, in face-aligned components (as `HllSide`).
#[derive(Clone, Copy)]
struct Star<S: Simd> {
    h: f64x8<S>,
    hun: f64x8<S>,
    hut: f64x8<S>,
    un: f64x8<S>,
    ut: f64x8<S>,
}

impl<S: Simd> Star<S> {
    /// The `side` of `hydrostatic_hll`: `h* = max(0, h + B − b_face)` with
    /// the velocity kept (zero on a side at or below `h_min`).
    #[inline(always)]
    fn new(
        simd: S,
        q: Side<S>,
        b_face: f64x8<S>,
        (nx, ny): (f64x8<S>, f64x8<S>),
        h_min: f64,
    ) -> Self {
        let zero = f64x8::<S>::splat(simd, 0.0);
        let h_min = f64x8::<S>::splat(simd, h_min);
        let h_star = max(q.h + q.b - b_face, zero);
        let wet = q.h.simd_gt(h_min);
        let (u, v) = (wet.select(q.u, zero), wet.select(q.v, zero));
        let (un, ut) = (u * nx + v * ny, -u * ny + v * nx);
        let (hun, hut) = (h_star * un, h_star * ut);
        let star_wet = h_star.simd_gt(h_min);
        Self {
            h: h_star,
            hun,
            hut,
            un: star_wet.select(un, zero),
            ut: star_wet.select(ut, zero),
        }
    }

    /// The physical flux in face-aligned components.
    #[inline(always)]
    fn flux(self, half_g: f64x8<S>) -> Flux<S> {
        Flux {
            h: self.h * self.un,
            hu: self.h * self.un * self.un + half_g * self.h * self.h,
            hv: self.h * self.un * self.ut,
        }
    }
}

/// The components of [`VOLUME_BATCH`] fluxes, one per lane.
#[derive(Clone, Copy)]
struct Flux<S: Simd> {
    h: f64x8<S>,
    hu: f64x8<S>,
    hv: f64x8<S>,
}

/// Largest face of P1–P4.
const MAX_FACE_NODES: usize = 5;

impl<BC: SWEBoundaryCondition2D> SplitFormSWE2D<'_, '_, BC> {
    /// [`Self::edge_fluxes`] of the `edges` (at most [`VOLUME_BATCH`]) at
    /// once, one edge per lane, into `slots` (one per edge). Boundary edges
    /// are left untouched, as by `edge_fluxes`. Returns false (and writes
    /// nothing) unless the surface flux is the hydrostatic HLL of `WetDry`
    /// and the order is P1–P4.
    pub(in crate::solver::rhs) fn edge_fluxes_simd(
        &self,
        edges: &[usize],
        slots: &mut [&mut [SWEState2D]],
    ) -> bool {
        debug_assert!(edges.len() <= L && slots.len() == edges.len());
        #[cfg(test)]
        if UNBATCHED.with(std::cell::Cell::get) {
            return false;
        }
        if self.surface != SurfaceFlux::HydrostaticHll {
            return false;
        }
        let mut mask = 0u8;
        for (l, &e) in edges.iter().enumerate() {
            if self.mesh.edges[e].right.is_some() {
                mask |= 1 << l;
            }
        }
        if mask == 0 {
            return true;
        }
        // Lanes without an interior edge repeat the first one
        let first = edges[mask.trailing_zeros() as usize];
        let mut lanes = [first; L];
        for (l, &e) in edges.iter().enumerate() {
            if mask & (1 << l) != 0 {
                lanes[l] = e;
            }
        }
        let mut out = [[SWEState2D::zero(); 2 * MAX_FACE_NODES]; L];
        let level = Level::new();
        let done = match self.ops.n_face_nodes {
            2 => dispatch!(level, simd => self.faces::<_, 2>(simd, &lanes, &mut out)),
            3 => dispatch!(level, simd => self.faces::<_, 3>(simd, &lanes, &mut out)),
            4 => dispatch!(level, simd => self.faces::<_, 4>(simd, &lanes, &mut out)),
            5 => dispatch!(level, simd => self.faces::<_, 5>(simd, &lanes, &mut out)),
            _ => false,
        };
        if !done {
            return false;
        }
        let per_edge = 2 * self.ops.n_face_nodes;
        for (l, slot) in slots.iter_mut().enumerate() {
            if mask & (1 << l) != 0 {
                slot.copy_from_slice(&out[l][..per_edge]);
            }
        }
        true
    }

    /// The face-node loop of [`Self::edge_fluxes`] for the interior edges
    /// `lanes` (`NF` nodes per face), lane by lane in the scalar order, into
    /// `out[l]`: the left side's `NF` values, then the right side's.
    #[inline(always)]
    fn faces<S: Simd, const NF: usize>(
        &self,
        simd: S,
        lanes: &[usize; L],
        out: &mut [[SWEState2D; 2 * MAX_FACE_NODES]; L],
    ) -> bool {
        let n = self.ops.n_nodes;
        let g = self.config.equation.g;
        let h_min = self.config.equation.h_min.meters();
        let [q_h, q_hu, q_hv] = &self.q.data;
        let bed = self.config.bathymetry.map(|b| &b.data[..]);
        let edges = lanes.map(|e| &self.mesh.edges[e]);
        let lefts = edges.map(|e| e.left);
        let rights = edges.map(|e| e.right.expect("interior edge"));
        let left_geometry = lefts.map(|left| self.geom.element_geometry(left.element));

        for fi in 0..NF {
            let rfi = NF - 1 - fi;
            let left = Side::gather(
                simd,
                [q_h, q_hu, q_hv],
                bed,
                lefts.map(|f| f.element * n + self.ops.face_nodes[f.face][fi]),
                h_min,
            );
            let right = Side::gather(
                simd,
                [q_h, q_hu, q_hv],
                bed,
                rights.map(|f| f.element * n + self.ops.face_nodes[f.face][rfi]),
                h_min,
            );
            // A parallelogram's faces are straight with a constant normal
            let mut normal = [[0.0; L]; 2];
            for l in 0..L {
                (normal[0][l], normal[1][l]) = if left_geometry[l].affine {
                    left_geometry[l].faces[lefts[l].face].0
                } else {
                    self.geom.normal(lefts[l].element, lefts[l].face, fi)
                };
            }
            let (mx, my) = (
                f64x8::<S>::from_slice(simd, &normal[0]),
                f64x8::<S>::from_slice(simd, &normal[1]),
            );
            let (f_left, f_right) = hydrostatic_hll(simd, left, right, (mx, my), g, h_min);
            // surface_flux: (F_a, −F_b)
            let (r_h, r_hu, r_hv) = (-f_right.h, -f_right.hu, -f_right.hv);
            let (lh, lhu, lhv) = (
                f_left.h.as_slice(),
                f_left.hu.as_slice(),
                f_left.hv.as_slice(),
            );
            let (rh, rhu, rhv) = (r_h.as_slice(), r_hu.as_slice(), r_hv.as_slice());
            for l in 0..L {
                out[l][fi] = SWEState2D::new(lh[l], lhu[l], lhv[l]);
                out[l][NF + rfi] = SWEState2D::new(rh[l], rhu[l], rhv[l]);
            }
        }
        true
    }
}

/// `f64::max(a, b)` for a non-NaN `b`: `a` if `a > b`, else `b` (a tie of
/// ±0 gives `b`).
#[inline(always)]
fn max<S: Simd>(a: f64x8<S>, b: f64x8<S>) -> f64x8<S> {
    a.simd_gt(b).select(a, b)
}

/// `f64::min(a, b)` for a non-NaN `b`: `a` if `a < b`, else `b`.
#[inline(always)]
fn min<S: Simd>(a: f64x8<S>, b: f64x8<S>) -> f64x8<S> {
    a.simd_lt(b).select(a, b)
}

/// [`SplitFormSWE2D::hydrostatic_hll`] with `hll_flux_face_aligned`,
/// `einfeldt_speeds_2d` and `rotate_from_normal`, lane by lane in the scalar
/// order: every branch is evaluated and selected.
#[inline(always)]
fn hydrostatic_hll<S: Simd>(
    simd: S,
    q_a: Side<S>,
    q_b: Side<S>,
    m: (f64x8<S>, f64x8<S>),
    g: f64,
    h_min: f64,
) -> (Flux<S>, Flux<S>) {
    let zero = f64x8::<S>::splat(simd, 0.0);
    let half = f64x8::<S>::splat(simd, 0.5);
    let two = f64x8::<S>::splat(simd, 2.0);
    let g_v = f64x8::<S>::splat(simd, g);
    let half_g = f64x8::<S>::splat(simd, 0.5 * g);
    let h_min_v = f64x8::<S>::splat(simd, h_min);
    let norm = (m.0 * m.0 + m.1 * m.1).sqrt();
    let (nx, ny) = (m.0 / norm, m.1 / norm);
    let b_face = max(q_a.b, q_b.b);
    let a = Star::new(simd, q_a, b_face, (nx, ny), h_min);
    let b = Star::new(simd, q_b, b_face, (nx, ny), h_min);
    let (h_l, h_r) = (a.h, b.h);

    // hll_flux_face_aligned, with einfeldt_speeds_2d
    let hll_dry = h_l.simd_le(h_min_v) & h_r.simd_le(h_min_v);
    let c_l = (g_v * max(h_l, zero)).sqrt();
    let c_r = (g_v * max(h_r, zero)).sqrt();
    let sqrt_h_l = max(h_l, zero).sqrt();
    let sqrt_h_r = max(h_r, zero).sqrt();
    let roe = (sqrt_h_l + sqrt_h_r).simd_gt(f64x8::<S>::splat(simd, 1e-10));
    let h_roe = half * (h_l + h_r);
    let un_roe = roe.select(
        (sqrt_h_l * a.un + sqrt_h_r * b.un) / (sqrt_h_l + sqrt_h_r),
        zero,
    );
    let c_roe = roe.select((g_v * h_roe).sqrt(), zero);
    let s_l = h_l
        .simd_gt(h_min_v)
        .select(min(a.un - c_l, un_roe - c_roe), b.un - two * c_r);
    let s_r = h_r
        .simd_gt(h_min_v)
        .select(max(b.un + c_r, un_roe + c_roe), a.un + two * c_l);
    let (f_l, f_r) = (a.flux(half_g), b.flux(half_g));
    let inv_ds = f64x8::<S>::splat(simd, 1.0) / (s_r - s_l);
    let s_lr = s_l * s_r;
    let star = Flux {
        h: inv_ds * (s_r * f_l.h - s_l * f_r.h + s_lr * (h_r - h_l)),
        hu: inv_ds * (s_r * f_l.hu - s_l * f_r.hu + s_lr * (b.hun - a.hun)),
        hv: inv_ds * (s_r * f_l.hv - s_l * f_r.hv + s_lr * (b.hut - a.hut)),
    };
    // s_l >= 0: F_l; else s_r <= 0: F_r; else the star flux; zero if dry
    let (from_l, from_r) = (s_l.simd_ge(zero), s_r.simd_le(zero));
    let hll = Flux {
        h: hll_dry.select(zero, from_l.select(f_l.h, from_r.select(f_r.h, star.h))),
        hu: hll_dry.select(zero, from_l.select(f_l.hu, from_r.select(f_r.hu, star.hu))),
        hv: hll_dry.select(zero, from_l.select(f_l.hv, from_r.select(f_r.hv, star.hv))),
    };
    // |m| rotate_from_normal
    let flux = Flux {
        h: norm * hll.h,
        hu: norm * (hll.hu * nx - hll.hv * ny),
        hv: norm * (hll.hu * ny + hll.hv * nx),
    };
    // + g/2 (h^2 - h*^2)(0, m), all of g h^2 / 2 where HLL is dry
    let (h_star_a, h_star_b) = (hll_dry.select(zero, h_l), hll_dry.select(zero, h_r));
    let dp_a = half_g * (q_a.h * q_a.h - h_star_a * h_star_a);
    let dp_b = half_g * (q_b.h * q_b.h - h_star_b * h_star_b);
    (
        Flux {
            h: flux.h + zero,
            hu: flux.hu + dp_a * m.0,
            hv: flux.hv + dp_a * m.1,
        },
        Flux {
            h: flux.h + zero,
            hu: flux.hu + dp_b * m.0,
            hv: flux.hv + dp_b * m.1,
        },
    )
}
