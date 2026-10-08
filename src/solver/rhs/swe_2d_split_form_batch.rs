//! The split-form kernel's hot loops for [`VOLUME_BATCH`] elements or edges
//! at once, one per SIMD lane (`fearless_simd`, `f64x8` at the CPU's best
//! level): the flux-differencing volume term of
//! [`SplitFormSWE2D::line_volume`], the shoreline subcells of
//! [`SplitFormSWE2D::line_subcells`], and the interior-face fluxes of
//! [`SplitFormSWE2D::edge_fluxes`] (the hydrostatic HLL of `WetDry`).
//!
//! A tensor-product element's lines are 2–5 nodes long at P1–P4, too short to
//! fill a vector, but every element runs the same arithmetic: lanes are
//! elements. [`SplitFormSWE2D::volume_chunk`] takes [`ELEMENT_CHUNK`]
//! consecutive elements, sorts them into the two kinds (a node shallower than
//! the subcell depth or not) so that both fill their lanes, and evaluates each
//! kind a batch at a time. The node states (and bed and contravariant
//! vectors) of a batch are transposed into `[node][lane]` vectors, the line
//! loops run on them in the scalar order, and each element's volume term
//! (times J) goes to [`SplitFormWorkspace::batch`], where
//! [`SplitFormSWE2D::element_rhs`] continues with the surface terms. The
//! subcells' branches (incomplete stencils at boundaries, the limiter's
//! extrema, the dry-node rule for `η`, dry HLL sides, upwind wave speeds)
//! become selects; so do the face pass's, which takes one edge per lane.
//!
//! **Bit for bit.** Every lane performs the scalar kernel's operations in the
//! scalar kernel's order (multiplies and adds, no fused multiply-add; `sqrt`
//! and division are correctly rounded in both), so each element's RHS is
//! identical to the per-element evaluation. Parallelogram and curved elements
//! share a batch: a parallelogram's lanes carry its constant contravariant
//! vector at every node, whose pair average `½(m + m) = m` is exact, and
//! where the two forms associate differently (the bed term, the subcell
//! interface metrics and balance term) both are computed and selected per
//! lane. `f64::max`/`min`/`signum` become compare-and-select, which agrees
//! with them except for NaN arguments and a tie of +0 with −0.
//!
//! Boundary edges stay on the scalar path; their lanes and the unused lanes
//! of a short batch are filled with a copy of a batched one and discarded.

use fearless_simd::{Level, dispatch, f64x8, mask64x8, prelude::*};

use super::{ELEMENT_CHUNK, SplitFormSWE2D, SplitFormWorkspace, SurfaceFlux, VOLUME_BATCH};
use crate::boundary::SWEBoundaryCondition2D;
use crate::operators::AffineMetric;
use crate::solver::SWEState2D;
use crate::solver::rhs::subcells::line_node;
use crate::types::ElementIndex;

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
    /// [`ELEMENT_CHUNK`]) into `ws.batch[j · n_nodes..]` for `ks[j]`, and the
    /// subcell interface mass fluxes of subcell elements into
    /// `ws.batch_mass[j · subcell_interfaces(n_1d)..]`. The elements are
    /// sorted into the flux-differencing and the subcell kind and each kind
    /// is evaluated [`VOLUME_BATCH`] at a time, one per lane. Bit `j` of the
    /// result is set if `ks[j]`'s volume term is there: all of them for
    /// P1–P4, none otherwise.
    pub(in crate::solver::rhs) fn volume_chunk(
        &self,
        ks: &[usize],
        ws: &mut SplitFormWorkspace,
    ) -> u64 {
        debug_assert!(ks.len() <= ELEMENT_CHUNK && ELEMENT_CHUNK <= 64);
        #[cfg(test)]
        if UNBATCHED.with(std::cell::Cell::get) {
            return 0;
        }
        if !(2..=5).contains(&self.ops.n_1d) || ks.is_empty() {
            return 0;
        }
        let n = self.ops.n_nodes;
        let h = &self.q.data[0];
        let (mut wet, mut n_wet) = ([0; ELEMENT_CHUNK], 0);
        let (mut shore, mut n_shore) = ([0; ELEMENT_CHUNK], 0);
        for (j, &k) in ks.iter().enumerate() {
            let subcells = self
                .subcells
                .is_some_and(|(depth, _)| h[k * n..(k + 1) * n].iter().any(|&h| h < depth));
            if subcells {
                shore[n_shore] = j;
                n_shore += 1;
            } else {
                wet[n_wet] = j;
                n_wet += 1;
            }
        }
        let level = Level::new();
        for (slots, subcells) in wet[..n_wet]
            .chunks(L)
            .map(|s| (s, false))
            .chain(shore[..n_shore].chunks(L).map(|s| (s, true)))
        {
            // Lanes without an element repeat the first one (and are dropped)
            let mut lanes = [ks[slots[0]]; L];
            for (lane, &j) in lanes.iter_mut().zip(slots) {
                *lane = ks[j];
            }
            match self.ops.n_1d {
                2 => {
                    dispatch!(level, simd => self.batch::<_, 2, 4>(simd, &lanes, slots, subcells, ws))
                }
                3 => {
                    dispatch!(level, simd => self.batch::<_, 3, 9>(simd, &lanes, slots, subcells, ws))
                }
                4 => {
                    dispatch!(level, simd => self.batch::<_, 4, 16>(simd, &lanes, slots, subcells, ws))
                }
                _ => {
                    dispatch!(level, simd => self.batch::<_, 5, 25>(simd, &lanes, slots, subcells, ws))
                }
            }
        }
        if ks.len() == 64 {
            u64::MAX
        } else {
            (1 << ks.len()) - 1
        }
    }

    /// The volume terms of the elements `lanes` (the first `slots.len()` of
    /// them, into `ws.batch` at `slots`): flux differencing, or the subcells.
    #[inline(always)]
    fn batch<S: Simd, const N1: usize, const NN: usize>(
        &self,
        simd: S,
        lanes: &[usize; L],
        slots: &[usize],
        subcells: bool,
        ws: &mut SplitFormWorkspace,
    ) {
        let data = self.gather::<S, NN>(simd, lanes);
        let zero = f64x8::<S>::splat(simd, 0.0);
        let mut rhs = [[zero; NN]; 3];
        let mut mass = [zero; MAX_SUBCELL_INTERFACES];
        if subcells {
            self.subcell_lines::<S, N1, NN>(simd, &data, lanes, &mut rhs, &mut mass);
        } else {
            self.lines::<S, N1, NN>(simd, &data, &mut rhs);
        }
        let [rh, rhu, rhv] = &rhs;
        for i in 0..NN {
            let (rh, rhu, rhv) = (rh[i].as_slice(), rhu[i].as_slice(), rhv[i].as_slice());
            for (l, &j) in slots.iter().enumerate() {
                ws.batch[j * NN + i] = SWEState2D::new(rh[l], rhu[l], rhv[l]);
            }
        }
        if subcells {
            let interfaces = 2 * N1 * (N1 - 1);
            for (s, mass) in mass[..interfaces].iter().enumerate() {
                let mass = mass.as_slice();
                for (l, &j) in slots.iter().enumerate() {
                    ws.batch_mass[j * interfaces + s] = mass[l];
                }
            }
        }
    }

    /// Node states, bed and contravariant vectors of the elements `lanes`.
    #[inline(always)]
    fn gather<S: Simd, const NN: usize>(&self, simd: S, lanes: &[usize; L]) -> Batch<S, NN> {
        // No closure below touches a vector: built without `target-cpu`
        // flags, a closure that is not inlined runs outside the vector
        // context `dispatch!` sets up (6 % slower than scalar, measured).
        let h_min = self.config.equation.h_min.meters();
        let [q_h, q_hu, q_hv] = &self.q.data;
        let zero = f64x8::<S>::splat(simd, 0.0);
        let geom = self.geom;
        let elements = lanes.map(|k| geom.element_geometry(k));
        let nodal = [None; L];

        let h = transpose::<S, NN>(simd, lanes, q_h, nodal);
        let hu = transpose::<S, NN>(simd, lanes, q_hu, nodal);
        let hv = transpose::<S, NN>(simd, lanes, q_hv, nodal);
        let b = match self.config.bathymetry {
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
        let affine = f64x8::<S>::from_slice(simd, &elements.map(|e| f64::from(u8::from(e.affine))))
            .simd_gt(zero);
        Batch {
            h,
            hu,
            hv,
            u,
            v,
            b,
            metric: [[mrx, mry], [msx, msy]],
            affine,
        }
    }

    /// [`Self::line_volume`] over every line of a batch (`N1` nodes per
    /// line, `NN = N1²` per element), lane by lane in the scalar order, into
    /// `rhs` (zero on entry).
    #[inline(always)]
    fn lines<S: Simd, const N1: usize, const NN: usize>(
        &self,
        simd: S,
        data: &Batch<S, NN>,
        rhs: &mut [[f64x8<S>; NN]; 3],
    ) {
        debug_assert_eq!(N1 * N1, NN);
        let g = self.config.equation.g;
        let d1 = &self.ops.dr_1d_row_major;
        let zero = f64x8::<S>::splat(simd, 0.0);
        let half = f64x8::<S>::splat(simd, 0.5);
        let g_v = f64x8::<S>::splat(simd, g);
        let nodes = data.nodes(simd, g);
        let [rh, rhu, rhv] = rhs;

        for line in 0..N1 {
            for direction in 0..2 {
                let idx = |a: usize| line_node(N1, direction, line, a);
                let [mx, my] = &data.metric[direction];
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

                    if self.config.bathymetry.is_some() {
                        let (mut db, mut db_x, mut db_y) = (zero, zero, zero);
                        for c in 0..N1 {
                            let ic = idx(c);
                            let d_b = f64x8::<S>::splat(simd, d1[a * N1 + c]) * data.b[ic];
                            db += d_b;
                            db_x += d_b * mx[ic];
                            db_y += d_b * my[ic];
                        }
                        let db_x = data.affine.select(db * mxa, half * (db * mxa + db_x));
                        let db_y = data.affine.select(db * mya, half * (db * mya + db_y));
                        let force = g_v * data.h[ia];
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
    }

    /// The outer nodes of [`Self::outer_nodes`] for a batch: per face node
    /// (`face · N1 + fi`), the neighbour's first interior node on the GLL
    /// line through it and its distance, where the face has a neighbour.
    #[inline(always)]
    fn gather_outer<S: Simd, const N1: usize>(
        &self,
        simd: S,
        lanes: &[usize; L],
    ) -> [Outer<S>; MAX_FACE_NODES * 4] {
        let zero = f64x8::<S>::splat(simd, 0.0);
        let empty = Outer {
            node: Side {
                h: zero,
                u: zero,
                v: zero,
                b: zero,
            },
            distance: zero,
            present: zero.simd_gt(zero),
        };
        let mut outer = [empty; MAX_FACE_NODES * 4];
        for face in 0..4 {
            for fi in 0..N1 {
                let mut t = [[0.0; L]; 6];
                for (l, &k) in lanes.iter().enumerate() {
                    if let Some((node, distance)) = self.outer_node(ElementIndex::new(k), face, fi)
                    {
                        (t[0][l], t[1][l], t[2][l], t[3][l]) = (node.h, node.u, node.v, node.b);
                        (t[4][l], t[5][l]) = (distance, 1.0);
                    }
                }
                outer[face * N1 + fi] = Outer {
                    node: Side {
                        h: f64x8::<S>::from_slice(simd, &t[0]),
                        u: f64x8::<S>::from_slice(simd, &t[1]),
                        v: f64x8::<S>::from_slice(simd, &t[2]),
                        b: f64x8::<S>::from_slice(simd, &t[3]),
                    },
                    distance: f64x8::<S>::from_slice(simd, &t[4]),
                    present: f64x8::<S>::from_slice(simd, &t[5]).simd_gt(zero),
                };
            }
        }
        outer
    }

    /// [`Self::line_subcells`] over every line of a batch of subcell
    /// elements (`N1` nodes per line), lane by lane in the scalar order, into
    /// `rhs` (zero on entry) and the interface mass fluxes into `mass`
    /// (`subcell_slot` order).
    #[inline(always)]
    fn subcell_lines<S: Simd, const N1: usize, const NN: usize>(
        &self,
        simd: S,
        data: &Batch<S, NN>,
        lanes: &[usize; L],
        rhs: &mut [[f64x8<S>; NN]; 3],
        mass: &mut [f64x8<S>; MAX_SUBCELL_INTERFACES],
    ) {
        let g = self.config.equation.g;
        let h_min = self.config.equation.h_min.meters();
        let h_dry = self.subcells.map_or(0.0, |(_, h_dry)| h_dry);
        let w = &self.ops.weights_1d;
        let xi = &self.ops.nodes_1d;
        let d1 = &self.ops.dr_1d_row_major;
        let zero = f64x8::<S>::splat(simd, 0.0);
        let half = f64x8::<S>::splat(simd, 0.5);
        let nodes = data.nodes(simd, g);
        let outer = self.gather_outer::<S, N1>(simd, lanes);
        let [rh, rhu, rhv] = rhs;

        for line in 0..N1 {
            // Line ends on faces 3/1 (r-lines) and 0/2 (s-lines); faces 2
            // and 3 list their nodes in reverse
            let (rev, fwd) = (N1 - 1 - line, line);
            for direction in 0..2 {
                let idx = |a: usize| line_node(N1, direction, line, a);
                let [mx, my] = &data.metric[direction];
                let ends = if direction == 0 {
                    [outer[3 * N1 + rev], outer[N1 + fwd]]
                } else {
                    [outer[fwd], outer[2 * N1 + rev]]
                };
                let mut metric = [(zero, zero); MAX_FACE_NODES];
                for a in 0..N1 {
                    metric[a] = (mx[idx(a)], my[idx(a)]);
                }

                // telescoped_interfaces: the line's first metric on
                // parallelograms, the telescoping sum on curved elements
                let mut interfaces = [(zero, zero); MAX_FACE_NODES + 1];
                interfaces[0] = metric[0];
                let mut m = (zero, zero);
                for a in 0..N1 - 1 {
                    for c in (0..N1).filter(|&c| c != a) {
                        let q2 = f64x8::<S>::splat(simd, 2.0 * w[a] * d1[a * N1 + c] * 0.5);
                        m.0 += q2 * (metric[a].0 + metric[c].0);
                        m.1 += q2 * (metric[a].1 + metric[c].1);
                    }
                    interfaces[a + 1] = (
                        data.affine.select(metric[0].0, m.0),
                        data.affine.select(metric[0].1, m.1),
                    );
                }
                interfaces[N1] = (
                    data.affine.select(metric[0].0, metric[N1 - 1].0),
                    data.affine.select(metric[0].1, metric[N1 - 1].1),
                );

                // The end nodes' physical fluxes
                let (first, last) = (idx(0), idx(N1 - 1));
                let (fh, fu, fv) = nodes.flux(first, first, metric[0].0, metric[0].1);
                let s = f64x8::<S>::splat(simd, 1.0 / w[0]);
                rh[first] += s * fh;
                rhu[first] += s * fu;
                rhv[first] += s * fv;
                let (m_last_x, m_last_y) = metric[N1 - 1];
                let (fh, fu, fv) = nodes.flux(last, last, m_last_x, m_last_y);
                let s = f64x8::<S>::splat(simd, 1.0 / w[N1 - 1]);
                rh[last] -= s * fh;
                rhu[last] -= s * fu;
                rhv[last] -= s * fv;

                // Face states of the subcells: limited reconstruction where
                // the stencil is complete, the node value otherwise
                let mut faces = [(data.side(0), data.side(0)); MAX_FACE_NODES];
                let mut x_left = -1.0;
                for a in 0..N1 {
                    let q = data.side(idx(a));
                    let x_right = x_left + w[a];
                    let (left, x_l, left_present) = if a > 0 {
                        let x = f64x8::<S>::splat(simd, xi[a - 1]);
                        (data.side(idx(a - 1)), x, zero.simd_ge(zero))
                    } else {
                        let end = ends[0];
                        let x = f64x8::<S>::splat(simd, -1.0) - end.distance;
                        (end.node, x, end.present)
                    };
                    let (right, x_r, right_present) = if a + 1 < N1 {
                        let x = f64x8::<S>::splat(simd, xi[a + 1]);
                        (data.side(idx(a + 1)), x, zero.simd_ge(zero))
                    } else {
                        let end = ends[1];
                        let x = f64x8::<S>::splat(simd, 1.0) + end.distance;
                        (end.node, x, end.present)
                    };
                    let x_c = f64x8::<S>::splat(simd, xi[a]);
                    let (face_l, face_r) = reconstruct_subcell(
                        simd,
                        [left, q, right],
                        [x_l, x_c, x_r],
                        (x_left, x_right),
                        h_dry,
                    );
                    let complete = left_present & right_present;
                    faces[a] = (q.select(complete, face_l), q.select(complete, face_r));
                    x_left = x_right;
                }

                // Interior subcell interfaces
                for a in 0..N1 - 1 {
                    let (ia, ib) = (idx(a), idx(a + 1));
                    let m = interfaces[a + 1];
                    let (f_a, f_b) =
                        hydrostatic_hll(simd, faces[a].1, faces[a + 1].0, (m.0, m.1), g, h_min);
                    mass[(direction * N1 + line) * (N1 - 1) + a] = f_a.h;
                    let s = f64x8::<S>::splat(simd, 1.0 / w[a]);
                    rh[ia] -= s * f_a.h;
                    rhu[ia] -= s * f_a.hu;
                    rhv[ia] -= s * f_a.hv;
                    let s = f64x8::<S>::splat(simd, 1.0 / w[a + 1]);
                    rh[ib] += s * f_b.h;
                    rhu[ib] += s * f_b.hu;
                    rhv[ib] += s * f_b.hv;
                }

                // Bed-slope term and the metric balance term (zero on
                // parallelograms)
                let minus_half_g = f64x8::<S>::splat(simd, -0.5 * g);
                let quarter_g = f64x8::<S>::splat(simd, 0.25 * g);
                for a in 0..N1 {
                    let ia = idx(a);
                    let (left, right) = faces[a];
                    let w_a = f64x8::<S>::splat(simd, w[a]);
                    let bed = minus_half_g * (left.h + right.h) * (right.b - left.b) / w_a;
                    let m0 = metric[0];
                    let (m_l, m_r) = (interfaces[a], interfaces[a + 1]);
                    let (h_a, b_a) = (data.h[ia], data.b[ia]);
                    let balance = quarter_g
                        * ((right.h + h_a) * (b_a - right.b) + (left.h + h_a) * (b_a - left.b))
                        / w_a;
                    let force_x = data.affine.select(
                        bed * m0.0,
                        bed * half * (m_l.0 + m_r.0) + balance * (m_r.0 - m_l.0),
                    );
                    let force_y = data.affine.select(
                        bed * m0.1,
                        bed * half * (m_l.1 + m_r.1) + balance * (m_r.1 - m_l.1),
                    );
                    rh[ia] += zero;
                    rhu[ia] += force_x;
                    rhv[ia] += force_y;
                }
            }
        }
    }
}

/// Largest per-element count of subcell interfaces, `2 N1 (N1 − 1)` at P4.
const MAX_SUBCELL_INTERFACES: usize = 2 * MAX_FACE_NODES * (MAX_FACE_NODES - 1);

/// The node data of a batch of elements, one per lane.
struct Batch<S: Simd, const NN: usize> {
    h: [f64x8<S>; NN],
    hu: [f64x8<S>; NN],
    hv: [f64x8<S>; NN],
    u: [f64x8<S>; NN],
    v: [f64x8<S>; NN],
    b: [f64x8<S>; NN],
    /// Contravariant vector of each line direction at every node,
    /// `[[J r_x, J r_y], [J s_x, J s_y]]`
    metric: [[[f64x8<S>; NN]; 2]; 2],
    /// Lanes of parallelograms
    affine: mask64x8<S>,
}

impl<S: Simd, const NN: usize> Batch<S, NN> {
    /// The state of node `i` as a face state.
    #[inline(always)]
    fn side(&self, i: usize) -> Side<S> {
        Side {
            h: self.h[i],
            u: self.u[i],
            v: self.v[i],
            b: self.b[i],
        }
    }

    #[inline(always)]
    fn nodes(&self, simd: S, g: f64) -> Nodes<'_, S, NN> {
        Nodes {
            h: &self.h,
            hu: &self.hu,
            hv: &self.hv,
            u: &self.u,
            v: &self.v,
            half: f64x8::<S>::splat(simd, 0.5),
            half_g: f64x8::<S>::splat(simd, 0.5 * g),
        }
    }
}

/// An outer node of a batch: the state, its distance from the face in the
/// element's reference coordinate, and the lanes where it exists.
#[derive(Clone, Copy)]
struct Outer<S: Simd> {
    node: Side<S>,
    distance: f64x8<S>,
    present: mask64x8<S>,
}

/// `f64::signum(x) · f64::min(|x|, bound)` for a nonzero `x`.
#[inline(always)]
fn signed_min<S: Simd>(simd: S, x: f64x8<S>, bound: f64x8<S>) -> f64x8<S> {
    let zero = f64x8::<S>::splat(simd, 0.0);
    let one = f64x8::<S>::splat(simd, 1.0);
    let sign = x.simd_lt(zero).select(-one, one);
    sign * min(x.abs(), bound)
}

/// `limited_slope`: the central slope of `q` at `xi[1]`, capped so that the
/// face values stay between the neighbouring node values; zero at an
/// extremum.
#[inline(always)]
fn limited_slope<S: Simd>(
    simd: S,
    q: [f64x8<S>; 3],
    xi: [f64x8<S>; 3],
    (x_l, x_r): (f64x8<S>, f64x8<S>),
) -> f64x8<S> {
    let zero = f64x8::<S>::splat(simd, 0.0);
    let half = f64x8::<S>::splat(simd, 0.5);
    let (dm, dp) = (q[1] - q[0], q[2] - q[1]);
    let extremum = (dm * dp).simd_le(zero);
    let central = (q[2] - q[0]) / (xi[2] - xi[0]);
    // bound(dq, face, node) = |dq / (face > 0 ? face : node / 2)|
    let (face_m, node_m) = (xi[1] - x_l, xi[1] - xi[0]);
    let (face_p, node_p) = (x_r - xi[1], xi[2] - xi[1]);
    let bound_m = (dm / face_m.simd_gt(zero).select(face_m, half * node_m)).abs();
    let bound_p = (dp / face_p.simd_gt(zero).select(face_p, half * node_p)).abs();
    let bound = min(bound_m, bound_p);
    extremum.select(zero, signed_min(simd, central, bound))
}

/// `reconstruct_subcell`: the face states of the subcell `[x_l, x_r]`
/// around `nodes[1]` from a limited linear reconstruction of `h`, `η`, `u`
/// and `v` (no `η` slope where a dry node lies below a wet surface of the
/// stencil).
#[inline(always)]
fn reconstruct_subcell<S: Simd>(
    simd: S,
    nodes: [Side<S>; 3],
    xi: [f64x8<S>; 3],
    (x_l, x_r): (f64, f64),
    h_dry: f64,
) -> (Side<S>, Side<S>) {
    let zero = f64x8::<S>::splat(simd, 0.0);
    let h_dry = f64x8::<S>::splat(simd, h_dry);
    let subcell = (f64x8::<S>::splat(simd, x_l), f64x8::<S>::splat(simd, x_r));
    let [m, c, p] = nodes;
    let eta = [m.h + m.b, c.h + c.b, p.h + p.b];
    let h = [m.h, c.h, p.h];
    let mut wet_surface = f64x8::<S>::splat(simd, f64::NEG_INFINITY);
    for (h, eta) in h.iter().zip(&eta) {
        // top.max(η) over the wet nodes
        let higher = eta.simd_gt(wet_surface);
        wet_surface = (h.simd_ge(h_dry) & higher).select(*eta, wet_surface);
    }
    let mut flooding = zero.simd_gt(zero);
    for (h, eta) in h.iter().zip(&eta) {
        flooding |= h.simd_lt(h_dry) & eta.simd_lt(wet_surface);
    }
    let s_h = limited_slope(simd, h, xi, subcell);
    let s_eta = flooding.select(zero, limited_slope(simd, eta, xi, subcell));
    let s_u = limited_slope(simd, [m.u, c.u, p.u], xi, subcell);
    let s_v = limited_slope(simd, [m.v, c.v, p.v], xi, subcell);
    let slopes = [s_h, s_eta, s_u, s_v];
    (
        reconstructed(simd, c, eta[1], slopes, subcell.0 - xi[1]),
        reconstructed(simd, c, eta[1], slopes, subcell.1 - xi[1]),
    )
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
    /// `other` in the lanes of `mask`, `self` elsewhere.
    #[inline(always)]
    fn select(self, mask: mask64x8<S>, other: Self) -> Self {
        Self {
            h: mask.select(other.h, self.h),
            u: mask.select(other.u, self.u),
            v: mask.select(other.v, self.v),
            b: mask.select(other.b, self.b),
        }
    }

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

/// The state at distance `d` from the node `c` (with surface `eta`) along
/// the limited slopes `[h, η, u, v]`, the depth kept ≥ 0 and the bed `η − h`.
#[inline(always)]
fn reconstructed<S: Simd>(
    simd: S,
    c: Side<S>,
    eta: f64x8<S>,
    [s_h, s_eta, s_u, s_v]: [f64x8<S>; 4],
    d: f64x8<S>,
) -> Side<S> {
    let h = max(c.h + s_h * d, f64x8::<S>::splat(simd, 0.0));
    Side {
        h,
        u: c.u + s_u * d,
        v: c.v + s_v * d,
        b: eta + s_eta * d - h,
    }
}
