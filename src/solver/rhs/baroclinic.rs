//! Baroclinic pressure gradient in σ-coordinates, computed at constant depth.
//!
//! # The σ-coordinate problem
//!
//! The horizontal pressure gradient force is `F = −(1/ρ₀) ∇p|_z`. On a
//! σ-surface the chain rule gives two large terms that must cancel,
//! `∇p|_z = ∇p|_σ + gρ ∇z|_σ`, and over steep bathymetry any inconsistency in
//! their discretisations appears as a spurious force as large as the physical
//! one (Haney 1991; Mellor et al. 1994). The previous discretisation — the
//! element-local derivative of `p` along σ plus `gρ∇z`, with a first-order
//! half-layer pressure — made ~1e-4 m/s² in a stratified fjord at rest.
//!
//! # Pressure differences at a common depth
//!
//! Instead, every term of the DG derivative is a pressure difference taken at
//! one depth (Stelling & van Kester 1994, in a DG form). With `D` the
//! element's nodal differentiation matrix (`Dx = r_x D_r + s_x D_s`),
//!
//! ```text
//!     (∂p/∂x|_z)_i ≈ Σ_j Dx_ij Δp_ij,
//!     Δp_ij = ½ [ (p_j(z_i) − p_i(z_i)) + (p_j(z_j) − p_i(z_j)) ],
//! ```
//!
//! where `p_c(z)` is the hydrostatic pressure of column `c` evaluated at depth
//! `z`, and `z_i`, `z_j` are the depths of nodes `i`, `j` on the σ-level.
//! `Δp_ij` is antisymmetric and a smooth function of the node positions, so
//! the sum is a high-order, consistent approximation of `∂p/∂x|_z` (`Σ_j Dx_ij
//! (x_j − x_i) = 1`). Where a node lies below the other column's bed (a
//! hydrostatically inconsistent pair, the discrete Haney condition), only the
//! difference at the shallower node's depth, which both columns reach, is
//! used; the scheme is first order there but stays balanced.
//!
//! Each column's density anomaly `ρ − ρ_ref` is a piecewise Hermite cubic in
//! `z` through the level values, with monotone harmonic-mean slopes
//! (Shchepetkin & McWilliams 2003, §5), continued linearly above the top and
//! below the bottom level. `p_c(z)` integrates it exactly from the surface.
//!
//! **Balance.** In a fluid at rest with `ρ = ρ(z)` every column samples the
//! same profile, so `Δp_ij` is the difference of two interpolants of one
//! profile: zero for any bathymetry and stretching when `ρ` is linear in `z`
//! (constant `N²`), and the interpolation error otherwise — not amplified by
//! the slope as in the σ form. A horizontally uniform `ρ(x)` over a flat bed
//! is exact too.
//!
//! **Faces.** The element-local sum ignores the neighbours. The jump of the
//! pressure across each face, taken at a common depth in the same way, is
//! lifted with a central flux (`p* − p⁻ = ½Δp`), so a density front (or an
//! `η` jump, for the full PGF) sitting on a face exerts its force. In a fluid
//! at rest the jump is zero.
//!
//! **Reference density.** The integrated density is `ρ − rho_ref`:
//! `rho_ref = 0` gives the full PGF, whose `ρ`-uniform part is `−g∇η`
//! (DG-coupled through the face lift); `rho_ref = ρ₀` gives the
//! baroclinic-only PGF of mode splitting, where `−g∇η` comes from the 2D
//! module.

use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::core::blocks::{Pooled, for_each_block};
use crate::solver::state::Solution3D;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// Level spacing (m) below which a column is treated as collapsed: its
/// density slope is taken as zero.
const MIN_LEVEL_SPACING: f64 = 1e-9;

/// One water column's density anomaly as a piecewise Hermite cubic in `z`,
/// and its hydrostatic pressure (per `g`) at the level centres.
#[derive(Clone, Copy)]
struct ColumnView<'a> {
    /// Level-centre depths, bottom first (ascending).
    z: &'a [f64],
    /// Density anomaly at the levels.
    rho: &'a [f64],
    /// `∂ρ/∂z` at the levels (monotone harmonic mean).
    slope: &'a [f64],
    /// `∫_z^η (ρ − ρ_ref) dz'` at the levels.
    pressure: &'a [f64],
    /// The same as [`Self::pressure_at`] evaluates it at the levels: equal
    /// to `pressure` but for rounding. The pressure differences take this,
    /// so that their own and cross-column terms round alike (round-off at
    /// rest stays ≤ 1.5e-14 m/s², not 3e-14).
    at_level: &'a [f64],
    /// Bed elevation.
    bed: f64,
    /// At least the minimum column depth deep.
    wet: bool,
}

impl ColumnView<'_> {
    /// `∫_z^η (ρ − ρ_ref) dz'` at depth `z`, and whether `z` is in the water
    /// column (above the bed). `cursor` is a level at or below the interval
    /// that holds `z` (`z[cursor] ≤ z`, or 0); it is moved up to that
    /// interval, so that a caller evaluating at rising depths finds each
    /// interval by stepping, not searching.
    #[inline]
    fn pressure_at(&self, z: f64, cursor: &mut usize) -> (f64, bool) {
        let n = self.z.len();
        let valid = z >= self.bed;
        let (top, bottom) = (n - 1, 0);
        let p = if z >= self.z[top] {
            let dz = z - self.z[top];
            self.pressure[top] - (self.rho[top] * dz + 0.5 * self.slope[top] * dz * dz)
        } else if z <= self.z[bottom] {
            let dz = self.z[bottom] - z;
            self.pressure[bottom] + self.rho[bottom] * dz - 0.5 * self.slope[bottom] * dz * dz
        } else {
            // z[m] ≤ z < z[m + 1]; z < z[top] ends the walk
            while self.z[*cursor + 1] <= z {
                *cursor += 1;
            }
            let m = *cursor;
            let h = self.z[m + 1] - self.z[m];
            let t = (z - self.z[m]) / h;
            self.pressure[m + 1]
                + h * (hermite_integral(1.0, self, m, h) - hermite_integral(t, self, m, h))
        };
        (p, valid)
    }
}

/// `(1/h) ∫_{z_m}^{z_m + t h}` of the Hermite cubic on level interval `m`.
#[inline]
fn hermite_integral(t: f64, column: &ColumnView, m: usize, h: f64) -> f64 {
    hermite_integral_of(t, column.rho, column.slope, m, h)
}

/// [`hermite_integral`] of the profile `rho` with slopes `slope`.
#[inline]
fn hermite_integral_of(t: f64, rho: &[f64], slope: &[f64], m: usize, h: f64) -> f64 {
    let (t2, t3, t4) = (t * t, t * t * t, t * t * t * t);
    let h00 = 0.5 * t4 - t3 + t;
    let h10 = 0.25 * t4 - 2.0 / 3.0 * t3 + 0.5 * t2;
    let h01 = -0.5 * t4 + t3;
    let h11 = 0.25 * t4 - t3 / 3.0;
    rho[m] * h00 + h * slope[m] * h10 + rho[m + 1] * h01 + h * slope[m + 1] * h11
}

/// `Δp_l = ½[(p_b(z_a) − p_a(z_a)) + (p_b(z_b) − p_a(z_b))]` (per `g`) at
/// every level `l`, with `z_a`, `z_b` the two columns' own depths of that
/// level, into `out`, using only the depths both columns reach; zero if
/// either column is thin (a dry bank exerts no pressure on the water beside
/// it).
///
/// A column's pressure at its own level is its table value, and the depths
/// at which each column is evaluated in the other rise with the level, so
/// one upward walk per column finds every interval.
fn pressure_differences(a: &ColumnView, b: &ColumnView, out: &mut [f64]) {
    if !(a.wet && b.wet) {
        out.fill(0.0);
        return;
    }
    let (mut cursor_a, mut cursor_b) = (0, 0);
    for (l, dp) in out.iter_mut().enumerate() {
        let (pb_at_a, b_reaches_a) = b.pressure_at(a.z[l], &mut cursor_b);
        let (pa_at_b, a_reaches_b) = a.pressure_at(b.z[l], &mut cursor_a);
        let (pa, pb) = (a.at_level[l], b.at_level[l]);
        *dp = match (b_reaches_a, a_reaches_b) {
            (true, true) => 0.5 * ((pb_at_a - pa) + (pb - pa_at_b)),
            (true, false) => pb_at_a - pa,
            (false, true) => pb - pa_at_b,
            // Impossible for columns with z ≥ bed at their own nodes (one of
            // the two depths is the shallower one, which both reach); keep
            // the symmetric form for degenerate input.
            (false, false) => 0.5 * ((pb_at_a - pa) + (pb - pa_at_b)),
        };
    }
}

/// Per-column profiles of the nodes of one element or face, stored flat as
/// `[node][level]`.
struct Columns {
    n_levels: usize,
    z: Vec<f64>,
    rho: Vec<f64>,
    slope: Vec<f64>,
    pressure: Vec<f64>,
    at_level: Vec<f64>,
    bed: Vec<f64>,
    wet: Vec<bool>,
    min_column_depth: f64,
}

impl Columns {
    fn new(n_columns: usize, n_levels: usize, min_column_depth: f64) -> Self {
        let n = n_columns * n_levels;
        Self {
            min_column_depth,
            wet: vec![true; n_columns],
            n_levels,
            z: vec![0.0; n],
            rho: vec![0.0; n],
            slope: vec![0.0; n],
            pressure: vec![0.0; n],
            at_level: vec![0.0; n],
            bed: vec![0.0; n_columns],
        }
    }

    fn view(&self, c: usize) -> ColumnView<'_> {
        let range = c * self.n_levels..(c + 1) * self.n_levels;
        ColumnView {
            z: &self.z[range.clone()],
            rho: &self.rho[range.clone()],
            slope: &self.slope[range.clone()],
            pressure: &self.pressure[range.clone()],
            at_level: &self.at_level[range],
            bed: self.bed[c],
            wet: self.wet[c],
        }
    }

    /// Fill column `c` from `ρ − rho_ref` of the state's column at (`el`,
    /// `node`).
    #[allow(clippy::too_many_arguments)]
    fn fill(
        &mut self,
        c: usize,
        state: &Solution3D,
        bathymetry: &Bathymetry2D,
        sigma: &SigmaGrid,
        el: ElementIndex,
        node: usize,
        rho_ref: f64,
    ) {
        let nl = self.n_levels;
        let range = c * nl..(c + 1) * nl;
        let eta = state.eta.get(el.as_usize(), node);
        let bed = bathymetry.get(el, node);
        let depth = eta - bed;
        self.bed[c] = bed;
        self.wet[c] = depth >= self.min_column_depth;
        let z = &mut self.z[range.clone()];
        let rho = &mut self.rho[range.clone()];
        let slope = &mut self.slope[range.clone()];
        let pressure = &mut self.pressure[range.clone()];
        let at_level = &mut self.at_level[range];
        for (l, (zl, &s)) in z.iter_mut().zip(sigma.sigma_rho()).enumerate() {
            *zl = eta + s * depth;
            rho[l] = state.rho_column(el, node)[l] - rho_ref;
        }

        // Monotone slopes: harmonic mean of the neighbouring secants, zero at
        // extrema; one-sided at the ends. Exact for linear ρ(z).
        let secant = |l: usize| {
            let h = z[l + 1] - z[l];
            if h > MIN_LEVEL_SPACING {
                (rho[l + 1] - rho[l]) / h
            } else {
                0.0
            }
        };
        if nl == 1 {
            slope[0] = 0.0;
        } else {
            slope[0] = secant(0);
            slope[nl - 1] = secant(nl - 2);
            for (l, s) in slope.iter_mut().enumerate().take(nl - 1).skip(1) {
                let (a, b) = (secant(l - 1), secant(l));
                *s = if a * b > 0.0 {
                    2.0 * a * b / (a + b)
                } else {
                    0.0
                };
            }
        }

        // ∫_z^η from the surface down: the top half-layer with the linear
        // continuation, then exact integrals of the cubic between levels
        let dz = eta - z[nl - 1];
        pressure[nl - 1] = rho[nl - 1] * dz + 0.5 * slope[nl - 1] * dz * dz;
        for l in (0..nl - 1).rev() {
            let h = z[l + 1] - z[l];
            pressure[l] = pressure[l + 1]
                + h * (0.5 * (rho[l] + rho[l + 1]) + h * (slope[l] - slope[l + 1]) / 12.0);
        }

        // `pressure_at(z[l])`: the end levels exactly, the others from the
        // interval above them at t = 0
        for (l, p) in at_level.iter_mut().enumerate() {
            *p = if l == 0 || l == nl - 1 {
                pressure[l]
            } else {
                let h = z[l + 1] - z[l];
                pressure[l + 1]
                    + h * (hermite_integral_of(1.0, rho, slope, l, h)
                        - hermite_integral_of(0.0, rho, slope, l, h))
            };
        }
    }
}

/// Compute the σ-coordinate pressure gradient force at constant depth (see
/// the module docs).
///
/// Output is stored in `grad_px` and `grad_py` with layout `[element][node][level]`.
/// The result is the force per unit mass: -1/ρ₀ ∇p.
///
/// The density used in the hydrostatic pressure and its gradient is `ρ − rho_ref`.
/// This lets the caller select which part of the PGF is computed:
///
/// - `rho_ref = 0` → **full** PGF (barotropic + baroclinic). The barotropic part
///   is exactly the free-surface term `−g∇η`.
/// - `rho_ref = ρ₀` → **baroclinic-only** PGF. The ρ₀ contribution integrates to
///   exactly `−g∇η`, so subtracting ρ₀ removes the barotropic term. Under mode
///   splitting the barotropic `−g∇η` is supplied by the 2D sub-model's `½gh²`
///   flux, so the 3D internal mode must use the baroclinic-only PGF to avoid
///   double-counting the surface pressure gradient (see `ModeSplitIntegrator`).
///
/// # Arguments
/// * `state` - 3D solution state (contains η and ρ)
/// * `mesh` - Mesh (face neighbours, for the lifted pressure jumps)
/// * `bathymetry` - Bed elevation B
/// * `sigma` - Vertical grid configuration
/// * `ops` - 2D DG operators (for gradients)
/// * `geom` - Geometric factors (affine elements)
/// * `g` - Gravitational acceleration (m/s²)
/// * `rho_0` - Reference density ρ₀ used in the `-1/ρ₀` normalization (kg/m³)
/// * `rho_ref` - Density subtracted from ρ before integrating (0 = full PGF, ρ₀ = baroclinic-only)
/// * `min_column_depth` - Columns shallower than this (m) exert and feel no
///   pressure difference (3D wetting and drying)
/// * `grad_px` - Output x-component of PGF (m/s²)
/// * `grad_py` - Output y-component of PGF (m/s²)
#[allow(clippy::too_many_arguments)]
pub fn compute_pressure_gradient(
    state: &Solution3D,
    mesh: &Mesh2D,
    bathymetry: &Bathymetry2D,
    sigma: &SigmaGrid,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    g: f64,
    rho_0: f64,
    rho_ref: f64,
    min_column_depth: f64,
    grad_px: &mut [f64],
    grad_py: &mut [f64],
) {
    let (nn, nfn) = (ops.n_nodes, ops.n_face_nodes);
    let nl = sigma.n_levels();
    let scale = -g / rho_0;

    let n = state.n_elements * nn * nl;
    for_each_block(
        state.n_elements,
        [&mut grad_px[..n], &mut grad_py[..n]],
        || {
            let mut scratch = Pooled::take(
                |s: &PressureScratch| s.fits(nn, nfn, nl),
                || PressureScratch::new(nn, nfn, nl),
            );
            scratch.own.min_column_depth = min_column_depth;
            scratch.across.min_column_depth = min_column_depth;
            scratch
        },
        |scratch, k, [out_x, out_y]| {
            let PressureScratch {
                own,
                across,
                px,
                py,
                dp,
            } = &mut **scratch;
            pressure_gradient_element(
                k, state, mesh, bathymetry, sigma, ops, geom, rho_ref, own, across, px, py, dp,
            );
            for ((fx, fy), (&dx, &dy)) in out_x.iter_mut().zip(out_y).zip(px.iter().zip(&*py)) {
                *fx = scale * dx;
                *fy = scale * dy;
            }
        },
    );
}

/// Buffers of one element of [`compute_pressure_gradient`].
struct PressureScratch {
    own: Columns,
    across: Columns,
    /// ∂p/∂x, ∂p/∂y per g, [node][level]
    px: Vec<f64>,
    py: Vec<f64>,
    /// Δp of one pair of columns, [level]
    dp: Vec<f64>,
}

impl PressureScratch {
    fn new(nn: usize, nfn: usize, nl: usize) -> Self {
        Self {
            own: Columns::new(nn, nl, 0.0),
            across: Columns::new(nfn, nl, 0.0),
            px: vec![0.0; nn * nl],
            py: vec![0.0; nn * nl],
            dp: vec![0.0; nl],
        }
    }

    fn fits(&self, nn: usize, nfn: usize, nl: usize) -> bool {
        self.own.n_levels == nl && self.own.bed.len() == nn && self.across.bed.len() == nfn
    }
}

/// `∂p/∂x`, `∂p/∂y` per `g` of element `k` into `px`, `py` (`[node][level]`),
/// with the columns of its nodes in `own` and of a face's neighbours in
/// `across`, and one pair's pressure differences in `dp` (see
/// [`compute_pressure_gradient`]).
#[allow(clippy::too_many_arguments)]
fn pressure_gradient_element(
    k: usize,
    state: &Solution3D,
    mesh: &Mesh2D,
    bathymetry: &Bathymetry2D,
    sigma: &SigmaGrid,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    rho_ref: f64,
    own: &mut Columns,
    across: &mut Columns,
    px: &mut [f64],
    py: &mut [f64],
    dp: &mut [f64],
) {
    let (nn, nfn) = (ops.n_nodes, ops.n_face_nodes);
    let nl = sigma.n_levels();
    {
        let el = ElementIndex::new(k);
        for i in 0..nn {
            own.fill(i, state, bathymetry, sigma, el, i, rho_ref);
        }
        px.fill(0.0);
        py.fill(0.0);

        // Volume term: Σ_j Dx_ij Δp_ij, with Δp_ji = −Δp_ij
        let metric = geom.affine_metric(k);
        let (rx, ry, sx, sy) = (metric.rx, metric.ry, metric.sx, metric.sy);
        for i in 0..nn {
            let a = own.view(i);
            for j in i + 1..nn {
                let b = own.view(j);
                let (dr_ij, ds_ij) = (ops.dr[(i, j)], ops.ds[(i, j)]);
                let (dr_ji, ds_ji) = (ops.dr[(j, i)], ops.ds[(j, i)]);
                let (dx_ij, dy_ij) = (rx * dr_ij + sx * ds_ij, ry * dr_ij + sy * ds_ij);
                let (dx_ji, dy_ji) = (rx * dr_ji + sx * ds_ji, ry * dr_ji + sy * ds_ji);
                pressure_differences(&a, &b, dp);
                for (l, &dp) in dp.iter().enumerate() {
                    px[i * nl + l] += dx_ij * dp;
                    py[i * nl + l] += dy_ij * dp;
                    px[j * nl + l] -= dx_ji * dp;
                    py[j * nl + l] -= dy_ji * dp;
                }
            }
        }

        // Face terms: LIFT (n·(p* − p⁻)), p* − p⁻ = ½ Δp at a common depth
        for f in 0..4 {
            let Some(nb) = mesh.neighbor(el, f) else {
                continue;
            };
            let nb_el = ElementIndex::new(nb.element);
            for fi in 0..nfn {
                let nb_node = ops.face_nodes[nb.face][nfn - 1 - fi];
                across.fill(fi, state, bathymetry, sigma, nb_el, nb_node, rho_ref);
            }
            let (normal, s_jac) = geom.affine_face(k, f);
            let lift_scale = s_jac * metric.det_j_inv;
            for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
                pressure_differences(&own.view(node), &across.view(fi), dp);
                for (l, &dp) in dp.iter().enumerate() {
                    let jump = 0.5 * dp;
                    if jump == 0.0 {
                        continue;
                    }
                    let (jx, jy) = (lift_scale * normal.0 * jump, lift_scale * normal.1 * jump);
                    for i in 0..nn {
                        let lift = ops.lift[f][(i, fi)];
                        px[i * nl + l] += lift * jx;
                        py[i * nl + l] += lift * jy;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vertical::{SongHaidvogelStretching, UniformStretching};

    const G: f64 = 9.81;
    const RHO0: f64 = 1025.0;

    /// A channel `[0, length] × [0, width]` with the bed `bed(x, y)`, the
    /// density `rho(x, z)` at every node and level, and `η = eta(x)`.
    struct Column3D {
        mesh: Mesh2D,
        ops: DGOperators2D,
        geom: GeometricFactors2D,
        sigma: SigmaGrid,
        bathymetry: Bathymetry2D,
        state: Solution3D,
    }

    impl Column3D {
        #[allow(clippy::too_many_arguments)]
        fn new(
            length: f64,
            nx: usize,
            order: usize,
            sigma: SigmaGrid,
            bed: impl Fn(f64, f64) -> f64,
            eta: impl Fn(f64) -> f64,
            rho: impl Fn(f64, f64) -> f64,
        ) -> Self {
            let mesh = Mesh2D::uniform_rectangle(0.0, length, 0.0, length / nx as f64, nx, 1);
            let ops = DGOperators2D::new(order);
            let geom = GeometricFactors2D::compute(&mesh, &ops);
            let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, bed);
            let nl = sigma.n_levels();
            let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, nl);
            for k in 0..mesh.n_elements {
                let el = ElementIndex::new(k);
                for i in 0..ops.n_nodes {
                    let [x, _] = mesh.reference_to_physical(el, ops.nodes_r[i], ops.nodes_s[i]);
                    let idx = k * ops.n_nodes + i;
                    let e = eta(x);
                    state.eta.data[idx] = e;
                    let depth = e - bathymetry.data[idx];
                    for (l, &s) in sigma.sigma_rho().iter().enumerate() {
                        state.rho[idx * nl + l] = rho(x, e + s * depth);
                    }
                }
            }
            Self {
                mesh,
                ops,
                geom,
                sigma,
                bathymetry,
                state,
            }
        }

        /// The PGF (x, y) at every node and level.
        fn pgf(&self, rho_ref: f64) -> (Vec<f64>, Vec<f64>) {
            let n = self.state.rho.len();
            let (mut fx, mut fy) = (vec![f64::NAN; n], vec![f64::NAN; n]);
            compute_pressure_gradient(
                &self.state,
                &self.mesh,
                &self.bathymetry,
                &self.sigma,
                &self.ops,
                &self.geom,
                G,
                RHO0,
                rho_ref,
                0.0,
                &mut fx,
                &mut fy,
            );
            (fx, fy)
        }

        /// Largest |F| over every node and level.
        fn max_force(&self, rho_ref: f64) -> f64 {
            let (fx, fy) = self.pgf(rho_ref);
            // NaN-propagating (f64::max would drop a NaN)
            fx.iter()
                .zip(&fy)
                .map(|(x, y)| x.hypot(*y))
                .fold(0.0, |m, f| if f.is_nan() || f > m { f } else { m })
        }
    }

    /// The review's fjord (REVIEW.md §3.3): the bed drops from 30 to 400 m
    /// over ≈ 1.5 km.
    fn fjord_bed(x: f64, _y: f64) -> f64 {
        -(30.0 + 185.0 * (1.0 + ((x - 1500.0) / 375.0).tanh()))
    }

    /// 5 kg/m³ lighter in the upper 20 m.
    fn pycnocline(z: f64) -> f64 {
        RHO0 - 2.5 + 2.5 * (-(z + 10.0) / 3.0).tanh()
    }

    /// TODO P4.3 gate: constant N² at rest over the steep fjord slope, at
    /// every order and for stretched levels, is balanced to round-off (the
    /// σ form: 2.6e-6 m/s² for N² = 1e-4 s⁻² on a 30→400 m slope at every
    /// resolution, REVIEW.md §3.3(a)).
    #[test]
    fn constant_n2_at_rest_is_balanced_over_a_steep_slope() {
        // N² = 1e-4 s⁻²: ∂ρ/∂z = −ρ₀N²/g
        let n2 = 1e-4;
        let linear = |_: f64, z: f64| RHO0 - RHO0 * n2 / G * z;
        for order in 1..=4 {
            for sigma in [
                SigmaGrid::new(30, UniformStretching),
                SigmaGrid::new(20, SongHaidvogelStretching::new(5.0, 0.4, 10.0)),
            ] {
                let case = Column3D::new(3000.0, 6, order, sigma, fjord_bed, |_| 0.0, linear);
                for rho_ref in [RHO0, 0.0] {
                    // Measured ≤ 1.5e-14 (the σ form: 1.3e-5 – 2.5e-3)
                    let force = case.max_force(rho_ref);
                    assert!(
                        force < 1e-13,
                        "P{order}, rho_ref {rho_ref}: spurious |F| = {force:.3e} m/s²"
                    );
                }
            }
        }
    }

    /// TODO P4.3 gate: the review's stratified fjord at rest (a 5 kg/m³
    /// pycnocline in the upper 20 m over a 30→400 m slope), on 30
    /// surface-stretched levels (θs = 5, θb = 0.4), against ≈ 5e-5 m/s² of
    /// real estuarine forcing. Spurious |F| (m/s²):
    ///
    /// | | P1, 500 m | P3, 250 m | P3, 100 m |
    /// |---|---|---|---|
    /// | σ form | 1.6e-3 | 1.1e-4 | 2.2e-5 |
    /// | constant depth | 2.0e-7 | 1.5e-6 | 3.3e-6 |
    ///
    /// On 30 *uniform* levels both schemes stay at 2e-4 – 1.6e-3 (new 2.0e-4
    /// – 4.1e-4): 13 m apart at 400 m, the deep columns do not sample the 3 m
    /// pycnocline, so no column-to-column comparison can match them. That is
    /// a vertical-resolution requirement (resolve the pycnocline in every
    /// column, as NorKyst's stretched levels do), not a PGF one.
    #[test]
    fn stratified_fjord_at_rest_has_a_small_spurious_force() {
        for (order, nx) in [(1, 6), (3, 12), (3, 30)] {
            let case = Column3D::new(
                3000.0,
                nx,
                order,
                SigmaGrid::new(30, SongHaidvogelStretching::new(5.0, 0.4, 10.0)),
                fjord_bed,
                |_| 0.0,
                |_, z| pycnocline(z),
            );
            let force = case.max_force(RHO0);
            assert!(
                force < 5e-6,
                "P{order}, {} m: spurious |F| = {force:.3e} m/s²",
                3000 / nx
            );
        }
    }

    /// A density that varies only horizontally, over a flat bed: the force
    /// is `−(g/ρ₀)(η − z) ∂ρ/∂x`, reproduced exactly for a polynomial `ρ(x)`
    /// of the element's degree.
    #[test]
    fn horizontal_density_gradient_over_a_flat_bed_is_exact() {
        let (a, depth) = (2e-4, 50.0);
        let case = Column3D::new(
            4000.0,
            4,
            2,
            SigmaGrid::new(10, UniformStretching),
            |_, _| -depth,
            |_| 0.0,
            |x, _| RHO0 + a * x,
        );
        let (fx, fy) = case.pgf(RHO0);
        let nl = case.sigma.n_levels();
        for (idx, (&x_force, &y_force)) in fx.iter().zip(&fy).enumerate() {
            let z = case.sigma.sigma_rho()[idx % nl] * depth;
            let expected = -G / RHO0 * (0.0 - z) * a;
            assert!(
                (x_force - expected).abs() < 1e-9 * expected.abs().max(1e-6),
                "{x_force:.6e} vs {expected:.6e}"
            );
            assert!(y_force.abs() < 1e-15, "{y_force:.3e}");
        }
    }

    /// Constant density over a sloping bed with a flat surface: no force, for
    /// the full PGF too (the old σ form needed exact `∇B` for this).
    #[test]
    fn constant_density_over_a_sloping_bed_has_no_force() {
        let case = Column3D::new(
            1000.0,
            2,
            1,
            SigmaGrid::new(5, UniformStretching),
            |x, _| -100.0 - 0.1 * x,
            |_| 0.0,
            |_, _| 1000.0,
        );
        for rho_ref in [0.0, 1000.0] {
            let force = case.max_force(rho_ref);
            assert!(force < 1e-12, "rho_ref {rho_ref}: |F| = {force:.3e}");
        }
    }

    /// Regression (TODO P0.6): with constant density ρ = ρ₀ and a tilted free
    /// surface over a flat bottom, the FULL PGF must be the barotropic term
    /// −g∇η, while the BAROCLINIC-ONLY PGF (rho_ref = ρ₀) must vanish. If the
    /// baroclinic path still carried −g∇η, the mode-split G-term would double
    /// the surface pressure gradient already supplied by the 2D ½gh² flux.
    #[test]
    fn baroclinic_only_pgf_excludes_surface_slope() {
        let slope = 1.0e-4;
        let case = Column3D::new(
            1000.0,
            4,
            1,
            SigmaGrid::new(4, UniformStretching),
            |_, _| -100.0,
            |x| slope * x,
            |_, _| RHO0,
        );
        let (full_x, full_y) = case.pgf(0.0);
        let expected = -G * slope;
        let x_err = full_x
            .iter()
            .map(|v| (v - expected).abs())
            .fold(0.0, f64::max);
        let y_err = full_y.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            x_err < 1e-9,
            "full PGF x error {x_err:.3e} (−g∇η = {expected:.3e})"
        );
        assert!(y_err < 1e-9, "full PGF y {y_err:.3e}");
        let baroclinic = case.max_force(RHO0);
        assert!(baroclinic < 1e-12, "baroclinic-only PGF {baroclinic:.3e}");
    }

    /// A density front on an element face: the element-local sums see two
    /// uniform elements and no force; the lifted pressure jump gives the
    /// front its force, and the force on the two sides balances (it only
    /// moves momentum across the face).
    #[test]
    fn density_front_on_a_face_exerts_a_force() {
        let depth = 20.0;
        let case = Column3D::new(
            2000.0,
            2,
            2,
            SigmaGrid::new(4, UniformStretching),
            |_, _| -depth,
            |_| 0.0,
            |_, _| RHO0,
        );
        let mut case = case;
        let nl = case.sigma.n_levels();
        let nn = case.ops.n_nodes;
        // Heavier water in the right element, up to the shared face
        case.state.rho[nn * nl..].fill(RHO0 + 1.0);
        let (fx, _) = case.pgf(RHO0);
        // Depth-integrated, area-weighted force on each element
        let force = |k: usize| -> f64 {
            let per_node: Vec<f64> = (0..nn)
                .map(|i| (0..nl).map(|l| fx[(k * nn + i) * nl + l]).sum::<f64>())
                .collect();
            case.geom.integrate_element(k, &per_node)
        };
        let (left, right) = (force(0), force(1));
        // Heavier water on the right pushes towards the left, on both sides
        assert!(
            left < 0.0 && right < 0.0,
            "left {left:.3e}, right {right:.3e}"
        );
        assert!(
            (left - right).abs() < 1e-12 * left.abs(),
            "the face force is shared: left {left:.6e}, right {right:.6e}"
        );
    }

    /// Smooth baroclinic field over a gentle slope: `ρ' = a sin(kx) e^{z/H}`,
    /// `∂p/∂x|_z = g a k H cos(kx) (1 − e^{z/H})`, at P2 under joint
    /// refinement of the mesh and the levels. The nodal P2 derivative is
    /// second order; measured ratios 1.69, 1.80, 1.75 (1.74 over a flat bed, so
    /// not the one-sided pairs at the bottom of the slope).
    #[test]
    fn smooth_baroclinic_force_converges() {
        let (a, h_scale, length) = (0.5, 40.0, 8000.0);
        let wavenumber = 2.0 * std::f64::consts::PI / length;
        let errors: Vec<f64> = [(4, 10), (8, 20), (16, 40), (32, 80)]
            .iter()
            .map(|&(nx, levels)| {
                let case = Column3D::new(
                    length,
                    nx,
                    2,
                    SigmaGrid::new(levels, UniformStretching),
                    |x, _| -100.0 - 0.005 * x,
                    |_| 0.0,
                    |x, z| RHO0 + a * (wavenumber * x).sin() * (z / h_scale).exp(),
                );
                let (fx, _) = case.pgf(RHO0);
                let nl = levels;
                let mut err = 0.0_f64;
                for k in 0..case.mesh.n_elements {
                    let el = ElementIndex::new(k);
                    for i in 0..case.ops.n_nodes {
                        let [x, _] = case.mesh.reference_to_physical(
                            el,
                            case.ops.nodes_r[i],
                            case.ops.nodes_s[i],
                        );
                        let idx = k * case.ops.n_nodes + i;
                        let depth = -case.bathymetry.data[idx];
                        for (l, &s) in case.sigma.sigma_rho().iter().enumerate() {
                            let z = s * depth;
                            let exact = -G / RHO0
                                * a
                                * wavenumber
                                * h_scale
                                * (wavenumber * x).cos()
                                * (1.0 - (z / h_scale).exp());
                            err = err.max((fx[idx * nl + l] - exact).abs());
                        }
                    }
                }
                err
            })
            .collect();
        for pair in errors.windows(2) {
            let order = (pair[0] / pair[1]).log2();
            assert!(order > 1.6, "order {order:.2} (errors {errors:?})");
        }
    }
}
