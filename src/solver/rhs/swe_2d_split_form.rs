//! Split-form DGSEM element kernel for the 2D SWE with bathymetry
//! (Wintermeyer, Winters, Gassner & Kopriva 2017).
//!
//! Selected with `SWEFormulation2D::EntropyStable` or `EntropyConservative`;
//! the two-point fluxes live in `flux::swe_2d_entropy`.
//!
//! # Semi-discrete form
//!
//! On a tensor-product GLL element with constant (affine) metrics, node
//! `i = (a, j)` lies on r-line `j` and s-line `a`. With `D` the 1D GLL
//! differentiation matrix, the element update is
//!
//! ```text
//! dq_i/dt = − Σ_c 2 D_ac F#(q_i, q_(c,j))·(r_x, r_y)
//!           − Σ_c 2 D_jc F#(q_i, q_(a,c))·(s_x, s_y)
//!           − g h_i (0, ∂_x B, ∂_y B)_i
//!           + J⁻¹ Σ_f LIFT_f sJ_f (F(q⁻)·n − F*)
//!
//! F* = F#(q⁻, q⁺)·n + ½g h⁻ (B⁺ − B⁻)(0, n)    [ − ½λ H̄[[w]]  for EntropyStable ]
//! ```
//!
//! where `(∂_x B, ∂_y B)_i = (r_x, r_y) Σ_c D_ac B_(c,j) + (s_x, s_y) Σ_c D_jc B_(a,c)`
//! is the collocated bed derivative taken with the same `D`, and `F(q⁻) = F#(q⁻, q⁻)`.
//!
//! # Properties
//!
//! For any nodal bathymetry, continuous or discontinuous across faces:
//! - **Mass conservation**: the mass component of `F#` is symmetric and `D`
//!   is SBP (`M D + (M D)ᵀ = diag(−1, 0, …, 0, 1)`), so the volume term
//!   telescopes to the face flux and neighbours exchange exactly `∓F*`.
//! - **Well-balanced**: at lake at rest the pressure part of the volume term is
//!   `g h_i Σ_c D_ac h_c`, cancelled by `g h_i Σ_c D_ac B_c` up to round-off
//!   (`Σ_c D_ac η = 0`); at faces `½g h⁻h⁺ + ½g h⁻(B⁺ − B⁻) = ½g h⁻²`.
//!   Unlike the collocated formulation this holds for bathymetry of any degree.
//! - **Entropy**: `EntropyConservative` conserves the total energy
//!   `∫ ½h|u|² + ½gh² + ghB` on periodic domains (up to round-off);
//!   `EntropyStable` dissipates it at faces. The split form also removes the
//!   aliasing of the collocated nonlinear flux derivative (REVIEW.md §1.6).
//!
//! Wetting and drying are not treated: dry nodes get zero velocity but the
//! scheme is not positivity preserving by itself.

use crate::boundary::SWEBoundaryCondition2D;
use crate::flux::{
    SWENodeState2D, entropy_stable_dissipation_2d, wintermeyer_bed_interface_term_2d,
    wintermeyer_flux_2d,
};
use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::{SWESolution2D, SWEState2D};
use crate::types::ElementIndex;

use super::swe_2d::{SWE2DRhsConfig, SWEFormulation2D};

/// Per-thread scratch space for [`SplitFormSWE2D::element_rhs`].
pub(super) struct SplitFormWorkspace {
    /// Node states (with bed and velocity) of the current element
    nodes: Vec<SWENodeState2D>,
    /// RHS accumulator of the current element
    rhs: Vec<SWEState2D>,
}

impl SplitFormWorkspace {
    pub(super) fn new(n_nodes: usize) -> Self {
        // Unused trailing capacity keeps workspaces of different threads off
        // shared cache lines (see `padded` in swe_2d.rs).
        const SLACK: usize = 4;
        let mut nodes = Vec::with_capacity(n_nodes + SLACK);
        nodes.resize(n_nodes, SWENodeState2D::default());
        let mut rhs = Vec::with_capacity(n_nodes + SLACK);
        rhs.resize(n_nodes, SWEState2D::zero());
        Self { nodes, rhs }
    }
}

/// Split-form operator for one RHS evaluation.
pub(super) struct SplitFormSWE2D<'a, 'c, BC: SWEBoundaryCondition2D> {
    q: &'a SWESolution2D,
    mesh: &'a Mesh2D,
    ops: &'a DGOperators2D,
    geom: &'a GeometricFactors2D,
    config: &'a SWE2DRhsConfig<'c, BC>,
    time: f64,
    /// Add entropy-stable dissipation at faces
    dissipative: bool,
}

impl<'a, 'c, BC: SWEBoundaryCondition2D> SplitFormSWE2D<'a, 'c, BC> {
    /// Returns `None` for the standard (collocated) formulation.
    ///
    /// # Panics
    /// If `config.source_terms` also discretizes the bed slope (e.g. contains
    /// `BathymetrySource2D`): the split form already includes it.
    pub(super) fn new(
        q: &'a SWESolution2D,
        mesh: &'a Mesh2D,
        ops: &'a DGOperators2D,
        geom: &'a GeometricFactors2D,
        config: &'a SWE2DRhsConfig<'c, BC>,
        time: f64,
    ) -> Option<Self> {
        let dissipative = match config.formulation {
            SWEFormulation2D::Standard => return None,
            SWEFormulation2D::EntropyConservative => false,
            SWEFormulation2D::EntropyStable => true,
        };
        assert!(
            !config
                .source_terms
                .is_some_and(|s| s.includes_bathymetry_slope()),
            "SWEFormulation2D::{:?} includes the bed slope −gh∇B in the DG operator; \
             remove BathymetrySource2D from source_terms",
            config.formulation
        );

        Some(Self {
            q,
            mesh,
            ops,
            geom,
            config,
            time,
            dissipative,
        })
    }

    #[inline]
    fn bed(&self, k: ElementIndex, node: usize) -> f64 {
        self.config.bathymetry.map_or(0.0, |b| b.get(k, node))
    }

    /// Volume, bed-slope and surface terms of element `k`, written to `out`
    /// (`[h, hu, hv]`, `n_nodes` values each). Source terms are not included.
    pub(super) fn element_rhs(
        &self,
        k: ElementIndex,
        ws: &mut SplitFormWorkspace,
        out: [&mut [f64]; 3],
    ) {
        let ops = self.ops;
        let n1 = ops.n_1d;
        let n_face_nodes = ops.n_face_nodes;
        let ki = k.as_usize();
        let g = self.config.equation.g;
        let h_min = self.config.equation.h_min.meters();

        for i in 0..ops.n_nodes {
            ws.nodes[i] = SWENodeState2D::new(&self.q.get_state(k, i), self.bed(k, i), h_min);
            ws.rhs[i] = SWEState2D::zero();
        }

        // 1. Flux-differencing volume term and collocated bed slope, line by line.
        //    Node ordering is i = j·n1 + a (r fastest).
        let dir_r = (self.geom.rx[ki], self.geom.ry[ki]);
        let dir_s = (self.geom.sx[ki], self.geom.sy[ki]);
        for line in 0..n1 {
            self.line_volume(ws, |a| line * n1 + a, dir_r, g);
            self.line_volume(ws, |j| j * n1 + line, dir_s, g);
        }

        // 2. Surface terms: J⁻¹ LIFT sJ (F(q⁻)·n − F*). For GLL collocation LIFT
        //    only couples a face node to itself.
        let j_inv = self.geom.det_j_inv[ki];
        for face in 0..4 {
            let normal = self.geom.normals[ki][face];
            let scale = j_inv * self.geom.surface_j[ki][face];
            let neighbor = self.mesh.neighbor(k, face);

            for (fi, &node) in ops.face_nodes[face].iter().enumerate() {
                let q_int = ws.nodes[node];
                let q_ext = match neighbor {
                    Some(nb) => {
                        let nb_k = ElementIndex::new(nb.element);
                        let nb_node = ops.face_nodes[nb.face][n_face_nodes - 1 - fi];
                        SWENodeState2D::new(
                            &self.q.get_state(nb_k, nb_node),
                            self.bed(nb_k, nb_node),
                            h_min,
                        )
                    }
                    // Ghost state with mirrored bathymetry: no bed step at boundaries
                    None => SWENodeState2D::new(
                        &self.ghost_state(k, face, node, normal),
                        q_int.b,
                        h_min,
                    ),
                };

                let mut f_star = wintermeyer_flux_2d(&q_int, &q_ext, normal, g)
                    + wintermeyer_bed_interface_term_2d(q_int.h, q_int.b, q_ext.b, normal, g);
                if self.dissipative {
                    f_star = f_star + entropy_stable_dissipation_2d(&q_int, &q_ext, normal, g);
                }
                let flux_diff = wintermeyer_flux_2d(&q_int, &q_int, normal, g) - f_star;

                ws.rhs[node] = ws.rhs[node] + (scale * ops.lift[face][(node, fi)]) * flux_diff;
            }
        }

        let [out_h, out_hu, out_hv] = out;
        for (i, r) in ws.rhs.iter().enumerate() {
            out_h[i] = r.h;
            out_hu[i] = r.hu;
            out_hv[i] = r.hv;
        }
    }

    /// Accumulate `−Σ_c 2 D_ac F#(q_a, q_c)·dir − g h_a (Σ_c D_ac B_c) dir` along
    /// one line of nodes `idx(0..n1)`.
    ///
    /// `F#` is symmetric, so each pair is evaluated once and applied to both ends.
    #[inline]
    fn line_volume(
        &self,
        ws: &mut SplitFormWorkspace,
        idx: impl Fn(usize) -> usize,
        dir: (f64, f64),
        g: f64,
    ) {
        let n1 = self.ops.n_1d;
        let d1 = &self.ops.dr_1d_row_major;
        for a in 0..n1 {
            let ia = idx(a);
            let q_a = ws.nodes[ia];

            // Diagonal pair F#(q_a, q_a) = F(q_a); D_aa ≠ 0 only at line ends
            let mut acc = (2.0 * d1[a * n1 + a]) * wintermeyer_flux_2d(&q_a, &q_a, dir, g);
            for c in (a + 1)..n1 {
                let ic = idx(c);
                let f = wintermeyer_flux_2d(&q_a, &ws.nodes[ic], dir, g);
                acc = acc + (2.0 * d1[a * n1 + c]) * f;
                ws.rhs[ic] = ws.rhs[ic] - (2.0 * d1[c * n1 + a]) * f;
            }

            if self.config.bathymetry.is_some() {
                let db: f64 = (0..n1).map(|c| d1[a * n1 + c] * ws.nodes[idx(c)].b).sum();
                let force = g * q_a.h * db;
                acc = acc + SWEState2D::new(0.0, force * dir.0, force * dir.1);
            }

            ws.rhs[ia] = ws.rhs[ia] - acc;
        }
    }

    /// Boundary ghost state from the configured boundary condition.
    fn ghost_state(
        &self,
        k: ElementIndex,
        face: usize,
        node: usize,
        normal: (f64, f64),
    ) -> SWEState2D {
        super::swe_2d::boundary_ghost_state(
            self.q,
            self.mesh,
            self.ops,
            self.config,
            self.time,
            k,
            face,
            node,
            normal,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::boundary::Reflective2D;
    use crate::equations::ShallowWater2D;
    use crate::flux::{SWENodeState2D, swe_entropy_variables_2d};
    use crate::mesh::{Bathymetry2D, Mesh2D};
    use crate::operators::{DGOperators2D, GeometricFactors2D};
    use crate::solver::rhs::swe_2d::{SWE2DRhsConfig, SWEFormulation2D, compute_rhs_swe_2d};
    use crate::solver::{SWESolution2D, SWEState2D};
    use crate::source::{BathymetrySource2D, CombinedSource2D, CoriolisSource2D};
    use crate::types::ElementIndex;

    const G: f64 = 9.81;
    const L: f64 = 20_000.0;
    const SPLIT_FORMS: [SWEFormulation2D; 2] = [
        SWEFormulation2D::EntropyConservative,
        SWEFormulation2D::EntropyStable,
    ];

    #[derive(Clone, Copy, Debug)]
    enum Bed {
        /// Smooth, nodal, degree ≥ p in every element; continuous across faces
        Smooth,
        /// Cell averages: constant per element, jumps at every face
        CellAverage,
        /// Smooth plus per-node noise: rough inside elements, jumps at faces
        Rough,
    }
    const BEDS: [Bed; 3] = [Bed::Smooth, Bed::CellAverage, Bed::Rough];

    fn smooth_bed(x: f64, y: f64) -> f64 {
        let tau = 2.0 * std::f64::consts::PI / L;
        -200.0 + 150.0 * (tau * x).sin() * (tau * y).cos()
    }

    fn setup(
        order: usize,
        periodic: bool,
        bed: Bed,
    ) -> (Mesh2D, DGOperators2D, GeometricFactors2D, Bathymetry2D) {
        let mesh = if periodic {
            Mesh2D::uniform_periodic(0.0, L, 0.0, L, 12, 12)
        } else {
            Mesh2D::uniform_rectangle(0.0, L, 0.0, 0.5 * L, 8, 5)
        };
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh);
        let mut bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, smooth_bed);
        match bed {
            Bed::Smooth => {}
            Bed::CellAverage => bathymetry.to_cell_average(),
            Bed::Rough => {
                let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
                for b in bathymetry.data.iter_mut() {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    *b += 30.0 * ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
                }
                bathymetry.compute_gradients(&ops, &geom);
            }
        }
        (mesh, ops, geom, bathymetry)
    }

    /// η = 0.3 m over `bathymetry`, velocity `speed`·(cos·cos, ½ sin·cos) —
    /// correlated with the bed slope so that no symmetry hides errors.
    fn state(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        bathymetry: &Bathymetry2D,
        speed: f64,
    ) -> SWESolution2D {
        let tau = 2.0 * std::f64::consts::PI / L;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let h = 0.3 - bathymetry.get(k, i);
                let u = speed * (tau * x).cos() * (tau * y).cos();
                let v = 0.5 * speed * (tau * x).sin() * (tau * y).cos();
                q.set_state(k, i, SWEState2D::from_primitives(h, u, v));
            }
        }
        q
    }

    /// Σ_k J_k Σ_i ω_i f(k, i)
    fn integrate(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        f: impl Fn(ElementIndex, usize) -> f64,
    ) -> f64 {
        let mut total = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            let j = geom.det_j[k.as_usize()];
            for (i, &w) in ops.weights.iter().enumerate() {
                total += w * j * f(k, i);
            }
        }
        total
    }

    /// Entropy rate dE/dt = ∫ w·dq/dt and the scale ∫ |w·dq/dt|.
    fn entropy_rate(
        q: &SWESolution2D,
        rhs: &SWESolution2D,
        bathymetry: &Bathymetry2D,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
    ) -> (f64, f64) {
        let w_dot_rhs = |k: ElementIndex, i: usize| {
            let node = SWENodeState2D::new(&q.get_state(k, i), bathymetry.get(k, i), 1e-6);
            let w = swe_entropy_variables_2d(&node, G);
            let r = rhs.get_state(k, i);
            w.h * r.h + w.hu * r.hu + w.hv * r.hv
        };
        (
            integrate(mesh, ops, geom, w_dot_rhs),
            integrate(mesh, ops, geom, |k, i| w_dot_rhs(k, i).abs()),
        )
    }

    fn max_momentum_rate(rhs: &SWESolution2D) -> f64 {
        rhs.data[1]
            .iter()
            .chain(rhs.data[2].iter())
            .fold(0.0_f64, |m, x| m.max(x.abs()))
    }

    #[test]
    fn test_lake_at_rest_any_nodal_bathymetry() {
        // REVIEW.md §1.1: the collocated scheme balances only deg B ≤ p/2. The split
        // form must balance smooth high-degree, cell-averaged and rough nodal beds,
        // with periodic and reflective-wall boundaries.
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        for order in 1..=4 {
            for periodic in [true, false] {
                for bed in BEDS {
                    let (mesh, ops, geom, bathymetry) = setup(order, periodic, bed);
                    let q = state(&mesh, &ops, &bathymetry, 0.0);
                    for formulation in SPLIT_FORMS {
                        let config = SWE2DRhsConfig::new(&equation, &bc)
                            .with_coriolis(false)
                            .with_formulation(formulation)
                            .with_bathymetry(&bathymetry);
                        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);

                        let max_h = rhs.data[0].iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                        let max_mom = max_momentum_rate(&rhs);
                        // O(1e-12) m/s of spurious acceleration in ~300 m of water
                        assert!(
                            max_h < 1e-12 && max_mom < 1e-9,
                            "p={order}, periodic={periodic}, {bed:?}, {formulation:?}: \
                             max |dh/dt| = {max_h:.3e}, max |d(hu)/dt| = {max_mom:.3e}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_standard_formulation_not_balanced_for_high_degree_bathymetry() {
        // Negative control for the test above: the collocated scheme with
        // BathymetrySource2D is far from balanced on the same smooth bed.
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let bathy_source = BathymetrySource2D::new(G);
        let (mesh, ops, geom, bathymetry) = setup(2, true, Bed::Smooth);
        let q = state(&mesh, &ops, &bathymetry, 0.0);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(true);
        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
        let max_mom = max_momentum_rate(&rhs);
        assert!(
            max_mom > 1e-4,
            "expected a lake-at-rest residual, got {max_mom:.3e}"
        );
    }

    #[test]
    fn test_mass_conservation() {
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        for order in 1..=3 {
            for bed in BEDS {
                let (mesh, ops, geom, bathymetry) = setup(order, true, bed);
                let q = state(&mesh, &ops, &bathymetry, 1.0);
                let mass = integrate(&mesh, &ops, &geom, |k, i| q.get_var(k, i, 0));
                for formulation in SPLIT_FORMS {
                    let config = SWE2DRhsConfig::new(&equation, &bc)
                        .with_coriolis(false)
                        .with_formulation(formulation)
                        .with_bathymetry(&bathymetry);
                    let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
                    let mass_rate = integrate(&mesh, &ops, &geom, |k, i| rhs.get_var(k, i, 0));
                    assert!(
                        mass_rate.abs() / mass < 1e-14,
                        "p={order}, {bed:?}, {formulation:?}: d(mass)/dt / mass = {:.3e}",
                        mass_rate / mass
                    );
                }
            }
        }
    }

    #[test]
    fn test_entropy_conservation_and_stability() {
        // EntropyConservative: dE/dt = 0 to round-off (periodic or walls).
        // EntropyStable: dE/dt < 0. The dissipation acts on jumps of the entropy
        // variables (η, u, v), not of h, so the free surface is given element-wise
        // offsets to make the states jump across every face.
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        for order in 1..=3 {
            for periodic in [true, false] {
                for bed in BEDS {
                    let (mesh, ops, geom, bathymetry) = setup(order, periodic, bed);
                    let mut q = state(&mesh, &ops, &bathymetry, 1.0);
                    for k in ElementIndex::iter(mesh.n_elements) {
                        let offset = 0.05 * (k.as_usize() % 3) as f64;
                        for i in 0..ops.n_nodes {
                            let s = q.get_state(k, i);
                            q.set_state(k, i, SWEState2D::new(s.h + offset, s.hu, s.hv));
                        }
                    }

                    let config = SWE2DRhsConfig::new(&equation, &bc)
                        .with_coriolis(false)
                        .with_formulation(SWEFormulation2D::EntropyConservative)
                        .with_bathymetry(&bathymetry);
                    let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
                    let (rate, scale) = entropy_rate(&q, &rhs, &bathymetry, &mesh, &ops, &geom);
                    assert!(
                        rate.abs() < 1e-12 * scale,
                        "EC p={order}, periodic={periodic}, {bed:?}: \
                         dE/dt = {rate:.3e} (scale {scale:.3e})"
                    );

                    let config = config.with_formulation(SWEFormulation2D::EntropyStable);
                    let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
                    let (rate, scale) = entropy_rate(&q, &rhs, &bathymetry, &mesh, &ops, &geom);
                    assert!(
                        rate < -1e-10 * scale,
                        "ES p={order}, periodic={periodic}, {bed:?}: \
                         dE/dt = {rate:.3e} (scale {scale:.3e})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_uniform_flow_flat_bottom_is_steady() {
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 4, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh);
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::from_primitives(2.0, 0.5, -0.2));
            }
        }
        for formulation in SPLIT_FORMS {
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_coriolis(false)
                .with_formulation(formulation);
            let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
            assert!(
                rhs.max_abs() < 1e-12,
                "{formulation:?}: {:.3e}",
                rhs.max_abs()
            );
        }
    }

    #[test]
    #[should_panic(expected = "remove BathymetrySource2D")]
    fn test_rejects_double_counted_bed_slope() {
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let (mesh, ops, geom, bathymetry) = setup(1, true, Bed::Smooth);
        let q = state(&mesh, &ops, &bathymetry, 0.0);
        let bathy_source = BathymetrySource2D::new(G);
        let coriolis = CoriolisSource2D::f_plane(1.2e-4);
        let combined = CombinedSource2D::new(vec![&coriolis, &bathy_source]);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_formulation(SWEFormulation2D::EntropyStable)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&combined);
        compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
    }

    #[test]
    #[cfg(all(feature = "parallel", feature = "simd"))]
    fn test_parallel_matches_serial() {
        use crate::solver::rhs::swe_2d::compute_rhs_swe_2d_parallel;

        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new();
        let coriolis = CoriolisSource2D::f_plane(1.2e-4);
        for periodic in [true, false] {
            let (mesh, ops, geom, bathymetry) = setup(3, periodic, Bed::Rough);
            let q = state(&mesh, &ops, &bathymetry, 1.0);
            for formulation in SPLIT_FORMS {
                let config = SWE2DRhsConfig::new(&equation, &bc)
                    .with_coriolis(false)
                    .with_formulation(formulation)
                    .with_bathymetry(&bathymetry)
                    .with_source_terms(&coriolis);
                let serial = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
                let parallel = compute_rhs_swe_2d_parallel(&q, &mesh, &ops, &geom, &config, 0.0);
                // Same kernel, same summation order: bitwise identical
                assert_eq!(
                    serial.data, parallel.data,
                    "periodic={periodic}, {formulation:?}"
                );
            }
        }
    }
}
