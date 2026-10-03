//! The force of the waves on the depth-averaged flow: the divergence of the
//! radiation stress (Longuet-Higgins & Stewart 1964), TODO F.4 coupling.
//!
//! Phase-averaged waves carry momentum; where they shoal, refract or break, the
//! change of their momentum flux, the radiation stress `S`, acts on the mean flow:
//!
//! ```text
//! ∂(hu)/∂t = … − (1/ρ) ∂S_xj/∂x_j,   ∂(hv)/∂t = … − (1/ρ) ∂S_yj/∂x_j
//! ```
//!
//! It sets the water down seaward of the breakers and up in the surf zone, and
//! drives longshore currents under oblique waves. [`WaveForce2D`] takes `S/(ρg)`
//! (m²) per node, as [`WaveModel2D::radiation_stress`] gives it on the same mesh
//! and order, and keeps `−g ∇·(S/ρg)` per node, differentiated within each
//! element by its own derivative matrices. The jumps of `S` between elements are
//! left out (they vanish as the wave field converges), so the force is not
//! exactly in flux form: its total over the domain differs from the boundary
//! integral of `S·n` by those jumps. Dry nodes (`h ≤ h_min`) get no force.
//!
//! The vortex-force form of McWilliams et al. (2004), which separates the
//! conservative part into a Bernoulli head and needs the Stokes drift in the
//! advection, is an alternative for 3D (TODO F.4).
//!
//! [`WaveCurrentFriction2D`] raises the bed friction of the current under
//! waves: the wave boundary layer's turbulence enhances the mean stress,
//! Soulsby's (1995) fit to the combined wave–current models,
//!
//! ```text
//! τ_m = τ_c [1 + 1.2 (τ_w/(τ_c + τ_w))^3.2],
//! ```
//!
//! with `τ_c` the stress of the current alone (any [`BottomFriction2D`] law) and
//! `τ_w` the waves' ([`WaveModel2D::bed_wave_stress`]). Up to 2.2× the current's
//! where the waves dominate; unchanged without them.

use crate::boundary::tidal_ramp;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::SWEState2D;
use crate::source::{BottomFriction2D, ElementSources, SourceContext2D, SourceTerm2D};
use crate::waves::model::nodal_gradient;
use crate::waves::{WaveModel2D, WaveSolution};

/// The radiation-stress force of waves on the 2D flow (see the module docs).
#[derive(Clone, Debug)]
pub struct WaveForce2D {
    n_nodes: usize,
    /// `−g ∇·(S/ρg)` per node (m²/s², force per unit area over ρ)
    force: Vec<[f64; 2]>,
    /// Node positions, for [`SourceTerm2D::evaluate`]
    positions: Vec<[f64; 2]>,
    ramp: Option<f64>,
    h_min: f64,
}

impl WaveForce2D {
    /// The force of the wave state `n` of `model`, on the model's mesh and order.
    pub fn new(model: &WaveModel2D, n: &WaveSolution) -> Self {
        let positions = (0..model.mesh.n_elements)
            .flat_map(|k| {
                let (mesh, ops) = (&model.mesh, &model.ops);
                (0..ops.n_nodes).map(move |i| {
                    mesh.reference_to_physical(
                        crate::types::ElementIndex::new(k),
                        ops.nodes_r[i],
                        ops.nodes_s[i],
                    )
                })
            })
            .collect();
        Self::from_radiation_stress(
            &model.ops,
            &model.geom,
            &model.radiation_stress(n),
            model.g(),
        )
        .with_positions(positions)
    }

    /// The force of the radiation stress `stress` (`[S_xx, S_xy, S_yy]/(ρg)`, m²,
    /// per node, element by element) under gravity `g`.
    pub fn from_radiation_stress(
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        stress: &[[f64; 3]],
        g: f64,
    ) -> Self {
        let np = stress.len();
        assert_eq!(np % ops.n_nodes, 0, "a nodal field element by element");
        let component = |c: usize| -> Vec<[f64; 2]> {
            let field: Vec<f64> = stress.iter().map(|s| s[c]).collect();
            let mut gradient = vec![[0.0; 2]; np];
            nodal_gradient(ops, geom, &field, &mut gradient);
            gradient
        };
        let (sxx, sxy, syy) = (component(0), component(1), component(2));
        let force = (0..np)
            .map(|p| [-g * (sxx[p][0] + sxy[p][1]), -g * (sxy[p][0] + syy[p][1])])
            .collect();
        Self {
            n_nodes: ops.n_nodes,
            force,
            positions: Vec::new(),
            ramp: None,
            h_min: 1e-6,
        }
    }

    fn with_positions(mut self, positions: Vec<[f64; 2]>) -> Self {
        self.positions = positions;
        self
    }

    /// Ramp the force up smoothly from 0 at t = 0 to full at `seconds` (the
    /// tidal ramp's `3τ² − 2τ³`), so that a run from rest does not ring.
    pub fn with_ramp(mut self, seconds: f64) -> Self {
        self.ramp = Some(seconds);
        self
    }

    /// No force where the water is shallower than `h_min` (m; default 1e-6).
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }

    /// `−g ∇·(S/ρg)` per node (m²/s²).
    pub fn force(&self) -> &[[f64; 2]] {
        &self.force
    }
}

impl SourceTerm2D for WaveForce2D {
    /// Per-node evaluation by position, a search over the nodes (slow; the RHS
    /// kernels use [`SourceTerm2D::add_element`]). Zero away from a node, and
    /// everywhere for a force built from a bare stress field (no positions).
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        if ctx.state.h <= self.h_min {
            return SWEState2D::zero();
        }
        let (x, y) = ctx.position;
        let scale = 1e-9 * (1.0 + x.abs() + y.abs());
        let Some(p) = self
            .positions
            .iter()
            .position(|q| (q[0] - x).abs() <= scale && (q[1] - y).abs() <= scale)
        else {
            return SWEState2D::zero();
        };
        let ramp = tidal_ramp(ctx.time, self.ramp);
        SWEState2D::new(0.0, ramp * self.force[p][0], ramp * self.force[p][1])
    }

    fn add_element(
        &self,
        element: &ElementSources<'_>,
        _h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
    ) {
        let ramp = tidal_ramp(element.time, self.ramp);
        let base = element.element.as_usize() * self.n_nodes;
        let depths = element.solution.element_h(element.element);
        for (i, &h) in depths.iter().enumerate() {
            if h > self.h_min {
                let [fx, fy] = self.force[base + i];
                hu[i] += ramp * fx;
                hv[i] += ramp * fy;
            }
        }
    }

    fn name(&self) -> &'static str {
        "wave_force_2d"
    }
}

/// Bottom friction of a current enhanced by waves (see the module docs):
/// `inner`'s damping rate times `1 + 1.2 (τ_w/(τ_c + τ_w))^3.2`.
#[derive(Clone, Debug)]
pub struct WaveCurrentFriction2D<F> {
    inner: F,
    /// The waves' bed stress per ρ (m²/s²) per node
    wave_stress: Vec<f64>,
}

impl<F: BottomFriction2D> WaveCurrentFriction2D<F> {
    /// `inner` under waves whose bed stress per ρ is `wave_stress` (m²/s², per
    /// node, element by element: [`WaveModel2D::bed_wave_stress`]).
    pub fn new(inner: F, wave_stress: Vec<f64>) -> Self {
        assert!(
            wave_stress.iter().all(|&t| t >= 0.0),
            "wave stresses must be non-negative"
        );
        if let Some(n) = inner.n_total_nodes() {
            assert_eq!(n, wave_stress.len(), "one wave stress per node of the law");
        }
        Self { inner, wave_stress }
    }

    /// Replace the waves' stress, as a coupling does every interval.
    pub fn set_wave_stress(&mut self, wave_stress: &[f64]) {
        assert_eq!(wave_stress.len(), self.wave_stress.len());
        self.wave_stress.copy_from_slice(wave_stress);
    }

    /// Soulsby's enhancement `1 + 1.2 (τ_w/(τ_c + τ_w))^3.2` of the current's
    /// stress `tau_c` under the waves' `tau_w` (any common units).
    pub fn enhancement(tau_c: f64, tau_w: f64) -> f64 {
        if tau_w <= 0.0 {
            return 1.0;
        }
        1.0 + 1.2 * (tau_w / (tau_c + tau_w)).powf(3.2)
    }
}

impl<F: BottomFriction2D> BottomFriction2D for WaveCurrentFriction2D<F> {
    /// The inner rate `Λ_c`, times the enhancement with `τ_c/ρ = Λ_c h |u|`.
    fn damping_rate(&self, node: usize, h: f64, speed: f64) -> f64 {
        let rate = self.inner.damping_rate(node, h, speed);
        rate * Self::enhancement(rate * h * speed, self.wave_stress[node])
    }

    fn n_total_nodes(&self) -> Option<usize> {
        Some(self.wave_stress.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh2D;
    use crate::types::ElementIndex;

    /// A radiation stress quadratic in x and y (exact at P2 on any
    /// parallelogram) gives its exact divergence at every node, and the force
    /// acts only on wet nodes, ramped.
    #[test]
    fn the_force_is_the_exact_divergence_of_a_polynomial_stress() {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 300.0, 0.0, 200.0, 3, 2);
        // Shear the mesh: the derivatives go through the metric terms
        for v in mesh.vertices.iter_mut() {
            v[0] += 0.3 * v[1];
        }
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let g = 9.81;
        let stress_at = |x: f64, y: f64| {
            [
                0.2 + 1e-3 * x - 2e-6 * x * x + 1e-6 * x * y,
                0.05 - 3e-4 * y + 1e-6 * x * y,
                0.1 + 2e-4 * x + 4e-6 * y * y,
            ]
        };
        // −g (∂S_xx/∂x + ∂S_xy/∂y), −g (∂S_xy/∂x + ∂S_yy/∂y)
        let exact = |x: f64, y: f64| {
            [
                -g * ((1e-3 - 4e-6 * x + 1e-6 * y) + (-3e-4 + 1e-6 * x)),
                -g * (1e-6 * y + 8e-6 * y),
            ]
        };
        let xy: Vec<[f64; 2]> = ElementIndex::iter(mesh.n_elements)
            .flat_map(|k| {
                let (mesh, ops) = (&mesh, &ops);
                (0..ops.n_nodes)
                    .map(move |i| mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]))
            })
            .collect();
        let stress: Vec<[f64; 3]> = xy.iter().map(|p| stress_at(p[0], p[1])).collect();
        let force = WaveForce2D::from_radiation_stress(&ops, &geom, &stress, g).with_ramp(100.0);
        for (p, f) in force.force().iter().enumerate() {
            let e = exact(xy[p][0], xy[p][1]);
            assert!(
                (f[0] - e[0]).abs() < 1e-14 && (f[1] - e[1]).abs() < 1e-14,
                "node {p}: {f:?} against {e:?}"
            );
        }
        // Half way up the ramp, and nothing on a dry node
        let mut q = crate::solver::SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        let k = ElementIndex::new(1);
        for i in 0..ops.n_nodes {
            let h = if i == 0 { 0.0 } else { 5.0 };
            q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
        }
        let element = ElementSources {
            element: k,
            time: 50.0,
            solution: &q,
            mesh: &mesh,
            ops: &ops,
            bathymetry: None,
            g,
            h_min: 1e-6,
        };
        let n = ops.n_nodes;
        let (mut h, mut hu, mut hv) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        force.add_element(&element, &mut h, &mut hu, &mut hv);
        assert_eq!((hu[0], hv[0]), (0.0, 0.0));
        for i in 1..n {
            let f = force.force()[n + i];
            assert!((hu[i] - 0.5 * f[0]).abs() < 1e-15 && (hv[i] - 0.5 * f[1]).abs() < 1e-15);
        }
        assert!(h.iter().all(|&x| x == 0.0));
    }

    /// Soulsby's enhancement: none without waves, 1 + 1.2·2^(−3.2) when the
    /// stresses are equal, 2.2 where the waves dominate; the friction law
    /// scales its inner rate by it, node by node.
    #[test]
    fn waves_enhance_the_bed_friction_as_soulsby_fits() {
        use crate::source::ChezyFriction2D;
        type Law = WaveCurrentFriction2D<ChezyFriction2D>;
        assert_eq!(Law::enhancement(0.3, 0.0), 1.0);
        assert!((Law::enhancement(0.3, 0.3) - (1.0 + 1.2 * 0.5f64.powf(3.2))).abs() < 1e-15);
        assert!((Law::enhancement(1e-9, 1.0) - 2.2).abs() < 1e-8);
        let (cd, h, speed) = (2.5e-3, 10.0, 0.4);
        let law = Law::new(ChezyFriction2D::new(cd), vec![0.0, 4e-4, 10.0]);
        let plain = cd * speed / h;
        let tau_c = cd * speed * speed;
        assert_eq!(law.damping_rate(0, h, speed), plain);
        let expected = plain * Law::enhancement(tau_c, 4e-4);
        assert!((law.damping_rate(1, h, speed) - expected).abs() < 1e-18);
        assert!((law.damping_rate(2, h, speed) / plain - 2.2).abs() < 1e-3);
        assert_eq!(law.n_total_nodes(), Some(3));
    }
}
