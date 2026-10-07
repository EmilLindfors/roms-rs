//! The waves and the circulation on meshes of their own (TODO F.4 cost).
//!
//! The spectral waves are smooth on the scale of a kilometre and costly per
//! node (hundreds of components), while the circulation needs the coastline
//! at tens of metres. At Frøya a 1 km P1 wave grid costs ≈ 3× the tidal
//! circulation per model hour, the coastline mesh at P2 ≈ 1000×. So the waves
//! run on a coarse mesh of their own and [`WaveCoupling2D`] carries the
//! exchanges across, by interpolation of the nodal fields
//! ([`MeshTransfer2D`]: each target node evaluates the source element's
//! polynomial there):
//!
//! - circulation → waves ([`WaveCoupling2D::update_waves`]): the surface
//!   elevation η and the depth-averaged current. The current is the ratio of
//!   the interpolated transport and depth, `(hu, hv)/h`, as a station samples
//!   it ([`crate::solver::Probe2D`]): a nearly dry node's `hu/h` does not
//!   count as much as a deep one's. On dry nodes η is the bed, which would
//!   pull a wave node at the shore up to the land; there η is the mean of
//!   the element's wet nodes, or 0 (still water) in a dry element. Wave
//!   nodes outside the circulation's mesh (water the circulation does not
//!   model) take the level at the nearest point of it, and no current.
//! - waves → circulation: the radiation stress `S/ρg`, interpolated and then
//!   differentiated on the circulation's mesh ([`WaveCoupling2D::force`]):
//!   the force `−g ∇·(S/ρg)` keeps its flux form there (its total is the
//!   boundary integral of `S` on the circulation's mesh), which a force
//!   interpolated from the coarse mesh would not, and it is exact for the
//!   interpolated stress. The waves' bed stress and surface roughness, and the
//!   Stokes drift ([`StokesDriftField::transferred`]), are interpolated as
//!   they are, the first two kept non-negative.
//!
//! On the same mesh and order every transfer is the identity (to round-off),
//! so the coupling reproduces the same-mesh one.

use std::sync::Arc;

use crate::mesh::{Bathymetry2D, Mesh2D};
use crate::operators::{DGOperators2D, GeometricFactors2D, MeshTransfer2D};
use crate::solver::SWESolution2D;
use crate::source::WaveForce2D;

use super::model::WaveModel2D;
use super::state::WaveSolution;
use super::stokes::StokesDriftField;

/// Default depth (m) below which a circulation node is dry for the coupling.
pub const DEFAULT_COUPLING_H_DRY: f64 = 1e-3;

/// The exchanges between a wave model and a circulation on different meshes
/// (see the module docs).
#[derive(Clone)]
pub struct WaveCoupling2D {
    /// Circulation nodes → wave nodes
    to_waves: MeshTransfer2D,
    /// Wave nodes → circulation nodes
    to_circulation: MeshTransfer2D,
    mesh: Arc<Mesh2D>,
    ops: Arc<DGOperators2D>,
    geom: Arc<GeometricFactors2D>,
    h_dry: f64,
}

impl WaveCoupling2D {
    /// The coupling of `waves` to a circulation on `mesh` at the order of
    /// `ops`.
    pub fn new(
        waves: &WaveModel2D,
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
    ) -> Self {
        Self {
            to_waves: MeshTransfer2D::new(&mesh, &ops, &waves.mesh, &waves.ops),
            to_circulation: MeshTransfer2D::new(&waves.mesh, &waves.ops, &mesh, &ops),
            mesh,
            ops,
            geom,
            h_dry: DEFAULT_COUPLING_H_DRY,
        }
    }

    /// Circulation nodes shallower than `h_dry` (m) are dry: they pass no
    /// level and no current to the waves.
    pub fn with_h_dry(mut self, h_dry: f64) -> Self {
        assert!(h_dry > 0.0);
        self.h_dry = h_dry;
        self
    }

    /// The transfer from the circulation's nodes onto the waves'.
    pub fn to_waves(&self) -> &MeshTransfer2D {
        &self.to_waves
    }

    /// The transfer from the waves' nodes onto the circulation's.
    pub fn to_circulation(&self) -> &MeshTransfer2D {
        &self.to_circulation
    }

    /// The surface elevation η (m) of the circulation state `q` over its bed
    /// `bathymetry` at the waves' nodes; dry nodes take their element's mean
    /// wet level, or 0 in a dry element.
    pub fn water_level(&self, q: &SWESolution2D, bathymetry: &Bathymetry2D) -> Vec<f64> {
        let nn = self.ops.n_nodes;
        assert_eq!(
            q.n_nodes, nn,
            "the state on the coupling's circulation order"
        );
        let mut eta = vec![0.0; q.h_data().len()];
        for ((eta, h), bed) in eta
            .chunks_exact_mut(nn)
            .zip(q.h_data().chunks_exact(nn))
            .zip(bathymetry.data.chunks_exact(nn))
        {
            let (mut wet, mut sum) = (0usize, 0.0);
            for (e, (&h, &b)) in eta.iter_mut().zip(h.iter().zip(bed)) {
                *e = h + b;
                if h > self.h_dry {
                    wet += 1;
                    sum += *e;
                }
            }
            let fill = if wet > 0 { sum / wet as f64 } else { 0.0 };
            for (e, &h) in eta.iter_mut().zip(h) {
                if h <= self.h_dry {
                    *e = fill;
                }
            }
        }
        self.to_waves.apply(&eta)
    }

    /// The depth-averaged current `(u, v)` (m/s, mesh axes) of `q` at the
    /// waves' nodes: the interpolated transport over the interpolated depth,
    /// 0 where that depth is below `h_dry` and at wave nodes outside the
    /// circulation's mesh (the circulation knows nothing of the flow there).
    pub fn currents(&self, q: &SWESolution2D) -> (Vec<f64>, Vec<f64>) {
        let t = &self.to_waves;
        let n = t.n_target_points();
        let (mut u, mut v) = (vec![0.0; n], vec![0.0; n]);
        for p in 0..n {
            let h = t.value_at(p, q.h_data());
            if h > self.h_dry && !t.is_outside(p) {
                u[p] = t.value_at(p, q.hu_data()) / h;
                v[p] = t.value_at(p, q.hv_data()) / h;
            }
        }
        (u, v)
    }

    /// Give `waves` the level and the current of the circulation state `q`
    /// over its bed `bathymetry` ([`WaveModel2D::set_water_level`],
    /// [`WaveModel2D::set_currents`]).
    pub fn update_waves(
        &self,
        waves: &mut WaveModel2D,
        q: &SWESolution2D,
        bathymetry: &Bathymetry2D,
    ) {
        waves.set_water_level(&self.water_level(q, bathymetry));
        let (u, v) = self.currents(q);
        waves.set_currents(&u, &v);
    }

    /// The radiation stress per ρg `[S_xx, S_xy, S_yy]` (m²) of the wave state
    /// `n` at the circulation's nodes.
    pub fn radiation_stress(&self, waves: &WaveModel2D, n: &WaveSolution) -> Vec<[f64; 3]> {
        self.to_circulation
            .apply_components(&waves.radiation_stress(n))
    }

    /// The force of the wave state `n` on the circulation: its radiation
    /// stress interpolated onto the circulation's nodes and differentiated
    /// there ([`WaveForce2D::from_radiation_stress`]).
    pub fn force(&self, waves: &WaveModel2D, n: &WaveSolution) -> WaveForce2D {
        WaveForce2D::from_radiation_stress(
            &self.mesh,
            &self.ops,
            &self.geom,
            &self.radiation_stress(waves, n),
            waves.g(),
        )
    }

    /// The waves' bed stress per ρ (m²/s²) on a bed of roughness length `z0`
    /// (m) at the circulation's nodes ([`WaveModel2D::bed_wave_stress`]), for
    /// [`crate::source::WaveCurrentFriction2D`].
    pub fn bed_wave_stress(&self, waves: &WaveModel2D, n: &WaveSolution, z0: f64) -> Vec<f64> {
        self.non_negative(&waves.bed_wave_stress(n, z0))
    }

    /// The surface roughness `alpha · H_s` (m) at the circulation's nodes
    /// ([`WaveModel2D::surface_roughness`]), for
    /// [`crate::physics::Hydrostatic3D::set_surface_roughness`].
    pub fn surface_roughness(&self, waves: &WaveModel2D, n: &WaveSolution, alpha: f64) -> Vec<f64> {
        self.non_negative(&waves.surface_roughness(n, alpha))
    }

    /// The Stokes drift of the wave state `n` as a field on the circulation's
    /// mesh, for the particle trackers.
    pub fn stokes_drift(&self, waves: &WaveModel2D, n: &WaveSolution) -> StokesDriftField {
        StokesDriftField::new(waves, n).transferred(&self.to_circulation, self.ops.n_nodes)
    }

    fn non_negative(&self, field: &[f64]) -> Vec<f64> {
        let mut out = self.to_circulation.apply(field);
        out.iter_mut().for_each(|x| *x = x.max(0.0));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SWEState2D;
    use crate::types::ElementIndex;
    use crate::waves::SpectralGrid;

    const G: f64 = 9.81;

    struct Circulation {
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
    }

    fn circulation(nx: usize, ny: usize, order: usize) -> Circulation {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 600.0, 0.0, 400.0, nx, ny);
        for v in &mut mesh.vertices {
            v[0] += 0.1 * v[1];
        }
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        Circulation {
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
        }
    }

    fn waves_on(c: &Circulation, bed: impl Fn(f64, f64) -> f64) -> WaveModel2D {
        let bathymetry = Bathymetry2D::from_function(&c.mesh, &c.ops, &c.geom, bed);
        WaveModel2D::new(
            c.mesh.clone(),
            c.ops.clone(),
            c.geom.clone(),
            &bathymetry,
            SpectralGrid::new(0.08, 0.3, 8, 12),
            G,
        )
    }

    fn positions(mesh: &Mesh2D, ops: &DGOperators2D) -> Vec<[f64; 2]> {
        ElementIndex::iter(mesh.n_elements)
            .flat_map(|k| {
                (0..ops.n_nodes)
                    .map(move |i| mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]))
            })
            .collect()
    }

    /// A sea that varies over the domain, so every exchange has something
    /// to carry.
    fn varied_sea(waves: &WaveModel2D) -> WaveSolution {
        let mut n = waves.uniform_state(&waves.grid.jonswap(1.2, 6.0, 3.3, 0.4, 6.0));
        let xy = positions(&waves.mesh, &waves.ops);
        for c in 0..waves.grid.n_components() {
            for (x, [px, py]) in n.component_mut(c).iter_mut().zip(&xy) {
                *x *= 1.0 + 0.5 * (px / 300.0).sin() * (py / 250.0).cos();
            }
        }
        n
    }

    /// On the same mesh every exchange is the same-mesh one: the force of
    /// `WaveForce2D::new`, the stresses, the roughness and the Stokes drift of
    /// the wave model itself, to round-off.
    #[test]
    fn on_the_same_mesh_the_coupling_is_the_direct_one() {
        let c = circulation(5, 4, 2);
        let waves = waves_on(&c, |x, _| -(4.0 + 0.02 * x));
        let n = varied_sea(&waves);
        let coupling = WaveCoupling2D::new(&waves, c.mesh.clone(), c.ops.clone(), c.geom.clone());
        let close = |a: f64, b: f64, scale: f64| (a - b).abs() <= 1e-12 * scale;

        let direct = WaveForce2D::new(&waves, &n);
        let scale = direct
            .force()
            .iter()
            .map(|f| f[0].hypot(f[1]))
            .fold(0.0, f64::max);
        assert!(scale > 0.0);
        for (a, b) in coupling
            .force(&waves, &n)
            .force()
            .iter()
            .zip(direct.force())
        {
            assert!(
                close(a[0], b[0], scale) && close(a[1], b[1], scale),
                "{a:?} {b:?}"
            );
        }
        for (a, b) in [
            (
                coupling.bed_wave_stress(&waves, &n, 1e-3),
                waves.bed_wave_stress(&n, 1e-3),
            ),
            (
                coupling.surface_roughness(&waves, &n, 0.6),
                waves.surface_roughness(&n, 0.6),
            ),
        ] {
            let scale = b.iter().cloned().fold(0.0, f64::max);
            assert!(scale > 0.0);
            assert!(a.iter().zip(&b).all(|(a, b)| close(*a, *b, scale)));
        }
        let field = coupling.stokes_drift(&waves, &n);
        let nn = c.ops.n_nodes;
        let mut w = vec![0.0; nn];
        let expected = waves.stokes_drift(&n, -0.5);
        for (p, e) in expected.iter().enumerate() {
            w.fill(0.0);
            w[p % nn] = 1.0;
            let got = field.at(ElementIndex::new(p / nn), &w, 0.5);
            let scale = e[0].hypot(e[1]);
            assert!(close(got[0], e[0], scale) && close(got[1], e[1], scale));
        }
    }

    /// From a P2 circulation onto a coarser P1 wave mesh: a linear level and a
    /// linear transport over a uniform depth arrive exactly; dry nodes pass
    /// their element's wet level, or 0 in a dry element, and no current.
    #[test]
    fn the_waves_get_the_level_and_the_current() {
        let c = circulation(6, 4, 2);
        let fine_bed = |x: f64, _: f64| if x > 450.0 { 2.0 } else { -10.0 };
        let bathymetry = Bathymetry2D::from_function(&c.mesh, &c.ops, &c.geom, fine_bed);
        let coarse = circulation(3, 2, 1);
        let waves = waves_on(&coarse, |_, _| -10.0);
        let coupling = WaveCoupling2D::new(&waves, c.mesh.clone(), c.ops.clone(), c.geom.clone());

        // Wet everywhere: η linear, a uniform depth of 10 m, transport linear
        let level = |x: f64, y: f64| 0.1 + 1e-4 * x - 2e-4 * y;
        let transport = |x: f64, y: f64| [0.5 + 1e-3 * x, -0.2 + 5e-4 * y];
        let flat = Bathymetry2D::from_function(&c.mesh, &c.ops, &c.geom, |_, _| -10.0);
        let mut q = SWESolution2D::new(c.mesh.n_elements, c.ops.n_nodes);
        let xy = positions(&c.mesh, &c.ops);
        for (p, &[x, y]) in xy.iter().enumerate() {
            let [hu, hv] = transport(x, y);
            let k = ElementIndex::new(p / c.ops.n_nodes);
            q.set_state(
                k,
                p % c.ops.n_nodes,
                SWEState2D::new(10.0 + level(x, y), hu, hv),
            );
        }
        // The depth is 10 + η, not uniform: compare the transport and depth
        let eta = coupling.water_level(&q, &flat);
        let (u, v) = coupling.currents(&q);
        for (p, &[x, y]) in positions(&waves.mesh, &waves.ops).iter().enumerate() {
            let h = 10.0 + level(x, y);
            let [hu, hv] = transport(x, y);
            assert!((eta[p] - level(x, y)).abs() < 1e-12);
            assert!((u[p] * h - hu).abs() < 1e-12 && (v[p] * h - hv).abs() < 1e-12);
        }

        // Over the land x > 450 m: still water 0.3 m up, dry above the bed
        for (p, &[x, _]) in xy.iter().enumerate() {
            let k = ElementIndex::new(p / c.ops.n_nodes);
            let state = if fine_bed(x, 0.0) < 0.0 {
                SWEState2D::new(10.3, 1.0, 0.0)
            } else {
                SWEState2D::new(0.0, 0.0, 0.0)
            };
            q.set_state(k, p % c.ops.n_nodes, state);
        }
        let eta = coupling.water_level(&q, &bathymetry);
        let (u, _) = coupling.currents(&q);
        for (p, &[x, y]) in positions(&waves.mesh, &waves.ops).iter().enumerate() {
            // The wave nodes at x = 600 m (sheared) are in fine elements whose
            // nodes are all on land; the rest in wet or partly wet ones
            let expected = if x - 0.1 * y > 550.0 { 0.0 } else { 0.3 };
            assert!((eta[p] - expected).abs() < 1e-12, "x = {x}: {}", eta[p]);
            assert!(u[p].is_finite() && u[p].abs() <= 0.1 + 1e-12);
        }

        // Waves beyond the circulation's mesh (the coarse grid moved 150 m
        // west): the level at its nearest point, no current
        let mut wide = coarse;
        let mut moved = (*wide.mesh).clone();
        moved.vertices.iter_mut().for_each(|v| v[0] -= 150.0);
        wide.geom = Arc::new(GeometricFactors2D::compute(&moved, &wide.ops));
        wide.mesh = Arc::new(moved);
        let waves = waves_on(&wide, |_, _| -10.0);
        let coupling = WaveCoupling2D::new(&waves, c.mesh.clone(), c.ops.clone(), c.geom.clone());
        let (eta, (u, _)) = (coupling.water_level(&q, &bathymetry), coupling.currents(&q));
        let mut outside = 0;
        for (p, &[x, y]) in positions(&waves.mesh, &waves.ops).iter().enumerate() {
            if x - 0.1 * y < 0.0 {
                outside += 1;
                assert!(coupling.to_waves().is_outside(p));
                assert_eq!(u[p], 0.0);
                assert!((eta[p] - 0.3).abs() < 1e-12);
            } else if x - 0.1 * y < 400.0 {
                assert!((u[p] - 1.0 / 10.3).abs() < 1e-12, "x = {x}: {}", u[p]);
            }
        }
        assert!(outside > 0);
    }
}
