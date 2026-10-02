//! Potential energy of a 3D state, and its reference potential energy: the
//! measure of diapycnal mixing.
//!
//! # Formulation
//!
//! The potential energy of a hydrostatic state is
//!
//! ```text
//! PE = g ∫ ρ z dV,
//! ```
//!
//! and its *reference* (background) potential energy RPE is the potential
//! energy of the same water rearranged adiabatically into its state of least
//! potential energy: sorted by density, densest at the bed, in level layers
//! that fill the basin (Winters et al. 1995). Advection only rearranges water,
//! so it leaves RPE unchanged; only diapycnal mixing raises it. With no
//! explicit diffusivity of the tracers, any growth of RPE is the advection
//! scheme's spurious mixing, which is how Ilıcak et al. (2012) compare ocean
//! models. The available potential energy `APE = PE − RPE ≥ 0` is what the
//! flow can draw on.
//!
//! # Discretisation
//!
//! Each node `i` of element `k` is a column of area `a = w_i J_i` (its GLL
//! quadrature weight, so `Σ a` is the quadrature of the area) over the bed
//! `B`, holding `n_levels` layers of thickness `D Δσ_l` (`D = η − B`) and
//! density `ρ_l`. The layers are the parcels:
//!
//! - `PE = g Σ ρ_l a D Δσ_l z̄_l`, with `z̄_l` the layer's mid-height, exact
//!   for a density constant within each layer;
//! - RPE sorts the parcels by density and stacks them from the lowest bed up
//!   in the basin made of the same columns, whose area at height `z` is the
//!   sum of the columns with `B < z`. Each parcel's `∫ z dV` is integrated
//!   exactly over the piecewise-constant area.
//!
//! Then `RPE ≤ PE` up to the free surface's own potential energy (the actual
//! columns are filled to `η`, the reference basin to a level surface of the
//! same volume), and for a horizontally uniform, stably stratified state over
//! a flat bed `RPE = PE` to round-off.
//!
//! The density is `state.rho` as it stands: refresh it (`update_density`)
//! before computing. With a pressure-dependent equation of state, it should
//! be the potential density.
//!
//! Both energies are of the density anomaly `ρ − ρ_r` for a reference
//! density `ρ_r` (the model's ρ₀, say). That shifts RPE by `g ρ_r ∫ z dV` of
//! the filled basin, a constant while the volume is conserved, and PE by
//! the same over the actual columns (which differs by the free surface's
//! potential energy): changes of RPE do not depend on `ρ_r`. With `ρ_r` near
//! the water's density they stay far above the round-off of `g ρ z V`, which
//! for a 20 m basin is ≈ 1e-10 of the change of a step of vertical diffusion.
//!
//! # References
//!
//! - Winters, Lombard, Riley & D'Asaro (1995), "Available potential energy
//!   and mixing in density-stratified fluids", J. Fluid Mech. 289.
//! - Ilıcak, Adcroft, Griffies & Hallberg (2012), "Spurious dianeutral mixing
//!   and the role of momentum closure", Ocean Modelling 45–46.

use crate::mesh::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::state::Solution3D;
use crate::vertical::SigmaGrid;

/// The potential energies of a 3D state (J; of the density anomaly
/// `ρ − ρ_r`, see the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PotentialEnergy {
    /// `g ∫ (ρ − ρ_r) z dV` of the state.
    pub total: f64,
    /// The potential energy of the same water sorted by density into level
    /// layers (the background or reference potential energy).
    pub reference: f64,
}

impl PotentialEnergy {
    /// The available potential energy `total − reference`.
    pub fn available(&self) -> f64 {
        self.total - self.reference
    }
}

/// Computes [`PotentialEnergy`] of 3D states on one mesh and bed.
///
/// Holds the basin's columns sorted by bed height, and the parcel buffer, so
/// that repeated calls (a time series) do not allocate.
#[derive(Clone, Debug)]
pub struct PotentialEnergy3D {
    /// `(B, a)` of every node, in the state's node order.
    nodes: Vec<(f64, f64)>,
    /// The same, by `B` ascending.
    columns: Vec<(f64, f64)>,
    /// `(ρ − ρ_r, V)` of every wet layer.
    parcels: Vec<(f64, f64)>,
    g: f64,
    rho_ref: f64,
}

impl PotentialEnergy3D {
    /// The basin of `bathymetry` (bed elevation `B`, negative under water) on
    /// the mesh of `ops`, `geom`, with gravity `g`, for energies of the
    /// density anomaly from `rho_ref` (see the module docs).
    pub fn new(
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        bathymetry: &Bathymetry2D,
        g: f64,
        rho_ref: f64,
    ) -> Self {
        let nn = ops.n_nodes;
        let nodes: Vec<(f64, f64)> = (0..bathymetry.n_elements * nn)
            .map(|idx| {
                let area = ops.weights[idx % nn] * geom.jacobian(idx / nn, idx % nn);
                (bathymetry.data[idx], area)
            })
            .collect();
        let mut columns = nodes.clone();
        columns.sort_unstable_by(|p, q| p.0.total_cmp(&q.0));
        Self {
            nodes,
            columns,
            parcels: Vec::new(),
            g,
            rho_ref,
        }
    }

    /// The potential energies of `state` on the levels of `sigma`, from its
    /// `rho` and `eta` (columns with `η ≤ B` hold no water).
    pub fn compute(&mut self, state: &Solution3D, sigma: &SigmaGrid) -> PotentialEnergy {
        let (nl, sigma_w, d_sigma) = (state.n_levels, sigma.sigma_w(), sigma.d_sigma());
        assert_eq!(
            self.nodes.len(),
            state.n_elements * state.n_nodes,
            "the state is not on this basin's mesh"
        );

        // The parcels, and the potential energy where they are
        self.parcels.clear();
        let mut total = 0.0;
        for (idx, (&(bed, area), &eta)) in self.nodes.iter().zip(&state.eta.data).enumerate() {
            let depth = eta - bed;
            if depth <= 0.0 {
                continue;
            }
            let rho = &state.rho[idx * nl..(idx + 1) * nl];
            for l in 0..nl {
                let (anomaly, volume) = (rho[l] - self.rho_ref, area * depth * d_sigma[l]);
                let z = eta + depth * 0.5 * (sigma_w[l] + sigma_w[l + 1]);
                total += anomaly * volume * z;
                self.parcels.push((anomaly, volume));
            }
        }

        // Densest first, stacked from the lowest bed up
        self.parcels.sort_unstable_by(|p, q| q.0.total_cmp(&p.0));
        let mut reference = 0.0;
        let mut z = self.columns.first().map_or(0.0, |c| c.0);
        let (mut area, mut next) = (0.0, 0);
        for &(rho, volume) in &self.parcels {
            // ∫ z dV over the parcel's slab
            let (mut left, mut moment) = (volume, 0.0);
            loop {
                while next < self.columns.len() && self.columns[next].0 <= z {
                    area += self.columns[next].1;
                    next += 1;
                }
                let top = self.columns.get(next).map_or(f64::INFINITY, |c| c.0);
                let room = area * (top - z);
                if room >= left {
                    let dz = left / area;
                    moment += left * (z + 0.5 * dz);
                    z += dz;
                    break;
                }
                moment += room * 0.5 * (z + top);
                left -= room;
                z = top;
            }
            reference += rho * moment;
        }

        PotentialEnergy {
            total: self.g * total,
            reference: self.g * reference,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh2D;
    use crate::types::ElementIndex;

    const G: f64 = 9.81;

    /// A `length × width` channel of `n_x` elements of order `order`, flat at
    /// `−depth`, on `levels` uniform levels: its operators, geometry, bed and
    /// a state at rest.
    fn channel(
        order: usize,
        n_x: usize,
        levels: usize,
        [length, width, depth]: [f64; 3],
    ) -> (
        DGOperators2D,
        GeometricFactors2D,
        Bathymetry2D,
        SigmaGrid,
        Solution3D,
    ) {
        let mesh = Mesh2D::uniform_rectangle(0.0, length, 0.0, width, n_x, 1);
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let bed = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -depth);
        let state = Solution3D::new(mesh.n_elements, ops.n_nodes, levels);
        (ops, geom, bed, SigmaGrid::uniform(levels), state)
    }

    /// A stable stratification that is the same in every column is its own
    /// sorted state: RPE = PE.
    #[test]
    fn a_level_stable_stratification_is_its_own_reference() {
        let (ops, geom, bed, sigma, mut state) = channel(2, 3, 8, [3e3, 500.0, 20.0]);
        let nl = sigma.n_levels();
        for column in state.rho.chunks_mut(nl) {
            for (l, rho) in column.iter_mut().enumerate() {
                *rho = 1027.0 - 0.3 * l as f64;
            }
        }
        let energy = PotentialEnergy3D::new(&ops, &geom, &bed, G, 0.0).compute(&state, &sigma);
        let error = energy.available().abs() / energy.total.abs();
        assert!(error < 1e-14, "APE / PE = {error:.3e}");
    }

    /// The lock exchange's initial state, dense water (ρ₁) left of the
    /// middle and light (ρ₂) right, over the full depth `H`: its sorted state
    /// is two layers of `H/2`, and `APE = g A H² (ρ₁ − ρ₂)/8`.
    #[test]
    fn a_lock_has_the_available_energy_of_its_two_layers() {
        let [length, width, depth] = [8e3, 500.0, 20.0];
        let (ops, geom, bed, sigma, mut state) = channel(2, 4, 5, [length, width, depth]);
        let (dense, light) = (1026.0, 1025.0);
        let column = ops.n_nodes * sigma.n_levels();
        for (k, rho) in state.rho.chunks_mut(column).enumerate() {
            rho.fill(if k < 2 { dense } else { light });
        }
        let energy = PotentialEnergy3D::new(&ops, &geom, &bed, G, 0.0).compute(&state, &sigma);
        let expected = G * length * width * depth * depth * (dense - light) / 8.0;
        let error = (energy.available() - expected).abs() / expected;
        assert!(
            error < 1e-10,
            "APE {:.6e} J against {expected:.6e} J",
            energy.available()
        );
        // The sorted layers' potential energy
        let area = length * width;
        let reference =
            G * area * (dense * 0.5 * depth * -0.75 * depth + light * 0.5 * depth * -0.25 * depth);
        let error = (energy.reference - reference).abs() / reference.abs();
        assert!(
            error < 1e-14,
            "RPE {:.6e} J against {reference:.6e} J",
            energy.reference
        );
    }

    /// The reference basin follows the bed. A channel half 20 m, half 10 m
    /// deep on two levels, with dense water (ρ₁) only in the deep half's bed
    /// layer, below 10 m: sorted, it fills the deep half up to 10 m below
    /// the surface, and the light water the whole channel above, which is
    /// where it is. RPE = PE = `g A (ρ₁·½·10·(−15) + ρ₂·10·(−5))`.
    #[test]
    fn the_reference_basin_follows_the_bed() {
        let [length, width] = [2e3, 500.0];
        let (ops, geom, mut bed, sigma, mut state) = channel(3, 2, 2, [length, width, 20.0]);
        for i in 0..ops.n_nodes {
            bed.set(ElementIndex::new(1), i, -10.0);
        }
        let (dense, light) = (1026.0, 1025.0);
        let nl = sigma.n_levels();
        for (idx, column) in state.rho.chunks_mut(nl).enumerate() {
            column[0] = if idx < ops.n_nodes { dense } else { light };
            column[1] = light;
        }
        let energy = PotentialEnergy3D::new(&ops, &geom, &bed, G, 0.0).compute(&state, &sigma);
        let area = length * width;
        let expected = G * area * (dense * 0.5 * 10.0 * -15.0 + light * 10.0 * -5.0);
        for (name, value) in [("PE", energy.total), ("RPE", energy.reference)] {
            let error = (value - expected).abs() / expected.abs();
            assert!(
                error < 1e-14,
                "{name} {value:.6e} J against {expected:.6e} J"
            );
        }
        // Dense water on the shallow half's surface layer (5 m) and light
        // water everywhere else: sorted, the dense water fills the deep half
        // from 20 to 15 m below the surface, the light water the rest
        for (idx, column) in state.rho.chunks_mut(nl).enumerate() {
            column[0] = light;
            column[1] = if idx >= ops.n_nodes { dense } else { light };
        }
        let energy = PotentialEnergy3D::new(&ops, &geom, &bed, G, 0.0).compute(&state, &sigma);
        let expected =
            G * area * (0.5 * (dense * 5.0 * -17.5 + light * 5.0 * -12.5) + light * 10.0 * -5.0);
        let error = (energy.reference - expected).abs() / expected.abs();
        assert!(
            error < 1e-14,
            "RPE {:.6e} J against {expected:.6e} J",
            energy.reference
        );
        assert!(energy.available() > 0.0);
    }

    /// A dry column (`η ≤ B`) holds no water, and the reference basin
    /// spreads what is left over the whole bed.
    #[test]
    fn dry_columns_hold_no_water() {
        let (ops, geom, bed, sigma, mut state) = channel(1, 2, 3, [1e3, 1e3, 10.0]);
        state.rho.fill(1025.0);
        let mut energy = PotentialEnergy3D::new(&ops, &geom, &bed, G, 0.0);
        let wet = energy.compute(&state, &sigma);
        // Dry the second element
        for i in 0..ops.n_nodes {
            state.eta.data[ops.n_nodes + i] = -10.0;
        }
        let half = energy.compute(&state, &sigma);
        // Half the water, in the same layer heights; sorted it spreads over
        // the whole basin, 5 m deep
        let error = (half.total - 0.5 * wet.total).abs() / wet.total.abs();
        assert!(
            error < 1e-14,
            "PE {:.6e} J of {:.6e} J",
            half.total,
            wet.total
        );
        let expected = G * 1025.0 * 1e6 * 5.0 * -7.5;
        let error = (half.reference - expected).abs() / expected.abs();
        assert!(
            error < 1e-14,
            "RPE {:.6e} J against {expected:.6e} J",
            half.reference
        );
    }
}
