//! A reference stratification and the pair density it defines.
//!
//! # Why
//!
//! The σ-pairs pressure gradient and the split-form tracer advection exchange
//! energy exactly (see [`crate::solver::rhs::baroclinic`]) for any symmetric
//! two-point density `ρ*_ij` that both use. With the arithmetic mean
//! `{{ρ}}`, the split form also conserves the tracer variance `Σ H_z ρ²`, and
//! for a stratification linear in `z` that makes the rest state stable:
//! energy plus a multiple of the variance (an energy–Casimir functional) has
//! its minimum there. For a curved profile no such functional exists. Where
//! one σ-level crosses a pycnocline between the nodes of one element (a
//! coastal cliff), a pair's chord `(ρ_j − ρ_i)/(z_j − z_i)` is far steeper
//! than the stratification at either node, and the stratified rest state is
//! unstable: on the Frøya bed round-off grew with an e-folding of 15 min.
//!
//! # The pair density
//!
//! Take `C(ρ)` with `C′(ρ) = −g Z(ρ)`, `Z(ρ)` the height at which the
//! reference profile `ρ_r(z)` has density `ρ` (convex, since `ρ_r` falls
//! with height). The two-point flux that conserves `Σ H_z C(ρ)` (Tadmor's
//! entropy-conservative flux for this entropy) is
//!
//! ```text
//!     ρ*_ij = [[ρC′ − C]] / [[C′]] = 1/(Z_j − Z_i) ∫_{Z_i}^{Z_j} ρ_r(z) dz,
//! ```
//!
//! the reference profile's mean between the two densities' reference
//! heights. Energy plus `Σ H_z C(ρ)` is then conserved by the volume terms,
//! and its first variation vanishes at rest in the reference profile, where
//! its second is the available potential energy `½ g ρ′²/|∂_z ρ_r|`: the
//! energy–Casimir argument of the linear case, for any stable profile. For a
//! linear `ρ_r` it is `{{ρ}}`. It lies between `ρ_i` and `ρ_j`, it is
//! symmetric, and it is a smooth function of the two densities, so the split
//! form keeps its order, conservation and bounds.
//!
//! Upwind face fluxes are entropy stable for this convex `C`: they dissipate
//! it. At rest the σ-pairs force with `ρ*` is the exact difference of the
//! reference's hydrostatic pressures between the two nodes, so it is
//! balanced up to the columns' own vertical interpolation.
//!
//! The tracers are advected, not the density: with a linear equation of
//! state `ρ = a + b T + c S`, `T* = {{T}} + θ_ij[[T]]` and likewise `S*`, with
//! `θ_ij = (ρ*_ij − {{ρ}})/[[ρ]]` (`|θ| ≤ ½`), give `a + b T* + c S* = ρ*`, and a
//! uniform tracer stays uniform.

/// A horizontally uniform reference density `ρ_r(z)`, strictly decreasing
/// with height: tabulated on uniform heights, linear between them and
/// continued linearly with the end slopes.
#[derive(Clone, Debug)]
pub struct ReferenceStratification {
    /// Lowest height of the table (m).
    z0: f64,
    /// Spacing of the table (m).
    dz: f64,
    /// `ρ_r` at the table heights, strictly decreasing (kg/m³).
    rho: Vec<f64>,
    /// `∫_{z0}^{z_k} (ρ_r − offset) dz` at the table heights.
    integral: Vec<f64>,
    /// Density subtracted in `integral` (the table's mean), for round-off.
    offset: f64,
}

/// The reference's quantities of one density, as [`ReferenceStratification::pair_deviation`]
/// takes them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReferencePoint {
    /// The density (kg/m³).
    pub rho: f64,
    /// Its reference height `Z(ρ)` (m).
    pub height: f64,
    /// `∫_{z0}^{Z} (ρ_r − offset) dz`.
    pub integral: f64,
}

impl ReferenceStratification {
    /// Weakest stratification of the table (kg/m⁴): `|∂_z ρ_r|` is raised to
    /// at least this where the profile is homogeneous or unstable (an `N²` of
    /// ≈ 1e-8 s⁻², the deep ocean's ≈ 1e-6 a hundredth of it), so that
    /// `Z(ρ)` is defined everywhere.
    pub const MIN_GRADIENT: f64 = 1e-6;

    /// Tabulate `profile(z)` (kg/m³) at `n ≥ 2` heights over `[z_min, z_max]`,
    /// made strictly stable from the top down: each value at least
    /// [`Self::MIN_GRADIENT`]`·dz` above the one over it.
    ///
    /// # Panics
    /// If `n < 2` or `z_max ≤ z_min`.
    pub fn from_fn(z_min: f64, z_max: f64, n: usize, profile: impl Fn(f64) -> f64) -> Self {
        assert!(
            n >= 2 && z_max > z_min,
            "ReferenceStratification: need n ≥ 2 heights over a positive range"
        );
        let dz = (z_max - z_min) / (n - 1) as f64;
        let mut rho: Vec<f64> = (0..n).map(|k| profile(z_min + k as f64 * dz)).collect();
        for k in (0..n - 1).rev() {
            rho[k] = rho[k].max(rho[k + 1] + Self::MIN_GRADIENT * dz);
        }
        let offset = rho.iter().sum::<f64>() / n as f64;
        let mut integral = vec![0.0; n];
        for k in 1..n {
            integral[k] = integral[k - 1] + 0.5 * dz * (rho[k - 1] + rho[k] - 2.0 * offset);
        }
        Self {
            z0: z_min,
            dz,
            rho,
            integral,
            offset,
        }
    }

    /// Number of table heights.
    fn len(&self) -> usize {
        self.rho.len()
    }

    /// `ρ_r(z)`.
    pub fn density(&self, z: f64) -> f64 {
        let n = self.len();
        let t = (z - self.z0) / self.dz;
        let k = (t.floor().max(0.0) as usize).min(n - 2);
        let f = t - k as f64;
        self.rho[k] + f * (self.rho[k + 1] - self.rho[k])
    }

    /// The reference height and integral of density `rho`.
    pub fn point(&self, rho: f64) -> ReferencePoint {
        let n = self.len();
        // The table cell k (ρ[k] ≥ ρ ≥ ρ[k + 1]), the end cells continued
        let k = if rho >= self.rho[0] {
            0
        } else if rho <= self.rho[n - 1] {
            n - 2
        } else {
            // ρ decreases: the last k with ρ[k] ≥ rho
            self.rho
                .partition_point(|&r| r >= rho)
                .saturating_sub(1)
                .min(n - 2)
        };
        let (r0, r1) = (self.rho[k], self.rho[k + 1]);
        let f = (r0 - rho) / (r0 - r1);
        let height = self.z0 + (k as f64 + f) * self.dz;
        // ∫ over the part of cell k below the height: trapezoid of the linear ρ_r
        let part = f * self.dz;
        let integral = self.integral[k] + 0.5 * part * (r0 + rho - 2.0 * self.offset);
        ReferencePoint {
            rho,
            height,
            integral,
        }
    }

    /// `ρ*_ij − {{ρ}}`: the reference's mean between the reference heights of
    /// `a` and `b` (see the module docs) less the arithmetic mean of their
    /// densities. Symmetric in `a` and `b`; zero when both lie in one table
    /// cell, where `ρ_r` is linear.
    pub fn pair_deviation(&self, a: &ReferencePoint, b: &ReferencePoint) -> f64 {
        let (a, b) = if a.height <= b.height { (a, b) } else { (b, a) };
        let cell = |z: f64| ((z - self.z0) / self.dz).floor();
        let (ka, kb) = (cell(a.height), cell(b.height));
        let n = self.len() as f64;
        // Within one cell, or both continued below or above the table
        if ka == kb || (ka < 0.0 && kb < 0.0) || (ka >= n - 1.0 && kb >= n - 1.0) {
            return 0.0;
        }
        let span = b.height - a.height;
        if kb == ka + 1.0 && ka >= 0.0 && kb < n - 1.0 {
            // One breakpoint between: the two trapezoids against the chord,
            // in differences of nearby values
            let k = kb as usize;
            let z_m = self.z0 + kb * self.dz;
            let rho_m = self.rho[k];
            let (u, w) = (z_m - a.height, b.height - z_m);
            return 0.5 * (u * (rho_m - b.rho) + w * (rho_m - a.rho)) / span;
        }
        (b.integral - a.integral) / span + self.offset - 0.5 * (a.rho + b.rho)
    }

    /// `θ_ij = (ρ*_ij − {{ρ}})/(ρ_j − ρ_i)` of the module docs, in `[−½, ½]`:
    /// the weight that turns arithmetic pair means of the tracers into the
    /// pair density's, `φ* = {{φ}} + θ_ij (φ_j − φ_i)`.
    #[inline]
    pub fn pair_weight(&self, i: &ReferencePoint, j: &ReferencePoint) -> f64 {
        let deviation = self.pair_deviation(i, j);
        let jump = j.rho - i.rho;
        if deviation == 0.0 || jump == 0.0 {
            0.0
        } else {
            (deviation / jump).clamp(-0.5, 0.5)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pycnocline: 4 kg/m³ across 15 m depth, weak gradient below.
    fn pycnocline(z: f64) -> f64 {
        1025.0 + 2.0 * (1.0 - ((z + 15.0) / 4.0).tanh()) - 0.002 * z
    }

    fn reference() -> ReferenceStratification {
        ReferenceStratification::from_fn(-310.0, 1.0, 3111, pycnocline)
    }

    #[test]
    fn heights_invert_the_table() {
        let r = reference();
        for z in [-305.0, -100.0, -17.3, -15.0, -12.1, -0.4] {
            let p = r.point(r.density(z));
            assert!((p.height - z).abs() < 1e-9, "{z}: {}", p.height);
        }
    }

    #[test]
    fn a_linear_profile_gives_the_arithmetic_mean() {
        let r = ReferenceStratification::from_fn(-100.0, 0.0, 101, |z| 1025.0 - 0.01 * z);
        let (a, b) = (r.point(1025.3), r.point(1025.9));
        assert!(r.pair_deviation(&a, &b).abs() < 1e-12);
        assert!(r.pair_weight(&a, &b).abs() < 1e-11);
    }

    /// The pair density is the reference's exact mean between the heights:
    /// across the pycnocline, far from the arithmetic mean.
    #[test]
    fn the_pair_density_is_the_mean_of_the_profile() {
        let r = reference();
        for (za, zb) in [
            (-300.0, -14.0),
            (-40.0, -2.0),
            (-15.05, -14.95),
            (-20.0, -19.9),
        ] {
            let (a, b) = (r.point(r.density(za)), r.point(r.density(zb)));
            let mean = {
                let n = 200_000;
                let h = (zb - za) / n as f64;
                (0..n)
                    .map(|m| r.density(za + (m as f64 + 0.5) * h))
                    .sum::<f64>()
                    / n as f64
            };
            let star = 0.5 * (a.rho + b.rho) + r.pair_deviation(&a, &b);
            assert!(
                (star - mean).abs() < 1e-8,
                "{za}..{zb}: {star} against {mean}"
            );
            assert!((r.pair_deviation(&b, &a) - r.pair_deviation(&a, &b)).abs() < 1e-15);
            let theta = r.pair_weight(&a, &b);
            assert!((-0.5..=0.5).contains(&theta));
            assert!((r.pair_weight(&b, &a) + theta).abs() < 1e-15);
        }
    }

    /// Densities beyond the table continue its end slopes.
    #[test]
    fn densities_beyond_the_table_continue_it() {
        let r = reference();
        let (lo, hi) = (
            r.point(r.density(-310.0) + 0.01),
            r.point(r.density(1.0) - 0.01),
        );
        assert!(lo.height < -310.0 && hi.height > 1.0);
        let d = r.pair_deviation(&lo, &hi);
        assert!(d.is_finite());
    }
}
