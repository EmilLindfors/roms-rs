//! Connectivity between sites from tracked particles (TODO F.2): which share
//! of the particles released at one site comes into contact with another,
//! for how long, and how sure that number is.
//!
//! # Contact and exposure
//!
//! A site is a [`ContactZone`]: a circular footprint (a net cage, a farm) and
//! a depth range below the surface (the net's depth, or the top few metres
//! where lice larvae meet the fish). A particle is in contact with a zone
//! while it is inside the footprint and within the depth range. Its exposure
//! to the zone is the weighted time it spends there,
//!
//! ```text
//! E_j = ∫ w(t) 1[x(t) ∈ Z_j] dt ≈ Σ_n w_n Δt_n 1[xⁿ ∈ Z_j],
//! ```
//!
//! summed over the tracking steps with the particle's position at the end of
//! each step ([`ConnectivityRecorder::record`]). The weight `w` is the
//! caller's: 1 for a passive tracer, 0 before larvae are infective, a
//! survival or infectivity factor otherwise. The sum is the rectangle rule,
//! first order in the step: a passage through a footprint of width `L` at
//! speed `U` is counted to within one step, `|ΔE| ≤ Δt`, so keep `UΔt ≪ L`.
//!
//! A particle released inside its own zone (larvae from a cage) is not in
//! contact with it until it has once been outside the footprint: the
//! diagonal of the matrix counts returns (self-infection), not the release.
//! A random walk near the edge of the footprint steps out and back in within
//! a step or two, which is not a return. With
//! [`ConnectivityRecorder::with_return_distance`] the particle must first
//! get that far beyond the footprint.
//!
//! # The matrix
//!
//! Each particle has a source zone and a group (a release batch, a tidal
//! phase, a season: the caller's). For the particles of a set of groups,
//! [`ConnectivityRecorder::matrix`] counts, per source `i` and receiver `j`,
//! the particles with exposure `E_j ≥ E_min`, and gives the share
//! `C_ij = n_ij / N_i` of the `N_i` released at `i` and the mean exposure
//! per released particle `Σ E_j / N_i`. [`ConnectivityRecorder::contacts`]
//! gives each contact of a pair (exposure, age at first contact) for their
//! distributions.
//!
//! # Uncertainty
//!
//! One run's `C_ij` is a sample: the random walk makes it vary from seed to
//! seed. [`ConnectivityEnsemble`] holds the matrices of runs that differ only
//! in the walk's seed (the same flow, the same releases), and gives the mean,
//! the spread `s_ij` across seeds and the standard error `s_ij/√R` of the
//! mean of `R` seeds. Two layouts (or diffusivities, or meshes) differ
//! beyond the noise when the difference of their means is several standard
//! errors ([`ConnectivityEnsemble::difference`]). Independent particles give
//! a binomial spread `√(C(1 − C)/N)` ([`ConnectivityMatrix::binomial_error`]);
//! particles released at fixed points in one flow spread less.

/// A site particles can come into contact with: a circular footprint and a
/// depth range below the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactZone {
    /// Centre of the footprint (mesh coordinates, m).
    pub center: [f64; 2],
    /// Radius of the footprint (m).
    pub radius: f64,
    /// Depth range below the surface, `[top, bottom]` (m, positive down).
    pub depth: [f64; 2],
}

impl ContactZone {
    /// A circular footprint of `radius` around `center` over `depth`
    /// (`[top, bottom]` below the surface, m).
    pub fn circular(center: [f64; 2], radius: f64, depth: [f64; 2]) -> Self {
        assert!(radius > 0.0, "a zone needs a positive radius");
        assert!(
            depth[0] <= depth[1],
            "the depth range is [top, bottom], top first"
        );
        Self {
            center,
            radius,
            depth,
        }
    }

    /// Whether `position` is inside the footprint.
    #[inline]
    pub fn covers(&self, position: [f64; 2]) -> bool {
        self.within(position, 0.0)
    }

    /// Whether `position` is no further than `margin` outside the footprint.
    #[inline]
    pub fn within(&self, position: [f64; 2], margin: f64) -> bool {
        let (dx, dy) = (position[0] - self.center[0], position[1] - self.center[1]);
        let r = self.radius + margin;
        dx * dx + dy * dy <= r * r
    }

    /// Whether `depth` (below the surface, m) is within the depth range.
    #[inline]
    pub fn spans(&self, depth: f64) -> bool {
        (self.depth[0]..=self.depth[1]).contains(&depth)
    }
}

/// One particle's contact with one zone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// The receiving zone.
    pub zone: usize,
    /// The particle's age (s since release) at the end of the first step
    /// it was in contact.
    pub first: f64,
    /// Weighted time in contact (s).
    pub exposure: f64,
}

/// What the recorder keeps of one particle.
#[derive(Clone, Debug)]
struct Track {
    source: u32,
    group: u32,
    released: f64,
    /// Whether it has been the return distance outside its source's
    /// footprint
    left_source: bool,
    /// Its contacts, at most one per zone, in the order they began
    contacts: Vec<Contact>,
}

/// Exposure of tracked particles to a set of zones (see the
/// [module docs](self)).
#[derive(Clone, Debug)]
pub struct ConnectivityRecorder {
    zones: Vec<ContactZone>,
    return_distance: f64,
    tracks: Vec<Track>,
}

impl ConnectivityRecorder {
    /// A recorder of contacts with `zones`; zone indices are positions in it.
    pub fn new(zones: Vec<ContactZone>) -> Self {
        assert!(!zones.is_empty(), "connectivity needs at least one zone");
        Self {
            zones,
            return_distance: 0.0,
            tracks: Vec::new(),
        }
    }

    /// How far beyond its source's footprint (m) a particle must have been
    /// before contact with the source counts as a return (default 0: any
    /// step outside).
    pub fn with_return_distance(mut self, distance: f64) -> Self {
        assert!(distance >= 0.0, "the return distance is a distance");
        self.return_distance = distance;
        self
    }

    /// The zones, in index order.
    pub fn zones(&self) -> &[ContactZone] {
        &self.zones
    }

    /// Number of particles added.
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Whether no particle has been added.
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// A particle released at time `released` from zone `source`, in
    /// `group`. Returns its index for [`Self::record`] (particles are
    /// numbered in the order they are added).
    pub fn add(&mut self, source: usize, group: usize, released: f64) -> usize {
        assert!(source < self.zones.len(), "source {source} is not a zone");
        self.tracks.push(Track {
            source: source as u32,
            group: group as u32,
            released,
            left_source: false,
            contacts: Vec::new(),
        });
        self.tracks.len() - 1
    }

    /// Particle `index` was at `position` at time `t`, the end of a step of
    /// `dt`, with weight `weight`; `depth` gives its depth below the surface
    /// (m) and is called only when the particle is inside a footprint.
    pub fn record(
        &mut self,
        index: usize,
        t: f64,
        dt: f64,
        position: [f64; 2],
        depth: impl FnOnce() -> f64,
        weight: f64,
    ) {
        let track = &mut self.tracks[index];
        let source = track.source as usize;
        if !track.left_source && !self.zones[source].within(position, self.return_distance) {
            track.left_source = true;
        }
        if weight <= 0.0 {
            return;
        }
        let mut depth = Some(depth);
        let mut below = f64::NAN;
        for (j, zone) in self.zones.iter().enumerate() {
            if !zone.covers(position) || (j == source && !track.left_source) {
                continue;
            }
            if let Some(depth) = depth.take() {
                below = depth();
            }
            if !zone.spans(below) {
                continue;
            }
            match track.contacts.iter_mut().find(|c| c.zone == j) {
                Some(contact) => contact.exposure += weight * dt,
                None => track.contacts.push(Contact {
                    zone: j,
                    first: t - track.released,
                    exposure: weight * dt,
                }),
            }
        }
    }

    /// The source zone, group and release time of particle `index`.
    pub fn particle(&self, index: usize) -> (usize, usize, f64) {
        let track = &self.tracks[index];
        (track.source as usize, track.group as usize, track.released)
    }

    /// The contacts of particle `index`, at most one per zone.
    pub fn particle_contacts(&self, index: usize) -> &[Contact] {
        &self.tracks[index].contacts
    }

    /// The connectivity of the particles whose group passes `groups`:
    /// contact counts those with exposure at least `min_exposure` (s).
    pub fn matrix(&self, min_exposure: f64, groups: impl Fn(usize) -> bool) -> ConnectivityMatrix {
        let n = self.zones.len();
        let mut matrix = ConnectivityMatrix {
            n_zones: n,
            released: vec![0; n],
            reached: vec![0; n * n],
            exposure: vec![0.0; n * n],
        };
        for track in self.tracks.iter().filter(|t| groups(t.group as usize)) {
            let i = track.source as usize;
            matrix.released[i] += 1;
            for c in &track.contacts {
                matrix.exposure[i * n + c.zone] += c.exposure;
                if c.exposure >= min_exposure {
                    matrix.reached[i * n + c.zone] += 1;
                }
            }
        }
        matrix
    }

    /// Every contact of a particle from `source` with `receiver`, for the
    /// particles whose group passes `groups`: the distributions of exposure
    /// and arrival age of the pair.
    pub fn contacts(
        &self,
        source: usize,
        receiver: usize,
        groups: impl Fn(usize) -> bool,
    ) -> Vec<Contact> {
        self.tracks
            .iter()
            .filter(|t| t.source as usize == source && groups(t.group as usize))
            .flat_map(|t| t.contacts.iter().filter(|c| c.zone == receiver).copied())
            .collect()
    }
}

/// Source-by-receiver connectivity of one run (see the [module docs](self)).
#[derive(Clone, Debug, PartialEq)]
pub struct ConnectivityMatrix {
    n_zones: usize,
    released: Vec<usize>,
    /// Particles in contact, `[source × receiver]`
    reached: Vec<usize>,
    /// Total exposure (s), `[source × receiver]`
    exposure: Vec<f64>,
}

impl ConnectivityMatrix {
    /// Number of zones (sources and receivers).
    pub fn n_zones(&self) -> usize {
        self.n_zones
    }

    /// Particles released at `source`.
    pub fn released(&self, source: usize) -> usize {
        self.released[source]
    }

    /// Particles from `source` in contact with `receiver`.
    pub fn reached(&self, source: usize, receiver: usize) -> usize {
        self.reached[source * self.n_zones + receiver]
    }

    /// Share of the particles released at `source` in contact with
    /// `receiver` (0 when none were released).
    pub fn share(&self, source: usize, receiver: usize) -> f64 {
        match self.released[source] {
            0 => 0.0,
            n => self.reached(source, receiver) as f64 / n as f64,
        }
    }

    /// Exposure to `receiver` per particle released at `source` (s).
    pub fn mean_exposure(&self, source: usize, receiver: usize) -> f64 {
        match self.released[source] {
            0 => 0.0,
            n => self.exposure[source * self.n_zones + receiver] / n as f64,
        }
    }

    /// The share's standard error were the particles independent,
    /// `√(C(1 − C)/N)`.
    pub fn binomial_error(&self, source: usize, receiver: usize) -> f64 {
        match self.released[source] {
            0 => 0.0,
            n => {
                let c = self.share(source, receiver);
                (c * (1.0 - c) / n as f64).sqrt()
            }
        }
    }
}

/// A difference between two estimates and its standard error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Difference {
    /// `a − b`.
    pub estimate: f64,
    /// `√(σ_a² + σ_b²)` of the two means.
    pub standard_error: f64,
}

impl Difference {
    /// The difference in standard errors (infinite for a nonzero difference
    /// without noise, 0 for none).
    pub fn z(&self) -> f64 {
        if self.estimate == 0.0 {
            0.0
        } else {
            self.estimate / self.standard_error
        }
    }
}

/// The matrices of runs that differ only in the random walk's seed (see the
/// [module docs](self)).
#[derive(Clone, Debug)]
pub struct ConnectivityEnsemble {
    members: Vec<ConnectivityMatrix>,
}

impl ConnectivityEnsemble {
    /// An ensemble of at least one matrix, all over the same zones.
    pub fn new(members: Vec<ConnectivityMatrix>) -> Self {
        assert!(!members.is_empty(), "an ensemble needs a member");
        let n = members[0].n_zones;
        assert!(
            members.iter().all(|m| m.n_zones == n),
            "the members have different zones"
        );
        Self { members }
    }

    /// The members, one per seed.
    pub fn members(&self) -> &[ConnectivityMatrix] {
        &self.members
    }

    /// Mean and sample variance across the members of `f`.
    fn moments(&self, f: impl Fn(&ConnectivityMatrix) -> f64) -> (f64, f64) {
        let r = self.members.len() as f64;
        let mean = self.members.iter().map(&f).sum::<f64>() / r;
        if self.members.len() < 2 {
            return (mean, 0.0);
        }
        let var = self
            .members
            .iter()
            .map(|m| (f(m) - mean).powi(2))
            .sum::<f64>()
            / (r - 1.0);
        (mean, var)
    }

    /// Mean share across the seeds.
    pub fn mean(&self, source: usize, receiver: usize) -> f64 {
        self.moments(|m| m.share(source, receiver)).0
    }

    /// Standard deviation of the share across the seeds (0 for one seed).
    pub fn spread(&self, source: usize, receiver: usize) -> f64 {
        self.moments(|m| m.share(source, receiver)).1.sqrt()
    }

    /// Standard error of the mean share, `spread/√R`.
    pub fn standard_error(&self, source: usize, receiver: usize) -> f64 {
        self.spread(source, receiver) / (self.members.len() as f64).sqrt()
    }

    /// Mean exposure per released particle across the seeds (s) and its
    /// standard error.
    pub fn mean_exposure(&self, source: usize, receiver: usize) -> (f64, f64) {
        let (mean, var) = self.moments(|m| m.mean_exposure(source, receiver));
        (mean, (var / self.members.len() as f64).sqrt())
    }

    /// This ensemble's mean share less `other`'s, with the standard error
    /// of the difference.
    pub fn difference(&self, other: &Self, source: usize, receiver: usize) -> Difference {
        Difference {
            estimate: self.mean(source, receiver) - other.mean(source, receiver),
            standard_error: self
                .standard_error(source, receiver)
                .hypot(other.standard_error(source, receiver)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cages() -> ConnectivityRecorder {
        ConnectivityRecorder::new(vec![
            ContactZone::circular([0.0, 0.0], 10.0, [0.0, 5.0]),
            ContactZone::circular([100.0, 0.0], 10.0, [0.0, 5.0]),
        ])
    }

    /// A particle released in its cage is not in contact with it until it
    /// has left the footprint; then a return counts.
    #[test]
    fn own_cage_counts_only_returns() {
        let mut rec = two_cages();
        let p = rec.add(0, 0, 0.0);
        rec.record(p, 10.0, 10.0, [1.0, 0.0], || 1.0, 1.0);
        assert!(rec.particle_contacts(p).is_empty());
        // Below the zone but outside the footprint: it has left
        rec.record(p, 20.0, 10.0, [20.0, 0.0], || 1.0, 1.0);
        rec.record(p, 30.0, 10.0, [5.0, 0.0], || 1.0, 1.0);
        assert_eq!(
            rec.particle_contacts(p),
            &[Contact {
                zone: 0,
                first: 30.0,
                exposure: 10.0
            }]
        );
        // With a return distance, a step just outside and back is no return
        let mut rec = two_cages().with_return_distance(10.0);
        let p = rec.add(0, 0, 0.0);
        rec.record(p, 10.0, 10.0, [15.0, 0.0], || 1.0, 1.0);
        rec.record(p, 20.0, 10.0, [5.0, 0.0], || 1.0, 1.0);
        assert!(rec.particle_contacts(p).is_empty());
        rec.record(p, 30.0, 10.0, [25.0, 0.0], || 1.0, 1.0);
        rec.record(p, 40.0, 10.0, [5.0, 0.0], || 1.0, 1.0);
        assert_eq!(rec.particle_contacts(p)[0].first, 40.0);
    }

    /// Exposure is weighted time inside the footprint and the depth range;
    /// the depth is sampled only inside a footprint.
    #[test]
    fn exposure_is_weighted_time_in_the_zone() {
        let mut rec = two_cages();
        let p = rec.add(0, 3, 100.0);
        let far = || -> f64 { panic!("depth sampled outside every footprint") };
        rec.record(p, 110.0, 10.0, [50.0, 0.0], far, 1.0);
        rec.record(p, 120.0, 10.0, [95.0, 0.0], || 2.0, 0.5);
        rec.record(p, 130.0, 10.0, [100.0, 0.0], || 7.0, 1.0); // too deep
        rec.record(p, 135.0, 5.0, [105.0, 0.0], || 0.0, 1.0);
        rec.record(p, 145.0, 10.0, [100.0, 0.0], || 1.0, 0.0); // not counted
        assert_eq!(
            rec.particle_contacts(p),
            &[Contact {
                zone: 1,
                first: 20.0,
                exposure: 10.0
            }]
        );
        let m = rec.matrix(0.0, |_| true);
        assert_eq!((m.released(0), m.reached(0, 1), m.reached(0, 0)), (1, 1, 0));
        assert_eq!(m.mean_exposure(0, 1), 10.0);
        // A threshold above the exposure, and a group filter
        assert_eq!(rec.matrix(10.5, |_| true).reached(0, 1), 0);
        assert_eq!(rec.matrix(0.0, |g| g != 3).released(0), 0);
    }

    #[test]
    fn matrix_shares_and_ensemble_statistics() {
        let mut members = Vec::new();
        for reached in [3, 5] {
            let mut rec = two_cages();
            for i in 0..10 {
                let p = rec.add(0, 0, 0.0);
                if i < reached {
                    rec.record(p, 1.0, 1.0, [100.0, 0.0], || 1.0, 1.0);
                }
            }
            let m = rec.matrix(0.0, |_| true);
            let c = reached as f64 / 10.0;
            assert_eq!(m.share(0, 1), c);
            assert!((m.binomial_error(0, 1) - (c * (1.0 - c) / 10.0).sqrt()).abs() < 1e-15);
            assert_eq!(m.share(1, 0), 0.0);
            members.push(m);
        }
        let ensemble = ConnectivityEnsemble::new(members);
        assert!((ensemble.mean(0, 1) - 0.4).abs() < 1e-15);
        // Sample standard deviation of {0.3, 0.5}
        assert!((ensemble.spread(0, 1) - 0.02_f64.sqrt()).abs() < 1e-15);
        assert!((ensemble.standard_error(0, 1) - 0.1).abs() < 1e-15);
        let one = ConnectivityEnsemble::new(vec![ensemble.members()[0].clone()]);
        let d = ensemble.difference(&one, 0, 1);
        assert!((d.estimate - 0.1).abs() < 1e-15 && (d.standard_error - 0.1).abs() < 1e-15);
        assert!((d.z() - 1.0).abs() < 1e-12);
    }
}
