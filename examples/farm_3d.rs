//! A fish farm in a stratified tidal channel, in 3D and in 2D (TODO F.1).
//!
//! A straight channel, 6 km long (periodic along it) and 600 m wide, 40 m
//! deep, carries a tidal current of ≈ 0.5 m/s: a depth-uniform M2 body force
//! in the 2D module stands in for the tidal surface slope, ramped up over
//! the first hour. A row of two circular cages (50 m across, nets 20 m deep)
//! stands across the flow in the middle of the channel. The water is weakly
//! stratified (2 °C warmer over the top 10 m), mixed by GLS k-ε, against a
//! log-layer bottom drag.
//!
//! Four runs from rest: 3D with and without the cages
//! (`Hydrostatic3D::with_cage_drag`), and the 2D model with and without them
//! (`SWEPhysics2DBuilder::with_cage_drag`, Chézy friction with the 3D
//! drag's `C_d` bound). At the end it reports
//! - the depth-mean current behind the farm and its deficit in 2D and 3D;
//! - the 3D profile behind a cage: the deficit sits in the layers the net
//!   reaches, and part of the flow goes underneath;
//! - the drag force on the cages.
//!
//! With `particles=N`, the 3D run with cages also releases N particles of
//! each kind in each cage at `release` hours (default 2, in the running
//! tide) and tracks them online (`ParticleTracker3D`, TODO F.2) between the
//! solver's states every `particle_seconds` (default 60), with a horizontal
//! walk `kh` (m²/s, default 0.1) and Visser's vertical walk in the model's
//! GLS diffusivity:
//! - lice larvae, neutrally buoyant, released over the top 5 m, swimming
//!   by `lice` (`SalmonLice`: `ladim`, the default, the operational IMR
//!   parameters; `johnsen`, Johnsen et al. 2014; `passive`): up towards the
//!   light, a clear sky at Mausund (63.87° N, 8.67° E) from `start` (UTC),
//!   and down out of water fresher than their threshold;
//! - faeces, sinking at 3 cm/s, released over the net's depth;
//! - feed pellets, sinking at 10 cm/s, released over the top 2 m.
//!
//! Faeces and feed settle on the bed; the report gives the larvae's depths,
//! spread and swimming, and where the rest landed.
//!
//! ```bash
//! cargo run --release --no-default-features --features parallel,simd \
//!     --example farm_3d -- [hours=3] [order=2] [levels=16] [dx=60] [nu=1] [cs=0.2] \
//!     [particles=0] [release=2] [particle_seconds=60] [kh=0.1] \
//!     [lice=ladim] [start=2025-06-15T00:00:00Z]
//! ```

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;
use std::time::Instant;

use dg_rs::boundary::Reflective2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::data::Bathymetry2D;
use dg_rs::mesh::{Mesh2D, PointLocator2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::particles::{
    ClearSkyLight, Particle3D, ParticleStatus, ParticleTracker3D, ParticleVelocity3D, SalmonLice,
    Solution3DVelocity,
};
use dg_rs::physics::cage_drag::{for_each_caged_node, layer_coefficient};
use dg_rs::physics::{
    BottomDrag3D, Forcing, GlsMixing, Hydrostatic3D, LinearEOS, PhysicsBuilder, SWEPhysics2D,
};
use dg_rs::simulation::{Simulation, Simulation3D};
use dg_rs::solver::state::Solution3D;
use dg_rs::solver::{Probe2D, SWESolution2D, SWEState2D};
use dg_rs::source::{
    CageDrag2D, ChezyFriction2D, CoriolisSource2D, HorizontalViscosity2D, NetCage, SourceContext2D,
    SourceTerm2D,
};
use dg_rs::time::{ModeSplitIntegrator, ModelClock, SSPRK3};
use dg_rs::types::ElementIndex;
use dg_rs::vertical::{SigmaGrid, UniformStretching};

const G: f64 = 9.81;
const RHO0: f64 = 1025.0;
const LX: f64 = 6_000.0;
const LY: f64 = 600.0;
const DEPTH: f64 = 40.0;
const NET_DEPTH: f64 = 20.0;
const RADIUS: f64 = 25.0;
/// Cage centres: a row across the flow in the middle of the channel
const CAGES: [[f64; 2]; 2] = [[3_000.0, 240.0], [3_000.0, 360.0]];
const M2: f64 = 2.0 * PI / 44_714.16;
/// Where the sun is: Mausund, off Frøya (longitude, latitude)
const SITE: [f64; 2] = [8.67, 63.87];
/// Tidal current amplitude without drag (m/s)
const U_TIDE: f64 = 0.5;
const RAMP: f64 = 3_600.0;
/// Bed roughness of the log-layer drag (m) and its lower `C_d` bound
const Z0: f64 = 0.005;
const CD_MIN: f64 = 2.5e-3;

/// The tidal surface slope as a depth-uniform body force along the
/// channel, `∂(hu)/∂t = h·a(t)`, `a = U ω r(t) cos ωt` with a smooth ramp
/// `r` over the first hour.
struct TidalForce;

impl TidalForce {
    fn acceleration(t: f64) -> f64 {
        let ramp = if t < RAMP {
            0.5 * (1.0 - (PI * t / RAMP).cos())
        } else {
            1.0
        };
        U_TIDE * M2 * ramp * (M2 * t).cos()
    }
}

impl SourceTerm2D for TidalForce {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        SWEState2D::new(0.0, ctx.state.h * Self::acceleration(ctx.time), 0.0)
    }

    fn name(&self) -> &'static str {
        "tidal_force"
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: HashMap<String, String> = std::env::args()
        .skip(1)
        .filter_map(|a| a.split_once('=').map(|(k, v)| (k.into(), v.into())))
        .collect();
    let get = |key: &str, default: f64| args.get(key).map_or(Ok(default), |v| v.parse());
    let hours: f64 = get("hours", 3.0)?;
    let order = get("order", 2.0)? as usize;
    let n_levels = get("levels", 16.0)? as usize;
    let dx: f64 = get("dx", 60.0)?;
    let nu: f64 = get("nu", 1.0)?;
    let cs: f64 = get("cs", 0.2)?;
    let particles_per_kind = get("particles", 0.0)? as usize;
    let release_time = get("release", 2.0)? * 3600.0;
    let particle_seconds: f64 = get("particle_seconds", 60.0)?;
    let kh: f64 = get("kh", 0.1)?;
    let clock = ModelClock::parse(
        args.get("start")
            .map_or("2025-06-15T00:00:00Z", String::as_str),
    )?;
    let light = ClearSkyLight::new(clock, SITE[0], SITE[1]);
    let lice = match args.get("lice").map_or("ladim", String::as_str) {
        "ladim" => Some(SalmonLice::ladim(light)),
        "johnsen" => Some(SalmonLice::johnsen_2014(light)),
        "passive" => None,
        other => return Err(format!("lice={other}: ladim, johnsen or passive").into()),
    };
    let t_end = hours * 3600.0;

    let (nx, ny) = ((LX / dx).round() as usize, (LY / dx).round() as usize);
    let mesh = Arc::new(Mesh2D::channel_periodic_x(0.0, LX, 0.0, LY, nx, ny));
    let ops = Arc::new(DGOperators2D::new(order));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let bathymetry = Arc::new(Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -DEPTH));
    let sigma = Arc::new(SigmaGrid::new(n_levels, UniformStretching));
    let cages: Vec<NetCage> = CAGES
        .iter()
        .map(|&c| NetCage::circular(c, RADIUS, NET_DEPTH, 0.25))
        .collect();
    println!(
        "Channel {:.0} × {:.0} m, {} quads of {dx:.0} m (P{order}), {n_levels} σ-levels, {DEPTH} m deep",
        LX, LY, mesh.n_elements
    );
    println!(
        "Cages: {} × R {RADIUS} m, nets {NET_DEPTH} m deep, C_d·a = {:.4} /m",
        cages.len(),
        cages[0].drag_per_length
    );

    // The 2D module: in 3D it owns the depth mean (tidal force, Coriolis,
    // viscosity of ⟨u⟩); friction and cages are the 3D model's
    let swe = |with_friction: bool, with_cages: bool| -> SWEPhysics2D<Reflective2D> {
        let builder = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::new(),
        )
        .with_bathymetry(bathymetry.clone())
        .with_source(TidalForce)
        .with_source(CoriolisSource2D::f_plane(1.2e-4))
        .with_viscosity(HorizontalViscosity2D::constant(nu));
        let builder = if with_friction {
            // The 3D log-layer drag of a well-mixed column: C_d at the
            // bottom-layer centre, bounded below as in BottomDrag3D
            let z_b = 0.5 * DEPTH / n_levels as f64;
            let cd = BottomDrag3D::log_layer(Z0)
                .drag_coefficient(z_b)
                .max(CD_MIN);
            builder.with_implicit_friction(ChezyFriction2D::new(cd))
        } else {
            builder
        };
        if with_cages {
            builder.with_cage_drag(CageDrag2D::new(&mesh, &ops, &cages))
        } else {
            builder
        }
        .build()
    };

    let eos = LinearEOS::default();
    let physics_3d = |with_cages: bool| {
        let physics = Hydrostatic3D::new(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            sigma.clone(),
            bathymetry.clone(),
            Arc::new(CoriolisSource2D::f_plane(1.2e-4)),
            eos,
            GlsMixing::k_epsilon().with_roughness(0.02, Z0),
            swe(false, false),
            Forcing {
                surface_stress: [0.0, 0.0],
                bottom_stress: [0.0, 0.0],
                surface_buoyancy_flux: 0.0,
            },
            G,
            RHO0,
        )
        .with_bottom_drag(BottomDrag3D::log_layer(Z0))
        .with_horizontal_viscosity(nu)
        .with_smagorinsky_viscosity(cs);
        if with_cages {
            physics.with_cage_drag(CageDrag2D::new(&mesh, &ops, &cages))
        } else {
            physics
        }
    };

    let run_3d = |with_cages: bool| -> Result<Solution3D, String> {
        let physics = physics_3d(with_cages);
        let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, n_levels);
        state.salt.fill(eos.s0);
        // 2 °C warmer over the top 10 m, a smooth step
        for column in state.temp.chunks_exact_mut(n_levels) {
            for (l, t) in column.iter_mut().enumerate() {
                let z = DEPTH * sigma.sigma_rho()[l];
                *t = eos.t0 + 1.0 + (0.5 * (z + 10.0)).tanh();
            }
        }
        physics.update_density(&mut state);
        let label = if with_cages { "3D cages" } else { "3D open" };
        let start = Instant::now();
        let mut tracking = (with_cages && particles_per_kind > 0)
            .then(|| FarmParticles::new(&mesh, &ops, particles_per_kind, release_time, kh, lice));
        let mut sim = Simulation3D::new(physics, ModeSplitIntegrator::new()).with_cfl(0.5);
        if tracking.is_some() {
            sim = sim.with_callback_interval(particle_seconds);
        }
        let result = sim.run_with_callback(&mut state, 0.0, t_end, |s, t| {
            if let Some(tracking) = tracking.as_mut() {
                tracking.advance(s, t, &sigma, &bathymetry);
            }
        });
        if let Some(tracking) = &tracking {
            tracking.report(&state, &sigma, &bathymetry);
        }
        println!(
            "{label:>9}: {:5} steps, dt {:.1}–{:.1} s, {:6.1} s wall",
            result.n_steps,
            result.dt_min,
            result.dt_max,
            start.elapsed().as_secs_f64()
        );
        if !result.success {
            return Err(format!("{label}: {:?}", result.error));
        }
        Ok(state)
    };

    let run_2d = |with_cages: bool| -> Result<SWESolution2D, String> {
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        q.data[0].fill(DEPTH);
        let label = if with_cages { "2D cages" } else { "2D open" };
        let start = Instant::now();
        let result = Simulation::new(swe(true, with_cages), SSPRK3)
            .with_cfl(0.5)
            .run(&mut q, 0.0, t_end);
        println!(
            "{label:>9}: {:5} steps, dt {:.2}–{:.2} s, {:6.1} s wall",
            result.n_steps,
            result.dt_min,
            result.dt_max,
            start.elapsed().as_secs_f64()
        );
        if !result.success {
            return Err(format!("{label}: {:?}", result.error));
        }
        Ok(q)
    };

    println!("\nM2 body force from rest, {hours} h:");
    let caged_3d = run_3d(true)?;
    let open_3d = run_3d(false)?;
    let caged_2d = run_2d(true)?;
    let open_2d = run_2d(false)?;

    // The flow direction at the end: report the lee side
    let locator = PointLocator2D::new(&mesh);
    let probe = |p: [f64; 2]| Probe2D::at(&locator, &ops, p).expect("the point is in the channel");
    let mean_3d = |s: &Solution3D, p: &Probe2D| p.evaluate_field(&s.ubar.data);
    let mean_2d = |q: &SWESolution2D, p: &Probe2D| {
        let sample = p.sample_swe(q, Some(&bathymetry), 1e-3);
        sample.u
    };
    let farm_u = mean_3d(&open_3d, &probe(CAGES[0]));
    let lee = farm_u.signum();
    println!(
        "\nUndisturbed current at the farm: 3D {:+.3} m/s, 2D {:+.3} m/s (the lee is {} of the farm)",
        farm_u,
        mean_2d(&open_2d, &probe(CAGES[0])),
        if lee > 0.0 { "east" } else { "west" }
    );

    println!("\nDepth-mean current behind cage 0 (deficit against the run without cages):");
    println!("  behind    3D (m/s)   deficit    2D (m/s)   deficit");
    for distance in [-150.0, 0.0, 50.0, 100.0, 200.0, 400.0, 800.0] {
        let p = probe([CAGES[0][0] + lee * distance, CAGES[0][1]]);
        let (u3, u3o) = (mean_3d(&caged_3d, &p), mean_3d(&open_3d, &p));
        let (u2, u2o) = (mean_2d(&caged_2d, &p), mean_2d(&open_2d, &p));
        println!(
            "  {distance:5.0} m  {u3:+9.4}   {:6.2} %   {u2:+9.4}   {:6.2} %",
            100.0 * (1.0 - u3 / u3o),
            100.0 * (1.0 - u2 / u2o)
        );
    }

    let column = |s: &Solution3D, p: &Probe2D| -> Vec<f64> {
        let k = p.element().as_usize();
        (0..n_levels)
            .map(|l| {
                let values: Vec<f64> = (0..ops.n_nodes)
                    .map(|i| s.u[(k * ops.n_nodes + i) * n_levels + l])
                    .collect();
                p.evaluate(&values)
            })
            .collect()
    };
    for distance in [0.0, 100.0] {
        let p = probe([CAGES[0][0] + lee * distance, CAGES[0][1]]);
        let (caged, open) = (column(&caged_3d, &p), column(&open_3d, &p));
        println!("\n3D profile {distance:.0} m behind cage 0 (z of the layer centre):");
        println!("      z (m)   cages (m/s)   open (m/s)   deficit");
        for l in (0..n_levels).rev() {
            let z = DEPTH * sigma.sigma_rho()[l];
            let marker = if -z < NET_DEPTH { "  net" } else { "" };
            println!(
                "  {z:9.2}   {:+11.4}   {:+10.4}   {:6.2} %{marker}",
                caged[l],
                open[l],
                100.0 * (1.0 - caged[l] / open[l])
            );
        }
    }

    // Drag force on the cages, F = ρ₀ ∫ Σ_l H_l λ_l u_l dA (3D) and
    // ρ₀ ∫ Λ hu dA (2D), along the channel
    let cage_drag = CageDrag2D::new(&mesh, &ops, &cages);
    let nn = ops.n_nodes;
    let mut force_3d = 0.0;
    for k in 0..mesh.n_elements {
        for_each_caged_node(&cage_drag, k, nn, |i, entries| {
            let idx = k * nn + i;
            let depth = caged_3d.eta.data[idx] + DEPTH;
            let mut drag = 0.0;
            for l in 0..n_levels {
                let (u, v) = (
                    caged_3d.u[idx * n_levels + l],
                    caged_3d.v[idx * n_levels + l],
                );
                let rate = layer_coefficient(entries, &sigma, l, depth) * u.hypot(v);
                drag += sigma.d_sigma()[l] * depth * rate * u;
            }
            force_3d += RHO0 * geom.mass[idx] * drag;
        });
    }
    let mut force_2d = 0.0;
    for k in ElementIndex::iter(mesh.n_elements) {
        for_each_caged_node(&cage_drag, k.as_usize(), nn, |i, entries| {
            let s = caged_2d.get_state(k, i);
            let speed = s.hu.hypot(s.hv) / s.h;
            let rate = CageDrag2D::damping_rate(entries, s.h, speed);
            force_2d += RHO0 * geom.mass[k.as_usize() * nn + i] * rate * s.hu;
        });
    }
    println!(
        "\nDrag on the farm along the channel: 3D {:.1} kN, 2D {:.1} kN",
        force_3d / 1e3,
        force_2d / 1e3
    );
    Ok(())
}

/// What a particle kind is: name, own vertical speed (m/s, positive up),
/// and the depth range it is released over (m below the surface).
const KINDS: [(&str, f64, [f64; 2]); 3] = [
    ("larvae", 0.0, [0.0, 5.0]),
    ("faeces", -0.03, [0.0, NET_DEPTH]),
    ("feed", -0.10, [0.0, 2.0]),
];

/// Index of the larvae in [`KINDS`].
const LARVAE: usize = 0;

/// Particles released from the cages and tracked online in the 3D flow.
struct FarmParticles<'a> {
    tracker: ParticleTracker3D<'a>,
    tracker_mesh: &'a Mesh2D,
    ops: &'a DGOperators2D,
    n: usize,
    release_time: f64,
    /// The larvae's behaviour (`None`: passive)
    lice: Option<SalmonLice<ClearSkyLight>>,
    /// Particles of each kind once released, with the cage of each
    particles: [Vec<Particle3D>; 3],
    cages: [Vec<usize>; 3],
    previous: Option<(f64, Solution3D)>,
    seconds: f64,
}

impl<'a> FarmParticles<'a> {
    fn new(
        mesh: &'a Mesh2D,
        ops: &'a DGOperators2D,
        n: usize,
        release_time: f64,
        kh: f64,
        lice: Option<SalmonLice<ClearSkyLight>>,
    ) -> Self {
        Self {
            tracker: ParticleTracker3D::new(mesh, ops)
                .with_horizontal_diffusivity(kh)
                .with_vertical_random_walk()
                .with_bed_settling(),
            tracker_mesh: mesh,
            ops,
            n,
            release_time,
            lice,
            particles: Default::default(),
            cages: Default::default(),
            previous: None,
            seconds: 0.0,
        }
    }

    fn released(&self) -> usize {
        self.particles.iter().map(Vec::len).sum()
    }

    /// `n` particles of each kind in each cage: a sunflower pattern over
    /// the footprint, evenly over the kind's depth range.
    fn release(&mut self) {
        let golden_angle = PI * (3.0 - 5.0_f64.sqrt());
        let mut id = 0;
        for (c, center) in CAGES.iter().enumerate() {
            for (kind, &(_, speed, [top, bottom])) in KINDS.iter().enumerate() {
                for i in 0..self.n {
                    let r = RADIUS * ((i as f64 + 0.5) / self.n as f64).sqrt();
                    let angle = i as f64 * golden_angle;
                    let p = [center[0] + r * angle.cos(), center[1] + r * angle.sin()];
                    let below = top + (bottom - top) * (i as f64 + 0.5) / self.n as f64;
                    let particle = self
                        .tracker
                        .release(id, p, -below / DEPTH, speed)
                        .expect("cages are in the channel");
                    id += 1;
                    self.particles[kind].push(particle);
                    self.cages[kind].push(c);
                }
            }
        }
    }

    /// Move the particles from the previous state to `state` at `t`, in
    /// steps of at most 10 s, linear in time between the two.
    fn advance(&mut self, state: &Solution3D, t: f64, sigma: &SigmaGrid, bed: &Bathymetry2D) {
        let start = Instant::now();
        if self.previous.is_some() && t > self.release_time && self.released() == 0 {
            self.release();
        }
        if let Some((t0, s0)) = &self.previous
            && t > self.release_time
        {
            let from = t0.max(self.release_time);
            let field = Solution3DVelocity::between(*t0, s0, t, state, sigma, bed, 0.05);
            let n = ((t - from) / 10.0).ceil().max(1.0) as usize;
            let dt = (t - from) / n as f64;
            for s in 0..n {
                let t = from + s as f64 * dt;
                for (kind, particles) in self.particles.iter_mut().enumerate() {
                    match &self.lice {
                        Some(lice) if kind == LARVAE => {
                            self.tracker.step_with(particles, &field, lice, t, dt)
                        }
                        _ => self.tracker.step(particles, &field, t, dt),
                    }
                }
            }
        }
        match &mut self.previous {
            Some((tp, sp)) => {
                *tp = t;
                sp.clone_from(state);
            }
            None => self.previous = Some((t, state.clone())),
        }
        self.seconds += start.elapsed().as_secs_f64();
    }

    /// Per kind: in the water and on the bed; depths, drift and spread of
    /// those in the water; where the settled ones landed.
    fn report(&self, state: &Solution3D, sigma: &SigmaGrid, bed: &Bathymetry2D) {
        if self.released() == 0 {
            println!("  (no particles released before the end)");
            return;
        }
        println!(
            "\nParticles: {} released at {:.1} h, {:.2} s tracking",
            self.released(),
            self.release_time / 3600.0,
            self.seconds
        );
        if let Some(lice) = &self.lice {
            let t = self.previous.as_ref().map_or(0.0, |(t, _)| *t);
            println!(
                "  Sun at the end: {:.1}° high, {:.0} µmol photons/m²/s at the surface; larvae swim up where the light exceeds {} (≈ {:.0} m deep)",
                lice.light.solar_height(t),
                lice.light.irradiance(t),
                lice.light_threshold[0],
                (lice.light.irradiance(t) / lice.light_threshold[0]).ln() / lice.attenuation
            );
        }
        let field = Solution3DVelocity::steady(state, sigma, bed, 0.05);
        // The vertical diffusivity the walk sees, 150 m up-current of the farm
        let locator = PointLocator2D::new(self.tracker_mesh);
        let ahead = [CAGES[0][0] - 150.0, CAGES[0][1]];
        if let Some(point) = locator.locate(ahead) {
            let w = self.ops.interpolation_weights(point.r, point.s);
            let depth = field.depth(point.element, &w, 0.0);
            let profile: Vec<String> = [1.0, 3.0, 6.0, 10.0, 15.0, 25.0, 35.0]
                .iter()
                .map(|&z| {
                    let k = field
                        .diffusivity(point.element, &w, -z / depth, 0.0)
                        .map_or(0.0, |(k, _)| k);
                    format!("{z:.0} m {k:.1e}")
                })
                .collect();
            println!("  K (m²/s) up-current of the farm: {}", profile.join(", "));
        }
        // Displacement from the cage, the nearest periodic image along x
        let offset = |p: &Particle3D, c: usize| {
            let [x, y] = p.position();
            let dx = (x - CAGES[c][0] + 0.5 * LX).rem_euclid(LX) - 0.5 * LX;
            [dx, y - CAGES[c][1]]
        };
        for (kind, &(name, ..)) in KINDS.iter().enumerate() {
            let of_kind: Vec<(&Particle3D, usize)> = self.particles[kind]
                .iter()
                .zip(self.cages[kind].iter().copied())
                .collect();
            let settled: Vec<[f64; 2]> = of_kind
                .iter()
                .filter(|(p, _)| p.status() == ParticleStatus::Settled)
                .map(|&(p, c)| offset(p, c))
                .collect();
            let water: Vec<(f64, [f64; 2])> = of_kind
                .iter()
                .filter(|(p, _)| p.status() == ParticleStatus::Active)
                .map(|&(p, c)| {
                    let point = p.point();
                    let w = self.ops.interpolation_weights(point.r, point.s);
                    let depth = field.depth(point.element, &w, 0.0);
                    (p.depth_below_surface(depth), offset(p, c))
                })
                .collect();
            print!(
                "  {name:>6}: {:4} in the water, {:4} on the bed",
                water.len(),
                settled.len()
            );
            if !water.is_empty() {
                let mut depths: Vec<f64> = water.iter().map(|(d, _)| *d).collect();
                depths.sort_by(f64::total_cmp);
                let pct = |q: f64| depths[(q * (depths.len() - 1) as f64).round() as usize];
                let n = water.len() as f64;
                let mean = |d: usize| water.iter().map(|(_, x)| x[d]).sum::<f64>() / n;
                let spread = (water
                    .iter()
                    .map(|(_, x)| (x[0] - mean(0)).powi(2) + (x[1] - mean(1)).powi(2))
                    .sum::<f64>()
                    / n)
                    .sqrt();
                print!(
                    "; depth 10/50/90 % {:.1}/{:.1}/{:.1} m, drift ({:+.0}, {:+.0}) m, spread {spread:.0} m",
                    pct(0.1),
                    pct(0.5),
                    pct(0.9),
                    mean(0),
                    mean(1)
                );
            }
            if kind == LARVAE && self.lice.is_some() && !water.is_empty() {
                let swimming = |up: bool| {
                    of_kind
                        .iter()
                        .filter(|(p, _)| p.status() == ParticleStatus::Active)
                        .filter(|(p, _)| {
                            let w = p.swimming_speed();
                            if up { w > 0.0 } else { w < 0.0 }
                        })
                        .count() as f64
                        / water.len() as f64
                };
                print!(
                    "; swimming up {:.0} %, down {:.0} %",
                    100.0 * swimming(true),
                    100.0 * swimming(false)
                );
            }
            if !settled.is_empty() {
                let mut distance: Vec<f64> = settled.iter().map(|x| x[0].hypot(x[1])).collect();
                distance.sort_by(f64::total_cmp);
                let mean_x = settled.iter().map(|x| x[0]).sum::<f64>() / settled.len() as f64;
                print!(
                    "; landed {mean_x:+.0} m along the channel on average, 50/90 % within {:.0}/{:.0} m of the cage",
                    distance[distance.len() / 2],
                    distance[(9 * distance.len() / 10).min(distance.len() - 1)]
                );
            }
            println!();
        }
    }
}
