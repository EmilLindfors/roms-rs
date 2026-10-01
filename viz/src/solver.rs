//! The solver, run on a thread of its own while the viewer draws.
//!
//! [`spawn`] builds the scenario's physics and runs `Simulation::run_with_callback`
//! with a callback every `interval` seconds of model time. Each callback reduces the
//! state to what the viewer draws, a [`Snapshot`] of the surface η and the
//! depth-averaged velocity (u, v) at every node, plus the particles released from the
//! cages ([`crate::particles::Cloud`], tracked here between snapshots), and sends it
//! down a channel. A scenario with a 3D model runs `Simulation3D` instead
//! (`Hydrostatic3D` with mode splitting), and its snapshots carry the layers too
//! ([`Layers`]: u, v and temperature on every σ-level) and particles tracked in 3D
//! ([`crate::cloud_3d::Cloud3D`]). The solver's own rayon pool leaves a couple of cores
//! to Bevy.

use std::sync::mpsc::{Receiver, channel};
use std::time::Instant;

use std::f64::consts::PI;
use std::sync::Arc;

use dg_rs::boundary::{
    CharacteristicOBC, HarmonicTide, MultiBoundaryCondition2D, Reflective2D, SWEBoundaryCondition2D,
};
use dg_rs::equations::ShallowWater2D;
use dg_rs::physics::{
    BottomDrag3D, Forcing as Forcing3D, GlsMixing, Hydrostatic3D, LinearEOS, PhysicsBuilder,
};
use dg_rs::simulation::{Simulation, Simulation3D};
use dg_rs::solver::state::Solution3D;
use dg_rs::solver::{SWESolution2D, SWEState2D, StandardLimiter2D, WetDryConfig};
use dg_rs::source::{
    CageDrag2D, CoriolisSource2D, HorizontalViscosity2D, ManningFriction2D, SourceContext2D,
    SourceTerm2D,
};
use dg_rs::time::{ModeSplitIntegrator, MultirateSSPRK3, SSPRK3};

use crate::cloud_3d::Cloud3D;
use crate::particles::{Cloud, ParticleConfig, ParticleSnapshot};
use crate::scenario::{G, Scenario, Tide};

/// Reference density of the 3D model (kg/m³).
const RHO0: f64 = 1025.0;
/// Angular frequency of M2 (1/s).
const M2: f64 = 2.0 * PI / 44_714.16;

/// The tidal surface slope of a periodic channel as a depth-uniform body force along
/// mesh x, `∂(hu)/∂t = h·a(t)`, `a = U ω r(t) cos ωt`, with a smooth ramp `r` over
/// `ramp` seconds: without friction it drives the current `U sin ωt`.
struct TidalForce {
    current: f64,
    ramp: f64,
}

impl TidalForce {
    fn acceleration(&self, t: f64) -> f64 {
        let ramp = if t < self.ramp {
            0.5 * (1.0 - (PI * t / self.ramp).cos())
        } else {
            1.0
        };
        self.current * M2 * ramp * (M2 * t).cos()
    }
}

impl SourceTerm2D for TidalForce {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        SWEState2D::new(0.0, ctx.state.h * self.acceleration(ctx.time), 0.0)
    }

    fn name(&self) -> &'static str {
        "tidal_force"
    }
}

/// Below this depth a node is dry: no velocity, and the viewer hides its surface.
pub const H_DRY: f32 = 0.01;

/// The model state at one instant, node by node in the solver's element-major order.
pub struct Snapshot {
    /// Model time (s)
    pub t: f64,
    /// Surface elevation η = h + B (m)
    pub eta: Vec<f32>,
    /// Depth-averaged velocity, mesh x (m/s)
    pub u: Vec<f32>,
    /// Depth-averaged velocity, mesh y (m/s)
    pub v: Vec<f32>,
    /// The particles released so far
    pub particles: ParticleSnapshot,
    /// The layers of a 3D run
    pub layers: Option<Layers>,
}

/// The 3D fields at one instant, `[node][level]` in the solver's element-major node
/// order, levels from the bed up.
pub struct Layers {
    pub n_levels: usize,
    /// Velocity along the σ-layers, mesh x and y (m/s)
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    /// Temperature (°C)
    pub temp: Vec<f32>,
}

impl Layers {
    fn bytes(&self) -> usize {
        4 * (self.u.len() + self.v.len() + self.temp.len())
    }
}

impl Snapshot {
    fn of(q: &SWESolution2D, bed: &[f64], t: f64, particles: ParticleSnapshot) -> Self {
        let [h, hu, hv] = &q.data;
        let velocity = |m: &[f64]| {
            h.iter()
                .zip(m)
                .map(|(&h, &m)| {
                    if h > H_DRY as f64 {
                        (m / h) as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        Self {
            t,
            eta: h.iter().zip(bed).map(|(h, b)| (h + b) as f32).collect(),
            u: velocity(hu),
            v: velocity(hv),
            particles,
            layers: None,
        }
    }

    /// The surface, the depth-averaged velocity and the layers of a 3D state.
    fn of_3d(state: &Solution3D, t: f64, particles: ParticleSnapshot) -> Self {
        let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        Self {
            t,
            eta: f32s(&state.eta.data),
            u: f32s(&state.ubar.data),
            v: f32s(&state.vbar.data),
            particles,
            layers: Some(Layers {
                n_levels: state.n_levels,
                u: f32s(&state.u),
                v: f32s(&state.v),
                temp: f32s(&state.temp),
            }),
        }
    }

    pub fn bytes(&self) -> usize {
        4 * (self.eta.len() + self.u.len() + self.v.len())
            + self.particles.bytes()
            + self.layers.as_ref().map_or(0, Layers::bytes)
    }
}

pub enum SolverMessage {
    Snapshot(Box<Snapshot>),
    Finished {
        steps: usize,
        wall: f64,
        error: Option<String>,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct SolverConfig {
    /// End of the run (s of model time)
    pub t_end: f64,
    /// Model time between snapshots (s)
    pub interval: f64,
    /// Local time stepping levels (`MultirateSSPRK3`); 0 steps every element globally
    pub levels: usize,
    /// Threads of the solver's rayon pool
    pub threads: usize,
    /// Whether the cages' nets drag on the flow (they are drawn either way)
    pub drag: bool,
    /// Particles released from the cages
    pub particles: ParticleConfig,
}

/// Start the run; snapshots arrive on the returned channel, the first at t = 0.
pub fn spawn(scenario: &Scenario, config: SolverConfig) -> Receiver<SolverMessage> {
    if scenario.three_d.is_some() {
        return spawn_3d(scenario, config);
    }
    let (tx, rx) = channel();
    let mesh = scenario.mesh.clone();
    let ops = scenario.ops.clone();
    let geom = scenario.geom.clone();
    let bathymetry = scenario.bathymetry.clone();
    let cages = scenario.cages.clone();
    let forcing = scenario.forcing.clone();
    let mut q = scenario.at_rest();
    std::thread::Builder::new()
        .name("dg-rs solver".into())
        .spawn(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(config.threads)
                .thread_name(|i| format!("dg-rs rayon {i}"))
                .build()
                .expect("solver thread pool");
            pool.install(|| {
                let wall = Reflective2D::new();
                let ramp = (forcing.ramp > 0.0).then_some(forcing.ramp);
                let body_force = match forcing.tide {
                    Tide::BodyForceM2(current) => current,
                    _ => 0.0,
                };
                let sea: Box<dyn SWEBoundaryCondition2D + Send + Sync> = match forcing.tide {
                    Tide::UniformM2(amplitude) => {
                        let tide = HarmonicTide::m2(amplitude, 0.0);
                        Box::new(CharacteristicOBC::new(match ramp {
                            Some(r) => tide.with_ramp_up(r),
                            None => tide,
                        }))
                    }
                    Tide::Atlas(tides) => Box::new(CharacteristicOBC::new(match ramp {
                        Some(r) => tides.with_ramp_up(r),
                        None => tides,
                    })),
                    // A body force needs no open boundary
                    Tide::BodyForceM2(_) => Box::new(Reflective2D::new()),
                };
                let physics = PhysicsBuilder::swe_2d(
                    mesh.clone(),
                    ops.clone(),
                    geom.clone(),
                    ShallowWater2D::new(G),
                    MultiBoundaryCondition2D::new(&wall).with_open(sea.as_ref()),
                )
                .with_bathymetry(bathymetry.clone())
                .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
                .with_wet_dry(WetDryConfig::default())
                .with_implicit_friction(ManningFriction2D::new(G, forcing.manning))
                .with_cage_drag(CageDrag2D::new(
                    &mesh,
                    &ops,
                    if config.drag { &cages } else { &[] },
                ))
                .with_source(CoriolisSource2D::f_plane(forcing.coriolis))
                .with_source(TidalForce {
                    current: body_force,
                    ramp: forcing.ramp,
                })
                .build();

                let started = Instant::now();
                let mut cloud = (config.particles.per_release > 0)
                    .then(|| Cloud::new(&mesh, &ops, &cages, config.particles));
                // A send fails only once the viewer has closed; the run then ends with the process.
                let mut send = |q: &SWESolution2D, t: f64| {
                    let particles = match cloud.as_mut() {
                        Some(cloud) => {
                            cloud.advance(q, t);
                            cloud.snapshot(q, &bathymetry.data)
                        }
                        None => ParticleSnapshot::default(),
                    };
                    let _ = tx.send(SolverMessage::Snapshot(Box::new(Snapshot::of(
                        q,
                        &bathymetry.data,
                        t,
                        particles,
                    ))));
                };
                send(&q, 0.0);
                let result = if config.levels > 0 {
                    Simulation::new(physics, MultirateSSPRK3::new(config.levels))
                        .with_cfl(1.0)
                        .with_callback_interval(config.interval)
                        .run_with_callback(&mut q, 0.0, config.t_end, &mut send)
                } else {
                    Simulation::new(physics, SSPRK3)
                        .with_cfl(1.0)
                        .with_callback_interval(config.interval)
                        .run_with_callback(&mut q, 0.0, config.t_end, &mut send)
                };
                let _ = tx.send(SolverMessage::Finished {
                    steps: result.n_steps,
                    wall: started.elapsed().as_secs_f64(),
                    error: result.error,
                });
            });
        })
        .expect("spawn the solver thread");
    rx
}

/// [`spawn`] for a scenario with a 3D model: the stratified channel of
/// `examples/farm_3d.rs` (GLS k-ε, log-layer bottom drag, background plus Smagorinsky
/// viscosity, net-cage drag per layer), from rest.
fn spawn_3d(scenario: &Scenario, config: SolverConfig) -> Receiver<SolverMessage> {
    let (tx, rx) = channel();
    let three_d = scenario
        .three_d
        .clone()
        .expect("spawn_3d runs a 3D scenario");
    let mesh = scenario.mesh.clone();
    let ops = scenario.ops.clone();
    let geom = scenario.geom.clone();
    let bathymetry = scenario.bathymetry.clone();
    let cages = scenario.cages.clone();
    let forcing = scenario.forcing.clone();
    std::thread::Builder::new()
        .name("dg-rs solver".into())
        .spawn(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(config.threads)
                .thread_name(|i| format!("dg-rs rayon {i}"))
                .build()
                .expect("solver thread pool");
            pool.install(|| {
                let current = match forcing.tide {
                    Tide::BodyForceM2(current) => current,
                    _ => 0.0,
                };
                // The 2D module owns the depth mean's forcing, Coriolis and viscosity;
                // friction and the cages are the 3D model's
                let swe = PhysicsBuilder::swe_2d(
                    mesh.clone(),
                    ops.clone(),
                    geom.clone(),
                    ShallowWater2D::new(G),
                    Reflective2D::new(),
                )
                .with_bathymetry(bathymetry.clone())
                .with_source(TidalForce {
                    current,
                    ramp: forcing.ramp,
                })
                .with_source(CoriolisSource2D::f_plane(forcing.coriolis))
                .with_viscosity(HorizontalViscosity2D::constant(three_d.viscosity))
                .build();
                let eos = LinearEOS::default();
                let z0 = three_d.roughness;
                let physics = Hydrostatic3D::new(
                    mesh.clone(),
                    ops.clone(),
                    geom.clone(),
                    three_d.sigma.clone(),
                    bathymetry.clone(),
                    Arc::new(CoriolisSource2D::f_plane(forcing.coriolis)),
                    eos,
                    GlsMixing::k_epsilon().with_roughness(0.02, z0),
                    swe,
                    Forcing3D {
                        surface_stress: [0.0, 0.0],
                        bottom_stress: [0.0, 0.0],
                        surface_buoyancy_flux: 0.0,
                    },
                    G,
                    RHO0,
                )
                .with_bottom_drag(BottomDrag3D::log_layer(z0))
                .with_horizontal_viscosity(three_d.viscosity)
                .with_smagorinsky_viscosity(three_d.smagorinsky);
                let physics = if config.drag {
                    physics.with_cage_drag(CageDrag2D::new(&mesh, &ops, &cages))
                } else {
                    physics
                };

                let n_levels = three_d.sigma.n_levels();
                let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, n_levels);
                state.salt.fill(three_d.salinity);
                for (k, column) in state.temp.chunks_exact_mut(n_levels).enumerate() {
                    let depth = -bathymetry.data[k];
                    for (t, &s) in column.iter_mut().zip(three_d.sigma.sigma_rho()) {
                        *t = (three_d.temperature)(s * depth);
                    }
                }
                physics.update_density(&mut state);

                let started = Instant::now();
                let mut cloud = (config.particles.per_release > 0).then(|| {
                    Cloud3D::new(
                        &mesh,
                        &ops,
                        &three_d.sigma,
                        &bathymetry,
                        &cages,
                        three_d.light,
                        config.particles,
                    )
                });
                let mut send = |s: &Solution3D, t: f64| {
                    let particles = match cloud.as_mut() {
                        Some(cloud) => {
                            cloud.advance(s, t);
                            cloud.snapshot(s)
                        }
                        None => ParticleSnapshot::default(),
                    };
                    let _ = tx.send(SolverMessage::Snapshot(Box::new(Snapshot::of_3d(
                        s, t, particles,
                    ))));
                };
                let result = Simulation3D::new(physics, ModeSplitIntegrator::new())
                    .with_cfl(0.5)
                    .with_callback_interval(config.interval)
                    .run_with_callback(&mut state, 0.0, config.t_end, &mut send);
                let _ = tx.send(SolverMessage::Finished {
                    steps: result.n_steps,
                    wall: started.elapsed().as_secs_f64(),
                    error: result.error,
                });
            });
        })
        .expect("spawn the solver thread");
    rx
}
