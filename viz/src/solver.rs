//! The solver, run on a thread of its own while the viewer draws.
//!
//! [`spawn`] builds the scenario's physics and runs `Simulation::run_with_callback`
//! with a callback every `interval` seconds of model time. Each callback reduces the
//! state to what the viewer draws, a [`Snapshot`] of the surface η and the
//! depth-averaged velocity (u, v) at every node, plus the particles released from the
//! cages ([`crate::particles::Cloud`], tracked here between snapshots), and sends it
//! down a channel. The
//! solver's own rayon pool leaves a couple of cores to Bevy.

use std::sync::mpsc::{Receiver, channel};
use std::time::Instant;

use dg_rs::boundary::{
    CharacteristicOBC, HarmonicTide, MultiBoundaryCondition2D, Reflective2D, SWEBoundaryCondition2D,
};
use dg_rs::equations::ShallowWater2D;
use dg_rs::physics::PhysicsBuilder;
use dg_rs::simulation::Simulation;
use dg_rs::solver::{SWESolution2D, StandardLimiter2D, WetDryConfig};
use dg_rs::source::{CageDrag2D, CoriolisSource2D, ManningFriction2D};
use dg_rs::time::{MultirateSSPRK3, SSPRK3};

use crate::particles::{Cloud, ParticleConfig, ParticleSnapshot};
use crate::scenario::{G, Scenario, Tide};

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
        }
    }

    pub fn bytes(&self) -> usize {
        4 * (self.eta.len() + self.u.len() + self.v.len()) + self.particles.bytes()
    }
}

pub enum SolverMessage {
    Snapshot(Snapshot),
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
                    let _ = tx.send(SolverMessage::Snapshot(Snapshot::of(
                        q,
                        &bathymetry.data,
                        t,
                        particles,
                    )));
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
