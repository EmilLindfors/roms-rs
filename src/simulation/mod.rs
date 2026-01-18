//! Simulation runner abstraction.
//!
//! This module provides a high-level interface for running DG simulations
//! that ties together:
//! - Physics modules (RHS computation, dt calculation)
//! - Time integrators (SSP-RK3, etc.)
//! - Diagnostics and callbacks
//!
//! # Example
//! ```ignore
//! use dg_rs::simulation::Simulation;
//! use dg_rs::physics::PhysicsBuilder;
//! use dg_rs::time::SSPRK3;
//!
//! let physics = PhysicsBuilder::swe_2d(mesh, ops, geom, equation, bc).build();
//! let integrator = SSPRK3;
//!
//! let result = Simulation::new(physics, integrator)
//!     .with_cfl(0.5)
//!     .with_callback(|state, time| println!("t = {:.2}", time))
//!     .run(&mut state, 0.0, 100.0);
//! ```

mod runner;

pub use runner::{Simulation, SimulationConfig, SimulationResult};
