//! Lagrangian particle tracking in the depth-averaged flow (TODO F.2), and
//! in the 3D flow with a σ-level per particle ([`tracker_3d`]), where
//! particles can swim by a behaviour ([`behaviour`]: salmon-lice larvae).
//!
//! Particles (sea-lice larvae, feed, faeces, drifters) move with a velocity
//! field sampled from the DG solution at the particle itself: the element
//! polynomial evaluated at the particle's reference coordinates, not the
//! nearest node ([`ParticleVelocity2D`]).
//!
//! # Advection
//!
//! A step from `t` to `t + Δt` integrates `dx/dt = u(x, t)` with the classical
//! fourth-order Runge–Kutta method:
//!
//! ```text
//! k₁ = u(xⁿ, t),              k₂ = u(xⁿ + ½Δt k₁, t + ½Δt),
//! k₃ = u(xⁿ + ½Δt k₂, t + ½Δt),   k₄ = u(xⁿ + Δt k₃, t + Δt),
//! xⁿ⁺¹ = xⁿ + Δt/6 (k₁ + 2k₂ + 2k₃ + k₄) + √(2KΔt) ξ,
//! ```
//!
//! so a field that the elements represent exactly (e.g. solid-body rotation,
//! linear in x and y, on bilinear elements of any order) is followed to the
//! RK4 error, O(Δt⁴) over a fixed time. Between two snapshots
//! ([`SWEVelocity2D::between`]) the velocity is linear in time, the usual
//! offline practice (OpenDrift, LADiM); the error of that interpolation is
//! then the output interval's, not the step's.
//!
//! # Dispersion
//!
//! With a horizontal diffusivity `K`, each step adds an independent Gaussian
//! displacement `√(2KΔt) ξ`, `ξ ~ N(0, I₂)` (the Itô form of
//! `∂c/∂t = ∇·(K∇c)` for constant `K`). Reflection at walls keeps a uniform
//! areal distribution uniform (Visser 1997's well-mixed condition in 2D).
//! This is the dispersion of a surface or depth-uniform concentration; a
//! depth-integrated tracer (`∂(hc)/∂t = ∇·(hK∇c)`) also needs the drift
//! `K∇h/h` (Dimou & Adams 1993), which is not implemented, nor is a
//! variable `K` (drift `∇K`).
//!
//! Each particle draws its random numbers from its own SplitMix64 stream,
//! seeded by the tracker's seed and the particle's id, so a run is
//! reproducible whatever the thread count or particle order.
//!
//! # Locating particles and boundaries
//!
//! A particle keeps its element and reference coordinates. Each move (the
//! RK stage points and the step itself) walks along the straight segment
//! from the particle's position, face by face ([`walk()`]), so tracking costs
//! O(faces crossed) per move and never searches the whole mesh. At a
//! boundary face the segment is mirrored in the face (walls, specular
//! reflection; no particle crosses a wall) or the particle leaves the
//! domain at the crossing ([`ParticleStatus::Exited`], open boundaries).
//! Periodic faces carry the particle across. With a stranding depth, a
//! particle on water shallower than it is held in place
//! ([`ParticleStatus::Stranded`], a drying tidal flat) until the water
//! there is deep enough again.
//!
//! # References
//!
//! - Visser, A. W. (1997). Using random walk models to simulate the vertical
//!   distribution of particles in a turbulent water column. *Mar. Ecol.
//!   Prog. Ser.* 158, 275–281.
//! - Dimou, K. N. & Adams, E. E. (1993). A random-walk, particle tracking
//!   model for well-mixed estuaries and coastal waters. *Estuar. Coast.
//!   Shelf Sci.* 37, 99–110.

pub mod behaviour;
mod tracker;
pub mod tracker_3d;
mod velocity;
pub mod velocity_3d;
pub mod walk;

pub use behaviour::{
    ClearSkyLight, ConstantLight, LiceStage, ParticleBehaviour3D, Passive, SalmonLice,
    SurfaceLight, Surroundings,
};
pub use tracker::{Particle2D, ParticleStatus, ParticleTracker2D};
pub use tracker_3d::{Particle3D, ParticleTracker3D};
pub use velocity::{NodalVelocity2D, ParticleVelocity2D, SWEVelocity2D};
pub use velocity_3d::{ParticleVelocity3D, Solution3DVelocity};
pub use walk::{WalkEnd, walk};
