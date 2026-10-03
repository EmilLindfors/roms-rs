//! Spectral wind waves: a phase-averaged wave model on the DG mesh.
//!
//! The sea state is the wave action density `N(x, y, σ, θ, t)`, one nodal DG field
//! per spectral component (frequency × direction), evolved by the action balance
//! (Komen et al. 1994, "Dynamics and Modelling of Ocean Waves"; Holthuijsen 2007,
//! "Waves in Oceanic and Coastal Waters"; the SWAN Scientific Documentation):
//!
//! - [`dispersion`]: the linear dispersion relation, group velocity, `∂σ/∂d`;
//! - [`spectrum`]: the spectral grid, JONSWAP spectra and the integrated parameters
//!   (H_s, periods, direction, spread), Stokes drift, radiation stress and the bed
//!   orbital velocity that the circulation, mixing and particles need;
//! - [`sources`]: wind input, whitecapping, bottom friction, depth-induced breaking;
//! - [`model`]: [`WaveModel2D`], the geographic DG propagation with refraction and
//!   current-induced frequency shifting, and the step that combines them.
//!
//! First stage of TODO "Waves" (P7): the four-wave (quadruplet) interactions, the
//! coupling to the circulation and the boundary spectra of a parent wave model
//! follow.

pub mod dispersion;
pub mod model;
pub mod sources;
pub mod spectrum;
pub mod state;

pub use dispersion::{dsigma_ddepth, group_velocity, group_velocity_ratio, wavenumber};
pub use model::{DEFAULT_DEPTH_MIN, WaveModel2D, WaveWorkspace};
pub use sources::{SourceTerms, Wind, breaking_fraction};
pub use spectrum::{SpectralGrid, WaveParameters, wrap_angle};
pub use state::WaveSolution;
