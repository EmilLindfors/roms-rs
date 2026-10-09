//! Runtime diagnostics and progress tracking.
//!
//! - [`SWEDiagnostics2D`]: Conservation diagnostics for 2D SWE
//! - [`PotentialEnergy3D`]: Potential and reference potential energy of a
//!   3D state (spurious mixing)
//! - [`DiagnosticsTracker`]: Time series tracking
//! - [`ProgressReporter`]: Progress output utilities

mod potential_energy_3d;
mod runtime;

pub use potential_energy_3d::{PotentialEnergy, PotentialEnergy3D};
pub use runtime::{
    DiagnosticsTracker, ProgressReporter, SWEDiagnostics2D, current_cfl_2d, total_energy_2d,
    total_mass_2d, total_momentum_2d,
};
