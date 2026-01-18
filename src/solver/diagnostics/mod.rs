//! Runtime diagnostics and progress tracking.
//!
//! - [`SWEDiagnostics2D`]: Conservation diagnostics for 2D SWE
//! - [`DiagnosticsTracker`]: Time series tracking
//! - [`ProgressReporter`]: Progress output utilities

mod diagnostics;

pub use diagnostics::{
    DiagnosticsTracker, ProgressReporter, SWEDiagnostics2D, current_cfl_2d, total_energy_2d,
    total_mass_2d, total_momentum_2d,
};
