//! Error types for Burn GPU operations.

use thiserror::Error;

/// Errors that can occur during GPU operations.
#[derive(Error, Debug)]
pub enum BurnError {
    /// Dimension mismatch between tensors.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    /// Backend initialization failed.
    #[error("Backend initialization failed: {0}")]
    BackendInit(String),

    /// Data transfer failed.
    #[error("Data transfer failed: {0}")]
    DataTransfer(String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Numerical error (NaN or Inf detected).
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

impl BurnError {
    /// Create a dimension mismatch error.
    pub fn dimension_mismatch(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::DimensionMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }
}
