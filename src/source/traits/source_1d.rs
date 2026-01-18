//! 1D source term trait and composition.

use crate::solver::SWEState;

/// Trait for source terms in shallow water equations.
///
/// Source terms modify the RHS of the equations:
/// dq/dt = -∂F/∂x + S(q, x, t)
pub trait SourceTerm: Send + Sync {
    /// Evaluate the source term contribution at a single node.
    ///
    /// # Arguments
    /// * `state` - Current state (h, hu)
    /// * `db_dx` - Local bathymetry gradient
    /// * `position` - Physical position x
    /// * `time` - Current time
    ///
    /// # Returns
    /// Source contribution as SWEState (S_h, S_hu)
    fn evaluate(&self, state: &SWEState, db_dx: f64, position: f64, time: f64) -> SWEState;

    /// Name of this source term for debugging.
    fn name(&self) -> &'static str;
}

/// Combine multiple source terms.
pub struct CombinedSource<'a> {
    sources: Vec<&'a dyn SourceTerm>,
}

impl<'a> CombinedSource<'a> {
    /// Create a new combined source from a list of source terms.
    pub fn new(sources: Vec<&'a dyn SourceTerm>) -> Self {
        Self { sources }
    }
}

impl<'a> SourceTerm for CombinedSource<'a> {
    fn evaluate(&self, state: &SWEState, db_dx: f64, position: f64, time: f64) -> SWEState {
        let mut total = SWEState::zero();
        for source in &self.sources {
            let contrib = source.evaluate(state, db_dx, position, time);
            total = total + contrib;
        }
        total
    }

    fn name(&self) -> &'static str {
        "combined"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ZeroSource;

    impl SourceTerm for ZeroSource {
        fn evaluate(&self, _: &SWEState, _: f64, _: f64, _: f64) -> SWEState {
            SWEState::zero()
        }

        fn name(&self) -> &'static str {
            "zero"
        }
    }

    struct ConstantSource {
        value: SWEState,
    }

    impl SourceTerm for ConstantSource {
        fn evaluate(&self, _: &SWEState, _: f64, _: f64, _: f64) -> SWEState {
            self.value
        }

        fn name(&self) -> &'static str {
            "constant"
        }
    }

    #[test]
    fn test_combined_source() {
        let s1 = ConstantSource {
            value: SWEState::new(1.0, 2.0),
        };
        let s2 = ConstantSource {
            value: SWEState::new(0.5, 1.0),
        };

        let combined = CombinedSource::new(vec![&s1, &s2]);

        let result = combined.evaluate(&SWEState::zero(), 0.0, 0.0, 0.0);

        assert!((result.h - 1.5).abs() < 1e-14);
        assert!((result.hu - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_zero_source() {
        let zero = ZeroSource;
        let result = zero.evaluate(&SWEState::new(1.0, 2.0), 0.5, 3.0, 1.0);

        assert!(result.h.abs() < 1e-14);
        assert!(result.hu.abs() < 1e-14);
    }
}
