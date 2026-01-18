//! 2D Source terms for shallow water equations.
//!
//! Source terms represent forces not part of the hyperbolic flux:
//! - Coriolis: S = (0, f·hv, -f·hu)
//! - Tidal potential: S = (0, -gh ∂Φ/∂x, -gh ∂Φ/∂y)
//! - Bottom friction (Manning): S = (0, -gn²|u|u/h^{1/3}, -gn²|u|v/h^{1/3})
//! - Sponge layer damping: S = γ(q_ref - q)
//!
//! Source terms are evaluated at each quadrature node and added to the RHS.

use crate::solver::SWEState2D;

/// Context for 2D source term evaluation.
///
/// Provides all information needed to evaluate a source term at a single node.
#[derive(Clone, Copy, Debug)]
pub struct SourceContext2D {
    /// Current simulation time
    pub time: f64,
    /// Physical position (x, y)
    pub position: (f64, f64),
    /// Current state (h, hu, hv)
    pub state: SWEState2D,
    /// Bathymetry (bottom elevation) at this point
    pub bathymetry: f64,
    /// Bathymetry gradients (∂B/∂x, ∂B/∂y)
    pub bathymetry_gradient: (f64, f64),
    /// Gravitational acceleration
    pub g: f64,
    /// Minimum depth threshold for wet/dry
    pub h_min: f64,
}

impl SourceContext2D {
    /// Create a new source context.
    pub fn new(
        time: f64,
        position: (f64, f64),
        state: SWEState2D,
        bathymetry: f64,
        bathymetry_gradient: (f64, f64),
        g: f64,
        h_min: f64,
    ) -> Self {
        Self {
            time,
            position,
            state,
            bathymetry,
            bathymetry_gradient,
            g,
            h_min,
        }
    }

    /// Get velocity (u, v) with desingularization for dry cells.
    pub fn velocity(&self) -> (f64, f64) {
        if self.state.h < self.h_min {
            (0.0, 0.0)
        } else {
            (self.state.hu / self.state.h, self.state.hv / self.state.h)
        }
    }

    /// Get water surface elevation (η = h + B).
    pub fn surface_elevation(&self) -> f64 {
        self.state.h + self.bathymetry
    }
}

/// Trait for 2D source terms in shallow water equations.
///
/// Source terms modify the RHS of the equations:
/// dq/dt = -∇·F + S(q, x, y, t)
///
/// Implementations must be thread-safe (`Send + Sync`) for parallel computation.
pub trait SourceTerm2D: Send + Sync {
    /// Evaluate the source term contribution at a single node.
    ///
    /// # Arguments
    /// * `ctx` - Context containing state, position, time, and bathymetry
    ///
    /// # Returns
    /// Source contribution as SWEState2D (S_h, S_hu, S_hv)
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D;

    /// Name of this source term for debugging and logging.
    fn name(&self) -> &'static str;

    /// Whether this source term requires special treatment (e.g., implicit).
    ///
    /// Stiff source terms (e.g., strong friction, large sponge damping)
    /// may require implicit or semi-implicit time integration.
    fn is_stiff(&self) -> bool {
        false
    }
}

/// Combine multiple 2D source terms into one.
///
/// The combined source evaluates all constituent sources and sums their contributions.
///
/// # Example
/// ```ignore
/// let coriolis = CoriolisSource2D::f_plane(1.2e-4);
/// let friction = ManningFriction2D::new(0.03, 9.81);
/// let combined = CombinedSource2D::new(vec![&coriolis, &friction]);
/// ```
pub struct CombinedSource2D<'a> {
    sources: Vec<&'a dyn SourceTerm2D>,
}

impl<'a> CombinedSource2D<'a> {
    /// Create a new combined source from a list of source terms.
    pub fn new(sources: Vec<&'a dyn SourceTerm2D>) -> Self {
        Self { sources }
    }

    /// Add a source term to the combination.
    pub fn add(&mut self, source: &'a dyn SourceTerm2D) {
        self.sources.push(source);
    }

    /// Number of source terms in the combination.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Whether the combination is empty.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}

impl<'a> SourceTerm2D for CombinedSource2D<'a> {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        let mut total = SWEState2D::zero();
        for source in &self.sources {
            total = total + source.evaluate(ctx);
        }
        total
    }

    fn name(&self) -> &'static str {
        "combined_2d"
    }

    fn is_stiff(&self) -> bool {
        self.sources.iter().any(|s| s.is_stiff())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ZeroSource2D;

    impl SourceTerm2D for ZeroSource2D {
        fn evaluate(&self, _ctx: &SourceContext2D) -> SWEState2D {
            SWEState2D::zero()
        }

        fn name(&self) -> &'static str {
            "zero_2d"
        }
    }

    struct ConstantSource2D {
        value: SWEState2D,
    }

    impl SourceTerm2D for ConstantSource2D {
        fn evaluate(&self, _ctx: &SourceContext2D) -> SWEState2D {
            self.value
        }

        fn name(&self) -> &'static str {
            "constant_2d"
        }
    }

    fn make_test_context() -> SourceContext2D {
        SourceContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(1.0, 0.5, 0.3),
            0.0,
            (0.0, 0.0),
            9.81,
            1e-6,
        )
    }

    #[test]
    fn test_zero_source_2d() {
        let source = ZeroSource2D;
        let ctx = make_test_context();
        let result = source.evaluate(&ctx);

        assert!(result.h.abs() < 1e-14);
        assert!(result.hu.abs() < 1e-14);
        assert!(result.hv.abs() < 1e-14);
    }

    #[test]
    fn test_constant_source_2d() {
        let source = ConstantSource2D {
            value: SWEState2D::new(1.0, 2.0, 3.0),
        };
        let ctx = make_test_context();
        let result = source.evaluate(&ctx);

        assert!((result.h - 1.0).abs() < 1e-14);
        assert!((result.hu - 2.0).abs() < 1e-14);
        assert!((result.hv - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_combined_source_2d() {
        let s1 = ConstantSource2D {
            value: SWEState2D::new(1.0, 2.0, 3.0),
        };
        let s2 = ConstantSource2D {
            value: SWEState2D::new(0.5, 1.0, 1.5),
        };

        let combined = CombinedSource2D::new(vec![&s1, &s2]);
        let ctx = make_test_context();
        let result = combined.evaluate(&ctx);

        assert!((result.h - 1.5).abs() < 1e-14);
        assert!((result.hu - 3.0).abs() < 1e-14);
        assert!((result.hv - 4.5).abs() < 1e-14);
    }

    #[test]
    fn test_combined_source_empty() {
        let combined: CombinedSource2D = CombinedSource2D::new(vec![]);
        let ctx = make_test_context();
        let result = combined.evaluate(&ctx);

        assert!(result.h.abs() < 1e-14);
        assert!(result.hu.abs() < 1e-14);
        assert!(result.hv.abs() < 1e-14);
        assert!(combined.is_empty());
    }

    #[test]
    fn test_source_context_velocity() {
        let ctx = SourceContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(2.0, 4.0, 6.0), // h=2, hu=4, hv=6 => u=2, v=3
            0.0,
            (0.0, 0.0),
            9.81,
            1e-6,
        );

        let (u, v) = ctx.velocity();
        assert!((u - 2.0).abs() < 1e-14);
        assert!((v - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_source_context_dry_velocity() {
        let ctx = SourceContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(1e-10, 0.1, 0.2), // Very shallow - should return zero velocity
            0.0,
            (0.0, 0.0),
            9.81,
            1e-6,
        );

        let (u, v) = ctx.velocity();
        assert!(u.abs() < 1e-14);
        assert!(v.abs() < 1e-14);
    }

    #[test]
    fn test_source_context_surface_elevation() {
        let ctx = SourceContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(5.0, 0.0, 0.0),
            -10.0, // Bottom at -10m
            (0.0, 0.0),
            9.81,
            1e-6,
        );

        let eta = ctx.surface_elevation();
        assert!((eta - (-5.0)).abs() < 1e-14); // Surface at h + B = 5 + (-10) = -5
    }

    #[test]
    fn test_combined_is_stiff() {
        struct StiffSource;
        impl SourceTerm2D for StiffSource {
            fn evaluate(&self, _: &SourceContext2D) -> SWEState2D {
                SWEState2D::zero()
            }
            fn name(&self) -> &'static str {
                "stiff"
            }
            fn is_stiff(&self) -> bool {
                true
            }
        }

        let normal = ZeroSource2D;
        let stiff = StiffSource;

        let combined_normal = CombinedSource2D::new(vec![&normal]);
        assert!(!combined_normal.is_stiff());

        let combined_stiff = CombinedSource2D::new(vec![&normal, &stiff]);
        assert!(combined_stiff.is_stiff());
    }
}
