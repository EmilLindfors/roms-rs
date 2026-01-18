//! Standard limiter implementations and composition utilities.

use super::super::state::SWESolution2D;
use super::swe_2d::{swe_kuzmin_limiter_2d, swe_positivity_limiter_2d, swe_tvb_limiter_2d};
use super::tracer_2d::{KuzminParameter2D, TVBParameter2D};
use super::traits::{BoxedLimiter2D, Limiter2D, LimiterContext2D};

/// TVB (Total Variation Bounded) slope limiter.
///
/// Limits slopes to prevent oscillations while maintaining high-order
/// accuracy in smooth regions through the M parameter.
#[derive(Clone, Debug)]
pub struct TVBLimiter2D {
    param: TVBParameter2D,
}

impl TVBLimiter2D {
    /// Create a new TVB limiter with the given parameters.
    pub fn new(param: TVBParameter2D) -> Self {
        Self { param }
    }

    /// Create a TVB limiter with the given M parameter and reference length.
    ///
    /// # Arguments
    /// * `m` - TVB parameter (typically 1-100; larger = less limiting)
    /// * `l_ref` - Reference length scale (e.g., domain size or h_min)
    pub fn with_params(m: f64, l_ref: f64) -> Self {
        Self {
            param: TVBParameter2D::new(m, l_ref),
        }
    }
}

impl Limiter2D for TVBLimiter2D {
    fn apply(&self, solution: &mut SWESolution2D, ctx: &LimiterContext2D) {
        swe_tvb_limiter_2d(solution, ctx.mesh, ctx.ops, &self.param);
    }

    fn name(&self) -> &'static str {
        "tvb"
    }
}

/// Kuzmin vertex-based slope limiter.
///
/// Uses vertex patches to compute local bounds, providing
/// more accurate limiting for unstructured meshes.
#[derive(Clone, Debug)]
pub struct KuzminLimiter2D {
    param: KuzminParameter2D,
}

impl KuzminLimiter2D {
    /// Create a new Kuzmin limiter with the given parameters.
    pub fn new(param: KuzminParameter2D) -> Self {
        Self { param }
    }

    /// Create a Kuzmin limiter with strict bounds.
    pub fn strict() -> Self {
        Self {
            param: KuzminParameter2D::strict(),
        }
    }

    /// Create a Kuzmin limiter with relaxed bounds.
    pub fn relaxed(relaxation: f64) -> Self {
        Self {
            param: KuzminParameter2D::relaxed(relaxation),
        }
    }
}

impl Limiter2D for KuzminLimiter2D {
    fn apply(&self, solution: &mut SWESolution2D, ctx: &LimiterContext2D) {
        swe_kuzmin_limiter_2d(solution, ctx.mesh, ctx.ops, &self.param);
    }

    fn name(&self) -> &'static str {
        "kuzmin"
    }
}

/// Positivity-preserving limiter.
///
/// Ensures water depth remains non-negative, which is essential
/// for physical validity and numerical stability.
#[derive(Clone, Debug)]
pub struct PositivityLimiter2D {
    h_min: f64,
}

impl PositivityLimiter2D {
    /// Create a new positivity limiter.
    ///
    /// # Arguments
    /// * `h_min` - Minimum allowed depth (typically 1e-6 to 1e-4)
    pub fn new(h_min: f64) -> Self {
        Self { h_min }
    }
}

impl Limiter2D for PositivityLimiter2D {
    fn apply(&self, solution: &mut SWESolution2D, ctx: &LimiterContext2D) {
        swe_positivity_limiter_2d(solution, ctx.ops, self.h_min);
    }

    fn name(&self) -> &'static str {
        "positivity"
    }

    fn preserves_positivity(&self) -> bool {
        true
    }
}

/// No-op limiter (does nothing).
///
/// Useful as a placeholder or for testing.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoLimiter2D;

impl Limiter2D for NoLimiter2D {
    fn apply(&self, _solution: &mut SWESolution2D, _ctx: &LimiterContext2D) {
        // No-op
    }

    fn name(&self) -> &'static str {
        "none"
    }
}

/// Chain of limiters applied in sequence.
///
/// Allows composing multiple limiters, e.g., slope limiting followed
/// by positivity preservation.
pub struct LimiterChain2D {
    limiters: Vec<Box<dyn Limiter2D>>,
}

impl LimiterChain2D {
    /// Create an empty limiter chain.
    pub fn new() -> Self {
        Self {
            limiters: Vec::new(),
        }
    }

    /// Add a limiter to the chain.
    ///
    /// Limiters are applied in the order they are added.
    pub fn then<L: Limiter2D + 'static>(mut self, limiter: L) -> Self {
        self.limiters.push(Box::new(limiter));
        self
    }

    /// Add a boxed limiter to the chain.
    pub fn then_boxed(mut self, limiter: Box<dyn Limiter2D>) -> Self {
        self.limiters.push(limiter);
        self
    }

    /// Returns true if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.limiters.is_empty()
    }

    /// Returns the number of limiters in the chain.
    pub fn len(&self) -> usize {
        self.limiters.len()
    }
}

impl Default for LimiterChain2D {
    fn default() -> Self {
        Self::new()
    }
}

impl Limiter2D for LimiterChain2D {
    fn apply(&self, solution: &mut SWESolution2D, ctx: &LimiterContext2D) {
        for limiter in &self.limiters {
            limiter.apply(solution, ctx);
        }
    }

    fn name(&self) -> &'static str {
        "chain"
    }

    fn preserves_positivity(&self) -> bool {
        self.limiters.iter().any(|l| l.preserves_positivity())
    }

    fn preserves_cell_average(&self) -> bool {
        self.limiters.iter().all(|l| l.preserves_cell_average())
    }
}

/// Enum wrapper for built-in limiter types.
///
/// Provides zero-cost dispatch when the limiter type is known at compile time.
#[derive(Clone, Debug, Default)]
pub enum StandardLimiter2D {
    /// No limiting
    #[default]
    None,
    /// TVB slope limiter
    Tvb(TVBParameter2D),
    /// Kuzmin vertex-based limiter
    Kuzmin(KuzminParameter2D),
    /// Positivity limiter only
    Positivity(f64),
    /// TVB + positivity (common combination)
    TvbWithPositivity { tvb: TVBParameter2D, h_min: f64 },
    /// Kuzmin + positivity (common combination)
    KuzminWithPositivity {
        kuzmin: KuzminParameter2D,
        h_min: f64,
    },
}

impl Limiter2D for StandardLimiter2D {
    fn apply(&self, solution: &mut SWESolution2D, ctx: &LimiterContext2D) {
        match self {
            StandardLimiter2D::None => {}
            StandardLimiter2D::Tvb(param) => {
                swe_tvb_limiter_2d(solution, ctx.mesh, ctx.ops, param);
            }
            StandardLimiter2D::Kuzmin(param) => {
                swe_kuzmin_limiter_2d(solution, ctx.mesh, ctx.ops, param);
            }
            StandardLimiter2D::Positivity(h_min) => {
                swe_positivity_limiter_2d(solution, ctx.ops, *h_min);
            }
            StandardLimiter2D::TvbWithPositivity { tvb, h_min } => {
                swe_tvb_limiter_2d(solution, ctx.mesh, ctx.ops, tvb);
                swe_positivity_limiter_2d(solution, ctx.ops, *h_min);
            }
            StandardLimiter2D::KuzminWithPositivity { kuzmin, h_min } => {
                swe_kuzmin_limiter_2d(solution, ctx.mesh, ctx.ops, kuzmin);
                swe_positivity_limiter_2d(solution, ctx.ops, *h_min);
            }
        }
    }

    fn name(&self) -> &'static str {
        match self {
            StandardLimiter2D::None => "none",
            StandardLimiter2D::Tvb(_) => "tvb",
            StandardLimiter2D::Kuzmin(_) => "kuzmin",
            StandardLimiter2D::Positivity(_) => "positivity",
            StandardLimiter2D::TvbWithPositivity { .. } => "tvb+positivity",
            StandardLimiter2D::KuzminWithPositivity { .. } => "kuzmin+positivity",
        }
    }

    fn preserves_positivity(&self) -> bool {
        matches!(
            self,
            StandardLimiter2D::Positivity(_)
                | StandardLimiter2D::TvbWithPositivity { .. }
                | StandardLimiter2D::KuzminWithPositivity { .. }
        )
    }
}

/// Create a boxed limiter from a standard limiter type.
pub fn create_limiter(limiter: StandardLimiter2D) -> BoxedLimiter2D {
    match limiter {
        StandardLimiter2D::None => Box::new(NoLimiter2D),
        StandardLimiter2D::Tvb(param) => Box::new(TVBLimiter2D::new(param)),
        StandardLimiter2D::Kuzmin(param) => Box::new(KuzminLimiter2D::new(param)),
        StandardLimiter2D::Positivity(h_min) => Box::new(PositivityLimiter2D::new(h_min)),
        StandardLimiter2D::TvbWithPositivity { tvb, h_min } => Box::new(
            LimiterChain2D::new()
                .then(TVBLimiter2D::new(tvb))
                .then(PositivityLimiter2D::new(h_min)),
        ),
        StandardLimiter2D::KuzminWithPositivity { kuzmin, h_min } => Box::new(
            LimiterChain2D::new()
                .then(KuzminLimiter2D::new(kuzmin))
                .then(PositivityLimiter2D::new(h_min)),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh2D;
    use crate::operators::DGOperators2D;
    use crate::solver::state::SWEState2D;
    use crate::types::ElementIndex;

    fn k(idx: usize) -> ElementIndex {
        ElementIndex::new(idx)
    }

    fn create_test_context<'a>(mesh: &'a Mesh2D, ops: &'a DGOperators2D) -> LimiterContext2D<'a> {
        LimiterContext2D::new(mesh, ops)
    }

    #[test]
    fn test_limiter_names() {
        assert_eq!(TVBLimiter2D::with_params(10.0, 0.1).name(), "tvb");
        assert_eq!(KuzminLimiter2D::strict().name(), "kuzmin");
        assert_eq!(PositivityLimiter2D::new(1e-6).name(), "positivity");
        assert_eq!(NoLimiter2D.name(), "none");
    }

    #[test]
    fn test_limiter_chain() {
        let chain = LimiterChain2D::new()
            .then(TVBLimiter2D::with_params(10.0, 0.1))
            .then(PositivityLimiter2D::new(1e-6));

        assert_eq!(chain.len(), 2);
        assert!(!chain.is_empty());
        assert_eq!(chain.name(), "chain");
        assert!(chain.preserves_positivity());
        assert!(chain.preserves_cell_average());
    }

    #[test]
    fn test_standard_limiter_enum() {
        let none = StandardLimiter2D::None;
        assert_eq!(none.name(), "none");
        assert!(!none.preserves_positivity());

        let pos = StandardLimiter2D::Positivity(1e-6);
        assert_eq!(pos.name(), "positivity");
        assert!(pos.preserves_positivity());

        let tvb_pos = StandardLimiter2D::TvbWithPositivity {
            tvb: TVBParameter2D::new(10.0, 0.1),
            h_min: 1e-6,
        };
        assert_eq!(tvb_pos.name(), "tvb+positivity");
        assert!(tvb_pos.preserves_positivity());
    }

    #[test]
    fn test_no_limiter_is_noop() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);
        let ops = DGOperators2D::new(2);
        let mut swe = SWESolution2D::new(mesh.n_elements, ops.n_nodes);

        // Set some values using set_state
        let test_state = SWEState2D::new(1.0, 0.5, 0.0);
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                swe.set_state(k(ki), i, test_state);
            }
        }

        let ctx = create_test_context(&mesh, &ops);
        let limiter = NoLimiter2D;

        // Apply no-op limiter
        limiter.apply(&mut swe, &ctx);

        // Values should be unchanged
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let state = swe.get_state(k(ki), i);
                assert!((state.h - 1.0).abs() < 1e-14);
                assert!((state.hu - 0.5).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_create_boxed_limiter() {
        let boxed = create_limiter(StandardLimiter2D::None);
        assert_eq!(boxed.name(), "none");

        let boxed = create_limiter(StandardLimiter2D::TvbWithPositivity {
            tvb: TVBParameter2D::new(10.0, 0.1),
            h_min: 1e-6,
        });
        assert_eq!(boxed.name(), "chain");
        assert!(boxed.preserves_positivity());
    }
}
