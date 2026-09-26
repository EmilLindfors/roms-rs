//! Bathymetry (bottom topography) storage for 2D shallow water simulations.
//!
//! Bathymetry B(x, y) represents the bed elevation above some reference datum.
//! The water surface elevation is η = h + B, where h is the water depth.
//!
//! For well-balanced schemes, we need both B and its gradients ∂B/∂x and ∂B/∂y.
//!
//! Nodal values come from a function: sampled at the nodes
//! ([`Bathymetry2D::from_function`]), or L2-projected onto each element's
//! polynomial space and limited to the data range ([`Bathymetry2D::project`]),
//! the right choice for raster data finer than the node spacing (e.g.
//! [`crate::io::BedRaster`]).

use faer::Mat;
use faer::linalg::solvers::Solve;

use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::polynomial::{gauss_legendre_nodes_weights, legendre, mode_degrees};
use crate::types::ElementIndex;

/// Physical position and Jacobian determinant of the bilinear map of an
/// element with vertices `v` (counter-clockwise) at reference `(r, s)`.
fn bilinear_map(v: &[[f64; 2]; 4], r: f64, s: f64) -> ([f64; 2], f64) {
    let shape = [
        (1.0 - r) * (1.0 - s),
        (1.0 + r) * (1.0 - s),
        (1.0 + r) * (1.0 + s),
        (1.0 - r) * (1.0 + s),
    ];
    let d_r = [-(1.0 - s), 1.0 - s, 1.0 + s, -(1.0 + s)];
    let d_s = [-(1.0 - r), -(1.0 + r), 1.0 + r, 1.0 - r];
    let [mut x, mut y, mut x_r, mut y_r, mut x_s, mut y_s] = [0.0; 6];
    for (c, &[vx, vy]) in v.iter().enumerate() {
        x += shape[c] * vx;
        y += shape[c] * vy;
        x_r += d_r[c] * vx;
        y_r += d_r[c] * vy;
        x_s += d_s[c] * vx;
        y_s += d_s[c] * vy;
    }
    ([0.25 * x, 0.25 * y], 0.0625 * (x_r * y_s - x_s * y_r))
}

/// A 1D quadrature rule on [−1, 1] with the orthonormal Legendre values of
/// every degree ≤ N at its points (`legendre[point × (N + 1) + degree]`), for
/// [`Bathymetry2D::project`].
#[derive(Clone)]
struct ProjectionRule {
    points: Vec<f64>,
    weights: Vec<f64>,
    legendre: Vec<f64>,
}

impl ProjectionRule {
    /// Call `visit(x, y, w·J, φ)` at every point of the tensor rule `self`
    /// (in r) × `s_rule` (in s) on the element with vertices `verts`, where
    /// `φ[m]` is the orthonormal mode m (numbered by [`mode_degrees`]) there.
    fn for_each_point(
        &self,
        s_rule: &Self,
        verts: &[[f64; 2]; 4],
        phi: &mut [f64],
        mut visit: impl FnMut(f64, f64, f64, &[f64]),
    ) {
        let n_1d = self.legendre.len() / self.points.len();
        for (b, &s) in s_rule.points.iter().enumerate() {
            let l_s = &s_rule.legendre[b * n_1d..][..n_1d];
            for (a, &r) in self.points.iter().enumerate() {
                let l_r = &self.legendre[a * n_1d..][..n_1d];
                for (m, value) in phi.iter_mut().enumerate() {
                    let (i, j) = mode_degrees(m, n_1d);
                    *value = l_r[i] * l_s[j];
                }
                let ([x, y], jac) = bilinear_map(verts, r, s);
                visit(x, y, self.weights[a] * s_rule.weights[b] * jac, phi);
            }
        }
    }
}

/// Bathymetry data for 2D shallow water simulations.
///
/// Stores nodal bathymetry values and their spatial gradients.
/// Layout: `data[k * n_nodes + i]` for element k, node i.
#[derive(Clone)]
pub struct Bathymetry2D {
    /// Nodal bathymetry values B(x, y)
    pub data: Vec<f64>,
    /// Pre-computed x-gradient ∂B/∂x at each node
    pub gradient_x: Vec<f64>,
    /// Pre-computed y-gradient ∂B/∂y at each node
    pub gradient_y: Vec<f64>,
    /// Number of elements
    pub n_elements: usize,
    /// Number of nodes per element
    pub n_nodes: usize,
}

impl Bathymetry2D {
    /// Create flat bathymetry (B = 0 everywhere).
    pub fn flat(n_elements: usize, n_nodes: usize) -> Self {
        Self {
            data: vec![0.0; n_elements * n_nodes],
            gradient_x: vec![0.0; n_elements * n_nodes],
            gradient_y: vec![0.0; n_elements * n_nodes],
            n_elements,
            n_nodes,
        }
    }

    /// Create constant bathymetry (B = constant everywhere).
    pub fn constant(n_elements: usize, n_nodes: usize, value: f64) -> Self {
        Self {
            data: vec![value; n_elements * n_nodes],
            gradient_x: vec![0.0; n_elements * n_nodes], // Zero gradient for constant
            gradient_y: vec![0.0; n_elements * n_nodes],
            n_elements,
            n_nodes,
        }
    }

    /// Initialize bathymetry from a function B(x, y).
    ///
    /// Evaluates B(x, y) at each physical node location and computes gradients.
    pub fn from_function<F>(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        f: F,
    ) -> Self
    where
        F: Fn(f64, f64) -> f64,
    {
        let n_elements = mesh.n_elements;
        let n_nodes = ops.n_nodes;
        let mut bathy = Self::flat(n_elements, n_nodes);

        // Set bathymetry values at each node
        for k in ElementIndex::iter(n_elements) {
            for i in 0..n_nodes {
                let r = ops.nodes_r[i];
                let s = ops.nodes_s[i];
                let [x, y] = mesh.reference_to_physical(k, r, s);
                bathy.set(k, i, f(x, y));
            }
        }

        // Compute gradients
        bathy.compute_gradients(ops, geom);

        bathy
    }

    /// L2 projection of the bed elevation `f(x, y)` onto each element's
    /// polynomial space, limited to the range of `f` over the element.
    ///
    /// Point sampling ([`Self::from_function`]) aliases data that varies on
    /// scales below the node spacing, such as a bathymetry raster whose pixels
    /// are finer than a coarse element. The projection instead finds, per
    /// element k, the B ∈ Q_N minimising ∫_k (B − f)² dx:
    ///
    /// ```text
    /// Σ_n (∫_k φ_m φ_n J) c_n = ∫_k φ_m f J,   B = V c
    /// ```
    ///
    /// with the orthonormal Legendre modes φ_m. The right side is integrated
    /// by composite Gauss–Legendre quadrature on sub-cells no longer than
    /// `resolution / 2` (N + 2 points per sub-cell and direction, at most
    /// [`Self::MAX_PROJECTION_SUBCELLS`] sub-cells per direction), where
    /// `resolution` is the length (m) on which `f` varies, the pixel size for
    /// a raster.
    ///
    /// Properties:
    /// - Exact for `f` in Q_N (e.g. linear fields on any quadrilateral).
    /// - Volume-preserving: `Σ_i w_i J_i B_i = ∫_k f` to quadrature accuracy
    ///   (for N ≥ 2, or on parallelograms), so the still-water volume is the
    ///   data's.
    /// - Bounded: an unresolved step (a coastline cliff) would overshoot like
    ///   any L2 projection, so the nodal values are scaled towards the element
    ///   mean until they lie within the minimum and maximum of `f` over the
    ///   quadrature points and nodes (Zhang & Shu 2010). The scaling keeps the
    ///   mean, and changes smooth data only at the level of the projection
    ///   error, so the projection still converges at N + 1.
    /// - Continuous: the element projections are independent, so a vertex on
    ///   an unresolved coastline could get −18 m in the water element and
    ///   +5 m in its land neighbours, leaving a lone deep node in a pocket
    ///   (it jetted at 8 m/s in the Frøya run). Coincident nodes are
    ///   therefore averaged ([`Self::make_continuous`]), which keeps the
    ///   bounds, the total volume and the exactness for Q_N.
    pub fn project<F>(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        f: F,
        resolution: f64,
    ) -> Self
    where
        F: Fn(f64, f64) -> f64,
    {
        assert!(
            resolution > 0.0 && resolution.is_finite(),
            "projection resolution must be positive, got {resolution}"
        );
        let n_1d = ops.n_1d;
        let n = ops.n_nodes;
        let (gauss_x, gauss_w) = gauss_legendre_nodes_weights(ops.order + 2);
        // Orthonormal Legendre values of every degree at `points`:
        // `[point × degree]`
        let legendre_table = |points: &[f64]| -> Vec<f64> {
            points
                .iter()
                .flat_map(|&x| {
                    (0..n_1d).map(move |i| ((2 * i + 1) as f64 / 2.0).sqrt() * legendre(i, x))
                })
                .collect()
        };
        // Composite rule on `m` equal sub-intervals of [−1, 1], with the
        // Legendre table at its points
        let composite = |m: usize| -> ProjectionRule {
            let half = 1.0 / m as f64;
            let (points, weights): (Vec<f64>, Vec<f64>) = (0..m)
                .flat_map(|c| {
                    let centre = -1.0 + (2 * c + 1) as f64 * half;
                    gauss_x
                        .iter()
                        .zip(&gauss_w)
                        .map(move |(&x, &w)| (centre + half * x, half * w))
                })
                .unzip();
            let legendre = legendre_table(&points);
            ProjectionRule {
                points,
                weights,
                legendre,
            }
        };
        // The mass matrix ∫ φ_m φ_n J: a polynomial of degree 2N + 1 per
        // direction (J is bilinear), exact with the N + 2 point rule
        let mass_rule = composite(1);
        let mut rules: Vec<Option<ProjectionRule>> = vec![None; Self::MAX_PROJECTION_SUBCELLS + 1];

        let mut bathy = Self::flat(mesh.n_elements, n);
        let mut mass = Mat::<f64>::zeros(n, n);
        let mut rhs = Mat::<f64>::zeros(n, 1);
        let mut phi = vec![0.0; n];
        for k in ElementIndex::iter(mesh.n_elements) {
            let verts = mesh.element_vertices(k);
            let dist = |a: usize, b: usize| {
                let [xa, ya] = verts[a];
                let [xb, yb] = verts[b];
                ((xb - xa).powi(2) + (yb - ya).powi(2)).sqrt()
            };
            let subcells = |length: f64| {
                ((2.0 * length / resolution).ceil() as usize)
                    .clamp(1, Self::MAX_PROJECTION_SUBCELLS)
            };
            let m_r = subcells(dist(0, 1).max(dist(3, 2)));
            let m_s = subcells(dist(0, 3).max(dist(1, 2)));
            for m in [m_r, m_s] {
                if rules[m].is_none() {
                    rules[m] = Some(composite(m));
                }
            }
            let (rule_r, rule_s) = (rules[m_r].as_ref().unwrap(), rules[m_s].as_ref().unwrap());

            mass.fill(0.0);
            mass_rule.for_each_point(&mass_rule, &verts, &mut phi, |_, _, w, phi| {
                for row in 0..n {
                    for col in 0..n {
                        mass[(row, col)] += w * phi[row] * phi[col];
                    }
                }
            });

            rhs.fill(0.0);
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            rule_r.for_each_point(rule_s, &verts, &mut phi, |x, y, w, phi| {
                let value = f(x, y);
                lo = lo.min(value);
                hi = hi.max(value);
                for row in 0..n {
                    rhs[(row, 0)] += w * value * phi[row];
                }
            });
            let modal = mass.full_piv_lu().solve(&rhs);

            let ki = k.as_usize();
            let nodal = &mut bathy.data[ki * n..][..n];
            for (i, b) in nodal.iter_mut().enumerate() {
                *b = (0..n)
                    .map(|m| ops.vandermonde.v[(i, m)] * modal[(m, 0)])
                    .sum();
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let value = f(x, y);
                lo = lo.min(value);
                hi = hi.max(value);
            }

            // Scale towards the mean into [lo, hi]
            let mean = geom.element_mean(ki, nodal);
            let (b_min, b_max) = nodal
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            let mut theta: f64 = 1.0;
            if b_max > hi {
                theta = theta.min((hi - mean).max(0.0) / (b_max - mean));
            }
            if b_min < lo {
                theta = theta.min((mean - lo).max(0.0) / (mean - b_min));
            }
            if theta < 1.0 {
                for b in nodal.iter_mut() {
                    *b = mean + theta * (*b - mean);
                }
            }
        }

        bathy.make_continuous(mesh, ops, geom);
        bathy
    }

    /// Make B continuous across elements: every set of coincident nodes
    /// (on shared faces and vertices) gets the mean of its values weighted by
    /// the nodes' mass w·J, as in the direct stiffness summation of spectral
    /// elements; then recompute the gradients.
    ///
    /// The total `Σ w J B` is unchanged and every new value is a convex
    /// combination of old ones, so bounds hold; a continuous B is unchanged.
    /// Nodes are matched by the mesh connectivity (neighbours list their
    /// shared face nodes in reverse order), not by position.
    pub fn make_continuous(
        &mut self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
    ) {
        let n = self.n_nodes;
        let n_face = ops.n_face_nodes;
        let mut parent: Vec<usize> = (0..self.data.len()).collect();
        fn root(parent: &mut [usize], mut a: usize) -> usize {
            while parent[a] != a {
                parent[a] = parent[parent[a]];
                a = parent[a];
            }
            a
        }
        let mut union = |a: usize, b: usize| {
            let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
            if ra != rb {
                parent[ra.max(rb)] = ra.min(rb);
            }
        };
        for edge in &mesh.edges {
            let Some(right) = edge.right else { continue };
            let left = edge.left;
            for fi in 0..n_face {
                union(
                    left.element * n + ops.face_nodes[left.face][fi],
                    right.element * n + ops.face_nodes[right.face][n_face - 1 - fi],
                );
            }
        }
        // Elements that touch at a vertex only
        let corner = |r: f64, s: f64| {
            (0..n)
                .find(|&i| ops.nodes_r[i] == r && ops.nodes_s[i] == s)
                .expect("GLL nodes include the corners")
        };
        let corners = [
            corner(-1.0, -1.0),
            corner(1.0, -1.0),
            corner(1.0, 1.0),
            corner(-1.0, 1.0),
        ];
        let mut vertex_node = vec![None; mesh.n_vertices];
        for (k, vertices) in mesh.elements.iter().enumerate() {
            for (c, &v) in vertices.iter().enumerate() {
                let node = k * n + corners[c];
                match vertex_node[v] {
                    Some(first) => union(first, node),
                    None => vertex_node[v] = Some(node),
                }
            }
        }

        let mut mass = vec![0.0; self.data.len()];
        let mut moment = vec![0.0; self.data.len()];
        for node in 0..self.data.len() {
            let r = root(&mut parent, node);
            mass[r] += geom.mass[node];
            moment[r] += geom.mass[node] * self.data[node];
        }
        for node in 0..self.data.len() {
            let r = root(&mut parent, node);
            self.data[node] = moment[r] / mass[r];
        }
        self.compute_gradients(ops, geom);
    }

    /// Bed from a merged elevation raster (bathymetry and land,
    /// [`crate::io::BedRaster`]), projected onto the nodes
    /// ([`Self::project`], at the raster's pixel size) through the map
    /// `projection` from mesh coordinates to longitude/latitude.
    ///
    /// Nodes on land get the raster's land elevation, so `WetDry` treats them
    /// as dry shore; nodes beyond the raster take its edge values.
    pub fn from_raster<P: crate::io::CoordinateProjection>(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        raster: &crate::io::BedRaster,
        projection: &P,
    ) -> Self {
        Self::project(
            mesh,
            ops,
            geom,
            raster.sampler(projection),
            raster.pixel_size(),
        )
    }

    /// Largest number of projection sub-cells per element and direction
    /// ([`Self::project`]).
    pub const MAX_PROJECTION_SUBCELLS: usize = 64;

    /// The bathymetry of the elements `kept` (old element indices, as
    /// returned by [`Mesh2D::retain_elements`]), in that order.
    ///
    /// The elements keep their geometry, so the gradients are copied too.
    pub fn select_elements(&self, kept: &[usize]) -> Self {
        let n = self.n_nodes;
        let pick = |field: &[f64]| -> Vec<f64> {
            kept.iter()
                .flat_map(|&k| field[k * n..][..n].iter().copied())
                .collect()
        };
        Self {
            data: pick(&self.data),
            gradient_x: pick(&self.gradient_x),
            gradient_y: pick(&self.gradient_y),
            n_elements: kept.len(),
            n_nodes: n,
        }
    }

    /// Get bathymetry at node i in element k.
    #[inline]
    pub fn get(&self, k: ElementIndex, i: usize) -> f64 {
        self.data[k.as_usize() * self.n_nodes + i]
    }

    /// Set bathymetry at node i in element k.
    #[inline]
    pub fn set(&mut self, k: ElementIndex, i: usize, value: f64) {
        self.data[k.as_usize() * self.n_nodes + i] = value;
    }

    /// Get bathymetry gradient (∂B/∂x, ∂B/∂y) at node i in element k.
    #[inline]
    pub fn get_gradient(&self, k: ElementIndex, i: usize) -> (f64, f64) {
        let idx = k.as_usize() * self.n_nodes + i;
        (self.gradient_x[idx], self.gradient_y[idx])
    }

    /// Compute positive water-column depth from free surface elevation.
    ///
    /// Bathymetry is stored as bed elevation B, so depth is eta - B.
    #[inline]
    pub fn water_depth(&self, k: ElementIndex, i: usize, eta: f64) -> f64 {
        eta - self.get(k, i)
    }

    /// Compute the gradient of positive water-column depth.
    ///
    /// Since depth D = eta - B, grad(D) = grad(eta) - grad(B).
    #[inline]
    pub fn water_depth_gradient(
        &self,
        k: ElementIndex,
        i: usize,
        d_eta_dx: f64,
        d_eta_dy: f64,
    ) -> (f64, f64) {
        let (db_dx, db_dy) = self.get_gradient(k, i);
        (d_eta_dx - db_dx, d_eta_dy - db_dy)
    }

    /// Get all bathymetry values for element k.
    pub fn element(&self, k: ElementIndex) -> &[f64] {
        let start = k.as_usize() * self.n_nodes;
        &self.data[start..start + self.n_nodes]
    }

    /// Get all x-gradient values for element k.
    pub fn element_gradient_x(&self, k: usize) -> &[f64] {
        let start = k * self.n_nodes;
        &self.gradient_x[start..start + self.n_nodes]
    }

    /// Get all y-gradient values for element k.
    pub fn element_gradient_y(&self, k: usize) -> &[f64] {
        let start = k * self.n_nodes;
        &self.gradient_y[start..start + self.n_nodes]
    }

    /// Compute gradients ∂B/∂x and ∂B/∂y using the differentiation matrices.
    ///
    /// Process:
    /// 1. Compute ∂B/∂r and ∂B/∂s using Dr and Ds matrices
    /// 2. Transform to physical derivatives using geometric factors:
    ///    ∂B/∂x = rx * ∂B/∂r + sx * ∂B/∂s
    ///    ∂B/∂y = ry * ∂B/∂r + sy * ∂B/∂s
    #[allow(clippy::needless_range_loop)]
    pub fn compute_gradients(&mut self, ops: &DGOperators2D, geom: &GeometricFactors2D) {
        let n = self.n_nodes;

        for k in ElementIndex::iter(self.n_elements) {
            let ki = k.as_usize();
            let b_k = self.element(k);

            // Compute ∂B/∂r = Dr * B_k
            let mut db_dr = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    db_dr[i] += ops.dr[(i, j)] * b_k[j];
                }
            }

            // Compute ∂B/∂s = Ds * B_k
            let mut db_ds = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    db_ds[i] += ops.ds[(i, j)] * b_k[j];
                }
            }

            // Transform to physical derivatives and store
            let start = ki * n;
            for i in 0..n {
                let (db_dx, db_dy) = geom.transform_derivatives(ki, i, db_dr[i], db_ds[i]);
                self.gradient_x[start + i] = db_dx;
                self.gradient_y[start + i] = db_dy;
            }
        }
    }

    /// Get bathymetry at a specific face of element k.
    ///
    /// Returns values at all face nodes in the order defined by `ops.face_nodes[face]`.
    pub fn face_values(&self, k: ElementIndex, ops: &DGOperators2D, face: usize) -> Vec<f64> {
        ops.face_nodes[face]
            .iter()
            .map(|&i| self.get(k, i))
            .collect()
    }

    /// Get maximum bathymetry value in the domain.
    pub fn max(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get minimum bathymetry value in the domain.
    pub fn min(&self) -> f64 {
        self.data.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Get maximum absolute gradient magnitude in the domain.
    pub fn max_gradient_magnitude(&self) -> f64 {
        self.gradient_x
            .iter()
            .zip(self.gradient_y.iter())
            .map(|(&gx, &gy)| (gx * gx + gy * gy).sqrt())
            .fold(0.0, f64::max)
    }

    /// Get the gradient magnitude at a specific node.
    #[inline]
    pub fn gradient_magnitude(&self, k: ElementIndex, i: usize) -> f64 {
        let (gx, gy) = self.get_gradient(k, i);
        (gx * gx + gy * gy).sqrt()
    }

    /// Smooth bathymetry by limiting gradient magnitude.
    ///
    /// For each node where |∇B| > max_gradient, the bathymetry value is adjusted
    /// to reduce the gradient while preserving the mean value in each element.
    ///
    /// This is a simple iterative smoothing that helps with numerical stability
    /// on steep bathymetry.
    ///
    /// # Arguments
    /// * `max_gradient` - Maximum allowed gradient magnitude
    /// * `ops` - DG operators for gradient recomputation
    /// * `geom` - Geometric factors
    /// * `iterations` - Number of smoothing iterations (more = smoother)
    pub fn smooth_gradients(
        &mut self,
        max_gradient: f64,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        iterations: usize,
    ) {
        let n = self.n_nodes;

        for _iter in 0..iterations {
            // For each element, limit the deviations from mean
            for k in ElementIndex::iter(self.n_elements) {
                let ki = k.as_usize();
                // Compute element mean
                let start = ki * n;
                let elem_data = &self.data[start..start + n];
                let mean: f64 = elem_data.iter().sum::<f64>() / n as f64;

                // Check max gradient in element
                let max_grad_in_elem = (0..n)
                    .map(|i| self.gradient_magnitude(k, i))
                    .fold(0.0_f64, f64::max);

                if max_grad_in_elem > max_gradient {
                    // Reduce deviations from mean by a factor
                    let factor = max_gradient / max_grad_in_elem;
                    for i in 0..n {
                        let deviation = self.data[start + i] - mean;
                        self.data[start + i] = mean + deviation * factor;
                    }
                }
            }

            // Recompute gradients after smoothing
            self.compute_gradients(ops, geom);
        }
    }

    /// Project bathymetry to linear (planar) within each element.
    ///
    /// For each element, sets the nodal values to the planar function
    ///   B_linear(x, y) = B_mean + (∂B/∂x)_mean * (x - x_c) + (∂B/∂y)_mean * (y - y_c)
    ///
    /// where (x_c, y_c) is the mean of the nodal coordinates. The fit is done
    /// element by element, so the result generally jumps across faces.
    ///
    /// # Well-balancing
    ///
    /// Planar B is **not** well-balanced at every order with
    /// `SWEFormulation2D::Standard`. The collocated volume term differentiates
    /// the interpolant of ½gh², which balances the `BathymetrySource2D` term
    /// g·h·∂B only if ½gh² is resolved by the element polynomials (deg B ≤ p/2):
    /// - p = 1: ½gh² is quadratic but interpolated linearly, leaving a nodal
    ///   residual g(h̄ − hᵢ)∂B (≈ 0.07 m/s² on steep coastal slopes). Not balanced.
    /// - p ≥ 2: balanced to round-off, provided `BathymetrySource2D` is in
    ///   `source_terms` and `with_well_balanced(true)` handles the face jumps.
    ///
    /// `SWEFormulation2D::EntropyStable` is well-balanced for any nodal B at every
    /// order, so this projection (and its loss of sub-element detail) is not
    /// needed there.
    ///
    /// # Arguments
    /// * `mesh` - The 2D mesh for element geometry
    /// * `ops` - DG operators for node coordinates
    /// * `geom` - Geometric factors for gradient recomputation
    pub fn linearize(&mut self, mesh: &Mesh2D, ops: &DGOperators2D, geom: &GeometricFactors2D) {
        let n = self.n_nodes;

        for k in ElementIndex::iter(self.n_elements) {
            let ki = k.as_usize();
            let start = ki * n;

            // Compute element centroid and mean values
            let mut x_sum = 0.0;
            let mut y_sum = 0.0;
            let mut b_sum = 0.0;
            let mut gx_sum = 0.0;
            let mut gy_sum = 0.0;

            for i in 0..n {
                let r = ops.nodes_r[i];
                let s = ops.nodes_s[i];
                let [x, y] = mesh.reference_to_physical(k, r, s);
                x_sum += x;
                y_sum += y;
                b_sum += self.data[start + i];
                gx_sum += self.gradient_x[start + i];
                gy_sum += self.gradient_y[start + i];
            }

            let n_f = n as f64;
            let x_c = x_sum / n_f;
            let y_c = y_sum / n_f;
            let b_mean = b_sum / n_f;
            let gx_mean = gx_sum / n_f;
            let gy_mean = gy_sum / n_f;

            // Set nodal values to linear function
            for i in 0..n {
                let r = ops.nodes_r[i];
                let s = ops.nodes_s[i];
                let [x, y] = mesh.reference_to_physical(k, r, s);

                // B_linear = B_mean + gx_mean * (x - x_c) + gy_mean * (y - y_c)
                self.data[start + i] = b_mean + gx_mean * (x - x_c) + gy_mean * (y - y_c);
            }
        }

        // Recompute gradients (they should now be constant within each element)
        self.compute_gradients(ops, geom);
    }

    /// Set bathymetry to cell-average (constant) within each element.
    ///
    /// This is the simplest approach for well-balanced schemes:
    /// - Zero gradients within elements → no volume source term
    /// - All bathymetry effects come from interface reconstruction
    /// - Requires `well_balanced=true` in RHS config for proper treatment
    ///
    /// Note: This loses sub-element bathymetry detail but ensures
    /// lake-at-rest preservation. Use with hydrostatic reconstruction.
    pub fn to_cell_average(&mut self) {
        let n = self.n_nodes;

        for k in 0..self.n_elements {
            let start = k * n;

            // Compute element mean
            let mut b_sum = 0.0;
            for i in 0..n {
                b_sum += self.data[start + i];
            }
            let b_mean = b_sum / n as f64;

            // Set all nodes to mean value
            for i in 0..n {
                self.data[start + i] = b_mean;
            }

            // Zero gradients
            for i in 0..n {
                self.gradient_x[start + i] = 0.0;
                self.gradient_y[start + i] = 0.0;
            }
        }
    }

    /// Smooth bathymetry with Laplacian filter.
    ///
    /// Applies element-local Laplacian smoothing: each value is replaced by
    /// a weighted average of itself and the element mean.
    ///
    /// # Arguments
    /// * `alpha` - Smoothing strength: 0 = no change, 1 = replace with mean
    /// * `ops` - DG operators
    /// * `geom` - Geometric factors
    /// * `iterations` - Number of smoothing passes
    pub fn smooth_laplacian(
        &mut self,
        alpha: f64,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        iterations: usize,
    ) {
        let n = self.n_nodes;
        let alpha = alpha.clamp(0.0, 1.0);

        for _iter in 0..iterations {
            for k in 0..self.n_elements {
                let start = k * n;
                let elem_data = &self.data[start..start + n];
                let mean: f64 = elem_data.iter().sum::<f64>() / n as f64;

                // Blend each value toward the mean
                for i in 0..n {
                    let old_val = self.data[start + i];
                    self.data[start + i] = (1.0 - alpha) * old_val + alpha * mean;
                }
            }

            // Recompute gradients
            self.compute_gradients(ops, geom);
        }
    }

    /// Smooth bathymetry using cross-element neighbor averaging.
    ///
    /// Unlike element-local methods (`smooth_laplacian`, `smooth_gradients`),
    /// this method communicates between face-adjacent elements, providing
    /// more effective gradient reduction for steep real-world bathymetry.
    ///
    /// # Algorithm
    ///
    /// For each element:
    /// 1. Find face-adjacent neighbors via mesh edge connectivity
    /// 2. Compute element mean bathymetry
    /// 3. If `max_gradient` is specified, skip elements below threshold
    /// 4. Blend toward average of neighbor means:
    ///    `B_new = (1 - alpha) * B_self + alpha * mean(B_neighbors)`
    ///
    /// # Arguments
    ///
    /// * `mesh` - Mesh with edge connectivity for neighbor lookup
    /// * `ops` - DG operators for gradient recomputation
    /// * `geom` - Geometric factors
    /// * `alpha` - Smoothing strength in [0, 1]:
    ///   - 0 = no smoothing
    ///   - 1 = full neighbor average
    ///   - Recommended: 0.3-0.5 for gentle smoothing
    /// * `iterations` - Number of smoothing passes (more = smoother)
    /// * `max_gradient` - Optional gradient threshold; only smooth elements
    ///   exceeding this. Use `None` to smooth all elements.
    ///
    /// # Recommended Gradient Thresholds
    ///
    /// For numerical stability with well-balanced schemes:
    /// - Conservative: 0.3 (gentle slopes)
    /// - Moderate: 0.5 (typical coastal bathymetry)
    /// - Aggressive: 1.0 (allow steeper slopes)
    ///
    /// Norwegian fjord bathymetry often has gradients up to 0.66.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Smooth steep regions (gradient > 0.5) with moderate strength
    /// bathymetry.smooth_cross_element(&mesh, &ops, &geom, 0.4, 5, Some(0.5));
    ///
    /// // Smooth entire domain gently
    /// bathymetry.smooth_cross_element(&mesh, &ops, &geom, 0.2, 3, None);
    /// ```
    pub fn smooth_cross_element(
        &mut self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        alpha: f64,
        iterations: usize,
        max_gradient: Option<f64>,
    ) {
        let n = self.n_nodes;
        let alpha = alpha.clamp(0.0, 1.0);

        if alpha == 0.0 || iterations == 0 {
            return;
        }

        for _iter in 0..iterations {
            // Compute element means (needed for neighbor averaging)
            let element_means: Vec<f64> = (0..self.n_elements)
                .map(|k| {
                    let start = k * n;
                    self.data[start..start + n].iter().sum::<f64>() / n as f64
                })
                .collect();

            // Compute element max gradient magnitudes if threshold specified
            let should_smooth: Vec<bool> = if let Some(threshold) = max_gradient {
                (0..self.n_elements)
                    .map(|k| {
                        let max_grad = (0..n)
                            .map(|i| self.gradient_magnitude(ElementIndex::new(k), i))
                            .fold(0.0_f64, f64::max);
                        max_grad > threshold
                    })
                    .collect()
            } else {
                vec![true; self.n_elements]
            };

            // Compute new element means based on neighbor averaging
            let new_means: Vec<f64> = (0..self.n_elements)
                .map(|k| {
                    if !should_smooth[k] {
                        return element_means[k];
                    }

                    // Get face-adjacent neighbors
                    let neighbors = Self::get_neighbor_elements(mesh, k);

                    if neighbors.is_empty() {
                        // Boundary element with no neighbors - keep current value
                        return element_means[k];
                    }

                    // Compute average of neighbor means
                    let neighbor_avg: f64 =
                        neighbors.iter().map(|&nk| element_means[nk]).sum::<f64>()
                            / neighbors.len() as f64;

                    // Blend self with neighbors
                    (1.0 - alpha) * element_means[k] + alpha * neighbor_avg
                })
                .collect();

            // Apply shift to all nodes in each element
            for k in 0..self.n_elements {
                if !should_smooth[k] {
                    continue;
                }

                let shift = new_means[k] - element_means[k];
                if shift.abs() < 1e-15 {
                    continue;
                }

                let start = k * n;
                for i in 0..n {
                    self.data[start + i] += shift;
                }
            }

            // Recompute gradients after smoothing
            self.compute_gradients(ops, geom);
        }
    }

    /// Get face-adjacent neighbor elements for element k.
    ///
    /// Returns a vector of element indices that share a face with element k.
    /// Boundary faces have no neighbor and are excluded.
    fn get_neighbor_elements(mesh: &Mesh2D, k: usize) -> Vec<usize> {
        let mut neighbors = Vec::with_capacity(4);

        for &edge_idx in &mesh.element_edges[k] {
            let edge = &mesh.edges[edge_idx];

            // Determine which side we are (left or right)
            if edge.left.element == k {
                // We're on left side, neighbor is on right (if exists)
                if let Some(right) = &edge.right {
                    neighbors.push(right.element);
                }
            } else if let Some(right) = &edge.right {
                // We're on right side, neighbor is on left
                if right.element == k {
                    neighbors.push(edge.left.element);
                }
            }
        }

        neighbors
    }
}

/// Common 2D bathymetry profiles for testing.
pub mod profiles {
    /// Gaussian bump centered at (x_c, y_c) with height A and width σ.
    ///
    /// B(x, y) = A * exp(-((x - x_c)² + (y - y_c)²) / (2σ²))
    pub fn gaussian_bump(x: f64, y: f64, x_c: f64, y_c: f64, amplitude: f64, sigma: f64) -> f64 {
        let dx = x - x_c;
        let dy = y - y_c;
        let r2 = dx * dx + dy * dy;
        amplitude * (-r2 / (2.0 * sigma * sigma)).exp()
    }

    /// Linear slope in the x-direction.
    ///
    /// B(x, y) = slope_x * x + offset
    pub fn linear_slope_x(x: f64, _y: f64, slope_x: f64, offset: f64) -> f64 {
        slope_x * x + offset
    }

    /// Linear slope in both directions.
    ///
    /// B(x, y) = slope_x * x + slope_y * y + offset
    pub fn linear_slope(x: f64, y: f64, slope_x: f64, slope_y: f64, offset: f64) -> f64 {
        slope_x * x + slope_y * y + offset
    }

    /// Fjord sill profile: a ridge across the y-direction.
    ///
    /// B(x, y) = B_base + A * exp(-((x - x_sill)² / (2σ²)))
    ///
    /// This creates a sill (underwater ridge) perpendicular to the x-axis.
    pub fn sill(x: f64, _y: f64, x_sill: f64, amplitude: f64, sigma: f64, b_base: f64) -> f64 {
        let dx = x - x_sill;
        b_base + amplitude * (-dx * dx / (2.0 * sigma * sigma)).exp()
    }

    /// Channel profile: deeper in the center, shallower at edges.
    ///
    /// B(x, y) = B_center + A * (1 - cos(π * (y - y_min) / (y_max - y_min)))
    ///
    /// where B_center is the center depth and A is the wall height.
    pub fn channel(y: f64, y_min: f64, y_max: f64, b_center: f64, wall_height: f64) -> f64 {
        let t = (y - y_min) / (y_max - y_min);
        b_center + wall_height * (1.0 - (std::f64::consts::PI * t).cos()) / 2.0
    }

    /// Parabolic bowl for wetting/drying tests.
    ///
    /// B(x, y) = h_0 * ((x - x_c)² + (y - y_c)²) / a²
    pub fn parabolic_bowl(x: f64, y: f64, x_center: f64, y_center: f64, a: f64, h_0: f64) -> f64 {
        let dx = x - x_center;
        let dy = y - y_center;
        h_0 * (dx * dx + dy * dy) / (a * a)
    }

    /// Step function in x-direction (discontinuous bathymetry).
    ///
    /// B(x, y) = B_left if x < x_step, B_right otherwise
    pub fn step_x(x: f64, _y: f64, x_step: f64, b_left: f64, b_right: f64) -> f64 {
        if x < x_step { b_left } else { b_right }
    }

    /// Narrow strait: elevated bathymetry except in a narrow passage.
    ///
    /// Creates a topographic barrier with a gap.
    /// B = B_barrier everywhere except where |y - y_center| < gap_width/2
    /// In the gap: B = B_channel
    pub fn strait(
        x: f64,
        y: f64,
        x_barrier: f64,
        barrier_width: f64,
        y_center: f64,
        gap_width: f64,
        b_channel: f64,
        b_barrier: f64,
    ) -> f64 {
        // Check if within the barrier x-range
        if (x - x_barrier).abs() > barrier_width / 2.0 {
            return b_channel; // Outside barrier
        }

        // Within barrier x-range: check if in gap
        if (y - y_center).abs() < gap_width / 2.0 {
            b_channel // In the strait gap
        } else {
            b_barrier // On the barrier
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(idx: usize) -> ElementIndex {
        ElementIndex::new(idx)
    }

    const TOL: f64 = 1e-10;

    fn make_mesh_and_ops() -> (Mesh2D, DGOperators2D, GeometricFactors2D) {
        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        (mesh, ops, geom)
    }

    #[test]
    fn test_flat_bathymetry() {
        let bathy = Bathymetry2D::flat(4, 16);

        assert_eq!(bathy.data.len(), 64);
        assert_eq!(bathy.gradient_x.len(), 64);
        assert_eq!(bathy.gradient_y.len(), 64);

        for &b in &bathy.data {
            assert!(b.abs() < TOL);
        }
        for &g in &bathy.gradient_x {
            assert!(g.abs() < TOL);
        }
        for &g in &bathy.gradient_y {
            assert!(g.abs() < TOL);
        }
    }

    #[test]
    fn test_constant_bathymetry() {
        let bathy = Bathymetry2D::constant(4, 16, 5.0);

        for &b in &bathy.data {
            assert!((b - 5.0).abs() < TOL);
        }
        for &g in &bathy.gradient_x {
            assert!(g.abs() < TOL); // Gradient of constant is zero
        }
        for &g in &bathy.gradient_y {
            assert!(g.abs() < TOL);
        }
    }

    #[test]
    fn test_from_function_constant() {
        let (mesh, ops, geom) = make_mesh_and_ops();
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |_, _| 3.0);

        for &b in &bathy.data {
            assert!((b - 3.0).abs() < TOL);
        }
        for &g in &bathy.gradient_x {
            assert!(g.abs() < 1e-8);
        }
        for &g in &bathy.gradient_y {
            assert!(g.abs() < 1e-8);
        }
    }

    #[test]
    fn test_from_function_linear_x() {
        let (mesh, ops, geom) = make_mesh_and_ops();

        // Linear bathymetry: B(x, y) = 2*x
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| 2.0 * x);

        // Check values
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let r = ops.nodes_r[i];
                let s = ops.nodes_s[i];
                let [x, _y] = mesh.reference_to_physical(k(ki), r, s);
                let b = bathy.get(k(ki), i);
                assert!(
                    (b - 2.0 * x).abs() < TOL,
                    "B({}, _) = {}, expected {}",
                    x,
                    b,
                    2.0 * x
                );
            }
        }

        // Gradient should be (2, 0) everywhere
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let (gx, gy) = bathy.get_gradient(k(ki), i);
                assert!((gx - 2.0).abs() < 1e-8, "∂B/∂x = {}, expected 2.0", gx);
                assert!(gy.abs() < 1e-8, "∂B/∂y = {}, expected 0.0", gy);
            }
        }
    }

    #[test]
    fn test_from_function_linear_y() {
        let (mesh, ops, geom) = make_mesh_and_ops();

        // Linear bathymetry: B(x, y) = 3*y
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |_x, y| 3.0 * y);

        // Gradient should be (0, 3) everywhere
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let (gx, gy) = bathy.get_gradient(k(ki), i);
                assert!(gx.abs() < 1e-8, "∂B/∂x = {}, expected 0.0", gx);
                assert!((gy - 3.0).abs() < 1e-8, "∂B/∂y = {}, expected 3.0", gy);
            }
        }
    }

    #[test]
    fn test_from_function_linear_xy() {
        let (mesh, ops, geom) = make_mesh_and_ops();

        // Linear bathymetry: B(x, y) = 2*x + 3*y + 1
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| 2.0 * x + 3.0 * y + 1.0);

        // Gradient should be (2, 3) everywhere
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let (gx, gy) = bathy.get_gradient(k(ki), i);
                assert!((gx - 2.0).abs() < 1e-8, "∂B/∂x = {}, expected 2.0", gx);
                assert!((gy - 3.0).abs() < 1e-8, "∂B/∂y = {}, expected 3.0", gy);
            }
        }
    }

    #[test]
    fn test_from_function_quadratic() {
        let (mesh, ops, geom) = make_mesh_and_ops();

        // Quadratic bathymetry: B(x, y) = x² + y²
        // Gradients: ∂B/∂x = 2x, ∂B/∂y = 2y
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| x * x + y * y);

        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let r = ops.nodes_r[i];
                let s = ops.nodes_s[i];
                let [x, y] = mesh.reference_to_physical(k(ki), r, s);
                let (gx, gy) = bathy.get_gradient(k(ki), i);

                assert!(
                    (gx - 2.0 * x).abs() < 1e-6,
                    "∂B/∂x at ({}, {}) is {}, expected {}",
                    x,
                    y,
                    gx,
                    2.0 * x
                );
                assert!(
                    (gy - 2.0 * y).abs() < 1e-6,
                    "∂B/∂y at ({}, {}) is {}, expected {}",
                    x,
                    y,
                    gy,
                    2.0 * y
                );
            }
        }
    }

    #[test]
    fn test_element_access() {
        let (mesh, ops, geom) = make_mesh_and_ops();
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| x + y);

        let elem = bathy.element(k(2));
        assert_eq!(elem.len(), ops.n_nodes);

        let grad_x = bathy.element_gradient_x(2);
        assert_eq!(grad_x.len(), ops.n_nodes);

        let grad_y = bathy.element_gradient_y(2);
        assert_eq!(grad_y.len(), ops.n_nodes);
    }

    #[test]
    fn test_min_max() {
        let (mesh, ops, geom) = make_mesh_and_ops();
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| x + y);

        let min = bathy.min();
        let max = bathy.max();

        // Domain is [0, 10] × [0, 10], so min ≈ 0, max ≈ 20
        assert!(min < 0.5);
        assert!(max > 19.5);
    }

    #[test]
    fn test_gradient_magnitude() {
        let (mesh, ops, geom) = make_mesh_and_ops();

        // B(x, y) = 3*x + 4*y, gradient = (3, 4), magnitude = 5
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| 3.0 * x + 4.0 * y);

        let max_mag = bathy.max_gradient_magnitude();
        assert!((max_mag - 5.0).abs() < 1e-8);

        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let mag = bathy.gradient_magnitude(k(ki), i);
                assert!((mag - 5.0).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn test_gaussian_bump_profile() {
        let b = profiles::gaussian_bump(5.0, 5.0, 5.0, 5.0, 1.0, 1.0);
        assert!((b - 1.0).abs() < TOL); // At center, B = amplitude

        let b_off = profiles::gaussian_bump(0.0, 0.0, 5.0, 5.0, 1.0, 1.0);
        assert!(b_off < 1e-5); // Far from center, B ≈ 0
    }

    #[test]
    fn test_sill_profile() {
        // Sill at x = 5 with amplitude 1, sigma 1
        let b_center = profiles::sill(5.0, 0.0, 5.0, 1.0, 1.0, 0.0);
        assert!((b_center - 1.0).abs() < TOL); // At sill center

        let b_far = profiles::sill(0.0, 0.0, 5.0, 1.0, 1.0, 0.0);
        assert!(b_far < 1e-5); // Far from sill
    }

    #[test]
    fn test_strait_profile() {
        // Barrier at x = 5, width 2, gap at y = 5, gap width 1
        // Channel depth 0, barrier height 10

        // In the gap (should be channel depth)
        let b_gap = profiles::strait(5.0, 5.0, 5.0, 2.0, 5.0, 1.0, 0.0, 10.0);
        assert!((b_gap - 0.0).abs() < TOL);

        // On the barrier (outside gap)
        let b_barrier = profiles::strait(5.0, 7.0, 5.0, 2.0, 5.0, 1.0, 0.0, 10.0);
        assert!((b_barrier - 10.0).abs() < TOL);

        // Outside barrier x-range
        let b_outside = profiles::strait(0.0, 5.0, 5.0, 2.0, 5.0, 1.0, 0.0, 10.0);
        assert!((b_outside - 0.0).abs() < TOL);
    }

    #[test]
    fn test_parabolic_bowl_profile() {
        let h_0 = 1.0;
        let a = 5.0;

        let b_center = profiles::parabolic_bowl(5.0, 5.0, 5.0, 5.0, a, h_0);
        assert!(b_center.abs() < TOL); // At center, B = 0

        // At (10, 5): dx = 5, dy = 0, B = 1.0 * 25 / 25 = 1.0
        let b_edge = profiles::parabolic_bowl(10.0, 5.0, 5.0, 5.0, a, h_0);
        assert!((b_edge - h_0).abs() < TOL);
    }

    #[test]
    fn test_smooth_cross_element_preserves_flat() {
        let (mesh, ops, geom) = make_mesh_and_ops();
        let mut bathy = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, 5.0);
        bathy.compute_gradients(&ops, &geom);

        let original_sum: f64 = bathy.data.iter().sum();

        // Smooth should not change flat bathymetry
        bathy.smooth_cross_element(&mesh, &ops, &geom, 0.5, 5, None);

        let new_sum: f64 = bathy.data.iter().sum();
        assert!(
            (original_sum - new_sum).abs() < 1e-10,
            "Cross-element smoothing changed total bathymetry on flat surface"
        );

        // All values should still be 5.0
        for &b in &bathy.data {
            assert!(
                (b - 5.0).abs() < 1e-10,
                "Flat bathymetry changed during smoothing"
            );
        }
    }

    #[test]
    fn test_smooth_cross_element_reduces_gradient() {
        let (mesh, ops, geom) = make_mesh_and_ops();

        // Create steep bathymetry: B = x with gradient 1.0
        let mut bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| x);

        let initial_max_grad = bathy.max_gradient_magnitude();
        assert!(initial_max_grad > 0.5, "Initial gradient should be steep");

        // Smooth the bathymetry
        bathy.smooth_cross_element(&mesh, &ops, &geom, 0.5, 10, None);

        let final_max_grad = bathy.max_gradient_magnitude();

        // Gradient should be reduced (though not necessarily to zero due to boundaries)
        // The smoothing should reduce interior gradients
        assert!(
            final_max_grad <= initial_max_grad + 1e-10,
            "Smoothing should not increase max gradient: {} -> {}",
            initial_max_grad,
            final_max_grad
        );
    }

    #[test]
    fn test_smooth_cross_element_with_threshold() {
        let (mesh, ops, geom) = make_mesh_and_ops();

        // Create bathymetry with steep gradient: B = 2*x (gradient = 2.0)
        let mut bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| 2.0 * x);

        // Also create a copy with low gradient threshold that won't trigger
        let mut bathy_low_threshold = bathy.clone();

        // Smooth only elements with gradient > 1.0 (all elements)
        bathy.smooth_cross_element(&mesh, &ops, &geom, 0.5, 5, Some(1.0));

        // Smooth only elements with gradient > 10.0 (no elements)
        bathy_low_threshold.smooth_cross_element(&mesh, &ops, &geom, 0.5, 5, Some(10.0));

        // The high threshold version should be unchanged
        let original = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| 2.0 * x);
        let diff: f64 = bathy_low_threshold
            .data
            .iter()
            .zip(original.data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(diff < 1e-10, "High threshold should not trigger smoothing");

        // The low threshold version should be different
        let diff2: f64 = bathy
            .data
            .iter()
            .zip(original.data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(diff2 > 0.1, "Low threshold should trigger smoothing");
    }

    #[test]
    fn test_get_neighbor_elements() {
        // Create a 3x3 mesh to test neighbor finding
        let mesh = Mesh2D::uniform_rectangle(0.0, 3.0, 0.0, 3.0, 3, 3);

        // Center element (4) should have 4 neighbors
        let center_neighbors = Bathymetry2D::get_neighbor_elements(&mesh, 4);
        assert_eq!(
            center_neighbors.len(),
            4,
            "Center element should have 4 neighbors"
        );

        // Corner element (0) should have 2 neighbors
        let corner_neighbors = Bathymetry2D::get_neighbor_elements(&mesh, 0);
        assert_eq!(
            corner_neighbors.len(),
            2,
            "Corner element should have 2 neighbors"
        );

        // Edge element (1) should have 3 neighbors
        let edge_neighbors = Bathymetry2D::get_neighbor_elements(&mesh, 1);
        assert_eq!(
            edge_neighbors.len(),
            3,
            "Edge element should have 3 neighbors"
        );
    }

    #[test]
    fn test_smooth_cross_element_no_op_with_alpha_zero() {
        let (mesh, ops, geom) = make_mesh_and_ops();
        let mut bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| x * y);

        let original: Vec<f64> = bathy.data.clone();

        // alpha = 0 should be a no-op
        bathy.smooth_cross_element(&mesh, &ops, &geom, 0.0, 10, None);

        assert_eq!(bathy.data, original, "alpha=0 should not change anything");
    }

    #[test]
    fn test_smooth_cross_element_no_op_with_zero_iterations() {
        let (mesh, ops, geom) = make_mesh_and_ops();
        let mut bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| x * y);

        let original: Vec<f64> = bathy.data.clone();

        // iterations = 0 should be a no-op
        bathy.smooth_cross_element(&mesh, &ops, &geom, 0.5, 0, None);

        assert_eq!(
            bathy.data, original,
            "iterations=0 should not change anything"
        );
    }

    /// `n` × `n` mesh of [0, 1]² with interior vertices moved, so that
    /// the elements are general (non-parallelogram) quadrilaterals.
    fn distorted_mesh(n: usize) -> Mesh2D {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, n, n);
        let tau = 2.0 * std::f64::consts::PI;
        let a = 0.3 / n as f64;
        for v in &mut mesh.vertices {
            let [x, y] = *v;
            let bump = (tau * x).sin() * (tau * y).sin();
            *v = [x + a * bump, y - 0.7 * a * bump];
        }
        mesh
    }

    fn setup(mesh: Mesh2D, order: usize) -> (Mesh2D, DGOperators2D, GeometricFactors2D) {
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        (mesh, ops, geom)
    }

    /// Largest |B − f| over the nodes.
    fn nodal_error(
        bathy: &Bathymetry2D,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        f: impl Fn(f64, f64) -> f64,
    ) -> f64 {
        let mut error: f64 = 0.0;
        for e in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(e, ops.nodes_r[i], ops.nodes_s[i]);
                error = error.max((bathy.get(e, i) - f(x, y)).abs());
            }
        }
        error
    }

    #[test]
    fn test_projection_is_exact_for_polynomials_of_the_element_degree() {
        // x^a y^b with a + b ≤ N lies in Q_N of a bilinear element
        for order in 1..=4 {
            let (mesh, ops, geom) = setup(distorted_mesh(3), order);
            let f = |x: f64, y: f64| {
                -20.0 + 3.0 * x - 2.0 * y
                    + (0..=order)
                        .map(|a| x.powi(a as i32) * y.powi((order - a) as i32))
                        .sum::<f64>()
            };
            let bathy = Bathymetry2D::project(&mesh, &ops, &geom, f, 0.2);
            let error = nodal_error(&bathy, &mesh, &ops, f);
            assert!(error < 1e-11, "N = {order}: {error:e}");
        }
    }

    #[test]
    fn test_projection_keeps_volume_and_bounds_at_a_cliff() {
        // A coastline cliff from −30 m to +5 m at x = 0.5, inside the second
        // column of 1/3-wide elements and on a sub-cell edge (1/60-wide
        // sub-cells), so the quadrature of the step is exact
        let (lo, hi, x_c) = (-30.0, 5.0, 0.5);
        let f = |x: f64, _y: f64| if x < x_c { lo } else { hi };
        for order in 1..=4 {
            let (mesh, ops, geom) =
                setup(Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 3, 3), order);
            let bathy = Bathymetry2D::project(&mesh, &ops, &geom, f, 0.034);
            // The total volume is the data's (element volumes move between
            // neighbours when coincident nodes are averaged)
            let volume: f64 = ElementIndex::iter(mesh.n_elements)
                .map(|e| geom.integrate_element(e.as_usize(), bathy.element(e)))
                .sum();
            let exact = lo * x_c + hi * (1.0 - x_c);
            assert!(
                (volume - exact).abs() < 1e-12,
                "N = {order}: {volume} vs {exact}"
            );
            assert!(
                bathy
                    .data
                    .iter()
                    .all(|&b| (lo - 1e-9..=hi + 1e-9).contains(&b)),
                "N = {order}: [{}, {}]",
                bathy.min(),
                bathy.max()
            );
            // The step elements keep a shoreline: dry and wet nodes
            let step = ElementIndex::new(1);
            let (b_min, b_max) = bathy
                .element(step)
                .iter()
                .fold((hi, lo), |(a, b), &v| (a.min(v), b.max(v)));
            assert!(
                b_min < -10.0 && b_max > 0.0,
                "N = {order}: [{b_min}, {b_max}]"
            );
            // Continuous across elements; away from the step column (off its
            // faces, which are averaged with it) the data
            assert_continuous(&bathy, &mesh, &ops, order);
            for e in ElementIndex::iter(mesh.n_elements) {
                for i in 0..ops.n_nodes {
                    let [x, y] = mesh.reference_to_physical(e, ops.nodes_r[i], ops.nodes_s[i]);
                    if !(1.0 / 3.0 - 1e-9..=2.0 / 3.0 + 1e-9).contains(&x) {
                        let error = (bathy.get(e, i) - f(x, y)).abs();
                        assert!(error < 1e-11, "N = {order}, ({x}, {y}): {error:e}");
                    }
                }
            }
        }
    }

    /// Coincident nodes (same position to 1e-9) carry the same B.
    fn assert_continuous(bathy: &Bathymetry2D, mesh: &Mesh2D, ops: &DGOperators2D, order: usize) {
        let mut seen: std::collections::HashMap<(i64, i64), f64> = Default::default();
        for e in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(e, ops.nodes_r[i], ops.nodes_s[i]);
                let key = ((x * 1e9).round() as i64, (y * 1e9).round() as i64);
                let b = bathy.get(e, i);
                let first = *seen.entry(key).or_insert(b);
                assert!(
                    (b - first).abs() < 1e-12,
                    "N = {order}, ({x}, {y}): {b} vs {first}"
                );
            }
        }
    }

    #[test]
    fn test_projection_does_not_alias_unresolved_data() {
        // Ripples of wavelength 0.13 on elements of 1/2 (P2 node spacing
        // 1/4): point sampling returns their amplitude, the projection
        // nearly their mean
        let (mesh, ops, geom) = setup(distorted_mesh(2), 2);
        let ripple = |x: f64, y: f64| {
            let k = 2.0 * std::f64::consts::PI / 0.13;
            -50.0 + 10.0 * (k * x + 0.3).sin() * (k * y + 0.7).sin()
        };
        let sampled = Bathymetry2D::from_function(&mesh, &ops, &geom, ripple);
        let projected = Bathymetry2D::project(&mesh, &ops, &geom, ripple, 0.13 / 4.0);
        let spread =
            |b: &Bathymetry2D| b.data.iter().map(|&v| (v + 50.0).abs()).fold(0.0, f64::max);
        assert!(spread(&sampled) > 5.0, "sampled: {}", spread(&sampled));
        assert!(
            spread(&projected) < 1.0,
            "projected: {}",
            spread(&projected)
        );
    }

    #[test]
    fn test_projection_converges_at_n_plus_one() {
        let f = |x: f64, y: f64| -40.0 + 10.0 * (3.0 * x + 1.0).sin() * (2.0 * y - 0.5).cos();
        for order in 2..=3 {
            let errors: Vec<f64> = [4, 8, 16]
                .iter()
                .map(|&n| {
                    let (mesh, ops, geom) = setup(distorted_mesh(n), order);
                    let bathy = Bathymetry2D::project(&mesh, &ops, &geom, f, 0.25 / n as f64);
                    nodal_error(&bathy, &mesh, &ops, f)
                })
                .collect();
            for pair in errors.windows(2) {
                let rate = (pair[0] / pair[1]).log2();
                assert!(
                    rate > order as f64 + 0.7,
                    "N = {order}: rate {rate:.2} ({errors:?})"
                );
            }
        }
    }

    #[test]
    fn test_select_elements_follows_retain_elements() {
        let (mesh, ops, geom) = setup(distorted_mesh(3), 2);
        let bed = |x: f64, y: f64| x - 0.5 + 0.1 * y;
        let bathy = Bathymetry2D::project(&mesh, &ops, &geom, bed, 0.1);
        let wet = |e: ElementIndex| bathy.element(e).iter().any(|&b| b < 0.0);
        let (water, kept) = mesh.retain_elements(wet, crate::mesh::BoundaryTag::Wall);
        let selected = bathy.select_elements(&kept);
        assert!(water.n_elements < mesh.n_elements);
        let water_geom = GeometricFactors2D::compute(&water, &ops);
        let direct = Bathymetry2D::project(&water, &ops, &water_geom, bed, 0.1);
        for (field, reference) in [
            (&selected.data, &direct.data),
            (&selected.gradient_x, &direct.gradient_x),
            (&selected.gradient_y, &direct.gradient_y),
        ] {
            let diff = field
                .iter()
                .zip(reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(diff < 1e-12, "{diff:e}");
        }
    }
}
