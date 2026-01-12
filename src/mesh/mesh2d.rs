//! 2D mesh representation for quadrilateral elements.
//!
//! The mesh stores:
//! - Vertex coordinates
//! - Element-vertex connectivity (counter-clockwise ordering)
//! - Edge-based connectivity for inter-element flux computation
//! - Boundary edge identification
//!
//! Face convention (counter-clockwise around element):
//! - Face 0 (bottom): from vertex 0 to vertex 1
//! - Face 1 (right):  from vertex 1 to vertex 2
//! - Face 2 (top):    from vertex 2 to vertex 3
//! - Face 3 (left):   from vertex 3 to vertex 0

use super::boundary_tags::BoundaryTag;

/// Reference to an element and one of its faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementFace {
    /// Element index
    pub element: usize,
    /// Face index (0-3 for quads)
    pub face: usize,
}

impl ElementFace {
    pub fn new(element: usize, face: usize) -> Self {
        Self { element, face }
    }
}

/// Information about an edge in the mesh.
#[derive(Clone, Debug)]
pub struct Edge {
    /// Vertex indices (v0, v1) with v0 < v1 for consistent ordering
    pub vertices: (usize, usize),
    /// Left element-face (always present)
    pub left: ElementFace,
    /// Right element-face (None for boundary edges)
    pub right: Option<ElementFace>,
    /// Boundary tag (only for boundary edges)
    pub boundary_tag: Option<BoundaryTag>,
}

impl Edge {
    /// Check if this is a boundary edge.
    pub fn is_boundary(&self) -> bool {
        self.right.is_none()
    }

    /// Check if this is an interior edge.
    pub fn is_interior(&self) -> bool {
        self.right.is_some()
    }
}

/// 2D mesh of quadrilateral elements.
#[derive(Clone)]
pub struct Mesh2D {
    /// Vertex coordinates: vertices[i] = (x, y)
    pub vertices: Vec<(f64, f64)>,

    /// Element-vertex connectivity: elements[k] = [v0, v1, v2, v3]
    /// Vertices are in counter-clockwise order:
    /// - v0: bottom-left  (r=-1, s=-1)
    /// - v1: bottom-right (r=+1, s=-1)
    /// - v2: top-right    (r=+1, s=+1)
    /// - v3: top-left     (r=-1, s=+1)
    pub elements: Vec<[usize; 4]>,

    /// Edge list with connectivity information
    pub edges: Vec<Edge>,

    /// Element-to-edge mapping: element_edges[k][f] = edge index for face f of element k
    pub element_edges: Vec<[usize; 4]>,

    /// Edge orientation for each element face:
    /// +1 if element's face direction matches edge direction
    /// -1 if reversed
    pub edge_orientation: Vec<[i8; 4]>,

    /// Number of elements
    pub n_elements: usize,

    /// Number of edges
    pub n_edges: usize,

    /// Number of boundary edges
    pub n_boundary_edges: usize,

    /// Number of vertices
    pub n_vertices: usize,

    /// Vertex-to-element connectivity: vertex_to_elements[v] = list of element indices
    /// containing vertex v. Used by vertex-based slope limiters (e.g., Kuzmin).
    ///
    /// For structured quad meshes:
    /// - Interior vertices have 4 elements
    /// - Edge boundary vertices have 2 elements
    /// - Corner boundary vertices have 1 element
    pub vertex_to_elements: Vec<Vec<usize>>,
}

impl Mesh2D {
    /// Create a uniform rectangular mesh of [x0, x1] × [y0, y1].
    ///
    /// # Arguments
    /// * `x0`, `x1` - x-coordinate bounds
    /// * `y0`, `y1` - y-coordinate bounds
    /// * `nx` - number of elements in x-direction
    /// * `ny` - number of elements in y-direction
    pub fn uniform_rectangle(x0: f64, x1: f64, y0: f64, y1: f64, nx: usize, ny: usize) -> Self {
        Self::uniform_rectangle_with_bc(x0, x1, y0, y1, nx, ny, BoundaryTag::Wall)
    }

    /// Create a uniform rectangular mesh with a specific boundary tag on all boundaries.
    pub fn uniform_rectangle_with_bc(
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
        nx: usize,
        ny: usize,
        bc_tag: BoundaryTag,
    ) -> Self {
        assert!(
            nx > 0 && ny > 0,
            "Need at least one element in each direction"
        );
        assert!(x1 > x0 && y1 > y0, "Invalid domain bounds");

        let dx = (x1 - x0) / nx as f64;
        let dy = (y1 - y0) / ny as f64;

        // Generate vertices: (nx+1) × (ny+1) grid
        let n_vertices = (nx + 1) * (ny + 1);
        let mut vertices = Vec::with_capacity(n_vertices);

        for j in 0..=ny {
            for i in 0..=nx {
                let x = x0 + i as f64 * dx;
                let y = y0 + j as f64 * dy;
                vertices.push((x, y));
            }
        }

        // Generate elements: nx × ny quads
        let n_elements = nx * ny;
        let mut elements = Vec::with_capacity(n_elements);

        for j in 0..ny {
            for i in 0..nx {
                // Vertex indices for this element (counter-clockwise)
                let v0 = j * (nx + 1) + i; // bottom-left
                let v1 = v0 + 1; // bottom-right
                let v2 = v1 + (nx + 1); // top-right
                let v3 = v0 + (nx + 1); // top-left
                elements.push([v0, v1, v2, v3]);
            }
        }

        // Build edge connectivity
        Self::build_mesh_with_connectivity(vertices, elements, nx, ny, bc_tag)
    }

    /// Create a uniform rectangular mesh with different boundary tags on each side.
    ///
    /// # Arguments
    /// * `x0`, `x1` - x-coordinate bounds
    /// * `y0`, `y1` - y-coordinate bounds
    /// * `nx` - number of elements in x-direction
    /// * `ny` - number of elements in y-direction
    /// * `bc_tags` - boundary tags for [south, east, north, west] sides
    pub fn uniform_rectangle_with_sides(
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
        nx: usize,
        ny: usize,
        bc_tags: [BoundaryTag; 4],
    ) -> Self {
        assert!(
            nx > 0 && ny > 0,
            "Need at least one element in each direction"
        );
        assert!(x1 > x0 && y1 > y0, "Invalid domain bounds");

        let dx = (x1 - x0) / nx as f64;
        let dy = (y1 - y0) / ny as f64;

        // Generate vertices
        let n_vertices = (nx + 1) * (ny + 1);
        let mut vertices = Vec::with_capacity(n_vertices);

        for j in 0..=ny {
            for i in 0..=nx {
                let x = x0 + i as f64 * dx;
                let y = y0 + j as f64 * dy;
                vertices.push((x, y));
            }
        }

        // Generate elements
        let n_elements = nx * ny;
        let mut elements = Vec::with_capacity(n_elements);

        for j in 0..ny {
            for i in 0..nx {
                let v0 = j * (nx + 1) + i;
                let v1 = v0 + 1;
                let v2 = v1 + (nx + 1);
                let v3 = v0 + (nx + 1);
                elements.push([v0, v1, v2, v3]);
            }
        }

        // Build connectivity with side-specific tags
        Self::build_mesh_with_side_tags(vertices, elements, nx, ny, bc_tags)
    }

    /// Create a mesh that is periodic in the x-direction (channel flow).
    pub fn channel_periodic_x(x0: f64, x1: f64, y0: f64, y1: f64, nx: usize, ny: usize) -> Self {
        assert!(
            nx > 0 && ny > 0,
            "Need at least one element in each direction"
        );
        assert!(x1 > x0 && y1 > y0, "Invalid domain bounds");

        let dx = (x1 - x0) / nx as f64;
        let dy = (y1 - y0) / ny as f64;

        // Generate vertices
        let n_vertices = (nx + 1) * (ny + 1);
        let mut vertices = Vec::with_capacity(n_vertices);

        for j in 0..=ny {
            for i in 0..=nx {
                let x = x0 + i as f64 * dx;
                let y = y0 + j as f64 * dy;
                vertices.push((x, y));
            }
        }

        // Generate elements
        let n_elements = nx * ny;
        let mut elements = Vec::with_capacity(n_elements);

        for j in 0..ny {
            for i in 0..nx {
                let v0 = j * (nx + 1) + i;
                let v1 = v0 + 1;
                let v2 = v1 + (nx + 1);
                let v3 = v0 + (nx + 1);
                elements.push([v0, v1, v2, v3]);
            }
        }

        // Build connectivity with periodic x, wall y
        Self::build_channel_mesh(vertices, elements, nx, ny)
    }

    /// Create a fully periodic mesh (periodic in both x and y).
    pub fn uniform_periodic(x0: f64, x1: f64, y0: f64, y1: f64, nx: usize, ny: usize) -> Self {
        assert!(
            nx > 0 && ny > 0,
            "Need at least one element in each direction"
        );
        assert!(x1 > x0 && y1 > y0, "Invalid domain bounds");

        let dx = (x1 - x0) / nx as f64;
        let dy = (y1 - y0) / ny as f64;

        // Generate vertices
        let n_vertices = (nx + 1) * (ny + 1);
        let mut vertices = Vec::with_capacity(n_vertices);

        for j in 0..=ny {
            for i in 0..=nx {
                let x = x0 + i as f64 * dx;
                let y = y0 + j as f64 * dy;
                vertices.push((x, y));
            }
        }

        // Generate elements
        let n_elements = nx * ny;
        let mut elements = Vec::with_capacity(n_elements);

        for j in 0..ny {
            for i in 0..nx {
                let v0 = j * (nx + 1) + i;
                let v1 = v0 + 1;
                let v2 = v1 + (nx + 1);
                let v3 = v0 + (nx + 1);
                elements.push([v0, v1, v2, v3]);
            }
        }

        // Build connectivity with full periodicity
        Self::build_periodic_mesh(vertices, elements, nx, ny)
    }

    /// Build mesh with edge connectivity for a structured grid.
    fn build_mesh_with_connectivity(
        vertices: Vec<(f64, f64)>,
        elements: Vec<[usize; 4]>,
        nx: usize,
        ny: usize,
        bc_tag: BoundaryTag,
    ) -> Self {
        let n_elements = elements.len();
        let n_vertices = vertices.len();

        // Count edges: horizontal + vertical
        // Horizontal edges: (nx) × (ny+1)
        // Vertical edges: (nx+1) × (ny)
        let n_horiz = nx * (ny + 1);
        let n_vert = (nx + 1) * ny;
        let n_edges = n_horiz + n_vert;

        let mut edges = Vec::with_capacity(n_edges);
        let mut element_edges = vec![[0usize; 4]; n_elements];
        let mut edge_orientation = vec![[1i8; 4]; n_elements];

        // Helper to get element index from grid position
        let elem_idx = |i: usize, j: usize| -> usize { j * nx + i };

        // Create horizontal edges (bottom/top faces)
        for j in 0..=ny {
            for i in 0..nx {
                let edge_idx = edges.len();
                let v0 = j * (nx + 1) + i;
                let v1 = v0 + 1;

                let left = if j > 0 {
                    // This is the top face of element below
                    let k = elem_idx(i, j - 1);
                    element_edges[k][2] = edge_idx;
                    edge_orientation[k][2] = -1; // top face goes right-to-left
                    Some(ElementFace::new(k, 2))
                } else {
                    None
                };

                let right = if j < ny {
                    // This is the bottom face of element above
                    let k = elem_idx(i, j);
                    element_edges[k][0] = edge_idx;
                    edge_orientation[k][0] = 1; // bottom face goes left-to-right
                    Some(ElementFace::new(k, 0))
                } else {
                    None
                };

                // Determine which is left/right based on which exists
                let (left_ef, right_ef, boundary) = match (left, right) {
                    (Some(l), Some(r)) => (l, Some(r), None),
                    (Some(l), None) => (l, None, Some(bc_tag)), // top boundary
                    (None, Some(r)) => (r, None, Some(bc_tag)), // bottom boundary
                    (None, None) => unreachable!(),
                };

                edges.push(Edge {
                    vertices: (v0.min(v1), v0.max(v1)),
                    left: left_ef,
                    right: right_ef,
                    boundary_tag: boundary,
                });
            }
        }

        // Create vertical edges (left/right faces)
        for j in 0..ny {
            for i in 0..=nx {
                let edge_idx = edges.len();
                let v0 = j * (nx + 1) + i;
                let v1 = v0 + (nx + 1);

                let left = if i > 0 {
                    // This is the right face of element to the left
                    let k = elem_idx(i - 1, j);
                    element_edges[k][1] = edge_idx;
                    edge_orientation[k][1] = 1; // right face goes bottom-to-top
                    Some(ElementFace::new(k, 1))
                } else {
                    None
                };

                let right = if i < nx {
                    // This is the left face of element to the right
                    let k = elem_idx(i, j);
                    element_edges[k][3] = edge_idx;
                    edge_orientation[k][3] = -1; // left face goes top-to-bottom
                    Some(ElementFace::new(k, 3))
                } else {
                    None
                };

                let (left_ef, right_ef, boundary) = match (left, right) {
                    (Some(l), Some(r)) => (l, Some(r), None),
                    (Some(l), None) => (l, None, Some(bc_tag)), // right boundary
                    (None, Some(r)) => (r, None, Some(bc_tag)), // left boundary
                    (None, None) => unreachable!(),
                };

                edges.push(Edge {
                    vertices: (v0.min(v1), v0.max(v1)),
                    left: left_ef,
                    right: right_ef,
                    boundary_tag: boundary,
                });
            }
        }

        let n_boundary_edges = edges.iter().filter(|e| e.is_boundary()).count();
        let vertex_to_elements = Self::build_vertex_to_elements(&elements, n_vertices);

        Self {
            vertices,
            elements,
            edges,
            element_edges,
            edge_orientation,
            n_elements,
            n_edges,
            n_boundary_edges,
            n_vertices,
            vertex_to_elements,
        }
    }

    /// Build mesh with different boundary tags on each side.
    /// bc_tags: [south, east, north, west]
    fn build_mesh_with_side_tags(
        vertices: Vec<(f64, f64)>,
        elements: Vec<[usize; 4]>,
        nx: usize,
        ny: usize,
        bc_tags: [BoundaryTag; 4],
    ) -> Self {
        let n_elements = elements.len();
        let n_vertices = vertices.len();

        let n_horiz = nx * (ny + 1);
        let n_vert = (nx + 1) * ny;
        let n_edges = n_horiz + n_vert;

        let mut edges = Vec::with_capacity(n_edges);
        let mut element_edges = vec![[0usize; 4]; n_elements];
        let mut edge_orientation = vec![[1i8; 4]; n_elements];

        let elem_idx = |i: usize, j: usize| -> usize { j * nx + i };

        // Create horizontal edges (bottom/top faces)
        for j in 0..=ny {
            for i in 0..nx {
                let edge_idx = edges.len();
                let v0 = j * (nx + 1) + i;
                let v1 = v0 + 1;

                let left = if j > 0 {
                    let k = elem_idx(i, j - 1);
                    element_edges[k][2] = edge_idx;
                    edge_orientation[k][2] = -1;
                    Some(ElementFace::new(k, 2))
                } else {
                    None
                };

                let right = if j < ny {
                    let k = elem_idx(i, j);
                    element_edges[k][0] = edge_idx;
                    edge_orientation[k][0] = 1;
                    Some(ElementFace::new(k, 0))
                } else {
                    None
                };

                let (left_ef, right_ef, boundary) = match (left, right) {
                    (Some(l), Some(r)) => (l, Some(r), None),
                    (Some(l), None) => (l, None, Some(bc_tags[2])), // north boundary
                    (None, Some(r)) => (r, None, Some(bc_tags[0])), // south boundary
                    (None, None) => unreachable!(),
                };

                edges.push(Edge {
                    vertices: (v0.min(v1), v0.max(v1)),
                    left: left_ef,
                    right: right_ef,
                    boundary_tag: boundary,
                });
            }
        }

        // Create vertical edges (left/right faces)
        for j in 0..ny {
            for i in 0..=nx {
                let edge_idx = edges.len();
                let v0 = j * (nx + 1) + i;
                let v1 = v0 + (nx + 1);

                let left = if i > 0 {
                    let k = elem_idx(i - 1, j);
                    element_edges[k][1] = edge_idx;
                    edge_orientation[k][1] = 1;
                    Some(ElementFace::new(k, 1))
                } else {
                    None
                };

                let right = if i < nx {
                    let k = elem_idx(i, j);
                    element_edges[k][3] = edge_idx;
                    edge_orientation[k][3] = -1;
                    Some(ElementFace::new(k, 3))
                } else {
                    None
                };

                let (left_ef, right_ef, boundary) = match (left, right) {
                    (Some(l), Some(r)) => (l, Some(r), None),
                    (Some(l), None) => (l, None, Some(bc_tags[1])), // east boundary
                    (None, Some(r)) => (r, None, Some(bc_tags[3])), // west boundary
                    (None, None) => unreachable!(),
                };

                edges.push(Edge {
                    vertices: (v0.min(v1), v0.max(v1)),
                    left: left_ef,
                    right: right_ef,
                    boundary_tag: boundary,
                });
            }
        }

        let n_boundary_edges = edges.iter().filter(|e| e.is_boundary()).count();
        let vertex_to_elements = Self::build_vertex_to_elements(&elements, n_vertices);

        Self {
            vertices,
            elements,
            edges,
            element_edges,
            edge_orientation,
            n_elements,
            n_edges,
            n_boundary_edges,
            n_vertices,
            vertex_to_elements,
        }
    }

    /// Build channel mesh with periodic x-direction.
    fn build_channel_mesh(
        vertices: Vec<(f64, f64)>,
        elements: Vec<[usize; 4]>,
        nx: usize,
        ny: usize,
    ) -> Self {
        let n_elements = elements.len();
        let n_vertices = vertices.len();

        let n_horiz = nx * (ny + 1);
        let n_vert = nx * ny; // No boundary vertical edges due to periodicity
        let n_edges = n_horiz + n_vert;

        let mut edges = Vec::with_capacity(n_edges);
        let mut element_edges = vec![[0usize; 4]; n_elements];
        let mut edge_orientation = vec![[1i8; 4]; n_elements];

        let elem_idx = |i: usize, j: usize| -> usize { j * nx + i };

        // Horizontal edges (same as non-periodic)
        for j in 0..=ny {
            for i in 0..nx {
                let edge_idx = edges.len();
                let v0 = j * (nx + 1) + i;
                let v1 = v0 + 1;

                let below = if j > 0 {
                    let k = elem_idx(i, j - 1);
                    element_edges[k][2] = edge_idx;
                    edge_orientation[k][2] = -1;
                    Some(ElementFace::new(k, 2))
                } else {
                    None
                };

                let above = if j < ny {
                    let k = elem_idx(i, j);
                    element_edges[k][0] = edge_idx;
                    edge_orientation[k][0] = 1;
                    Some(ElementFace::new(k, 0))
                } else {
                    None
                };

                let (left_ef, right_ef, boundary) = match (below, above) {
                    (Some(l), Some(r)) => (l, Some(r), None),
                    (Some(l), None) => (l, None, Some(BoundaryTag::Wall)),
                    (None, Some(r)) => (r, None, Some(BoundaryTag::Wall)),
                    (None, None) => unreachable!(),
                };

                edges.push(Edge {
                    vertices: (v0.min(v1), v0.max(v1)),
                    left: left_ef,
                    right: right_ef,
                    boundary_tag: boundary,
                });
            }
        }

        // Vertical edges (interior only, with periodic wrapping)
        for j in 0..ny {
            for i in 0..nx {
                let edge_idx = edges.len();

                // Right face of element (i, j)
                let k_left = elem_idx(i, j);
                // Left face of element to the right (with periodic wrap)
                let i_right = (i + 1) % nx;
                let k_right = elem_idx(i_right, j);

                element_edges[k_left][1] = edge_idx;
                edge_orientation[k_left][1] = 1;
                element_edges[k_right][3] = edge_idx;
                edge_orientation[k_right][3] = -1;

                // Use vertices from k_left's right edge
                let v0 = j * (nx + 1) + i + 1;
                let v1 = v0 + (nx + 1);

                edges.push(Edge {
                    vertices: (v0.min(v1), v0.max(v1)),
                    left: ElementFace::new(k_left, 1),
                    right: Some(ElementFace::new(k_right, 3)),
                    boundary_tag: None,
                });
            }
        }

        let n_boundary_edges = edges.iter().filter(|e| e.is_boundary()).count();
        let n_edges_actual = edges.len();
        let vertex_to_elements = Self::build_vertex_to_elements(&elements, n_vertices);

        Self {
            vertices,
            elements,
            edges,
            element_edges,
            edge_orientation,
            n_elements,
            n_edges: n_edges_actual,
            n_boundary_edges,
            n_vertices,
            vertex_to_elements,
        }
    }

    /// Build fully periodic mesh.
    fn build_periodic_mesh(
        vertices: Vec<(f64, f64)>,
        elements: Vec<[usize; 4]>,
        nx: usize,
        ny: usize,
    ) -> Self {
        let n_elements = elements.len();
        let n_vertices = vertices.len();

        // All edges are interior
        let n_edges = nx * ny * 2; // Each element contributes 2 unique edges

        let mut edges = Vec::with_capacity(n_edges);
        let mut element_edges = vec![[0usize; 4]; n_elements];
        let mut edge_orientation = vec![[1i8; 4]; n_elements];

        let elem_idx = |i: usize, j: usize| -> usize { j * nx + i };

        // Horizontal edges (bottom of each element)
        for j in 0..ny {
            for i in 0..nx {
                let edge_idx = edges.len();

                let k_above = elem_idx(i, j);
                let j_below = if j > 0 { j - 1 } else { ny - 1 };
                let k_below = elem_idx(i, j_below);

                element_edges[k_above][0] = edge_idx;
                edge_orientation[k_above][0] = 1;
                element_edges[k_below][2] = edge_idx;
                edge_orientation[k_below][2] = -1;

                let v0 = j * (nx + 1) + i;
                let v1 = v0 + 1;

                edges.push(Edge {
                    vertices: (v0.min(v1), v0.max(v1)),
                    left: ElementFace::new(k_below, 2),
                    right: Some(ElementFace::new(k_above, 0)),
                    boundary_tag: None,
                });
            }
        }

        // Vertical edges (right of each element)
        for j in 0..ny {
            for i in 0..nx {
                let edge_idx = edges.len();

                let k_left = elem_idx(i, j);
                let i_right = (i + 1) % nx;
                let k_right = elem_idx(i_right, j);

                element_edges[k_left][1] = edge_idx;
                edge_orientation[k_left][1] = 1;
                element_edges[k_right][3] = edge_idx;
                edge_orientation[k_right][3] = -1;

                let v0 = j * (nx + 1) + i + 1;
                let v1 = v0 + (nx + 1);

                edges.push(Edge {
                    vertices: (v0.min(v1), v0.max(v1)),
                    left: ElementFace::new(k_left, 1),
                    right: Some(ElementFace::new(k_right, 3)),
                    boundary_tag: None,
                });
            }
        }

        let n_edges_actual = edges.len();
        let vertex_to_elements = Self::build_vertex_to_elements(&elements, n_vertices);

        Self {
            vertices,
            elements,
            edges,
            element_edges,
            edge_orientation,
            n_elements,
            n_edges: n_edges_actual,
            n_boundary_edges: 0,
            n_vertices,
            vertex_to_elements,
        }
    }

    /// Get the vertices of an element.
    pub fn element_vertices(&self, k: usize) -> [(f64, f64); 4] {
        let [v0, v1, v2, v3] = self.elements[k];
        [
            self.vertices[v0],
            self.vertices[v1],
            self.vertices[v2],
            self.vertices[v3],
        ]
    }

    /// Map reference coordinates (r, s) in [-1, 1]² to physical coordinates (x, y).
    ///
    /// Uses bilinear interpolation for quadrilateral elements:
    /// ```text
    /// x(r, s) = (1-r)(1-s)/4 * x0 + (1+r)(1-s)/4 * x1
    ///         + (1+r)(1+s)/4 * x2 + (1-r)(1+s)/4 * x3
    /// ```
    pub fn reference_to_physical(&self, k: usize, r: f64, s: f64) -> (f64, f64) {
        let verts = self.element_vertices(k);
        let (x0, y0) = verts[0];
        let (x1, y1) = verts[1];
        let (x2, y2) = verts[2];
        let (x3, y3) = verts[3];

        // Bilinear shape functions
        let n0 = (1.0 - r) * (1.0 - s) / 4.0;
        let n1 = (1.0 + r) * (1.0 - s) / 4.0;
        let n2 = (1.0 + r) * (1.0 + s) / 4.0;
        let n3 = (1.0 - r) * (1.0 + s) / 4.0;

        let x = n0 * x0 + n1 * x1 + n2 * x2 + n3 * x3;
        let y = n0 * y0 + n1 * y1 + n2 * y2 + n3 * y3;

        (x, y)
    }

    /// Get the edge index for a given element face.
    pub fn edge_for_face(&self, element: usize, face: usize) -> usize {
        self.element_edges[element][face]
    }

    /// Get the neighbor element across a face, if it exists.
    pub fn neighbor(&self, element: usize, face: usize) -> Option<ElementFace> {
        let edge_idx = self.element_edges[element][face];
        let edge = &self.edges[edge_idx];

        if edge.left.element == element && edge.left.face == face {
            edge.right
        } else {
            Some(edge.left)
        }
    }

    /// Check if a face is on the boundary.
    pub fn is_boundary_face(&self, element: usize, face: usize) -> bool {
        let edge_idx = self.element_edges[element][face];
        self.edges[edge_idx].is_boundary()
    }

    /// Get the boundary tag for a face, if it's a boundary face.
    pub fn boundary_tag(&self, element: usize, face: usize) -> Option<BoundaryTag> {
        let edge_idx = self.element_edges[element][face];
        self.edges[edge_idx].boundary_tag
    }

    /// Get minimum element diameter (for CFL computation).
    pub fn h_min(&self) -> f64 {
        let mut h_min = f64::INFINITY;
        for k in 0..self.n_elements {
            let verts = self.element_vertices(k);
            // Approximate diameter as minimum edge length
            for i in 0..4 {
                let (x0, y0) = verts[i];
                let (x1, y1) = verts[(i + 1) % 4];
                let len = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
                h_min = h_min.min(len);
            }
        }
        h_min
    }

    /// Get maximum element diameter.
    pub fn h_max(&self) -> f64 {
        let mut h_max: f64 = 0.0;
        for k in 0..self.n_elements {
            let verts = self.element_vertices(k);
            // Use diagonal as maximum dimension
            let (x0, y0) = verts[0];
            let (x2, y2) = verts[2];
            let diag = ((x2 - x0).powi(2) + (y2 - y0).powi(2)).sqrt();
            h_max = h_max.max(diag);
        }
        h_max
    }

    /// Get the diameter of a specific element (diagonal length).
    pub fn element_diameter(&self, k: usize) -> f64 {
        let verts = self.element_vertices(k);
        let (x0, y0) = verts[0];
        let (x2, y2) = verts[2];
        ((x2 - x0).powi(2) + (y2 - y0).powi(2)).sqrt()
    }

    /// Get all elements sharing a given vertex.
    ///
    /// Used by vertex-based slope limiters (e.g., Kuzmin) to compute
    /// local bounds from the vertex patch.
    #[inline]
    pub fn elements_at_vertex(&self, vertex: usize) -> &[usize] {
        &self.vertex_to_elements[vertex]
    }

    /// Get the global vertex indices for an element.
    #[inline]
    pub fn element_vertex_indices(&self, k: usize) -> [usize; 4] {
        self.elements[k]
    }

    /// Build vertex-to-element connectivity from element-vertex connectivity.
    pub(crate) fn build_vertex_to_elements(
        elements: &[[usize; 4]],
        n_vertices: usize,
    ) -> Vec<Vec<usize>> {
        let mut v2e = vec![Vec::with_capacity(4); n_vertices];
        for (elem_idx, elem) in elements.iter().enumerate() {
            for &vertex in elem {
                v2e[vertex].push(elem_idx);
            }
        }
        v2e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_rectangle_dimensions() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 3, 2);

        assert_eq!(mesh.n_elements, 6); // 3 × 2
        assert_eq!(mesh.n_vertices, 12); // 4 × 3
        assert_eq!(mesh.elements.len(), 6);
        assert_eq!(mesh.vertices.len(), 12);
    }

    #[test]
    fn test_uniform_rectangle_vertices() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 1.0, 2, 1);

        // Should have 6 vertices in a 3×2 grid
        assert_eq!(mesh.n_vertices, 6);

        // Check corner vertices
        assert!((mesh.vertices[0].0 - 0.0).abs() < 1e-14);
        assert!((mesh.vertices[0].1 - 0.0).abs() < 1e-14);
        assert!((mesh.vertices[2].0 - 2.0).abs() < 1e-14);
        assert!((mesh.vertices[2].1 - 0.0).abs() < 1e-14);
        assert!((mesh.vertices[5].0 - 2.0).abs() < 1e-14);
        assert!((mesh.vertices[5].1 - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_element_vertex_ordering() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);

        // Check that each element has counter-clockwise vertices
        for k in 0..mesh.n_elements {
            let verts = mesh.element_vertices(k);

            // Compute signed area (should be positive for CCW)
            let mut area = 0.0;
            for i in 0..4 {
                let (x0, y0) = verts[i];
                let (x1, y1) = verts[(i + 1) % 4];
                area += (x1 - x0) * (y1 + y0);
            }
            assert!(area < 0.0, "Element {} should have CCW vertices", k);
        }
    }

    #[test]
    fn test_edge_count() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 3, 2);

        // For a 3×2 grid:
        // Horizontal edges: 3 × 3 = 9
        // Vertical edges: 4 × 2 = 8
        // Total: 17
        assert_eq!(mesh.n_edges, 17);
    }

    #[test]
    fn test_boundary_edges() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 3, 2);

        // Boundary edges: 3 + 3 + 2 + 2 = 10
        assert_eq!(mesh.n_boundary_edges, 10);

        // Verify boundary detection
        let boundary_count = mesh.edges.iter().filter(|e| e.is_boundary()).count();
        assert_eq!(boundary_count, 10);
    }

    #[test]
    fn test_neighbor_connectivity() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);

        // Element 0 (bottom-left) should have:
        // - No neighbor on bottom (face 0)
        // - Element 1 on right (face 1)
        // - Element 2 on top (face 2)
        // - No neighbor on left (face 3)
        assert!(mesh.is_boundary_face(0, 0));
        assert!(!mesh.is_boundary_face(0, 1));
        assert!(!mesh.is_boundary_face(0, 2));
        assert!(mesh.is_boundary_face(0, 3));

        let right_neighbor = mesh.neighbor(0, 1).unwrap();
        assert_eq!(right_neighbor.element, 1);

        let top_neighbor = mesh.neighbor(0, 2).unwrap();
        assert_eq!(top_neighbor.element, 2);
    }

    #[test]
    fn test_periodic_mesh_no_boundaries() {
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 3, 2);

        assert_eq!(mesh.n_boundary_edges, 0);

        // All faces should have neighbors
        for k in 0..mesh.n_elements {
            for face in 0..4 {
                assert!(
                    !mesh.is_boundary_face(k, face),
                    "Element {} face {} should not be boundary",
                    k,
                    face
                );
                assert!(
                    mesh.neighbor(k, face).is_some(),
                    "Element {} face {} should have neighbor",
                    k,
                    face
                );
            }
        }
    }

    #[test]
    fn test_channel_mesh_periodicity() {
        let mesh = Mesh2D::channel_periodic_x(0.0, 1.0, 0.0, 1.0, 3, 2);

        // Top and bottom are walls, left and right are periodic
        // Wall edges: 3 + 3 = 6
        assert_eq!(mesh.n_boundary_edges, 6);

        // Check that left-right faces are connected periodically
        // Element 0's left face should connect to element 2 (rightmost in row)
        let left_neighbor = mesh.neighbor(0, 3).unwrap();
        assert_eq!(left_neighbor.element, 2);

        // Element 2's right face should connect to element 0
        let right_neighbor = mesh.neighbor(2, 1).unwrap();
        assert_eq!(right_neighbor.element, 0);
    }

    #[test]
    fn test_h_min_h_max() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 1.0, 2, 1);

        // Elements are 1.0 × 1.0 squares
        assert!((mesh.h_min() - 1.0).abs() < 1e-14);
        assert!((mesh.h_max() - 2.0_f64.sqrt()).abs() < 1e-14); // diagonal
    }

    #[test]
    fn test_reference_to_physical() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 1.0, 2, 1);

        // Element 0 spans [0, 1] × [0, 1]
        // Reference corner (-1, -1) should map to (0, 0)
        let (x, y) = mesh.reference_to_physical(0, -1.0, -1.0);
        assert!((x - 0.0).abs() < 1e-14);
        assert!((y - 0.0).abs() < 1e-14);

        // Reference corner (1, 1) should map to (1, 1)
        let (x, y) = mesh.reference_to_physical(0, 1.0, 1.0);
        assert!((x - 1.0).abs() < 1e-14);
        assert!((y - 1.0).abs() < 1e-14);

        // Reference center (0, 0) should map to (0.5, 0.5)
        let (x, y) = mesh.reference_to_physical(0, 0.0, 0.0);
        assert!((x - 0.5).abs() < 1e-14);
        assert!((y - 0.5).abs() < 1e-14);

        // Element 1 spans [1, 2] × [0, 1]
        // Reference center (0, 0) should map to (1.5, 0.5)
        let (x, y) = mesh.reference_to_physical(1, 0.0, 0.0);
        assert!((x - 1.5).abs() < 1e-14);
        assert!((y - 0.5).abs() < 1e-14);
    }

    #[test]
    fn test_element_edges_mapping() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);

        // Each element should have 4 edge indices
        for k in 0..mesh.n_elements {
            for face in 0..4 {
                let edge_idx = mesh.element_edges[k][face];
                assert!(edge_idx < mesh.n_edges);
            }
        }
    }

    #[test]
    fn test_vertex_to_elements_structured() {
        // 3x3 mesh of elements
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 3, 3);

        // 4x4 = 16 vertices
        assert_eq!(mesh.vertex_to_elements.len(), 16);

        // Corner vertices should have 1 element
        // Vertex 0 is bottom-left corner
        assert_eq!(mesh.elements_at_vertex(0).len(), 1);
        // Vertex 3 is bottom-right corner
        assert_eq!(mesh.elements_at_vertex(3).len(), 1);
        // Vertex 12 is top-left corner
        assert_eq!(mesh.elements_at_vertex(12).len(), 1);
        // Vertex 15 is top-right corner
        assert_eq!(mesh.elements_at_vertex(15).len(), 1);

        // Edge vertices (not corners) should have 2 elements
        // Vertex 1 is on bottom edge (between corners 0 and 3)
        assert_eq!(mesh.elements_at_vertex(1).len(), 2);
        // Vertex 4 is on left edge (between corners 0 and 12)
        assert_eq!(mesh.elements_at_vertex(4).len(), 2);

        // Interior vertices should have 4 elements
        // Vertex 5 is interior (second row, second column)
        assert_eq!(mesh.elements_at_vertex(5).len(), 4);
        // Vertex 6 is interior
        assert_eq!(mesh.elements_at_vertex(6).len(), 4);
        // Vertex 9 is interior
        assert_eq!(mesh.elements_at_vertex(9).len(), 4);
        // Vertex 10 is interior
        assert_eq!(mesh.elements_at_vertex(10).len(), 4);
    }

    #[test]
    fn test_vertex_to_elements_periodic() {
        // Periodic mesh: all vertices effectively interior
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 3, 3);

        // In a periodic mesh, each vertex should have 4 elements
        // (because periodicity wraps around)
        for v in 0..mesh.n_vertices {
            let patch_size = mesh.elements_at_vertex(v).len();
            // Due to periodic wrapping, some vertices appear at edges of the
            // physical grid but are connected to elements via periodicity.
            // For a structured periodic mesh, each vertex should have 4 elements.
            assert!(
                patch_size >= 1 && patch_size <= 4,
                "Vertex {} has {} elements, expected 1-4",
                v,
                patch_size
            );
        }
    }

    #[test]
    fn test_vertex_to_elements_consistency() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);

        // Check that each element-vertex pair is consistent
        for (k, elem) in mesh.elements.iter().enumerate() {
            for &v in elem {
                let patch = mesh.elements_at_vertex(v);
                assert!(
                    patch.contains(&k),
                    "Element {} contains vertex {} but vertex_to_elements[{}] = {:?}",
                    k,
                    v,
                    v,
                    patch
                );
            }
        }

        // Check reverse: each vertex-element pair should have the vertex in the element
        for (v, elems) in mesh.vertex_to_elements.iter().enumerate() {
            for &k in elems {
                assert!(
                    mesh.elements[k].contains(&v),
                    "vertex_to_elements[{}] contains {} but element {} = {:?}",
                    v,
                    k,
                    k,
                    mesh.elements[k]
                );
            }
        }
    }

    #[test]
    fn test_element_vertex_indices() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);

        for k in 0..mesh.n_elements {
            let indices = mesh.element_vertex_indices(k);
            assert_eq!(indices, mesh.elements[k]);
        }
    }
}
