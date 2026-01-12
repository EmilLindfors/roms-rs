//! Gmsh mesh file I/O.
//!
//! Supports reading and writing Gmsh MSH format version 2.2 (ASCII).
//! This is the most widely supported format for Gmsh meshes.
//!
//! ## Supported Element Types
//! - 3 = Quadrilateral (4-node)
//! - 1 = Line (2-node, for boundary edges)
//!
//! ## Example
//! ```no_run
//! use dg_rs::mesh::gmsh::read_gmsh_mesh;
//! use std::path::Path;
//!
//! let mesh = read_gmsh_mesh(Path::new("mesh.msh")).expect("Failed to read mesh");
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use thiserror::Error;

use super::mesh2d::{Edge, ElementFace, Mesh2D};
use crate::mesh::BoundaryTag;

/// Error type for Gmsh I/O operations.
#[derive(Debug, Error)]
pub enum GmshError {
    /// File could not be opened.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Invalid file format.
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Unsupported mesh format version.
    #[error("Unsupported Gmsh version: {0}")]
    UnsupportedVersion(String),

    /// Unsupported element type.
    #[error("Unsupported element type: {0}")]
    UnsupportedElement(i32),

    /// Missing required section.
    #[error("Missing section: {0}")]
    MissingSection(String),
}

/// Gmsh element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GmshElementType {
    Line = 1,
    Triangle = 2,
    Quadrilateral = 3,
}

impl TryFrom<i32> for GmshElementType {
    type Error = GmshError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(GmshElementType::Line),
            2 => Ok(GmshElementType::Triangle),
            3 => Ok(GmshElementType::Quadrilateral),
            _ => Err(GmshError::UnsupportedElement(value)),
        }
    }
}

/// Read a Gmsh MSH file (format 2.2).
///
/// # Arguments
/// * `path` - Path to the MSH file
///
/// # Returns
/// * `Ok(Mesh2D)` - The parsed mesh
/// * `Err(GmshError)` - If reading or parsing fails
pub fn read_gmsh_mesh(path: &Path) -> Result<Mesh2D, GmshError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();
    let mut vertices: Vec<(f64, f64)> = Vec::new();
    let mut elements: Vec<[usize; 4]> = Vec::new();
    let mut boundary_line_tags: HashMap<(usize, usize), BoundaryTag> = HashMap::new();

    // Parse sections
    while let Some(line_result) = lines.next() {
        let line = line_result?;
        let line = line.trim();

        if line.starts_with("$MeshFormat") {
            parse_mesh_format(&mut lines)?;
        } else if line.starts_with("$Nodes") {
            vertices = parse_nodes(&mut lines)?;
        } else if line.starts_with("$Elements") {
            let (quads, line_tags) = parse_elements(&mut lines)?;
            elements = quads;
            boundary_line_tags = line_tags;
        }
    }

    if vertices.is_empty() {
        return Err(GmshError::MissingSection("Nodes".to_string()));
    }
    if elements.is_empty() {
        return Err(GmshError::MissingSection(
            "Elements (quadrilaterals)".to_string(),
        ));
    }

    // Build mesh from vertices and elements
    let mesh = build_mesh_from_gmsh(vertices, elements, boundary_line_tags)?;
    Ok(mesh)
}

/// Parse the $MeshFormat section.
fn parse_mesh_format<I>(lines: &mut I) -> Result<(), GmshError>
where
    I: Iterator<Item = std::io::Result<String>>,
{
    if let Some(line_result) = lines.next() {
        let line = line_result?;
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        if parts.is_empty() {
            return Err(GmshError::ParseError("Empty MeshFormat line".to_string()));
        }

        let version = parts[0];
        if !version.starts_with("2.") && !version.starts_with("4.") {
            return Err(GmshError::UnsupportedVersion(version.to_string()));
        }

        // Skip to end of section
        for line_result in lines.by_ref() {
            let line = line_result?;
            if line.trim().starts_with("$EndMeshFormat") {
                break;
            }
        }
    }
    Ok(())
}

/// Parse the $Nodes section.
fn parse_nodes<I>(lines: &mut I) -> Result<Vec<(f64, f64)>, GmshError>
where
    I: Iterator<Item = std::io::Result<String>>,
{
    let n_nodes = match lines.next() {
        Some(Ok(line)) => line
            .trim()
            .parse::<usize>()
            .map_err(|_| GmshError::ParseError("Invalid node count".to_string()))?,
        _ => return Err(GmshError::ParseError("Missing node count".to_string())),
    };

    let mut vertices = Vec::with_capacity(n_nodes);

    for _ in 0..n_nodes {
        if let Some(line_result) = lines.next() {
            let line = line_result?;
            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            if parts.len() < 4 {
                return Err(GmshError::ParseError(format!(
                    "Invalid node line: {}",
                    line
                )));
            }

            // Format: node_id x y z
            let x: f64 = parts[1].parse().map_err(|_| {
                GmshError::ParseError(format!("Invalid x coordinate: {}", parts[1]))
            })?;
            let y: f64 = parts[2].parse().map_err(|_| {
                GmshError::ParseError(format!("Invalid y coordinate: {}", parts[2]))
            })?;

            vertices.push((x, y));
        }
    }

    // Skip to end of section
    for line_result in lines.by_ref() {
        let line = line_result?;
        if line.trim().starts_with("$EndNodes") {
            break;
        }
    }

    Ok(vertices)
}

/// Parse the $Elements section.
///
/// Returns (quadrilaterals, boundary_line_tags).
fn parse_elements<I>(
    lines: &mut I,
) -> Result<(Vec<[usize; 4]>, HashMap<(usize, usize), BoundaryTag>), GmshError>
where
    I: Iterator<Item = std::io::Result<String>>,
{
    let n_elements = match lines.next() {
        Some(Ok(line)) => line
            .trim()
            .parse::<usize>()
            .map_err(|_| GmshError::ParseError("Invalid element count".to_string()))?,
        _ => return Err(GmshError::ParseError("Missing element count".to_string())),
    };

    let mut quads = Vec::new();
    let mut boundary_line_tags: HashMap<(usize, usize), BoundaryTag> = HashMap::new();

    for _ in 0..n_elements {
        if let Some(line_result) = lines.next() {
            let line = line_result?;
            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            if parts.len() < 3 {
                return Err(GmshError::ParseError(format!(
                    "Invalid element line: {}",
                    line
                )));
            }

            // Format: elem_id elem_type n_tags tag1 ... tagN node1 node2 ...
            let elem_type: i32 = parts[1].parse().map_err(|_| {
                GmshError::ParseError(format!("Invalid element type: {}", parts[1]))
            })?;
            let n_tags: usize = parts[2]
                .parse()
                .map_err(|_| GmshError::ParseError(format!("Invalid tag count: {}", parts[2])))?;

            // Get physical tag (first tag) for boundary condition
            let physical_tag = if n_tags > 0 && parts.len() > 3 {
                parts[3].parse::<u32>().unwrap_or(0)
            } else {
                0
            };

            let node_start = 3 + n_tags;
            let elem_type = GmshElementType::try_from(elem_type);

            match elem_type {
                Ok(GmshElementType::Quadrilateral) => {
                    if parts.len() < node_start + 4 {
                        return Err(GmshError::ParseError(
                            "Quadrilateral needs 4 nodes".to_string(),
                        ));
                    }
                    // Gmsh uses 1-based indexing
                    let n0: usize = parts[node_start].parse::<usize>().unwrap() - 1;
                    let n1: usize = parts[node_start + 1].parse::<usize>().unwrap() - 1;
                    let n2: usize = parts[node_start + 2].parse::<usize>().unwrap() - 1;
                    let n3: usize = parts[node_start + 3].parse::<usize>().unwrap() - 1;
                    quads.push([n0, n1, n2, n3]);
                }
                Ok(GmshElementType::Line) => {
                    if parts.len() < node_start + 2 {
                        return Err(GmshError::ParseError("Line needs 2 nodes".to_string()));
                    }
                    let n0: usize = parts[node_start].parse::<usize>().unwrap() - 1;
                    let n1: usize = parts[node_start + 1].parse::<usize>().unwrap() - 1;

                    // Convert physical tag to boundary tag
                    let tag = physical_tag_to_boundary_tag(physical_tag);
                    // Store as sorted tuple for consistent lookup
                    let edge = if n0 < n1 { (n0, n1) } else { (n1, n0) };
                    boundary_line_tags.insert(edge, tag);
                }
                Ok(GmshElementType::Triangle) => {
                    // We don't support triangles yet, skip silently
                }
                Err(_) => {
                    // Unknown element type, skip silently
                }
            }
        }
    }

    // Skip to end of section
    for line_result in lines.by_ref() {
        let line = line_result?;
        if line.trim().starts_with("$EndElements") {
            break;
        }
    }

    Ok((quads, boundary_line_tags))
}

/// Convert a Gmsh physical tag to a BoundaryTag.
fn physical_tag_to_boundary_tag(tag: u32) -> BoundaryTag {
    match tag {
        1 => BoundaryTag::Wall,
        2 => BoundaryTag::Open,
        3 => BoundaryTag::TidalForcing,
        4 => BoundaryTag::River,
        _ => BoundaryTag::Custom(tag),
    }
}

/// Convert a BoundaryTag to a Gmsh physical tag.
fn boundary_tag_to_physical(tag: &BoundaryTag) -> u32 {
    match tag {
        BoundaryTag::Wall => 1,
        BoundaryTag::Open => 2,
        BoundaryTag::TidalForcing => 3,
        BoundaryTag::River => 4,
        BoundaryTag::Periodic(_) => 0, // Periodic not supported in Gmsh export
        BoundaryTag::Dirichlet => 5,
        BoundaryTag::Neumann => 6,
        BoundaryTag::Custom(t) => *t,
    }
}

/// Build a Mesh2D from Gmsh data.
fn build_mesh_from_gmsh(
    vertices: Vec<(f64, f64)>,
    elements: Vec<[usize; 4]>,
    boundary_line_tags: HashMap<(usize, usize), BoundaryTag>,
) -> Result<Mesh2D, GmshError> {
    let n_elements = elements.len();
    let n_vertices = vertices.len();

    // Build edge connectivity: map from sorted vertex pair to edge info
    let mut edge_map: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();

    for (k, elem) in elements.iter().enumerate() {
        // Quadrilateral edges: 0-1 (bottom), 1-2 (right), 2-3 (top), 3-0 (left)
        let face_edges = [
            (elem[0], elem[1]),
            (elem[1], elem[2]),
            (elem[2], elem[3]),
            (elem[3], elem[0]),
        ];

        for (face, &(v0, v1)) in face_edges.iter().enumerate() {
            let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            edge_map.entry(key).or_default().push((k, face));
        }
    }

    // Build edge list and element_edges
    let n_edges = edge_map.len();
    let mut edges: Vec<Edge> = Vec::with_capacity(n_edges);
    let mut element_edges = vec![[0usize; 4]; n_elements];
    let mut edge_orientation = vec![[1i8; 4]; n_elements];

    for (key, elem_faces) in &edge_map {
        let edge_idx = edges.len();

        // First element-face is "left"
        let left = ElementFace::new(elem_faces[0].0, elem_faces[0].1);

        // Second element-face (if exists) is "right"
        let right = if elem_faces.len() > 1 {
            Some(ElementFace::new(elem_faces[1].0, elem_faces[1].1))
        } else {
            None
        };

        // Get boundary tag if this is a boundary edge
        let boundary_tag = if right.is_none() {
            boundary_line_tags
                .get(key)
                .cloned()
                .or(Some(BoundaryTag::Wall))
        } else {
            None
        };

        edges.push(Edge {
            vertices: *key,
            left,
            right,
            boundary_tag,
        });

        // Set element_edges and orientation
        for &(elem, face) in elem_faces {
            element_edges[elem][face] = edge_idx;

            // Determine orientation: +1 if element's face goes in same direction as edge
            let elem_nodes = elements[elem];
            let face_vertices = [
                (elem_nodes[0], elem_nodes[1]),
                (elem_nodes[1], elem_nodes[2]),
                (elem_nodes[2], elem_nodes[3]),
                (elem_nodes[3], elem_nodes[0]),
            ];
            let (v0, v1) = face_vertices[face];
            let sorted_key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            edge_orientation[elem][face] = if (v0, v1) == sorted_key { 1 } else { -1 };
        }
    }

    let n_boundary_edges = edges.iter().filter(|e| e.is_boundary()).count();
    let vertex_to_elements = Mesh2D::build_vertex_to_elements(&elements, n_vertices);

    Ok(Mesh2D {
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
    })
}

/// Write a Mesh2D to Gmsh MSH format 2.2.
///
/// # Arguments
/// * `mesh` - The mesh to write
/// * `path` - Output file path
pub fn write_gmsh_mesh(mesh: &Mesh2D, path: &Path) -> Result<(), GmshError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "$MeshFormat")?;
    writeln!(writer, "2.2 0 8")?;
    writeln!(writer, "$EndMeshFormat")?;

    // Write nodes
    writeln!(writer, "$Nodes")?;
    writeln!(writer, "{}", mesh.vertices.len())?;
    for (i, &(x, y)) in mesh.vertices.iter().enumerate() {
        writeln!(writer, "{} {} {} 0", i + 1, x, y)?;
    }
    writeln!(writer, "$EndNodes")?;

    // Count boundary edges
    let boundary_edges: Vec<_> = mesh.edges.iter().filter(|e| e.is_boundary()).collect();
    let n_total = boundary_edges.len() + mesh.n_elements;

    writeln!(writer, "$Elements")?;
    writeln!(writer, "{}", n_total)?;

    // Write boundary edges first
    let mut elem_id = 1;
    for edge in &boundary_edges {
        let (n0, n1) = edge.vertices;
        let physical_tag = edge
            .boundary_tag
            .as_ref()
            .map(boundary_tag_to_physical)
            .unwrap_or(1);
        // Format: elem_id type n_tags physical_tag geometrical_tag node1 node2
        writeln!(
            writer,
            "{} 1 2 {} {} {} {}",
            elem_id,
            physical_tag,
            physical_tag,
            n0 + 1,
            n1 + 1
        )?;
        elem_id += 1;
    }

    // Write quadrilaterals
    for elem in &mesh.elements {
        // Format: elem_id type n_tags physical_tag geometrical_tag node1 node2 node3 node4
        writeln!(
            writer,
            "{} 3 2 0 0 {} {} {} {}",
            elem_id,
            elem[0] + 1,
            elem[1] + 1,
            elem[2] + 1,
            elem[3] + 1
        )?;
        elem_id += 1;
    }

    writeln!(writer, "$EndElements")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_simple_mesh() {
        // Create a simple mesh file
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
$EndNodes
$Elements
1
1 3 2 0 0 1 2 3 4
$EndElements"#
        )
        .unwrap();

        let mesh = read_gmsh_mesh(file.path()).unwrap();
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.n_elements, 1);
        assert_eq!(mesh.n_edges, 4);
        assert_eq!(mesh.n_boundary_edges, 4);
    }

    #[test]
    fn test_read_mesh_with_boundary() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
$EndNodes
$Elements
5
1 1 2 1 1 1 2
2 1 2 2 2 2 3
3 1 2 1 1 3 4
4 1 2 2 2 4 1
5 3 2 0 0 1 2 3 4
$EndElements"#
        )
        .unwrap();

        let mesh = read_gmsh_mesh(file.path()).unwrap();
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.n_elements, 1);
        assert_eq!(mesh.n_boundary_edges, 4);

        // Check boundary tags are assigned
        let wall_count = mesh
            .edges
            .iter()
            .filter(|e| e.boundary_tag == Some(BoundaryTag::Wall))
            .count();
        let open_count = mesh
            .edges
            .iter()
            .filter(|e| e.boundary_tag == Some(BoundaryTag::Open))
            .count();
        assert_eq!(wall_count, 2);
        assert_eq!(open_count, 2);
    }

    #[test]
    fn test_roundtrip() {
        // Create a simple mesh
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);

        // Write to file
        let file = NamedTempFile::new().unwrap();
        write_gmsh_mesh(&mesh, file.path()).unwrap();

        // Read back
        let mesh2 = read_gmsh_mesh(file.path()).unwrap();

        assert_eq!(mesh.vertices.len(), mesh2.vertices.len());
        assert_eq!(mesh.n_elements, mesh2.n_elements);
        assert_eq!(mesh.n_edges, mesh2.n_edges);
    }

    #[test]
    fn test_physical_tag_conversion() {
        assert_eq!(physical_tag_to_boundary_tag(1), BoundaryTag::Wall);
        assert_eq!(physical_tag_to_boundary_tag(2), BoundaryTag::Open);
        assert_eq!(physical_tag_to_boundary_tag(3), BoundaryTag::TidalForcing);
        assert_eq!(physical_tag_to_boundary_tag(4), BoundaryTag::River);
        assert_eq!(physical_tag_to_boundary_tag(99), BoundaryTag::Custom(99));

        assert_eq!(boundary_tag_to_physical(&BoundaryTag::Wall), 1);
        assert_eq!(boundary_tag_to_physical(&BoundaryTag::Open), 2);
    }

    #[test]
    fn test_error_missing_nodes() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Elements
1
1 3 2 0 0 1 2 3 4
$EndElements"#
        )
        .unwrap();

        let result = read_gmsh_mesh(file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_two_element_mesh() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
6
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 2.0 0.0 0.0
4 0.0 1.0 0.0
5 1.0 1.0 0.0
6 2.0 1.0 0.0
$EndNodes
$Elements
2
1 3 2 0 0 1 2 5 4
2 3 2 0 0 2 3 6 5
$EndElements"#
        )
        .unwrap();

        let mesh = read_gmsh_mesh(file.path()).unwrap();
        assert_eq!(mesh.n_elements, 2);
        assert_eq!(mesh.vertices.len(), 6);

        // Should have 7 edges: 4 boundary + 3 = 7? Let's count:
        // Bottom: 2 edges (boundary)
        // Top: 2 edges (boundary)
        // Left: 1 edge (boundary)
        // Right: 1 edge (boundary)
        // Middle: 1 edge (interior)
        // Total: 7 edges, 6 boundary
        assert_eq!(mesh.n_edges, 7);
        assert_eq!(mesh.n_boundary_edges, 6);

        // Interior edges should have both left and right neighbors
        let interior_edges: Vec<_> = mesh.edges.iter().filter(|e| e.is_interior()).collect();
        assert_eq!(interior_edges.len(), 1);
    }
}
