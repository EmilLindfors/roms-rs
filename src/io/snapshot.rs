//! Snapshot files: a 2D shallow-water run's surface and depth-averaged velocity,
//! frame after frame, in one compact binary file that also carries the mesh and the
//! bed, so a run can be replayed (e.g. by the `viz/` viewer) without the data and
//! the code that built its domain.
//!
//! A frame is the surface elevation η = h + B and the velocity (u, v) at every DG
//! node in the solution's element-major order, as `f32`: 12 bytes per node, a tenth
//! of an ASCII VTU frame. The velocity is `SWEState2D::velocity` with the writer's
//! `h_min` (desingularised in thin water), as `write_vtk_swe` writes it.
//!
//! # Format (little-endian)
//!
//! ```text
//! magic        8 bytes   b"DGSNAP\0\x01" (format version 1)
//! header_len   u64       bytes of the header that follows; frames start after it
//! header:
//!   order                u32
//!   n_vertices           u64, then [x, y] f64 per vertex
//!   n_elements           u64, then 4 vertex indices u32 per element (counter-clockwise)
//!   n_periodic           u64, then (element, face, element, face) u32 per periodic pair
//!   n_boundary           u64, then (element, face, kind, value) u32 per boundary face
//!   bed                  f64 per node (n_elements × (order + 1)²), B (negative under water)
//!   clock                f64: Unix time of model time 0, NaN for none
//!   metadata             u64 bytes of UTF-8, `key=value` lines
//! frames, each:
//!   t                    f64 (model time, s)
//!   eta, u, v            f32 per node each
//! ```
//!
//! Boundary kinds: 0 wall, 1 open, 2 tidal forcing, 3 periodic (value: group),
//! 4 river, 5 Dirichlet, 6 Neumann, 7 custom (value). Frames are appended and
//! flushed one at a time, so a reader can follow a run that is still writing: it
//! counts the complete frames from the file's length and ignores a partial last one.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use thiserror::Error;

use crate::mesh::{BoundaryTag, ElementFace, Mesh2D, QuadMeshError};
use crate::operators::DGOperators2D;
use crate::solver::{SWESolution2D, SWEState2D};
use crate::time::ModelClock;
use crate::types::Depth;

const MAGIC: &[u8; 8] = b"DGSNAP\0\x01";

/// Error reading or writing a snapshot file.
#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("snapshot I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a snapshot file: {0}")]
    Format(String),
    #[error("snapshot mesh: {0}")]
    Mesh(#[from] QuadMeshError),
}

/// What a snapshot file holds besides its frames.
#[derive(Clone)]
pub struct SnapshotHeader {
    /// Polynomial order of the run
    pub order: usize,
    pub mesh: Mesh2D,
    /// Bed elevation B at every node (m, negative under water), element-major
    pub bathymetry: Vec<f64>,
    /// UTC of model time 0, if the run had one
    pub clock: Option<ModelClock>,
    /// Free-form `key=value` pairs (title, points of interest, …), in file order
    pub metadata: Vec<(String, String)>,
}

impl SnapshotHeader {
    /// The first value of `key` in the metadata.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Nodes in a frame.
    pub fn n_points(&self) -> usize {
        self.bathymetry.len()
    }
}

/// One frame: model time, η, u, v at every node.
#[derive(Clone, Debug, Default)]
pub struct SnapshotFrame {
    pub t: f64,
    pub eta: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
}

/// Writes a snapshot file: the header on creation, then one frame per call.
pub struct SnapshotWriter {
    out: BufWriter<File>,
    bed: Vec<f64>,
    h_min: Depth,
    /// Frame buffer, reused
    bytes: Vec<u8>,
    eta: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
}

impl SnapshotWriter {
    /// Create `path` for a run on `mesh` at `ops`' order over the bed `bathymetry`
    /// (nodal, element-major), with velocities desingularised below `h_min` (m).
    pub fn create(
        path: impl AsRef<Path>,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        bathymetry: &[f64],
        clock: Option<&ModelClock>,
        metadata: &[(&str, &str)],
        h_min: f64,
    ) -> Result<Self, SnapshotError> {
        let n = mesh.n_elements * ops.n_nodes;
        if bathymetry.len() != n {
            return Err(SnapshotError::Format(format!(
                "the bed has {} values for {n} nodes",
                bathymetry.len()
            )));
        }
        if let Some((k, v)) = metadata
            .iter()
            .find(|(k, v)| k.contains(['=', '\n']) || v.contains('\n'))
        {
            return Err(SnapshotError::Format(format!(
                "metadata {k:?} = {v:?}: no '=' in keys, no newlines"
            )));
        }
        let header = encode_header(mesh, ops.order, bathymetry, clock, metadata);
        let mut out = BufWriter::new(File::create(path)?);
        out.write_all(MAGIC)?;
        out.write_all(&(header.len() as u64).to_le_bytes())?;
        out.write_all(&header)?;
        out.flush()?;
        Ok(Self {
            out,
            bed: bathymetry.to_vec(),
            h_min: Depth::new(h_min),
            bytes: Vec::with_capacity(8 + 12 * n),
            eta: vec![0.0; n],
            u: vec![0.0; n],
            v: vec![0.0; n],
        })
    }

    /// Append the state `q` at model time `t`.
    pub fn write_state(&mut self, t: f64, q: &SWESolution2D) -> Result<(), SnapshotError> {
        let [h, hu, hv] = &q.data;
        if h.len() != self.bed.len() {
            return Err(SnapshotError::Format(format!(
                "the state has {} nodes, the file {}",
                h.len(),
                self.bed.len()
            )));
        }
        for i in 0..h.len() {
            let (u, v) = SWEState2D::new(h[i], hu[i], hv[i]).velocity(self.h_min);
            self.eta[i] = (h[i] + self.bed[i]) as f32;
            self.u[i] = u as f32;
            self.v[i] = v as f32;
        }
        let (eta, u, v) = (
            std::mem::take(&mut self.eta),
            std::mem::take(&mut self.u),
            std::mem::take(&mut self.v),
        );
        let written = self.write_fields(t, &eta, &u, &v);
        (self.eta, self.u, self.v) = (eta, u, v);
        written
    }

    /// Append a frame of η, u and v (one value per node each) at model time `t`.
    pub fn write_fields(
        &mut self,
        t: f64,
        eta: &[f32],
        u: &[f32],
        v: &[f32],
    ) -> Result<(), SnapshotError> {
        let n = self.bed.len();
        if eta.len() != n || u.len() != n || v.len() != n {
            return Err(SnapshotError::Format(format!(
                "a frame needs {n} values per field, not {}, {}, {}",
                eta.len(),
                u.len(),
                v.len()
            )));
        }
        self.bytes.clear();
        self.bytes.extend_from_slice(&t.to_le_bytes());
        for field in [eta, u, v] {
            for x in field {
                self.bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        // One whole frame per flush, so that a reader sees frames complete or not at all
        self.out.write_all(&self.bytes)?;
        self.out.flush()?;
        Ok(())
    }
}

/// Reads a snapshot file: the header on opening, then any frame by its index.
pub struct SnapshotReader {
    file: File,
    header: SnapshotHeader,
    frames_start: u64,
    bytes: Vec<u8>,
}

impl SnapshotReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SnapshotError> {
        let mut file = File::open(path)?;
        let mut start = [0u8; 16];
        file.read_exact(&mut start)
            .map_err(|_| SnapshotError::Format("shorter than its magic number".into()))?;
        if &start[..8] != MAGIC {
            return Err(SnapshotError::Format(
                "wrong magic number (not a dg-rs snapshot, or another version)".into(),
            ));
        }
        let header_len = u64::from_le_bytes(start[8..].try_into().unwrap());
        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header)
            .map_err(|_| SnapshotError::Format("truncated header".into()))?;
        let header = decode_header(&header)?;
        let n = header.n_points();
        Ok(Self {
            file,
            header,
            frames_start: 16 + header_len,
            bytes: vec![0u8; 8 + 12 * n],
        })
    }

    pub fn header(&self) -> &SnapshotHeader {
        &self.header
    }

    pub fn into_header(self) -> SnapshotHeader {
        self.header
    }

    fn frame_bytes(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// Complete frames in the file now (a run may still be appending).
    pub fn n_frames(&self) -> Result<usize, SnapshotError> {
        let len = self.file.metadata()?.len();
        Ok((len.saturating_sub(self.frames_start) / self.frame_bytes()) as usize)
    }

    /// Model time of frame `i`.
    pub fn time(&mut self, i: usize) -> Result<f64, SnapshotError> {
        let mut t = [0u8; 8];
        self.file.seek(SeekFrom::Start(
            self.frames_start + i as u64 * self.frame_bytes(),
        ))?;
        self.file.read_exact(&mut t)?;
        Ok(f64::from_le_bytes(t))
    }

    /// Frame `i`.
    pub fn read_frame(&mut self, i: usize) -> Result<SnapshotFrame, SnapshotError> {
        let mut frame = SnapshotFrame::default();
        self.read_frame_into(i, &mut frame)?;
        Ok(frame)
    }

    /// Frame `i` into `frame`, reusing its buffers.
    pub fn read_frame_into(
        &mut self,
        i: usize,
        frame: &mut SnapshotFrame,
    ) -> Result<(), SnapshotError> {
        let n = self.header.n_points();
        self.file.seek(SeekFrom::Start(
            self.frames_start + i as u64 * self.frame_bytes(),
        ))?;
        self.file.read_exact(&mut self.bytes)?;
        frame.t = f64::from_le_bytes(self.bytes[..8].try_into().unwrap());
        for (f, field) in [&mut frame.eta, &mut frame.u, &mut frame.v]
            .into_iter()
            .enumerate()
        {
            let at = 8 + 4 * n * f;
            field.clear();
            field.extend(
                self.bytes[at..at + 4 * n]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&b| f32::from_le_bytes(b)),
            );
        }
        Ok(())
    }
}

fn tag_code(tag: BoundaryTag) -> (u32, u32) {
    match tag {
        BoundaryTag::Wall => (0, 0),
        BoundaryTag::Open => (1, 0),
        BoundaryTag::TidalForcing => (2, 0),
        BoundaryTag::Periodic(group) => (3, group),
        BoundaryTag::River => (4, 0),
        BoundaryTag::Dirichlet => (5, 0),
        BoundaryTag::Neumann => (6, 0),
        BoundaryTag::Custom(value) => (7, value),
    }
}

fn tag_of(kind: u32, value: u32) -> Result<BoundaryTag, SnapshotError> {
    Ok(match kind {
        0 => BoundaryTag::Wall,
        1 => BoundaryTag::Open,
        2 => BoundaryTag::TidalForcing,
        3 => BoundaryTag::Periodic(value),
        4 => BoundaryTag::River,
        5 => BoundaryTag::Dirichlet,
        6 => BoundaryTag::Neumann,
        7 => BoundaryTag::Custom(value),
        _ => return Err(SnapshotError::Format(format!("boundary kind {kind}"))),
    })
}

fn encode_header(
    mesh: &Mesh2D,
    order: usize,
    bed: &[f64],
    clock: Option<&ModelClock>,
    metadata: &[(&str, &str)],
) -> Vec<u8> {
    let mut b = Vec::new();
    let u32_ = |b: &mut Vec<u8>, x: usize| b.extend_from_slice(&(x as u32).to_le_bytes());
    let u64_ = |b: &mut Vec<u8>, x: usize| b.extend_from_slice(&(x as u64).to_le_bytes());
    u32_(&mut b, order);
    u64_(&mut b, mesh.vertices.len());
    for p in &mesh.vertices {
        b.extend_from_slice(&p[0].to_le_bytes());
        b.extend_from_slice(&p[1].to_le_bytes());
    }
    u64_(&mut b, mesh.elements.len());
    for quad in &mesh.elements {
        quad.iter().for_each(|&v| u32_(&mut b, v));
    }
    // An interior edge whose two faces do not share their vertices is periodic
    let face_key = |f: ElementFace| {
        let quad = mesh.elements[f.element];
        let (a, b) = (quad[f.face], quad[(f.face + 1) % 4]);
        (a.min(b), a.max(b))
    };
    let periodic: Vec<_> = mesh
        .edges
        .iter()
        .filter_map(|e| e.right.map(|r| (e.left, r)))
        .filter(|&(l, r)| face_key(l) != face_key(r))
        .collect();
    u64_(&mut b, periodic.len());
    for (l, r) in periodic {
        [l.element, l.face, r.element, r.face]
            .iter()
            .for_each(|&x| u32_(&mut b, x));
    }
    let boundary: Vec<_> = mesh.edges.iter().filter(|e| e.right.is_none()).collect();
    u64_(&mut b, boundary.len());
    for edge in boundary {
        let (kind, value) = tag_code(edge.boundary_tag.unwrap_or(BoundaryTag::Wall));
        u32_(&mut b, edge.left.element);
        u32_(&mut b, edge.left.face);
        b.extend_from_slice(&kind.to_le_bytes());
        b.extend_from_slice(&value.to_le_bytes());
    }
    bed.iter()
        .for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
    let epoch = clock.map_or(f64::NAN, |c| c.epoch_unix);
    b.extend_from_slice(&epoch.to_le_bytes());
    let text: String = metadata.iter().map(|(k, v)| format!("{k}={v}\n")).collect();
    u64_(&mut b, text.len());
    b.extend_from_slice(text.as_bytes());
    b
}

/// A cursor over the header's bytes.
struct Bytes<'a>(&'a [u8]);

impl Bytes<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], SnapshotError> {
        if self.0.len() < n {
            return Err(SnapshotError::Format("truncated header".into()));
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }
    fn u32(&mut self) -> Result<u32, SnapshotError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn index(&mut self) -> Result<usize, SnapshotError> {
        self.u32().map(|x| x as usize)
    }
    /// A count of items of `size` bytes each, checked against what is left
    fn count(&mut self, size: usize) -> Result<usize, SnapshotError> {
        let n = u64::from_le_bytes(self.take(8)?.try_into().unwrap()) as usize;
        if n.saturating_mul(size) > self.0.len() {
            return Err(SnapshotError::Format("truncated header".into()));
        }
        Ok(n)
    }
    fn f64(&mut self) -> Result<f64, SnapshotError> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

fn decode_header(bytes: &[u8]) -> Result<SnapshotHeader, SnapshotError> {
    let mut b = Bytes(bytes);
    let order = b.index()?;
    let vertices = (0..b.count(16)?)
        .map(|_| Ok([b.f64()?, b.f64()?]))
        .collect::<Result<Vec<_>, SnapshotError>>()?;
    let elements = (0..b.count(16)?)
        .map(|_| Ok([b.index()?, b.index()?, b.index()?, b.index()?]))
        .collect::<Result<Vec<_>, SnapshotError>>()?;
    let periodic = (0..b.count(16)?)
        .map(|_| {
            Ok((
                ElementFace::new(b.index()?, b.index()?),
                ElementFace::new(b.index()?, b.index()?),
            ))
        })
        .collect::<Result<Vec<_>, SnapshotError>>()?;
    let mut tags = std::collections::HashMap::new();
    for _ in 0..b.count(16)? {
        let face = (b.index()?, b.index()?);
        tags.insert(face, tag_of(b.u32()?, b.u32()?)?);
    }
    let n_elements = elements.len();
    let mesh = Mesh2D::from_quads(vertices, elements, &periodic, |f| {
        tags.get(&(f.element, f.face))
            .copied()
            .unwrap_or(BoundaryTag::Wall)
    })?;
    let n_points = n_elements * (order + 1) * (order + 1);
    if b.0.len() < 8 * n_points {
        return Err(SnapshotError::Format("truncated header".into()));
    }
    let bathymetry = (0..n_points)
        .map(|_| b.f64())
        .collect::<Result<Vec<_>, _>>()?;
    let epoch = b.f64()?;
    let n_text = b.count(1)?;
    let text = std::str::from_utf8(b.take(n_text)?)
        .map_err(|_| SnapshotError::Format("metadata is not UTF-8".into()))?;
    let metadata = text
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    Ok(SnapshotHeader {
        order,
        mesh,
        bathymetry,
        clock: epoch.is_finite().then(|| ModelClock::new(epoch)),
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ElementIndex;

    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("dg-rs-snapshot-{}-{name}", std::process::id()))
    }

    /// A periodic channel with an open end tag: every kind of face
    fn mesh() -> Mesh2D {
        let mut mesh = Mesh2D::channel_periodic_x(0.0, 300.0, 0.0, 100.0, 3, 2);
        let wall = mesh.edges.iter().position(|e| e.right.is_none()).unwrap();
        mesh.edges[wall].boundary_tag = Some(BoundaryTag::Open);
        mesh
    }

    #[test]
    fn a_run_reads_back_as_written() {
        let path = scratch("roundtrip");
        let mesh = mesh();
        let ops = DGOperators2D::new(2);
        let n = mesh.n_elements * ops.n_nodes;
        let bed: Vec<f64> = (0..n).map(|i| -10.0 - i as f64 * 0.1).collect();
        let clock = ModelClock::parse("2025-05-31T00:00:00Z").unwrap();
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let h = -bed[k.as_usize() * ops.n_nodes + i] + 0.25;
                q.set_state(k, i, SWEState2D::new(h, 0.5 * h, -0.25 * h));
            }
        }
        let mut writer = SnapshotWriter::create(
            &path,
            &mesh,
            &ops,
            &bed,
            Some(&clock),
            &[("title", "test: a=b"), ("point_of_interest", "150,50")],
            1e-3,
        )
        .unwrap();
        writer.write_state(0.0, &q).unwrap();
        writer.write_state(3600.0, &q).unwrap();
        // A partial frame, as a run being written leaves it, is not counted
        writer.out.write_all(&[0u8; 20]).unwrap();
        writer.out.flush().unwrap();

        let mut reader = SnapshotReader::open(&path).unwrap();
        assert_eq!(reader.n_frames().unwrap(), 2);
        let header = reader.header().clone();
        assert_eq!(header.order, 2);
        assert_eq!(header.bathymetry, bed);
        assert_eq!(header.clock, Some(clock));
        assert_eq!(header.metadata("title"), Some("test: a=b"));
        assert_eq!(header.metadata("point_of_interest"), Some("150,50"));
        assert_eq!(header.mesh.vertices, mesh.vertices);
        assert_eq!(header.mesh.elements, mesh.elements);
        for k in ElementIndex::iter(mesh.n_elements) {
            for f in 0..4 {
                let (a, b) = (
                    mesh.element_edges[k.as_usize()][f],
                    header.mesh.element_edges[k.as_usize()][f],
                );
                let (a, b) = (&mesh.edges[a], &header.mesh.edges[b]);
                assert_eq!(a.right.is_some(), b.right.is_some(), "face {k:?} {f}");
                assert_eq!(a.boundary_tag, b.boundary_tag, "face {k:?} {f}");
            }
        }
        assert_eq!(header.mesh.n_boundary_edges, mesh.n_boundary_edges);

        assert_eq!(reader.time(1).unwrap(), 3600.0);
        let frame = reader.read_frame(1).unwrap();
        assert_eq!(frame.t, 3600.0);
        // η = h + B = 0.25 everywhere, (u, v) = (0.5, −0.25) in water metres deep
        for j in 0..n {
            assert!((frame.eta[j] - 0.25).abs() < 1e-6, "η at node {j}");
            assert!((frame.u[j] - 0.5).abs() < 1e-6 && (frame.v[j] + 0.25).abs() < 1e-6);
        }
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn other_files_are_refused() {
        let path = scratch("bad");
        std::fs::write(&path, b"<?xml version=\"1.0\"?> not a snapshot").unwrap();
        assert!(matches!(
            SnapshotReader::open(&path),
            Err(SnapshotError::Format(_))
        ));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn frames_of_the_wrong_size_are_refused() {
        let path = scratch("size");
        let mesh = mesh();
        let ops = DGOperators2D::new(1);
        let bed = vec![-5.0; mesh.n_elements * ops.n_nodes];
        let mut writer = SnapshotWriter::create(&path, &mesh, &ops, &bed, None, &[], 1e-3).unwrap();
        assert!(
            writer
                .write_fields(0.0, &[0.0; 3], &[0.0; 3], &[0.0; 3])
                .is_err()
        );
        assert!(SnapshotWriter::create(&path, &mesh, &ops, &bed[1..], None, &[], 1e-3).is_err());
        assert!(
            SnapshotWriter::create(&path, &mesh, &ops, &bed, None, &[("a=b", "c")], 1e-3).is_err()
        );
        std::fs::remove_file(&path).unwrap();
    }
}
