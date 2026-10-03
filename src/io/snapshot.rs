//! Snapshot files: a run's surface and depth-averaged velocity, and in 3D its
//! fields on every σ-level, frame after frame, in one compact binary file that also
//! carries the mesh, the bed and the σ-grid, so a run can be replayed (e.g. by the
//! `viz/` viewer) without the data and the code that built its domain.
//!
//! A frame is the surface elevation η and the depth-averaged velocity (u, v) at
//! every DG node in the solution's element-major order, as `f32`: 12 bytes per node,
//! a tenth of an ASCII VTU frame. In 2D (`write_state`) η = h + B and the velocity
//! is `SWEState2D::velocity` with the writer's `h_min` (desingularised in thin
//! water), as `write_vtk_swe` writes it. In 3D (`create_3d`, `write_solution_3d`)
//! they are `Solution3D`'s η, ū, v̄, followed by its u, v, T and S on every level,
//! `[node][level]` from the bed up ([`SOLUTION_3D_FIELDS`]).
//!
//! # Format (little-endian)
//!
//! ```text
//! magic        8 bytes   b"DGSNAP\0\x02" (format version 2; version 1 is read too)
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
//!   n_levels             u32 (version 2; 0 in 2D, and nothing more), then
//!     sigma_w            f64 × (n_levels + 1), bed (−1) to surface (0)
//!     sigma_rho          f64 × n_levels, the layer centres
//!     stretching         u64 bytes of UTF-8, its name
//!     n_fields           u32, then per field u64 bytes of UTF-8, its name
//! frames, each:
//!   t                    f64 (model time, s)
//!   eta, u, v            f32 per node each
//!   per field            f32 per node and level, [node][level]
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
use crate::solver::state::Solution3D;
use crate::solver::{SWESolution2D, SWEState2D};
use crate::time::ModelClock;
use crate::types::Depth;
use crate::vertical::SigmaGrid;

/// The magic number without its version byte, and the version written.
const MAGIC: &[u8; 7] = b"DGSNAP\0";
const VERSION: u8 = 2;

/// The layered fields of a 3D snapshot ([`SnapshotWriter::create_3d`]): velocity
/// along the σ-layers (m/s), temperature (°C) and salinity.
pub const SOLUTION_3D_FIELDS: [&str; 4] = ["u", "v", "temp", "salt"];

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
    /// The σ-grid and the layered fields of a 3D run
    pub levels: Option<SnapshotLevels>,
}

/// The vertical of a 3D snapshot.
#[derive(Clone)]
pub struct SnapshotLevels {
    pub sigma: SigmaGrid,
    /// Names of the fields stored on every level, in frame order
    pub fields: Vec<String>,
}

impl SnapshotLevels {
    /// Index of the layered field `name` in a frame's `layers`.
    pub fn field(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f == name)
    }
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

    /// Levels of a 3D run, 0 in 2D.
    pub fn n_levels(&self) -> usize {
        self.levels.as_ref().map_or(0, |l| l.sigma.n_levels())
    }

    /// Bytes of one frame.
    fn frame_bytes(&self) -> usize {
        let n_fields = self.levels.as_ref().map_or(0, |l| l.fields.len());
        8 + 4 * self.n_points() * (3 + n_fields * self.n_levels())
    }
}

/// One frame: model time, η, u, v at every node, and in 3D the layered fields.
#[derive(Clone, Debug, Default)]
pub struct SnapshotFrame {
    pub t: f64,
    pub eta: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    /// Per layered field ([`SnapshotLevels::fields`]), `[node][level]`
    pub layers: Vec<Vec<f32>>,
}

/// Writes a snapshot file: the header on creation, then one frame per call.
pub struct SnapshotWriter {
    out: BufWriter<File>,
    bed: Vec<f64>,
    h_min: Depth,
    /// Levels and layered fields per frame (0 in 2D)
    n_levels: usize,
    n_fields: usize,
    /// Frame buffer, reused
    bytes: Vec<u8>,
    eta: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
}

impl SnapshotWriter {
    /// Create `path` for a 2D run on `mesh` at `ops`' order over the bed `bathymetry`
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
        Self::create_with(path, mesh, ops, bathymetry, clock, metadata, h_min, None)
    }

    /// Create `path` for a 3D run on the σ-grid `sigma`, with the layered fields
    /// [`SOLUTION_3D_FIELDS`] ([`Self::write_solution_3d`]).
    pub fn create_3d(
        path: impl AsRef<Path>,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        bathymetry: &[f64],
        sigma: &SigmaGrid,
        clock: Option<&ModelClock>,
        metadata: &[(&str, &str)],
    ) -> Result<Self, SnapshotError> {
        Self::create_with(
            path,
            mesh,
            ops,
            bathymetry,
            clock,
            metadata,
            0.0,
            Some((sigma, &SOLUTION_3D_FIELDS)),
        )
    }

    /// Create `path` with any layered fields on `levels`' σ-grid (or none), for
    /// [`Self::write_fields`].
    #[allow(clippy::too_many_arguments)]
    pub fn create_with(
        path: impl AsRef<Path>,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        bathymetry: &[f64],
        clock: Option<&ModelClock>,
        metadata: &[(&str, &str)],
        h_min: f64,
        levels: Option<(&SigmaGrid, &[&str])>,
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
        if let Some((_, fields)) = levels
            && let Some(f) = fields.iter().find(|f| f.is_empty())
        {
            return Err(SnapshotError::Format(format!(
                "layered field name {f:?} is empty"
            )));
        }
        let header = encode_header(mesh, ops.order, bathymetry, clock, metadata, levels);
        let mut out = BufWriter::new(File::create(path)?);
        out.write_all(MAGIC)?;
        out.write_all(&[VERSION])?;
        out.write_all(&(header.len() as u64).to_le_bytes())?;
        out.write_all(&header)?;
        out.flush()?;
        Ok(Self {
            out,
            bed: bathymetry.to_vec(),
            h_min: Depth::new(h_min.max(0.0)),
            n_levels: levels.map_or(0, |(sigma, _)| sigma.n_levels()),
            n_fields: levels.map_or(0, |(_, fields)| fields.len()),
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
        let written = self.write_fields(t, &eta, &u, &v, &[]);
        (self.eta, self.u, self.v) = (eta, u, v);
        written
    }

    /// Append the 3D state `state` at model time `t` (a file from [`Self::create_3d`]).
    pub fn write_solution_3d(&mut self, t: f64, state: &Solution3D) -> Result<(), SnapshotError> {
        let n = self.bed.len();
        if state.n_levels != self.n_levels
            || self.n_fields != SOLUTION_3D_FIELDS.len()
            || state.eta.data.len() != n
        {
            return Err(SnapshotError::Format(format!(
                "a state of {} nodes on {} levels, the file has {n} nodes and {} levels of {} fields",
                state.eta.data.len(),
                state.n_levels,
                self.n_levels,
                self.n_fields
            )));
        }
        self.begin_frame(t);
        for field in [&state.eta.data, &state.ubar.data, &state.vbar.data]
            .into_iter()
            .chain([&state.u, &state.v, &state.temp, &state.salt])
        {
            for &x in field.iter() {
                self.bytes.extend_from_slice(&(x as f32).to_le_bytes());
            }
        }
        self.end_frame()
    }

    /// Append a frame of η, u and v (one value per node each) at model time `t`, and
    /// the layered fields (`[node][level]` each, in the order the file was created
    /// with; none in 2D).
    pub fn write_fields(
        &mut self,
        t: f64,
        eta: &[f32],
        u: &[f32],
        v: &[f32],
        layers: &[&[f32]],
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
        if layers.len() != self.n_fields || layers.iter().any(|l| l.len() != n * self.n_levels) {
            return Err(SnapshotError::Format(format!(
                "a frame needs {} layered fields of {} values",
                self.n_fields,
                n * self.n_levels
            )));
        }
        self.begin_frame(t);
        for field in [eta, u, v].into_iter().chain(layers.iter().copied()) {
            for x in field {
                self.bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        self.end_frame()
    }

    fn begin_frame(&mut self, t: f64) {
        self.bytes.clear();
        self.bytes.extend_from_slice(&t.to_le_bytes());
    }

    fn end_frame(&mut self) -> Result<(), SnapshotError> {
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
        if &start[..7] != MAGIC {
            return Err(SnapshotError::Format(
                "wrong magic number (not a dg-rs snapshot)".into(),
            ));
        }
        let version = start[7];
        if !(1..=VERSION).contains(&version) {
            return Err(SnapshotError::Format(format!(
                "format version {version}; this build reads 1 to {VERSION}"
            )));
        }
        let header_len = u64::from_le_bytes(start[8..].try_into().unwrap());
        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header)
            .map_err(|_| SnapshotError::Format("truncated header".into()))?;
        let header = decode_header(&header, version)?;
        let frame_bytes = header.frame_bytes();
        Ok(Self {
            file,
            header,
            frames_start: 16 + header_len,
            bytes: vec![0u8; frame_bytes],
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
        let layer = n * self.header.n_levels();
        let n_fields = self.header.levels.as_ref().map_or(0, |l| l.fields.len());
        self.file.seek(SeekFrom::Start(
            self.frames_start + i as u64 * self.frame_bytes(),
        ))?;
        self.file.read_exact(&mut self.bytes)?;
        frame.t = f64::from_le_bytes(self.bytes[..8].try_into().unwrap());
        frame.layers.resize_with(n_fields, Vec::new);
        let mut at = 8;
        for (field, len) in [&mut frame.eta, &mut frame.u, &mut frame.v]
            .into_iter()
            .map(|f| (f, n))
            .chain(frame.layers.iter_mut().map(|f| (f, layer)))
        {
            field.clear();
            field.extend(
                self.bytes[at..at + 4 * len]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&b| f32::from_le_bytes(b)),
            );
            at += 4 * len;
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
    levels: Option<(&SigmaGrid, &[&str])>,
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
    let string = |b: &mut Vec<u8>, s: &str| {
        u64_(b, s.len());
        b.extend_from_slice(s.as_bytes());
    };
    string(&mut b, &text);
    match levels {
        None => u32_(&mut b, 0),
        Some((sigma, fields)) => {
            u32_(&mut b, sigma.n_levels());
            sigma
                .sigma_w()
                .iter()
                .chain(sigma.sigma_rho())
                .for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
            string(&mut b, sigma.stretching_name());
            u32_(&mut b, fields.len());
            fields.iter().for_each(|f| string(&mut b, f));
        }
    }
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
    fn string(&mut self) -> Result<&str, SnapshotError> {
        let n = self.count(1)?;
        std::str::from_utf8(self.take(n)?)
            .map_err(|_| SnapshotError::Format("a string is not UTF-8".into()))
    }
}

fn decode_header(bytes: &[u8], version: u8) -> Result<SnapshotHeader, SnapshotError> {
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
    let metadata = b
        .string()?
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let n_levels = if version >= 2 { b.index()? } else { 0 };
    let levels = if n_levels > 0 {
        if b.0.len() < 8 * (2 * n_levels + 1) {
            return Err(SnapshotError::Format("truncated header".into()));
        }
        let sigma_w = (0..=n_levels)
            .map(|_| b.f64())
            .collect::<Result<Vec<_>, _>>()?;
        let sigma_rho = (0..n_levels)
            .map(|_| b.f64())
            .collect::<Result<Vec<_>, _>>()?;
        let stretching = b.string()?.to_string();
        let sigma = SigmaGrid::from_levels(sigma_rho, sigma_w, &stretching)
            .map_err(|e| SnapshotError::Format(format!("σ-levels: {e}")))?;
        let n_fields = b.index()?;
        let fields = (0..n_fields)
            .map(|_| b.string().map(str::to_string))
            .collect::<Result<Vec<_>, _>>()?;
        Some(SnapshotLevels { sigma, fields })
    } else {
        None
    };
    Ok(SnapshotHeader {
        order,
        mesh,
        bathymetry,
        clock: epoch.is_finite().then(|| ModelClock::new(epoch)),
        metadata,
        levels,
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
    fn a_3d_run_reads_back_with_its_levels() {
        use crate::vertical::SongHaidvogelStretching;

        let path = scratch("3d");
        let mesh = mesh();
        let ops = DGOperators2D::new(1);
        let n = mesh.n_elements * ops.n_nodes;
        let sigma = SigmaGrid::new(5, SongHaidvogelStretching::new(5.0, 0.4, 10.0));
        let bed = vec![-20.0; n];
        let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, 5);
        // Every value its own, so a misplaced one shows
        let value = |field: usize, i: usize| field as f64 * 1000.0 + i as f64 * 0.125;
        for (f, data) in [
            &mut state.eta.data,
            &mut state.ubar.data,
            &mut state.vbar.data,
            &mut state.u,
            &mut state.v,
            &mut state.temp,
            &mut state.salt,
        ]
        .into_iter()
        .enumerate()
        {
            for (i, x) in data.iter_mut().enumerate() {
                *x = value(f, i);
            }
        }
        let mut writer =
            SnapshotWriter::create_3d(&path, &mesh, &ops, &bed, &sigma, None, &[("cage", "1,2,3")])
                .unwrap();
        writer.write_solution_3d(0.0, &state).unwrap();
        writer.write_solution_3d(60.0, &state).unwrap();
        let mut reader = SnapshotReader::open(&path).unwrap();
        assert_eq!(reader.n_frames().unwrap(), 2);
        let header = reader.header().clone();
        let levels = header.levels.as_ref().unwrap();
        assert_eq!(levels.sigma.sigma_rho(), sigma.sigma_rho());
        assert_eq!(levels.sigma.sigma_w(), sigma.sigma_w());
        assert_eq!(levels.sigma.d_sigma(), sigma.d_sigma());
        assert_eq!(levels.sigma.stretching_name(), sigma.stretching_name());
        assert_eq!(levels.fields, SOLUTION_3D_FIELDS);
        assert_eq!(levels.field("temp"), Some(2));
        assert_eq!(header.metadata("cage"), Some("1,2,3"));

        let frame = reader.read_frame(1).unwrap();
        assert_eq!(frame.t, 60.0);
        for (f, data) in [&frame.eta, &frame.u, &frame.v]
            .into_iter()
            .chain(&frame.layers)
            .enumerate()
        {
            let len = if f < 3 { n } else { 5 * n };
            assert_eq!(data.len(), len, "field {f}");
            for (i, &x) in data.iter().enumerate() {
                assert_eq!(x, value(f, i) as f32, "field {f}, value {i}");
            }
        }
        // A 2D state does not fit a 3D file
        let q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        assert!(writer.write_state(0.0, &q).is_err());
        assert!(
            writer
                .write_solution_3d(0.0, &Solution3D::new(mesh.n_elements, ops.n_nodes, 4))
                .is_err()
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn version_1_files_still_read() {
        let path = scratch("v1");
        let mesh = mesh();
        let ops = DGOperators2D::new(1);
        let n = mesh.n_elements * ops.n_nodes;
        let bed = vec![-5.0; n];
        let mut writer =
            SnapshotWriter::create(&path, &mesh, &ops, &bed, None, &[("title", "old")], 1e-3)
                .unwrap();
        let eta = vec![0.5f32; n];
        writer.write_fields(7.0, &eta, &eta, &eta, &[]).unwrap();
        drop(writer);
        // Version 1 is version 2 without the levels' count at the end of the header
        let mut bytes = std::fs::read(&path).unwrap();
        let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        assert_eq!(bytes[16 + header_len - 4..16 + header_len], [0; 4]);
        bytes.drain(16 + header_len - 4..16 + header_len);
        bytes[7] = 1;
        bytes[8..16].copy_from_slice(&(header_len as u64 - 4).to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let mut reader = SnapshotReader::open(&path).unwrap();
        assert!(reader.header().levels.is_none());
        assert_eq!(reader.header().metadata("title"), Some("old"));
        assert_eq!(reader.n_frames().unwrap(), 1);
        let frame = reader.read_frame(0).unwrap();
        assert_eq!((frame.t, frame.eta[n - 1]), (7.0, 0.5));
        assert!(frame.layers.is_empty());
        // A version from the future is refused
        bytes[7] = VERSION + 1;
        std::fs::write(&path, &bytes).unwrap();
        assert!(SnapshotReader::open(&path).is_err());
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
                .write_fields(0.0, &[0.0; 3], &[0.0; 3], &[0.0; 3], &[])
                .is_err()
        );
        assert!(SnapshotWriter::create(&path, &mesh, &ops, &bed[1..], None, &[], 1e-3).is_err());
        assert!(
            SnapshotWriter::create(&path, &mesh, &ops, &bed, None, &[("a=b", "c")], 1e-3).is_err()
        );
        std::fs::remove_file(&path).unwrap();
    }
}
