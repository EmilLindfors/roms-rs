//! Restart files of 3D runs: everything a mode-split run carries from one
//! step to the next, so that a run can stop (a reboot, a crash, a long run in
//! pieces) and resume where it left off, bit for bit, or several experiments
//! can start from one spun-up state.
//!
//! A [`Restart3D`] holds, at a time between two steps:
//! - the whole [`Solution3D`]: η, ū, v̄, u, v, T, S, and also `w` and `ρ`
//!   (the next step's time-step bound reads them before it recomputes them),
//!   the eddy viscosity and diffusivity, and a turbulence closure's `k`, `ψ`;
//! - the mode splitter's slow forcing `G` of the last steps, which its AB3
//!   step average continues from ([`SlowForcingRecord`]);
//! - the vertical tracer advection's reference stratification, if the run
//!   has one (it relaxes towards the state when given a time scale);
//! - a fingerprint of the domain (mesh, bed, σ-grid), checked on resume, so a
//!   restart is not continued on another bed (e.g. without the 3D smoothing);
//! - the driver's own data: `key=value` metadata and named `f64` arrays (e.g.
//!   station records), carried along unread.
//!
//! [`crate::simulation::Simulation3D::restart`] and
//! [`crate::simulation::RunContext3D::restart`] take one;
//! [`crate::simulation::Simulation3D::resume`] continues from one. Everything
//! else a run holds is scratch, or a function of time (forcing, tides, the
//! parent model) evaluated afresh.
//!
//! # Format (little-endian)
//!
//! ```text
//! magic        8 bytes   b"DGRST3D" and the format version (1)
//! n_elements   u64
//! n_nodes      u64       horizontal nodes per element
//! n_levels     u64
//! domain       u64       fingerprint of the mesh, bed and σ-grid (`domain_fingerprint`)
//! time         f64       model time (s) of the state
//! n_sections   u64, then per section:
//!   name       u32 bytes of UTF-8
//!   kind       u8        0: f64 values, 1: UTF-8 text
//!   len        u64       values (kind 0) or bytes (kind 1)
//!   payload
//! checksum     u64       FNV-1a (64 bit) of every byte before it
//! ```
//!
//! Sections: `state.<field>` for the fields of [`Solution3D`] (`eta`, `ubar`,
//! `vbar`, `u`, `v`, `w`, `temp`, `salt`, `rho`, `eddy_viscosity`,
//! `eddy_diffusivity`, `tke`, `gls`), `slow_forcing.t`, `slow_forcing.next_t`
//! and `slow_forcing.<j>.h|hu|hv` (newest first), `vertical_reference.temp`
//! and `.salt` if any, `metadata` (text, `key=value` lines) and `extra.<name>`.
//! [`Restart3D::write`] writes a temporary file beside the target and renames
//! it over the target, so an interrupted write leaves the previous restart.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::mesh::Mesh2D;
use crate::solver::{DGSolution2D, SWESolution2D};
use crate::solver::state::Solution3D;
use crate::time::SlowForcingRecord;
use crate::vertical::SigmaGrid;

/// The magic number without its version byte, and the version written.
const MAGIC: &[u8; 7] = b"DGRST3D";
const VERSION: u8 = 1;

const KIND_VALUES: u8 = 0;
const KIND_TEXT: u8 = 1;

/// Values per chunk when streaming `f64` arrays.
const CHUNK: usize = 8192;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Error reading, writing or resuming from a restart file.
#[derive(Debug, Error)]
pub enum RestartError {
    #[error("restart I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a valid restart file: {0}")]
    Format(String),
    #[error("the restart is of another domain: {0}")]
    Domain(String),
}

/// A 3D run's state between two steps (see the [module docs](self)).
#[derive(Clone)]
pub struct Restart3D {
    /// Model time (s) of the state.
    pub time: f64,
    /// [`domain_fingerprint`] of the run's mesh, bed and σ-grid.
    pub domain: u64,
    /// The state after the step that ended at `time`.
    pub state: Solution3D,
    /// The mode splitter's slow forcing history.
    pub slow_forcing: SlowForcingRecord,
    /// The vertical tracer advection's reference `[temperature, salinity]`,
    /// `[element][node][level]`, if the run has one.
    pub vertical_reference: Option<[Vec<f64>; 2]>,
    /// The driver's `key=value` pairs (no newlines; no `=` in keys).
    pub metadata: Vec<(String, String)>,
    /// The driver's named arrays.
    pub extra: Vec<(String, Vec<f64>)>,
}

impl Restart3D {
    /// The driver's value of `key`, if any.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// The driver's array `name`, if any.
    pub fn extra(&self, name: &str) -> Option<&[f64]> {
        self.extra
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
    }

    /// Add (or replace) the driver's value of `key`.
    ///
    /// # Panics
    /// If `key` is empty or holds `=` or a newline, or `value` a newline.
    pub fn set_metadata(&mut self, key: &str, value: impl Into<String>) {
        let value = value.into();
        assert!(
            !key.is_empty() && !key.contains(['=', '\n', '\r']),
            "a restart metadata key must be non-empty, without '=' or newlines: {key:?}"
        );
        assert!(
            !value.contains(['\n', '\r']),
            "a restart metadata value must not hold newlines: {value:?}"
        );
        match self.metadata.iter_mut().find(|(k, _)| k == key) {
            Some((_, v)) => *v = value,
            None => self.metadata.push((key.to_string(), value)),
        }
    }

    /// Add (or replace) the driver's array `name`.
    pub fn set_extra(&mut self, name: &str, values: Vec<f64>) {
        match self.extra.iter_mut().find(|(n, _)| n == name) {
            Some((_, v)) => *v = values,
            None => self.extra.push((name.to_string(), values)),
        }
    }

    /// Write to `path`: to a temporary file beside it first, synced and then
    /// renamed over `path`, so a crash during the write keeps the previous
    /// file whole.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), RestartError> {
        let path = path.as_ref();
        let temporary = temporary_path(path);
        {
            let file = File::create(&temporary)?;
            let mut out = HashingWriter::new(BufWriter::new(file));
            self.write_body(&mut out)?;
            let checksum = out.hash;
            let mut inner = out.inner;
            inner.write_all(&checksum.to_le_bytes())?;
            inner.flush()?;
            inner
                .into_inner()
                .map_err(|e| e.into_error())?
                .sync_all()?;
        }
        std::fs::rename(&temporary, path)?;
        Ok(())
    }

    fn write_body<W: Write>(&self, out: &mut HashingWriter<W>) -> std::io::Result<()> {
        let s = &self.state;
        out.write_all(MAGIC)?;
        out.write_all(&[VERSION])?;
        for n in [s.n_elements, s.n_nodes, s.n_levels] {
            out.write_all(&(n as u64).to_le_bytes())?;
        }
        out.write_all(&self.domain.to_le_bytes())?;
        out.write_all(&self.time.to_le_bytes())?;

        let mut sections: Vec<(String, &[f64])> = state_fields(s)
            .into_iter()
            .map(|(name, values)| (format!("state.{name}"), values))
            .collect();
        let next_t = [self.slow_forcing.next_t];
        sections.push(("slow_forcing.t".into(), &self.slow_forcing.t));
        sections.push(("slow_forcing.next_t".into(), &next_t));
        for (j, g) in self.slow_forcing.g.iter().enumerate() {
            for (component, values) in ["h", "hu", "hv"].iter().zip(&g.data) {
                sections.push((format!("slow_forcing.{j}.{component}"), values));
            }
        }
        if let Some([temp, salt]) = &self.vertical_reference {
            sections.push(("vertical_reference.temp".into(), temp));
            sections.push(("vertical_reference.salt".into(), salt));
        }
        for (name, values) in &self.extra {
            sections.push((format!("extra.{name}"), values));
        }
        let metadata: String = self
            .metadata
            .iter()
            .map(|(k, v)| format!("{k}={v}\n"))
            .collect();

        out.write_all(&(sections.len() as u64 + 1).to_le_bytes())?;
        for (name, values) in sections {
            write_name(out, &name, KIND_VALUES, values.len())?;
            let mut bytes = Vec::with_capacity(8 * CHUNK.min(values.len()));
            for chunk in values.chunks(CHUNK) {
                bytes.clear();
                for x in chunk {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                out.write_all(&bytes)?;
            }
        }
        write_name(out, "metadata", KIND_TEXT, metadata.len())?;
        out.write_all(metadata.as_bytes())
    }

    /// Read a restart written by [`Self::write`], checking its checksum.
    pub fn read(path: impl AsRef<Path>) -> Result<Self, RestartError> {
        let path = path.as_ref();
        let length = std::fs::metadata(path)?.len();
        if length < 8 {
            return Err(RestartError::Format(format!(
                "{} is too short for a restart ({length} bytes)",
                path.display()
            )));
        }
        let mut input = HashingReader {
            inner: BufReader::new(File::open(path)?),
            hash: FNV_OFFSET,
            remaining: length - 8,
        };
        let restart = Self::read_body(&mut input)?;
        if input.remaining != 0 {
            return Err(RestartError::Format(format!(
                "{} trailing bytes before the checksum",
                input.remaining
            )));
        }
        let computed = input.hash;
        let mut stored = [0; 8];
        input.inner.read_exact(&mut stored)?;
        if u64::from_le_bytes(stored) != computed {
            return Err(RestartError::Format(format!(
                "{}: checksum mismatch (the file is damaged or truncated)",
                path.display()
            )));
        }
        Ok(restart)
    }

    fn read_body<R: Read>(input: &mut HashingReader<R>) -> Result<Self, RestartError> {
        let format = |message: String| RestartError::Format(message);
        let mut magic = [0; 8];
        input.read_exact(&mut magic)?;
        if &magic[..7] != MAGIC {
            return Err(format("bad magic number".into()));
        }
        if magic[7] != VERSION {
            return Err(format(format!(
                "format version {} (this build reads {VERSION})",
                magic[7]
            )));
        }
        let n_elements = input.u64()? as usize;
        let n_nodes = input.u64()? as usize;
        let n_levels = input.u64()? as usize;
        let domain = input.u64()?;
        let time = f64::from_bits(input.u64()?);

        let n_sections = input.u64()?;
        let mut values: HashMap<String, Vec<f64>> = HashMap::new();
        let mut order = Vec::new();
        let mut metadata = Vec::new();
        for _ in 0..n_sections {
            let mut name_len = [0; 4];
            input.read_exact(&mut name_len)?;
            let name_len = u32::from_le_bytes(name_len) as u64;
            if name_len > input.remaining {
                return Err(format("section name past the end of the file".into()));
            }
            let mut name = vec![0; name_len as usize];
            input.read_exact(&mut name)?;
            let name =
                String::from_utf8(name).map_err(|_| format("section name not UTF-8".into()))?;
            let mut kind = [0; 1];
            input.read_exact(&mut kind)?;
            let len = input.u64()?;
            match kind[0] {
                KIND_VALUES => {
                    if len.saturating_mul(8) > input.remaining {
                        return Err(format(format!("section {name} past the end of the file")));
                    }
                    let mut data = vec![0.0; len as usize];
                    let mut bytes = vec![0; 8 * CHUNK.min(data.len())];
                    for chunk in data.chunks_mut(CHUNK) {
                        let bytes = &mut bytes[..8 * chunk.len()];
                        input.read_exact(bytes)?;
                        for (x, b) in chunk.iter_mut().zip(bytes.as_chunks::<8>().0) {
                            *x = f64::from_le_bytes(*b);
                        }
                    }
                    order.push(name.clone());
                    values.insert(name, data);
                }
                KIND_TEXT if name == "metadata" => {
                    if len > input.remaining {
                        return Err(format("metadata past the end of the file".into()));
                    }
                    let mut text = vec![0; len as usize];
                    input.read_exact(&mut text)?;
                    let text =
                        String::from_utf8(text).map_err(|_| format("metadata not UTF-8".into()))?;
                    for line in text.lines().filter(|l| !l.is_empty()) {
                        let (k, v) = line
                            .split_once('=')
                            .ok_or_else(|| format(format!("metadata line without '=': {line}")))?;
                        metadata.push((k.to_string(), v.to_string()));
                    }
                }
                other => return Err(format(format!("section {name} of unknown kind {other}"))),
            }
        }

        let n_2d = n_elements * n_nodes;
        let n_3d = n_2d * n_levels;
        let n_w = n_2d * (n_levels + 1);
        let mut take = |name: &str, expected| take_section(&mut values, name, expected);
        let mut field_2d = |name: &str| -> Result<DGSolution2D, RestartError> {
            Ok(DGSolution2D {
                data: take(name, Some(n_2d))?,
                n_elements,
                n_nodes,
            })
        };
        let (eta, ubar, vbar) = (
            field_2d("state.eta")?,
            field_2d("state.ubar")?,
            field_2d("state.vbar")?,
        );
        let state = Solution3D {
            n_elements,
            n_nodes,
            n_levels,
            eta,
            ubar,
            vbar,
            u: take("state.u", Some(n_3d))?,
            v: take("state.v", Some(n_3d))?,
            w: take("state.w", Some(n_3d))?,
            temp: take("state.temp", Some(n_3d))?,
            salt: take("state.salt", Some(n_3d))?,
            rho: take("state.rho", Some(n_3d))?,
            eddy_viscosity: take("state.eddy_viscosity", Some(n_w))?,
            eddy_diffusivity: take("state.eddy_diffusivity", Some(n_w))?,
            tke: take("state.tke", None)?,
            gls: take("state.gls", None)?,
        };
        for (name, field) in [("tke", &state.tke), ("gls", &state.gls)] {
            if !field.is_empty() && field.len() != n_w {
                return Err(format(format!(
                    "section state.{name} has {} values, expected 0 or {n_w}",
                    field.len()
                )));
            }
        }

        let t = take("slow_forcing.t", None)?;
        if t.len() > 3 {
            return Err(format(format!("{} slow-forcing steps (at most 3)", t.len())));
        }
        let next_t = take("slow_forcing.next_t", Some(1))?[0];
        let mut g = Vec::with_capacity(t.len());
        for j in 0..t.len() {
            let mut gj = SWESolution2D::new(n_elements, n_nodes);
            for (component, data) in ["h", "hu", "hv"].iter().zip(gj.data.iter_mut()) {
                *data = take(&format!("slow_forcing.{j}.{component}"), Some(n_2d))?;
            }
            g.push(gj);
        }
        let has_reference = order.iter().any(|n| n.starts_with("vertical_reference."));
        let vertical_reference = if has_reference {
            Some([
                take("vertical_reference.temp", Some(n_3d))?,
                take("vertical_reference.salt", Some(n_3d))?,
            ])
        } else {
            None
        };
        let extra = order
            .iter()
            .filter_map(|name| {
                let data = values.remove(name)?;
                Some(name.strip_prefix("extra.").map(|n| (n.to_string(), data)))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| format("an unknown section".into()))?;

        Ok(Self {
            time,
            domain,
            state,
            slow_forcing: SlowForcingRecord { g, t, next_t },
            vertical_reference,
            metadata,
            extra,
        })
    }
}

/// Remove section `name` from `values`, checking its length if `expected`.
fn take_section(
    values: &mut HashMap<String, Vec<f64>>,
    name: &str,
    expected: Option<usize>,
) -> Result<Vec<f64>, RestartError> {
    let data = values
        .remove(name)
        .ok_or_else(|| RestartError::Format(format!("section {name} missing")))?;
    match expected {
        Some(n) if data.len() != n => Err(RestartError::Format(format!(
            "section {name} has {} values, expected {n}",
            data.len()
        ))),
        _ => Ok(data),
    }
}

/// A fingerprint of a domain: the mesh's vertices and elements, the bed at
/// every node and the σ-grid's surfaces (FNV-1a over their bits). Two runs
/// with the same fingerprint step the same columns.
pub fn domain_fingerprint(mesh: &Mesh2D, bed: &[f64], sigma: &SigmaGrid) -> u64 {
    let mut hash = FNV_OFFSET;
    let mut add = |word: u64| {
        for byte in word.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
        }
    };
    add(mesh.vertices.len() as u64);
    for [x, y] in &mesh.vertices {
        add(x.to_bits());
        add(y.to_bits());
    }
    add(mesh.elements.len() as u64);
    for element in &mesh.elements {
        element.iter().for_each(|&v| add(v as u64));
    }
    add(bed.len() as u64);
    bed.iter().for_each(|b| add(b.to_bits()));
    add(sigma.sigma_w().len() as u64);
    sigma.sigma_w().iter().for_each(|s| add(s.to_bits()));
    hash
}

/// The fields of a [`Solution3D`] by name, in the file's order.
fn state_fields(s: &Solution3D) -> [(&'static str, &[f64]); 13] {
    [
        ("eta", &s.eta.data),
        ("ubar", &s.ubar.data),
        ("vbar", &s.vbar.data),
        ("u", &s.u),
        ("v", &s.v),
        ("w", &s.w),
        ("temp", &s.temp),
        ("salt", &s.salt),
        ("rho", &s.rho),
        ("eddy_viscosity", &s.eddy_viscosity),
        ("eddy_diffusivity", &s.eddy_diffusivity),
        ("tke", &s.tke),
        ("gls", &s.gls),
    ]
}

fn write_name<W: Write>(
    out: &mut HashingWriter<W>,
    name: &str,
    kind: u8,
    len: usize,
) -> std::io::Result<()> {
    out.write_all(&(name.len() as u32).to_le_bytes())?;
    out.write_all(name.as_bytes())?;
    out.write_all(&[kind])?;
    out.write_all(&(len as u64).to_le_bytes())
}

/// `path` with `.tmp` appended to its file name.
fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

/// A writer that hashes (FNV-1a) what passes through it.
struct HashingWriter<W> {
    inner: W,
    hash: u64,
}

impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hash: FNV_OFFSET,
        }
    }

    fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        for &byte in bytes {
            self.hash = (self.hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
        }
        self.inner.write_all(bytes)
    }
}

/// A reader that hashes (FNV-1a) what it reads and stops `remaining` bytes in.
struct HashingReader<R> {
    inner: R,
    hash: u64,
    remaining: u64,
}

impl<R: Read> HashingReader<R> {
    fn read_exact(&mut self, bytes: &mut [u8]) -> Result<(), RestartError> {
        if bytes.len() as u64 > self.remaining {
            return Err(RestartError::Format("truncated file".into()));
        }
        self.inner.read_exact(bytes)?;
        self.remaining -= bytes.len() as u64;
        for &byte in bytes.iter() {
            self.hash = (self.hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
        }
        Ok(())
    }

    fn u64(&mut self) -> Result<u64, RestartError> {
        let mut bytes = [0; 8];
        self.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small restart with every kind of section, its values distinct.
    fn sample() -> Restart3D {
        let (ne, nn, nl) = (3, 4, 2);
        let mut state = Solution3D::new(ne, nn, nl);
        let mut x = 0.0;
        let mut fill = |field: &mut Vec<f64>| {
            for v in field.iter_mut() {
                x += 0.37;
                *v = x;
            }
        };
        fill(&mut state.eta.data);
        fill(&mut state.u);
        fill(&mut state.temp);
        fill(&mut state.eddy_diffusivity);
        state.tke = vec![0.0; ne * nn * (nl + 1)];
        state.gls = vec![0.0; ne * nn * (nl + 1)];
        fill(&mut state.gls);
        let mut g = SWESolution2D::new(ne, nn);
        fill(&mut g.data[1]);
        let mut restart = Restart3D {
            time: 1234.5,
            domain: 0xdead_beef,
            state,
            slow_forcing: SlowForcingRecord {
                g: vec![g.clone(), g],
                t: vec![1200.0, 1165.25],
                next_t: 1234.5,
            },
            vertical_reference: Some([vec![1.5; ne * nn * nl], vec![-0.25; ne * nn * nl]]),
            metadata: Vec::new(),
            extra: Vec::new(),
        };
        restart.set_metadata("command", "froya_real_data levels=20 tide3d=1");
        restart.set_extra("station.0.eta", vec![0.1, f64::NAN, -0.3]);
        restart
    }

    fn assert_same(a: &Restart3D, b: &Restart3D) {
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(a.time.to_bits(), b.time.to_bits());
        assert_eq!(a.domain, b.domain);
        for ((name, x), (_, y)) in state_fields(&a.state).into_iter().zip(state_fields(&b.state)) {
            assert_eq!(bits(x), bits(y), "state.{name}");
        }
        assert_eq!(bits(&a.slow_forcing.t), bits(&b.slow_forcing.t));
        assert_eq!(
            a.slow_forcing.next_t.to_bits(),
            b.slow_forcing.next_t.to_bits()
        );
        for (ga, gb) in a.slow_forcing.g.iter().zip(&b.slow_forcing.g) {
            for (x, y) in ga.data.iter().zip(&gb.data) {
                assert_eq!(bits(x), bits(y));
            }
        }
        assert_eq!(a.slow_forcing.g.len(), b.slow_forcing.g.len());
        match (&a.vertical_reference, &b.vertical_reference) {
            (Some([ta, sa]), Some([tb, sb])) => {
                assert_eq!(bits(ta), bits(tb));
                assert_eq!(bits(sa), bits(sb));
            }
            (None, None) => {}
            _ => panic!("vertical reference lost"),
        }
        assert_eq!(a.metadata, b.metadata);
        assert_eq!(a.extra.len(), b.extra.len());
        for ((na, va), (nb, vb)) in a.extra.iter().zip(&b.extra) {
            assert_eq!(na, nb);
            assert_eq!(bits(va), bits(vb));
        }
    }

    #[test]
    fn a_restart_reads_back_bit_for_bit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.restart");
        let restart = sample();
        restart.write(&path).unwrap();
        assert!(!temporary_path(&path).exists(), "the temporary file is left");
        assert_same(&restart, &Restart3D::read(&path).unwrap());

        // Without the optional parts
        let mut bare = sample();
        bare.vertical_reference = None;
        bare.slow_forcing = SlowForcingRecord::empty();
        bare.state.tke.clear();
        bare.state.gls.clear();
        bare.metadata.clear();
        bare.extra.clear();
        bare.write(&path).unwrap();
        assert_same(&bare, &Restart3D::read(&path).unwrap());
    }

    #[test]
    fn a_damaged_or_truncated_restart_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.restart");
        sample().write(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();

        let mut flipped = bytes.clone();
        flipped[bytes.len() / 2] ^= 0x10;
        std::fs::write(&path, &flipped).unwrap();
        assert!(matches!(
            Restart3D::read(&path),
            Err(RestartError::Format(_))
        ));

        for cut in [7, 40, bytes.len() / 2, bytes.len() - 1] {
            std::fs::write(&path, &bytes[..cut]).unwrap();
            assert!(
                matches!(Restart3D::read(&path), Err(RestartError::Format(_))),
                "truncated at {cut} of {} bytes",
                bytes.len()
            );
        }
    }

    #[test]
    fn the_domain_fingerprint_sees_the_bed_and_the_levels() {
        use crate::vertical::UniformStretching;
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);
        let bed = vec![-10.0; 4 * 9];
        let sigma = SigmaGrid::new(4, UniformStretching);
        let base = domain_fingerprint(&mesh, &bed, &sigma);
        assert_eq!(base, domain_fingerprint(&mesh, &bed, &sigma));
        let mut deeper = bed.clone();
        deeper[17] = -10.000000001;
        assert_ne!(base, domain_fingerprint(&mesh, &deeper, &sigma));
        assert_ne!(
            base,
            domain_fingerprint(&mesh, &bed, &SigmaGrid::new(5, UniformStretching))
        );
    }
}
