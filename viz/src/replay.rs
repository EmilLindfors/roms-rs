//! A finished run played back from its output instead of the solver.
//!
//! Two kinds of output replay:
//! - A snapshot file (`dg_rs::io::SnapshotWriter`, `.dgsnap`), as
//!   `examples/froya_real_data.rs snapshot_minutes=N` writes it: f32 η, u, v per node,
//!   with the mesh, the bed, the clock and the stations in its header; a 3D file
//!   (`examples/farm_3d.rs snapshot=`, or a live 3D run saved by the viewer) adds the
//!   σ-grid and u, v, T on every level, which become the snapshots' [`Layers`]. The
//!   scenario is built from the file alone ([`Scenario::from_snapshot`]: mesh, bed,
//!   cages, σ-grid, section), so the run's domain data and builder are not needed.
//!   [`writer`] and [`save`] write such a file from the viewer's own snapshots
//!   (`--save-snapshot`, live or replayed).
//! - A directory of VTU frames (`froya_NNNN.vtu`, `dg_rs::io::write_vtk_swe`): every
//!   element's nodes in the solver's order, with `eta`, `u`, `v`, `bathymetry` and the
//!   model time (`TimeValue`), in ASCII. These carry no mesh connectivity, so the
//!   scenario is built as the run was (Frøya), and [`Replay::vtu`] checks the first
//!   frame's nodes and bed against it. `--save-snapshot` writes the frames to a
//!   snapshot file while they are read, a tenth of their size.
//!
//! [`spawn`] reads the frames on a thread of its own (VTU a batch at a time in
//! parallel) and sends them down the solver's channel ([`SolverMessage`]), so the
//! playback treats a replay as a solver that runs ahead of the view: the first frame
//! shows at once. Between frames the field is linear in time
//! ([`crate::field::Field`]), as between the solver's snapshots.

use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};
use std::time::Instant;

use dg_rs::io::{SnapshotError, SnapshotFrame, SnapshotReader, SnapshotWriter};
use dg_rs::source::CageFootprint;
use dg_rs::time::ModelClock;
use dg_rs::types::ElementIndex;
use rayon::prelude::*;

use crate::particles::ParticleSnapshot;
use crate::scenario::Scenario;
use crate::solver::{H_DRY, Layers, Snapshot, SolverMessage};

/// How far (m) the frames' node positions and bed may lie from the scenario's: the
/// files carry ten significant digits, ≈ 1e-6 m over a domain tens of km wide.
const TOLERANCE: f64 = 1e-3;

/// Where the frames are.
enum Frames {
    /// VTU files, in time order
    Vtu(Vec<PathBuf>),
    /// A snapshot file, its number of frames when opened, and its layers in 3D
    Snapshot(Box<SnapshotReader>, usize, Option<LayerFields>),
}

/// The frames of a finished run.
pub struct Replay {
    frames: Frames,
    /// Model time of the first and the last frame (s)
    pub t_first: f64,
    pub t_last: f64,
    /// Model time between frames (s)
    pub interval: f64,
    /// The run's clock (UTC of model time 0)
    pub clock: Option<ModelClock>,
    /// The scenario's bed at the nodes, for the dry nodes
    bed: Vec<f64>,
    /// Where the frames are also written as they are read
    save: Option<SnapshotWriter>,
}

impl Replay {
    /// The `*.vtu` frames in `dir` (in file-name order, which is time order for
    /// zero-padded numbers), checked against `scenario`. The clock is that of the
    /// run's `run.log` (its `Clock:` line), if it has one.
    pub fn vtu(dir: &Path, scenario: &Scenario) -> Result<Self, Box<dyn Error>> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "vtu"))
            .collect();
        files.sort();
        let (Some(first), Some(last)) = (files.first(), files.last()) else {
            return Err(format!("no .vtu frames in {}", dir.display()).into());
        };

        // The nodes and the bed of the first frame against the scenario's
        let text = std::fs::read_to_string(first)?;
        let (mesh, ops) = (&scenario.mesh, &scenario.ops);
        let n = mesh.n_elements * ops.n_nodes;
        let points = floats(array(&text, "<Points>")?)?;
        if points.len() != 3 * n {
            return Err(format!(
                "{} has {} points, but the scenario has {n} nodes ({} elements of {}): \
                 the run used another mesh or order",
                first.display(),
                points.len() / 3,
                mesh.n_elements,
                ops.n_nodes
            )
            .into());
        }
        let mut off = 0.0f64;
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let p = &points[3 * (k.as_usize() * ops.n_nodes + i)..];
                off = off.max((p[0] - x).hypot(p[1] - y));
            }
        }
        if off > TOLERANCE {
            return Err(format!(
                "the nodes of {} lie up to {off:.3} m from the scenario's: the run used another mesh",
                first.display()
            )
            .into());
        }
        let bed = floats(array(&text, "Name=\"bathymetry\"")?)?;
        let bed_off = bed
            .iter()
            .zip(&scenario.bathymetry.data)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        if bed.len() != n || bed_off > TOLERANCE {
            return Err(format!(
                "the bed of {} differs from the scenario's by up to {bed_off:.3} m: \
                 the run built its bed otherwise",
                first.display()
            )
            .into());
        }

        let t_first = time_value(&text)?;
        let t_last = time_value(&tail(last)?)?;
        let interval = match files.get(1) {
            Some(second) => time_value(&tail(second)?)? - t_first,
            None => 0.0,
        };
        Ok(Self {
            frames: Frames::Vtu(files),
            t_first,
            t_last,
            interval,
            clock: run_clock(&dir.join("run.log")),
            bed: scenario.bathymetry.data.clone(),
            save: None,
        })
    }

    /// The snapshot file at `path`, and the scenario of its header.
    pub fn snapshot(path: &Path) -> Result<(Self, Scenario), Box<dyn Error>> {
        let mut reader = SnapshotReader::open(path)?;
        let n = reader.n_frames()?;
        if n == 0 {
            return Err("the file has no frames yet".into());
        }
        let t_first = reader.time(0)?;
        let t_last = reader.time(n - 1)?;
        // The mean spacing: a run's callbacks need not be evenly spaced
        let interval = if n > 1 {
            (t_last - t_first) / (n - 1) as f64
        } else {
            0.0
        };
        let header = reader.header().clone();
        let clock = header.clock;
        // The layered fields the viewer draws: u, v and T on every level
        let layers = match &header.levels {
            Some(levels) => {
                let field = |name: &str| {
                    levels
                        .field(name)
                        .ok_or_else(|| format!("the file's levels have no {name} field"))
                };
                Some(LayerFields {
                    n_levels: levels.sigma.n_levels(),
                    u: field("u")?,
                    v: field("v")?,
                    temp: levels.field("temp"),
                })
            }
            None => None,
        };
        let scenario = Scenario::from_snapshot(header);
        let replay = Self {
            frames: Frames::Snapshot(Box::new(reader), n, layers),
            t_first,
            t_last,
            interval,
            clock,
            bed: scenario.bathymetry.data.clone(),
            save: None,
        };
        Ok((replay, scenario))
    }

    /// Also write the frames to the snapshot file `path` as they are read
    /// ([`writer`]).
    pub fn save_to(&mut self, path: &Path, scenario: &Scenario) -> Result<(), Box<dyn Error>> {
        self.save = Some(writer(path, scenario, self.clock.as_ref())?);
        Ok(())
    }

    pub fn frames(&self) -> usize {
        match &self.frames {
            Frames::Vtu(files) => files.len(),
            Frames::Snapshot(_, n, _) => *n,
        }
    }
}

/// Where a 3D snapshot file's frames hold what the viewer's [`Layers`] draw.
#[derive(Clone, Copy)]
struct LayerFields {
    n_levels: usize,
    /// Indices into the frame's layered fields; no temperature is drawn as 0 °C
    u: usize,
    v: usize,
    temp: Option<usize>,
}

/// The layered fields the viewer writes: what [`Layers`] holds.
const VIEWER_FIELDS: [&str; 3] = ["u", "v", "temp"];

/// A snapshot file at `path` for the scenario's runs: its mesh, bed and (in 3D)
/// σ-grid, `clock`, and as metadata the title, the point of interest, the cages
/// (`cage=x,y,radius,net_depth,drag_per_length`), the section of a 3D run
/// (`section=x0,y0,x1,y1`) and the periods of a periodic mesh (`periodic=x,y`), so
/// that [`Scenario::from_snapshot`] rebuilds what the viewer draws. Frames go in with
/// [`save`].
pub fn writer(
    path: &Path,
    scenario: &Scenario,
    clock: Option<&ModelClock>,
) -> Result<SnapshotWriter, Box<dyn Error>> {
    let title = scenario
        .name
        .split(':')
        .next()
        .unwrap_or_default()
        .to_string();
    let mut metadata = vec![
        ("title", title),
        ("source", "dg-viz --save-snapshot".to_string()),
        (
            "point_of_interest",
            format!("{:.1},{:.1}", scenario.farm[0], scenario.farm[1]),
        ),
    ];
    for cage in &scenario.cages {
        if let CageFootprint::Circle { center, radius } = cage.footprint {
            metadata.push((
                "cage",
                format!(
                    "{},{},{radius},{},{}",
                    center[0], center[1], cage.net_depth, cage.drag_per_length
                ),
            ));
        }
    }
    if let Some([x, y]) = scenario.periodic {
        metadata.push(("periodic", format!("{x},{y}")));
    }
    if let Some(three_d) = &scenario.three_d {
        let [[x0, y0], [x1, y1]] = three_d.section;
        metadata.push(("section", format!("{x0},{y0},{x1},{y1}")));
    }
    let metadata: Vec<(&str, &str)> = metadata.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let levels = scenario
        .three_d
        .as_ref()
        .map(|three_d| (three_d.sigma.as_ref(), &VIEWER_FIELDS[..]));
    Ok(SnapshotWriter::create_with(
        path,
        &scenario.mesh,
        &scenario.ops,
        &scenario.bathymetry.data,
        clock,
        &metadata,
        H_DRY as f64,
        levels,
    )?)
}

/// Append the viewer's snapshot `s` to a file from [`writer`].
pub fn save(writer: &mut SnapshotWriter, s: &Snapshot) -> Result<(), SnapshotError> {
    match &s.layers {
        Some(l) => writer.write_fields(s.t, &s.eta, &s.u, &s.v, &[&l.u, &l.v, &l.temp]),
        None => writer.write_fields(s.t, &s.eta, &s.u, &s.v, &[]),
    }
}

/// Read the frames on a thread of its own (VTU files `threads` at a time); they
/// arrive on the returned channel in time order, then [`SolverMessage::Finished`]
/// with the number of frames as its steps.
pub fn spawn(replay: Replay, threads: usize) -> Receiver<SolverMessage> {
    let (tx, rx) = channel();
    std::thread::Builder::new()
        .name("dg-viz replay".into())
        .spawn(move || {
            let started = Instant::now();
            let n = replay.frames();
            let error = read_all(replay, threads, |snapshot| {
                tx.send(SolverMessage::Snapshot(Box::new(snapshot))).is_ok()
            })
            .err();
            let _ = tx.send(SolverMessage::Finished {
                steps: n,
                wall: started.elapsed().as_secs_f64(),
                error,
            });
        })
        .expect("spawn the replay thread");
    rx
}

/// Hand every frame to `send` in time order (saving it too if asked), until `send`
/// returns false (the viewer has closed).
fn read_all(
    mut replay: Replay,
    threads: usize,
    mut send: impl FnMut(Snapshot) -> bool,
) -> Result<(), String> {
    let mut deliver = |snapshot: Snapshot, writer: &mut Option<SnapshotWriter>| {
        if let Some(writer) = writer {
            save(writer, &snapshot).map_err(|e| format!("saving the snapshot file: {e}"))?;
        }
        Ok::<bool, String>(send(snapshot))
    };
    match replay.frames {
        Frames::Vtu(files) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("dg-viz replay {i}"))
                .build()
                .map_err(|e| e.to_string())?;
            for batch in files.chunks(threads.max(1)) {
                let frames: Vec<_> = pool.install(|| {
                    batch
                        .par_iter()
                        .map(|path| {
                            vtu_frame(path, &replay.bed)
                                .map_err(|e| format!("{}: {e}", path.display()))
                        })
                        .collect()
                });
                for frame in frames {
                    if !deliver(frame?, &mut replay.save)? {
                        return Ok(());
                    }
                }
            }
        }
        Frames::Snapshot(mut reader, n, layer_fields) => {
            let mut frame = SnapshotFrame::default();
            for i in 0..n {
                reader
                    .read_frame_into(i, &mut frame)
                    .map_err(|e| format!("frame {i}: {e}"))?;
                let mut snapshot = snapshot(frame.t, &frame.eta, &frame.u, &frame.v, &replay.bed);
                snapshot.layers = layer_fields.map(|f| Layers {
                    n_levels: f.n_levels,
                    u: frame.layers[f.u].clone(),
                    v: frame.layers[f.v].clone(),
                    temp: match f.temp {
                        Some(t) => frame.layers[t].clone(),
                        None => vec![0.0; frame.layers[f.u].len()],
                    },
                });
                if !deliver(snapshot, &mut replay.save)? {
                    return Ok(());
                }
            }
        }
    }
    Ok(())
}

/// The viewer's snapshot of η and (u, v), the velocity zeroed where the water is no
/// deeper than [`H_DRY`] over the bed, as the solver's snapshots have it.
fn snapshot<T: Copy + Into<f64>>(t: f64, eta: &[T], u: &[T], v: &[T], bed: &[f64]) -> Snapshot {
    let wet = |i: usize| eta[i].into() - bed[i] > H_DRY as f64;
    let velocity = |c: &[T]| {
        (0..bed.len())
            .map(|i| if wet(i) { c[i].into() as f32 } else { 0.0 })
            .collect()
    };
    Snapshot {
        t,
        eta: eta.iter().map(|&x| x.into() as f32).collect(),
        u: velocity(u),
        v: velocity(v),
        particles: ParticleSnapshot::default(),
        layers: None,
    }
}

/// One VTU frame as a snapshot.
fn vtu_frame(path: &Path, bed: &[f64]) -> Result<Snapshot, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    let n = bed.len();
    let field = |name: &str| -> Result<Vec<f64>, Box<dyn Error>> {
        let values = floats(array(&text, &format!("Name=\"{name}\""))?)?;
        if values.len() != n {
            return Err(format!("{name} has {} values, not {n}", values.len()).into());
        }
        Ok(values)
    };
    let (eta, u, v) = (field("eta")?, field("u")?, field("v")?);
    Ok(snapshot(time_value(&text)?, &eta, &u, &v, bed))
}

/// The body of a data array: the text from the first `>` after `key` (an attribute of
/// the array's tag, such as `Name="eta"`, or the element holding it, `<Points>`) to
/// the next `</DataArray>`.
fn array<'a>(text: &'a str, key: &str) -> Result<&'a str, String> {
    let missing = || format!("no {key} data array");
    let after = text.find(key).ok_or_else(missing)? + key.len();
    let body = after + text[after..].find('>').ok_or_else(missing)? + 1;
    let end = body + text[body..].find("</DataArray>").ok_or_else(missing)?;
    Ok(&text[body..end])
}

fn floats(body: &str) -> Result<Vec<f64>, String> {
    body.split_ascii_whitespace()
        .map(|s| s.parse().map_err(|_| format!("bad number {s}")))
        .collect()
}

/// The model time of a frame (its `TimeValue` field).
fn time_value(text: &str) -> Result<f64, String> {
    match floats(array(text, "Name=\"TimeValue\"")?)?.as_slice() {
        [t] => Ok(*t),
        _ => Err("TimeValue is not one number".into()),
    }
}

/// The end of a frame file, which holds its `TimeValue`.
fn tail(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    file.seek(SeekFrom::Start(len.saturating_sub(1024)))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// The clock of a `froya_real_data` run, from its log's `Clock: t = 0 at … UTC` line.
fn run_clock(log: &Path) -> Option<ModelClock> {
    let text = std::fs::read_to_string(log).ok()?;
    let line = text
        .lines()
        .find_map(|l| l.trim().strip_prefix("Clock: t = 0 at "))?;
    ModelClock::parse(line.trim_end_matches(" UTC")).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: &str = r#"<VTKFile>
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          1.0e0 2.0e0 0.0 3.0e0 4.0e0 0.0
        </DataArray>
      </Points>
      <PointData Scalars="h">
        <DataArray type="Float64" Name="u" format="ascii">
          5.0e-1 -2.5e-1
        </DataArray>
        <DataArray type="Float64" Name="eta" format="ascii">
          1.0e-1 -1.0e0
        </DataArray>
      </PointData>
    <FieldData>
      <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1" format="ascii">
        3.6000000000e3
      </DataArray>
    </FieldData>
    </VTKFile>"#;

    #[test]
    fn reads_the_arrays_of_a_frame() {
        assert_eq!(
            floats(array(FRAME, "<Points>").unwrap()).unwrap(),
            [1.0, 2.0, 0.0, 3.0, 4.0, 0.0]
        );
        // The array named, not the next one; `Name="u"` is not matched inside another name
        assert_eq!(
            floats(array(FRAME, "Name=\"u\"").unwrap()).unwrap(),
            [0.5, -0.25]
        );
        assert_eq!(
            floats(array(FRAME, "Name=\"eta\"").unwrap()).unwrap(),
            [0.1, -1.0]
        );
        assert_eq!(time_value(FRAME).unwrap(), 3600.0);
        assert!(array(FRAME, "Name=\"v\"").is_err());
    }
}
