//! A finished run played back from its VTU output instead of the solver.
//!
//! `examples/froya_real_data.rs` writes a `froya_NNNN.vtu` frame every `output_minutes`
//! (`dg_rs::io::write_vtk_swe`): every element's nodes in the solver's order, so the
//! scenario's own node numbering applies, with `eta`, `u`, `v`, `bathymetry` and the
//! model time (`TimeValue`), in ASCII. [`Replay::open`] lists the frames of a directory
//! and checks the first one's nodes and bed against the scenario, which must be built
//! as the run was (the same mesh, order and bed). [`spawn`] then reads the frames on a
//! thread of its own, a batch at a time in parallel, and sends them down the solver's
//! channel ([`SolverMessage`]), so the playback treats a replay as a solver that runs
//! ahead of the view: the first frame shows at once, and the 15-day Frøya run (385
//! frames of 18 MB) is read in well under a minute, faster than it is played back.
//! Between frames the field is linear in time ([`crate::field::Field`]), as between
//! the solver's snapshots.

use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};
use std::time::Instant;

use dg_rs::time::ModelClock;
use dg_rs::types::ElementIndex;
use rayon::prelude::*;

use crate::particles::ParticleSnapshot;
use crate::scenario::Scenario;
use crate::solver::{H_DRY, Snapshot, SolverMessage};

/// How far (m) the frames' node positions and bed may lie from the scenario's: the
/// files carry ten significant digits, ≈ 1e-6 m over a domain tens of km wide.
const TOLERANCE: f64 = 1e-3;

/// The frames of a finished run.
pub struct Replay {
    pub dir: PathBuf,
    /// The frame files, in time order
    files: Vec<PathBuf>,
    /// Model time of the first and the last frame (s)
    pub t_first: f64,
    pub t_last: f64,
    /// Model time between frames (s)
    pub interval: f64,
    /// The run's clock (UTC of model time 0), from the `Clock:` line of its `run.log`
    pub clock: Option<ModelClock>,
    /// The scenario's bed at the nodes, for the dry nodes
    bed: Vec<f64>,
}

impl Replay {
    /// The `*.vtu` frames in `dir` (in file-name order, which is time order for
    /// zero-padded numbers), checked against `scenario`.
    pub fn open(dir: &Path, scenario: &Scenario) -> Result<Self, Box<dyn Error>> {
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
            dir: dir.to_path_buf(),
            t_first,
            t_last,
            interval,
            clock: run_clock(&dir.join("run.log")),
            bed: scenario.bathymetry.data.clone(),
            files,
        })
    }

    pub fn frames(&self) -> usize {
        self.files.len()
    }
}

/// Read the frames on a thread of its own, `threads` files at a time; they arrive on
/// the returned channel in time order, then [`SolverMessage::Finished`] with the
/// number of frames as its steps.
pub fn spawn(replay: Replay, threads: usize) -> Receiver<SolverMessage> {
    let (tx, rx) = channel();
    std::thread::Builder::new()
        .name("dg-viz replay".into())
        .spawn(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("dg-viz replay {i}"))
                .build()
                .expect("replay thread pool");
            let started = Instant::now();
            for batch in replay.files.chunks(threads.max(1)) {
                let frames: Vec<_> = pool.install(|| {
                    batch
                        .par_iter()
                        .map(|path| {
                            frame(path, &replay.bed).map_err(|e| format!("{}: {e}", path.display()))
                        })
                        .collect()
                });
                for frame in frames {
                    let message = match frame {
                        Ok(snapshot) => SolverMessage::Snapshot(Box::new(snapshot)),
                        Err(e) => {
                            let _ = tx.send(SolverMessage::Finished {
                                steps: 0,
                                wall: started.elapsed().as_secs_f64(),
                                error: Some(e),
                            });
                            return;
                        }
                    };
                    // Fails only once the viewer has closed
                    if tx.send(message).is_err() {
                        return;
                    }
                }
            }
            let _ = tx.send(SolverMessage::Finished {
                steps: replay.files.len(),
                wall: started.elapsed().as_secs_f64(),
                error: None,
            });
        })
        .expect("spawn the replay thread");
    rx
}

/// One frame as a snapshot: η, and (u, v) where the water is deeper than [`H_DRY`]
/// over the bed, as the solver's snapshots have them.
fn frame(path: &Path, bed: &[f64]) -> Result<Snapshot, Box<dyn Error>> {
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
    let wet = |i: usize| eta[i] - bed[i] > H_DRY as f64;
    let velocity = |c: &[f64]| {
        (0..n)
            .map(|i| if wet(i) { c[i] as f32 } else { 0.0 })
            .collect()
    };
    Ok(Snapshot {
        t: time_value(&text)?,
        eta: eta.iter().map(|&x| x as f32).collect(),
        u: velocity(&u),
        v: velocity(&v),
        particles: ParticleSnapshot::default(),
        layers: None,
    })
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
