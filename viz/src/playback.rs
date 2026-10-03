//! The snapshots received so far and the model time the viewer shows.
//!
//! [`Playback`] keeps the solver's snapshots in arrival order, dropping the oldest
//! beyond a memory budget, and a clock that runs `rate` seconds of model time per
//! second. The clock never passes the newest snapshot: while the solver is behind, the
//! view waits for it. Between two snapshots the field is interpolated linearly in
//! time ([`crate::field::Field`]). A replay ([`Source::Replay`]) sends the frames of a
//! finished run down the same channel.
//!
//! Keys: Space pauses, `[` and `]` halve and double the rate, ← and → step back and
//! forward by ten snapshots, Home and End jump to the oldest and newest.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Instant;

use bevy::prelude::*;

use crate::solver::{Snapshot, SolverMessage};

/// Where the snapshots come from.
#[derive(Resource)]
pub enum Source {
    /// The solver, running while the viewer draws
    Solver,
    /// The frames of a finished run ([`crate::replay`])
    Replay {
        /// The run's output directory, as named on the command line
        name: String,
        /// Frames in the run
        frames: usize,
        /// Model time of the last frame (s)
        t_last: f64,
    },
}

/// How the run on the solver thread ended.
pub enum SolverState {
    Running,
    Finished { steps: usize, wall: f64 },
    Failed(String),
}

#[derive(Resource)]
pub struct Playback {
    pub frames: VecDeque<Snapshot>,
    /// Model time shown (s)
    pub t: f64,
    /// Model seconds per second
    pub rate: f64,
    pub paused: bool,
    pub solver: SolverState,
    /// Model time between snapshots (s)
    pub interval: f64,
    /// Bytes of snapshots kept at most
    budget: usize,
    /// Set when the shown model time or the snapshots around it changed this frame
    pub changed: bool,
    started: Instant,
    /// Wall time (s) at which the newest snapshot arrived
    pub newest_wall: f64,
}

impl Playback {
    pub fn new(rate: f64, interval: f64, budget_bytes: usize) -> Self {
        Self {
            frames: VecDeque::new(),
            t: 0.0,
            rate,
            paused: false,
            solver: SolverState::Running,
            interval,
            budget: budget_bytes,
            changed: true,
            started: Instant::now(),
            newest_wall: 0.0,
        }
    }

    pub fn oldest(&self) -> Option<f64> {
        self.frames.front().map(|s| s.t)
    }

    pub fn newest(&self) -> Option<f64> {
        self.frames.back().map(|s| s.t)
    }

    /// Model seconds the solver advances per second of wall time.
    pub fn solver_speed(&self) -> f64 {
        self.newest().unwrap_or(0.0) / self.newest_wall.max(1e-9)
    }

    /// Bytes held in snapshots.
    pub fn bytes(&self) -> usize {
        self.frames.iter().map(Snapshot::bytes).sum()
    }

    /// Whether the clock is held at the newest snapshot waiting for the solver.
    pub fn waiting(&self) -> bool {
        !self.paused
            && matches!(self.solver, SolverState::Running)
            && self.newest().is_none_or(|t| self.t >= t)
    }

    /// The snapshots around `self.t` and the weight of the later one.
    pub fn bracket(&self) -> Option<(&Snapshot, &Snapshot, f32)> {
        let later = self.frames.partition_point(|s| s.t < self.t);
        let later = later.min(self.frames.len().checked_sub(1)?);
        let earlier = later.saturating_sub(1);
        let (a, b) = (&self.frames[earlier], &self.frames[later]);
        let w = if b.t > a.t {
            ((self.t - a.t) / (b.t - a.t)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        Some((a, b, w as f32))
    }

    fn clamp(&mut self) {
        if let (Some(lo), Some(hi)) = (self.oldest(), self.newest()) {
            self.t = self.t.clamp(lo, hi);
        }
    }
}

/// The receiving end of the solver's channel (a `Receiver` is `Send` but not `Sync`).
#[derive(Resource)]
pub struct SolverChannel(pub Mutex<Receiver<SolverMessage>>);

pub struct PlaybackPlugin;

impl Plugin for PlaybackPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreUpdate, (receive, keys, advance).chain());
    }
}

fn receive(channel: Res<SolverChannel>, mut playback: ResMut<Playback>) {
    let rx = channel.0.lock().expect("solver channel");
    loop {
        match rx.try_recv() {
            Ok(SolverMessage::Snapshot(snapshot)) => {
                // The budget counts snapshots of this size; all are the same size.
                let fits = (playback.budget / snapshot.bytes().max(1)).max(2);
                playback.frames.push_back(*snapshot);
                while playback.frames.len() > fits {
                    playback.frames.pop_front();
                }
                playback.newest_wall = playback.started.elapsed().as_secs_f64();
                playback.clamp();
                playback.changed = true;
            }
            Ok(SolverMessage::Finished { steps, wall, error }) => {
                playback.solver = match error {
                    None => SolverState::Finished { steps, wall },
                    Some(e) => SolverState::Failed(e),
                };
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                if matches!(playback.solver, SolverState::Running) {
                    playback.solver = SolverState::Failed("the solver thread stopped".into());
                }
                break;
            }
        }
    }
}

fn keys(keys: Res<ButtonInput<KeyCode>>, mut playback: ResMut<Playback>) {
    if keys.just_pressed(KeyCode::Space) {
        playback.paused = !playback.paused;
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        playback.rate = (playback.rate / 2.0).max(1.0);
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        playback.rate = (playback.rate * 2.0).min(1e5);
    }
    let step = 10.0 * playback.interval;
    let seek = if keys.just_pressed(KeyCode::ArrowLeft) {
        Some(playback.t - step)
    } else if keys.just_pressed(KeyCode::ArrowRight) {
        Some(playback.t + step)
    } else if keys.just_pressed(KeyCode::Home) {
        playback.oldest()
    } else if keys.just_pressed(KeyCode::End) {
        playback.newest()
    } else {
        None
    };
    if let Some(t) = seek {
        playback.t = t;
        playback.clamp();
        playback.changed = true;
    }
}

fn advance(time: Res<Time<Real>>, mut playback: ResMut<Playback>) {
    if playback.paused {
        return;
    }
    let before = playback.t;
    playback.t += playback.rate * time.delta_secs_f64();
    playback.clamp();
    if playback.t != before {
        playback.changed = true;
    }
}

/// Clears the change flag once every system has seen it.
pub fn settle(mut playback: ResMut<Playback>) {
    playback.changed = false;
}
