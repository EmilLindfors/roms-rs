//! Wall-clock phase timers for profiling a run (`DG_PROFILE=1`).
//!
//! [`scope`] times a closure under a name when the environment variable
//! `DG_PROFILE` is set, and runs it untimed otherwise (one relaxed atomic load).
//! [`report`] lists the totals, [`reset`] clears them.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

static TIMES: Mutex<Vec<(&'static str, Duration, u64)>> = Mutex::new(Vec::new());
/// 0: not read yet, 1: off, 2: on.
static ENABLED: AtomicU8 = AtomicU8::new(0);

/// Whether `DG_PROFILE` is set.
#[inline]
pub fn enabled() -> bool {
    match ENABLED.load(Ordering::Relaxed) {
        0 => {
            let on = std::env::var_os("DG_PROFILE").is_some();
            ENABLED.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
        v => v == 2,
    }
}

/// Run `f`, adding its wall time to `name`'s total when profiling.
#[inline]
pub fn scope<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let start = Instant::now();
    let r = f();
    add(name, start.elapsed());
    r
}

/// The start of a timed section ([`stop`]), `None` when not profiling.
#[inline]
pub fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}

/// Add the time since `start` to `name`'s total (nothing for `None`).
#[inline]
pub fn stop(name: &'static str, start: Option<Instant>) {
    if let Some(start) = start {
        add(name, start.elapsed());
    }
}

/// Add `elapsed` to `name`'s total.
pub fn add(name: &'static str, elapsed: Duration) {
    let mut times = TIMES.lock().unwrap_or_else(|e| e.into_inner());
    match times.iter_mut().find(|(n, _, _)| *n == name) {
        Some((_, t, count)) => {
            *t += elapsed;
            *count += 1;
        }
        None => times.push((name, elapsed, 1)),
    }
}

/// Clear the totals.
pub fn reset() {
    TIMES.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

/// The totals, one line per name in order of first use: total seconds, calls,
/// milliseconds per call.
pub fn report() -> String {
    let times = TIMES.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = String::new();
    for (name, t, count) in times.iter() {
        out.push_str(&format!(
            "  {name:<32} {:>9.3} s {count:>9} calls {:>10.3} ms/call\n",
            t.as_secs_f64(),
            1e3 * t.as_secs_f64() / *count as f64
        ));
    }
    out
}
