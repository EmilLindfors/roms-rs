//! Element-block loops of the 3D kernels, parallel with the `parallel`
//! feature, and the per-thread scratch they work in.
//!
//! The 3D kernels write each element's `[node][level]` block of their output
//! fields from shared inputs. [`for_each_block`] runs such a kernel over every
//! block, on rayon's pool with the `parallel` feature and in order without
//! it. Every block is computed by the same code from the same inputs either
//! way, so the result does not depend on the feature or the thread count, bit
//! for bit; reductions across blocks ([`max_over_blocks`]) use only
//! exact operations.
//!
//! A kernel's per-element buffers come from [`Pooled`], a cache of this
//! thread's scratch: rayon's workers persist, so after the first call nothing
//! is allocated.

use std::any::Any;
use std::cell::RefCell;

use super::disjoint::DisjointChunks;

/// Scratch values cached per thread (of any type; a handful at most).
const POOL_CAPACITY: usize = 16;

thread_local! {
    static POOL: RefCell<Vec<Box<dyn Any>>> = const { RefCell::new(Vec::new()) };
}

/// A scratch value taken from this thread's cache, returned to it on drop.
///
/// Take it once per loop or rayon job, not per element: a thread-local
/// lookup per element costs more than many kernels' work.
pub(crate) struct Pooled<T: 'static>(Option<Box<T>>);

impl<T: 'static> Pooled<T> {
    /// A cached `T` for which `fits` holds, or else `make()`.
    pub(crate) fn take(fits: impl Fn(&T) -> bool, make: impl FnOnce() -> T) -> Self {
        let cached = POOL
            .try_with(|pool| {
                let mut pool = pool.try_borrow_mut().ok()?;
                let i = pool
                    .iter()
                    .position(|v| v.downcast_ref::<T>().is_some_and(&fits))?;
                pool.swap_remove(i).downcast::<T>().ok()
            })
            .ok()
            .flatten();
        Self(Some(cached.unwrap_or_else(|| Box::new(make()))))
    }
}

impl<T: 'static> std::ops::Deref for Pooled<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.0.as_ref().expect("present until drop")
    }
}

impl<T: 'static> std::ops::DerefMut for Pooled<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.0.as_mut().expect("present until drop")
    }
}

impl<T: 'static> Drop for Pooled<T> {
    fn drop(&mut self) {
        let Some(value) = self.0.take() else {
            return;
        };
        // `try_with`: the thread-local may already be gone at thread exit
        let _ = POOL.try_with(|pool| {
            if let Ok(mut pool) = pool.try_borrow_mut()
                && pool.len() < POOL_CAPACITY
            {
                pool.push(value);
            }
        });
    }
}

/// Run `f(scratch, block, chunks)` for every `block` in `0..n_blocks`, with
/// `chunks[i]` the `block`-th of `n_blocks` equal chunks of `outputs[i]`
/// (in parallel with the `parallel` feature). `init` makes a worker's
/// scratch, once per loop or rayon job.
pub(crate) fn for_each_block<T, S, const N: usize>(
    n_blocks: usize,
    outputs: [&mut [T]; N],
    init: impl Fn() -> S + Sync + Send,
    f: impl Fn(&mut S, usize, [&mut [T]; N]) + Sync + Send,
) where
    T: Send,
{
    max_over_blocks(n_blocks, outputs, init, |scratch, block, chunks| {
        f(scratch, block, chunks);
        f64::NEG_INFINITY
    });
}

/// [`for_each_block`] for a kernel that returns a value per block: their
/// largest (`−∞` for none; `max` is exact, so this too is independent of the
/// thread count).
pub(crate) fn max_over_blocks<T, S, const N: usize>(
    n_blocks: usize,
    outputs: [&mut [T]; N],
    init: impl Fn() -> S + Sync + Send,
    f: impl Fn(&mut S, usize, [&mut [T]; N]) -> f64 + Sync + Send,
) -> f64
where
    T: Send,
{
    reduce_blocks(n_blocks, outputs, init, f, || f64::NEG_INFINITY, f64::max)
}

/// [`for_each_block`] for a kernel that returns a value per block, combined
/// with `combine` from `identity()`. The grouping of the combinations
/// depends on the thread count: `combine` must be exact (and associative and
/// commutative) for the result not to, e.g. a maximum or an integer count.
pub(crate) fn reduce_blocks<T, S, R, const N: usize>(
    n_blocks: usize,
    outputs: [&mut [T]; N],
    init: impl Fn() -> S + Sync + Send,
    f: impl Fn(&mut S, usize, [&mut [T]; N]) -> R + Sync + Send,
    identity: impl Fn() -> R + Sync + Send,
    combine: impl Fn(R, R) -> R + Sync + Send,
) -> R
where
    T: Send,
    R: Send,
{
    let chunks = outputs.map(|out| {
        assert!(
            out.len().is_multiple_of(n_blocks.max(1)),
            "output of length {} in {n_blocks} blocks",
            out.len()
        );
        let chunk = out.len() / n_blocks.max(1);
        (chunk > 0).then(|| DisjointChunks::new(out, chunk))
    });
    // SAFETY: every block index is visited exactly once (a range), so no
    // chunk is handed out twice. An empty output has no chunks: an empty slice
    let block_of = |block: usize| {
        std::array::from_fn(|i| match &chunks[i] {
            Some(c) => unsafe { c.chunk(block) },
            None => &mut [][..],
        })
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..n_blocks)
            .into_par_iter()
            .map_init(init, |scratch, block| f(scratch, block, block_of(block)))
            .reduce(identity, combine)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = init();
        (0..n_blocks).fold(identity(), |r, block| {
            combine(r, f(&mut scratch, block, block_of(block)))
        })
    }
}

/// Values per job of the streaming loops ([`update_values`],
/// [`update_with`]): large enough that dispatch is negligible next to the
/// memory traffic.
const STREAM_CHUNK: usize = 1 << 14;

/// `f(x)` for every value of `out` (in parallel with the `parallel`
/// feature; elementwise, so the result is the serial one).
pub(crate) fn update_values(out: &mut [f64], f: impl Fn(&mut f64) + Sync + Send) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(STREAM_CHUNK)
            .for_each(|chunk| chunk.iter_mut().for_each(&f));
    }
    #[cfg(not(feature = "parallel"))]
    out.iter_mut().for_each(f);
}

/// `f(x, y)` for every value `x` of `out` and `y` of `input` at the same
/// index (as [`update_values`]).
pub(crate) fn update_with(out: &mut [f64], input: &[f64], f: impl Fn(&mut f64, f64) + Sync + Send) {
    assert_eq!(out.len(), input.len(), "lengths of the streamed fields");
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(STREAM_CHUNK)
            .zip(input.par_chunks(STREAM_CHUNK))
            .for_each(|(out, input)| {
                for (x, &y) in out.iter_mut().zip(input) {
                    f(x, y);
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    for (x, &y) in out.iter_mut().zip(input) {
        f(x, y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_block_is_written_once_from_its_own_chunks() {
        let (mut a, mut b) = (vec![0.0; 12], vec![0.0; 6]);
        let largest = max_over_blocks(
            3,
            [&mut a[..], &mut b[..]],
            || 0usize,
            |calls, block, [a, b]| {
                *calls += 1;
                assert_eq!((a.len(), b.len()), (4, 2));
                a.fill(block as f64);
                b.fill(10.0 * block as f64);
                block as f64
            },
        );
        assert_eq!(largest, 2.0);
        assert_eq!(a, [0., 0., 0., 0., 1., 1., 1., 1., 2., 2., 2., 2.]);
        assert_eq!(b, [0., 0., 10., 10., 20., 20.]);
    }

    #[test]
    fn pooled_scratch_is_reused_on_the_same_thread() {
        let first = Pooled::take(|v: &Vec<f64>| v.len() == 7, || vec![1.0; 7]);
        let address = first.as_ptr();
        drop(first);
        let again = Pooled::take(|v: &Vec<f64>| v.len() == 7, || vec![2.0; 7]);
        assert_eq!(again.as_ptr(), address);
        assert_eq!(again[0], 1.0);
        // Another size is not taken from the cache
        let other = Pooled::take(|v: &Vec<f64>| v.len() == 3, || vec![3.0; 3]);
        assert_eq!(other[0], 3.0);
    }
}
