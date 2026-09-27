//! Mutable access to chosen chunks of a slice from parallel workers.
//!
//! Local time stepping updates a changing subset of elements (and of faces)
//! many times per step. Splitting the whole slice into chunks to reach a few
//! of them costs a pass over every element, which dominated multirate steps
//! with many levels. [`DisjointChunks`] hands out the chunks by index
//! instead; the caller guarantees that no chunk is handed out twice at once
//! (the element and edge lists it works through are free of duplicates).

use std::marker::PhantomData;

/// Chunks of `chunk` elements of a mutable slice, handed out by index.
pub(crate) struct DisjointChunks<'a, T> {
    ptr: *mut T,
    n_chunks: usize,
    chunk: usize,
    _slice: PhantomData<&'a mut [T]>,
}

// The chunks are disjoint and handed out at most once each (see `chunk`),
// so sharing the handle between threads is sharing `&mut` to distinct data.
unsafe impl<T: Send> Send for DisjointChunks<'_, T> {}
unsafe impl<T: Send> Sync for DisjointChunks<'_, T> {}

impl<'a, T> DisjointChunks<'a, T> {
    /// `slice` in chunks of `chunk` elements (its length a multiple).
    pub(crate) fn new(slice: &'a mut [T], chunk: usize) -> Self {
        assert!(chunk > 0 && slice.len().is_multiple_of(chunk));
        Self {
            ptr: slice.as_mut_ptr(),
            n_chunks: slice.len() / chunk,
            chunk,
            _slice: PhantomData,
        }
    }

    /// Chunk `i`.
    ///
    /// # Safety
    /// No other reference to chunk `i` obtained from this handle may be
    /// alive while the result is.
    #[inline]
    pub(crate) unsafe fn chunk(&self, i: usize) -> &'a mut [T] {
        assert!(i < self.n_chunks, "chunk {i} of {}", self.n_chunks);
        // SAFETY: in bounds (checked); exclusive by the caller's guarantee
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(i * self.chunk), self.chunk) }
    }
}

/// Whether `items` has no duplicates (for debug assertions).
pub(crate) fn all_distinct(items: &[u32], bound: usize) -> bool {
    let mut seen = vec![false; bound];
    items
        .iter()
        .all(|&i| !std::mem::replace(&mut seen[i as usize], true))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_are_disjoint_and_writable_in_parallel() {
        let mut data = vec![0usize; 12];
        {
            let chunks = DisjointChunks::new(&mut data, 3);
            let picked = [3u32, 1, 0];
            assert!(all_distinct(&picked, 4));
            std::thread::scope(|s| {
                for &i in &picked {
                    let chunks = &chunks;
                    s.spawn(move || {
                        // SAFETY: each index once
                        let c = unsafe { chunks.chunk(i as usize) };
                        c.iter_mut().for_each(|v| *v = i as usize + 1);
                    });
                }
            });
        }
        assert_eq!(data, [1, 1, 1, 2, 2, 2, 0, 0, 0, 4, 4, 4]);
        assert!(!all_distinct(&[1, 2, 1], 3));
    }
}
