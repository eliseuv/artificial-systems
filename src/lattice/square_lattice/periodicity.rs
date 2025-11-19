//! Periodicity
//!

/// Periodicity
///
/// prev = [n-1, 0, ..., n-3, n-2]
/// next = [1, 2, ..., n-1, 0]
#[derive(Debug, Clone)]
pub struct Periodicity {
    prev: Vec<usize>,
    next: Vec<usize>,
}

impl Periodicity {
    /// Create new periodicity with a given length
    pub fn new(length: usize) -> Self {
        // Create vectors
        let mut prev: Vec<usize> = (0..length).map(|k| k.wrapping_sub(1)).collect();
        let mut next: Vec<usize> = (0..length).map(|k| k + 1).collect();
        // Periodic boundaries
        prev[0] = length - 1;
        next[length - 1] = 0;

        Self { prev, next }
    }

    /// Get previous index
    #[inline(always)]
    pub fn prev(&self, k: usize) -> usize {
        self.prev[k]
    }

    /// Get next index
    #[inline(always)]
    pub fn next(&self, k: usize) -> usize {
        self.next[k]
    }
}
