//! Square Lattices
//!

use ndarray::{Array, Axis, Dimension, Ix1, Ix2, Ix3};
use rand_distr::Uniform;

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

/// Square Lattice
/// TODO: Generalize all lattice methods to an arbitrary dimensionality
/// TODO: Is there a way to avoid these auxiliary fields, maybe lazily creating and caching them.
#[derive(Debug)]
pub struct SquareLattice<T, D>
where
    T: Clone + Copy,
    D: Dimension,
{
    /// Lattice state $\in \mathbb{R}^n$
    pub(crate) state: Array<T, D>,
    /// Periodicity
    pub(crate) period: Periodicity,
    /// Uniform distribution over all sites
    site_dist: Uniform<usize>,
}

impl<T, D> SquareLattice<T, D>
where
    D: Dimension,
    T: Clone + Copy,
{
    /// Side length of the square lattice
    #[inline(always)]
    pub fn length(&self) -> usize {
        self.state.len_of(Axis(0))
    }

    /// Total number of sites in the lattice
    #[inline(always)]
    pub fn site_count(&self) -> usize {
        self.state.len()
    }

    /// Iterator over all sites of the lattice
    #[inline(always)]
    pub fn sites(&self) -> impl Iterator<Item = &T> {
        self.state.iter()
    }

    /// Iterator over all sites of the lattice
    #[inline(always)]
    pub fn sites_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.state.iter_mut()
    }

    /// Set all sites to same state
    #[inline(always)]
    pub fn fill(&mut self, site: T) {
        self.state.fill(site)
    }
}

/// One-dimensional Lattice
pub type SquareLattice1D<T> = SquareLattice<T, Ix1>;
pub mod impl_1d;

/// Two-dimensional square lattice
pub type SquareLattice2D<T> = SquareLattice<T, Ix2>;
pub mod impl_2d;

/// Three-dimensional square lattice
pub type SquareLattice3D<T> = SquareLattice<T, Ix3>;
pub mod impl_3d;
