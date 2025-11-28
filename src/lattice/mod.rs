//! Lattice
//!

use std::{
    mem::MaybeUninit,
    ops::{Index, IndexMut},
};

use crate::{
    lattice::initial_state::InitialStateSpec,
    systems::{Measurement, StateResetSpec},
};

/// Arbitrary lattice
pub trait Lattice:
    Index<Self::Index, Output = Self::Site> + IndexMut<Self::Index, Output = Self::Site>
{
    /// Lattice shape specification
    type Shape;

    /// Index for sites
    type Index: Copy;

    /// Single site state
    type Site;

    /// Total number of sites
    fn site_count(&self) -> usize;

    /// Iterator over all indices of the lattice
    fn indices(&self) -> impl Iterator<Item = Self::Index>;

    /// Iterator over all sites
    fn sites<'a>(&'a self) -> impl Iterator<Item = &'a Self::Site>
    where
        Self::Site: 'a;

    /// Mutable iterator over all sites
    fn sites_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Self::Site>
    where
        Self::Site: 'a;

    /// Iterate over indices and sites
    fn indexed_sites<'a>(&'a self) -> impl Iterator<Item = (Self::Index, &'a Self::Site)>
    where
        Self::Site: 'a;

    /// Mutable iterator over indices and sites
    fn indexed_sites_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (Self::Index, &'a mut Self::Site)>
    where
        Self::Site: 'a;
    /// Swap sites
    fn swap(&mut self, i: Self::Index, j: Self::Index);

    /// Iterator over the indices of all nearest neighbors of a given site
    fn nearest_neighbors_indices(&self, idx: Self::Index) -> impl Iterator<Item = Self::Index>;

    /// Iterator over the state of all nearest neighbors of a given site
    fn nearest_neighbors(&self, idx: Self::Index) -> impl Iterator<Item = &Self::Site>;

    /// Iterator over the indices of all pairs of nearest neighbors
    fn nearest_neighbors_pairs_indices(&self) -> impl Iterator<Item = (Self::Index, Self::Index)>;

    /// Iterator over the states of all pairs of nearest neighbors
    fn nearest_neighbors_pairs(&self) -> impl Iterator<Item = (&Self::Site, &Self::Site)>;

    /// Construct a new uninitialized lattice
    fn new_uninit(shape: Self::Shape) -> MaybeUninit<Self>
    where
        Self: Sized;

    /// Lattice construction
    #[inline(always)]
    fn new<I>(shape: Self::Shape, spec: &mut I) -> Self
    where
        Self: Sized,
        I: InitialStateSpec<Self>,
    {
        spec.construct(shape)
    }

    /// Lattice reset
    #[inline(always)]
    fn reset<I>(&mut self, spec: &mut I)
    where
        Self: Sized,
        I: StateResetSpec<Self>,
    {
        spec.reset(self)
    }

    /// Measurement on the lattice
    #[inline(always)]
    fn measure<M>(&self) -> M::Result
    where
        Self: Sized,
        M: Measurement<Self>,
    {
        M::measure(self)
    }
}

/// Initial lattice state specifications
pub mod initial_state;

/// Diffusion on the lattice
pub mod diffusion;

/// Square lattices
pub mod square_lattice;
