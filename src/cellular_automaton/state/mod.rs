//! Cellular Automaton State
//!

use std::ops::{Index, IndexMut};

/// Cellular Automaton State
pub trait CellularAutomatonState:
    Index<Self::Index, Output = Self::Site> + IndexMut<Self::Index, Output = Self::Site>
{
    /// Single site state
    type Site;

    /// Index for an individual site
    type Index: Copy;

    /// Total number of sites
    fn site_count(&self) -> usize;

    /// Iterator over all indices of the sites
    fn indices(&self) -> impl Iterator<Item = Self::Index>;

    /// Iterator over all sites
    fn sites<'a>(&'a self) -> impl Iterator<Item = &'a Self::Site>
    where
        Self::Site: 'a;

    /// Mutable iterator over all sites
    fn sites_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Self::Site>
    where
        Self::Site: 'a;

    /// Iterator over indices and sites pairs
    fn indexed_sites<'a>(&'a self) -> impl Iterator<Item = (Self::Index, &'a Self::Site)>
    where
        Self::Site: 'a;

    /// Iterator over indices and sites pairs
    fn indexed_sites_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (Self::Index, &'a mut Self::Site)>
    where
        Self::Site: 'a;
}

/// Lattice Cellular Automaton State
pub mod lattice;
