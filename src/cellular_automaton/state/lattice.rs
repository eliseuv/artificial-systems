//! Lattice Cellular Automaton State
//!

use crate::{cellular_automaton::state::CellularAutomatonState, lattice::Lattice};

impl<L: Lattice> CellularAutomatonState for L {
    type Site = L::Site;

    type Index = L::Index;

    #[inline(always)]
    fn site_count(&self) -> usize {
        Lattice::site_count(self)
    }

    #[inline(always)]
    fn indices(&self) -> impl Iterator<Item = Self::Index> {
        Lattice::indices(self)
    }

    #[inline(always)]
    fn sites<'a>(&'a self) -> impl Iterator<Item = &'a Self::Site>
    where
        Self::Site: 'a,
    {
        Lattice::sites(self)
    }

    #[inline(always)]
    fn sites_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Self::Site>
    where
        Self::Site: 'a,
    {
        Lattice::sites_mut(self)
    }

    #[inline(always)]
    fn indexed_sites<'a>(&'a self) -> impl Iterator<Item = (Self::Index, &'a Self::Site)>
    where
        Self::Site: 'a,
    {
        Lattice::indexed_sites(self)
    }

    #[inline(always)]
    fn indexed_sites_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (Self::Index, &'a mut Self::Site)>
    where
        Self::Site: 'a,
    {
        Lattice::indexed_sites_mut(self)
    }
}
