//! Contact Process State
//!

use crate::{
    cellular_automaton::{contact_process::cell::Binary, state::CellularAutomatonState},
    lattice::Lattice,
};

pub trait ContactProcessState: CellularAutomatonState<Site = Binary> {
    /// Total number of active sites
    fn total_active(&self) -> usize;

    /// Fraction of active sites
    #[inline(always)]
    fn active(&self) -> f64 {
        self.total_active() as f64 / self.site_count() as f64
    }
}

impl<L> ContactProcessState for L
where
    L: CellularAutomatonState<Site = Binary> + Lattice<Site = Binary>,
{
    #[inline(always)]
    fn total_active(&self) -> usize {
        Lattice::sites(self).map(|s| *s as usize).sum()
    }
}
