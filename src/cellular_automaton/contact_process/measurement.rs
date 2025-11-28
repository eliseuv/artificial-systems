//! Contact Process Measurements
//!

use crate::{
    cellular_automaton::contact_process::{cell::Binary, lattice::LatticeContactProcess},
    lattice::Lattice,
    systems::Measurement,
};

/// Total number of active sites
pub struct TotalActiveSites;

impl<L> Measurement<LatticeContactProcess<L>> for TotalActiveSites
where
    L: Lattice<Site = Binary>,
{
    type Result = usize;

    fn measure(system: &LatticeContactProcess<L>) -> Self::Result {
        system.total_active()
    }
}

/// Fraction of active sites
pub struct ActiveSites;

impl<L> Measurement<LatticeContactProcess<L>> for ActiveSites
where
    L: Lattice<Site = Binary>,
{
    type Result = f64;

    fn measure(system: &LatticeContactProcess<L>) -> Self::Result {
        system.active()
    }
}
