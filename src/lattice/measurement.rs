//! Measurements over the state of the lattice
//!

use crate::lattice::Lattice;

/// Measurement over the lattice
pub trait Measurement {
    /// Type of the measured quantity
    type Result;

    /// Perform measurement on a given lattice
    fn measure(&self, lattice: &impl Lattice) -> Self::Result;
}
