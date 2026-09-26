//! Contact Process State
//!

use rand::Rng;

/// Contact Process State
pub trait ContactProcessState {
    /// Total number of active sites
    fn total_active(&self) -> usize;

    /// Fraction of active sites
    fn active(&self) -> f64;
}

/// All sites active initial state
pub struct AllActive;

/// Random initial state
pub struct Random<'a, R>
where
    R: Rng + ?Sized,
{
    rng: &'a mut R,
}

impl<'a, R> Random<'a, R>
where
    R: Rng + ?Sized,
{
    pub fn new(rng: &'a mut R) -> Self {
        Self { rng }
    }
}

/// Lattice Contact Process State
pub mod lattice;
