//! System configurations.
//!
//! - [`LatticeState`]: explicit site values on a [`Topology`](crate::topology::Topology).
//! - [`MeanFieldState`]: fully connected system, where only the number of sites in each state
//!   matters, stored in `O(K)` memory.
//!
//! Both keep per value counts up to date on every modification, so macroscopic observables
//! (magnetisation, density of active sites, …) are `O(1)` to measure.

mod init;
mod lattice;
mod mean_field;

pub use init::{CustomInit, Init, Position, Prepare};
pub use lattice::LatticeState;
pub use mean_field::MeanFieldState;

use crate::site::{Site, Spin};

/// Configuration of `len` sites summarised by the number of sites in each state.
pub trait Configuration {
    /// Type of the single site states.
    type Site: Site;

    /// Number of sites.
    fn len(&self) -> usize;

    /// Whether there are no sites.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of sites in each state, indexed by [`Site::index`].
    fn counts(&self) -> &[u32];

    /// Number of sites in state `s`.
    #[inline]
    fn count(&self, s: Self::Site) -> u32 {
        self.counts()[s.index()]
    }

    /// Sum of spin projections `M = Σᵢ sᵢ`.
    #[inline]
    fn total_magnetization(&self) -> i64
    where
        Self::Site: Spin,
    {
        Self::Site::VALUES
            .iter()
            .zip(self.counts())
            .map(|(s, &n)| s.value() as i64 * n as i64)
            .sum()
    }

    /// Sum of squared spin projections `Q = Σᵢ sᵢ²`.
    #[inline]
    fn total_quadrupole(&self) -> i64
    where
        Self::Site: Spin,
    {
        Self::Site::VALUES
            .iter()
            .zip(self.counts())
            .map(|(s, &n)| (s.value() as i64).pow(2) * n as i64)
            .sum()
    }
}
