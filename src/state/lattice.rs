//! Explicit site configurations on a topology.

use std::{
    fmt::{self, Display},
    sync::Arc,
};

use super::Configuration;
use crate::{site::Site, topology::Topology};

/// Site values on a topology, with per value counts kept up to date.
///
/// The topology is shared through an [`Arc`], so cloning a state (e.g. one per Markov chain of an
/// ensemble) only copies the site values.
#[derive(Debug, Clone)]
pub struct LatticeState<S: Site, T: Topology> {
    sites: Vec<S>,
    counts: Box<[u32]>,
    topology: Arc<T>,
}

impl<S: Site, T: Topology> LatticeState<S, T> {
    /// State with every site set to `fill`.
    pub fn uniform(topology: Arc<T>, fill: S) -> Self {
        Self::from_sites(topology.clone(), vec![fill; topology.len()])
    }

    /// State with given site values.
    ///
    /// # Panics
    /// If the number of values differs from the number of sites or does not fit in `u32`.
    pub fn from_sites(topology: Arc<T>, sites: Vec<S>) -> Self {
        assert_eq!(
            sites.len(),
            topology.len(),
            "Number of site values must match the number of sites"
        );
        assert!(u32::try_from(sites.len()).is_ok(), "Too many sites");
        let counts = count_values(&sites);
        Self {
            sites,
            counts,
            topology,
        }
    }

    /// Underlying topology.
    #[inline]
    pub fn topology(&self) -> &Arc<T> {
        &self.topology
    }

    /// Site values.
    #[inline(always)]
    pub fn sites(&self) -> &[S] {
        &self.sites
    }

    /// Value of site `i`.
    #[inline(always)]
    pub fn get(&self, i: usize) -> S {
        self.sites[i]
    }

    /// Set site `i` to `s`.
    #[inline(always)]
    pub fn set(&mut self, i: usize, s: S) {
        let old = std::mem::replace(&mut self.sites[i], s);
        self.counts[old.index()] -= 1;
        self.counts[s.index()] += 1;
    }

    /// Exchange the values of sites `i` and `j`.
    #[inline(always)]
    pub fn swap(&mut self, i: usize, j: usize) {
        self.sites.swap(i, j);
    }

    /// Neighbours of site `i`.
    #[inline(always)]
    pub fn neighbors(&self, i: usize) -> &[u32] {
        self.topology.neighbors(i)
    }

    /// Set every site to `s`.
    pub fn fill(&mut self, s: S) {
        self.sites.fill(s);
        self.counts.fill(0);
        self.counts[s.index()] = self.sites.len() as u32;
    }

    /// Overwrite the site values through `f`, which receives the value slice.
    pub fn assign_with(&mut self, f: impl FnOnce(&mut [S])) {
        f(&mut self.sites);
        self.counts = count_values(&self.sites);
    }

    /// Exchange the site values with `buffer` (of equal length), e.g. after a synchronous update.
    ///
    /// # Panics
    /// If `buffer` has a different length.
    pub fn swap_buffer(&mut self, buffer: &mut Vec<S>) {
        assert_eq!(buffer.len(), self.sites.len(), "Buffer length mismatch");
        std::mem::swap(&mut self.sites, buffer);
        self.counts = count_values(&self.sites);
    }
}

fn count_values<S: Site>(sites: &[S]) -> Box<[u32]> {
    let mut counts = vec![0u32; S::COUNT].into_boxed_slice();
    for s in sites {
        counts[s.index()] += 1;
    }
    counts
}

impl<S: Site, T: Topology> Configuration for LatticeState<S, T> {
    type Site = S;

    #[inline(always)]
    fn len(&self) -> usize {
        self.sites.len()
    }

    #[inline(always)]
    fn counts(&self) -> &[u32] {
        &self.counts
    }
}

impl<S: Site + Display, const D: usize> Display
    for LatticeState<S, crate::topology::Hypercubic<D>>
{
    /// One row per line for `D >= 2` (higher dimensional lattices are printed slice by slice).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let row = self.topology.lengths()[D - 1];
        for (k, chunk) in self.sites.chunks(row).enumerate() {
            if k > 0 {
                writeln!(f)?;
            }
            for s in chunk {
                write!(f, "{s}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        site::{SpinHalf, SpinOne},
        topology::{Chain, Square},
    };

    #[test]
    fn counts_track_modifications() {
        let mut st = LatticeState::uniform(Arc::new(Square::periodic_cube(4)), SpinOne::Zero);
        assert_eq!(st.counts(), &[0, 16, 0]);
        st.set(3, SpinOne::Up);
        st.set(5, SpinOne::Down);
        st.set(5, SpinOne::Up);
        assert_eq!(st.counts(), &[0, 14, 2]);
        assert_eq!(st.total_magnetization(), 2);
        assert_eq!(st.total_quadrupole(), 2);
        st.swap(3, 0);
        assert_eq!(st.get(0), SpinOne::Up);
        assert_eq!(st.counts(), &[0, 14, 2]);
        st.fill(SpinOne::Down);
        assert_eq!(st.counts(), &[16, 0, 0]);
        st.assign_with(|s| s[..4].fill(SpinOne::Up));
        assert_eq!(st.counts(), &[12, 0, 4]);
    }

    #[test]
    fn display_rows() {
        let mut st = LatticeState::uniform(Arc::new(Square::periodic([2, 3])), SpinHalf::Down);
        st.set(1, SpinHalf::Up);
        assert_eq!(st.to_string(), " █ \n   ");
        let st = LatticeState::uniform(Arc::new(Chain::periodic([3])), SpinHalf::Up);
        assert_eq!(st.to_string(), "███");
    }
}
