//! Fully connected (mean-field) configurations.

use std::marker::PhantomData;

use rand::Rng;

use super::Configuration;
use crate::{rng::random_index, site::Site};

/// Fully connected system of `N` exchangeable sites.
///
/// Only the number of sites in each state is stored. Each site interacts with every other site
/// with strength rescaled so that it has an effective coordination number `z`, i.e. the lattice
/// sum `Σ_⟨ij⟩` becomes `(z / N) Σ_{i<j}`.
#[derive(Debug, Clone, PartialEq)]
pub struct MeanFieldState<S: Site> {
    counts: Box<[u32]>,
    len: u32,
    coordination: f64,
    _site: PhantomData<S>,
}

impl<S: Site> MeanFieldState<S> {
    /// System of `n` sites, all in state `fill`, with effective coordination number `z`.
    ///
    /// # Panics
    /// If `n` is zero or does not fit in `u32`, or if `z` is not positive.
    pub fn uniform(n: usize, z: f64, fill: S) -> Self {
        let mut counts = vec![0; S::COUNT];
        counts[fill.index()] = u32::try_from(n).expect("Too many sites");
        Self::from_counts(counts, z)
    }

    /// System with given number of sites in each state, indexed by [`Site::index`].
    ///
    /// # Panics
    /// If `counts` has the wrong length, the total is zero or overflows, or `z` is not positive.
    pub fn from_counts(counts: Vec<u32>, z: f64) -> Self {
        assert_eq!(counts.len(), S::COUNT, "One count per site value required");
        assert!(z > 0.0, "Coordination number must be positive");
        let len = counts
            .iter()
            .try_fold(0u32, |acc, &n| acc.checked_add(n))
            .expect("Too many sites");
        assert!(len > 0, "Mean-field system must have at least one site");
        Self {
            counts: counts.into_boxed_slice(),
            len,
            coordination: z,
            _site: PhantomData,
        }
    }

    /// Effective coordination number `z`.
    #[inline(always)]
    pub fn coordination(&self) -> f64 {
        self.coordination
    }

    /// Change one site from state `from` to state `to`.
    ///
    /// # Panics
    /// If no site is in state `from`.
    #[inline(always)]
    pub fn transfer(&mut self, from: S, to: S) {
        assert!(self.counts[from.index()] > 0, "No site in state {from:?}");
        self.counts[from.index()] -= 1;
        self.counts[to.index()] += 1;
    }

    /// State of a uniformly chosen site.
    #[inline]
    pub fn random_site<R: Rng + ?Sized>(&self, rng: &mut R) -> S {
        let mut u = random_index(rng, self.len as usize) as u32;
        for (k, &n) in self.counts.iter().enumerate() {
            if u < n {
                return S::from_index(k);
            }
            u -= n;
        }
        unreachable!("Counts sum to the number of sites")
    }

    /// Replace the counts, keeping the number of sites.
    ///
    /// # Panics
    /// If the new counts do not sum to the current number of sites.
    pub fn set_counts(&mut self, counts: &[u32]) {
        assert_eq!(counts.len(), S::COUNT, "One count per site value required");
        assert_eq!(
            counts.iter().map(|&n| n as u64).sum::<u64>(),
            self.len as u64,
            "Counts must sum to the number of sites"
        );
        self.counts.copy_from_slice(counts);
    }
}

impl<S: Site> Configuration for MeanFieldState<S> {
    type Site = S;

    #[inline(always)]
    fn len(&self) -> usize {
        self.len as usize
    }

    #[inline(always)]
    fn counts(&self) -> &[u32] {
        &self.counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{rng::stream, site::SpinHalf};

    #[test]
    fn random_site_follows_counts() {
        let st = MeanFieldState::<SpinHalf>::from_counts(vec![1, 3], 4.0);
        let mut rng = stream(0, &[]);
        let n = 40_000;
        let ups = (0..n)
            .filter(|_| st.random_site(&mut rng) == SpinHalf::Up)
            .count();
        let p = ups as f64 / n as f64;
        assert!((p - 0.75).abs() < 0.01, "p = {p}");
    }

    #[test]
    fn transfer_updates_counts() {
        let mut st = MeanFieldState::uniform(10, 4.0, SpinHalf::Up);
        st.transfer(SpinHalf::Up, SpinHalf::Down);
        assert_eq!(st.counts(), &[1, 9]);
        assert_eq!(st.total_magnetization(), 8);
    }
}
