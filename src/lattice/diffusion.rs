//! Diffusion on the Lattice
//!

use crate::lattice::Lattice;
use rand::{
    Rng,
    seq::{IteratorRandom, SliceRandom},
};
use rand_distr::{Bernoulli, BernoulliError, Distribution};

/// Arbitrary diffusion on the lattice
pub trait Diffusion<L>
where
    L: Lattice,
{
    /// Apply the diffusion on the lattice
    fn apply<R: Rng + ?Sized>(&self, lattice: &mut L, rng: &mut R);
}

/// Simple Swap Diffusion
pub struct SimpleSwapDiffusion {
    coin: Bernoulli,
}

impl SimpleSwapDiffusion {
    /// Simple swap diffusion with a given diffusion coefficient
    pub fn with_gamma(gamma: f64) -> Result<Self, BernoulliError> {
        let coin = Bernoulli::new(gamma)?;
        Ok(Self { coin })
    }
}

impl<L> Diffusion<L> for SimpleSwapDiffusion
where
    L: Lattice,
    L::Site: Copy,
{
    fn apply<R: Rng + ?Sized>(&self, lattice: &mut L, rng: &mut R) {
        for _ in 0..lattice.site_count() {
            // Test diffusion probability
            if self.coin.sample(rng) {
                // Select random pair
                let (i, j) = lattice
                    .nearest_neighbors_pairs_indices()
                    .choose(rng)
                    .expect("No nearest neighbors pair available");
                // Perform swap
                lattice.swap(i, j);
            }
        }
    }
}

/// Simple Swap Diffusion
pub struct SimpleSwapDiffusion2 {
    coin: Bernoulli,
}

impl SimpleSwapDiffusion2 {
    /// Simple swap diffusion with a given diffusion coefficient
    pub fn with_gamma(gamma: f64) -> Result<Self, BernoulliError> {
        let coin = Bernoulli::new(gamma)?;
        Ok(Self { coin })
    }
}

impl<L> Diffusion<L> for SimpleSwapDiffusion2
where
    L: Lattice,
    L::Site: Copy,
{
    fn apply<R: Rng + ?Sized>(&self, lattice: &mut L, rng: &mut R) {
        let mut indices: Vec<_> = lattice.indices().collect();
        indices.shuffle(rng);
        let coins: Vec<_> = self
            .coin
            .sample_iter(&mut *rng)
            .take(indices.len())
            .collect();
        for (i, do_swap) in indices.iter().zip(coins) {
            // Test diffusion probability
            if do_swap {
                // Select random nearest neighbor
                let j = lattice
                    .nearest_neighbors_indices(*i)
                    .choose(rng)
                    .expect("No nearest neighbor available");
                // Perform swap
                lattice.swap(*i, j);
            }
        }
    }
}
