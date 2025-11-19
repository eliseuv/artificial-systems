//! Diffusion on the Lattice
//!

use crate::lattice::Lattice;
use rand::{Rng, seq::IteratorRandom};
use rand_distr::{Bernoulli, BernoulliError, Distribution};

/// Arbitrary diffusion on the lattice
pub trait Diffusion<L>
where
    L: Lattice,
{
    /// Apply the diffusion the lattice
    fn apply<R: Rng + ?Sized>(&self, lattice: &mut L, rng: &mut R);
}

pub struct SimpleSwapDiffusion {
    coin: Bernoulli,
}

impl SimpleSwapDiffusion {
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
                let s_i = lattice[i];
                lattice[i] = lattice[j];
                lattice[j] = s_i;
            }
        }
    }
}
