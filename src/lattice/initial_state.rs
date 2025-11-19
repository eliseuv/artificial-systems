//! Initial lattice state specification
//!

use std::marker::PhantomData;

use rand::Rng;
use rand_distr::Distribution;

use crate::lattice::Lattice;

/// Specification for an initial state of a lattice
pub trait InitialStateSpec<L>
where
    L: Lattice,
{
    /// Construct a new lattice according to specification
    fn construct(&mut self, shape: L::Shape) -> L {
        let mut lattice = unsafe { L::new_uninit(shape).assume_init() };

        self.reset(&mut lattice);
        lattice
    }

    /// Reset state of the lattice according to specification
    fn reset(&mut self, lattice: &mut L);
}

/// Uniform initial state specification
#[derive(Debug, Clone, Copy)]
pub struct UniformSites<T: Copy>(pub T);

impl<L, T> InitialStateSpec<L> for UniformSites<T>
where
    T: Copy,
    L: Lattice<Site = T>,
{
    fn reset(&mut self, lattice: &mut L) {
        for s in lattice.sites_mut() {
            *s = self.0;
        }
    }
}

/// Random initial state specification
#[derive(Debug)]
pub struct RandomSites<'a, T, D, R>
where
    D: Distribution<T>,
    R: Rng + ?Sized,
{
    _site: PhantomData<T>,
    pub(crate) dist: D,
    pub(crate) rng: &'a mut R,
}

impl<'a, T, D, R> RandomSites<'a, T, D, R>
where
    D: Distribution<T>,
    R: Rng + ?Sized,
{
    pub fn with_dist(dist: D, rng: &'a mut R) -> Self {
        Self {
            _site: PhantomData,
            dist,
            rng,
        }
    }
}

impl<'a, L, T, D, R> InitialStateSpec<L> for RandomSites<'a, T, D, R>
where
    L: Lattice<Site = T>,
    D: Distribution<T>,
    R: Rng + ?Sized,
{
    fn reset(&mut self, lattice: &mut L) {
        for (s, s_prime) in lattice
            .sites_mut()
            .zip((&mut self.rng).sample_iter(&self.dist))
        {
            *s = s_prime;
        }
    }
}
