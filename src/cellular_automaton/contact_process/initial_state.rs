//! Contact Process Initial States
//!

use rand::Rng;
use rand_distr::StandardUniform;

use crate::{
    cellular_automaton::contact_process::{cell::Binary, lattice::LatticeContactProcess},
    lattice::{
        Lattice,
        initial_state::{InitialStateSpec, RandomSites, UniformSites},
    },
    systems::StateResetSpec,
};

/// All sites active
pub struct AllActive;

impl<L: Lattice<Site = Binary>> InitialStateSpec<L> for AllActive {
    fn construct(&mut self, shape: <L as Lattice>::Shape) -> L {
        L::new(shape, &mut UniformSites(Binary::Active))
    }
}

impl<L> StateResetSpec<LatticeContactProcess<L>> for AllActive
where
    L: Lattice<Site = Binary>,
{
    fn reset(&mut self, system: &mut LatticeContactProcess<L>) {
        for s in Lattice::sites_mut(&mut system.state) {
            *s = Binary::Active;
        }
    }
}

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

impl<'a, L, R> InitialStateSpec<L> for Random<'a, R>
where
    L: Lattice<Site = Binary>,
    R: Rng + ?Sized,
{
    fn construct(&mut self, shape: <L as Lattice>::Shape) -> L {
        L::new(
            shape,
            &mut RandomSites::with_dist(StandardUniform, self.rng),
        )
    }
}

impl<'a, R, L> StateResetSpec<LatticeContactProcess<L>> for Random<'a, R>
where
    L: Lattice<Site = Binary>,
    R: Rng + ?Sized,
{
    fn reset(&mut self, system: &mut LatticeContactProcess<L>) {
        for (s, s_prime) in
            Lattice::sites_mut(&mut system.state).zip(self.rng.sample_iter(StandardUniform))
        {
            *s = s_prime;
        }
    }
}
