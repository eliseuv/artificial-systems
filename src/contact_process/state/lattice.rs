//! Lattice Contact Process State
//!

use std::{
    fmt::Display,
    mem::MaybeUninit,
    ops::{Index, IndexMut},
};

use rand::Rng;
use rand_distr::StandardUniform;

use crate::{
    contact_process::{
        ContactProcessSystem,
        cell::Binary,
        state::{AllActive, ContactProcessState, Random},
    },
    lattice::{
        Lattice,
        initial_state::{InitialStateSpec, RandomSites, UniformSites},
    },
    method::SystemResetSpec,
};

/// Lattice Contact Process State Wrapper
pub struct LatticeContactProcessState<L>(pub(crate) L)
where
    L: Lattice<Site = Binary>;

impl<L> Index<L::Index> for LatticeContactProcessState<L>
where
    L: Lattice<Site = Binary>,
{
    type Output = L::Site;

    #[inline(always)]
    fn index(&self, i: L::Index) -> &Self::Output {
        &self.0[i]
    }
}

impl<L> IndexMut<L::Index> for LatticeContactProcessState<L>
where
    L: Lattice<Site = Binary>,
{
    #[inline(always)]
    fn index_mut(&mut self, i: L::Index) -> &mut Self::Output {
        &mut self.0[i]
    }
}

impl<L> Lattice for LatticeContactProcessState<L>
where
    L: Lattice<Site = Binary>,
{
    type Shape = L::Shape;

    type Index = L::Index;

    type Site = L::Site;

    #[inline(always)]
    fn site_count(&self) -> usize {
        self.0.site_count()
    }

    #[inline(always)]
    fn indices(&self) -> impl Iterator<Item = Self::Index> {
        self.0.indices()
    }

    #[inline(always)]
    fn sites<'a>(&'a self) -> impl Iterator<Item = &'a Self::Site>
    where
        Self::Site: 'a,
    {
        self.0.sites()
    }

    #[inline(always)]
    fn sites_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Self::Site>
    where
        Self::Site: 'a,
    {
        self.0.sites_mut()
    }

    #[inline(always)]
    fn indexed_sites<'a>(&'a self) -> impl Iterator<Item = (Self::Index, &'a Self::Site)>
    where
        Self::Site: 'a,
    {
        self.0.indexed_sites()
    }

    #[inline(always)]
    fn indexed_sites_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (Self::Index, &'a mut Self::Site)>
    where
        Self::Site: 'a,
    {
        self.0.indexed_sites_mut()
    }

    #[inline(always)]
    fn swap(&mut self, i: Self::Index, j: Self::Index) {
        self.0.swap(i, j);
    }

    #[inline(always)]
    fn nearest_neighbors_indices(&self, idx: Self::Index) -> impl Iterator<Item = Self::Index> {
        self.0.nearest_neighbors_indices(idx)
    }

    #[inline(always)]
    fn nearest_neighbors(&self, idx: Self::Index) -> impl Iterator<Item = &Self::Site> {
        self.0.nearest_neighbors(idx)
    }

    #[inline(always)]
    fn nearest_neighbors_pairs_indices(&self) -> impl Iterator<Item = (Self::Index, Self::Index)> {
        self.0.nearest_neighbors_pairs_indices()
    }

    #[inline(always)]
    fn nearest_neighbors_pairs(&self) -> impl Iterator<Item = (&Self::Site, &Self::Site)> {
        self.0.nearest_neighbors_pairs()
    }

    #[inline(always)]
    fn new_uninit(shape: Self::Shape) -> MaybeUninit<Self>
    where
        Self: Sized,
    {
        MaybeUninit::new(LatticeContactProcessState(unsafe {
            L::new_uninit(shape).assume_init()
        }))
    }
}

impl<L> ContactProcessState for LatticeContactProcessState<L>
where
    L: Lattice<Site = Binary>,
{
    #[inline(always)]
    fn total_active(&self) -> usize {
        self.0.sites().map(|s| *s as usize).sum()
    }

    #[inline(always)]
    fn active(&self) -> f64 {
        self.total_active() as f64 / self.0.site_count() as f64
    }
}

impl<L> InitialStateSpec<LatticeContactProcessState<L>> for AllActive
where
    L: Lattice<Site = Binary>,
{
    #[inline(always)]
    fn construct(&mut self, shape: <L as Lattice>::Shape) -> LatticeContactProcessState<L> {
        LatticeContactProcessState(L::new(shape, &mut UniformSites(Binary::Active)))
    }
}

impl<L, S> SystemResetSpec<S> for AllActive
where
    S: ContactProcessSystem<State = LatticeContactProcessState<L>>,
    L: Lattice<Site = Binary>,
{
    #[inline(always)]
    fn reset(&mut self, system: &mut S) {
        for s in system.state_mut().sites_mut() {
            *s = Binary::Active;
        }
    }
}

impl<'a, L, R> InitialStateSpec<LatticeContactProcessState<L>> for Random<'a, R>
where
    L: Lattice<Site = Binary>,
    R: Rng + ?Sized,
{
    #[inline(always)]
    fn construct(&mut self, shape: <L as Lattice>::Shape) -> LatticeContactProcessState<L> {
        LatticeContactProcessState(L::new(
            shape,
            &mut RandomSites::with_dist(StandardUniform, self.rng),
        ))
    }
}

impl<'a, R, L, S> SystemResetSpec<S> for Random<'a, R>
where
    S: ContactProcessSystem<State = LatticeContactProcessState<L>>,
    L: Lattice<Site = Binary>,
    R: Rng + ?Sized,
{
    #[inline(always)]
    fn reset(&mut self, system: &mut S) {
        for (s, s_prime) in system
            .state_mut()
            .sites_mut()
            .zip(self.rng.sample_iter(StandardUniform))
        {
            *s = s_prime;
        }
    }
}

impl<L> Display for LatticeContactProcessState<L>
where
    L: Lattice<Site = Binary> + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
