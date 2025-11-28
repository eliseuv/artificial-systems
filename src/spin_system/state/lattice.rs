use std::{
    fmt::Display,
    ops::{Index, IndexMut, Neg},
};

use rand::Rng;
use rand_distr::{Distribution, StandardUniform};

use crate::{
    lattice::{
        Lattice,
        initial_state::{InitialStateSpec, RandomSites, UniformSites},
    },
    spin_system::{
        spin::Spin,
        state::{Ferromagnetic, Paramagnetic, SpinState},
    },
};

/// Lattice Spin State Wrapper
pub struct LatticeSpinState<L: Lattice>(pub L);

impl<L: Lattice> Index<L::Index> for LatticeSpinState<L> {
    type Output = L::Site;

    fn index(&self, i: L::Index) -> &Self::Output {
        &self.0[i]
    }
}

impl<L: Lattice> IndexMut<L::Index> for LatticeSpinState<L> {
    fn index_mut(&mut self, i: L::Index) -> &mut Self::Output {
        &mut self.0[i]
    }
}

impl<L> SpinState for LatticeSpinState<L>
where
    L: Lattice,
    L::Site: Spin,
{
    type Spin = L::Site;

    type Index = L::Index;

    #[inline(always)]
    fn spin_count(&self) -> usize {
        self.0.site_count()
    }

    #[inline(always)]
    fn indices(&self) -> impl Iterator<Item = Self::Index> {
        self.0.indices()
    }

    #[inline(always)]
    fn spins<'a>(&'a self) -> impl Iterator<Item = &'a Self::Spin>
    where
        Self::Spin: 'a,
    {
        self.0.sites()
    }

    #[inline(always)]
    fn spins_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Self::Spin>
    where
        Self::Spin: 'a,
    {
        self.0.sites_mut()
    }

    #[inline(always)]
    fn indexed_spins<'a>(&'a self) -> impl Iterator<Item = (Self::Index, &'a Self::Spin)>
    where
        Self::Spin: 'a,
    {
        self.0.indexed_sites()
    }

    #[inline(always)]
    fn indexed_spins_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (Self::Index, &'a mut Self::Spin)>
    where
        Self::Spin: 'a,
    {
        self.0.indexed_sites_mut()
    }

    #[inline(always)]
    fn total_magnet(&self) -> i32 {
        self.0.sites().map(|&s| s.into()).sum::<i32>()
    }

    #[inline(always)]
    fn nn_sum(&self, i: Self::Index) -> i32 {
        self.0.nearest_neighbors(i).map(|&s| s.into()).sum()
    }

    #[inline(always)]
    fn interaction(&self, i: Self::Index) -> i32 {
        let s_i: i32 = self[i].into();

        (s_i * self.nn_sum(i)).neg()
    }

    #[inline(always)]
    fn total_interaction(&self) -> i32 {
        self.0
            .nearest_neighbors_pairs()
            .map(|(&s_i, &s_j)| (s_i * s_j).into())
            .sum::<i32>()
            .neg()
    }
}

impl<L, T> InitialStateSpec<L> for Ferromagnetic<T>
where
    T: Spin,
    L: Lattice<Site = T>,
{
    fn construct(&mut self, shape: <L as Lattice>::Shape) -> L {
        UniformSites(self.0).construct(shape)
    }
}

impl<'a, L, T, R> InitialStateSpec<L> for Paramagnetic<'a, T, R>
where
    T: Spin,
    L: Lattice<Site = T>,
    R: Rng + ?Sized,
    StandardUniform: Distribution<T>,
{
    fn construct(&mut self, shape: <L as Lattice>::Shape) -> L {
        RandomSites::with_dist(StandardUniform, self.rng).construct(shape)
    }
}

impl<L> Display for LatticeSpinState<L>
where
    L: Lattice + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
