use std::ops::Neg;

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

/// Lattice Spin State
impl<L> SpinState for L
where
    L: Lattice,
    L::Site: Spin,
{
    type Spin = L::Site;

    type Index = L::Index;

    #[inline(always)]
    fn spin_count(&self) -> usize {
        self.site_count()
    }

    #[inline(always)]
    fn indices(&self) -> impl Iterator<Item = Self::Index> {
        self.indices()
    }

    #[inline(always)]
    fn indexed_spins<'a>(&'a self) -> impl Iterator<Item = (Self::Index, &'a Self::Spin)>
    where
        Self::Spin: 'a,
    {
        self.indexed_sites()
    }

    #[inline(always)]
    fn indexed_spins_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (Self::Index, &'a mut Self::Spin)>
    where
        Self::Spin: 'a,
    {
        self.indexed_sites_mut()
    }

    #[inline(always)]
    fn total_magnet(&self) -> i32 {
        self.sites().map(|&s| s.into()).sum::<i32>()
    }

    #[inline(always)]
    fn nn_sum(&self, i: Self::Index) -> i32 {
        self.nearest_neighbors(i).map(|&s| s.into()).sum()
    }

    #[inline(always)]
    fn interaction(&self, i: Self::Index) -> i32 {
        let s_i: i32 = self[i].into();

        (s_i * self.nn_sum(i)).neg()
    }

    #[inline(always)]
    fn total_interaction(&self) -> i32 {
        self.nearest_neighbors_pairs()
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
    fn reset(&mut self, lattice: &mut L) {
        UniformSites(self.0).reset(lattice);
    }
}

impl<'a, L, T, R> InitialStateSpec<L> for Paramagnetic<'a, T, R>
where
    T: Spin,
    L: Lattice<Site = T>,
    R: Rng + ?Sized,
    StandardUniform: Distribution<T>,
{
    fn reset(&mut self, lattice: &mut L) {
        RandomSites::with_dist(StandardUniform, self.rng).reset(lattice);
    }
}
