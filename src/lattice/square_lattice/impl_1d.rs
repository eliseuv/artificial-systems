//! 1D Square Lattice
//!
//!
use std::{
    mem::MaybeUninit,
    ops::{Index, IndexMut},
};

use rand::Rng;
use rand_distr::Distribution;

use crate::lattice::{
    Lattice,
    square_lattice::{SquareLattice, periodicity::Periodicity},
};

/// 1D Square Lattice
#[derive(Debug, Clone)]
pub struct SquareLattice1D<T> {
    state: Vec<T>,
    period: Periodicity,
}

impl<T> Index<usize> for SquareLattice1D<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.state[index]
    }
}

impl<T> IndexMut<usize> for SquareLattice1D<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.state[index]
    }
}

impl<T> Distribution<<SquareLattice1D<T> as Lattice>::Index> for SquareLattice1D<T> {
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <SquareLattice1D<T> as Lattice>::Index {
        rng.random_range(0..self.state.len())
    }
}

impl<T> Lattice for SquareLattice1D<T> {
    type Shape = usize;

    type Index = usize;

    type Site = T;

    #[inline(always)]
    fn site_count(&self) -> usize {
        self.state.len()
    }

    fn indices(&self) -> impl Iterator<Item = Self::Index> {
        0..self.state.len()
    }

    #[inline(always)]
    fn sites<'a>(&'a self) -> impl Iterator<Item = &'a Self::Site>
    where
        Self::Site: 'a,
    {
        self.state.iter()
    }

    #[inline(always)]
    fn sites_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Self::Site>
    where
        Self::Site: 'a,
    {
        self.state.iter_mut()
    }

    #[inline(always)]
    fn indexed_sites<'a>(&'a self) -> impl Iterator<Item = (Self::Index, &'a Self::Site)>
    where
        Self::Site: 'a,
    {
        self.state.iter().enumerate()
    }

    #[inline(always)]
    fn indexed_sites_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (Self::Index, &'a mut Self::Site)>
    where
        Self::Site: 'a,
    {
        self.state.iter_mut().enumerate()
    }

    #[inline(always)]
    fn nearest_neighbors_indices(&self, i: Self::Index) -> impl Iterator<Item = Self::Index> {
        [self.period.prev(i), self.period.next(i)].into_iter()
    }

    #[inline(always)]
    fn nearest_neighbors(&self, i: Self::Index) -> impl Iterator<Item = &Self::Site> {
        [self.period.prev(i), self.period.next(i)]
            .map(|i| &self.state[i])
            .into_iter()
    }

    #[inline(always)]
    fn nearest_neighbors_pairs_indices(&self) -> impl Iterator<Item = (Self::Index, Self::Index)> {
        (0..self.site_count()).map(|i| (i, self.period.next(i)))
    }

    #[inline(always)]
    fn nearest_neighbors_pairs(&self) -> impl Iterator<Item = (&Self::Site, &Self::Site)> {
        (0..self.site_count()).map(|i| (&self.state[i], &self.state[self.period.next(i)]))
    }

    #[allow(clippy::uninit_vec)]
    fn new_uninit(length: Self::Shape) -> MaybeUninit<Self>
    where
        Self: Sized,
    {
        let mut state = Vec::with_capacity(length);
        unsafe { state.set_len(length) };

        MaybeUninit::new(Self {
            state,
            period: Periodicity::new(length),
        })
    }
}

impl<T> SquareLattice<1> for SquareLattice1D<T> {
    #[inline(always)]
    fn length(&self) -> usize {
        self.state.len()
    }
}
