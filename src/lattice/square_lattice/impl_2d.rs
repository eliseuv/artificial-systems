//! 2D Square Lattice
//!

use std::{
    mem::MaybeUninit,
    ops::{Index, IndexMut},
};

use itertools::Itertools;
use ndarray::{Array2, Axis};
use rand::Rng;
use rand_distr::Distribution;

use crate::lattice::{
    Lattice,
    square_lattice::{SquareLattice, periodicity::Periodicity},
};

/// 2D Square Lattice
#[derive(Debug, Clone)]
pub struct SquareLattice2D<T> {
    state: Array2<T>,
    period: Periodicity,
}

impl<T> SquareLattice2D<T> {
    #[inline(always)]
    pub fn rows(&self) -> ndarray::iter::Lanes<'_, T, ndarray::Dim<[usize; 1]>> {
        self.state.rows()
    }
}

impl<T> Index<[usize; 2]> for SquareLattice2D<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.state[index]
    }
}

impl<T> IndexMut<[usize; 2]> for SquareLattice2D<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.state[index]
    }
}

impl<T> Distribution<[usize; 2]> for SquareLattice2D<T> {
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [usize; 2] {
        [
            rng.random_range(0..self.length()),
            rng.random_range(0..self.length()),
        ]
    }
}

impl<T> Lattice for SquareLattice2D<T> {
    type Shape = usize;

    type Index = [usize; 2];

    type Site = T;

    #[inline(always)]
    fn site_count(&self) -> usize {
        self.state.len()
    }

    #[inline(always)]
    fn indices(&self) -> impl Iterator<Item = Self::Index> {
        (0..self.site_count()).map(|n| {
            let i = n % self.length();
            let j = n - (i * self.length());
            [i, j]
        })
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
        self.state.indexed_iter().map(|(i, s)| (i.into(), s))
    }

    #[inline(always)]
    fn indexed_sites_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (Self::Index, &'a mut Self::Site)>
    where
        Self::Site: 'a,
    {
        self.state.indexed_iter_mut().map(|(i, s)| (i.into(), s))
    }

    #[inline(always)]
    fn nearest_neighbors_indices(&self, [i, j]: Self::Index) -> impl Iterator<Item = Self::Index> {
        [
            [i, self.period.prev(j)],
            [i, self.period.next(j)],
            [self.period.prev(i), j],
            [self.period.next(i), j],
        ]
        .into_iter()
    }

    #[inline(always)]
    fn nearest_neighbors(&self, idx: Self::Index) -> impl Iterator<Item = &Self::Site> {
        self.nearest_neighbors_indices(idx)
            .map(|idx_nn| &self.state[idx_nn])
    }

    #[inline(always)]
    fn nearest_neighbors_pairs_indices(&self) -> impl Iterator<Item = (Self::Index, Self::Index)> {
        let length = self.length();
        (0..length)
            .cartesian_product(0..length)
            .flat_map(|idx @ (i, j)| {
                [
                    (idx.into(), [i, self.period.next(j)]),
                    (idx.into(), [self.period.next(i), j]),
                ]
            })
    }

    #[inline(always)]
    fn nearest_neighbors_pairs(&self) -> impl Iterator<Item = (&Self::Site, &Self::Site)> {
        self.state.indexed_iter().flat_map(|((i, j), s)| {
            [
                (s, &self[[i, self.period.next(j)]]),
                (s, &self[[self.period.next(i), j]]),
            ]
        })
    }

    #[inline(always)]
    fn new_uninit(length: Self::Shape) -> MaybeUninit<Self>
    where
        Self: Sized,
    {
        MaybeUninit::new(Self {
            state: unsafe { Array2::uninit([length; 2]).assume_init() },
            period: Periodicity::new(length),
        })
    }
}

impl<T> SquareLattice<2> for SquareLattice2D<T> {
    #[inline(always)]
    fn length(&self) -> usize {
        self.state.len_of(Axis(0))
    }
}
