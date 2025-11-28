//! 3D Square Lattice
//!

use std::{
    mem::MaybeUninit,
    ops::{Index, IndexMut},
};

use itertools::Itertools;
use ndarray::{Array3, Axis};
use rand::Rng;
use rand_distr::Distribution;

use crate::{
    hypercube_index,
    lattice::{
        Lattice,
        square_lattice::{SquareLattice, periodicity::Periodicity},
    },
};

/// 2D Square Lattice
#[derive(Debug, Clone)]
pub struct SquareLattice3D<T> {
    state: Array3<T>,
    period: Periodicity,
}

impl<T> Index<[usize; 3]> for SquareLattice3D<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: [usize; 3]) -> &Self::Output {
        &self.state[index]
    }
}

impl<T> IndexMut<[usize; 3]> for SquareLattice3D<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: [usize; 3]) -> &mut Self::Output {
        &mut self.state[index]
    }
}

impl<T> Distribution<[usize; 3]> for SquareLattice3D<T> {
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [usize; 3] {
        [
            rng.random_range(0..self.length()),
            rng.random_range(0..self.length()),
            rng.random_range(0..self.length()),
        ]
    }
}

impl<T> Lattice for SquareLattice3D<T> {
    type Shape = usize;

    type Index = [usize; 3];

    type Site = T;

    #[inline(always)]
    fn site_count(&self) -> usize {
        self.state.len()
    }

    #[inline(always)]
    fn indices(&self) -> impl Iterator<Item = Self::Index> {
        (0..self.site_count()).map(|n| hypercube_index!(n, self.length(); x, y, z))
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
    fn indexed_sites<'a>(&self) -> impl Iterator<Item = (Self::Index, &Self::Site)>
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
    fn swap(&mut self, i: Self::Index, j: Self::Index) {
        self.state.swap(i, j);
    }

    #[inline(always)]
    fn nearest_neighbors_indices(
        &self,
        [i, j, k]: Self::Index,
    ) -> impl Iterator<Item = Self::Index> {
        [
            [i, j, self.period.prev(k)],
            [i, j, self.period.next(k)],
            [i, self.period.prev(j), k],
            [i, self.period.next(j), k],
            [self.period.prev(i), j, k],
            [self.period.next(i), j, k],
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
            .cartesian_product(0..length)
            .flat_map(|((i, j), k)| {
                let idx = [i, j, k];
                [
                    (idx, [i, j, self.period.next(k)]),
                    (idx, [i, self.period.next(j), k]),
                    (idx, [self.period.next(i), j, k]),
                ]
            })
    }

    #[inline(always)]
    fn nearest_neighbors_pairs(&self) -> impl Iterator<Item = (&Self::Site, &Self::Site)> {
        self.state.indexed_iter().flat_map(|((i, j, k), s)| {
            [
                (s, &self[[i, j, self.period.next(k)]]),
                (s, &self[[i, self.period.next(j), k]]),
                (s, &self[[self.period.next(i), j, k]]),
            ]
        })
    }

    #[inline(always)]
    fn new_uninit(length: Self::Shape) -> MaybeUninit<Self>
    where
        Self: Sized,
    {
        MaybeUninit::new(Self {
            state: unsafe { Array3::uninit([length; 3]).assume_init() },
            period: Periodicity::new(length),
        })
    }
}

impl<T> SquareLattice<3> for SquareLattice3D<T> {
    #[inline(always)]
    fn length(&self) -> usize {
        self.state.len_of(Axis(0))
    }
}
