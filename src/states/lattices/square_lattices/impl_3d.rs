//! Three-dimensional square lattice
//!

use super::{Periodicity, SquareLattice3D};
use crate::states::{SimpleSwapDiffusion, SiteState, SiteStateNN, lattices::Lattice};
use itertools::Itertools;
use ndarray::{Array3, Axis};
use ndarray_rand::RandomExt;
use rand::Rng;
use rand_distr::{Bernoulli, Distribution, Uniform};
use std::ops::{Index, IndexMut};

impl<T> Lattice for SquareLattice3D<T> where T: Clone + Copy {}

impl<T> Index<<Self as SiteState>::Index> for SquareLattice3D<T>
where
    T: Clone + Copy,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: <Self as SiteState>::Index) -> &Self::Output {
        &self.state[index]
    }
}

impl<T> IndexMut<<Self as SiteState>::Index> for SquareLattice3D<T>
where
    T: Clone + Copy,
{
    #[inline(always)]
    fn index_mut(&mut self, index: <Self as SiteState>::Index) -> &mut Self::Output {
        &mut self.state[index]
    }
}

impl<T> Distribution<<Self as SiteState>::Index> for SquareLattice3D<T>
where
    T: Clone + Copy,
{
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as SiteState>::Index {
        [
            self.site_dist.sample(rng),
            self.site_dist.sample(rng),
            self.site_dist.sample(rng),
        ]
    }
}

impl<T> SiteState for SquareLattice3D<T>
where
    T: Clone + Copy,
{
    /// Side length of the lattice
    type Shape = usize;

    type Index = [usize; 3];

    type Site = T;

    #[inline(always)]
    fn site_count(&self) -> usize {
        self.site_count()
    }

    #[inline(always)]
    fn sites(&self) -> impl Iterator<Item = &Self::Site> {
        self.sites()
    }

    #[inline(always)]
    fn sites_mut(&mut self) -> impl Iterator<Item = &mut Self::Site> {
        self.sites_mut()
    }

    #[inline(always)]
    fn uniform(side_length: Self::Shape, site: Self::Site) -> Self {
        Self {
            state: Array3::from_elem([side_length; 3], site),
            period: Periodicity::new(side_length),
            site_dist: Uniform::new(0, side_length),
        }
    }

    fn random<D, R>(side_length: Self::Shape, dist: &D, rng: &mut R) -> Self
    where
        D: Distribution<Self::Site>,
        R: rand::Rng + ?Sized,
    {
        Self {
            state: Array3::random_using([side_length; 3], dist, rng),
            period: Periodicity::new(side_length),
            site_dist: Uniform::new(0, side_length),
        }
    }

    fn set_uniform(&mut self, site: Self::Site) {
        self.state.fill(site);
    }

    fn set_random<D: Distribution<Self::Site>, R: rand::Rng + ?Sized>(
        &mut self,
        dist: &D,
        rng: &mut R,
    ) {
        for (s, x) in self.sites_mut().zip((&dist).sample_iter(rng)) {
            *s = x;
        }
    }
}

impl<T> SiteStateNN for SquareLattice3D<T>
where
    T: Clone + Copy,
{
    fn nearest_neighbors_index_pairs(&self) -> impl Iterator<Item = (Self::Index, Self::Index)> {
        let side_length = self.state.len_of(Axis(0));
        (0..side_length)
            .cartesian_product(0..side_length)
            .cartesian_product(0..side_length)
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
    fn nearest_neighbors_index(&self, [i, j, k]: Self::Index) -> impl Iterator<Item = Self::Index> {
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
    fn nearest_neighbors_pairs(&self) -> impl Iterator<Item = (&Self::Site, &Self::Site)> {
        self.state.indexed_iter().flat_map(|((i, j, k), s)| {
            [
                (s, &self.state[[i, j, self.period.next(k)]]),
                (s, &self.state[[i, self.period.next(j), k]]),
                (s, &self.state[[self.period.next(i), j, k]]),
            ]
        })
    }

    #[inline(always)]
    fn nearest_neighbors(&self, [i, j, k]: Self::Index) -> impl Iterator<Item = &Self::Site> {
        [
            &self.state[[i, j, self.period.prev(k)]],
            &self.state[[i, self.period.prev(j), k]],
            &self.state[[self.period.prev(i), j, k]],
            &self.state[[i, j, self.period.next(k)]],
            &self.state[[i, self.period.next(j), k]],
            &self.state[[self.period.next(i), j, k]],
        ]
        .into_iter()
    }
}

impl<T> SimpleSwapDiffusion for SquareLattice3D<T>
where
    T: Clone + Copy,
{
    fn diffuse<R: rand::prelude::Rng + ?Sized>(&mut self, diffusion_coin: Bernoulli, rng: &mut R) {
        let direction_dist = Uniform::new(0, 3);
        // Loop on random sites
        for _ in 0..self.site_count() {
            let idx @ [i, j, k] = self.sample(rng);
            // Select random nearest neighbor
            let nn_idx = match direction_dist.sample(&mut *rng) {
                0 => [i, j, self.period.next(k)],
                1 => [i, self.period.next(j), k],
                _ => [self.period.next(i), j, k],
            };
            // Diffuse with coin flip
            if diffusion_coin.sample(rng) {
                self.state.swap(idx, nn_idx)
            }
        }
    }
}
