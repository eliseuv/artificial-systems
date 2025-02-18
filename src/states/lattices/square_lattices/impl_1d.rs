//! One-dimensional square lattice
//!

use super::{Periodicity, SquareLattice1D};
use crate::states::{SimpleSwapDiffusion, SiteCharRepr, SiteState, SiteStateNN, lattices::Lattice};
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand::Rng;
use rand_distr::{Bernoulli, Distribution, Uniform};
use std::{
    fmt::Display,
    ops::{Index, IndexMut},
};

impl<T> Lattice for SquareLattice1D<T> where T: Clone + Copy {}

impl<T> Index<<Self as SiteState>::Index> for SquareLattice1D<T>
where
    T: Clone + Copy,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: <Self as SiteState>::Index) -> &Self::Output {
        &self.state[index]
    }
}

impl<T> IndexMut<<Self as SiteState>::Index> for SquareLattice1D<T>
where
    T: Clone + Copy,
{
    #[inline(always)]
    fn index_mut(&mut self, index: <Self as SiteState>::Index) -> &mut Self::Output {
        &mut self.state[index]
    }
}

impl<T> Distribution<<Self as SiteState>::Index> for SquareLattice1D<T>
where
    T: Clone + Copy,
{
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as SiteState>::Index {
        self.site_dist.sample(rng)
    }
}

impl<T> SiteState for SquareLattice1D<T>
where
    T: Clone + Copy,
{
    type Shape = usize;

    type Index = usize;

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
    fn uniform(length: Self::Shape, site: Self::Site) -> Self {
        Self {
            state: Array1::from_elem(length, site),
            period: Periodicity::new(length),
            site_dist: Uniform::new(0, length),
        }
    }

    fn random<D, R>(length: Self::Shape, dist: &D, rng: &mut R) -> Self
    where
        D: Distribution<Self::Site>,
        R: Rng + ?Sized,
    {
        Self {
            state: Array1::random_using(length, dist, rng),
            period: Periodicity::new(length),
            site_dist: Uniform::new(0, length),
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

impl<T> SiteStateNN for SquareLattice1D<T>
where
    T: Clone + Copy,
{
    fn nearest_neighbors_index_pairs(&self) -> impl Iterator<Item = (Self::Index, Self::Index)> {
        (0..self.site_count()).map(|i| (i, self.period.next(i)))
    }

    #[inline(always)]
    fn nearest_neighbors_index(&self, i: Self::Index) -> impl Iterator<Item = Self::Index> {
        [self.period.prev(i), self.period.next(i)].into_iter()
    }

    #[inline(always)]
    fn nearest_neighbors_pairs(&self) -> impl Iterator<Item = (&Self::Site, &Self::Site)> {
        self.state.indexed_iter().map(|(i, s)| {
            let i_next = self.period.next(i);
            (s, &self.state[i_next])
        })
    }

    #[inline(always)]
    fn nearest_neighbors(&self, i: Self::Index) -> impl Iterator<Item = &Self::Site> {
        let i_prev = self.period.prev(i);
        let i_next = self.period.next(i);
        [&self.state[i_prev], &self.state[i_next]].into_iter()
    }
}

/// Display 1D state
impl<T> Display for SquareLattice1D<T>
where
    T: Clone + Copy + SiteCharRepr,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let repr: String = {
            let mut char_vec: Vec<char> = self.sites().map(|s| s.char()).collect();
            char_vec.rotate_right(self.site_count() / 2);
            char_vec.into_iter().collect()
        };
        write!(f, "▕{repr}▏")
    }
}

impl<T> SimpleSwapDiffusion for SquareLattice1D<T>
where
    T: Clone + Copy,
{
    fn diffuse<R: rand::prelude::Rng + ?Sized>(&mut self, diffusion_coin: Bernoulli, rng: &mut R) {
        // Loop on random sites
        for _ in 0..self.site_count() {
            // Select random site
            let i = self.sample(rng);
            // Select random nearest neighbor
            // let nn_idx = self.nearest_neighbors_index(i).choose(rng).unwrap();
            // let nn_idx = match rng.gen::<bool>() {
            //     false => self.period.prev(i),
            //     true => self.period.next(i),
            // };
            let nn_idx = self.period.next(i);
            // Diffuse with coin flip
            if diffusion_coin.sample(rng) {
                self.state.swap(i, nn_idx)
            }
        }
    }
}
