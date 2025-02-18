//! Two-dimensional square lattice
//!

use super::{Periodicity, SquareLattice2D};
use crate::states::{SimpleSwapDiffusion, SiteCharRepr, SiteState, SiteStateNN, lattices::Lattice};
use itertools::Itertools;
use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use rand::Rng;
use rand_distr::{Bernoulli, Distribution, Uniform};
use std::{
    fmt::Display,
    ops::{Index, IndexMut},
};

impl<T> Lattice for SquareLattice2D<T> where T: Clone + Copy {}

impl<T> Index<<Self as SiteState>::Index> for SquareLattice2D<T>
where
    T: Clone + Copy,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: <Self as SiteState>::Index) -> &Self::Output {
        &self.state[index]
    }
}

impl<T> IndexMut<<Self as SiteState>::Index> for SquareLattice2D<T>
where
    T: Clone + Copy,
{
    #[inline(always)]
    fn index_mut(&mut self, index: <Self as SiteState>::Index) -> &mut Self::Output {
        &mut self.state[index]
    }
}

impl<T> Distribution<<Self as SiteState>::Index> for SquareLattice2D<T>
where
    T: Clone + Copy,
{
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self as SiteState>::Index {
        [self.site_dist.sample(rng), self.site_dist.sample(rng)]
    }
}

impl<T> SiteState for SquareLattice2D<T>
where
    T: Clone + Copy,
{
    /// Side length of the lattice
    type Shape = usize;

    type Index = [usize; 2];

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
            state: Array2::from_elem([side_length; 2], site),
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
            state: Array2::random_using([side_length; 2], dist, rng),
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

impl<T> SiteStateNN for SquareLattice2D<T>
where
    T: Clone + Copy,
{
    fn nearest_neighbors_index_pairs(&self) -> impl Iterator<Item = (Self::Index, Self::Index)> {
        let side_length = self.state.len_of(Axis(0));
        (0..side_length)
            .cartesian_product(0..side_length)
            .flat_map(|idx @ (i, j)| {
                [
                    (idx.into(), [i, self.period.next(j)]),
                    (idx.into(), [self.period.next(i), j]),
                ]
            })
    }

    #[inline(always)]
    fn nearest_neighbors_index(&self, [i, j]: Self::Index) -> impl Iterator<Item = Self::Index> {
        [
            [i, self.period.prev(j)],
            [i, self.period.next(j)],
            [self.period.prev(i), j],
            [self.period.next(i), j],
        ]
        .into_iter()
    }

    #[inline(always)]
    fn nearest_neighbors_pairs(&self) -> impl Iterator<Item = (&Self::Site, &Self::Site)> {
        self.state.indexed_iter().flat_map(|((i, j), s)| {
            [
                (s, &self.state[[i, self.period.next(j)]]),
                (s, &self.state[[self.period.next(i), j]]),
            ]
        })
    }

    #[inline(always)]
    fn nearest_neighbors(&self, [i, j]: Self::Index) -> impl Iterator<Item = &Self::Site> {
        [
            &self.state[[i, self.period.prev(j)]],
            &self.state[[self.period.prev(i), j]],
            &self.state[[i, self.period.next(j)]],
            &self.state[[self.period.next(i), j]],
        ]
        .into_iter()
    }
}

/// Display Contact Process 2D state
impl<T> Display for SquareLattice2D<T>
where
    T: Clone + Copy + SiteCharRepr,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (nrows, ncols) = self.state.dim();
        writeln!(f, "{}", "_".repeat(nrows + 2))?;
        for i in 0..nrows {
            let i_off = (i + (nrows / 2)) % nrows;
            write!(f, "|")?;
            for j in 0..ncols {
                let j_off = (j + (ncols / 2)) % ncols;
                let c = self.state[[i_off, j_off]].char();
                write!(f, "{c}")?;
            }
            writeln!(f, "|")?;
        }
        write!(f, "{}", "‾".repeat(nrows + 2))?;
        Ok(())
    }
}

// impl<T> Display for SquareLattice2D<T>
// where
//     T: Clone + Copy + CharacterRepresentation,
// {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let (nrows, _ncols) = self.state.dim();
//         writeln!(f, "{}", "_".repeat(nrows + 2))?;
//         let repr = self
//             .state
//             .rows()
//             .into_iter()
//             .map(|x| {
//                 ['|']
//                     .into_iter()
//                     .chain(x.into_iter().map(|s| s.char_repr()))
//                     .chain(['|'])
//                     .collect::<String>()
//             })
//             .join("\n");
//         writeln!(f, "{repr}")?;
//         write!(f, "{}", "‾".repeat(nrows))?;
//         Ok(())
//     }
// }

impl<T> SimpleSwapDiffusion for SquareLattice2D<T>
where
    T: Clone + Copy,
{
    fn diffuse<R: rand::prelude::Rng + ?Sized>(&mut self, diffusion_coin: Bernoulli, rng: &mut R) {
        // Loop on random sites
        for _ in 0..self.site_count() {
            // Select random site
            let idx @ [i, j] = self.sample(rng);
            // Select random nearest neighbor
            let nn_idx = match rng.r#gen() {
                true => [i, self.period.next(j)],
                false => [self.period.next(i), j],
            };
            // Diffuse with coin flip
            if diffusion_coin.sample(rng) {
                self.state.swap(idx, nn_idx)
            }
        }
    }
}
