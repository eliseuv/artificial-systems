//! Spin States
//!

use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
};

use rand::Rng;
use rand_distr::{Distribution, StandardUniform};

use crate::{spin_system::spin::Spin, systems::StateResetSpec};

/// Spin States
pub trait SpinState:
    Index<Self::Index, Output = Self::Spin> + IndexMut<Self::Index, Output = Self::Spin>
{
    /// Single Spin State
    type Spin: Spin;

    /// Index for an individual spin
    type Index: Copy;

    /// Total number of spins
    fn spin_count(&self) -> usize;

    /// Iterator over all indices of the spins
    fn indices(&self) -> impl Iterator<Item = Self::Index>;

    /// Iterator over all spins
    fn spins<'a>(&'a self) -> impl Iterator<Item = &'a Self::Spin>
    where
        Self::Spin: 'a;

    /// Mutable iterator over all spins
    fn spins_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Self::Spin>
    where
        Self::Spin: 'a;

    /// Iterator over indices and spins pairs
    fn indexed_spins<'a>(&'a self) -> impl Iterator<Item = (Self::Index, &'a Self::Spin)>
    where
        Self::Spin: 'a;

    /// Iterator over indices and spins pairs
    fn indexed_spins_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (Self::Index, &'a mut Self::Spin)>
    where
        Self::Spin: 'a;

    /// Total magnetization
    #[inline(always)]
    fn total_magnet(&self) -> i32 {
        self.spins().map(|&s| s.into()).sum()
    }

    /// Magnetization
    #[inline(always)]
    fn magnet(&self) -> f64 {
        self.total_magnet() as f64 / self.spin_count() as f64
    }

    /// Sum of the nearest neighbors
    fn nn_sum(&self, i: Self::Index) -> i32;

    /// Interaction energy local to a single spin
    /// h_i = - s_i  \sum_<i,j> s_j
    fn interaction(&self, i: Self::Index) -> i32;

    /// Total interaction energy
    /// h = - \sum_<i,j> s_i s_j
    fn total_interaction(&self) -> i32;
}

/// Ferromagnetic (ordered) spin configuration
#[derive(Debug, Clone, Copy)]
pub struct Ferromagnetic<T: Spin>(pub T);

impl<S, T> StateResetSpec<S> for Ferromagnetic<T>
where
    T: Spin,
    S: SpinState<Spin = T>,
{
    fn reset(&mut self, system: &mut S) {
        for s in system.spins_mut() {
            *s = self.0
        }
    }
}

/// Paramagentic (disordered) spin configuration
pub struct Paramagnetic<'a, T, R>
where
    T: Spin,
    R: Rng + ?Sized,
{
    _spin: PhantomData<T>,
    rng: &'a mut R,
}

impl<'a, T, R> Paramagnetic<'a, T, R>
where
    T: Spin,
    R: Rng + ?Sized,
{
    pub fn with_rng(rng: &'a mut R) -> Self {
        Self {
            _spin: PhantomData,
            rng,
        }
    }
}

impl<'a, S, T, R> StateResetSpec<S> for Paramagnetic<'a, T, R>
where
    T: Spin,
    S: SpinState<Spin = T>,
    R: Rng + ?Sized,
    StandardUniform: Distribution<T>,
{
    fn reset(&mut self, system: &mut S) {
        for (s, s_prime) in system
            .spins_mut()
            .zip(self.rng.sample_iter(StandardUniform))
        {
            *s = s_prime;
        }
    }
}

/// Lattice Spin States
pub mod lattice;
