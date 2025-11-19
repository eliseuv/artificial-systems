//! Spin States
//!

use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
};

use rand::Rng;

use crate::spin_system::spin::Spin;

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
    fn total_magnet(&self) -> i32;

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

/// Specification of spin state configuration
pub trait SpinStateSpec {}

/// Ferromagnetic (ordered) spin configuration
#[derive(Debug, Clone, Copy)]
pub struct Ferromagnetic<T: Spin>(pub T);

impl<T> SpinStateSpec for Ferromagnetic<T> where T: Spin {}

/// Paramagentic (disordered) spin configuration
pub struct Paramagnetic<'a, T, R>
where
    T: Spin,
    R: Rng + ?Sized,
{
    _spin: PhantomData<T>,
    rng: &'a mut R,
}

impl<'a, T, R> SpinStateSpec for Paramagnetic<'a, T, R>
where
    T: Spin,
    R: Rng + ?Sized,
{
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

/// Lattice Spin States
pub mod lattice;
