//! Ising Model
//!

use std::ops::{Index, IndexMut};

use crate::{
    hamiltonian::HamiltonianSystem,
    spin_system::{SpinSystem, state::SpinState},
};

/// Critical temperature for Ising model on the 2D square lattice
/// $\beta_C = 2 / \ln(1 + \sqrt{2})$
pub const ISING_SQ2D_BETA_CRITICAL: f64 = 2.269185314213022;

/// Ising Model
#[derive(Debug)]
pub struct Ising<S: SpinState> {
    state: S,
}

impl<S: SpinState> Ising<S> {
    /// New Ising system with given initial state
    pub fn with_initial_state(state: S) -> Self {
        Self { state }
    }
}

impl<S: SpinState> Index<S::Index> for Ising<S> {
    type Output = S::Spin;

    fn index(&self, i: S::Index) -> &Self::Output {
        &self.state[i]
    }
}

impl<S: SpinState> IndexMut<S::Index> for Ising<S> {
    fn index_mut(&mut self, i: S::Index) -> &mut Self::Output {
        &mut self.state[i]
    }
}

impl<S> HamiltonianSystem for Ising<S>
where
    S: SpinState,
{
    type H = i32;

    fn hamiltonian(&self) -> Self::H {
        self.total_interaction()
    }
}

impl<S> SpinSystem for Ising<S>
where
    S: SpinState,
{
    type State = S;

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn to_state(self) -> Self::State {
        self.state
    }
}

impl<S> SpinState for Ising<S>
where
    S: SpinState,
{
    type Spin = S::Spin;

    type Index = S::Index;

    fn spin_count(&self) -> usize {
        self.state.spin_count()
    }

    fn indices(&self) -> impl Iterator<Item = Self::Index> {
        self.state.indices()
    }

    fn indexed_spins<'a>(&'a self) -> impl Iterator<Item = (Self::Index, &'a Self::Spin)>
    where
        Self::Spin: 'a,
    {
        self.state.indexed_spins()
    }

    fn indexed_spins_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (Self::Index, &'a mut Self::Spin)>
    where
        Self::Spin: 'a,
    {
        self.state.indexed_spins_mut()
    }

    fn total_magnet(&self) -> i32 {
        self.state.total_magnet()
    }

    fn nn_sum(&self, i: Self::Index) -> i32 {
        self.state.nn_sum(i)
    }

    fn interaction(&self, i: Self::Index) -> i32 {
        self.state.interaction(i)
    }

    fn total_interaction(&self) -> i32 {
        self.state.total_interaction()
    }
}
