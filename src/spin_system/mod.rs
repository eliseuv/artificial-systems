//! Spin Systems
//!

use crate::{
    hamiltonian::HamiltonianSystem,
    method::Measurement,
    spin_system::{spin::spin_half::SpinHalf, state::SpinState},
};

/// Single spins states
pub mod spin;

/// Spin States
pub mod state;

/// Spin System
pub trait SpinSystem:
    HamiltonianSystem
    + SpinState<Spin = <Self::State as SpinState>::Spin, Index = <Self::State as SpinState>::Index>
{
    /// Underlying system state
    type State: SpinState;

    /// Reference to underlying state
    fn state(&self) -> &Self::State;

    /// Get underlying state
    fn to_state(self) -> Self::State;

    /// Perform a measurement over the system
    #[inline(always)]
    fn measure<M>(&self) -> M::Result
    where
        Self: Sized,
        M: Measurement<Self>,
    {
        M::measure(self)
    }
}

pub(crate) trait SpinHalfSystem: SpinSystem
where
    <Self as SpinSystem>::State: SpinState<Spin = SpinHalf>,
{
    /// Energy difference of single spin flip
    fn flip_energy(&self, i: <Self::State as SpinState>::Index) -> Self::H;
}

impl<S> SpinHalfSystem for S
where
    S: SpinSystem,
    S::State: SpinState<Spin = SpinHalf>,
{
    fn flip_energy(&self, i: <Self::State as SpinState>::Index) -> Self::H {
        let s_i = self[i] as i32;
        let nn_sum = self.nn_sum(i);
        (2 * s_i * nn_sum).into()
    }
}

/// Measurements over spin systems
pub mod measurement;

/// Sampling of spin systems
pub mod sampling;

/// Ising Model
pub mod ising;
