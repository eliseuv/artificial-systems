//! Ising Model
//!

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

impl<S> HamiltonianSystem for Ising<S>
where
    S: SpinState,
{
    type H = i32;

    fn hamiltonian(&self) -> Self::H {
        self.state().total_interaction()
    }
}

impl<S> SpinSystem for Ising<S>
where
    S: SpinState,
{
    type State = S;

    #[inline(always)]
    fn state(&self) -> &Self::State {
        &self.state
    }

    #[inline(always)]
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}
