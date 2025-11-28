//! Cellular Automata
//!

use ndarray::Array2;
use rand::Rng;

use crate::{
    cellular_automaton::state::CellularAutomatonState,
    systems::{Measurement, StateResetSpec},
};

/// Cellular Automaton State
pub mod state;

/// Stochastic Cellular Automaton
pub trait StochasticCellularAutomaton:
    CellularAutomatonState<
        Site = <Self::State as CellularAutomatonState>::Site,
        Index = <Self::State as CellularAutomatonState>::Index,
    >
{
    /// Underlying system state
    type State: CellularAutomatonState;

    /// Reference to underlying state
    fn state(&self) -> &Self::State;

    /// Get underlying state
    fn to_state(self) -> Self::State;

    /// Single step
    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R);

    /// Advance multiple steps
    #[inline(always)]
    fn advance<R: Rng + ?Sized>(&mut self, n_steps: usize, rng: &mut R) {
        for _ in 0..n_steps {
            self.step(rng);
        }
    }

    /// Perform a measurement over the system
    fn measure<M, R>(&mut self, n_steps: usize, rng: &mut R) -> Vec<M::Result>
    where
        Self: Sized,
        M: Measurement<Self>,
        R: Rng + ?Sized;

    /// Perform multiple measurements over the system
    fn measure_multiple<M, U, R>(
        &mut self,
        n_steps: usize,
        n_runs: usize,
        reset_spec: U,
        rng: &mut R,
    ) -> Array2<M::Result>
    where
        Self: Sized,
        M: Measurement<Self>,
        U: StateResetSpec<Self>,
        R: Rng + ?Sized;
}

/// Contact Process
pub mod contact_process;
