//! Contact Process Model
//!

use num_traits::Inv;
use rand::Rng;
use rand_distr::Bernoulli;

use crate::{contact_process::state::ContactProcessState, mcmc::MarkovChain};

/// Contact Process Cell
pub mod cell;

/// Contact Process State
pub mod state;

/// Contact Process
pub trait ContactProcessSystem {
    type State: ContactProcessState;

    /// Reference to underlying state
    fn state(&self) -> &Self::State;

    /// Mutable reference to underlying state
    fn state_mut(&mut self) -> &mut Self::State;

    /// Single Step
    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R);
}

/// Contact Process Measurements
pub mod measurement;

#[derive(Debug)]
pub struct ContactProcessDiffusion<S: ContactProcessState> {
    pub(crate) rate_inv: f64,
    pub(crate) diffusion_coin: Bernoulli,
    pub(crate) state: S,
}

impl<S: ContactProcessState> ContactProcessDiffusion<S> {
    #[inline(always)]
    pub fn with_state(state: S, rate: f64, diffusion: f64) -> Self {
        assert!(0.0 <= rate, "Rate must be positive");
        assert!(
            (0.0..=1.0).contains(&diffusion),
            "Diffusion rate must be in the interval [0, 1]"
        );
        Self {
            rate_inv: rate.inv(),
            diffusion_coin: Bernoulli::new(diffusion).unwrap(),
            state,
        }
    }

    #[inline(always)]
    pub fn rate(&self) -> f64 {
        self.rate_inv.inv()
    }

    #[inline(always)]
    pub fn set_rate(&mut self, rate: f64) {
        assert!(rate >= 0.0, "Rate must be positive");
        self.rate_inv = rate.inv();
    }

    #[inline(always)]
    pub fn diff_rate(&self) -> f64 {
        self.diffusion_coin.p()
    }

    #[inline(always)]
    pub fn set_diff_rate(&mut self, diff_rate: f64) {
        assert!(
            (0.0..=1.0).contains(&diff_rate),
            "Diffusion rate must be in the interval [0, 1]"
        );
        self.diffusion_coin = Bernoulli::new(diff_rate).unwrap();
    }
}

/// Lattice Contact Process
pub mod lattice;

pub struct ContactProcessMarkovChain;

impl<S: ContactProcessSystem> MarkovChain<S> for ContactProcessMarkovChain {
    #[inline(always)]
    fn step<R>(&mut self, system: &mut S, rng: &mut R)
    where
        R: Rng + ?Sized,
    {
        system.step(rng);
    }
}
