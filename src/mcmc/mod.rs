//! Markov Chain Monte Carlo
//!

use ndarray::Array2;
use num_traits::Inv;
use rand::Rng;

use crate::{
    hamiltonian::HamiltonianSystem,
    method::{Measurement, SystemResetSpec},
};

/// Arbitrary Markov Chain
pub trait MarkovChain<S> {
    fn step<R>(&mut self, system: &mut S, rng: &mut R)
    where
        R: Rng + ?Sized;
}

/// Metropolis Sampling
pub trait MetropolisSampling<S: HamiltonianSystem> {
    fn step<R: Rng + ?Sized>(&self, system: &mut S, rng: &mut R);

    fn advance<R: Rng + ?Sized>(&self, system: &mut S, n_steps: usize, rng: &mut R);

    fn sample<M, R>(&self, system: &mut S, n_steps: usize, rng: &mut R) -> Vec<M::Result>
    where
        M: Measurement<S>,
        R: Rng + ?Sized;

    fn sample_multiple<M, U, R>(
        &self,
        system: &mut S,
        n_steps: usize,
        n_runs: usize,
        reset_spec: U,
        rng: &mut R,
    ) -> Array2<M::Result>
    where
        M: Measurement<S>,
        U: SystemResetSpec<S>,
        R: Rng + ?Sized;
}

/// Metropolis Sampler
pub struct MetropolisSampler {
    pub(crate) minus_beta: f64,
}

impl MetropolisSampler {
    pub fn with_beta(beta: f64) -> Self {
        Self { minus_beta: -beta }
    }

    pub fn with_temperature(temperature: f64) -> Self {
        let beta = temperature.inv();
        Self::with_beta(beta)
    }

    pub fn set_beta(&mut self, beta: f64) {
        self.minus_beta = -beta
    }

    pub fn set_temperature(&mut self, temperature: f64) {
        self.set_beta(temperature.inv())
    }

    pub fn beta(&self) -> f64 {
        -self.minus_beta
    }

    pub fn temperature(&self) -> f64 {
        self.beta().inv()
    }
}
