//! Markov Chain Monte Carlo
//!

use rand::{Rng, seq::IteratorRandom};

use crate::{
    hamiltonian::HamiltonianSystem,
    spin_system::{SpinSystem, UpDownSymmetry, spin::spin_half::SpinHalf, state::SpinState},
};

/// Arbitrary Markov Chain
pub trait MarkovChain {
    /// A
    fn step<R>(&mut self, rng: &mut R)
    where
        R: Rng + ?Sized;

    fn advance<R>(&mut self, n_steps: usize, rng: &mut R)
    where
        R: Rng + ?Sized,
    {
        assert!(n_steps > 0, "Number of steps must be positive!");
        for _t in 0..n_steps {
            self.step(rng);
        }
    }
}

pub struct MetropolisSampler<S: HamiltonianSystem> {
    minus_beta: f64,
    system: S,
}

impl<S: HamiltonianSystem> MetropolisSampler<S> {
    pub fn with_system(system: S, beta: f64) -> Self {
        Self {
            minus_beta: -beta,
            system,
        }
    }
    /// Reference to inner system
    pub fn system(&self) -> &S {
        &self.system
    }

    pub fn get_system(self) -> S {
        self.system
    }
}

impl<S> MarkovChain for MetropolisSampler<S>
where
    S: SpinSystem,
    S::State: SpinState<Spin = SpinHalf>,
    S::H: Into<f64>,
{
    fn step<R>(&mut self, rng: &mut R)
    where
        R: Rng + ?Sized,
    {
        let random_indices: Vec<_> = (0..self.system.state().spin_count())
            .map(|_| {
                self.system
                    .state()
                    .indices()
                    .choose(rng)
                    .expect("Empty spin state")
            })
            .collect();
        for i in random_indices {
            // Calculate flip energy
            let dh = self.system.flip_energy(i).into();
            // Metropolis prescription
            if dh <= 0f64 || f64::exp(self.minus_beta * dh) > rng.random() {
                // Flip
                self.system[i] = self.system[i].flipped();
            }
        }
    }
}
