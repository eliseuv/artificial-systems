//! Markov Chain Monte Carlo
//!

use rand::{Rng, seq::IteratorRandom};

use crate::{
    hamiltonian::HamiltonianSystem,
    spin_system::{SpinSystem, UpDownSymmetry, spin::spin_half::SpinHalf, state::SpinState},
};

/// Arbitrary Markov Chain
pub trait MarkovChain<S> {
    /// A
    fn step<R: Rng + ?Sized>(&mut self, system: &mut S, rng: &mut R);

    fn advance<R: Rng + ?Sized>(&mut self, system: &mut S, n_steps: usize, rng: &mut R) {
        assert!(n_steps > 0, "Number of steps must be positive!");
        for _t in 0..n_steps {
            self.step(system, rng);
        }
    }
}

/// Metropolis Sampler
pub trait MetropolisSampling: HamiltonianSystem {
    /// Step at a given temperature
    fn step<R>(&mut self, beta: Self::H, rng: &mut R)
    where
        R: Rng + ?Sized;

    /// Step at infinite temperature
    fn step_beta_infty<R>(&mut self, rng: &mut R)
    where
        R: Rng + ?Sized;

    /// Step at zero temperature
    fn step_beta_zero<R>(&mut self, rng: &mut R)
    where
        R: Rng + ?Sized;
}

impl<S> MetropolisSampling for S
where
    S: SpinSystem,
    S::State: SpinState<Spin = SpinHalf>,
    S::H: PartialOrd<i32>,
{
    fn step<R>(&mut self, beta: Self::H, rng: &mut R)
    where
        R: Rng + ?Sized,
    {
        let random_indices: Vec<_> = (0..self.state().spin_count())
            .map(|_| {
                self.state()
                    .indices()
                    .choose(rng)
                    .expect("Empty spin state")
            })
            .collect();
        for i in random_indices {
            // Calculate flip energy
            let dh = self.flip_energy(i);
            // Metropolis prescription
            if dh <= 0 || f64::exp((-beta * dh).into()) > rng.random() {
                // Flip
                self.state_mut()[i] = self.state()[i].flipped();
            }
        }
    }

    fn step_beta_infty<R>(&mut self, rng: &mut R)
    where
        R: Rng + ?Sized,
    {
        let random_indices: Vec<_> = (0..self.state().spin_count())
            .map(|_| {
                self.state()
                    .indices()
                    .choose(rng)
                    .expect("Empty spin state")
            })
            .collect();
        for i in random_indices {
            let dh = self.flip_energy(i);
            // Metropolis prescription
            if dh <= 0 {
                // Flip
                self.state_mut()[i] = self.state()[i].flipped();
            }
        }
    }

    fn step_beta_zero<R>(&mut self, rng: &mut R)
    where
        R: Rng + ?Sized,
    {
        let random_indices: Vec<_> = (0..self.state().spin_count())
            .map(|_| {
                self.state()
                    .indices()
                    .choose(rng)
                    .expect("Empty spin state")
            })
            .collect();
        for i in random_indices {
            // Flip
            self.state_mut()[i] = self.state()[i].flipped();
        }
    }
}
