//! Spin System Sampling
//!

use ndarray::Array2;
use rand::{Rng, seq::SliceRandom};

use crate::{
    mcmc::{MetropolisSampler, MetropolisSampling},
    spin_system::{SpinHalfSystem, SpinSystem, spin::spin_half::SpinHalf, state::SpinState},
    systems::{Measurement, StateResetSpec},
};

fn sweep<S, R>(sampler: &MetropolisSampler, system: &mut S, indices: &mut [S::Index], rng: &mut R)
where
    S: SpinSystem,
    S::State: SpinState<Spin = SpinHalf>,
    S::H: Into<f64>,
    R: Rng + ?Sized,
{
    indices.shuffle(rng);
    for i in indices.iter() {
        // Calculate flip energy
        let dh = system.flip_energy(*i).into();
        // Metropolis prescription
        if dh <= 0f64 || f64::exp(sampler.minus_beta * dh) > rng.random() {
            // Flip
            system[*i] = system[*i].flipped();
        }
    }
}

/// Metropolis sampling of Spin`1/2` system
impl<S> MetropolisSampling<S> for MetropolisSampler
where
    S: SpinSystem,
    S::State: SpinState<Spin = SpinHalf>,
    S::H: Into<f64>,
{
    fn step<R>(&self, system: &mut S, rng: &mut R)
    where
        R: Rng + ?Sized,
    {
        let mut indices: Vec<_> = system.indices().collect();
        sweep(self, system, &mut indices, rng);
    }

    fn advance<R>(&self, system: &mut S, n_steps: usize, rng: &mut R)
    where
        R: Rng + ?Sized,
    {
        assert!(n_steps > 0, "Number of steps must be positive!");
        let mut indices: Vec<_> = system.indices().collect();
        for _t in 0..n_steps {
            sweep(self, system, &mut indices, rng);
        }
    }

    fn sample<M, R>(&self, system: &mut S, n_steps: usize, rng: &mut R) -> Vec<M::Result>
    where
        M: Measurement<S>,
        R: Rng + ?Sized,
    {
        assert!(n_steps > 0, "Number of steps must be positive!");
        let mut result = Vec::with_capacity(n_steps + 1);
        result.push(M::measure(system));
        let mut indices: Vec<_> = system.indices().collect();
        for _t in 0..n_steps {
            sweep(self, system, &mut indices, rng);
            result.push(M::measure(system));
        }

        result
    }

    fn sample_multiple<M, U, R>(
        &self,
        system: &mut S,
        n_steps: usize,
        n_runs: usize,
        mut reset_spec: U,
        rng: &mut R,
    ) -> Array2<M::Result>
    where
        M: Measurement<S>,
        U: StateResetSpec<S>,
        R: Rng + ?Sized,
    {
        assert!(n_runs > 0, "Number of runs must be positive!");
        assert!(n_steps > 0, "Number of steps must be positive!");
        // Pre-allocate
        let mut indices: Vec<_> = system.indices().collect();
        let mut result = Array2::uninit((n_runs, n_steps + 1));
        // Runs loop
        for n in 0..n_runs {
            // Reset system
            reset_spec.reset(system);
            // Initial measurement
            result[(n, 0)].write(M::measure(system));
            // Time loop
            for t in 1..(n_steps + 1) {
                sweep(self, system, &mut indices, rng);
                result[(n, t)].write(M::measure(system));
            }
        }

        unsafe { result.assume_init() }
    }
}
