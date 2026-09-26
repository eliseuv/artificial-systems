//! Time Series Matrices from Systems
//!

use ndarray::Array2;
use rand::Rng;

use crate::mcmc::MarkovChain;

/// Specification to reset the system
pub trait SystemResetSpec<S> {
    fn reset(&mut self, system: &mut S);
}

/// Measurement over a system
pub trait Measurement<S> {
    /// Type of the measured quantity
    type Result;

    /// Perform measurement on system
    fn measure(system: &S) -> Self::Result;
}

pub trait TimeSeries<S, U>: Measurement<S>
where
    U: MarkovChain<S>,
{
    #[inline(always)]
    fn next<R: Rng + ?Sized>(system: &mut S, mcmc: &mut U, rng: &mut R) -> Self::Result {
        mcmc.step(system, rng);
        Self::measure(system)
    }

    /// Sample a time series
    fn time_series<R: Rng + ?Sized>(
        system: &mut S,
        mcmc: &mut U,
        n_steps: usize,
        rng: &mut R,
    ) -> Vec<Self::Result> {
        assert!(n_steps > 0, "Number of steps must be non-zero");
        let mut ts = Vec::with_capacity(n_steps + 1);
        ts.push(Self::measure(system));
        for x in ts.spare_capacity_mut().iter_mut().take(n_steps) {
            x.write(Self::next(system, mcmc, rng));
        }
        unsafe { ts.set_len(n_steps + 1) };

        ts
    }
}

impl<M, S, U> TimeSeries<S, U> for M
where
    M: Measurement<S>,
    U: MarkovChain<S>,
{
}

pub trait TimeSeriesMatrix<S, U>: TimeSeries<S, U>
where
    U: MarkovChain<S>,
{
    fn time_series_matrix<I, R>(
        system: &mut S,
        mcmc: &mut U,
        reset_spec: &mut I,
        n_steps: usize,
        n_samples: usize,
        rng: &mut R,
    ) -> Array2<Self::Result>
    where
        I: SystemResetSpec<S>,
        R: Rng + ?Sized,
    {
        assert!(n_steps > 0, "Number of steps must be non-zero");
        assert!(n_samples > 0, "Number of samples must be non-zero");
        let mut ts_matrix = Array2::<Self::Result>::uninit((n_samples, n_steps + 1));
        for mut ts in ts_matrix.rows_mut() {
            ts[0].write(Self::measure(system));
            for i in 1..(n_steps + 1) {
                ts[i].write(Self::next(system, mcmc, rng));
            }
            reset_spec.reset(system);
        }

        unsafe { ts_matrix.assume_init() }
    }
}

impl<M, S, U> TimeSeriesMatrix<S, U> for M
where
    M: TimeSeries<S, U>,
    U: MarkovChain<S>,
{
}
