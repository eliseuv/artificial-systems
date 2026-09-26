//! Ensembles of independent Markov chains and their time series.
//!
//! Chain `(run, sample)` draws all its randomness from
//! [`rng::stream(seed, [run, sample])`](crate::rng::stream), so every time series is determined
//! by the master seed and its coordinates only, whatever the number of threads.

use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{dynamics::Dynamics, observable::Observable, rng::stream, state::Prepare};

/// When to measure along a chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schedule {
    /// Number of measurements after the initial one (time series length is `n_steps + 1`).
    pub n_steps: usize,
    /// Steps discarded before the initial measurement.
    pub burn_in: usize,
    /// Steps between consecutive measurements (`>= 1`).
    pub stride: usize,
}

impl Schedule {
    /// Measure at every step from the initial state on (`t = 0, 1, …, n_steps`).
    pub const fn new(n_steps: usize) -> Self {
        Self {
            n_steps,
            burn_in: 0,
            stride: 1,
        }
    }
}

/// Ensemble of independent chains of a system under some dynamics, each started from a fresh
/// initial state and measured with an observable.
#[derive(Debug, Clone)]
pub struct Ensemble<Sys, D, P, O> {
    /// Prototype system, cloned for each chain (its state is overwritten by `prepare`).
    pub system: Sys,
    /// Dynamics, cloned for each chain.
    pub dynamics: D,
    /// Initial state of every chain.
    pub prepare: P,
    /// Measured quantity.
    pub observable: O,
    /// Measurement schedule.
    pub schedule: Schedule,
    /// Master seed.
    pub seed: u64,
}

impl<Sys, D, P, O> Ensemble<Sys, D, P, O>
where
    Sys: Clone + Send + Sync,
    D: Dynamics<Sys>,
    P: Prepare<Sys>,
    O: Observable<Sys>,
{
    /// Time series `[x(0), x(stride), …, x(n_steps · stride)]` of chain `(run, sample)`.
    ///
    /// Once the system reaches an absorbing configuration (see [`Dynamics::is_frozen`]) the
    /// remaining measurements are copies of the last one and no further steps are simulated.
    pub fn time_series(&self, run: u64, sample: u64) -> Vec<O::Output> {
        let Schedule {
            n_steps,
            burn_in,
            stride,
        } = self.schedule;
        assert!(stride >= 1, "Measurement stride must be positive");
        let mut rng = stream(self.seed, &[run, sample]);
        let mut sys = self.system.clone();
        let mut dynamics = self.dynamics.clone();
        self.prepare.prepare(&mut sys, &mut rng);
        let mut frozen = false;
        let mut advance = |sys: &mut Sys, steps: usize, frozen: &mut bool| {
            for _ in 0..steps {
                if dynamics.is_frozen(sys) {
                    *frozen = true;
                    return;
                }
                dynamics.step(sys, &mut rng);
            }
        };
        advance(&mut sys, burn_in, &mut frozen);
        let mut series = Vec::with_capacity(n_steps + 1);
        series.push(self.observable.measure(&sys));
        for _ in 0..n_steps {
            if !frozen {
                advance(&mut sys, stride, &mut frozen);
            }
            if frozen {
                let last = series.last().expect("initial measurement").clone();
                series.resize(n_steps + 1, last);
                break;
            }
            series.push(self.observable.measure(&sys));
        }
        series
    }

    /// Time series matrix of run `run`: row `k` is the series of sample `k`, so the shape is
    /// `(n_samples, n_steps + 1)`. Samples are simulated in parallel.
    pub fn matrix(&self, run: u64, n_samples: usize) -> Array2<O::Output> {
        let rows: Vec<Vec<O::Output>> = (0..n_samples as u64)
            .into_par_iter()
            .map(|sample| self.time_series(run, sample))
            .collect();
        let n_cols = self.schedule.n_steps + 1;
        Array2::from_shape_vec((n_samples, n_cols), rows.into_iter().flatten().collect())
            .expect("every series has n_steps + 1 entries")
    }

    /// Apply `reduce` to the time series matrix of each of `n_runs` runs, in parallel, keeping
    /// only the reduced results (e.g. correlation matrix spectra) in memory.
    pub fn map_runs<T, F>(&self, n_runs: usize, n_samples: usize, reduce: F) -> Vec<T>
    where
        T: Send,
        F: Fn(u64, Array2<O::Output>) -> T + Send + Sync,
    {
        (0..n_runs as u64)
            .into_par_iter()
            .map(|run| reduce(run, self.matrix(run, n_samples)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        dynamics::HeatBath,
        model::Beg,
        observable::Magnetization,
        site::SpinHalf,
        state::{Init, LatticeState},
        system::SpinSystem,
        topology::Square,
    };

    fn ising_ensemble(seed: u64) -> impl Fn(Schedule) -> Array2<f64> {
        move |schedule| {
            let top = Arc::new(Square::periodic_cube(8));
            Ensemble {
                system: SpinSystem::new(
                    LatticeState::uniform(top, SpinHalf::Up),
                    Beg::ising(1.0, 0.0),
                ),
                dynamics: HeatBath::new(0.4),
                prepare: Init::IidUniform,
                observable: Magnetization,
                schedule,
                seed,
            }
            .matrix(3, 6)
        }
    }

    #[test]
    fn matrices_are_reproducible_and_thread_independent() {
        let schedule = Schedule::new(20);
        let a = ising_ensemble(11)(schedule);
        let b = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| ising_ensemble(11)(schedule));
        assert_eq!(a.dim(), (6, 21));
        assert_eq!(a, b);
        assert_ne!(a, ising_ensemble(12)(schedule));
        // Rows are distinct chains
        assert_ne!(a.row(0), a.row(1));
    }

    #[test]
    fn burn_in_and_stride_shift_the_same_chain() {
        let full = ising_ensemble(5)(Schedule::new(12));
        let strided = ising_ensemble(5)(Schedule {
            n_steps: 4,
            burn_in: 2,
            stride: 2,
        });
        for k in 0..6 {
            for t in 0..=4 {
                assert_eq!(strided[[k, t]], full[[k, 2 + 2 * t]]);
            }
        }
    }
}
