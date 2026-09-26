//! Markov chain dynamics.
//!
//! One [`Dynamics::step`] is one Monte Carlo step, i.e. `N` single site update attempts for a
//! system of `N` sites, so time is measured in the same units for every model.

use rand::{Rng, RngExt as _, seq::SliceRandom as _};
use serde::{Deserialize, Serialize};

use crate::{site::Site, topology::Topology};

mod heat_bath;
mod metropolis;

pub use heat_bath::{Glauber, HeatBath};
pub use metropolis::Metropolis;

/// Stochastic time evolution of a system of type `Sys`.
pub trait Dynamics<Sys>: Clone + Send + Sync {
    /// Advance `sys` by one time step.
    fn step<R: Rng + ?Sized>(&mut self, sys: &mut Sys, rng: &mut R);

    /// Whether `sys` is in an absorbing configuration that no further step can change.
    fn is_frozen(&self, _sys: &Sys) -> bool {
        false
    }
}

/// Order in which sites are visited during one Monte Carlo step of `N` attempts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[serde(rename_all = "lowercase")]
pub enum SiteOrder {
    /// `N` sites drawn uniformly with replacement (random sequential update).
    #[default]
    Random,
    /// Every site once, in index order (typewriter sweep).
    Sequential,
    /// Every site once, in a fresh random order each step.
    Permutation,
    /// Every site once, first one sublattice of a bipartite topology, then the other.
    Checkerboard,
}

/// Site visiting schedule, keeping any buffer it needs across steps.
#[derive(Debug, Clone, Default)]
pub struct Sweep {
    order: SiteOrder,
    buffer: Vec<u32>,
}

impl Sweep {
    /// Schedule visiting sites in the given order.
    pub fn new(order: SiteOrder) -> Self {
        Self {
            order,
            buffer: Vec::new(),
        }
    }

    /// Site order.
    pub fn order(&self) -> SiteOrder {
        self.order
    }

    /// Call `f(i, rng)` for the `N = topology.len()` site visits of one step.
    ///
    /// # Panics
    /// For [`SiteOrder::Checkerboard`] on a topology that is not bipartite.
    #[inline(always)]
    pub fn run<T, R, F>(&mut self, topology: &T, rng: &mut R, mut f: F)
    where
        T: Topology + ?Sized,
        R: Rng + ?Sized,
        F: FnMut(usize, &mut R),
    {
        let n = topology.len();
        match self.order {
            SiteOrder::Random => {
                for _ in 0..n {
                    let i = rng.random_range(0..n);
                    f(i, rng);
                }
            }
            SiteOrder::Sequential => (0..n).for_each(|i| f(i, rng)),
            SiteOrder::Permutation => {
                if self.buffer.len() != n {
                    self.buffer = (0..n as u32).collect();
                }
                self.buffer.shuffle(rng);
                self.buffer.iter().for_each(|&i| f(i as usize, rng));
            }
            SiteOrder::Checkerboard => {
                if self.buffer.len() != n {
                    let colour = topology
                        .bipartition()
                        .expect("Checkerboard order requires a bipartite topology");
                    self.buffer = (0..n as u32)
                        .filter(|&i| !colour[i as usize])
                        .chain((0..n as u32).filter(|&i| colour[i as usize]))
                        .collect();
                }
                self.buffer.iter().for_each(|&i| f(i as usize, rng));
            }
        }
    }
}

/// Uniformly random state different from `old`; a flip for two-state sites.
#[inline(always)]
pub(crate) fn propose_other<S: Site, R: Rng + ?Sized>(old: S, rng: &mut R) -> S {
    if S::COUNT == 2 {
        S::from_index(1 - old.index())
    } else {
        let k = rng.random_range(0..S::COUNT - 1);
        S::from_index(if k >= old.index() { k + 1 } else { k })
    }
}

/// Inverse temperature `β = 1/T` (with `T = 0` giving `β = ∞`).
///
/// # Panics
/// If `temperature` is negative or NaN.
pub fn beta_from_temperature(temperature: f64) -> f64 {
    assert!(
        temperature >= 0.0,
        "Temperature must be non-negative, got {temperature}"
    );
    temperature.recip()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{rng::stream, site::SpinOne, topology::Square};

    #[test]
    fn orders_visit_every_site_once() {
        let top = Square::periodic_cube(4);
        let mut rng = stream(0, &[]);
        for order in [
            SiteOrder::Sequential,
            SiteOrder::Permutation,
            SiteOrder::Checkerboard,
        ] {
            let mut sweep = Sweep::new(order);
            for _ in 0..2 {
                let mut visits = [0; 16];
                sweep.run(&top, &mut rng, |i, _| visits[i] += 1);
                assert!(visits.iter().all(|&v| v == 1), "{order:?}");
            }
        }
        let mut count = 0;
        Sweep::new(SiteOrder::Random).run(&top, &mut rng, |_, _| count += 1);
        assert_eq!(count, 16);
    }

    #[test]
    fn checkerboard_alternates_sublattices() {
        let top = Square::periodic_cube(4);
        let mut sweep = Sweep::new(SiteOrder::Checkerboard);
        let mut seen = Vec::new();
        sweep.run(&top, &mut stream(0, &[]), |i, _| seen.push(i));
        let parity = |i: usize| top.coords(i).iter().sum::<usize>() % 2;
        assert!(seen[..8].iter().all(|&i| parity(i) == parity(seen[0])));
        assert!(seen[8..].iter().all(|&i| parity(i) != parity(seen[0])));
    }

    #[test]
    fn proposals_differ_from_old() {
        let mut rng = stream(1, &[]);
        let mut hits = [0i32; 3];
        for _ in 0..3000 {
            let new = propose_other(SpinOne::Zero, &mut rng);
            assert_ne!(new, SpinOne::Zero);
            hits[new.index()] += 1;
        }
        assert!((hits[0] - 1500).abs() < 150);
    }

    #[test]
    fn temperatures() {
        assert_eq!(beta_from_temperature(0.0), f64::INFINITY);
        assert_eq!(beta_from_temperature(2.0), 0.5);
    }
}
