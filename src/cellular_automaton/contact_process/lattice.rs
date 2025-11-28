//! Lattice Contact Process
//!

use std::ops::{Index, IndexMut};

use num_traits::Inv;
use rand::Rng;
use rand::seq::{IteratorRandom, SliceRandom};
use rand_distr::Bernoulli;

use crate::cellular_automaton::contact_process::cell::Binary;
use crate::{
    cellular_automaton::{CellularAutomatonState, StochasticCellularAutomaton},
    lattice::{Lattice, square_lattice::impl_1d::SquareLattice1D},
    systems::{Measurement, StateResetSpec},
};
use ndarray::Array2;

/// Lattice Contact Process
pub struct LatticeContactProcess<L>
where
    L: Lattice<Site = Binary>,
{
    pub(crate) rate_inv: f64,
    pub(crate) diff_coin: Bernoulli,
    pub(crate) state: L,
}

impl<L> Index<L::Index> for LatticeContactProcess<L>
where
    L: Lattice<Site = Binary>,
{
    type Output = Binary;

    #[inline(always)]
    fn index(&self, i: L::Index) -> &Self::Output {
        &self.state[i]
    }
}

impl<L> IndexMut<L::Index> for LatticeContactProcess<L>
where
    L: Lattice<Site = Binary>,
{
    #[inline(always)]
    fn index_mut(&mut self, i: L::Index) -> &mut Self::Output {
        &mut self.state[i]
    }
}

impl<L> LatticeContactProcess<L>
where
    L: Lattice<Site = Binary>,
{
    pub fn new(state: L, rate: f64, diff_rate: f64) -> Self {
        assert!(0.0 <= rate, "Rate must be positive");
        assert!(
            (0.0..=1.0).contains(&diff_rate),
            "Diffusion rate must be in the interval [0, 1]"
        );
        Self {
            rate_inv: rate.inv(),
            diff_coin: Bernoulli::new(diff_rate).unwrap(),
            state,
        }
    }

    #[inline(always)]
    pub fn site_count(&self) -> usize {
        self.state.site_count()
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
        self.diff_coin.p()
    }

    #[inline(always)]
    pub fn set_diff_rate(&mut self, diff_rate: f64) {
        assert!(
            (0.0..=1.0).contains(&diff_rate),
            "Diffusion rate must be in the interval [0, 1]"
        );
        self.diff_coin = Bernoulli::new(diff_rate).unwrap();
    }

    #[inline(always)]
    pub fn total_active(&self) -> usize {
        Lattice::sites(&self.state)
            .filter(|&&s| s == Binary::Active)
            .count()
    }

    #[inline(always)]
    pub fn active(&self) -> f64 {
        self.total_active() as f64 / self.site_count() as f64
    }

    /// Perform a measurement over the system
    #[inline(always)]
    pub fn measure<M>(&self) -> M::Result
    where
        Self: Sized,
        M: Measurement<Self>,
    {
        M::measure(self)
    }
    fn get_indices(&mut self) -> Vec<(L::Index, *mut Binary)> {
        Lattice::indexed_sites_mut(&mut self.state)
            .map(|(i, s)| (i, std::ptr::from_mut(s)))
            .collect()
    }

    fn dynamics_sweep<R: Rng + ?Sized>(
        &mut self,
        indices: &mut [(L::Index, *mut Binary)],
        rng: &mut R,
    ) {
        indices.shuffle(rng);
        for &(i, s) in indices.iter() {
            // Update site based on its state
            unsafe {
                *s = match *s {
                    Binary::Active => {
                        // Test $\xi \in [0,1)$ against $1/\alpha$
                        let xi: f64 = rng.random();
                        match xi {
                            xi if xi < self.rate_inv => Binary::Inactive,
                            _ => Binary::Active,
                        }
                    }
                    // Copy state of random nearest neighbor
                    Binary::Inactive => *self.state.nearest_neighbors(i).choose(rng).unwrap(),
                }
            }
        }
    }

    fn apply_diffusion<R: Rng + ?Sized>(
        &mut self,
        indices: &mut [(L::Index, *mut Binary)],
        rng: &mut R,
    ) {
        indices.shuffle(rng);
    }
}

impl<L> CellularAutomatonState for LatticeContactProcess<L>
where
    L: Lattice<Site = Binary>,
{
    type Site = Binary;
    type Index = L::Index;

    #[inline(always)]
    fn site_count(&self) -> usize {
        Lattice::site_count(&self.state)
    }

    #[inline(always)]
    fn indices(&self) -> impl Iterator<Item = Self::Index> {
        Lattice::indices(&self.state)
    }

    #[inline(always)]
    fn sites<'a>(&'a self) -> impl Iterator<Item = &'a Self::Site>
    where
        Self::Site: 'a,
    {
        Lattice::sites(&self.state)
    }

    #[inline(always)]
    fn sites_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Self::Site>
    where
        Self::Site: 'a,
    {
        Lattice::sites_mut(&mut self.state)
    }

    #[inline(always)]
    fn indexed_sites<'a>(&'a self) -> impl Iterator<Item = (Self::Index, &'a Self::Site)>
    where
        Self::Site: 'a,
    {
        Lattice::indexed_sites(&self.state)
    }

    #[inline(always)]
    fn indexed_sites_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (Self::Index, &'a mut Self::Site)>
    where
        Self::Site: 'a,
    {
        Lattice::indexed_sites_mut(&mut self.state)
    }
}

/// 1D Contact Process
pub type ContactProcess1D = LatticeContactProcess<SquareLattice1D<Binary>>;

impl StochasticCellularAutomaton for ContactProcess1D {
    type State = SquareLattice1D<Binary>;

    #[inline(always)]
    fn state(&self) -> &Self::State {
        &self.state
    }

    #[inline(always)]
    fn to_state(self) -> Self::State {
        self.state
    }

    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let mut indices = self.get_indices();
        self.dynamics_sweep(&mut indices, rng);
    }

    fn advance<R: Rng + ?Sized>(&mut self, n_steps: usize, rng: &mut R) {
        let mut indices = self.get_indices();
        for _ in 0..n_steps {
            self.dynamics_sweep(&mut indices, rng);
        }
    }

    fn measure<M, R>(&mut self, n_steps: usize, rng: &mut R) -> Vec<M::Result>
    where
        Self: Sized,
        M: Measurement<Self>,
        R: Rng + ?Sized,
    {
        assert!(n_steps > 0, "Number of steps must be positive!");
        let mut result = Vec::with_capacity(n_steps + 1);
        let mut indices = self.get_indices();

        result.push(M::measure(self));
        for _ in 0..n_steps {
            self.dynamics_sweep(&mut indices, rng);
            result.push(M::measure(self));
        }

        result
    }

    fn measure_multiple<M, U, R>(
        &mut self,
        n_steps: usize,
        n_runs: usize,
        mut reset_spec: U,
        rng: &mut R,
    ) -> Array2<M::Result>
    where
        Self: Sized,
        M: Measurement<Self>,
        U: StateResetSpec<Self>,
        R: Rng + ?Sized,
    {
        assert!(n_runs > 0, "Number of runs must be positive!");
        assert!(n_steps > 0, "Number of steps must be positive!");
        let mut result = Array2::uninit((n_runs, n_steps + 1));
        let mut indices = self.get_indices();

        // Runs loop
        for n in 0..n_runs {
            // Reset system
            reset_spec.reset(self);
            // Initial measurement
            result[(n, 0)].write(M::measure(self));
            // Steps loop
            for t in 1..(n_steps + 1) {
                self.dynamics_sweep(&mut indices, rng);
                result[(n, t)].write(M::measure(self));
            }
        }

        unsafe { result.assume_init() }
    }
}
