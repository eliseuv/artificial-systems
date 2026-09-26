//! Lattice Contact Process
//!

use rand::{Rng, seq::IteratorRandom};
use rand_distr::Distribution;

use crate::{
    contact_process::{
        ContactProcessDiffusion, ContactProcessSystem, cell::Binary,
        state::lattice::LatticeContactProcessState,
    },
    lattice::{Lattice, initial_state::InitialStateSpec, square_lattice::impl_1d::SquareLattice1D},
};

/// Lattice Contact Process 1D with diffusion
pub type LatticeContactProcessDiffusion1D =
    ContactProcessDiffusion<LatticeContactProcessState<SquareLattice1D<Binary>>>;

impl LatticeContactProcessDiffusion1D {
    #[inline(always)]
    pub fn new<I>(length: usize, rate: f64, diffusion: f64, initial_state: &mut I) -> Self
    where
        I: InitialStateSpec<LatticeContactProcessState<SquareLattice1D<Binary>>>,
    {
        let state = initial_state.construct(length);
        Self::with_state(state, rate, diffusion)
    }

    #[inline(always)]
    fn dynamics_sweep<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let n = self.state.site_count();
        for _ in 0..n {
            // Select random site
            let i = rng.random_range(0..n);
            // Update site based on its state
            self.state[i] = match self.state[i] {
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

    #[inline(always)]
    fn apply_diffusion<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let n = self.state.site_count();
        for _ in 0..n {
            if self.diffusion_coin.sample(rng) {
                let i = rng.random_range(0..n);
                let nn_idx = self.state.nearest_neighbors_indices(i).next().unwrap();
                self.state.swap(i, nn_idx);
            }
        }
    }
}

impl ContactProcessSystem for LatticeContactProcessDiffusion1D {
    type State = LatticeContactProcessState<SquareLattice1D<Binary>>;

    #[inline(always)]
    fn state(&self) -> &Self::State {
        &self.state
    }

    #[inline(always)]
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }

    #[inline(always)]
    fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        self.dynamics_sweep(rng);
        self.apply_diffusion(rng);
    }
}

// fn advance<R: Rng + ?Sized>(&mut self, n_steps: usize, rng: &mut R) {
//     let mut indices: Vec<_> = self.indices().collect();
//     for _ in 0..n_steps {
//         self.dynamics_sweep(&mut indices, rng);
//         self.apply_diffusion(&mut indices, rng);
//     }
// }
//
// fn measure<M, R>(&mut self, n_steps: usize, rng: &mut R) -> Vec<M::Result>
// where
//     Self: Sized,
//     M: Measurement<Self>,
//     R: Rng + ?Sized,
// {
//     assert!(n_steps > 0, "Number of steps must be positive!");
//     let mut result = Vec::with_capacity(n_steps + 1);
//     let mut indices: Vec<_> = self.indices().collect();
//
//     result.push(M::measure(self));
//     for _ in 0..n_steps {
//         self.dynamics_sweep(&mut indices, rng);
//         self.apply_diffusion(&mut indices, rng);
//         result.push(M::measure(self));
//     }
//
//     result
// }
//
// fn measure_multiple<M, U, R>(
//     &mut self,
//     n_steps: usize,
//     n_runs: usize,
//     mut reset_spec: U,
//     rng: &mut R,
// ) -> Array2<M::Result>
// where
//     Self: Sized,
//     M: Measurement<Self>,
//     U: StateResetSpec<Self>,
//     R: Rng + ?Sized,
// {
//     assert!(n_runs > 0, "Number of runs must be positive!");
//     assert!(n_steps > 0, "Number of steps must be positive!");
//     let mut result = Array2::uninit((n_runs, n_steps + 1));
//     let mut indices: Vec<_> = self.indices().collect();
//
//     // Runs loop
//     for n in 0..n_runs {
//         // Reset system
//         reset_spec.reset(self);
//         // Initial measurement
//         result[(n, 0)].write(M::measure(self));
//         // Steps loop
//         for t in 1..(n_steps + 1) {
//             self.dynamics_sweep(&mut indices, rng);
//             self.apply_diffusion(&mut indices, rng);
//             result[(n, t)].write(M::measure(self));
//         }
//     }
//
//     unsafe { result.assume_init() }
// }
