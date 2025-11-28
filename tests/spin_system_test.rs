use artificial_systems::lattice::Lattice;
use artificial_systems::lattice::initial_state::RandomSites;
use artificial_systems::lattice::square_lattice::impl_1d::SquareLattice1D;
use artificial_systems::mcmc::{MetropolisSampler, MetropolisSampling};
use artificial_systems::spin_system::ising::Ising;
use artificial_systems::spin_system::spin::spin_half::SpinHalf;
use artificial_systems::spin_system::state::lattice::LatticeSpinState;
use rand::SeedableRng;
use rand_distr::StandardUniform;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn test_ising_model_runs() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut spec = RandomSites::with_dist(StandardUniform, &mut rng);
    let lattice = SquareLattice1D::<SpinHalf>::new(100, &mut spec);
    let state = LatticeSpinState(lattice);
    let mut ising = Ising::with_initial_state(state);

    let sampler = MetropolisSampler::with_beta(1.0); // beta = 1.0

    // Test step
    sampler.step(&mut ising, &mut rng);

    // Test advance
    sampler.advance(&mut ising, 10, &mut rng);

    // We just want to ensure it runs without panic and types align
}
