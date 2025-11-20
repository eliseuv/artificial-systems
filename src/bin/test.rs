use artificial_systems::{
    lattice::{Lattice, square_lattice::impl_2d::SquareLattice2D},
    mcmc::{MarkovChain, MetropolisSampler},
    spin_system::{
        SpinSystem,
        ising::Ising,
        measurement::{Magnetization, TotalEnergy, TotalMagnetization},
        spin::spin_half::SpinHalf,
        state::Paramagnetic,
    },
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn main() {
    let mut rng = Xoshiro256PlusPlus::from_os_rng();
    let mut init_spec = Paramagnetic::<SpinHalf, _>::with_rng(&mut rng);
    let length = 128;
    let lattice = SquareLattice2D::new(length, &mut init_spec);
    let ising = Ising::with_initial_state(lattice);

    println!(
        "{state}\nL = {length}\tN = {N}\nM = {M}\tm = {m}\nE = {E}",
        state = ising.state(),
        N = ising.state().site_count(),
        M = ising.measure::<TotalMagnetization>(),
        m = ising.measure::<Magnetization>(),
        E = ising.measure::<TotalEnergy>()
    );

    let beta = 1000.0;
    let mut sampler = MetropolisSampler::with_system(ising, beta);
    let n_steps = 8;
    for _ in 0..n_steps {
        sampler.step(&mut rng);
        println!("{state}", state = sampler.system().state());
    }
}
