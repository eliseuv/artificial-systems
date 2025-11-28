use artificial_systems::{
    hamiltonian::TotalEnergy,
    lattice::{Lattice, square_lattice::impl_2d::SquareLattice2D},
    mcmc::{MetropolisSampler, MetropolisSampling},
    spin_system::{
        SpinSystem,
        ising::Ising,
        measurement::{Magnetization, TotalMagnetization},
        spin::spin_half::SpinHalf,
        state::{Ferromagnetic, Paramagnetic, SpinState, lattice::LatticeSpinState},
    },
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn main() {
    let length = 70;
    let mut ising = Ising::with_initial_state(LatticeSpinState(SquareLattice2D::new(
        length,
        &mut Ferromagnetic(SpinHalf::Down),
    )));

    println!(
        "{state}\nL = {length}\tN = {N}\nM = {M}\tm = {m}\nE = {E}",
        state = ising.state(),
        N = ising.spin_count(),
        M = ising.measure::<TotalMagnetization>(),
        m = ising.measure::<Magnetization>(),
        E = ising.measure::<TotalEnergy>()
    );

    let mut rng = Xoshiro256PlusPlus::from_os_rng();

    let temperature = 0.0;
    let sampler = MetropolisSampler::with_temperature(temperature);

    let n_samples = 8;
    let n_steps = 16;
    let x = sampler.sample_multiple::<TotalMagnetization, _, _>(
        &mut ising,
        n_steps,
        n_samples,
        Paramagnetic::with_rng(&mut Xoshiro256PlusPlus::from_os_rng()),
        &mut rng,
    );

    dbg!(&x);

    println!(
        "{state}\nL = {length}\tN = {N}\nM = {M}\tm = {m}\nE = {E}",
        state = ising.state(),
        N = ising.spin_count(),
        M = ising.measure::<TotalMagnetization>(),
        m = ising.measure::<Magnetization>(),
        E = ising.measure::<TotalEnergy>()
    );
}
