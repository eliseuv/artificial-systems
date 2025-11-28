use artificial_systems::{
    cellular_automaton::{
        StochasticCellularAutomaton,
        contact_process::{
            cell::Binary, initial_state::Random, lattice::ContactProcess1D,
            measurement::TotalActiveSites,
        },
    },
    lattice::{Lattice, square_lattice::impl_1d::SquareLattice1D},
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn main() {
    let length = 64;
    let alpha = 2.5;
    let gamma = 0.1;
    let n_steps = 32;
    let n_samples = 16;
    let mut rng = Xoshiro256PlusPlus::from_os_rng();
    let mut system = ContactProcess1D::new(
        SquareLattice1D::<Binary>::new(length, &mut Random::new(&mut rng)),
        alpha,
        gamma,
    );

    println!("{}", system.state());
    for _t in 0..n_steps {
        system.step(&mut rng);
        println!("{}", system.state());
    }

    let x = system.measure_multiple::<TotalActiveSites, _, _>(
        n_steps,
        n_samples,
        Random::new(&mut rng.clone()),
        &mut rng,
    );

    dbg!(&x);
}
