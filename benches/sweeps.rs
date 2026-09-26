//! Throughput of one Monte Carlo step for the thesis workloads, and of the spectral analysis.

use std::{hint::black_box, sync::Arc};

use artificial_systems::{
    analysis::{ZeroVariance, correlation_spectrum},
    automaton::contact_process,
    dynamics::{Dynamics, HeatBath, Metropolis},
    model::{Beg, Potts},
    rng::stream,
    site::{Binary, PottsState, SpinHalf, SpinOne},
    state::{Init, LatticeState, MeanFieldState, Prepare},
    system::SpinSystem,
    topology::{Chain, Cubic, Square, Topology},
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ndarray::Array2;
use rand::RngExt as _;

fn spin_sweeps(c: &mut Criterion) {
    let mut group = c.benchmark_group("sweep");

    let square = Arc::new(Square::periodic_cube(100));
    group.throughput(Throughput::Elements(square.len() as u64));
    let mut rng = stream(0, &[]);
    let mut state = LatticeState::uniform(square.clone(), SpinHalf::Up);
    Init::IidUniform.prepare(&mut state, &mut rng);
    let mut ising = SpinSystem::new(state, Beg::ising(1.0, 0.0));
    let mut metropolis = Metropolis::new(1.0 / 2.269);
    group.bench_function("ising_square_L100_metropolis", |b| {
        b.iter(|| metropolis.step(black_box(&mut ising), &mut rng))
    });
    let mut heat_bath = HeatBath::new(1.0 / 2.269);
    group.bench_function("ising_square_L100_heat_bath", |b| {
        b.iter(|| heat_bath.step(black_box(&mut ising), &mut rng))
    });

    let cubic = Arc::new(Cubic::periodic_cube(22));
    group.throughput(Throughput::Elements(cubic.len() as u64));
    let mut state = LatticeState::uniform(cubic, SpinOne::Zero);
    Init::IidUniform.prepare(&mut state, &mut rng);
    let mut bc = SpinSystem::new(state, Beg::blume_capel(1.0, 1.0, 0.0));
    let mut heat_bath = HeatBath::new(1.0 / 3.2);
    group.bench_function("blume_capel_cubic_L22_heat_bath", |b| {
        b.iter(|| heat_bath.step(black_box(&mut bc), &mut rng))
    });

    let mut state =
        LatticeState::uniform(Arc::new(Square::periodic_cube(64)), PottsState::<3>::new(0));
    group.throughput(Throughput::Elements(64 * 64));
    Init::IidUniform.prepare(&mut state, &mut rng);
    let mut potts = SpinSystem::new(state, Potts::new(1.0, 0.0));
    let mut heat_bath = HeatBath::new(1.0);
    group.bench_function("potts3_square_L64_heat_bath", |b| {
        b.iter(|| heat_bath.step(black_box(&mut potts), &mut rng))
    });

    group.throughput(Throughput::Elements(10_000));
    let mut state = MeanFieldState::uniform(10_000, 4.0, SpinHalf::Up);
    Init::IidUniform.prepare(&mut state, &mut rng);
    let mut mean_field = SpinSystem::new(state, Beg::ising(1.0, 0.0));
    let mut heat_bath = HeatBath::new(0.25);
    group.bench_function("ising_mean_field_N10000_heat_bath", |b| {
        b.iter(|| heat_bath.step(black_box(&mut mean_field), &mut rng))
    });
    let mut metropolis = Metropolis::new(0.25);
    group.bench_function("ising_mean_field_N10000_metropolis", |b| {
        b.iter(|| metropolis.step(black_box(&mut mean_field), &mut rng))
    });

    group.throughput(Throughput::Elements(128));
    let chain = Arc::new(Chain::periodic([128]));
    let mut cp = contact_process(4.0, 0.5);
    let mut state = LatticeState::uniform(chain, Binary::Active);
    group.bench_function("contact_process_chain_L128_diffusion", |b| {
        b.iter(|| {
            if cp.is_frozen(&state) {
                state.fill(Binary::Active);
            }
            cp.step(black_box(&mut state), &mut rng)
        })
    });
    group.finish();
}

fn spectrum(c: &mut Criterion) {
    let mut rng = stream(1, &[]);
    let x = Array2::from_shape_fn((100, 301), |_| rng.random::<f64>());
    c.bench_function("correlation_spectrum_100x301", |b| {
        b.iter(|| correlation_spectrum(black_box(x.view()), 0, ZeroVariance::Zero).unwrap())
    });
}

criterion_group!(benches, spin_sweeps, spectrum);
criterion_main!(benches);
