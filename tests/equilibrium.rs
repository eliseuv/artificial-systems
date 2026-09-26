//! Statistical correctness of the dynamics: sampled distributions against exact Boltzmann
//! distributions on systems small enough to enumerate, and the 2D Ising model against Onsager.

use std::sync::Arc;

use artificial_systems::{
    dynamics::{Dynamics, HeatBath, Metropolis, SiteOrder},
    model::{Beg, Clock, MeanFieldModel, Model, Potts},
    rng::stream,
    site::{ClockState, PottsState, Site, SpinHalf, SpinOne},
    state::{Configuration, Init, LatticeState, MeanFieldState, Prepare},
    system::SpinSystem,
    topology::{Graph, Square, Topology},
};

/// Total variation distance between two distributions.
fn total_variation(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.0
}

fn encode<S: Site>(sites: &[S]) -> usize {
    sites
        .iter()
        .rev()
        .fold(0, |acc, s| acc * S::COUNT + s.index())
}

fn decode<S: Site>(mut code: usize, n: usize) -> Vec<S> {
    (0..n)
        .map(|_| {
            let s = S::from_index(code % S::COUNT);
            code /= S::COUNT;
            s
        })
        .collect()
}

/// Exact Boltzmann distribution over all configurations of a small lattice system.
fn exact_lattice<S, T, M>(topology: &Arc<T>, model: &M, beta: f64) -> Vec<f64>
where
    S: Site,
    T: Topology,
    M: Model<LatticeState<S, T>>,
{
    let n = topology.len();
    let states = S::COUNT.pow(n as u32);
    let energies: Vec<f64> = (0..states)
        .map(|c| model.energy(&LatticeState::from_sites(topology.clone(), decode(c, n))))
        .collect();
    let e_min = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let weights: Vec<f64> = energies
        .iter()
        .map(|e| (-beta * (e - e_min)).exp())
        .collect();
    let z: f64 = weights.iter().sum();
    weights.into_iter().map(|w| w / z).collect()
}

/// Empirical distribution of configurations along a chain.
fn sample_lattice<S, T, M, D>(
    topology: &Arc<T>,
    model: M,
    mut dynamics: D,
    n_steps: usize,
    seed: u64,
) -> Vec<f64>
where
    S: Site,
    T: Topology,
    M: Model<LatticeState<S, T>>,
    D: Dynamics<SpinSystem<LatticeState<S, T>, M>>,
{
    let mut rng = stream(seed, &[]);
    let mut state = LatticeState::uniform(topology.clone(), S::VALUES[0]);
    Init::IidUniform.prepare(&mut state, &mut rng);
    let mut sys = SpinSystem::new(state, model);
    let mut hist = vec![0.0; S::COUNT.pow(topology.len() as u32)];
    for _ in 0..1000 {
        dynamics.step(&mut sys, &mut rng);
    }
    for _ in 0..n_steps {
        dynamics.step(&mut sys, &mut rng);
        hist[encode(sys.state().sites())] += 1.0;
    }
    let tracked = sys.energy();
    sys.refresh_energy();
    assert!(
        (tracked - sys.energy()).abs() < 1e-8,
        "tracked energy {tracked} drifted from {}",
        sys.energy()
    );
    hist.iter().map(|h| h / n_steps as f64).collect()
}

fn check_lattice<S, T, M>(topology: Arc<T>, model: M, beta: f64, order: SiteOrder)
where
    S: Site,
    T: Topology,
    M: Model<LatticeState<S, T>>,
    Metropolis: Dynamics<SpinSystem<LatticeState<S, T>, M>>,
    HeatBath: Dynamics<SpinSystem<LatticeState<S, T>, M>>,
{
    let exact = exact_lattice(&topology, &model, beta);
    let steps = 60 * exact.len().max(1000);
    let metropolis = sample_lattice(
        &topology,
        model.clone(),
        Metropolis::with_order(beta, order),
        steps,
        1,
    );
    let heat_bath = sample_lattice(
        &topology,
        model.clone(),
        HeatBath::with_order(beta, order),
        steps,
        2,
    );
    let tol = 0.03;
    let (dm, dh) = (
        total_variation(&metropolis, &exact),
        total_variation(&heat_bath, &exact),
    );
    assert!(
        dm < tol,
        "Metropolis TV distance {dm} for {model:?} at β={beta}"
    );
    assert!(
        dh < tol,
        "Heat bath TV distance {dh} for {model:?} at β={beta}"
    );
}

fn ring(n: usize) -> Arc<Graph> {
    Arc::new(Graph::from_edges(n, (0..n).map(|i| (i, (i + 1) % n))))
}

#[test]
fn ising_matches_boltzmann() {
    for beta in [0.0, 0.4, 1.0] {
        check_lattice::<SpinHalf, _, _>(ring(5), Beg::ising(1.0, 0.3), beta, SiteOrder::Random);
    }
    // Antiferromagnet on a frustrated triangle plus a pendant site
    let frustrated = Arc::new(Graph::from_edges(4, [(0, 1), (1, 2), (2, 0), (2, 3)]));
    check_lattice::<SpinHalf, _, _>(frustrated, Beg::ising(-1.0, 0.1), 0.7, SiteOrder::Random);
}

#[test]
fn site_orders_preserve_boltzmann() {
    let square = Arc::new(Square::periodic([2, 2]));
    for order in [
        SiteOrder::Sequential,
        SiteOrder::Permutation,
        SiteOrder::Checkerboard,
    ] {
        check_lattice::<SpinHalf, _, _>(square.clone(), Beg::ising(1.0, 0.2), 0.5, order);
    }
}

#[test]
fn blume_capel_and_beg_match_boltzmann() {
    check_lattice::<SpinOne, _, _>(
        ring(4),
        Beg::blume_capel(1.0, 0.5, 0.2),
        0.8,
        SiteOrder::Random,
    );
    let beg = Beg {
        j: 1.0,
        k: -0.5,
        h3: 0.3,
        d: 0.4,
        h: -0.2,
    };
    check_lattice::<SpinOne, _, _>(ring(4), beg, 0.6, SiteOrder::Random);
}

#[test]
fn potts_and_clock_match_boltzmann() {
    check_lattice::<PottsState<3>, _, _>(ring(4), Potts::new(1.0, 0.3), 0.9, SiteOrder::Random);
    check_lattice::<ClockState<4>, _, _>(ring(4), Clock::new(4, 1.0, 0.2), 0.7, SiteOrder::Random);
}

/// Mean-field: probability of each composition is multinomial degeneracy times Boltzmann.
fn check_mean_field<S, M>(n: u32, model: M, beta: f64)
where
    S: Site,
    M: MeanFieldModel<S>,
    Metropolis: Dynamics<SpinSystem<MeanFieldState<S>, M>>,
    HeatBath: Dynamics<SpinSystem<MeanFieldState<S>, M>>,
{
    assert!(S::COUNT <= 3);
    let compositions: Vec<Vec<u32>> = (0..=n)
        .flat_map(|a| {
            (0..=n - a).map(move |b| match S::COUNT {
                2 => vec![a, n - a],
                _ => vec![a, b, n - a - b],
            })
        })
        .filter(|c| c.iter().sum::<u32>() == n)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let ln_fact = |k: u32| (1..=k).map(|x| (x as f64).ln()).sum::<f64>();
    let log_w: Vec<f64> = compositions
        .iter()
        .map(|c| {
            let st = MeanFieldState::<S>::from_counts(c.clone(), 4.0);
            ln_fact(n)
                - c.iter().map(|&k| ln_fact(k)).sum::<f64>()
                - beta * model.mean_field_energy(&st)
        })
        .collect();
    let max = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = log_w.iter().map(|l| (l - max).exp()).sum();
    let exact: Vec<f64> = log_w.iter().map(|l| (l - max).exp() / z).collect();

    fn run<S: Site, M: MeanFieldModel<S>, D: Dynamics<SpinSystem<MeanFieldState<S>, M>>>(
        n: u32,
        model: M,
        mut dynamics: D,
        compositions: &[Vec<u32>],
        seed: u64,
    ) -> Vec<f64> {
        let mut rng = stream(seed, &[]);
        let mut sys = SpinSystem::new(
            MeanFieldState::uniform(n as usize, 4.0, S::VALUES[0]),
            model,
        );
        let steps = 200_000;
        let mut hist = vec![0.0; compositions.len()];
        for _ in 0..steps {
            dynamics.step(&mut sys, &mut rng);
            let k = compositions
                .iter()
                .position(|c| c.as_slice() == sys.counts())
                .unwrap();
            hist[k] += 1.0 / steps as f64;
        }
        hist
    }
    let dm = total_variation(
        &run(n, model.clone(), Metropolis::new(beta), &compositions, 3),
        &exact,
    );
    let dh = total_variation(
        &run(n, model.clone(), HeatBath::new(beta), &compositions, 4),
        &exact,
    );
    assert!(dm < 0.02, "Metropolis TV distance {dm} for {model:?}");
    assert!(dh < 0.02, "Heat bath TV distance {dh} for {model:?}");
}

#[test]
fn mean_field_matches_boltzmann() {
    check_mean_field::<SpinHalf, _>(8, Beg::ising(1.0, 0.1), 0.3);
    check_mean_field::<SpinOne, _>(6, Beg::blume_capel(1.0, 0.4, 0.0), 0.5);
    check_mean_field::<PottsState<3>, _>(6, Potts::new(1.0, 0.2), 0.6);
}

/// Complete elliptic integral of the first kind `K(k)` through the arithmetic-geometric mean.
fn elliptic_k(k: f64) -> f64 {
    let (mut a, mut b) = (1.0, (1.0 - k * k).sqrt());
    while (a - b).abs() > 1e-15 {
        (a, b) = ((a + b) / 2.0, (a * b).sqrt());
    }
    std::f64::consts::PI / (2.0 * a)
}

/// Onsager energy per spin of the infinite square lattice Ising model (J = 1).
fn onsager_energy(beta: f64) -> f64 {
    let t = (2.0 * beta).tanh();
    let k = 2.0 * (2.0 * beta).sinh() / (2.0 * beta).cosh().powi(2);
    -(1.0 / t) * (1.0 + (2.0 / std::f64::consts::PI) * (2.0 * t * t - 1.0) * elliptic_k(k))
}

#[test]
fn square_ising_matches_onsager() {
    let topology = Arc::new(Square::periodic_cube(32));
    let n = topology.len() as f64;
    for (temperature, ordered) in [(1.5, true), (3.5, false)] {
        let beta = 1.0 / temperature;
        let fill = if ordered {
            SpinHalf::Up
        } else {
            SpinHalf::Down
        };
        let mut sys = SpinSystem::new(
            LatticeState::uniform(topology.clone(), fill),
            Beg::ising(1.0, 0.0),
        );
        let mut dynamics = HeatBath::new(beta);
        let mut rng = stream(7, &[]);
        for _ in 0..500 {
            dynamics.step(&mut sys, &mut rng);
        }
        let (mut e, mut m, steps) = (0.0, 0.0, 3000);
        for _ in 0..steps {
            dynamics.step(&mut sys, &mut rng);
            e += sys.energy() / n / steps as f64;
            m += (sys.state().total_magnetization() as f64 / n).abs() / steps as f64;
        }
        let exact = onsager_energy(beta);
        assert!(
            (e - exact).abs() < 0.01 * exact.abs(),
            "T = {temperature}: e = {e}, Onsager {exact}"
        );
        if ordered {
            let yang = (1.0 - (2.0 * beta).sinh().powi(-4)).powf(0.125);
            assert!(
                (m - yang).abs() < 0.005,
                "T = {temperature}: m = {m}, Yang {yang}"
            );
        }
    }
}
