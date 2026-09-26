//! Heat bath (Glauber) single site dynamics.

use rand::{Rng, RngExt as _};

use super::{Dynamics, SiteOrder, Sweep, beta_from_temperature};
use crate::{
    model::{Kernel, LocalModel, MeanFieldModel, boltzmann_weight, check_beta},
    site::Site,
    state::{Configuration, LatticeState, MeanFieldState},
    system::SpinSystem,
    topology::Topology,
};

/// Heat bath dynamics: the new state of a site is drawn from its local Boltzmann distribution,
/// independently of its current state.
///
/// For two-state sites this is Glauber dynamics, flipping with probability
/// `1 / (1 + exp(β ΔE))`.
#[derive(Debug, Clone)]
pub struct HeatBath {
    beta: f64,
    sweep: Sweep,
    /// Scratch space for mean-field energy changes and weights.
    scratch: (Vec<f64>, Vec<f64>),
}

/// Glauber dynamics, i.e. heat bath dynamics of two-state sites.
pub type Glauber = HeatBath;

impl HeatBath {
    /// Random sequential heat bath dynamics at inverse temperature `beta` (`∞` allowed).
    pub fn new(beta: f64) -> Self {
        Self::with_order(beta, SiteOrder::Random)
    }

    /// Heat bath dynamics at inverse temperature `beta` visiting sites in `order`.
    pub fn with_order(beta: f64, order: SiteOrder) -> Self {
        check_beta(beta);
        Self {
            beta,
            sweep: Sweep::new(order),
            scratch: (Vec::new(), Vec::new()),
        }
    }

    /// Random sequential heat bath dynamics at temperature `temperature` (`0` allowed).
    pub fn at_temperature(temperature: f64) -> Self {
        Self::new(beta_from_temperature(temperature))
    }

    /// Inverse temperature.
    pub fn beta(&self) -> f64 {
        self.beta
    }
}

impl<S, T, M> Dynamics<SpinSystem<LatticeState<S, T>, M>> for HeatBath
where
    S: Site,
    T: Topology,
    M: LocalModel<S>,
{
    #[inline]
    fn step<R: Rng + ?Sized>(&mut self, sys: &mut SpinSystem<LatticeState<S, T>, M>, rng: &mut R) {
        let (state, model, kernel, energy) = sys.parts_mut(self.beta);
        let topology = state.topology().clone();
        self.sweep.run(&*topology, rng, |i, rng| {
            let field = model.field(state.sites(), topology.neighbors(i));
            let old = state.get(i);
            let new = kernel.heat_bath(&field, rng.random());
            if new != old {
                *energy += kernel.delta_energy(old, new, &field);
                state.set(i, new);
            }
        });
    }
}

impl<S, M> Dynamics<SpinSystem<MeanFieldState<S>, M>> for HeatBath
where
    S: Site,
    M: MeanFieldModel<S>,
{
    #[inline]
    fn step<R: Rng + ?Sized>(&mut self, sys: &mut SpinSystem<MeanFieldState<S>, M>, rng: &mut R) {
        let (state, model, kernel, energy) = sys.parts_mut(self.beta);
        let (deltas, weights) = &mut self.scratch;
        deltas.resize(S::COUNT, 0.0);
        weights.resize(S::COUNT, 0.0);
        for _ in 0..state.len() {
            let old = state.random_site(rng);
            model.mean_field_deltas(state, old, deltas);
            let d_min = deltas.iter().copied().fold(f64::INFINITY, f64::min);
            let mut total = 0.0;
            for (d, w) in deltas.iter().zip(weights.iter_mut()) {
                *w = boltzmann_weight(kernel.beta, d - d_min);
                total += *w;
            }
            let k = sample_weights(weights, rng.random::<f64>() * total);
            if k != old.index() {
                *energy += deltas[k];
                state.transfer(old, S::from_index(k));
            }
        }
    }
}

/// Index of the bin of `target ∈ [0, Σw)` in the cumulative sum of `weights`.
#[inline(always)]
fn sample_weights(weights: &[f64], target: f64) -> usize {
    let mut acc = 0.0;
    for (k, w) in weights.iter().enumerate() {
        acc += w;
        if target < acc {
            return k;
        }
    }
    weights
        .iter()
        .rposition(|&w| w > 0.0)
        .expect("some state has positive weight")
}
