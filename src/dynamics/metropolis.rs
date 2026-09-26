//! Metropolis single site dynamics.

use rand::{Rng, RngExt as _};

use super::{Dynamics, SiteOrder, Sweep, beta_from_temperature, propose_other};
use crate::{
    model::{Kernel, LocalModel, MeanFieldModel, check_beta, metropolis_acceptance},
    site::Site,
    state::{Configuration, LatticeState, MeanFieldState},
    system::SpinSystem,
    topology::Topology,
};

/// Metropolis dynamics: propose a uniformly random different state for a site and accept with
/// probability `min(1, exp(-β ΔE))`. Two-state sites are always proposed a flip.
#[derive(Debug, Clone)]
pub struct Metropolis {
    beta: f64,
    sweep: Sweep,
}

impl Metropolis {
    /// Random sequential Metropolis dynamics at inverse temperature `beta` (`∞` allowed).
    pub fn new(beta: f64) -> Self {
        Self::with_order(beta, SiteOrder::Random)
    }

    /// Metropolis dynamics at inverse temperature `beta` visiting sites in `order`.
    pub fn with_order(beta: f64, order: SiteOrder) -> Self {
        check_beta(beta);
        Self {
            beta,
            sweep: Sweep::new(order),
        }
    }

    /// Random sequential Metropolis dynamics at temperature `temperature` (`0` allowed).
    pub fn at_temperature(temperature: f64) -> Self {
        Self::new(beta_from_temperature(temperature))
    }

    /// Inverse temperature.
    pub fn beta(&self) -> f64 {
        self.beta
    }
}

impl<S, T, M> Dynamics<SpinSystem<LatticeState<S, T>, M>> for Metropolis
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
            let new = propose_other(old, rng);
            let a = kernel.acceptance(old, new, &field);
            if a >= 1.0 || rng.random::<f64>() < a {
                *energy += kernel.delta_energy(old, new, &field);
                state.set(i, new);
            }
        });
    }
}

impl<S, M> Dynamics<SpinSystem<MeanFieldState<S>, M>> for Metropolis
where
    S: Site,
    M: MeanFieldModel<S>,
{
    #[inline]
    fn step<R: Rng + ?Sized>(&mut self, sys: &mut SpinSystem<MeanFieldState<S>, M>, rng: &mut R) {
        let (state, model, kernel, energy) = sys.parts_mut(self.beta);
        for _ in 0..state.len() {
            let old = state.random_site(rng);
            let new = propose_other(old, rng);
            let delta = model.mean_field_delta(state, old, new);
            let a = metropolis_acceptance(kernel.beta, delta);
            if a >= 1.0 || rng.random::<f64>() < a {
                *energy += delta;
                state.transfer(old, new);
            }
        }
    }
}
