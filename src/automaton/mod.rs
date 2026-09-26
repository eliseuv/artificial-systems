//! Stochastic cellular automata.
//!
//! A [`LocalRule`] gives the new value of a site from its current value and its neighbourhood.
//! It can be applied
//! - [`Synchronous`]ly: every site at once from the previous configuration (double buffered), or
//! - [`Asynchronous`]ly: `N` in-place single site updates per step, in a [`SiteOrder`].
//!
//! [`Diffusion`] exchanges neighbouring sites and [`Compose`] chains dynamics, so e.g. the contact
//! process with diffusion is `Compose(Asynchronous(ContactRule), Diffusion)` (see
//! [`contact_process`]).

use rand::Rng;
use rand_distr::{Binomial, Distribution as _};

use crate::{
    dynamics::{Dynamics, SiteOrder, Sweep},
    rng::random_index,
    site::Site,
    state::{Configuration, LatticeState},
    topology::Topology,
};

mod rules;

pub use rules::{BrassRule, ContactRule, Elementary, TotalisticBinary};

/// Stochastic local update rule.
pub trait LocalRule<S: Site>: Clone + Send + Sync {
    /// New value of a site currently in state `current` with neighbours `neighbors`.
    fn apply<R: Rng + ?Sized>(&self, current: S, sites: &[S], neighbors: &[u32], rng: &mut R) -> S;

    /// Whether a configuration with these per value counts can never change under the rule.
    fn is_absorbing(&self, _counts: &[u32]) -> bool {
        false
    }
}

/// Parallel update of every site from the previous configuration.
#[derive(Debug, Clone)]
pub struct Synchronous<Rule, S> {
    /// Update rule.
    pub rule: Rule,
    buffer: Vec<S>,
}

impl<Rule, S> Synchronous<Rule, S> {
    /// Synchronous application of `rule`.
    pub fn new(rule: Rule) -> Self {
        Self {
            rule,
            buffer: Vec::new(),
        }
    }
}

impl<S, T, Rule> Dynamics<LatticeState<S, T>> for Synchronous<Rule, S>
where
    S: Site,
    T: Topology,
    Rule: LocalRule<S>,
{
    fn step<R: Rng + ?Sized>(&mut self, state: &mut LatticeState<S, T>, rng: &mut R) {
        let topology = state.topology().clone();
        let sites = state.sites();
        self.buffer.clear();
        self.buffer.extend(
            sites
                .iter()
                .enumerate()
                .map(|(i, &s)| self.rule.apply(s, sites, topology.neighbors(i), rng)),
        );
        state.swap_buffer(&mut self.buffer);
    }

    fn is_frozen(&self, state: &LatticeState<S, T>) -> bool {
        self.rule.is_absorbing(state.counts())
    }
}

/// Sequence of `N` in-place single site updates.
#[derive(Debug, Clone)]
pub struct Asynchronous<Rule> {
    /// Update rule.
    pub rule: Rule,
    sweep: Sweep,
}

impl<Rule> Asynchronous<Rule> {
    /// Random sequential application of `rule`.
    pub fn new(rule: Rule) -> Self {
        Self::with_order(rule, SiteOrder::Random)
    }

    /// Asynchronous application of `rule` visiting sites in `order`.
    pub fn with_order(rule: Rule, order: SiteOrder) -> Self {
        Self {
            rule,
            sweep: Sweep::new(order),
        }
    }
}

impl<S, T, Rule> Dynamics<LatticeState<S, T>> for Asynchronous<Rule>
where
    S: Site,
    T: Topology,
    Rule: LocalRule<S>,
{
    #[inline]
    fn step<R: Rng + ?Sized>(&mut self, state: &mut LatticeState<S, T>, rng: &mut R) {
        let topology = state.topology().clone();
        self.sweep.run(&*topology, rng, |i, rng| {
            let old = state.get(i);
            let new = self
                .rule
                .apply(old, state.sites(), topology.neighbors(i), rng);
            if new != old {
                state.set(i, new);
            }
        });
    }

    fn is_frozen(&self, state: &LatticeState<S, T>) -> bool {
        self.rule.is_absorbing(state.counts())
    }
}

/// Exchange of neighbouring sites: `N` attempts per step, each swapping, with probability
/// `gamma`, a uniformly random site with a uniformly random neighbour of it.
///
/// Failed attempts do nothing, so each step performs a `Binomial(N, γ)` number of swaps, which is
/// how it is simulated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Diffusion {
    gamma: f64,
    /// Swap count distribution for the last system size.
    swaps: Option<(usize, Binomial)>,
}

impl Diffusion {
    /// Diffusion with swap probability `gamma` per attempt.
    ///
    /// # Panics
    /// If `gamma` is not in `[0, 1]`.
    pub fn new(gamma: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&gamma),
            "Diffusion probability must be in [0, 1], got {gamma}"
        );
        Self { gamma, swaps: None }
    }

    /// Swap probability per attempt.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

impl<S: Site, T: Topology> Dynamics<LatticeState<S, T>> for Diffusion {
    #[inline]
    fn step<R: Rng + ?Sized>(&mut self, state: &mut LatticeState<S, T>, rng: &mut R) {
        if self.gamma == 0.0 {
            return;
        }
        let n = state.len();
        let swaps = match self.swaps {
            Some((len, dist)) if len == n => dist,
            _ => {
                let dist = Binomial::new(n as u64, self.gamma).expect("valid probability");
                self.swaps = Some((n, dist));
                dist
            }
        };
        let topology = state.topology().clone();
        for _ in 0..swaps.sample(rng) {
            let i = random_index(rng, n);
            let neighbors = topology.neighbors(i);
            if !neighbors.is_empty() {
                let j = neighbors[random_index(rng, neighbors.len())];
                state.swap(i, j as usize);
            }
        }
    }

    /// Frozen when every site is in the same state (or there is no diffusion).
    fn is_frozen(&self, state: &LatticeState<S, T>) -> bool {
        self.gamma == 0.0 || state.counts().iter().filter(|&&n| n > 0).count() <= 1
    }
}

/// Dynamics `A` followed by dynamics `B` within each step.
#[derive(Debug, Clone)]
pub struct Compose<A, B>(pub A, pub B);

impl<Sys, A: Dynamics<Sys>, B: Dynamics<Sys>> Dynamics<Sys> for Compose<A, B> {
    #[inline]
    fn step<R: Rng + ?Sized>(&mut self, sys: &mut Sys, rng: &mut R) {
        self.0.step(sys, rng);
        self.1.step(sys, rng);
    }

    fn is_frozen(&self, sys: &Sys) -> bool {
        self.0.is_frozen(sys) && self.1.is_frozen(sys)
    }
}

/// Contact process with diffusion.
pub type ContactProcess = Compose<Asynchronous<ContactRule>, Diffusion>;

/// Contact process with infection rate `alpha` (per active site, spread over its neighbours;
/// `∞` disables recovery) and diffusion probability `gamma`.
///
/// Each step makes `N` random sequential updates: an active site becomes inactive with
/// probability `1/α`, an inactive site copies the state of a uniformly random neighbour. Then `N`
/// diffusion attempts follow (see [`Diffusion`]). An inactive site with `k` of its `z`
/// neighbours active therefore activates at rate `α k/z` relative to recoveries, which is the
/// standard contact process with `λ = α`.
pub fn contact_process(alpha: f64, gamma: f64) -> ContactProcess {
    Compose(
        Asynchronous::new(ContactRule::new(alpha)),
        Diffusion::new(gamma),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        rng::stream,
        site::Binary,
        state::{Init, Prepare},
        topology::Chain,
    };

    fn chain(n: usize, fill: Binary) -> LatticeState<Binary, Chain> {
        LatticeState::uniform(Arc::new(Chain::periodic([n])), fill)
    }

    fn density(st: &LatticeState<Binary, Chain>) -> f64 {
        st.count(Binary::Active) as f64 / st.len() as f64
    }

    #[test]
    fn contact_process_phases() {
        let mut rng = stream(0, &[]);
        // Absorbing phase
        let mut st = chain(512, Binary::Active);
        let mut cp = contact_process(2.5, 0.0);
        let mut t = 0;
        while !cp.is_frozen(&st) {
            cp.step(&mut st, &mut rng);
            t += 1;
            assert!(t < 5000, "Subcritical contact process should die out");
        }
        assert_eq!(density(&st), 0.0);
        // Active phase
        let mut st = chain(512, Binary::Active);
        let mut cp = contact_process(4.0, 0.5);
        for _ in 0..1000 {
            cp.step(&mut st, &mut rng);
        }
        assert!(density(&st) > 0.2, "density = {}", density(&st));
        assert!(!cp.is_frozen(&st));
    }

    #[test]
    fn diffusion_conserves_and_mixes() {
        let mut rng = stream(1, &[]);
        let mut st = chain(1000, Binary::Inactive);
        Init::Exact(vec![500, 500]).prepare(&mut st, &mut rng);
        let before = st.sites().to_vec();
        let mut diffusion = Diffusion::new(1.0);
        diffusion.step(&mut st, &mut rng);
        assert_eq!(st.count(Binary::Active), 500);
        assert_ne!(st.sites(), before.as_slice());
        let mut none = Diffusion::new(0.0);
        let before = st.sites().to_vec();
        none.step(&mut st, &mut rng);
        assert_eq!(st.sites(), before.as_slice());
    }

    #[test]
    fn uniform_states_freeze_diffusion() {
        let st = chain(10, Binary::Active);
        assert!(Diffusion::new(0.5).is_frozen(&st));
        assert!(!contact_process(3.0, 0.5).is_frozen(&st));
        assert!(contact_process(3.0, 0.5).is_frozen(&chain(10, Binary::Inactive)));
    }
}
