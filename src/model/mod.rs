//! Spin models (Hamiltonians).
//!
//! Lattice models are pairwise interactions plus an on-site term,
//! `H = Σ_⟨ij⟩ b(sᵢ, sⱼ) + Σᵢ u(sᵢ)`, described by [`LocalModel`]. All the information a single
//! site update needs about the neighbourhood of a site is summarised in a small *local field*
//! (e.g. `Σⱼ sⱼ`). For a given inverse temperature a model builds a [`Kernel`], typically a table
//! of Boltzmann factors indexed by the local field, so the hot loop of the dynamics avoids `exp`.
//!
//! Mean-field models ([`MeanFieldModel`]) express the energy through the number of sites in each
//! state, with lattice sums `Σ_⟨ij⟩` replaced by `(z / N) Σ_{i<j}`.

use std::{fmt::Debug, marker::PhantomData};

use crate::{
    site::Site,
    state::{LatticeState, MeanFieldState},
    topology::Topology,
};

mod beg;
mod clock;
mod potts;

pub use beg::{Beg, BegKernel};
pub use clock::Clock;
pub use potts::{Potts, PottsKernel};

/// Pairwise plus on-site model on an explicit topology.
pub trait LocalModel<S: Site>: Clone + PartialEq + Debug + Send + Sync + 'static {
    /// Summary of the neighbourhood of a site sufficient to compute its energy.
    type Field: Copy;

    /// Precomputed single site update probabilities for a given inverse temperature.
    type Kernel: Kernel<S, Self::Field>;

    /// Local field of a site from the values of its neighbours.
    fn field(&self, sites: &[S], neighbors: &[u32]) -> Self::Field;

    /// Energy `b(a, b)` of a bond between sites in states `a` and `b` (symmetric).
    fn bond_energy(&self, a: S, b: S) -> f64;

    /// On-site energy `u(s)`.
    fn onsite_energy(&self, s: S) -> f64;

    /// Energy of every term involving a site in state `s` with local field `field`:
    /// `u(s) + Σⱼ b(s, sⱼ)`.
    fn site_energy(&self, s: S, field: &Self::Field) -> f64;

    /// Kernel for inverse temperature `beta` on a topology whose sites have at most
    /// `max_degree` neighbours.
    fn kernel(&self, beta: f64, max_degree: usize) -> Self::Kernel;
}

/// Single site update probabilities at fixed inverse temperature.
pub trait Kernel<S: Site, F>: Clone + Debug + Send + Sync {
    /// Energy change of a site with local field `field` going from `old` to `new`.
    fn delta_energy(&self, old: S, new: S, field: &F) -> f64;

    /// Metropolis acceptance probability `min(1, exp(-β ΔE))`.
    fn acceptance(&self, old: S, new: S, field: &F) -> f64;

    /// New state drawn from the local Boltzmann distribution, given `u` uniform in `[0, 1)`.
    fn heat_bath(&self, field: &F, u: f64) -> S;
}

/// Kernel evaluating Boltzmann factors on the fly from [`LocalModel::site_energy`].
#[derive(Debug, Clone)]
pub struct DirectKernel<M, S> {
    model: M,
    beta: f64,
    _site: PhantomData<fn() -> S>,
}

impl<M, S> DirectKernel<M, S> {
    /// Direct kernel for `model` at inverse temperature `beta`.
    pub fn new(model: M, beta: f64) -> Self {
        check_beta(beta);
        Self {
            model,
            beta,
            _site: PhantomData,
        }
    }
}

impl<S: Site, M: LocalModel<S>> Kernel<S, M::Field> for DirectKernel<M, S> {
    #[inline]
    fn delta_energy(&self, old: S, new: S, field: &M::Field) -> f64 {
        self.model.site_energy(new, field) - self.model.site_energy(old, field)
    }

    #[inline]
    fn acceptance(&self, old: S, new: S, field: &M::Field) -> f64 {
        metropolis_acceptance(self.beta, self.delta_energy(old, new, field))
    }

    #[inline]
    fn heat_bath(&self, field: &M::Field, u: f64) -> S {
        // Three passes avoid scratch storage: minimum energy, normalisation, then inversion
        let energy = |s: &S| self.model.site_energy(*s, field);
        let e_min = S::VALUES.iter().map(energy).fold(f64::INFINITY, f64::min);
        let weight = |s: &S| boltzmann_weight(self.beta, energy(s) - e_min);
        let total: f64 = S::VALUES.iter().map(weight).sum();
        let target = u * total;
        let mut acc = 0.0;
        for s in S::VALUES {
            acc += weight(s);
            if target < acc {
                return *s;
            }
        }
        *S::VALUES
            .iter()
            .rev()
            .find(|s| weight(s) > 0.0)
            .expect("some state has positive weight")
    }
}

/// Model on fully connected systems, as a function of the number of sites in each state.
pub trait MeanFieldModel<S: Site>: Clone + PartialEq + Debug + Send + Sync + 'static {
    /// Total energy.
    fn mean_field_energy(&self, state: &MeanFieldState<S>) -> f64;

    /// Energy change when one site goes from state `from` to state `to`.
    fn mean_field_delta(&self, state: &MeanFieldState<S>, from: S, to: S) -> f64;

    /// Energy changes when one site goes from state `from` to each state, indexed by
    /// [`Site::index`] (zero for `from` itself).
    fn mean_field_deltas(&self, state: &MeanFieldState<S>, from: S, out: &mut [f64]) {
        for (&to, d) in S::VALUES.iter().zip(out.iter_mut()) {
            *d = self.mean_field_delta(state, from, to);
        }
    }
}

/// Energy function and update kernel of a model on a given kind of configuration.
///
/// Implemented automatically for [`LocalModel`]s on [`LatticeState`]s and [`MeanFieldModel`]s on
/// [`MeanFieldState`]s.
pub trait Model<St>: Clone + PartialEq + Debug + Send + Sync + 'static {
    /// Precomputed update data for a given inverse temperature.
    type Kernel: Clone + Debug + Send + Sync;

    /// Total energy of `state`.
    fn energy(&self, state: &St) -> f64;

    /// Kernel for inverse temperature `beta` on `state`'s topology.
    fn kernel(&self, state: &St, beta: f64) -> Self::Kernel;
}

impl<S: Site, T: Topology, M: LocalModel<S>> Model<LatticeState<S, T>> for M {
    type Kernel = M::Kernel;

    fn energy(&self, state: &LatticeState<S, T>) -> f64 {
        let sites = state.sites();
        let bonds: f64 = state
            .topology()
            .bonds()
            .map(|(i, j)| self.bond_energy(sites[i], sites[j]))
            .sum();
        let onsite: f64 = sites.iter().map(|&s| self.onsite_energy(s)).sum();
        bonds + onsite
    }

    fn kernel(&self, state: &LatticeState<S, T>, beta: f64) -> Self::Kernel {
        LocalModel::kernel(self, beta, state.topology().max_degree())
    }
}

/// Kernel of mean-field models, whose local field is continuous and evaluated directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeanFieldKernel {
    /// Inverse temperature.
    pub beta: f64,
}

impl<S: Site, M: MeanFieldModel<S>> Model<MeanFieldState<S>> for M {
    type Kernel = MeanFieldKernel;

    fn energy(&self, state: &MeanFieldState<S>) -> f64 {
        self.mean_field_energy(state)
    }

    fn kernel(&self, _state: &MeanFieldState<S>, beta: f64) -> Self::Kernel {
        check_beta(beta);
        MeanFieldKernel { beta }
    }
}

/// # Panics
/// If `beta` is negative or NaN (`+∞`, i.e. zero temperature, is allowed).
#[inline]
pub(crate) fn check_beta(beta: f64) {
    assert!(
        beta >= 0.0,
        "Inverse temperature must be non-negative, got {beta}"
    );
}

/// `min(1, exp(-β ΔE))`, well defined for `β = 0` and `β = ∞`.
#[inline]
pub(crate) fn metropolis_acceptance(beta: f64, delta: f64) -> f64 {
    if delta <= 0.0 {
        1.0
    } else {
        boltzmann_weight(beta, delta)
    }
}

/// `exp(-β ΔE)` for `ΔE >= 0`, with `0 · ∞ = 0` (so `β = ∞` gives `1` only for `ΔE = 0`).
#[inline]
pub(crate) fn boltzmann_weight(beta: f64, delta: f64) -> f64 {
    if delta == 0.0 {
        1.0
    } else {
        (-beta * delta).exp()
    }
}

/// Cumulative Boltzmann distribution over `energies`, written to `out` (same length).
///
/// Energies are shifted by their minimum, so no weight overflows and the minimum energy state
/// always has weight one; the last entry is exactly `1`.
pub(crate) fn boltzmann_cumulative(beta: f64, energies: &[f64], out: &mut [f64]) {
    debug_assert_eq!(energies.len(), out.len());
    let e_min = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let mut acc = 0.0;
    let mut last_positive = 0;
    for (k, (e, c)) in energies.iter().zip(out.iter_mut()).enumerate() {
        let w = boltzmann_weight(beta, e - e_min);
        if w > 0.0 {
            last_positive = k;
        }
        acc += w;
        *c = acc;
    }
    for c in out.iter_mut() {
        *c /= acc;
    }
    // Exact upper bound so every `u < 1` maps to a state with positive weight
    for c in &mut out[last_positive..] {
        *c = 1.0;
    }
}

/// Index `k` of the first cumulative probability exceeding `u`.
#[inline(always)]
pub(crate) fn sample_cumulative(cumulative: &[f64], u: f64) -> usize {
    cumulative
        .iter()
        .position(|&c| u < c)
        .unwrap_or(cumulative.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acceptance_limits() {
        assert_eq!(metropolis_acceptance(f64::INFINITY, 1.0), 0.0);
        assert_eq!(metropolis_acceptance(f64::INFINITY, 0.0), 1.0);
        assert_eq!(metropolis_acceptance(f64::INFINITY, -1.0), 1.0);
        assert_eq!(metropolis_acceptance(0.0, 1e300), 1.0);
        approx::assert_relative_eq!(metropolis_acceptance(2.0, 0.5), (-1.0f64).exp());
    }

    #[test]
    fn cumulative_is_stable() {
        let mut out = [0.0; 3];
        boltzmann_cumulative(1e3, &[-1e3, 0.0, 1e3], &mut out);
        assert_eq!(out, [1.0, 1.0, 1.0]);
        assert_eq!(sample_cumulative(&out, 0.999), 0);

        boltzmann_cumulative(f64::INFINITY, &[1.0, 0.0, 0.0], &mut out);
        assert_eq!(out, [0.0, 0.5, 1.0]);
        assert_eq!(sample_cumulative(&out, 0.0), 1);
        assert_eq!(sample_cumulative(&out, 0.7), 2);

        boltzmann_cumulative(0.0, &[5.0, -3.0, 1.0], &mut out);
        approx::assert_relative_eq!(out[0], 1.0 / 3.0);
        approx::assert_relative_eq!(out[1], 2.0 / 3.0);
        assert_eq!(out[2], 1.0);
    }
}
