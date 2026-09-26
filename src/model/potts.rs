//! `Q`-state Potts model.

use serde::{Deserialize, Serialize};

use super::{
    Kernel, LocalModel, MeanFieldModel, boltzmann_weight, check_beta, metropolis_acceptance,
};
use crate::{
    site::PottsState,
    state::{Configuration, MeanFieldState},
};

/// Potts Hamiltonian `H = -J Σ_⟨ij⟩ δ(sᵢ, sⱼ) - h Σᵢ δ(sᵢ, 0)`.
///
/// The local field of a site is the number of neighbours in each of the `Q` states.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Potts {
    /// Coupling `J` (ferromagnetic for `J > 0`).
    pub j: f64,
    /// Field `h` favouring state `0`.
    pub h: f64,
}

impl Potts {
    /// Potts model with coupling `j` and field `h` on state `0`.
    pub const fn new(j: f64, h: f64) -> Self {
        Self { j, h }
    }

    /// Energy of a site in a state with `n` equal neighbours, `is_zero` telling whether it is
    /// state `0`.
    #[inline(always)]
    fn energy_of(&self, n: i64, is_zero: bool) -> f64 {
        -self.j * n as f64 - if is_zero { self.h } else { 0.0 }
    }
}

impl<const Q: usize> LocalModel<PottsState<Q>> for Potts {
    type Field = [u32; Q];
    type Kernel = PottsKernel<Q>;

    #[inline(always)]
    fn field(&self, sites: &[PottsState<Q>], neighbors: &[u32]) -> Self::Field {
        let mut counts = [0; Q];
        for &j in neighbors {
            counts[sites[j as usize].q()] += 1;
        }
        counts
    }

    #[inline]
    fn bond_energy(&self, a: PottsState<Q>, b: PottsState<Q>) -> f64 {
        if a == b { -self.j } else { 0.0 }
    }

    #[inline]
    fn onsite_energy(&self, s: PottsState<Q>) -> f64 {
        if s.q() == 0 { -self.h } else { 0.0 }
    }

    #[inline(always)]
    fn site_energy(&self, s: PottsState<Q>, field: &Self::Field) -> f64 {
        self.energy_of(field[s.q()] as i64, s.q() == 0)
    }

    fn kernel(&self, beta: f64, max_degree: usize) -> Self::Kernel {
        PottsKernel::new(*self, beta, max_degree)
    }
}

/// Potts kernel with Boltzmann factors tabulated by the difference in number of equal
/// neighbours `Δn ∈ [-z, z]` and in `δ(s, 0)`.
#[derive(Debug, Clone)]
pub struct PottsKernel<const Q: usize> {
    model: Potts,
    z: i64,
    /// `acceptance[(Δn + z) * 3 + Δδ + 1]`
    acceptance: Box<[f64]>,
    /// `weights[(Δn + z) * 3 + Δδ + 1] = exp(-β ΔE)` for `ΔE >= 0`, used relative to the local
    /// minimum energy state
    weights: Box<[f64]>,
}

impl<const Q: usize> PottsKernel<Q> {
    fn new(model: Potts, beta: f64, max_degree: usize) -> Self {
        check_beta(beta);
        let z = max_degree as i64;
        let deltas: Vec<f64> = (-z..=z)
            .flat_map(|dn| (-1..=1).map(move |dd| (dn, dd)))
            .map(|(dn, dd)| -model.j * dn as f64 - model.h * dd as f64)
            .collect();
        Self {
            model,
            z,
            acceptance: deltas
                .iter()
                .map(|&de| metropolis_acceptance(beta, de))
                .collect(),
            // Negative differences never occur relative to the minimum; clamp keeps them finite
            weights: deltas
                .iter()
                .map(|&de| boltzmann_weight(beta, de.max(0.0)))
                .collect(),
        }
    }

    #[inline(always)]
    fn index(&self, dn: i64, dd: i64) -> usize {
        ((dn + self.z) * 3 + dd + 1) as usize
    }
}

impl<const Q: usize> Kernel<PottsState<Q>, [u32; Q]> for PottsKernel<Q> {
    #[inline(always)]
    fn delta_energy(&self, old: PottsState<Q>, new: PottsState<Q>, field: &[u32; Q]) -> f64 {
        self.model.energy_of(field[new.q()] as i64, new.q() == 0)
            - self.model.energy_of(field[old.q()] as i64, old.q() == 0)
    }

    #[inline(always)]
    fn acceptance(&self, old: PottsState<Q>, new: PottsState<Q>, field: &[u32; Q]) -> f64 {
        let dn = field[new.q()] as i64 - field[old.q()] as i64;
        let dd = (new.q() == 0) as i64 - (old.q() == 0) as i64;
        self.acceptance[self.index(dn, dd)]
    }

    #[inline(always)]
    fn heat_bath(&self, field: &[u32; Q], u: f64) -> PottsState<Q> {
        let energy = |q: usize| self.model.energy_of(field[q] as i64, q == 0);
        let q_min = (0..Q)
            .min_by(|&a, &b| energy(a).total_cmp(&energy(b)))
            .expect("Q >= 2");
        let (n_min, d_min) = (field[q_min] as i64, (q_min == 0) as i64);
        let mut weights = [0.0; Q];
        let mut total = 0.0;
        for (q, w) in weights.iter_mut().enumerate() {
            *w = self.weights[self.index(field[q] as i64 - n_min, (q == 0) as i64 - d_min)];
            total += *w;
        }
        let target = u * total;
        let mut acc = 0.0;
        for (q, w) in weights.iter().enumerate() {
            acc += w;
            if target < acc {
                return PottsState::new(q);
            }
        }
        PottsState::new(q_min)
    }
}

impl<const Q: usize> MeanFieldModel<PottsState<Q>> for Potts {
    /// `H = -(Jz/2N) Σ_q nq(nq - 1) - h n₀`
    fn mean_field_energy(&self, state: &MeanFieldState<PottsState<Q>>) -> f64 {
        let pairs: i128 = state
            .counts()
            .iter()
            .map(|&n| n as i128 * (n as i128 - 1))
            .sum();
        let z_over_n = state.coordination() / state.len() as f64;
        -self.j * z_over_n * pairs as f64 / 2.0 - self.h * state.counts()[0] as f64
    }

    fn mean_field_delta(
        &self,
        state: &MeanFieldState<PottsState<Q>>,
        from: PottsState<Q>,
        to: PottsState<Q>,
    ) -> f64 {
        if from == to {
            return 0.0;
        }
        let counts = state.counts();
        let dn = counts[to.q()] as i64 - counts[from.q()] as i64 + 1;
        let dd = (to.q() == 0) as i64 - (from.q() == 0) as i64;
        let z_over_n = state.coordination() / state.len() as f64;
        -self.j * z_over_n * dn as f64 - self.h * dd as f64
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use proptest::prelude::*;

    use super::*;
    use crate::{
        model::{DirectKernel, Model},
        rng::stream,
        site::Site,
        state::{Init, LatticeState, Prepare},
        topology::Square,
    };

    type P3 = PottsState<3>;

    proptest! {
        #[test]
        fn kernel_matches_direct(j in -2.0..2.0, h in -2.0..2.0, beta in 0.0..3.0, seed: u64, i in 0usize..16, new in 0usize..3, u in 0.0..1.0) {
            let model = Potts::new(j, h);
            let mut st = LatticeState::uniform(Arc::new(Square::periodic([4, 4])), P3::new(0));
            Init::IidUniform.prepare(&mut st, &mut stream(seed, &[]));
            let (old, new) = (st.get(i), P3::from_index(new));
            let field = LocalModel::<P3>::field(&model, st.sites(), st.neighbors(i));
            let kernel = Model::kernel(&model, &st, beta);
            let direct = DirectKernel::<Potts, P3>::new(model, beta);
            prop_assert!((kernel.acceptance(old, new, &field) - direct.acceptance(old, new, &field)).abs() < 1e-12);
            prop_assert_eq!(kernel.heat_bath(&field, u), direct.heat_bath(&field, u));
            let e0 = model.energy(&st);
            let delta = kernel.delta_energy(old, new, &field);
            st.set(i, new);
            prop_assert!((model.energy(&st) - e0 - delta).abs() < 1e-9);
        }

        #[test]
        fn mean_field_delta_is_exact(j in -2.0..2.0, h in -2.0..2.0, counts in prop::array::uniform3(0u32..40), from in 0usize..3, to in 0usize..3) {
            prop_assume!(counts[from] > 0);
            let model = Potts::new(j, h);
            let mut st = MeanFieldState::<P3>::from_counts(counts.to_vec(), 4.0);
            let (from, to) = (P3::new(from), P3::new(to));
            let e0 = model.mean_field_energy(&st);
            let delta = model.mean_field_delta(&st, from, to);
            st.transfer(from, to);
            prop_assert!((model.mean_field_energy(&st) - e0 - delta).abs() < 1e-9);
        }
    }
}
