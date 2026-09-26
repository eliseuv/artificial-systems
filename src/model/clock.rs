//! `Q`-state clock (discrete XY) model.

use std::f64::consts::TAU;

use serde::{Deserialize, Serialize};

use super::{DirectKernel, LocalModel, MeanFieldModel};
use crate::{
    site::ClockState,
    state::{Configuration, MeanFieldState},
};

/// Clock Hamiltonian `H = -J Σ_⟨ij⟩ cos(θᵢ - θⱼ) - h Σᵢ cos θᵢ` with `θ = 2πq/Q`.
///
/// The local field of a site is `(Σⱼ cos θⱼ, Σⱼ sin θⱼ)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Clock {
    /// Coupling `J`.
    pub j: f64,
    /// Field `h` along `θ = 0`.
    pub h: f64,
    cos: Box<[f64]>,
    sin: Box<[f64]>,
}

impl Clock {
    /// Clock model for `q` states with coupling `j` and field `h`.
    pub fn new(q: usize, j: f64, h: f64) -> Self {
        assert!(q >= 2, "Clock model needs at least two states");
        let angles = (0..q).map(|k| TAU * k as f64 / q as f64);
        Self {
            j,
            h,
            cos: angles.clone().map(f64::cos).collect(),
            sin: angles.map(f64::sin).collect(),
        }
    }

    /// Number of states `Q` the model was built for.
    pub fn states(&self) -> usize {
        self.cos.len()
    }

    #[inline(always)]
    fn check<const Q: usize>(&self) {
        debug_assert_eq!(self.cos.len(), Q, "Clock model built for a different Q");
    }

    /// Magnetisation vector `(Σᵢ cos θᵢ, Σᵢ sin θᵢ)` from per state counts.
    pub fn magnetization(&self, counts: &[u32]) -> (f64, f64) {
        counts
            .iter()
            .zip(self.cos.iter().zip(self.sin.iter()))
            .fold((0.0, 0.0), |(x, y), (&n, (c, s))| {
                (x + n as f64 * c, y + n as f64 * s)
            })
    }
}

impl<const Q: usize> LocalModel<ClockState<Q>> for Clock {
    type Field = (f64, f64);
    type Kernel = DirectKernel<Clock, ClockState<Q>>;

    #[inline(always)]
    fn field(&self, sites: &[ClockState<Q>], neighbors: &[u32]) -> Self::Field {
        self.check::<Q>();
        neighbors.iter().fold((0.0, 0.0), |(x, y), &j| {
            let q = sites[j as usize].q();
            (x + self.cos[q], y + self.sin[q])
        })
    }

    #[inline]
    fn bond_energy(&self, a: ClockState<Q>, b: ClockState<Q>) -> f64 {
        let (a, b) = (a.q(), b.q());
        -self.j * (self.cos[a] * self.cos[b] + self.sin[a] * self.sin[b])
    }

    #[inline]
    fn onsite_energy(&self, s: ClockState<Q>) -> f64 {
        -self.h * self.cos[s.q()]
    }

    #[inline(always)]
    fn site_energy(&self, s: ClockState<Q>, &(x, y): &Self::Field) -> f64 {
        let q = s.q();
        -self.j * (self.cos[q] * x + self.sin[q] * y) - self.h * self.cos[q]
    }

    fn kernel(&self, beta: f64, _max_degree: usize) -> Self::Kernel {
        self.check::<Q>();
        DirectKernel::new(self.clone(), beta)
    }
}

impl<const Q: usize> MeanFieldModel<ClockState<Q>> for Clock {
    /// `H = -(Jz/2N) (|M|² - N) - h Mₓ` with `M = Σᵢ (cos θᵢ, sin θᵢ)`
    fn mean_field_energy(&self, state: &MeanFieldState<ClockState<Q>>) -> f64 {
        self.check::<Q>();
        let (mx, my) = self.magnetization(state.counts());
        let n = state.len() as f64;
        -self.j * state.coordination() / (2.0 * n) * (mx * mx + my * my - n) - self.h * mx
    }

    fn mean_field_delta(
        &self,
        state: &MeanFieldState<ClockState<Q>>,
        from: ClockState<Q>,
        to: ClockState<Q>,
    ) -> f64 {
        let (mx, my) = self.magnetization(state.counts());
        let (f, t) = (from.q(), to.q());
        let (dx, dy) = (self.cos[t] - self.cos[f], self.sin[t] - self.sin[f]);
        // |M + ΔM|² - |M|² = 2 M·ΔM + |ΔM|², avoiding cancellation of O(N²) terms
        let d_sq = 2.0 * (mx * dx + my * dy) + dx * dx + dy * dy;
        -self.j * state.coordination() / (2.0 * state.len() as f64) * d_sq - self.h * dx
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use proptest::prelude::*;

    use super::*;
    use crate::{
        model::{Kernel, Model},
        rng::stream,
        site::Site,
        state::{Init, LatticeState, Prepare},
        topology::Square,
    };

    type C6 = ClockState<6>;

    proptest! {
        #[test]
        fn local_delta_matches_total_energy(j in -2.0..2.0, h in -2.0..2.0, seed: u64, i in 0usize..16, new in 0usize..6) {
            let model = Clock::new(6, j, h);
            let mut st = LatticeState::uniform(Arc::new(Square::periodic([4, 4])), C6::new(0));
            Init::IidUniform.prepare(&mut st, &mut stream(seed, &[]));
            let (old, new) = (st.get(i), C6::from_index(new));
            let field = LocalModel::<C6>::field(&model, st.sites(), st.neighbors(i));
            let delta = Model::kernel(&model, &st, 1.0).delta_energy(old, new, &field);
            let e0 = model.energy(&st);
            st.set(i, new);
            prop_assert!((model.energy(&st) - e0 - delta).abs() < 1e-9);
        }

        #[test]
        fn mean_field_delta_matches(j in -2.0..2.0, h in -2.0..2.0, counts in prop::array::uniform6(0u32..30), from in 0usize..6, to in 0usize..6) {
            prop_assume!(counts[from] > 0);
            let model = Clock::new(6, j, h);
            let mut st = MeanFieldState::<C6>::from_counts(counts.to_vec(), 4.0);
            let (from, to) = (C6::new(from), C6::new(to));
            let e0 = model.mean_field_energy(&st);
            let delta = model.mean_field_delta(&st, from, to);
            st.transfer(from, to);
            prop_assert!((model.mean_field_energy(&st) - e0 - delta).abs() < 1e-9);
        }
    }
}
