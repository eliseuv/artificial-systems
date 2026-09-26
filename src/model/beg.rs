//! Blume-Emery-Griffiths family: Ising, Blume-Capel and the full BEG model.

use serde::{Deserialize, Serialize};

use super::{
    DirectKernel, Kernel, LocalModel, MeanFieldModel, boltzmann_cumulative, check_beta,
    metropolis_acceptance, sample_cumulative,
};
use crate::{
    site::Spin,
    state::{Configuration, MeanFieldState},
};

/// Blume-Emery-Griffiths Hamiltonian
///
/// ```text
/// H = - J Σ_⟨ij⟩ sᵢsⱼ - K Σ_⟨ij⟩ sᵢ²sⱼ² - H₃ Σ_⟨ij⟩ sᵢsⱼ(sᵢ + sⱼ) + D Σᵢ sᵢ² - H Σᵢ sᵢ
/// ```
///
/// for any integer valued [`Spin`]:
/// - Ising: spin-½ with `K = H₃ = D = 0` (see [`Beg::ising`]).
/// - Blume-Capel: spin-1 with `K = H₃ = 0` (see [`Beg::blume_capel`]).
///
/// The local field of a site is `(Σⱼ sⱼ, Σⱼ sⱼ²)` over its neighbours.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Beg {
    /// Bilinear exchange coupling `J`.
    pub j: f64,
    /// Biquadratic coupling `K`.
    pub k: f64,
    /// Cubic (multispin) coupling `H₃`.
    pub h3: f64,
    /// Crystal field (anisotropy) `D`.
    pub d: f64,
    /// External magnetic field `H`.
    pub h: f64,
}

impl Beg {
    /// Ising model `H = -J Σ_⟨ij⟩ sᵢsⱼ - h Σᵢ sᵢ`.
    pub const fn ising(j: f64, h: f64) -> Self {
        Self {
            j,
            k: 0.0,
            h3: 0.0,
            d: 0.0,
            h,
        }
    }

    /// Blume-Capel model `H = -J Σ_⟨ij⟩ sᵢsⱼ + D Σᵢ sᵢ² - h Σᵢ sᵢ`.
    pub const fn blume_capel(j: f64, d: f64, h: f64) -> Self {
        Self {
            j,
            k: 0.0,
            h3: 0.0,
            d,
            h,
        }
    }

    /// Energy of a site with value `s` and local field `(a, b) = (Σⱼ sⱼ, Σⱼ sⱼ²)`.
    #[inline(always)]
    fn energy_of(&self, s: i32, a: i32, b: i32) -> f64 {
        let (s, a, b) = (s as f64, a as f64, b as f64);
        let s2 = s * s;
        -self.j * s * a - self.k * s2 * b - self.h3 * (s2 * a + s * b) + self.d * s2 - self.h * s
    }
}

impl<S: Spin> LocalModel<S> for Beg {
    type Field = (i32, i32);
    type Kernel = BegKernel<S>;

    #[inline(always)]
    fn field(&self, sites: &[S], neighbors: &[u32]) -> Self::Field {
        neighbors.iter().fold((0, 0), |(a, b), &j| {
            let v = sites[j as usize].value();
            (a + v, b + v * v)
        })
    }

    #[inline]
    fn bond_energy(&self, a: S, b: S) -> f64 {
        let (a, b) = (a.value() as f64, b.value() as f64);
        -self.j * a * b - self.k * a * a * b * b - self.h3 * a * b * (a + b)
    }

    #[inline]
    fn onsite_energy(&self, s: S) -> f64 {
        let s = s.value() as f64;
        self.d * s * s - self.h * s
    }

    #[inline(always)]
    fn site_energy(&self, s: S, &(a, b): &Self::Field) -> f64 {
        self.energy_of(s.value(), a, b)
    }

    fn kernel(&self, beta: f64, max_degree: usize) -> Self::Kernel {
        BegKernel::new(*self, beta, max_degree)
    }
}

/// Largest number of table entries before falling back to direct evaluation.
const MAX_TABLE_LEN: usize = 1 << 22;

/// Tabulated energies, Metropolis acceptances and heat bath distributions of [`Beg`] indexed by
/// the local field.
#[derive(Debug, Clone)]
pub struct BegKernel<S> {
    table: Option<BegTable>,
    direct: DirectKernel<Beg, S>,
}

#[derive(Debug, Clone)]
struct BegTable {
    /// Offset of `Σⱼ sⱼ` so indices are non-negative.
    a_offset: i32,
    /// Number of possible values of `Σⱼ sⱼ²`.
    b_count: i32,
    /// `energies[f * K + s]`
    energies: Box<[f64]>,
    /// `acceptance[(f * K + old) * K + new]`
    acceptance: Box<[f64]>,
    /// `cumulative[f * K + s]`
    cumulative: Box<[f64]>,
}

impl<S: Spin> BegKernel<S> {
    fn new(model: Beg, beta: f64, max_degree: usize) -> Self {
        check_beta(beta);
        let s_max = S::VALUES.iter().map(|s| s.value().abs()).max().unwrap_or(0);
        let z = i32::try_from(max_degree).expect("Degree too large");
        let a_max = z * s_max;
        let b_max = z * s_max * s_max;
        let fields = (2 * a_max as usize + 1) * (b_max as usize + 1);
        let k = S::COUNT;
        let table = (fields * k * k <= MAX_TABLE_LEN).then(|| {
            let mut energies = Vec::with_capacity(fields * k);
            for a in -a_max..=a_max {
                for b in 0..=b_max {
                    energies.extend(S::VALUES.iter().map(|s| model.energy_of(s.value(), a, b)));
                }
            }
            let mut acceptance = Vec::with_capacity(fields * k * k);
            let mut cumulative = vec![0.0; fields * k];
            for (e, c) in energies.chunks(k).zip(cumulative.chunks_mut(k)) {
                for old in e {
                    acceptance.extend(e.iter().map(|new| metropolis_acceptance(beta, new - old)));
                }
                boltzmann_cumulative(beta, e, c);
            }
            BegTable {
                a_offset: a_max,
                b_count: b_max + 1,
                energies: energies.into(),
                acceptance: acceptance.into(),
                cumulative: cumulative.into(),
            }
        });
        Self {
            table,
            direct: DirectKernel::new(model, beta),
        }
    }
}

impl BegTable {
    #[inline(always)]
    fn field_index(&self, (a, b): (i32, i32)) -> usize {
        ((a + self.a_offset) * self.b_count + b) as usize
    }
}

impl<S: Spin> Kernel<S, (i32, i32)> for BegKernel<S> {
    #[inline(always)]
    fn delta_energy(&self, old: S, new: S, &(a, b): &(i32, i32)) -> f64 {
        match &self.table {
            Some(t) => {
                let base = t.field_index((a, b)) * S::COUNT;
                t.energies[base + new.index()] - t.energies[base + old.index()]
            }
            None => self.direct.delta_energy(old, new, &(a, b)),
        }
    }

    #[inline(always)]
    fn acceptance(&self, old: S, new: S, field: &(i32, i32)) -> f64 {
        match &self.table {
            Some(t) => {
                let k = S::COUNT;
                t.acceptance[(t.field_index(*field) * k + old.index()) * k + new.index()]
            }
            None => self.direct.acceptance(old, new, field),
        }
    }

    #[inline(always)]
    fn heat_bath(&self, &(a, b): &(i32, i32), u: f64) -> S {
        let k = S::COUNT;
        match &self.table {
            Some(t) => {
                let base = t.field_index((a, b)) * k;
                S::from_index(sample_cumulative(&t.cumulative[base..base + k], u))
            }
            None => self.direct.heat_bath(&(a, b), u),
        }
    }
}

/// Integer power sums `Pₖ = Σᵢ sᵢᵏ` for `k = 1..=4`.
#[inline]
fn power_sums<S: Spin>(counts: &[u32]) -> [i128; 4] {
    let mut p = [0i128; 4];
    for (s, &n) in S::VALUES.iter().zip(counts) {
        let (v, n) = (s.value() as i128, n as i128);
        p[0] += n * v;
        p[1] += n * v * v;
        p[2] += n * v * v * v;
        p[3] += n * v * v * v * v;
    }
    p
}

impl Beg {
    /// Mean-field energy from power sums, split into the (exact, integer) pair sums and the
    /// on-site sums: `Σ_{i≠j} sᵢsⱼ = P₁² - P₂`, `Σ_{i≠j} sᵢ²sⱼ² = P₂² - P₄` and
    /// `Σ_{i≠j} sᵢsⱼ(sᵢ + sⱼ) = 2 (P₁P₂ - P₃)`.
    #[inline]
    fn mean_field_terms([p1, p2, p3, p4]: [i128; 4]) -> [i128; 5] {
        [p1 * p1 - p2, p2 * p2 - p4, 2 * (p1 * p2 - p3), p2, p1]
    }

    #[inline]
    fn mean_field_combine(&self, terms: [i128; 5], z: f64, n: f64) -> f64 {
        let [pair_j, pair_k, pair_h3, p2, p1] = terms.map(|t| t as f64);
        -(z / (2.0 * n)) * (self.j * pair_j + self.k * pair_k + self.h3 * pair_h3) + self.d * p2
            - self.h * p1
    }
}

impl<S: Spin> MeanFieldModel<S> for Beg {
    /// `H = -(z/2N) Σ_{i≠j} [J sᵢsⱼ + K sᵢ²sⱼ² + H₃ sᵢsⱼ(sᵢ + sⱼ)] + D Σᵢ sᵢ² - H Σᵢ sᵢ`
    fn mean_field_energy(&self, state: &MeanFieldState<S>) -> f64 {
        let terms = Self::mean_field_terms(power_sums::<S>(state.counts()));
        self.mean_field_combine(terms, state.coordination(), state.len() as f64)
    }

    fn mean_field_delta(&self, state: &MeanFieldState<S>, from: S, to: S) -> f64 {
        if from == to {
            return 0.0;
        }
        let sums = low_power_sums::<S>(state.counts());
        self.mean_field_delta_with(sums, state, from, to)
    }

    fn mean_field_deltas(&self, state: &MeanFieldState<S>, from: S, out: &mut [f64]) {
        let sums = low_power_sums::<S>(state.counts());
        for (&to, d) in S::VALUES.iter().zip(out.iter_mut()) {
            *d = if to == from {
                0.0
            } else {
                self.mean_field_delta_with(sums, state, from, to)
            };
        }
    }
}

/// `(P₁, P₂)` in `i64`.
#[inline(always)]
fn low_power_sums<S: Spin>(counts: &[u32]) -> (i64, i64) {
    S::VALUES
        .iter()
        .zip(counts)
        .fold((0, 0), |(p1, p2), (s, &n)| {
            let (v, n) = (s.value() as i64, n as i64);
            (p1 + n * v, p2 + n * v * v)
        })
}

impl Beg {
    /// Energy change of moving a site from `from` to `to` given `(P₁, P₂)`, from exact integer
    /// differences of the pair sums (each `O(N)`, so they fit in `i64`):
    /// `Δ(P₁²) = ΔP₁ (2P₁ + ΔP₁)`, `Δ(P₂²) = ΔP₂ (2P₂ + ΔP₂)` and
    /// `Δ(P₁P₂) = ΔP₁P₂ + P₁ΔP₂ + ΔP₁ΔP₂`.
    #[inline(always)]
    fn mean_field_delta_with<S: Spin>(
        &self,
        (p1, p2): (i64, i64),
        state: &MeanFieldState<S>,
        from: S,
        to: S,
    ) -> f64 {
        let (f, t) = (from.value() as i64, to.value() as i64);
        let d = |k: u32| t.pow(k) - f.pow(k);
        let (d1, d2, d3, d4) = (d(1), d(2), d(3), d(4));
        let terms = [
            d1 * (2 * p1 + d1) - d2,
            d2 * (2 * p2 + d2) - d4,
            2 * (d1 * p2 + p1 * d2 + d1 * d2 - d3),
            d2,
            d1,
        ];
        self.mean_field_combine(
            terms.map(i128::from),
            state.coordination(),
            state.len() as f64,
        )
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
        site::{Site, SpinHalf, SpinOne},
        state::{Init, LatticeState, Prepare},
        topology::{Graph, Square, Topology},
    };

    fn beg() -> impl Strategy<Value = Beg> {
        (-2.0..2.0, -2.0..2.0, -2.0..2.0, -3.0..3.0, -2.0..2.0).prop_map(|(j, k, h3, d, h)| Beg {
            j,
            k,
            h3,
            d,
            h,
        })
    }

    proptest! {
        #[test]
        fn local_delta_matches_total_energy(model in beg(), seed: u64, i in 0usize..16, new in 0usize..3) {
            let top = Arc::new(Square::periodic([4, 4]));
            let mut st = LatticeState::uniform(top, SpinOne::Zero);
            Init::IidUniform.prepare(&mut st, &mut stream(seed, &[]));
            let old = st.get(i);
            let new = SpinOne::from_index(new);
            let field = LocalModel::<SpinOne>::field(&model, st.sites(), st.neighbors(i));
            let e0 = model.energy(&st);
            let kernel = Model::kernel(&model, &st, 1.0);
            let delta = kernel.delta_energy(old, new, &field);
            let direct = DirectKernel::<Beg, SpinOne>::new(model, 1.0).delta_energy(old, new, &field);
            st.set(i, new);
            let e1 = model.energy(&st);
            prop_assert!((e1 - e0 - delta).abs() < 1e-9, "{} vs {}", e1 - e0, delta);
            prop_assert!((direct - delta).abs() < 1e-9);
        }

        #[test]
        fn mean_field_delta_is_exact(model in beg(), counts in prop::array::uniform3(0u32..50), from in 0usize..3, to in 0usize..3) {
            prop_assume!(counts[from] > 0);
            let mut st = MeanFieldState::<SpinOne>::from_counts(counts.to_vec(), 4.0);
            let (from, to) = (SpinOne::from_index(from), SpinOne::from_index(to));
            let e0 = model.mean_field_energy(&st);
            let delta = model.mean_field_delta(&st, from, to);
            st.transfer(from, to);
            let e1 = model.mean_field_energy(&st);
            prop_assert!((e1 - e0 - delta).abs() < 1e-9 * (1.0 + e0.abs()));
        }
    }

    #[test]
    fn ising_energy_of_ordered_states() {
        let top = Arc::new(Square::periodic_cube(8));
        let n = top.len() as f64;
        let st = LatticeState::uniform(top, SpinHalf::Up);
        assert_eq!(Beg::ising(1.0, 0.0).energy(&st), -2.0 * n);
        assert_eq!(Beg::ising(1.0, 0.5).energy(&st), -2.5 * n);
        let mf = MeanFieldState::uniform(100, 4.0, SpinHalf::Up);
        // -(z/2N)(N² - N) = -(z/2)(N - 1)
        approx::assert_relative_eq!(Beg::ising(1.0, 0.0).mean_field_energy(&mf), -2.0 * 99.0);
    }

    #[test]
    fn ising_mean_field_flip_matches_known_form() {
        // ΔH = 2Jz(sM - 1)/N for flipping a spin s
        let st = MeanFieldState::<SpinHalf>::from_counts(vec![30, 70], 4.0);
        let model = Beg::ising(1.0, 0.0);
        let delta = model.mean_field_delta(&st, SpinHalf::Up, SpinHalf::Down);
        approx::assert_relative_eq!(delta, 2.0 * 4.0 * (40.0 - 1.0) / 100.0);
    }

    #[test]
    fn large_degree_falls_back_to_direct_evaluation() {
        let g = Graph::complete(1001);
        let kernel =
            LocalModel::<SpinOne>::kernel(&Beg::blume_capel(1.0, 0.5, 0.0), 0.3, g.max_degree());
        assert!(kernel.table.is_none());
        let field = (10, 100);
        let direct = DirectKernel::<Beg, SpinOne>::new(Beg::blume_capel(1.0, 0.5, 0.0), 0.3);
        for &old in SpinOne::VALUES {
            for &new in SpinOne::VALUES {
                assert_eq!(
                    kernel.acceptance(old, new, &field),
                    direct.acceptance(old, new, &field)
                );
            }
        }
        for u in [0.0, 0.3, 0.99] {
            assert_eq!(kernel.heat_bath(&field, u), direct.heat_bath(&field, u));
        }
    }
}
