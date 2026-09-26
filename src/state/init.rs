//! Initial state specifications.

use std::{fmt, sync::Arc};

use rand::{Rng, RngExt as _, seq::SliceRandom as _};
use rand_distr::{Binomial, Distribution as _};

use super::{Configuration, LatticeState, MeanFieldState};
use crate::{site::Site, topology::Topology};

/// Prepare a system in some initial state, drawing any randomness from `rng`.
///
/// All randomness must come from `rng` so ensembles stay reproducible.
pub trait Prepare<Sys>: Send + Sync {
    /// Overwrite the state of `sys`.
    fn prepare<R: Rng>(&self, sys: &mut Sys, rng: &mut R);
}

/// Arbitrary initialisation of lattice site values.
pub type CustomInit<S> = Arc<dyn Fn(&mut [S], &mut dyn Rng) + Send + Sync>;

/// Location of a distinguished site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    /// Site with the given flat index.
    Index(usize),
    /// Center of the topology, see [`Topology::center`].
    Center,
}

/// Initial state specification.
#[derive(Clone)]
pub enum Init<S: Site> {
    /// Every site in the same state.
    Uniform(S),
    /// Every site independently and uniformly random.
    IidUniform,
    /// Every site independently random with (unnormalised) weights indexed by [`Site::index`].
    Iid(Vec<f64>),
    /// Every site in state `background` except a single one in state `value`.
    Single {
        background: S,
        value: S,
        at: Position,
    },
    /// Exact number of sites in each state (indexed by [`Site::index`]), at random positions.
    Exact(Vec<u32>),
    /// Arbitrary initialisation of the site values (lattice states only).
    Custom(CustomInit<S>),
}

impl<S: Site> Init<S> {
    /// Independent sites with given weights.
    ///
    /// # Panics
    /// If there is not one weight per site value, a weight is negative or not finite, or all
    /// weights are zero.
    pub fn iid(weights: Vec<f64>) -> Self {
        assert_eq!(
            weights.len(),
            S::COUNT,
            "One weight per site value required"
        );
        assert!(
            weights.iter().all(|w| w.is_finite() && *w >= 0.0),
            "Weights must be finite and non-negative"
        );
        assert!(
            weights.iter().any(|&w| w > 0.0),
            "Some weight must be positive"
        );
        Self::Iid(weights)
    }

    /// Sites independently in state `value` with probability `p`, and `other` otherwise.
    pub fn bernoulli(p: f64, value: S, other: S) -> Self {
        assert!((0.0..=1.0).contains(&p), "Probability must be in [0, 1]");
        let mut weights = vec![0.0; S::COUNT];
        weights[other.index()] += 1.0 - p;
        weights[value.index()] += p;
        Self::iid(weights)
    }

    /// Cumulative probabilities of the independent site distribution, if any.
    fn cumulative(&self) -> Option<Vec<f64>> {
        match self {
            Self::IidUniform => Some((1..=S::COUNT).map(|k| k as f64 / S::COUNT as f64).collect()),
            Self::Iid(weights) => {
                let total: f64 = weights.iter().sum();
                let mut acc = 0.0;
                Some(
                    weights
                        .iter()
                        .map(|w| {
                            acc += w / total;
                            acc
                        })
                        .collect(),
                )
            }
            _ => None,
        }
    }
}

impl<S: Site> fmt::Debug for Init<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uniform(s) => f.debug_tuple("Uniform").field(s).finish(),
            Self::IidUniform => f.write_str("IidUniform"),
            Self::Iid(w) => f.debug_tuple("Iid").field(w).finish(),
            Self::Single {
                background,
                value,
                at,
            } => f
                .debug_struct("Single")
                .field("background", background)
                .field("value", value)
                .field("at", at)
                .finish(),
            Self::Exact(c) => f.debug_tuple("Exact").field(c).finish(),
            Self::Custom(_) => f.write_str("Custom(..)"),
        }
    }
}

/// Sample an index from cumulative probabilities (last entry is 1 up to rounding).
#[inline]
fn sample_cumulative<R: Rng + ?Sized>(cumulative: &[f64], rng: &mut R) -> usize {
    let u: f64 = rng.random();
    cumulative
        .iter()
        .position(|&c| u < c)
        .unwrap_or(cumulative.len() - 1)
}

impl<S: Site, T: Topology> Prepare<LatticeState<S, T>> for Init<S> {
    fn prepare<R: Rng>(&self, state: &mut LatticeState<S, T>, rng: &mut R) {
        match self {
            Self::Uniform(s) => state.fill(*s),
            Self::IidUniform => {
                state.assign_with(|sites| sites.iter_mut().for_each(|s| *s = S::random(rng)))
            }
            Self::Iid(_) => {
                let cumulative = self.cumulative().expect("independent distribution");
                state.assign_with(|sites| {
                    sites
                        .iter_mut()
                        .for_each(|s| *s = S::from_index(sample_cumulative(&cumulative, rng)))
                });
            }
            Self::Single {
                background,
                value,
                at,
            } => {
                let i = match at {
                    Position::Index(i) => *i,
                    Position::Center => state.topology().center(),
                };
                state.fill(*background);
                state.set(i, *value);
            }
            Self::Exact(counts) => {
                assert_eq!(counts.len(), S::COUNT, "One count per site value required");
                assert_eq!(
                    counts.iter().map(|&n| n as usize).sum::<usize>(),
                    state.len(),
                    "Counts must sum to the number of sites"
                );
                state.assign_with(|sites| {
                    let values = counts
                        .iter()
                        .enumerate()
                        .flat_map(|(k, &n)| std::iter::repeat_n(S::from_index(k), n as usize));
                    for (s, v) in sites.iter_mut().zip(values) {
                        *s = v;
                    }
                    sites.shuffle(rng);
                });
            }
            Self::Custom(f) => state.assign_with(|sites| f(sites, rng)),
        }
    }
}

impl<S: Site> Prepare<MeanFieldState<S>> for Init<S> {
    /// # Panics
    /// For [`Init::Custom`], which has no meaning without site positions.
    fn prepare<R: Rng>(&self, state: &mut MeanFieldState<S>, rng: &mut R) {
        let n = state.len() as u32;
        let mut counts = vec![0u32; S::COUNT];
        match self {
            Self::Uniform(s) => counts[s.index()] = n,
            Self::IidUniform | Self::Iid(_) => {
                // Multinomial through sequential conditional binomials
                let weights = match self {
                    Self::Iid(w) => w.clone(),
                    _ => vec![1.0; S::COUNT],
                };
                let mut remaining = n as u64;
                let mut mass: f64 = weights.iter().sum();
                for (k, w) in weights.iter().enumerate() {
                    if remaining == 0 || mass <= 0.0 {
                        break;
                    }
                    let p = (w / mass).clamp(0.0, 1.0);
                    let x = if k + 1 == S::COUNT || p >= 1.0 {
                        remaining
                    } else {
                        Binomial::new(remaining, p)
                            .expect("valid binomial parameters")
                            .sample(rng)
                    };
                    counts[k] = x as u32;
                    remaining -= x;
                    mass -= w;
                }
            }
            Self::Single {
                background, value, ..
            } => {
                counts[background.index()] += n - 1;
                counts[value.index()] += 1;
            }
            Self::Exact(c) => counts.clone_from(c),
            Self::Custom(_) => panic!("Custom initial states require site positions"),
        }
        state.set_counts(&counts);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        rng::stream,
        site::{Binary, SpinOne},
        topology::Chain,
    };

    fn chain(n: usize) -> LatticeState<SpinOne, Chain> {
        LatticeState::uniform(Arc::new(Chain::periodic([n])), SpinOne::Zero)
    }

    #[test]
    fn lattice_initial_states() {
        let mut rng = stream(1, &[]);
        let mut st = chain(11);
        Init::Uniform(SpinOne::Up).prepare(&mut st, &mut rng);
        assert_eq!(st.counts(), &[0, 0, 11]);
        Init::Single {
            background: SpinOne::Zero,
            value: SpinOne::Down,
            at: Position::Center,
        }
        .prepare(&mut st, &mut rng);
        assert_eq!(st.get(5), SpinOne::Down);
        assert_eq!(st.counts(), &[1, 10, 0]);
        Init::Exact(vec![3, 0, 8]).prepare(&mut st, &mut rng);
        assert_eq!(st.counts(), &[3, 0, 8]);
        Init::iid(vec![0.0, 1.0, 0.0]).prepare(&mut st, &mut rng);
        assert_eq!(st.counts(), &[0, 11, 0]);
    }

    #[test]
    fn iid_frequencies() {
        let mut rng = stream(2, &[]);
        let mut st = LatticeState::uniform(Arc::new(Chain::periodic([100_000])), Binary::Inactive);
        Init::bernoulli(0.3, Binary::Active, Binary::Inactive).prepare(&mut st, &mut rng);
        let p = st.count(Binary::Active) as f64 / st.len() as f64;
        assert!((p - 0.3).abs() < 0.01, "p = {p}");
    }

    #[test]
    fn mean_field_multinomial() {
        let mut rng = stream(3, &[]);
        let mut st = MeanFieldState::uniform(90_000, 4.0, SpinOne::Zero);
        Init::IidUniform.prepare(&mut st, &mut rng);
        assert_eq!(st.counts().iter().sum::<u32>(), 90_000);
        for &c in st.counts() {
            assert!(
                (c as f64 - 30_000.0).abs() < 600.0,
                "counts = {:?}",
                st.counts()
            );
        }
        Init::Single {
            background: SpinOne::Up,
            value: SpinOne::Down,
            at: Position::Center,
        }
        .prepare(&mut st, &mut rng);
        assert_eq!(st.counts(), &[1, 0, 89_999]);
    }

    #[test]
    fn same_seed_same_state() {
        let init = Init::<SpinOne>::IidUniform;
        let (mut a, mut b) = (chain(64), chain(64));
        init.prepare(&mut a, &mut stream(9, &[1]));
        init.prepare(&mut b, &mut stream(9, &[1]));
        assert_eq!(a.sites(), b.sites());
    }
}
