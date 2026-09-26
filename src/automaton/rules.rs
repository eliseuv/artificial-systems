//! Local update rules.

use rand::{Rng, RngExt as _};
use serde::{Deserialize, Serialize};

use super::LocalRule;
use crate::site::{Binary, Brass, Site, Spin};

/// Contact process rule: an active site becomes inactive with probability `1/α`, an inactive site
/// copies the state of a uniformly random neighbour.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactRule {
    recovery: f64,
}

impl ContactRule {
    /// Rule with infection rate `alpha` (`∞` allowed).
    ///
    /// # Panics
    /// If `alpha` is not positive.
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0, "Infection rate must be positive, got {alpha}");
        Self {
            recovery: alpha.recip(),
        }
    }

    /// Infection rate `α`.
    pub fn alpha(&self) -> f64 {
        self.recovery.recip()
    }
}

impl LocalRule<Binary> for ContactRule {
    #[inline(always)]
    fn apply<R: Rng + ?Sized>(
        &self,
        current: Binary,
        sites: &[Binary],
        neighbors: &[u32],
        rng: &mut R,
    ) -> Binary {
        match current {
            Binary::Active if rng.random::<f64>() < self.recovery => Binary::Inactive,
            Binary::Active => Binary::Active,
            Binary::Inactive if neighbors.is_empty() => Binary::Inactive,
            Binary::Inactive => sites[neighbors[rng.random_range(0..neighbors.len())] as usize],
        }
    }

    fn is_absorbing(&self, counts: &[u32]) -> bool {
        counts[Binary::Active.index()] == 0
    }
}

/// Totalistic binary rule: a site becomes active with probability `p[k]`, where `k` is its number
/// of active neighbours (independent of its own state).
///
/// The Domany-Kinzel automaton is `p = [0, p₁, p₂]` on a chain ([`TotalisticBinary::domany_kinzel`]);
/// `p₂ = p₁(2 - p₁)` is bond and `p₂ = p₁` site directed percolation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TotalisticBinary {
    probabilities: Vec<f64>,
}

impl TotalisticBinary {
    /// Rule with activation probabilities `probabilities[k]` for `k` active neighbours. Sites
    /// must have at most `probabilities.len() - 1` neighbours.
    ///
    /// # Panics
    /// If a probability is outside `[0, 1]` or there are none.
    pub fn new(probabilities: Vec<f64>) -> Self {
        assert!(
            !probabilities.is_empty(),
            "At least one probability required"
        );
        assert!(
            probabilities.iter().all(|p| (0.0..=1.0).contains(p)),
            "Probabilities must be in [0, 1]"
        );
        Self { probabilities }
    }

    /// Domany-Kinzel automaton with probabilities `p1` (one active neighbour) and `p2` (two).
    pub fn domany_kinzel(p1: f64, p2: f64) -> Self {
        Self::new(vec![0.0, p1, p2])
    }

    /// Activation probabilities by number of active neighbours.
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }
}

impl LocalRule<Binary> for TotalisticBinary {
    #[inline(always)]
    fn apply<R: Rng + ?Sized>(
        &self,
        _current: Binary,
        sites: &[Binary],
        neighbors: &[u32],
        rng: &mut R,
    ) -> Binary {
        let k = neighbors
            .iter()
            .filter(|&&j| sites[j as usize].is_active())
            .count();
        let p = self.probabilities[k];
        Binary::from(p >= 1.0 || (p > 0.0 && rng.random::<f64>() < p))
    }

    fn is_absorbing(&self, counts: &[u32]) -> bool {
        self.probabilities[0] == 0.0 && counts[Binary::Active.index()] == 0
    }
}

/// Wolfram elementary cellular automaton `rule` (deterministic).
///
/// Requires exactly two neighbours per site ordered `[left, right]`, i.e. a periodic
/// [`Chain`](crate::topology::Chain).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Elementary {
    /// Rule number: bit `4l + 2c + r` is the new state for neighbourhood `(l, c, r)`.
    pub rule: u8,
}

impl LocalRule<Binary> for Elementary {
    #[inline(always)]
    fn apply<R: Rng + ?Sized>(
        &self,
        current: Binary,
        sites: &[Binary],
        neighbors: &[u32],
        _rng: &mut R,
    ) -> Binary {
        let &[left, right] = neighbors else {
            panic!("Elementary automata need exactly two neighbours per site");
        };
        let pattern =
            (sites[left as usize] as u8) << 2 | (current as u8) << 1 | sites[right as usize] as u8;
        Binary::from((self.rule >> pattern) & 1 == 1)
    }

    fn is_absorbing(&self, counts: &[u32]) -> bool {
        let (inactive, active) = (counts[0], counts[1]);
        (active == 0 && self.rule & 1 == 0) || (inactive == 0 && self.rule & 0x80 != 0)
    }
}

/// Brass immune network automaton (Tomé & Drugowich de Felício).
///
/// With `σ = sign(Σⱼ sⱼ)` over the neighbours (`TH1 = +1`, `TH2 = -1`):
/// - `TH0` stays with probability `1 - p`, otherwise becomes `TH1` if `σ > 0`, `TH2` if `σ < 0`
///   and either with probability ½ if `σ = 0`.
/// - `TH1` and `TH2` decay to `TH0` with probability `r`, and stay otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BrassRule {
    /// Antigen probability `p`.
    pub p: f64,
    /// Decay probability `r`.
    pub r: f64,
}

impl BrassRule {
    /// Brass rule with probabilities `p` and `r`.
    ///
    /// # Panics
    /// If a probability is outside `[0, 1]`.
    pub fn new(p: f64, r: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p) && (0.0..=1.0).contains(&r),
            "Probabilities must be in [0, 1]"
        );
        Self { p, r }
    }
}

impl LocalRule<Brass> for BrassRule {
    #[inline(always)]
    fn apply<R: Rng + ?Sized>(
        &self,
        current: Brass,
        sites: &[Brass],
        neighbors: &[u32],
        rng: &mut R,
    ) -> Brass {
        let u: f64 = rng.random();
        match current {
            Brass::TH0 => {
                let w0 = 1.0 - self.p;
                if u < w0 {
                    return Brass::TH0;
                }
                let sum: i32 = neighbors.iter().map(|&j| sites[j as usize].value()).sum();
                let w1 = w0
                    + match sum.signum() {
                        1 => self.p,
                        0 => self.p / 2.0,
                        _ => 0.0,
                    };
                if u < w1 { Brass::TH1 } else { Brass::TH2 }
            }
            helper => {
                if u < self.r {
                    Brass::TH0
                } else {
                    helper
                }
            }
        }
    }

    fn is_absorbing(&self, counts: &[u32]) -> bool {
        self.p == 0.0 && counts[Brass::TH1.index()] == 0 && counts[Brass::TH2.index()] == 0
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        automaton::Synchronous,
        dynamics::Dynamics,
        rng::stream,
        state::{Configuration, Init, LatticeState, Position, Prepare},
        topology::Chain,
    };

    fn binomial_odd(n: usize, k: usize) -> bool {
        // Lucas: C(n, k) is odd iff the bits of k are a subset of those of n
        k & !n == 0
    }

    #[test]
    fn rule_90_draws_sierpinski() {
        let n = 129;
        let mut st = LatticeState::uniform(Arc::new(Chain::periodic([n])), Binary::Inactive);
        Init::Single {
            background: Binary::Inactive,
            value: Binary::Active,
            at: Position::Center,
        }
        .prepare(&mut st, &mut stream(0, &[]));
        let mut ca = Synchronous::new(Elementary { rule: 90 });
        let c = n / 2;
        for t in 0..60 {
            for x in 0..n {
                let d = x as isize - c as isize;
                let expected = (t as isize + d) % 2 == 0
                    && d.unsigned_abs() <= t
                    && binomial_odd(t, (t as isize + d) as usize / 2);
                assert_eq!(st.get(x).is_active(), expected, "t = {t}, x = {x}");
            }
            ca.step(&mut st, &mut stream(0, &[]));
        }
    }

    #[test]
    fn domany_kinzel_limits() {
        let mut rng = stream(3, &[]);
        let mut st = LatticeState::uniform(Arc::new(Chain::periodic([50])), Binary::Active);
        let mut dead = Synchronous::new(TotalisticBinary::domany_kinzel(0.0, 0.0));
        dead.step(&mut st, &mut rng);
        assert_eq!(st.count(Binary::Active), 0);
        assert!(dead.is_frozen(&st));
        st.set(10, Binary::Active);
        let mut full = Synchronous::new(TotalisticBinary::domany_kinzel(1.0, 1.0));
        for _ in 0..30 {
            full.step(&mut st, &mut rng);
        }
        // Light cone of a single seed on an even ring fills one sublattice
        assert_eq!(st.count(Binary::Active), 25);
    }

    #[test]
    fn brass_decay_without_antigen() {
        let mut rng = stream(4, &[]);
        let mut st = LatticeState::uniform(Arc::new(Chain::periodic([100_000])), Brass::TH1);
        let mut ca = Synchronous::new(BrassRule::new(0.0, 0.3));
        ca.step(&mut st, &mut rng);
        let survivors = st.count(Brass::TH1) as f64 / st.len() as f64;
        assert!((survivors - 0.7).abs() < 0.01, "{survivors}");
        assert_eq!(st.count(Brass::TH2), 0);
        for _ in 0..100 {
            ca.step(&mut st, &mut rng);
        }
        assert!(ca.is_frozen(&st));
    }

    #[test]
    fn brass_activation_follows_neighbour_sign() {
        let rule = BrassRule::new(1.0, 0.0);
        let sites = [Brass::TH0, Brass::TH1, Brass::TH1, Brass::TH2];
        let mut rng = stream(5, &[]);
        assert_eq!(
            rule.apply(Brass::TH0, &sites, &[1, 2], &mut rng),
            Brass::TH1
        );
        assert_eq!(
            rule.apply(Brass::TH0, &sites, &[3, 0], &mut rng),
            Brass::TH2
        );
        let tie = (0..10_000)
            .filter(|_| rule.apply(Brass::TH0, &sites, &[1, 3], &mut rng) == Brass::TH1)
            .count();
        assert!((tie as f64 / 10_000.0 - 0.5).abs() < 0.02);
    }
}
