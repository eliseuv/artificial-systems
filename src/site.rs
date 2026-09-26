//! Single site states.
//!
//! Every site type has a finite set of values, enumerated by [`Site::VALUES`]. The position of a
//! value in that list is its [`Site::index`], which the states use for bookkeeping (per value
//! counts) and the models use to index precomputed tables.

use std::{
    f64::consts::TAU,
    fmt::{self, Debug, Display},
};

use rand::{Rng, RngExt as _};

/// Single site state with finitely many values.
pub trait Site: Copy + Eq + Debug + Send + Sync + 'static {
    /// All possible values, ordered by [`Site::index`].
    const VALUES: &'static [Self];

    /// Number of possible values.
    const COUNT: usize = Self::VALUES.len();

    /// Position of `self` in [`Site::VALUES`].
    fn index(self) -> usize;

    /// Value at position `i` of [`Site::VALUES`].
    ///
    /// # Panics
    /// If `i >= Self::COUNT`.
    #[inline(always)]
    fn from_index(i: usize) -> Self {
        Self::VALUES[i]
    }

    /// Uniformly random value.
    #[inline]
    fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::from_index(rng.random_range(0..Self::COUNT))
    }
}

/// Site with an integer valued spin projection `s`.
pub trait Spin: Site {
    /// Spin projection.
    fn value(self) -> i32;
}

/// Spin whose values come in `±s` pairs.
pub trait Flip: Spin {
    /// Value with the opposite projection.
    fn flipped(self) -> Self;
}

/// Spin-½ with projections `s ∈ {-1, +1}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum SpinHalf {
    Down = -1,
    Up = 1,
}

impl Site for SpinHalf {
    const VALUES: &'static [Self] = &[Self::Down, Self::Up];

    #[inline(always)]
    fn index(self) -> usize {
        match self {
            Self::Down => 0,
            Self::Up => 1,
        }
    }
}

impl Spin for SpinHalf {
    #[inline(always)]
    fn value(self) -> i32 {
        self as i32
    }
}

impl Flip for SpinHalf {
    #[inline(always)]
    fn flipped(self) -> Self {
        match self {
            Self::Down => Self::Up,
            Self::Up => Self::Down,
        }
    }
}

/// Spin-1 with projections `s ∈ {-1, 0, +1}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum SpinOne {
    Down = -1,
    Zero = 0,
    Up = 1,
}

impl Site for SpinOne {
    const VALUES: &'static [Self] = &[Self::Down, Self::Zero, Self::Up];

    #[inline(always)]
    fn index(self) -> usize {
        (self as i8 + 1) as usize
    }
}

impl Spin for SpinOne {
    #[inline(always)]
    fn value(self) -> i32 {
        self as i32
    }
}

impl Flip for SpinOne {
    #[inline(always)]
    fn flipped(self) -> Self {
        match self {
            Self::Down => Self::Up,
            Self::Zero => Self::Zero,
            Self::Up => Self::Down,
        }
    }
}

/// State `q ∈ {0, …, Q-1}` of a `Q`-state Potts model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PottsState<const Q: usize>(u8);

/// State `q ∈ {0, …, Q-1}` of a `Q`-state clock model, pointing at angle `θ_q = 2πq/Q`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockState<const Q: usize>(u8);

macro_rules! impl_q_state {
    ($ty:ident) => {
        impl<const Q: usize> $ty<Q> {
            const ALL: [Self; Q] = {
                assert!(Q >= 2 && Q <= u8::MAX as usize + 1, "Q must be in 2..=256");
                let mut all = [Self(0); Q];
                let mut q = 0;
                while q < Q {
                    all[q] = Self(q as u8);
                    q += 1;
                }
                all
            };

            /// State `q`.
            ///
            /// # Panics
            /// If `q >= Q`.
            #[inline]
            pub fn new(q: usize) -> Self {
                assert!(q < Q, "State {q} out of range for Q = {Q}");
                Self(q as u8)
            }

            /// Label `q` of the state.
            #[inline(always)]
            pub fn q(self) -> usize {
                self.0 as usize
            }
        }

        impl<const Q: usize> Site for $ty<Q> {
            const VALUES: &'static [Self] = &Self::ALL;

            #[inline(always)]
            fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

impl_q_state!(PottsState);
impl_q_state!(ClockState);

impl<const Q: usize> ClockState<Q> {
    /// Angle `θ_q = 2πq/Q`.
    #[inline]
    pub fn angle(self) -> f64 {
        TAU * self.0 as f64 / Q as f64
    }
}

/// Binary site of absorbing state cellular automata such as the contact process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Binary {
    Inactive = 0,
    Active = 1,
}

impl Binary {
    /// Whether the site is active.
    #[inline(always)]
    pub fn is_active(self) -> bool {
        self == Self::Active
    }
}

impl From<bool> for Binary {
    #[inline(always)]
    fn from(active: bool) -> Self {
        if active { Self::Active } else { Self::Inactive }
    }
}

impl Site for Binary {
    const VALUES: &'static [Self] = &[Self::Inactive, Self::Active];

    #[inline(always)]
    fn index(self) -> usize {
        self as usize
    }
}

impl Spin for Binary {
    #[inline(always)]
    fn value(self) -> i32 {
        self as i32
    }
}

/// Cell of the Brass (immune network) cellular automaton: virgin (`TH0`), `TH1` (`+1`) or `TH2`
/// (`-1`) helper cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum Brass {
    TH0 = 0,
    TH1 = 1,
    TH2 = -1,
}

impl Site for Brass {
    const VALUES: &'static [Self] = &[Self::TH0, Self::TH1, Self::TH2];

    #[inline(always)]
    fn index(self) -> usize {
        match self {
            Self::TH0 => 0,
            Self::TH1 => 1,
            Self::TH2 => 2,
        }
    }
}

impl Spin for Brass {
    #[inline(always)]
    fn value(self) -> i32 {
        self as i32
    }
}

impl Flip for Brass {
    #[inline(always)]
    fn flipped(self) -> Self {
        match self {
            Self::TH0 => Self::TH0,
            Self::TH1 => Self::TH2,
            Self::TH2 => Self::TH1,
        }
    }
}

impl Display for SpinHalf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Down => " ",
            Self::Up => "█",
        })
    }
}

impl Display for SpinOne {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Down => "░",
            Self::Zero => " ",
            Self::Up => "█",
        })
    }
}

impl Display for Binary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Inactive => " ",
            Self::Active => "█",
        })
    }
}

impl Display for Brass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::TH0 => " ",
            Self::TH1 => "█",
            Self::TH2 => "░",
        })
    }
}

impl<const Q: usize> Display for PottsState<Q> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            char::from_digit(self.0 as u32 % 36, 36).unwrap_or('?')
        )
    }
}

impl<const Q: usize> Display for ClockState<Q> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            char::from_digit(self.0 as u32 % 36, 36).unwrap_or('?')
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_indices<S: Site>() {
        assert_eq!(S::COUNT, S::VALUES.len());
        for (i, &s) in S::VALUES.iter().enumerate() {
            assert_eq!(s.index(), i);
            assert_eq!(S::from_index(i), s);
        }
    }

    #[test]
    fn indices_are_consistent() {
        check_indices::<SpinHalf>();
        check_indices::<SpinOne>();
        check_indices::<PottsState<3>>();
        check_indices::<PottsState<256>>();
        check_indices::<ClockState<6>>();
        check_indices::<Binary>();
        check_indices::<Brass>();
    }

    #[test]
    fn flips_negate_values() {
        fn check<S: Flip>() {
            for &s in S::VALUES {
                assert_eq!(s.flipped().value(), -s.value());
                assert_eq!(s.flipped().flipped(), s);
            }
        }
        check::<SpinHalf>();
        check::<SpinOne>();
        check::<Brass>();
    }

    #[test]
    fn clock_angles() {
        assert_eq!(ClockState::<4>::new(0).angle(), 0.0);
        approx::assert_relative_eq!(ClockState::<4>::new(1).angle(), std::f64::consts::FRAC_PI_2);
    }
}
