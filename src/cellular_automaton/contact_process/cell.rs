//! Contact Process Cell
//!

use std::fmt::Display;

use rand::Rng;
use rand_distr::{Distribution, StandardUniform};

/// Binary state of a Contact Process
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binary {
    Inactive = 0,
    Active = 1,
}

impl Display for Binary {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Inactive => ' ',
                Self::Active => '█',
            }
        )
    }
}

impl Distribution<Binary> for StandardUniform {
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Binary {
        match rng.random::<bool>() {
            true => Binary::Active,
            false => Binary::Inactive,
        }
    }
}
