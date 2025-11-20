//! Spin-`1/2`
//!

use std::{
    fmt::Display,
    mem::transmute,
    ops::{Mul, Neg},
};

use rand::Rng;
use rand_distr::{Distribution, StandardUniform};

use crate::spin_system::spin::{Spin, SpinFlip};

/// Spin-`1/2`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SpinHalf {
    Down = -1,
    Up = 1,
}

impl SpinHalf {
    #[inline(always)]
    pub fn flipped(&self) -> Self {
        unsafe { transmute((*self as i32).neg()) }
    }
}

impl Spin for SpinHalf {}

impl From<SpinHalf> for i32 {
    fn from(value: SpinHalf) -> Self {
        unsafe { transmute(value) }
    }
}

impl Mul for SpinHalf {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { transmute((self as i32) * (rhs as i32)) }
    }
}

impl Display for SpinHalf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                // SpinHalf::Down => '↓',
                // SpinHalf::Up => '↑',
                SpinHalf::Down => ' ',
                SpinHalf::Up => '█',
            }
        )
    }
}

impl SpinFlip for SpinHalf {
    #[inline(always)]
    fn flip(&mut self) {
        *self = unsafe { transmute::<i32, Self>(i32::from(*self).neg()) };
    }
}

/// Random spin-`1/2` state
impl Distribution<SpinHalf> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SpinHalf {
        match rng.random() {
            true => SpinHalf::Up,
            false => SpinHalf::Down,
        }
    }
}
