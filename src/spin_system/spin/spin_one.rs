use std::fmt::Display;
use std::ops::Neg;
use std::{mem::transmute, ops::Mul};

use rand::Rng;
use rand_distr::{Distribution, StandardUniform};

use crate::spin_system::spin::{Spin, SpinFlip};

/// Spin-`1`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SpinOne {
    Down = -1,
    Zero = 0,
    Up = 1,
}

impl Spin for SpinOne {}

impl From<SpinOne> for i32 {
    fn from(value: SpinOne) -> Self {
        unsafe { transmute(value) }
    }
}

impl Mul for SpinOne {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { transmute((self as i32) * (rhs as i32)) }
    }
}

impl Display for SpinOne {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                SpinOne::Down => '↓',
                SpinOne::Zero => '0',
                SpinOne::Up => '↑',
            }
        )
    }
}

/// Random spin-`1/2` state
impl Distribution<SpinOne> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SpinOne {
        match rng.random_range(-1..=1) {
            -1 => SpinOne::Down,
            0 => SpinOne::Zero,
            1 => SpinOne::Up,
            _ => unreachable!(),
        }
    }
}

impl SpinFlip for SpinOne {
    #[inline(always)]
    fn flip(&mut self) {
        *self = unsafe { transmute::<i32, Self>(i32::from(*self).neg()) };
    }
}
