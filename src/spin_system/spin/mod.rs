//! Single Spin States
//!

use std::{fmt::Display, ops::Mul};

/// Single Spin State
pub trait Spin: Copy + Into<i32> + Mul<Output = Self> + Display {}

/// Spin Flip
pub trait SpinFlip: Spin {
    fn flip(&mut self);
}

/// Spin-`1/2`
pub mod spin_half;

/// Spin-`1``
pub mod spin_one;
