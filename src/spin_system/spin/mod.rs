//! Single Spin States
//!

use std::{fmt::Display, ops::Mul};

use crate::lattice::square_lattice::impl_2d::SquareLattice2D;

/// Single Spin State
pub trait Spin: Copy + Into<i32> + Mul<Output = Self> {
    /// Character representation
    fn char(&self) -> char;
}

/// Spin Flip
pub trait SpinFlip: Spin {
    fn flip(&mut self);
}

/// Spin-`1/2`
pub mod spin_half;

/// Spin-`1``
pub mod spin_one;

impl<T: Spin> Display for SquareLattice2D<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in self.rows() {
            for site in row {
                write!(f, "{}", site.char())?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
