//! Square Lattices
//!

use crate::lattice::Lattice;

/// Periodic boundaries
pub mod periodicity;

/// N-dimensional square lattice
pub trait SquareLattice<const N: usize>: Lattice<Shape = usize> {
    /// Get lattice side length
    fn length(&self) -> usize;
}

/// 1D Square Lattice
pub mod impl_1d;

/// 2D Square Lattice
pub mod impl_2d;

/// 3D Square Lattice
pub mod impl_3d;
