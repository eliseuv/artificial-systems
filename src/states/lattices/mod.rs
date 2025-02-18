//! Lattices
//!

use super::SiteStateNN;

/// Arbitrary lattice
pub trait Lattice: SiteStateNN {}

/// Square Lattices
pub mod square_lattices;
