//! Artificial Systems
//!

/// General maths utilities
pub mod maths;
pub(crate) use maths::hypercube_index;

/// Lattices
pub mod lattice;

/// Hamiltonian systems
pub mod hamiltonian;

/// Spin Systems
pub mod spin_system;

/// Markov Chain Monte Carlo
pub mod mcmc;
