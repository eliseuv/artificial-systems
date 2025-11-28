//! Artificial Systems
//!

/// General maths utilities
pub mod maths;
pub(crate) use maths::hypercube_index;

/// Abstract concepts across all systems
pub mod systems;

/// Lattices
pub mod lattice;

/// Hamiltonian systems
pub mod hamiltonian;

/// Markov Chain Monte Carlo
pub mod mcmc;

/// Cellular Automata
pub mod cellular_automaton;

/// Spin Systems
pub mod spin_system;
