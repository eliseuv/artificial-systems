//! Artificial Systems
//!

/// General Utilities
pub mod utils;

/// General maths utilities
pub mod maths;
pub(crate) use maths::hypercube_index;

/// Lattices
pub mod lattice;

/// Hamiltonian systems
pub mod hamiltonian;

/// Markov Chain Monte Carlo
pub mod mcmc;

/// Spin Systems
pub mod spin_system;

/// Contact Process
pub mod contact_process;

/// Time Series Matrices
pub mod method;

/// Data Files Interface
pub mod data_io;
