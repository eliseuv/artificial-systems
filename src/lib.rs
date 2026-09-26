//! # Artificial Systems
//!
//! High performance simulations of artificial systems: lattice and mean-field spin models,
//! stochastic cellular automata, ensemble time series generation and random matrix analysis
//! of the resulting time series matrices.

#[cfg(feature = "analysis")]
pub mod analysis;
pub mod automaton;
pub mod constants;
pub mod dynamics;
pub mod ensemble;
pub mod model;
pub mod observable;
pub mod rng;
pub mod site;
pub mod state;
pub mod system;
pub mod topology;
