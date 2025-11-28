//! Abstract Systems Concepts
//!

/// Specification to reset the system
pub trait StateResetSpec<S> {
    fn reset(&mut self, system: &mut S);
}

/// Measurement over a system
pub trait Measurement<S> {
    /// Type of the measured quantity
    type Result;

    /// Perform measurement on system
    fn measure(system: &S) -> Self::Result;
}
