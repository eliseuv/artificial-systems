//! Contact Process Measurements
//!

use crate::{
    contact_process::{ContactProcessSystem, state::ContactProcessState},
    method::Measurement,
};

/// Total number of active sites
pub struct TotalActiveSites;

impl<S> Measurement<S> for TotalActiveSites
where
    S: ContactProcessSystem,
{
    type Result = usize;

    fn measure(system: &S) -> Self::Result {
        system.state().total_active()
    }
}

/// Fraction of active sites
pub struct ActiveSites;

impl<S> Measurement<S> for ActiveSites
where
    S: ContactProcessSystem,
{
    type Result = f64;

    fn measure(system: &S) -> Self::Result {
        system.state().active()
    }
}
