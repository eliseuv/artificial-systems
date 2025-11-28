use crate::{spin_system::SpinSystem, systems::Measurement};

/// Total Magnetization of the system
pub struct TotalMagnetization;

impl<S> Measurement<S> for TotalMagnetization
where
    S: SpinSystem,
{
    type Result = i32;

    #[inline(always)]
    fn measure(system: &S) -> Self::Result {
        system.total_magnet()
    }
}

/// Magnetization of the system
pub struct Magnetization;

impl<S> Measurement<S> for Magnetization
where
    S: SpinSystem,
{
    type Result = f64;

    #[inline(always)]
    fn measure(system: &S) -> Self::Result {
        system.magnet()
    }
}
