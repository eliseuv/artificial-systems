use crate::spin_system::SpinSystem;

/// Measurement over a spin system
pub trait Measurement<S: SpinSystem> {
    type Result;

    fn measure(system: &S) -> Self::Result;
}

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

/// Total system energy
pub struct TotalEnergy;

impl<S> Measurement<S> for TotalEnergy
where
    S: SpinSystem,
{
    type Result = S::H;

    #[inline(always)]
    fn measure(system: &S) -> Self::Result {
        system.hamiltonian()
    }
}
