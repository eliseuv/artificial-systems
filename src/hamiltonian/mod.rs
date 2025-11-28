use std::ops::Neg;

use num_traits::Num;

use crate::systems::Measurement;

pub trait HamiltonianSystem {
    /// Type of the Hamiltonian
    type H: Copy + Num + Neg<Output = Self::H> + From<i32> + Into<f64>;

    /// Hamiltonian function
    fn hamiltonian(&self) -> Self::H;
}

/// Total system energy
pub struct TotalEnergy;

impl<S> Measurement<S> for TotalEnergy
where
    S: HamiltonianSystem,
{
    type Result = S::H;

    #[inline(always)]
    fn measure(system: &S) -> Self::Result {
        system.hamiltonian()
    }
}
