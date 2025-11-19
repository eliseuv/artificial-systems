use std::ops::Neg;

use num_traits::Num;

pub trait HamiltonianSystem {
    /// Type of the Hamiltonian
    type H: Copy + Num + Neg<Output = Self::H> + From<i32> + Into<f64>;

    /// Hamiltonian function
    fn hamiltonian(&self) -> Self::H;
}
