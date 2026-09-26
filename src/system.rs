//! Spin systems: a configuration together with a Hamiltonian.

use rand::Rng;

use crate::{
    model::Model,
    state::{Configuration, Prepare},
};

/// Configuration `St` evolving under the Hamiltonian `M`.
///
/// The total energy is tracked incrementally by the dynamics, so measuring it is `O(1)`. The
/// update kernel for the last used inverse temperature is cached.
#[derive(Debug, Clone)]
pub struct SpinSystem<St, M: Model<St>> {
    state: St,
    model: M,
    energy: f64,
    kernel: Option<(f64, M::Kernel)>,
}

impl<St, M: Model<St>> SpinSystem<St, M> {
    /// System in configuration `state` with Hamiltonian `model`.
    pub fn new(state: St, model: M) -> Self {
        let energy = model.energy(&state);
        Self {
            state,
            model,
            energy,
            kernel: None,
        }
    }

    /// Current configuration.
    #[inline(always)]
    pub fn state(&self) -> &St {
        &self.state
    }

    /// Hamiltonian.
    #[inline(always)]
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Total energy of the current configuration.
    #[inline(always)]
    pub fn energy(&self) -> f64 {
        self.energy
    }

    /// Replace the Hamiltonian.
    pub fn set_model(&mut self, model: M) {
        self.model = model;
        self.kernel = None;
        self.refresh_energy();
    }

    /// Modify the configuration arbitrarily; the energy is recomputed afterwards.
    pub fn modify_state<U>(&mut self, f: impl FnOnce(&mut St) -> U) -> U {
        let out = f(&mut self.state);
        self.refresh_energy();
        out
    }

    /// Recompute the total energy from scratch, discarding accumulated rounding errors.
    pub fn refresh_energy(&mut self) {
        self.energy = self.model.energy(&self.state);
    }

    /// Decompose into configuration and Hamiltonian.
    pub fn into_parts(self) -> (St, M) {
        (self.state, self.model)
    }

    /// Mutable access for dynamics: configuration, Hamiltonian, kernel at `beta` and tracked
    /// energy. Any change to the configuration must be reflected in the energy.
    pub(crate) fn parts_mut(&mut self, beta: f64) -> (&mut St, &M, &M::Kernel, &mut f64) {
        if !matches!(&self.kernel, Some((b, _)) if b.to_bits() == beta.to_bits()) {
            self.kernel = Some((beta, self.model.kernel(&self.state, beta)));
        }
        let (_, kernel) = self.kernel.as_ref().expect("kernel was just built");
        (&mut self.state, &self.model, kernel, &mut self.energy)
    }
}

impl<St: Configuration, M: Model<St>> Configuration for SpinSystem<St, M> {
    type Site = St::Site;

    #[inline(always)]
    fn len(&self) -> usize {
        self.state.len()
    }

    #[inline(always)]
    fn counts(&self) -> &[u32] {
        self.state.counts()
    }
}

impl<St, M: Model<St>, P: Prepare<St>> Prepare<SpinSystem<St, M>> for P {
    fn prepare<R: Rng>(&self, sys: &mut SpinSystem<St, M>, rng: &mut R) {
        sys.modify_state(|state| self.prepare(state, rng));
    }
}
