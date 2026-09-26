//! Measurements on systems.
//!
//! Macroscopic observables are computed from the per value counts every state keeps up to date,
//! so they cost `O(K)` for `K` site values regardless of the system size. Closures
//! `Fn(&Sys) -> T` are observables too.

use std::f64::consts::TAU;

use crate::{
    model::Model,
    site::{ClockState, Site, Spin},
    state::{Configuration, LatticeState},
    system::SpinSystem,
    topology::Topology,
};

/// Quantity measured on a system of type `Sys`.
pub trait Observable<Sys>: Send + Sync {
    /// Measured value.
    type Output: Clone + Send + Sync + 'static;

    /// Measure on `sys`.
    fn measure(&self, sys: &Sys) -> Self::Output;
}

impl<Sys, F, T> Observable<Sys> for F
where
    F: Fn(&Sys) -> T + Send + Sync,
    T: Clone + Send + Sync + 'static,
{
    type Output = T;

    #[inline]
    fn measure(&self, sys: &Sys) -> T {
        self(sys)
    }
}

/// Magnetisation per site `m = (1/N) Σᵢ sᵢ`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Magnetization;

impl<Sys: Configuration<Site: Spin>> Observable<Sys> for Magnetization {
    type Output = f64;

    #[inline]
    fn measure(&self, sys: &Sys) -> f64 {
        sys.total_magnetization() as f64 / sys.len() as f64
    }
}

/// Absolute magnetisation per site `|m|`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AbsMagnetization;

impl<Sys: Configuration<Site: Spin>> Observable<Sys> for AbsMagnetization {
    type Output = f64;

    #[inline]
    fn measure(&self, sys: &Sys) -> f64 {
        (sys.total_magnetization() as f64 / sys.len() as f64).abs()
    }
}

/// Total magnetisation `M = Σᵢ sᵢ`.
#[derive(Debug, Clone, Copy, Default)]
pub struct TotalMagnetization;

impl<Sys: Configuration<Site: Spin>> Observable<Sys> for TotalMagnetization {
    type Output = i64;

    #[inline]
    fn measure(&self, sys: &Sys) -> i64 {
        sys.total_magnetization()
    }
}

/// Quadrupole moment per site `q = (1/N) Σᵢ sᵢ²` (density of non-zero spins for spin-1).
#[derive(Debug, Clone, Copy, Default)]
pub struct Quadrupole;

impl<Sys: Configuration<Site: Spin>> Observable<Sys> for Quadrupole {
    type Output = f64;

    #[inline]
    fn measure(&self, sys: &Sys) -> f64 {
        sys.total_quadrupole() as f64 / sys.len() as f64
    }
}

/// Fraction of sites in a given state, e.g. the density of active sites.
#[derive(Debug, Clone, Copy)]
pub struct Density<S>(pub S);

impl<Sys: Configuration> Observable<Sys> for Density<Sys::Site> {
    type Output = f64;

    #[inline]
    fn measure(&self, sys: &Sys) -> f64 {
        sys.count(self.0) as f64 / sys.len() as f64
    }
}

/// Number of sites in a given state, e.g. the number of active sites.
#[derive(Debug, Clone, Copy)]
pub struct Count<S>(pub S);

impl<Sys: Configuration> Observable<Sys> for Count<Sys::Site> {
    type Output = u32;

    #[inline]
    fn measure(&self, sys: &Sys) -> u32 {
        sys.count(self.0)
    }
}

/// Potts order parameter `(Q max_q n_q / N - 1) / (Q - 1)`, zero when disordered and one when
/// fully ordered.
#[derive(Debug, Clone, Copy, Default)]
pub struct PottsOrder;

impl<Sys: Configuration> Observable<Sys> for PottsOrder {
    type Output = f64;

    #[inline]
    fn measure(&self, sys: &Sys) -> f64 {
        let q = Sys::Site::COUNT as f64;
        let max = sys.counts().iter().copied().max().unwrap_or(0) as f64;
        (q * max / sys.len() as f64 - 1.0) / (q - 1.0)
    }
}

/// Modulus of the clock magnetisation per site `|(1/N) Σᵢ (cos θᵢ, sin θᵢ)|`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClockMagnetization;

impl<const Q: usize, Sys: Configuration<Site = ClockState<Q>>> Observable<Sys>
    for ClockMagnetization
{
    type Output = f64;

    #[inline]
    fn measure(&self, sys: &Sys) -> f64 {
        let (x, y) = sys
            .counts()
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(x, y), (q, &n)| {
                let theta = TAU * q as f64 / Q as f64;
                (x + n as f64 * theta.cos(), y + n as f64 * theta.sin())
            });
        x.hypot(y) / sys.len() as f64
    }
}

/// Total energy (tracked incrementally).
#[derive(Debug, Clone, Copy, Default)]
pub struct Energy;

impl<St, M: Model<St>> Observable<SpinSystem<St, M>> for Energy {
    type Output = f64;

    #[inline]
    fn measure(&self, sys: &SpinSystem<St, M>) -> f64 {
        sys.energy()
    }
}

/// Energy per site.
#[derive(Debug, Clone, Copy, Default)]
pub struct EnergyPerSite;

impl<St: Configuration, M: Model<St>> Observable<SpinSystem<St, M>> for EnergyPerSite {
    type Output = f64;

    #[inline]
    fn measure(&self, sys: &SpinSystem<St, M>) -> f64 {
        sys.energy() / sys.len() as f64
    }
}

/// Complete configuration, as the spin projection of every site (for state time series).
#[derive(Debug, Clone, Copy, Default)]
pub struct Snapshot;

impl<S: Spin, T: Topology> Observable<LatticeState<S, T>> for Snapshot {
    type Output = Vec<i8>;

    fn measure(&self, state: &LatticeState<S, T>) -> Vec<i8> {
        state.sites().iter().map(|s| s.value() as i8).collect()
    }
}

impl<S: Spin, T: Topology, M: Model<LatticeState<S, T>>>
    Observable<SpinSystem<LatticeState<S, T>, M>> for Snapshot
{
    type Output = Vec<i8>;

    fn measure(&self, sys: &SpinSystem<LatticeState<S, T>, M>) -> Vec<i8> {
        self.measure(sys.state())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        site::{Binary, SpinOne},
        state::MeanFieldState,
        topology::Chain,
    };

    #[test]
    fn macroscopic_observables() {
        let mut st = LatticeState::uniform(Arc::new(Chain::periodic([4])), SpinOne::Up);
        st.set(0, SpinOne::Down);
        st.set(1, SpinOne::Zero);
        assert_eq!(Magnetization.measure(&st), 0.25);
        assert_eq!(TotalMagnetization.measure(&st), 1);
        assert_eq!(Quadrupole.measure(&st), 0.75);
        assert_eq!(Density(SpinOne::Up).measure(&st), 0.5);
        assert_eq!(Count(SpinOne::Zero).measure(&st), 1);
        assert_eq!(PottsOrder.measure(&st), (3.0 * 0.5 - 1.0) / 2.0);
        assert_eq!(Snapshot.measure(&st), vec![-1, 0, 1, 1]);
        let closure = |s: &LatticeState<SpinOne, Chain>| s.get(3);
        assert_eq!(closure.measure(&st), SpinOne::Up);

        let mf = MeanFieldState::<Binary>::from_counts(vec![3, 1], 2.0);
        assert_eq!(Density(Binary::Active).measure(&mf), 0.25);
    }

    #[test]
    fn clock_magnetization() {
        let st = MeanFieldState::<ClockState<4>>::from_counts(vec![2, 0, 0, 0], 4.0);
        approx::assert_relative_eq!(ClockMagnetization.measure(&st), 1.0);
        let st = MeanFieldState::<ClockState<4>>::from_counts(vec![1, 0, 1, 0], 4.0);
        approx::assert_abs_diff_eq!(ClockMagnetization.measure(&st), 0.0, epsilon = 1e-12);
    }
}
