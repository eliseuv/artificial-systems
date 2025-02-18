//! System States
//!

use rand::Rng;
use rand_distr::{Bernoulli, Distribution};
use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
};

/// State composed of sites
/// State with shape specified by a variable of type `Shape` composed of a finite number of sites with a given type `Site` and providing:
/// - Indexing of sites for access and modification
/// - Distribution of site indices for selecting random site
/// - Iteration over all sites
/// - Distribution over indices to select random site
pub trait SiteState:
    Index<Self::Index, Output = Self::Site>
    + IndexMut<Self::Index, Output = Self::Site>
    + Distribution<Self::Index> // where
//     UniformSitesDistribution: Distribution<Self::Index>,
{
    /// Shape of the state
    type Shape: Clone + Copy;

    /// Index for sites
    type Index: Clone + Copy;

    /// Type for each site
    type Site: Clone + Copy;

    /// Total number of sites
    fn site_count(&self) -> usize;

    /// Iterator over all sites
    fn sites(&self) -> impl Iterator<Item = &Self::Site>;

    /// Mutable iterator over all sites
    fn sites_mut(&mut self) -> impl Iterator<Item = &mut Self::Site>;

    /// Create new state with sites in the same state
    fn uniform(shape: Self::Shape, site: Self::Site) -> Self;

    /// Create new state with sites states drawn from a random distribution
    fn random<D, R>(shape: Self::Shape, dist: &D, rng: &mut R) -> Self
    where
        D: Distribution<Self::Site>,
        R: Rng + ?Sized;

    /// Set all sites to the same state
    fn set_uniform(&mut self, site: Self::Site);

    /// Set all sites with values drawn from a given distribution
    fn set_random<D: Distribution<Self::Site>, R: Rng + ?Sized>(&mut self, dist: &D, rng: &mut R);

    #[inline(always)]
    fn new<I: InitialStateSpec<Self>>(shape: Self::Shape, spec: &mut I) -> Self
    where
        Self: Sized,
    {
        spec.construct(shape)
    }

    #[inline(always)]
    fn reset<I: InitialStateSpec<Self>>(&mut self, spec: &mut I)
    where
        Self: Sized,
    {
        spec.reset(self);
    }
}

/// Character representation of a site
pub trait SiteCharRepr {
    fn char(&self) -> char;
}

/// Trait for types representing site states reset specifications
pub trait InitialStateSpec<S: SiteState> {
    /// Constructs a new state with given shape with this specification
    fn construct(&mut self, shape: S::Shape) -> S;
    /// Resets a given state with this specification
    fn reset(&mut self, state: &mut S);
}

/// Set all sites uniformly
#[derive(Debug)]
pub struct UniformSites<T: Clone + Copy>(pub T);

impl<S: SiteState<Site = T>, T: Clone + Copy> InitialStateSpec<S> for UniformSites<T> {
    fn construct(&mut self, shape: <S as SiteState>::Shape) -> S {
        S::uniform(shape, self.0)
    }

    fn reset(&mut self, state: &mut S) {
        state.set_uniform(self.0);
    }
}

/// Set all sites to a random state according to a given distribution
#[derive(Debug)]
pub struct RandomSites<'a, T, D, R>
where
    T: Clone + Copy,
    D: Distribution<T>,
    R: Rng + ?Sized,
{
    _site: PhantomData<T>,
    dist: D,
    rng: &'a mut R,
}

impl<'a, T, D, R> RandomSites<'a, T, D, R>
where
    T: Clone + Copy,
    D: Distribution<T>,
    R: Rng + ?Sized,
{
    pub fn with_dist(dist: D, rng: &'a mut R) -> Self {
        Self {
            _site: PhantomData,
            dist,
            rng,
        }
    }

    // pub fn standard(rng: &'a mut R) -> Self
    // where
    //     Standard: Distribution<T>,
    // {
    //     Self::with_dist(Standard, rng);
    // }
}

impl<S, T, D, R> InitialStateSpec<S> for RandomSites<'_, T, D, R>
where
    S: SiteState<Site = T>,
    T: Clone + Copy,
    D: Distribution<T>,
    R: Rng + ?Sized,
{
    fn construct(&mut self, shape: <S as SiteState>::Shape) -> S {
        S::random(shape, &self.dist, self.rng)
    }

    fn reset(&mut self, state: &mut S) {
        state.set_random(&self.dist, self.rng);
    }
}

/// Defines default site reset specification
pub trait DefaultSiteStateReset: SiteState
where
    Self: Sized,
{
    type Spec: InitialStateSpec<Self>;

    #[inline(always)]
    fn reset(&mut self, spec: &mut Self::Spec) {
        spec.reset(self);
    }
}

/// Measurement over a state
pub trait StateMeasurement<S: SiteState> {
    /// Resulting type of the measurement
    type Type;

    /// Perform measurement on a given state
    fn measure(&self, state: &S) -> Self::Type;
}

/// State swap diffusion
pub trait SimpleSwapDiffusion {
    fn diffuse<R: Rng + ?Sized>(&mut self, diffusion_coin: Bernoulli, rng: &mut R);
}

/// States composed of sites with a notion of "nearest neighborhood"
/// - Iteration over all nearest neighbors pairs
/// - Iteration over the nearest neighbors of given site
pub trait SiteStateNN: SiteState {
    /// Iterator over all indices of nearest neighbors pairs
    fn nearest_neighbors_index_pairs(&self) -> impl Iterator<Item = (Self::Index, Self::Index)>;

    /// Iterator over the indices of nearest neighbors of a given site
    fn nearest_neighbors_index(&self, idx: Self::Index) -> impl Iterator<Item = Self::Index>;

    /// Iterator over all nearest neighbors pairs
    fn nearest_neighbors_pairs(&self) -> impl Iterator<Item = (&Self::Site, &Self::Site)>;

    /// Iterator over the nearest neighbors of a given site
    fn nearest_neighbors(&self, idx: Self::Index) -> impl Iterator<Item = &Self::Site>;
}

/// Lattices
pub mod lattices;
