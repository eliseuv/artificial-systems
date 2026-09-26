//! Hypercubic lattices of arbitrary dimension.

use super::{Adjacency, Topology};

/// Boundary condition along one axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Boundary {
    /// Opposite faces are neighbours.
    #[default]
    Periodic,
    /// Sites on a face have no neighbour beyond it.
    Open,
}

/// `D`-dimensional hypercubic lattice with nearest neighbour interactions.
///
/// Sites are stored in row-major order (last axis fastest). The neighbours of a site are listed
/// axis by axis, backward before forward: `[-x₀, +x₀, -x₁, +x₁, …]`, omitting those beyond an
/// open boundary. In particular, for a [`Chain`] the neighbours are `[left, right]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hypercubic<const D: usize> {
    lengths: [usize; D],
    boundaries: [Boundary; D],
    strides: [usize; D],
    adjacency: Adjacency,
}

/// One-dimensional lattice.
pub type Chain = Hypercubic<1>;
/// Two-dimensional square lattice.
pub type Square = Hypercubic<2>;
/// Three-dimensional simple cubic lattice.
pub type Cubic = Hypercubic<3>;

impl<const D: usize> Hypercubic<D> {
    /// Lattice with given side lengths and boundary conditions per axis.
    ///
    /// # Panics
    /// If a side length is zero, or a periodic side has length one (which would make a site
    /// its own neighbour).
    pub fn new(lengths: [usize; D], boundaries: [Boundary; D]) -> Self {
        assert!(D > 0, "Lattice dimension must be positive");
        for (d, (&l, &b)) in lengths.iter().zip(&boundaries).enumerate() {
            assert!(l > 0, "Side length along axis {d} must be positive");
            assert!(
                b == Boundary::Open || l > 1,
                "Periodic side along axis {d} must have length > 1"
            );
        }
        let mut strides = [1; D];
        for d in (0..D - 1).rev() {
            strides[d] = strides[d + 1] * lengths[d + 1];
        }
        let n = lengths.iter().product::<usize>();
        let lists = (0..n)
            .map(|i| {
                let c = Self::coords_with(i, &lengths, &strides);
                let mut list = Vec::with_capacity(2 * D);
                for d in 0..D {
                    let (l, s) = (lengths[d], strides[d]);
                    let base = i - c[d] * s;
                    let periodic = boundaries[d] == Boundary::Periodic;
                    if c[d] > 0 {
                        list.push((base + (c[d] - 1) * s) as u32);
                    } else if periodic {
                        list.push((base + (l - 1) * s) as u32);
                    }
                    if c[d] + 1 < l {
                        list.push((base + (c[d] + 1) * s) as u32);
                    } else if periodic {
                        list.push(base as u32);
                    }
                }
                list
            })
            .collect();
        Self {
            lengths,
            boundaries,
            strides,
            adjacency: Adjacency::from_lists(lists),
        }
    }

    /// Periodic lattice with given side lengths.
    pub fn periodic(lengths: [usize; D]) -> Self {
        Self::new(lengths, [Boundary::Periodic; D])
    }

    /// Periodic hypercube with all sides of length `length`.
    pub fn periodic_cube(length: usize) -> Self {
        Self::periodic([length; D])
    }

    /// Open lattice with given side lengths.
    pub fn open(lengths: [usize; D]) -> Self {
        Self::new(lengths, [Boundary::Open; D])
    }

    /// Side lengths.
    #[inline]
    pub fn lengths(&self) -> [usize; D] {
        self.lengths
    }

    /// Boundary conditions.
    #[inline]
    pub fn boundaries(&self) -> [Boundary; D] {
        self.boundaries
    }

    /// Coordinates of site `i`.
    #[inline]
    pub fn coords(&self, i: usize) -> [usize; D] {
        Self::coords_with(i, &self.lengths, &self.strides)
    }

    /// Flat index of the site at `coords`.
    #[inline]
    pub fn index(&self, coords: [usize; D]) -> usize {
        debug_assert!(coords.iter().zip(&self.lengths).all(|(c, l)| c < l));
        coords.iter().zip(&self.strides).map(|(c, s)| c * s).sum()
    }

    #[inline]
    fn coords_with(i: usize, lengths: &[usize; D], strides: &[usize; D]) -> [usize; D] {
        std::array::from_fn(|d| (i / strides[d]) % lengths[d])
    }
}

impl<const D: usize> Topology for Hypercubic<D> {
    #[inline]
    fn len(&self) -> usize {
        self.adjacency.len()
    }

    #[inline(always)]
    fn neighbors(&self, i: usize) -> &[u32] {
        self.adjacency.neighbors(i)
    }

    #[inline]
    fn max_degree(&self) -> usize {
        self.adjacency.max_degree()
    }

    fn center(&self) -> usize {
        self.index(self.lengths.map(|l| l / 2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn chain_neighbors_are_left_right() {
        let chain = Chain::periodic([5]);
        assert_eq!(chain.neighbors(0), &[4, 1]);
        assert_eq!(chain.neighbors(2), &[1, 3]);
        assert_eq!(chain.neighbors(4), &[3, 0]);
        assert_eq!(chain.bonds().count(), 5);
        assert_eq!(chain.center(), 2);
    }

    #[test]
    fn square_neighbor_order() {
        let sq = Square::periodic([3, 4]);
        assert_eq!(sq.len(), 12);
        let i = sq.index([1, 2]);
        assert_eq!(sq.coords(i), [1, 2]);
        let expected = [[0, 2], [2, 2], [1, 1], [1, 3]].map(|c| sq.index(c) as u32);
        assert_eq!(sq.neighbors(i), &expected);
        let corner = sq.index([0, 0]);
        let expected = [[2, 0], [1, 0], [0, 3], [0, 1]].map(|c| sq.index(c) as u32);
        assert_eq!(sq.neighbors(corner), &expected);
    }

    #[test]
    fn open_boundaries_reduce_degree() {
        let sq = Square::open([3, 3]);
        assert_eq!(sq.degree(sq.index([0, 0])), 2);
        assert_eq!(sq.degree(sq.index([0, 1])), 3);
        assert_eq!(sq.degree(sq.index([1, 1])), 4);
        assert_eq!(sq.bonds().count(), 12);
        let chain = Chain::open([1]);
        assert_eq!(chain.degree(0), 0);
    }

    #[test]
    fn length_two_periodic_counts_double_bonds() {
        let chain = Chain::periodic([2]);
        assert_eq!(chain.neighbors(0), &[1, 1]);
        assert_eq!(chain.bonds().count(), 2);
    }

    #[test]
    fn bipartition_depends_on_parity() {
        assert!(Square::periodic([4, 6]).bipartition().is_some());
        assert!(Square::periodic([3, 4]).bipartition().is_none());
        assert!(Square::open([3, 3]).bipartition().is_some());
    }

    proptest! {
        #[test]
        fn neighbourhoods_are_symmetric(
            lx in 2usize..6, ly in 1usize..6, lz in 1usize..5,
            bx: bool, by: bool, bz: bool,
        ) {
            let b = |open: bool, l: usize| if open || l == 1 { Boundary::Open } else { Boundary::Periodic };
            let lat = Cubic::new([lx, ly, lz], [b(bx, lx), b(by, ly), b(bz, lz)]);
            prop_assert_eq!(lat.len(), lx * ly * lz);
            for i in 0..lat.len() {
                prop_assert_eq!(lat.coords(lat.index(lat.coords(i))), lat.coords(i));
                for &j in lat.neighbors(i) {
                    let fwd = lat.neighbors(i).iter().filter(|&&k| k == j).count();
                    let bwd = lat.neighbors(j as usize).iter().filter(|&&k| k as usize == i).count();
                    prop_assert_eq!(fwd, bwd);
                    // Neighbours differ in exactly one coordinate
                    let (ci, cj) = (lat.coords(i), lat.coords(j as usize));
                    prop_assert_eq!(ci.iter().zip(&cj).filter(|(a, b)| a != b).count(), 1);
                }
            }
            let degree_sum: usize = (0..lat.len()).map(|i| lat.degree(i)).sum();
            prop_assert_eq!(lat.bonds().count() * 2, degree_sum);
        }
    }
}
