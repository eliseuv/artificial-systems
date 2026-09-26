//! Interaction topologies.
//!
//! A topology only describes *which* sites interact. Site states live elsewhere (see
//! [`crate::state`]) and refer to sites by their flat index `0..len`. Neighbour lists are
//! precomputed into a flat table so the hot loops of the dynamics only perform a slice lookup.

use std::{collections::VecDeque, fmt::Debug};

mod graph;
mod hypercubic;

pub use graph::Graph;
pub use hypercubic::{Boundary, Chain, Cubic, Hypercubic, Square};

/// Set of sites together with their neighbourhoods.
///
/// Neighbourhoods are symmetric (with multiplicity): `j` appears `k` times in `neighbors(i)` iff
/// `i` appears `k` times in `neighbors(j)`. Self loops are not allowed.
pub trait Topology: Debug + Send + Sync {
    /// Total number of sites.
    fn len(&self) -> usize;

    /// Whether the topology has no sites.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Indices of the neighbours of site `i`, in a stable, topology-specific order.
    fn neighbors(&self, i: usize) -> &[u32];

    /// Number of neighbours of site `i`.
    #[inline]
    fn degree(&self, i: usize) -> usize {
        self.neighbors(i).len()
    }

    /// Largest number of neighbours of any site.
    fn max_degree(&self) -> usize;

    /// Site considered the "center" of the topology (e.g. for single seed initial states).
    fn center(&self) -> usize {
        self.len() / 2
    }

    /// Every bond `(i, j)` with `i < j` exactly as many times as it appears in the neighbour lists.
    fn bonds(&self) -> impl Iterator<Item = (usize, usize)> + '_
    where
        Self: Sized,
    {
        (0..self.len()).flat_map(move |i| {
            self.neighbors(i)
                .iter()
                .map(|&j| j as usize)
                .filter(move |&j| j > i)
                .map(move |j| (i, j))
        })
    }

    /// Two-colouring of the sites (`true`/`false`) such that neighbours always differ, if one exists.
    fn bipartition(&self) -> Option<Vec<bool>> {
        let n = self.len();
        let mut colour: Vec<Option<bool>> = vec![None; n];
        let mut queue = VecDeque::new();
        for root in 0..n {
            if colour[root].is_some() {
                continue;
            }
            colour[root] = Some(false);
            queue.push_back(root);
            while let Some(i) = queue.pop_front() {
                let c = colour[i].expect("queued sites are coloured");
                for &j in self.neighbors(i) {
                    match colour[j as usize] {
                        None => {
                            colour[j as usize] = Some(!c);
                            queue.push_back(j as usize);
                        }
                        Some(cj) if cj == c => return None,
                        Some(_) => {}
                    }
                }
            }
        }
        Some(colour.into_iter().map(Option::unwrap).collect())
    }
}

/// Flat neighbour table.
///
/// Regular topologies (every site with the same degree) are stored without offsets so the
/// neighbourhood of `i` is simply `targets[i * degree..(i + 1) * degree]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Adjacency {
    len: usize,
    /// Common degree when regular, unused otherwise.
    degree: usize,
    /// CSR offsets (`len + 1` entries) for irregular topologies, empty when regular.
    offsets: Vec<usize>,
    targets: Vec<u32>,
    max_degree: usize,
}

impl Adjacency {
    /// Build from per-site neighbour lists.
    ///
    /// # Panics
    /// If there are more than `u32::MAX` sites, a neighbour index is out of range, a site lists
    /// itself or the lists are not symmetric.
    pub fn from_lists(lists: Vec<Vec<u32>>) -> Self {
        let n = lists.len();
        assert!(u32::try_from(n).is_ok(), "Too many sites for u32 indices: {n}");
        for (i, list) in lists.iter().enumerate() {
            for &j in list {
                assert!((j as usize) < n, "Neighbour {j} of site {i} out of range");
                assert_ne!(j as usize, i, "Self loop at site {i}");
            }
        }
        debug_assert!(
            (0..n).all(|i| lists[i].iter().all(|&j| {
                let forward = lists[i].iter().filter(|&&k| k == j).count();
                let backward = lists[j as usize].iter().filter(|&&k| k as usize == i).count();
                forward == backward
            })),
            "Neighbour lists are not symmetric"
        );
        let max_degree = lists.iter().map(Vec::len).max().unwrap_or(0);
        let regular = lists.iter().all(|l| l.len() == max_degree);
        let offsets = if regular {
            Vec::new()
        } else {
            std::iter::once(0)
                .chain(lists.iter().scan(0, |acc, l| {
                    *acc += l.len();
                    Some(*acc)
                }))
                .collect()
        };
        Self {
            len: n,
            degree: max_degree,
            offsets,
            targets: lists.into_iter().flatten().collect(),
            max_degree,
        }
    }

    /// Number of sites.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no sites.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Neighbours of site `i`.
    #[inline(always)]
    pub fn neighbors(&self, i: usize) -> &[u32] {
        if self.offsets.is_empty() {
            &self.targets[i * self.degree..(i + 1) * self.degree]
        } else {
            &self.targets[self.offsets[i]..self.offsets[i + 1]]
        }
    }

    /// Largest degree.
    #[inline]
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    /// Whether every site has the same degree.
    #[inline]
    pub fn is_regular(&self) -> bool {
        self.offsets.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacency_regular_and_irregular() {
        let regular = Adjacency::from_lists(vec![vec![1, 2], vec![0, 2], vec![0, 1]]);
        assert!(regular.is_regular());
        assert_eq!(regular.len(), 3);
        assert_eq!(regular.neighbors(1), &[0, 2]);

        let path = Adjacency::from_lists(vec![vec![1], vec![0, 2], vec![1]]);
        assert!(!path.is_regular());
        assert_eq!(path.len(), 3);
        assert_eq!(path.neighbors(0), &[1]);
        assert_eq!(path.neighbors(1), &[0, 2]);
        assert_eq!(path.max_degree(), 2);
    }

    #[test]
    #[should_panic(expected = "Self loop")]
    fn adjacency_rejects_self_loops() {
        Adjacency::from_lists(vec![vec![0]]);
    }
}
