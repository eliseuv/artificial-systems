//! Arbitrary undirected graphs.

use super::{Adjacency, Topology};

/// Arbitrary undirected (multi)graph without self loops.
///
/// Neighbours are listed in the order edges were given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graph {
    adjacency: Adjacency,
}

impl Graph {
    /// Graph with `n` vertices and the given undirected edges.
    ///
    /// # Panics
    /// If an edge refers to a vertex `>= n` or is a self loop.
    pub fn from_edges(n: usize, edges: impl IntoIterator<Item = (usize, usize)>) -> Self {
        let mut lists = vec![Vec::new(); n];
        for (i, j) in edges {
            assert!(i < n && j < n, "Edge ({i}, {j}) out of range for {n} vertices");
            assert_ne!(i, j, "Self loop at vertex {i}");
            lists[i].push(j as u32);
            lists[j].push(i as u32);
        }
        Self {
            adjacency: Adjacency::from_lists(lists),
        }
    }

    /// Graph from explicit, symmetric neighbour lists.
    pub fn from_neighbor_lists(lists: Vec<Vec<u32>>) -> Self {
        Self {
            adjacency: Adjacency::from_lists(lists),
        }
    }

    /// Complete graph on `n` vertices.
    ///
    /// Memory grows as `n²`; for large fully connected systems use
    /// [`MeanFieldState`](crate::state::MeanFieldState) instead.
    pub fn complete(n: usize) -> Self {
        Self::from_edges(n, (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))))
    }
}

impl Topology for Graph {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_graph() {
        let g = Graph::complete(5);
        assert_eq!(g.len(), 5);
        assert!((0..5).all(|i| g.degree(i) == 4));
        assert_eq!(g.bonds().count(), 10);
        assert!(g.bipartition().is_none());
    }

    #[test]
    fn star_graph_is_irregular_and_bipartite() {
        let g = Graph::from_edges(4, [(0, 1), (0, 2), (0, 3)]);
        assert_eq!(g.neighbors(0), &[1, 2, 3]);
        assert_eq!(g.neighbors(3), &[0]);
        assert_eq!(g.max_degree(), 3);
        let colour = g.bipartition().unwrap();
        assert!(colour[1..].iter().all(|&c| c != colour[0]));
    }
}
