//! Square lattice with Moore (8 neighbour) neighbourhoods.

use super::{Adjacency, Topology};

/// Periodic two-dimensional square lattice where each site interacts with its 8 nearest and
/// next-nearest neighbours, as in Life-like cellular automata.
///
/// Sites are stored in row-major order, like [`Square`](super::Square). The neighbours of a site
/// are listed row by row, top to bottom and left to right, skipping the site itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Moore {
    lengths: [usize; 2],
    adjacency: Adjacency,
}

impl Moore {
    /// Periodic lattice with `lengths = [rows, columns]`.
    ///
    /// # Panics
    /// If a side is shorter than 3, where wrapping would make sites their own or repeated
    /// neighbours.
    pub fn periodic(lengths: [usize; 2]) -> Self {
        let [rows, cols] = lengths;
        assert!(
            rows >= 3 && cols >= 3,
            "Moore lattice sides must be at least 3, got {rows}x{cols}"
        );
        let lists = (0..rows * cols)
            .map(|i| {
                let (r, c) = (i / cols, i % cols);
                let mut list = Vec::with_capacity(8);
                for dr in [rows - 1, 0, 1] {
                    for dc in [cols - 1, 0, 1] {
                        if dr == 0 && dc == 0 {
                            continue;
                        }
                        list.push((((r + dr) % rows) * cols + (c + dc) % cols) as u32);
                    }
                }
                list
            })
            .collect();
        Self {
            lengths,
            adjacency: Adjacency::from_lists(lists),
        }
    }

    /// Side lengths `[rows, columns]`.
    pub fn lengths(&self) -> [usize; 2] {
        self.lengths
    }

    /// Coordinates `[row, column]` of site `i`.
    #[inline]
    pub fn coords(&self, i: usize) -> [usize; 2] {
        [i / self.lengths[1], i % self.lengths[1]]
    }

    /// Flat index of the site at `[row, column]`.
    #[inline]
    pub fn index(&self, [r, c]: [usize; 2]) -> usize {
        debug_assert!(r < self.lengths[0] && c < self.lengths[1]);
        r * self.lengths[1] + c
    }
}

impl Topology for Moore {
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

    #[test]
    fn neighbor_order_and_wrapping() {
        let m = Moore::periodic([3, 4]);
        assert_eq!(m.len(), 12);
        assert!((0..m.len()).all(|i| m.degree(i) == 8));
        let idx = |cs: &[[usize; 2]]| cs.iter().map(|&c| m.index(c) as u32).collect::<Vec<_>>();
        let i = m.index([1, 2]);
        assert_eq!(m.coords(i), [1, 2]);
        let expected = idx(&[
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 1],
            [1, 3],
            [2, 1],
            [2, 2],
            [2, 3],
        ]);
        assert_eq!(m.neighbors(i), expected);
        let expected = idx(&[
            [2, 3],
            [2, 0],
            [2, 1],
            [0, 3],
            [0, 1],
            [1, 3],
            [1, 0],
            [1, 1],
        ]);
        assert_eq!(m.neighbors(m.index([0, 0])), expected);
        assert_eq!(m.bonds().count(), 4 * 12);
    }

    #[test]
    #[should_panic(expected = "at least 3")]
    fn rejects_short_sides() {
        Moore::periodic([2, 5]);
    }
}
