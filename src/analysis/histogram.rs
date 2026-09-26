//! Fixed width histograms.

use serde::{Deserialize, Serialize};

/// Abscissa representing each bin when computing moments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[serde(rename_all = "kebab-case")]
pub enum BinPosition {
    /// Bin center (unbiased for smooth densities).
    #[default]
    Center,
    /// Left edge, as in the original thesis analysis scripts (biased by `-width/2`).
    LeftEdge,
}

/// Histogram with `n_bins` equal width bins over `[low, high]`.
///
/// Values outside the range are counted separately and ignored by the statistics; the upper edge
/// belongs to the last bin.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Histogram {
    low: f64,
    high: f64,
    counts: Vec<u64>,
    underflow: u64,
    overflow: u64,
}

impl Histogram {
    /// Empty histogram over `[low, high]`.
    ///
    /// # Panics
    /// If the range is empty or not finite, or `n_bins` is zero.
    pub fn new(low: f64, high: f64, n_bins: usize) -> Self {
        assert!(n_bins > 0, "Histogram needs at least one bin");
        assert!(
            low.is_finite() && high.is_finite() && low < high,
            "Invalid histogram range [{low}, {high}]"
        );
        Self {
            low,
            high,
            counts: vec![0; n_bins],
            underflow: 0,
            overflow: 0,
        }
    }

    /// Histogram of `data` over its own range.
    ///
    /// A degenerate range (all values equal, e.g. a spectrum of an absorbed system) is widened
    /// to a unit interval centered on the value.
    ///
    /// # Panics
    /// If `data` is empty or contains non-finite values.
    pub fn from_data(data: &[f64], n_bins: usize) -> Self {
        assert!(!data.is_empty(), "Cannot histogram empty data");
        let (low, high) = data
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                (lo.min(x), hi.max(x))
            });
        let (low, high) = if low < high {
            (low, high)
        } else {
            (low - 0.5, high + 0.5)
        };
        let mut hist = Self::new(low, high, n_bins);
        hist.extend(data.iter().copied());
        hist
    }

    /// Record `x`.
    #[inline]
    pub fn add(&mut self, x: f64) {
        if x < self.low || x.is_nan() {
            self.underflow += 1;
        } else if x > self.high {
            self.overflow += 1;
        } else {
            let n = self.counts.len();
            let k = (((x - self.low) / self.width()) as usize).min(n - 1);
            self.counts[k] += 1;
        }
    }

    /// Record every value of `data`.
    pub fn extend(&mut self, data: impl IntoIterator<Item = f64>) {
        data.into_iter().for_each(|x| self.add(x));
    }

    /// Add the counts of `other`, which must have the same binning.
    ///
    /// # Panics
    /// If the binning differs.
    pub fn merge(&mut self, other: &Self) {
        assert!(
            self.low == other.low && self.high == other.high && self.n_bins() == other.n_bins(),
            "Histograms must have the same binning"
        );
        for (a, b) in self.counts.iter_mut().zip(&other.counts) {
            *a += b;
        }
        self.underflow += other.underflow;
        self.overflow += other.overflow;
    }

    /// Number of bins.
    pub fn n_bins(&self) -> usize {
        self.counts.len()
    }

    /// Bin width.
    pub fn width(&self) -> f64 {
        (self.high - self.low) / self.counts.len() as f64
    }

    /// Bin edges (`n_bins + 1` values).
    pub fn edges(&self) -> Vec<f64> {
        (0..=self.n_bins())
            .map(|k| self.low + k as f64 * self.width())
            .collect()
    }

    /// Bin representatives.
    pub fn positions(&self, position: BinPosition) -> Vec<f64> {
        let shift = match position {
            BinPosition::Center => 0.5,
            BinPosition::LeftEdge => 0.0,
        };
        (0..self.n_bins())
            .map(|k| self.low + (k as f64 + shift) * self.width())
            .collect()
    }

    /// Counts per bin.
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Number of values inside the range.
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// Number of values below and above the range.
    pub fn outliers(&self) -> (u64, u64) {
        (self.underflow, self.overflow)
    }

    /// Probability density per bin (integrates to one over the range).
    pub fn density(&self) -> Vec<f64> {
        let norm = self.total() as f64 * self.width();
        self.counts.iter().map(|&c| c as f64 / norm).collect()
    }

    /// Raw moment `⟨xᵏ⟩ = Σ_b x_bᵏ n_b / Σ_b n_b` from the binned data.
    pub fn moment(&self, k: i32, position: BinPosition) -> f64 {
        let total = self.total() as f64;
        self.positions(position)
            .iter()
            .zip(&self.counts)
            .map(|(x, &c)| x.powi(k) * c as f64)
            .sum::<f64>()
            / total
    }

    /// Mean from the binned data.
    pub fn mean(&self, position: BinPosition) -> f64 {
        self.moment(1, position)
    }

    /// Variance `⟨x²⟩ - ⟨x⟩²` from the binned data (independent of the bin position).
    pub fn variance(&self, position: BinPosition) -> f64 {
        let mean = self.mean(position);
        let total = self.total() as f64;
        self.positions(position)
            .iter()
            .zip(&self.counts)
            .map(|(x, &c)| (x - mean).powi(2) * c as f64)
            .sum::<f64>()
            / total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binning_and_moments() {
        let mut h = Histogram::new(0.0, 4.0, 4);
        h.extend([0.0, 0.5, 1.5, 3.9, 4.0, -1.0, 5.0]);
        assert_eq!(h.counts(), &[2, 1, 0, 2]);
        assert_eq!(h.outliers(), (1, 1));
        assert_eq!(h.edges(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        approx::assert_relative_eq!(
            h.mean(BinPosition::Center),
            (0.5 * 2.0 + 1.5 + 3.5 * 2.0) / 5.0
        );
        approx::assert_relative_eq!(
            h.mean(BinPosition::LeftEdge),
            h.mean(BinPosition::Center) - 0.5
        );
        approx::assert_relative_eq!(
            h.variance(BinPosition::LeftEdge),
            h.variance(BinPosition::Center)
        );
        let density: f64 = h.density().iter().map(|d| d * h.width()).sum();
        approx::assert_relative_eq!(density, 1.0);
    }

    #[test]
    fn fine_histogram_moments_match_data() {
        let data: Vec<f64> = (0..10_000).map(|k| (k as f64 * 0.7).sin() + 1.0).collect();
        let h = Histogram::from_data(&data, 1000);
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        approx::assert_relative_eq!(h.mean(BinPosition::Center), mean, epsilon = 1e-3);
        approx::assert_relative_eq!(h.variance(BinPosition::Center), var, epsilon = 1e-3);
    }

    #[test]
    fn degenerate_data() {
        let h = Histogram::from_data(&[0.0; 10], 5);
        assert_eq!(h.total(), 10);
        assert_eq!(h.mean(BinPosition::Center), 0.0);
    }
}
