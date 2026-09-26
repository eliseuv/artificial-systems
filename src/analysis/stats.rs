//! Summary statistics, fits and reference distributions.

use std::f64::consts::PI;

use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};

/// Streaming mean and variance (Welford), mergeable across threads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Moments {
    count: u64,
    mean: f64,
    m2: f64,
}

impl Moments {
    /// Empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `x`.
    #[inline]
    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (x - self.mean);
    }

    /// Combine with another accumulator (Chan et al.).
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        let count = self.count + other.count;
        let delta = other.mean - self.mean;
        self.mean += delta * other.count as f64 / count as f64;
        self.m2 +=
            other.m2 + delta * delta * (self.count as f64 * other.count as f64) / count as f64;
        self.count = count;
    }

    /// Number of values.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Mean (`NaN` when empty).
    pub fn mean(&self) -> f64 {
        if self.count == 0 { f64::NAN } else { self.mean }
    }

    /// Variance with `count - ddof` degrees of freedom.
    pub fn variance(&self, ddof: u64) -> f64 {
        self.m2 / (self.count as f64 - ddof as f64)
    }
}

impl FromIterator<f64> for Moments {
    fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        let mut m = Self::new();
        iter.into_iter().for_each(|x| m.push(x));
        m
    }
}

/// Ordinary least squares fit `y = slope · x + intercept`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LinearFit {
    /// Slope.
    pub slope: f64,
    /// Intercept.
    pub intercept: f64,
    /// Coefficient of determination `R² = 1 - SS_res / SS_tot`.
    pub r_squared: f64,
    /// Number of points used.
    pub points: usize,
}

/// Least squares line through the points `(x, y)`, if there are at least two distinct `x`.
pub fn linear_fit(x: &[f64], y: &[f64]) -> Option<LinearFit> {
    assert_eq!(x.len(), y.len(), "Mismatched fit data");
    let n = x.len();
    if n < 2 {
        return None;
    }
    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;
    let (sxx, sxy, syy) = x
        .iter()
        .zip(y)
        .fold((0.0, 0.0, 0.0), |(sxx, sxy, syy), (a, b)| {
            let (dx, dy) = (a - mean_x, b - mean_y);
            (sxx + dx * dx, sxy + dx * dy, syy + dy * dy)
        });
    if sxx == 0.0 {
        return None;
    }
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    let ss_res = syy - slope * sxy;
    let r_squared = if syy == 0.0 { 1.0 } else { 1.0 - ss_res / syy };
    Some(LinearFit {
        slope,
        intercept,
        r_squared,
        points: n,
    })
}

/// Power law fit `y = A tᶿ` as a linear fit of `ln y` against `ln t`.
///
/// Points with `t < t_min` or non-positive values are skipped (e.g. `t = 0` or an absorbed
/// ensemble average). The slope is the exponent `θ` and `R²` measures how close the data is to a
/// power law, which is maximal at criticality.
pub fn power_law_fit(t: &[f64], y: &[f64], t_min: f64) -> Option<LinearFit> {
    assert_eq!(t.len(), y.len(), "Mismatched fit data");
    let (lx, ly): (Vec<f64>, Vec<f64>) = t
        .iter()
        .zip(y)
        .filter(|&(&t, &y)| t >= t_min && t > 0.0 && y > 0.0)
        .map(|(t, y)| (t.ln(), y.ln()))
        .unzip();
    linear_fit(&lx, &ly)
}

/// Average over samples (rows) at each time (column), e.g. `⟨ρ(t)⟩` from a time series matrix.
pub fn column_means(x: ArrayView2<f64>) -> Vec<f64> {
    let n = x.nrows() as f64;
    x.columns().into_iter().map(|c| c.sum() / n).collect()
}

/// Dynamic exponent `z` from the short time relaxation `F₂(t) = ⟨M²⟩ / ⟨M⟩² ~ t^{d/z}`, where
/// `⟨M²⟩` comes from `random_start` series and `⟨M⟩` from `ordered_start` series (rows are
/// samples, columns times `t = 0, 1, …`). Returns `z` and the fit.
pub fn dynamic_exponent(
    dimension: usize,
    random_start: ArrayView2<f64>,
    ordered_start: ArrayView2<f64>,
    t_min: f64,
) -> Option<(f64, LinearFit)> {
    let second: Vec<f64> = column_means(random_start.mapv(|m| m * m).view());
    let first = column_means(ordered_start);
    let f2: Vec<f64> = second
        .iter()
        .zip(&first)
        .map(|(s, f)| s / (f * f))
        .collect();
    let t: Vec<f64> = (0..f2.len()).map(|t| t as f64).collect();
    let fit = power_law_fit(&t, &f2, t_min)?;
    Some((dimension as f64 / fit.slope, fit))
}

/// Gaps between consecutive (sorted) eigenvalues.
pub fn spacings(sorted: &[f64]) -> Vec<f64> {
    sorted.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Spectral entropy `-Σ pᵢ ln pᵢ / ln n` with `pᵢ = λᵢ / Σλ` (one for a flat spectrum).
pub fn spectral_entropy(eigenvalues: &[f64]) -> f64 {
    let total: f64 = eigenvalues.iter().sum();
    let h: f64 = eigenvalues
        .iter()
        .map(|l| l / total)
        .filter(|&p| p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    h / (eigenvalues.len() as f64).ln()
}

/// Marchenko-Pastur law for the spectrum of `G = X Xᵀ / n_steps` with `X` an
/// `n_samples × n_steps` matrix of independent standardised entries.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MarchenkoPastur {
    /// Ratio `q = n_samples / n_steps`.
    pub q: f64,
}

impl MarchenkoPastur {
    /// Law for `n_samples` series of length `n_steps`.
    pub fn new(n_samples: usize, n_steps: usize) -> Self {
        Self {
            q: n_samples as f64 / n_steps as f64,
        }
    }

    /// Support `[λ₋, λ₊]` with `λ± = (1 ± √q)²`.
    pub fn bounds(&self) -> (f64, f64) {
        let s = self.q.sqrt();
        ((1.0 - s).powi(2), (1.0 + s).powi(2))
    }

    /// Density `√((λ₊ - λ)(λ - λ₋)) / (2π q λ)` of the continuous part (for `q > 1` there is
    /// additionally a mass `1 - 1/q` at zero).
    pub fn density(&self, lambda: f64) -> f64 {
        let (lo, hi) = self.bounds();
        if lambda <= lo || lambda >= hi || lambda <= 0.0 {
            0.0
        } else {
            ((hi - lambda) * (lambda - lo)).sqrt() / (2.0 * PI * self.q * lambda)
        }
    }

    /// Mean eigenvalue (one).
    pub fn mean(&self) -> f64 {
        1.0
    }

    /// Eigenvalue variance (`q`).
    pub fn variance(&self) -> f64 {
        self.q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_matches_two_pass_and_merges() {
        let data: Vec<f64> = (0..1000).map(|k| ((k * 37) % 101) as f64 * 0.3).collect();
        let all: Moments = data.iter().copied().collect();
        let mean = data.iter().sum::<f64>() / 1000.0;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 999.0;
        approx::assert_relative_eq!(all.mean(), mean, epsilon = 1e-12);
        approx::assert_relative_eq!(all.variance(1), var, epsilon = 1e-12);
        let mut left: Moments = data[..300].iter().copied().collect();
        left.merge(&data[300..].iter().copied().collect());
        approx::assert_relative_eq!(left.mean(), all.mean(), epsilon = 1e-12);
        approx::assert_relative_eq!(left.variance(0), all.variance(0), epsilon = 1e-10);
    }

    #[test]
    fn power_law_recovers_exponent() {
        let t: Vec<f64> = (0..200).map(|t| t as f64).collect();
        let y: Vec<f64> = t.iter().map(|t| 3.0 * t.powf(-0.159)).collect();
        let fit = power_law_fit(&t, &y, 1.0).unwrap();
        approx::assert_relative_eq!(fit.slope, -0.159, epsilon = 1e-12);
        approx::assert_relative_eq!(fit.intercept.exp(), 3.0, epsilon = 1e-10);
        approx::assert_relative_eq!(fit.r_squared, 1.0, epsilon = 1e-12);
        assert_eq!(fit.points, 199);
        let noisy: Vec<f64> = y
            .iter()
            .enumerate()
            .map(|(k, v)| v * (1.0 + 0.1 * (k as f64).sin()))
            .collect();
        assert!(power_law_fit(&t, &noisy, 1.0).unwrap().r_squared < 0.99);
    }

    #[test]
    fn marchenko_pastur_is_normalised() {
        let mp = MarchenkoPastur::new(100, 300);
        let (lo, hi) = mp.bounds();
        let n = 200_000;
        let h = (hi - lo) / n as f64;
        let (mut mass, mut mean) = (0.0, 0.0);
        for k in 0..n {
            let l = lo + (k as f64 + 0.5) * h;
            mass += mp.density(l) * h;
            mean += l * mp.density(l) * h;
        }
        approx::assert_relative_eq!(mass, 1.0, epsilon = 1e-3);
        approx::assert_relative_eq!(mean, mp.mean(), epsilon = 1e-3);
    }

    #[test]
    fn entropy_and_spacings() {
        approx::assert_relative_eq!(spectral_entropy(&[1.0, 1.0, 1.0, 1.0]), 1.0);
        assert!(spectral_entropy(&[0.0, 0.0, 0.0, 4.0]).abs() < 1e-12);
        assert_eq!(spacings(&[0.0, 1.0, 3.0]), vec![1.0, 2.0]);
    }
}
