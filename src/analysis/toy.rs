//! Toy model: Gaussian time series with prescribed pairwise correlation.

use ndarray::Array2;
use rand::{Rng, RngExt as _};
use rand_distr::StandardNormal;

/// Time series matrix with `2 n_pairs` rows of length `n_steps` of standard normal values, where
/// rows `2k` and `2k + 1` have correlation `rho` and all other pairs are independent.
///
/// With `θ = ½ asin ρ` and independent `φ₁, φ₂`, the pair is `(φ₁ sin θ + φ₂ cos θ,
/// φ₁ cos θ + φ₂ sin θ)`, which has unit variances and covariance `sin 2θ = ρ`. The expected
/// correlation matrix is block diagonal with eigenvalues `1 ± ρ`.
///
/// # Panics
/// If `rho` is not in `[-1, 1]`.
pub fn correlated_pairs<R: Rng + ?Sized>(
    rho: f64,
    n_steps: usize,
    n_pairs: usize,
    rng: &mut R,
) -> Array2<f64> {
    assert!(
        (-1.0..=1.0).contains(&rho),
        "Correlation must be in [-1, 1]"
    );
    let (sin, cos) = (0.5 * rho.asin()).sin_cos();
    let mut x = Array2::zeros((2 * n_pairs, n_steps));
    for k in 0..n_pairs {
        for t in 0..n_steps {
            let a: f64 = rng.sample(StandardNormal);
            let b: f64 = rng.sample(StandardNormal);
            x[[2 * k, t]] = a * sin + b * cos;
            x[[2 * k + 1, t]] = a * cos + b * sin;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::{
            MarchenkoPastur, Moments, ZeroVariance, correlation_matrix, correlation_spectrum,
            eigenvalues, standardize_rows,
        },
        rng::stream,
    };

    #[test]
    fn pairs_have_prescribed_correlation() {
        let mut rng = stream(0, &[]);
        let mut x = correlated_pairs(0.6, 20_000, 2, &mut rng);
        standardize_rows(&mut x, 0, ZeroVariance::Zero).unwrap();
        let g = correlation_matrix(x.view());
        approx::assert_abs_diff_eq!(g[[0, 1]], 0.6, epsilon = 0.02);
        approx::assert_abs_diff_eq!(g[[2, 3]], 0.6, epsilon = 0.02);
        approx::assert_abs_diff_eq!(g[[0, 2]], 0.0, epsilon = 0.02);
        let eig = eigenvalues(g.view()).unwrap();
        approx::assert_abs_diff_eq!(eig[0], 0.4, epsilon = 0.03);
        approx::assert_abs_diff_eq!(eig[3], 1.6, epsilon = 0.03);
    }

    #[test]
    fn uncorrelated_spectrum_follows_marchenko_pastur() {
        let mut rng = stream(1, &[]);
        let (n_samples, n_steps) = (100, 300);
        let mp = MarchenkoPastur::new(n_samples, n_steps);
        let (lo, hi) = mp.bounds();
        let mut moments = Moments::new();
        for _ in 0..20 {
            let x = correlated_pairs(0.0, n_steps, n_samples / 2, &mut rng);
            let eig = correlation_spectrum(x.view(), 0, ZeroVariance::Zero).unwrap();
            // Population standardisation makes the trace, hence the mean eigenvalue, exactly one
            approx::assert_relative_eq!(
                eig.iter().sum::<f64>() / n_samples as f64,
                1.0,
                epsilon = 1e-10
            );
            assert!(eig[0] > lo * 0.8 && eig[n_samples - 1] < hi * 1.1);
            eig.iter().for_each(|&l| moments.push(l));
        }
        // Finite size corrections of the variance are O(1/n_steps)
        approx::assert_relative_eq!(moments.variance(0), mp.variance(), epsilon = 0.05);
    }
}
