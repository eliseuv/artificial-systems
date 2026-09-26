//! Random matrix analysis of time series matrices.
//!
//! Given a time series matrix `X` with one series per row (shape `(n_samples, n_steps)`, as
//! produced by [`Ensemble::matrix`](crate::ensemble::Ensemble::matrix)):
//! 1. [`standardize_rows`]: `x*ₖₜ = (xₖₜ - ⟨xₖ⟩) / σₖ` with moments averaged over time.
//! 2. [`correlation_matrix`]: `G* = X* X*ᵀ / n_steps`, the `n_samples × n_samples` Wishart
//!    correlation matrix of the series.
//! 3. [`eigenvalues`] of `G*`, whose density is compared with the [`MarchenkoPastur`] law of
//!    uncorrelated series and summarised through [`Histogram`] moments and [`Moments`] of the
//!    extreme eigenvalues.

use faer::{
    Accum, MatRef, Par,
    diag::Diag,
    dyn_stack::{MemBuffer, MemStack},
    linalg::{
        evd::{ComputeEigenvectors, self_adjoint_evd, self_adjoint_evd_scratch},
        matmul::matmul,
    },
};
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

mod histogram;
mod stats;
pub mod toy;

pub use histogram::{BinPosition, Histogram};
pub use stats::{
    LinearFit, MarchenkoPastur, Moments, column_means, dynamic_exponent, linear_fit, power_law_fit,
    spacings, spectral_entropy,
};

/// Errors of the analysis routines.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AnalysisError {
    /// A time series is constant, so it cannot be standardised.
    #[error("time series {row} has zero variance")]
    ZeroVariance {
        /// Row of the offending series.
        row: usize,
    },
    /// The eigenvalue decomposition did not converge.
    #[error("eigenvalue decomposition did not converge")]
    NoConvergence,
}

/// Treatment of constant time series (e.g. contact process samples that reached the absorbing
/// state before the first measurement) during standardisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[serde(rename_all = "lowercase")]
pub enum ZeroVariance {
    /// Replace the series by zeros (uncorrelated with everything, contributing a zero eigenvalue).
    #[default]
    Zero,
    /// Fail with [`AnalysisError::ZeroVariance`].
    Error,
}

/// Standardise each row to zero mean and unit variance, with `σ² = Σₜ (xₜ - ⟨x⟩)² / (n - ddof)`.
///
/// `ddof = 0` (population variance) makes the diagonal of [`correlation_matrix`] exactly one.
/// A series whose standard deviation is below rounding level relative to its mean is constant.
pub fn standardize_rows(
    x: &mut Array2<f64>,
    ddof: usize,
    zero: ZeroVariance,
) -> Result<(), AnalysisError> {
    let n = x.ncols();
    assert!(n > ddof, "Series must be longer than ddof");
    for (row, mut series) in x.rows_mut().into_iter().enumerate() {
        let mean = series.sum() / n as f64;
        let var = series.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - ddof) as f64;
        let std = var.sqrt();
        if std <= 64.0 * f64::EPSILON * mean.abs() || std == 0.0 {
            match zero {
                ZeroVariance::Zero => series.fill(0.0),
                ZeroVariance::Error => return Err(AnalysisError::ZeroVariance { row }),
            }
        } else {
            series.mapv_inplace(|v| (v - mean) / std);
        }
    }
    Ok(())
}

/// Parallelism of a single faer operation.
///
/// Parallel only outside rayon workers: the ensemble routines already analyse one matrix per
/// worker, and nesting faer's parallelism inside them would oversubscribe the pool.
fn linalg_par() -> Par {
    #[cfg(feature = "parallel-linalg")]
    if rayon::current_thread_index().is_none() {
        return Par::rayon(0);
    }
    Par::Seq
}

/// Correlation matrix `G = X Xᵀ / n_steps` of (standardised) series in the rows of `x`.
pub fn correlation_matrix(x: ArrayView2<f64>) -> Array2<f64> {
    let (n_samples, n_steps) = x.dim();
    let x = x.as_standard_layout();
    let slice = x.as_slice().expect("standard layout");
    let x = MatRef::from_row_major_slice(slice, n_samples, n_steps);
    let mut g = faer::Mat::<f64>::zeros(n_samples, n_samples);
    matmul(
        g.as_mut(),
        Accum::Replace,
        x,
        x.transpose(),
        1.0 / n_steps as f64,
        linalg_par(),
    );
    Array2::from_shape_fn((n_samples, n_samples), |(i, j)| {
        // Exact symmetry regardless of accumulation order
        if i <= j { g[(i, j)] } else { g[(j, i)] }
    })
}

/// Eigenvalues of the symmetric matrix `g`, in ascending order.
pub fn eigenvalues(g: ArrayView2<f64>) -> Result<Vec<f64>, AnalysisError> {
    let n = g.nrows();
    assert_eq!(n, g.ncols(), "Matrix must be square");
    let g = g.as_standard_layout();
    let a = MatRef::from_row_major_slice(g.as_slice().expect("standard layout"), n, n);
    let mut s = Diag::<f64>::zeros(n);
    let par = linalg_par();
    let mut buffer = MemBuffer::new(self_adjoint_evd_scratch::<f64>(
        n,
        ComputeEigenvectors::No,
        par,
        Default::default(),
    ));
    self_adjoint_evd(
        a,
        s.as_mut(),
        None,
        par,
        MemStack::new(&mut buffer),
        Default::default(),
    )
    .map_err(|_| AnalysisError::NoConvergence)?;
    let mut values: Vec<f64> = s.column_vector().iter().copied().collect();
    values.sort_by(f64::total_cmp);
    Ok(values)
}

/// Eigenvalues (ascending) of the correlation matrix of the time series in the rows of `x`.
pub fn correlation_spectrum(
    x: ArrayView2<f64>,
    ddof: usize,
    zero: ZeroVariance,
) -> Result<Vec<f64>, AnalysisError> {
    let mut x = x.to_owned();
    standardize_rows(&mut x, ddof, zero)?;
    eigenvalues(correlation_matrix(x.view()).view())
}

/// Entries of the upper triangle of `g`, row by row, with (`strict = false`) or without the
/// diagonal.
pub fn upper_triangle(g: ArrayView2<f64>, strict: bool) -> Vec<f64> {
    let n = g.nrows();
    let offset = usize::from(strict);
    (0..n)
        .flat_map(|i| (i + offset..n).map(move |j| (i, j)))
        .map(|(i, j)| g[[i, j]])
        .collect()
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use rand::RngExt as _;
    use rand_distr::StandardNormal;

    use super::*;
    use crate::rng::stream;

    #[test]
    fn standardization() {
        let mut x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]];
        standardize_rows(&mut x, 0, ZeroVariance::Zero).unwrap();
        let row = x.row(0);
        approx::assert_abs_diff_eq!(row.sum(), 0.0, epsilon = 1e-12);
        approx::assert_relative_eq!(row.mapv(|v| v * v).sum() / 4.0, 1.0);
        assert!(x.row(1).iter().all(|&v| v == 0.0));
        let mut y = array![[0.1, 0.1, 0.1]];
        assert_eq!(
            standardize_rows(&mut y, 0, ZeroVariance::Error),
            Err(AnalysisError::ZeroVariance { row: 0 })
        );
    }

    #[test]
    fn correlation_of_standardized_series_has_unit_diagonal() {
        let mut rng = stream(0, &[]);
        let mut x = Array2::from_shape_fn((20, 300), |_| rng.sample::<f64, _>(StandardNormal));
        standardize_rows(&mut x, 0, ZeroVariance::Zero).unwrap();
        let g = correlation_matrix(x.view());
        for i in 0..20 {
            approx::assert_relative_eq!(g[[i, i]], 1.0, epsilon = 1e-12);
            for j in 0..20 {
                assert_eq!(g[[i, j]], g[[j, i]]);
            }
        }
        let eig = eigenvalues(g.view()).unwrap();
        assert!(eig.windows(2).all(|w| w[0] <= w[1]));
        approx::assert_relative_eq!(eig.iter().sum::<f64>(), 20.0, epsilon = 1e-9);
    }

    #[test]
    fn eigenvalues_of_known_matrix() {
        let g = array![[2.0, 1.0], [1.0, 2.0]];
        let eig = eigenvalues(g.view()).unwrap();
        approx::assert_relative_eq!(eig[0], 1.0, epsilon = 1e-12);
        approx::assert_relative_eq!(eig[1], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn spectrum_independent_of_calling_thread() {
        use rayon::prelude::*;
        let mut rng = stream(2, &[]);
        let x = Array2::from_shape_fn((300, 600), |_| rng.sample::<f64, _>(StandardNormal));
        let spectrum = || correlation_spectrum(x.view(), 0, ZeroVariance::Zero).unwrap();
        let outside = spectrum();
        let inside = (0..2)
            .into_par_iter()
            .map(|_| spectrum())
            .collect::<Vec<_>>();
        assert!(rayon::current_thread_index().is_none());
        for spectrum in inside {
            for (a, b) in outside.iter().zip(&spectrum) {
                approx::assert_relative_eq!(a, b, epsilon = 1e-10, max_relative = 1e-10);
            }
        }
    }

    #[test]
    fn triangles() {
        let g = array![[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]];
        assert_eq!(upper_triangle(g.view(), true), vec![2.0, 3.0, 5.0]);
        assert_eq!(
            upper_triangle(g.view(), false),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }
}
