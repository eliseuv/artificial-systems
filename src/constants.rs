//! Known critical points (unit couplings).

/// Critical temperature of the square lattice Ising model, `T_c = 2 / ln(1 + √2)` (Onsager).
pub const ISING_SQUARE_T_CRITICAL: f64 = 2.269_185_314_213_022;

/// Critical inverse temperature of the square lattice Ising model, `β_c = ln(1 + √2) / 2`.
pub const ISING_SQUARE_BETA_CRITICAL: f64 = 0.440_686_793_509_771_5;

/// Critical temperature of the simple cubic Ising model (Monte Carlo estimate).
pub const ISING_CUBIC_T_CRITICAL: f64 = 4.511_523_2;

/// Critical infection rate of the one-dimensional contact process, `α_c ≈ 3.29785(2)`.
pub const CONTACT_PROCESS_CHAIN_ALPHA_CRITICAL: f64 = 3.297_85;

/// Critical inverse temperature `β_c = 1 / (zJ)` of the mean-field Ising model with effective
/// coordination number `z`.
pub fn ising_mean_field_beta_critical(z: f64, j: f64) -> f64 {
    1.0 / (z * j)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onsager_values() {
        approx::assert_relative_eq!(
            ISING_SQUARE_T_CRITICAL,
            2.0 / (1.0 + 2f64.sqrt()).ln(),
            epsilon = 1e-15
        );
        approx::assert_relative_eq!(
            ISING_SQUARE_BETA_CRITICAL * ISING_SQUARE_T_CRITICAL,
            1.0,
            epsilon = 1e-15
        );
    }
}
