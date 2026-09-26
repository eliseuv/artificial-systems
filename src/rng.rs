//! Random number generation and reproducible stream derivation.
//!
//! Every Markov chain in an ensemble owns an independent generator derived from a master seed
//! and the chain's coordinates (e.g. run and sample index). Results are therefore independent
//! of scheduling and bit-identical for any number of threads.

use rand::{RngExt as _, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Default pseudo random number generator.
pub type DefaultRng = Xoshiro256PlusPlus;

/// SplitMix64 finaliser, a bijective 64-bit mixer with good avalanche properties.
#[inline]
pub const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Hash a master seed together with a path of stream coordinates into a single 64-bit seed.
///
/// Each coordinate is mixed sequentially, so `[a, b]` and `[b, a]` give different streams.
#[inline]
pub const fn derive_seed(seed: u64, path: &[u64]) -> u64 {
    let mut h = splitmix64(seed);
    let mut k = 0;
    while k < path.len() {
        h = splitmix64(h ^ splitmix64(path[k] ^ (k as u64).rotate_left(32)));
        k += 1;
    }
    h
}

/// Independent generator for the stream identified by `path` under the master `seed`.
#[inline]
pub fn stream(seed: u64, path: &[u64]) -> DefaultRng {
    DefaultRng::seed_from_u64(derive_seed(seed, path))
}

/// Fresh master seed from operating system entropy.
///
/// Callers are expected to record the returned value so the run can be reproduced.
pub fn entropy_seed() -> u64 {
    rand::rng().random()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng as _;

    #[test]
    fn streams_are_deterministic() {
        let mut a = stream(42, &[1, 2]);
        let mut b = stream(42, &[1, 2]);
        for _ in 0..16 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn streams_differ_by_path_and_order() {
        let seeds = [
            derive_seed(42, &[]),
            derive_seed(42, &[0]),
            derive_seed(42, &[1]),
            derive_seed(42, &[0, 1]),
            derive_seed(42, &[1, 0]),
            derive_seed(43, &[0, 1]),
        ];
        for (i, a) in seeds.iter().enumerate() {
            for b in &seeds[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }
}
