# artificial-systems

High performance simulations of artificial systems in Rust: lattice and mean-field spin models,
stochastic cellular automata, ensembles of their time series, and the random matrix analysis of
the resulting time series matrices (spectra of Wishart correlation matrices), as used in the
thesis *Random matrices approaches for correlated time series: statistical physics and other
applications*.

## Contents

| Area | What is available |
|---|---|
| Topologies | Hypercubic lattices of any dimension (`Chain`, `Square`, `Cubic`), per-axis periodic/open boundaries, non-square shapes; square lattice with Moore neighbourhoods (`Moore`); arbitrary graphs (`Graph`); fully connected systems (`MeanFieldState`) |
| Spin models | Blume-Emery-Griffiths `H = -JΣsᵢsⱼ - KΣsᵢ²sⱼ² - H₃Σsᵢsⱼ(sᵢ+sⱼ) + DΣsᵢ² - HΣsᵢ` covering Ising (spin-½) and Blume-Capel (spin-1); `Q`-state Potts; `Q`-state clock. All on lattices, graphs and mean field |
| Dynamics | Metropolis, heat bath (= Glauber for two-state spins); random, sequential, permutation or checkerboard site order; `T = 0` and `T = ∞` handled exactly |
| Cellular automata | Synchronous and asynchronous application of local rules, neighbour-swap diffusion, composition of dynamics. Contact process with diffusion, Domany-Kinzel (and any totalistic binary rule), Wolfram elementary rules, Life-like rules (Conway's Game of Life), Brass immune network automaton |
| Ensembles | Time series matrices `(n_samples, n_steps + 1)` with burn-in and stride, parallel over samples and runs, reproducible from a single seed |
| Analysis | Standardisation, correlation matrix `G* = X*X*ᵀ/n_steps`, eigenvalues (faer), histograms and their moments, Welford moments, Marchenko-Pastur law, spacings, spectral entropy, power-law fits with `R²`, dynamic exponent from `F₂`, correlated-pairs toy model |
| IO | CBOR/JSON (optionally gzipped) and CSV, parameter encoded file names |

## Command line

```sh
nix develop            # toolchain from the flake
cargo build --release  # target/release/artsys
```

Every subcommand writes one file holding its arguments, provenance metadata (including the seed)
and the requested data. `--emit ts-matrix` stores the time series matrices, `--emit spectrum` only
the correlation matrix spectra (`eigenvalues`, shape `(n_runs, n_samples)`, ascending per row),
`--emit all` both; `--correlations` adds the off-diagonal correlation values.

```sh
# Contact process spectra, as in the thesis (L = 128, 100 × 501 series, 1000 matrices)
artsys contact-process --dim 1 --length 128 --alpha 3.29785 --gamma 0.5 \
    --n-steps 500 --n-samples 100 --n-runs 1000 --emit spectrum

# Blume-Capel on the simple cubic lattice with heat bath dynamics from random states
artsys blume-capel --dim 3 --length 22 --d 1 --temperature 3.2 --dynamics heat-bath \
    --n-steps 300 --n-samples 100 --n-runs 1000 --emit spectrum

# Mean-field Ising with Glauber dynamics
artsys ising --mean-field --sites 10000 --coordination 4 --temperature 4 --dynamics heat-bath \
    --n-steps 300 --n-samples 100 --n-runs 1000 --emit spectrum

# Space-time diagrams of one-dimensional automata
artsys elementary --dim 1 --length 79 --rule 30 --init single --n-steps 40 --show
artsys domany-kinzel --dim 1 --length 100 --p1 0.7 --p2 0.7 --init single --n-steps 50 --show

# What is in a data file
artsys inspect data/ContactProcess_…cbor.gz
```

Run `artsys <command> --help` for all options (initial states, observables, site order, burn-in,
stride, output format, standardisation convention, …). `--threads` limits the worker threads.
`scripts/run.py` queues parameter scans with `pueue`.

## Library

```rust,no_run
use std::sync::Arc;
use artificial_systems::{
    analysis::{ZeroVariance, correlation_spectrum},
    dynamics::HeatBath,
    ensemble::{Ensemble, Schedule},
    model::Beg,
    observable::Magnetization,
    site::SpinOne,
    state::{Init, LatticeState},
    system::SpinSystem,
    topology::Cubic,
};

let lattice = Arc::new(Cubic::periodic_cube(22));
let ensemble = Ensemble {
    system: SpinSystem::new(LatticeState::uniform(lattice, SpinOne::Zero), Beg::blume_capel(1.0, 1.0, 0.0)),
    dynamics: HeatBath::at_temperature(3.2),
    prepare: Init::IidUniform,
    observable: Magnetization,
    schedule: Schedule::new(300),
    seed: 42,
};
let spectra = ensemble.map_runs(1000, 100, |_, m| {
    correlation_spectrum(m.view(), 0, ZeroVariance::Zero).unwrap()
});
```

New models implement `LocalModel` (lattice: bond and on-site energies plus a local field) and/or
`MeanFieldModel` (energy as a function of per-state counts); new automata implement `LocalRule`.
Any `Fn(&System) -> T` is an observable.

## Conventions

- **Time**: one step is `N` single site update attempts (`N` = number of sites). Series include
  the initial state, so they have `n_steps + 1` entries.
- **Contact process**: an active site becomes inactive with probability `1/α`, an inactive site
  copies a uniformly random neighbour; then `N` diffusion attempts swap, with probability `γ`, a
  random site with a random neighbour. This is the contact process with `λ = α`
  (`α_c ≈ 3.29785` in one dimension).
- **Mean field**: lattice sums become `Σ_⟨ij⟩ → (z/N) Σ_{i<j}` over distinct pairs, so e.g. the
  Ising energy change of a flip is `2Jz(sM - 1)/N` and `β_c = 1/(zJ)`.
- **Standardisation**: `x* = (x - ⟨x⟩)/σ` with the population variance by default (`--ddof 0`),
  so `G*` has unit diagonal and `⟨λ⟩ = 1` exactly. Constant series (absorbed contact process
  samples) are set to zero by default (`--zero-variance error` to fail instead).
- **Histogram moments** use bin centers. `BinPosition::LeftEdge` reproduces the (biased) left
  edge convention of the original analysis scripts for comparisons.
- **Reproducibility**: chain `(run, sample)` uses its own generator derived from the master seed,
  so results do not depend on the number of threads. The seed is always stored in the output.

## Performance

Single core throughput of one Monte Carlo step (`cargo bench`), in site updates per second:

| Workload | Updates/s |
|---|---|
| Ising, square `L = 100`, Metropolis | 67 M |
| Ising, square `L = 100`, heat bath | 84 M |
| Blume-Capel, cubic `L = 22`, heat bath | 43 M |
| Potts `Q = 3`, square `L = 64`, heat bath | 34 M |
| Ising, mean field `N = 10⁴`, Metropolis | 58 M |
| Ising, mean field `N = 10⁴`, heat bath | 30 M |
| Contact process with diffusion (`γ = 0.5`), chain `L = 128` | 82 M |

A `100 × 301` correlation spectrum takes about 0.6 ms.

Ensembles run in parallel over samples and runs. Hot loops avoid `exp` through per-temperature
tables indexed by the local field, observables are `O(1)` thanks to incrementally maintained
per-state counts, diffusion draws its number of swaps from a binomial instead of flipping a coin
per attempt, and absorbed chains stop being simulated. Binaries are portable by default; for a machine
specific build use `RUSTFLAGS="-C target-cpu=native" cargo build --release`.

## Development

```sh
cargo test --release   # unit, property and statistical tests
cargo clippy --all-targets -- -D warnings
cargo bench
```

The statistical tests compare sampled distributions with exact Boltzmann distributions of small
systems (every model, both dynamics, every site order, mean field) and the square lattice Ising
model with the Onsager energy and Yang magnetisation.
