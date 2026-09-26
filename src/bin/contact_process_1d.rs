use std::path::PathBuf;

use artificial_systems::{
    contact_process::{
        ContactProcessMarkovChain, lattice::LatticeContactProcessDiffusion1D,
        measurement::TotalActiveSites, state::AllActive,
    },
    data_io::{DataFile, DataFileSpec},
    dbg,
    method::TimeSeriesMatrix,
    utils::{AutoTimer, DefaultRNG},
};
use clap::Parser;
use log::info;
use ndarray::Array2;
use rand::SeedableRng;
use serde::Serialize;

#[derive(Debug, Parser, Serialize)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, short)]
    length: usize,

    #[arg(long, short)]
    rate: f64,

    #[arg(long, short)]
    diffusion: f64,

    #[arg(long)]
    n_steps: usize,

    #[arg(long)]
    n_samples: usize,

    #[arg(long, short)]
    output: PathBuf,

    #[arg(long, value_enum, default_value_t = DataFileSpec::default())]
    data_spec: DataFileSpec,

    #[arg(long, default_value_t = true)]
    gzip: bool,

    #[arg(long, default_value_t = false)]
    no_overwrite: bool,

    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Debug, Serialize)]
struct Output {
    args: Args,
    time_series_matrix: Array2<usize>,
}

fn main() -> anyhow::Result<()> {
    let _timer = AutoTimer::new("Contact Process 1D time series matrix");

    // Parse arguments
    let args = dbg!(Args::parse());

    // Output datafile
    let datafile = DataFile::new(&args.output, args.data_spec, args.gzip);
    info!("Output datafile:\n{datafile:#?}");

    // Test overwrite if needed
    if args.no_overwrite {
        datafile.no_overwrite_test()?;
    }
    datafile.create_parent_dir()?;

    // Prepare PRNG
    let mut rng = if let Some(seed) = args.seed {
        DefaultRNG::seed_from_u64(seed)
    } else {
        DefaultRNG::from_os_rng()
    };

    // Prepare system
    let mut system = LatticeContactProcessDiffusion1D::new(
        args.length,
        args.rate,
        args.diffusion,
        &mut AllActive,
    );

    // Run simulation
    info!("Running...");
    let time_series_matrix = TotalActiveSites::time_series_matrix(
        &mut system,
        &mut ContactProcessMarkovChain,
        &mut AllActive,
        args.n_steps,
        args.n_samples,
        &mut rng,
    );

    info!("Writing...");
    datafile.write(&Output {
        args,
        time_series_matrix: dbg!(time_series_matrix),
    })?;

    Ok(())
}
