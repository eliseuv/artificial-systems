//! `artsys`: ensembles of artificial systems from the command line.

mod automaton;
mod common;
mod inspect;
mod spin;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Number of worker threads (default: all cores)
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Ising model (spin-½)
    Ising(spin::IsingArgs),
    /// Blume-Capel model (spin-1), or Blume-Emery-Griffiths with --k/--h3
    BlumeCapel(spin::BlumeCapelArgs),
    /// Q-state Potts model
    Potts(spin::PottsArgs),
    /// Q-state clock model
    Clock(spin::ClockArgs),
    /// Contact process with diffusion
    ContactProcess(automaton::ContactProcessArgs),
    /// Domany-Kinzel automaton
    DomanyKinzel(automaton::DomanyKinzelArgs),
    /// Wolfram elementary cellular automaton
    Elementary(automaton::ElementaryArgs),
    /// Brass immune network automaton
    Brass(automaton::BrassArgs),
    /// Summarise the contents of a data file
    Inspect(inspect::InspectArgs),
}

fn main() -> anyhow::Result<()> {
    colog::init();
    let cli = Cli::parse();
    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?;
    }
    match &cli.command {
        Command::Ising(args) => spin::ising(args),
        Command::BlumeCapel(args) => spin::blume_capel(args),
        Command::Potts(args) => spin::potts(args),
        Command::Clock(args) => spin::clock(args),
        Command::ContactProcess(args) => automaton::contact_process_cmd(args),
        Command::DomanyKinzel(args) => automaton::domany_kinzel(args),
        Command::Elementary(args) => automaton::elementary(args),
        Command::Brass(args) => automaton::brass(args),
        Command::Inspect(args) => inspect::inspect(args),
    }
}
