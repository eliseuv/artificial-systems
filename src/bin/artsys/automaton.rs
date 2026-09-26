//! Cellular automaton subcommands.

use std::fmt::Display;

use artificial_systems::{
    automaton::{
        Asynchronous, BrassRule, Elementary, Synchronous, TotalisticBinary, contact_process,
    },
    dynamics::Dynamics,
    ensemble::Ensemble,
    io::ParamValue,
    observable::{Count, Density, Magnetization},
    rng::stream,
    site::{Binary, Brass, Site},
    state::{Configuration, Init, LatticeState, Position, Prepare},
    topology::Hypercubic,
};
use clap::{Args, ValueEnum};
use serde::Serialize;

use crate::common::{
    LatticeArgs, Obs, ObservableKind, OutputArgs, SeriesArgs, boxed, execute, with_lattice,
};

/// Arguments common to every automaton.
#[derive(Debug, Clone, Args, Serialize)]
pub struct AutomatonArgs {
    #[command(flatten)]
    pub lattice: LatticeArgs,
    #[command(flatten)]
    pub series: SeriesArgs,
    #[command(flatten)]
    pub output: OutputArgs,
    /// Print the evolution of one sample instead of storing an ensemble
    #[arg(long)]
    pub show: bool,
}

/// Initial states of binary automata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BinaryInit {
    /// Every site active
    AllActive,
    /// A single active site at the center
    Single,
    /// Every site independently active with probability --density
    Random,
}

/// Observables of binary automata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BinaryObservable {
    /// Fraction of active sites
    Density,
    /// Number of active sites
    Count,
}

impl<Sys: Configuration<Site = Binary> + 'static> ObservableKind<Sys> for BinaryObservable {
    fn build(self) -> Obs<Sys> {
        match self {
            Self::Density => boxed(Density(Binary::Active)),
            Self::Count => boxed(Count(Binary::Active)),
        }
    }
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct BinaryArgs {
    #[command(flatten)]
    pub common: AutomatonArgs,
    /// Initial state
    #[arg(long, value_enum, default_value_t = BinaryInit::AllActive)]
    pub init: BinaryInit,
    /// Density of active sites of random initial states
    #[arg(long, default_value_t = 0.5)]
    pub density: f64,
    /// Measured quantity
    #[arg(long, value_enum, default_value_t = BinaryObservable::Density)]
    pub observable: BinaryObservable,
}

impl BinaryArgs {
    fn init(&self) -> Init<Binary> {
        match self.init {
            BinaryInit::AllActive => Init::Uniform(Binary::Active),
            BinaryInit::Single => Init::Single {
                background: Binary::Inactive,
                value: Binary::Active,
                at: Position::Center,
            },
            BinaryInit::Random => Init::bernoulli(self.density, Binary::Active, Binary::Inactive),
        }
    }

    fn params(&self) -> Vec<(&'static str, ParamValue)> {
        let init = match self.init {
            BinaryInit::AllActive => "all-active".to_owned(),
            BinaryInit::Single => "single".to_owned(),
            BinaryInit::Random => format!("random{}", self.density),
        };
        let mut params = vec![
            ("init", init.into()),
            (
                "obs",
                match self.observable {
                    BinaryObservable::Density => "density",
                    BinaryObservable::Count => "count",
                }
                .into(),
            ),
        ];
        params.extend(self.common.lattice.params());
        params.extend(self.common.series.params());
        params
    }
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct ContactProcessArgs {
    #[command(flatten)]
    pub binary: BinaryArgs,
    /// Infection rate α (inf disables recovery)
    #[arg(long)]
    pub alpha: f64,
    /// Diffusion probability γ per attempt
    #[arg(long, default_value_t = 0.0)]
    pub gamma: f64,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct DomanyKinzelArgs {
    #[command(flatten)]
    pub binary: BinaryArgs,
    /// Activation probability with one active neighbour
    #[arg(long)]
    pub p1: f64,
    /// Activation probability with two active neighbours
    #[arg(long)]
    pub p2: f64,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct ElementaryArgs {
    #[command(flatten)]
    pub binary: BinaryArgs,
    /// Wolfram rule number
    #[arg(long)]
    pub rule: u8,
}

/// Update scheme of automata that support both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Update {
    Synchronous,
    Asynchronous,
}

/// Initial states of the Brass automaton.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BrassInit {
    /// Independent uniformly random cells
    Random,
    /// Every cell virgin (TH0)
    Th0,
    /// Every cell TH1
    Th1,
}

/// Observables of the Brass automaton.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BrassObservable {
    /// (n_TH1 - n_TH2) / N
    Magnetization,
    DensityTh0,
    DensityTh1,
    DensityTh2,
}

impl<Sys: Configuration<Site = Brass> + 'static> ObservableKind<Sys> for BrassObservable {
    fn build(self) -> Obs<Sys> {
        match self {
            Self::Magnetization => boxed(Magnetization),
            Self::DensityTh0 => boxed(Density(Brass::TH0)),
            Self::DensityTh1 => boxed(Density(Brass::TH1)),
            Self::DensityTh2 => boxed(Density(Brass::TH2)),
        }
    }
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct BrassArgs {
    #[command(flatten)]
    pub common: AutomatonArgs,
    /// Antigen probability p
    #[arg(long)]
    pub p: f64,
    /// Decay probability r
    #[arg(long)]
    pub r: f64,
    /// Update scheme
    #[arg(long, value_enum, default_value_t = Update::Synchronous)]
    pub update: Update,
    /// Initial state
    #[arg(long, value_enum, default_value_t = BrassInit::Random)]
    pub init: BrassInit,
    /// Measured quantity
    #[arg(long, value_enum, default_value_t = BrassObservable::Magnetization)]
    pub observable: BrassObservable,
}

/// Print the evolution of a single sample, one configuration per step.
fn show<S, D, const N: usize>(
    state: LatticeState<S, Hypercubic<N>>,
    mut dynamics: D,
    init: &Init<S>,
    n_steps: usize,
    seed: u64,
) where
    S: Site + Display,
    D: Dynamics<LatticeState<S, Hypercubic<N>>>,
{
    let mut state = state;
    let mut rng = stream(seed, &[0, 0]);
    init.prepare(&mut state, &mut rng);
    let separator = if N > 1 { "\n\n" } else { "\n" };
    print!("{state}{separator}");
    for _ in 0..n_steps {
        dynamics.step(&mut state, &mut rng);
        print!("{state}{separator}");
    }
}

/// Run `dynamics` from `init` on the lattice of `common`, storing or showing the result.
macro_rules! run_automaton {
    ($common:expr, $fill:expr, $dynamics:expr, $init:expr, $observable:expr, $prefix:expr, $params:expr, $args:expr) => {{
        let common: &AutomatonArgs = $common;
        let seed = common.output.seed();
        with_lattice!(common.lattice, |topology| {
            let state = LatticeState::uniform(topology, $fill);
            if common.show {
                show(state, $dynamics, &$init, common.series.n_steps, seed);
                Ok(())
            } else {
                execute(
                    Ensemble {
                        system: state,
                        dynamics: $dynamics,
                        prepare: $init.clone(),
                        observable: $observable.build(),
                        schedule: common.series.schedule(),
                        seed,
                    },
                    &common.series,
                    &common.output,
                    $prefix,
                    $params,
                    $args,
                )
            }
        })
    }};
}

pub fn contact_process_cmd(args: &ContactProcessArgs) -> anyhow::Result<()> {
    let binary = &args.binary;
    let mut params = vec![("alpha", args.alpha.into()), ("gamma", args.gamma.into())];
    params.extend(binary.params());
    let init = binary.init();
    run_automaton!(
        &binary.common,
        Binary::Inactive,
        contact_process(args.alpha, args.gamma),
        init,
        binary.observable,
        "ContactProcess",
        params,
        args
    )
}

pub fn domany_kinzel(args: &DomanyKinzelArgs) -> anyhow::Result<()> {
    let binary = &args.binary;
    let mut params = vec![("p1", args.p1.into()), ("p2", args.p2.into())];
    params.extend(binary.params());
    let init = binary.init();
    run_automaton!(
        &binary.common,
        Binary::Inactive,
        Synchronous::new(TotalisticBinary::domany_kinzel(args.p1, args.p2)),
        init,
        binary.observable,
        "DomanyKinzel",
        params,
        args
    )
}

pub fn elementary(args: &ElementaryArgs) -> anyhow::Result<()> {
    let binary = &args.binary;
    anyhow::ensure!(
        binary.common.lattice.dim == 1
            && binary.common.lattice.boundary == artificial_systems::topology::Boundary::Periodic,
        "Elementary automata need a periodic chain (--dim 1)"
    );
    let mut params = vec![("rule", args.rule.into())];
    params.extend(binary.params());
    let init = binary.init();
    run_automaton!(
        &binary.common,
        Binary::Inactive,
        Synchronous::new(Elementary { rule: args.rule }),
        init,
        binary.observable,
        "Elementary",
        params,
        args
    )
}

pub fn brass(args: &BrassArgs) -> anyhow::Result<()> {
    let rule = BrassRule::new(args.p, args.r);
    let init = match args.init {
        BrassInit::Random => Init::IidUniform,
        BrassInit::Th0 => Init::Uniform(Brass::TH0),
        BrassInit::Th1 => Init::Uniform(Brass::TH1),
    };
    let label = |v: &dyn erased::Label| v.label();
    let mut params = vec![
        ("p", args.p.into()),
        ("r", args.r.into()),
        ("update", label(&args.update).into()),
        ("init", label(&args.init).into()),
        ("obs", label(&args.observable).into()),
    ];
    params.extend(args.common.lattice.params());
    params.extend(args.common.series.params());
    match args.update {
        Update::Synchronous => run_automaton!(
            &args.common,
            Brass::TH0,
            Synchronous::new(rule),
            init,
            args.observable,
            "Brass",
            params,
            args
        ),
        Update::Asynchronous => run_automaton!(
            &args.common,
            Brass::TH0,
            Asynchronous::new(rule),
            init,
            args.observable,
            "Brass",
            params,
            args
        ),
    }
}

mod erased {
    use serde::Serialize;

    /// Kebab-case name of a serialisable unit enum variant.
    pub trait Label {
        fn label(&self) -> String;
    }

    impl<T: Serialize> Label for T {
        fn label(&self) -> String {
            serde_json::to_value(self)
                .ok()
                .and_then(|v| v.as_str().map(str::to_owned))
                .unwrap_or_default()
        }
    }
}
