//! Spin model subcommands.

use artificial_systems::{
    dynamics::{Dynamics, HeatBath, Metropolis},
    ensemble::Ensemble,
    io::ParamValue,
    model::{Beg, Clock, LocalModel, MeanFieldModel, Potts},
    observable::{
        AbsMagnetization, ClockMagnetization, Density, Energy, EnergyPerSite, Magnetization,
        PottsOrder, Quadrupole,
    },
    site::{ClockState, PottsState, Site, SpinHalf, SpinOne},
    state::{Configuration, Init, LatticeState, MeanFieldState, Prepare},
    system::SpinSystem,
};
use clap::{Args, ValueEnum};
use serde::Serialize;

use crate::common::{
    DynamicsArgs, DynamicsKind, LatticeArgs, Obs, ObservableKind, OutputArgs, SeriesArgs,
    ThermalArgs, boxed, execute, with_lattice,
};

#[derive(Debug, Clone, Args, Serialize)]
pub struct MeanFieldArgs {
    /// Fully connected (mean-field) system instead of a lattice
    #[arg(long)]
    pub mean_field: bool,
    /// Number of sites of the mean-field system
    #[arg(long, required_if_eq("mean_field", "true"))]
    pub sites: Option<usize>,
    /// Effective coordination number z of the mean-field system
    #[arg(long, default_value_t = 4.0)]
    pub coordination: f64,
}

/// Arguments common to every spin model.
#[derive(Debug, Clone, Args, Serialize)]
pub struct SpinArgs {
    #[command(flatten)]
    pub lattice: LatticeArgs,
    #[command(flatten)]
    pub mean_field: MeanFieldArgs,
    #[command(flatten)]
    pub thermal: ThermalArgs,
    #[command(flatten)]
    pub dynamics: DynamicsArgs,
    #[command(flatten)]
    pub series: SeriesArgs,
    #[command(flatten)]
    pub output: OutputArgs,
}

impl SpinArgs {
    fn params(&self) -> Vec<(&'static str, ParamValue)> {
        let mut params = if self.mean_field.mean_field {
            vec![
                ("N", self.mean_field.sites.unwrap_or(0).into()),
                ("z", self.mean_field.coordination.into()),
            ]
        } else {
            self.lattice.params()
        };
        params.extend(self.thermal.params());
        params.extend(self.dynamics.params());
        params.extend(self.series.params());
        params
    }
}

/// Observables of spin systems with scalar spins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SpinObservable {
    Magnetization,
    AbsMagnetization,
    /// Mean squared spin
    Quadrupole,
    Energy,
    EnergyPerSite,
}

impl<Sys: 'static> ObservableKind<Sys> for SpinObservable
where
    Magnetization: artificial_systems::observable::Observable<Sys, Output = f64>,
    AbsMagnetization: artificial_systems::observable::Observable<Sys, Output = f64>,
    Quadrupole: artificial_systems::observable::Observable<Sys, Output = f64>,
    Energy: artificial_systems::observable::Observable<Sys, Output = f64>,
    EnergyPerSite: artificial_systems::observable::Observable<Sys, Output = f64>,
{
    fn build(self) -> Obs<Sys> {
        match self {
            Self::Magnetization => boxed(Magnetization),
            Self::AbsMagnetization => boxed(AbsMagnetization),
            Self::Quadrupole => boxed(Quadrupole),
            Self::Energy => boxed(Energy),
            Self::EnergyPerSite => boxed(EnergyPerSite),
        }
    }
}

/// Initial states of scalar spin systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SpinInit {
    /// Independent uniformly random spins (infinite temperature)
    Random,
    /// All spins up
    Up,
    /// All spins down
    Down,
    /// All spins zero (spin-1 only)
    Zero,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct IsingArgs {
    #[command(flatten)]
    pub common: SpinArgs,
    /// Exchange coupling J
    #[arg(long, default_value_t = 1.0)]
    pub j: f64,
    /// External field h
    #[arg(long, default_value_t = 0.0)]
    pub h: f64,
    /// Initial state
    #[arg(long, value_enum, default_value_t = SpinInit::Random)]
    pub init: SpinInit,
    /// Measured quantity
    #[arg(long, value_enum, default_value_t = SpinObservable::Magnetization)]
    pub observable: SpinObservable,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct BlumeCapelArgs {
    #[command(flatten)]
    pub common: SpinArgs,
    /// Exchange coupling J
    #[arg(long, default_value_t = 1.0)]
    pub j: f64,
    /// Crystal field (anisotropy) D
    #[arg(long, default_value_t = 0.0)]
    pub d: f64,
    /// External field H
    #[arg(long, default_value_t = 0.0)]
    pub h: f64,
    /// Biquadratic coupling K (Blume-Emery-Griffiths)
    #[arg(long, default_value_t = 0.0)]
    pub k: f64,
    /// Cubic coupling H3 (Blume-Emery-Griffiths)
    #[arg(long, default_value_t = 0.0)]
    pub h3: f64,
    /// Initial state
    #[arg(long, value_enum, default_value_t = SpinInit::Random)]
    pub init: SpinInit,
    /// Measured quantity
    #[arg(long, value_enum, default_value_t = SpinObservable::Magnetization)]
    pub observable: SpinObservable,
}

/// Observables of Potts systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PottsObservable {
    /// (Q max_q n_q / N - 1) / (Q - 1)
    Order,
    /// Fraction of sites in state 0
    Density0,
    Energy,
    EnergyPerSite,
}

/// Initial states of Potts and clock systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum QInit {
    /// Independent uniformly random states
    Random,
    /// All sites in state 0
    Ordered,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct PottsArgs {
    #[command(flatten)]
    pub common: SpinArgs,
    /// Number of states Q (2 to 8)
    #[arg(long, value_parser = clap::value_parser!(u8).range(2..=8))]
    pub q: u8,
    /// Coupling J
    #[arg(long, default_value_t = 1.0)]
    pub j: f64,
    /// Field h on state 0
    #[arg(long, default_value_t = 0.0)]
    pub h: f64,
    /// Initial state
    #[arg(long, value_enum, default_value_t = QInit::Random)]
    pub init: QInit,
    /// Measured quantity
    #[arg(long, value_enum, default_value_t = PottsObservable::Order)]
    pub observable: PottsObservable,
}

/// Observables of clock systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ClockObservable {
    /// Modulus of the magnetisation vector per site
    Magnetization,
    Energy,
    EnergyPerSite,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct ClockArgs {
    #[command(flatten)]
    pub common: SpinArgs,
    /// Number of states Q (2 to 8)
    #[arg(long, value_parser = clap::value_parser!(u8).range(2..=8))]
    pub q: u8,
    /// Coupling J
    #[arg(long, default_value_t = 1.0)]
    pub j: f64,
    /// Field h along θ = 0
    #[arg(long, default_value_t = 0.0)]
    pub h: f64,
    /// Initial state
    #[arg(long, value_enum, default_value_t = QInit::Random)]
    pub init: QInit,
    /// Measured quantity
    #[arg(long, value_enum, default_value_t = ClockObservable::Magnetization)]
    pub observable: ClockObservable,
}

/// Build the ensemble for `system` and run it with the selected dynamics.
fn run_system<Sys, S, K, A>(
    system: Sys,
    init: Init<S>,
    observable: K,
    common: &SpinArgs,
    prefix: &str,
    params: Vec<(&'static str, ParamValue)>,
    args: &A,
) -> anyhow::Result<()>
where
    Sys: Clone + Send + Sync,
    S: Site,
    Init<S>: Prepare<Sys>,
    K: ObservableKind<Sys>,
    Metropolis: Dynamics<Sys>,
    HeatBath: Dynamics<Sys>,
    A: Serialize,
{
    let beta = common.thermal.beta()?;
    let order = common.dynamics.site_order;
    let seed = common.output.seed();
    let schedule = common.series.schedule();
    macro_rules! go {
        ($dynamics:expr) => {
            execute(
                Ensemble {
                    system,
                    dynamics: $dynamics,
                    prepare: init,
                    observable: observable.build(),
                    schedule,
                    seed,
                },
                &common.series,
                &common.output,
                prefix,
                params,
                args,
            )
        };
    }
    match common.dynamics.dynamics {
        DynamicsKind::Metropolis => go!(Metropolis::with_order(beta, order)),
        DynamicsKind::HeatBath => go!(HeatBath::with_order(beta, order)),
    }
}

/// Run a model on the lattice or mean-field system selected in `common`.
fn run_model<S, M, K, A>(
    model: M,
    init: Init<S>,
    observable: K,
    common: &SpinArgs,
    prefix: &str,
    model_params: Vec<(&'static str, ParamValue)>,
    args: &A,
) -> anyhow::Result<()>
where
    S: Site,
    M: LocalModel<S> + MeanFieldModel<S>,
    K: ObservableKind<SpinSystem<MeanFieldState<S>, M>>
        + ObservableKind<SpinSystem<LatticeState<S, artificial_systems::topology::Chain>, M>>
        + ObservableKind<SpinSystem<LatticeState<S, artificial_systems::topology::Square>, M>>
        + ObservableKind<SpinSystem<LatticeState<S, artificial_systems::topology::Cubic>, M>>,
    A: Serialize,
{
    let mut params = model_params;
    params.extend(common.params());
    if common.mean_field.mean_field {
        let n = common.mean_field.sites.expect("required with --mean-field");
        let state = MeanFieldState::uniform(n, common.mean_field.coordination, S::VALUES[0]);
        let prefix = format!("{prefix}MeanField");
        run_system(
            SpinSystem::new(state, model),
            init,
            observable,
            common,
            &prefix,
            params,
            args,
        )
    } else {
        with_lattice!(common.lattice, |topology| {
            let system = SpinSystem::new(LatticeState::uniform(topology, S::VALUES[0]), model);
            run_system(system, init, observable, common, prefix, params, args)
        })
    }
}

fn spin_init<S: Site>(init: SpinInit, up: S, down: S, zero: Option<S>) -> anyhow::Result<Init<S>> {
    Ok(match init {
        SpinInit::Random => Init::IidUniform,
        SpinInit::Up => Init::Uniform(up),
        SpinInit::Down => Init::Uniform(down),
        SpinInit::Zero => {
            Init::Uniform(zero.ok_or_else(|| anyhow::anyhow!("Spin-½ systems have no zero state"))?)
        }
    })
}

fn init_name(init: impl Serialize) -> ParamValue {
    serde_json::to_value(init)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
        .into()
}

pub fn ising(args: &IsingArgs) -> anyhow::Result<()> {
    let init = spin_init(args.init, SpinHalf::Up, SpinHalf::Down, None)?;
    run_model(
        Beg::ising(args.j, args.h),
        init,
        args.observable,
        &args.common,
        "Ising",
        vec![
            ("J", args.j.into()),
            ("h", args.h.into()),
            ("init", init_name(args.init)),
            ("obs", init_name(args.observable)),
        ],
        args,
    )
}

pub fn blume_capel(args: &BlumeCapelArgs) -> anyhow::Result<()> {
    let init = spin_init(args.init, SpinOne::Up, SpinOne::Down, Some(SpinOne::Zero))?;
    let model = Beg {
        j: args.j,
        k: args.k,
        h3: args.h3,
        d: args.d,
        h: args.h,
    };
    let beg = args.k != 0.0 || args.h3 != 0.0;
    let mut params = vec![
        ("J", args.j.into()),
        ("D", args.d.into()),
        ("H", args.h.into()),
        ("init", init_name(args.init)),
        ("obs", init_name(args.observable)),
    ];
    if beg {
        params.extend([("K", args.k.into()), ("H3", args.h3.into())]);
    }
    run_model(
        model,
        init,
        args.observable,
        &args.common,
        if beg {
            "BlumeEmeryGriffiths"
        } else {
            "BlumeCapel"
        },
        params,
        args,
    )
}

/// Run `$body` with the const `$Q` bound to the run time value `$q` (2 to 8).
macro_rules! with_q {
    ($q:expr, $Q:ident => $body:expr) => {
        match $q {
            2 => {
                const $Q: usize = 2;
                $body
            }
            3 => {
                const $Q: usize = 3;
                $body
            }
            4 => {
                const $Q: usize = 4;
                $body
            }
            5 => {
                const $Q: usize = 5;
                $body
            }
            6 => {
                const $Q: usize = 6;
                $body
            }
            7 => {
                const $Q: usize = 7;
                $body
            }
            8 => {
                const $Q: usize = 8;
                $body
            }
            q => anyhow::bail!("Unsupported number of states {q}"),
        }
    };
}

impl<const Q: usize, Sys: Configuration<Site = PottsState<Q>> + 'static> ObservableKind<Sys>
    for PottsObservable
where
    Energy: artificial_systems::observable::Observable<Sys, Output = f64>,
    EnergyPerSite: artificial_systems::observable::Observable<Sys, Output = f64>,
{
    fn build(self) -> Obs<Sys> {
        match self {
            Self::Order => boxed(PottsOrder),
            Self::Density0 => boxed(Density(PottsState::<Q>::new(0))),
            Self::Energy => boxed(Energy),
            Self::EnergyPerSite => boxed(EnergyPerSite),
        }
    }
}

impl<const Q: usize, Sys: Configuration<Site = ClockState<Q>> + 'static> ObservableKind<Sys>
    for ClockObservable
where
    Energy: artificial_systems::observable::Observable<Sys, Output = f64>,
    EnergyPerSite: artificial_systems::observable::Observable<Sys, Output = f64>,
{
    fn build(self) -> Obs<Sys> {
        match self {
            Self::Magnetization => boxed(ClockMagnetization),
            Self::Energy => boxed(Energy),
            Self::EnergyPerSite => boxed(EnergyPerSite),
        }
    }
}

pub fn potts(args: &PottsArgs) -> anyhow::Result<()> {
    with_q!(args.q, Q => {
        let init = match args.init {
            QInit::Random => Init::IidUniform,
            QInit::Ordered => Init::Uniform(PottsState::<Q>::new(0)),
        };
        run_model(
            Potts::new(args.j, args.h),
            init,
            args.observable,
            &args.common,
            "Potts",
            vec![
                ("q", Q.into()),
                ("J", args.j.into()),
                ("h", args.h.into()),
                ("init", init_name(args.init)),
                ("obs", init_name(args.observable)),
            ],
            args,
        )
    })
}

pub fn clock(args: &ClockArgs) -> anyhow::Result<()> {
    with_q!(args.q, Q => {
        let init = match args.init {
            QInit::Random => Init::IidUniform,
            QInit::Ordered => Init::Uniform(ClockState::<Q>::new(0)),
        };
        run_model(
            Clock::new(Q, args.j, args.h),
            init,
            args.observable,
            &args.common,
            "Clock",
            vec![
                ("q", Q.into()),
                ("J", args.j.into()),
                ("h", args.h.into()),
                ("init", init_name(args.init)),
                ("obs", init_name(args.observable)),
            ],
            args,
        )
    })
}
