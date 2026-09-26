//! Arguments and output pipeline shared by all subcommands.

use std::path::PathBuf;

use anyhow::{Context, bail};
use artificial_systems::{
    analysis::{Moments, ZeroVariance, correlation_matrix, standardize_rows, upper_triangle},
    dynamics::{Dynamics, SiteOrder},
    ensemble::{Ensemble, Schedule},
    io::{DataFile, Format, Metadata, ParamValue, file_name},
    observable::Observable,
    rng::entropy_seed,
    state::Prepare,
    topology::Boundary,
};
use clap::{Args, ValueEnum};
use log::info;
use ndarray::{Array2, Axis};
use serde::Serialize;

/// Observable producing `f64` values, chosen at run time.
pub type Obs<Sys> = Box<dyn Fn(&Sys) -> f64 + Send + Sync>;

/// Run time choice of observable for systems of type `Sys`.
pub trait ObservableKind<Sys>: Copy {
    /// Boxed observable.
    fn build(self) -> Obs<Sys>;
}

/// Boxed observable from a library observable.
pub fn boxed<Sys, O>(observable: O) -> Obs<Sys>
where
    O: Observable<Sys> + 'static,
    O::Output: Into<f64>,
{
    Box::new(move |sys| observable.measure(sys).into())
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct LatticeArgs {
    /// Lattice dimension (1, 2 or 3)
    #[arg(long, default_value_t = 2)]
    pub dim: usize,
    /// Side length along every axis
    #[arg(long, conflicts_with = "shape")]
    pub length: Option<usize>,
    /// Side lengths per axis, e.g. `64,32`
    #[arg(long, value_delimiter = ',')]
    pub shape: Option<Vec<usize>>,
    /// Boundary conditions
    #[arg(long, value_enum, default_value_t = Boundary::Periodic)]
    pub boundary: Boundary,
}

impl LatticeArgs {
    /// Side lengths of a `D`-dimensional lattice.
    pub fn lengths<const D: usize>(&self) -> anyhow::Result<[usize; D]> {
        match (&self.length, &self.shape) {
            (Some(l), None) => Ok([*l; D]),
            (None, Some(shape)) => shape
                .as_slice()
                .try_into()
                .with_context(|| format!("--shape needs {D} lengths, got {}", shape.len())),
            _ => bail!("Specify the lattice size with --length or --shape"),
        }
    }

    /// Parameters identifying the lattice in file names.
    pub fn params(&self) -> Vec<(&'static str, ParamValue)> {
        let size = match (&self.length, &self.shape) {
            (Some(l), _) => ParamValue::from(*l),
            (_, Some(s)) => s
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("x")
                .into(),
            _ => ParamValue::from("unknown"),
        };
        let mut params = vec![("dim", self.dim.into()), ("L", size)];
        if self.boundary == Boundary::Open {
            params.push(("boundary", "open".into()));
        }
        params
    }
}

/// Run `$body` with `$top` bound to an `Arc` of the hypercubic lattice described by `$args`.
macro_rules! with_lattice {
    ($args:expr, |$top:ident| $body:expr) => {{
        use artificial_systems::topology::Hypercubic;
        use std::sync::Arc;
        let args = &$args;
        match args.dim {
            1 => {
                let $top = Arc::new(Hypercubic::<1>::new(
                    args.lengths::<1>()?,
                    [args.boundary; 1],
                ));
                $body
            }
            2 => {
                let $top = Arc::new(Hypercubic::<2>::new(
                    args.lengths::<2>()?,
                    [args.boundary; 2],
                ));
                $body
            }
            3 => {
                let $top = Arc::new(Hypercubic::<3>::new(
                    args.lengths::<3>()?,
                    [args.boundary; 3],
                ));
                $body
            }
            d => anyhow::bail!("Unsupported lattice dimension {d} (use 1, 2 or 3)"),
        }
    }};
}
pub(crate) use with_lattice;

#[derive(Debug, Clone, Args, Serialize)]
pub struct SeriesArgs {
    /// Measurements after the initial one (series length is n_steps + 1)
    #[arg(long)]
    pub n_steps: usize,
    /// Independent series per time series matrix
    #[arg(long, default_value_t = 1)]
    pub n_samples: usize,
    /// Number of time series matrices
    #[arg(long, default_value_t = 1)]
    pub n_runs: usize,
    /// Steps discarded before the initial measurement
    #[arg(long, default_value_t = 0)]
    pub burn_in: usize,
    /// Steps between measurements
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u64).range(1..))]
    pub stride: u64,
}

impl SeriesArgs {
    pub fn schedule(&self) -> Schedule {
        Schedule {
            n_steps: self.n_steps,
            burn_in: self.burn_in,
            stride: self.stride as usize,
        }
    }

    pub fn params(&self) -> Vec<(&'static str, ParamValue)> {
        let mut params = vec![
            ("n_steps", self.n_steps.into()),
            ("n_samples", self.n_samples.into()),
            ("n_runs", self.n_runs.into()),
        ];
        if self.burn_in > 0 {
            params.push(("burn_in", self.burn_in.into()));
        }
        if self.stride > 1 {
            params.push(("stride", self.stride.into()));
        }
        params
    }
}

/// What to store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Emit {
    /// Time series matrices
    TsMatrix,
    /// Correlation matrix spectra (and optionally correlation values)
    Spectrum,
    /// Both
    All,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct OutputArgs {
    /// Output path without extension (default: parameter encoded name inside --dir)
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Directory for automatically named outputs
    #[arg(long, default_value = "data")]
    pub dir: PathBuf,
    /// File format
    #[arg(long, value_enum, default_value_t = Format::Cbor)]
    pub format: Format,
    /// Do not gzip the output
    #[arg(long)]
    pub no_gzip: bool,
    /// Fail instead of overwriting an existing file
    #[arg(long)]
    pub no_overwrite: bool,
    /// Master seed (drawn from the OS, and recorded, if omitted)
    #[arg(long)]
    pub seed: Option<u64>,
    /// Data to store
    #[arg(long, value_enum, default_value_t = Emit::TsMatrix)]
    pub emit: Emit,
    /// Delta degrees of freedom of the series standardisation (0: population variance)
    #[arg(long, default_value_t = 0)]
    pub ddof: usize,
    /// Treatment of constant series in the spectral analysis
    #[arg(long, value_enum, default_value_t = ZeroVariance::Zero)]
    pub zero_variance: ZeroVariance,
    /// Also store the off-diagonal correlation values of each matrix
    #[arg(long)]
    pub correlations: bool,
}

impl OutputArgs {
    /// Master seed, drawing a fresh one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed.unwrap_or_else(entropy_seed)
    }
}

#[derive(Debug, Clone, Args, Serialize)]
#[group(required = true, multiple = false)]
pub struct ThermalArgs {
    /// Temperature (0 allowed)
    #[arg(long)]
    pub temperature: Option<f64>,
    /// Inverse temperature (inf allowed)
    #[arg(long)]
    pub beta: Option<f64>,
}

impl ThermalArgs {
    pub fn beta(&self) -> anyhow::Result<f64> {
        match (self.temperature, self.beta) {
            (Some(t), None) if t >= 0.0 => Ok(t.recip()),
            (None, Some(b)) if b >= 0.0 => Ok(b),
            _ => bail!("Temperature or inverse temperature must be non-negative"),
        }
    }

    pub fn params(&self) -> Vec<(&'static str, ParamValue)> {
        match (self.temperature, self.beta) {
            (Some(t), _) => vec![("T", t.into())],
            (_, Some(b)) => vec![("beta", b.into())],
            _ => vec![],
        }
    }
}

/// Single site dynamics of spin models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DynamicsKind {
    Metropolis,
    /// Heat bath (Glauber for two-state spins)
    HeatBath,
}

#[derive(Debug, Clone, Args, Serialize)]
pub struct DynamicsArgs {
    /// Single site update rule
    #[arg(long, value_enum, default_value_t = DynamicsKind::Metropolis)]
    pub dynamics: DynamicsKind,
    /// Order in which sites are visited within a step
    #[arg(long, value_enum, default_value_t = SiteOrder::Random)]
    pub site_order: SiteOrder,
}

impl DynamicsArgs {
    pub fn params(&self) -> Vec<(&'static str, ParamValue)> {
        let mut params = vec![(
            "dynamics",
            match self.dynamics {
                DynamicsKind::Metropolis => "metropolis",
                DynamicsKind::HeatBath => "heatbath",
            }
            .into(),
        )];
        if self.site_order != SiteOrder::Random {
            params.push((
                "order",
                format!("{:?}", self.site_order).to_lowercase().into(),
            ));
        }
        params
    }
}

#[derive(Serialize)]
struct Payload<'a, A> {
    args: &'a A,
    metadata: Metadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    time_series_matrix: Option<Array2<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    time_series_matrices: Option<Vec<Array2<f64>>>,
    /// `(n_runs, n_samples)`, ascending within each row
    #[serde(skip_serializing_if = "Option::is_none")]
    eigenvalues: Option<Array2<f64>>,
    /// `(n_runs, n_samples (n_samples - 1) / 2)`, strict upper triangles row by row
    #[serde(skip_serializing_if = "Option::is_none")]
    correlations: Option<Array2<f64>>,
}

struct RunResult {
    matrix: Option<Array2<f64>>,
    eigenvalues: Option<Vec<f64>>,
    correlations: Option<Vec<f64>>,
}

/// Simulate the ensemble and store what `output` asks for.
pub fn execute<Sys, D, P, A>(
    ensemble: Ensemble<Sys, D, P, Obs<Sys>>,
    series: &SeriesArgs,
    output: &OutputArgs,
    prefix: &str,
    params: Vec<(&'static str, ParamValue)>,
    args: &A,
) -> anyhow::Result<()>
where
    Sys: Clone + Send + Sync,
    D: Dynamics<Sys>,
    P: Prepare<Sys>,
    A: Serialize,
{
    let stem = output
        .output
        .clone()
        .unwrap_or_else(|| output.dir.join(file_name(prefix, params)));
    let file = DataFile::new(stem, output.format, !output.no_gzip);
    if output.no_overwrite {
        file.ensure_absent()?;
    }
    let spectrum = matches!(output.emit, Emit::Spectrum | Emit::All);
    let keep_matrix = matches!(output.emit, Emit::TsMatrix | Emit::All);
    if spectrum && series.n_samples < 2 {
        bail!("Spectral analysis needs --n-samples >= 2");
    }
    info!(
        "Simulating {} run(s) of {} sample(s) x {} steps (seed {})",
        series.n_runs,
        series.n_samples,
        series.n_steps + 1,
        ensemble.seed
    );
    let timer = std::time::Instant::now();
    let results = ensemble.map_runs(series.n_runs, series.n_samples, |_, mut matrix| {
        let kept = keep_matrix.then(|| matrix.clone());
        let (eigenvalues, correlations) = if spectrum {
            standardize_rows(&mut matrix, output.ddof, output.zero_variance)?;
            let g = correlation_matrix(matrix.view());
            let eig = artificial_systems::analysis::eigenvalues(g.view())?;
            (
                Some(eig),
                output.correlations.then(|| upper_triangle(g.view(), true)),
            )
        } else {
            (None, None)
        };
        Ok::<_, anyhow::Error>(RunResult {
            matrix: kept,
            eigenvalues,
            correlations,
        })
    });
    let results: Vec<RunResult> = results.into_iter().collect::<Result<_, _>>()?;
    info!("Simulated in {:.3} s", timer.elapsed().as_secs_f64());

    let stack = |rows: Vec<Vec<f64>>| -> anyhow::Result<Array2<f64>> {
        let n = rows.first().map_or(0, Vec::len);
        Ok(Array2::from_shape_vec((rows.len(), n), rows.concat())?)
    };
    let mut matrices: Vec<Array2<f64>> = Vec::new();
    let mut eigenvalues = Vec::new();
    let mut correlations = Vec::new();
    for r in results {
        matrices.extend(r.matrix);
        eigenvalues.extend(r.eigenvalues);
        correlations.extend(r.correlations);
    }
    let eigenvalues = (!eigenvalues.is_empty())
        .then(|| stack(eigenvalues))
        .transpose()?;
    let correlations = (!correlations.is_empty())
        .then(|| stack(correlations))
        .transpose()?;
    if let Some(eig) = &eigenvalues {
        let all: Moments = eig.iter().copied().collect();
        let max: Moments = eig
            .index_axis(Axis(1), eig.ncols() - 1)
            .iter()
            .copied()
            .collect();
        info!(
            "Spectrum: <λ> = {:.6}, var(λ) = {:.6}, <λ_max> = {:.6}, var(λ_max) = {:.6}",
            all.mean(),
            all.variance(0),
            max.mean(),
            max.variance(0)
        );
    }
    let (time_series_matrix, time_series_matrices) = if matrices.len() == 1 {
        (matrices.pop(), None)
    } else {
        (None, (!matrices.is_empty()).then_some(matrices))
    };
    let payload = Payload {
        args,
        metadata: Metadata::new(ensemble.seed),
        time_series_matrix,
        time_series_matrices,
        eigenvalues,
        correlations,
    };

    if output.format == Format::Csv {
        let primary = payload
            .time_series_matrix
            .as_ref()
            .or(payload.eigenvalues.as_ref())
            .context("CSV output holds a single matrix: use --n-runs 1 or --emit spectrum")?;
        file.write_csv(primary)?;
        let meta = DataFile::new(file.path().with_extension("meta"), Format::Json, false);
        meta.write(&Payload {
            args,
            metadata: payload.metadata.clone(),
            time_series_matrix: None,
            time_series_matrices: None,
            eigenvalues: None,
            correlations: None,
        })?;
        info!("Metadata written to {}", meta.path().display());
    } else {
        file.write(&payload)?;
    }
    info!("Written to {}", file.path().display());
    Ok(())
}
