//! End-to-end runs of the `artsys` binary, read back through the library.

#![cfg(feature = "cli")]

use std::{path::Path, process::Command};

use artificial_systems::io::DataFile;
use ndarray::Array2;
use serde::Deserialize;

#[derive(Deserialize)]
struct Metadata {
    seed: u64,
}

#[derive(Deserialize)]
struct Output {
    metadata: Metadata,
    time_series_matrices: Vec<Array2<f64>>,
    eigenvalues: Array2<f64>,
    correlations: Array2<f64>,
}

fn run(stem: &Path, extra: &[&str]) -> Output {
    let status = Command::new(env!("CARGO_BIN_EXE_artsys"))
        .args([
            "contact-process",
            "--dim",
            "1",
            "--length",
            "64",
            "--alpha",
            "3.3",
            "--gamma",
            "0.2",
            "--n-steps",
            "40",
            "--n-samples",
            "5",
            "--n-runs",
            "2",
            "--emit",
            "all",
            "--correlations",
            "--seed",
            "17",
            "--output",
        ])
        .arg(stem)
        .args(extra)
        .status()
        .expect("artsys runs");
    assert!(status.success());
    DataFile::from_path(stem.with_extension("cbor.gz"))
        .unwrap()
        .read()
        .unwrap()
}

#[test]
fn contact_process_output_is_complete_and_reproducible() {
    let dir = std::env::temp_dir().join(format!("artsys-cli-{}", std::process::id()));
    let a = run(&dir.join("a"), &[]);
    let b = run(&dir.join("b"), &["--threads", "1"]);
    std::fs::remove_dir_all(&dir).unwrap();

    assert_eq!(a.metadata.seed, 17);
    assert_eq!(a.time_series_matrices.len(), 2);
    assert_eq!(a.time_series_matrices[0].dim(), (5, 41));
    // All-active initial state
    assert!(
        a.time_series_matrices[0]
            .column(0)
            .iter()
            .all(|&d| d == 1.0)
    );
    assert_eq!(a.eigenvalues.dim(), (2, 5));
    assert_eq!(a.correlations.dim(), (2, 10));
    for row in a.eigenvalues.rows() {
        assert!(row.iter().zip(row.iter().skip(1)).all(|(x, y)| x <= y));
    }
    assert_eq!(a.time_series_matrices, b.time_series_matrices);
    assert_eq!(a.eigenvalues, b.eigenvalues);
}
