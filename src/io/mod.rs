//! Data files.
//!
//! Serialised outputs are CBOR (default) or JSON, optionally gzip compressed, with the format
//! given by the file extension (`.cbor`, `.json`, `.cbor.gz`, …). Numeric matrices can also be
//! written as CSV. Matrices use the `ndarray` serde layout `{v, dim, data}` (row-major data).

use std::{
    ffi::OsString,
    fmt,
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use ndarray::Array2;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

mod naming;

pub use naming::{ParamValue, file_name, parse_file_name};

/// Errors of the data file routines.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// Underlying I/O failure.
    #[error("I/O error on {path}: {source}")]
    Io {
        /// File involved.
        path: PathBuf,
        /// Cause.
        source: io::Error,
    },
    /// The file extension does not identify a supported format.
    #[error("unrecognised data file extension: {0}")]
    UnknownFormat(PathBuf),
    /// The file already exists and overwriting was not allowed.
    #[error("refusing to overwrite existing file {0}")]
    Exists(PathBuf),
    /// Serialisation or deserialisation failure.
    #[error("(de)serialisation error on {path}: {message}")]
    Serde {
        /// File involved.
        path: PathBuf,
        /// Cause.
        message: String,
    },
    /// The requested operation is not available for the format.
    #[error("{0} files do not support this operation")]
    Unsupported(Format),
}

/// Serialisation format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[serde(rename_all = "lowercase")]
pub enum Format {
    /// Concise Binary Object Representation.
    #[default]
    Cbor,
    /// JSON (non-finite numbers, e.g. `β = ∞`, are written as `null`).
    Json,
    /// Comma separated values (numeric matrices only).
    Csv,
}

impl Format {
    /// File extension.
    pub fn extension(self) -> &'static str {
        match self {
            Self::Cbor => "cbor",
            Self::Json => "json",
            Self::Csv => "csv",
        }
    }

    fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "cbor" => Some(Self::Cbor),
            "json" => Some(Self::Json),
            "csv" => Some(Self::Csv),
            _ => None,
        }
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.extension())
    }
}

/// Data file with a known format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataFile {
    path: PathBuf,
    format: Format,
    gzip: bool,
}

impl DataFile {
    /// File at `stem` with the extension of `format` (and `.gz` if `gzip`) appended.
    pub fn new(stem: impl AsRef<Path>, format: Format, gzip: bool) -> Self {
        let mut name = OsString::from(stem.as_ref().as_os_str());
        name.push(".");
        name.push(format.extension());
        if gzip {
            name.push(".gz");
        }
        Self {
            path: name.into(),
            format,
            gzip,
        }
    }

    /// File whose format is determined by the extension(s) of `path`.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path = path.as_ref();
        let unknown = || IoError::UnknownFormat(path.to_owned());
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(unknown)?;
        let (format, gzip) = if ext.eq_ignore_ascii_case("gz") {
            let inner = Path::new(path.file_stem().ok_or_else(unknown)?)
                .extension()
                .and_then(|e| e.to_str())
                .ok_or_else(unknown)?;
            (Format::from_extension(inner).ok_or_else(unknown)?, true)
        } else {
            (Format::from_extension(ext).ok_or_else(unknown)?, false)
        };
        Ok(Self {
            path: path.to_owned(),
            format,
            gzip,
        })
    }

    /// Full path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Format.
    pub fn format(&self) -> Format {
        self.format
    }

    /// Whether the file is gzip compressed.
    pub fn is_gzip(&self) -> bool {
        self.gzip
    }

    fn io_error(&self, source: io::Error) -> IoError {
        IoError::Io {
            path: self.path.clone(),
            source,
        }
    }

    fn serde_error(&self, e: impl fmt::Display) -> IoError {
        IoError::Serde {
            path: self.path.clone(),
            message: e.to_string(),
        }
    }

    /// Fail if the file exists.
    pub fn ensure_absent(&self) -> Result<(), IoError> {
        match self.path.try_exists() {
            Ok(false) => Ok(()),
            Ok(true) => Err(IoError::Exists(self.path.clone())),
            Err(e) => Err(self.io_error(e)),
        }
    }

    /// Create the parent directory if needed.
    pub fn create_parent_dir(&self) -> Result<(), IoError> {
        match self.path.parent() {
            Some(dir) if !dir.as_os_str().is_empty() => {
                fs::create_dir_all(dir).map_err(|e| self.io_error(e))
            }
            _ => Ok(()),
        }
    }

    /// Run `f` on a (compressing) writer to the file, finishing the stream explicitly so trailer
    /// write errors are reported.
    fn with_writer(
        &self,
        f: impl FnOnce(&mut dyn Write) -> Result<(), IoError>,
    ) -> Result<(), IoError> {
        self.create_parent_dir()?;
        let mut file = BufWriter::new(File::create(&self.path).map_err(|e| self.io_error(e))?);
        if self.gzip {
            // Serialisers issue many tiny writes, which are slow to feed to the compressor directly
            let mut encoder = BufWriter::new(GzEncoder::new(&mut file, Compression::default()));
            f(&mut encoder)?;
            encoder
                .into_inner()
                .map_err(|e| self.io_error(e.into_error()))?
                .finish()
                .map_err(|e| self.io_error(e))?;
        } else {
            f(&mut file)?;
        }
        file.flush().map_err(|e| self.io_error(e))
    }

    fn reader(&self) -> Result<Box<dyn Read>, IoError> {
        let file = BufReader::new(File::open(&self.path).map_err(|e| self.io_error(e))?);
        Ok(if self.gzip {
            Box::new(GzDecoder::new(file))
        } else {
            Box::new(file)
        })
    }

    /// Serialise `data` (CBOR or JSON).
    pub fn write<T: Serialize + ?Sized>(&self, data: &T) -> Result<(), IoError> {
        match self.format {
            Format::Cbor => self
                .with_writer(|w| ciborium::into_writer(data, w).map_err(|e| self.serde_error(e))),
            Format::Json => self
                .with_writer(|w| serde_json::to_writer(w, data).map_err(|e| self.serde_error(e))),
            Format::Csv => Err(IoError::Unsupported(Format::Csv)),
        }
    }

    /// Deserialise the file contents (CBOR or JSON).
    pub fn read<T: DeserializeOwned>(&self) -> Result<T, IoError> {
        let reader = self.reader()?;
        match self.format {
            Format::Cbor => ciborium::from_reader(reader).map_err(|e| self.serde_error(e)),
            Format::Json => serde_json::from_reader(reader).map_err(|e| self.serde_error(e)),
            Format::Csv => Err(IoError::Unsupported(Format::Csv)),
        }
    }

    /// Write a matrix as CSV, one row per line and no header.
    pub fn write_csv<T: fmt::Display>(&self, matrix: &Array2<T>) -> Result<(), IoError> {
        if self.format != Format::Csv {
            return Err(IoError::Unsupported(self.format));
        }
        self.with_writer(|w| {
            let mut writer = csv::WriterBuilder::new().has_headers(false).from_writer(w);
            for row in matrix.rows() {
                writer
                    .write_record(row.iter().map(|x| x.to_string()))
                    .map_err(|e| self.serde_error(e))?;
            }
            writer.flush().map_err(|e| self.io_error(e))
        })
    }
}

/// Provenance recorded alongside every output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metadata {
    /// Crate name and version that produced the data.
    pub generator: String,
    /// Master seed of the ensemble.
    pub seed: u64,
    /// Seconds since the Unix epoch at creation.
    pub created: u64,
}

impl Metadata {
    /// Metadata for data generated now with master seed `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            generator: concat!(env!("CARGO_PKG_NAME"), " ", env!("CARGO_PKG_VERSION")).to_owned(),
            seed,
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn extensions_round_trip() {
        let f = DataFile::new("out/run_L=128", Format::Cbor, true);
        assert_eq!(f.path(), Path::new("out/run_L=128.cbor.gz"));
        assert_eq!(DataFile::from_path(f.path()).unwrap(), f);
        let f = DataFile::from_path("a/b.rate=3.5.json").unwrap();
        assert_eq!((f.format(), f.is_gzip()), (Format::Json, false));
        assert!(matches!(
            DataFile::from_path("noext"),
            Err(IoError::UnknownFormat(_))
        ));
        assert!(matches!(
            DataFile::from_path("x.gz"),
            Err(IoError::UnknownFormat(_))
        ));
        assert!(DataFile::from_path("x.txt.gz").is_err());
    }

    #[test]
    fn write_read_round_trip() {
        let dir = std::env::temp_dir().join(format!("artsys-io-{}", std::process::id()));
        let matrix = array![[1.5, 2.0], [3.0, -4.25]];
        for (format, gzip) in [
            (Format::Cbor, true),
            (Format::Cbor, false),
            (Format::Json, true),
        ] {
            let f = DataFile::new(dir.join("nested/matrix"), format, gzip);
            f.write(&matrix).unwrap();
            assert!(f.ensure_absent().is_err());
            let back: Array2<f64> = DataFile::from_path(f.path()).unwrap().read().unwrap();
            assert_eq!(back, matrix);
        }
        let f = DataFile::new(dir.join("matrix"), Format::Csv, false);
        f.write_csv(&matrix).unwrap();
        assert_eq!(fs::read_to_string(f.path()).unwrap(), "1.5,2\n3,-4.25\n");
        assert!(matches!(f.write(&matrix), Err(IoError::Unsupported(_))));
        fs::remove_dir_all(dir).unwrap();
    }
}
