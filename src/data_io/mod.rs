//! Data Files Interface
//!

use anyhow::Context;
use flate2::{Compression, write::GzEncoder};
use serde::Serialize;
use std::{
    ffi::{OsStr, OsString},
    fmt::{self, Display},
    fs::{File, create_dir_all},
    io::{self, Write},
    path::{Path, PathBuf},
    str::FromStr,
};

/// Data file specification
#[derive(Debug, Clone, Copy, Default, Serialize, clap::ValueEnum)]
pub enum DataFileSpec {
    #[default]
    CBOR,
    CSV,
}

impl DataFileSpec {
    /// Extension associated with specification
    fn extension(&self) -> OsString {
        match self {
            DataFileSpec::CBOR => "cbor".into(),
            DataFileSpec::CSV => "csv".into(),
        }
    }

    /// Get data file specification from extension
    fn from_extension(extension: &OsStr) -> Option<Self> {
        match extension
            .to_str()
            .expect("Unable to convert extension OS string into string!")
            .to_lowercase()
            .as_str()
        {
            "cbor" => Some(Self::CBOR),
            "csv" => Some(Self::CSV),
            _ => None,
        }
    }

    /// Create data file format from specification
    pub fn file_format(&self, compressed: bool) -> DataFileFormat {
        DataFileFormat {
            spec: *self,
            compression: if compressed {
                Some(Compression::default())
            } else {
                None
            },
        }
    }

    /// Serialize data to specification
    pub fn into_writer(
        &self,
        data: &impl Serialize,
        writer: &mut impl Write,
    ) -> anyhow::Result<()> {
        match self {
            DataFileSpec::CBOR => {
                ciborium::into_writer(data, writer).context("Unable to write CBOR file")
            }
            DataFileSpec::CSV => csv::Writer::from_writer(writer)
                .serialize(data)
                .context("Unable to write CSV file"),
        }
    }
}

impl Display for DataFileSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            DataFileSpec::CBOR => "CBOR",
            DataFileSpec::CSV => "CSV",
        };
        write!(f, "{name}")
    }
}

/// Data file format
#[derive(Debug, Clone, Copy)]
pub struct DataFileFormat {
    pub spec: DataFileSpec,
    pub compression: Option<Compression>,
}

impl DataFileFormat {
    /// Get extension associated with data file format
    pub fn extension(&self) -> OsString {
        let mut extension = self.spec.extension();

        if self.compression.is_some() {
            extension.push(".gz");
        }

        extension
    }

    /// Parse data file format from path
    pub fn from_path(path: &Path) -> Option<Self> {
        let mut extensions = Vec::new();
        while let Some(ext) = path.extension() {
            extensions.push(ext);
        }
        extensions.reverse();

        match extensions
            .into_iter()
            .map(|x| x.to_str().expect(""))
            .collect::<Vec<_>>()[..]
        {
            [.., ext, "gz"] => {
                let compression = Some(Compression::default());
                let specification = DataFileSpec::from_extension(&OsString::from_str(ext).unwrap());
                specification.map(|spec| DataFileFormat { spec, compression })
            }
            [.., ext] => {
                let compression = None;
                let specification = DataFileSpec::from_extension(&OsString::from_str(ext).unwrap());
                specification.map(|spec| DataFileFormat { spec, compression })
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct DataFile {
    path: PathBuf,
    format: DataFileFormat,
}

impl DataFile {
    /// Create new data file with given specification
    pub fn new(path: &Path, spec: DataFileSpec, compessed: bool) -> Self {
        let format = spec.file_format(compessed);
        let path = path.with_extension(format.extension());
        Self { path, format }
    }

    /// Test `path` and return an error if it is a file to avoid overwriting
    pub fn no_overwrite_test(&self) -> io::Result<()> {
        if self.path.try_exists()? {
            log::error!("File already exists: {path}", path = self.path.display());
            Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "Skipping already existing file!",
            ))
        } else {
            Ok(())
        }
    }

    /// Create parent directory
    pub fn create_parent_dir(&self) -> io::Result<()> {
        let parent_dir = self.path.parent().ok_or(io::Error::new(
            io::ErrorKind::NotFound,
            "Unable to determine parent directory!",
        ))?;
        log::debug!("Parent directory: {}", parent_dir.display());
        if !parent_dir.try_exists()? {
            log::debug!("Creating parent directory...");
            create_dir_all(parent_dir)?;
        }
        Ok(())
    }

    /// Write data to file
    pub fn write(&self, data: &impl Serialize) -> anyhow::Result<()> {
        log::debug!("Writing data to file: {path}", path = self.path.display());
        let mut file = File::create(&self.path)?;
        if let Some(compression) = self.format.compression {
            let mut encoder = GzEncoder::new(file, compression);
            self.format.spec.into_writer(data, &mut encoder)
        } else {
            self.format.spec.into_writer(data, &mut file)
        }
    }
}
