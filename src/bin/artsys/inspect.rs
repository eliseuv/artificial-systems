//! Data file inspection.

use std::path::PathBuf;

use artificial_systems::io::{DataFile, Format};
use ciborium::Value;
use clap::Args;

#[derive(Debug, Clone, Args)]
pub struct InspectArgs {
    /// Data file (.cbor, .cbor.gz, .json, .json.gz)
    pub path: PathBuf,
}

/// Print the structure of a data file: scalars in full, arrays by shape.
pub fn inspect(args: &InspectArgs) -> anyhow::Result<()> {
    let file = DataFile::from_path(&args.path)?;
    let value: Value = match file.format() {
        Format::Json => {
            let json: serde_json::Value = file.read()?;
            Value::serialized(&json)?
        }
        _ => file.read()?,
    };
    print(&value, 0);
    Ok(())
}

fn print(value: &Value, indent: usize) {
    let pad = "  ".repeat(indent);
    match value {
        Value::Map(entries) => {
            if let Some(shape) = ndarray_shape(entries) {
                println!("array {shape:?}");
                return;
            }
            println!();
            for (k, v) in entries {
                let key = k.as_text().map_or_else(|| format!("{k:?}"), str::to_owned);
                print!("{pad}{key}: ");
                print(v, indent + 1);
            }
        }
        Value::Array(items) => {
            let shapes: Vec<_> = items
                .iter()
                .filter_map(|v| v.as_map().and_then(|m| ndarray_shape(m)))
                .collect();
            if !shapes.is_empty() && shapes.len() == items.len() {
                println!("{} arrays, first {:?}", items.len(), shapes[0]);
            } else {
                println!("list of {} items", items.len());
            }
        }
        Value::Text(s) => println!("{s}"),
        Value::Integer(i) => println!("{}", i128::from(*i)),
        Value::Float(f) => println!("{f}"),
        Value::Bool(b) => println!("{b}"),
        Value::Null => println!("null"),
        other => println!("{other:?}"),
    }
}

/// Shape of an `ndarray` serialised as `{v, dim, data}`.
fn ndarray_shape(entries: &[(Value, Value)]) -> Option<Vec<u64>> {
    let get = |key: &str| {
        entries
            .iter()
            .find(|(k, _)| k.as_text() == Some(key))
            .map(|(_, v)| v)
    };
    get("data")?;
    get("v")?;
    get("dim")?
        .as_array()?
        .iter()
        .map(|d| d.as_integer().and_then(|i| u64::try_from(i).ok()))
        .collect()
}
