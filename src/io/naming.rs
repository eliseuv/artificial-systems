//! Parameter encoding in file names: `Prefix_key1=value1_key2=value2`.

use std::{collections::BTreeMap, fmt};

use serde::{Deserialize, Serialize};

/// Parameter value in a file name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValue {
    /// Integer.
    Int(i64),
    /// Floating point number (written in shortest round-trip form).
    Float(f64),
    /// Boolean.
    Bool(bool),
    /// Anything else.
    Str(String),
}

impl fmt::Display for ParamValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v:?}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Str(v) => f.write_str(v),
        }
    }
}

impl ParamValue {
    /// Typed value of a string: integer, then float, then boolean, else string.
    pub fn parse(s: &str) -> Self {
        if let Ok(v) = s.parse::<i64>() {
            Self::Int(v)
        } else if let Ok(v) = s.parse::<f64>() {
            Self::Float(v)
        } else if let Ok(v) = s.parse::<bool>() {
            Self::Bool(v)
        } else {
            Self::Str(s.to_owned())
        }
    }

    /// Numeric value, if any.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int(v) => Some(*v as f64),
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }
}

macro_rules! impl_from {
    ($variant:ident: $($ty:ty),+) => {
        $(impl From<$ty> for ParamValue {
            fn from(v: $ty) -> Self {
                Self::$variant(v.into())
            }
        })+
    };
}

impl_from!(Int: i64, i32, u32, u16, u8);
impl_from!(Float: f64, f32);
impl_from!(Bool: bool);
impl_from!(Str: String, &str);

impl From<usize> for ParamValue {
    fn from(v: usize) -> Self {
        Self::Int(i64::try_from(v).expect("parameter too large"))
    }
}

impl From<u64> for ParamValue {
    fn from(v: u64) -> Self {
        Self::Int(i64::try_from(v).expect("parameter too large"))
    }
}

/// File name `prefix_k1=v1_k2=v2…` with keys sorted alphabetically.
///
/// Floats are written in shortest round-trip form (`0.1`, `3.29785`, `1e-5`), so the name holds
/// the exact parameter values.
///
/// # Panics
/// If the prefix contains `_` or `=`, a key contains `=`, or a value contains `_` or `=` (they
/// would make the name ambiguous).
pub fn file_name<'a>(
    prefix: &str,
    params: impl IntoIterator<Item = (&'a str, ParamValue)>,
) -> String {
    assert!(
        !prefix.contains(['_', '=']),
        "Prefix must not contain '_' or '=': {prefix}"
    );
    let params: BTreeMap<&str, ParamValue> = params.into_iter().collect();
    let mut name = prefix.to_owned();
    for (key, value) in params {
        let value = value.to_string();
        assert!(!key.contains('='), "Key must not contain '=': {key}");
        assert!(
            !value.contains(['_', '=']),
            "Value must not contain '_' or '=': {value}"
        );
        name.push('_');
        name.push_str(key);
        name.push('=');
        name.push_str(&value);
    }
    name
}

/// Prefix and parameters of a name produced by [`file_name`] (extensions must be removed first).
///
/// Keys may contain `_` (e.g. `n_steps`): chunks without `=` are joined to the following key.
pub fn parse_file_name(name: &str) -> (String, BTreeMap<String, ParamValue>) {
    let mut chunks = name.split('_');
    let prefix = chunks.next().unwrap_or_default().to_owned();
    let mut params = BTreeMap::new();
    let mut pending: Vec<&str> = Vec::new();
    for chunk in chunks {
        match chunk.split_once('=') {
            Some((key, value)) => {
                pending.push(key);
                params.insert(pending.join("_"), ParamValue::parse(value));
                pending.clear();
            }
            None => pending.push(chunk),
        }
    }
    (prefix, params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        let name = file_name(
            "ContactProcess",
            [
                ("n_steps", 500usize.into()),
                ("L", 128usize.into()),
                ("alpha", 3.29785.into()),
                ("gamma", 0.1.into()),
                ("tiny", 1e-5.into()),
                ("big", 1e20.into()),
                ("init", "all-active".into()),
                ("gzip", true.into()),
            ],
        );
        assert_eq!(
            name,
            "ContactProcess_L=128_alpha=3.29785_big=1e20_gamma=0.1_gzip=true_init=all-active_n_steps=500_tiny=1e-5"
        );
        let (prefix, params) = parse_file_name(&name);
        assert_eq!(prefix, "ContactProcess");
        assert_eq!(params["n_steps"], ParamValue::Int(500));
        assert_eq!(params["alpha"], ParamValue::Float(3.29785));
        assert_eq!(params["tiny"], ParamValue::Float(1e-5));
        assert_eq!(params["big"], ParamValue::Float(1e20));
        assert_eq!(params["gzip"], ParamValue::Bool(true));
        assert_eq!(params["init"], ParamValue::Str("all-active".into()));
    }

    #[test]
    fn parses_legacy_names() {
        let (prefix, params) = parse_file_name(
            "ContactProcessDiffusion1D_L=128_rate=2.0_diffusion=0.05_n_steps=500_n_samples=100000",
        );
        assert_eq!(prefix, "ContactProcessDiffusion1D");
        assert_eq!(params["rate"].as_f64(), Some(2.0));
        assert_eq!(params["n_samples"], ParamValue::Int(100_000));
    }

    #[test]
    #[should_panic(expected = "Value must not contain")]
    fn rejects_ambiguous_values() {
        file_name("X", [("k", "a_b".into())]);
    }
}
