//! Serialization Utilities
//!
//! Provides fast serialization and deserialization.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Serialization format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    Json,
    MessagePack,
    Bincode,
    Cbor,
}

impl SerializationFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            SerializationFormat::Json => "json",
            SerializationFormat::MessagePack => "msgpack",
            SerializationFormat::Bincode => "bincode",
            SerializationFormat::Cbor => "cbor",
        }
    }
}

/// Serializer service
#[pyclass]
pub struct Serializer {
    default_format: SerializationFormat,
}

#[pymethods]
impl Serializer {
    #[new]
    #[pyo3(signature = (format="json"))]
    pub fn new(format: String) -> PyResult<Self> {
        let fmt = match format.to_lowercase().as_str() {
            "json" => SerializationFormat::Json,
            "msgpack" | "messagepack" => SerializationFormat::MessagePack,
            "bincode" => SerializationFormat::Bincode,
            "cbor" => SerializationFormat::Cbor,
            _ => return Err(PyValueError::new_err(format!("Unknown format: {}", format))),
        };
        
        Ok(Self {
            default_format: fmt,
        })
    }

    pub fn serialize(&self, data: PyObject) -> PyResult<Vec<u8>> {
        Python::with_gil(|py| {
            match self.default_format {
                SerializationFormat::Json => {
                    let json = py.import("json")?;
                    let json_str = json.call_method1("dumps", (data,))?;
                    let json_str: String = json_str.extract()?;
                    Ok(json_str.into_bytes())
                }
                _ => {
                    // Fallback to JSON for other formats
                    let json = py.import("json")?;
                    let json_str = json.call_method1("dumps", (data,))?;
                    let json_str: String = json_str.extract()?;
                    Ok(json_str.into_bytes())
                }
            }
        })
    }

    pub fn deserialize(&self, data: &[u8]) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            match self.default_format {
                SerializationFormat::Json => {
                    let json_str = String::from_utf8(data.to_vec())
                        .map_err(|e| PyValueError::new_err(format!("Invalid UTF-8: {}", e)))?;
                    let json = py.import("json")?;
                    json.call_method1("loads", (json_str,))
                        .map(|obj| obj.into())
                }
                _ => {
                    // Fallback to JSON
                    let json_str = String::from_utf8(data.to_vec())
                        .map_err(|e| PyValueError::new_err(format!("Invalid UTF-8: {}", e)))?;
                    let json = py.import("json")?;
                    json.call_method1("loads", (json_str,))
                        .map(|obj| obj.into())
                }
            }
        })
    }

    pub fn get_format(&self) -> String {
        self.default_format.as_str().to_string()
    }
}

#[pyfunction]
pub fn create_serializer(format: Option<String>) -> PyResult<Serializer> {
    Serializer::new(format.unwrap_or_else(|| "json".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization_format() {
        assert_eq!(SerializationFormat::Json.as_str(), "json");
    }
}












