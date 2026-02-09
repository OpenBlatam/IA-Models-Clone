//! Serialization Module - High Performance Data Serialization
//!
//! Provides blazing fast serialization options:
//! - simd-json: SIMD-accelerated JSON parsing (~3x faster than serde_json)
//! - Bincode: Ultra-fast binary serialization
//! - MessagePack: Compact binary format
//! - CBOR: Concise Binary Object Representation

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

use crate::error::CoreError;

/// High-performance serialization service
#[pyclass]
pub struct SerializationService {
    pretty_print: bool,
}

#[pymethods]
impl SerializationService {
    #[new]
    #[pyo3(signature = (pretty_print=false))]
    fn new(pretty_print: bool) -> Self {
        Self { pretty_print }
    }

    // ==================== JSON ====================

    /// Serialize to JSON using serde_json (fast)
    fn to_json(&self, data: &str) -> PyResult<String> {
        // Parse the input as JSON to validate it
        let value: serde_json::Value = serde_json::from_str(data)
            .map_err(|e| CoreError::serialization_error(format!("Invalid JSON: {}", e)))?;
        
        if self.pretty_print {
            serde_json::to_string_pretty(&value)
                .map_err(|e| CoreError::serialization_error(format!("JSON serialization failed: {}", e)).into())
        } else {
            serde_json::to_string(&value)
                .map_err(|e| CoreError::serialization_error(format!("JSON serialization failed: {}", e)).into())
        }
    }

    /// Parse JSON using simd-json (SIMD accelerated, ~3x faster)
    fn parse_json_simd(&self, data: &str) -> PyResult<String> {
        // Make a mutable copy for simd-json
        let mut data_copy = data.as_bytes().to_vec();
        
        let value: simd_json::OwnedValue = simd_json::to_owned_value(&mut data_copy)
            .map_err(|e| CoreError::serialization_error(format!("SIMD JSON parse failed: {}", e)))?;
        
        // Convert back to string for Python
        serde_json::to_string(&value)
            .map_err(|e| CoreError::serialization_error(format!("JSON serialization failed: {}", e)).into())
    }

    /// Parse multiple JSON strings in parallel
    fn parse_json_batch(&self, items: Vec<String>) -> PyResult<Vec<String>> {
        let results: Result<Vec<String>, CoreError> = items
            .par_iter()
            .map(|data| {
                let mut data_copy = data.as_bytes().to_vec();
                let value: simd_json::OwnedValue = simd_json::to_owned_value(&mut data_copy)
                    .map_err(|e| CoreError::serialization_error(format!("Parse failed: {}", e)))?;
                serde_json::to_string(&value)
                    .map_err(|e| CoreError::serialization_error(format!("Serialize failed: {}", e)))
            })
            .collect();
        
        results.map_err(|e| e.into())
    }

    /// Validate JSON without full parsing (fast validation)
    fn validate_json(&self, data: &str) -> bool {
        let mut data_copy = data.as_bytes().to_vec();
        simd_json::to_owned_value(&mut data_copy).is_ok()
    }

    /// Validate multiple JSON strings in parallel
    fn validate_json_batch(&self, items: Vec<String>) -> Vec<bool> {
        items
            .par_iter()
            .map(|data| {
                let mut data_copy = data.as_bytes().to_vec();
                simd_json::to_owned_value(&mut data_copy).is_ok()
            })
            .collect()
    }

    // ==================== MSGPACK ====================

    /// Serialize to MessagePack (compact binary format)
    fn to_msgpack<'py>(&self, py: Python<'py>, json_data: &str) -> PyResult<Bound<'py, PyBytes>> {
        let value: serde_json::Value = serde_json::from_str(json_data)
            .map_err(|e| CoreError::serialization_error(format!("Invalid JSON: {}", e)))?;
        
        let msgpack_bytes = rmp_serde::to_vec(&value)
            .map_err(|e| CoreError::serialization_error(format!("MessagePack serialization failed: {}", e)))?;
        
        Ok(PyBytes::new_bound(py, &msgpack_bytes))
    }

    /// Deserialize from MessagePack
    fn from_msgpack(&self, data: &[u8]) -> PyResult<String> {
        let value: serde_json::Value = rmp_serde::from_slice(data)
            .map_err(|e| CoreError::serialization_error(format!("MessagePack deserialization failed: {}", e)))?;
        
        serde_json::to_string(&value)
            .map_err(|e| CoreError::serialization_error(format!("JSON serialization failed: {}", e)).into())
    }

    /// Batch serialize to MessagePack
    fn to_msgpack_batch<'py>(&self, py: Python<'py>, items: Vec<String>) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let results: Result<Vec<Vec<u8>>, CoreError> = items
            .par_iter()
            .map(|json_data| {
                let value: serde_json::Value = serde_json::from_str(json_data)
                    .map_err(|e| CoreError::serialization_error(format!("Invalid JSON: {}", e)))?;
                rmp_serde::to_vec(&value)
                    .map_err(|e| CoreError::serialization_error(format!("MessagePack failed: {}", e)))
            })
            .collect();
        
        let bytes_vec = results?;
        Ok(bytes_vec.iter().map(|b| PyBytes::new_bound(py, b)).collect())
    }

    // ==================== BINCODE ====================

    /// Serialize to Bincode (fastest binary format)
    fn to_bincode<'py>(&self, py: Python<'py>, json_data: &str) -> PyResult<Bound<'py, PyBytes>> {
        let value: serde_json::Value = serde_json::from_str(json_data)
            .map_err(|e| CoreError::serialization_error(format!("Invalid JSON: {}", e)))?;
        
        let bincode_bytes = bincode::serialize(&value)
            .map_err(|e| CoreError::serialization_error(format!("Bincode serialization failed: {}", e)))?;
        
        Ok(PyBytes::new_bound(py, &bincode_bytes))
    }

    /// Deserialize from Bincode
    fn from_bincode(&self, data: &[u8]) -> PyResult<String> {
        let value: serde_json::Value = bincode::deserialize(data)
            .map_err(|e| CoreError::serialization_error(format!("Bincode deserialization failed: {}", e)))?;
        
        serde_json::to_string(&value)
            .map_err(|e| CoreError::serialization_error(format!("JSON serialization failed: {}", e)).into())
    }

    // ==================== CBOR ====================

    /// Serialize to CBOR (Concise Binary Object Representation)
    fn to_cbor<'py>(&self, py: Python<'py>, json_data: &str) -> PyResult<Bound<'py, PyBytes>> {
        let value: serde_json::Value = serde_json::from_str(json_data)
            .map_err(|e| CoreError::serialization_error(format!("Invalid JSON: {}", e)))?;
        
        let mut cbor_bytes = Vec::new();
        ciborium::into_writer(&value, &mut cbor_bytes)
            .map_err(|e| CoreError::serialization_error(format!("CBOR serialization failed: {}", e)))?;
        
        Ok(PyBytes::new_bound(py, &cbor_bytes))
    }

    /// Deserialize from CBOR
    fn from_cbor(&self, data: &[u8]) -> PyResult<String> {
        let value: serde_json::Value = ciborium::from_reader(data)
            .map_err(|e| CoreError::serialization_error(format!("CBOR deserialization failed: {}", e)))?;
        
        serde_json::to_string(&value)
            .map_err(|e| CoreError::serialization_error(format!("JSON serialization failed: {}", e)).into())
    }

    // ==================== UTILITIES ====================

    /// Compare sizes of different serialization formats
    fn compare_formats(&self, json_data: &str) -> PyResult<Vec<(String, usize)>> {
        let value: serde_json::Value = serde_json::from_str(json_data)
            .map_err(|e| CoreError::serialization_error(format!("Invalid JSON: {}", e)))?;
        
        let json_size = serde_json::to_string(&value)
            .map(|s| s.len())
            .unwrap_or(0);
        
        let msgpack_size = rmp_serde::to_vec(&value)
            .map(|v| v.len())
            .unwrap_or(0);
        
        let bincode_size = bincode::serialize(&value)
            .map(|v| v.len())
            .unwrap_or(0);
        
        let mut cbor_bytes = Vec::new();
        let cbor_size = if ciborium::into_writer(&value, &mut cbor_bytes).is_ok() {
            cbor_bytes.len()
        } else {
            0
        };
        
        Ok(vec![
            ("json".to_string(), json_size),
            ("msgpack".to_string(), msgpack_size),
            ("bincode".to_string(), bincode_size),
            ("cbor".to_string(), cbor_size),
        ])
    }

    /// Benchmark serialization performance
    fn benchmark_serialization(&self, json_data: &str, iterations: usize) -> PyResult<Vec<(String, f64)>> {
        let value: serde_json::Value = serde_json::from_str(json_data)
            .map_err(|e| CoreError::serialization_error(format!("Invalid JSON: {}", e)))?;
        
        let mut results = Vec::new();
        
        // Benchmark JSON
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = serde_json::to_string(&value);
        }
        let json_time = start.elapsed().as_secs_f64() * 1000.0;
        results.push(("json".to_string(), json_time));
        
        // Benchmark MessagePack
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = rmp_serde::to_vec(&value);
        }
        let msgpack_time = start.elapsed().as_secs_f64() * 1000.0;
        results.push(("msgpack".to_string(), msgpack_time));
        
        // Benchmark Bincode
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = bincode::serialize(&value);
        }
        let bincode_time = start.elapsed().as_secs_f64() * 1000.0;
        results.push(("bincode".to_string(), bincode_time));
        
        // Benchmark CBOR
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let mut bytes = Vec::new();
            let _ = ciborium::into_writer(&value, &mut bytes);
        }
        let cbor_time = start.elapsed().as_secs_f64() * 1000.0;
        results.push(("cbor".to_string(), cbor_time));
        
        Ok(results)
    }

    /// Minify JSON (remove whitespace)
    fn minify_json(&self, json_data: &str) -> PyResult<String> {
        let value: serde_json::Value = serde_json::from_str(json_data)
            .map_err(|e| CoreError::serialization_error(format!("Invalid JSON: {}", e)))?;
        
        serde_json::to_string(&value)
            .map_err(|e| CoreError::serialization_error(format!("Minify failed: {}", e)).into())
    }

    /// Pretty print JSON with indentation
    fn prettify_json(&self, json_data: &str) -> PyResult<String> {
        let value: serde_json::Value = serde_json::from_str(json_data)
            .map_err(|e| CoreError::serialization_error(format!("Invalid JSON: {}", e)))?;
        
        serde_json::to_string_pretty(&value)
            .map_err(|e| CoreError::serialization_error(format!("Prettify failed: {}", e)).into())
    }

    /// Deep copy JSON (useful for ensuring no references)
    fn deep_copy(&self, json_data: &str) -> PyResult<String> {
        let value: serde_json::Value = serde_json::from_str(json_data)
            .map_err(|e| CoreError::serialization_error(format!("Invalid JSON: {}", e)))?;
        
        let cloned = value.clone();
        serde_json::to_string(&cloned)
            .map_err(|e| CoreError::serialization_error(format!("Clone failed: {}", e)).into())
    }

    /// Get JSON path value (simple implementation)
    fn get_json_path(&self, json_data: &str, path: &str) -> PyResult<String> {
        let value: serde_json::Value = serde_json::from_str(json_data)
            .map_err(|e| CoreError::serialization_error(format!("Invalid JSON: {}", e)))?;
        
        let parts: Vec<&str> = path.split('.').filter(|s| !s.is_empty()).collect();
        let mut current = &value;
        
        for part in parts {
            current = current.get(part)
                .or_else(|| {
                    // Try as array index
                    part.parse::<usize>().ok().and_then(|i| current.get(i))
                })
                .ok_or_else(|| CoreError::serialization_error(format!("Path not found: {}", part)))?;
        }
        
        serde_json::to_string(current)
            .map_err(|e| CoreError::serialization_error(format!("Serialize failed: {}", e)).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_roundtrip() {
        let service = SerializationService::new(false);
        let json = r#"{"name": "test", "value": 42}"#;
        let result = service.to_json(json).unwrap();
        assert!(result.contains("test"));
    }

    #[test]
    fn test_msgpack_roundtrip() {
        Python::with_gil(|py| {
            let service = SerializationService::new(false);
            let json = r#"{"name": "test", "value": 42}"#;
            let packed = service.to_msgpack(py, json).unwrap();
            let unpacked = service.from_msgpack(packed.as_bytes()).unwrap();
            assert!(unpacked.contains("test"));
        });
    }

    #[test]
    fn test_json_validation() {
        let service = SerializationService::new(false);
        assert!(service.validate_json(r#"{"valid": true}"#));
        assert!(!service.validate_json(r#"{"invalid": }"#));
    }
}












