//! SIMD-accelerated JSON Processing
//!
//! Uses simd-json for ultra-fast JSON parsing on supported CPUs.
//! Falls back to serde_json on unsupported platforms.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[pyclass]
pub struct SimdJsonService {
    use_simd: bool,
}

#[pymethods]
impl SimdJsonService {
    #[new]
    pub fn new() -> Self {
        let use_simd = is_simd_available();
        Self { use_simd }
    }

    pub fn parse(&self, json_str: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            if self.use_simd {
                let mut bytes = json_str.as_bytes().to_vec();
                match simd_json::serde::from_slice::<serde_json::Value>(&mut bytes) {
                    Ok(value) => value_to_pyobject(py, &value),
                    Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e.to_string())),
                }
            } else {
                match serde_json::from_str::<serde_json::Value>(json_str) {
                    Ok(value) => value_to_pyobject(py, &value),
                    Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e.to_string())),
                }
            }
        })
    }

    pub fn parse_to_string(&self, json_str: &str) -> PyResult<String> {
        if self.use_simd {
            let mut bytes = json_str.as_bytes().to_vec();
            let value: serde_json::Value = simd_json::serde::from_slice(&mut bytes)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            serde_json::to_string(&value)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        } else {
            let value: serde_json::Value = serde_json::from_str(json_str)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            serde_json::to_string(&value)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        }
    }

    pub fn stringify(&self, obj: &PyAny) -> PyResult<String> {
        let value = pyobject_to_value(obj)?;
        serde_json::to_string(&value)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    pub fn stringify_pretty(&self, obj: &PyAny) -> PyResult<String> {
        let value = pyobject_to_value(obj)?;
        serde_json::to_string_pretty(&value)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    pub fn is_valid(&self, json_str: &str) -> bool {
        if self.use_simd {
            let mut bytes = json_str.as_bytes().to_vec();
            simd_json::serde::from_slice::<serde_json::Value>(&mut bytes).is_ok()
        } else {
            serde_json::from_str::<serde_json::Value>(json_str).is_ok()
        }
    }

    pub fn parse_batch(&self, json_strings: Vec<String>) -> Vec<PyResult<PyObject>> {
        Python::with_gil(|py| {
            if self.use_simd {
                json_strings
                    .into_iter()
                    .map(|s| {
                        let mut bytes = s.as_bytes().to_vec();
                        match simd_json::serde::from_slice::<serde_json::Value>(&mut bytes) {
                            Ok(value) => value_to_pyobject(py, &value),
                            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e.to_string())),
                        }
                    })
                    .collect()
            } else {
                json_strings
                    .into_iter()
                    .map(|s| {
                        match serde_json::from_str::<serde_json::Value>(&s) {
                            Ok(value) => value_to_pyobject(py, &value),
                            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e.to_string())),
                        }
                    })
                    .collect()
            }
        })
    }

    pub fn using_simd(&self) -> bool {
        self.use_simd
    }

    pub fn benchmark_parse(&self, json_str: &str, iterations: u32) -> JsonBenchmark {
        let mut simd_time = 0.0;
        let mut serde_time = 0.0;

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let mut bytes = json_str.as_bytes().to_vec();
            let _ = simd_json::serde::from_slice::<serde_json::Value>(&mut bytes);
        }
        simd_time = start.elapsed().as_micros() as f64;

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = serde_json::from_str::<serde_json::Value>(json_str);
        }
        serde_time = start.elapsed().as_micros() as f64;

        JsonBenchmark {
            iterations,
            simd_time_us: simd_time,
            serde_time_us: serde_time,
            json_size: json_str.len(),
        }
    }
}

impl Default for SimdJsonService {
    fn default() -> Self {
        Self::new()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct JsonBenchmark {
    #[pyo3(get)]
    pub iterations: u32,
    #[pyo3(get)]
    pub simd_time_us: f64,
    #[pyo3(get)]
    pub serde_time_us: f64,
    #[pyo3(get)]
    pub json_size: usize,
}

#[pymethods]
impl JsonBenchmark {
    pub fn speedup(&self) -> f64 {
        if self.simd_time_us == 0.0 {
            return 0.0;
        }
        self.serde_time_us / self.simd_time_us
    }

    pub fn simd_ops_per_sec(&self) -> f64 {
        if self.simd_time_us == 0.0 {
            return 0.0;
        }
        (self.iterations as f64 / self.simd_time_us) * 1_000_000.0
    }

    pub fn serde_ops_per_sec(&self) -> f64 {
        if self.serde_time_us == 0.0 {
            return 0.0;
        }
        (self.iterations as f64 / self.serde_time_us) * 1_000_000.0
    }

    pub fn simd_throughput_mb_s(&self) -> f64 {
        if self.simd_time_us == 0.0 {
            return 0.0;
        }
        let total_bytes = self.json_size * self.iterations as usize;
        (total_bytes as f64 / 1_000_000.0) / (self.simd_time_us / 1_000_000.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "JsonBenchmark(simd={:.0}μs, serde={:.0}μs, speedup={:.1}x, simd_ops={:.0}/s)",
            self.simd_time_us,
            self.serde_time_us,
            self.speedup(),
            self.simd_ops_per_sec()
        )
    }
}

fn is_simd_available() -> bool {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

fn value_to_pyobject(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_py(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_py(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_py(py))
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_py(py)),
        serde_json::Value::Array(arr) => {
            let list: Vec<PyObject> = arr
                .iter()
                .filter_map(|v| value_to_pyobject(py, v).ok())
                .collect();
            Ok(list.into_py(py))
        }
        serde_json::Value::Object(obj) => {
            let dict = pyo3::types::PyDict::new(py);
            for (k, v) in obj {
                if let Ok(py_v) = value_to_pyobject(py, v) {
                    let _ = dict.set_item(k, py_v);
                }
            }
            Ok(dict.into())
        }
    }
}

fn pyobject_to_value(obj: &PyAny) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(serde_json::Value::Bool(b));
    }
    
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(serde_json::Value::Number(i.into()));
    }
    
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(serde_json::json!(f));
    }
    
    if let Ok(s) = obj.extract::<String>() {
        return Ok(serde_json::Value::String(s));
    }
    
    if let Ok(list) = obj.extract::<Vec<&PyAny>>() {
        let arr: Vec<serde_json::Value> = list
            .into_iter()
            .filter_map(|item| pyobject_to_value(item).ok())
            .collect();
        return Ok(serde_json::Value::Array(arr));
    }
    
    if let Ok(dict) = obj.downcast::<pyo3::types::PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            if let (Ok(key), Ok(value)) = (k.extract::<String>(), pyobject_to_value(v)) {
                map.insert(key, value);
            }
        }
        return Ok(serde_json::Value::Object(map));
    }
    
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Unsupported type for JSON serialization",
    ))
}

#[pyfunction]
pub fn parse_json_fast(json_str: &str) -> PyResult<PyObject> {
    let service = SimdJsonService::new();
    service.parse(json_str)
}

#[pyfunction]
pub fn is_json_valid(json_str: &str) -> bool {
    let service = SimdJsonService::new();
    service.is_valid(json_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let service = SimdJsonService::new();
        let result = service.parse_to_string(r#"{"name": "test", "value": 123}"#);
        assert!(result.is_ok());
    }

    #[test]
    fn test_is_valid() {
        let service = SimdJsonService::new();
        assert!(service.is_valid(r#"{"valid": true}"#));
        assert!(!service.is_valid("not json"));
    }

    #[test]
    fn test_benchmark() {
        let service = SimdJsonService::new();
        let json = r#"{"name": "test", "values": [1, 2, 3, 4, 5]}"#;
        let benchmark = service.benchmark_parse(json, 1000);
        assert!(benchmark.speedup() > 0.0);
    }
}












