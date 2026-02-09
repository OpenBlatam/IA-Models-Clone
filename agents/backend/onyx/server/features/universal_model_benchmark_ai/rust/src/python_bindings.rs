//! Python Bindings
//!
//! PyO3 bindings for Python integration.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::inference::{InferenceEngine, InferenceConfig, InferenceStats, MetricsCollector};
use crate::data::DataProcessor;
use crate::utils::{format_duration, format_bytes, percentile};

/// Python module for benchmark core.
#[pymodule]
fn benchmark_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyInferenceEngine>()?;
    m.add_class::<PyDataProcessor>()?;
    m.add_class::<PyMetricsCollector>()?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(get_system_info, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_metrics_py, m)?)?;
    Ok(())
}

/// Python wrapper for InferenceEngine.
#[pyclass]
pub struct PyInferenceEngine {
    engine: InferenceEngine,
}

#[pymethods]
impl PyInferenceEngine {
    #[new]
    fn new(
        model_path: String,
        device_str: Option<String>,
        config: Option<PyObject>,
    ) -> PyResult<Self> {
        use candle_core::Device;
        
        let device = match device_str.as_deref() {
            Some("cuda") | Some("gpu") => Device::Cpu, // TODO: Support CUDA
            _ => Device::Cpu,
        };
        
        let rust_config = if let Some(cfg) = config {
            // Parse Python dict to InferenceConfig
            // For now, use default
            None
        } else {
            None
        };
        
        let engine = InferenceEngine::new(model_path, device, rust_config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to create engine: {}", e)
            ))?;
        
        Ok(Self { engine })
    }
    
    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        self.engine.encode(text)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Encoding error: {}", e)
            ))
    }
    
    fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        self.engine.decode(&tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Decoding error: {}", e)
            ))
    }
    
    fn infer(&self, prompt: &str) -> PyResult<(Vec<u32>, PyDict)> {
        let (tokens, stats) = self.engine.infer(prompt, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Inference error: {}", e)
            ))?;
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("latency_ms", stats.latency_ms)?;
            dict.set_item("tokens_per_second", stats.tokens_per_second)?;
            dict.set_item("num_tokens", stats.num_tokens)?;
            Ok((tokens, dict))
        })
    }
}

/// Python wrapper for DataProcessor.
#[pyclass]
pub struct PyDataProcessor {
    processor: DataProcessor,
}

#[pymethods]
impl PyDataProcessor {
    #[new]
    fn new(config: Option<PyObject>) -> PyResult<Self> {
        // For now, use default config
        let processor = DataProcessor::new(None);
        Ok(Self { processor })
    }
    
    fn process_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        self.processor.process_batch(&texts)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Processing error: {}", e)
            ))
    }
    
    fn format_prompt(
        &self,
        template: &str,
        variables: &PyDict,
    ) -> PyResult<String> {
        use std::collections::HashMap;
        
        let mut vars = HashMap::new();
        for (key, value) in variables.iter() {
            let key_str = key.extract::<String>()?;
            let value_str = value.extract::<String>()?;
            vars.insert(key_str, value_str);
        }
        
        self.processor.format_prompt(template, &vars)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Format error: {}", e)
            ))
    }
}

/// Python wrapper for MetricsCollector.
#[pyclass]
pub struct PyMetricsCollector {
    collector: MetricsCollector,
}

#[pymethods]
impl PyMetricsCollector {
    #[new]
    fn new(max_samples: usize) -> Self {
        Self {
            collector: MetricsCollector::new(max_samples),
        }
    }
    
    fn record(&self, latency_ms: f64, tokens: usize) {
        self.collector.record(latency_ms, tokens);
    }
    
    fn get_metrics(&self) -> PyResult<PyDict> {
        let metrics = self.collector.get_metrics();
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("total_requests", metrics.total_requests)?;
            dict.set_item("total_tokens", metrics.total_tokens)?;
            dict.set_item("avg_latency_ms", metrics.avg_latency_ms)?;
            dict.set_item("p50_latency_ms", metrics.p50_latency_ms)?;
            dict.set_item("p95_latency_ms", metrics.p95_latency_ms)?;
            dict.set_item("p99_latency_ms", metrics.p99_latency_ms)?;
            dict.set_item("tokens_per_second", metrics.tokens_per_second)?;
            Ok(dict)
        })
    }
    
    fn reset(&self) {
        self.collector.reset();
    }
}

/// Get library version.
#[pyfunction]
fn get_version() -> &'static str {
    crate::get_version()
}

/// Get system information.
#[pyfunction]
fn get_system_info() -> PyResult<PyDict> {
    let info = crate::get_system_info();
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        for (key, value) in info {
            dict.set_item(key, value)?;
        }
        Ok(dict)
    })
}

/// Calculate metrics from Python.
#[pyfunction]
fn calculate_metrics_py(
    latencies: Vec<f64>,
    accuracies: Vec<bool>,
    total_tokens: usize,
    total_time: f64,
) -> PyResult<PyDict> {
    use crate::metrics::calculate_metrics;
    use crate::Metrics;
    
    let metrics = calculate_metrics(&latencies, &accuracies, total_tokens, total_time);
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("accuracy", metrics.accuracy)?;
        dict.set_item("latency_p50", metrics.latency_p50)?;
        dict.set_item("latency_p95", metrics.latency_p95)?;
        dict.set_item("latency_p99", metrics.latency_p99)?;
        dict.set_item("throughput", metrics.throughput)?;
        dict.set_item("memory_peak_mb", metrics.memory_peak_mb)?;
        Ok(dict)
    })
}

