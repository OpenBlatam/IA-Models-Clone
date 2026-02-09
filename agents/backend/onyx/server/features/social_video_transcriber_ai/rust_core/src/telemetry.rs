//! Telemetry and Observability
//!
//! Provides telemetry collection and observability features.

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Telemetry event
#[derive(Debug, Clone)]
pub struct TelemetryEvent {
    pub name: String,
    pub timestamp: u64,
    pub duration_ms: Option<f64>,
    pub metadata: HashMap<String, String>,
    pub tags: HashMap<String, String>,
}

/// Telemetry collector
#[pyclass]
pub struct TelemetryCollector {
    events: Arc<Mutex<Vec<TelemetryEvent>>>,
    traces: Arc<Mutex<Vec<TraceSpan>>>,
    metrics: Arc<Mutex<HashMap<String, MetricValue>>>,
}

#[derive(Debug, Clone)]
struct TraceSpan {
    name: String,
    start: Instant,
    end: Option<Instant>,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram(Vec<f64>),
}

#[pymethods]
impl TelemetryCollector {
    #[new]
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            traces: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn record_event(&self, name: String, duration_ms: Option<f64>, metadata: Option<PyObject>) -> PyResult<()> {
        Python::with_gil(|py| {
            let mut meta = HashMap::new();
            if let Some(meta_obj) = metadata {
                if let Ok(dict) = meta_obj.downcast::<PyDict>(py) {
                    for (key, value) in dict.iter() {
                        if let (Ok(k), Ok(v)) = (key.extract::<String>(), value.extract::<String>()) {
                            meta.insert(k, v);
                        }
                    }
                }
            }
            
            let event = TelemetryEvent {
                name,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                duration_ms,
                metadata: meta,
                tags: HashMap::new(),
            };
            
            self.events.lock().unwrap().push(event);
            Ok(())
        })
    }

    pub fn start_trace(&self, name: String) -> PyResult<String> {
        let span = TraceSpan {
            name: name.clone(),
            start: Instant::now(),
            end: None,
            metadata: HashMap::new(),
        };
        
        let trace_id = format!("trace_{}", self.traces.lock().unwrap().len());
        self.traces.lock().unwrap().push(span);
        Ok(trace_id)
    }

    pub fn end_trace(&self, trace_id: String) -> PyResult<f64> {
        let mut traces = self.traces.lock().unwrap();
        if let Some(span) = traces.iter_mut().find(|s| s.name == trace_id) {
            span.end = Some(Instant::now());
            Ok(span.start.elapsed().as_secs_f64() * 1000.0)
        } else {
            Err(PyValueError::new_err(format!("Trace {} not found", trace_id)))
        }
    }

    pub fn increment_counter(&self, name: String, value: Option<u64>) -> PyResult<()> {
        let mut metrics = self.metrics.lock().unwrap();
        let entry = metrics.entry(name).or_insert_with(|| MetricValue::Counter(0));
        if let MetricValue::Counter(ref mut count) = entry {
            *count += value.unwrap_or(1);
        }
        Ok(())
    }

    pub fn set_gauge(&self, name: String, value: f64) -> PyResult<()> {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.insert(name, MetricValue::Gauge(value));
        Ok(())
    }

    pub fn record_histogram(&self, name: String, value: f64) -> PyResult<()> {
        let mut metrics = self.metrics.lock().unwrap();
        let entry = metrics.entry(name).or_insert_with(|| MetricValue::Histogram(Vec::new()));
        if let MetricValue::Histogram(ref mut values) = entry {
            values.push(value);
            // Keep only last 1000 values
            if values.len() > 1000 {
                values.remove(0);
            }
        }
        Ok(())
    }

    pub fn get_metrics(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let metrics = self.metrics.lock().unwrap();
            let dict = PyDict::new(py);
            
            for (name, value) in metrics.iter() {
                match value {
                    MetricValue::Counter(count) => {
                        dict.set_item(format!("{}_count", name), *count)?;
                    }
                    MetricValue::Gauge(gauge) => {
                        dict.set_item(format!("{}_gauge", name), *gauge)?;
                    }
                    MetricValue::Histogram(values) => {
                        if !values.is_empty() {
                            let sum: f64 = values.iter().sum();
                            let avg = sum / values.len() as f64;
                            let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                            let max = values.iter().fold(0.0, |a, &b| a.max(b));
                            dict.set_item(format!("{}_avg", name), avg)?;
                            dict.set_item(format!("{}_min", name), min)?;
                            dict.set_item(format!("{}_max", name), max)?;
                            dict.set_item(format!("{}_count", name), values.len())?;
                        }
                    }
                }
            }
            
            Ok(dict.into())
        })
    }

    pub fn export_events(&self, limit: Option<usize>) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            let events = self.events.lock().unwrap();
            let limit = limit.unwrap_or(events.len());
            let result: Vec<PyObject> = events.iter()
                .take(limit)
                .map(|event| {
                    let dict = PyDict::new(py);
                    dict.set_item("name", &event.name).unwrap();
                    dict.set_item("timestamp", event.timestamp).unwrap();
                    if let Some(duration) = event.duration_ms {
                        dict.set_item("duration_ms", duration).unwrap();
                    }
                    dict.into()
                })
                .collect();
            Ok(result)
        })
    }

    pub fn clear(&self) -> PyResult<()> {
        self.events.lock().unwrap().clear();
        self.traces.lock().unwrap().clear();
        self.metrics.lock().unwrap().clear();
        Ok(())
    }
}

#[pyfunction]
pub fn create_telemetry_collector() -> TelemetryCollector {
    TelemetryCollector::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_collector() {
        let collector = TelemetryCollector::new();
        collector.record_event("test".to_string(), Some(10.0), None).unwrap();
        assert!(collector.export_events(None).unwrap().len() > 0);
    }
}












