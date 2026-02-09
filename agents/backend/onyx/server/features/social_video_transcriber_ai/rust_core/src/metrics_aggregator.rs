//! Metrics Aggregator
//!
//! Provides metrics aggregation and reporting.

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Metric type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Timer,
}

/// Aggregated metric
#[derive(Debug, Clone)]
struct AggregatedMetric {
    name: String,
    metric_type: MetricType,
    count: usize,
    sum: f64,
    min: f64,
    max: f64,
    values: Vec<f64>,
    last_updated: Instant,
}

/// Metrics aggregator
#[pyclass]
pub struct MetricsAggregator {
    metrics: Arc<Mutex<HashMap<String, AggregatedMetric>>>,
    aggregation_window_ms: u64,
}

#[pymethods]
impl MetricsAggregator {
    #[new]
    pub fn new(aggregation_window_ms: u64) -> Self {
        Self {
            metrics: Arc::new(Mutex::new(HashMap::new())),
            aggregation_window_ms,
        }
    }

    pub fn record(&self, name: String, value: f64, metric_type: Option<String>) -> PyResult<()> {
        let mut metrics = self.metrics.lock().unwrap();
        let mtype = if let Some(t) = metric_type {
            match t.to_lowercase().as_str() {
                "counter" => MetricType::Counter,
                "gauge" => MetricType::Gauge,
                "histogram" => MetricType::Histogram,
                "timer" => MetricType::Timer,
                _ => MetricType::Gauge,
            }
        } else {
            MetricType::Gauge
        };
        
        let metric = metrics.entry(name.clone()).or_insert_with(|| {
            AggregatedMetric {
                name: name.clone(),
                metric_type: mtype,
                count: 0,
                sum: 0.0,
                min: f64::INFINITY,
                max: f64::NEG_INFINITY,
                values: Vec::new(),
                last_updated: Instant::now(),
            }
        });
        
        metric.count += 1;
        metric.sum += value;
        metric.min = metric.min.min(value);
        metric.max = metric.max.max(value);
        metric.last_updated = Instant::now();
        
        // Keep only recent values for histograms
        if metric.metric_type == MetricType::Histogram {
            metric.values.push(value);
            if metric.values.len() > 1000 {
                metric.values.remove(0);
            }
        }
        
        Ok(())
    }

    pub fn increment(&self, name: String, value: Option<f64>) -> PyResult<()> {
        self.record(name, value.unwrap_or(1.0), Some("counter".to_string()))
    }

    pub fn set_gauge(&self, name: String, value: f64) -> PyResult<()> {
        self.record(name, value, Some("gauge".to_string()))
    }

    pub fn record_timer(&self, name: String, duration_ms: f64) -> PyResult<()> {
        self.record(name, duration_ms, Some("timer".to_string()))
    }

    pub fn get_metric(&self, name: String) -> PyResult<Option<PyObject>> {
        Python::with_gil(|py| {
            let metrics = self.metrics.lock().unwrap();
            if let Some(metric) = metrics.get(&name) {
                let dict = PyDict::new(py);
                dict.set_item("name", &metric.name)?;
                dict.set_item("type", match metric.metric_type {
                    MetricType::Counter => "counter",
                    MetricType::Gauge => "gauge",
                    MetricType::Histogram => "histogram",
                    MetricType::Timer => "timer",
                })?;
                dict.set_item("count", metric.count)?;
                dict.set_item("sum", metric.sum)?;
                dict.set_item("min", if metric.min == f64::INFINITY { 0.0 } else { metric.min })?;
                dict.set_item("max", if metric.max == f64::NEG_INFINITY { 0.0 } else { metric.max })?;
                dict.set_item("avg", if metric.count > 0 { metric.sum / metric.count as f64 } else { 0.0 })?;
                Ok(Some(dict.into()))
            } else {
                Ok(None)
            }
        })
    }

    pub fn get_all_metrics(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let metrics = self.metrics.lock().unwrap();
            let dict = PyDict::new(py);
            
            for (name, metric) in metrics.iter() {
                let metric_dict = PyDict::new(py);
                metric_dict.set_item("type", match metric.metric_type {
                    MetricType::Counter => "counter",
                    MetricType::Gauge => "gauge",
                    MetricType::Histogram => "histogram",
                    MetricType::Timer => "timer",
                })?;
                metric_dict.set_item("count", metric.count)?;
                metric_dict.set_item("sum", metric.sum)?;
                metric_dict.set_item("min", if metric.min == f64::INFINITY { 0.0 } else { metric.min })?;
                metric_dict.set_item("max", if metric.max == f64::NEG_INFINITY { 0.0 } else { metric.max })?;
                metric_dict.set_item("avg", if metric.count > 0 { metric.sum / metric.count as f64 } else { 0.0 })?;
                dict.set_item(name, metric_dict)?;
            }
            
            Ok(dict.into())
        })
    }

    pub fn cleanup_old_metrics(&self) -> usize {
        let mut metrics = self.metrics.lock().unwrap();
        let window = Duration::from_millis(self.aggregation_window_ms);
        let now = Instant::now();
        
        let before = metrics.len();
        metrics.retain(|_, metric| {
            now.duration_since(metric.last_updated) < window
        });
        before - metrics.len()
    }

    pub fn reset(&self) -> PyResult<()> {
        self.metrics.lock().unwrap().clear();
        Ok(())
    }
}

#[pyfunction]
pub fn create_metrics_aggregator(aggregation_window_ms: Option<u64>) -> MetricsAggregator {
    MetricsAggregator::new(aggregation_window_ms.unwrap_or(60000))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_aggregator() {
        let aggregator = MetricsAggregator::new(60000);
        aggregator.record("test".to_string(), 10.0, None).unwrap();
        assert!(aggregator.get_metric("test".to_string()).unwrap().is_some());
    }
}












