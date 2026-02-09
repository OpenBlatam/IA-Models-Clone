//! Performance Metrics Collection
//!
//! Provides comprehensive metrics collection:
//! - Operation timing
//! - Throughput measurement
//! - Resource usage tracking
//! - Statistical aggregation

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use parking_lot::RwLock;

#[pyclass]
pub struct MetricsCollector {
    counters: RwLock<HashMap<String, AtomicU64>>,
    timers: RwLock<HashMap<String, Vec<f64>>>,
    gauges: RwLock<HashMap<String, f64>>,
    histograms: RwLock<HashMap<String, Histogram>>,
    start_time: Instant,
}

#[pymethods]
impl MetricsCollector {
    #[new]
    pub fn new() -> Self {
        Self {
            counters: RwLock::new(HashMap::new()),
            timers: RwLock::new(HashMap::new()),
            gauges: RwLock::new(HashMap::new()),
            histograms: RwLock::new(HashMap::new()),
            start_time: Instant::now(),
        }
    }

    pub fn increment(&self, name: &str) {
        self.increment_by(name, 1);
    }

    pub fn increment_by(&self, name: &str, value: u64) {
        let counters = self.counters.read();
        if let Some(counter) = counters.get(name) {
            counter.fetch_add(value, Ordering::Relaxed);
        } else {
            drop(counters);
            let mut counters = self.counters.write();
            counters
                .entry(name.to_string())
                .or_insert_with(|| AtomicU64::new(0))
                .fetch_add(value, Ordering::Relaxed);
        }
    }

    pub fn get_counter(&self, name: &str) -> u64 {
        self.counters
            .read()
            .get(name)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    pub fn record_time(&self, name: &str, duration_ms: f64) {
        let mut timers = self.timers.write();
        timers
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(duration_ms);
    }

    pub fn time_operation(&self, name: &str) -> OperationTimer {
        OperationTimer {
            name: name.to_string(),
            start: Instant::now(),
            collector: self as *const MetricsCollector,
        }
    }

    pub fn set_gauge(&self, name: &str, value: f64) {
        self.gauges.write().insert(name.to_string(), value);
    }

    pub fn get_gauge(&self, name: &str) -> Option<f64> {
        self.gauges.read().get(name).copied()
    }

    pub fn record_histogram(&self, name: &str, value: f64) {
        let mut histograms = self.histograms.write();
        histograms
            .entry(name.to_string())
            .or_insert_with(Histogram::new)
            .record(value);
    }

    pub fn get_timer_stats(&self, name: &str) -> Option<TimerStats> {
        let timers = self.timers.read();
        timers.get(name).map(|values| calculate_stats(values))
    }

    pub fn get_histogram_stats(&self, name: &str) -> Option<HistogramStats> {
        let histograms = self.histograms.read();
        histograms.get(name).map(|h| h.get_stats())
    }

    pub fn get_all_counters(&self) -> HashMap<String, u64> {
        self.counters
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed)))
            .collect()
    }

    pub fn get_all_gauges(&self) -> HashMap<String, f64> {
        self.gauges.read().clone()
    }

    pub fn uptime_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    pub fn reset(&self) {
        self.counters.write().clear();
        self.timers.write().clear();
        self.gauges.write().clear();
        self.histograms.write().clear();
    }

    pub fn get_summary(&self) -> MetricsSummary {
        MetricsSummary {
            counters: self.get_all_counters(),
            gauges: self.get_all_gauges(),
            timer_names: self.timers.read().keys().cloned().collect(),
            histogram_names: self.histograms.read().keys().cloned().collect(),
            uptime_seconds: self.uptime_seconds(),
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

pub struct OperationTimer {
    name: String,
    start: Instant,
    collector: *const MetricsCollector,
}

impl Drop for OperationTimer {
    fn drop(&mut self) {
        let duration = self.start.elapsed().as_secs_f64() * 1000.0;
        unsafe {
            if let Some(collector) = self.collector.as_ref() {
                collector.record_time(&self.name, duration);
            }
        }
    }
}

unsafe impl Send for OperationTimer {}
unsafe impl Sync for OperationTimer {}

struct Histogram {
    values: Vec<f64>,
    min: f64,
    max: f64,
    sum: f64,
    count: usize,
}

impl Histogram {
    fn new() -> Self {
        Self {
            values: Vec::new(),
            min: f64::MAX,
            max: f64::MIN,
            sum: 0.0,
            count: 0,
        }
    }

    fn record(&mut self, value: f64) {
        self.values.push(value);
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.sum += value;
        self.count += 1;
    }

    fn get_stats(&self) -> HistogramStats {
        if self.count == 0 {
            return HistogramStats::default();
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = percentile(&sorted, 50.0);
        let p90 = percentile(&sorted, 90.0);
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);

        HistogramStats {
            count: self.count,
            min: self.min,
            max: self.max,
            mean: self.sum / self.count as f64,
            p50,
            p90,
            p95,
            p99,
        }
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn calculate_stats(values: &[f64]) -> TimerStats {
    if values.is_empty() {
        return TimerStats::default();
    }

    let sum: f64 = values.iter().sum();
    let mean = sum / values.len() as f64;
    let min = values.iter().cloned().fold(f64::MAX, f64::min);
    let max = values.iter().cloned().fold(f64::MIN, f64::max);

    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    TimerStats {
        count: values.len(),
        min,
        max,
        mean,
        std_dev,
        p50: percentile(&sorted, 50.0),
        p95: percentile(&sorted, 95.0),
        p99: percentile(&sorted, 99.0),
    }
}

#[pyclass]
#[derive(Clone, Default)]
pub struct TimerStats {
    #[pyo3(get)]
    pub count: usize,
    #[pyo3(get)]
    pub min: f64,
    #[pyo3(get)]
    pub max: f64,
    #[pyo3(get)]
    pub mean: f64,
    #[pyo3(get)]
    pub std_dev: f64,
    #[pyo3(get)]
    pub p50: f64,
    #[pyo3(get)]
    pub p95: f64,
    #[pyo3(get)]
    pub p99: f64,
}

#[pymethods]
impl TimerStats {
    fn __repr__(&self) -> String {
        format!(
            "TimerStats(count={}, mean={:.2}ms, p50={:.2}ms, p99={:.2}ms)",
            self.count, self.mean, self.p50, self.p99
        )
    }
}

#[pyclass]
#[derive(Clone, Default)]
pub struct HistogramStats {
    #[pyo3(get)]
    pub count: usize,
    #[pyo3(get)]
    pub min: f64,
    #[pyo3(get)]
    pub max: f64,
    #[pyo3(get)]
    pub mean: f64,
    #[pyo3(get)]
    pub p50: f64,
    #[pyo3(get)]
    pub p90: f64,
    #[pyo3(get)]
    pub p95: f64,
    #[pyo3(get)]
    pub p99: f64,
}

#[pymethods]
impl HistogramStats {
    fn __repr__(&self) -> String {
        format!(
            "HistogramStats(count={}, mean={:.2}, p50={:.2}, p99={:.2})",
            self.count, self.mean, self.p50, self.p99
        )
    }
}

#[pyclass]
#[derive(Clone)]
pub struct MetricsSummary {
    #[pyo3(get)]
    pub counters: HashMap<String, u64>,
    #[pyo3(get)]
    pub gauges: HashMap<String, f64>,
    #[pyo3(get)]
    pub timer_names: Vec<String>,
    #[pyo3(get)]
    pub histogram_names: Vec<String>,
    #[pyo3(get)]
    pub uptime_seconds: f64,
}

#[pymethods]
impl MetricsSummary {
    fn __repr__(&self) -> String {
        format!(
            "MetricsSummary(counters={}, gauges={}, timers={}, histograms={}, uptime={:.1}s)",
            self.counters.len(),
            self.gauges.len(),
            self.timer_names.len(),
            self.histogram_names.len(),
            self.uptime_seconds
        )
    }
}

#[pyclass]
pub struct Stopwatch {
    start: Option<Instant>,
    elapsed: Duration,
    running: bool,
}

#[pymethods]
impl Stopwatch {
    #[new]
    pub fn new() -> Self {
        Self {
            start: None,
            elapsed: Duration::ZERO,
            running: false,
        }
    }

    pub fn start(&mut self) {
        if !self.running {
            self.start = Some(Instant::now());
            self.running = true;
        }
    }

    pub fn stop(&mut self) {
        if self.running {
            if let Some(start) = self.start {
                self.elapsed += start.elapsed();
            }
            self.running = false;
        }
    }

    pub fn reset(&mut self) {
        self.start = None;
        self.elapsed = Duration::ZERO;
        self.running = false;
    }

    pub fn elapsed_ms(&self) -> f64 {
        let mut total = self.elapsed;
        if self.running {
            if let Some(start) = self.start {
                total += start.elapsed();
            }
        }
        total.as_secs_f64() * 1000.0
    }

    pub fn elapsed_us(&self) -> f64 {
        let mut total = self.elapsed;
        if self.running {
            if let Some(start) = self.start {
                total += start.elapsed();
            }
        }
        total.as_secs_f64() * 1_000_000.0
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn lap(&mut self) -> f64 {
        let elapsed = self.elapsed_ms();
        self.reset();
        self.start();
        elapsed
    }
}

impl Default for Stopwatch {
    fn default() -> Self {
        Self::new()
    }
}

#[pyfunction]
pub fn create_metrics_collector() -> MetricsCollector {
    MetricsCollector::new()
}

#[pyfunction]
pub fn create_stopwatch() -> Stopwatch {
    Stopwatch::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let collector = MetricsCollector::new();
        collector.increment("test");
        collector.increment("test");
        assert_eq!(collector.get_counter("test"), 2);
    }

    #[test]
    fn test_timer() {
        let collector = MetricsCollector::new();
        collector.record_time("op", 10.0);
        collector.record_time("op", 20.0);
        collector.record_time("op", 30.0);

        let stats = collector.get_timer_stats("op").unwrap();
        assert_eq!(stats.count, 3);
        assert!((stats.mean - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_gauge() {
        let collector = MetricsCollector::new();
        collector.set_gauge("memory", 1024.0);
        assert_eq!(collector.get_gauge("memory"), Some(1024.0));
    }

    #[test]
    fn test_stopwatch() {
        let mut sw = Stopwatch::new();
        sw.start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        sw.stop();
        assert!(sw.elapsed_ms() >= 10.0);
    }
}












