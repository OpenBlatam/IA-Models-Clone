//! Profiling and Performance Monitoring
//!
//! Provides tools for profiling Rust code and monitoring performance metrics.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};

#[pyclass]
#[derive(Clone)]
pub struct Profiler {
    timings: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
    counters: Arc<Mutex<HashMap<String, u64>>>,
}

#[pymethods]
impl Profiler {
    #[new]
    pub fn new() -> Self {
        Self {
            timings: Arc::new(Mutex::new(HashMap::new())),
            counters: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn start_timer(&self, name: String) -> PyResult<u64> {
        let start = Instant::now();
        let id = start.elapsed().as_nanos() as u64;
        Ok(id)
    }

    pub fn record_time(&self, name: String, duration_ms: f64) -> PyResult<()> {
        let duration = Duration::from_secs_f64(duration_ms / 1000.0);
        let mut timings = self.timings.lock().unwrap();
        timings.entry(name).or_insert_with(Vec::new).push(duration);
        Ok(())
    }

    pub fn increment(&self, name: String) -> PyResult<()> {
        let mut counters = self.counters.lock().unwrap();
        *counters.entry(name).or_insert(0) += 1;
        Ok(())
    }

    pub fn increment_by(&self, name: String, value: u64) -> PyResult<()> {
        let mut counters = self.counters.lock().unwrap();
        *counters.entry(name).or_insert(0) += value;
        Ok(())
    }

    pub fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            // Timing stats
            let timings = self.timings.lock().unwrap();
            let timing_dict = PyDict::new(py);
            for (name, durations) in timings.iter() {
                if !durations.is_empty() {
                    let total: f64 = durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum();
                    let avg = total / durations.len() as f64;
                    let min = durations.iter().map(|d| d.as_secs_f64() * 1000.0).fold(f64::INFINITY, f64::min);
                    let max = durations.iter().map(|d| d.as_secs_f64() * 1000.0).fold(0.0, f64::max);
                    
                    let stats = PyDict::new(py);
                    stats.set_item("count", durations.len())?;
                    stats.set_item("total_ms", total)?;
                    stats.set_item("avg_ms", avg)?;
                    stats.set_item("min_ms", min)?;
                    stats.set_item("max_ms", max)?;
                    timing_dict.set_item(name, stats)?;
                }
            }
            dict.set_item("timings", timing_dict)?;
            
            // Counter stats
            let counters = self.counters.lock().unwrap();
            let counter_dict = PyDict::new(py);
            for (name, value) in counters.iter() {
                counter_dict.set_item(name, *value)?;
            }
            dict.set_item("counters", counter_dict)?;
            
            Ok(dict.into())
        })
    }

    pub fn reset(&self) -> PyResult<()> {
        let mut timings = self.timings.lock().unwrap();
        timings.clear();
        let mut counters = self.counters.lock().unwrap();
        counters.clear();
        Ok(())
    }

    pub fn export_report(&self) -> PyResult<String> {
        let stats = self.get_stats()?;
        Python::with_gil(|py| {
            let json = py.import("json")?;
            let report = json.call_method1("dumps", (stats,))?;
            report.extract()
        })
    }
}

#[pyclass]
pub struct Timer {
    start: Instant,
    name: String,
}

#[pymethods]
impl Timer {
    #[new]
    pub fn new(name: String) -> Self {
        Self {
            start: Instant::now(),
            name,
        }
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    pub fn elapsed_us(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1_000_000.0
    }

    pub fn elapsed_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    pub fn reset(&mut self) {
        self.start = Instant::now();
    }

    fn __repr__(&self) -> String {
        format!("Timer(name={}, elapsed={:.2}ms)", self.name, self.elapsed_ms())
    }
}

#[pyfunction]
pub fn create_profiler() -> Profiler {
    Profiler::new()
}

#[pyfunction]
pub fn create_timer(name: String) -> Timer {
    Timer::new(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler() {
        let profiler = Profiler::new();
        profiler.record_time("test".to_string(), 10.0).unwrap();
        profiler.increment("counter".to_string()).unwrap();
        
        let stats = profiler.get_stats().unwrap();
        assert!(stats.to_string().contains("test"));
    }

    #[test]
    fn test_timer() {
        let mut timer = Timer::new("test".to_string());
        std::thread::sleep(Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10.0);
    }
}












