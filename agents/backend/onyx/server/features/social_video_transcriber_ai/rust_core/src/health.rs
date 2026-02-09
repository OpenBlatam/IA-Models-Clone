//! Health Check and System Monitoring
//!
//! Provides health check endpoints and system monitoring capabilities.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[pyclass]
pub struct HealthChecker {
    start_time: SystemTime,
    request_count: AtomicU64,
    error_count: AtomicU64,
}

#[pymethods]
impl HealthChecker {
    #[new]
    pub fn new() -> Self {
        Self {
            start_time: SystemTime::now(),
            request_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
        }
    }

    pub fn record_request(&self) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn get_health(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            let uptime = self.start_time.elapsed().unwrap_or(Duration::ZERO);
            let requests = self.request_count.load(Ordering::Relaxed);
            let errors = self.error_count.load(Ordering::Relaxed);
            let success_rate = if requests > 0 {
                1.0 - (errors as f64 / requests as f64)
            } else {
                1.0
            };
            
            dict.set_item("status", "healthy")?;
            dict.set_item("uptime_seconds", uptime.as_secs())?;
            dict.set_item("total_requests", requests)?;
            dict.set_item("total_errors", errors)?;
            dict.set_item("success_rate", success_rate)?;
            dict.set_item("timestamp", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs())?;
            
            Ok(dict.into())
        })
    }

    pub fn get_metrics(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            let uptime = self.start_time.elapsed().unwrap_or(Duration::ZERO);
            let requests = self.request_count.load(Ordering::Relaxed);
            let errors = self.error_count.load(Ordering::Relaxed);
            
            let requests_per_sec = if uptime.as_secs() > 0 {
                requests as f64 / uptime.as_secs() as f64
            } else {
                0.0
            };
            
            dict.set_item("uptime_seconds", uptime.as_secs())?;
            dict.set_item("total_requests", requests)?;
            dict.set_item("total_errors", errors)?;
            dict.set_item("requests_per_second", requests_per_sec)?;
            dict.set_item("error_rate", if requests > 0 { errors as f64 / requests as f64 } else { 0.0 })?;
            
            Ok(dict.into())
        })
    }

    pub fn reset(&self) {
        self.request_count.store(0, Ordering::Relaxed);
        self.error_count.store(0, Ordering::Relaxed);
    }
}

#[pyclass]
pub struct SystemMonitor {
    cpu_usage: f64,
    memory_usage: f64,
    timestamp: u64,
}

#[pymethods]
impl SystemMonitor {
    #[new]
    pub fn new() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        }
    }

    pub fn update(&mut self, cpu: f64, memory: f64) {
        self.cpu_usage = cpu;
        self.memory_usage = memory;
        self.timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    }

    pub fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("cpu_usage_percent", self.cpu_usage)?;
            dict.set_item("memory_usage_percent", self.memory_usage)?;
            dict.set_item("timestamp", self.timestamp)?;
            Ok(dict.into())
        })
    }
}

#[pyfunction]
pub fn create_health_checker() -> HealthChecker {
    HealthChecker::new()
}

#[pyfunction]
pub fn create_system_monitor() -> SystemMonitor {
    SystemMonitor::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_checker() {
        let checker = HealthChecker::new();
        checker.record_request();
        checker.record_request();
        checker.record_error();
        
        let health = checker.get_health().unwrap();
        assert!(health.to_string().contains("healthy"));
    }

    #[test]
    fn test_system_monitor() {
        let mut monitor = SystemMonitor::new();
        monitor.update(50.0, 75.0);
        
        let stats = monitor.get_stats().unwrap();
        assert!(stats.to_string().contains("50"));
    }
}












