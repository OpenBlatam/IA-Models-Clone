//! Utilities Module - Common Helper Functions
//!
//! Provides various utility functions for the Cursor Agent Core.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

/// Timer utility for performance measurement
#[pyclass]
pub struct Timer {
    start: Instant,
    checkpoints: Vec<(String, f64)>,
}

#[pymethods]
impl Timer {
    #[new]
    fn new() -> Self {
        Self {
            start: Instant::now(),
            checkpoints: Vec::new(),
        }
    }

    /// Get elapsed time in milliseconds
    fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    /// Get elapsed time in seconds
    fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Get elapsed time in microseconds
    fn elapsed_us(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1_000_000.0
    }

    /// Get elapsed time in nanoseconds
    fn elapsed_ns(&self) -> u128 {
        self.start.elapsed().as_nanos()
    }

    /// Reset the timer
    fn reset(&mut self) {
        self.start = Instant::now();
        self.checkpoints.clear();
    }

    /// Add a checkpoint
    fn checkpoint(&mut self, name: &str) {
        self.checkpoints
            .push((name.to_string(), self.elapsed_ms()));
    }

    /// Get all checkpoints
    fn get_checkpoints(&self) -> Vec<(String, f64)> {
        self.checkpoints.clone()
    }

    /// Lap - get time since last checkpoint or start
    fn lap(&mut self) -> f64 {
        let current = self.elapsed_ms();
        let last = self
            .checkpoints
            .last()
            .map(|(_, t)| *t)
            .unwrap_or(0.0);
        let lap_time = current - last;
        self.checkpoints
            .push((format!("lap_{}", self.checkpoints.len()), current));
        lap_time
    }

    fn __repr__(&self) -> String {
        format!("Timer(elapsed={:.2}ms)", self.elapsed_ms())
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// System information utilities
#[pyclass]
pub struct SystemInfo;

#[pymethods]
impl SystemInfo {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Get number of CPU cores
    fn cpu_count(&self) -> usize {
        num_cpus::get()
    }

    /// Get number of physical CPU cores
    fn cpu_count_physical(&self) -> usize {
        num_cpus::get_physical()
    }

    /// Get Rayon thread pool size
    fn rayon_threads(&self) -> usize {
        rayon::current_num_threads()
    }

    /// Get Rust version
    fn rust_version(&self) -> String {
        env!("CARGO_PKG_RUST_VERSION").to_string()
    }

    /// Get package version
    fn package_version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Get system info as dictionary
    fn to_dict(&self) -> HashMap<String, String> {
        HashMap::from([
            ("cpu_count".to_string(), self.cpu_count().to_string()),
            (
                "cpu_count_physical".to_string(),
                self.cpu_count_physical().to_string(),
            ),
            (
                "rayon_threads".to_string(),
                self.rayon_threads().to_string(),
            ),
            ("rust_version".to_string(), self.rust_version()),
            ("package_version".to_string(), self.package_version()),
        ])
    }

    fn __repr__(&self) -> String {
        format!(
            "SystemInfo(cpus={}, threads={})",
            self.cpu_count(),
            self.rayon_threads()
        )
    }
}

/// Size formatting utilities
#[pyclass]
pub struct SizeFormatter;

#[pymethods]
impl SizeFormatter {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Format bytes as human-readable string
    #[pyo3(signature = (bytes, binary=true))]
    fn format_bytes(&self, bytes: u64, binary: bool) -> String {
        let (divisor, units): (f64, &[&str]) = if binary {
            (1024.0, &["B", "KiB", "MiB", "GiB", "TiB", "PiB"])
        } else {
            (1000.0, &["B", "KB", "MB", "GB", "TB", "PB"])
        };

        let mut value = bytes as f64;
        let mut unit_index = 0;

        while value >= divisor && unit_index < units.len() - 1 {
            value /= divisor;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{} {}", bytes, units[0])
        } else {
            format!("{:.2} {}", value, units[unit_index])
        }
    }

    /// Parse human-readable size string to bytes
    fn parse_size(&self, size_str: &str) -> PyResult<u64> {
        let size_str = size_str.trim().to_uppercase();

        let (num_part, unit) = if size_str.ends_with("PIB") || size_str.ends_with("PB") {
            let len = if size_str.ends_with("PIB") { 3 } else { 2 };
            (&size_str[..size_str.len() - len], &size_str[size_str.len() - len..])
        } else if size_str.ends_with("TIB") || size_str.ends_with("TB") {
            let len = if size_str.ends_with("TIB") { 3 } else { 2 };
            (&size_str[..size_str.len() - len], &size_str[size_str.len() - len..])
        } else if size_str.ends_with("GIB") || size_str.ends_with("GB") {
            let len = if size_str.ends_with("GIB") { 3 } else { 2 };
            (&size_str[..size_str.len() - len], &size_str[size_str.len() - len..])
        } else if size_str.ends_with("MIB") || size_str.ends_with("MB") {
            let len = if size_str.ends_with("MIB") { 3 } else { 2 };
            (&size_str[..size_str.len() - len], &size_str[size_str.len() - len..])
        } else if size_str.ends_with("KIB") || size_str.ends_with("KB") {
            let len = if size_str.ends_with("KIB") { 3 } else { 2 };
            (&size_str[..size_str.len() - len], &size_str[size_str.len() - len..])
        } else if size_str.ends_with('B') {
            (&size_str[..size_str.len() - 1], "B")
        } else {
            (size_str.as_str(), "B")
        };

        let num: f64 = num_part.trim().parse().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid size: {}", size_str))
        })?;

        let multiplier: u64 = match unit {
            "PIB" => 1024 * 1024 * 1024 * 1024 * 1024,
            "PB" => 1000 * 1000 * 1000 * 1000 * 1000,
            "TIB" => 1024 * 1024 * 1024 * 1024,
            "TB" => 1000 * 1000 * 1000 * 1000,
            "GIB" => 1024 * 1024 * 1024,
            "GB" => 1000 * 1000 * 1000,
            "MIB" => 1024 * 1024,
            "MB" => 1000 * 1000,
            "KIB" => 1024,
            "KB" => 1000,
            "B" | "" => 1,
            _ => 1,
        };

        Ok((num * multiplier as f64) as u64)
    }
}

/// Duration formatting utilities
#[pyclass]
pub struct DurationFormatter;

#[pymethods]
impl DurationFormatter {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Format milliseconds as human-readable string
    fn format_ms(&self, ms: f64) -> String {
        if ms < 1.0 {
            format!("{:.2}µs", ms * 1000.0)
        } else if ms < 1000.0 {
            format!("{:.2}ms", ms)
        } else if ms < 60000.0 {
            format!("{:.2}s", ms / 1000.0)
        } else if ms < 3600000.0 {
            let mins = (ms / 60000.0).floor();
            let secs = (ms % 60000.0) / 1000.0;
            format!("{}m {:.1}s", mins, secs)
        } else {
            let hours = (ms / 3600000.0).floor();
            let mins = ((ms % 3600000.0) / 60000.0).floor();
            format!("{}h {}m", hours, mins)
        }
    }

    /// Format seconds as human-readable string
    fn format_seconds(&self, secs: f64) -> String {
        self.format_ms(secs * 1000.0)
    }

    /// Parse duration string to milliseconds
    fn parse_duration(&self, duration_str: &str) -> PyResult<f64> {
        let duration_str = duration_str.trim().to_lowercase();

        if duration_str.ends_with("ms") {
            duration_str[..duration_str.len() - 2]
                .trim()
                .parse()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid duration"))
        } else if duration_str.ends_with("us") || duration_str.ends_with("µs") {
            let len = if duration_str.ends_with("µs") { 2 } else { 2 };
            let value: f64 = duration_str[..duration_str.len() - len]
                .trim()
                .parse()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid duration"))?;
            Ok(value / 1000.0)
        } else if duration_str.ends_with('s') {
            let value: f64 = duration_str[..duration_str.len() - 1]
                .trim()
                .parse()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid duration"))?;
            Ok(value * 1000.0)
        } else if duration_str.ends_with('m') {
            let value: f64 = duration_str[..duration_str.len() - 1]
                .trim()
                .parse()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid duration"))?;
            Ok(value * 60000.0)
        } else if duration_str.ends_with('h') {
            let value: f64 = duration_str[..duration_str.len() - 1]
                .trim()
                .parse()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid duration"))?;
            Ok(value * 3600000.0)
        } else {
            duration_str
                .parse()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid duration"))
        }
    }
}

/// Get system information as a Python function
#[pyfunction]
pub fn get_system_info() -> HashMap<String, String> {
    HashMap::from([
        ("cpu_count".to_string(), num_cpus::get().to_string()),
        (
            "cpu_count_physical".to_string(),
            num_cpus::get_physical().to_string(),
        ),
        (
            "rayon_threads".to_string(),
            rayon::current_num_threads().to_string(),
        ),
        (
            "rust_version".to_string(),
            "1.70+".to_string(),
        ),
        (
            "package_version".to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
        ),
    ])
}

/// Create a new timer
#[pyfunction]
pub fn create_timer() -> Timer {
    Timer::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer() {
        let mut timer = Timer::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10.0);
        timer.checkpoint("test");
        assert_eq!(timer.get_checkpoints().len(), 1);
    }

    #[test]
    fn test_size_formatter() {
        let formatter = SizeFormatter::new();
        assert_eq!(formatter.format_bytes(1024, true), "1.00 KiB");
        assert_eq!(formatter.format_bytes(1000, false), "1.00 KB");
        assert_eq!(formatter.format_bytes(500, true), "500 B");
    }

    #[test]
    fn test_duration_formatter() {
        let formatter = DurationFormatter::new();
        assert!(formatter.format_ms(0.5).contains("µs"));
        assert!(formatter.format_ms(500.0).contains("ms"));
        assert!(formatter.format_ms(5000.0).contains("s"));
    }
}
