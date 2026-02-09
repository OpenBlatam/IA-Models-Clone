//! Advanced Logging System
//!
//! Provides structured logging with levels, formatting, and filtering.

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// Log level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "TRACE" => Some(LogLevel::Trace),
            "DEBUG" => Some(LogLevel::Debug),
            "INFO" => Some(LogLevel::Info),
            "WARN" => Some(LogLevel::Warn),
            "ERROR" => Some(LogLevel::Error),
            _ => None,
        }
    }
}

/// Log entry
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level: LogLevel,
    pub message: String,
    pub module: String,
    pub timestamp: u64,
    pub metadata: HashMap<String, String>,
}

/// Logger with filtering and formatting
#[pyclass]
pub struct Logger {
    level: Arc<Mutex<LogLevel>>,
    entries: Arc<Mutex<Vec<LogEntry>>>,
    max_entries: usize,
}

#[pymethods]
impl Logger {
    #[new]
    #[pyo3(signature = (level="INFO", max_entries=1000))]
    pub fn new(level: String, max_entries: usize) -> PyResult<Self> {
        let log_level = LogLevel::from_str(&level)
            .ok_or_else(|| PyValueError::new_err(format!("Invalid log level: {}", level)))?;
        
        Ok(Self {
            level: Arc::new(Mutex::new(log_level)),
            entries: Arc::new(Mutex::new(Vec::new())),
            max_entries,
        })
    }

    pub fn set_level(&self, level: String) -> PyResult<()> {
        let log_level = LogLevel::from_str(&level)
            .ok_or_else(|| PyValueError::new_err(format!("Invalid log level: {}", level)))?;
        *self.level.lock().unwrap() = log_level;
        Ok(())
    }

    pub fn get_level(&self) -> String {
        self.level.lock().unwrap().as_str().to_string()
    }

    pub fn trace(&self, message: String, module: Option<String>) -> PyResult<()> {
        self.log(LogLevel::Trace, message, module, HashMap::new())
    }

    pub fn debug(&self, message: String, module: Option<String>) -> PyResult<()> {
        self.log(LogLevel::Debug, message, module, HashMap::new())
    }

    pub fn info(&self, message: String, module: Option<String>) -> PyResult<()> {
        self.log(LogLevel::Info, message, module, HashMap::new())
    }

    pub fn warn(&self, message: String, module: Option<String>) -> PyResult<()> {
        self.log(LogLevel::Warn, message, module, HashMap::new())
    }

    pub fn error(&self, message: String, module: Option<String>) -> PyResult<()> {
        self.log(LogLevel::Error, message, module, HashMap::new())
    }

    pub fn get_entries(&self, level: Option<String>, limit: Option<usize>) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            let entries = self.entries.lock().unwrap();
            let filtered: Vec<&LogEntry> = if let Some(level_str) = level {
                let filter_level = LogLevel::from_str(&level_str)
                    .ok_or_else(|| PyValueError::new_err(format!("Invalid log level: {}", level_str)))?;
                entries.iter()
                    .filter(|e| e.level == filter_level)
                    .collect()
            } else {
                entries.iter().collect()
            };

            let limit = limit.unwrap_or(filtered.len());
            let result: Vec<PyObject> = filtered.iter()
                .take(limit)
                .map(|entry| {
                    let dict = PyDict::new(py);
                    dict.set_item("level", entry.level.as_str()).unwrap();
                    dict.set_item("message", &entry.message).unwrap();
                    dict.set_item("module", &entry.module).unwrap();
                    dict.set_item("timestamp", entry.timestamp).unwrap();
                    dict.into()
                })
                .collect();
            
            Ok(result)
        })
    }

    pub fn clear(&self) -> PyResult<()> {
        let mut entries = self.entries.lock().unwrap();
        entries.clear();
        Ok(())
    }

    pub fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let entries = self.entries.lock().unwrap();
            let mut stats = HashMap::new();
            
            for entry in entries.iter() {
                *stats.entry(entry.level.as_str()).or_insert(0) += 1;
            }
            
            let dict = PyDict::new(py);
            dict.set_item("total", entries.len())?;
            for (level, count) in stats {
                dict.set_item(level, count)?;
            }
            Ok(dict.into())
        })
    }
}

impl Logger {
    fn log(&self, level: LogLevel, message: String, module: Option<String>, metadata: HashMap<String, String>) -> PyResult<()> {
        let current_level = *self.level.lock().unwrap();
        if level < current_level {
            return Ok(());
        }

        let entry = LogEntry {
            level,
            message,
            module: module.unwrap_or_else(|| "default".to_string()),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metadata,
        };

        // Also log to standard logger
        match level {
            LogLevel::Trace => log::trace!("[{}] {}", entry.module, entry.message),
            LogLevel::Debug => log::debug!("[{}] {}", entry.module, entry.message),
            LogLevel::Info => log::info!("[{}] {}", entry.module, entry.message),
            LogLevel::Warn => log::warn!("[{}] {}", entry.module, entry.message),
            LogLevel::Error => log::error!("[{}] {}", entry.module, entry.message),
        }

        let mut entries = self.entries.lock().unwrap();
        entries.push(entry);
        
        // Limit entries
        if entries.len() > self.max_entries {
            entries.remove(0);
        }

        Ok(())
    }
}

#[pyfunction]
pub fn create_logger(level: Option<String>, max_entries: Option<usize>) -> Logger {
    Logger::new(
        level.unwrap_or_else(|| "INFO".to_string()),
        max_entries.unwrap_or(1000),
    ).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level() {
        assert_eq!(LogLevel::Info.as_str(), "INFO");
        assert_eq!(LogLevel::from_str("INFO"), Some(LogLevel::Info));
    }

    #[test]
    fn test_logger() {
        let logger = Logger::new("INFO".to_string(), 100).unwrap();
        logger.info("Test message".to_string(), None).unwrap();
        assert!(logger.get_entries(None, None).unwrap().len() > 0);
    }
}












