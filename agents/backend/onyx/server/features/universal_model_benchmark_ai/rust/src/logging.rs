//! Logging and observability utilities.
//!
//! Provides structured logging and observability features for benchmarks.

use std::time::Instant;
use crate::types::Metrics;
use crate::benchmark::runner::BenchmarkResult;
use crate::config::BenchmarkConfig;

/// Log level for benchmark operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Trace level - very detailed information.
    Trace,
    /// Debug level - debugging information.
    Debug,
    /// Info level - general information.
    Info,
    /// Warn level - warning messages.
    Warn,
    /// Error level - error messages.
    Error,
}

/// Simple logger for benchmarks.
pub struct BenchmarkLogger {
    level: LogLevel,
    enabled: bool,
}

impl BenchmarkLogger {
    /// Create a new logger with default level.
    pub fn new() -> Self {
        Self {
            level: LogLevel::Info,
            enabled: true,
        }
    }
    
    /// Create a logger with custom level.
    pub fn with_level(level: LogLevel) -> Self {
        Self {
            level,
            enabled: true,
        }
    }
    
    /// Enable or disable logging.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Log a message at trace level.
    pub fn trace(&self, message: &str) {
        if self.enabled && self.level <= LogLevel::Trace {
            eprintln!("[TRACE] {}", message);
        }
    }
    
    /// Log a message at debug level.
    pub fn debug(&self, message: &str) {
        if self.enabled && self.level <= LogLevel::Debug {
            eprintln!("[DEBUG] {}", message);
        }
    }
    
    /// Log a message at info level.
    pub fn info(&self, message: &str) {
        if self.enabled && self.level <= LogLevel::Info {
            eprintln!("[INFO] {}", message);
        }
    }
    
    /// Log a message at warn level.
    pub fn warn(&self, message: &str) {
        if self.enabled && self.level <= LogLevel::Warn {
            eprintln!("[WARN] {}", message);
        }
    }
    
    /// Log a message at error level.
    pub fn error(&self, message: &str) {
        if self.enabled && self.level <= LogLevel::Error {
            eprintln!("[ERROR] {}", message);
        }
    }
    
    /// Log benchmark start.
    pub fn log_benchmark_start(&self, name: &str) {
        self.info(&format!("Starting benchmark: {}", name));
    }
    
    /// Log benchmark completion.
    pub fn log_benchmark_complete(&self, name: &str, duration_ms: f64) {
        self.info(&format!("Benchmark complete: {} (took {:.2}ms)", name, duration_ms));
    }
    
    /// Log metrics.
    pub fn log_metrics(&self, name: &str, metrics: &Metrics) {
        self.info(&format!(
            "Metrics for {}: accuracy={:.2}%, latency_p50={:.2}ms, throughput={:.2}",
            name,
            metrics.accuracy * 100.0,
            metrics.latency_p50,
            metrics.throughput
        ));
    }
    
    /// Log benchmark result.
    pub fn log_benchmark_result(&self, name: &str, result: &BenchmarkResult) {
        self.info(&format!(
            "Benchmark result for {}: iterations={}, avg_latency={:.2}ms, throughput={:.2}, success_rate={:.1}%",
            name,
            result.iterations,
            result.avg_latency_ms,
            result.throughput,
            result.success_rate * 100.0
        ));
    }
    
    /// Log configuration.
    pub fn log_config(&self, config: &BenchmarkConfig) {
        self.debug(&format!(
            "Config: model={}, batch={}, max_tokens={}, temp={:.2}",
            config.model_path,
            config.batch_size,
            config.max_tokens,
            config.temperature
        ));
    }
}

impl Default for BenchmarkLogger {
    fn default() -> Self {
        Self::new()
    }
}

/// Scoped timer that logs on drop.
pub struct ScopedLogger {
    logger: BenchmarkLogger,
    name: String,
    start: Instant,
    level: LogLevel,
}

impl ScopedLogger {
    /// Create a new scoped logger.
    pub fn new(logger: BenchmarkLogger, name: String) -> Self {
        let start = Instant::now();
        logger.debug(&format!("Starting: {}", name));
        Self {
            logger,
            name,
            start,
            level: LogLevel::Debug,
        }
    }
    
    /// Create with custom log level.
    pub fn with_level(logger: BenchmarkLogger, name: String, level: LogLevel) -> Self {
        let start = Instant::now();
        match level {
            LogLevel::Trace => logger.trace(&format!("Starting: {}", name)),
            LogLevel::Debug => logger.debug(&format!("Starting: {}", name)),
            LogLevel::Info => logger.info(&format!("Starting: {}", name)),
            _ => {},
        }
        Self {
            logger,
            name,
            start,
            level,
        }
    }
}

impl Drop for ScopedLogger {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        let message = format!("Completed: {} (took {:.2}ms)", self.name, duration.as_secs_f64() * 1000.0);
        match self.level {
            LogLevel::Trace => self.logger.trace(&message),
            LogLevel::Debug => self.logger.debug(&message),
            LogLevel::Info => self.logger.info(&message),
            _ => {},
        }
    }
}

/// Global logger instance (thread-local).
thread_local! {
    static GLOBAL_LOGGER: std::cell::RefCell<Option<BenchmarkLogger>> = std::cell::RefCell::new(None);
}

/// Set the global logger.
pub fn set_global_logger(logger: BenchmarkLogger) {
    GLOBAL_LOGGER.with(|l| {
        *l.borrow_mut() = Some(logger);
    });
}

/// Get the global logger or create a default one.
pub fn get_logger() -> BenchmarkLogger {
    GLOBAL_LOGGER.with(|l| {
        l.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(BenchmarkLogger::new)
    })
}

/// Log a message using the global logger.
pub fn log(level: LogLevel, message: &str) {
    let logger = get_logger();
    match level {
        LogLevel::Trace => logger.trace(message),
        LogLevel::Debug => logger.debug(message),
        LogLevel::Info => logger.info(message),
        LogLevel::Warn => logger.warn(message),
        LogLevel::Error => logger.error(message),
    }
}

/// Convenience functions for global logging.
pub fn trace(message: &str) {
    log(LogLevel::Trace, message);
}

pub fn debug(message: &str) {
    log(LogLevel::Debug, message);
}

pub fn info(message: &str) {
    log(LogLevel::Info, message);
}

pub fn warn(message: &str) {
    log(LogLevel::Warn, message);
}

pub fn error(message: &str) {
    log(LogLevel::Error, message);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_logger() {
        let logger = BenchmarkLogger::new();
        logger.info("Test message");
    }
    
    #[test]
    fn test_scoped_logger() {
        let logger = BenchmarkLogger::new();
        let _scoped = ScopedLogger::new(logger, "test".to_string());
        // Will log on drop
    }
}












