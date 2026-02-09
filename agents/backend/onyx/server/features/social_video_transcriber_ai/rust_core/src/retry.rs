//! Retry Logic and Circuit Breaker
//!
//! Provides retry mechanisms and circuit breaker pattern.

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Retry strategy
#[derive(Debug, Clone, Copy)]
pub enum RetryStrategy {
    Exponential,
    Linear,
    Fixed,
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_attempts: usize,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub strategy: RetryStrategy,
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 100,
            max_delay_ms: 5000,
            strategy: RetryStrategy::Exponential,
            backoff_multiplier: 2.0,
        }
    }
}

/// Retry executor
#[pyclass]
pub struct RetryExecutor {
    config: Arc<Mutex<RetryConfig>>,
    stats: Arc<Mutex<RetryStats>>,
}

#[derive(Debug, Default)]
struct RetryStats {
    total_attempts: usize,
    successful_attempts: usize,
    failed_attempts: usize,
    total_retries: usize,
}

#[pymethods]
impl RetryExecutor {
    #[new]
    pub fn new() -> Self {
        Self {
            config: Arc::new(Mutex::new(RetryConfig::default())),
            stats: Arc::new(Mutex::new(RetryStats::default())),
        }
    }

    pub fn with_config(
        max_attempts: usize,
        initial_delay_ms: u64,
        max_delay_ms: u64,
        strategy: String,
    ) -> PyResult<Self> {
        let retry_strategy = match strategy.to_lowercase().as_str() {
            "exponential" => RetryStrategy::Exponential,
            "linear" => RetryStrategy::Linear,
            "fixed" => RetryStrategy::Fixed,
            _ => return Err(PyValueError::new_err(format!("Unknown strategy: {}", strategy))),
        };

        Ok(Self {
            config: Arc::new(Mutex::new(RetryConfig {
                max_attempts,
                initial_delay_ms,
                max_delay_ms,
                strategy: retry_strategy,
                backoff_multiplier: 2.0,
            })),
            stats: Arc::new(Mutex::new(RetryStats::default())),
        })
    }

    pub fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stats = self.stats.lock().unwrap();
            let dict = PyDict::new(py);
            dict.set_item("total_attempts", stats.total_attempts)?;
            dict.set_item("successful_attempts", stats.successful_attempts)?;
            dict.set_item("failed_attempts", stats.failed_attempts)?;
            dict.set_item("total_retries", stats.total_retries)?;
            Ok(dict.into())
        })
    }

    pub fn reset_stats(&self) -> PyResult<()> {
        let mut stats = self.stats.lock().unwrap();
        *stats = RetryStats::default();
        Ok(())
    }
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker
#[pyclass]
pub struct CircuitBreaker {
    state: Arc<Mutex<CircuitState>>,
    failure_threshold: usize,
    success_threshold: usize,
    timeout_ms: u64,
    failures: Arc<Mutex<usize>>,
    successes: Arc<Mutex<usize>>,
    last_failure_time: Arc<Mutex<Option<Instant>>>,
}

#[pymethods]
impl CircuitBreaker {
    #[new]
    pub fn new(failure_threshold: usize, timeout_ms: u64) -> Self {
        Self {
            state: Arc::new(Mutex::new(CircuitState::Closed)),
            failure_threshold,
            success_threshold: 1,
            timeout_ms,
            failures: Arc::new(Mutex::new(0)),
            successes: Arc::new(Mutex::new(0)),
            last_failure_time: Arc::new(Mutex::new(None)),
        }
    }

    pub fn call(&self) -> PyResult<bool> {
        let state = *self.state.lock().unwrap();
        
        match state {
            CircuitState::Open => {
                // Check if timeout has passed
                if let Some(last_failure) = *self.last_failure_time.lock().unwrap() {
                    if last_failure.elapsed().as_millis() as u64 >= self.timeout_ms {
                        *self.state.lock().unwrap() = CircuitState::HalfOpen;
                        *self.successes.lock().unwrap() = 0;
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            CircuitState::HalfOpen => Ok(true),
            CircuitState::Closed => Ok(true),
        }
    }

    pub fn record_success(&self) -> PyResult<()> {
        let mut state = self.state.lock().unwrap();
        let mut successes = self.successes.lock().unwrap();
        let mut failures = self.failures.lock().unwrap();
        
        match *state {
            CircuitState::HalfOpen => {
                *successes += 1;
                if *successes >= self.success_threshold {
                    *state = CircuitState::Closed;
                    *failures = 0;
                }
            }
            CircuitState::Closed => {
                *failures = 0;
            }
            _ => {}
        }
        
        Ok(())
    }

    pub fn record_failure(&self) -> PyResult<()> {
        let mut state = self.state.lock().unwrap();
        let mut failures = self.failures.lock().unwrap();
        
        *failures += 1;
        *self.last_failure_time.lock().unwrap() = Some(Instant::now());
        
        if *failures >= self.failure_threshold {
            *state = CircuitState::Open;
        }
        
        Ok(())
    }

    pub fn get_state(&self) -> String {
        match *self.state.lock().unwrap() {
            CircuitState::Closed => "closed".to_string(),
            CircuitState::Open => "open".to_string(),
            CircuitState::HalfOpen => "half_open".to_string(),
        }
    }

    pub fn reset(&self) -> PyResult<()> {
        *self.state.lock().unwrap() = CircuitState::Closed;
        *self.failures.lock().unwrap() = 0;
        *self.successes.lock().unwrap() = 0;
        *self.last_failure_time.lock().unwrap() = None;
        Ok(())
    }
}

#[pyfunction]
pub fn create_retry_executor() -> RetryExecutor {
    RetryExecutor::new()
}

#[pyfunction]
pub fn create_circuit_breaker(failure_threshold: usize, timeout_ms: u64) -> CircuitBreaker {
    CircuitBreaker::new(failure_threshold, timeout_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_executor() {
        let executor = RetryExecutor::new();
        assert!(executor.get_stats().is_ok());
    }

    #[test]
    fn test_circuit_breaker() {
        let breaker = CircuitBreaker::new(5, 1000);
        assert!(breaker.call().unwrap());
        assert_eq!(breaker.get_state(), "closed");
    }
}












