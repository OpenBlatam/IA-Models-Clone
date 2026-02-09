//! Async Utilities
//!
//! Provides async/await utilities and helpers.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration, Instant};

/// Async semaphore for rate limiting
#[pyclass]
pub struct AsyncSemaphore {
    semaphore: Arc<Semaphore>,
}

#[pymethods]
impl AsyncSemaphore {
    #[new]
    pub fn new(permits: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(permits)),
        }
    }

    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    pub fn add_permits(&self, n: usize) {
        self.semaphore.add_permits(n);
    }
}

/// Async rate limiter
#[pyclass]
pub struct AsyncRateLimiter {
    semaphore: Arc<Semaphore>,
    refill_interval: Duration,
    max_permits: usize,
}

#[pymethods]
impl AsyncRateLimiter {
    #[new]
    pub fn new(permits_per_second: usize) -> Self {
        let semaphore = Arc::new(Semaphore::new(permits_per_second));
        let refill_interval = Duration::from_millis(1000 / permits_per_second as u64);
        
        Self {
            semaphore: semaphore.clone(),
            refill_interval,
            max_permits: permits_per_second,
        }
    }

    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }
}

/// Async timer
#[pyclass]
pub struct AsyncTimer {
    start: Instant,
}

#[pymethods]
impl AsyncTimer {
    #[new]
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    pub fn reset(&mut self) {
        self.start = Instant::now();
    }
}

/// Async batch processor
#[pyclass]
pub struct AsyncBatchProcessor {
    batch_size: usize,
    concurrency: usize,
}

#[pymethods]
impl AsyncBatchProcessor {
    #[new]
    pub fn new(batch_size: usize, concurrency: usize) -> Self {
        Self {
            batch_size,
            concurrency,
        }
    }

    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn get_concurrency(&self) -> usize {
        self.concurrency
    }
}

#[pyfunction]
pub fn create_async_semaphore(permits: usize) -> AsyncSemaphore {
    AsyncSemaphore::new(permits)
}

#[pyfunction]
pub fn create_async_rate_limiter(permits_per_second: usize) -> AsyncRateLimiter {
    AsyncRateLimiter::new(permits_per_second)
}

#[pyfunction]
pub fn create_async_timer() -> AsyncTimer {
    AsyncTimer::new()
}

#[pyfunction]
pub fn create_async_batch_processor(batch_size: usize, concurrency: usize) -> AsyncBatchProcessor {
    AsyncBatchProcessor::new(batch_size, concurrency)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_async_semaphore() {
        let sem = AsyncSemaphore::new(10);
        assert_eq!(sem.available_permits(), 10);
    }

    #[test]
    fn test_async_rate_limiter() {
        let limiter = AsyncRateLimiter::new(10);
        assert!(limiter.available_permits() > 0);
    }

    #[test]
    fn test_async_timer() {
        let timer = AsyncTimer::new();
        std::thread::sleep(Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10.0);
    }
}












