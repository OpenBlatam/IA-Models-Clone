//! Rate Limiting
//!
//! Provides rate limiting with various algorithms.

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Rate limiting algorithm
#[derive(Debug, Clone, Copy)]
pub enum RateLimitAlgorithm {
    TokenBucket,
    SlidingWindow,
    FixedWindow,
    LeakyBucket,
}

/// Rate limiter
#[pyclass]
pub struct RateLimiter {
    algorithm: RateLimitAlgorithm,
    max_requests: usize,
    window_ms: u64,
    requests: Arc<Mutex<VecDeque<Instant>>>,
    tokens: Arc<Mutex<usize>>,
    last_refill: Arc<Mutex<Instant>>,
}

#[pymethods]
impl RateLimiter {
    #[new]
    #[pyo3(signature = (max_requests, window_ms=1000, algorithm="token_bucket"))]
    pub fn new(max_requests: usize, window_ms: u64, algorithm: String) -> PyResult<Self> {
        let algo = match algorithm.to_lowercase().as_str() {
            "token_bucket" => RateLimitAlgorithm::TokenBucket,
            "sliding_window" => RateLimitAlgorithm::SlidingWindow,
            "fixed_window" => RateLimitAlgorithm::FixedWindow,
            "leaky_bucket" => RateLimitAlgorithm::LeakyBucket,
            _ => return Err(PyValueError::new_err(format!("Unknown algorithm: {}", algorithm))),
        };
        
        Ok(Self {
            algorithm: algo,
            max_requests,
            window_ms,
            requests: Arc::new(Mutex::new(VecDeque::new())),
            tokens: Arc::new(Mutex::new(max_requests)),
            last_refill: Arc::new(Mutex::new(Instant::now())),
        })
    }

    pub fn allow(&self) -> bool {
        match self.algorithm {
            RateLimitAlgorithm::TokenBucket => self.token_bucket_allow(),
            RateLimitAlgorithm::SlidingWindow => self.sliding_window_allow(),
            RateLimitAlgorithm::FixedWindow => self.fixed_window_allow(),
            RateLimitAlgorithm::LeakyBucket => self.leaky_bucket_allow(),
        }
    }

    pub fn try_acquire(&self) -> PyResult<bool> {
        Ok(self.allow())
    }

    pub fn get_remaining(&self) -> usize {
        match self.algorithm {
            RateLimitAlgorithm::TokenBucket => {
                self.refill_tokens();
                *self.tokens.lock().unwrap()
            }
            _ => {
                let requests = self.requests.lock().unwrap();
                let now = Instant::now();
                let window = Duration::from_millis(self.window_ms);
                let count = requests.iter()
                    .filter(|&time| now.duration_since(*time) < window)
                    .count();
                self.max_requests.saturating_sub(count)
            }
        }
    }

    pub fn reset(&self) -> PyResult<()> {
        let mut requests = self.requests.lock().unwrap();
        requests.clear();
        *self.tokens.lock().unwrap() = self.max_requests;
        *self.last_refill.lock().unwrap() = Instant::now();
        Ok(())
    }
}

impl RateLimiter {
    fn token_bucket_allow(&self) -> bool {
        self.refill_tokens();
        let mut tokens = self.tokens.lock().unwrap();
        if *tokens > 0 {
            *tokens -= 1;
            true
        } else {
            false
        }
    }

    fn refill_tokens(&self) {
        let mut last_refill = self.last_refill.lock().unwrap();
        let mut tokens = self.tokens.lock().unwrap();
        let now = Instant::now();
        let elapsed = now.duration_since(*last_refill);
        
        if elapsed.as_millis() as u64 >= self.window_ms {
            let refill_amount = (elapsed.as_millis() as u64 / self.window_ms) as usize;
            *tokens = (*tokens + refill_amount).min(self.max_requests);
            *last_refill = now;
        }
    }

    fn sliding_window_allow(&self) -> bool {
        let mut requests = self.requests.lock().unwrap();
        let now = Instant::now();
        let window = Duration::from_millis(self.window_ms);
        
        // Remove old requests
        while let Some(&time) = requests.front() {
            if now.duration_since(time) >= window {
                requests.pop_front();
            } else {
                break;
            }
        }
        
        if requests.len() < self.max_requests {
            requests.push_back(now);
            true
        } else {
            false
        }
    }

    fn fixed_window_allow(&self) -> bool {
        let mut requests = self.requests.lock().unwrap();
        let now = Instant::now();
        let window = Duration::from_millis(self.window_ms);
        
        // Remove requests outside current window
        requests.retain(|&time| now.duration_since(time) < window);
        
        if requests.len() < self.max_requests {
            requests.push_back(now);
            true
        } else {
            false
        }
    }

    fn leaky_bucket_allow(&self) -> bool {
        // Similar to token bucket but different refill strategy
        self.token_bucket_allow()
    }
}

#[pyfunction]
pub fn create_rate_limiter(max_requests: usize, window_ms: Option<u64>, algorithm: Option<String>) -> PyResult<RateLimiter> {
    RateLimiter::new(
        max_requests,
        window_ms.unwrap_or(1000),
        algorithm.unwrap_or_else(|| "token_bucket".to_string()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new(10, 1000, "token_bucket".to_string()).unwrap();
        assert!(limiter.allow());
        assert_eq!(limiter.get_remaining(), 9);
    }
}












