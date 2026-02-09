//! Inference Metrics
//!
//! Performance metrics collection for inference operations.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::collections::VecDeque;

use serde::{Serialize, Deserialize};

/// Performance metrics for inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    pub total_requests: u64,
    pub total_tokens: u64,
    pub total_latency_ms: f64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub tokens_per_second: f64,
    pub memory_usage_mb: f64,
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            total_tokens: 0,
            total_latency_ms: 0.0,
            avg_latency_ms: 0.0,
            p50_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            tokens_per_second: 0.0,
            memory_usage_mb: 0.0,
        }
    }
}

/// Metrics collector with percentile tracking.
pub struct MetricsCollector {
    latencies: Arc<Mutex<VecDeque<f64>>>,
    max_samples: usize,
    total_requests: Arc<Mutex<u64>>,
    total_tokens: Arc<Mutex<u64>>,
    total_latency: Arc<Mutex<f64>>,
}

impl MetricsCollector {
    /// Create a new metrics collector.
    pub fn new(max_samples: usize) -> Self {
        Self {
            latencies: Arc::new(Mutex::new(VecDeque::with_capacity(max_samples))),
            max_samples,
            total_requests: Arc::new(Mutex::new(0)),
            total_tokens: Arc::new(Mutex::new(0)),
            total_latency: Arc::new(Mutex::new(0.0)),
        }
    }
    
    /// Record an inference operation.
    pub fn record(&self, latency_ms: f64, tokens: usize) {
        // Use expect instead of unwrap for better error messages
        let mut latencies = self.latencies.lock().expect("Latencies lock poisoned");
        let mut total_requests = self.total_requests.lock().expect("Total requests lock poisoned");
        let mut total_tokens = self.total_tokens.lock().expect("Total tokens lock poisoned");
        let mut total_latency = self.total_latency.lock().expect("Total latency lock poisoned");
        
        // Add latency
        if latencies.len() >= self.max_samples {
            latencies.pop_front();
        }
        latencies.push_back(latency_ms);
        
        // Update totals
        *total_requests += 1;
        *total_tokens += tokens as u64;
        *total_latency += latency_ms;
    }
    
    /// Get current metrics.
    pub fn get_metrics(&self) -> InferenceMetrics {
        let latencies = self.latencies.lock().expect("Latencies lock poisoned");
        let total_requests = *self.total_requests.lock().expect("Total requests lock poisoned");
        let total_tokens = *self.total_tokens.lock().expect("Total tokens lock poisoned");
        let total_latency = *self.total_latency.lock().expect("Total latency lock poisoned");
        
        let mut sorted_latencies: Vec<f64> = latencies.iter().copied().collect();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let count = sorted_latencies.len();
        let p50 = if count > 0 {
            sorted_latencies[count * 50 / 100]
        } else {
            0.0
        };
        let p95 = if count > 0 {
            sorted_latencies[count * 95 / 100.min(count - 1)]
        } else {
            0.0
        };
        let p99 = if count > 0 {
            sorted_latencies[count * 99 / 100.min(count - 1)]
        } else {
            0.0
        };
        
        let avg_latency = if total_requests > 0 {
            total_latency / total_requests as f64
        } else {
            0.0
        };
        
        let tokens_per_second = if total_latency > 0.0 {
            (total_tokens as f64) / (total_latency / 1000.0)
        } else {
            0.0
        };
        
        InferenceMetrics {
            total_requests,
            total_tokens,
            total_latency_ms: total_latency,
            avg_latency_ms: avg_latency,
            p50_latency_ms: p50,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
            tokens_per_second,
            memory_usage_mb: 0.0, // TODO: Get actual memory usage
        }
    }
    
    /// Reset all metrics.
    pub fn reset(&self) {
        let mut latencies = self.latencies.lock().expect("Latencies lock poisoned");
        let mut total_requests = self.total_requests.lock().expect("Total requests lock poisoned");
        let mut total_tokens = self.total_tokens.lock().expect("Total tokens lock poisoned");
        let mut total_latency = self.total_latency.lock().expect("Total latency lock poisoned");
        
        latencies.clear();
        *total_requests = 0;
        *total_tokens = 0;
        *total_latency = 0.0;
    }
}

/// Timer for measuring operation duration.
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Start a new timer.
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }
    
    /// Get elapsed time in milliseconds.
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
    
    /// Get elapsed time as Duration.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

