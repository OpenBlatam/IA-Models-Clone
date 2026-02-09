//! Core Traits
//!
//! Fundamental traits for validation, summarization, and statistics.

use crate::error::Result;

/// Trait for types that can be validated.
pub trait Validate {
    /// Validate the value and return an error if invalid.
    fn validate(&self) -> Result<()>;
}

/// Trait for types that can provide a summary or description.
pub trait Summarize {
    /// Get a summary string of the value.
    fn summary(&self) -> String;
    
    /// Get a detailed description.
    fn description(&self) -> String {
        self.summary()
    }
    
    /// Get a short summary (one line).
    fn short_summary(&self) -> String {
        self.summary()
            .lines()
            .next()
            .unwrap_or("")
            .to_string()
    }
}

/// Trait for types that can provide performance statistics.
pub trait PerformanceStats {
    /// Get average latency in milliseconds.
    fn avg_latency_ms(&self) -> f64;
    
    /// Get P50 latency in milliseconds.
    fn p50_latency_ms(&self) -> f64;
    
    /// Get P95 latency in milliseconds.
    fn p95_latency_ms(&self) -> f64;
    
    /// Get P99 latency in milliseconds.
    fn p99_latency_ms(&self) -> f64;
    
    /// Get throughput (operations per second).
    fn throughput(&self) -> f64;
    
    /// Get all latency percentiles.
    fn latency_percentiles(&self) -> (f64, f64, f64, f64) {
        (
            self.avg_latency_ms(),
            self.p50_latency_ms(),
            self.p95_latency_ms(),
            self.p99_latency_ms(),
        )
    }
}

/// Trait for types that can be reset to initial state.
pub trait Reset {
    /// Reset to initial/default state.
    fn reset(&mut self);
    
    /// Create a new instance in default state.
    fn reset_to_default() -> Self
    where
        Self: Default,
    {
        Self::default()
    }
}

/// Trait for types that can provide statistics.
pub trait Statistics {
    /// Get count of items.
    fn count(&self) -> usize;
    
    /// Get total value.
    fn total(&self) -> f64;
    
    /// Get average value.
    fn average(&self) -> f64 {
        let count = self.count();
        if count == 0 {
            0.0
        } else {
            self.total() / count as f64
        }
    }
    
    /// Get minimum value.
    fn min(&self) -> Option<f64> {
        None
    }
    
    /// Get maximum value.
    fn max(&self) -> Option<f64> {
        None
    }
}




