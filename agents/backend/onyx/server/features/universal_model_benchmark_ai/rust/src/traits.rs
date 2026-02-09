//! Useful traits for common operations.
//!
//! Provides traits that can be implemented for various types to add
//! common functionality and reduce boilerplate.

/// Trait for types that can be validated.
pub trait Validate {
    /// Validate the value and return an error if invalid.
    fn validate(&self) -> crate::error::Result<()>;
}

/// Trait for types that can provide a summary or description.
pub trait Summarize {
    /// Get a summary string of the value.
    fn summary(&self) -> String;
    
    /// Get a detailed description.
    fn description(&self) -> String {
        self.summary()
    }
}

/// Trait for types that can be converted to metrics.
pub trait ToMetrics {
    /// Convert to Metrics struct.
    fn to_metrics(&self) -> crate::types::Metrics;
}

/// Trait for types that can be serialized to JSON.
pub trait ToJson {
    /// Convert to JSON string.
    fn to_json(&self) -> crate::error::Result<String>
    where
        Self: serde::Serialize,
    {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::error::BenchmarkError::serialization(e.to_string()))
    }
}

impl<T: serde::Serialize> ToJson for T {}

/// Trait for types that can be created from JSON.
pub trait FromJson: Sized {
    /// Create from JSON string.
    fn from_json(json: &str) -> crate::error::Result<Self>
    where
        Self: serde::de::DeserializeOwned,
    {
        serde_json::from_str(json)
            .map_err(|e| crate::error::BenchmarkError::serialization(e.to_string()))
    }
}

impl<T: serde::de::DeserializeOwned> FromJson for T {}

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
}

/// Trait for types that can be reset to initial state.
pub trait Reset {
    /// Reset to initial/default state.
    fn reset(&mut self);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_to_json() {
        use serde::Serialize;
        
        #[derive(Serialize)]
        struct Test {
            value: i32,
        }
        
        let test = Test { value: 42 };
        let json = test.to_json().unwrap();
        assert!(json.contains("42"));
    }
    
    #[test]
    fn test_from_json() {
        use serde::{Deserialize, Serialize};
        
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Test {
            value: i32,
        }
        
        let json = r#"{"value": 42}"#;
        let test: Test = FromJson::from_json(json).unwrap();
        assert_eq!(test.value, 42);
    }
}












