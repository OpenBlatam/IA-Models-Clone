//! Conversion Traits
//!
//! Traits for converting between types.

use crate::error::Result;
use crate::types::Metrics;

/// Trait for types that can be converted to metrics.
pub trait ToMetrics {
    /// Convert to Metrics struct.
    fn to_metrics(&self) -> Metrics;
    
    /// Convert to Metrics with custom accuracy.
    fn to_metrics_with_accuracy(&self, accuracy: f64) -> Metrics {
        let mut metrics = self.to_metrics();
        metrics.accuracy = accuracy;
        metrics
    }
}

/// Trait for types that can be serialized to JSON.
pub trait ToJson {
    /// Convert to JSON string.
    fn to_json(&self) -> Result<String>
    where
        Self: serde::Serialize,
    {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::error::BenchmarkError::serialization(e.to_string()))
    }
    
    /// Convert to JSON string (compact).
    fn to_json_compact(&self) -> Result<String>
    where
        Self: serde::Serialize,
    {
        serde_json::to_string(self)
            .map_err(|e| crate::error::BenchmarkError::serialization(e.to_string()))
    }
}

impl<T: serde::Serialize> ToJson for T {}

/// Trait for types that can be created from JSON.
pub trait FromJson: Sized {
    /// Create from JSON string.
    fn from_json(json: &str) -> Result<Self>
    where
        Self: serde::de::DeserializeOwned,
    {
        serde_json::from_str(json)
            .map_err(|e| crate::error::BenchmarkError::serialization(e.to_string()))
    }
    
    /// Create from JSON bytes.
    fn from_json_bytes(bytes: &[u8]) -> Result<Self>
    where
        Self: serde::de::DeserializeOwned,
    {
        serde_json::from_slice(bytes)
            .map_err(|e| crate::error::BenchmarkError::serialization(e.to_string()))
    }
}

impl<T: serde::de::DeserializeOwned> FromJson for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    
    #[test]
    fn test_to_json() {
        #[derive(Serialize)]
        struct Test {
            value: i32,
        }
        
        let test = Test { value: 42 };
        let json = test.to_json().unwrap();
        assert!(json.contains("42"));
        
        let compact = test.to_json_compact().unwrap();
        assert_eq!(compact, r#"{"value":42}"#);
    }
    
    #[test]
    fn test_from_json() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Test {
            value: i32,
        }
        
        let json = r#"{"value": 42}"#;
        let test: Test = FromJson::from_json(json).unwrap();
        assert_eq!(test.value, 42);
        
        let bytes = json.as_bytes();
        let test2: Test = FromJson::from_json_bytes(bytes).unwrap();
        assert_eq!(test2.value, 42);
    }
}




