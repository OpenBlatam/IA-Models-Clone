//! Advanced validation utilities.
//!
//! Provides comprehensive validation functions for various types and scenarios.

use crate::error::{Result, BenchmarkError};
use crate::types::Metrics;
use crate::config::BenchmarkConfig;
use crate::constants::thresholds;

// For num_traits::Zero, we'll implement a simple version
trait Zero {
    fn zero() -> Self;
}

impl Zero for f64 {
    fn zero() -> Self { 0.0 }
}

impl Zero for i32 {
    fn zero() -> Self { 0 }
}

impl Zero for usize {
    fn zero() -> Self { 0 }
}

/// Validate that a value is within a range.
pub fn validate_in_range<T: PartialOrd + std::fmt::Display>(
    value: T,
    min: T,
    max: T,
    name: &str,
) -> Result<()> {
    if value < min || value > max {
        return Err(BenchmarkError::invalid_input(
            format!("{} must be in range [{}, {}], got {}", name, min, max, value)
        ));
    }
    Ok(())
}

/// Validate that a value is positive.
pub fn validate_positive<T: PartialOrd + std::fmt::Display + Zero>(
    value: T,
    name: &str,
) -> Result<()> {
    if value <= T::zero() {
        return Err(BenchmarkError::invalid_input(
            format!("{} must be positive, got {}", name, value)
        ));
    }
    Ok(())
}

/// Validate that a value is non-negative.
pub fn validate_non_negative<T: PartialOrd + std::fmt::Display + Zero>(
    value: T,
    name: &str,
) -> Result<()> {
    if value < T::zero() {
        return Err(BenchmarkError::invalid_input(
            format!("{} must be non-negative, got {}", name, value)
        ));
    }
    Ok(())
}

/// Validate that a string is not empty.
pub fn validate_not_empty(value: &str, name: &str) -> Result<()> {
    if value.is_empty() {
        return Err(BenchmarkError::invalid_input(
            format!("{} cannot be empty", name)
        ));
    }
    Ok(())
}

/// Validate that a slice is not empty.
pub fn validate_not_empty_slice<T>(value: &[T], name: &str) -> Result<()> {
    if value.is_empty() {
        return Err(BenchmarkError::invalid_input(
            format!("{} cannot be empty", name)
        ));
    }
    Ok(())
}

/// Validate that a value is finite.
pub fn validate_finite(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(BenchmarkError::invalid_input(
            format!("{} must be finite, got {}", name, value)
        ));
    }
    Ok(())
}

/// Validate metrics comprehensively.
pub fn validate_metrics_comprehensive(metrics: &Metrics) -> Result<()> {
    validate_in_range(metrics.accuracy, 0.0, 1.0, "accuracy")?;
    validate_non_negative(metrics.latency_p50, "latency_p50")?;
    validate_non_negative(metrics.latency_p95, "latency_p95")?;
    validate_non_negative(metrics.latency_p99, "latency_p99")?;
    validate_non_negative(metrics.throughput, "throughput")?;
    validate_non_negative(metrics.memory_peak_mb, "memory_peak_mb")?;
    
    validate_finite(metrics.accuracy, "accuracy")?;
    validate_finite(metrics.latency_p50, "latency_p50")?;
    validate_finite(metrics.latency_p95, "latency_p95")?;
    validate_finite(metrics.latency_p99, "latency_p99")?;
    validate_finite(metrics.throughput, "throughput")?;
    validate_finite(metrics.memory_peak_mb, "memory_peak_mb")?;
    
    // Validate logical constraints
    if metrics.latency_p95 < metrics.latency_p50 {
        return Err(BenchmarkError::invalid_input(
            format!("latency_p95 ({}) must be >= latency_p50 ({})", 
                metrics.latency_p95, metrics.latency_p50)
        ));
    }
    
    if metrics.latency_p99 < metrics.latency_p95 {
        return Err(BenchmarkError::invalid_input(
            format!("latency_p99 ({}) must be >= latency_p95 ({})", 
                metrics.latency_p99, metrics.latency_p95)
        ));
    }
    
    Ok(())
}

/// Validate that metrics meet performance thresholds.
pub fn validate_metrics_performance(metrics: &Metrics) -> Result<()> {
    if metrics.accuracy < thresholds::MIN_ACCURACY {
        return Err(BenchmarkError::invalid_input(
            format!("Accuracy {} is below minimum threshold {}", 
                metrics.accuracy, thresholds::MIN_ACCURACY)
        ));
    }
    
    if metrics.latency_p50 > thresholds::MAX_LATENCY_P50 {
        return Err(BenchmarkError::invalid_input(
            format!("Latency P50 {} exceeds maximum threshold {}", 
                metrics.latency_p50, thresholds::MAX_LATENCY_P50)
        ));
    }
    
    if metrics.throughput < thresholds::MIN_THROUGHPUT {
        return Err(BenchmarkError::invalid_input(
            format!("Throughput {} is below minimum threshold {}", 
                metrics.throughput, thresholds::MIN_THROUGHPUT)
        ));
    }
    
    Ok(())
}

/// Validate configuration comprehensively.
pub fn validate_config_comprehensive(config: &BenchmarkConfig) -> Result<()> {
    validate_not_empty(&config.model_path, "model_path")?;
    
    validate_in_range(
        config.batch_size,
        crate::config::limits::MIN_BATCH_SIZE,
        crate::config::limits::MAX_BATCH_SIZE,
        "batch_size"
    )?;
    
    validate_in_range(
        config.max_tokens,
        crate::config::limits::MIN_TOKENS,
        crate::config::limits::MAX_TOKENS_LIMIT,
        "max_tokens"
    )?;
    
    validate_in_range(
        config.temperature,
        crate::config::limits::MIN_TEMPERATURE,
        crate::config::limits::MAX_TEMPERATURE,
        "temperature"
    )?;
    
    validate_in_range(
        config.top_p,
        crate::config::limits::MIN_TOP_P,
        crate::config::limits::MAX_TOP_P,
        "top_p"
    )?;
    
    validate_positive(config.top_k as i32, "top_k")?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validate_in_range() {
        assert!(validate_in_range(5, 1, 10, "value").is_ok());
        assert!(validate_in_range(0, 1, 10, "value").is_err());
        assert!(validate_in_range(11, 1, 10, "value").is_err());
    }
    
    #[test]
    fn test_validate_positive() {
        assert!(validate_positive(1.0, "value").is_ok());
        assert!(validate_positive(0.0, "value").is_err());
        assert!(validate_positive(-1.0, "value").is_err());
    }
    
    #[test]
    fn test_validate_metrics_comprehensive() {
        let metrics = Metrics::builder()
            .accuracy(0.9)
            .latency_p50(100.0)
            .latency_p95(200.0)
            .latency_p99(300.0)
            .throughput(50.0)
            .build();
        
        assert!(validate_metrics_comprehensive(&metrics).is_ok());
    }
}

