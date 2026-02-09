//! Conversion utilities for common type conversions.
//!
//! Provides convenient conversion functions between common types.

use crate::error::{Result, BenchmarkError};
use crate::types::Metrics;
use crate::benchmark::runner::BenchmarkResult;

/// Convert BenchmarkResult to Metrics.
impl From<&BenchmarkResult> for Metrics {
    fn from(result: &BenchmarkResult) -> Self {
        Metrics::builder()
            .latency_p50(result.p50_latency_ms)
            .latency_p95(result.p95_latency_ms)
            .latency_p99(result.p99_latency_ms)
            .throughput(result.throughput)
            .build()
    }
}

/// Convert BenchmarkResult to Metrics with accuracy.
pub fn benchmark_result_to_metrics(
    result: &BenchmarkResult,
    accuracy: f64,
) -> Metrics {
    Metrics::builder()
        .accuracy(accuracy)
        .latency_p50(result.p50_latency_ms)
        .latency_p95(result.p95_latency_ms)
        .latency_p99(result.p99_latency_ms)
        .throughput(result.throughput)
        .build()
}

/// Convert latency vector to Metrics.
pub fn latencies_to_metrics(
    latencies: &[f64],
    accuracy: f64,
    throughput: f64,
) -> Result<Metrics> {
    if latencies.is_empty() {
        return Err(BenchmarkError::invalid_input("Latencies cannot be empty"));
    }
    
    let mut sorted = latencies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let len = sorted.len();
    let p50 = if len > 0 {
        sorted[len * 50 / 100.min(len - 1)]
    } else {
        0.0
    };
    let p95 = if len > 0 {
        sorted[len * 95 / 100.min(len - 1)]
    } else {
        0.0
    };
    let p99 = if len > 0 {
        sorted[len * 99 / 100.min(len - 1)]
    } else {
        0.0
    };
    
    Ok(Metrics::builder()
        .accuracy(accuracy)
        .latency_p50(p50)
        .latency_p95(p95)
        .latency_p99(p99)
        .throughput(throughput)
        .build())
}

/// Convert Metrics to BenchmarkResult.
impl From<&Metrics> for BenchmarkResult {
    fn from(metrics: &Metrics) -> Self {
        BenchmarkResult {
            iterations: 0,
            total_time_ms: 0.0,
            avg_latency_ms: metrics.latency_p50,
            p50_latency_ms: metrics.latency_p50,
            p95_latency_ms: metrics.latency_p95,
            p99_latency_ms: metrics.latency_p99,
            throughput: metrics.throughput,
            success_rate: if metrics.accuracy > 0.0 { 1.0 } else { 0.0 },
            errors: Vec::new(),
        }
    }
}

/// Convert f64 to usize safely.
pub fn f64_to_usize(value: f64) -> Result<usize> {
    if value < 0.0 {
        return Err(BenchmarkError::invalid_input(
            format!("Cannot convert negative f64 to usize: {}", value)
        ));
    }
    if value > usize::MAX as f64 {
        return Err(BenchmarkError::invalid_input(
            format!("f64 value too large for usize: {}", value)
        ));
    }
    Ok(value as usize)
}

/// Convert usize to f64.
pub fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

/// Convert i64 to f64.
pub fn i64_to_f64(value: i64) -> f64 {
    value as f64
}

/// Convert f64 to i64 safely.
pub fn f64_to_i64(value: f64) -> Result<i64> {
    if value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err(BenchmarkError::invalid_input(
            format!("f64 value out of range for i64: {}", value)
        ));
    }
    Ok(value as i64)
}

/// Convert string to number with error handling.
pub fn str_to_f64(s: &str) -> Result<f64> {
    s.parse::<f64>()
        .map_err(|e| BenchmarkError::invalid_input(
            format!("Failed to parse '{}' as f64: {}", s, e)
        ))
}

/// Convert string to usize with error handling.
pub fn str_to_usize(s: &str) -> Result<usize> {
    s.parse::<usize>()
        .map_err(|e| BenchmarkError::invalid_input(
            format!("Failed to parse '{}' as usize: {}", s, e)
        ))
}

/// Convert Option<T> to Result<T> with custom error message.
pub fn option_to_result<T>(opt: Option<T>, msg: &str) -> Result<T> {
    opt.ok_or_else(|| BenchmarkError::Other(msg.to_string()))
}

/// Convert Vec<T> to array of fixed size.
pub fn vec_to_array<T, const N: usize>(vec: Vec<T>) -> Result<[T; N]> {
    if vec.len() != N {
        return Err(BenchmarkError::invalid_input(
            format!("Vector length {} does not match array size {}", vec.len(), N)
        ));
    }
    vec.try_into()
        .map_err(|_| BenchmarkError::Other("Failed to convert vector to array".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_f64_to_usize() {
        assert_eq!(f64_to_usize(42.0).unwrap(), 42);
        assert!(f64_to_usize(-1.0).is_err());
    }
    
    #[test]
    fn test_str_to_f64() {
        assert_eq!(str_to_f64("42.5").unwrap(), 42.5);
        assert!(str_to_f64("invalid").is_err());
    }
    
    #[test]
    fn test_benchmark_result_to_metrics() {
        let result = BenchmarkResult {
            iterations: 10,
            total_time_ms: 1000.0,
            avg_latency_ms: 100.0,
            p50_latency_ms: 95.0,
            p95_latency_ms: 150.0,
            p99_latency_ms: 200.0,
            throughput: 10.0,
            success_rate: 1.0,
            errors: Vec::new(),
        };
        
        let metrics: Metrics = (&result).into();
        assert_eq!(metrics.latency_p50, 95.0);
        assert_eq!(metrics.throughput, 10.0);
    }
}

