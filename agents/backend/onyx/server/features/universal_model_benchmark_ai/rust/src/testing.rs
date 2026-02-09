//! Testing utilities for benchmarks.
//!
//! Provides helper functions and utilities for writing tests and test fixtures.

#[cfg(test)]
use crate::error::Result;
#[cfg(test)]
use crate::types::Metrics;
#[cfg(test)]
use crate::config::BenchmarkConfig;
#[cfg(test)]
use crate::benchmark::runner::BenchmarkResult;
#[cfg(test)]
use crate::constants::*;

/// Create a test benchmark configuration.
#[cfg(test)]
pub fn test_config() -> BenchmarkConfig {
    BenchmarkConfig::builder()
        .model_path("test_model".to_string())
        .batch_size(batch_sizes::SMALL)
        .max_tokens(token_limits::SHORT)
        .temperature(temperatures::MEDIUM)
        .top_p(top_p_values::BALANCED)
        .top_k(top_k_values::BALANCED)
        .build()
        .expect("Failed to create test config")
}

/// Create test metrics with default values.
#[cfg(test)]
pub fn test_metrics() -> Metrics {
    Metrics::builder()
        .accuracy(0.9)
        .latency_p50(100.0)
        .latency_p95(200.0)
        .latency_p99(300.0)
        .throughput(50.0)
        .memory_peak_mb(1000.0)
        .build()
}

/// Create test metrics with custom values.
#[cfg(test)]
pub fn test_metrics_custom(
    accuracy: f64,
    latency_p50: f64,
    throughput: f64,
) -> Metrics {
    Metrics::builder()
        .accuracy(accuracy)
        .latency_p50(latency_p50)
        .latency_p95(latency_p50 * 2.0)
        .latency_p99(latency_p50 * 3.0)
        .throughput(throughput)
        .memory_peak_mb(1000.0)
        .build()
}

/// Create a test benchmark result.
#[cfg(test)]
pub fn test_benchmark_result() -> BenchmarkResult {
    BenchmarkResult {
        iterations: 10,
        total_time_ms: 1000.0,
        avg_latency_ms: 100.0,
        p50_latency_ms: 95.0,
        p95_latency_ms: 150.0,
        p99_latency_ms: 200.0,
        throughput: 10.0,
        success_rate: 1.0,
        errors: Vec::new(),
    }
}

/// Create a test benchmark result with errors.
#[cfg(test)]
pub fn test_benchmark_result_with_errors() -> BenchmarkResult {
    BenchmarkResult {
        iterations: 10,
        total_time_ms: 1000.0,
        avg_latency_ms: 100.0,
        p50_latency_ms: 95.0,
        p95_latency_ms: 150.0,
        p99_latency_ms: 200.0,
        throughput: 10.0,
        success_rate: 0.8,
        errors: vec!["Error 1".to_string(), "Error 2".to_string()],
    }
}

/// Generate a vector of test latencies.
#[cfg(test)]
pub fn test_latencies(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| 50.0 + (i as f64 * 10.0))
        .collect()
}

/// Generate a vector of test accuracies.
#[cfg(test)]
pub fn test_accuracies(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| 0.8 + (i as f64 * 0.02))
        .collect()
}

/// Assert that metrics are approximately equal.
#[cfg(test)]
pub fn assert_metrics_approx_eq(
    expected: &Metrics,
    actual: &Metrics,
    epsilon: f64,
) {
    assert!(
        (expected.accuracy - actual.accuracy).abs() < epsilon,
        "Accuracy mismatch: expected {}, got {}",
        expected.accuracy,
        actual.accuracy
    );
    assert!(
        (expected.latency_p50 - actual.latency_p50).abs() < epsilon,
        "Latency P50 mismatch: expected {}, got {}",
        expected.latency_p50,
        actual.latency_p50
    );
    assert!(
        (expected.throughput - actual.throughput).abs() < epsilon,
        "Throughput mismatch: expected {}, got {}",
        expected.throughput,
        actual.throughput
    );
}

/// Assert that benchmark result is successful.
#[cfg(test)]
pub fn assert_benchmark_successful(result: &BenchmarkResult) {
    assert!(
        result.is_successful(),
        "Benchmark should be successful, but got success_rate={}, errors={:?}",
        result.success_rate,
        result.errors
    );
}

/// Assert that metrics meet good performance criteria.
#[cfg(test)]
pub fn assert_good_performance(metrics: &Metrics) {
    assert!(
        metrics.accuracy >= thresholds::GOOD_ACCURACY,
        "Accuracy should be >= {}, got {}",
        thresholds::GOOD_ACCURACY,
        metrics.accuracy
    );
    assert!(
        metrics.latency_p50 <= thresholds::MAX_LATENCY_P50,
        "Latency P50 should be <= {}, got {}",
        thresholds::MAX_LATENCY_P50,
        metrics.latency_p50
    );
    assert!(
        metrics.throughput >= thresholds::MIN_THROUGHPUT,
        "Throughput should be >= {}, got {}",
        thresholds::MIN_THROUGHPUT,
        metrics.throughput
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_test_config() {
        let config = test_config();
        assert_eq!(config.model_path, "test_model");
        assert!(config.is_valid());
    }
    
    #[test]
    fn test_test_metrics() {
        let metrics = test_metrics();
        assert_eq!(metrics.accuracy, 0.9);
        assert_eq!(metrics.latency_p50, 100.0);
    }
    
    #[test]
    fn test_test_latencies() {
        let latencies = test_latencies(10);
        assert_eq!(latencies.len(), 10);
        assert_eq!(latencies[0], 50.0);
    }
}












