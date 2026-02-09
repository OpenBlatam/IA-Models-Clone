//! Metrics Calculation
//!
//! Functions for calculating various performance metrics.

use crate::Metrics;
use crate::error::Result;
use crate::utils::statistics::{percentile, mean};

/// Calculate benchmark metrics from raw data.
pub fn calculate_metrics(
    latencies: &[f64],
    accuracies: &[bool],
    total_tokens: usize,
    total_time: f64,
) -> Metrics {
    if latencies.is_empty() {
        return Metrics {
            accuracy: 0.0,
            latency_p50: 0.0,
            latency_p95: 0.0,
            latency_p99: 0.0,
            throughput: 0.0,
            memory_peak_mb: 0.0,
        };
    }
    
    // Calculate accuracy
    let accuracy = calculate_accuracy(accuracies);
    
    // Calculate latency percentiles
    let mut sorted_latencies = latencies.to_vec();
    sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let latency_p50 = percentile(&sorted_latencies, 0.50);
    let latency_p95 = percentile(&sorted_latencies, 0.95);
    let latency_p99 = percentile(&sorted_latencies, 0.99);
    
    // Calculate throughput
    let throughput = calculate_throughput(total_tokens, total_time);
    
    Metrics {
        accuracy,
        latency_p50,
        latency_p95,
        latency_p99,
        throughput,
        memory_peak_mb: 0.0, // Would be measured separately
    }
}

/// Calculate accuracy from boolean results.
pub fn calculate_accuracy(results: &[bool]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    
    let correct = results.iter().filter(|&&x| x).count();
    correct as f64 / results.len() as f64
}

/// Calculate accuracy from correct/total counts.
pub fn calculate_accuracy_from_counts(correct: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    correct as f64 / total as f64
}

/// Calculate throughput (tokens per second).
pub fn calculate_throughput(total_tokens: usize, total_time_seconds: f64) -> f64 {
    if total_time_seconds <= 0.0 {
        return 0.0;
    }
    total_tokens as f64 / total_time_seconds
}

/// Calculate throughput from requests and time.
pub fn calculate_throughput_requests(requests: usize, total_time_seconds: f64) -> f64 {
    if total_time_seconds <= 0.0 {
        return 0.0;
    }
    requests as f64 / total_time_seconds
}

/// Calculate latency statistics.
pub fn calculate_latency_stats(latencies: &[f64]) -> Result<(f64, f64, f64, f64)> {
    if latencies.is_empty() {
        return Err(crate::error::BenchmarkError::invalid_input("Empty latencies"));
    }
    
    let mut sorted = latencies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mean_val = mean(&sorted);
    let p50 = percentile(&sorted, 0.50);
    let p95 = percentile(&sorted, 0.95);
    let p99 = percentile(&sorted, 0.99);
    
    Ok((mean_val, p50, p95, p99))
}

/// Calculate latency percentiles.
pub fn calculate_latency_percentiles(latencies: &[f64], percentiles: &[f64]) -> Result<Vec<f64>> {
    if latencies.is_empty() {
        return Err(crate::error::BenchmarkError::invalid_input("Empty latencies"));
    }
    
    let mut sorted = latencies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    Ok(percentiles.iter()
        .map(|&p| percentile(&sorted, p))
        .collect())
}

/// Calculate memory efficiency (tokens per MB).
pub fn calculate_memory_efficiency(tokens: usize, memory_mb: f64) -> f64 {
    if memory_mb <= 0.0 {
        return 0.0;
    }
    tokens as f64 / memory_mb
}

/// Calculate cost efficiency (tokens per dollar).
pub fn calculate_cost_efficiency(tokens: usize, cost: f64) -> f64 {
    if cost <= 0.0 {
        return 0.0;
    }
    tokens as f64 / cost
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calculate_accuracy() {
        let results = vec![true, true, false, true, false];
        assert_eq!(calculate_accuracy(&results), 0.6);
    }
    
    #[test]
    fn test_calculate_throughput() {
        assert_eq!(calculate_throughput(1000, 1.0), 1000.0);
        assert_eq!(calculate_throughput(1000, 2.0), 500.0);
    }
    
    #[test]
    fn test_calculate_latency_stats() {
        let latencies = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, p50, p95, p99) = calculate_latency_stats(&latencies).unwrap();
        assert_eq!(mean, 3.0);
        assert!(p50 > 0.0);
    }
}
