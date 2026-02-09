//! Metrics Aggregation
//!
//! Functions for aggregating and analyzing multiple metrics.

use std::collections::HashMap;
use crate::error::Result;
use crate::Metrics;
use crate::utils::statistics::{calculate_statistics as calc_stats, percentile};

/// Calculate statistics for a set of values.
pub fn calculate_statistics(values: &[f64]) -> HashMap<String, f64> {
    if values.is_empty() {
        return HashMap::new();
    }
    
    calc_stats(values)
}

/// Aggregate metrics from multiple runs.
pub fn aggregate_metrics(metrics_list: &[Metrics]) -> Result<HashMap<String, f64>> {
    if metrics_list.is_empty() {
        return Err(crate::error::BenchmarkError::invalid_input("Empty metrics list"));
    }
    
    let mut aggregated = HashMap::new();
    
    // Collect all values
    let accuracies: Vec<f64> = metrics_list.iter().map(|m| m.accuracy).collect();
    let latencies_p50: Vec<f64> = metrics_list.iter().map(|m| m.latency_p50).collect();
    let latencies_p95: Vec<f64> = metrics_list.iter().map(|m| m.latency_p95).collect();
    let latencies_p99: Vec<f64> = metrics_list.iter().map(|m| m.latency_p99).collect();
    let throughputs: Vec<f64> = metrics_list.iter().map(|m| m.throughput).collect();
    let memory_peaks: Vec<f64> = metrics_list.iter().map(|m| m.memory_peak_mb).collect();
    
    // Calculate statistics for each metric
    let accuracy_stats = calculate_statistics(&accuracies);
    let p50_stats = calculate_statistics(&latencies_p50);
    let p95_stats = calculate_statistics(&latencies_p95);
    let p99_stats = calculate_statistics(&latencies_p99);
    let throughput_stats = calculate_statistics(&throughputs);
    let memory_stats = calculate_statistics(&memory_peaks);
    
    // Aggregate into result
    for (key, value) in accuracy_stats.iter() {
        aggregated.insert(format!("accuracy_{}", key), *value);
    }
    for (key, value) in p50_stats.iter() {
        aggregated.insert(format!("latency_p50_{}", key), *value);
    }
    for (key, value) in p95_stats.iter() {
        aggregated.insert(format!("latency_p95_{}", key), *value);
    }
    for (key, value) in p99_stats.iter() {
        aggregated.insert(format!("latency_p99_{}", key), *value);
    }
    for (key, value) in throughput_stats.iter() {
        aggregated.insert(format!("throughput_{}", key), *value);
    }
    for (key, value) in memory_stats.iter() {
        aggregated.insert(format!("memory_peak_{}", key), *value);
    }
    
    Ok(aggregated)
}

/// Calculate weighted average of metrics.
pub fn weighted_average_metrics(
    metrics_list: &[Metrics],
    weights: &[f64],
) -> Result<Metrics> {
    if metrics_list.is_empty() {
        return Err(crate::error::BenchmarkError::invalid_input("Empty metrics list"));
    }
    
    if metrics_list.len() != weights.len() {
        return Err(crate::error::BenchmarkError::invalid_input(
            "Metrics and weights must have same length"
        ));
    }
    
    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        return Err(crate::error::BenchmarkError::invalid_input(
            "Total weight must be positive"
        ));
    }
    
    let mut weighted_accuracy = 0.0;
    let mut weighted_p50 = 0.0;
    let mut weighted_p95 = 0.0;
    let mut weighted_p99 = 0.0;
    let mut weighted_throughput = 0.0;
    let mut weighted_memory = 0.0;
    
    for (metric, &weight) in metrics_list.iter().zip(weights.iter()) {
        let normalized_weight = weight / total_weight;
        weighted_accuracy += metric.accuracy * normalized_weight;
        weighted_p50 += metric.latency_p50 * normalized_weight;
        weighted_p95 += metric.latency_p95 * normalized_weight;
        weighted_p99 += metric.latency_p99 * normalized_weight;
        weighted_throughput += metric.throughput * normalized_weight;
        weighted_memory += metric.memory_peak_mb * normalized_weight;
    }
    
    Ok(Metrics {
        accuracy: weighted_accuracy,
        latency_p50: weighted_p50,
        latency_p95: weighted_p95,
        latency_p99: weighted_p99,
        throughput: weighted_throughput,
        memory_peak_mb: weighted_memory,
    })
}

/// Calculate median metrics from multiple runs.
pub fn median_metrics(metrics_list: &[Metrics]) -> Result<Metrics> {
    if metrics_list.is_empty() {
        return Err(crate::error::BenchmarkError::invalid_input("Empty metrics list"));
    }
    
    let accuracies: Vec<f64> = metrics_list.iter().map(|m| m.accuracy).collect();
    let latencies_p50: Vec<f64> = metrics_list.iter().map(|m| m.latency_p50).collect();
    let latencies_p95: Vec<f64> = metrics_list.iter().map(|m| m.latency_p95).collect();
    let latencies_p99: Vec<f64> = metrics_list.iter().map(|m| m.latency_p99).collect();
    let throughputs: Vec<f64> = metrics_list.iter().map(|m| m.throughput).collect();
    let memory_peaks: Vec<f64> = metrics_list.iter().map(|m| m.memory_peak_mb).collect();
    
    let mut sorted_acc = accuracies.clone();
    sorted_acc.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut sorted_p50 = latencies_p50.clone();
    sorted_p50.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut sorted_p95 = latencies_p95.clone();
    sorted_p95.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut sorted_p99 = latencies_p99.clone();
    sorted_p99.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut sorted_throughput = throughputs.clone();
    sorted_throughput.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut sorted_memory = memory_peaks.clone();
    sorted_memory.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mid = sorted_acc.len() / 2;
    let median_val = |sorted: &[f64]| {
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    };
    
    Ok(Metrics {
        accuracy: median_val(&sorted_acc),
        latency_p50: median_val(&sorted_p50),
        latency_p95: median_val(&sorted_p95),
        latency_p99: median_val(&sorted_p99),
        throughput: median_val(&sorted_throughput),
        memory_peak_mb: median_val(&sorted_memory),
    })
}

/// Compare two metrics and return differences.
pub fn compare_metrics(base: &Metrics, other: &Metrics) -> HashMap<String, f64> {
    let mut diff = HashMap::new();
    
    diff.insert("accuracy_diff".to_string(), other.accuracy - base.accuracy);
    diff.insert("latency_p50_diff".to_string(), other.latency_p50 - base.latency_p50);
    diff.insert("latency_p95_diff".to_string(), other.latency_p95 - base.latency_p95);
    diff.insert("latency_p99_diff".to_string(), other.latency_p99 - base.latency_p99);
    diff.insert("throughput_diff".to_string(), other.throughput - base.throughput);
    diff.insert("memory_peak_diff".to_string(), other.memory_peak_mb - base.memory_peak_mb);
    
    // Relative differences
    if base.accuracy > 0.0 {
        diff.insert("accuracy_relative".to_string(), (other.accuracy - base.accuracy) / base.accuracy);
    }
    if base.latency_p50 > 0.0 {
        diff.insert("latency_p50_relative".to_string(), (other.latency_p50 - base.latency_p50) / base.latency_p50);
    }
    if base.throughput > 0.0 {
        diff.insert("throughput_relative".to_string(), (other.throughput - base.throughput) / base.throughput);
    }
    
    diff
}
