//! Helper functions for common operations.
//!
//! Provides convenient functions that combine multiple operations.

use crate::error::Result;
use crate::types::Metrics;
use crate::benchmark::runner::BenchmarkResult;
use crate::constants::thresholds;

/// Check if metrics meet good performance criteria.
pub fn is_good_performance(metrics: &Metrics) -> bool {
    metrics.accuracy >= thresholds::GOOD_ACCURACY &&
    metrics.latency_p50 <= thresholds::MAX_LATENCY_P50 &&
    metrics.throughput >= thresholds::MIN_THROUGHPUT
}

/// Check if metrics meet excellent performance criteria.
pub fn is_excellent_performance(metrics: &Metrics) -> bool {
    metrics.accuracy >= thresholds::EXCELLENT_ACCURACY &&
    metrics.latency_p50 <= thresholds::MAX_LATENCY_P50 &&
    metrics.throughput >= thresholds::GOOD_THROUGHPUT
}

/// Get performance rating as string.
pub fn performance_rating(metrics: &Metrics) -> &'static str {
    if is_excellent_performance(metrics) {
        "excellent"
    } else if is_good_performance(metrics) {
        "good"
    } else if metrics.accuracy >= thresholds::MIN_ACCURACY {
        "acceptable"
    } else {
        "poor"
    }
}

/// Calculate improvement percentage between two metrics.
pub fn improvement_percentage(before: &Metrics, after: &Metrics) -> f64 {
    if before.accuracy == 0.0 {
        return 0.0;
    }
    ((after.accuracy - before.accuracy) / before.accuracy) * 100.0
}

/// Check if benchmark result indicates success.
pub fn is_benchmark_successful(result: &BenchmarkResult) -> bool {
    result.is_successful() && result.error_count() == 0
}

/// Get summary string for benchmark result.
pub fn benchmark_summary(result: &BenchmarkResult) -> String {
    format!(
        "iterations={}, avg_latency={:.2}ms, throughput={:.2}, success_rate={:.1}%",
        result.iterations,
        result.avg_latency_ms,
        result.throughput,
        result.success_rate * 100.0
    )
}

/// Get detailed summary string for metrics.
pub fn metrics_summary(metrics: &Metrics) -> String {
    format!(
        "accuracy={:.2}%, latency_p50={:.2}ms, latency_p95={:.2}ms, throughput={:.2}, rating={}",
        metrics.accuracy * 100.0,
        metrics.latency_p50,
        metrics.latency_p95,
        metrics.throughput,
        performance_rating(metrics)
    )
}

/// Compare two metrics and return comparison result.
pub fn compare_metrics(baseline: &Metrics, candidate: &Metrics) -> String {
    let accuracy_diff = candidate.accuracy - baseline.accuracy;
    let latency_diff = candidate.latency_p50 - baseline.latency_p50;
    let throughput_diff = candidate.throughput - baseline.throughput;
    
    let mut parts = Vec::new();
    
    if accuracy_diff > 0.0 {
        parts.push(format!("accuracy +{:.2}%", accuracy_diff * 100.0));
    } else if accuracy_diff < 0.0 {
        parts.push(format!("accuracy {:.2}%", accuracy_diff * 100.0));
    }
    
    if latency_diff < 0.0 {
        parts.push(format!("latency {:.2}ms faster", -latency_diff));
    } else if latency_diff > 0.0 {
        parts.push(format!("latency +{:.2}ms slower", latency_diff));
    }
    
    if throughput_diff > 0.0 {
        parts.push(format!("throughput +{:.2}", throughput_diff));
    } else if throughput_diff < 0.0 {
        parts.push(format!("throughput {:.2}", throughput_diff));
    }
    
    if parts.is_empty() {
        "no significant difference".to_string()
    } else {
        parts.join(", ")
    }
}

/// Validate that metrics are within acceptable ranges.
pub fn validate_metrics(metrics: &Metrics) -> Result<()> {
    if metrics.accuracy < 0.0 || metrics.accuracy > 1.0 {
        return Err(crate::error::BenchmarkError::invalid_input(
            format!("Accuracy must be between 0.0 and 1.0, got {}", metrics.accuracy)
        ));
    }
    
    if metrics.latency_p50 < 0.0 {
        return Err(crate::error::BenchmarkError::invalid_input(
            format!("Latency P50 must be non-negative, got {}", metrics.latency_p50)
        ));
    }
    
    if metrics.throughput < 0.0 {
        return Err(crate::error::BenchmarkError::invalid_input(
            format!("Throughput must be non-negative, got {}", metrics.throughput)
        ));
    }
    
    Ok(())
}

/// Normalize metrics to [0, 1] range for comparison.
pub fn normalize_metrics(metrics: &Metrics) -> Metrics {
    Metrics::builder()
        .accuracy(metrics.accuracy.clamp(0.0, 1.0))
        .latency_p50(metrics.latency_p50 / 1000.0) // Normalize to seconds
        .latency_p95(metrics.latency_p95 / 1000.0)
        .latency_p99(metrics.latency_p99 / 1000.0)
        .throughput(metrics.throughput / 1000.0) // Normalize to k tokens/sec
        .memory_peak_mb(metrics.memory_peak_mb / 1000.0) // Normalize to GB
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_is_good_performance() {
        let metrics = Metrics::builder()
            .accuracy(0.95)
            .latency_p50(50.0)
            .throughput(100.0)
            .build();
        
        assert!(is_good_performance(&metrics));
    }
    
    #[test]
    fn test_performance_rating() {
        let metrics = Metrics::builder()
            .accuracy(0.95)
            .latency_p50(50.0)
            .throughput(100.0)
            .build();
        
        assert_eq!(performance_rating(&metrics), "excellent");
    }
    
    #[test]
    fn test_compare_metrics() {
        let baseline = Metrics::builder()
            .accuracy(0.9)
            .latency_p50(100.0)
            .throughput(50.0)
            .build();
        
        let candidate = Metrics::builder()
            .accuracy(0.95)
            .latency_p50(80.0)
            .throughput(60.0)
            .build();
        
        let comparison = compare_metrics(&baseline, &candidate);
        assert!(comparison.contains("accuracy"));
        assert!(comparison.contains("faster"));
    }
}












