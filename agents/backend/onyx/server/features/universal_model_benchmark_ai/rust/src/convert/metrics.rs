//! Metrics Conversions
//!
//! Conversion functions for metrics types.

use crate::error::Result;
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
        return Err(crate::error::BenchmarkError::invalid_input("Latencies cannot be empty"));
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

#[cfg(test)]
mod tests {
    use super::*;
    
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




