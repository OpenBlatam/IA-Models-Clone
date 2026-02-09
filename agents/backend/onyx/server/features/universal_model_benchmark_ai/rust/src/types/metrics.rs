//! Metrics Types
//!
//! Metrics structures and builders.

use serde::{Deserialize, Serialize};

/// Benchmark metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub accuracy: f64,
    pub latency_p50: f64,
    pub latency_p95: f64,
    pub latency_p99: f64,
    pub throughput: f64,
    pub memory_peak_mb: f64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            accuracy: 0.0,
            latency_p50: 0.0,
            latency_p95: 0.0,
            latency_p99: 0.0,
            throughput: 0.0,
            memory_peak_mb: 0.0,
        }
    }
}

impl Metrics {
    /// Create a new metrics instance.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create metrics with builder pattern.
    pub fn builder() -> MetricsBuilder {
        MetricsBuilder::new()
    }
    
    /// Calculate a composite score (weighted combination of metrics).
    pub fn composite_score(&self, weights: Option<MetricsWeights>) -> f64 {
        let w = weights.unwrap_or_default();
        w.accuracy_weight * self.accuracy +
        w.latency_weight * (1.0 / (self.latency_p50 + 0.001)) +
        w.throughput_weight * (self.throughput / 1000.0)
    }
    
    /// Check if metrics indicate good performance.
    pub fn is_good_performance(&self, thresholds: Option<PerformanceThresholds>) -> bool {
        let t = thresholds.unwrap_or_default();
        self.accuracy >= t.min_accuracy &&
        self.latency_p50 <= t.max_latency_p50 &&
        self.throughput >= t.min_throughput
    }
    
    /// Calculate improvement percentage compared to another metrics.
    pub fn improvement_percentage(&self, other: &Metrics) -> f64 {
        if other.accuracy == 0.0 {
            return 0.0;
        }
        ((self.accuracy - other.accuracy) / other.accuracy) * 100.0
    }
    
    /// Check if these metrics are better than another set.
    pub fn is_better_than(&self, other: &Metrics) -> bool {
        self.accuracy > other.accuracy &&
        self.latency_p50 < other.latency_p50 &&
        self.throughput > other.throughput
    }
    
    /// Get normalized metrics (0-1 scale).
    pub fn normalized(&self) -> Metrics {
        let max_throughput = 1000.0; // Normalize to reasonable max
        Metrics {
            accuracy: self.accuracy.min(1.0).max(0.0),
            latency_p50: (self.latency_p50 / 10.0).min(1.0).max(0.0),
            latency_p95: (self.latency_p95 / 10.0).min(1.0).max(0.0),
            latency_p99: (self.latency_p99 / 10.0).min(1.0).max(0.0),
            throughput: (self.throughput / max_throughput).min(1.0).max(0.0),
            memory_peak_mb: self.memory_peak_mb,
        }
    }
}

/// Builder for Metrics.
pub struct MetricsBuilder {
    metrics: Metrics,
}

impl MetricsBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            metrics: Metrics::default(),
        }
    }
    
    /// Set accuracy.
    pub fn accuracy(mut self, acc: f64) -> Self {
        self.metrics.accuracy = acc;
        self
    }
    
    /// Set latency P50.
    pub fn latency_p50(mut self, lat: f64) -> Self {
        self.metrics.latency_p50 = lat;
        self
    }
    
    /// Set latency P95.
    pub fn latency_p95(mut self, lat: f64) -> Self {
        self.metrics.latency_p95 = lat;
        self
    }
    
    /// Set latency P99.
    pub fn latency_p99(mut self, lat: f64) -> Self {
        self.metrics.latency_p99 = lat;
        self
    }
    
    /// Set throughput.
    pub fn throughput(mut self, thr: f64) -> Self {
        self.metrics.throughput = thr;
        self
    }
    
    /// Set memory peak.
    pub fn memory_peak_mb(mut self, mem: f64) -> Self {
        self.metrics.memory_peak_mb = mem;
        self
    }
    
    /// Build the metrics.
    pub fn build(self) -> Metrics {
        self.metrics
    }
}

impl Default for MetricsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Weights for composite score calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsWeights {
    pub accuracy_weight: f64,
    pub latency_weight: f64,
    pub throughput_weight: f64,
}

impl Default for MetricsWeights {
    fn default() -> Self {
        Self {
            accuracy_weight: 0.5,
            latency_weight: 0.3,
            throughput_weight: 0.2,
        }
    }
}

impl MetricsWeights {
    /// Create with custom weights.
    pub fn new(accuracy: f64, latency: f64, throughput: f64) -> Self {
        Self {
            accuracy_weight: accuracy,
            latency_weight: latency,
            throughput_weight: throughput,
        }
    }
    
    /// Normalize weights to sum to 1.0.
    pub fn normalized(self) -> Self {
        let total = self.accuracy_weight + self.latency_weight + self.throughput_weight;
        if total > 0.0 {
            Self {
                accuracy_weight: self.accuracy_weight / total,
                latency_weight: self.latency_weight / total,
                throughput_weight: self.throughput_weight / total,
            }
        } else {
            Self::default()
        }
    }
}

/// Performance thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub min_accuracy: f64,
    pub max_latency_p50: f64,
    pub min_throughput: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            min_accuracy: 0.8,
            max_latency_p50: 1.0,
            min_throughput: 100.0,
        }
    }
}

impl PerformanceThresholds {
    /// Create with custom thresholds.
    pub fn new(min_accuracy: f64, max_latency_p50: f64, min_throughput: f64) -> Self {
        Self {
            min_accuracy,
            max_latency_p50,
            min_throughput,
        }
    }
}

/// Performance summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub metrics: Metrics,
    pub composite_score: f64,
    pub meets_thresholds: bool,
    pub improvement_vs_baseline: Option<f64>,
}

impl PerformanceSummary {
    /// Create from metrics.
    pub fn from_metrics(metrics: Metrics, baseline: Option<&Metrics>) -> Self {
        let composite_score = metrics.composite_score(None);
        let meets_thresholds = metrics.is_good_performance(None);
        let improvement_vs_baseline = baseline.map(|b| metrics.improvement_percentage(b));
        
        Self {
            metrics,
            composite_score,
            meets_thresholds,
            improvement_vs_baseline,
        }
    }
}




