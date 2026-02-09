//! Shared Types - Common type definitions and aliases.
//!
//! Provides:
//! - Type aliases for common types
//! - Result types
//! - Common structures

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ════════════════════════════════════════════════════════════════════════════════
// TYPE ALIASES
// ════════════════════════════════════════════════════════════════════════════════

/// Token ID type.
pub type TokenId = u32;

/// Token sequence.
pub type TokenSequence = Vec<TokenId>;

/// Batch of token sequences.
pub type TokenBatch = Vec<TokenSequence>;

/// Metadata map.
pub type Metadata = HashMap<String, String>;

/// Configuration map.
pub type ConfigMap = HashMap<String, String>;

// ════════════════════════════════════════════════════════════════════════════════
// COMMON STRUCTURES
// ════════════════════════════════════════════════════════════════════════════════

/// Version information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    pub version: String,
    pub name: String,
    pub description: String,
}

impl VersionInfo {
    /// Create new version info.
    pub fn new(version: String, name: String, description: String) -> Self {
        Self {
            version,
            name,
            description,
        }
    }
}

/// System information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub version: String,
    pub name: String,
    pub description: String,
    pub hostname: Option<String>,
    pub features: Vec<String>,
}

impl SystemInfo {
    /// Create new system info.
    pub fn new(
        version: String,
        name: String,
        description: String,
    ) -> Self {
        Self {
            version,
            name,
            description,
            hostname: std::env::var("HOSTNAME").ok(),
            features: Vec::new(),
        }
    }
    
    /// Add a feature.
    pub fn with_feature(mut self, feature: String) -> Self {
        self.features.push(feature);
        self
    }
}

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
    pub fn accuracy(mut self, value: f64) -> Self {
        self.metrics.accuracy = value;
        self
    }
    
    /// Set latency P50.
    pub fn latency_p50(mut self, value: f64) -> Self {
        self.metrics.latency_p50 = value;
        self
    }
    
    /// Set latency P95.
    pub fn latency_p95(mut self, value: f64) -> Self {
        self.metrics.latency_p95 = value;
        self
    }
    
    /// Set latency P99.
    pub fn latency_p99(mut self, value: f64) -> Self {
        self.metrics.latency_p99 = value;
        self
    }
    
    /// Set throughput.
    pub fn throughput(mut self, value: f64) -> Self {
        self.metrics.throughput = value;
        self
    }
    
    /// Set memory peak.
    pub fn memory_peak_mb(mut self, value: f64) -> Self {
        self.metrics.memory_peak_mb = value;
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
#[derive(Debug, Clone)]
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

/// Performance thresholds for evaluation.
#[derive(Debug, Clone)]
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
            min_throughput: 10.0,
        }
    }
}

/// Performance metrics summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_operations: usize,
    pub successful_operations: usize,
    pub failed_operations: usize,
    pub average_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput: f64,
}

impl Default for PerformanceSummary {
    fn default() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            average_latency_ms: 0.0,
            p50_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            throughput: 0.0,
        }
    }
}

impl PerformanceSummary {
    /// Calculate success rate.
    pub fn success_rate(&self) -> f64 {
        if self.total_operations == 0 {
            return 0.0;
        }
        self.successful_operations as f64 / self.total_operations as f64
    }
    
    /// Check if performance is acceptable.
    pub fn is_acceptable(&self, max_latency_ms: f64) -> bool {
        self.p95_latency_ms <= max_latency_ms && self.success_rate() >= 0.95
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_performance_summary() {
        let summary = PerformanceSummary {
            total_operations: 100,
            successful_operations: 95,
            failed_operations: 5,
            average_latency_ms: 100.0,
            p50_latency_ms: 95.0,
            p95_latency_ms: 150.0,
            p99_latency_ms: 200.0,
            throughput: 10.0,
        };
        
        assert_eq!(summary.success_rate(), 0.95);
        assert!(summary.is_acceptable(200.0));
        assert!(!summary.is_acceptable(100.0));
    }
}

