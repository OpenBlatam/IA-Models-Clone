//! Reporting Types
//!
//! Core data structures for reports.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::Metrics;
use crate::error::Result;

/// Comprehensive benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub report_id: String,
    pub timestamp: String,
    pub model_name: String,
    pub benchmark_name: String,
    pub metrics: Metrics,
    pub config: HashMap<String, String>,
    pub samples: ReportSamples,
    pub performance: PerformanceBreakdown,
}

impl BenchmarkReport {
    /// Create a new report.
    pub fn new(
        model_name: String,
        benchmark_name: String,
        metrics: Metrics,
    ) -> Self {
        Self {
            report_id: format!("{}_{}_{}", model_name, benchmark_name, 
                chrono::Utc::now().timestamp()),
            timestamp: chrono::Utc::now().to_rfc3339(),
            model_name,
            benchmark_name,
            metrics,
            config: HashMap::new(),
            samples: ReportSamples {
                total: 0,
                correct: 0,
                incorrect: 0,
                accuracy: 0.0,
            },
            performance: PerformanceBreakdown {
                total_time_seconds: 0.0,
                inference_time_seconds: 0.0,
                evaluation_time_seconds: 0.0,
                overhead_time_seconds: 0.0,
                tokens_per_second: 0.0,
                memory_peak_mb: 0.0,
            },
        }
    }
    
    /// Generate summary string.
    pub fn summary(&self) -> String {
        format!(
            "Benchmark: {}\nModel: {}\nAccuracy: {:.2}%\nLatency P50: {:.3}s\nThroughput: {:.2} tok/s",
            self.benchmark_name,
            self.model_name,
            self.metrics.accuracy * 100.0,
            self.metrics.latency_p50,
            self.metrics.throughput
        )
    }
    
    /// Export to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::error::BenchmarkError::serialization(e.to_string()))
    }
    
    /// Set configuration.
    pub fn with_config(mut self, config: HashMap<String, String>) -> Self {
        self.config = config;
        self
    }
    
    /// Set samples.
    pub fn with_samples(mut self, samples: ReportSamples) -> Self {
        self.samples = samples;
        self
    }
    
    /// Set performance breakdown.
    pub fn with_performance(mut self, performance: PerformanceBreakdown) -> Self {
        self.performance = performance;
        self
    }
}

/// Report samples information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSamples {
    pub total: usize,
    pub correct: usize,
    pub incorrect: usize,
    pub accuracy: f64,
}

impl ReportSamples {
    /// Create from counts.
    pub fn from_counts(correct: usize, total: usize) -> Self {
        let accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };
        
        Self {
            total,
            correct,
            incorrect: total - correct,
            accuracy,
        }
    }
    
    /// Calculate accuracy.
    pub fn calculate_accuracy(&self) -> f64 {
        if self.total > 0 {
            self.correct as f64 / self.total as f64
        } else {
            0.0
        }
    }
}

/// Performance breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBreakdown {
    pub total_time_seconds: f64,
    pub inference_time_seconds: f64,
    pub evaluation_time_seconds: f64,
    pub overhead_time_seconds: f64,
    pub tokens_per_second: f64,
    pub memory_peak_mb: f64,
}

impl PerformanceBreakdown {
    /// Calculate overhead percentage.
    pub fn overhead_percentage(&self) -> f64 {
        if self.total_time_seconds > 0.0 {
            self.overhead_time_seconds / self.total_time_seconds * 100.0
        } else {
            0.0
        }
    }
    
    /// Calculate inference percentage.
    pub fn inference_percentage(&self) -> f64 {
        if self.total_time_seconds > 0.0 {
            self.inference_time_seconds / self.total_time_seconds * 100.0
        } else {
            0.0
        }
    }
}

/// Model comparison data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    pub model_name: String,
    pub metrics: Metrics,
    pub rank: usize,
    pub score: f64, // Composite score
}

impl ModelComparison {
    /// Calculate composite score.
    pub fn calculate_score(metrics: &Metrics, weights: Option<(f64, f64, f64)>) -> f64 {
        let (w_acc, w_lat, w_thr) = weights.unwrap_or((0.5, 0.3, 0.2));
        
        let accuracy_score = metrics.accuracy * w_acc;
        let latency_score = (1.0 / (metrics.latency_p50 + 0.001)) * w_lat;
        let throughput_score = metrics.throughput / 1000.0 * w_thr;
        
        accuracy_score + latency_score + throughput_score
    }
    
    /// Create with calculated score.
    pub fn new(model_name: String, metrics: Metrics, rank: usize) -> Self {
        let score = Self::calculate_score(&metrics, None);
        Self {
            model_name,
            metrics,
            rank,
            score,
        }
    }
}

/// Comparison report between multiple models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub benchmark_name: String,
    pub timestamp: String,
    pub models: Vec<ModelComparison>,
    pub best_model: String,
    pub rankings: HashMap<String, usize>,
}

impl ComparisonReport {
    /// Create a new comparison report.
    pub fn new(benchmark_name: String, reports: Vec<BenchmarkReport>) -> Self {
        let mut models: Vec<ModelComparison> = reports
            .into_iter()
            .enumerate()
            .map(|(idx, r)| {
                ModelComparison::new(r.model_name, r.metrics, idx + 1)
            })
            .collect();
        
        // Sort by score (descending)
        models.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        // Assign ranks
        let mut rankings = HashMap::new();
        for (idx, model) in models.iter_mut().enumerate() {
            model.rank = idx + 1;
            rankings.insert(model.model_name.clone(), idx + 1);
        }
        
        let best_model = models.first()
            .map(|m| m.model_name.clone())
            .unwrap_or_default();
        
        Self {
            benchmark_name,
            timestamp: chrono::Utc::now().to_rfc3339(),
            models,
            best_model,
            rankings,
        }
    }
    
    /// Export to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::error::BenchmarkError::serialization(e.to_string()))
    }
    
    /// Get model by rank.
    pub fn get_model_by_rank(&self, rank: usize) -> Option<&ModelComparison> {
        self.models.iter().find(|m| m.rank == rank)
    }
    
    /// Get model by name.
    pub fn get_model_by_name(&self, name: &str) -> Option<&ModelComparison> {
        self.models.iter().find(|m| m.model_name == name)
    }
}




