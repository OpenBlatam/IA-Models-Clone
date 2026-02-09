//! Reporting and visualization utilities.
//!
//! Provides:
//! - Report generation
//! - Data export
//! - Visualization data preparation
//! - Comparison reports

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

/// Report samples information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSamples {
    pub total: usize,
    pub correct: usize,
    pub incorrect: usize,
    pub accuracy: f64,
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

/// Model comparison data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    pub model_name: String,
    pub metrics: Metrics,
    pub rank: usize,
    pub score: f64, // Composite score
}

impl ComparisonReport {
    /// Create a new comparison report.
    pub fn new(benchmark_name: String, reports: Vec<BenchmarkReport>) -> Self {
        let mut models: Vec<ModelComparison> = reports
            .into_iter()
            .map(|r| {
                // Calculate composite score (weighted)
                let score = r.metrics.accuracy * 0.5 +
                           (1.0 / (r.metrics.latency_p50 + 0.001)) * 0.3 +
                           r.metrics.throughput / 1000.0 * 0.2;
                
                ModelComparison {
                    model_name: r.model_name.clone(),
                    metrics: r.metrics,
                    rank: 0, // Will be set after sorting
                    score,
                }
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
    
    /// Generate markdown report.
    pub fn to_markdown(&self) -> String {
        let mut md = format!("# Benchmark Comparison: {}\n\n", self.benchmark_name);
        md.push_str(&format!("Generated: {}\n\n", self.timestamp));
        md.push_str(&format!("Best Model: **{}**\n\n", self.best_model));
        
        md.push_str("## Results\n\n");
        md.push_str("| Rank | Model | Accuracy | Latency P50 | Throughput | Score |\n");
        md.push_str("|------|-------|----------|-------------|------------|-------|\n");
        
        for model in &self.models {
            md.push_str(&format!(
                "| {} | {} | {:.2}% | {:.3}s | {:.2} tok/s | {:.3} |\n",
                model.rank,
                model.model_name,
                model.metrics.accuracy * 100.0,
                model.metrics.latency_p50,
                model.metrics.throughput,
                model.score
            ));
        }
        
        md
    }
}

/// Report generator.
pub struct ReportGenerator;

impl ReportGenerator {
    /// Generate single model report.
    pub fn generate_report(
        model_name: &str,
        benchmark_name: &str,
        metrics: &Metrics,
    ) -> BenchmarkReport {
        BenchmarkReport::new(
            model_name.to_string(),
            benchmark_name.to_string(),
            metrics.clone(),
        )
    }
    
    /// Generate comparison report.
    pub fn generate_comparison(
        benchmark_name: &str,
        reports: Vec<BenchmarkReport>,
    ) -> ComparisonReport {
        ComparisonReport::new(benchmark_name.to_string(), reports)
    }
    
    /// Export reports to file.
    pub fn export_reports(
        reports: &[BenchmarkReport],
        output_path: &str,
    ) -> Result<()> {
        use std::fs::File;
        use std::io::Write;
        
        let json = serde_json::to_string_pretty(reports)
            .map_err(|e| crate::error::BenchmarkError::serialization(e.to_string()))?;
        
        let mut file = File::create(output_path)
            .map_err(|e| crate::error::BenchmarkError::io(e))?;
        
        file.write_all(json.as_bytes())
            .map_err(|e| crate::error::BenchmarkError::io(e))?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_report_generation() {
        let metrics = Metrics {
            accuracy: 0.85,
            latency_p50: 0.1,
            latency_p95: 0.2,
            latency_p99: 0.3,
            throughput: 100.0,
            memory_peak_mb: 1000.0,
        };
        
        let report = ReportGenerator::generate_report("test-model", "test-benchmark", &metrics);
        assert_eq!(report.model_name, "test-model");
        assert_eq!(report.benchmark_name, "test-benchmark");
    }
}












