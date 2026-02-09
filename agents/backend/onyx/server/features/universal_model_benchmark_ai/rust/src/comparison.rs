//! Benchmark comparison utilities.
//!
//! Provides tools for comparing multiple benchmarks, models, or configurations.

use crate::error::Result;
use crate::types::Metrics;
use crate::config::BenchmarkConfig;
use crate::benchmark::runner::BenchmarkResult;
use crate::helpers::compare_metrics;
use crate::analysis::{StatisticalSummary, compare_summaries};

/// Comparison between two benchmark results.
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    pub name_a: String,
    pub name_b: String,
    pub metrics_a: Metrics,
    pub metrics_b: Metrics,
    pub accuracy_diff: f64,
    pub accuracy_diff_pct: f64,
    pub latency_diff: f64,
    pub latency_diff_pct: f64,
    pub throughput_diff: f64,
    pub throughput_diff_pct: f64,
    pub winner: ComparisonWinner,
}

/// Winner of a comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonWinner {
    A,
    B,
    Tie,
}

impl BenchmarkComparison {
    /// Create a new comparison.
    pub fn new(name_a: String, metrics_a: Metrics, name_b: String, metrics_b: Metrics) -> Self {
        let accuracy_diff = metrics_b.accuracy - metrics_a.accuracy;
        let accuracy_diff_pct = if metrics_a.accuracy > 0.0 {
            (accuracy_diff / metrics_a.accuracy) * 100.0
        } else {
            0.0
        };
        
        let latency_diff = metrics_b.latency_p50 - metrics_a.latency_p50;
        let latency_diff_pct = if metrics_a.latency_p50 > 0.0 {
            (latency_diff / metrics_a.latency_p50) * 100.0
        } else {
            0.0
        };
        
        let throughput_diff = metrics_b.throughput - metrics_a.throughput;
        let throughput_diff_pct = if metrics_a.throughput > 0.0 {
            (throughput_diff / metrics_a.throughput) * 100.0
        } else {
            0.0
        };
        
        // Determine winner based on composite score
        let score_a = metrics_a.composite_score(None);
        let score_b = metrics_b.composite_score(None);
        
        let winner = if score_b > score_a * 1.05 {
            ComparisonWinner::B
        } else if score_a > score_b * 1.05 {
            ComparisonWinner::A
        } else {
            ComparisonWinner::Tie
        };
        
        Self {
            name_a,
            name_b,
            metrics_a,
            metrics_b,
            accuracy_diff,
            accuracy_diff_pct,
            latency_diff,
            latency_diff_pct,
            throughput_diff,
            throughput_diff_pct,
            winner,
        }
    }
    
    /// Get a summary string of the comparison.
    pub fn summary(&self) -> String {
        match self.winner {
            ComparisonWinner::A => format!(
                "{} wins: accuracy {:.2}%, latency {:.2}ms, throughput {:.2}",
                self.name_a,
                self.accuracy_diff_pct,
                -self.latency_diff,
                self.throughput_diff
            ),
            ComparisonWinner::B => format!(
                "{} wins: accuracy {:.2}%, latency {:.2}ms, throughput {:.2}",
                self.name_b,
                -self.accuracy_diff_pct,
                self.latency_diff,
                -self.throughput_diff
            ),
            ComparisonWinner::Tie => "Results are very similar (tie)".to_string(),
        }
    }
    
    /// Check if A is better than B.
    pub fn a_is_better(&self) -> bool {
        self.winner == ComparisonWinner::A
    }
    
    /// Check if B is better than A.
    pub fn b_is_better(&self) -> bool {
        self.winner == ComparisonWinner::B
    }
}

/// Compare multiple benchmarks.
pub struct BenchmarkComparator {
    results: Vec<(String, Metrics)>,
}

impl BenchmarkComparator {
    /// Create a new comparator.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }
    
    /// Add a benchmark result.
    pub fn add(&mut self, name: String, metrics: Metrics) {
        self.results.push((name, metrics));
    }
    
    /// Compare all benchmarks and return rankings.
    pub fn compare_all(&self) -> Vec<BenchmarkRanking> {
        let mut rankings: Vec<BenchmarkRanking> = self.results
            .iter()
            .enumerate()
            .map(|(idx, (name, metrics))| {
                let score = metrics.composite_score(None);
                BenchmarkRanking {
                    rank: 0, // Will be set later
                    name: name.clone(),
                    metrics: metrics.clone(),
                    score,
                    index: idx,
                }
            })
            .collect();
        
        // Sort by score (descending)
        rankings.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Assign ranks
        for (rank, ranking) in rankings.iter_mut().enumerate() {
            ranking.rank = rank + 1;
        }
        
        rankings
    }
    
    /// Get the best benchmark.
    pub fn best(&self) -> Option<&(String, Metrics)> {
        self.results
            .iter()
            .max_by(|a, b| {
                a.1.composite_score(None)
                    .partial_cmp(&b.1.composite_score(None))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
    
    /// Get pairwise comparisons.
    pub fn pairwise_comparisons(&self) -> Vec<BenchmarkComparison> {
        let mut comparisons = Vec::new();
        
        for i in 0..self.results.len() {
            for j in (i + 1)..self.results.len() {
                let (name_a, metrics_a) = &self.results[i];
                let (name_b, metrics_b) = &self.results[j];
                
                comparisons.push(BenchmarkComparison::new(
                    name_a.clone(),
                    metrics_a.clone(),
                    name_b.clone(),
                    metrics_b.clone(),
                ));
            }
        }
        
        comparisons
    }
}

impl Default for BenchmarkComparator {
    fn default() -> Self {
        Self::new()
    }
}

/// Ranking of a benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkRanking {
    pub rank: usize,
    pub name: String,
    pub metrics: Metrics,
    pub score: f64,
    pub index: usize,
}

/// Compare configurations by running benchmarks.
pub struct ConfigComparison {
    pub config_a: BenchmarkConfig,
    pub config_b: BenchmarkConfig,
    pub result_a: Option<BenchmarkResult>,
    pub result_b: Option<BenchmarkResult>,
}

impl ConfigComparison {
    /// Create a new configuration comparison.
    pub fn new(config_a: BenchmarkConfig, config_b: BenchmarkConfig) -> Self {
        Self {
            config_a,
            config_b,
            result_a: None,
            result_b: None,
        }
    }
    
    /// Set result for configuration A.
    pub fn with_result_a(mut self, result: BenchmarkResult) -> Self {
        self.result_a = Some(result);
        self
    }
    
    /// Set result for configuration B.
    pub fn with_result_b(mut self, result: BenchmarkResult) -> Self {
        self.result_b = Some(result);
        self
    }
    
    /// Compare the configurations.
    pub fn compare(&self) -> Result<BenchmarkComparison> {
        let metrics_a: Metrics = self.result_a
            .as_ref()
            .ok_or_else(|| crate::error::BenchmarkError::invalid_input(
                "Result A is missing"
            ))?
            .into();
        
        let metrics_b: Metrics = self.result_b
            .as_ref()
            .ok_or_else(|| crate::error::BenchmarkError::invalid_input(
                "Result B is missing"
            ))?
            .into();
        
        Ok(BenchmarkComparison::new(
            "Config A".to_string(),
            metrics_a,
            "Config B".to_string(),
            metrics_b,
        ))
    }
}

/// Compare latency distributions.
pub fn compare_latency_distributions(
    name_a: &str,
    latencies_a: &[f64],
    name_b: &str,
    latencies_b: &[f64],
) -> String {
    let summary_a = StatisticalSummary::from_data(latencies_a);
    let summary_b = StatisticalSummary::from_data(latencies_b);
    let comparison = compare_summaries(&summary_a, &summary_b);
    
    format!(
        "{} vs {}: mean {:.2}% ({:.2}ms), std {:.2}%, P95 {:.2}ms vs {:.2}ms",
        name_a,
        name_b,
        comparison.mean_diff_pct,
        comparison.mean_diff,
        comparison.std_dev_diff_pct,
        summary_a.p95,
        summary_b.p95
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_comparison() {
        let metrics_a = Metrics::builder()
            .accuracy(0.9)
            .latency_p50(100.0)
            .throughput(50.0)
            .build();
        
        let metrics_b = Metrics::builder()
            .accuracy(0.95)
            .latency_p50(80.0)
            .throughput(60.0)
            .build();
        
        let comparison = BenchmarkComparison::new(
            "A".to_string(),
            metrics_a,
            "B".to_string(),
            metrics_b,
        );
        
        assert!(comparison.b_is_better());
    }
    
    #[test]
    fn test_benchmark_comparator() {
        let mut comparator = BenchmarkComparator::new();
        comparator.add("A".to_string(), Metrics::builder().accuracy(0.9).build());
        comparator.add("B".to_string(), Metrics::builder().accuracy(0.95).build());
        
        let rankings = comparator.compare_all();
        assert_eq!(rankings[0].name, "B");
    }
}












