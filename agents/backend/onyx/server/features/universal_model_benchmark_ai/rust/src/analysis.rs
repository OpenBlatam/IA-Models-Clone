//! Statistical analysis utilities.
//!
//! Provides advanced statistical analysis functions for benchmark data.

use crate::error::Result;
use crate::types::Metrics;

/// Statistical summary of a dataset.
#[derive(Debug, Clone)]
pub struct StatisticalSummary {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub p25: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

impl StatisticalSummary {
    /// Create a new statistical summary from data.
    pub fn from_data(data: &[f64]) -> Self {
        if data.is_empty() {
            return Self::empty();
        }
        
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let count = sorted.len();
        let mean = sorted.iter().sum::<f64>() / count as f64;
        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };
        
        let variance = sorted.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();
        
        let min = sorted[0];
        let max = sorted[count - 1];
        
        let p25 = percentile(&sorted, 0.25);
        let p75 = percentile(&sorted, 0.75);
        let p90 = percentile(&sorted, 0.90);
        let p95 = percentile(&sorted, 0.95);
        let p99 = percentile(&sorted, 0.99);
        
        Self {
            count,
            mean,
            median,
            std_dev,
            min,
            max,
            p25,
            p75,
            p90,
            p95,
            p99,
        }
    }
    
    /// Create an empty summary.
    pub fn empty() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            p25: 0.0,
            p75: 0.0,
            p90: 0.0,
            p95: 0.0,
            p99: 0.0,
        }
    }
    
    /// Calculate coefficient of variation.
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean == 0.0 {
            return 0.0;
        }
        self.std_dev / self.mean
    }
    
    /// Check if data has low variance (CV < 0.1).
    pub fn is_low_variance(&self) -> bool {
        self.coefficient_of_variation() < 0.1
    }
    
    /// Check if data has high variance (CV > 0.5).
    pub fn is_high_variance(&self) -> bool {
        self.coefficient_of_variation() > 0.5
    }
    
    /// Calculate interquartile range (IQR).
    pub fn iqr(&self) -> f64 {
        self.p75 - self.p25
    }
    
    /// Detect outliers using IQR method.
    pub fn detect_outliers(&self, data: &[f64]) -> Vec<usize> {
        let q1 = self.p25;
        let q3 = self.p75;
        let iqr = self.iqr();
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;
        
        data.iter()
            .enumerate()
            .filter(|(_, &value)| value < lower_bound || value > upper_bound)
            .map(|(idx, _)| idx)
            .collect()
    }
}

/// Calculate percentile from sorted data.
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    
    let index = (p * (sorted_data.len() - 1) as f64).round() as usize;
    sorted_data[index.min(sorted_data.len() - 1)]
}

/// Compare two statistical summaries.
pub fn compare_summaries(baseline: &StatisticalSummary, candidate: &StatisticalSummary) -> ComparisonResult {
    let mean_diff = candidate.mean - baseline.mean;
    let mean_diff_pct = if baseline.mean != 0.0 {
        (mean_diff / baseline.mean) * 100.0
    } else {
        0.0
    };
    
    let std_dev_diff = candidate.std_dev - baseline.std_dev;
    let std_dev_diff_pct = if baseline.std_dev != 0.0 {
        (std_dev_diff / baseline.std_dev) * 100.0
    } else {
        0.0
    };
    
    ComparisonResult {
        mean_diff,
        mean_diff_pct,
        std_dev_diff,
        std_dev_diff_pct,
        is_better: mean_diff > 0.0 && std_dev_diff < 0.0,
        is_worse: mean_diff < 0.0 || std_dev_diff > 0.0,
    }
}

/// Result of comparing two statistical summaries.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub mean_diff: f64,
    pub mean_diff_pct: f64,
    pub std_dev_diff: f64,
    pub std_dev_diff_pct: f64,
    pub is_better: bool,
    pub is_worse: bool,
}

/// Analyze latency distribution.
pub fn analyze_latency_distribution(latencies: &[f64]) -> LatencyAnalysis {
    let summary = StatisticalSummary::from_data(latencies);
    
    let tail_latency = summary.p99 - summary.p50;
    let tail_ratio = if summary.p50 > 0.0 {
        tail_latency / summary.p50
    } else {
        0.0
    };
    
    LatencyAnalysis {
        summary,
        tail_latency,
        tail_ratio,
        is_consistent: summary.is_low_variance(),
        has_outliers: !summary.detect_outliers(latencies).is_empty(),
    }
}

/// Analysis of latency distribution.
#[derive(Debug, Clone)]
pub struct LatencyAnalysis {
    pub summary: StatisticalSummary,
    pub tail_latency: f64,
    pub tail_ratio: f64,
    pub is_consistent: bool,
    pub has_outliers: bool,
}

/// Calculate correlation between two datasets.
pub fn correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(crate::error::BenchmarkError::invalid_input(
            "Datasets must have the same length"
        ));
    }
    
    if x.is_empty() {
        return Err(crate::error::BenchmarkError::invalid_input(
            "Datasets cannot be empty"
        ));
    }
    
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    
    let numerator: f64 = x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();
    
    let sum_sq_x: f64 = x.iter()
        .map(|xi| (xi - mean_x).powi(2))
        .sum();
    
    let sum_sq_y: f64 = y.iter()
        .map(|yi| (yi - mean_y).powi(2))
        .sum();
    
    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    
    if denominator == 0.0 {
        return Ok(0.0);
    }
    
    Ok(numerator / denominator)
}

/// Calculate trend in data (increasing, decreasing, stable).
pub fn calculate_trend(data: &[f64]) -> Trend {
    if data.len() < 2 {
        return Trend::Stable;
    }
    
    let mut increasing = 0;
    let mut decreasing = 0;
    
    for i in 1..data.len() {
        if data[i] > data[i - 1] {
            increasing += 1;
        } else if data[i] < data[i - 1] {
            decreasing += 1;
        }
    }
    
    let total = data.len() - 1;
    let threshold = (total as f64 * 0.6) as usize;
    
    if increasing >= threshold {
        Trend::Increasing
    } else if decreasing >= threshold {
        Trend::Decreasing
    } else {
        Trend::Stable
    }
}

/// Trend in data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    Increasing,
    Decreasing,
    Stable,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_statistical_summary() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let summary = StatisticalSummary::from_data(&data);
        
        assert_eq!(summary.count, 5);
        assert_eq!(summary.mean, 3.0);
        assert_eq!(summary.median, 3.0);
    }
    
    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_trend() {
        let increasing = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(calculate_trend(&increasing), Trend::Increasing);
        
        let decreasing = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(calculate_trend(&decreasing), Trend::Decreasing);
    }
}












