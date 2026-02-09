//! Report Generator
//!
//! Functions for generating various types of reports.

use crate::Metrics;
use crate::error::Result;
use super::types::{BenchmarkReport, ComparisonReport};

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
    
    /// Generate report with samples.
    pub fn generate_with_samples(
        model_name: &str,
        benchmark_name: &str,
        metrics: &Metrics,
        correct: usize,
        total: usize,
    ) -> BenchmarkReport {
        let mut report = BenchmarkReport::new(
            model_name.to_string(),
            benchmark_name.to_string(),
            metrics.clone(),
        );
        
        report.samples = super::types::ReportSamples::from_counts(correct, total);
        report
    }
    
    /// Generate report with performance breakdown.
    pub fn generate_with_performance(
        model_name: &str,
        benchmark_name: &str,
        metrics: &Metrics,
        performance: super::types::PerformanceBreakdown,
    ) -> BenchmarkReport {
        BenchmarkReport::new(
            model_name.to_string(),
            benchmark_name.to_string(),
            metrics.clone(),
        )
        .with_performance(performance)
    }
}




