//! Report Formatters
//!
//! Functions for formatting reports in various text formats.

use super::types::ComparisonReport;

impl ComparisonReport {
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
    
    /// Generate CSV report.
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("Rank,Model,Accuracy,Latency_P50,Latency_P95,Latency_P99,Throughput,Score\n");
        
        for model in &self.models {
            csv.push_str(&format!(
                "{},{},{:.4},{:.4},{:.4},{:.4},{:.2},{:.4}\n",
                model.rank,
                model.model_name,
                model.metrics.accuracy,
                model.metrics.latency_p50,
                model.metrics.latency_p95,
                model.metrics.latency_p99,
                model.metrics.throughput,
                model.score
            ));
        }
        
        csv
    }
    
    /// Generate summary text.
    pub fn to_summary(&self) -> String {
        let mut summary = format!("Benchmark Comparison: {}\n", self.benchmark_name);
        summary.push_str(&format!("Best Model: {}\n\n", self.best_model));
        
        summary.push_str("Top 3 Models:\n");
        for model in self.models.iter().take(3) {
            summary.push_str(&format!(
                "  {}. {} - Accuracy: {:.2}%, Latency: {:.3}s, Score: {:.3}\n",
                model.rank,
                model.model_name,
                model.metrics.accuracy * 100.0,
                model.metrics.latency_p50,
                model.score
            ));
        }
        
        summary
    }
}




