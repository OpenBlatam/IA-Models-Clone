//! Report Export
//!
//! Functions for exporting reports to various formats.

use std::fs::File;
use std::io::Write;
use crate::error::Result;
use super::types::{BenchmarkReport, ComparisonReport};

/// Export reports to JSON file.
pub fn export_reports_json(
    reports: &[BenchmarkReport],
    output_path: &str,
) -> Result<()> {
    let json = serde_json::to_string_pretty(reports)
        .map_err(|e| crate::error::BenchmarkError::serialization(e.to_string()))?;
    
    let mut file = File::create(output_path)
        .map_err(|e| crate::error::BenchmarkError::io(e))?;
    
    file.write_all(json.as_bytes())
        .map_err(|e| crate::error::BenchmarkError::io(e))?;
    
    Ok(())
}

/// Export comparison report to JSON file.
pub fn export_comparison_json(
    comparison: &ComparisonReport,
    output_path: &str,
) -> Result<()> {
    let json = comparison.to_json()?;
    
    let mut file = File::create(output_path)
        .map_err(|e| crate::error::BenchmarkError::io(e))?;
    
    file.write_all(json.as_bytes())
        .map_err(|e| crate::error::BenchmarkError::io(e))?;
    
    Ok(())
}

/// Export comparison report to Markdown.
pub fn export_comparison_markdown(
    comparison: &ComparisonReport,
    output_path: &str,
) -> Result<()> {
    let md = comparison.to_markdown();
    
    let mut file = File::create(output_path)
        .map_err(|e| crate::error::BenchmarkError::io(e))?;
    
    file.write_all(md.as_bytes())
        .map_err(|e| crate::error::BenchmarkError::io(e))?;
    
    Ok(())
}

/// Export single report to JSON file.
pub fn export_report_json(
    report: &BenchmarkReport,
    output_path: &str,
) -> Result<()> {
    let json = report.to_json()?;
    
    let mut file = File::create(output_path)
        .map_err(|e| crate::error::BenchmarkError::io(e))?;
    
    file.write_all(json.as_bytes())
        .map_err(|e| crate::error::BenchmarkError::io(e))?;
    
    Ok(())
}




