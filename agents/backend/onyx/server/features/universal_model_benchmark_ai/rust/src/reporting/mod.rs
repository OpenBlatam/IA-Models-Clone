//! Reporting Module
//!
//! Comprehensive report generation and export utilities.

pub mod types;
pub mod generator;
pub mod export;
pub mod formatters;

// Re-exports
pub use types::{
    BenchmarkReport,
    ReportSamples,
    PerformanceBreakdown,
    ModelComparison,
    ComparisonReport,
};

pub use generator::ReportGenerator;

pub use export::{
    export_reports_json,
    export_comparison_json,
    export_comparison_markdown,
    export_report_json,
};




