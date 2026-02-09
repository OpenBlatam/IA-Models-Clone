//! Metrics Module
//!
//! Comprehensive metrics calculation and analysis.

mod calculation;
mod aggregation;

// Re-exports from calculation
pub use calculation::{
    calculate_metrics,
    calculate_accuracy,
    calculate_accuracy_from_counts,
    calculate_throughput,
    calculate_throughput_requests,
    calculate_latency_stats,
    calculate_latency_percentiles,
    calculate_memory_efficiency,
    calculate_cost_efficiency,
};

// Re-exports from aggregation
pub use aggregation::{
    calculate_statistics,
    aggregate_metrics,
    weighted_average_metrics,
    median_metrics,
    compare_metrics,
};
