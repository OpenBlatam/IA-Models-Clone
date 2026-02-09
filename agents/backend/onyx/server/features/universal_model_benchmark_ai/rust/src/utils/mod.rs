//! Utilities Module
//!
//! Collection of utility functions organized by category.

pub mod formatting;
pub mod statistics;
pub mod validation;
pub mod timing;

// Re-exports from formatting
pub use formatting::{
    format_duration,
    format_bytes,
    format_number,
    format_percentage,
    format_latency_ms,
    format_throughput,
};

// Re-exports from statistics
pub use statistics::{
    percentile,
    percentiles,
    mean,
    median,
    std_dev,
    summary_stats,
};

// Re-exports from validation
pub use validation::{
    clamp,
    clamp_i64,
    clamp_usize,
    in_range,
    in_range_i64,
    is_positive,
    is_non_negative,
    is_finite,
    validate_range,
    validate_positive,
};

// Re-exports from timing
pub use timing::{
    measure_time,
    measure_duration,
    Timer,
    ScopedTimer,
};

// Legacy re-exports for backward compatibility
#[deprecated(note = "Use measure_time instead")]
pub use measure_time as measure_time_legacy;




