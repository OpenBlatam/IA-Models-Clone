//! Convert Module
//!
//! Conversion utilities for common type conversions.

pub mod metrics;
pub mod numeric;
pub mod string;
pub mod collections;

// Re-exports
pub use metrics::{
    benchmark_result_to_metrics,
    latencies_to_metrics,
};
pub use numeric::{
    f64_to_usize,
    usize_to_f64,
    i64_to_f64,
    f64_to_i64,
};
pub use string::{
    str_to_f64,
    str_to_usize,
};
pub use collections::{
    option_to_result,
    vec_to_array,
};




