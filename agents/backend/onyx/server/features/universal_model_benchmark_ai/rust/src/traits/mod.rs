//! Traits Module
//!
//! Useful traits for common operations.

pub mod core;
pub mod conversion;
pub mod statistics;

// Re-exports
pub use core::{Validate, Summarize, PerformanceStats, Reset, Statistics};
pub use conversion::{ToMetrics, ToJson, FromJson};




