//! Profiling Module
//!
//! Performance profiling with timing, memory tracking, and reporting.

pub mod profiler;

// Re-exports
pub use profiler::{
    Profiler,
    MemorySnapshot,
    TimingStats,
    PerformanceReport,
    Timer,
};




