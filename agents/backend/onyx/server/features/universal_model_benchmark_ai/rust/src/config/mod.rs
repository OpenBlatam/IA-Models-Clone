//! Configuration Module
//!
//! Centralized configuration management with validation and builders.

pub mod constants;
pub mod benchmark_config;

// Re-exports
pub use constants::{defaults, limits};
pub use benchmark_config::{BenchmarkConfig, BenchmarkConfigBuilder};




