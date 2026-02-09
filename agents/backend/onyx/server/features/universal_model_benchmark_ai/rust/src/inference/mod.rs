//! Inference Module - High-performance inference engine
//!
//! This module provides a modular inference system with:
//! - Engine abstraction
//! - Tokenization
//! - Batch processing
//! - Performance metrics
//! - Configuration management
//! - Error handling
//! - Advanced batching

pub mod engine;
pub mod tokenizer;
pub mod config;
pub mod stats;
pub mod sampling;
pub mod error;
pub mod batch;
pub mod metrics;
pub mod validators;
pub mod utils;

// Re-exports
pub use engine::InferenceEngine;
pub use config::{InferenceConfig, SamplingConfig, SamplingStrategy};
pub use stats::InferenceStats;
pub use tokenizer::TokenizerWrapper;
pub use error::{InferenceError, InferenceResult};
// Batch types are available from the general batching module
// pub use batch::{DynamicBatcher, ContinuousBatcher, BatchItem, BatchPriority};
pub use metrics::{MetricsCollector, InferenceMetrics, Timer};
pub use validators::{
    validate_config,
    validate_temperature,
    validate_top_p,
    validate_top_k,
    validate_batch_size,
};
pub use utils::{
    validate_range,
    validate_positive,
    calculate_tokens_per_second,
    format_latency,
    format_memory,
    clamp,
    lerp,
    ExponentialMovingAverage,
};

