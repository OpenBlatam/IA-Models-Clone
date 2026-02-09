//! Configuration presets for common use cases.
//!
//! Provides pre-configured BenchmarkConfig instances for typical scenarios.

use crate::config::BenchmarkConfig;
use crate::error::Result;
use crate::constants::{
    batch_sizes, token_limits, temperatures, top_p_values, top_k_values,
};

/// Create a configuration preset for fast inference.
///
/// Optimized for low latency with small batch size and short context.
pub fn fast_inference(model_path: String) -> Result<BenchmarkConfig> {
    BenchmarkConfig::builder()
        .model_path(model_path)
        .batch_size(batch_sizes::SMALL)
        .max_tokens(token_limits::SHORT)
        .temperature(temperatures::LOW)
        .top_p(top_p_values::FOCUSED)
        .top_k(top_k_values::FOCUSED)
        .build()
}

/// Create a configuration preset for high throughput.
///
/// Optimized for processing many requests with large batch size.
pub fn high_throughput(model_path: String) -> Result<BenchmarkConfig> {
    BenchmarkConfig::builder()
        .model_path(model_path)
        .batch_size(batch_sizes::VERY_LARGE)
        .max_tokens(token_limits::MEDIUM)
        .temperature(temperatures::MEDIUM)
        .top_p(top_p_values::BALANCED)
        .top_k(top_k_values::BALANCED)
        .build()
}

/// Create a configuration preset for creative generation.
///
/// Optimized for diverse and creative outputs with high temperature.
pub fn creative_generation(model_path: String) -> Result<BenchmarkConfig> {
    BenchmarkConfig::builder()
        .model_path(model_path)
        .batch_size(batch_sizes::MEDIUM)
        .max_tokens(token_limits::LONG)
        .temperature(temperatures::HIGH)
        .top_p(top_p_values::DIVERSE)
        .top_k(top_k_values::DIVERSE)
        .build()
}

/// Create a configuration preset for deterministic outputs.
///
/// Optimized for consistent, reproducible results with low temperature.
pub fn deterministic(model_path: String) -> Result<BenchmarkConfig> {
    BenchmarkConfig::builder()
        .model_path(model_path)
        .batch_size(batch_sizes::SMALL)
        .max_tokens(token_limits::MEDIUM)
        .temperature(temperatures::VERY_LOW)
        .top_p(top_p_values::VERY_FOCUSED)
        .top_k(top_k_values::VERY_FOCUSED)
        .build()
}

/// Create a configuration preset for long context processing.
///
/// Optimized for processing long documents with maximum context.
pub fn long_context(model_path: String) -> Result<BenchmarkConfig> {
    BenchmarkConfig::builder()
        .model_path(model_path)
        .batch_size(batch_sizes::SMALL)
        .max_tokens(token_limits::MAXIMUM)
        .temperature(temperatures::MEDIUM)
        .top_p(top_p_values::BALANCED)
        .top_k(top_k_values::BALANCED)
        .build()
}

/// Create a configuration preset for balanced performance.
///
/// Good default for most use cases with balanced settings.
pub fn balanced(model_path: String) -> Result<BenchmarkConfig> {
    BenchmarkConfig::builder()
        .model_path(model_path)
        .batch_size(batch_sizes::MEDIUM)
        .max_tokens(token_limits::MEDIUM)
        .temperature(temperatures::MEDIUM)
        .top_p(top_p_values::BALANCED)
        .top_k(top_k_values::BALANCED)
        .build()
}

/// Create a configuration preset for code generation.
///
/// Optimized for generating code with focused sampling.
pub fn code_generation(model_path: String) -> Result<BenchmarkConfig> {
    BenchmarkConfig::builder()
        .model_path(model_path)
        .batch_size(batch_sizes::MEDIUM)
        .max_tokens(token_limits::LONG)
        .temperature(temperatures::LOW)
        .top_p(top_p_values::FOCUSED)
        .top_k(top_k_values::FOCUSED)
        .build()
}

/// Create a configuration preset for conversational AI.
///
/// Optimized for natural conversations with balanced creativity.
pub fn conversational(model_path: String) -> Result<BenchmarkConfig> {
    BenchmarkConfig::builder()
        .model_path(model_path)
        .batch_size(batch_sizes::SMALL)
        .max_tokens(token_limits::MEDIUM)
        .temperature(temperatures::MEDIUM)
        .top_p(top_p_values::BALANCED)
        .top_k(top_k_values::BALANCED)
        .build()
}

/// Create a configuration preset for summarization.
///
/// Optimized for creating concise summaries.
pub fn summarization(model_path: String) -> Result<BenchmarkConfig> {
    BenchmarkConfig::builder()
        .model_path(model_path)
        .batch_size(batch_sizes::MEDIUM)
        .max_tokens(token_limits::SHORT)
        .temperature(temperatures::LOW)
        .top_p(top_p_values::FOCUSED)
        .top_k(top_k_values::FOCUSED)
        .build()
}

/// Create a configuration preset for question answering.
///
/// Optimized for accurate, focused answers.
pub fn question_answering(model_path: String) -> Result<BenchmarkConfig> {
    BenchmarkConfig::builder()
        .model_path(model_path)
        .batch_size(batch_sizes::SMALL)
        .max_tokens(token_limits::MEDIUM)
        .temperature(temperatures::VERY_LOW)
        .top_p(top_p_values::VERY_FOCUSED)
        .top_k(top_k_values::VERY_FOCUSED)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fast_inference() {
        let config = fast_inference("test".to_string()).unwrap();
        assert_eq!(config.batch_size, batch_sizes::SMALL);
        assert_eq!(config.max_tokens, token_limits::SHORT);
    }
    
    #[test]
    fn test_high_throughput() {
        let config = high_throughput("test".to_string()).unwrap();
        assert_eq!(config.batch_size, batch_sizes::VERY_LARGE);
    }
    
    #[test]
    fn test_balanced() {
        let config = balanced("test".to_string()).unwrap();
        assert_eq!(config.batch_size, batch_sizes::MEDIUM);
        assert_eq!(config.temperature, temperatures::MEDIUM);
    }
}












