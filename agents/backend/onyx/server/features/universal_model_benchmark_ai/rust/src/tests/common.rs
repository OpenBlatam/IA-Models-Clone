//! Common test utilities and fixtures.
//!
//! Shared test helpers for all test modules.

use crate::inference::{InferenceEngine, InferenceConfig, SamplingConfig};
use crate::data::DataProcessor;
use crate::inference::MetricsCollector;
use std::path::PathBuf;

/// Create a test inference engine.
pub fn create_test_inference_engine() -> InferenceEngine {
    // Use a dummy model path for testing
    let model_path = PathBuf::from("test_model");
    let device = candle_core::Device::Cpu;
    let config = InferenceConfig::default();
    
    InferenceEngine::new(model_path, device, Some(config))
        .expect("Failed to create test inference engine")
}

/// Create a test inference config.
pub fn create_test_inference_config() -> InferenceConfig {
    InferenceConfig {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        repetition_penalty: 1.0,
        batch_size: 1,
        sampling: SamplingConfig::default(),
    }
}

/// Create a test data processor.
pub fn create_test_data_processor() -> DataProcessor {
    DataProcessor::new(None)
}

/// Create a test metrics collector.
pub fn create_test_metrics_collector() -> MetricsCollector {
    MetricsCollector::new(1000)
}

/// Generate test token IDs.
pub fn generate_test_tokens(count: usize) -> Vec<u32> {
    (0..count as u32).collect()
}

/// Generate test text batch.
pub fn generate_test_texts(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("Test text {}", i))
        .collect()
}

/// Assert two float values are approximately equal.
pub fn assert_approx_eq(a: f64, b: f64, epsilon: f64) {
    assert!(
        (a - b).abs() < epsilon,
        "Values not approximately equal: {} vs {} (epsilon: {})",
        a,
        b,
        epsilon
    );
}

/// Assert latency is within reasonable range.
pub fn assert_reasonable_latency(latency_ms: f64) {
    assert!(
        latency_ms >= 0.0 && latency_ms < 1_000_000.0,
        "Latency out of reasonable range: {}ms",
        latency_ms
    );
}

