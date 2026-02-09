//! Inference Configuration
//!
//! Configuration structures for inference engine.

use serde::{Serialize, Deserialize};

/// Main inference configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub batch_size: usize,
    pub sampling: SamplingConfig,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.0,
            batch_size: 1,
            sampling: SamplingConfig::default(),
        }
    }
}

/// Sampling configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub strategy: SamplingStrategy,
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategy::TopP,
            seed: None,
        }
    }
}

/// Sampling strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    Greedy,
    TopK,
    TopP,
    Temperature,
    Nucleus,
}












