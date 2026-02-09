//! Inference Statistics
//!
//! Performance metrics and statistics for inference operations.

use serde::{Serialize, Deserialize};

/// Inference statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStats {
    pub latency_ms: f64,
    pub tokens_per_second: f64,
    pub memory_usage_mb: f64,
    pub num_tokens: usize,
    pub num_input_tokens: usize,
    pub num_output_tokens: usize,
}

impl InferenceStats {
    /// Create new stats.
    pub fn new(
        latency_ms: f64,
        num_input_tokens: usize,
        num_output_tokens: usize,
    ) -> Self {
        let num_tokens = num_input_tokens + num_output_tokens;
        let tokens_per_second = if latency_ms > 0.0 {
            (num_tokens as f64) / (latency_ms / 1000.0)
        } else {
            0.0
        };
        
        Self {
            latency_ms,
            tokens_per_second,
            memory_usage_mb: 0.0, // TODO: Get actual memory usage
            num_tokens,
            num_input_tokens,
            num_output_tokens,
        }
    }
    
    /// Merge multiple stats.
    pub fn merge(stats: &[Self]) -> Self {
        if stats.is_empty() {
            return Self::new(0.0, 0, 0);
        }
        
        let total_latency = stats.iter().map(|s| s.latency_ms).sum::<f64>();
        let total_input_tokens = stats.iter().map(|s| s.num_input_tokens).sum::<usize>();
        let total_output_tokens = stats.iter().map(|s| s.num_output_tokens).sum::<usize>();
        let avg_latency = total_latency / stats.len() as f64;
        
        Self::new(avg_latency, total_input_tokens, total_output_tokens)
    }
}












