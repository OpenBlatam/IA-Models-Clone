//! Configuration Module - Centralized configuration management.
//!
//! Provides:
//! - Configuration structures
//! - Default values
//! - Validation
//! - Builder patterns

use serde::{Deserialize, Serialize};
use crate::error::{BenchmarkError, Result};
use crate::traits::Validate;

// ════════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ════════════════════════════════════════════════════════════════════════════════

/// Default values for configuration.
pub mod defaults {
    pub const MAX_TOKENS: usize = 512;
    pub const TEMPERATURE: f32 = 0.7;
    pub const TOP_P: f32 = 0.9;
    pub const TOP_K: usize = 50;
    pub const BATCH_SIZE: usize = 1;
    pub const REPETITION_PENALTY: f32 = 1.0;
}

/// Limits for configuration values.
pub mod limits {
    pub const MAX_TOKENS_LIMIT: usize = 32768;
    pub const MIN_TOKENS: usize = 1;
    pub const MAX_TEMPERATURE: f32 = 2.0;
    pub const MIN_TEMPERATURE: f32 = 0.0;
    pub const MAX_TOP_P: f32 = 1.0;
    pub const MIN_TOP_P: f32 = 0.0;
    pub const MIN_TOP_K: usize = 1;
    pub const MAX_BATCH_SIZE: usize = 128;
    pub const MIN_BATCH_SIZE: usize = 1;
    pub const MAX_REPETITION_PENALTY: f32 = 2.0;
    pub const MIN_REPETITION_PENALTY: f32 = 1.0;
}

// ════════════════════════════════════════════════════════════════════════════════
// BENCHMARK CONFIGURATION
// ════════════════════════════════════════════════════════════════════════════════

/// Benchmark configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub model_path: String,
    pub batch_size: usize,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            batch_size: defaults::BATCH_SIZE,
            max_tokens: defaults::MAX_TOKENS,
            temperature: defaults::TEMPERATURE,
            top_p: defaults::TOP_P,
            top_k: defaults::TOP_K,
        }
    }
}

impl BenchmarkConfig {
    /// Create a new benchmark configuration.
    pub fn new(model_path: String) -> Self {
        Self {
            model_path,
            ..Default::default()
        }
    }
    
    /// Create with builder pattern.
    pub fn builder() -> BenchmarkConfigBuilder {
        BenchmarkConfigBuilder::new()
    }
    
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        <Self as Validate>::validate(self)
    }
    
    /// Create a copy with modified batch size.
    pub fn with_batch_size(&self, batch_size: usize) -> Self {
        let mut config = self.clone();
        config.batch_size = batch_size;
        config
    }
    
    /// Create a copy with modified max tokens.
    pub fn with_max_tokens(&self, max_tokens: usize) -> Self {
        let mut config = self.clone();
        config.max_tokens = max_tokens;
        config
    }
    
    /// Create a copy with modified temperature.
    pub fn with_temperature(&self, temperature: f32) -> Self {
        let mut config = self.clone();
        config.temperature = temperature;
        config
    }
    
    /// Check if configuration is valid without returning error.
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }
    
    /// Get a summary string of the configuration.
    pub fn summary(&self) -> String {
        format!(
            "model={}, batch={}, max_tokens={}, temp={:.2}",
            self.model_path,
            self.batch_size,
            self.max_tokens,
            self.temperature
        )
    }
}

impl Validate for BenchmarkConfig {
    fn validate(&self) -> Result<()> {
        if self.model_path.is_empty() {
            return Err(BenchmarkError::invalid_input("model_path cannot be empty"));
        }
        if self.batch_size < limits::MIN_BATCH_SIZE || self.batch_size > limits::MAX_BATCH_SIZE {
            return Err(BenchmarkError::invalid_input(
                format!("batch_size must be between {} and {}", 
                    limits::MIN_BATCH_SIZE, limits::MAX_BATCH_SIZE)
            ));
        }
        if self.max_tokens < limits::MIN_TOKENS || self.max_tokens > limits::MAX_TOKENS_LIMIT {
            return Err(BenchmarkError::invalid_input(
                format!("max_tokens must be between {} and {}", 
                    limits::MIN_TOKENS, limits::MAX_TOKENS_LIMIT)
            ));
        }
        if !(limits::MIN_TEMPERATURE..=limits::MAX_TEMPERATURE).contains(&self.temperature) {
            return Err(BenchmarkError::invalid_input(
                format!("temperature must be between {} and {}", 
                    limits::MIN_TEMPERATURE, limits::MAX_TEMPERATURE)
            ));
        }
        if !(limits::MIN_TOP_P..=limits::MAX_TOP_P).contains(&self.top_p) {
            return Err(BenchmarkError::invalid_input(
                format!("top_p must be between {} and {}", 
                    limits::MIN_TOP_P, limits::MAX_TOP_P)
            ));
        }
        if self.top_k < limits::MIN_TOP_K {
            return Err(BenchmarkError::invalid_input(
                format!("top_k must be at least {}", limits::MIN_TOP_K)
            ));
        }
        Ok(())
    }
}

/// Builder for BenchmarkConfig.
pub struct BenchmarkConfigBuilder {
    config: BenchmarkConfig,
}

impl BenchmarkConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
        }
    }
    
    /// Set model path.
    pub fn model_path(mut self, path: String) -> Self {
        self.config.model_path = path;
        self
    }
    
    /// Set batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }
    
    /// Set max tokens.
    pub fn max_tokens(mut self, tokens: usize) -> Self {
        self.config.max_tokens = tokens;
        self
    }
    
    /// Set temperature.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = temp;
        self
    }
    
    /// Set top_p.
    pub fn top_p(mut self, p: f32) -> Self {
        self.config.top_p = p;
        self
    }
    
    /// Set top_k.
    pub fn top_k(mut self, k: usize) -> Self {
        self.config.top_k = k;
        self
    }
    
    /// Build the configuration.
    pub fn build(self) -> Result<BenchmarkConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for BenchmarkConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.batch_size, defaults::BATCH_SIZE);
        assert_eq!(config.max_tokens, defaults::MAX_TOKENS);
    }
    
    #[test]
    fn test_builder() {
        let config = BenchmarkConfig::builder()
            .model_path("test_model".to_string())
            .batch_size(32)
            .max_tokens(1024)
            .build()
            .unwrap();
        
        assert_eq!(config.model_path, "test_model");
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.max_tokens, 1024);
    }
    
    #[test]
    fn test_validation() {
        let config = BenchmarkConfig::new(String::new());
        assert!(config.validate().is_err());
        
        let config = BenchmarkConfig::builder()
            .model_path("test".to_string())
            .batch_size(200)  // Exceeds limit
            .build();
        assert!(config.is_err());
    }
}

