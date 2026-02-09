//! Benchmark Configuration
//!
//! Configuration structure and validation for benchmarks.

use serde::{Deserialize, Serialize};
use crate::error::{BenchmarkError, Result};
use crate::traits::Validate;
use super::constants::{defaults, limits};

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
    
    /// Create a copy with modified top_p.
    pub fn with_top_p(&self, top_p: f32) -> Self {
        let mut config = self.clone();
        config.top_p = top_p;
        config
    }
    
    /// Create a copy with modified top_k.
    pub fn with_top_k(&self, top_k: usize) -> Self {
        let mut config = self.clone();
        config.top_k = top_k;
        config
    }
    
    /// Check if configuration is valid without returning error.
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }
    
    /// Get a summary string of the configuration.
    pub fn summary(&self) -> String {
        format!(
            "model={}, batch={}, max_tokens={}, temp={:.2}, top_p={:.2}, top_k={}",
            self.model_path,
            self.batch_size,
            self.max_tokens,
            self.temperature,
            self.top_p,
            self.top_k
        )
    }
    
    /// Merge with another config (other takes precedence).
    pub fn merge(&self, other: &BenchmarkConfig) -> Self {
        Self {
            model_path: if !other.model_path.is_empty() {
                other.model_path.clone()
            } else {
                self.model_path.clone()
            },
            batch_size: other.batch_size,
            max_tokens: other.max_tokens,
            temperature: other.temperature,
            top_p: other.top_p,
            top_k: other.top_k,
        }
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
    
    /// Build without validation (use with caution).
    pub fn build_unchecked(self) -> BenchmarkConfig {
        self.config
    }
}

impl Default for BenchmarkConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}




