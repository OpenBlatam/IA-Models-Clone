//! Inference Error Types
//!
//! Custom error types for inference operations with detailed context.

use thiserror::Error;

/// Inference engine errors.
#[derive(Debug, Error)]
pub enum InferenceError {
    /// Model loading error.
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),
    
    /// Tokenizer error.
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
    
    /// Encoding error.
    #[error("Encoding error: {0}")]
    EncodingError(String),
    
    /// Decoding error.
    #[error("Decoding error: {0}")]
    DecodingError(String),
    
    /// Inference error.
    #[error("Inference error: {0}")]
    InferenceError(String),
    
    /// Batch processing error.
    #[error("Batch processing error: {0}")]
    BatchError(String),
    
    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    /// Device error.
    #[error("Device error: {0}")]
    DeviceError(String),
    
    /// Memory error.
    #[error("Memory error: {0}")]
    MemoryError(String),
    
    /// Validation error.
    #[error("Validation error: {0}")]
    ValidationError(String),
}

/// Result type for inference operations.
pub type InferenceResult<T> = Result<T, InferenceError>;

/// Helper to convert anyhow errors to InferenceError.
impl From<anyhow::Error> for InferenceError {
    fn from(err: anyhow::Error) -> Self {
        InferenceError::InferenceError(err.to_string())
    }
}

/// Helper to convert std::io::Error to InferenceError.
impl From<std::io::Error> for InferenceError {
    fn from(err: std::io::Error) -> Self {
        InferenceError::ModelLoadError(err.to_string())
    }
}




