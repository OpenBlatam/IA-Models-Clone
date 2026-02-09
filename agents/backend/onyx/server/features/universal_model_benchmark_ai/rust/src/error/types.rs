//! Error Types
//!
//! Comprehensive error types for benchmark operations.

use thiserror::Error;

/// Result type alias for benchmark operations.
pub type Result<T> = std::result::Result<T, BenchmarkError>;

/// Comprehensive error type for benchmark operations.
#[derive(Error, Debug)]
pub enum BenchmarkError {
    /// Model loading error.
    #[error("Failed to load model: {0}")]
    ModelLoad(String),
    
    /// Inference error.
    #[error("Inference failed: {0}")]
    Inference(String),
    
    /// Tokenization error.
    #[error("Tokenization failed: {0}")]
    Tokenization(String),
    
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    /// Data processing error.
    #[error("Data processing failed: {0}")]
    DataProcessing(String),
    
    /// Metrics calculation error.
    #[error("Metrics calculation failed: {0}")]
    Metrics(String),
    
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    /// Deserialization error.
    #[error("Deserialization error: {0}")]
    Deserialization(String),
    
    /// Invalid input error.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Resource not found error.
    #[error("Resource not found: {0}")]
    NotFound(String),
    
    /// Timeout error.
    #[error("Operation timed out: {0}")]
    Timeout(String),
    
    /// Generic error with message.
    #[error("{0}")]
    Other(String),
}

impl BenchmarkError {
    /// Create a model load error.
    pub fn model_load(msg: impl Into<String>) -> Self {
        Self::ModelLoad(msg.into())
    }
    
    /// Create an inference error.
    pub fn inference(msg: impl Into<String>) -> Self {
        Self::Inference(msg.into())
    }
    
    /// Create a tokenization error.
    pub fn tokenization(msg: impl Into<String>) -> Self {
        Self::Tokenization(msg.into())
    }
    
    /// Create a configuration error.
    pub fn configuration(msg: impl Into<String>) -> Self {
        Self::Configuration(msg.into())
    }
    
    /// Create a data processing error.
    pub fn data_processing(msg: impl Into<String>) -> Self {
        Self::DataProcessing(msg.into())
    }
    
    /// Create a metrics error.
    pub fn metrics(msg: impl Into<String>) -> Self {
        Self::Metrics(msg.into())
    }
    
    /// Create an invalid input error.
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }
    
    /// Create a not found error.
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }
    
    /// Create a timeout error.
    pub fn timeout(msg: impl Into<String>) -> Self {
        Self::Timeout(msg.into())
    }
    
    /// Create a serialization error.
    pub fn serialization(msg: impl Into<String>) -> Self {
        Self::Serialization(msg.into())
    }
    
    /// Create a deserialization error.
    pub fn deserialization(msg: impl Into<String>) -> Self {
        Self::Deserialization(msg.into())
    }
    
    /// Create a generic error.
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }
}

/// Helper to convert serde_json errors.
impl From<serde_json::Error> for BenchmarkError {
    fn from(err: serde_json::Error) -> Self {
        BenchmarkError::Serialization(err.to_string())
    }
}

/// Helper to convert anyhow errors.
impl From<anyhow::Error> for BenchmarkError {
    fn from(err: anyhow::Error) -> Self {
        BenchmarkError::Other(err.to_string())
    }
}




