//! Error handling for benchmark core.
//!
//! Provides comprehensive error types and error handling utilities.

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
    
    /// Invalid input error.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Resource not found error.
    #[error("Resource not found: {0}")]
    NotFound(String),
    
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
    
    /// Create a serialization error.
    pub fn serialization(msg: impl Into<String>) -> Self {
        Self::Serialization(msg.into())
    }
    
    /// Create a generic error.
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }
    
    /// Check if error is a model load error.
    pub fn is_model_load(&self) -> bool {
        matches!(self, Self::ModelLoad(_))
    }
    
    /// Check if error is an inference error.
    pub fn is_inference(&self) -> bool {
        matches!(self, Self::Inference(_))
    }
    
    /// Check if error is a configuration error.
    pub fn is_configuration(&self) -> bool {
        matches!(self, Self::Configuration(_))
    }
    
    /// Check if error is an invalid input error.
    pub fn is_invalid_input(&self) -> bool {
        matches!(self, Self::InvalidInput(_))
    }
    
    /// Get error message as string.
    pub fn message(&self) -> String {
        format!("{}", self)
    }
    
    /// Convert to a user-friendly error message.
    pub fn user_message(&self) -> String {
        match self {
            Self::ModelLoad(msg) => format!("Failed to load model: {}", msg),
            Self::Inference(msg) => format!("Inference error: {}", msg),
            Self::Tokenization(msg) => format!("Tokenization error: {}", msg),
            Self::Configuration(msg) => format!("Configuration error: {}", msg),
            Self::DataProcessing(msg) => format!("Data processing error: {}", msg),
            Self::Metrics(msg) => format!("Metrics error: {}", msg),
            Self::Io(e) => format!("I/O error: {}", e),
            Self::Serialization(msg) => format!("Serialization error: {}", msg),
            Self::InvalidInput(msg) => format!("Invalid input: {}", msg),
            Self::NotFound(msg) => format!("Not found: {}", msg),
            Self::Other(msg) => msg.clone(),
        }
    }
}

impl From<anyhow::Error> for BenchmarkError {
    fn from(err: anyhow::Error) -> Self {
        Self::Other(err.to_string())
    }
}

impl From<serde_json::Error> for BenchmarkError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}

// Additional error conversions can be added here when needed
// For example:
// impl From<serde_yaml::Error> for BenchmarkError { ... }
// impl From<toml::de::Error> for BenchmarkError { ... }
// impl From<url::ParseError> for BenchmarkError { ... }
