//! Error Module - Unified Error Handling
//!
//! Provides consistent error types that map to Python exceptions

use pyo3::prelude::*;
use pyo3::exceptions::{PyException, PyRuntimeError, PyValueError, PyIOError};
use thiserror::Error;

/// Core error type for the Rust library
#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Compression error: {0}")]
    Compression(String),
    
    #[error("Cryptography error: {0}")]
    Crypto(String),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Batch processing error: {0}")]
    Batch(String),
    
    #[error("ID generation error: {0}")]
    IdGeneration(String),
    
    #[error("Text processing error: {0}")]
    TextProcessing(String),
    
    #[error("IO error: {0}")]
    Io(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Timeout error: {0}")]
    Timeout(String),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl CoreError {
    /// Create a compression error
    pub fn compression_error(msg: String) -> Self {
        CoreError::Compression(msg)
    }
    
    /// Create a crypto error
    pub fn crypto_error(msg: String) -> Self {
        CoreError::Crypto(msg)
    }
    
    /// Create a serialization error
    pub fn serialization_error(msg: String) -> Self {
        CoreError::Serialization(msg)
    }
    
    /// Create a batch error
    pub fn batch_error(msg: String) -> Self {
        CoreError::Batch(msg)
    }
    
    /// Create an ID error
    pub fn id_error(msg: String) -> Self {
        CoreError::IdGeneration(msg)
    }
    
    /// Create a text processing error
    pub fn text_error(msg: String) -> Self {
        CoreError::TextProcessing(msg)
    }
    
    /// Create an IO error
    pub fn io_error(msg: String) -> Self {
        CoreError::Io(msg)
    }
    
    /// Create a validation error
    pub fn validation_error(msg: String) -> Self {
        CoreError::Validation(msg)
    }
    
    /// Create a configuration error
    pub fn config_error(msg: String) -> Self {
        CoreError::Configuration(msg)
    }
    
    /// Create a timeout error
    pub fn timeout_error(msg: String) -> Self {
        CoreError::Timeout(msg)
    }
}

/// Convert CoreError to PyErr
impl From<CoreError> for PyErr {
    fn from(err: CoreError) -> PyErr {
        match &err {
            CoreError::Compression(msg) => PyRuntimeError::new_err(format!("CompressionError: {}", msg)),
            CoreError::Crypto(msg) => PyRuntimeError::new_err(format!("CryptoError: {}", msg)),
            CoreError::Serialization(msg) => PyValueError::new_err(format!("SerializationError: {}", msg)),
            CoreError::Batch(msg) => PyRuntimeError::new_err(format!("BatchError: {}", msg)),
            CoreError::IdGeneration(msg) => PyValueError::new_err(format!("IdError: {}", msg)),
            CoreError::TextProcessing(msg) => PyValueError::new_err(format!("TextError: {}", msg)),
            CoreError::Io(msg) => PyIOError::new_err(format!("IOError: {}", msg)),
            CoreError::Validation(msg) => PyValueError::new_err(format!("ValidationError: {}", msg)),
            CoreError::Configuration(msg) => PyValueError::new_err(format!("ConfigError: {}", msg)),
            CoreError::Timeout(msg) => PyRuntimeError::new_err(format!("TimeoutError: {}", msg)),
            CoreError::Unknown(msg) => PyException::new_err(format!("UnknownError: {}", msg)),
        }
    }
}

/// Result type alias for convenience
pub type CoreResult<T> = Result<T, CoreError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CoreError::compression_error("test error".to_string());
        assert!(err.to_string().contains("Compression error"));
    }
}
