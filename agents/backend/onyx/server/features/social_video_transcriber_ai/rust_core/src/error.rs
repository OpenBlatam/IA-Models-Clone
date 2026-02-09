//! Error types for Transcriber Core

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TranscriberError {
    #[error("Text processing error: {0}")]
    TextProcessing(String),

    #[error("Search error: {0}")]
    Search(String),

    #[error("Cache error: {0}")]
    Cache(String),

    #[error("Batch processing error: {0}")]
    BatchProcessing(String),

    #[error("Hash error: {0}")]
    Hash(String),

    #[error("Similarity error: {0}")]
    Similarity(String),

    #[error("Language detection error: {0}")]
    LanguageDetection(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Operation timed out")]
    Timeout,
}

impl From<TranscriberError> for PyErr {
    fn from(err: TranscriberError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, TranscriberError>;












