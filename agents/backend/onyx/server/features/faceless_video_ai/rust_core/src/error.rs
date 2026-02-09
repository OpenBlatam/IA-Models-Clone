//! Módulo de errores personalizados para Faceless Video Core
//! 
//! Proporciona tipos de error específicos y conversiones a errores de Python

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError, PyIOError};
use thiserror::Error;

/// Tipo de resultado para operaciones del core
pub type CoreResult<T> = Result<T, CoreError>;

/// Errores principales del sistema
#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Video processing error: {0}")]
    VideoProcessing(String),

    #[error("Image processing error: {0}")]
    ImageProcessing(String),

    #[error("Encryption error: {0}")]
    Encryption(String),

    #[error("Decryption error: {0}")]
    Decryption(String),

    #[error("Text processing error: {0}")]
    TextProcessing(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("FFmpeg error: {0}")]
    FFmpeg(String),

    #[error("Batch processing error: {0}")]
    BatchProcessing(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Timeout error: operation exceeded {0} seconds")]
    Timeout(u64),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<CoreError> for PyErr {
    fn from(err: CoreError) -> PyErr {
        match err {
            CoreError::VideoProcessing(msg) => PyRuntimeError::new_err(msg),
            CoreError::ImageProcessing(msg) => PyRuntimeError::new_err(msg),
            CoreError::Encryption(msg) => PyRuntimeError::new_err(msg),
            CoreError::Decryption(msg) => PyRuntimeError::new_err(msg),
            CoreError::TextProcessing(msg) => PyRuntimeError::new_err(msg),
            CoreError::InvalidConfig(msg) => PyValueError::new_err(msg),
            CoreError::FileNotFound(msg) => PyIOError::new_err(msg),
            CoreError::Io(err) => PyIOError::new_err(err.to_string()),
            CoreError::Serialization(msg) => PyValueError::new_err(msg),
            CoreError::FFmpeg(msg) => PyRuntimeError::new_err(msg),
            CoreError::BatchProcessing(msg) => PyRuntimeError::new_err(msg),
            CoreError::InvalidInput(msg) => PyValueError::new_err(msg),
            CoreError::Timeout(secs) => PyRuntimeError::new_err(format!("Timeout after {} seconds", secs)),
            CoreError::ResourceExhausted(msg) => PyRuntimeError::new_err(msg),
            CoreError::Internal(msg) => PyRuntimeError::new_err(msg),
        }
    }
}

impl From<image::ImageError> for CoreError {
    fn from(err: image::ImageError) -> Self {
        CoreError::ImageProcessing(err.to_string())
    }
}

impl From<serde_json::Error> for CoreError {
    fn from(err: serde_json::Error) -> Self {
        CoreError::Serialization(err.to_string())
    }
}

impl From<regex::Error> for CoreError {
    fn from(err: regex::Error) -> Self {
        CoreError::TextProcessing(err.to_string())
    }
}




