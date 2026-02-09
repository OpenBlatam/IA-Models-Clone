use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VideoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Processing failed: {0}")]
    ProcessingFailed(String),
    
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),
}

impl From<VideoError> for PyErr {
    fn from(err: VideoError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}
