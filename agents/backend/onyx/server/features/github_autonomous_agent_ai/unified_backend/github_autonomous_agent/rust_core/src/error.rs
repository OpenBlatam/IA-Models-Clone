//! Error types for Agent Core
//!
//! Definiciones de errores personalizados para el módulo Rust.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

/// Errores del procesador de lotes
#[derive(Error, Debug)]
pub enum BatchError {
    #[error("Invalid batch size: {0}")]
    InvalidBatchSize(usize),

    #[error("Invalid concurrency limit: {0}")]
    InvalidConcurrency(usize),

    #[error("Task processing failed: {0}")]
    ProcessingFailed(String),

    #[error("Batch timeout exceeded after {0}ms")]
    Timeout(u64),

    #[error("Queue is full, max capacity: {0}")]
    QueueFull(usize),
}

/// Errores del servicio de caché
#[derive(Error, Debug)]
pub enum CacheError {
    #[error("Invalid cache key: {0}")]
    InvalidKey(String),

    #[error("Invalid TTL value: {0}")]
    InvalidTTL(i64),

    #[error("Cache capacity exceeded: {0}")]
    CapacityExceeded(usize),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Entry not found: {0}")]
    NotFound(String),
}

/// Errores del motor de búsqueda
#[derive(Error, Debug)]
pub enum SearchError {
    #[error("Invalid regex pattern: {0}")]
    InvalidRegex(String),

    #[error("Invalid field name: {0}")]
    InvalidField(String),

    #[error("Invalid operator: {0}")]
    InvalidOperator(String),

    #[error("Search timeout after {0}ms")]
    Timeout(u64),

    #[error("Invalid search query: {0}")]
    InvalidQuery(String),

    #[error("Internal search error: {0}")]
    Internal(String),
}

/// Errores del procesador de texto
#[derive(Error, Debug)]
pub enum TextError {
    #[error("Invalid instruction format: {0}")]
    InvalidInstruction(String),

    #[error("Parse error at position {position}: {message}")]
    ParseError { position: usize, message: String },

    #[error("Empty instruction")]
    EmptyInstruction,

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Errores de la cola de tareas
#[derive(Error, Debug)]
pub enum QueueError {
    #[error("Queue is empty")]
    Empty,

    #[error("Invalid priority: {0}")]
    InvalidPriority(i32),

    #[error("Task not found: {0}")]
    TaskNotFound(String),

    #[error("Queue capacity exceeded: max {0}")]
    CapacityExceeded(usize),
}

/// Errores de hashing/crypto
#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("Invalid input for hashing")]
    InvalidInput,

    #[error("Hash algorithm not supported: {0}")]
    UnsupportedAlgorithm(String),

    #[error("Encoding error: {0}")]
    EncodingError(String),
}

/// Error genérico del módulo
#[derive(Error, Debug)]
pub enum AgentCoreError {
    #[error("Batch error: {0}")]
    Batch(#[from] BatchError),

    #[error("Cache error: {0}")]
    Cache(#[from] CacheError),

    #[error("Search error: {0}")]
    Search(#[from] SearchError),

    #[error("Text error: {0}")]
    Text(#[from] TextError),

    #[error("Queue error: {0}")]
    Queue(#[from] QueueError),

    #[error("Crypto error: {0}")]
    Crypto(#[from] CryptoError),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<BatchError> for PyErr {
    fn from(err: BatchError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

impl From<CacheError> for PyErr {
    fn from(err: CacheError) -> PyErr {
        match err {
            CacheError::InvalidKey(_) | CacheError::InvalidTTL(_) => {
                PyValueError::new_err(err.to_string())
            }
            _ => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

impl From<SearchError> for PyErr {
    fn from(err: SearchError) -> PyErr {
        match err {
            SearchError::InvalidRegex(_)
            | SearchError::InvalidField(_)
            | SearchError::InvalidOperator(_)
            | SearchError::InvalidQuery(_) => PyValueError::new_err(err.to_string()),
            _ => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

impl From<TextError> for PyErr {
    fn from(err: TextError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

impl From<QueueError> for PyErr {
    fn from(err: QueueError) -> PyErr {
        match err {
            QueueError::InvalidPriority(_) => PyValueError::new_err(err.to_string()),
            _ => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

impl From<CryptoError> for PyErr {
    fn from(err: CryptoError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

impl From<AgentCoreError> for PyErr {
    fn from(err: AgentCoreError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

