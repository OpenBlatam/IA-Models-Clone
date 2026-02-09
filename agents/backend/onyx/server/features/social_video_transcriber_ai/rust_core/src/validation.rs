//! Validation utilities
//!
//! Provides validation functions for common use cases.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::constants;

/// Validate that a value is within a range
pub fn validate_range<T: PartialOrd + std::fmt::Display>(
    value: T,
    min: T,
    max: T,
    name: &str,
) -> PyResult<()> {
    if value < min || value > max {
        return Err(PyValueError::new_err(format!(
            "{} must be between {} and {}, got {}",
            name, min, max, value
        )));
    }
    Ok(())
}

/// Validate that a string is not empty
pub fn validate_not_empty(s: &str, name: &str) -> PyResult<()> {
    if s.is_empty() {
        return Err(PyValueError::new_err(format!("{} cannot be empty", name)));
    }
    Ok(())
}

/// Validate that a collection is not empty
pub fn validate_not_empty_collection<T>(items: &[T], name: &str) -> PyResult<()> {
    if items.is_empty() {
        return Err(PyValueError::new_err(format!("{} cannot be empty", name)));
    }
    Ok(())
}

/// Validate that a number is positive
pub fn validate_positive<T: PartialOrd + std::fmt::Display + From<i32>>(
    value: T,
    name: &str,
) -> PyResult<()> {
    if value <= T::from(0) {
        return Err(PyValueError::new_err(format!("{} must be positive, got {}", name, value)));
    }
    Ok(())
}

/// Validate that a number is non-negative
pub fn validate_non_negative<T: PartialOrd + std::fmt::Display + From<i32>>(
    value: T,
    name: &str,
) -> PyResult<()> {
    if value < T::from(0) {
        return Err(PyValueError::new_err(format!("{} must be non-negative, got {}", name, value)));
    }
    Ok(())
}

/// Validate cache size
pub fn validate_cache_size(size: usize) -> PyResult<()> {
    validate_range(
        size,
        constants::MIN_CACHE_SIZE,
        constants::MAX_CACHE_SIZE,
        "cache_size",
    )
    .map_err(|_| PyValueError::new_err(constants::errors::INVALID_CACHE_SIZE))
}

/// Validate TTL
pub fn validate_ttl(ttl: u64) -> PyResult<()> {
    validate_range(ttl, constants::MIN_TTL, constants::MAX_TTL, "ttl")
        .map_err(|_| PyValueError::new_err(constants::errors::INVALID_TTL))
}

/// Validate batch size
pub fn validate_batch_size(size: usize) -> PyResult<()> {
    validate_range(
        size,
        constants::MIN_BATCH_SIZE,
        constants::MAX_BATCH_SIZE,
        "batch_size",
    )
    .map_err(|_| PyValueError::new_err(constants::errors::INVALID_BATCH_SIZE))
}

/// Validate number of workers
pub fn validate_workers(workers: usize) -> PyResult<()> {
    validate_range(
        workers,
        constants::MIN_NUM_WORKERS,
        constants::MAX_NUM_WORKERS,
        "num_workers",
    )
    .map_err(|_| PyValueError::new_err(constants::errors::INVALID_WORKERS))
}

/// Validate compression level
pub fn validate_compression_level(level: u32) -> PyResult<()> {
    validate_range(
        level,
        constants::MIN_COMPRESSION_LEVEL,
        constants::MAX_COMPRESSION_LEVEL,
        "compression_level",
    )
    .map_err(|_| PyValueError::new_err(constants::errors::INVALID_COMPRESSION_LEVEL))
}

/// Validate similarity threshold
pub fn validate_similarity_threshold(threshold: f64) -> PyResult<()> {
    validate_range(
        threshold,
        constants::MIN_SIMILARITY_THRESHOLD,
        constants::MAX_SIMILARITY_THRESHOLD,
        "similarity_threshold",
    )
    .map_err(|_| PyValueError::new_err(constants::errors::INVALID_SIMILARITY_THRESHOLD))
}

#[pyclass]
pub struct Validator;

#[pymethods]
impl Validator {
    #[new]
    pub fn new() -> Self {
        Self
    }

    pub fn validate_range(&self, value: f64, min: f64, max: f64, name: String) -> PyResult<()> {
        validate_range(value, min, max, &name)
    }

    pub fn validate_not_empty(&self, s: String, name: String) -> PyResult<()> {
        validate_not_empty(&s, &name)
    }

    pub fn validate_positive(&self, value: f64, name: String) -> PyResult<()> {
        validate_positive(value, &name)
    }

    pub fn validate_cache_size(&self, size: usize) -> PyResult<()> {
        validate_cache_size(size)
    }

    pub fn validate_ttl(&self, ttl: u64) -> PyResult<()> {
        validate_ttl(ttl)
    }

    pub fn validate_batch_size(&self, size: usize) -> PyResult<()> {
        validate_batch_size(size)
    }

    pub fn validate_workers(&self, workers: usize) -> PyResult<()> {
        validate_workers(workers)
    }

    pub fn validate_compression_level(&self, level: u32) -> PyResult<()> {
        validate_compression_level(level)
    }
}

#[pyfunction]
pub fn create_validator() -> Validator {
    Validator::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_range() {
        assert!(validate_range(5, 1, 10, "value").is_ok());
        assert!(validate_range(0, 1, 10, "value").is_err());
        assert!(validate_range(11, 1, 10, "value").is_err());
    }

    #[test]
    fn test_validate_not_empty() {
        assert!(validate_not_empty("test", "value").is_ok());
        assert!(validate_not_empty("", "value").is_err());
    }

    #[test]
    fn test_validate_positive() {
        assert!(validate_positive(1, "value").is_ok());
        assert!(validate_positive(0, "value").is_err());
        assert!(validate_positive(-1, "value").is_err());
    }
}

