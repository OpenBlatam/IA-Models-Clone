//! Safety utilities and error handling helpers.
//!
//! Provides:
//! - Safe unwrap alternatives
//! - Result helpers
//! - Validation utilities
//! - Error context builders

use crate::error::{BenchmarkError, Result};

/// Safely unwrap a lock, converting poison errors to benchmark errors.
pub fn safe_lock<T>(lock_result: std::sync::LockResult<T>) -> Result<T> {
    lock_result.map_err(|_| BenchmarkError::Other("Lock poisoned".to_string()))
}

/// Safely unwrap a RwLock read, converting poison errors to benchmark errors.
pub fn safe_read_lock<T>(lock_result: std::sync::LockResult<T>) -> Result<T> {
    lock_result.map_err(|_| BenchmarkError::Other("RwLock read poisoned".to_string()))
}

/// Safely unwrap a RwLock write, converting poison errors to benchmark errors.
pub fn safe_write_lock<T>(lock_result: std::sync::LockResult<T>) -> Result<T> {
    lock_result.map_err(|_| BenchmarkError::Other("RwLock write poisoned".to_string()))
}

/// Safely unwrap an Option, converting None to an error.
pub fn safe_unwrap<T>(opt: Option<T>, error_msg: impl Into<String>) -> Result<T> {
    opt.ok_or_else(|| BenchmarkError::Other(error_msg.into()))
}

/// Safely unwrap a Result, converting errors to BenchmarkError.
pub fn safe_result<T, E: std::fmt::Display>(result: std::result::Result<T, E>) -> Result<T> {
    result.map_err(|e| BenchmarkError::Other(e.to_string()))
}

/// Validate that a value is within a range.
pub fn validate_range<T: PartialOrd>(
    value: T,
    min: T,
    max: T,
    field_name: &str,
) -> Result<()> {
    if value < min || value > max {
        Err(BenchmarkError::invalid_input(format!(
            "{} must be between {:?} and {:?}, got {:?}",
            field_name, min, max, value
        )))
    } else {
        Ok(())
    }
}

/// Validate that a value is positive.
pub fn validate_positive<T: PartialOrd + Default>(
    value: T,
    field_name: &str,
) -> Result<()> {
    if value <= T::default() {
        Err(BenchmarkError::invalid_input(format!(
            "{} must be positive, got {:?}",
            field_name, value
        )))
    } else {
        Ok(())
    }
}

/// Validate that a value is non-negative.
pub fn validate_non_negative<T: PartialOrd + Default>(
    value: T,
    field_name: &str,
) -> Result<()> {
    if value < T::default() {
        Err(BenchmarkError::invalid_input(format!(
            "{} must be non-negative, got {:?}",
            field_name, value
        )))
    } else {
        Ok(())
    }
}

/// Validate that a string is not empty.
pub fn validate_non_empty_string(s: &str, field_name: &str) -> Result<()> {
    if s.is_empty() {
        Err(BenchmarkError::invalid_input(format!(
            "{} cannot be empty",
            field_name
        )))
    } else {
        Ok(())
    }
}

/// Validate that a slice is not empty.
pub fn validate_non_empty_slice<T>(slice: &[T], field_name: &str) -> Result<()> {
    if slice.is_empty() {
        Err(BenchmarkError::invalid_input(format!(
            "{} cannot be empty",
            field_name
        )))
    } else {
        Ok(())
    }
}

/// Validate that a number is finite (not NaN or Infinity).
pub fn validate_finite(value: f64, field_name: &str) -> Result<()> {
    if !value.is_finite() {
        Err(BenchmarkError::invalid_input(format!(
            "{} must be finite, got {}",
            field_name, value
        )))
    } else {
        Ok(())
    }
}

/// Context builder for adding error context.
pub struct ErrorContext {
    context: String,
}

impl ErrorContext {
    /// Create a new error context.
    pub fn new(context: impl Into<String>) -> Self {
        Self {
            context: context.into(),
        }
    }
    
    /// Add additional context.
    pub fn with(mut self, additional: impl Into<String>) -> Self {
        self.context.push_str(": ");
        self.context.push_str(&additional.into());
        self
    }
    
    /// Convert to BenchmarkError.
    pub fn into_error(self) -> BenchmarkError {
        BenchmarkError::Other(self.context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validate_range() {
        assert!(validate_range(5, 0, 10, "value").is_ok());
        assert!(validate_range(15, 0, 10, "value").is_err());
    }
    
    #[test]
    fn test_validate_positive() {
        assert!(validate_positive(5, "value").is_ok());
        assert!(validate_positive(0, "value").is_err());
        assert!(validate_positive(-5, "value").is_err());
    }
    
    #[test]
    fn test_validate_finite() {
        assert!(validate_finite(5.0, "value").is_ok());
        assert!(validate_finite(f64::NAN, "value").is_err());
        assert!(validate_finite(f64::INFINITY, "value").is_err());
    }
}












