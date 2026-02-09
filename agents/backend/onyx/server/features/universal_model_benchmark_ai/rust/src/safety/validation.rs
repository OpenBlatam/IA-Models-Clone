//! Validation Utilities
//!
//! Validation functions for common types and values.

use crate::error::{BenchmarkError, Result};

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

/// Validate that a value is not null/None.
pub fn validate_not_null<T>(value: Option<T>, field_name: &str) -> Result<T> {
    value.ok_or_else(|| BenchmarkError::invalid_input(format!(
        "{} cannot be null",
        field_name
    )))
}

/// Validate that a value matches a predicate.
pub fn validate_predicate<T, F>(
    value: T,
    predicate: F,
    field_name: &str,
    error_msg: &str,
) -> Result<()>
where
    F: FnOnce(&T) -> bool,
{
    if predicate(&value) {
        Ok(())
    } else {
        Err(BenchmarkError::invalid_input(format!(
            "{}: {}",
            field_name, error_msg
        )))
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
    
    #[test]
    fn test_validate_predicate() {
        assert!(validate_predicate(5, |x| *x > 0, "value", "must be positive").is_ok());
        assert!(validate_predicate(-5, |x| *x > 0, "value", "must be positive").is_err());
    }
}




