//! Numeric Conversions
//!
//! Conversion functions for numeric types.

use crate::error::{Result, BenchmarkError};

/// Convert f64 to usize safely.
pub fn f64_to_usize(value: f64) -> Result<usize> {
    if value < 0.0 {
        return Err(BenchmarkError::invalid_input(
            format!("Cannot convert negative f64 to usize: {}", value)
        ));
    }
    if value > usize::MAX as f64 {
        return Err(BenchmarkError::invalid_input(
            format!("f64 value too large for usize: {}", value)
        ));
    }
    Ok(value as usize)
}

/// Convert usize to f64.
pub fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

/// Convert i64 to f64.
pub fn i64_to_f64(value: i64) -> f64 {
    value as f64
}

/// Convert f64 to i64 safely.
pub fn f64_to_i64(value: f64) -> Result<i64> {
    if value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err(BenchmarkError::invalid_input(
            format!("f64 value out of range for i64: {}", value)
        ));
    }
    Ok(value as i64)
}

/// Convert u32 to f64.
pub fn u32_to_f64(value: u32) -> f64 {
    value as f64
}

/// Convert f64 to u32 safely.
pub fn f64_to_u32(value: f64) -> Result<u32> {
    if value < 0.0 || value > u32::MAX as f64 {
        return Err(BenchmarkError::invalid_input(
            format!("f64 value out of range for u32: {}", value)
        ));
    }
    Ok(value as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_f64_to_usize() {
        assert_eq!(f64_to_usize(42.0).unwrap(), 42);
        assert!(f64_to_usize(-1.0).is_err());
    }
    
    #[test]
    fn test_f64_to_i64() {
        assert_eq!(f64_to_i64(42.0).unwrap(), 42);
        assert!(f64_to_i64(i64::MAX as f64 + 1.0).is_err());
    }
}




