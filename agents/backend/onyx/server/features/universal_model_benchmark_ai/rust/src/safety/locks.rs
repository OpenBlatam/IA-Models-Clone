//! Lock Safety Utilities
//!
//! Safe unwrap alternatives for locks and results.

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

/// Safely unwrap with default value.
pub fn safe_unwrap_or<T>(opt: Option<T>, default: T) -> T {
    opt.unwrap_or(default)
}

/// Safely unwrap or return error.
pub fn safe_unwrap_or_err<T>(opt: Option<T>, error: BenchmarkError) -> Result<T> {
    opt.ok_or(error)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_safe_unwrap() {
        assert!(safe_unwrap(Some(42), "error").is_ok());
        assert!(safe_unwrap(None::<i32>, "error").is_err());
    }
    
    #[test]
    fn test_safe_unwrap_or() {
        assert_eq!(safe_unwrap_or(Some(42), 0), 42);
        assert_eq!(safe_unwrap_or(None, 0), 0);
    }
}




