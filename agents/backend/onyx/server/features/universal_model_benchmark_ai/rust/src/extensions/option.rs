//! Option Extensions
//!
//! Extension trait for Option.

use crate::error::{Result, BenchmarkError};

/// Extension trait for Option.
pub trait OptionExt<T> {
    /// Convert to Result with error message.
    fn ok_or_err(self, msg: &str) -> Result<T>;
    
    /// Convert to Result with error function.
    fn ok_or_err_with<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
    
    /// Convert to Result with BenchmarkError.
    fn ok_or_benchmark_err(self, error: BenchmarkError) -> Result<T>;
}

impl<T> OptionExt<T> for Option<T> {
    fn ok_or_err(self, msg: &str) -> Result<T> {
        self.ok_or_else(|| BenchmarkError::Other(msg.to_string()))
    }
    
    fn ok_or_err_with<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.ok_or_else(|| BenchmarkError::Other(f()))
    }
    
    fn ok_or_benchmark_err(self, error: BenchmarkError) -> Result<T> {
        self.ok_or(Err(error))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_option_ext() {
        let some = Some(42);
        assert_eq!(some.ok_or_err("error").unwrap(), 42);
        
        let none: Option<i32> = None;
        assert!(none.ok_or_err("error").is_err());
    }
}




