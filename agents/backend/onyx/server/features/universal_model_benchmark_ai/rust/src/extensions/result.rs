//! Result Extensions
//!
//! Extension trait for Result types.

use crate::error::{Result, BenchmarkError};

/// Extension trait for Result types.
pub trait ResultExt<T> {
    /// Convert to Option, logging error.
    fn ok_or_log(self, msg: &str) -> Option<T>;
    
    /// Map error with context.
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
    
    /// Map error with message.
    fn map_err_msg(self, msg: &str) -> Result<T>;
}

impl<T, E> ResultExt<T> for std::result::Result<T, E>
where
    E: std::fmt::Display,
{
    fn ok_or_log(self, msg: &str) -> Option<T> {
        match self {
            Ok(val) => Some(val),
            Err(e) => {
                eprintln!("{}: {}", msg, e);
                None
            }
        }
    }
    
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| BenchmarkError::Other(format!("{}: {}", f(), e)))
    }
    
    fn map_err_msg(self, msg: &str) -> Result<T> {
        self.map_err(|e| BenchmarkError::Other(format!("{}: {}", msg, e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_result_ext() {
        let ok: std::result::Result<i32, &str> = Ok(42);
        assert_eq!(ok.ok_or_log("error"), Some(42));
        
        let err: std::result::Result<i32, &str> = Err("failed");
        assert_eq!(err.ok_or_log("error"), None);
    }
}




