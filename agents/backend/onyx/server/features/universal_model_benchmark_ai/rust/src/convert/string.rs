//! String Conversions
//!
//! Conversion functions for string types.

use crate::error::{Result, BenchmarkError};

/// Convert string to number with error handling.
pub fn str_to_f64(s: &str) -> Result<f64> {
    s.parse::<f64>()
        .map_err(|e| BenchmarkError::invalid_input(
            format!("Failed to parse '{}' as f64: {}", s, e)
        ))
}

/// Convert string to usize with error handling.
pub fn str_to_usize(s: &str) -> Result<usize> {
    s.parse::<usize>()
        .map_err(|e| BenchmarkError::invalid_input(
            format!("Failed to parse '{}' as usize: {}", s, e)
        ))
}

/// Convert string to i64 with error handling.
pub fn str_to_i64(s: &str) -> Result<i64> {
    s.parse::<i64>()
        .map_err(|e| BenchmarkError::invalid_input(
            format!("Failed to parse '{}' as i64: {}", s, e)
        ))
}

/// Convert string to u32 with error handling.
pub fn str_to_u32(s: &str) -> Result<u32> {
    s.parse::<u32>()
        .map_err(|e| BenchmarkError::invalid_input(
            format!("Failed to parse '{}' as u32: {}", s, e)
        ))
}

/// Convert string to bool with error handling.
pub fn str_to_bool(s: &str) -> Result<bool> {
    match s.to_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(BenchmarkError::invalid_input(
            format!("Failed to parse '{}' as bool", s)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_str_to_f64() {
        assert_eq!(str_to_f64("42.5").unwrap(), 42.5);
        assert!(str_to_f64("invalid").is_err());
    }
    
    #[test]
    fn test_str_to_bool() {
        assert_eq!(str_to_bool("true").unwrap(), true);
        assert_eq!(str_to_bool("false").unwrap(), false);
        assert!(str_to_bool("maybe").is_err());
    }
}




