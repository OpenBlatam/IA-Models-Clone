//! Extension traits for common types.
//!
//! Provides convenient methods for working with common types.

use crate::error::{Result, BenchmarkError};
use crate::types::Metrics;

/// Extension trait for slices of f64.
pub trait F64SliceExt {
    /// Calculate mean.
    fn mean(&self) -> f64;
    
    /// Calculate variance.
    fn variance(&self) -> f64;
    
    /// Calculate standard deviation.
    fn std_dev(&self) -> f64;
    
    /// Calculate median.
    fn median(&self) -> f64;
    
    /// Calculate min.
    fn min_value(&self) -> f64;
    
    /// Calculate max.
    fn max_value(&self) -> f64;
    
    /// Calculate sum.
    fn sum(&self) -> f64;
    
    /// Check if all values are finite.
    fn all_finite(&self) -> bool;
    
    /// Check if all values are positive.
    fn all_positive(&self) -> bool;
    
    /// Normalize values to [0, 1] range.
    fn normalize(&self) -> Vec<f64>;
}

impl F64SliceExt for [f64] {
    fn mean(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        self.sum() / self.len() as f64
    }
    
    fn variance(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        let mean = self.mean();
        let sum_sq_diff: f64 = self.iter()
            .map(|&x| (x - mean).powi(2))
            .sum();
        sum_sq_diff / self.len() as f64
    }
    
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    
    fn median(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        let mut sorted = self.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }
    
    fn min_value(&self) -> f64 {
        self.iter()
            .copied()
            .fold(f64::INFINITY, |a, b| a.min(b))
    }
    
    fn max_value(&self) -> f64 {
        self.iter()
            .copied()
            .fold(f64::NEG_INFINITY, |a, b| a.max(b))
    }
    
    fn sum(&self) -> f64 {
        self.iter().sum()
    }
    
    fn all_finite(&self) -> bool {
        self.iter().all(|&x| x.is_finite())
    }
    
    fn all_positive(&self) -> bool {
        self.iter().all(|&x| x > 0.0)
    }
    
    fn normalize(&self) -> Vec<f64> {
        if self.is_empty() {
            return Vec::new();
        }
        let min = self.min_value();
        let max = self.max_value();
        let range = max - min;
        
        if range == 0.0 {
            return vec![0.0; self.len()];
        }
        
        self.iter()
            .map(|&x| (x - min) / range)
            .collect()
    }
}

/// Extension trait for Result types.
pub trait ResultExt<T> {
    /// Convert to Option, logging error.
    fn ok_or_log(self, msg: &str) -> Option<T>;
    
    /// Map error with context.
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
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
}

/// Extension trait for String.
pub trait StringExt {
    /// Check if string is not empty.
    fn is_not_empty(&self) -> bool;
    
    /// Truncate to max length with ellipsis.
    fn truncate_with_ellipsis(&self, max_len: usize) -> String;
    
    /// Remove whitespace from both ends.
    fn trim_whitespace(&self) -> String;
}

impl StringExt for String {
    fn is_not_empty(&self) -> bool {
        !self.is_empty()
    }
    
    fn truncate_with_ellipsis(&self, max_len: usize) -> String {
        if self.len() <= max_len {
            return self.clone();
        }
        format!("{}...", &self[..max_len.saturating_sub(3)])
    }
    
    fn trim_whitespace(&self) -> String {
        self.trim().to_string()
    }
}

impl StringExt for &str {
    fn is_not_empty(&self) -> bool {
        !self.is_empty()
    }
    
    fn truncate_with_ellipsis(&self, max_len: usize) -> String {
        if self.len() <= max_len {
            return self.to_string();
        }
        format!("{}...", &self[..max_len.saturating_sub(3)])
    }
    
    fn trim_whitespace(&self) -> String {
        self.trim().to_string()
    }
}

/// Extension trait for Vec.
pub trait VecExt<T> {
    /// Check if vector is not empty.
    fn is_not_empty(&self) -> bool;
    
    /// Get first element or error.
    fn first_or_err(&self, msg: &str) -> Result<&T>;
    
    /// Get last element or error.
    fn last_or_err(&self, msg: &str) -> Result<&T>;
}

impl<T> VecExt<T> for Vec<T> {
    fn is_not_empty(&self) -> bool {
        !self.is_empty()
    }
    
    fn first_or_err(&self, msg: &str) -> Result<&T> {
        self.first()
            .ok_or_else(|| BenchmarkError::Other(msg.to_string()))
    }
    
    fn last_or_err(&self, msg: &str) -> Result<&T> {
        self.last()
            .ok_or_else(|| BenchmarkError::Other(msg.to_string()))
    }
}

impl<T> VecExt<T> for [T] {
    fn is_not_empty(&self) -> bool {
        !self.is_empty()
    }
    
    fn first_or_err(&self, msg: &str) -> Result<&T> {
        self.first()
            .ok_or_else(|| BenchmarkError::Other(msg.to_string()))
    }
    
    fn last_or_err(&self, msg: &str) -> Result<&T> {
        self.last()
            .ok_or_else(|| BenchmarkError::Other(msg.to_string()))
    }
}

/// Extension trait for Option.
pub trait OptionExt<T> {
    /// Convert to Result with error message.
    fn ok_or_err(self, msg: &str) -> Result<T>;
    
    /// Convert to Result with error function.
    fn ok_or_err_with<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_f64_slice_ext() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(data.mean(), 3.0);
        assert_eq!(data.median(), 3.0);
        assert_eq!(data.min_value(), 1.0);
        assert_eq!(data.max_value(), 5.0);
        assert_eq!(data.sum(), 15.0);
        assert!(data.all_finite());
        assert!(data.all_positive());
    }
    
    #[test]
    fn test_string_ext() {
        let s = "  hello world  ";
        assert!(s.is_not_empty());
        assert_eq!(s.truncate_with_ellipsis(5), "he...");
        assert_eq!(s.trim_whitespace(), "hello world");
    }
    
    #[test]
    fn test_vec_ext() {
        let v = vec![1, 2, 3];
        assert!(v.is_not_empty());
        assert_eq!(*v.first_or_err("empty").unwrap(), 1);
        assert_eq!(*v.last_or_err("empty").unwrap(), 3);
    }
    
    #[test]
    fn test_option_ext() {
        let some = Some(42);
        assert_eq!(some.ok_or_err("error").unwrap(), 42);
        
        let none: Option<i32> = None;
        assert!(none.ok_or_err("error").is_err());
    }
}












