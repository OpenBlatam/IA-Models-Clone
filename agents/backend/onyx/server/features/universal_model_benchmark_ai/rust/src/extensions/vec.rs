//! Vec Extensions
//!
//! Extension trait for Vec and slices.

use crate::error::{Result, BenchmarkError};

/// Extension trait for Vec.
pub trait VecExt<T> {
    /// Check if vector is not empty.
    fn is_not_empty(&self) -> bool;
    
    /// Get first element or error.
    fn first_or_err(&self, msg: &str) -> Result<&T>;
    
    /// Get last element or error.
    fn last_or_err(&self, msg: &str) -> Result<&T>;
    
    /// Get element at index or error.
    fn get_or_err(&self, index: usize, msg: &str) -> Result<&T>;
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
    
    fn get_or_err(&self, index: usize, msg: &str) -> Result<&T> {
        self.get(index)
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
    
    fn get_or_err(&self, index: usize, msg: &str) -> Result<&T> {
        self.get(index)
            .ok_or_else(|| BenchmarkError::Other(msg.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vec_ext() {
        let v = vec![1, 2, 3];
        assert!(v.is_not_empty());
        assert_eq!(*v.first_or_err("empty").unwrap(), 1);
        assert_eq!(*v.last_or_err("empty").unwrap(), 3);
        assert_eq!(*v.get_or_err(1, "out of bounds").unwrap(), 2);
    }
}




