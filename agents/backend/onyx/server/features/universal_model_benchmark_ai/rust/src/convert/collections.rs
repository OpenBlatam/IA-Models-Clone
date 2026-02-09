//! Collection Conversions
//!
//! Conversion functions for collections.

use crate::error::{Result, BenchmarkError};

/// Convert Option<T> to Result<T> with custom error message.
pub fn option_to_result<T>(opt: Option<T>, msg: &str) -> Result<T> {
    opt.ok_or_else(|| BenchmarkError::Other(msg.to_string()))
}

/// Convert Vec<T> to array of fixed size.
pub fn vec_to_array<T, const N: usize>(vec: Vec<T>) -> Result<[T; N]> {
    if vec.len() != N {
        return Err(BenchmarkError::invalid_input(
            format!("Vector length {} does not match array size {}", vec.len(), N)
        ));
    }
    vec.try_into()
        .map_err(|_| BenchmarkError::Other("Failed to convert vector to array".to_string()))
}

/// Convert array to Vec.
pub fn array_to_vec<T, const N: usize>(arr: [T; N]) -> Vec<T> {
    arr.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_option_to_result() {
        assert!(option_to_result(Some(42), "error").is_ok());
        assert!(option_to_result(None::<i32>, "error").is_err());
    }
    
    #[test]
    fn test_vec_to_array() {
        let vec = vec![1, 2, 3];
        let arr: [i32; 3] = vec_to_array(vec).unwrap();
        assert_eq!(arr, [1, 2, 3]);
        
        let vec = vec![1, 2];
        assert!(vec_to_array::<i32, 3>(vec).is_err());
    }
}




