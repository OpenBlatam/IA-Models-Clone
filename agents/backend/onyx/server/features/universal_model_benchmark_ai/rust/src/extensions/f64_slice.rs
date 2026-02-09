//! F64 Slice Extensions
//!
//! Extension trait for slices of f64.

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
}




