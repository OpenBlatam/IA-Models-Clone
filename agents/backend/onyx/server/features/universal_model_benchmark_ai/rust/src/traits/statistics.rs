//! Statistics Traits
//!
//! Additional traits for statistical operations.

/// Trait for types that can calculate percentiles.
pub trait Percentiles {
    /// Calculate percentile value.
    fn percentile(&self, p: f64) -> Option<f64>;
    
    /// Calculate multiple percentiles.
    fn percentiles(&self, ps: &[f64]) -> Vec<Option<f64>> {
        ps.iter().map(|&p| self.percentile(p)).collect()
    }
    
    /// Get P50 (median).
    fn p50(&self) -> Option<f64> {
        self.percentile(50.0)
    }
    
    /// Get P95.
    fn p95(&self) -> Option<f64> {
        self.percentile(95.0)
    }
    
    /// Get P99.
    fn p99(&self) -> Option<f64> {
        self.percentile(99.0)
    }
}

/// Trait for types that can calculate distribution statistics.
pub trait Distribution {
    /// Calculate mean.
    fn mean(&self) -> f64;
    
    /// Calculate variance.
    fn variance(&self) -> f64;
    
    /// Calculate standard deviation.
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    
    /// Calculate coefficient of variation.
    fn cv(&self) -> f64 {
        let mean = self.mean();
        if mean == 0.0 {
            return 0.0;
        }
        self.std_dev() / mean
    }
}




