//! Utility functions for common operations.
//!
//! Provides:
//! - Formatting utilities
//! - Validation helpers
//! - Math utilities
//! - Range checking

use std::time::Duration;

/// Format duration to human-readable string.
pub fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    
    if secs < 1.0 {
        format!("{:.0}ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{:.2}s", secs)
    } else if secs < 3600.0 {
        format!("{:.1}m", secs / 60.0)
    } else {
        format!("{:.1}h", secs / 3600.0)
    }
}

/// Format bytes to human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB", "PB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    
    format!("{:.2} {}", size, UNITS[unit_idx])
}

/// Calculate percentile from sorted data.
pub fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    
    let index = (p * (sorted_data.len() - 1) as f64).round() as usize;
    sorted_data[index.min(sorted_data.len() - 1)]
}

/// Calculate multiple percentiles.
pub fn percentiles(sorted_data: &[f64], percentiles: &[f64]) -> Vec<f64> {
    percentiles.iter().map(|&p| percentile(sorted_data, p)).collect()
}

/// Measure execution time of a function.
pub fn measure_time<F, R>(f: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = std::time::Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    (result, elapsed)
}

/// Clamp value to range.
pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

/// Check if value is in range.
pub fn in_range(value: f64, min: f64, max: f64) -> bool {
    value >= min && value <= max
}

/// Calculate mean of a slice of values.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Calculate variance of a slice of values.
pub fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean_val = mean(values);
    let sum_squared_diff: f64 = values.iter()
        .map(|x| (x - mean_val).powi(2))
        .sum();
    sum_squared_diff / values.len() as f64
}

/// Calculate standard deviation of a slice of values.
pub fn std_dev(values: &[f64]) -> f64 {
    variance(values).sqrt()
}

/// Calculate median of a slice of values.
pub fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Normalize value to range [0, 1] based on min and max.
pub fn normalize(value: f64, min: f64, max: f64) -> f64 {
    if max == min {
        return 0.0;
    }
    (value - min) / (max - min)
}

/// Denormalize value from range [0, 1] to [min, max].
pub fn denormalize(value: f64, min: f64, max: f64) -> f64 {
    value * (max - min) + min
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_millis(500)), "500ms");
        assert_eq!(format_duration(Duration::from_secs(30)), "30.00s");
    }
    
    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
    }
    
    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&data, 0.5), 3.0);
        assert_eq!(percentile(&data, 0.0), 1.0);
        assert_eq!(percentile(&data, 1.0), 5.0);
    }
    
    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(clamp(-5.0, 0.0, 10.0), 0.0);
        assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
    }
    
    #[test]
    fn test_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&values), 3.0);
        assert_eq!(mean(&[]), 0.0);
    }
    
    #[test]
    fn test_variance() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let var = variance(&values);
        assert!(var > 0.0);
    }
    
    #[test]
    fn test_std_dev() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std = std_dev(&values);
        assert!(std > 0.0);
    }
    
    #[test]
    fn test_median() {
        let mut values = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(median(&mut values), 3.0);
        
        let mut even = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&mut even), 2.5);
    }
    
    #[test]
    fn test_normalize() {
        assert_eq!(normalize(5.0, 0.0, 10.0), 0.5);
        assert_eq!(normalize(0.0, 0.0, 10.0), 0.0);
        assert_eq!(normalize(10.0, 0.0, 10.0), 1.0);
    }
    
    #[test]
    fn test_denormalize() {
        assert_eq!(denormalize(0.5, 0.0, 10.0), 5.0);
        assert_eq!(denormalize(0.0, 0.0, 10.0), 0.0);
        assert_eq!(denormalize(1.0, 0.0, 10.0), 10.0);
    }
}
