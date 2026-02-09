//! Statistical Utilities
//!
//! Functions for calculating percentiles and statistical measures.

/// Calculate percentile from sorted data.
pub fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    
    if p <= 0.0 {
        return sorted_data[0];
    }
    
    if p >= 1.0 {
        return sorted_data[sorted_data.len() - 1];
    }
    
    let index = (sorted_data.len() as f64 * p).floor() as usize;
    let index = index.min(sorted_data.len() - 1);
    
    sorted_data[index]
}

/// Calculate multiple percentiles from sorted data.
pub fn percentiles(sorted_data: &[f64], percentiles: &[f64]) -> Vec<f64> {
    percentiles.iter()
        .map(|&p| percentile(sorted_data, p))
        .collect()
}

/// Calculate mean.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    data.iter().sum::<f64>() / data.len() as f64
}

/// Calculate median.
pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Calculate standard deviation.
pub fn std_dev(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    let mean_val = mean(data);
    let variance = data.iter()
        .map(|&x| (x - mean_val).powi(2))
        .sum::<f64>() / data.len() as f64;
    
    variance.sqrt()
}

/// Calculate min, max, mean, median, std_dev.
pub fn summary_stats(data: &[f64]) -> (f64, f64, f64, f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    
    let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mean_val = mean(data);
    let median_val = median(data);
    let std_dev_val = std_dev(data);
    
    (min, max, mean_val, median_val, std_dev_val)
}




