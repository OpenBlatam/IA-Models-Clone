//! Inference Utilities
//!
//! Utility functions for inference operations.

use super::error::{InferenceError, InferenceResult};

/// Validate that a value is within a range.
pub fn validate_range<T: PartialOrd>(
    value: T,
    min: T,
    max: T,
    field_name: &str,
) -> InferenceResult<()> {
    if value < min || value > max {
        Err(InferenceError::ValidationError(
            format!(
                "{} must be between {:?} and {:?}, got {:?}",
                field_name, min, max, value
            )
        ))
    } else {
        Ok(())
    }
}

/// Validate that a value is positive.
pub fn validate_positive<T: PartialOrd + Default>(
    value: T,
    field_name: &str,
) -> InferenceResult<()> {
    if value <= T::default() {
        Err(InferenceError::ValidationError(
            format!("{} must be positive, got {:?}", field_name, value)
        ))
    } else {
        Ok(())
    }
}

/// Calculate tokens per second from latency and token count.
pub fn calculate_tokens_per_second(latency_ms: f64, tokens: usize) -> f64 {
    if latency_ms > 0.0 {
        (tokens as f64) / (latency_ms / 1000.0)
    } else {
        0.0
    }
}

/// Format latency in human-readable format.
pub fn format_latency(latency_ms: f64) -> String {
    if latency_ms < 1.0 {
        format!("{:.2}μs", latency_ms * 1000.0)
    } else if latency_ms < 1000.0 {
        format!("{:.2}ms", latency_ms)
    } else {
        format!("{:.2}s", latency_ms / 1000.0)
    }
}

/// Format memory usage in human-readable format.
pub fn format_memory(mb: f64) -> String {
    if mb < 1.0 {
        format!("{:.2}KB", mb * 1024.0)
    } else if mb < 1024.0 {
        format!("{:.2}MB", mb)
    } else {
        format!("{:.2}GB", mb / 1024.0)
    }
}

/// Clamp a value between min and max.
pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Linear interpolation.
pub fn lerp(start: f32, end: f32, t: f32) -> f32 {
    start + (end - start) * t.clamp(0.0, 1.0)
}

/// Exponential moving average.
pub struct ExponentialMovingAverage {
    alpha: f64,
    value: Option<f64>,
}

impl ExponentialMovingAverage {
    /// Create a new EMA with given smoothing factor.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            value: None,
        }
    }
    
    /// Update EMA with new value.
    pub fn update(&mut self, new_value: f64) -> f64 {
        match self.value {
            Some(current) => {
                let updated = self.alpha * new_value + (1.0 - self.alpha) * current;
                self.value = Some(updated);
                updated
            }
            None => {
                self.value = Some(new_value);
                new_value
            }
        }
    }
    
    /// Get current EMA value.
    pub fn value(&self) -> Option<f64> {
        self.value
    }
    
    /// Reset EMA.
    pub fn reset(&mut self) {
        self.value = None;
    }
}




