//! Validation Utilities
//!
//! Functions for validating and constraining values.

/// Clamp value between min and max.
pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

/// Clamp value between min and max (integer version).
pub fn clamp_i64(value: i64, min: i64, max: i64) -> i64 {
    value.max(min).min(max)
}

/// Clamp value between min and max (usize version).
pub fn clamp_usize(value: usize, min: usize, max: usize) -> usize {
    value.max(min).min(max)
}

/// Check if value is in range [min, max].
pub fn in_range(value: f64, min: f64, max: f64) -> bool {
    value >= min && value <= max
}

/// Check if value is in range [min, max] (integer version).
pub fn in_range_i64(value: i64, min: i64, max: i64) -> bool {
    value >= min && value <= max
}

/// Check if value is positive.
pub fn is_positive(value: f64) -> bool {
    value > 0.0
}

/// Check if value is non-negative.
pub fn is_non_negative(value: f64) -> bool {
    value >= 0.0
}

/// Check if value is finite.
pub fn is_finite(value: f64) -> bool {
    value.is_finite()
}

/// Validate that value is in valid range, return error if not.
pub fn validate_range(value: f64, min: f64, max: f64, name: &str) -> Result<(), String> {
    if !in_range(value, min, max) {
        Err(format!("{} must be in range [{}, {}], got {}", name, min, max, value))
    } else {
        Ok(())
    }
}

/// Validate that value is positive, return error if not.
pub fn validate_positive(value: f64, name: &str) -> Result<(), String> {
    if !is_positive(value) {
        Err(format!("{} must be positive, got {}", name, value))
    } else {
        Ok(())
    }
}




