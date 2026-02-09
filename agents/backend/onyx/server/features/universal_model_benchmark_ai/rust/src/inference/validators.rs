//! Configuration Validators
//!
//! Validation functions for inference configuration.

use crate::inference::error::{InferenceError, InferenceResult};
use crate::inference::config::InferenceConfig;

/// Validate inference configuration.
pub fn validate_config(config: &InferenceConfig) -> InferenceResult<()> {
    // Validate max_tokens
    if config.max_tokens == 0 {
        return Err(InferenceError::ValidationError(
            "max_tokens must be greater than 0".to_string()
        ));
    }
    
    if config.max_tokens > 32768 {
        return Err(InferenceError::ValidationError(
            "max_tokens exceeds maximum of 32768".to_string()
        ));
    }
    
    // Validate temperature
    if config.temperature < 0.0 || config.temperature > 2.0 {
        return Err(InferenceError::ValidationError(
            "temperature must be between 0.0 and 2.0".to_string()
        ));
    }
    
    // Validate top_p
    if config.top_p <= 0.0 || config.top_p > 1.0 {
        return Err(InferenceError::ValidationError(
            "top_p must be between 0.0 and 1.0".to_string()
        ));
    }
    
    // Validate top_k
    if config.top_k == 0 {
        return Err(InferenceError::ValidationError(
            "top_k must be greater than 0".to_string()
        ));
    }
    
    // Validate repetition_penalty
    if config.repetition_penalty < 1.0 || config.repetition_penalty > 2.0 {
        return Err(InferenceError::ValidationError(
            "repetition_penalty must be between 1.0 and 2.0".to_string()
        ));
    }
    
    // Validate batch_size
    if config.batch_size == 0 {
        return Err(InferenceError::ValidationError(
            "batch_size must be greater than 0".to_string()
        ));
    }
    
    if config.batch_size > 128 {
        return Err(InferenceError::ValidationError(
            "batch_size exceeds maximum of 128".to_string()
        ));
    }
    
    Ok(())
}

/// Validate temperature value.
pub fn validate_temperature(temperature: f32) -> InferenceResult<()> {
    if temperature < 0.0 || temperature > 2.0 {
        Err(InferenceError::ValidationError(
            format!("Temperature {} is out of range [0.0, 2.0]", temperature)
        ))
    } else {
        Ok(())
    }
}

/// Validate top_p value.
pub fn validate_top_p(top_p: f32) -> InferenceResult<()> {
    if top_p <= 0.0 || top_p > 1.0 {
        Err(InferenceError::ValidationError(
            format!("Top-p {} is out of range (0.0, 1.0]", top_p)
        ))
    } else {
        Ok(())
    }
}

/// Validate top_k value.
pub fn validate_top_k(top_k: usize) -> InferenceResult<()> {
    if top_k == 0 {
        Err(InferenceError::ValidationError(
            "Top-k must be greater than 0".to_string()
        ))
    } else {
        Ok(())
    }
}

/// Validate batch size.
pub fn validate_batch_size(batch_size: usize) -> InferenceResult<()> {
    if batch_size == 0 {
        Err(InferenceError::ValidationError(
            "Batch size must be greater than 0".to_string()
        ))
    } else if batch_size > 128 {
        Err(InferenceError::ValidationError(
            "Batch size exceeds maximum of 128".to_string()
        ))
    } else {
        Ok(())
    }
}




