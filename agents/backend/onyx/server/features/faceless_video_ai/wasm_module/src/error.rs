//! Error types for the WASM module.
//!
//! Provides unified error handling with JavaScript interoperability.

use thiserror::Error;
use wasm_bindgen::prelude::*;

/// Main error type for WASM operations.
#[derive(Error, Debug)]
pub enum WasmError {
    /// Canvas-related errors (context acquisition, drawing).
    #[error("Canvas error: {0}")]
    Canvas(String),

    /// Image processing errors (invalid data, unsupported format).
    #[error("Image processing error: {0}")]
    ImageProcessing(String),

    /// Invalid input parameters.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Parsing errors (SRT, VTT, JSON).
    #[error("Parse error: {0}")]
    Parse(String),

    /// JavaScript/WASM interop errors.
    #[error("JavaScript error: {0}")]
    JavaScript(String),

    /// Filter not found or unsupported.
    #[error("Unknown filter: {0}")]
    UnknownFilter(String),

    /// Image not loaded before operation.
    #[error("No image loaded")]
    NoImageLoaded,

    /// Invalid dimensions for operation.
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// Operation out of bounds.
    #[error("Out of bounds: {0}")]
    OutOfBounds(String),
}

impl WasmError {
    /// Create a canvas error.
    pub fn canvas(msg: impl Into<String>) -> Self {
        Self::Canvas(msg.into())
    }

    /// Create an image processing error.
    pub fn image(msg: impl Into<String>) -> Self {
        Self::ImageProcessing(msg.into())
    }

    /// Create an invalid input error.
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create a parse error.
    pub fn parse(msg: impl Into<String>) -> Self {
        Self::Parse(msg.into())
    }
}

impl From<WasmError> for JsValue {
    fn from(err: WasmError) -> Self {
        JsValue::from_str(&err.to_string())
    }
}

impl From<JsValue> for WasmError {
    fn from(value: JsValue) -> Self {
        WasmError::JavaScript(
            value
                .as_string()
                .unwrap_or_else(|| "Unknown JavaScript error".to_string()),
        )
    }
}

impl From<serde_json::Error> for WasmError {
    fn from(err: serde_json::Error) -> Self {
        WasmError::Parse(err.to_string())
    }
}

impl From<std::num::ParseIntError> for WasmError {
    fn from(err: std::num::ParseIntError) -> Self {
        WasmError::Parse(format!("Integer parse error: {}", err))
    }
}

impl From<std::num::ParseFloatError> for WasmError {
    fn from(err: std::num::ParseFloatError) -> Self {
        WasmError::Parse(format!("Float parse error: {}", err))
    }
}

/// Result type alias for WASM operations.
pub type WasmResult<T> = Result<T, WasmError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = WasmError::canvas("Failed to get context");
        assert_eq!(err.to_string(), "Canvas error: Failed to get context");
    }

    #[test]
    fn test_error_constructors() {
        assert!(matches!(
            WasmError::canvas("test"),
            WasmError::Canvas(_)
        ));
        assert!(matches!(
            WasmError::image("test"),
            WasmError::ImageProcessing(_)
        ));
        assert!(matches!(
            WasmError::invalid_input("test"),
            WasmError::InvalidInput(_)
        ));
        assert!(matches!(WasmError::parse("test"), WasmError::Parse(_)));
    }
}
