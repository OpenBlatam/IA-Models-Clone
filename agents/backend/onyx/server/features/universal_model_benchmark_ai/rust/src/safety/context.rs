//! Error Context Builder
//!
//! Context builder for adding error context.

use crate::error::BenchmarkError;

/// Context builder for adding error context.
pub struct ErrorContext {
    context: String,
}

impl ErrorContext {
    /// Create a new error context.
    pub fn new(context: impl Into<String>) -> Self {
        Self {
            context: context.into(),
        }
    }
    
    /// Add additional context.
    pub fn with(mut self, additional: impl Into<String>) -> Self {
        self.context.push_str(": ");
        self.context.push_str(&additional.into());
        self
    }
    
    /// Add context with format.
    pub fn with_format(mut self, format: &str, args: &[&dyn std::fmt::Display]) -> Self {
        self.context.push_str(": ");
        // Simple format implementation
        let formatted = format!("{}", format);
        self.context.push_str(&formatted);
        self
    }
    
    /// Convert to BenchmarkError.
    pub fn into_error(self) -> BenchmarkError {
        BenchmarkError::Other(self.context)
    }
    
    /// Get the context string.
    pub fn as_str(&self) -> &str {
        &self.context
    }
    
    /// Convert to string.
    pub fn to_string(&self) -> String {
        self.context.clone()
    }
}

impl From<ErrorContext> for BenchmarkError {
    fn from(ctx: ErrorContext) -> Self {
        ctx.into_error()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_context() {
        let ctx = ErrorContext::new("Operation failed")
            .with("File not found")
            .with("Path: /tmp/test");
        
        let error: BenchmarkError = ctx.into();
        assert!(matches!(error, BenchmarkError::Other(_)));
    }
    
    #[test]
    fn test_error_context_as_str() {
        let ctx = ErrorContext::new("Test");
        assert_eq!(ctx.as_str(), "Test");
    }
}




