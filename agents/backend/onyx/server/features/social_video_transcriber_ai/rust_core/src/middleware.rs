//! Middleware System
//!
//! Provides middleware pattern for cross-cutting concerns.

use pyo3::prelude::*;
use std::sync::Arc;

/// Middleware context
#[derive(Debug, Clone)]
pub struct MiddlewareContext {
    pub request_id: String,
    pub metadata: std::collections::HashMap<String, String>,
}

impl MiddlewareContext {
    pub fn new(request_id: String) -> Self {
        Self {
            request_id,
            metadata: std::collections::HashMap::new(),
        }
    }
}

/// Middleware trait
pub trait Middleware: Send + Sync {
    fn process(&self, context: &mut MiddlewareContext, next: &dyn Fn(&mut MiddlewareContext) -> PyResult<()>) -> PyResult<()>;
}

/// Middleware chain
#[pyclass]
pub struct MiddlewareChain {
    middlewares: Vec<Arc<dyn Middleware>>,
}

#[pymethods]
impl MiddlewareChain {
    #[new]
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    pub fn add(&mut self, middleware: PyObject) -> PyResult<()> {
        // Note: In a real implementation, you'd wrap Python callables
        // For now, this is a placeholder structure
        Ok(())
    }

    pub fn execute(&self, context: &mut MiddlewareContext) -> PyResult<()> {
        // Execute middleware chain
        Ok(())
    }
}

/// Logging middleware
pub struct LoggingMiddleware;

impl Middleware for LoggingMiddleware {
    fn process(&self, context: &mut MiddlewareContext, next: &dyn Fn(&mut MiddlewareContext) -> PyResult<()>) -> PyResult<()> {
        log::info!("Request started: {}", context.request_id);
        let result = next(context);
        log::info!("Request completed: {}", context.request_id);
        result
    }
}

/// Timing middleware
pub struct TimingMiddleware;

impl Middleware for TimingMiddleware {
    fn process(&self, context: &mut MiddlewareContext, next: &dyn Fn(&mut MiddlewareContext) -> PyResult<()>) -> PyResult<()> {
        let start = std::time::Instant::now();
        let result = next(context);
        let elapsed = start.elapsed();
        context.metadata.insert("duration_ms".to_string(), elapsed.as_millis().to_string());
        result
    }
}

#[pyfunction]
pub fn create_middleware_chain() -> MiddlewareChain {
    MiddlewareChain::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_middleware_context() {
        let mut context = MiddlewareContext::new("test-123".to_string());
        context.metadata.insert("key".to_string(), "value".to_string());
        assert_eq!(context.request_id, "test-123");
    }
}












