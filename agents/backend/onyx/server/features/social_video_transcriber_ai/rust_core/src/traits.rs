//! Traits and Interfaces
//!
//! Defines common traits for extensibility and polymorphism.

use pyo3::prelude::*;

/// Trait for modules that can be registered with Python
pub trait PyModuleRegistrar {
    fn register(&self, module: &Bound<'_, PyModule>) -> PyResult<()>;
}

/// Trait for services that can be configured
pub trait Configurable {
    fn configure(&mut self, config: &crate::config::CoreConfig) -> PyResult<()>;
}

/// Trait for services that provide statistics
pub trait StatProvider {
    fn get_stats(&self) -> PyResult<PyObject>;
}

/// Trait for services that can be reset
pub trait Resettable {
    fn reset(&mut self) -> PyResult<()>;
}

/// Trait for services that support health checks
pub trait HealthCheckable {
    fn is_healthy(&self) -> bool;
    fn get_health_status(&self) -> PyResult<PyObject>;
}

/// Trait for services that can be profiled
pub trait Profilable {
    fn start_profiling(&mut self) -> PyResult<()>;
    fn stop_profiling(&mut self) -> PyResult<()>;
    fn get_profile_data(&self) -> PyResult<PyObject>;
}

/// Trait for batch processors
pub trait BatchProcessable<T> {
    fn process_batch(&self, items: Vec<T>) -> PyResult<Vec<T>>;
    fn get_batch_stats(&self) -> PyResult<PyObject>;
}

/// Trait for cache services
pub trait Cacheable {
    fn get(&self, key: &str) -> PyResult<Option<String>>;
    fn set(&self, key: &str, value: &str, ttl: Option<u64>) -> PyResult<()>;
    fn remove(&self, key: &str) -> PyResult<()>;
    fn clear(&self) -> PyResult<()>;
}

/// Trait for compression services
pub trait Compressible {
    fn compress(&self, data: &[u8]) -> PyResult<Vec<u8>>;
    fn decompress(&self, data: &[u8]) -> PyResult<Vec<u8>>;
    fn get_compression_ratio(&self) -> f64;
}

/// Trait for search services
pub trait Searchable {
    fn search(&self, query: &str) -> PyResult<Vec<String>>;
    fn index(&self, document: &str) -> PyResult<()>;
    fn clear_index(&self) -> PyResult<()>;
}

/// Trait for text processors
pub trait TextProcessable {
    fn process(&self, text: &str) -> PyResult<String>;
    fn analyze(&self, text: &str) -> PyResult<PyObject>;
}












