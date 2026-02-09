//! Factory Pattern Implementation
//!
//! Provides factories for creating service instances with different configurations.

use pyo3::prelude::*;
use crate::config::CoreConfig;
use crate::cache::CacheService;
use crate::compression::CompressionService;
use crate::batch::BatchProcessor;
use crate::text::TextProcessor;
use crate::search::SearchEngine;
use crate::profiling::Profiler;
use crate::health::HealthChecker;

/// Factory for creating service instances
pub struct ServiceFactory;

impl ServiceFactory {
    /// Create a cache service with default configuration
    pub fn create_cache_service(max_size: usize, ttl: u64) -> PyResult<crate::cache::CacheService> {
        Ok(crate::cache::CacheService::new(max_size, ttl))
    }

    /// Create a cache service with custom configuration
    pub fn create_cache_service_with_config(
        config: &CoreConfig,
    ) -> PyResult<crate::cache::CacheService> {
        Ok(crate::cache::CacheService::new(
            config.max_cache_size,
            config.default_ttl,
        ))
    }

    /// Create a compression service with default settings
    pub fn create_compression_service() -> PyResult<crate::compression::CompressionService> {
        Ok(crate::compression::CompressionService::new(3, 4, 6))
    }

    /// Create a compression service with custom settings
    pub fn create_compression_service_with_config(
        config: &CoreConfig,
    ) -> PyResult<crate::compression::CompressionService> {
        Ok(crate::compression::CompressionService::new(
            config.compression_level as u32,
            4,
            6,
        ))
    }

    /// Create a batch processor with default settings
    pub fn create_batch_processor() -> PyResult<crate::batch::BatchProcessor> {
        let num_workers = num_cpus::get();
        Ok(crate::batch::BatchProcessor::new(num_workers, 100))
    }

    /// Create a batch processor with custom settings
    pub fn create_batch_processor_with_config(
        config: &CoreConfig,
    ) -> PyResult<crate::batch::BatchProcessor> {
        Ok(crate::batch::BatchProcessor::new(
            config.num_workers,
            config.batch_size,
        ))
    }

    /// Create a text processor
    pub fn create_text_processor() -> PyResult<crate::text::TextProcessor> {
        Ok(crate::text::TextProcessor::new())
    }

    /// Create a search engine with default settings
    pub fn create_search_engine() -> PyResult<crate::search::SearchEngine> {
        Ok(crate::search::SearchEngine::new(1000))
    }

    /// Create a profiler
    pub fn create_profiler() -> PyResult<crate::profiling::Profiler> {
        Ok(crate::profiling::Profiler::new())
    }

    /// Create a health checker
    pub fn create_health_checker() -> PyResult<crate::health::HealthChecker> {
        Ok(crate::health::HealthChecker::new())
    }

    /// Create all services with a configuration
    pub fn create_all_services(
        config: &CoreConfig,
    ) -> ServiceBundle {
        ServiceBundle {
            cache: Self::create_cache_service_with_config(config).unwrap_or_else(|_| CacheService::new(1000, 3600)),
            compression: Self::create_compression_service_with_config(config).unwrap_or_else(|_| CompressionService::new(3, 4, 6)),
            batch: Self::create_batch_processor_with_config(config).unwrap_or_else(|_| BatchProcessor::new(4, 100)),
            text: Self::create_text_processor().unwrap_or_else(|_| TextProcessor::new()),
            search: Self::create_search_engine().unwrap_or_else(|_| SearchEngine::new(1000)),
            profiler: Self::create_profiler().unwrap_or_else(|_| Profiler::new()),
            health: Self::create_health_checker().unwrap_or_else(|_| HealthChecker::new()),
        }
    }
}

/// Bundle of all services
#[pyclass]
pub struct ServiceBundle {
    #[pyo3(get, set)]
    pub cache: crate::cache::CacheService,
    #[pyo3(get, set)]
    pub compression: crate::compression::CompressionService,
    #[pyo3(get, set)]
    pub batch: crate::batch::BatchProcessor,
    #[pyo3(get, set)]
    pub text: crate::text::TextProcessor,
    #[pyo3(get, set)]
    pub search: crate::search::SearchEngine,
    #[pyo3(get, set)]
    pub profiler: crate::profiling::Profiler,
    #[pyo3(get, set)]
    pub health: crate::health::HealthChecker,
}

#[pymethods]
impl ServiceBundle {
    #[new]
    pub fn new() -> Self {
        let config = CoreConfig::default();
        ServiceFactory::create_all_services(&config)
    }

    pub fn with_config(_config: &crate::config::Config) -> Self {
        let core_config = CoreConfig::default(); // Convert from Config if needed
        ServiceFactory::create_all_services(&core_config)
    }

    pub fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("cache", self.cache.get_stats()?)?;
            dict.set_item("health", self.health.get_health()?)?;
            Ok(dict.into())
        })
    }
}

#[pyfunction]
pub fn create_service_factory() -> ServiceFactory {
    ServiceFactory
}

#[pyfunction]
pub fn create_service_bundle() -> ServiceBundle {
    ServiceBundle::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_factory() {
        let cache = ServiceFactory::create_cache_service(1000, 3600).unwrap();
        assert!(cache.get_stats().is_ok());
    }

    #[test]
    fn test_service_bundle() {
        let bundle = ServiceBundle::new();
        assert!(bundle.get_stats().is_ok());
    }
}

