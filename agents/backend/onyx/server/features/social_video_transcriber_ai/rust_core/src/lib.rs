//! # Transcriber Core - Ultra High-Performance Rust Core
//!
//! High-performance Rust implementation for Social Video Transcriber AI.
//!
//! ## Features
//!
//! - **Text Processing**: Segmentation, analysis, NLP (10-20x faster)
//! - **Search Engine**: Regex, similarity, full-text (3-5x faster)
//! - **Cache**: LRU, TTL, concurrent (20x faster)
//! - **Batch Processing**: Rayon parallelization (5-10x faster)
//! - **Compression**: LZ4, Zstd, Snappy, Brotli (500+ MB/s)
//! - **SIMD JSON**: 3-5x faster JSON parsing
//! - **ID Generation**: UUID, ULID, Snowflake (1M+ IDs/s)
//! - **Memory Management**: Object pools, ring buffers
//! - **Streaming**: Chunked text processing
//! - **Metrics**: Performance tracking and histograms
//!
//! ## Example
//!
//! ```python
//! from transcriber_core import (
//!     TextProcessor, SearchEngine, CacheService,
//!     CompressionService, SimdJsonService, IdGenerator
//! )
//!
//! # Text processing
//! processor = TextProcessor()
//! stats = processor.analyze("Your text here...")
//!
//! # Compression
//! compressor = CompressionService()
//! compressed = compressor.compress_lz4(data)
//!
//! # SIMD JSON
//! json_service = SimdJsonService()
//! parsed = json_service.parse('{"key": "value"}')
//! ```

use pyo3::prelude::*;

// Organized module structure
pub mod core {
    pub mod batch;
    pub mod cache;
    pub mod search;
    pub mod text;
}

pub mod processing {
    pub mod crypto;
    pub mod similarity;
    pub mod language;
    pub mod streaming;
}

pub mod optimization {
    pub mod compression;
    pub mod simd_json;
    pub mod memory;
    pub mod metrics;
}

pub mod utility {
    pub mod id_gen;
    pub mod utils;
    pub mod profiling;
    pub mod health;
    pub mod logger;
    pub mod async_utils;
    pub mod serialization;
    pub mod retry;
    pub mod pool;
    pub mod rate_limiter;
    pub mod backpressure;
    pub mod telemetry;
}

pub mod enterprise {
    pub mod context;
    pub mod cache_strategies;
    pub mod scheduler;
    pub mod workflow;
    pub mod distributed_lock;
    pub mod state_machine;
    pub mod feature_flags;
    pub mod metrics_aggregator;
}

pub mod infrastructure {
    pub mod builder;
    pub mod config;
    pub mod constants;
    pub mod error;
    pub mod events;
    pub mod factory;
    #[macro_use]
    pub mod macros;
    pub mod middleware;
    pub mod module_registry;
    pub mod observer;
    pub mod plugin;
    pub mod prelude;
    pub mod reexports;
    pub mod traits;
    pub mod types;
    pub mod validation;
}

// Re-export for backward compatibility
pub use core::*;
pub use processing::*;
pub use optimization::*;
pub use utility::*;
pub use enterprise::*;
pub use infrastructure::*;

// Infrastructure imports
use infrastructure::config::Config;
use infrastructure::module_registry::register_all_modules;
use infrastructure::factory::{ServiceFactory, ServiceBundle, create_service_factory, create_service_bundle};
use infrastructure::builder::{ConfigBuilder, ServiceBundleBuilder, create_config_builder, create_service_bundle_builder};
use infrastructure::validation::{Validator, create_validator};
use infrastructure::events::{EventBus, create_event_bus};
use infrastructure::middleware::{MiddlewareChain, create_middleware_chain};
use infrastructure::observer::{Observable, create_observable};
use infrastructure::plugin::{PluginManager, create_plugin_manager};
use utility::logger::{Logger, create_logger};
use utility::async_utils::{AsyncSemaphore, AsyncRateLimiter, AsyncTimer, AsyncBatchProcessor, create_async_semaphore, create_async_rate_limiter, create_async_timer, create_async_batch_processor};
use utility::serialization::{Serializer, create_serializer};
use utility::retry::{RetryExecutor, CircuitBreaker, create_retry_executor, create_circuit_breaker};
use utility::pool::{ResourcePool, create_resource_pool};
use utility::rate_limiter::{RateLimiter, create_rate_limiter};
use utility::backpressure::{BackpressureController, create_backpressure_controller};
use utility::telemetry::{TelemetryCollector, create_telemetry_collector};
use enterprise::context::{RequestContext, ContextManager, create_context_manager};
use enterprise::cache_strategies::{AdvancedCache, create_advanced_cache};
use enterprise::scheduler::{TaskScheduler, create_task_scheduler};
use enterprise::workflow::{Workflow, create_workflow};
use enterprise::distributed_lock::{DistributedLock, LockManager, create_lock_manager};
use enterprise::state_machine::{StateMachine, create_state_machine};
use enterprise::feature_flags::{FeatureFlagManager, create_feature_flag_manager};
use enterprise::metrics_aggregator::{MetricsAggregator, create_metrics_aggregator};

// Re-export commonly used items at the top level
pub use infrastructure::reexports::*;

// Re-export constants and types
pub use infrastructure::constants::*;
pub use infrastructure::types::*;

#[pymodule]
fn transcriber_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_all_modules(m)?;
    
    m.add_class::<Config>()?;
    m.add_class::<ServiceBundle>()?;
    m.add_class::<ConfigBuilder>()?;
    m.add_class::<ServiceBundleBuilder>()?;
    m.add_class::<Validator>()?;
    m.add_class::<EventBus>()?;
    m.add_class::<MiddlewareChain>()?;
    m.add_class::<Observable>()?;
    m.add_class::<PluginManager>()?;
    m.add_class::<Logger>()?;
    m.add_class::<AsyncSemaphore>()?;
    m.add_class::<AsyncRateLimiter>()?;
    m.add_class::<AsyncTimer>()?;
    m.add_class::<AsyncBatchProcessor>()?;
    m.add_class::<Serializer>()?;
    m.add_class::<RetryExecutor>()?;
    m.add_class::<CircuitBreaker>()?;
    m.add_class::<ResourcePool>()?;
    m.add_class::<RateLimiter>()?;
    m.add_class::<BackpressureController>()?;
    m.add_class::<TelemetryCollector>()?;
    m.add_class::<RequestContext>()?;
    m.add_class::<ContextManager>()?;
    m.add_class::<AdvancedCache>()?;
    m.add_class::<TaskScheduler>()?;
    m.add_class::<Workflow>()?;
    m.add_class::<DistributedLock>()?;
    m.add_class::<LockManager>()?;
    m.add_class::<StateMachine>()?;
    m.add_class::<FeatureFlagManager>()?;
    m.add_class::<MetricsAggregator>()?;
    
    m.add_function(wrap_pyfunction!(is_rust_available, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(get_module_info, m)?)?;
    m.add_function(wrap_pyfunction!(create_service_factory, m)?)?;
    m.add_function(wrap_pyfunction!(create_service_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(create_config_builder, m)?)?;
    m.add_function(wrap_pyfunction!(create_service_bundle_builder, m)?)?;
    m.add_function(wrap_pyfunction!(create_validator, m)?)?;
    m.add_function(wrap_pyfunction!(create_event_bus, m)?)?;
    m.add_function(wrap_pyfunction!(create_middleware_chain, m)?)?;
    m.add_function(wrap_pyfunction!(create_observable, m)?)?;
    m.add_function(wrap_pyfunction!(create_plugin_manager, m)?)?;
    m.add_function(wrap_pyfunction!(create_logger, m)?)?;
    m.add_function(wrap_pyfunction!(create_async_semaphore, m)?)?;
    m.add_function(wrap_pyfunction!(create_async_rate_limiter, m)?)?;
    m.add_function(wrap_pyfunction!(create_async_timer, m)?)?;
    m.add_function(wrap_pyfunction!(create_async_batch_processor, m)?)?;
    m.add_function(wrap_pyfunction!(create_serializer, m)?)?;
    m.add_function(wrap_pyfunction!(create_retry_executor, m)?)?;
    m.add_function(wrap_pyfunction!(create_circuit_breaker, m)?)?;
    m.add_function(wrap_pyfunction!(create_resource_pool, m)?)?;
    m.add_function(wrap_pyfunction!(create_rate_limiter, m)?)?;
    m.add_function(wrap_pyfunction!(create_backpressure_controller, m)?)?;
    m.add_function(wrap_pyfunction!(create_telemetry_collector, m)?)?;
    m.add_function(wrap_pyfunction!(create_context_manager, m)?)?;
    m.add_function(wrap_pyfunction!(create_advanced_cache, m)?)?;
    m.add_function(wrap_pyfunction!(create_task_scheduler, m)?)?;
    m.add_function(wrap_pyfunction!(create_workflow, m)?)?;
    m.add_function(wrap_pyfunction!(create_lock_manager, m)?)?;
    m.add_function(wrap_pyfunction!(create_state_machine, m)?)?;
    m.add_function(wrap_pyfunction!(create_feature_flag_manager, m)?)?;
    m.add_function(wrap_pyfunction!(create_metrics_aggregator, m)?)?;
    
    m.add("__version__", infrastructure::constants::VERSION)?;
    m.add("__author__", infrastructure::constants::LIBRARY_AUTHOR)?;
    m.add("__description__", infrastructure::constants::LIBRARY_DESCRIPTION)?;
    m.add("__name__", infrastructure::constants::LIBRARY_NAME)?;
    m.add("RUST_AVAILABLE", true)?;
    
    // Add constants
    m.add("DEFAULT_CACHE_SIZE", infrastructure::constants::DEFAULT_CACHE_SIZE)?;
    m.add("DEFAULT_TTL", infrastructure::constants::DEFAULT_TTL)?;
    m.add("DEFAULT_BATCH_SIZE", infrastructure::constants::DEFAULT_BATCH_SIZE)?;
    m.add("MAX_CACHE_SIZE", infrastructure::constants::MAX_CACHE_SIZE)?;
    m.add("MAX_TTL", infrastructure::constants::MAX_TTL)?;
    m.add("MAX_BATCH_SIZE", infrastructure::constants::MAX_BATCH_SIZE)?;

    Ok(())
}

#[pyfunction]
pub fn is_rust_available() -> bool {
    true
}

#[pyfunction]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[pyfunction]
pub fn get_module_info() -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("version", env!("CARGO_PKG_VERSION"))?;
        dict.set_item("author", "Social Video Transcriber AI Team")?;
        dict.set_item("rust_available", true)?;
        dict.set_item("modules", vec![
            infrastructure::constants::modules::TEXT,
            infrastructure::constants::modules::SEARCH,
            infrastructure::constants::modules::CACHE,
            infrastructure::constants::modules::BATCH,
            infrastructure::constants::modules::CRYPTO,
            infrastructure::constants::modules::SIMILARITY,
            infrastructure::constants::modules::LANGUAGE,
            infrastructure::constants::modules::COMPRESSION,
            infrastructure::constants::modules::SIMD_JSON,
            infrastructure::constants::modules::ID_GEN,
            infrastructure::constants::modules::MEMORY,
            infrastructure::constants::modules::STREAMING,
            infrastructure::constants::modules::METRICS,
            infrastructure::constants::modules::UTILS,
            infrastructure::constants::modules::PROFILING,
            infrastructure::constants::modules::HEALTH,
        ])?;
    dict.set_item("version", infrastructure::constants::VERSION)?;
    dict.set_item("name", infrastructure::constants::LIBRARY_NAME)?;
    dict.set_item("author", infrastructure::constants::LIBRARY_AUTHOR)?;
    dict.set_item("description", infrastructure::constants::LIBRARY_DESCRIPTION)?;
        Ok(dict.into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_initialization() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = PyModule::new_bound(py, "transcriber_core").unwrap();
            assert!(transcriber_core(&module).is_ok());
        });
    }

    #[test]
    fn test_config() {
        let config = infrastructure::config::Config::new();
        assert!(config.get_max_cache_size() > 0);
        assert!(config.get_num_workers() > 0);
    }

    #[test]
    fn test_module_info() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let info = get_module_info().unwrap();
            let dict = info.downcast::<pyo3::types::PyDict>(py).unwrap();
            assert!(dict.contains("version").unwrap());
            assert!(dict.contains("rust_available").unwrap());
        });
    }
}

