//! Python Module Registry
//!
//! Centralized module registration for all Rust submodules.
//! Provides a clean, maintainable way to register Python bindings.

use pyo3::prelude::*;

use crate::core::batch::BatchProcessor;
use crate::core::cache::CacheService;
use crate::optimization::compression::{CompressionService, CompressionBenchmark};
use crate::processing::crypto::HashService;
use crate::utility::id_gen::{IdGenerator, IdBenchmark};
use crate::processing::language::LanguageDetector;
use crate::optimization::memory::{
    ChunkedBuffer, MemoryStats, MemoryTracker, ObjectPool, PoolStatsResult, RingBuffer,
};
use crate::optimization::metrics::{
    HistogramStats, MetricsCollector, MetricsSummary, Stopwatch, TimerStats,
};
use crate::core::search::SearchEngine;
use crate::optimization::simd_json::{JsonBenchmark, SimdJsonService};
use crate::processing::similarity::SimilarityEngine;
use crate::processing::streaming::{
    ChunkResult, LineItem, LineIterator, ParallelProcessor, ProcessingProgress, StreamChunk,
    TextStream,
};
use crate::core::text::TextProcessor;
use crate::utility::utils::{
    create_timer, get_system_info, DateUtils, JsonUtils, SRTEntry, StringUtils, SubtitleUtils,
    Timer,
};
use crate::utility::profiling::{Profiler, Timer as ProfilingTimer, create_profiler, create_timer as create_profiling_timer};
use crate::utility::health::{HealthChecker, SystemMonitor, create_health_checker, create_system_monitor};
use crate::infrastructure::factory::{ServiceFactory, ServiceBundle, create_service_factory, create_service_bundle};
use crate::infrastructure::builder::{ConfigBuilder, ServiceBundleBuilder, create_config_builder, create_service_bundle_builder};
use crate::infrastructure::validation::{Validator, create_validator};
use crate::infrastructure::events::{EventBus, create_event_bus};
use crate::infrastructure::middleware::{MiddlewareChain, create_middleware_chain};
use crate::infrastructure::observer::{Observable, create_observable};
use crate::infrastructure::plugin::{PluginManager, create_plugin_manager};
use crate::utility::logger::{Logger, create_logger};
use crate::utility::async_utils::{AsyncSemaphore, AsyncRateLimiter, AsyncTimer, AsyncBatchProcessor, create_async_semaphore, create_async_rate_limiter, create_async_timer, create_async_batch_processor};
use crate::utility::serialization::{Serializer, create_serializer};
use crate::utility::retry::{RetryExecutor, CircuitBreaker, create_retry_executor, create_circuit_breaker};
use crate::utility::pool::{ResourcePool, create_resource_pool};
use crate::utility::rate_limiter::{RateLimiter, create_rate_limiter};
use crate::utility::backpressure::{BackpressureController, create_backpressure_controller};
use crate::utility::telemetry::{TelemetryCollector, create_telemetry_collector};
use crate::enterprise::context::{RequestContext, ContextManager, create_context_manager};
use crate::enterprise::cache_strategies::{AdvancedCache, create_advanced_cache};
use crate::enterprise::scheduler::{TaskScheduler, create_task_scheduler};
use crate::enterprise::workflow::{Workflow, create_workflow};
use crate::enterprise::distributed_lock::{DistributedLock, LockManager, create_lock_manager};
use crate::enterprise::state_machine::{StateMachine, create_state_machine};
use crate::enterprise::feature_flags::{FeatureFlagManager, create_feature_flag_manager};
use crate::enterprise::metrics_aggregator::{MetricsAggregator, create_metrics_aggregator};

pub fn register_all_modules(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    register_core_modules(parent)?;
    register_processing_modules(parent)?;
    register_optimization_modules(parent)?;
    register_utility_modules(parent)?;
    Ok(())
}

fn register_core_modules(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    register_text_module(parent)?;
    register_search_module(parent)?;
    register_cache_module(parent)?;
    register_batch_module(parent)?;
    Ok(())
}

fn register_processing_modules(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    register_crypto_module(parent)?;
    register_similarity_module(parent)?;
    register_language_module(parent)?;
    register_streaming_module(parent)?;
    Ok(())
}

fn register_optimization_modules(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    register_compression_module(parent)?;
    register_simd_json_module(parent)?;
    register_memory_module(parent)?;
    register_metrics_module(parent)?;
    Ok(())
}

fn register_utility_modules(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    register_id_gen_module(parent)?;
    register_utils_module(parent)?;
    register_profiling_module(parent)?;
    register_health_module(parent)?;
    register_factory_module(parent)?;
    register_builder_module(parent)?;
    register_validation_module(parent)?;
    register_events_module(parent)?;
    register_middleware_module(parent)?;
    register_observer_module(parent)?;
    register_plugin_module(parent)?;
    register_logger_module(parent)?;
    register_async_utils_module(parent)?;
    register_serialization_module(parent)?;
    register_retry_module(parent)?;
    register_pool_module(parent)?;
    register_rate_limiter_module(parent)?;
    register_backpressure_module(parent)?;
    register_telemetry_module(parent)?;
    register_context_module(parent)?;
    register_cache_strategies_module(parent)?;
    register_scheduler_module(parent)?;
    register_workflow_module(parent)?;
    register_distributed_lock_module(parent)?;
    register_state_machine_module(parent)?;
    register_feature_flags_module(parent)?;
    register_metrics_aggregator_module(parent)?;
    Ok(())
}

fn register_text_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "text")?;
    m.add_class::<TextProcessor>()?;
    m.add_class::<crate::core::text::TextSegment>()?;
    m.add_class::<crate::core::text::TextStats>()?;
    m.add_class::<crate::core::text::Keyword>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_search_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "search")?;
    m.add_class::<SearchEngine>()?;
    m.add_class::<crate::core::search::SearchResult>()?;
    m.add_class::<crate::core::search::SearchFilter>()?;
    m.add_class::<crate::core::search::IndexedDocument>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_cache_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "cache")?;
    m.add_class::<CacheService>()?;
    m.add_class::<crate::core::cache::CacheEntry>()?;
    m.add_class::<crate::core::cache::CacheStats>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_batch_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "batch")?;
    m.add_class::<BatchProcessor>()?;
    m.add_class::<crate::core::batch::BatchJob>()?;
    m.add_class::<crate::core::batch::BatchResult>()?;
    m.add_class::<crate::core::batch::BatchStats>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_crypto_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "crypto")?;
    m.add_class::<HashService>()?;
    m.add_class::<crate::processing::crypto::HashResult>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_similarity_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "similarity")?;
    m.add_class::<SimilarityEngine>()?;
    m.add_class::<crate::processing::similarity::SimilarityResult>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_language_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "language")?;
    m.add_class::<LanguageDetector>()?;
    m.add_class::<crate::processing::language::DetectionResult>()?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_compression_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "compression")?;
    m.add_class::<CompressionService>()?;
    m.add_class::<CompressionBenchmark>()?;
    m.add_function(wrap_pyfunction!(crate::optimization::compression::compress_fast, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::optimization::compression::decompress_fast, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::optimization::compression::compress_best, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_simd_json_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "simd_json")?;
    m.add_class::<SimdJsonService>()?;
    m.add_class::<JsonBenchmark>()?;
    m.add_function(wrap_pyfunction!(crate::optimization::simd_json::parse_json_fast, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::optimization::simd_json::is_json_valid, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_id_gen_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "id_gen")?;
    m.add_class::<IdGenerator>()?;
    m.add_class::<IdBenchmark>()?;
    m.add_function(wrap_pyfunction!(crate::utility::id_gen::uuid_v4, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::utility::id_gen::uuid_v7, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::utility::id_gen::ulid, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::utility::id_gen::nanoid, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_memory_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "memory")?;
    m.add_class::<ObjectPool>()?;
    m.add_class::<PoolStatsResult>()?;
    m.add_class::<RingBuffer>()?;
    m.add_class::<ChunkedBuffer>()?;
    m.add_class::<MemoryTracker>()?;
    m.add_class::<MemoryStats>()?;
    m.add_function(wrap_pyfunction!(crate::optimization::memory::create_pool, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::optimization::memory::create_ring_buffer, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::optimization::memory::create_chunked_buffer, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_streaming_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "streaming")?;
    m.add_class::<TextStream>()?;
    m.add_class::<StreamChunk>()?;
    m.add_class::<ParallelProcessor>()?;
    m.add_class::<ChunkResult>()?;
    m.add_class::<ProcessingProgress>()?;
    m.add_class::<LineIterator>()?;
    m.add_class::<LineItem>()?;
    m.add_function(wrap_pyfunction!(crate::processing::streaming::create_text_stream, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::processing::streaming::create_line_iterator, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_metrics_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "metrics")?;
    m.add_class::<MetricsCollector>()?;
    m.add_class::<TimerStats>()?;
    m.add_class::<HistogramStats>()?;
    m.add_class::<MetricsSummary>()?;
    m.add_class::<Stopwatch>()?;
    m.add_function(wrap_pyfunction!(crate::optimization::metrics::create_metrics_collector, &m)?)?;
    m.add_function(wrap_pyfunction!(crate::optimization::metrics::create_stopwatch, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_utils_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "utils")?;
    m.add_class::<Timer>()?;
    m.add_class::<DateUtils>()?;
    m.add_class::<StringUtils>()?;
    m.add_class::<JsonUtils>()?;
    m.add_class::<SubtitleUtils>()?;
    m.add_class::<SRTEntry>()?;
    m.add_function(wrap_pyfunction!(get_system_info, &m)?)?;
    m.add_function(wrap_pyfunction!(create_timer, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_profiling_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "profiling")?;
    m.add_class::<Profiler>()?;
    m.add_class::<ProfilingTimer>()?;
    m.add_function(wrap_pyfunction!(create_profiler, &m)?)?;
    m.add_function(wrap_pyfunction!(create_profiling_timer, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_health_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "health")?;
    m.add_class::<HealthChecker>()?;
    m.add_class::<SystemMonitor>()?;
    m.add_function(wrap_pyfunction!(create_health_checker, &m)?)?;
    m.add_function(wrap_pyfunction!(create_system_monitor, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_factory_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "factory")?;
    m.add_class::<ServiceBundle>()?;
    m.add_function(wrap_pyfunction!(create_service_factory, &m)?)?;
    m.add_function(wrap_pyfunction!(create_service_bundle, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_builder_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "builder")?;
    m.add_class::<ConfigBuilder>()?;
    m.add_class::<ServiceBundleBuilder>()?;
    m.add_function(wrap_pyfunction!(create_config_builder, &m)?)?;
    m.add_function(wrap_pyfunction!(create_service_bundle_builder, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_validation_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "validation")?;
    m.add_class::<Validator>()?;
    m.add_function(wrap_pyfunction!(create_validator, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_events_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "events")?;
    m.add_class::<EventBus>()?;
    m.add_function(wrap_pyfunction!(create_event_bus, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_middleware_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "middleware")?;
    m.add_class::<MiddlewareChain>()?;
    m.add_function(wrap_pyfunction!(create_middleware_chain, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_observer_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "observer")?;
    m.add_class::<Observable>()?;
    m.add_function(wrap_pyfunction!(create_observable, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_plugin_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "plugin")?;
    m.add_class::<PluginManager>()?;
    m.add_function(wrap_pyfunction!(create_plugin_manager, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_logger_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "logger")?;
    m.add_class::<Logger>()?;
    m.add_function(wrap_pyfunction!(create_logger, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_async_utils_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "async_utils")?;
    m.add_class::<AsyncSemaphore>()?;
    m.add_class::<AsyncRateLimiter>()?;
    m.add_class::<AsyncTimer>()?;
    m.add_class::<AsyncBatchProcessor>()?;
    m.add_function(wrap_pyfunction!(create_async_semaphore, &m)?)?;
    m.add_function(wrap_pyfunction!(create_async_rate_limiter, &m)?)?;
    m.add_function(wrap_pyfunction!(create_async_timer, &m)?)?;
    m.add_function(wrap_pyfunction!(create_async_batch_processor, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_serialization_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "serialization")?;
    m.add_class::<Serializer>()?;
    m.add_function(wrap_pyfunction!(create_serializer, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_retry_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "retry")?;
    m.add_class::<RetryExecutor>()?;
    m.add_class::<CircuitBreaker>()?;
    m.add_function(wrap_pyfunction!(create_retry_executor, &m)?)?;
    m.add_function(wrap_pyfunction!(create_circuit_breaker, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_pool_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "pool")?;
    m.add_class::<ResourcePool>()?;
    m.add_function(wrap_pyfunction!(create_resource_pool, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_rate_limiter_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "rate_limiter")?;
    m.add_class::<RateLimiter>()?;
    m.add_function(wrap_pyfunction!(create_rate_limiter, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_backpressure_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "backpressure")?;
    m.add_class::<BackpressureController>()?;
    m.add_function(wrap_pyfunction!(create_backpressure_controller, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_telemetry_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "telemetry")?;
    m.add_class::<TelemetryCollector>()?;
    m.add_function(wrap_pyfunction!(create_telemetry_collector, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_context_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "context")?;
    m.add_class::<RequestContext>()?;
    m.add_class::<ContextManager>()?;
    m.add_function(wrap_pyfunction!(create_context_manager, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_cache_strategies_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "cache_strategies")?;
    m.add_class::<AdvancedCache>()?;
    m.add_function(wrap_pyfunction!(create_advanced_cache, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_scheduler_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "scheduler")?;
    m.add_class::<TaskScheduler>()?;
    m.add_function(wrap_pyfunction!(create_task_scheduler, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_workflow_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "workflow")?;
    m.add_class::<Workflow>()?;
    m.add_function(wrap_pyfunction!(create_workflow, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_distributed_lock_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "distributed_lock")?;
    m.add_class::<DistributedLock>()?;
    m.add_class::<LockManager>()?;
    m.add_function(wrap_pyfunction!(create_lock_manager, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_state_machine_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "state_machine")?;
    m.add_class::<StateMachine>()?;
    m.add_function(wrap_pyfunction!(create_state_machine, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_feature_flags_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "feature_flags")?;
    m.add_class::<FeatureFlagManager>()?;
    m.add_function(wrap_pyfunction!(create_feature_flag_manager, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

fn register_metrics_aggregator_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "metrics_aggregator")?;
    m.add_class::<MetricsAggregator>()?;
    m.add_function(wrap_pyfunction!(create_metrics_aggregator, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

