//! Re-exports for convenient access
//!
//! Provides convenient re-exports of commonly used types and functions.

// Core services
pub use crate::cache::CacheService;
pub use crate::compression::CompressionService;
pub use crate::batch::BatchProcessor;
pub use crate::text::TextProcessor;
pub use crate::search::SearchEngine;
pub use crate::crypto::HashService;
pub use crate::similarity::SimilarityEngine;
pub use crate::language::LanguageDetector;
pub use crate::id_gen::IdGenerator;
pub use crate::memory::{ObjectPool, RingBuffer, ChunkedBuffer, MemoryTracker};
pub use crate::streaming::{TextStream, ParallelProcessor, LineIterator};
pub use crate::metrics::MetricsCollector;
pub use crate::profiling::Profiler;
pub use crate::health::{HealthChecker, SystemMonitor};

// Configuration
pub use crate::config::{Config, CoreConfig};

// Factory and Builder
pub use crate::factory::{ServiceFactory, ServiceBundle};
pub use crate::builder::{ConfigBuilder, ServiceBundleBuilder};

// Validation
pub use crate::validation::Validator;

// Events, Middleware, Observer, Plugin
pub use crate::events::EventBus;
pub use crate::middleware::MiddlewareChain;
pub use crate::observer::Observable;
pub use crate::plugin::PluginManager;

// Logger, Async, Serialization, Retry
pub use crate::logger::Logger;
pub use crate::async_utils::{AsyncSemaphore, AsyncRateLimiter, AsyncTimer, AsyncBatchProcessor};
pub use crate::serialization::Serializer;
pub use crate::retry::{RetryExecutor, CircuitBreaker};

// Pool, Rate Limiting, Backpressure, Telemetry
pub use crate::pool::ResourcePool;
pub use crate::rate_limiter::RateLimiter;
pub use crate::backpressure::BackpressureController;
pub use crate::telemetry::TelemetryCollector;

// Context, Cache Strategies, Scheduler, Workflow
pub use crate::context::{RequestContext, ContextManager};
pub use crate::cache_strategies::AdvancedCache;
pub use crate::scheduler::TaskScheduler;
pub use crate::workflow::Workflow;

// Distributed Lock, State Machine, Feature Flags, Metrics Aggregator
pub use crate::distributed_lock::{DistributedLock, LockManager};
pub use crate::state_machine::StateMachine;
pub use crate::feature_flags::FeatureFlagManager;
pub use crate::metrics_aggregator::MetricsAggregator;

// Error types
pub use crate::error::TranscriberError;

// Common types
pub use crate::text::{TextStats, TextSegment, Keyword};
pub use crate::search::{SearchResult, SearchFilter, IndexedDocument};
pub use crate::cache::{CacheEntry, CacheStats};
pub use crate::batch::{BatchJob, BatchResult, BatchStats};
pub use crate::crypto::HashResult;
pub use crate::similarity::SimilarityResult;
pub use crate::language::DetectionResult;
pub use crate::memory::{PoolStatsResult, MemoryStats};
pub use crate::streaming::{StreamChunk, ChunkResult, ProcessingProgress, LineItem};
pub use crate::metrics::{TimerStats, HistogramStats, MetricsSummary, Stopwatch};
pub use crate::profiling::Timer as ProfilingTimer;
pub use crate::utils::{Timer, DateUtils, StringUtils, JsonUtils, SubtitleUtils, SRTEntry};

// Utility functions
pub use crate::utils::{get_system_info, create_timer};
pub use crate::factory::{create_service_factory, create_service_bundle};
pub use crate::builder::{create_config_builder, create_service_bundle_builder};
pub use crate::validation::create_validator;
pub use crate::profiling::{create_profiler, create_timer as create_profiling_timer};
pub use crate::health::{create_health_checker, create_system_monitor};
pub use crate::events::create_event_bus;
pub use crate::middleware::create_middleware_chain;
pub use crate::observer::create_observable;
pub use crate::plugin::create_plugin_manager;
pub use crate::logger::create_logger;
pub use crate::async_utils::{create_async_semaphore, create_async_rate_limiter, create_async_timer, create_async_batch_processor};
pub use crate::serialization::create_serializer;
pub use crate::retry::{create_retry_executor, create_circuit_breaker};
pub use crate::pool::create_resource_pool;
pub use crate::rate_limiter::create_rate_limiter;
pub use crate::backpressure::create_backpressure_controller;
pub use crate::telemetry::create_telemetry_collector;
pub use crate::context::create_context_manager;
pub use crate::cache_strategies::create_advanced_cache;
pub use crate::scheduler::create_task_scheduler;
pub use crate::workflow::create_workflow;
pub use crate::distributed_lock::create_lock_manager;
pub use crate::state_machine::create_state_machine;
pub use crate::feature_flags::create_feature_flag_manager;
pub use crate::metrics_aggregator::create_metrics_aggregator;

// Compression functions
pub use crate::compression::{compress_fast, decompress_fast, compress_best};

// ID generation functions
pub use crate::id_gen::{uuid_v4, uuid_v7, ulid, nanoid};

// SIMD JSON functions
pub use crate::simd_json::{parse_json_fast, is_json_valid};

// Memory functions
pub use crate::memory::{create_pool, create_ring_buffer, create_chunked_buffer};

// Streaming functions
pub use crate::streaming::{create_text_stream, create_line_iterator};

// Metrics functions
pub use crate::metrics::{create_metrics_collector, create_stopwatch};

