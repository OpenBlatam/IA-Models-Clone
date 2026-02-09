//! Benchmark Core - High-performance Rust library for model benchmarking
//!
//! This library provides a comprehensive suite of tools for benchmarking machine learning models,
//! with a focus on performance, accuracy, and ease of use.
//!
//! # Features
//!
//! - **Fast Inference Engine**: High-performance inference using Candle framework
//! - **Efficient Metrics Calculation**: Comprehensive metrics with statistical analysis
//! - **Data Processing**: High-performance batch processing and template engines
//! - **Error Handling**: Unified error handling system
//! - **Advanced Batching**: Dynamic and continuous batching with priority queues
//! - **Caching**: LRU cache for tokenization and results
//! - **Profiling**: Performance profiling and memory tracking
//! - **Reporting**: Comprehensive report generation and comparison
//! - **Benchmark Runner**: High-level API for running benchmarks
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use benchmark_core::prelude::*;
//!
//! // Option 1: Use a preset for quick configuration
//! let config = fast_inference("path/to/model".to_string())?;
//!
//! // Option 2: Use builder pattern for custom configuration
//! let config = BenchmarkConfig::builder()
//!     .model_path("path/to/model".to_string())
//!     .batch_size(batch_sizes::LARGE)
//!     .max_tokens(token_limits::LONG)
//!     .temperature(temperatures::MEDIUM)
//!     .build()?;
//!
//! // Option 3: Use macro for concise syntax
//! let config = benchmark_config! {
//!     model_path: "path/to/model",
//!     batch_size: batch_sizes::MEDIUM,
//!     max_tokens: token_limits::MEDIUM,
//! }?;
//!
//! // Run benchmark
//! let runner = BenchmarkRunner::new(engine, processor, None)?;
//! let result = runner.run_single("prompt", None)?;
//!
//! // Check if successful
//! if is_benchmark_successful(&result) {
//!     println!("Benchmark successful: {}", benchmark_summary(&result));
//! }
//!
//! // Convert to metrics
//! let metrics: Metrics = (&result).into();
//!
//! // Or use builder
//! let metrics = Metrics::builder()
//!     .accuracy(0.95)
//!     .latency_p50(50.0)
//!     .throughput(100.0)
//!     .build();
//!
//! // Check performance
//! if is_good_performance(&metrics) {
//!     println!("Performance rating: {}", performance_rating(&metrics));
//!     println!("Metrics: {}", metrics_summary(&metrics));
//! }
//!
//! // Generate report
//! let report = ReportGenerator::generate_report(
//!     "model-name",
//!     "benchmark-name",
//!     &metrics
//! );
//! ```
//!
//! # Modules
//!
//! - [`inference`]: Inference engine and tokenization
//! - [`metrics`]: Metrics calculation and aggregation
//! - [`data`]: Data processing and templates
//! - [`error`]: Error handling
//! - [`cache`]: Caching utilities
//! - [`profiling`]: Performance profiling
//! - [`reporting`]: Report generation
//! - [`batching`]: Batch processing
//! - [`utils`]: Utility functions
//! - [`config`]: Configuration management
//! - [`types`]: Common type definitions
//! - [`benchmark`]: Benchmark runner
//! - [`traits`]: Useful traits for common operations
//! - [`iterators`]: Iterator utilities and adapters
//! - [`safety`]: Safety utilities and error handling helpers
//! - [`extensions`]: Extension traits for common types
//! - [`convert`]: Type conversion utilities
//! - [`constants`]: Centralized constants and presets
//! - [`presets`]: Pre-configured settings for common use cases
//! - [`helpers`]: Helper functions for common operations
//! - [`validation`]: Advanced validation utilities
//! - [`analysis`]: Statistical analysis utilities
//! - [`comparison`]: Benchmark comparison utilities
//! - [`logging`]: Logging and observability utilities
//! - [`macros`]: Useful macros for common operations
//! - [`prelude`]: Convenient re-exports

pub mod inference;
pub mod metrics;
pub mod data;
pub mod error;
pub mod cache;
pub mod profiling;
pub mod reporting;
pub mod batching;
pub mod utils;
pub mod config;
pub mod types;
pub mod benchmark;
pub mod traits;
pub mod iterators;
pub mod safety;
pub mod extensions;
pub mod convert;
pub mod constants;
pub mod presets;
pub mod helpers;
pub mod validation;
pub mod analysis;
pub mod comparison;
pub mod logging;

#[cfg(test)]
mod testing;

// Macros are exported at crate root level
#[macro_use]
mod macros;

// Re-export config module types at top level for convenience
pub use config::{BenchmarkConfig, BenchmarkConfigBuilder};

/// Prelude module for convenient imports.
///
/// This module re-exports commonly used types and functions.
/// Use `use benchmark_core::prelude::*;` for easy access to common types.
pub mod prelude {
    pub use crate::{
        BenchmarkConfig,
        BenchmarkConfigBuilder,
        Metrics,
        MetricsWeights,
        PerformanceThresholds,
        TokenId,
        TokenSequence,
        TokenBatch,
        SystemInfo,
        PerformanceSummary,
        get_version,
        get_name,
        get_description,
        get_system_info,
        has_python_support,
    };
    
    pub use crate::error::{
        BenchmarkError,
        Result as BenchmarkResult,
    };
    
    pub use crate::inference::{
        InferenceEngine,
        InferenceConfig,
        InferenceStats,
        TokenizerWrapper,
        SamplingConfig,
        SamplingStrategy,
        InferenceError,
        InferenceResult,
        MetricsCollector,
        InferenceMetrics,
        Timer,
    };
    
    pub use crate::metrics::{
        calculate_metrics,
        calculate_statistics,
        calculate_accuracy,
        calculate_throughput,
        calculate_latency_stats,
        aggregate_metrics,
    };
    
    pub use crate::data::{
        DataProcessor,
        DataProcessorConfig,
        TemplateEngine,
    };
    
    pub use crate::benchmark::{
        BenchmarkRunner,
        BenchmarkRunnerConfig,
        BenchmarkResult,
    };
    
    // Formatting
    pub use crate::utils::{
        format_duration,
        format_bytes,
        format_number,
        format_percentage,
        format_latency_ms,
        format_throughput,
    };
    
    // Statistics
    pub use crate::utils::{
        percentile,
        percentiles,
        mean,
        median,
        std_dev,
        summary_stats,
    };
    
    // Validation
    pub use crate::utils::{
        clamp,
        clamp_i64,
        clamp_usize,
        in_range,
        in_range_i64,
        is_positive,
        is_non_negative,
        is_finite,
        validate_range,
        validate_positive,
    };
    
    // Timing
    pub use crate::utils::{
        measure_time,
        measure_duration,
        Timer,
        ScopedTimer,
    };
    
    // Types
    pub use crate::batching::{
        BatchPriority,
        BatchItem,
        BatchStats,
    };
    
    // Dynamic batching
    pub use crate::batching::DynamicBatcher;
    
    // Continuous batching
    pub use crate::batching::{
        ContinuousBatcher,
        BatchManager,
        create_batch_manager,
    };
    
    pub use crate::cache::{
        LRUCache,
        CacheStats,
        TokenizationCache,
        ResultCache,
        create_tokenization_cache,
        create_result_cache,
    };
    
    pub use crate::profiling::{
        Profiler,
        MemorySnapshot,
        TimingStats,
        PerformanceReport,
        Timer,
    };
    
    // Types
    pub use crate::reporting::{
        BenchmarkReport,
        ReportSamples,
        PerformanceBreakdown,
        ModelComparison,
        ComparisonReport,
    };
    
    // Generator
    pub use crate::reporting::ReportGenerator;
    
    // Export
    pub use crate::reporting::{
        export_reports_json,
        export_comparison_json,
        export_comparison_markdown,
        export_report_json,
    };
    
    pub use crate::traits::{
        Validate,
        Summarize,
        ToMetrics,
        ToJson,
        FromJson,
        PerformanceStats,
        Reset,
        Statistics,
    };
    
    pub use crate::iterators::{
        BatchExt,
        WindowExt,
        EnumerateFromExt,
        TakeWhileInclusiveExt,
    };
    
    pub use crate::safety::{
        safe_lock,
        safe_unwrap,
        validate_range,
        validate_positive,
        validate_non_empty_string,
        validate_finite,
    };
    
    pub use crate::types::MetricsBuilder;
    
    pub use crate::extensions::{
        F64SliceExt,
        ResultExt,
        StringExt,
        VecExt,
        OptionExt,
    };
    
    pub use crate::convert::{
        benchmark_result_to_metrics,
        latencies_to_metrics,
        f64_to_usize,
        usize_to_f64,
        i64_to_f64,
        f64_to_i64,
        str_to_f64,
        str_to_usize,
        option_to_result,
    };
    
    pub use crate::constants::{
        percentiles,
        time_ms,
        size_bytes,
        thresholds,
        batch_sizes,
        token_limits,
        temperatures,
        top_p_values,
        top_k_values,
        retry,
        cache,
        benchmark as benchmark_constants,
    };
    
    pub use crate::presets::{
        fast_inference,
        high_throughput,
        creative_generation,
        deterministic,
        long_context,
        balanced,
        code_generation,
        conversational,
        summarization,
        question_answering,
    };
    
    pub use crate::helpers::{
        is_good_performance,
        is_excellent_performance,
        performance_rating,
        improvement_percentage,
        is_benchmark_successful,
        benchmark_summary,
        metrics_summary,
        compare_metrics,
        validate_metrics,
        normalize_metrics,
    };
    
    pub use crate::validation::{
        validate_in_range,
        validate_positive,
        validate_non_negative,
        validate_not_empty,
        validate_not_empty_slice,
        validate_finite,
        validate_metrics_comprehensive,
        validate_metrics_performance,
        validate_config_comprehensive,
    };
    
    pub use crate::analysis::{
        StatisticalSummary,
        ComparisonResult,
        LatencyAnalysis,
        Trend,
        compare_summaries,
        analyze_latency_distribution,
        correlation,
        calculate_trend,
    };
    
    pub use crate::comparison::{
        BenchmarkComparison,
        ComparisonWinner,
        BenchmarkComparator,
        BenchmarkRanking,
        ConfigComparison,
        compare_latency_distributions,
    };
    
    pub use crate::logging::{
        BenchmarkLogger,
        LogLevel,
        ScopedLogger,
        set_global_logger,
        get_logger,
        log,
        trace,
        debug,
        info,
        warn,
        error,
    };
}

#[cfg(feature = "python")]
pub mod python_bindings;

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ════════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - CONFIG
// ════════════════════════════════════════════════════════════════════════════════

pub use config::{
    BenchmarkConfig,
    BenchmarkConfigBuilder,
    defaults,
    limits,
};

// ════════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - TYPES
// ════════════════════════════════════════════════════════════════════════════════

// Type aliases
pub use types::{
    TokenId,
    TokenSequence,
    TokenBatch,
    Metadata,
    ConfigMap,
};

// Metrics types
pub use types::{
    Metrics,
    MetricsBuilder,
    MetricsWeights,
    PerformanceThresholds,
    PerformanceSummary,
};

// System types
pub use types::{
    VersionInfo,
    SystemInfo,
};

// ════════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - INFERENCE
// ════════════════════════════════════════════════════════════════════════════════

// Core inference types
pub use inference::{
    InferenceEngine,
    InferenceConfig,
    InferenceStats,
    TokenizerWrapper,
    SamplingConfig,
    SamplingStrategy,
    InferenceError,
    InferenceResult,
};

// Advanced batching (from inference module - specialized)
// Note: General batching types are exported from batching module below

// Metrics and monitoring
pub use inference::{
    MetricsCollector,
    InferenceMetrics,
    Timer,
};

// ════════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - METRICS
// ════════════════════════════════════════════════════════════════════════════════

pub use metrics::{
    calculate_metrics,
    calculate_statistics,
    calculate_accuracy,
    calculate_throughput,
    calculate_latency_stats,
    aggregate_metrics,
};

// ════════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - DATA
// ════════════════════════════════════════════════════════════════════════════════

pub use data::{
    DataProcessor,
    DataProcessorConfig,
    TemplateEngine,
    validate_non_empty,
    validate_length,
    validate_batch_size,
    validate_batch_not_empty,
    validate_template,
};

// ════════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - ERROR
// ════════════════════════════════════════════════════════════════════════════════

pub use error::{
    BenchmarkError,
    Result as BenchmarkResult,
};

// ════════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - UTILS
// ════════════════════════════════════════════════════════════════════════════════

pub use utils::{
    // Formatting
    format_duration,
    format_bytes,
    format_number,
    format_percentage,
    format_latency_ms,
    format_throughput,
    // Statistics
    percentile,
    percentiles,
    mean,
    median,
    std_dev,
    summary_stats,
    // Validation
    clamp,
    clamp_i64,
    clamp_usize,
    in_range,
    in_range_i64,
    is_positive,
    is_non_negative,
    is_finite,
    validate_range,
    validate_positive,
    // Timing
    measure_time,
    measure_duration,
    Timer,
    ScopedTimer,
    median,
    normalize,
    denormalize,
};

// ════════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - CACHE
// ════════════════════════════════════════════════════════════════════════════════

pub use cache::{
    LRUCache,
    CacheStats,
    TokenizationCache,
    ResultCache,
    create_tokenization_cache,
    create_result_cache,
    cached_tokenize,
    cached_result,
};

// ════════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - PROFILING
// ════════════════════════════════════════════════════════════════════════════════

pub use profiling::{
    Profiler,
    MemorySnapshot,
    TimingStats,
    PerformanceReport,
    Timer,
};

// ════════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - BATCHING
// ════════════════════════════════════════════════════════════════════════════════

pub use batching::{
    // Types
    BatchPriority,
    BatchItem,
    BatchStats,
    // Dynamic batching
    DynamicBatcher,
    // Continuous batching
    ContinuousBatcher,
    BatchManager,
    create_batch_manager,
};

// ════════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - REPORTING
// ════════════════════════════════════════════════════════════════════════════════

pub use reporting::{
    // Types
    BenchmarkReport,
    ReportSamples,
    PerformanceBreakdown,
    ModelComparison,
    ComparisonReport,
    // Generator
    ReportGenerator,
    // Export
    export_reports_json,
    export_comparison_json,
    export_comparison_markdown,
    export_report_json,
};

// ════════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - BENCHMARK
// ════════════════════════════════════════════════════════════════════════════════

pub use benchmark::{
    BenchmarkRunner,
    BenchmarkRunnerConfig,
    BenchmarkResult,
};

// ════════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - TRAITS
// ════════════════════════════════════════════════════════════════════════════════

pub use traits::{
    Validate,
    Summarize,
    ToMetrics,
    ToJson,
    FromJson,
    PerformanceStats,
    Reset,
    Statistics,
};

// ════════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - ITERATORS
// ════════════════════════════════════════════════════════════════════════════════

pub use iterators::{
    BatchIterator,
    WindowIterator,
    EnumerateFrom,
    TakeWhileInclusive,
    BatchExt,
    WindowExt,
    EnumerateFromExt,
    TakeWhileInclusiveExt,
};

// ════════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS - SAFETY
// ════════════════════════════════════════════════════════════════════════════════

pub use safety::{
    safe_lock,
    safe_read_lock,
    safe_write_lock,
    safe_unwrap,
    safe_result,
    validate_range,
    validate_positive,
    validate_non_negative,
    validate_non_empty_string,
    validate_non_empty_slice,
    validate_finite,
    ErrorContext,
};

// ════════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════════

/// Get library version.
pub fn get_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Get library name.
pub fn get_name() -> &'static str {
    env!("CARGO_PKG_NAME")
}

/// Get library description.
pub fn get_description() -> &'static str {
    env!("CARGO_PKG_DESCRIPTION")
}

/// Get system information.
pub fn get_system_info() -> HashMap<String, String> {
    let mut info = HashMap::new();
    
    info.insert("version".to_string(), get_version().to_string());
    info.insert("name".to_string(), get_name().to_string());
    info.insert("description".to_string(), get_description().to_string());
    
    // Add more system info if needed
    if let Ok(hostname) = std::env::var("HOSTNAME") {
        info.insert("hostname".to_string(), hostname);
    }
    
    info
}

/// Check if a feature is enabled.
#[cfg(feature = "python")]
pub fn has_python_support() -> bool {
    true
}

#[cfg(not(feature = "python"))]
pub fn has_python_support() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version() {
        let version = get_version();
        assert!(!version.is_empty());
    }
    
    #[test]
    fn test_name() {
        let name = get_name();
        assert_eq!(name, "benchmark-core");
    }
    
    #[test]
    fn test_description() {
        let desc = get_description();
        assert!(!desc.is_empty());
    }
}
