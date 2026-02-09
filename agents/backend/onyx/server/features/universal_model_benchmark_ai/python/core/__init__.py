"""
Core Module - Centralized exports for core functionality.

This module provides a clean, organized interface to all core functionality.
Imports are organized by category for better maintainability.

Organization:
- Configuration & Constants
- Utilities & Validation
- Logging
- Model Loading
- Results Management
- Analytics & Monitoring
- Experiments & Registry
- Distributed & Cost
- Queue & Scheduling
- Performance & Metrics
- Security & Auth
- Export & Documentation
- Database & Migrations
- Resilience (Circuit Breaker, Retry, Timeout)
- Feature Management
- Backup & Recovery
- Event System
- Middleware
- Dynamic Configuration
- Health Checks
- Rust Integration (Optional)
- Lazy Imports (Optional Modules)
"""

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

from .config import (
    DeviceType,
    QuantizationType,
    ModelConfig,
    BenchmarkConfig,
    ExecutionConfig,
    SystemConfig,
    load_config,
    save_config,
)

from .constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_BATCH_SIZE,
    MAX_TOKENS_LIMIT,
    MAX_BATCH_SIZE,
    MIN_BATCH_SIZE,
    get_default_inference_config,
    get_default_benchmark_config,
    get_benchmark_config,
    get_context_length,
)

# ════════════════════════════════════════════════════════════════════════════════
# UTILITIES & VALIDATION
# ════════════════════════════════════════════════════════════════════════════════

from .utils import (
    measure_time,
    retry_on_failure,
    format_size,
    format_duration,
    save_results,
    load_results,
    get_memory_usage,
    calculate_throughput,
    calculate_percentiles,
)

from .validation import (
    BasicValidationError as ValidationError,
    validate_path,
    validate_range,
    validate_positive,
    validate_in_list,
    validate_type,
    validate_inference_config,
    validate_benchmark_config,
    sanitize_text,
    validate_and_sanitize_prompt,
    ValidationLevel,
    AdvancedValidationError,
    ValidationResult,
    Validator,
    RequiredValidator,
    TypeValidator,
    RangeValidator,
    RegexValidator,
    LengthValidator,
    CustomValidator,
    ValidationSchema,
    AdvancedValidator,
)

# ════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════════

from .logging_config import (
    LogLevel,
    setup_logging,
    get_logger,
    configure_module_logging,
    LoggingContext,
    log_performance,
    log_error_with_context,
)

# ════════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (OPTIONAL - Heavy Dependencies)
# ════════════════════════════════════════════════════════════════════════════════

try:
    from .model_loader import (
        ModelType,
        BackendType,
        ModelConfig as ModelLoaderConfig,
        GenerationConfig,
        BaseBackend,
        ModelLoader,
        create_backend,
        auto_select_backend,
        get_available_backends,
    )
    MODEL_LOADER_AVAILABLE = True
except ImportError:
    MODEL_LOADER_AVAILABLE = False
    ModelType = None
    BackendType = None
    ModelLoaderConfig = None
    GenerationConfig = None
    BaseBackend = None
    ModelLoader = None
    create_backend = None
    auto_select_backend = None
    get_available_backends = None

# ════════════════════════════════════════════════════════════════════════════════
# RESULTS MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════════

from .results import (
    ResultStatus,
    BenchmarkResult,
    ModelResults,
    ComparisonResults,
    ResultsManager,
)

# ════════════════════════════════════════════════════════════════════════════════
# ANALYTICS & MONITORING
# ════════════════════════════════════════════════════════════════════════════════

from .analytics import (
    TrendAnalysis,
    Anomaly,
    AnalyticsEngine,
)

from .monitoring import (
    AlertLevel,
    Alert,
    MetricCollector,
    AlertManager,
    HealthMonitor,
)

# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS & REGISTRY
# ════════════════════════════════════════════════════════════════════════════════

from .experiments import (
    ExperimentStatus,
    ExperimentConfig,
    Experiment,
    ExperimentManager,
)

from .model_registry import (
    ModelStatus,
    ModelMetadata,
    ModelVersion,
    ModelRegistry,
)

# ════════════════════════════════════════════════════════════════════════════════
# DISTRIBUTED & COST
# ════════════════════════════════════════════════════════════════════════════════

from .distributed import (
    NodeStatus,
    Node,
    DistributedTask,
    DistributedExecutor,
)

from .cost_tracking import (
    ResourceType,
    ResourceUsage,
    CostRecord,
    CostTracker,
)

# ════════════════════════════════════════════════════════════════════════════════
# QUEUE & SCHEDULING
# ════════════════════════════════════════════════════════════════════════════════

from .infrastructure import (
    TaskStatus,
    Task,
    TaskQueue,
    ScheduleType,
    ScheduledTask,
    CronParser,
    TaskScheduler,
    CacheBackend,
    CacheEntry,
    MemoryCache,
    DistributedCache,
    ServiceStatus,
    Service,
    ServiceRegistry,
    LoadBalancer,
)

# ════════════════════════════════════════════════════════════════════════════════
# PERFORMANCE & METRICS
# ════════════════════════════════════════════════════════════════════════════════

from .rate_limiter import (
    RateLimit,
    TokenBucket,
    SlidingWindowLimiter,
    RateLimiter,
)

from .metrics import (
    MetricsCollector,
    metrics_collector,
)

from .performance import (
    PerformanceProfile,
    PerformanceProfiler,
    MemoryOptimizer,
    CPUOptimizer,
    PerformanceOptimizer,
    performance_optimizer,
)

# ════════════════════════════════════════════════════════════════════════════════
# RESILIENCE (CIRCUIT BREAKER, RETRY, TIMEOUT)
# ════════════════════════════════════════════════════════════════════════════════

from .resilience import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitBreakerManager,
    RetryStrategy,
    RetryPolicy,
    RetryResult,
    RetryExecutor,
    retry,
    RetryManager,
    get_retry_manager,
    TimeoutError,
    TimeoutManager,
    with_timeout,
    get_timeout_manager,
)

# ════════════════════════════════════════════════════════════════════════════════
# SECURITY & AUTHENTICATION
# ════════════════════════════════════════════════════════════════════════════════

from .auth import (
    UserRole,
    User,
    Token,
    AuthManager,
)

# ════════════════════════════════════════════════════════════════════════════════
# EXPORT & DOCUMENTATION
# ════════════════════════════════════════════════════════════════════════════════

from .export import (
    ExportFormat,
    Exporter,
)

from .documentation import (
    DocSection,
    DocumentationGenerator,
)

# ════════════════════════════════════════════════════════════════════════════════
# DATABASE & MIGRATIONS
# ════════════════════════════════════════════════════════════════════════════════

from .migrations import (
    Migration,
    MigrationManager,
)

# ════════════════════════════════════════════════════════════════════════════════
# FEATURE MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════════

from .feature_flags import (
    FlagType,
    FeatureFlag,
    FeatureFlagManager,
)

# ════════════════════════════════════════════════════════════════════════════════
# BACKUP & RECOVERY
# ════════════════════════════════════════════════════════════════════════════════

from .backup import (
    BackupInfo,
    BackupManager,
)

# ════════════════════════════════════════════════════════════════════════════════
# EVENT SYSTEM
# ════════════════════════════════════════════════════════════════════════════════

from .event_bus import (
    EventType,
    Event,
    EventBus,
    event_bus,
)

# ════════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE
# ════════════════════════════════════════════════════════════════════════════════

from .middleware import (
    Request,
    Response,
    MiddlewareFunc,
    MiddlewarePipeline,
    logging_middleware,
    timing_middleware,
    cors_middleware,
    error_handling_middleware,
    rate_limit_middleware,
    authentication_middleware,
)

# ════════════════════════════════════════════════════════════════════════════════
# DYNAMIC CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

from .dynamic_config import (
    ConfigChange,
    DynamicConfigManager,
)

# ════════════════════════════════════════════════════════════════════════════════
# HEALTH CHECKS
# ════════════════════════════════════════════════════════════════════════════════

from .health_check import (
    HealthStatus,
    HealthCheckResult,
    HealthChecker,
    database_health_check,
    memory_health_check,
)

# Advanced validation now imported from .validation above

# Infrastructure modules now imported above from .infrastructure

from .serialization import (
    SerializationFormat,
    SerializationOptions,
    Serializer,
)

from .testing_utils import (
    MockDataGenerator,
    PerformanceTestHelper,
    AssertionHelper,
)

from .environment import (
    Environment,
    EnvironmentConfig,
    EnvironmentManager,
    env_manager,
)

from .errors import (
    ErrorSeverity,
    RecoveryStrategy,
    ErrorContext,
    RecoveryPolicy,
    ErrorRecoveryManager,
    ErrorInfo,
    ErrorHandler,
    ErrorAggregator,
    format_error,
    log_error,
    handle_error,
    recover_from_error,
    classify_error,
)

from .types import (
    ConfigDict,
    ResultDict,
    MetadataDict,
    PayloadDict,
    MaybeResult,
    MaybeList,
    MaybeDict,
    HandlerFunc,
    ValidatorFunc,
    TransformerFunc,
    Configurable,
    Validatable,
    Serializable,
    Timestamped,
    Identifiable,
)

from .factory import (
    Factory,
    benchmark_factory,
    model_factory,
    backend_factory,
    create_benchmark,
    create_model,
    create_backend,
)

# ════════════════════════════════════════════════════════════════════════════════
# RUST INTEGRATION (OPTIONAL)
# ════════════════════════════════════════════════════════════════════════════════

try:
    from .rust_integration import (
        RustInferenceWrapper,
        RustDataProcessorWrapper,
        RustMetricsWrapper,
        calculate_metrics_rust,
        get_rust_version,
        is_rust_available,
        RUST_AVAILABLE,
    )
except ImportError:
    RUST_AVAILABLE = False
    RustInferenceWrapper = None
    RustDataProcessorWrapper = None
    RustMetricsWrapper = None
    calculate_metrics_rust = None
    get_rust_version = None
    is_rust_available = lambda: False

# ════════════════════════════════════════════════════════════════════════════════
# LAZY IMPORTS (OPTIONAL MODULES)
# ════════════════════════════════════════════════════════════════════════════════

_LAZY_IMPORTS = {
    'ReportGenerator': '.reporting',
    'ReportConfig': '.reporting',
    'generate_summary_report': '.reporting',
    'VisualizationGenerator': '.visualization',
    'ChartData': '.visualization',
    'prepare_dashboard_data': '.visualization',
    'ModelOptimizer': '.optimizer',
    'OptimizationConfig': '.optimizer',
    'OptimizationLevel': '.optimizer',
    'create_optimization_config': '.optimizer',
}

_import_cache = {}

def __getattr__(name: str):
    """Lazy import system for optional modules."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name in _import_cache:
        return _import_cache[name]
    
    module_path = _LAZY_IMPORTS[name]
    try:
        module = __import__(module_path, fromlist=[name], level=1)
        if hasattr(module, name):
            _import_cache[name] = getattr(module, name)
            return _import_cache[name]
        else:
            _import_cache[name] = module
            return module
    except (ImportError, AttributeError) as e:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Failed to import: {e}"
        ) from e

# ════════════════════════════════════════════════════════════════════════════════
# PUBLIC API - __all__
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Configuration
    "DeviceType",
    "QuantizationType",
    "ModelConfig",
    "BenchmarkConfig",
    "ExecutionConfig",
    "SystemConfig",
    "load_config",
    "save_config",
    # Constants
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DEFAULT_TOP_K",
    "DEFAULT_BATCH_SIZE",
    "MAX_TOKENS_LIMIT",
    "MAX_BATCH_SIZE",
    "MIN_BATCH_SIZE",
    "get_default_inference_config",
    "get_default_benchmark_config",
    "get_benchmark_config",
    "get_context_length",
    # Utils
    "measure_time",
    "retry_on_failure",
    "format_size",
    "format_duration",
    "save_results",
    "load_results",
    "get_memory_usage",
    "calculate_throughput",
    "calculate_percentiles",
    # Validation
    "ValidationError",
    "validate_path",
    "validate_range",
    "validate_positive",
    "validate_in_list",
    "validate_type",
    "validate_inference_config",
    "validate_benchmark_config",
    "sanitize_text",
    "validate_and_sanitize_prompt",
    # Logging
    "LogLevel",
    "setup_logging",
    "get_logger",
    "configure_module_logging",
    "LoggingContext",
    "log_performance",
    "log_error_with_context",
    # Model Loader
    "MODEL_LOADER_AVAILABLE",
    "ModelType",
    "BackendType",
    "ModelLoaderConfig",
    "GenerationConfig",
    "BaseBackend",
    "ModelLoader",
    "create_backend",
    "auto_select_backend",
    "get_available_backends",
    # Results
    "ResultStatus",
    "BenchmarkResult",
    "ModelResults",
    "ComparisonResults",
    "ResultsManager",
    # Analytics
    "TrendAnalysis",
    "Anomaly",
    "AnalyticsEngine",
    # Monitoring
    "AlertLevel",
    "Alert",
    "MetricCollector",
    "AlertManager",
    "HealthMonitor",
    # Experiments
    "ExperimentStatus",
    "ExperimentConfig",
    "Experiment",
    "ExperimentManager",
    # Model Registry
    "ModelStatus",
    "ModelMetadata",
    "ModelVersion",
    "ModelRegistry",
    # Distributed
    "NodeStatus",
    "Node",
    "DistributedTask",
    "DistributedExecutor",
    # Cost Tracking
    "ResourceType",
    "ResourceUsage",
    "CostRecord",
    "CostTracker",
    # Queue
    "TaskStatus",
    "Task",
    "TaskQueue",
    # Scheduler
    "ScheduleType",
    "ScheduledTask",
    "CronParser",
    "TaskScheduler",
    # Rate Limiting
    "RateLimit",
    "TokenBucket",
    "SlidingWindowLimiter",
    "RateLimiter",
    # Metrics
    "MetricsCollector",
    "metrics_collector",
    # Performance
    "PerformanceProfile",
    "PerformanceProfiler",
    "MemoryOptimizer",
    "CPUOptimizer",
    "PerformanceOptimizer",
    "performance_optimizer",
    # Circuit Breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitBreakerManager",
    # Retry
    "RetryStrategy",
    "RetryPolicy",
    "RetryResult",
    "RetryExecutor",
    "retry",
    # Timeout
    "TimeoutError",
    "TimeoutPolicy",
    "TimeoutManager",
    "with_timeout",
    # Auth
    "UserRole",
    "User",
    "Token",
    "AuthManager",
    # Export
    "ExportFormat",
    "Exporter",
    # Documentation
    "DocSection",
    "DocumentationGenerator",
    # Migrations
    "Migration",
    "MigrationManager",
    # Feature Flags
    "FlagType",
    "FeatureFlag",
    "FeatureFlagManager",
    # Backup
    "BackupInfo",
    "BackupManager",
    # Event Bus
    "EventType",
    "Event",
    "EventBus",
    "event_bus",
    # Middleware
    "Request",
    "Response",
    "MiddlewareFunc",
    "MiddlewarePipeline",
    "logging_middleware",
    "timing_middleware",
    "cors_middleware",
    "error_handling_middleware",
    "rate_limit_middleware",
    "authentication_middleware",
    # Dynamic Config
    "ConfigChange",
    "DynamicConfigManager",
    # Health Check
    "HealthStatus",
    "HealthCheckResult",
    "HealthChecker",
    "database_health_check",
    "memory_health_check",
    # Advanced Validation
    "ValidationLevel",
    "AdvancedValidationError",
    "ValidationResult",
    "Validator",
    "RequiredValidator",
    "TypeValidator",
    "RangeValidator",
    "RegexValidator",
    "LengthValidator",
    "CustomValidator",
    "ValidationSchema",
    "AdvancedValidator",
    # Distributed Cache
    "CacheBackend",
    "CacheEntry",
    "MemoryCache",
    "DistributedCache",
    # Service Discovery
    "ServiceStatus",
    "Service",
    "ServiceRegistry",
    "LoadBalancer",
    # Serialization
    "SerializationFormat",
    "SerializationOptions",
    "Serializer",
    # Testing Utils
    "MockDataGenerator",
    "PerformanceTestHelper",
    "AssertionHelper",
    # Environment
    "Environment",
    "EnvironmentConfig",
    "EnvironmentManager",
    "env_manager",
    # Error Recovery
    "ErrorSeverity",
    "RecoveryStrategy",
    "ErrorContext",
    "RecoveryPolicy",
    "ErrorRecoveryManager",
    # Types
    "ConfigDict",
    "ResultDict",
    "MetadataDict",
    "PayloadDict",
    "MaybeResult",
    "MaybeList",
    "MaybeDict",
    "HandlerFunc",
    "ValidatorFunc",
    "TransformerFunc",
    "Configurable",
    "Validatable",
    "Serializable",
    "Timestamped",
    "Identifiable",
    # Factory
    "Factory",
    "benchmark_factory",
    "model_factory",
    "backend_factory",
    "create_benchmark",
    "create_model",
    "create_backend",
    # Rust Integration
    "RUST_AVAILABLE",
    "RustInferenceWrapper",
    "RustDataProcessorWrapper",
    "RustMetricsWrapper",
    "calculate_metrics_rust",
    "get_rust_version",
    "is_rust_available",
    # Reporting (lazy)
    "ReportGenerator",
    "ReportConfig",
    "generate_summary_report",
    # Visualization (lazy)
    "VisualizationGenerator",
    "ChartData",
    "prepare_dashboard_data",
    # Optimizer (lazy)
    "ModelOptimizer",
    "OptimizationConfig",
    "OptimizationLevel",
    "create_optimization_config",
]
