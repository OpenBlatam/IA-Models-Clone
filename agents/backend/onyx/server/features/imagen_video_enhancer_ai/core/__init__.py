"""
Core Module
===========

Core components and utilities.
"""

# Base classes
from .base_models import (
    BaseModel,
    TimestampedModel,
    IdentifiedModel,
    StatusModel
)

from .repository_base import (
    BaseRepository,
    RepositoryMixin
)

from .manager_base import (
    BaseManager,
    ManagerRegistry
)

from .lifecycle import (
    LifecycleManager,
    LifecycleState,
    LifecycleComponent
)

# Dependency injection
from .dependency_injection import (
    DependencyContainer,
    get_container,
    register,
    get as get_service,
    has as has_service
)

# Component registry
from .component_registry import ComponentRegistry

# Constants
from .constants import (
    DEFAULT_MAX_PARALLEL_TASKS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_CACHE_TTL_HOURS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RATE_LIMIT_RPS,
    DEFAULT_RATE_LIMIT_BURST,
    OUTPUT_DIRECTORIES,
    DEFAULT_MAX_IMAGE_SIZE_MB,
    DEFAULT_MAX_VIDEO_SIZE_MB,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    ENHANCEMENT_TYPES
)

# Types and interfaces
from .types import *
from .interfaces import *

# Core components (imported from imports.py for better organization)
from .imports import (
    EnhancerAgent,
    TaskManager,
    Task,
    TaskStatus,
    TaskEvent,
    ServiceHandler,
    ServiceType,
    ServiceConfig,
    ServiceConfigRegistry,
    ServiceResult,
    ParallelExecutor,
    BatchProcessor,
    CacheManager,
    WebhookManager,
    RetryManager,
    MetricsCollector,
    EventBus,
    VideoProcessor,
    SecurityManager,
    TokenManager,
    Token,
    AuditLogger,
    AuditLog,
    AuditAction,
    AuditLevel,
    Throttler,
    ThrottleConfig,
    ThrottleStrategy,
    QueueManager,
    QueueItem,
    QueuePriority,
    ResourcePool,
    PoolConfig,
    PoolState,
    StrategyManager,
    ValidationManager,
    ValidationRule,
    ValidationResult,
    ValidationLevel,
    timed_operation,
    retry_operation,
    rate_limited_operation,
    cached_operation,
    monitored_operation,
    OperationContext
)

# Code generation
from .code_generator import (
    CodeGenerator,
    CodeTemplate,
    CodeTemplateType,
    GeneratedCode
)

# Seeds system
from .seeds import (
    SeedRunner,
    Seed,
    SeedStatus
)

# Automatic backup
from .auto_backup import (
    AutoBackupManager,
    BackupConfig,
    BackupResult,
    BackupType,
    BackupStatus
)

# Deployment utilities
from .deployment_utils import (
    DeploymentManager,
    DeploymentConfig,
    DeploymentStep,
    DeploymentStage,
    DeploymentStatus,
    EnvironmentChecker
)

# Migrations
from .migrations import (
    MigrationRunner,
    Migration,
    MigrationStatus
)

# API Versioning
from .api_versioning import (
    APIVersionManager,
    APIVersion,
    VersionedEndpoint,
    VersionStrategy
)

# Distributed Cache
from .cache_distributed import (
    DistributedCache,
    CacheEntry,
    CacheBackend,
    CacheConsistency
)

# Advanced Logging
from .logging_advanced import (
    AdvancedLogger,
    LogManager,
    ContextLogger,
    LogConfig,
    LogLevel,
    LogFormat
)

# Advanced Testing
from .testing_advanced import (
    AsyncTestCase,
    TestRunner,
    TestFixture,
    TestResult,
    TestConfig,
    MockBuilder,
    TestDecorator,
    temp_directory,
    temp_file
)

# Advanced Request Validation
from .request_validation_advanced import (
    AdvancedRequestValidator,
    ValidationRule,
    ValidationResult,
    ValidationLevel
)

# Advanced Data Transformation
from .data_transformation_advanced import (
    AdvancedDataTransformer,
    TransformRule,
    TransformResult,
    TransformDirection
)

# Advanced Middleware
from .middleware_advanced import (
    AdvancedMiddleware,
    MiddlewareManager,
    MiddlewareConfig,
    MiddlewareType
)

# Advanced Rate Limiting
from .rate_limiting_advanced import (
    AdvancedRateLimiter,
    RateLimitConfig,
    RateLimitResult,
    RateLimitStrategy
)

# Advanced Circuit Breaker
from .circuit_breaker_advanced import (
    AdvancedCircuitBreaker,
    CircuitBreakerManager,
    CircuitBreakerConfig,
    CircuitBreakerStats,
    CircuitState
)

# Telemetry System
from .telemetry import (
    TelemetryCollector,
    TelemetryManager,
    TelemetryData,
    TelemetryType
)

# Advanced Performance Profiler
from .performance_profiler_advanced import (
    AdvancedPerformanceProfiler,
    PerformanceProfile,
    ProfileResult
)

# Real-time Metrics
from .metrics_realtime import (
    RealTimeMetricsCollector,
    MetricSeries,
    MetricPoint,
    MetricType
)

# Advanced Permissions
from .permissions_advanced import (
    AdvancedPermissionsManager,
    Role,
    User,
    Permission,
    PermissionAction
)

# Advanced Encryption
from .encryption_advanced import (
    AdvancedEncryptionManager,
    EncryptionKey,
    EncryptionAlgorithm
)

# Security Validator
from .security_validator import (
    SecurityValidator,
    SecurityIssue,
    SecurityLevel,
    ValidationResult as SecurityValidationResult
)

# Advanced Audit
from .audit_advanced import (
    AdvancedAuditLogger,
    AuditEntry,
    AuditAction,
    AuditLevel
)

# Advanced Health Monitoring
from .health_monitoring_advanced import (
    AdvancedHealthMonitor,
    HealthCheck,
    HealthCheckResult,
    SystemHealth,
    HealthStatus
)

# Advanced Retry
from .retry_advanced import (
    AdvancedRetryManager,
    RetryConfig,
    RetryResult,
    RetryStrategy
)

# Advanced Queue
from .queue_advanced import (
    AdvancedQueue,
    QueueItem,
    QueuePriority,
    QueueItemStatus
)

# Advanced Event Bus
from .event_bus_advanced import (
    AdvancedEventBus,
    Event,
    EventHandler,
    EventPriority
)

# Advanced Cache Strategy
from .cache_strategy_advanced import (
    AdvancedCacheStrategy,
    CacheEntry,
    CacheEvictionPolicy
)

# Advanced Validation
from .validation_advanced import (
    AdvancedValidator,
    ValidationRule,
    ValidationResult,
    ValidationLevel
)

__all__ = [
    # Base classes
    "BaseModel",
    "TimestampedModel",
    "IdentifiedModel",
    "StatusModel",
    "BaseRepository",
    "RepositoryMixin",
    "BaseManager",
    "ManagerRegistry",
    "LifecycleManager",
    "LifecycleState",
    "LifecycleComponent",
    # Dependency injection
    "DependencyContainer",
    "get_container",
    "register",
    "get_service",
    "has_service",
    # Component registry
    "ComponentRegistry",
    # Core components
    "EnhancerAgent",
    "TaskManager",
    "Task",
    "TaskStatus",
    "TaskEvent",
    "ServiceHandler",
    "ServiceType",
    "ServiceConfig",
    "ServiceConfigRegistry",
    "ServiceResult",
    "ParallelExecutor",
    "BatchProcessor",
    "CacheManager",
    "WebhookManager",
    "RetryManager",
    "MetricsCollector",
    "EventBus",
    "VideoProcessor",
    # Security systems
    "SecurityManager",
    "TokenManager",
    "Token",
    "AuditLogger",
    "AuditLog",
    "AuditAction",
    "AuditLevel",
    "Throttler",
    "ThrottleConfig",
    "ThrottleStrategy",
    "QueueManager",
    "QueueItem",
    "QueuePriority",
    "ResourcePool",
    "PoolConfig",
    "PoolState",
    # Strategy and validation
    "StrategyManager",
    "ValidationManager",
    "ValidationRule",
    "ValidationResult",
    "ValidationLevel",
    # Context managers
    "timed_operation",
    "retry_operation",
    "rate_limited_operation",
    "cached_operation",
    "monitored_operation",
    "OperationContext",
    # Benchmark and optimization
    "BenchmarkRunner",
    "BenchmarkResult",
    "PerformanceProfiler",
    "PerformanceOptimizer",
    "ResourceMonitor",
    "OptimizationResult",
    # Documentation
    "DocumentationGenerator",
    "DocSection",
    # Dynamic configuration
    "DynamicConfigManager",
    "ConfigChange",
    # Advanced health
    "AdvancedHealthChecker",
    "HealthCheck",
    "HealthCheckResult",
    "HealthStatus",
    # Manager registry
    "ManagerRegistry",
    "BaseManager",
    # System integrator
    "SystemIntegrator",
    "SystemComponent",
    # Error recovery
    "RecoveryManager",
    "ErrorRecovery",
    "RecoveryConfig",
    "RecoveryResult",
    "RecoveryStrategy",
    # Async utilities
    "gather_with_exceptions",
    "timeout_after",
    "retry_async",
    "async_to_sync",
    "ensure_async",
    "AsyncLock",
    "AsyncSemaphore",
    # Testing helpers
    "TestRunner",
    "TestResult",
    "AsyncTestCase",
    "temp_directory",
    "mock_service",
    # CI/CD helpers
    "CIHelper",
    "DeploymentHelper",
    "BuildInfo",
    # Analytics
    "AnalyticsCollector",
    "AnalyticsReporter",
    "AnalyticsEvent",
    "EventType",
    # Reporting
    "ReportGenerator",
    "Report",
    "ReportType",
    # Data validator
    "DataValidator",
    "ValidationSchema",
    "ValidationRule",
    "SchemaBuilder",
    # Registry base
    "BaseRegistry",
    "FactoryRegistry",
    # Executor base
    "BaseExecutor",
    "AsyncExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    # Storage base
    "BaseStorage",
    "FileStorage",
    # Workflow
    "Workflow",
    "WorkflowManager",
    "WorkflowStep",
    "WorkflowResult",
    "WorkflowStatus",
    # Pipeline
    "Pipeline",
    "PipelineManager",
    "PipelineStep",
    "PipelineResult",
    "PipelineStage",
    # Orchestrator
    "Orchestrator",
    "OrchestrationTask",
    "OrchestrationResult",
    "OrchestrationStatus",
    # State management
    "StateManager",
    "StateChange",
    "StateEvent",
    # Advanced cache
    "AdvancedCache",
    "CacheEntry",
    "CacheStrategy",
    # Service base
    "BaseService",
    "AsyncService",
    "ServiceRegistry",
    "ServiceConfig",
    "ServiceResult",
    "ServiceStatus",
    # Handler base
    "BaseHandler",
    "AsyncHandler",
    "HandlerChain",
    "HandlerConfig",
    "HandlerResult",
    # Processor base
    "BaseProcessor",
    "AsyncProcessor",
    "ProcessingConfig",
    "ProcessingResult",
    "ProcessingStatus",
    # Coordinator
    "Coordinator",
    "CoordinationTask",
    "CoordinationResult",
    "CoordinationStatus",
    # Integration
    "IntegrationAdapter",
    "IntegrationManager",
    "IntegrationConfig",
    "IntegrationResult",
    "IntegrationStatus",
    # Data pipeline
    "DataPipeline",
    "DataPipelineManager",
    "TransformStep",
    "TransformResult",
    # Serializer
    "Serializer",
    "SerializationResult",
    "SerializationFormat",
    # Structured logging
    "StructuredLogger",
    "LogEntry",
    "LogLevel",
    # Config builder
    "ConfigBuilder",
    "ConfigSection",
    # Final utilities
    "UtilityHelper",
    "AsyncHelper",
    "FileHelper",
    # Agent component
    "AgentComponent",
    "ComponentManager",
    "ComponentConfig",
    "ComponentHealth",
    "ComponentStatus",
    # Event handler
    "EventHandler",
    "EventDispatcher",
    "Event",
    "EventPriority",
    # Factory base
    "BaseFactory",
    "BuilderFactory",
    # Middleware base
    "BaseMiddleware",
    "MiddlewarePipeline",
    "Request",
    "Response",
    # Batch operations
    "BatchOperationManager",
    "BatchOperation",
    "BatchItem",
    "BatchResult",
    "BatchStatus",
    # Scheduler
    "Scheduler",
    "ScheduledTask",
    "ScheduleType",
    # Advanced queue
    "AdvancedQueue",
    "QueueItem",
    "QueuePriority",
    "QueueStatus",
    # Result aggregator
    "ResultAggregator",
    "AggregationResult",
    # Performance tuner
    "PerformanceTuner",
    "TuningRecommendation",
    "TuningAction",
    # Resource manager
    "ResourceManager",
    "ResourceLimit",
    "ResourceUsage",
    "ResourceType",
    # Advanced service base
    "AdvancedServiceBase",
    "ServiceRegistry",
    "ServiceState",
    "ServiceMetrics",
    # Execution context
    "ExecutionContext",
    "ContextManager",
    # Advanced error handler
    "ErrorHandler",
    "ErrorHandlerDecorator",
    "ErrorInfo",
    "ErrorSeverity",
    # Constants
    "DEFAULT_MAX_PARALLEL_TASKS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_CACHE_TTL_HOURS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RATE_LIMIT_RPS",
    "DEFAULT_RATE_LIMIT_BURST",
    "OUTPUT_DIRECTORIES",
    "DEFAULT_MAX_IMAGE_SIZE_MB",
    "DEFAULT_MAX_VIDEO_SIZE_MB",
    "SUPPORTED_IMAGE_FORMATS",
    "SUPPORTED_VIDEO_FORMATS",
    "ENHANCEMENT_TYPES",
    # Code generation
    "CodeGenerator",
    "CodeTemplate",
    "CodeTemplateType",
    "GeneratedCode",
    # Seeds system
    "SeedRunner",
    "Seed",
    "SeedStatus",
    # Automatic backup
    "AutoBackupManager",
    "BackupConfig",
    "BackupResult",
    "BackupType",
    "BackupStatus",
    # Deployment utilities
    "DeploymentManager",
    "DeploymentConfig",
    "DeploymentStep",
    "DeploymentStage",
    "DeploymentStatus",
    "EnvironmentChecker",
    # Migrations
    "MigrationRunner",
    "Migration",
    "MigrationStatus",
    # API Versioning
    "APIVersionManager",
    "APIVersion",
    "VersionedEndpoint",
    "VersionStrategy",
    # Distributed Cache
    "DistributedCache",
    "CacheEntry",
    "CacheBackend",
    "CacheConsistency",
    # Advanced Logging
    "AdvancedLogger",
    "LogManager",
    "ContextLogger",
    "LogConfig",
    "LogLevel",
    "LogFormat",
    # Advanced Testing
    "AsyncTestCase",
    "TestRunner",
    "TestFixture",
    "TestResult",
    "TestConfig",
    "MockBuilder",
    "TestDecorator",
    "temp_directory",
    "temp_file",
    # Advanced Request Validation
    "AdvancedRequestValidator",
    "ValidationRule",
    "ValidationResult",
    "ValidationLevel",
    # Advanced Data Transformation
    "AdvancedDataTransformer",
    "TransformRule",
    "TransformResult",
    "TransformDirection",
    # Advanced Middleware
    "AdvancedMiddleware",
    "MiddlewareManager",
    "MiddlewareConfig",
    "MiddlewareType",
    # Advanced Rate Limiting
    "AdvancedRateLimiter",
    "RateLimitConfig",
    "RateLimitResult",
    "RateLimitStrategy",
    # Advanced Circuit Breaker
    "AdvancedCircuitBreaker",
    "CircuitBreakerManager",
    "CircuitBreakerConfig",
    "CircuitBreakerStats",
    "CircuitState",
    # Telemetry System
    "TelemetryCollector",
    "TelemetryManager",
    "TelemetryData",
    "TelemetryType",
    # Advanced Performance Profiler
    "AdvancedPerformanceProfiler",
    "PerformanceProfile",
    "ProfileResult",
    # Real-time Metrics
    "RealTimeMetricsCollector",
    "MetricSeries",
    "MetricPoint",
    "MetricType",
    # Advanced Permissions
    "AdvancedPermissionsManager",
    "Role",
    "User",
    "Permission",
    "PermissionAction",
    # Advanced Encryption
    "AdvancedEncryptionManager",
    "EncryptionKey",
    "EncryptionAlgorithm",
    # Security Validator
    "SecurityValidator",
    "SecurityIssue",
    "SecurityLevel",
    "SecurityValidationResult",
    # Advanced Audit
    "AdvancedAuditLogger",
    "AuditEntry",
    "AuditAction",
    "AuditLevel",
    # Advanced Health Monitoring
    "AdvancedHealthMonitor",
    "HealthCheck",
    "HealthCheckResult",
    "SystemHealth",
    "HealthStatus",
    # Advanced Retry
    "AdvancedRetryManager",
    "RetryConfig",
    "RetryResult",
    "RetryStrategy",
    # Advanced Queue
    "AdvancedQueue",
    "QueueItem",
    "QueuePriority",
    "QueueItemStatus",
    # Advanced Event Bus
    "AdvancedEventBus",
    "Event",
    "EventHandler",
    "EventPriority",
    # Advanced Cache Strategy
    "AdvancedCacheStrategy",
    "CacheEntry",
    "CacheEvictionPolicy",
    # Advanced Validation
    "AdvancedValidator",
    "ValidationRule",
    "ValidationResult",
    "ValidationLevel",
]
