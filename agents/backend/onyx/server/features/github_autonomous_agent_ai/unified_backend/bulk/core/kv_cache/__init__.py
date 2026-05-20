"""
Ultra-Adaptive Key-Value Cache Engine - Modular Package

This package provides a modular, extensible, and production-ready KV cache
implementation following best practices for PyTorch, Transformers, and LLMs.

Architecture:
- Foundation Layer: Types, constants, interfaces, exceptions, config
- Core Layer: Base cache implementation, storage, statistics, strategies
- Processing Layer: Quantization, compression, memory management
- Utility Layer: Device management, validation, error handling, profiling
- Adapter Layer: Adaptive and paged cache implementations
- Advanced Layer: Batch operations, monitoring, persistence, integrations
- Development Layer: Decorators, helpers, builders, testing, performance tools

Usage:
    # Basic usage
    from kv_cache import BaseKVCache, KVCacheConfig, CacheStrategy
    
    config = KVCacheConfig(max_tokens=4096, cache_strategy=CacheStrategy.ADAPTIVE)
    cache = BaseKVCache(config)
    
    # Layer-based imports (optional, backward compatible)
    from kv_cache.core import BaseKVCache
    from kv_cache.processing import Quantizer, Compressor
    from kv_cache.utilities import DeviceManager, CacheValidator
"""

# Import types and constants first
from kv_cache.types import (
    TensorPair, CacheEntry, CacheDict,
    AccessTimesDict, AccessCountsDict,
    StatsDict, MetricsDict, ConfigDict
)
from kv_cache.constants import (
    DEFAULT_MAX_TOKENS, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM,
    QUANTIZATION_BITS_SUPPORTED, EVICTION_FRACTION
)
from kv_cache.interfaces import (
    IQuantizer, ICompressor, IStorage, IMemoryManager,
    IProfiler, IMonitorable, ICache
)

from kv_cache.config import (
    CacheStrategy,
    CacheMode,
    KVCacheConfig,
)
from kv_cache.quantization import Quantizer
from kv_cache.compression import Compressor
from kv_cache.memory_manager import MemoryManager
from kv_cache.strategies import (
    BaseEvictionStrategy,
    LRUEvictionStrategy,
    LFUEvictionStrategy,
    AdaptiveEvictionStrategy,
    create_eviction_strategy,
)

# Import base classes
from kv_cache.base import BaseKVCache
from kv_cache.stats import CacheStatsTracker
from kv_cache.device_manager import DeviceManager
from kv_cache.cache_storage import CacheStorage
from kv_cache.validators import CacheValidator
from kv_cache.utils import (
    get_device_info,
    validate_tensor_shapes,
    format_memory_size,
    safe_device_transfer,
    calculate_tensor_memory_mb,
    get_tensor_info,
)

# Import adapters
from kv_cache.adapters.adaptive_cache import AdaptiveKVCache
from kv_cache.adapters.paged_cache import PagedKVCache

# Import error handling and profiling
from kv_cache.error_handler import ErrorHandler
from kv_cache.exceptions import (
    CacheError, CacheMemoryError, CacheValidationError,
    CacheDeviceError, CacheConfigError, CacheOperationError, CacheStrategyError
)
from kv_cache.profiler import CacheProfiler

# Import optimizations
from kv_cache.optimizations import (
    FastQuantizer, FastCompressor, enable_torch_optimizations
)
from kv_cache.batch_operations import BatchCacheOperations

# Import additional features
from kv_cache.transformers_integration import TransformersKVCache, ModelCacheWrapper
from kv_cache.monitoring import CacheMonitor, CacheMetrics, MetricsExporter
from kv_cache.persistence import CachePersistence, save_cache_checkpoint, load_cache_checkpoint

# Import helpers
from kv_cache.helpers import (
    create_cache_from_config,
    batch_process_cache_operations,
    estimate_cache_memory,
    validate_cache_config,
    get_cache_recommendations,
    format_cache_info,
)

# Import decorators
from kv_cache.decorators import (
    profile_cache_operation,
    retry_on_failure,
    validate_inputs,
    cache_result,
    synchronized,
)

# Import testing utilities (optional)
try:
    from kv_cache.testing import (
        create_test_cache,
        create_test_tensors,
        benchmark_cache_operation,
        validate_cache_integrity,
        test_cache_basic_operations,
        compare_cache_strategies,
    )
    _TESTING_AVAILABLE = True
except ImportError:
    _TESTING_AVAILABLE = False

# Import prelude utilities
from kv_cache.prelude import (
    setup_logging,
    enable_optimizations,
    check_environment,
    print_environment_info,
    suppress_warnings,
    get_cache_info,
)

# Import builders
from kv_cache.builders import (
    CacheConfigBuilder,
    create_default_config,
    create_inference_config,
    create_training_config,
    create_memory_efficient_config,
    create_high_performance_config,
)

# Import performance utilities
from kv_cache.performance import (
    measure_latency,
    calculate_throughput,
    estimate_cache_efficiency,
    optimize_cache_size,
    analyze_bottlenecks,
    benchmark_cache_operations,
)

# Import cache operations
from kv_cache.cache_operations import CacheOperations

# Import lifecycle management
from kv_cache.lifecycle import LifecycleManager, CacheState

# Import strategy registry
from kv_cache.strategies.registry import (
    register_strategy,
    get_registered_strategies,
    is_strategy_registered,
)

# Import async operations
from kv_cache.async_operations import AsyncCacheOperations

# Import memory pool
from kv_cache.memory_pool import TensorMemoryPool, KVCacheMemoryPool

# Import warmup strategies
from kv_cache.warmup_strategies import (
    WarmupStrategy,
    CacheWarmup
)

# Import advanced metrics
from kv_cache.metrics_advanced import AdvancedMetrics

# Import batch processor
from kv_cache.batch_processor import BatchProcessor

# Import advanced compression
from kv_cache.compression_advanced import (
    SVDCompressor,
    LowRankCompressor,
    BlockSparseCompressor
)

# Import cache prefetching
from kv_cache.cache_prefetch import (
    CachePrefetcher,
    SequentialPrefetcher
)

# Import cache analyzer
from kv_cache.cache_analyzer import CacheAnalyzer

# Import cache optimizer
from kv_cache.cache_optimizer import CacheOptimizer

# Import distributed cache
from kv_cache.distributed_cache import (
    DistributedCacheCoordinator,
    ConsistentHashingCache
)

# Import cache serializer
from kv_cache.cache_serializer import CacheSerializer

# Import cache health
from kv_cache.cache_health import CacheHealthMonitor, HealthStatus

# Import cache benchmark
from kv_cache.cache_benchmark import CacheBenchmark

# Import advanced validator
from kv_cache.cache_validator_advanced import AdvancedCacheValidator

# Import ML utilities
from kv_cache.cache_ml import (
    CacheMLPredictor,
    CacheMLOptimizer
)

# Import telemetry
from kv_cache.cache_telemetry import (
    CacheTelemetry,
    PrometheusExporter,
    StatsDExporter
)

# Import cache guard
from kv_cache.cache_guard import (
    CacheGuard,
    CircuitState,
    CacheRateLimiter
)

# Import advanced utilities
from kv_cache.cache_utils_advanced import (
    CacheInspector,
    CacheRepair,
    CacheMigrator
)

# Import advanced optimizer
from kv_cache.cache_optimizer_advanced import AdvancedCacheOptimizer

# Import documentation generator
from kv_cache.cache_documentation import CacheDocumentationGenerator

# Import security
from kv_cache.cache_security import CacheSecurity, CacheEncryption

# Import backup
from kv_cache.cache_backup import (
    CacheBackupManager,
    CacheSnapshot
)

# Import metrics export
from kv_cache.cache_metrics_export import MetricsExporter

# Import analytics
from kv_cache.cache_analytics import CacheAnalytics

# Import events
from kv_cache.cache_events import (
    CacheEventEmitter,
    CacheEvent,
    CacheEventType,
    CacheEventListener
)

# Import automation
from kv_cache.cache_automation import (
    CacheAutomation,
    CacheAutoBackup,
    CacheAutoOptimization
)

# Import clustering
from kv_cache.cache_clustering import (
    CacheClustering,
    CacheGrouping
)

# Import scheduler
from kv_cache.cache_scheduler import (
    CacheScheduler,
    CacheMaintenanceScheduler
)

# Import compliance
from kv_cache.cache_compliance import (
    CacheCompliance,
    ComplianceLevel,
    CacheAuditor
)

# Import plugin system
from kv_cache.cache_plugin import (
    PluginManager,
    CachePlugin,
    LoggingPlugin,
    MetricsPlugin
)

# Import experiments
from kv_cache.cache_experiments import (
    CacheExperiment,
    ExperimentType,
    CacheABTesting
)

# Import ML optimization
from kv_cache.cache_optimization_ml import (
    MLBasedOptimizer,
    OptimizationTarget,
    ReinforcementLearningOptimizer
)

# Import adapters
from kv_cache.cache_adapters import (
    CacheAdapter,
    DictAdapter,
    ContextManagerAdapter,
    AsyncAdapter,
    BatchAdapter,
    TransformerAdapter
)

# Import synchronization
from kv_cache.cache_sync import (
    CacheSynchronizer,
    SyncStrategy,
    DistributedLock,
    CacheBarrier,
    CacheMutex
)

# Import pool
from kv_cache.cache_pool import (
    CachePool,
    CachePoolManager,
    PoolConfig
)

# Import advanced decorators
from kv_cache.cache_decorators_advanced import (
    cache_result,
    retry_on_failure,
    rate_limit,
    timeout,
    circuit_breaker
)

# Import versioning
from kv_cache.cache_versioning import (
    CacheVersioning,
    VersionStrategy,
    VersionedEntry,
    CacheReplication
)

# Import sharding
from kv_cache.cache_sharding import (
    CacheSharding,
    ShardConfig,
    ConsistentHashingSharding
)

# Import tuning
from kv_cache.cache_tuning import (
    CacheTuner,
    TuningRecommendation,
    CacheProfiler
)

# Import streaming
from kv_cache.cache_streaming import (
    CacheStreamer,
    CachePipeline,
    CacheBatchProcessor
)

# Import advanced compression
from kv_cache.cache_compression_advanced import (
    DeltaCompression,
    DictionaryCompression,
    PredictiveCompression
)

# Import final optimization
from kv_cache.cache_optimization_final import (
    CacheOptimizerFinal,
    OptimizationLevel,
    OptimizationResult,
    CacheWarmupAdvanced
)

# Import invalidation
from kv_cache.cache_invalidation import (
    CacheInvalidator,
    InvalidationStrategy,
    InvalidationRule,
    CacheInvalidationManager
)

# Import routing
from kv_cache.cache_routing import (
    CacheRouter,
    CacheNode,
    RoutingStrategy,
    CacheLoadBalancer
)

# Import federation
from kv_cache.cache_federation import (
    CacheFederation,
    CacheCluster
)

# Import observability
from kv_cache.cache_observability import (
    CacheObservability,
    MetricType,
    Metric,
    CacheMetricsExporter
)

# Import testing framework
from kv_cache.cache_testing_framework import (
    CacheTestSuite,
    TestCase,
    TestResult,
    CacheTestHelpers
)

# Import disaster recovery
from kv_cache.cache_disaster_recovery import (
    CacheDisasterRecovery,
    RecoveryStrategy,
    RecoveryPlan
)

# Import encryption
from kv_cache.cache_encryption import (
    CacheEncryption,
    EncryptionAlgorithm,
    CacheSecurity
)

# Import cost optimization
from kv_cache.cache_cost_optimization import (
    CacheCostOptimizer,
    CostMetric,
    CostProfile
)

# Import multi-tenancy
from kv_cache.cache_multitenancy import (
    CacheMultiTenancy,
    Tenant,
    TenantCacheRouter
)

# Import migration
from kv_cache.cache_migration import (
    CacheMigrator,
    MigrationStrategy,
    MigrationPlan
)

# Import advanced analytics
from kv_cache.cache_analytics_advanced import (
    CacheAnalyticsAdvanced,
    AnalyticsInsight
)

# Import API
from kv_cache.cache_api import CacheAPI

# Import CLI
from kv_cache.cache_cli import CacheCLI

# Import anomaly detection
from kv_cache.cache_anomaly_detection import (
    CacheAnomalyDetector,
    AnomalyType,
    Anomaly
)

# Import auto-scaling
from kv_cache.cache_autoscaling import (
    CacheAutoScaler,
    ScalingAction,
    ScalingDecision
)

# Import GraphQL
from kv_cache.cache_graphql import CacheGraphQL

# Import WebSocket
from kv_cache.cache_websocket import CacheWebSocket

# Import distributed tracing
from kv_cache.cache_distributed_tracing import (
    CacheDistributedTracing,
    Span,
    SpanKind
)

# Import middleware
from kv_cache.cache_middleware import (
    CacheMiddleware,
    MiddlewarePipeline,
    LoggingMiddleware,
    MetricsMiddleware,
    ValidationMiddleware,
    TransformMiddleware
)

# Import versioning
from kv_cache.cache_versioning import (
    CacheVersionManager,
    VersionStrategy,
    CacheVersion,
    VersionedCache
)

# Import transformation
from kv_cache.cache_transformation import (
    CacheTransformer,
    TransformationPipeline,
    NormalizationTransformer,
    QuantizationTransformer,
    CompressionTransformer,
    CustomTransformer
)

# Import advanced profiling
from kv_cache.cache_profiling import (
    CacheProfiler as AdvancedCacheProfiler,
    ProfilingLevel,
    PerformanceMetric,
    ProfilingResult,
    FunctionProfile,
    MemorySnapshot,
    CachePerformanceAnalyzer
)

# Import query engine
from kv_cache.cache_query_engine import (
    CacheQueryEngine,
    CacheQuery,
    QueryOperator,
    LogicalOperator,
    QueryCondition,
    SortSpec,
    QueryResult,
    AggregationResult
)

# Import adaptive intelligence
from kv_cache.cache_adaptive_intelligence import (
    CacheAdaptiveIntelligence,
    AdaptationStrategy,
    AdaptationAction,
    AdaptationDecision,
    PerformancePattern,
    AdaptationHistory
)

# Import event sourcing
from kv_cache.cache_event_sourcing import (
    CacheEventSourcing,
    CacheEventStore,
    CacheEventProjector,
    CacheEvent,
    EventType,
    EventStream
)

# Import schema management
from kv_cache.cache_schema_manager import (
    CacheSchemaManager,
    SchemaAwareCache,
    SchemaDefinition,
    SchemaEvolution,
    SchemaVersion,
    ValidationResult
)

# Import partitioning
from kv_cache.cache_partitioning import (
    CachePartitioning,
    PartitionedCache,
    Partition,
    PartitionConfig,
    PartitionStrategy
)

# Import consistency models
from kv_cache.cache_consistency import (
    CacheConsistencyManager,
    ConsistentCache,
    ConsistencyLevel,
    ConsistencyProtocol,
    VersionVector,
    ConsistencyConfig,
    CacheEntry as ConsistencyCacheEntry
)

# Import indexing
from kv_cache.cache_indexing import (
    CacheIndexManager,
    IndexedCache,
    IndexDefinition,
    IndexType,
    HashIndex,
    SortedIndex,
    FullTextIndex
)

# Import conflict resolution
from kv_cache.cache_conflict_resolution import (
    CacheConflictManager,
    ConflictAwareCache,
    ConflictResolver,
    ConflictResolutionStrategy,
    ConflictEntry,
    ConflictResolution
)

# Import data deduplication
from kv_cache.cache_data_deduplication import (
    CacheDeduplication,
    DeduplicatedCache,
    ContentDeduplicator,
    SimilarityDeduplicator,
    DeduplicationStrategy,
    DeduplicationEntry,
    DeduplicationStats
)

# Import tiering
from kv_cache.cache_tiering import (
    MultiTierCache,
    CacheTier,
    TierLevel,
    TierPolicy,
    TierConfig,
    TierStats
)

# Import advanced prefetching
from kv_cache.cache_prefetching_advanced import (
    AdvancedCachePrefetcher,
    PrefetchStrategy,
    PrefetchPrediction,
    AccessPattern
)

# Import lifecycle management
from kv_cache.cache_lifecycle import (
    CacheLifecycleManager,
    LifecycleAwareCache,
    LifecycleStage,
    ExpirationPolicy,
    LifecycleEntry,
    LifecyclePolicy
)

# Import cache patterns
from kv_cache.cache_patterns import (
    CacheAside,
    WriteThrough,
    WriteBack,
    RefreshAhead,
    PatternCacheFactory,
    CachePattern,
    PatternConfig
)

# Import rate limiting
from kv_cache.cache_rate_limiting import (
    CacheRateLimiter,
    RateLimiter,
    RateLimitStrategy,
    RateLimitConfig,
    RateLimitResult,
    RateLimitExceededError
)

# Import advanced compression v2
from kv_cache.cache_compression_advanced_v2 import (
    AdvancedCompressor,
    CompressedCache,
    CompressionAlgorithm,
    CompressionResult,
    CompressionStats
)

# Import analytics dashboard
from kv_cache.cache_analytics_dashboard import (
    CacheAnalyticsDashboard,
    CacheMonitoringDashboard,
    DashboardReport,
    Metric,
    MetricType
)

# Import performance tuning
from kv_cache.cache_performance_tuning import (
    CachePerformanceTuner,
    TuningTarget,
    TuningParameter,
    TuningRecommendation,
    TuningResult
)

# Import advanced security
from kv_cache.cache_security_advanced import (
    CacheSecurityManager,
    SecureCache,
    SecurityLevel,
    SecurityPolicy,
    AccessPermission,
    AccessControlEntry,
    AuditLogEntry
)

# Import snapshot and backup
from kv_cache.cache_snapshot_backup import (
    CacheSnapshotManager,
    BackupManager,
    Snapshot,
    SnapshotFormat,
    BackupConfig
)

# Import health check
from kv_cache.cache_health_check import (
    CacheHealthChecker,
    HealthMonitor,
    HealthStatus,
    HealthCheck,
    HealthReport
)

# Import advanced clustering
from kv_cache.cache_clustering_advanced import (
    CacheCluster,
    ClusterNode,
    ClusterConfig,
    ClusterRole,
    ClusterState
)

# Import distributed locking
from kv_cache.cache_distributed_lock import (
    DistributedLockManager,
    LockContext,
    Lock,
    LockType
)

# Import transactions
from kv_cache.cache_transaction import (
    CacheTransactionManager,
    TransactionalCache,
    Transaction,
    TransactionOperation,
    TransactionStatus
)

# Import warmup strategies
from kv_cache.cache_warmup_strategies import (
    CacheWarmer,
    IntelligentWarmer,
    WarmupStrategy,
    WarmupConfig,
    WarmupResult
)

# Import advanced metrics
from kv_cache.cache_metrics_advanced import (
    AdvancedMetricsCollector,
    CacheMetricsAdvanced,
    MetricAggregation,
    MetricDataPoint,
    AggregatedMetric
)

# Import ML optimization
from kv_cache.cache_optimization_ml import (
    MLOptimizer,
    OptimizationTarget,
    FeatureVector,
    Prediction
)

# Import advanced batch operations
from kv_cache.cache_batch_operations_advanced import (
    AdvancedBatchProcessor,
    BatchStrategy,
    BatchOperation,
    BatchResult
)

# Import notification system
from kv_cache.cache_notification_system import (
    CacheNotificationSystem,
    NotifiedCache,
    NotificationSubscriber,
    Notification,
    NotificationType
)

# Import advanced validation
from kv_cache.cache_validation_advanced import (
    AdvancedCacheValidator,
    ValidatedCache,
    ValidationLevel,
    ValidationRule,
    ValidationResult
)

# Import load balancing
from kv_cache.cache_load_balancing import (
    CacheLoadBalancer,
    CacheNode,
    LoadBalanceStrategy,
    LoadBalanceStats
)

# Import real-time monitoring
from kv_cache.cache_monitoring_realtime import (
    RealTimeMonitor,
    CacheMonitorAdvanced,
    Alert,
    AlertLevel,
    MetricSnapshot
)

# Import advanced serialization
from kv_cache.cache_serialization_advanced import (
    AdvancedSerializer,
    SerializedCache,
    SerializationFormat,
    SerializationResult
)

# Import cache coherence
from kv_cache.cache_cache_coherence import (
    CacheCoherenceManager,
    CacheLine,
    CacheState,
    CoherenceProtocol
)

# Import async operations
from kv_cache.cache_async_operations import (
    AsyncCacheOperations,
    AsyncCache,
    AsyncOperation,
    AsyncOperationType
)

# Import advanced expiration
from kv_cache.cache_expiration_advanced import (
    AdvancedExpirationManager,
    ExpiringCache,
    ExpirationPolicy,
    ExpirationEntry
)

# Try to import engine if available (may not exist yet)
try:
    from kv_cache.engine import UltraAdaptiveKVCacheEngine
except ImportError:
    UltraAdaptiveKVCacheEngine = None

__version__ = "6.6.0"  # Production Ready - Cache Coherence, Async Operations, Advanced Expiration
__all__ = [
    # Types and Constants
    "TensorPair",
    "CacheEntry",
    "CacheDict",
    "AccessTimesDict",
    "AccessCountsDict",
    "StatsDict",
    "MetricsDict",
    "ConfigDict",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_NUM_HEADS",
    "DEFAULT_HEAD_DIM",
    "QUANTIZATION_BITS_SUPPORTED",
    "EVICTION_FRACTION",
    "IQuantizer",
    "ICompressor",
    "IStorage",
    "IMemoryManager",
    "IProfiler",
    "IMonitorable",
    "ICache",
    # Config
    "CacheStrategy",
    "CacheMode",
    "KVCacheConfig",
    # Components
    "Quantizer",
    "Compressor",
    "MemoryManager",
    # Strategies
    "BaseEvictionStrategy",
    "LRUEvictionStrategy",
    "LFUEvictionStrategy",
    "AdaptiveEvictionStrategy",
    "create_eviction_strategy",
    # Statistics
    "CacheStatsTracker",
    # Storage and Device
    "CacheStorage",
    "DeviceManager",
    # Validators
    "CacheValidator",
    # Utilities
    "get_device_info",
    "validate_tensor_shapes",
    "format_memory_size",
    "safe_device_transfer",
    "calculate_tensor_memory_mb",
    "get_tensor_info",
    # Main classes
    "BaseKVCache",
    "AdaptiveKVCache",
    "PagedKVCache",
    "UltraAdaptiveKVCacheEngine",
    # Error handling
    "ErrorHandler",
    "CacheError",
    "CacheMemoryError",
    "CacheValidationError",
    "CacheDeviceError",
    "CacheConfigError",
    "CacheOperationError",
    "CacheStrategyError",
    # Helpers
    "create_cache_from_config",
    "batch_process_cache_operations",
    "estimate_cache_memory",
    "validate_cache_config",
    "get_cache_recommendations",
    "format_cache_info",
    # Decorators
    "profile_cache_operation",
    "retry_on_failure",
    "validate_inputs",
    "cache_result",
    "synchronized",
    # Testing utilities (if available)
    "create_test_cache",
    "create_test_tensors",
    "benchmark_cache_operation",
    "validate_cache_integrity",
    "test_cache_basic_operations",
    "compare_cache_strategies",
    # Prelude utilities
    "setup_logging",
    "enable_optimizations",
    "check_environment",
    "print_environment_info",
    "suppress_warnings",
    "get_cache_info",
    # Builders
    "CacheConfigBuilder",
    "create_default_config",
    "create_inference_config",
    "create_training_config",
    "create_memory_efficient_config",
    "create_high_performance_config",
    # Performance utilities
    "measure_latency",
    "calculate_throughput",
    "estimate_cache_efficiency",
    "optimize_cache_size",
    "analyze_bottlenecks",
    "benchmark_cache_operations",
    # Cache operations
    "CacheOperations",
    # Lifecycle management
    "LifecycleManager",
    "CacheState",
    # Strategy registry
    "register_strategy",
    "get_registered_strategies",
    "is_strategy_registered",
    # Async operations
    "AsyncCacheOperations",
    # Memory pool
    "TensorMemoryPool",
    "KVCacheMemoryPool",
    # Warmup strategies
    "WarmupStrategy",
    "CacheWarmup",
    # Advanced metrics
    "AdvancedMetrics",
    # Batch processor
    "BatchProcessor",
    # Advanced compression
    "SVDCompressor",
    "LowRankCompressor",
    "BlockSparseCompressor",
    # Cache prefetching
    "CachePrefetcher",
    "SequentialPrefetcher",
    # Cache analyzer
    "CacheAnalyzer",
    # Cache optimizer
    "CacheOptimizer",
    # Distributed cache
    "DistributedCacheCoordinator",
    "ConsistentHashingCache",
    # Cache serializer
    "CacheSerializer",
    # Cache health
    "CacheHealthMonitor",
    "HealthStatus",
    # Cache benchmark
    "CacheBenchmark",
    # Advanced validator
    "AdvancedCacheValidator",
    # ML utilities
    "CacheMLPredictor",
    "CacheMLOptimizer",
    # Telemetry
    "CacheTelemetry",
    "PrometheusExporter",
    "StatsDExporter",
    # Cache guard
    "CacheGuard",
    "CircuitState",
    "CacheRateLimiter",
    # Advanced utilities
    "CacheInspector",
    "CacheRepair",
    "CacheMigrator",
    # Advanced optimizer
    "AdvancedCacheOptimizer",
    # Documentation generator
    "CacheDocumentationGenerator",
    # Security
    "CacheSecurity",
    "CacheEncryption",
    # Backup
    "CacheBackupManager",
    "CacheSnapshot",
    # Metrics export
    "MetricsExporter",
    # Analytics
    "CacheAnalytics",
    # Events
    "CacheEventEmitter",
    "CacheEvent",
    "CacheEventType",
    "CacheEventListener",
    # Automation
    "CacheAutomation",
    "CacheAutoBackup",
    "CacheAutoOptimization",
    # Clustering
    "CacheClustering",
    "CacheGrouping",
    # Scheduler
    "CacheScheduler",
    "CacheMaintenanceScheduler",
    # Compliance
    "CacheCompliance",
    "ComplianceLevel",
    "CacheAuditor",
    # Plugin system
    "PluginManager",
    "CachePlugin",
    "LoggingPlugin",
    "MetricsPlugin",
    # Experiments
    "CacheExperiment",
    "ExperimentType",
    "CacheABTesting",
    # ML Optimization
    "MLBasedOptimizer",
    "OptimizationTarget",
    "ReinforcementLearningOptimizer",
    # Adapters
    "CacheAdapter",
    "DictAdapter",
    "ContextManagerAdapter",
    "AsyncAdapter",
    "BatchAdapter",
    "TransformerAdapter",
    # Synchronization
    "CacheSynchronizer",
    "SyncStrategy",
    "DistributedLock",
    "CacheBarrier",
    "CacheMutex",
    # Pool
    "CachePool",
    "CachePoolManager",
    "PoolConfig",
    # Advanced Decorators
    "cache_result",
    "retry_on_failure",
    "rate_limit",
    "timeout",
    "circuit_breaker",
    # Versioning
    "CacheVersioning",
    "VersionStrategy",
    "VersionedEntry",
    "CacheReplication",
    # Sharding
    "CacheSharding",
    "ShardConfig",
    "ConsistentHashingSharding",
    # Tuning
    "CacheTuner",
    "TuningRecommendation",
    # Profiling
    "CacheProfiler",
    # Streaming
    "CacheStreamer",
    "CachePipeline",
    "CacheBatchProcessor",
    # Advanced Compression
    "DeltaCompression",
    "DictionaryCompression",
    "PredictiveCompression",
    # Final Optimization
    "CacheOptimizerFinal",
    "OptimizationLevel",
    "OptimizationResult",
    "CacheWarmupAdvanced",
    # Invalidation
    "CacheInvalidator",
    "InvalidationStrategy",
    "InvalidationRule",
    "CacheInvalidationManager",
    # Routing
    "CacheRouter",
    "CacheNode",
    "RoutingStrategy",
    "CacheLoadBalancer",
    # Federation
    "CacheFederation",
    "CacheCluster",
    # Observability
    "CacheObservability",
    "MetricType",
    "Metric",
    "CacheMetricsExporter",
    # Testing
    "CacheTestSuite",
    "TestCase",
    "TestResult",
    "CacheTestHelpers",
    # Disaster Recovery
    "CacheDisasterRecovery",
    "RecoveryStrategy",
    "RecoveryPlan",
    # Encryption
    "CacheEncryption",
    "EncryptionAlgorithm",
    "CacheSecurity",
    # Cost Optimization
    "CacheCostOptimizer",
    "CostMetric",
    "CostProfile",
    # Multi-tenancy
    "CacheMultiTenancy",
    "Tenant",
    "TenantCacheRouter",
    # Migration
    "CacheMigrator",
    "MigrationStrategy",
    "MigrationPlan",
    # Advanced Analytics
    "CacheAnalyticsAdvanced",
    "AnalyticsInsight",
    # API
    "CacheAPI",
    # CLI
    "CacheCLI",
    # Anomaly Detection
    "CacheAnomalyDetector",
    "AnomalyType",
    "Anomaly",
    # Auto-scaling
    "CacheAutoScaler",
    "ScalingAction",
    "ScalingDecision",
    # GraphQL
    "CacheGraphQL",
    # WebSocket
    "CacheWebSocket",
    # Distributed Tracing
    "CacheDistributedTracing",
    "Span",
    "SpanKind",
    # Optimizations
    "FastQuantizer",
    "FastCompressor",
    "BatchCacheOperations",
    "enable_torch_optimizations",
    # Transformers Integration
    "TransformersKVCache",
    "ModelCacheWrapper",
    # Monitoring
    "CacheMonitor",
    "CacheMetrics",
    "MetricsExporter",
    # Persistence
    "CachePersistence",
    "save_cache_checkpoint",
    "load_cache_checkpoint",
    # Advanced Profiling
    "AdvancedCacheProfiler",
    "ProfilingLevel",
    "PerformanceMetric",
    "ProfilingResult",
    "FunctionProfile",
    "MemorySnapshot",
    "CachePerformanceAnalyzer",
    # Query Engine
    "CacheQueryEngine",
    "CacheQuery",
    "QueryOperator",
    "LogicalOperator",
    "QueryCondition",
    "SortSpec",
    "QueryResult",
    "AggregationResult",
    # Adaptive Intelligence
    "CacheAdaptiveIntelligence",
    "AdaptationStrategy",
    "AdaptationAction",
    "AdaptationDecision",
    "PerformancePattern",
    "AdaptationHistory",
    # Event Sourcing
    "CacheEventSourcing",
    "CacheEventStore",
    "CacheEventProjector",
    "CacheEvent",
    "EventType",
    "EventStream",
    # Schema Management
    "CacheSchemaManager",
    "SchemaAwareCache",
    "SchemaDefinition",
    "SchemaEvolution",
    "SchemaVersion",
    "ValidationResult",
    # Partitioning
    "CachePartitioning",
    "PartitionedCache",
    "Partition",
    "PartitionConfig",
    "PartitionStrategy",
    # Consistency Models
    "CacheConsistencyManager",
    "ConsistentCache",
    "ConsistencyLevel",
    "ConsistencyProtocol",
    "VersionVector",
    "ConsistencyConfig",
    # Indexing
    "CacheIndexManager",
    "IndexedCache",
    "IndexDefinition",
    "IndexType",
    "HashIndex",
    "SortedIndex",
    "FullTextIndex",
    # Conflict Resolution
    "CacheConflictManager",
    "ConflictAwareCache",
    "ConflictResolver",
    "ConflictResolutionStrategy",
    "ConflictEntry",
    "ConflictResolution",
    # Data Deduplication
    "CacheDeduplication",
    "DeduplicatedCache",
    "ContentDeduplicator",
    "SimilarityDeduplicator",
    "DeduplicationStrategy",
    "DeduplicationEntry",
    "DeduplicationStats",
    # Tiering
    "MultiTierCache",
    "CacheTier",
    "TierLevel",
    "TierPolicy",
    "TierConfig",
    "TierStats",
    # Advanced Prefetching
    "AdvancedCachePrefetcher",
    "PrefetchStrategy",
    "PrefetchPrediction",
    "AccessPattern",
    # Lifecycle Management
    "CacheLifecycleManager",
    "LifecycleAwareCache",
    "LifecycleStage",
    "ExpirationPolicy",
    "LifecycleEntry",
    "LifecyclePolicy",
    # Cache Patterns
    "CacheAside",
    "WriteThrough",
    "WriteBack",
    "RefreshAhead",
    "PatternCacheFactory",
    "CachePattern",
    "PatternConfig",
    # Rate Limiting
    "CacheRateLimiter",
    "RateLimiter",
    "RateLimitStrategy",
    "RateLimitConfig",
    "RateLimitResult",
    "RateLimitExceededError",
    # Advanced Compression V2
    "AdvancedCompressor",
    "CompressedCache",
    "CompressionAlgorithm",
    "CompressionResult",
    "CompressionStats",
    # Analytics Dashboard
    "CacheAnalyticsDashboard",
    "CacheMonitoringDashboard",
    "DashboardReport",
    "Metric",
    "MetricType",
    # Performance Tuning
    "CachePerformanceTuner",
    "TuningTarget",
    "TuningParameter",
    "TuningRecommendation",
    "TuningResult",
    # Advanced Security
    "CacheSecurityManager",
    "SecureCache",
    "SecurityLevel",
    "SecurityPolicy",
    "AccessPermission",
    "AccessControlEntry",
    "AuditLogEntry",
    # Snapshot and Backup
    "CacheSnapshotManager",
    "BackupManager",
    "Snapshot",
    "SnapshotFormat",
    "BackupConfig",
    # Health Check
    "CacheHealthChecker",
    "HealthMonitor",
    "HealthStatus",
    "HealthCheck",
    "HealthReport",
    # Advanced Clustering
    "CacheCluster",
    "ClusterNode",
    "ClusterConfig",
    "ClusterRole",
    "ClusterState",
    # Distributed Locking
    "DistributedLockManager",
    "LockContext",
    "Lock",
    "LockType",
    # Transactions
    "CacheTransactionManager",
    "TransactionalCache",
    "Transaction",
    "TransactionOperation",
    "TransactionStatus",
    # Warmup Strategies
    "CacheWarmer",
    "IntelligentWarmer",
    "WarmupStrategy",
    "WarmupConfig",
    "WarmupResult",
    # Advanced Metrics
    "AdvancedMetricsCollector",
    "CacheMetricsAdvanced",
    "MetricAggregation",
    "MetricDataPoint",
    "AggregatedMetric",
    # ML Optimization
    "MLOptimizer",
    "OptimizationTarget",
    "FeatureVector",
    "Prediction",
    # Advanced Batch Operations
    "AdvancedBatchProcessor",
    "BatchStrategy",
    "BatchOperation",
    "BatchResult",
    # Notification System
    "CacheNotificationSystem",
    "NotifiedCache",
    "NotificationSubscriber",
    "Notification",
    "NotificationType",
    # Advanced Validation
    "AdvancedCacheValidator",
    "ValidatedCache",
    "ValidationLevel",
    "ValidationRule",
    "ValidationResult",
    # Load Balancing
    "CacheLoadBalancer",
    "CacheNode",
    "LoadBalanceStrategy",
    "LoadBalanceStats",
    # Real-time Monitoring
    "RealTimeMonitor",
    "CacheMonitorAdvanced",
    "Alert",
    "AlertLevel",
    "MetricSnapshot",
    # Advanced Serialization
    "AdvancedSerializer",
    "SerializedCache",
    "SerializationFormat",
    "SerializationResult",
    # Cache Coherence
    "CacheCoherenceManager",
    "CacheLine",
    "CacheState",
    "CoherenceProtocol",
    # Async Operations
    "AsyncCacheOperations",
    "AsyncCache",
    "AsyncOperation",
    "AsyncOperationType",
    # Advanced Expiration
    "AdvancedExpirationManager",
    "ExpiringCache",
    "ExpirationPolicy",
    "ExpirationEntry",
]



This package provides a modular, extensible, and production-ready KV cache
implementation following best practices for PyTorch, Transformers, and LLMs.

Architecture:
- Foundation Layer: Types, constants, interfaces, exceptions, config
- Core Layer: Base cache implementation, storage, statistics, strategies
- Processing Layer: Quantization, compression, memory management
- Utility Layer: Device management, validation, error handling, profiling
- Adapter Layer: Adaptive and paged cache implementations
- Advanced Layer: Batch operations, monitoring, persistence, integrations
- Development Layer: Decorators, helpers, builders, testing, performance tools

Usage:
    # Basic usage
    from kv_cache import BaseKVCache, KVCacheConfig, CacheStrategy
    
    config = KVCacheConfig(max_tokens=4096, cache_strategy=CacheStrategy.ADAPTIVE)
    cache = BaseKVCache(config)
    
    # Layer-based imports (optional, backward compatible)
    from kv_cache.core import BaseKVCache
    from kv_cache.processing import Quantizer, Compressor
    from kv_cache.utilities import DeviceManager, CacheValidator
"""

# Import types and constants first
from kv_cache.types import (
    TensorPair, CacheEntry, CacheDict,
    AccessTimesDict, AccessCountsDict,
    StatsDict, MetricsDict, ConfigDict
)
from kv_cache.constants import (
    DEFAULT_MAX_TOKENS, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM,
    QUANTIZATION_BITS_SUPPORTED, EVICTION_FRACTION
)
from kv_cache.interfaces import (
    IQuantizer, ICompressor, IStorage, IMemoryManager,
    IProfiler, IMonitorable, ICache
)

from kv_cache.config import (
    CacheStrategy,
    CacheMode,
    KVCacheConfig,
)
from kv_cache.quantization import Quantizer
from kv_cache.compression import Compressor
from kv_cache.memory_manager import MemoryManager
from kv_cache.strategies import (
    BaseEvictionStrategy,
    LRUEvictionStrategy,
    LFUEvictionStrategy,
    AdaptiveEvictionStrategy,
    create_eviction_strategy,
)

# Import base classes
from kv_cache.base import BaseKVCache
from kv_cache.stats import CacheStatsTracker
from kv_cache.device_manager import DeviceManager
from kv_cache.cache_storage import CacheStorage
from kv_cache.validators import CacheValidator
from kv_cache.utils import (
    get_device_info,
    validate_tensor_shapes,
    format_memory_size,
    safe_device_transfer,
    calculate_tensor_memory_mb,
    get_tensor_info,
)

# Import adapters
from kv_cache.adapters.adaptive_cache import AdaptiveKVCache
from kv_cache.adapters.paged_cache import PagedKVCache

# Import error handling and profiling
from kv_cache.error_handler import ErrorHandler
from kv_cache.exceptions import (
    CacheError, CacheMemoryError, CacheValidationError,
    CacheDeviceError, CacheConfigError, CacheOperationError, CacheStrategyError
)
from kv_cache.profiler import CacheProfiler

# Import optimizations
from kv_cache.optimizations import (
    FastQuantizer, FastCompressor, enable_torch_optimizations
)
from kv_cache.batch_operations import BatchCacheOperations

# Import additional features
from kv_cache.transformers_integration import TransformersKVCache, ModelCacheWrapper
from kv_cache.monitoring import CacheMonitor, CacheMetrics, MetricsExporter
from kv_cache.persistence import CachePersistence, save_cache_checkpoint, load_cache_checkpoint

# Import helpers
from kv_cache.helpers import (
    create_cache_from_config,
    batch_process_cache_operations,
    estimate_cache_memory,
    validate_cache_config,
    get_cache_recommendations,
    format_cache_info,
)

# Import decorators
from kv_cache.decorators import (
    profile_cache_operation,
    retry_on_failure,
    validate_inputs,
    cache_result,
    synchronized,
)

# Import testing utilities (optional)
try:
    from kv_cache.testing import (
        create_test_cache,
        create_test_tensors,
        benchmark_cache_operation,
        validate_cache_integrity,
        test_cache_basic_operations,
        compare_cache_strategies,
    )
    _TESTING_AVAILABLE = True
except ImportError:
    _TESTING_AVAILABLE = False

# Import prelude utilities
from kv_cache.prelude import (
    setup_logging,
    enable_optimizations,
    check_environment,
    print_environment_info,
    suppress_warnings,
    get_cache_info,
)

# Import builders
from kv_cache.builders import (
    CacheConfigBuilder,
    create_default_config,
    create_inference_config,
    create_training_config,
    create_memory_efficient_config,
    create_high_performance_config,
)

# Import performance utilities
from kv_cache.performance import (
    measure_latency,
    calculate_throughput,
    estimate_cache_efficiency,
    optimize_cache_size,
    analyze_bottlenecks,
    benchmark_cache_operations,
)

# Import cache operations
from kv_cache.cache_operations import CacheOperations

# Import lifecycle management
from kv_cache.lifecycle import LifecycleManager, CacheState

# Import strategy registry
from kv_cache.strategies.registry import (
    register_strategy,
    get_registered_strategies,
    is_strategy_registered,
)

# Import async operations
from kv_cache.async_operations import AsyncCacheOperations

# Import memory pool
from kv_cache.memory_pool import TensorMemoryPool, KVCacheMemoryPool

# Import warmup strategies
from kv_cache.warmup_strategies import (
    WarmupStrategy,
    CacheWarmup
)

# Import advanced metrics
from kv_cache.metrics_advanced import AdvancedMetrics

# Import batch processor
from kv_cache.batch_processor import BatchProcessor

# Import advanced compression
from kv_cache.compression_advanced import (
    SVDCompressor,
    LowRankCompressor,
    BlockSparseCompressor
)

# Import cache prefetching
from kv_cache.cache_prefetch import (
    CachePrefetcher,
    SequentialPrefetcher
)

# Import cache analyzer
from kv_cache.cache_analyzer import CacheAnalyzer

# Import cache optimizer
from kv_cache.cache_optimizer import CacheOptimizer

# Import distributed cache
from kv_cache.distributed_cache import (
    DistributedCacheCoordinator,
    ConsistentHashingCache
)

# Import cache serializer
from kv_cache.cache_serializer import CacheSerializer

# Import cache health
from kv_cache.cache_health import CacheHealthMonitor, HealthStatus

# Import cache benchmark
from kv_cache.cache_benchmark import CacheBenchmark

# Import advanced validator
from kv_cache.cache_validator_advanced import AdvancedCacheValidator

# Import ML utilities
from kv_cache.cache_ml import (
    CacheMLPredictor,
    CacheMLOptimizer
)

# Import telemetry
from kv_cache.cache_telemetry import (
    CacheTelemetry,
    PrometheusExporter,
    StatsDExporter
)

# Import cache guard
from kv_cache.cache_guard import (
    CacheGuard,
    CircuitState,
    CacheRateLimiter
)

# Import advanced utilities
from kv_cache.cache_utils_advanced import (
    CacheInspector,
    CacheRepair,
    CacheMigrator
)

# Import advanced optimizer
from kv_cache.cache_optimizer_advanced import AdvancedCacheOptimizer

# Import documentation generator
from kv_cache.cache_documentation import CacheDocumentationGenerator

# Import security
from kv_cache.cache_security import CacheSecurity, CacheEncryption

# Import backup
from kv_cache.cache_backup import (
    CacheBackupManager,
    CacheSnapshot
)

# Import metrics export
from kv_cache.cache_metrics_export import MetricsExporter

# Import analytics
from kv_cache.cache_analytics import CacheAnalytics

# Import events
from kv_cache.cache_events import (
    CacheEventEmitter,
    CacheEvent,
    CacheEventType,
    CacheEventListener
)

# Import automation
from kv_cache.cache_automation import (
    CacheAutomation,
    CacheAutoBackup,
    CacheAutoOptimization
)

# Import clustering
from kv_cache.cache_clustering import (
    CacheClustering,
    CacheGrouping
)

# Import scheduler
from kv_cache.cache_scheduler import (
    CacheScheduler,
    CacheMaintenanceScheduler
)

# Import compliance
from kv_cache.cache_compliance import (
    CacheCompliance,
    ComplianceLevel,
    CacheAuditor
)

# Import plugin system
from kv_cache.cache_plugin import (
    PluginManager,
    CachePlugin,
    LoggingPlugin,
    MetricsPlugin
)

# Import experiments
from kv_cache.cache_experiments import (
    CacheExperiment,
    ExperimentType,
    CacheABTesting
)

# Import ML optimization
from kv_cache.cache_optimization_ml import (
    MLBasedOptimizer,
    OptimizationTarget,
    ReinforcementLearningOptimizer
)

# Import adapters
from kv_cache.cache_adapters import (
    CacheAdapter,
    DictAdapter,
    ContextManagerAdapter,
    AsyncAdapter,
    BatchAdapter,
    TransformerAdapter
)

# Import synchronization
from kv_cache.cache_sync import (
    CacheSynchronizer,
    SyncStrategy,
    DistributedLock,
    CacheBarrier,
    CacheMutex
)

# Import pool
from kv_cache.cache_pool import (
    CachePool,
    CachePoolManager,
    PoolConfig
)

# Import advanced decorators
from kv_cache.cache_decorators_advanced import (
    cache_result,
    retry_on_failure,
    rate_limit,
    timeout,
    circuit_breaker
)

# Import middleware
from kv_cache.cache_middleware import (
    CacheMiddleware,
    MiddlewarePipeline,
    LoggingMiddleware,
    MetricsMiddleware,
    ValidationMiddleware,
    TransformMiddleware
)

# Import versioning
from kv_cache.cache_versioning import (
    CacheVersionManager,
    VersionStrategy,
    CacheVersion,
    VersionedCache
)

# Import transformation
from kv_cache.cache_transformation import (
    CacheTransformer,
    TransformationPipeline,
    NormalizationTransformer,
    QuantizationTransformer,
    CompressionTransformer,
    CustomTransformer
)

# Import advanced profiling
from kv_cache.cache_profiling import (
    CacheProfiler as AdvancedCacheProfiler,
    ProfilingLevel,
    PerformanceMetric,
    ProfilingResult,
    FunctionProfile,
    MemorySnapshot,
    CachePerformanceAnalyzer
)

# Import query engine
from kv_cache.cache_query_engine import (
    CacheQueryEngine,
    CacheQuery,
    QueryOperator,
    LogicalOperator,
    QueryCondition,
    SortSpec,
    QueryResult,
    AggregationResult
)

# Import adaptive intelligence
from kv_cache.cache_adaptive_intelligence import (
    CacheAdaptiveIntelligence,
    AdaptationStrategy,
    AdaptationAction,
    AdaptationDecision,
    PerformancePattern,
    AdaptationHistory
)

# Import event sourcing
from kv_cache.cache_event_sourcing import (
    CacheEventSourcing,
    CacheEventStore,
    CacheEventProjector,
    CacheEvent,
    EventType,
    EventStream
)

# Import schema management
from kv_cache.cache_schema_manager import (
    CacheSchemaManager,
    SchemaAwareCache,
    SchemaDefinition,
    SchemaEvolution,
    SchemaVersion,
    ValidationResult
)

# Import partitioning
from kv_cache.cache_partitioning import (
    CachePartitioning,
    PartitionedCache,
    Partition,
    PartitionConfig,
    PartitionStrategy
)

# Import consistency models
from kv_cache.cache_consistency import (
    CacheConsistencyManager,
    ConsistentCache,
    ConsistencyLevel,
    ConsistencyProtocol,
    VersionVector,
    ConsistencyConfig,
    CacheEntry as ConsistencyCacheEntry
)

# Import indexing
from kv_cache.cache_indexing import (
    CacheIndexManager,
    IndexedCache,
    IndexDefinition,
    IndexType,
    HashIndex,
    SortedIndex,
    FullTextIndex
)

# Import conflict resolution
from kv_cache.cache_conflict_resolution import (
    CacheConflictManager,
    ConflictAwareCache,
    ConflictResolver,
    ConflictResolutionStrategy,
    ConflictEntry,
    ConflictResolution
)

# Import data deduplication
from kv_cache.cache_data_deduplication import (
    CacheDeduplication,
    DeduplicatedCache,
    ContentDeduplicator,
    SimilarityDeduplicator,
    DeduplicationStrategy,
    DeduplicationEntry,
    DeduplicationStats
)

# Import tiering
from kv_cache.cache_tiering import (
    MultiTierCache,
    CacheTier,
    TierLevel,
    TierPolicy,
    TierConfig,
    TierStats
)

# Import advanced prefetching
from kv_cache.cache_prefetching_advanced import (
    AdvancedCachePrefetcher,
    PrefetchStrategy,
    PrefetchPrediction,
    AccessPattern
)

# Import lifecycle management
from kv_cache.cache_lifecycle import (
    CacheLifecycleManager,
    LifecycleAwareCache,
    LifecycleStage,
    ExpirationPolicy,
    LifecycleEntry,
    LifecyclePolicy
)

# Import cache patterns
from kv_cache.cache_patterns import (
    CacheAside,
    WriteThrough,
    WriteBack,
    RefreshAhead,
    PatternCacheFactory,
    CachePattern,
    PatternConfig
)

# Import rate limiting
from kv_cache.cache_rate_limiting import (
    CacheRateLimiter,
    RateLimiter,
    RateLimitStrategy,
    RateLimitConfig,
    RateLimitResult,
    RateLimitExceededError
)

# Import advanced compression v2
from kv_cache.cache_compression_advanced_v2 import (
    AdvancedCompressor,
    CompressedCache,
    CompressionAlgorithm,
    CompressionResult,
    CompressionStats
)

# Import analytics dashboard
from kv_cache.cache_analytics_dashboard import (
    CacheAnalyticsDashboard,
    CacheMonitoringDashboard,
    DashboardReport,
    Metric,
    MetricType
)

# Import performance tuning
from kv_cache.cache_performance_tuning import (
    CachePerformanceTuner,
    TuningTarget,
    TuningParameter,
    TuningRecommendation,
    TuningResult
)

# Import advanced security
from kv_cache.cache_security_advanced import (
    CacheSecurityManager,
    SecureCache,
    SecurityLevel,
    SecurityPolicy,
    AccessPermission,
    AccessControlEntry,
    AuditLogEntry
)

# Import snapshot and backup
from kv_cache.cache_snapshot_backup import (
    CacheSnapshotManager,
    BackupManager,
    Snapshot,
    SnapshotFormat,
    BackupConfig
)

# Import health check
from kv_cache.cache_health_check import (
    CacheHealthChecker,
    HealthMonitor,
    HealthStatus,
    HealthCheck,
    HealthReport
)

# Import advanced clustering
from kv_cache.cache_clustering_advanced import (
    CacheCluster,
    ClusterNode,
    ClusterConfig,
    ClusterRole,
    ClusterState
)

# Import distributed locking
from kv_cache.cache_distributed_lock import (
    DistributedLockManager,
    LockContext,
    Lock,
    LockType
)

# Import transactions
from kv_cache.cache_transaction import (
    CacheTransactionManager,
    TransactionalCache,
    Transaction,
    TransactionOperation,
    TransactionStatus
)

# Import warmup strategies
from kv_cache.cache_warmup_strategies import (
    CacheWarmer,
    IntelligentWarmer,
    WarmupStrategy,
    WarmupConfig,
    WarmupResult
)

# Import advanced metrics
from kv_cache.cache_metrics_advanced import (
    AdvancedMetricsCollector,
    CacheMetricsAdvanced,
    MetricAggregation,
    MetricDataPoint,
    AggregatedMetric
)

# Import ML optimization
from kv_cache.cache_optimization_ml import (
    MLOptimizer,
    OptimizationTarget,
    FeatureVector,
    Prediction
)

# Import advanced batch operations
from kv_cache.cache_batch_operations_advanced import (
    AdvancedBatchProcessor,
    BatchStrategy,
    BatchOperation,
    BatchResult
)

# Import notification system
from kv_cache.cache_notification_system import (
    CacheNotificationSystem,
    NotifiedCache,
    NotificationSubscriber,
    Notification,
    NotificationType
)

# Import advanced validation
from kv_cache.cache_validation_advanced import (
    AdvancedCacheValidator,
    ValidatedCache,
    ValidationLevel,
    ValidationRule,
    ValidationResult
)

# Import load balancing
from kv_cache.cache_load_balancing import (
    CacheLoadBalancer,
    CacheNode,
    LoadBalanceStrategy,
    LoadBalanceStats
)

# Import real-time monitoring
from kv_cache.cache_monitoring_realtime import (
    RealTimeMonitor,
    CacheMonitorAdvanced,
    Alert,
    AlertLevel,
    MetricSnapshot
)

# Import advanced serialization
from kv_cache.cache_serialization_advanced import (
    AdvancedSerializer,
    SerializedCache,
    SerializationFormat,
    SerializationResult
)

# Import cache coherence
from kv_cache.cache_cache_coherence import (
    CacheCoherenceManager,
    CacheLine,
    CacheState,
    CoherenceProtocol
)

# Import async operations
from kv_cache.cache_async_operations import (
    AsyncCacheOperations,
    AsyncCache,
    AsyncOperation,
    AsyncOperationType
)

# Import advanced expiration
from kv_cache.cache_expiration_advanced import (
    AdvancedExpirationManager,
    ExpiringCache,
    ExpirationPolicy,
    ExpirationEntry
)

# Try to import engine if available (may not exist yet)
try:
    from kv_cache.engine import UltraAdaptiveKVCacheEngine
except ImportError:
    UltraAdaptiveKVCacheEngine = None

__version__ = "6.6.0"  # Production Ready - Cache Coherence, Async Operations, Advanced Expiration
__all__ = [
    # Types and Constants
    "TensorPair",
    "CacheEntry",
    "CacheDict",
    "AccessTimesDict",
    "AccessCountsDict",
    "StatsDict",
    "MetricsDict",
    "ConfigDict",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_NUM_HEADS",
    "DEFAULT_HEAD_DIM",
    "QUANTIZATION_BITS_SUPPORTED",
    "EVICTION_FRACTION",
    "IQuantizer",
    "ICompressor",
    "IStorage",
    "IMemoryManager",
    "IProfiler",
    "IMonitorable",
    "ICache",
    # Config
    "CacheStrategy",
    "CacheMode",
    "KVCacheConfig",
    # Components
    "Quantizer",
    "Compressor",
    "MemoryManager",
    # Strategies
    "BaseEvictionStrategy",
    "LRUEvictionStrategy",
    "LFUEvictionStrategy",
    "AdaptiveEvictionStrategy",
    "create_eviction_strategy",
    # Statistics
    "CacheStatsTracker",
    # Storage and Device
    "CacheStorage",
    "DeviceManager",
    # Validators
    "CacheValidator",
    # Utilities
    "get_device_info",
    "validate_tensor_shapes",
    "format_memory_size",
    "safe_device_transfer",
    "calculate_tensor_memory_mb",
    "get_tensor_info",
    # Main classes
    "BaseKVCache",
    "AdaptiveKVCache",
    "PagedKVCache",
    "UltraAdaptiveKVCacheEngine",
    # Error handling
    "ErrorHandler",
    "CacheError",
    "CacheMemoryError",
    "CacheValidationError",
    "CacheDeviceError",
    "CacheConfigError",
    "CacheOperationError",
    "CacheStrategyError",
    # Helpers
    "create_cache_from_config",
    "batch_process_cache_operations",
    "estimate_cache_memory",
    "validate_cache_config",
    "get_cache_recommendations",
    "format_cache_info",
    # Decorators
    "profile_cache_operation",
    "retry_on_failure",
    "validate_inputs",
    "cache_result",
    "synchronized",
    # Testing utilities (if available)
    "create_test_cache",
    "create_test_tensors",
    "benchmark_cache_operation",
    "validate_cache_integrity",
    "test_cache_basic_operations",
    "compare_cache_strategies",
    # Prelude utilities
    "setup_logging",
    "enable_optimizations",
    "check_environment",
    "print_environment_info",
    "suppress_warnings",
    "get_cache_info",
    # Builders
    "CacheConfigBuilder",
    "create_default_config",
    "create_inference_config",
    "create_training_config",
    "create_memory_efficient_config",
    "create_high_performance_config",
    # Performance utilities
    "measure_latency",
    "calculate_throughput",
    "estimate_cache_efficiency",
    "optimize_cache_size",
    "analyze_bottlenecks",
    "benchmark_cache_operations",
    # Cache operations
    "CacheOperations",
    # Lifecycle management
    "LifecycleManager",
    "CacheState",
    # Strategy registry
    "register_strategy",
    "get_registered_strategies",
    "is_strategy_registered",
    # Async operations
    "AsyncCacheOperations",
    # Memory pool
    "TensorMemoryPool",
    "KVCacheMemoryPool",
    # Warmup strategies
    "WarmupStrategy",
    "CacheWarmup",
    # Advanced metrics
    "AdvancedMetrics",
    # Batch processor
    "BatchProcessor",
    # Advanced compression
    "SVDCompressor",
    "LowRankCompressor",
    "BlockSparseCompressor",
    # Cache prefetching
    "CachePrefetcher",
    "SequentialPrefetcher",
    # Cache analyzer
    "CacheAnalyzer",
    # Cache optimizer
    "CacheOptimizer",
    # Distributed cache
    "DistributedCacheCoordinator",
    "ConsistentHashingCache",
    # Cache serializer
    "CacheSerializer",
    # Cache health
    "CacheHealthMonitor",
    "HealthStatus",
    # Cache benchmark
    "CacheBenchmark",
    # Advanced validator
    "AdvancedCacheValidator",
    # ML utilities
    "CacheMLPredictor",
    "CacheMLOptimizer",
    # Telemetry
    "CacheTelemetry",
    "PrometheusExporter",
    "StatsDExporter",
    # Cache guard
    "CacheGuard",
    "CircuitState",
    "CacheRateLimiter",
    # Advanced utilities
    "CacheInspector",
    "CacheRepair",
    "CacheMigrator",
    # Advanced optimizer
    "AdvancedCacheOptimizer",
    # Documentation generator
    "CacheDocumentationGenerator",
    # Security
    "CacheSecurity",
    "CacheEncryption",
    # Backup
    "CacheBackupManager",
    "CacheSnapshot",
    # Metrics export
    "MetricsExporter",
    # Analytics
    "CacheAnalytics",
    # Events
    "CacheEventEmitter",
    "CacheEvent",
    "CacheEventType",
    "CacheEventListener",
    # Automation
    "CacheAutomation",
    "CacheAutoBackup",
    "CacheAutoOptimization",
    # Clustering
    "CacheClustering",
    "CacheGrouping",
    # Scheduler
    "CacheScheduler",
    "CacheMaintenanceScheduler",
    # Compliance
    "CacheCompliance",
    "ComplianceLevel",
    "CacheAuditor",
    # Plugin system
    "PluginManager",
    "CachePlugin",
    "LoggingPlugin",
    "MetricsPlugin",
    # Experiments
    "CacheExperiment",
    "ExperimentType",
    "CacheABTesting",
    # ML Optimization
    "MLBasedOptimizer",
    "OptimizationTarget",
    "ReinforcementLearningOptimizer",
    # Adapters
    "CacheAdapter",
    "DictAdapter",
    "ContextManagerAdapter",
    "AsyncAdapter",
    "BatchAdapter",
    "TransformerAdapter",
    # Synchronization
    "CacheSynchronizer",
    "SyncStrategy",
    "DistributedLock",
    "CacheBarrier",
    "CacheMutex",
    # Pool
    "CachePool",
    "CachePoolManager",
    "PoolConfig",
    # Advanced Decorators
    "cache_result",
    "retry_on_failure",
    "rate_limit",
    "timeout",
    "circuit_breaker",
    # Middleware
    "CacheMiddleware",
    "MiddlewarePipeline",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "ValidationMiddleware",
    "TransformMiddleware",
    # Versioning
    "CacheVersionManager",
    "VersionStrategy",
    "CacheVersion",
    "VersionedCache",
    # Transformation
    "CacheTransformer",
    "TransformationPipeline",
    "NormalizationTransformer",
    "QuantizationTransformer",
    "CompressionTransformer",
    "CustomTransformer",
    # Profiling
    "CacheProfiler",
    # Optimizations
    "FastQuantizer",
    "FastCompressor",
    "BatchCacheOperations",
    "enable_torch_optimizations",
    # Transformers Integration
    "TransformersKVCache",
    "ModelCacheWrapper",
    # Monitoring
    "CacheMonitor",
    "CacheMetrics",
    "MetricsExporter",
    # Persistence
    "CachePersistence",
    "save_cache_checkpoint",
    "load_cache_checkpoint",
    # Advanced Profiling
    "AdvancedCacheProfiler",
    "ProfilingLevel",
    "PerformanceMetric",
    "ProfilingResult",
    "FunctionProfile",
    "MemorySnapshot",
    "CachePerformanceAnalyzer",
    # Query Engine
    "CacheQueryEngine",
    "CacheQuery",
    "QueryOperator",
    "LogicalOperator",
    "QueryCondition",
    "SortSpec",
    "QueryResult",
    "AggregationResult",
    # Adaptive Intelligence
    "CacheAdaptiveIntelligence",
    "AdaptationStrategy",
    "AdaptationAction",
    "AdaptationDecision",
    "PerformancePattern",
    "AdaptationHistory",
    # Event Sourcing
    "CacheEventSourcing",
    "CacheEventStore",
    "CacheEventProjector",
    "CacheEvent",
    "EventType",
    "EventStream",
    # Schema Management
    "CacheSchemaManager",
    "SchemaAwareCache",
    "SchemaDefinition",
    "SchemaEvolution",
    "SchemaVersion",
    "ValidationResult",
    # Partitioning
    "CachePartitioning",
    "PartitionedCache",
    "Partition",
    "PartitionConfig",
    "PartitionStrategy",
    # Consistency Models
    "CacheConsistencyManager",
    "ConsistentCache",
    "ConsistencyLevel",
    "ConsistencyProtocol",
    "VersionVector",
    "ConsistencyConfig",
    # Indexing
    "CacheIndexManager",
    "IndexedCache",
    "IndexDefinition",
    "IndexType",
    "HashIndex",
    "SortedIndex",
    "FullTextIndex",
    # Conflict Resolution
    "CacheConflictManager",
    "ConflictAwareCache",
    "ConflictResolver",
    "ConflictResolutionStrategy",
    "ConflictEntry",
    "ConflictResolution",
    # Data Deduplication
    "CacheDeduplication",
    "DeduplicatedCache",
    "ContentDeduplicator",
    "SimilarityDeduplicator",
    "DeduplicationStrategy",
    "DeduplicationEntry",
    "DeduplicationStats",
    # Tiering
    "MultiTierCache",
    "CacheTier",
    "TierLevel",
    "TierPolicy",
    "TierConfig",
    "TierStats",
    # Advanced Prefetching
    "AdvancedCachePrefetcher",
    "PrefetchStrategy",
    "PrefetchPrediction",
    "AccessPattern",
    # Lifecycle Management
    "CacheLifecycleManager",
    "LifecycleAwareCache",
    "LifecycleStage",
    "ExpirationPolicy",
    "LifecycleEntry",
    "LifecyclePolicy",
    # Cache Patterns
    "CacheAside",
    "WriteThrough",
    "WriteBack",
    "RefreshAhead",
    "PatternCacheFactory",
    "CachePattern",
    "PatternConfig",
    # Rate Limiting
    "CacheRateLimiter",
    "RateLimiter",
    "RateLimitStrategy",
    "RateLimitConfig",
    "RateLimitResult",
    "RateLimitExceededError",
    # Advanced Compression V2
    "AdvancedCompressor",
    "CompressedCache",
    "CompressionAlgorithm",
    "CompressionResult",
    "CompressionStats",
    # Analytics Dashboard
    "CacheAnalyticsDashboard",
    "CacheMonitoringDashboard",
    "DashboardReport",
    "Metric",
    "MetricType",
    # Performance Tuning
    "CachePerformanceTuner",
    "TuningTarget",
    "TuningParameter",
    "TuningRecommendation",
    "TuningResult",
    # Advanced Security
    "CacheSecurityManager",
    "SecureCache",
    "SecurityLevel",
    "SecurityPolicy",
    "AccessPermission",
    "AccessControlEntry",
    "AuditLogEntry",
    # Snapshot and Backup
    "CacheSnapshotManager",
    "BackupManager",
    "Snapshot",
    "SnapshotFormat",
    "BackupConfig",
    # Health Check
    "CacheHealthChecker",
    "HealthMonitor",
    "HealthStatus",
    "HealthCheck",
    "HealthReport",
    # Advanced Clustering
    "CacheCluster",
    "ClusterNode",
    "ClusterConfig",
    "ClusterRole",
    "ClusterState",
    # Distributed Locking
    "DistributedLockManager",
    "LockContext",
    "Lock",
    "LockType",
    # Transactions
    "CacheTransactionManager",
    "TransactionalCache",
    "Transaction",
    "TransactionOperation",
    "TransactionStatus",
    # Warmup Strategies
    "CacheWarmer",
    "IntelligentWarmer",
    "WarmupStrategy",
    "WarmupConfig",
    "WarmupResult",
    # Advanced Metrics
    "AdvancedMetricsCollector",
    "CacheMetricsAdvanced",
    "MetricAggregation",
    "MetricDataPoint",
    "AggregatedMetric",
    # ML Optimization
    "MLOptimizer",
    "OptimizationTarget",
    "FeatureVector",
    "Prediction",
    # Advanced Batch Operations
    "AdvancedBatchProcessor",
    "BatchStrategy",
    "BatchOperation",
    "BatchResult",
    # Notification System
    "CacheNotificationSystem",
    "NotifiedCache",
    "NotificationSubscriber",
    "Notification",
    "NotificationType",
    # Advanced Validation
    "AdvancedCacheValidator",
    "ValidatedCache",
    "ValidationLevel",
    "ValidationRule",
    "ValidationResult",
    # Load Balancing
    "CacheLoadBalancer",
    "CacheNode",
    "LoadBalanceStrategy",
    "LoadBalanceStats",
    # Real-time Monitoring
    "RealTimeMonitor",
    "CacheMonitorAdvanced",
    "Alert",
    "AlertLevel",
    "MetricSnapshot",
    # Advanced Serialization
    "AdvancedSerializer",
    "SerializedCache",
    "SerializationFormat",
    "SerializationResult",
    # Cache Coherence
    "CacheCoherenceManager",
    "CacheLine",
    "CacheState",
    "CoherenceProtocol",
    # Async Operations
    "AsyncCacheOperations",
    "AsyncCache",
    "AsyncOperation",
    "AsyncOperationType",
    # Advanced Expiration
    "AdvancedExpirationManager",
    "ExpiringCache",
    "ExpirationPolicy",
    "ExpirationEntry",
]

