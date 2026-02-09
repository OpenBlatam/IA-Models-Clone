"""
Core Module - Robot Movement AI
================================

Módulo principal con todos los componentes core.
"""

# Robot Core Components
from .robot import (
    RobotMovementEngine,
    TrajectoryOptimizer,
    TrajectoryPoint,
    InverseKinematicsSolver,
    EndEffectorPose,
    JointState,
    RealTimeFeedbackSystem,
    FeedbackData
)

# Algorithms
from .algorithms import (
    BaseOptimizationAlgorithm,
    PPOAlgorithm,
    DQNAlgorithm,
    AStarAlgorithm,
    RRTAlgorithm,
    HeuristicAlgorithm
)

# Constants
from .constants.constants import (
    OptimizationAlgorithm,
    ErrorMessages,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_LEARNING_RATE,
    # ... más constantes
)

# Exceptions
from .exceptions import (
    RobotMovementError,
    TrajectoryError,
    TrajectoryEmptyError,
    TrajectoryCollisionError,
    TrajectoryInvalidError,
    IKError,
    RobotConnectionError,
    RobotNotConnectedError,
    RobotMovementInProgressError,
    AlgorithmNotFoundError,
    ConfigurationError,
    ValidationError
)

# Validators
from .validators import (
    validate_position,
    validate_orientation,
    validate_trajectory_point,
    validate_trajectory,
    validate_obstacle,
    validate_obstacles
)

# Decorators
from .decorators import (
    log_execution_time,
    log_execution_time_async,
    handle_robot_errors,
    handle_robot_errors_async,
    validate_inputs,
    retry_on_failure,
    retry_on_failure_async,
    cache_result
)

# Metrics
from .metrics import (
    MetricsCollector,
    get_metrics_collector,
    record_value,
    increment_counter,
    record_timing
)

# Performance
from .performance import (
    measure_time,
    timeit,
    timeit_async,
    PerformanceProfiler,
    CacheManager
)

# Helpers
from .helpers.helpers import (
    clamp,
    lerp,
    normalize_angle,
    euclidean_distance,
    format_duration,
    format_distance
)

# Utils
from .utils import (
    quaternion_slerp,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    calculate_trajectory_distance,
    calculate_trajectory_curvature,
    smooth_trajectory,
    validate_trajectory_continuity
)

# Types
from .types.types import (
    Position3D,
    Orientation,
    OptimizationResult,
    ValidationResult
)

# Serialization
from .serialization.serialization import (
    serialize_trajectory,
    deserialize_trajectory,
    serialize_config,
    deserialize_config,
    to_json,
    from_json
)

# Extensions
from .extensions import (
    Extension,
    ExtensionManager,
    get_extension_manager
)

# Compatibility
from .compatibility.compatibility import (
    get_system_info,
    check_python_version,
    check_dependencies,
    get_feature_flags
)

# Factories
try:
    from .factories.factories import (
        TrajectoryOptimizerFactory,
        MovementEngineFactory,
        ComponentBuilder
    )
except ImportError:
    pass

# Cache
try:
    from .cache import (
        IntelligentCache,
        CacheEntry,
        CacheStats
    )
    # Aliases for compatibility
    LRUCache = IntelligentCache
    TTLCache = IntelligentCache
    CacheManager = IntelligentCache
except ImportError:
    LRUCache = None
    TTLCache = None
    CacheManager = None

# Try to import additional cache utilities
try:
    from .cache import get_cache_manager, cached
except ImportError:
    get_cache_manager = None
    cached = None

# Async Utils
try:
    from ..utils.async_utils import (
        AsyncBatchProcessor,
        run_in_background,
        timeout_after,
        retry_async,
        AsyncQueue,
        gather_with_limit,
        AsyncLock
    )
except ImportError:
    AsyncBatchProcessor = None
    run_in_background = None
    timeout_after = None
    retry_async = None
    AsyncQueue = None
    gather_with_limit = None
    AsyncLock = None

# Logging
try:
    from .config.logging_config import (
        setup_logging,
        get_logger,
        LoggerAdapter,
        log_function_call,
        log_function_call_async
    )
except ImportError:
    # Fallback if logging_config not available
    setup_logging = None
    get_logger = None
    LoggerAdapter = None
    log_function_call = None
    log_function_call_async = None

# Initialization
from .initialization import (
    SystemInitializer,
    get_system_initializer,
    initialize_system,
    InitStage
)

# Quality
from .quality import (
    QualityChecker,
    QualityMetric,
    check_performance_quality,
    check_system_health_quality
)

# Resource Manager
try:
    from .resource_manager import (
        ResourceManager,
        get_resource_manager
    )
except ImportError:
    ResourceManager = None
    get_resource_manager = None

# Testing Utils
try:
    from .testing_utils import (
    create_mock_trajectory_point,
    create_mock_trajectory,
    assert_trajectory_valid,
    AsyncTestCase
    )
except ImportError:
    create_mock_trajectory_point = None
    create_mock_trajectory = None
    assert_trajectory_valid = None
    AsyncTestCase = None

# Debug Utils
try:
    from .debug_utils import (
    debug_print,
    trace_function,
    trace_function_async,
    DebugContext,
    profile_function
    )
except ImportError:
    debug_print = None
    trace_function = None
    trace_function_async = None
    DebugContext = None
    profile_function = None

# Versioning
try:
    from .versioning.versioning import (
    Version,
    VersionInfo,
    VersionManager,
    get_version_manager,
    get_current_version
    )
except ImportError:
    Version = None
    VersionInfo = None
    VersionManager = None
    get_version_manager = None
    get_current_version = None

# Backup
try:
    from .backup.backup import (
    BackupManager,
    get_backup_manager
    )
except ImportError:
    BackupManager = None
    get_backup_manager = None

# Dynamic Config
try:
    from .dynamic_config import (
    DynamicConfigManager,
    ConfigChange,
    get_dynamic_config_manager
    )
except ImportError:
    DynamicConfigManager = None
    ConfigChange = None
    get_dynamic_config_manager = None

# Analytics
try:
    from .analytics.analytics import (
    AnalyticsEngine,
    PerformanceReport,
    get_analytics_engine
    )
except ImportError:
    AnalyticsEngine = None
    PerformanceReport = None
    get_analytics_engine = None

# Auto Optimizer
try:
    from .auto_optimizer import (
        AutoOptimizer,
        OptimizationRule,
        get_auto_optimizer
    )
except ImportError:
    AutoOptimizer = None
    OptimizationRule = None
    get_auto_optimizer = None

# Continuous Learning
try:
    from .continuous_learning import (
    ContinuousLearningSystem,
    LearningPattern,
    get_continuous_learning
    )
except ImportError:
    ContinuousLearningSystem = None
    LearningPattern = None
    get_continuous_learning = None

# Benchmarking
try:
    from .benchmarking import (
    BenchmarkRunner,
    BenchmarkResult,
    get_benchmark_runner
    )
except ImportError:
    BenchmarkRunner = None
    BenchmarkResult = None
    get_benchmark_runner = None

# Task Scheduler
try:
    from .scheduler import (
    TaskScheduler,
    ScheduledTask,
    TaskStatus,
    get_task_scheduler
    )
except ImportError:
    TaskScheduler = None
    ScheduledTask = None
    TaskStatus = None
    get_task_scheduler = None

# Workflow
try:
    from .workflow import (
    Workflow,
    WorkflowStep,
    WorkflowManager,
    get_workflow_manager
    )
except ImportError:
    Workflow = None
    WorkflowStep = None
    WorkflowManager = None
    get_workflow_manager = None

# Rate Limiter
try:
    from .rate_limiter import (
    RateLimiter,
    RateLimit,
    get_rate_limiter
    )
except ImportError:
    RateLimiter = None
    RateLimit = None
    get_rate_limiter = None

# Validation Engine
try:
    from .validation_engine import (
    ValidationEngine,
    ValidationRule,
    ValidationResult,
    ValidationLevel,
    get_validation_engine
    )
except ImportError:
    ValidationEngine = None
    ValidationRule = None
    ValidationResult = None
    ValidationLevel = None
    get_validation_engine = None

# Notification System
try:
    from .notification_system import (
    NotificationSystem,
    Notification,
    NotificationType,
    NotificationChannel,
    get_notification_system
    )
except ImportError:
    NotificationSystem = None
    Notification = None
    NotificationType = None
    NotificationChannel = None
    get_notification_system = None

# API Client
try:
    from .api_client import (
    APIClient,
    APIRequest,
    APIResponse,
    get_api_client
    )
except ImportError:
    APIClient = None
    APIRequest = None
    APIResponse = None
    get_api_client = None

# Data Pipeline
try:
    from .data_pipeline import (
    DataPipeline,
    PipelineStep,
    PipelineStage,
    PipelineManager,
    get_pipeline_manager
    )
except ImportError:
    DataPipeline = None
    PipelineStep = None
    PipelineStage = None
    PipelineManager = None
    get_pipeline_manager = None

# State Machine
try:
    from .state_machine import (
        StateMachine,
        State,
        Transition,
        StateMachineManager,
        get_state_machine_manager
    )
except ImportError:
    StateMachine = None
    State = None
    Transition = None
    StateMachineManager = None
    get_state_machine_manager = None

# Documentation Generator
try:
    from .documentation_generator import (
    DocumentationGenerator,
    get_documentation_generator
    )
except ImportError:
    DocumentationGenerator = None
    get_documentation_generator = None

# Code Quality
try:
    from .code_quality import (
    CodeQualityAnalyzer,
    CodeQualityReport,
    CodeMetric,
    get_code_quality_analyzer
    )
except ImportError:
    CodeQualityAnalyzer = None
    CodeQualityReport = None
    CodeMetric = None
    get_code_quality_analyzer = None

# Performance Monitor - already imported above, skip duplicate
# from .performance import (
#     PerformanceMonitor,
#     PerformanceSnapshot,
#     get_performance_monitor
# )

# Error Tracker
try:
    from .error_tracker import (
        ErrorTracker,
        ErrorRecord,
        get_error_tracker
    )
except ImportError:
    ErrorTracker = None
    ErrorRecord = None
    get_error_tracker = None

# Summary Generator
try:
    from .summary_generator import (
        ProjectSummaryGenerator,
        get_summary_generator
    )
except ImportError:
    ProjectSummaryGenerator = None
    get_summary_generator = None

# Export/Import Utilities
try:
    from .export_utils import (
    ExportManager,
    get_export_manager
    )
except ImportError:
    ExportManager = None
    get_export_manager = None

try:
    from .import_utils import (
    ImportManager,
    get_import_manager
    )
except ImportError:
    ImportManager = None
    get_import_manager = None

# Test Runner
try:
    from .test_runner import (
    TestRunner,
    TestResult,
    TestSuiteResult,
    get_test_runner
    )
except ImportError:
    TestRunner = None
    TestResult = None
    TestSuiteResult = None
    get_test_runner = None

# Deployment Utilities
try:
    from .deployment_utils import (
    DeploymentManager,
    DeploymentConfig,
    get_deployment_manager
    )
except ImportError:
    DeploymentManager = None
    DeploymentConfig = None
    get_deployment_manager = None

# CLI Tools
try:
    from .cli_tools import (
    CLITool,
    ProjectCLI,
    main as cli_main
    )
except ImportError:
    CLITool = None
    ProjectCLI = None
    cli_main = None

# Maintenance Utilities
try:
    from .maintenance_utils import (
    MaintenanceManager,
    get_maintenance_manager
    )
except ImportError:
    MaintenanceManager = None
    get_maintenance_manager = None

# Security Audit
try:
    from .security_audit import (
    SecurityAuditor,
    SecurityIssue,
    get_security_auditor
    )
except ImportError:
    SecurityAuditor = None
    SecurityIssue = None
    get_security_auditor = None

# Compliance Checker
try:
    from .compliance_checker import (
    ComplianceChecker,
    ComplianceRule,
    ComplianceResult,
    ComplianceStandard,
    get_compliance_checker
    )
except ImportError:
    ComplianceChecker = None
    ComplianceRule = None
    ComplianceResult = None
    ComplianceStandard = None
    get_compliance_checker = None

# Optimization Profiler
try:
    from .optimization_profiler import (
        OptimizationProfiler,
        ProfileResult,
        OptimizationProfile,
        get_optimization_profiler
    )
except ImportError:
    OptimizationProfiler = None
    ProfileResult = None
    OptimizationProfile = None
    get_optimization_profiler = None

# Data Validator
try:
    from .data_validator import (
    DataValidator,
    ValidationRule,
    ValidationResult,
    ValidationType,
    get_data_validator
    )
except ImportError:
    DataValidator = None
    ValidationRule = None
    ValidationResult = None
    ValidationType = None
    get_data_validator = None

# Data Transformer
try:
    from .data_transformer import (
    DataTransformer,
    TransformationRule,
    get_data_transformer
    )
except ImportError:
    DataTransformer = None
    TransformationRule = None
    get_data_transformer = None

# Report Generator
try:
    from .report_generator import (
    ReportGenerator,
    Report,
    ReportSection,
    get_report_generator
    )
except ImportError:
    ReportGenerator = None
    Report = None
    ReportSection = None
    get_report_generator = None

# Dashboard Builder
try:
    from .dashboard_builder import (
    DashboardBuilder,
    Dashboard,
    DashboardWidget,
    WidgetType,
    get_dashboard_builder
    )
except ImportError:
    DashboardBuilder = None
    Dashboard = None
    DashboardWidget = None
    WidgetType = None
    get_dashboard_builder = None

# Feature Flags
try:
    from .feature_flags import (
    FeatureFlagManager,
    FeatureFlag,
    FeatureStatus,
    get_feature_flag_manager
    )
except ImportError:
    FeatureFlagManager = None
    FeatureFlag = None
    FeatureStatus = None
    get_feature_flag_manager = None

# Experiment Manager
try:
    from .experiment_manager import (
    ExperimentManager,
    Experiment,
    ExperimentVariant,
    ExperimentResult,
    get_experiment_manager
    )
except ImportError:
    ExperimentManager = None
    Experiment = None
    ExperimentVariant = None
    ExperimentResult = None
    get_experiment_manager = None

# Knowledge Base
try:
    from .knowledge_base import (
    KnowledgeBase,
    KnowledgeEntry,
    get_knowledge_base
    )
except ImportError:
    KnowledgeBase = None
    KnowledgeEntry = None
    get_knowledge_base = None

# Recommendation Engine
try:
    from .analytics.recommendation_engine import (
    RecommendationEngine,
    Recommendation,
    get_recommendation_engine
    )
except ImportError:
    RecommendationEngine = None
    Recommendation = None
    get_recommendation_engine = None

# Collaboration System
try:
    from .collaboration_system import (
    CollaborationSystem,
    User,
    Task,
    Comment,
    TaskStatus,
    UserRole,
    get_collaboration_system
    )
except ImportError:
    CollaborationSystem = None
    User = None
    Task = None
    Comment = None
    TaskStatus = None
    UserRole = None
    get_collaboration_system = None

# Audit Log
try:
    from .audit_log import (
    AuditLogger,
    AuditEntry,
    AuditAction,
    AuditLevel,
    get_audit_logger
    )
except ImportError:
    AuditLogger = None
    AuditEntry = None
    AuditAction = None
    AuditLevel = None
    get_audit_logger = None

# Version Control
try:
    from .versioning.version_control import (
    VersionControl,
    Version,
    get_version_control
    )
except ImportError:
    VersionControl = None
    Version = None
    get_version_control = None

# Snapshot Manager
try:
    from .analytics.snapshot_manager import (
    SnapshotManager,
    Snapshot,
    get_snapshot_manager
    )
except ImportError:
    SnapshotManager = None
    Snapshot = None
    get_snapshot_manager = None

# Webhook System
try:
    from .webhook_system import (
    WebhookSystem,
    Webhook,
    WebhookDelivery,
    WebhookStatus,
    get_webhook_system
    )
except ImportError:
    WebhookSystem = None
    Webhook = None
    WebhookDelivery = None
    WebhookStatus = None
    get_webhook_system = None

# API Gateway
try:
    from .api_gateway import (
        APIGateway,
        APIEndpoint,
        APIRequest,
        APIStatus,
        get_api_gateway
    )
except ImportError:
    APIGateway = None
    APIEndpoint = None
    APIRequest = None
    APIStatus = None
    get_api_gateway = None

# GraphQL API
try:
    from .graphql_api import (
    GraphQLAPI,
    GraphQLQuery,
    GraphQLResolver,
    get_graphql_api
    )
except ImportError:
    GraphQLAPI = None
    GraphQLQuery = None
    GraphQLResolver = None
    get_graphql_api = None

# WebSocket Manager
try:
    from .websocket_manager import (
    WebSocketManager,
    WebSocketConnection,
    ConnectionStatus,
    get_websocket_manager
    )
except ImportError:
    WebSocketManager = None
    WebSocketConnection = None
    ConnectionStatus = None
    get_websocket_manager = None

# Streaming System
try:
    from .streaming_system import (
    StreamingSystem,
    Stream,
    StreamStatus,
    get_streaming_system
    )
except ImportError:
    StreamingSystem = None
    Stream = None
    StreamStatus = None
    get_streaming_system = None

# Event Bus
try:
    from .event_bus import (
    EventBus,
    Event,
    get_event_bus
    )
except ImportError:
    EventBus = None
    Event = None
    get_event_bus = None

# Cache Warming
try:
    from .cache_warming import (
    CacheWarmer,
    CacheWarmingTask,
    get_cache_warmer
    )
except ImportError:
    CacheWarmer = None
    CacheWarmingTask = None
    get_cache_warmer = None

# Connection Pool
try:
    from .connection_pool import (
    ConnectionPool,
    Connection,
    ConnectionStatus,
    create_connection_pool,
    get_connection_pool
    )
except ImportError:
    ConnectionPool = None
    Connection = None
    ConnectionStatus = None
    create_connection_pool = None
    get_connection_pool = None

# Load Balancer
try:
    from .load_balancer import (
    LoadBalancer,
    Server,
    LoadBalanceStrategy,
    create_load_balancer,
    get_load_balancer
    )
except ImportError:
    LoadBalancer = None
    Server = None
    LoadBalanceStrategy = None
    create_load_balancer = None
    get_load_balancer = None

# Circuit Breaker
try:
    from .circuit_breaker import (
    CircuitBreakerManager,
    CircuitBreaker,
    CircuitState,
    get_circuit_breaker_manager
    )
except ImportError:
    CircuitBreakerManager = None
    CircuitBreaker = None
    CircuitState = None
    get_circuit_breaker_manager = None

# Service Discovery
try:
    from .service_discovery import (
    ServiceDiscovery,
    Service,
    ServiceStatus,
    get_service_discovery
    )
except ImportError:
    ServiceDiscovery = None
    Service = None
    ServiceStatus = None
    get_service_discovery = None

# Distributed Lock
try:
    from .distributed_lock import (
    DistributedLockManager,
    Lock,
    LockStatus,
    get_distributed_lock_manager
    )
except ImportError:
    DistributedLockManager = None
    Lock = None
    LockStatus = None
    get_distributed_lock_manager = None

# Message Queue
try:
    from .message_queue import (
    MessageQueueManager,
    MessageQueue,
    Message,
    MessagePriority,
    get_message_queue_manager
    )
except ImportError:
    MessageQueueManager = None
    MessageQueue = None
    Message = None
    MessagePriority = None
    get_message_queue_manager = None

# Job Queue
try:
    from .job_queue import (
    JobQueueManager,
    JobQueue,
    Job,
    JobStatus,
    get_job_queue_manager
    )
except ImportError:
    JobQueueManager = None
    JobQueue = None
    Job = None
    JobStatus = None
    get_job_queue_manager = None

# Advanced Rate Limiter
try:
    from .rate_limiter_advanced import (
    AdvancedRateLimiter,
    RateLimitRule,
    RateLimitResult,
    RateLimitStrategy,
    get_advanced_rate_limiter
    )
except ImportError:
    AdvancedRateLimiter = None
    RateLimitRule = None
    RateLimitResult = None
    RateLimitStrategy = None
    get_advanced_rate_limiter = None

# Throttle Manager
try:
    from .throttle_manager import (
    ThrottleManager,
    ThrottleRule,
    get_throttle_manager
    )
except ImportError:
    ThrottleManager = None
    ThrottleRule = None
    get_throttle_manager = None

# Retry Manager
try:
    from .retry_manager import (
        RetryManager,
        RetryConfig,
        RetryResult,
        RetryStrategy,
        get_retry_manager
    )
except ImportError:
    RetryManager = None
    RetryConfig = None
    RetryResult = None
    RetryStrategy = None
    get_retry_manager = None

# Timeout Manager
try:
    from .timeout_manager import (
    TimeoutManager,
    TimeoutConfig,
    get_timeout_manager
    )
except ImportError:
    TimeoutManager = None
    TimeoutConfig = None
    get_timeout_manager = None

# Health Monitor
try:
    from .health_monitor import (
    HealthMonitor,
    HealthCheck,
    HealthReport,
    HealthStatus,
    get_health_monitor
    )
except ImportError:
    HealthMonitor = None
    HealthCheck = None
    HealthReport = None
    HealthStatus = None
    get_health_monitor = None

# Graceful Shutdown
try:
    from .graceful_shutdown import (
    GracefulShutdownManager,
    ShutdownHandler,
    get_graceful_shutdown_manager
    )
except ImportError:
    GracefulShutdownManager = None
    ShutdownHandler = None
    get_graceful_shutdown_manager = None

# Resource Pool
try:
    from .resource_pool import (
    ResourcePool,
    Resource,
    ResourceStatus,
    create_resource_pool,
    get_resource_pool
    )
except ImportError:
    ResourcePool = None
    Resource = None
    ResourceStatus = None
    create_resource_pool = None
    get_resource_pool = None

# Advanced Batch Processor
try:
    from .batch_processor_advanced import (
    AdvancedBatchProcessor,
    Batch,
    BatchItem,
    BatchStatus,
    get_advanced_batch_processor
    )
except ImportError:
    AdvancedBatchProcessor = None
    Batch = None
    BatchItem = None
    BatchStatus = None
    get_advanced_batch_processor = None

# Observability System
try:
    from .observability_system import (
    ObservabilitySystem,
    Trace,
    Span,
    ObservabilityLevel,
    get_observability_system
    )
except ImportError:
    ObservabilitySystem = None
    Trace = None
    Span = None
    ObservabilityLevel = None
    get_observability_system = None

# Tracing System
try:
    from .tracing_system import (
    TracingSystem,
    TraceContext,
    get_tracing_system
    )
except ImportError:
    TracingSystem = None
    TraceContext = None
    get_tracing_system = None

# Distributed Cache
try:
    from .distributed_cache import (
        DistributedCache,
        CacheEntry,
        CacheStrategy,
        create_distributed_cache,
        get_distributed_cache
    )
except ImportError:
    DistributedCache = None
    CacheEntry = None
    CacheStrategy = None
    create_distributed_cache = None
    get_distributed_cache = None

# Distributed State
try:
    from .distributed_state import (
    DistributedState,
    StateEntry,
    StateStatus,
    create_distributed_state,
    get_distributed_state
    )
except ImportError:
    DistributedState = None
    StateEntry = None
    StateStatus = None
    create_distributed_state = None
    get_distributed_state = None

# Async Task Manager
try:
    from .async_task_manager import (
    AsyncTaskManager,
    AsyncTask,
    TaskStatus,
    get_async_task_manager
    )
except ImportError:
    AsyncTaskManager = None
    AsyncTask = None
    TaskStatus = None
    get_async_task_manager = None

# Event Sourcing
try:
    from .event_sourcing import (
    EventStore,
    Event,
    EventType,
    get_event_store
    )
except ImportError:
    EventStore = None
    Event = None
    EventType = None
    get_event_store = None

# CQRS Pattern
try:
    from .cqrs_pattern import (
    CQRSSystem,
    Command,
    Query,
    CommandResult,
    QueryResult,
    get_cqrs_system
    )
except ImportError:
    CQRSSystem = None
    Command = None
    Query = None
    CommandResult = None
    QueryResult = None
    get_cqrs_system = None

# Saga Pattern
try:
    from .saga_pattern import (
    SagaManager,
    Saga,
    SagaStep,
    SagaStatus,
    StepStatus,
    get_saga_manager
    )
except ImportError:
    SagaManager = None
    Saga = None
    SagaStep = None
    SagaStatus = None
    StepStatus = None
    get_saga_manager = None

# Microservices Orchestrator
try:
    from .microservices_orchestrator import (
    MicroservicesOrchestrator,
    Microservice,
    ServiceStatus,
    get_microservices_orchestrator
    )
except ImportError:
    MicroservicesOrchestrator = None
    Microservice = None
    ServiceStatus = None
    get_microservices_orchestrator = None

# API Composition
try:
    from .api_composition import (
    APIComposer,
    APIComposition,
    CompositionStrategy,
    get_api_composer
    )
except ImportError:
    APIComposer = None
    APIComposition = None
    CompositionStrategy = None
    get_api_composer = None

# Data Replication
try:
    from .data_replication import (
        DataReplicationManager,
        ReplicationTarget,
        ReplicationJob,
        ReplicationStatus,
        get_data_replication_manager
    )
except ImportError:
    DataReplicationManager = None
    ReplicationTarget = None
    ReplicationJob = None
    ReplicationStatus = None
    get_data_replication_manager = None

# Data Synchronization
try:
    from .data_synchronization import (
    DataSynchronizationManager,
    SyncEndpoint,
    SyncRule,
    SyncResult,
    SyncStatus,
    SyncDirection,
    get_data_synchronization_manager
    )
except ImportError:
    DataSynchronizationManager = None
    SyncEndpoint = None
    SyncRule = None
    SyncResult = None
    SyncStatus = None
    SyncDirection = None
    get_data_synchronization_manager = None

# Adaptive Learning
try:
    from .adaptive_learning import (
    AdaptiveLearningSystem,
    LearningMetric,
    LearningPattern,
    get_adaptive_learning_system
    )
except ImportError:
    AdaptiveLearningSystem = None
    LearningMetric = None
    LearningPattern = None
    get_adaptive_learning_system = None

# Predictive Analytics - already wrapped above (see line ~1172)

# Deep Learning Models
try:
    from .deep_learning_models import (
    DeepLearningModelManager,
    TrajectoryPredictor,
    MotionController,
    ObstacleDetector,
    ModelType,
    ModelConfig,
    TrainingMetrics,
    ModelCheckpoint,
    get_dl_model_manager
    )
except ImportError:
    DeepLearningModelManager = None
    TrajectoryPredictor = None
    MotionController = None
    ObstacleDetector = None
    ModelType = None
    ModelConfig = None
    TrainingMetrics = None
    ModelCheckpoint = None
    get_dl_model_manager = None

# LLM Processor
try:
    from .llm_processor import (
    LLMProcessor,
    LLMTask,
    LLMConfig,
    LLMRequest,
    LLMResponse,
    CommandIntent,
    get_llm_processor
    )
except ImportError:
    LLMProcessor = None
    LLMTask = None
    LLMConfig = None
    LLMRequest = None
    LLMResponse = None
    CommandIntent = None
    get_llm_processor = None

# Model Training
try:
    from .model_training import (
        ModelTrainer,
        TrainingStrategy,
        TrainingConfig,
        TrainingProgress,
        TrainingResult,
        get_model_trainer
    )
except ImportError:
    ModelTrainer = None
    TrainingStrategy = None
    TrainingConfig = None
    TrainingProgress = None
    TrainingResult = None
    get_model_trainer = None

# Gradio Interface
try:
    from .gradio_interface import (
        GradioInterfaceManager,
        GradioInterface,
        get_gradio_manager
    )
except ImportError:
    GradioInterfaceManager = None
    GradioInterface = None
    get_gradio_manager = None

__all__ = [
    # Core classes
    "TrajectoryOptimizer",
    "TrajectoryPoint",
    "OptimizationParams",
    "RobotMovementEngine",
    # Algorithms
    "BaseOptimizationAlgorithm",
    "PPOAlgorithm",
    "DQNAlgorithm",
    "AStarAlgorithm",
    "RRTAlgorithm",
    "HeuristicAlgorithm",
    # Constants
    "OptimizationAlgorithm",
    # Exceptions
    "RobotMovementError",
    "TrajectoryError",
    # Validators
    "validate_trajectory",
    "validate_trajectory_point",
    # Decorators
    "log_execution_time",
    "handle_robot_errors",
    # Metrics
    "get_metrics_collector",
    "record_value",
    # Performance
    "measure_time",
    "PerformanceProfiler",
    # Helpers
    "clamp",
    "lerp",
    "euclidean_distance",
    # Utils
    "quaternion_slerp",
    # Types
    "Position3D",
    "OptimizationResult",
    # Serialization
    "serialize_trajectory",
    # Extensions
    "Extension",
    "get_extension_manager",
    # Factories
    "TrajectoryOptimizerFactory",
    "ComponentBuilder",
    # Cache
    "get_cache_manager",
    "cached",
    # Async
    "AsyncBatchProcessor",
    "retry_async",
    # Logging
    "setup_logging",
    "get_logger",
    # Initialization
    "SystemInitializer",
    "get_system_initializer",
    "initialize_system",
    "InitStage",
    # Quality
    "QualityChecker",
    "check_performance_quality",
    "check_system_health_quality",
    # Resource Manager
    "ResourceManager",
    "get_resource_manager",
    # Testing
    "create_mock_trajectory_point",
    "assert_trajectory_valid",
    "AsyncTestCase",
    # Debug
    "debug_print",
    "trace_function",
    "DebugContext",
    # Versioning
    "Version",
    "VersionManager",
    "get_version_manager",
    "get_current_version",
    # Backup
    "BackupManager",
    "get_backup_manager",
    # Dynamic Config
    "DynamicConfigManager",
    "get_dynamic_config_manager",
    # Analytics
    "AnalyticsEngine",
    "get_analytics_engine",
    # Auto Optimizer
    "AutoOptimizer",
    "get_auto_optimizer",
    # Continuous Learning
    "ContinuousLearningSystem",
    "get_continuous_learning",
    # Benchmarking
    "BenchmarkRunner",
    "get_benchmark_runner",
    # Task Scheduler
    "TaskScheduler",
    "get_task_scheduler",
    # Workflow
    "Workflow",
    "WorkflowManager",
    "get_workflow_manager",
    # Rate Limiter
    "RateLimiter",
    "get_rate_limiter",
    # Validation Engine
    "ValidationEngine",
    "get_validation_engine",
    # Notification System
    "NotificationSystem",
    "get_notification_system",
    # API Client
    "APIClient",
    "get_api_client",
    # Data Pipeline
    "DataPipeline",
    "PipelineManager",
    "get_pipeline_manager",
    # State Machine
    "StateMachine",
    "StateMachineManager",
    "get_state_machine_manager",
    # Documentation Generator
    "DocumentationGenerator",
    "get_documentation_generator",
    # Code Quality
    "CodeQualityAnalyzer",
    "get_code_quality_analyzer",
    # Performance Monitor
    "PerformanceMonitor",
    "get_performance_monitor",
    # Error Tracker
    "ErrorTracker",
    "get_error_tracker",
    # Summary Generator
    "ProjectSummaryGenerator",
    "get_summary_generator",
    # Export/Import Utilities
    "ExportManager",
    "get_export_manager",
    "ImportManager",
    "get_import_manager",
    # Test Runner
    "TestRunner",
    "get_test_runner",
    # Deployment Utilities
    "DeploymentManager",
    "get_deployment_manager",
    # CLI Tools
    "ProjectCLI",
    "CLITool",
    # Maintenance Utilities
    "MaintenanceManager",
    "get_maintenance_manager",
    # Security Audit
    "SecurityAuditor",
    "get_security_auditor",
    # Compliance Checker
    "ComplianceChecker",
    "get_compliance_checker",
    # Performance Tuner
    "PerformanceTuner",
    "get_performance_tuner",
    # Optimization Profiler
    "OptimizationProfiler",
    "get_optimization_profiler",
    # Data Validator
    "DataValidator",
    "get_data_validator",
    # Data Transformer
    "DataTransformer",
    "get_data_transformer",
    # Report Generator
    "ReportGenerator",
    "get_report_generator",
    # Dashboard Builder
    "DashboardBuilder",
    "get_dashboard_builder",
    # Feature Flags
    "FeatureFlagManager",
    "get_feature_flag_manager",
    # Experiment Manager
    "ExperimentManager",
    "get_experiment_manager",
    # Knowledge Base
    "KnowledgeBase",
    "get_knowledge_base",
    # Recommendation Engine
    "RecommendationEngine",
    "get_recommendation_engine",
    # Collaboration System
    "CollaborationSystem",
    "get_collaboration_system",
    # Audit Log
    "AuditLogger",
    "get_audit_logger",
    # Version Control
    "VersionControl",
    "get_version_control",
    # Snapshot Manager
    "SnapshotManager",
    "get_snapshot_manager",
    # Webhook System
    "WebhookSystem",
    "get_webhook_system",
    # API Gateway
    "APIGateway",
    "get_api_gateway",
    # GraphQL API
    "GraphQLAPI",
    "get_graphql_api",
    # WebSocket Manager
    "WebSocketManager",
    "get_websocket_manager",
    # Streaming System
    "StreamingSystem",
    "get_streaming_system",
    # Event Bus
    "EventBus",
    "get_event_bus",
    # Cache Warming
    "CacheWarmer",
    "get_cache_warmer",
    # Connection Pool
    "ConnectionPool",
    "create_connection_pool",
    "get_connection_pool",
    # Load Balancer
    "LoadBalancer",
    "create_load_balancer",
    "get_load_balancer",
    # Circuit Breaker
    "CircuitBreakerManager",
    "get_circuit_breaker_manager",
    # Service Discovery
    "ServiceDiscovery",
    "get_service_discovery",
    # Distributed Lock
    "DistributedLockManager",
    "get_distributed_lock_manager",
    # Message Queue
    "MessageQueueManager",
    "get_message_queue_manager",
    # Job Queue
    "JobQueueManager",
    "get_job_queue_manager",
    # Advanced Rate Limiter
    "AdvancedRateLimiter",
    "get_advanced_rate_limiter",
    # Throttle Manager
    "ThrottleManager",
    "get_throttle_manager",
    # Retry Manager
    "RetryManager",
    "get_retry_manager",
    # Timeout Manager
    "TimeoutManager",
    "get_timeout_manager",
    # Health Monitor
    "HealthMonitor",
    "get_health_monitor",
    # Graceful Shutdown
    "GracefulShutdownManager",
    "get_graceful_shutdown_manager",
    # Resource Pool
    "ResourcePool",
    "create_resource_pool",
    "get_resource_pool",
    # Advanced Batch Processor
    "AdvancedBatchProcessor",
    "get_advanced_batch_processor",
    # Observability System
    "ObservabilitySystem",
    "get_observability_system",
    # Tracing System
    "TracingSystem",
    "get_tracing_system",
    # Distributed Cache
    "DistributedCache",
    "create_distributed_cache",
    "get_distributed_cache",
    # Distributed State
    "DistributedState",
    "create_distributed_state",
    "get_distributed_state",
    # Async Task Manager
    "AsyncTaskManager",
    "get_async_task_manager",
    # Event Sourcing
    "EventStore",
    "get_event_store",
    # CQRS Pattern
    "CQRSSystem",
    "get_cqrs_system",
    # Saga Pattern
    "SagaManager",
    "get_saga_manager",
    # Microservices Orchestrator
    "MicroservicesOrchestrator",
    "get_microservices_orchestrator",
    # API Composition
    "APIComposer",
    "get_api_composer",
    # Data Replication
    "DataReplicationManager",
    "get_data_replication_manager",
    # Data Synchronization
    "DataSynchronizationManager",
    "get_data_synchronization_manager",
    # Adaptive Learning
    "AdaptiveLearningSystem",
    "get_adaptive_learning_system",
    # Predictive Analytics
    "PredictiveAnalyticsSystem",
    "get_predictive_analytics_system",
    # Deep Learning Models
    "DeepLearningModelManager",
    "TrajectoryPredictor",
    "MotionController",
    "ObstacleDetector",
    "ModelType",
    "ModelConfig",
    "get_dl_model_manager",
    # LLM Processor
    "LLMProcessor",
    "LLMTask",
    "LLMConfig",
    "CommandIntent",
    "get_llm_processor",
    # Model Training
    "ModelTrainer",
    "TrainingStrategy",
    "TrainingConfig",
    "TrainingProgress",
    "get_model_trainer",
    # Gradio Interface
    "GradioInterfaceManager",
    "get_gradio_manager",
]
