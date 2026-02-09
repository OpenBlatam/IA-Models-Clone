"""
Core Module - Módulo core refactorizado
======================================

Módulo core con clases base y utilidades comunes.
"""

from .base_service import BaseService
from .base_repository import BaseRepository
from .exceptions import (
    AIProjectGeneratorError,
    ProjectNotFoundError,
    ProjectGenerationError,
    ValidationError,
    CacheError,
    ServiceUnavailableError,
    ConfigurationError,
    RepositoryError
)
from .error_handler import ErrorHandler, get_error_handler, setup_error_handlers
from .utils import (
    generate_id,
    hash_data,
    sanitize_dict,
    merge_dicts,
    format_duration,
    validate_required_fields,
    parse_datetime
)
from .validators import (
    validate_project_name,
    validate_description,
    validate_email,
    validate_url,
    ProjectNameValidator,
    DescriptionValidator
)
from .decorators import (
    timed,
    logged,
    cached,
    profiled,
    retry_on_failure
)
from .robustness import (
    RobustService,
    RobustRepository,
    RobustHealthChecker,
    get_health_checker,
    RobustValidator,
    ProjectDataModel,
    DependencyValidator,
    get_dependency_validator,
    FallbackManager,
    get_fallback_manager,
    TimeoutManager,
    get_timeout_manager
)
from .advanced_api_gateway import (
    AdvancedAPIGatewayClient,
    get_advanced_api_gateway_client,
    RateLimitStrategy
)
from .advanced_serverless import (
    AdvancedServerlessOptimizer,
    get_advanced_serverless_optimizer,
    ServerlessEnvironment
)
from .advanced_security import (
    DDoSProtection,
    AdvancedRateLimiter,
    SecurityHeaders,
    ContentValidator,
    AdvancedSecurityMiddleware,
    get_advanced_security_middleware
)
from .cloud_services import (
    CloudDatabase,
    DynamoDBClient,
    CosmosDBClient,
    get_cloud_database
)
from .service_mesh import (
    ServiceMeshType,
    ServiceMeshClient,
    IstioClient,
    LinkerdClient,
    get_service_mesh_client
)
from .load_balancer import (
    LoadBalanceStrategy,
    BackendServer,
    LoadBalancer,
    get_load_balancer
)
from .search_engine import (
    SearchEngine,
    ElasticsearchClient,
    get_search_engine
)
from .centralized_logging import (
    LoggingBackend,
    CentralizedLogger,
    ELKLogger,
    CloudWatchLogger,
    StructuredLogger,
    get_centralized_logger
)
from .container_optimizer import (
    ContainerOptimizer,
    get_container_optimizer
)
from .advanced_message_broker import (
    MessageBrokerType,
    MessageBroker,
    RabbitMQBroker,
    KafkaBroker,
    EventSourcing,
    get_message_broker
)
from .distributed_tracing import (
    TracingBackend,
    DistributedTracer,
    get_distributed_tracer
)
from .reverse_proxy import (
    ReverseProxyType,
    ReverseProxyConfig,
    generate_reverse_proxy_config
)
from .auto_scaling import (
    ScalingPolicy,
    AutoScaler,
    get_auto_scaler
)
from .api_versioning import (
    VersioningStrategy,
    APIVersionRouter,
    create_versioned_route,
    get_api_version_router,
    get_api_version_manager
)
from .websocket_manager import (
    ConnectionManager,
    WebSocketRateLimiter,
    get_connection_manager,
    get_websocket_rate_limiter
)
from .performance_profiler import (
    PerformanceProfiler,
    get_performance_profiler
)
from .advanced_caching import (
    CacheStrategy,
    CacheEntry,
    AdvancedCache,
    get_advanced_cache
)
from .graphql_support import (
    GraphQLType,
    GraphQLSchemaBuilder,
    GraphQLExecutor,
    get_graphql_schema_builder,
    get_graphql_executor
)
from .grpc_integration import (
    GRPCServiceType,
    GRPCService,
    GRPCClient,
    get_grpc_service,
    get_grpc_client
)
from .feature_flags import (
    FeatureFlagType,
    FeatureFlag,
    FeatureFlagManager,
    get_feature_flag_manager
)
from .database_migrations import (
    MigrationStatus,
    Migration,
    MigrationManager,
    get_migration_manager
)
from .backup_recovery import (
    BackupType,
    Backup,
    BackupManager,
    get_backup_manager
)
from .testing_utilities import (
    TestDataFactory,
    MockService,
    TestFixture,
    IntegrationTestHelper,
    get_test_data_factory,
    get_integration_test_helper
)
from .advanced_openapi import (
    AdvancedOpenAPIConfig,
    customize_openapi,
    add_api_examples,
    get_openapi_config
)
from .distributed_rate_limiter import (
    DistributedRateLimiter,
    get_distributed_rate_limiter
)
from .queue_manager import (
    JobStatus,
    JobPriority,
    Job,
    QueueManager,
    get_queue_manager
)
from .cqrs_pattern import (
    Command,
    Query,
    CommandHandler,
    QueryHandler,
    CQRSBus,
    CreateProjectCommand,
    GetProjectQuery,
    get_cqrs_bus
)
from .saga_pattern import (
    SagaStatus,
    SagaStep,
    Saga,
    SagaOrchestrator,
    get_saga_orchestrator
)
from .service_discovery import (
    ServiceStatus,
    ServiceInstance,
    ServiceRegistry,
    get_service_registry
)
from .config_manager import (
    ConfigSource,
    ConfigManager,
    get_config_manager
)
from .monitoring_alerting import (
    AlertSeverity,
    AlertChannel,
    AlertRule,
    Alert,
    MonitoringSystem,
    get_monitoring_system
)
from .data_encryption import (
    EncryptionAlgorithm,
    DataEncryptor,
    get_data_encryptor
)
from .audit_logging import (
    AuditEventType,
    AuditLog,
    AuditLogger,
    get_audit_logger
)
from .multi_tenancy import (
    TenantIsolation,
    Tenant,
    TenantManager,
    get_tenant_manager
)
from .api_testing import (
    APITestClient,
    LoadTestRunner,
    get_api_test_client,
    get_load_test_runner
)
from .webhook_manager import (
    WebhookStatus,
    Webhook,
    WebhookManager,
    get_webhook_manager
)
from .performance_benchmark import (
    BenchmarkResult,
    PerformanceBenchmark,
    get_performance_benchmark
)
from .deployment_automation import (
    DeploymentStrategy,
    DeploymentStatus,
    Deployment,
    DeploymentManager,
    get_deployment_manager
)
from .cache_warming import (
    WarmingStrategy,
    CacheWarmer,
    get_cache_warmer
)
from .api_documentation_generator import (
    DocumentationFormat,
    APIDocumentationGenerator,
    get_api_documentation_generator
)
from .api_analytics import (
    APIRequest,
    APIAnalytics,
    get_api_analytics
)
from .request_response_transformer import (
    TransformationType,
    RequestResponseTransformer,
    get_request_response_transformer
)
from .advanced_error_recovery import (
    ErrorType,
    RecoveryStrategy,
    ErrorRecoveryManager,
    get_error_recovery_manager
)
from .user_rate_limiting import (
    RateLimitTier,
    UserRateLimiter,
    get_user_rate_limiter
)
from .schema_validation import (
    ValidationLevel,
    SchemaValidator,
    get_schema_validator
)
from .api_version_manager import (
    VersionStatus,
    APIVersion,
    APIVersionManager,
    get_api_version_manager
)
from .compression_manager import (
    CompressionAlgorithm,
    CompressionManager,
    get_compression_manager
)
from .api_throttling import (
    ThrottlePolicy,
    RequestPriority,
    APIThrottler,
    get_api_throttler
)
from .query_optimizer import (
    QueryType,
    QueryOptimizer,
    get_query_optimizer
)
from .request_interceptor import (
    InterceptorType,
    RequestInterceptor,
    get_request_interceptor
)
from .resource_pool import (
    PoolStatus,
    ResourcePool,
    create_resource_pool
)
from .advanced_caching_strategies import (
    CachePattern,
    AdvancedCacheStrategy,
    get_advanced_cache_strategy
)
from .api_mocking import (
    MockMatchType,
    APIMock,
    APIMockServer,
    get_api_mock_server
)
from .advanced_request_validation import (
    ValidationRule,
    ValidationError,
    AdvancedRequestValidator,
    get_advanced_request_validator
)
from .data_transformation_pipeline import (
    TransformationStage,
    DataTransformationPipeline,
    create_transformation_pipeline
)
from .shared_utils import (
    get_logger,
    validate_path,
    ensure_directory,
    safe_write_file,
    sanitize_filename,
    format_project_name,
    merge_dicts,
    get_nested_value,
    set_nested_value
)
from .json_utils import (
    json_dumps,
    json_loads,
    json_dumps_str,
    json_dumps_pretty
)

__all__ = [
    "BaseService",
    "BaseRepository",
    "AIProjectGeneratorError",
    "ProjectNotFoundError",
    "ProjectGenerationError",
    "ValidationError",
    "CacheError",
    "ServiceUnavailableError",
    "ConfigurationError",
    "RepositoryError",
    "ErrorHandler",
    "get_error_handler",
    "setup_error_handlers",
    "generate_id",
    "hash_data",
    "sanitize_dict",
    "merge_dicts",
    "format_duration",
    "validate_required_fields",
    "parse_datetime",
    "validate_project_name",
    "validate_description",
    "validate_email",
    "validate_url",
    "ProjectNameValidator",
    "DescriptionValidator",
    "timed",
    "logged",
    "cached",
    "profiled",
    "retry_on_failure",
    # Robustness
    "RobustService",
    "RobustRepository",
    "RobustHealthChecker",
    "get_health_checker",
    "RobustValidator",
    "ProjectDataModel",
    "DependencyValidator",
    "get_dependency_validator",
    "FallbackManager",
    "get_fallback_manager",
    "TimeoutManager",
    "get_timeout_manager",
    # Advanced API Gateway
    "AdvancedAPIGatewayClient",
    "get_advanced_api_gateway_client",
    "RateLimitStrategy",
    # Advanced Serverless
    "AdvancedServerlessOptimizer",
    "get_advanced_serverless_optimizer",
    "ServerlessEnvironment",
    # Advanced Security
    "DDoSProtection",
    "AdvancedRateLimiter",
    "SecurityHeaders",
    "ContentValidator",
    "AdvancedSecurityMiddleware",
    "get_advanced_security_middleware",
    # Cloud Services
    "CloudDatabase",
    "DynamoDBClient",
    "CosmosDBClient",
    "get_cloud_database",
    # Service Mesh
    "ServiceMeshType",
    "ServiceMeshClient",
    "IstioClient",
    "LinkerdClient",
    "get_service_mesh_client",
    # Load Balancer
    "LoadBalanceStrategy",
    "BackendServer",
    "LoadBalancer",
    "get_load_balancer",
    # Search Engine
    "SearchEngine",
    "ElasticsearchClient",
    "get_search_engine",
    # Centralized Logging
    "LoggingBackend",
    "CentralizedLogger",
    "ELKLogger",
    "CloudWatchLogger",
    "StructuredLogger",
    "get_centralized_logger",
    # Container Optimizer
    "ContainerOptimizer",
    "get_container_optimizer",
    # Advanced Message Broker
    "MessageBrokerType",
    "MessageBroker",
    "RabbitMQBroker",
    "KafkaBroker",
    "EventSourcing",
    "get_message_broker",
    # Distributed Tracing
    "TracingBackend",
    "DistributedTracer",
    "get_distributed_tracer",
    # Reverse Proxy
    "ReverseProxyType",
    "ReverseProxyConfig",
    "generate_reverse_proxy_config",
    # Auto Scaling
    "ScalingPolicy",
    "AutoScaler",
    "get_auto_scaler",
    # API Versioning
    "VersioningStrategy",
    "APIVersionRouter",
    "create_versioned_route",
    "get_api_version_router",
    "get_api_version_manager",
    # WebSocket Manager
    "ConnectionManager",
    "WebSocketRateLimiter",
    "get_connection_manager",
    "get_websocket_rate_limiter",
    # Performance Profiler
    "PerformanceProfiler",
    "get_performance_profiler",
    # Advanced Caching
    "CacheStrategy",
    "CacheEntry",
    "AdvancedCache",
    "get_advanced_cache",
    # GraphQL Support
    "GraphQLType",
    "GraphQLSchemaBuilder",
    "GraphQLExecutor",
    "get_graphql_schema_builder",
    "get_graphql_executor",
    # gRPC Integration
    "GRPCServiceType",
    "GRPCService",
    "GRPCClient",
    "get_grpc_service",
    "get_grpc_client",
    # Feature Flags
    "FeatureFlagType",
    "FeatureFlag",
    "FeatureFlagManager",
    "get_feature_flag_manager",
    # Database Migrations
    "MigrationStatus",
    "Migration",
    "MigrationManager",
    "get_migration_manager",
    # Backup and Recovery
    "BackupType",
    "Backup",
    "BackupManager",
    "get_backup_manager",
    # Testing Utilities
    "TestDataFactory",
    "MockService",
    "TestFixture",
    "IntegrationTestHelper",
    "get_test_data_factory",
    "get_integration_test_helper",
    # Advanced OpenAPI
    "AdvancedOpenAPIConfig",
    "customize_openapi",
    "add_api_examples",
    "get_openapi_config",
    # Distributed Rate Limiter
    "DistributedRateLimiter",
    "get_distributed_rate_limiter",
    # Queue Manager
    "JobStatus",
    "JobPriority",
    "Job",
    "QueueManager",
    "get_queue_manager",
    # CQRS Pattern
    "Command",
    "Query",
    "CommandHandler",
    "QueryHandler",
    "CQRSBus",
    "CreateProjectCommand",
    "GetProjectQuery",
    "get_cqrs_bus",
    # Saga Pattern
    "SagaStatus",
    "SagaStep",
    "Saga",
    "SagaOrchestrator",
    "get_saga_orchestrator",
    # Service Discovery
    "ServiceStatus",
    "ServiceInstance",
    "ServiceRegistry",
    "get_service_registry",
    # Config Manager
    "ConfigSource",
    "ConfigManager",
    "get_config_manager",
    # Monitoring and Alerting
    "AlertSeverity",
    "AlertChannel",
    "AlertRule",
    "Alert",
    "MonitoringSystem",
    "get_monitoring_system",
    # Data Encryption
    "EncryptionAlgorithm",
    "DataEncryptor",
    "get_data_encryptor",
    # Audit Logging
    "AuditEventType",
    "AuditLog",
    "AuditLogger",
    "get_audit_logger",
    # Multi-Tenancy
    "TenantIsolation",
    "Tenant",
    "TenantManager",
    "get_tenant_manager",
    # API Testing
    "APITestClient",
    "LoadTestRunner",
    "get_api_test_client",
    "get_load_test_runner",
    # Webhook Manager
    "WebhookStatus",
    "Webhook",
    "WebhookManager",
    "get_webhook_manager",
    # Performance Benchmark
    "BenchmarkResult",
    "PerformanceBenchmark",
    "get_performance_benchmark",
    # Deployment Automation
    "DeploymentStrategy",
    "DeploymentStatus",
    "Deployment",
    "DeploymentManager",
    "get_deployment_manager",
    # Cache Warming
    "WarmingStrategy",
    "CacheWarmer",
    "get_cache_warmer",
    # API Documentation Generator
    "DocumentationFormat",
    "APIDocumentationGenerator",
    "get_api_documentation_generator",
    # API Analytics
    "APIRequest",
    "APIAnalytics",
    "get_api_analytics",
    # Request/Response Transformer
    "TransformationType",
    "RequestResponseTransformer",
    "get_request_response_transformer",
    # Advanced Error Recovery
    "ErrorType",
    "RecoveryStrategy",
    "ErrorRecoveryManager",
    "get_error_recovery_manager",
    # User Rate Limiting
    "RateLimitTier",
    "UserRateLimiter",
    "get_user_rate_limiter",
    # Schema Validation
    "ValidationLevel",
    "SchemaValidator",
    "get_schema_validator",
    # API Version Manager
    "VersionStatus",
    "APIVersion",
    "APIVersionManager",
    "get_api_version_manager",
    # Compression Manager
    "CompressionAlgorithm",
    "CompressionManager",
    "get_compression_manager",
    # API Throttling
    "ThrottlePolicy",
    "RequestPriority",
    "APIThrottler",
    "get_api_throttler",
    # Query Optimizer
    "QueryType",
    "QueryOptimizer",
    "get_query_optimizer",
    # Request Interceptor
    "InterceptorType",
    "RequestInterceptor",
    "get_request_interceptor",
    # Resource Pool
    "PoolStatus",
    "ResourcePool",
    "create_resource_pool",
    # Advanced Caching Strategies
    "CachePattern",
    "AdvancedCacheStrategy",
    "get_advanced_cache_strategy",
    # API Mocking
    "MockMatchType",
    "APIMock",
    "APIMockServer",
    "get_api_mock_server",
    # Advanced Request Validation
    "ValidationRule",
    "ValidationError",
    "AdvancedRequestValidator",
    "get_advanced_request_validator",
    # Data Transformation Pipeline
    "TransformationStage",
    "DataTransformationPipeline",
    "create_transformation_pipeline",
    # Shared Utilities
    "get_logger",
    "validate_path",
    "ensure_directory",
    "safe_write_file",
    "sanitize_filename",
    "format_project_name",
    "merge_dicts",
    "get_nested_value",
    "set_nested_value",
    # JSON Utilities
    "json_dumps",
    "json_loads",
    "json_dumps_str",
    "json_dumps_pretty",
]
