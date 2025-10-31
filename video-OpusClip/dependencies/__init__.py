"""
Dependencies Module for Video-OpusClip
Dependency injection and management system
"""

from .dependency_container import (
    DependencyContainer, ServiceProvider, ServiceLifetime,
    register_singleton, register_transient, register_scoped,
    resolve, resolve_all, get_service_provider
)

from .service_registration import (
    register_services, register_database_services,
    register_auth_services, register_logging_services,
    register_cache_services, register_queue_services,
    register_storage_services, register_monitoring_services
)

from .dependency_resolution import (
    resolve_dependencies, inject_dependencies,
    create_dependency_graph, validate_dependencies,
    DependencyResolver, CircularDependencyError
)

from .configuration_dependencies import (
    ConfigurationService, DatabaseConfiguration,
    AuthConfiguration, LoggingConfiguration,
    CacheConfiguration, QueueConfiguration,
    StorageConfiguration, MonitoringConfiguration
)

from .database_dependencies import (
    DatabaseService, DatabaseConnection,
    DatabaseSession, DatabaseTransaction,
    DatabaseMigration, DatabaseHealthCheck
)

from .auth_dependencies import (
    AuthService, JWTService, PasswordService,
    PermissionService, RoleService, UserService
)

from .logging_dependencies import (
    LoggingService, StructuredLogger,
    LogFormatter, LogHandler, LogFilter
)

from .cache_dependencies import (
    CacheService, RedisCache, MemoryCache,
    CacheKey, CacheValue, CachePolicy
)

from .queue_dependencies import (
    QueueService, MessageQueue, TaskQueue,
    QueueProducer, QueueConsumer, QueueHandler
)

from .storage_dependencies import (
    StorageService, FileStorage, ObjectStorage,
    StorageProvider, StorageConfig, StorageHealthCheck
)

from .monitoring_dependencies import (
    MonitoringService, MetricsService, HealthCheckService,
    AlertService, DashboardService, ReportingService
)

from .fastapi_integration import (
    FastAPIDependencyContainer, FastAPIServiceProvider,
    get_fastapi_dependencies, inject_fastapi_dependencies,
    FastAPIDependencyResolver
)

__all__ = [
    # Core Dependency Container
    'DependencyContainer', 'ServiceProvider', 'ServiceLifetime',
    'register_singleton', 'register_transient', 'register_scoped',
    'resolve', 'resolve_all', 'get_service_provider',
    
    # Service Registration
    'register_services', 'register_database_services',
    'register_auth_services', 'register_logging_services',
    'register_cache_services', 'register_queue_services',
    'register_storage_services', 'register_monitoring_services',
    
    # Dependency Resolution
    'resolve_dependencies', 'inject_dependencies',
    'create_dependency_graph', 'validate_dependencies',
    'DependencyResolver', 'CircularDependencyError',
    
    # Configuration Dependencies
    'ConfigurationService', 'DatabaseConfiguration',
    'AuthConfiguration', 'LoggingConfiguration',
    'CacheConfiguration', 'QueueConfiguration',
    'StorageConfiguration', 'MonitoringConfiguration',
    
    # Database Dependencies
    'DatabaseService', 'DatabaseConnection',
    'DatabaseSession', 'DatabaseTransaction',
    'DatabaseMigration', 'DatabaseHealthCheck',
    
    # Auth Dependencies
    'AuthService', 'JWTService', 'PasswordService',
    'PermissionService', 'RoleService', 'UserService',
    
    # Logging Dependencies
    'LoggingService', 'StructuredLogger',
    'LogFormatter', 'LogHandler', 'LogFilter',
    
    # Cache Dependencies
    'CacheService', 'RedisCache', 'MemoryCache',
    'CacheKey', 'CacheValue', 'CachePolicy',
    
    # Queue Dependencies
    'QueueService', 'MessageQueue', 'TaskQueue',
    'QueueProducer', 'QueueConsumer', 'QueueHandler',
    
    # Storage Dependencies
    'StorageService', 'FileStorage', 'ObjectStorage',
    'StorageProvider', 'StorageConfig', 'StorageHealthCheck',
    
    # Monitoring Dependencies
    'MonitoringService', 'MetricsService', 'HealthCheckService',
    'AlertService', 'DashboardService', 'ReportingService',
    
    # FastAPI Integration
    'FastAPIDependencyContainer', 'FastAPIServiceProvider',
    'get_fastapi_dependencies', 'inject_fastapi_dependencies',
    'FastAPIDependencyResolver'
] 