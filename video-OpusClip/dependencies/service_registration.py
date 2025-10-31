#!/usr/bin/env python3
"""
Service Registration for Video-OpusClip
Service registration utilities for different service types
"""

from typing import Dict, List, Any, Optional, Type, Callable
from dataclasses import dataclass

from .dependency_container import DependencyContainer, register_singleton, register_transient, register_scoped


@dataclass
class ServiceRegistration:
    """Service registration configuration"""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    lifetime: str = "singleton"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def register_services(container: DependencyContainer, services: List[ServiceRegistration]) -> DependencyContainer:
    """
    Register multiple services at once
    
    Args:
        container: Dependency container
        services: List of service registrations
        
    Returns:
        Updated container
    """
    for service in services:
        if service.lifetime == "singleton":
            container.register_singleton(
                service.service_type,
                service.implementation_type,
                service.factory,
                **service.metadata
            )
        elif service.lifetime == "transient":
            container.register_transient(
                service.service_type,
                service.implementation_type,
                service.factory,
                **service.metadata
            )
        elif service.lifetime == "scoped":
            container.register_scoped(
                service.service_type,
                service.implementation_type,
                service.factory,
                **service.metadata
            )
    
    return container


def register_database_services(container: DependencyContainer) -> DependencyContainer:
    """
    Register database-related services
    
    Args:
        container: Dependency container
        
    Returns:
        Updated container
    """
    # Database connection services
    container.register_singleton(
        "DatabaseConnection",
        factory=lambda config: create_database_connection(config)
    )
    
    # Database session services
    container.register_scoped(
        "DatabaseSession",
        factory=lambda connection: create_database_session(connection)
    )
    
    # Database transaction services
    container.register_transient(
        "DatabaseTransaction",
        factory=lambda session: create_database_transaction(session)
    )
    
    # Database migration services
    container.register_singleton(
        "DatabaseMigration",
        factory=lambda config: create_database_migration(config)
    )
    
    # Database health check services
    container.register_singleton(
        "DatabaseHealthCheck",
        factory=lambda connection: create_database_health_check(connection)
    )
    
    return container


def register_auth_services(container: DependencyContainer) -> DependencyContainer:
    """
    Register authentication and authorization services
    
    Args:
        container: Dependency container
        
    Returns:
        Updated container
    """
    # JWT service
    container.register_singleton(
        "JWTService",
        factory=lambda config: create_jwt_service(config)
    )
    
    # Password service
    container.register_singleton(
        "PasswordService",
        factory=lambda config: create_password_service(config)
    )
    
    # Permission service
    container.register_singleton(
        "PermissionService",
        factory=lambda config: create_permission_service(config)
    )
    
    # Role service
    container.register_singleton(
        "RoleService",
        factory=lambda config: create_role_service(config)
    )
    
    # User service
    container.register_singleton(
        "UserService",
        factory=lambda config: create_user_service(config)
    )
    
    # Auth service
    container.register_singleton(
        "AuthService",
        factory=lambda jwt_service, password_service, user_service: create_auth_service(
            jwt_service, password_service, user_service
        )
    )
    
    return container


def register_logging_services(container: DependencyContainer) -> DependencyContainer:
    """
    Register logging services
    
    Args:
        container: Dependency container
        
    Returns:
        Updated container
    """
    # Structured logger
    container.register_singleton(
        "StructuredLogger",
        factory=lambda config: create_structured_logger(config)
    )
    
    # Log formatter
    container.register_singleton(
        "LogFormatter",
        factory=lambda config: create_log_formatter(config)
    )
    
    # Log handler
    container.register_singleton(
        "LogHandler",
        factory=lambda config: create_log_handler(config)
    )
    
    # Log filter
    container.register_singleton(
        "LogFilter",
        factory=lambda config: create_log_filter(config)
    )
    
    # Logging service
    container.register_singleton(
        "LoggingService",
        factory=lambda logger, formatter, handler, filter: create_logging_service(
            logger, formatter, handler, filter
        )
    )
    
    return container


def register_cache_services(container: DependencyContainer) -> DependencyContainer:
    """
    Register caching services
    
    Args:
        container: Dependency container
        
    Returns:
        Updated container
    """
    # Redis cache
    container.register_singleton(
        "RedisCache",
        factory=lambda config: create_redis_cache(config)
    )
    
    # Memory cache
    container.register_singleton(
        "MemoryCache",
        factory=lambda config: create_memory_cache(config)
    )
    
    # Cache service
    container.register_singleton(
        "CacheService",
        factory=lambda redis_cache, memory_cache: create_cache_service(
            redis_cache, memory_cache
        )
    )
    
    return container


def register_queue_services(container: DependencyContainer) -> DependencyContainer:
    """
    Register queue services
    
    Args:
        container: Dependency container
        
    Returns:
        Updated container
    """
    # Message queue
    container.register_singleton(
        "MessageQueue",
        factory=lambda config: create_message_queue(config)
    )
    
    # Task queue
    container.register_singleton(
        "TaskQueue",
        factory=lambda config: create_task_queue(config)
    )
    
    # Queue producer
    container.register_singleton(
        "QueueProducer",
        factory=lambda message_queue: create_queue_producer(message_queue)
    )
    
    # Queue consumer
    container.register_singleton(
        "QueueConsumer",
        factory=lambda message_queue: create_queue_consumer(message_queue)
    )
    
    # Queue handler
    container.register_singleton(
        "QueueHandler",
        factory=lambda producer, consumer: create_queue_handler(producer, consumer)
    )
    
    # Queue service
    container.register_singleton(
        "QueueService",
        factory=lambda message_queue, task_queue, producer, consumer, handler: create_queue_service(
            message_queue, task_queue, producer, consumer, handler
        )
    )
    
    return container


def register_storage_services(container: DependencyContainer) -> DependencyContainer:
    """
    Register storage services
    
    Args:
        container: Dependency container
        
    Returns:
        Updated container
    """
    # File storage
    container.register_singleton(
        "FileStorage",
        factory=lambda config: create_file_storage(config)
    )
    
    # Object storage
    container.register_singleton(
        "ObjectStorage",
        factory=lambda config: create_object_storage(config)
    )
    
    # Storage provider
    container.register_singleton(
        "StorageProvider",
        factory=lambda file_storage, object_storage: create_storage_provider(
            file_storage, object_storage
        )
    )
    
    # Storage service
    container.register_singleton(
        "StorageService",
        factory=lambda provider: create_storage_service(provider)
    )
    
    return container


def register_monitoring_services(container: DependencyContainer) -> DependencyContainer:
    """
    Register monitoring services
    
    Args:
        container: Dependency container
        
    Returns:
        Updated container
    """
    # Metrics service
    container.register_singleton(
        "MetricsService",
        factory=lambda config: create_metrics_service(config)
    )
    
    # Health check service
    container.register_singleton(
        "HealthCheckService",
        factory=lambda config: create_health_check_service(config)
    )
    
    # Alert service
    container.register_singleton(
        "AlertService",
        factory=lambda config: create_alert_service(config)
    )
    
    # Dashboard service
    container.register_singleton(
        "DashboardService",
        factory=lambda config: create_dashboard_service(config)
    )
    
    # Reporting service
    container.register_singleton(
        "ReportingService",
        factory=lambda config: create_reporting_service(config)
    )
    
    # Monitoring service
    container.register_singleton(
        "MonitoringService",
        factory=lambda metrics, health_check, alert, dashboard, reporting: create_monitoring_service(
            metrics, health_check, alert, dashboard, reporting
        )
    )
    
    return container


# Factory functions for service creation
def create_database_connection(config: Dict[str, Any]):
    """Create database connection"""
    return {"type": "database_connection", "config": config}


def create_database_session(connection):
    """Create database session"""
    return {"type": "database_session", "connection": connection}


def create_database_transaction(session):
    """Create database transaction"""
    return {"type": "database_transaction", "session": session}


def create_database_migration(config: Dict[str, Any]):
    """Create database migration"""
    return {"type": "database_migration", "config": config}


def create_database_health_check(connection):
    """Create database health check"""
    return {"type": "database_health_check", "connection": connection}


def create_jwt_service(config: Dict[str, Any]):
    """Create JWT service"""
    return {"type": "jwt_service", "config": config}


def create_password_service(config: Dict[str, Any]):
    """Create password service"""
    return {"type": "password_service", "config": config}


def create_permission_service(config: Dict[str, Any]):
    """Create permission service"""
    return {"type": "permission_service", "config": config}


def create_role_service(config: Dict[str, Any]):
    """Create role service"""
    return {"type": "role_service", "config": config}


def create_user_service(config: Dict[str, Any]):
    """Create user service"""
    return {"type": "user_service", "config": config}


def create_auth_service(jwt_service, password_service, user_service):
    """Create auth service"""
    return {
        "type": "auth_service",
        "jwt_service": jwt_service,
        "password_service": password_service,
        "user_service": user_service
    }


def create_structured_logger(config: Dict[str, Any]):
    """Create structured logger"""
    return {"type": "structured_logger", "config": config}


def create_log_formatter(config: Dict[str, Any]):
    """Create log formatter"""
    return {"type": "log_formatter", "config": config}


def create_log_handler(config: Dict[str, Any]):
    """Create log handler"""
    return {"type": "log_handler", "config": config}


def create_log_filter(config: Dict[str, Any]):
    """Create log filter"""
    return {"type": "log_filter", "config": config}


def create_logging_service(logger, formatter, handler, filter):
    """Create logging service"""
    return {
        "type": "logging_service",
        "logger": logger,
        "formatter": formatter,
        "handler": handler,
        "filter": filter
    }


def create_redis_cache(config: Dict[str, Any]):
    """Create Redis cache"""
    return {"type": "redis_cache", "config": config}


def create_memory_cache(config: Dict[str, Any]):
    """Create memory cache"""
    return {"type": "memory_cache", "config": config}


def create_cache_service(redis_cache, memory_cache):
    """Create cache service"""
    return {
        "type": "cache_service",
        "redis_cache": redis_cache,
        "memory_cache": memory_cache
    }


def create_message_queue(config: Dict[str, Any]):
    """Create message queue"""
    return {"type": "message_queue", "config": config}


def create_task_queue(config: Dict[str, Any]):
    """Create task queue"""
    return {"type": "task_queue", "config": config}


def create_queue_producer(message_queue):
    """Create queue producer"""
    return {"type": "queue_producer", "message_queue": message_queue}


def create_queue_consumer(message_queue):
    """Create queue consumer"""
    return {"type": "queue_consumer", "message_queue": message_queue}


def create_queue_handler(producer, consumer):
    """Create queue handler"""
    return {"type": "queue_handler", "producer": producer, "consumer": consumer}


def create_queue_service(message_queue, task_queue, producer, consumer, handler):
    """Create queue service"""
    return {
        "type": "queue_service",
        "message_queue": message_queue,
        "task_queue": task_queue,
        "producer": producer,
        "consumer": consumer,
        "handler": handler
    }


def create_file_storage(config: Dict[str, Any]):
    """Create file storage"""
    return {"type": "file_storage", "config": config}


def create_object_storage(config: Dict[str, Any]):
    """Create object storage"""
    return {"type": "object_storage", "config": config}


def create_storage_provider(file_storage, object_storage):
    """Create storage provider"""
    return {
        "type": "storage_provider",
        "file_storage": file_storage,
        "object_storage": object_storage
    }


def create_storage_service(provider):
    """Create storage service"""
    return {"type": "storage_service", "provider": provider}


def create_metrics_service(config: Dict[str, Any]):
    """Create metrics service"""
    return {"type": "metrics_service", "config": config}


def create_health_check_service(config: Dict[str, Any]):
    """Create health check service"""
    return {"type": "health_check_service", "config": config}


def create_alert_service(config: Dict[str, Any]):
    """Create alert service"""
    return {"type": "alert_service", "config": config}


def create_dashboard_service(config: Dict[str, Any]):
    """Create dashboard service"""
    return {"type": "dashboard_service", "config": config}


def create_reporting_service(config: Dict[str, Any]):
    """Create reporting service"""
    return {"type": "reporting_service", "config": config}


def create_monitoring_service(metrics, health_check, alert, dashboard, reporting):
    """Create monitoring service"""
    return {
        "type": "monitoring_service",
        "metrics": metrics,
        "health_check": health_check,
        "alert": alert,
        "dashboard": dashboard,
        "reporting": reporting
    }


# Example usage
if __name__ == "__main__":
    # Example service registration
    print("🔧 Service Registration Example")
    
    # Create container
    from .dependency_container import DependencyContainer
    container = DependencyContainer()
    
    # Register different service types
    container = register_database_services(container)
    container = register_auth_services(container)
    container = register_logging_services(container)
    container = register_cache_services(container)
    container = register_queue_services(container)
    container = register_storage_services(container)
    container = register_monitoring_services(container)
    
    # Test service resolution
    try:
        auth_service = container.resolve("AuthService")
        print(f"✅ Auth service resolved: {auth_service}")
        
        cache_service = container.resolve("CacheService")
        print(f"✅ Cache service resolved: {cache_service}")
        
        monitoring_service = container.resolve("MonitoringService")
        print(f"✅ Monitoring service resolved: {monitoring_service}")
        
    except Exception as e:
        print(f"❌ Service resolution failed: {e}")
    
    # Validate container
    errors = container.validate()
    if errors:
        print(f"❌ Container validation errors: {errors}")
    else:
        print("✅ Container validation passed")
    
    print("✅ Service registration example completed!") 