"""
Service Container for Dependency Injection
Provides centralized service management and dependency injection
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar
from functools import lru_cache

from core.interfaces import (
    IStorageService, ICacheService, IFileStorageService,
    IMessageQueueService, INotificationService, IMetricsService,
    ITracingService, IAuthenticationService, IServiceFactory
)
from infrastructure.storage import StorageServiceFactory
from infrastructure.cache import CacheServiceFactory
from infrastructure.messaging import MessagingServiceFactory
from infrastructure.observability import ObservabilityServiceFactory
from infrastructure.security import SecurityServiceFactory

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceContainer:
    """
    Service container for dependency injection
    
    Manages service instances and provides factory methods
    """
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, IServiceFactory] = {}
        self._setup_factories()
    
    def _setup_factories(self) -> None:
        """Setup service factories"""
        self._factories = {
            "storage": StorageServiceFactory(),
            "cache": CacheServiceFactory(),
            "messaging": MessagingServiceFactory(),
            "observability": ObservabilityServiceFactory(),
            "security": SecurityServiceFactory()
        }
    
    def register_service(self, service_name: str, service_instance: Any) -> None:
        """Register a service instance"""
        self._services[service_name] = service_instance
        logger.info(f"Registered service: {service_name}")
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """Get service instance by name"""
        return self._services.get(service_name)
    
    def get_storage_service(self) -> IStorageService:
        """Get storage service instance"""
        if "storage" not in self._services:
            factory = self._factories["storage"]
            self._services["storage"] = factory.create_storage_service()
        return self._services["storage"]
    
    def get_cache_service(self) -> ICacheService:
        """Get cache service instance"""
        if "cache" not in self._services:
            factory = self._factories["cache"]
            self._services["cache"] = factory.create_cache_service()
        return self._services["cache"]
    
    def get_file_storage_service(self) -> IFileStorageService:
        """Get file storage service instance"""
        if "file_storage" not in self._services:
            # Implementation would go here
            pass
        return self._services.get("file_storage")
    
    def get_message_queue_service(self) -> IMessageQueueService:
        """Get message queue service instance"""
        if "message_queue" not in self._services:
            factory = self._factories["messaging"]
            self._services["message_queue"] = factory.create_message_queue_service()
        return self._services["message_queue"]
    
    def get_notification_service(self) -> INotificationService:
        """Get notification service instance"""
        if "notification" not in self._services:
            factory = self._factories["messaging"]
            self._services["notification"] = factory.create_notification_service()
        return self._services["notification"]
    
    def get_metrics_service(self) -> IMetricsService:
        """Get metrics service instance"""
        if "metrics" not in self._services:
            factory = self._factories["observability"]
            self._services["metrics"] = factory.create_metrics_service()
        return self._services["metrics"]
    
    def get_tracing_service(self) -> ITracingService:
        """Get tracing service instance"""
        if "tracing" not in self._services:
            factory = self._factories["observability"]
            self._services["tracing"] = factory.create_tracing_service()
        return self._services["tracing"]
    
    def get_authentication_service(self) -> IAuthenticationService:
        """Get authentication service instance"""
        if "authentication" not in self._services:
            factory = self._factories["security"]
            self._services["authentication"] = factory.create_authentication_service()
        return self._services["authentication"]
    
    def reset(self) -> None:
        """Reset all services (useful for testing)"""
        self._services.clear()
        logger.info("Service container reset")


# Global service container instance
_container: Optional[ServiceContainer] = None


@lru_cache()
def get_container() -> ServiceContainer:
    """Get global service container instance (singleton)"""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


# Convenience functions for FastAPI dependency injection
def get_storage_service() -> IStorageService:
    """FastAPI dependency for storage service"""
    return get_container().get_storage_service()


def get_cache_service() -> ICacheService:
    """FastAPI dependency for cache service"""
    return get_container().get_cache_service()


def get_message_queue_service() -> IMessageQueueService:
    """FastAPI dependency for message queue service"""
    return get_container().get_message_queue_service()


def get_notification_service() -> INotificationService:
    """FastAPI dependency for notification service"""
    return get_container().get_notification_service()


def get_metrics_service() -> IMetricsService:
    """FastAPI dependency for metrics service"""
    return get_container().get_metrics_service()


def get_tracing_service() -> ITracingService:
    """FastAPI dependency for tracing service"""
    return get_container().get_tracing_service()


def get_authentication_service() -> IAuthenticationService:
    """FastAPI dependency for authentication service"""
    return get_container().get_authentication_service()















