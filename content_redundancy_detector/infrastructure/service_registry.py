"""
Service Registry - Centralized service lifecycle management
Implements dependency injection pattern for microservices
"""

import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from abc import ABC, abstractmethod

from ..core.config import get_settings
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class Service(ABC):
    """Base service interface"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the service"""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        pass


class ServiceRegistry:
    """
    Centralized service registry with dependency injection
    Manages service lifecycle following microservices principles
    """
    
    def __init__(self):
        self._services: Dict[str, Service] = {}
        self._initialized: bool = False
        self.settings = get_settings()
        logger.info("Service registry initialized")
    
    def register(self, name: str, service: Service) -> None:
        """Register a service"""
        if self._initialized:
            raise RuntimeError("Cannot register services after initialization")
        
        if name in self._services:
            raise ValueError(f"Service '{name}' already registered")
        
        self._services[name] = service
        logger.debug(f"Registered service: {name}")
    
    def get(self, name: str) -> Optional[Service]:
        """Get a service by name"""
        return self._services.get(name)
    
    def get_all(self) -> Dict[str, Service]:
        """Get all registered services"""
        return self._services.copy()
    
    async def initialize_all(self) -> None:
        """Initialize all registered services"""
        if self._initialized:
            logger.warning("Services already initialized")
            return
        
        logger.info(f"Initializing {len(self._services)} services...")
        
        initialized_count = 0
        for name, service in self._services.items():
            try:
                await service.initialize()
                initialized_count += 1
                logger.info(f"✅ Initialized: {name}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {name}: {e}", exc_info=True)
                raise
        
        self._initialized = True
        logger.info(f"✅ All {initialized_count} services initialized successfully")
    
    async def shutdown_all(self) -> None:
        """Shutdown all registered services"""
        if not self._initialized:
            return
        
        logger.info(f"Shutting down {len(self._services)} services...")
        
        shutdown_count = 0
        for name, service in reversed(list(self._services.items())):
            try:
                await service.shutdown()
                shutdown_count += 1
                logger.info(f"✅ Shutdown: {name}")
            except Exception as e:
                logger.error(f"❌ Error shutting down {name}: {e}", exc_info=True)
        
        self._initialized = False
        logger.info(f"✅ All {shutdown_count} services shut down")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all services"""
        status = {
            "initialized": self._initialized,
            "services": {}
        }
        
        for name, service in self._services.items():
            try:
                status["services"][name] = {
                    "healthy": service.is_healthy(),
                    "registered": True
                }
            except Exception as e:
                status["services"][name] = {
                    "healthy": False,
                    "registered": True,
                    "error": str(e)
                }
        
        return status


# Factory functions for lazy service creation
async def create_cache_service(settings) -> Service:
    """Factory for cache service"""
    from .cache import CacheService
    return CacheService(settings)


async def create_database_service(settings) -> Service:
    """Factory for database service"""
    from .database import DatabaseService
    return DatabaseService(settings)


async def create_ml_service(settings) -> Service:
    """Factory for ML service"""
    from .ml_service import MLService
    return MLService(settings)


# Service registration helpers
SERVICE_FACTORIES = {
    "cache": create_cache_service,
    "database": create_database_service,
    "ml": create_ml_service,
}






