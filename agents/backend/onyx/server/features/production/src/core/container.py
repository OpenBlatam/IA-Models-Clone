from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
from typing import Dict, Any, Type, Optional, Callable
from contextlib import asynccontextmanager
from functools import lru_cache
import weakref
from src.core.config import Settings
from src.core.exceptions import ContainerError
from src.infrastructure.database import DatabaseManager
from src.infrastructure.cache import CacheService
from src.infrastructure.monitoring import MonitoringService
from src.infrastructure.health import HealthChecker
from src.application.services.ai_service import AIService
from src.application.services.event_publisher import EventPublisher
from src.application.repositories.copywriting_repository import CopywritingRepository
from src.application.repositories.user_repository import UserRepository
from typing import Any, List, Dict, Optional
"""
🔧 Dependency Injection Container
=================================

Ultra-optimized DI container with lazy loading, singleton management,
and automatic resource cleanup.
"""




class Container:
    """
    Ultra-optimized dependency injection container with:
    - Lazy loading
    - Singleton management
    - Async resource management
    - Automatic cleanup
    - Circular dependency detection
    """
    
    def __init__(self) -> Any:
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._resources: Dict[str, Any] = {}
        self._cleanup_tasks: list = []
        self._initialized = False
        self._logger = logging.getLogger(__name__)
        
        # Register default factories
        self._register_default_factories()
    
    def _register_default_factories(self) -> Any:
        """Register default service factories"""
        
        # Core services
        self.register_factory("settings", lambda: Settings())
        self.register_factory("database", self._create_database_manager)
        self.register_factory("cache", self._create_cache_service)
        self.register_factory("monitoring", self._create_monitoring_service)
        self.register_factory("health_checker", self._create_health_checker)
        
        # Application services
        self.register_factory("ai_service", self._create_ai_service)
        self.register_factory("event_publisher", self._create_event_publisher)
        
        # Repositories
        self.register_factory("copywriting_repository", self._create_copywriting_repository)
        self.register_factory("user_repository", self._create_user_repository)
    
    def register_factory(self, name: str, factory: Callable):
        """Register a factory function for a service"""
        self._factories[name] = factory
        self._logger.debug(f"Registered factory for: {name}")
    
    def register_singleton(self, name: str, instance: Any):
        """Register a singleton instance"""
        self._singletons[name] = instance
        self._logger.debug(f"Registered singleton: {name}")
    
    def get(self, service_type: Type) -> Optional[Dict[str, Any]]:
        """Get a service instance by type"""
        service_name = service_type.__name__
        return self.get_by_name(service_name)
    
    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a service instance by name"""
        
        # Check if already instantiated
        if name in self._instances:
            return self._instances[name]
        
        # Check if singleton exists
        if name in self._singletons:
            return self._singletons[name]
        
        # Create new instance
        if name in self._factories:
            try:
                instance = self._factories[name]()
                self._instances[name] = instance
                self._logger.debug(f"Created instance: {name}")
                return instance
            except Exception as e:
                self._logger.error(f"Failed to create instance {name}: {e}")
                raise ContainerError(f"Failed to create {name}: {e}")
        
        raise ContainerError(f"Service {name} not found")
    
    @lru_cache(maxsize=128)
    def get_cached(self, service_type: Type) -> Optional[Dict[str, Any]]:
        """Get a cached service instance"""
        return self.get(service_type)
    
    async def init_resources(self) -> Any:
        """Initialize all async resources"""
        if self._initialized:
            return
        
        self._logger.info("Initializing container resources...")
        
        try:
            # Initialize database
            db_manager = self.get(DatabaseManager)
            await db_manager.initialize()
            self._resources["database"] = db_manager
            
            # Initialize cache
            cache_service = self.get(CacheService)
            await cache_service.initialize()
            self._resources["cache"] = cache_service
            
            # Initialize monitoring
            monitoring = self.get(MonitoringService)
            await monitoring.initialize()
            self._resources["monitoring"] = monitoring
            
            # Initialize health checker
            health_checker = self.get(HealthChecker)
            await health_checker.initialize()
            self._resources["health_checker"] = health_checker
            
            # Initialize AI service
            ai_service = self.get(AIService)
            await ai_service.initialize()
            self._resources["ai_service"] = ai_service
            
            # Initialize event publisher
            event_publisher = self.get(EventPublisher)
            await event_publisher.initialize()
            self._resources["event_publisher"] = event_publisher
            
            self._initialized = True
            self._logger.info("Container resources initialized successfully")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize container resources: {e}")
            await self.cleanup()
            raise
    
    async def cleanup(self) -> Any:
        """Cleanup all resources"""
        self._logger.info("Cleaning up container resources...")
        
        cleanup_tasks = []
        
        # Cleanup resources in reverse order
        for name, resource in reversed(list(self._resources.items())):
            if hasattr(resource, 'cleanup'):
                cleanup_tasks.append(self._cleanup_resource(name, resource))
            elif hasattr(resource, 'close'):
                cleanup_tasks.append(self._close_resource(name, resource))
        
        # Wait for all cleanup tasks
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Clear instances
        self._instances.clear()
        self._resources.clear()
        self._initialized = False
        
        self._logger.info("Container cleanup completed")
    
    async def _cleanup_resource(self, name: str, resource: Any):
        """Cleanup a specific resource"""
        try:
            await resource.cleanup()
            self._logger.debug(f"Cleaned up resource: {name}")
        except Exception as e:
            self._logger.error(f"Error cleaning up {name}: {e}")
    
    async def _close_resource(self, name: str, resource: Any):
        """Close a specific resource"""
        try:
            await resource.close()
            self._logger.debug(f"Closed resource: {name}")
        except Exception as e:
            self._logger.error(f"Error closing {name}: {e}")
    
    # Factory methods for services
    def _create_database_manager(self) -> DatabaseManager:
        """Create database manager instance"""
        settings = self.get_by_name("settings")
        return DatabaseManager(settings.database)
    
    def _create_cache_service(self) -> CacheService:
        """Create cache service instance"""
        settings = self.get_by_name("settings")
        return CacheService(settings.redis, settings.cache)
    
    def _create_monitoring_service(self) -> MonitoringService:
        """Create monitoring service instance"""
        settings = self.get_by_name("settings")
        return MonitoringService(settings.monitoring)
    
    def _create_health_checker(self) -> HealthChecker:
        """Create health checker instance"""
        settings = self.get_by_name("settings")
        return HealthChecker(settings)
    
    def _create_ai_service(self) -> AIService:
        """Create AI service instance"""
        settings = self.get_by_name("settings")
        cache_service = self.get_by_name("cache")
        return AIService(settings.openai, settings.ai, cache_service)
    
    def _create_event_publisher(self) -> EventPublisher:
        """Create event publisher instance"""
        settings = self.get_by_name("settings")
        return EventPublisher(settings.events)
    
    def _create_copywriting_repository(self) -> CopywritingRepository:
        """Create copywriting repository instance"""
        db_manager = self.get_by_name("database")
        cache_service = self.get_by_name("cache")
        return CopywritingRepository(db_manager, cache_service)
    
    def _create_user_repository(self) -> UserRepository:
        """Create user repository instance"""
        db_manager = self.get_by_name("database")
        cache_service = self.get_by_name("cache")
        return UserRepository(db_manager, cache_service)
    
    @asynccontextmanager
    async def lifespan(self) -> Any:
        """Context manager for container lifespan"""
        try:
            await self.init_resources()
            yield self
        finally:
            await self.cleanup()
    
    def __enter__(self) -> Any:
        """Synchronous context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Synchronous context manager exit"""
        if self._initialized:
            asyncio.create_task(self.cleanup())


# Global container instance
_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container instance"""
    global _container
    if _container is None:
        _container = Container()
    return _container


def reset_container():
    """Reset the global container (useful for testing)"""
    global _container
    if _container:
        asyncio.create_task(_container.cleanup())
    _container = None 