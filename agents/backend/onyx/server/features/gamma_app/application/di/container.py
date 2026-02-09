"""
Dependency Injection Container
Manages service dependencies and lifecycle
"""

import logging
from typing import Dict, Type, TypeVar, Callable, Optional, Any
from functools import lru_cache

from ...utils.config import get_settings
from ...core.content_generator import ContentGenerator
from ...services.collaboration_service import CollaborationService
from ...services.analytics_service import AnalyticsService
from ...services.cache_service import AdvancedCacheService
from ...infrastructure.database.session import DatabaseSessionManager, get_db_manager

logger = logging.getLogger(__name__)

T = TypeVar('T')

class DIContainer:
    """Dependency Injection Container"""
    
    def __init__(self):
        """Initialize DI container"""
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
        self._initialized = False
    
    def register_singleton(self, service_type: Type[T], instance: T):
        """Register a singleton instance"""
        self._singletons[service_type] = instance
        logger.debug(f"Registered singleton: {service_type.__name__}")
    
    def register_factory(self, service_type: Type[T], factory: Callable[[], T]):
        """Register a factory function"""
        self._factories[service_type] = factory
        logger.debug(f"Registered factory: {service_type.__name__}")
    
    def register_transient(self, service_type: Type[T], implementation: Type[T]):
        """Register a transient service (new instance each time)"""
        self._services[service_type] = implementation
        logger.debug(f"Registered transient: {service_type.__name__}")
    
    def get(self, service_type: Type[T]) -> T:
        """Get a service instance"""
        if service_type in self._singletons:
            return self._singletons[service_type]
        
        if service_type in self._factories:
            if service_type not in self._singletons:
                self._singletons[service_type] = self._factories[service_type]()
            return self._singletons[service_type]
        
        if service_type in self._services:
            implementation = self._services[service_type]
            return implementation()
        
        raise ValueError(f"Service {service_type.__name__} not registered")
    
    def initialize(self):
        """Initialize all registered services"""
        if self._initialized:
            return
        
        settings = get_settings()
        db_manager = get_db_manager()
        
        content_generator = ContentGenerator({
            'openai_api_key': settings.openai_api_key,
            'anthropic_api_key': settings.anthropic_api_key,
            'openai_model': settings.openai_model,
            'anthropic_model': settings.anthropic_model
        })
        
        collaboration_service = CollaborationService({
            'database_url': settings.database_url,
            'redis_url': settings.redis_url
        })
        
        analytics_service = AnalyticsService({
            'database_url': settings.database_url,
            'redis_url': settings.redis_url
        })
        
        cache_service = AdvancedCacheService({
            'redis_url': settings.redis_url
        })
        
        self.register_singleton(ContentGenerator, content_generator)
        self.register_singleton(CollaborationService, collaboration_service)
        self.register_singleton(AnalyticsService, analytics_service)
        self.register_singleton(AdvancedCacheService, cache_service)
        self.register_singleton(DatabaseSessionManager, db_manager)
        
        self._initialized = True
        logger.info("DI Container initialized")
    
    async def shutdown(self):
        """Shutdown all services"""
        for service in self._singletons.values():
            if hasattr(service, 'close'):
                try:
                    close_method = service.close
                    if hasattr(close_method, '__code__'):
                        import inspect
                        if inspect.iscoroutinefunction(close_method):
                            await close_method()
                        else:
                            close_method()
                    else:
                        close_method()
                except Exception as e:
                    logger.error(f"Error closing service {type(service).__name__}: {e}")
        
        self._singletons.clear()
        self._initialized = False
        logger.info("DI Container shut down")

_container: Optional[DIContainer] = None

@lru_cache()
def get_container() -> DIContainer:
    """Get the global DI container"""
    global _container
    if _container is None:
        _container = DIContainer()
        _container.initialize()
    return _container

def reset_container():
    """Reset the global container (mainly for testing)"""
    global _container
    _container = None
    get_container.cache_clear()

