"""
Cache Module
Independent caching module
"""

from typing import List
from modules.base_module import BaseModule
from infrastructure.cache import CacheServiceFactory
from core.service_container import get_container

logger = __import__("logging").getLogger(__name__)


class CacheModule(BaseModule):
    """Cache feature module"""
    
    def __init__(self):
        super().__init__("cache", "1.0.0")
        self._factory = None
    
    def get_dependencies(self) -> List[str]:
        """Cache module has no dependencies"""
        return []
    
    def _on_initialize(self) -> None:
        """Initialize cache module"""
        self._factory = CacheServiceFactory()
        
        # Register cache service in container
        container = get_container()
        container.register_service("cache", self._factory.create_cache_service())
        
        logger.info("Cache module initialized")
    
    def _on_shutdown(self) -> None:
        """Shutdown cache module"""
        # Cleanup connections if needed
        logger.info("Cache module shut down")
    
    def get_cache_service(self):
        """Get cache service instance"""
        if not self._factory:
            raise RuntimeError("Cache module not initialized")
        return self._factory.create_cache_service()















