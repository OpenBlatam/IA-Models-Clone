"""
Service caching utilities for routers
"""

from typing import Dict, Any, Optional
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class ServiceCache:
    """Cache for service instances"""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    def get(self, service_name: str) -> Optional[Any]:
        """Get cached service"""
        return self._cache.get(service_name)
    
    def set(self, service_name: str, service: Any):
        """Cache a service"""
        self._cache[service_name] = service
    
    def clear(self):
        """Clear all cached services"""
        self._cache.clear()
    
    def has(self, service_name: str) -> bool:
        """Check if service is cached"""
        return service_name in self._cache


# Global service cache instance
_service_cache = ServiceCache()


def get_cached_service(service_name: str, getter_func: callable) -> Any:
    """
    Get a service with caching
    
    Args:
        service_name: Name of the service
        getter_func: Function to get the service if not cached
    
    Returns:
        Service instance
    """
    if _service_cache.has(service_name):
        return _service_cache.get(service_name)
    
    service = getter_func(service_name)
    _service_cache.set(service_name, service)
    return service


def clear_service_cache():
    """Clear the service cache"""
    _service_cache.clear()

