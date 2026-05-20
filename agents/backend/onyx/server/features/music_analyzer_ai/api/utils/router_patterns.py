"""
Common router patterns and utilities
"""

from typing import Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)


def cache_service_in_endpoint(service_name: str):
    """
    Decorator to cache service instance within an endpoint
    
    Args:
        service_name: Name of the service to cache
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            # Cache service in router instance if not already cached
            cache_key = f"_cached_{service_name}"
            if not hasattr(self, cache_key):
                setattr(self, cache_key, self.get_service(service_name))
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def get_or_cache_service(router_instance: Any, service_name: str) -> Any:
    """
    Get service from cache or retrieve it
    
    Args:
        router_instance: Router instance
        service_name: Name of the service
    
    Returns:
        Service instance
    """
    cache_key = f"_cached_{service_name}"
    if hasattr(router_instance, cache_key):
        return getattr(router_instance, cache_key)
    
    service = router_instance.get_service(service_name)
    setattr(router_instance, cache_key, service)
    return service

