"""
Proxy Pattern Utilities for Piel Mejorador AI SAM3
==================================================

Unified proxy pattern implementation utilities.
"""

import logging
from typing import TypeVar, Callable, Any, Optional, Dict
from abc import ABC, abstractmethod
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class Proxy(ABC):
    """Base proxy interface."""
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Proxy call."""
        pass


class LazyProxy(Proxy):
    """Lazy loading proxy."""
    
    def __init__(
        self,
        factory: Callable[[], T],
        name: Optional[str] = None
    ):
        """
        Initialize lazy proxy.
        
        Args:
            factory: Factory function to create object
            name: Optional proxy name
        """
        self._factory = factory
        self._instance: Optional[T] = None
        self.name = name or factory.__name__
    
    def __call__(self, *args, **kwargs) -> T:
        """Get instance (lazy load)."""
        if self._instance is None:
            self._instance = self._factory()
            logger.debug(f"Lazy loaded instance: {self.name}")
        return self._instance
    
    @property
    def instance(self) -> T:
        """Get instance."""
        return self.__call__()
    
    def reset(self):
        """Reset instance (force reload)."""
        self._instance = None


class CachingProxy(Proxy):
    """Caching proxy."""
    
    def __init__(
        self,
        target: Callable[..., R],
        cache_size: int = 100,
        name: Optional[str] = None
    ):
        """
        Initialize caching proxy.
        
        Args:
            target: Target function
            cache_size: Maximum cache size
            name: Optional proxy name
        """
        self._target = target
        self._cache: Dict[tuple, R] = {}
        self._cache_size = cache_size
        self.name = name or target.__name__
    
    def __call__(self, *args, **kwargs) -> R:
        """Call with caching."""
        key = (args, tuple(sorted(kwargs.items())))
        
        if key in self._cache:
            logger.debug(f"Cache hit: {self.name}")
            return self._cache[key]
        
        result = self._target(*args, **kwargs)
        
        # Manage cache size
        if len(self._cache) >= self._cache_size:
            # Remove oldest (simple FIFO)
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        
        self._cache[key] = result
        logger.debug(f"Cache miss: {self.name}")
        return result
    
    def clear_cache(self):
        """Clear cache."""
        self._cache.clear()
    
    @property
    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class LoggingProxy(Proxy):
    """Logging proxy."""
    
    def __init__(
        self,
        target: Callable[..., R],
        name: Optional[str] = None,
        log_args: bool = True,
        log_result: bool = True
    ):
        """
        Initialize logging proxy.
        
        Args:
            target: Target function
            name: Optional proxy name
            log_args: Whether to log arguments
            log_result: Whether to log result
        """
        self._target = target
        self.name = name or target.__name__
        self._log_args = log_args
        self._log_result = log_result
    
    def __call__(self, *args, **kwargs) -> R:
        """Call with logging."""
        if self._log_args:
            logger.info(f"Calling {self.name} with args={args}, kwargs={kwargs}")
        
        try:
            result = self._target(*args, **kwargs)
            if self._log_result:
                logger.info(f"{self.name} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"{self.name} raised exception: {e}")
            raise


class ProxyUtils:
    """Unified proxy pattern utilities."""
    
    @staticmethod
    def create_lazy_proxy(
        factory: Callable[[], T],
        name: Optional[str] = None
    ) -> LazyProxy:
        """
        Create lazy loading proxy.
        
        Args:
            factory: Factory function
            name: Optional proxy name
            
        Returns:
            LazyProxy
        """
        return LazyProxy(factory, name)
    
    @staticmethod
    def create_caching_proxy(
        target: Callable[..., R],
        cache_size: int = 100,
        name: Optional[str] = None
    ) -> CachingProxy:
        """
        Create caching proxy.
        
        Args:
            target: Target function
            cache_size: Maximum cache size
            name: Optional proxy name
            
        Returns:
            CachingProxy
        """
        return CachingProxy(target, cache_size, name)
    
    @staticmethod
    def create_logging_proxy(
        target: Callable[..., R],
        name: Optional[str] = None,
        **kwargs
    ) -> LoggingProxy:
        """
        Create logging proxy.
        
        Args:
            target: Target function
            name: Optional proxy name
            **kwargs: Additional options
            
        Returns:
            LoggingProxy
        """
        return LoggingProxy(target, name, **kwargs)
    
    @staticmethod
    def proxy_decorator(
        proxy_type: str = "logging",
        **kwargs
    ):
        """
        Create proxy decorator.
        
        Args:
            proxy_type: Type of proxy ("lazy", "caching", "logging")
            **kwargs: Proxy-specific options
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable[..., R]) -> Proxy:
            if proxy_type == "lazy":
                return ProxyUtils.create_lazy_proxy(lambda: func, name=func.__name__)
            elif proxy_type == "caching":
                return ProxyUtils.create_caching_proxy(func, name=func.__name__, **kwargs)
            elif proxy_type == "logging":
                return ProxyUtils.create_logging_proxy(func, name=func.__name__, **kwargs)
            else:
                raise ValueError(f"Unknown proxy type: {proxy_type}")
        
        return decorator


# Convenience functions
def create_lazy_proxy(factory: Callable[[], T], **kwargs) -> LazyProxy:
    """Create lazy proxy."""
    return ProxyUtils.create_lazy_proxy(factory, **kwargs)


def create_caching_proxy(target: Callable[..., R], **kwargs) -> CachingProxy:
    """Create caching proxy."""
    return ProxyUtils.create_caching_proxy(target, **kwargs)


def create_logging_proxy(target: Callable[..., R], **kwargs) -> LoggingProxy:
    """Create logging proxy."""
    return ProxyUtils.create_logging_proxy(target, **kwargs)




