"""
Cache middleware system.

Provides middleware pipeline for cache operations.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Callable, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheMiddleware:
    """
    Base cache middleware.
    
    Provides base class for middleware implementations.
    """
    
    def on_before_get(self, position: int, cache: Any) -> Optional[int]:
        """
        Called before get operation.
        
        Args:
            position: Cache position
            cache: Cache instance
            
        Returns:
            Modified position or None
        """
        return position
    
    def on_after_get(self, position: int, result: Any, cache: Any) -> Any:
        """
        Called after get operation.
        
        Args:
            position: Cache position
            result: Get result
            cache: Cache instance
            
        Returns:
            Modified result
        """
        return result
    
    def on_before_put(self, position: int, value: Any, cache: Any) -> tuple:
        """
        Called before put operation.
        
        Args:
            position: Cache position
            value: Value to put
            cache: Cache instance
            
        Returns:
            Modified (position, value) tuple
        """
        return position, value
    
    def on_after_put(self, position: int, value: Any, cache: Any) -> None:
        """
        Called after put operation.
        
        Args:
            position: Cache position
            value: Value that was put
            cache: Cache instance
        """
        pass
    
    def on_before_clear(self, cache: Any) -> None:
        """
        Called before clear operation.
        
        Args:
            cache: Cache instance
        """
        pass
    
    def on_after_clear(self, cache: Any) -> None:
        """
        Called after clear operation.
        
        Args:
            cache: Cache instance
        """
        pass


class MiddlewarePipeline:
    """
    Middleware pipeline.
    
    Manages and executes middleware chain.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize pipeline.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.middlewares: List[CacheMiddleware] = []
    
    def add_middleware(self, middleware: CacheMiddleware) -> None:
        """
        Add middleware to pipeline.
        
        Args:
            middleware: Middleware instance
        """
        self.middlewares.append(middleware)
        logger.info(f"Added middleware: {middleware.__class__.__name__}")
    
    def remove_middleware(self, middleware: CacheMiddleware) -> bool:
        """
        Remove middleware from pipeline.
        
        Args:
            middleware: Middleware instance
            
        Returns:
            True if removed
        """
        if middleware in self.middlewares:
            self.middlewares.remove(middleware)
            logger.info(f"Removed middleware: {middleware.__class__.__name__}")
            return True
        return False
    
    def execute_get(self, position: int) -> Optional[Any]:
        """
        Execute get with middleware.
        
        Args:
            position: Cache position
            
        Returns:
            Cached value or None
        """
        # Before hooks
        for middleware in self.middlewares:
            position = middleware.on_before_get(position, self.cache) or position
        
        # Execute get
        result = self.cache.get(position)
        
        # After hooks
        for middleware in reversed(self.middlewares):
            result = middleware.on_after_get(position, result, self.cache)
        
        return result
    
    def execute_put(self, position: int, value: Any) -> None:
        """
        Execute put with middleware.
        
        Args:
            position: Cache position
            value: Value to put
        """
        # Before hooks
        for middleware in self.middlewares:
            position, value = middleware.on_before_put(position, value, self.cache)
        
        # Execute put
        self.cache.put(position, value)
        
        # After hooks
        for middleware in reversed(self.middlewares):
            middleware.on_after_put(position, value, self.cache)
    
    def execute_clear(self) -> None:
        """Execute clear with middleware."""
        # Before hooks
        for middleware in self.middlewares:
            middleware.on_before_clear(self.cache)
        
        # Execute clear
        self.cache.clear()
        
        # After hooks
        for middleware in reversed(self.middlewares):
            middleware.on_after_clear(self.cache)


class LoggingMiddleware(CacheMiddleware):
    """Logging middleware."""
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize logging middleware.
        
        Args:
            log_level: Log level
        """
        self.log_level = log_level
    
    def on_after_get(self, position: int, result: Any, cache: Any) -> Any:
        """Log get operation."""
        if result is not None:
            logger.log(getattr(logging, self.log_level), f"Cache hit: {position}")
        else:
            logger.log(getattr(logging, self.log_level), f"Cache miss: {position}")
        return result
    
    def on_after_put(self, position: int, value: Any, cache: Any) -> None:
        """Log put operation."""
        logger.debug(f"Cache put: {position}")


class MetricsMiddleware(CacheMiddleware):
    """Metrics middleware."""
    
    def __init__(self):
        """Initialize metrics middleware."""
        self.hits = 0
        self.misses = 0
        self.puts = 0
    
    def on_after_get(self, position: int, result: Any, cache: Any) -> Any:
        """Track metrics on get."""
        if result is not None:
            self.hits += 1
        else:
            self.misses += 1
        return result
    
    def on_after_put(self, position: int, value: Any, cache: Any) -> None:
        """Track metrics on put."""
        self.puts += 1
    
    def get_metrics(self) -> Dict[str, int]:
        """
        Get collected metrics.
        
        Returns:
            Metrics dictionary
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "puts": self.puts,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.0
        }


class ValidationMiddleware(CacheMiddleware):
    """Validation middleware."""
    
    def on_before_put(self, position: int, value: Any, cache: Any) -> tuple:
        """Validate before put."""
        if position < 0:
            raise ValueError(f"Invalid position: {position}")
        if value is None:
            raise ValueError("Cannot put None value")
        return position, value


class TransformMiddleware(CacheMiddleware):
    """Transform middleware."""
    
    def __init__(self, transform_fn: Callable[[Any], Any]):
        """
        Initialize transform middleware.
        
        Args:
            transform_fn: Transformation function
        """
        self.transform_fn = transform_fn
    
    def on_after_get(self, position: int, result: Any, cache: Any) -> Any:
        """Transform result."""
        if result is not None:
            return self.transform_fn(result)
        return result
    
    def on_before_put(self, position: int, value: Any, cache: Any) -> tuple:
        """Transform value before put."""
        transformed = self.transform_fn(value)
        return position, transformed

