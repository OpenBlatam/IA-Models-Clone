"""
Middleware System
Middleware pattern for cross-cutting concerns.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class Middleware(ABC):
    """Base middleware class."""
    
    @abstractmethod
    def process(self, request: Any, next_handler: Callable) -> Any:
        """Process the request."""
        pass


class LoggingMiddleware(Middleware):
    """Middleware for logging requests."""
    
    def process(self, request: Any, next_handler: Callable) -> Any:
        logger.info(f"Processing request: {type(request).__name__}")
        try:
            response = next_handler(request)
            logger.info(f"Request completed successfully")
            return response
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise


class ValidationMiddleware(Middleware):
    """Middleware for request validation."""
    
    def __init__(self, validator: Callable):
        self.validator = validator
    
    def process(self, request: Any, next_handler: Callable) -> Any:
        if not self.validator(request):
            raise ValueError("Request validation failed")
        return next_handler(request)


class TimingMiddleware(Middleware):
    """Middleware for timing requests."""
    
    def process(self, request: Any, next_handler: Callable) -> Any:
        import time
        start_time = time.time()
        response = next_handler(request)
        duration = time.time() - start_time
        logger.info(f"Request took {duration:.4f} seconds")
        return response


class CachingMiddleware(Middleware):
    """Middleware for caching responses."""
    
    def __init__(self, cache: Dict[str, Any], key_func: Callable):
        self.cache = cache
        self.key_func = key_func
    
    def process(self, request: Any, next_handler: Callable) -> Any:
        cache_key = self.key_func(request)
        if cache_key in self.cache:
            logger.info("Cache hit")
            return self.cache[cache_key]
        
        response = next_handler(request)
        self.cache[cache_key] = response
        return response


class MiddlewarePipeline:
    """Pipeline for chaining middleware."""
    
    def __init__(self):
        self.middleware_stack: List[Middleware] = []
    
    def add(self, middleware: Middleware):
        """Add middleware to the pipeline."""
        self.middleware_stack.append(middleware)
        return self
    
    def process(self, request: Any, handler: Callable) -> Any:
        """Process request through middleware pipeline."""
        def create_handler(index: int) -> Callable:
            if index >= len(self.middleware_stack):
                return handler
            
            current_middleware = self.middleware_stack[index]
            next_handler = create_handler(index + 1)
            
            return lambda req: current_middleware.process(req, next_handler)
        
        return create_handler(0)(request)


class MiddlewareManager:
    """Manager for middleware configuration."""
    
    def __init__(self):
        self.pipelines: Dict[str, MiddlewarePipeline] = {}
    
    def create_pipeline(self, name: str) -> MiddlewarePipeline:
        """Create a named middleware pipeline."""
        pipeline = MiddlewarePipeline()
        self.pipelines[name] = pipeline
        return pipeline
    
    def get_pipeline(self, name: str) -> Optional[MiddlewarePipeline]:
        """Get a middleware pipeline by name."""
        return self.pipelines.get(name)
    
    def add_to_pipeline(self, name: str, middleware: Middleware):
        """Add middleware to a pipeline."""
        if name not in self.pipelines:
            self.create_pipeline(name)
        self.pipelines[name].add(middleware)



