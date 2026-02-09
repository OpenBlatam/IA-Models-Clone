"""
Middleware System
===============

Ultra-modular middleware system with advanced patterns.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, Type, List, Callable, Union, get_type_hints
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import inspect
import functools
from collections import deque

logger = logging.getLogger(__name__)

class MiddlewareType(str, Enum):
    """Middleware types."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    LOGGING = "logging"
    METRICS = "metrics"
    CACHING = "caching"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MONITORING = "monitoring"

class MiddlewareScope(str, Enum):
    """Middleware scopes."""
    GLOBAL = "global"
    ROUTE = "route"
    BLUEPRINT = "blueprint"
    NAMESPACE = "namespace"

@dataclass
class MiddlewareInfo:
    """Middleware information."""
    name: str
    middleware: Callable
    middleware_type: MiddlewareType
    scope: MiddlewareScope
    priority: int
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_count: int = 0
    total_time: float = 0.0
    error_count: int = 0

class BaseMiddleware(ABC):
    """Base middleware class."""
    
    @abstractmethod
    def process_request(self, request: Any) -> Any:
        """Process request."""
        pass
    
    @abstractmethod
    def process_response(self, request: Any, response: Any) -> Any:
        """Process response."""
        pass
    
    def process_error(self, request: Any, error: Exception) -> Any:
        """Process error."""
        return None

class RequestMiddleware:
    """Request middleware."""
    
    def __init__(self, func: Callable, name: str = None, priority: int = 0):
        self.func = func
        self.name = name or func.__name__
        self.priority = priority
        self.execution_count = 0
        self.total_time = 0.0
        self.error_count = 0
    
    def __call__(self, request: Any) -> Any:
        """Execute middleware."""
        try:
            start_time = time.time()
            self.execution_count += 1
            
            result = self.func(request)
            
            self.total_time += time.time() - start_time
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Middleware {self.name} failed: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get middleware statistics."""
        return {
            'name': self.name,
            'execution_count': self.execution_count,
            'total_time': self.total_time,
            'average_time': self.total_time / max(self.execution_count, 1),
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.execution_count, 1)
        }

class ResponseMiddleware:
    """Response middleware."""
    
    def __init__(self, func: Callable, name: str = None, priority: int = 0):
        self.func = func
        self.name = name or func.__name__
        self.priority = priority
        self.execution_count = 0
        self.total_time = 0.0
        self.error_count = 0
    
    def __call__(self, request: Any, response: Any) -> Any:
        """Execute middleware."""
        try:
            start_time = time.time()
            self.execution_count += 1
            
            result = self.func(request, response)
            
            self.total_time += time.time() - start_time
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Middleware {self.name} failed: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get middleware statistics."""
        return {
            'name': self.name,
            'execution_count': self.execution_count,
            'total_time': self.total_time,
            'average_time': self.total_time / max(self.execution_count, 1),
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.execution_count, 1)
        }

class MiddlewareManager:
    """Middleware manager."""
    
    def __init__(self):
        self._request_middleware: List[RequestMiddleware] = []
        self._response_middleware: List[ResponseMiddleware] = []
        self._error_middleware: List[Callable] = []
        self._lock = threading.RLock()
        self._enabled: bool = True
        self._stats: Dict[str, Any] = {}
    
    def add_request_middleware(self, func: Callable, name: str = None, priority: int = 0) -> None:
        """Add request middleware."""
        try:
            with self._lock:
                middleware = RequestMiddleware(func, name, priority)
                self._request_middleware.append(middleware)
                # Sort by priority
                self._request_middleware.sort(key=lambda m: m.priority, reverse=True)
                logger.info(f"Request middleware {middleware.name} added")
        except Exception as e:
            logger.error(f"Failed to add request middleware: {str(e)}")
            raise
    
    def add_response_middleware(self, func: Callable, name: str = None, priority: int = 0) -> None:
        """Add response middleware."""
        try:
            with self._lock:
                middleware = ResponseMiddleware(func, name, priority)
                self._response_middleware.append(middleware)
                # Sort by priority
                self._response_middleware.sort(key=lambda m: m.priority, reverse=True)
                logger.info(f"Response middleware {middleware.name} added")
        except Exception as e:
            logger.error(f"Failed to add response middleware: {str(e)}")
            raise
    
    def add_error_middleware(self, func: Callable, name: str = None, priority: int = 0) -> None:
        """Add error middleware."""
        try:
            with self._lock:
                self._error_middleware.append(func)
                logger.info(f"Error middleware {name or func.__name__} added")
        except Exception as e:
            logger.error(f"Failed to add error middleware: {str(e)}")
            raise
    
    def remove_request_middleware(self, name: str) -> None:
        """Remove request middleware."""
        try:
            with self._lock:
                self._request_middleware = [m for m in self._request_middleware if m.name != name]
                logger.info(f"Request middleware {name} removed")
        except Exception as e:
            logger.error(f"Failed to remove request middleware {name}: {str(e)}")
            raise
    
    def remove_response_middleware(self, name: str) -> None:
        """Remove response middleware."""
        try:
            with self._lock:
                self._response_middleware = [m for m in self._response_middleware if m.name != name]
                logger.info(f"Response middleware {name} removed")
        except Exception as e:
            logger.error(f"Failed to remove response middleware {name}: {str(e)}")
            raise
    
    def remove_error_middleware(self, func: Callable) -> None:
        """Remove error middleware."""
        try:
            with self._lock:
                if func in self._error_middleware:
                    self._error_middleware.remove(func)
                    logger.info(f"Error middleware {func.__name__} removed")
        except Exception as e:
            logger.error(f"Failed to remove error middleware: {str(e)}")
            raise
    
    def process_request(self, request: Any) -> Any:
        """Process request through middleware chain."""
        try:
            if not self._enabled:
                return request
            
            with self._lock:
                for middleware in self._request_middleware:
                    try:
                        request = middleware(request)
                    except Exception as e:
                        logger.error(f"Request middleware {middleware.name} failed: {str(e)}")
                        # Continue with next middleware
                
                return request
        except Exception as e:
            logger.error(f"Failed to process request: {str(e)}")
            return request
    
    def process_response(self, request: Any, response: Any) -> Any:
        """Process response through middleware chain."""
        try:
            if not self._enabled:
                return response
            
            with self._lock:
                for middleware in self._response_middleware:
                    try:
                        response = middleware(request, response)
                    except Exception as e:
                        logger.error(f"Response middleware {middleware.name} failed: {str(e)}")
                        # Continue with next middleware
                
                return response
        except Exception as e:
            logger.error(f"Failed to process response: {str(e)}")
            return response
    
    def process_error(self, request: Any, error: Exception) -> Any:
        """Process error through middleware chain."""
        try:
            if not self._enabled:
                return None
            
            with self._lock:
                for middleware in self._error_middleware:
                    try:
                        result = middleware(request, error)
                        if result is not None:
                            return result
                    except Exception as e:
                        logger.error(f"Error middleware {middleware.__name__} failed: {str(e)}")
                        # Continue with next middleware
                
                return None
        except Exception as e:
            logger.error(f"Failed to process error: {str(e)}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get middleware statistics."""
        try:
            with self._lock:
                request_stats = [m.get_stats() for m in self._request_middleware]
                response_stats = [m.get_stats() for m in self._response_middleware]
                
                return {
                    'enabled': self._enabled,
                    'request_middleware_count': len(self._request_middleware),
                    'response_middleware_count': len(self._response_middleware),
                    'error_middleware_count': len(self._error_middleware),
                    'request_middleware_stats': request_stats,
                    'response_middleware_stats': response_stats
                }
        except Exception as e:
            logger.error(f"Failed to get middleware stats: {str(e)}")
            return {}
    
    def enable(self) -> None:
        """Enable middleware."""
        try:
            with self._lock:
                self._enabled = True
                logger.info("Middleware enabled")
        except Exception as e:
            logger.error(f"Failed to enable middleware: {str(e)}")
            raise
    
    def disable(self) -> None:
        """Disable middleware."""
        try:
            with self._lock:
                self._enabled = False
                logger.info("Middleware disabled")
        except Exception as e:
            logger.error(f"Failed to disable middleware: {str(e)}")
            raise
    
    def clear(self) -> None:
        """Clear all middleware."""
        try:
            with self._lock:
                self._request_middleware.clear()
                self._response_middleware.clear()
                self._error_middleware.clear()
                logger.info("All middleware cleared")
        except Exception as e:
            logger.error(f"Failed to clear middleware: {str(e)}")
            raise

# Global middleware manager
middleware_manager = MiddlewareManager()

# Middleware decorators
def request_middleware(name: str = None, priority: int = 0):
    """Decorator to register request middleware."""
    def decorator(func):
        middleware_manager.add_request_middleware(func, name, priority)
        return func
    return decorator

def response_middleware(name: str = None, priority: int = 0):
    """Decorator to register response middleware."""
    def decorator(func):
        middleware_manager.add_response_middleware(func, name, priority)
        return func
    return decorator

def error_middleware(name: str = None, priority: int = 0):
    """Decorator to register error middleware."""
    def decorator(func):
        middleware_manager.add_error_middleware(func, name, priority)
        return func
    return decorator

# Built-in middleware
@request_middleware("request_logging", priority=100)
def request_logging_middleware(request):
    """Log request details."""
    try:
        logger.info(f"Request: {request.method} {request.path}")
        return request
    except Exception as e:
        logger.error(f"Request logging middleware failed: {str(e)}")
        return request

@response_middleware("response_logging", priority=100)
def response_logging_middleware(request, response):
    """Log response details."""
    try:
        logger.info(f"Response: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Response logging middleware failed: {str(e)}")
        return response

@request_middleware("performance_monitoring", priority=90)
def performance_monitoring_middleware(request):
    """Monitor request performance."""
    try:
        request.start_time = time.time()
        return request
    except Exception as e:
        logger.error(f"Performance monitoring middleware failed: {str(e)}")
        return request

@response_middleware("performance_monitoring", priority=90)
def performance_monitoring_response_middleware(request, response):
    """Monitor response performance."""
    try:
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            logger.info(f"Request duration: {duration:.3f}s")
        return response
    except Exception as e:
        logger.error(f"Performance monitoring response middleware failed: {str(e)}")
        return response

@error_middleware("error_logging")
def error_logging_middleware(request, error):
    """Log errors."""
    try:
        logger.error(f"Error in request: {str(error)}")
        return None
    except Exception as e:
        logger.error(f"Error logging middleware failed: {str(e)}")
        return None









