"""
Base Middleware Implementation

Ultra-specialized base middleware with advanced features for
request/response processing and cross-cutting concerns.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import weakref
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class MiddlewareType(Enum):
    """Middleware type enumeration"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    LOGGING = "logging"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CACHING = "caching"
    VALIDATION = "validation"
    MONITORING = "monitoring"
    ERROR_HANDLING = "error_handling"
    COMPRESSION = "compression"
    CORS = "cors"


class MiddlewarePriority(Enum):
    """Middleware priority enumeration"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MiddlewareConfig:
    """Middleware configuration"""
    name: str
    middleware_type: MiddlewareType
    priority: MiddlewarePriority = MiddlewarePriority.NORMAL
    enabled: bool = True
    timeout: Optional[float] = None
    retry_count: int = 0
    cache_ttl: Optional[float] = None
    rate_limit: Optional[int] = None
    rate_window: Optional[float] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestContext:
    """Request context for middleware"""
    request_id: str
    method: str
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, Any]
    path_params: Dict[str, Any]
    body: Any
    start_time: datetime
    client_ip: str
    user_agent: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseContext:
    """Response context for middleware"""
    status_code: int
    headers: Dict[str, str]
    body: Any
    end_time: datetime
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMiddleware(ABC):
    """Base middleware with advanced features"""
    
    def __init__(self, config: MiddlewareConfig):
        self.config = config
        self._enabled = config.enabled
        self._request_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_duration = 0.0
        self._callbacks: List[Callable] = []
        self._error_handlers: List[Callable] = []
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def process_request(self, request: RequestContext) -> Optional[ResponseContext]:
        """Process incoming request (override in subclasses)"""
        pass
    
    @abstractmethod
    async def process_response(self, request: RequestContext, response: ResponseContext) -> ResponseContext:
        """Process outgoing response (override in subclasses)"""
        pass
    
    async def __call__(self, request: RequestContext, next_middleware: Callable) -> ResponseContext:
        """Execute middleware in chain"""
        if not self._enabled:
            return await next_middleware(request)
        
        start_time = time.time()
        
        try:
            # Process request
            early_response = await self.process_request(request)
            if early_response:
                # Middleware handled request, return early response
                self._update_metrics(True, start_time)
                return early_response
            
            # Continue to next middleware
            response = await next_middleware(request)
            
            # Process response
            processed_response = await self.process_response(request, response)
            
            self._update_metrics(True, start_time)
            return processed_response
            
        except Exception as e:
            self._update_metrics(False, start_time)
            await self._handle_error(request, e)
            raise
    
    def _update_metrics(self, success: bool, start_time: float) -> None:
        """Update middleware metrics"""
        duration = time.time() - start_time
        
        self._request_count += 1
        if success:
            self._success_count += 1
        else:
            self._error_count += 1
        
        self._total_duration += duration
    
    async def _handle_error(self, request: RequestContext, error: Exception) -> None:
        """Handle middleware errors"""
        # Call error handlers
        for handler in self._error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(request, error)
                else:
                    handler(request, error)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
    
    def add_callback(self, callback: Callable) -> None:
        """Add callback for events"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def add_error_handler(self, handler: Callable) -> None:
        """Add error handler"""
        self._error_handlers.append(handler)
    
    def remove_error_handler(self, handler: Callable) -> None:
        """Remove error handler"""
        if handler in self._error_handlers:
            self._error_handlers.remove(handler)
    
    def enable(self) -> None:
        """Enable middleware"""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable middleware"""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if middleware is enabled"""
        return self._enabled
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get middleware metrics"""
        avg_duration = (
            self._total_duration / self._request_count
            if self._request_count > 0 else 0
        )
        
        success_rate = (
            self._success_count / self._request_count
            if self._request_count > 0 else 0
        )
        
        return {
            "name": self.config.name,
            "type": self.config.middleware_type.value,
            "priority": self.config.priority.value,
            "enabled": self._enabled,
            "request_count": self._request_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate": success_rate,
            "total_duration": self._total_duration,
            "average_duration": avg_duration
        }
    
    def reset_metrics(self) -> None:
        """Reset middleware metrics"""
        self._request_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_duration = 0.0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}', enabled={self._enabled})"


class MiddlewareChain:
    """Chain of middleware with priority ordering"""
    
    def __init__(self):
        self._middleware: List[BaseMiddleware] = []
        self._lock = asyncio.Lock()
    
    def add_middleware(self, middleware: BaseMiddleware) -> None:
        """Add middleware to chain"""
        self._middleware.append(middleware)
        # Sort by priority (higher priority first)
        self._middleware.sort(
            key=lambda m: m.config.priority.value,
            reverse=True
        )
    
    def remove_middleware(self, name: str) -> None:
        """Remove middleware from chain"""
        self._middleware = [
            m for m in self._middleware
            if m.config.name != name
        ]
    
    async def process_request(self, request: RequestContext) -> ResponseContext:
        """Process request through middleware chain"""
        if not self._middleware:
            # No middleware, return default response
            return ResponseContext(
                status_code=200,
                headers={},
                body=None,
                end_time=datetime.utcnow(),
                duration=0.0
            )
        
        # Create middleware chain
        async def middleware_chain(req: RequestContext, index: int = 0) -> ResponseContext:
            if index >= len(self._middleware):
                # End of chain, return default response
                return ResponseContext(
                    status_code=200,
                    headers={},
                    body=None,
                    end_time=datetime.utcnow(),
                    duration=0.0
                )
            
            middleware = self._middleware[index]
            return await middleware(req, lambda r: middleware_chain(r, index + 1))
        
        return await middleware_chain(request)
    
    def get_middleware(self) -> List[BaseMiddleware]:
        """Get all middleware"""
        return self._middleware.copy()
    
    def get_middleware_by_type(self, middleware_type: MiddlewareType) -> List[BaseMiddleware]:
        """Get middleware by type"""
        return [
            m for m in self._middleware
            if m.config.middleware_type == middleware_type
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all middleware"""
        return {
            middleware.config.name: middleware.get_metrics()
            for middleware in self._middleware
        }


class MiddlewareRegistry:
    """Registry for managing middleware"""
    
    def __init__(self):
        self._middleware: Dict[str, BaseMiddleware] = {}
        self._chains: Dict[str, MiddlewareChain] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, middleware: BaseMiddleware) -> None:
        """Register middleware"""
        async with self._lock:
            self._middleware[middleware.config.name] = middleware
            logger.info(f"Registered middleware: {middleware.config.name}")
    
    async def unregister(self, name: str) -> None:
        """Unregister middleware"""
        async with self._lock:
            if name in self._middleware:
                del self._middleware[name]
                logger.info(f"Unregistered middleware: {name}")
    
    def get(self, name: str) -> Optional[BaseMiddleware]:
        """Get middleware by name"""
        return self._middleware.get(name)
    
    def get_by_type(self, middleware_type: MiddlewareType) -> List[BaseMiddleware]:
        """Get middleware by type"""
        return [
            middleware for middleware in self._middleware.values()
            if middleware.config.middleware_type == middleware_type
        ]
    
    def create_chain(self, name: str, middleware_names: List[str]) -> MiddlewareChain:
        """Create middleware chain"""
        chain = MiddlewareChain()
        
        for middleware_name in middleware_names:
            middleware = self.get(middleware_name)
            if middleware:
                chain.add_middleware(middleware)
        
        self._chains[name] = chain
        return chain
    
    def get_chain(self, name: str) -> Optional[MiddlewareChain]:
        """Get middleware chain"""
        return self._chains.get(name)
    
    def list_all(self) -> List[BaseMiddleware]:
        """List all middleware"""
        return list(self._middleware.values())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all middleware"""
        return {
            name: middleware.get_metrics()
            for name, middleware in self._middleware.items()
        }


# Global middleware registry
middleware_registry = MiddlewareRegistry()


# Convenience functions
async def register_middleware(middleware: BaseMiddleware):
    """Register middleware"""
    await middleware_registry.register(middleware)


def get_middleware(name: str) -> Optional[BaseMiddleware]:
    """Get middleware by name"""
    return middleware_registry.get(name)


def create_middleware_chain(name: str, middleware_names: List[str]) -> MiddlewareChain:
    """Create middleware chain"""
    return middleware_registry.create_chain(name, middleware_names)


# Middleware factory functions
def create_middleware(middleware_type: MiddlewareType, name: str, **kwargs) -> BaseMiddleware:
    """Create middleware by type"""
    config = MiddlewareConfig(
        name=name,
        middleware_type=middleware_type,
        **kwargs
    )
    
    # This would be implemented with specific middleware classes
    # For now, return a placeholder
    raise NotImplementedError(f"Middleware type {middleware_type} not implemented yet")


# Common middleware combinations
def security_chain(name: str = "security_chain") -> MiddlewareChain:
    """Create security middleware chain"""
    return create_middleware_chain(name, [
        "authentication",
        "authorization",
        "csrf_protection",
        "rate_limiting"
    ])


def performance_chain(name: str = "performance_chain") -> MiddlewareChain:
    """Create performance middleware chain"""
    return create_middleware_chain(name, [
        "timing",
        "profiling",
        "caching",
        "compression"
    ])


def monitoring_chain(name: str = "monitoring_chain") -> MiddlewareChain:
    """Create monitoring middleware chain"""
    return create_middleware_chain(name, [
        "logging",
        "audit",
        "metrics",
        "health_check"
    ])


def error_handling_chain(name: str = "error_handling_chain") -> MiddlewareChain:
    """Create error handling middleware chain"""
    return create_middleware_chain(name, [
        "error_handling",
        "exception_handling",
        "fallback",
        "logging"
    ])





















