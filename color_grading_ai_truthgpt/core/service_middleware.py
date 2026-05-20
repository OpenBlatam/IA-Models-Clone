"""
Service Middleware for Color Grading AI
========================================

Middleware system for services with request/response interception.
"""

import logging
import inspect
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MiddlewareType(Enum):
    """Middleware types."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    BOTH = "both"


@dataclass
class MiddlewareContext:
    """Middleware context."""
    service_name: str
    method_name: str
    args: tuple
    kwargs: dict
    request_metadata: Dict[str, Any] = field(default_factory=dict)
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    result: Any = None


class ServiceMiddleware:
    """
    Service middleware.
    
    Features:
    - Request interception
    - Response interception
    - Error handling
    - Metadata propagation
    - Chain execution
    """
    
    def __init__(self, middleware_type: MiddlewareType = MiddlewareType.BOTH):
        """
        Initialize service middleware.
        
        Args:
            middleware_type: Middleware type
        """
        self.middleware_type = middleware_type
        self._request_middleware: List[Callable] = []
        self._response_middleware: List[Callable] = []
        self._error_middleware: List[Callable] = []
    
    def add_request_middleware(self, middleware: Callable):
        """
        Add request middleware.
        
        Args:
            middleware: Middleware function (context: MiddlewareContext) -> Optional[Any]
        """
        self._request_middleware.append(middleware)
        logger.debug("Added request middleware")
    
    def add_response_middleware(self, middleware: Callable):
        """
        Add response middleware.
        
        Args:
            middleware: Middleware function (context: MiddlewareContext) -> Optional[Any]
        """
        self._response_middleware.append(middleware)
        logger.debug("Added response middleware")
    
    def add_error_middleware(self, middleware: Callable):
        """
        Add error middleware.
        
        Args:
            middleware: Middleware function (context: MiddlewareContext) -> Optional[Any]
        """
        self._error_middleware.append(middleware)
        logger.debug("Added error middleware")
    
    async def execute(
        self,
        service_name: str,
        method_name: str,
        method: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute method with middleware.
        
        Args:
            service_name: Service name
            method_name: Method name
            method: Method to execute
            *args: Method arguments
            **kwargs: Method keyword arguments
            
        Returns:
            Method result
        """
        context = MiddlewareContext(
            service_name=service_name,
            method_name=method_name,
            args=args,
            kwargs=kwargs
        )
        
        try:
            # Execute request middleware
            for middleware in self._request_middleware:
                result = middleware(context)
                if inspect.iscoroutinefunction(middleware):
                    result = await result
                if result is not None:
                    return result  # Short-circuit
            
            # Execute method
            if inspect.iscoroutinefunction(method):
                context.result = await method(*args, **kwargs)
            else:
                context.result = method(*args, **kwargs)
            
            # Execute response middleware
            for middleware in self._response_middleware:
                result = middleware(context)
                if inspect.iscoroutinefunction(middleware):
                    result = await result
                if result is not None:
                    return result  # Override response
            
            return context.result
        
        except Exception as e:
            context.error = e
            
            # Execute error middleware
            for middleware in self._error_middleware:
                try:
                    result = middleware(context)
                    if inspect.iscoroutinefunction(middleware):
                        result = await result
                    if result is not None:
                        return result  # Error handled
                except Exception as middleware_error:
                    logger.error(f"Error in error middleware: {middleware_error}")
            
            # Re-raise if not handled
            raise
    
    def wrap_service(self, service: Any, service_name: str) -> Any:
        """
        Wrap service with middleware.
        
        Args:
            service: Service instance
            service_name: Service name
            
        Returns:
            Wrapped service
        """
        class MiddlewareWrapper:
            def __init__(self, original_service, middleware, name):
                self._original = original_service
                self._middleware = middleware
                self._name = name
                
                # Copy attributes
                for attr in dir(original_service):
                    if not attr.startswith('_') and not callable(getattr(original_service, attr)):
                        setattr(self, attr, getattr(original_service, attr))
            
            def __getattr__(self, name):
                attr = getattr(self._original, name)
                
                if callable(attr):
                    async def wrapped(*args, **kwargs):
                        return await self._middleware.execute(
                            self._name,
                            name,
                            attr,
                            *args,
                            **kwargs
                        )
                    return wrapped
                
                return attr
        
        return MiddlewareWrapper(service, self, service_name)

