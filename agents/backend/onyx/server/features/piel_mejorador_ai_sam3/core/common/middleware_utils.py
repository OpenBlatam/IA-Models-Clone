"""
Middleware Utilities for Piel Mejorador AI SAM3
===============================================

Unified middleware and interceptor pattern utilities.
"""

import asyncio
import logging
from typing import Callable, Any, Optional, List, TypeVar, Awaitable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class MiddlewareContext:
    """Middleware execution context."""
    request: Any
    response: Optional[Any] = None
    metadata: dict = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()


class Middleware:
    """Base middleware class."""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize middleware.
        
        Args:
            name: Optional middleware name
        """
        self.name = name or self.__class__.__name__
    
    async def process(
        self,
        context: MiddlewareContext,
        next_handler: Callable[[MiddlewareContext], Awaitable[Any]]
    ) -> Any:
        """
        Process middleware.
        
        Args:
            context: Middleware context
            next_handler: Next handler in chain
            
        Returns:
            Response
        """
        # Default: just call next
        return await next_handler(context)


class MiddlewareChain:
    """Chain of middleware handlers."""
    
    def __init__(self, *middlewares: Middleware):
        """
        Initialize middleware chain.
        
        Args:
            *middlewares: Middleware instances
        """
        self._middlewares: List[Middleware] = list(middlewares)
    
    def add(self, middleware: Middleware) -> "MiddlewareChain":
        """
        Add middleware to chain.
        
        Args:
            middleware: Middleware to add
            
        Returns:
            Self for chaining
        """
        self._middlewares.append(middleware)
        return self
    
    def add_first(self, middleware: Middleware) -> "MiddlewareChain":
        """
        Add middleware at beginning of chain.
        
        Args:
            middleware: Middleware to add
            
        Returns:
            Self for chaining
        """
        self._middlewares.insert(0, middleware)
        return self
    
    async def execute(
        self,
        request: Any,
        final_handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        """
        Execute middleware chain.
        
        Args:
            request: Request object
            final_handler: Final handler to call
            
        Returns:
            Response
        """
        context = MiddlewareContext(request=request)
        
        # Build chain from end to beginning
        handler = final_handler
        
        for middleware in reversed(self._middlewares):
            current_middleware = middleware
            next_handler = handler
            
            async def create_handler(mw, nh):
                async def handler_wrapper(ctx):
                    return await mw.process(ctx, nh)
                return handler_wrapper
            
            handler = await create_handler(current_middleware, next_handler)
        
        # Execute chain
        async def chain_handler(ctx):
            return await handler(ctx)
        
        return await chain_handler(context)


class MiddlewareUtils:
    """Unified middleware utilities."""
    
    @staticmethod
    def create_chain(*middlewares: Middleware) -> MiddlewareChain:
        """
        Create middleware chain.
        
        Args:
            *middlewares: Middleware instances
            
        Returns:
            MiddlewareChain
        """
        return MiddlewareChain(*middlewares)
    
    @staticmethod
    def create_middleware(
        process_func: Callable[[MiddlewareContext, Callable], Awaitable[Any]],
        name: Optional[str] = None
    ) -> Middleware:
        """
        Create middleware from function.
        
        Args:
            process_func: Processing function
            name: Optional middleware name
            
        Returns:
            Middleware instance
        """
        class FunctionMiddleware(Middleware):
            async def process(self, context, next_handler):
                return await process_func(context, next_handler)
        
        return FunctionMiddleware(name or process_func.__name__)
    
    @staticmethod
    def create_logging_middleware(name: str = "logging") -> Middleware:
        """
        Create logging middleware.
        
        Args:
            name: Middleware name
            
        Returns:
            Logging middleware
        """
        class LoggingMiddleware(Middleware):
            async def process(self, context, next_handler):
                logger.info(f"Request: {context.request}")
                try:
                    response = await next_handler(context)
                    logger.info(f"Response: {response}")
                    return response
                except Exception as e:
                    logger.error(f"Error in middleware chain: {e}")
                    raise
        
        return LoggingMiddleware(name)
    
    @staticmethod
    def create_timing_middleware(name: str = "timing") -> Middleware:
        """
        Create timing middleware.
        
        Args:
            name: Middleware name
            
        Returns:
            Timing middleware
        """
        class TimingMiddleware(Middleware):
            async def process(self, context, next_handler):
                start = datetime.now()
                try:
                    response = await next_handler(context)
                    elapsed = (datetime.now() - start).total_seconds()
                    context.metadata["execution_time"] = elapsed
                    logger.debug(f"Execution time: {elapsed:.3f}s")
                    return response
                except Exception as e:
                    elapsed = (datetime.now() - start).total_seconds()
                    context.metadata["execution_time"] = elapsed
                    raise
        
        return TimingMiddleware(name)
    
    @staticmethod
    def create_error_handling_middleware(
        name: str = "error_handling",
        error_handler: Optional[Callable[[Exception, MiddlewareContext], Any]] = None
    ) -> Middleware:
        """
        Create error handling middleware.
        
        Args:
            name: Middleware name
            error_handler: Optional custom error handler
            
        Returns:
            Error handling middleware
        """
        class ErrorHandlingMiddleware(Middleware):
            async def process(self, context, next_handler):
                try:
                    return await next_handler(context)
                except Exception as e:
                    if error_handler:
                        return await error_handler(e, context)
                    logger.error(f"Error in middleware: {e}")
                    raise
        
        return ErrorHandlingMiddleware(name)


# Convenience functions
def create_chain(*middlewares: Middleware) -> MiddlewareChain:
    """Create middleware chain."""
    return MiddlewareUtils.create_chain(*middlewares)


def create_middleware(process_func: Callable, **kwargs) -> Middleware:
    """Create middleware."""
    return MiddlewareUtils.create_middleware(process_func, **kwargs)




