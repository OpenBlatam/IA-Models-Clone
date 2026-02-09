"""
Context Utilities for Piel Mejorador AI SAM3
===========================================

Unified context manager utilities.
"""

import asyncio
import logging
from typing import Callable, Any, Optional, TypeVar, AsyncContextManager, ContextManager
from contextlib import contextmanager, asynccontextmanager
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ContextUtils:
    """Unified context manager utilities."""
    
    @staticmethod
    @contextmanager
    def timer(operation_name: str = "operation"):
        """
        Context manager for timing operations.
        
        Args:
            operation_name: Name of operation
            
        Yields:
            Timer context
        """
        import time
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            logger.debug(f"{operation_name} took {elapsed:.3f}s")
    
    @staticmethod
    @asynccontextmanager
    async def async_timer(operation_name: str = "operation"):
        """
        Async context manager for timing operations.
        
        Args:
            operation_name: Name of operation
            
        Yields:
            Timer context
        """
        import time
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            logger.debug(f"{operation_name} took {elapsed:.3f}s")
    
    @staticmethod
    @contextmanager
    def suppress_exceptions(*exception_types: type[Exception]):
        """
        Context manager to suppress specific exceptions.
        
        Args:
            *exception_types: Exception types to suppress
            
        Yields:
            Suppress context
        """
        try:
            yield
        except exception_types as e:
            logger.debug(f"Suppressed exception: {e}")
    
    @staticmethod
    @asynccontextmanager
    async def async_suppress_exceptions(*exception_types: type[Exception]):
        """
        Async context manager to suppress specific exceptions.
        
        Args:
            *exception_types: Exception types to suppress
            
        Yields:
            Suppress context
        """
        try:
            yield
        except exception_types as e:
            logger.debug(f"Suppressed exception: {e}")
    
    @staticmethod
    @contextmanager
    def resource_cleanup(cleanup_func: Callable[[Any], None], resource: Any):
        """
        Context manager for resource cleanup.
        
        Args:
            cleanup_func: Cleanup function
            resource: Resource to cleanup
            
        Yields:
            Resource
        """
        try:
            yield resource
        finally:
            try:
                cleanup_func(resource)
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
    
    @staticmethod
    @asynccontextmanager
    async def async_resource_cleanup(
        cleanup_func: Callable[[Any], Any],
        resource: Any
    ):
        """
        Async context manager for resource cleanup.
        
        Args:
            cleanup_func: Cleanup function (can be async or sync)
            resource: Resource to cleanup
            
        Yields:
            Resource
        """
        try:
            yield resource
        finally:
            try:
                if asyncio.iscoroutinefunction(cleanup_func):
                    await cleanup_func(resource)
                else:
                    cleanup_func(resource)
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
    
    @staticmethod
    def make_context_manager(func: Callable) -> Callable:
        """
        Convert function to context manager.
        
        Args:
            func: Function to convert
            
        Returns:
            Context manager
        """
        if asyncio.iscoroutinefunction(func):
            @asynccontextmanager
            async def async_cm(*args, **kwargs):
                result = await func(*args, **kwargs)
                try:
                    yield result
                finally:
                    # Try to call cleanup if exists
                    if hasattr(result, 'close'):
                        close_method = result.close
                        if asyncio.iscoroutinefunction(close_method):
                            await close_method()
                        else:
                            close_method()
            
            return async_cm
        else:
            @contextmanager
            def sync_cm(*args, **kwargs):
                result = func(*args, **kwargs)
                try:
                    yield result
                finally:
                    # Try to call cleanup if exists
                    if hasattr(result, 'close'):
                        result.close()
            
            return sync_cm


# Convenience functions
@contextmanager
def timer(operation_name: str = "operation"):
    """Time an operation."""
    with ContextUtils.timer(operation_name):
        yield


@asynccontextmanager
async def async_timer(operation_name: str = "operation"):
    """Time an async operation."""
    async with ContextUtils.async_timer(operation_name):
        yield


@contextmanager
def suppress_exceptions(*exception_types: type[Exception]):
    """Suppress specific exceptions."""
    with ContextUtils.suppress_exceptions(*exception_types):
        yield


@asynccontextmanager
async def async_suppress_exceptions(*exception_types: type[Exception]):
    """Suppress specific exceptions in async context."""
    async with ContextUtils.async_suppress_exceptions(*exception_types):
        yield




