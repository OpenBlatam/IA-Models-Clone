"""
Timing Utilities for Piel Mejorador AI SAM3
===========================================

Unified timing and performance measurement utilities.
"""

import time
import logging
from typing import Callable, Any, Optional, TypeVar
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class TimingResult:
    """Timing result."""
    elapsed: float
    operation_name: str
    success: bool = True
    error: Optional[Exception] = None


class TimingUtils:
    """Unified timing utilities."""
    
    @staticmethod
    def time_function(func: Callable) -> Callable:
        """
        Decorator to time function execution.
        
        Args:
            func: Function to time
            
        Returns:
            Decorated function
        """
        import asyncio
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    elapsed = time.time() - start
                    logger.debug(f"{func.__name__} took {elapsed:.3f}s")
                    return result
                except Exception as e:
                    elapsed = time.time() - start
                    logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
                    raise
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start
                    logger.debug(f"{func.__name__} took {elapsed:.3f}s")
                    return result
                except Exception as e:
                    elapsed = time.time() - start
                    logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
                    raise
            
            return sync_wrapper
    
    @staticmethod
    @contextmanager
    def measure(operation_name: str = "operation"):
        """
        Context manager to measure execution time.
        
        Args:
            operation_name: Name of operation
            
        Yields:
            TimingResult
        """
        start = time.time()
        result = TimingResult(
            elapsed=0.0,
            operation_name=operation_name,
            success=True
        )
        
        try:
            yield result
            result.success = True
        except Exception as e:
            result.success = False
            result.error = e
            raise
        finally:
            result.elapsed = time.time() - start
            logger.debug(
                f"{operation_name} took {result.elapsed:.3f}s "
                f"({'success' if result.success else 'failed'})"
            )
    
    @staticmethod
    def measure_sync(func: Callable, *args, **kwargs) -> tuple[Any, TimingResult]:
        """
        Measure sync function execution time.
        
        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, timing_result)
        """
        start = time.time()
        result = TimingResult(
            elapsed=0.0,
            operation_name=func.__name__,
            success=True
        )
        
        try:
            func_result = func(*args, **kwargs)
            result.success = True
            return func_result, result
        except Exception as e:
            result.success = False
            result.error = e
            raise
        finally:
            result.elapsed = time.time() - start
    
    @staticmethod
    async def measure_async(func: Callable, *args, **kwargs) -> tuple[Any, TimingResult]:
        """
        Measure async function execution time.
        
        Args:
            func: Async function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, timing_result)
        """
        start = time.time()
        result = TimingResult(
            elapsed=0.0,
            operation_name=func.__name__,
            success=True
        )
        
        try:
            func_result = await func(*args, **kwargs)
            result.success = True
            return func_result, result
        except Exception as e:
            result.success = False
            result.error = e
            raise
        finally:
            result.elapsed = time.time() - start
    
    @staticmethod
    def format_elapsed(seconds: float) -> str:
        """
        Format elapsed time in human-readable format.
        
        Args:
            seconds: Elapsed seconds
            
        Returns:
            Formatted string (e.g., "2h 30m 15s")
        """
        if seconds < 0.001:
            return f"{seconds * 1000000:.2f}μs"
        elif seconds < 1:
            return f"{seconds * 1000:.2f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.2f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.2f}s"


# Convenience functions
def time_function(func: Callable) -> Callable:
    """Time function execution."""
    return TimingUtils.time_function(func)


@contextmanager
def measure(operation_name: str = "operation"):
    """Measure execution time."""
    with TimingUtils.measure(operation_name) as result:
        yield result


def format_elapsed(seconds: float) -> str:
    """Format elapsed time."""
    return TimingUtils.format_elapsed(seconds)




