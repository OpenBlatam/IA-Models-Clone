"""
Timeout utilities and decorators
"""

import asyncio
import functools
import signal
from typing import Callable, Any, TypeVar, ParamSpec, Optional
from contextlib import contextmanager

from .logging_config import get_logger
from .exceptions import TimeoutError

logger = get_logger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


def timeout(seconds: float):
    """
    Decorator to add timeout to async functions
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.warning(
                    f"{func.__name__} timed out after {seconds}s",
                    extra={"function": func.__name__, "timeout": seconds}
                )
                raise TimeoutError(func.__name__, seconds)
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # For sync functions, use threading timeout
            import threading
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=seconds)
            
            if thread.is_alive():
                logger.warning(
                    f"{func.__name__} timed out after {seconds}s",
                    extra={"function": func.__name__, "timeout": seconds}
                )
                raise TimeoutError(func.__name__, seconds)
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


@contextmanager
def timeout_context(seconds: float, operation_name: str = "operation"):
    """Context manager for timeout operations"""
    try:
        yield
    except asyncio.TimeoutError:
        logger.warning(f"{operation_name} timed out after {seconds}s")
        raise TimeoutError(operation_name, seconds)


async def with_timeout(coro, seconds: float, operation_name: str = "operation"):
    """Execute coroutine with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        logger.warning(f"{operation_name} timed out after {seconds}s")
        raise TimeoutError(operation_name, seconds)

