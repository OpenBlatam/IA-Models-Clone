"""
Advanced Timeout Handling - Timeout management and deadline enforcement
Production-ready timeout system
"""

import asyncio
import signal
import time
import threading
from typing import Any, Callable, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TimeoutConfig:
    """Timeout configuration"""
    timeout: float
    on_timeout: Optional[Callable[[], Any]] = None
    raise_on_timeout: bool = True
    default_value: Any = None

class TimeoutError(Exception):
    """Timeout exception"""
    pass

class TimeoutHandler:
    """Advanced timeout handler"""
    
    def __init__(self, default_timeout: float = 30.0):
        self.default_timeout = default_timeout
        self.timeout_count = 0
        self.total_operations = 0

    async def with_timeout(
        self,
        func: Callable,
        timeout: float = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with timeout"""
        timeout = timeout or self.default_timeout
        self.total_operations += 1
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout
                )
            else:
                # For sync functions, run in executor with timeout
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                    timeout=timeout
                )
            
            return result
            
        except asyncio.TimeoutError:
            self.timeout_count += 1
            logger.warning(f"Operation timed out after {timeout}s")
            
            if isinstance(func, Callable):
                func_name = getattr(func, '__name__', str(func))
            else:
                func_name = str(func)
            
            raise TimeoutError(
                f"Operation '{func_name}' timed out after {timeout}s"
            )

    def get_stats(self) -> dict:
        """Get timeout statistics"""
        timeout_rate = self.timeout_count / max(self.total_operations, 1)
        
        return {
            "timeout_count": self.timeout_count,
            "total_operations": self.total_operations,
            "timeout_rate": timeout_rate,
            "default_timeout": self.default_timeout
        }

def with_timeout(timeout: float, default_value: Any = None):
    """Decorator for automatic timeout handling"""
    handler = TimeoutHandler(timeout)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await handler.with_timeout(func, timeout, *args, **kwargs)
            except TimeoutError:
                if default_value is not None:
                    return default_value
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, use threading timeout
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout)
            
            if thread.is_alive():
                handler.timeout_count += 1
                handler.total_operations += 1
                logger.warning(f"Operation timed out after {timeout}s")
                if default_value is not None:
                    return default_value
                raise TimeoutError(f"Operation timed out after {timeout}s")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

import functools






