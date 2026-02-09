"""
Performance Utilities

Utility functions for performance monitoring and optimization.
"""

import time
import functools
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
    
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} took {duration:.4f} seconds")
    
    return wrapper


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
):
    """
    Decorator to retry function on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {str(e)}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            
            raise last_exception
        
        return wrapper
    return decorator


def throttle(calls: int, period: float):
    """
    Decorator to throttle function calls.
    
    Args:
        calls: Maximum number of calls
        period: Time period in seconds
    
    Returns:
        Decorator function
    """
    import threading
    
    lock = threading.Lock()
    call_times = []
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                now = time.time()
                # Remove old call times
                call_times[:] = [t for t in call_times if now - t < period]
                
                if len(call_times) >= calls:
                    sleep_time = period - (now - call_times[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        now = time.time()
                        call_times[:] = [t for t in call_times if now - t < period]
                
                call_times.append(now)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class PerformanceMonitor:
    """
    Context manager for monitoring code block performance.
    """
    
    def __init__(self, name: str):
        """
        Initialize performance monitor.
        
        Args:
            name: Name of the monitored block
        """
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start monitoring."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and log."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {duration:.4f} seconds")
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0



