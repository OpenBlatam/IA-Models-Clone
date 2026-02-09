"""
Advanced decorator utilities
Reusable decorators for common patterns
"""

from typing import Callable, TypeVar, Any
from functools import wraps
import time
import logging

T = TypeVar('T')

logger = logging.getLogger(__name__)


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay in seconds
        backoff: Backoff multiplier
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def timeout(seconds: float):
    """
    Timeout decorator
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            
            return result
        
        return wrapper
    return decorator


def log_execution(func: Callable) -> Callable:
    """
    Log function execution
    
    Args:
        func: Function to log
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Executing {func.__name__} with args={args}, kwargs={kwargs}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
    
    return wrapper


def validate_args(*validators: Callable):
    """
    Validate function arguments
    
    Args:
        *validators: Validator functions
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate positional arguments
            for i, validator in enumerate(validators):
                if i < len(args):
                    if not validator(args[i]):
                        raise ValueError(f"Argument {i} failed validation")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def rate_limit(calls: int, period: float):
    """
    Rate limit decorator
    
    Args:
        calls: Number of calls allowed
        period: Time period in seconds
    """
    import time
    from collections import deque
    
    call_times = deque()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove old calls
            while call_times and call_times[0] < now - period:
                call_times.popleft()
            
            if len(call_times) >= calls:
                raise RuntimeError(f"Rate limit exceeded: {calls} calls per {period}s")
            
            call_times.append(now)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

