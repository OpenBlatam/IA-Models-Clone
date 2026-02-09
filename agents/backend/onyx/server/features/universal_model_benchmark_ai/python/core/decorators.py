"""
Decorators Module - Reusable decorators for common patterns.

Provides:
- Performance monitoring decorators
- Error handling decorators
- Caching decorators
- Retry decorators
- Validation decorators
"""

import time
import functools
import logging
from typing import Callable, Any, Optional, Dict, TypeVar, ParamSpec
from collections import defaultdict

logger = logging.getLogger(__name__)

P = ParamSpec('P')
R = TypeVar('R')


# ════════════════════════════════════════════════════════════════════════════════
# PERFORMANCE DECORATORS
# ════════════════════════════════════════════════════════════════════════════════

def timed(func: Callable[P, R]) -> Callable[P, tuple[R, float]]:
    """
    Decorator to measure function execution time.
    
    Returns:
        Tuple of (result, elapsed_seconds)
    
    Example:
        >>> @timed
        >>> def my_function():
        >>>     return "result"
        >>> result, elapsed = my_function()
    """
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[R, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.4f}s")
        return result, elapsed
    return wrapper


def log_performance(threshold_seconds: float = 1.0):
    """
    Decorator to log slow function calls.
    
    Args:
        threshold_seconds: Log if execution time exceeds this threshold
    
    Example:
        >>> @log_performance(threshold_seconds=0.5)
        >>> def slow_function():
        >>>     time.sleep(1)
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            
            if elapsed > threshold_seconds:
                logger.warning(
                    f"{func.__name__} took {elapsed:.4f}s (threshold: {threshold_seconds}s)"
                )
            
            return result
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING DECORATORS
# ════════════════════════════════════════════════════════════════════════════════

def handle_errors(
    default_return: Any = None,
    log_error: bool = True,
    reraise: bool = False,
):
    """
    Decorator to handle errors gracefully.
    
    Args:
        default_return: Value to return on error
        log_error: Whether to log errors
        reraise: Whether to re-raise exceptions
    
    Example:
        >>> @handle_errors(default_return=0, log_error=True)
        >>> def risky_function():
        >>>     raise ValueError("Error")
        >>> result = risky_function()  # Returns 0, logs error
    """
    def decorator(func: Callable[P, R]) -> Callable[P, Optional[R]]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Optional[R]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                
                if reraise:
                    raise
                
                return default_return
        return wrapper
    return decorator


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator to retry function on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
    
    Example:
        >>> @retry_on_failure(max_attempts=3, delay=1.0)
        >>> def unreliable_function():
        >>>     # May fail
        >>>     pass
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            
            raise last_exception
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# CACHING DECORATORS
# ════════════════════════════════════════════════════════════════════════════════

def memoize(maxsize: Optional[int] = None, ttl: Optional[float] = None):
    """
    Decorator to cache function results.
    
    Args:
        maxsize: Maximum cache size (None for unlimited)
        ttl: Time-to-live in seconds (None for no expiration)
    
    Example:
        >>> @memoize(maxsize=100, ttl=3600)
        >>> def expensive_function(x):
        >>>     return x * 2
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        cache: Dict[Any, tuple[R, float]] = {}
        cache_times: Dict[Any, float] = {}
        
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Create cache key
            key = (args, tuple(sorted(kwargs.items())))
            
            # Check cache
            if key in cache:
                if ttl is None:
                    return cache[key][0]
                
                # Check TTL
                if time.time() - cache_times[key] < ttl:
                    return cache[key][0]
                else:
                    # Expired
                    del cache[key]
                    del cache_times[key]
            
            # Call function
            result = func(*args, **kwargs)
            
            # Store in cache
            if maxsize is None or len(cache) < maxsize:
                cache[key] = (result, time.time())
                cache_times[key] = time.time()
            else:
                # Remove oldest entry
                oldest_key = min(cache_times.items(), key=lambda x: x[1])[0]
                del cache[oldest_key]
                del cache_times[oldest_key]
                cache[key] = (result, time.time())
                cache_times[key] = time.time()
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: (cache.clear(), cache_times.clear())
        wrapper.cache_info = lambda: {
            "size": len(cache),
            "maxsize": maxsize,
            "ttl": ttl,
        }
        
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# VALIDATION DECORATORS
# ════════════════════════════════════════════════════════════════════════════════

def validate_args(**validators: Callable):
    """
    Decorator to validate function arguments.
    
    Args:
        **validators: Mapping of argument names to validator functions
    
    Example:
        >>> @validate_args(x=lambda x: x > 0, y=lambda y: isinstance(y, str))
        >>> def my_function(x, y):
        >>>     return x + len(y)
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate arguments
            for arg_name, validator in validators.items():
                if arg_name in bound.arguments:
                    value = bound.arguments[arg_name]
                    if not validator(value):
                        raise ValueError(
                            f"Validation failed for argument '{arg_name}': {value}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# METRICS DECORATORS
# ════════════════════════════════════════════════════════════════════════════════

_call_counts: Dict[str, int] = defaultdict(int)
_call_times: Dict[str, list[float]] = defaultdict(list)


def track_metrics(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator to track function call metrics.
    
    Tracks:
    - Call count
    - Execution times
    - Average execution time
    
    Example:
        >>> @track_metrics
        >>> def my_function():
        >>>     pass
        >>> 
        >>> # Get metrics
        >>> metrics = get_function_metrics("my_function")
    """
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        _call_counts[func.__name__] += 1
        _call_times[func.__name__].append(elapsed)
        
        return result
    return wrapper


def get_function_metrics(func_name: str) -> Dict[str, Any]:
    """
    Get metrics for a function.
    
    Args:
        func_name: Function name
    
    Returns:
        Dictionary with metrics
    """
    times = _call_times.get(func_name, [])
    
    if not times:
        return {
            "count": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "min_time": 0.0,
            "max_time": 0.0,
        }
    
    return {
        "count": _call_counts.get(func_name, 0),
        "total_time": sum(times),
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
    }


def reset_metrics():
    """Reset all function metrics."""
    _call_counts.clear()
    _call_times.clear()


__all__ = [
    # Performance
    "timed",
    "log_performance",
    # Error handling
    "handle_errors",
    "retry_on_failure",
    # Caching
    "memoize",
    # Validation
    "validate_args",
    # Metrics
    "track_metrics",
    "get_function_metrics",
    "reset_metrics",
]












