"""Advanced decorators for common patterns."""

from functools import wraps
from typing import Callable, Any, Optional
import asyncio
import time
import functools
import warnings
from datetime import datetime
from threading import Lock

from utils.logger import get_logger
from utils.metrics import metrics_collector

logger = get_logger(__name__)


def timeout(seconds: float):
    """
    Decorator to add timeout to async functions.
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {seconds}s")
                raise TimeoutError(f"Operation timed out after {seconds} seconds")
        return wrapper
    return decorator


def cache_result(ttl_seconds: Optional[float] = None, max_size: int = 128):
    """
    Decorator to cache function results.
    
    Args:
        ttl_seconds: Time to live in seconds (None for no expiration)
        max_size: Maximum cache size
    """
    cache = {}
    cache_times = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            key = str((args, tuple(sorted(kwargs.items()))))
            
            # Check cache
            if key in cache:
                if ttl_seconds is None:
                    return cache[key]
                
                elapsed = time.time() - cache_times.get(key, 0)
                if elapsed < ttl_seconds:
                    return cache[key]
                else:
                    # Expired, remove
                    del cache[key]
                    del cache_times[key]
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Store in cache
            if len(cache) >= max_size:
                # Remove oldest entry
                oldest_key = min(cache_times.items(), key=lambda x: x[1])[0]
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            cache[key] = result
            cache_times[key] = time.time()
            
            return result
        
        # Add cache management methods
        wrapper.clear_cache = lambda: (cache.clear(), cache_times.clear())
        wrapper.cache_size = lambda: len(cache)
        
        return wrapper
    return decorator


def throttle(calls: int, period: float):
    """
    Decorator to throttle function calls.
    
    Args:
        calls: Maximum number of calls
        period: Time period in seconds
    """
    call_times = []
    lock = asyncio.Lock()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            async with lock:
                now = time.time()
                
                # Remove old calls
                call_times[:] = [t for t in call_times if now - t < period]
                
                if len(call_times) >= calls:
                    sleep_time = period - (now - call_times[0])
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                        now = time.time()
                        call_times[:] = [t for t in call_times if now - t < period]
                
                call_times.append(now)
            
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def track_performance(metric_name: Optional[str] = None):
    """
    Decorator to track function performance.
    
    Args:
        metric_name: Custom metric name (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        metric = metric_name or f"function.{func.__name__}"
        
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                metrics_collector.record_timing(metric, duration)
                metrics_collector.increment(f"{metric}.success")
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.record_timing(metric, duration)
                metrics_collector.increment(f"{metric}.error")
                raise
        
        return wrapper
    return decorator


def validate_input(validator: Callable):
    """
    Decorator to validate function inputs.
    
    Args:
        validator: Function that validates args and kwargs
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                validator(*args, **kwargs)
            except Exception as e:
                logger.error(f"Validation failed for {func.__name__}: {e}")
                raise ValueError(f"Invalid input: {e}")
            
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def log_execution(log_args: bool = False, log_result: bool = False):
    """
    Decorator to log function execution.
    
    Args:
        log_args: Whether to log arguments
        log_result: Whether to log result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            logger.info(f"Executing {func.__name__}")
            
            if log_args:
                logger.debug(f"Args: {args}, Kwargs: {kwargs}")
            
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger.info(f"{func.__name__} completed in {duration:.2f}s")
                
                if log_result:
                    logger.debug(f"Result: {result}")
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator


def singleton(cls):
    """
    Decorator to make a class a singleton.
    
    Args:
        cls: Class to make singleton
    """
    instances = {}
    
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def memoize(ttl_seconds: Optional[float] = None):
    """
    Decorator to memoize function results (synchronous only).
    
    Args:
        ttl_seconds: Time to live in seconds
    """
    cache = {}
    cache_times = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            key = str((args, tuple(sorted(kwargs.items()))))
            
            if key in cache:
                if ttl_seconds is None:
                    return cache[key]
                
                elapsed = time.time() - cache_times.get(key, 0)
                if elapsed < ttl_seconds:
                    return cache[key]
            
            result = func(*args, **kwargs)
            cache[key] = result
            cache_times[key] = time.time()
            
            return result
        
        wrapper.clear_cache = lambda: (cache.clear(), cache_times.clear())
        return wrapper
    return decorator


# Additional decorators from decorator_utils.py
def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    Retry decorator (sync).
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Delay between retries
        exceptions: Exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                    else:
                        raise
            if last_exception:
                raise last_exception
        return wrapper
    return decorator


def retry_async(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    Retry decorator (async).
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Delay between retries
        exceptions: Exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay)
                    else:
                        raise
            if last_exception:
                raise last_exception
        return wrapper
    return decorator


def synchronized(lock: Optional[Lock] = None):
    """
    Synchronized decorator for thread safety.
    
    Args:
        lock: Lock instance (creates new if None)
    """
    if lock is None:
        lock = Lock()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def cached_property(func: Callable) -> property:
    """Cached property decorator."""
    attr_name = f'_cached_{func.__name__}'
    
    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    
    return wrapper


def deprecated(reason: str = ""):
    """
    Deprecated decorator.
    
    Args:
        reason: Deprecation reason
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated"
            if reason:
                message += f": {reason}"
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_args(validator: Callable):
    """
    Validate arguments decorator (alias for validate_input).
    
    Args:
        validator: Validation function
    """
    return validate_input(validator)


def log_calls(custom_logger: Optional[Any] = None):
    """
    Log function calls decorator.
    
    Args:
        custom_logger: Logger instance (uses default if None)
    """
    def decorator(func: Callable) -> Callable:
        log = custom_logger or logger
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if log:
                log.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            result = func(*args, **kwargs)
            if log:
                log.info(f"{func.__name__} returned {result}")
            return result
        return wrapper
    return decorator


# Decorators from decorators.py
def track_metrics(metric_name: str):
    """
    Decorator to track metrics for function calls.
    
    Args:
        metric_name: Base name for the metric
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics_collector.increment(f"{metric_name}.calls")
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_timing(f"{metric_name}.duration", duration)
                return result
            except Exception as e:
                metrics_collector.increment(f"{metric_name}.errors")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics_collector.increment(f"{metric_name}.calls")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_timing(f"{metric_name}.duration", duration)
                return result
            except Exception as e:
                metrics_collector.increment(f"{metric_name}.errors")
                raise
        
        if functools.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def handle_exceptions(default_message: str = "An error occurred"):
    """
    Decorator to handle exceptions with logging.
    
    Args:
        default_message: Default error message
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                raise
        
        if functools.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

