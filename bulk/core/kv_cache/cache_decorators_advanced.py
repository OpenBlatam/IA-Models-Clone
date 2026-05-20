"""
Advanced cache decorators.

Provides advanced decorators for cache operations.
"""
from __future__ import annotations

import logging
import time
import threading
import functools
from typing import Dict, Any, Optional, Callable, TypeVar, ParamSpec, List

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


def cache_result(
    ttl: Optional[float] = None,
    max_size: int = 1000,
    key_func: Optional[Callable] = None
):
    """
    Cache function result.
    
    Args:
        ttl: Time to live in seconds
        max_size: Maximum cache size
        key_func: Optional key function
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        cache: Dict[str, tuple] = {}
        
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = str((args, tuple(sorted(kwargs.items()))))
            
            # Check cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                
                # Check TTL
                if ttl is None or (time.time() - timestamp) < ttl:
                    return result
                else:
                    del cache[cache_key]
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            if len(cache) >= max_size:
                # Remove oldest entry
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
            
            cache[cache_key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry on failure decorator.
    
    Args:
        max_retries: Maximum retries
        backoff_factor: Backoff factor
        exceptions: Exceptions to catch
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor * (2 ** attempt)
                        time.sleep(wait_time)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
                    else:
                        logger.error(f"All retries exhausted for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator


def rate_limit(
    max_calls: int = 10,
    period: float = 1.0
):
    """
    Rate limit decorator.
    
    Args:
        max_calls: Maximum calls per period
        period: Period in seconds
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        calls: List[float] = []
        lock = threading.Lock()
        
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with lock:
                now = time.time()
                
                # Remove old calls
                calls[:] = [c for c in calls if now - c < period]
                
                # Check rate limit
                if len(calls) >= max_calls:
                    sleep_time = period - (now - calls[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        calls.pop(0)
                
                calls.append(time.time())
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def timeout(timeout_seconds: float):
    """
    Timeout decorator.
    
    Args:
        timeout_seconds: Timeout in seconds
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out")
            
            # Set signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        
        return wrapper
    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception
):
    """
    Circuit breaker decorator.
    
    Args:
        failure_threshold: Failure threshold
        recovery_timeout: Recovery timeout
        expected_exception: Expected exception type
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        failure_count = 0
        last_failure_time: Optional[float] = None
        state = "closed"  # closed, open, half_open
        lock = threading.Lock()
        
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            nonlocal failure_count, last_failure_time, state
            
            with lock:
                # Check state
                if state == "open":
                    if last_failure_time and (time.time() - last_failure_time) > recovery_timeout:
                        state = "half_open"
                    else:
                        raise RuntimeError("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                
                # Success
                with lock:
                    if state == "half_open":
                        state = "closed"
                        failure_count = 0
                
                return result
            
            except expected_exception as e:
                with lock:
                    failure_count += 1
                    last_failure_time = time.time()
                    
                    if failure_count >= failure_threshold:
                        state = "open"
                        logger.error(f"Circuit breaker opened for {func.__name__}")
                
                raise
        
        return wrapper
    return decorator

