from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import functools
import time
import inspect
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, TypeVar, Generic
from datetime import datetime, timezone
import structlog
from dataclasses import dataclass
from enum import Enum
import weakref
                import hashlib
                import json
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Async Decorators and Utilities for HeyGen AI API
Ensure all I/O operations are non-blocking and optimized.
"""


logger = structlog.get_logger()

# =============================================================================
# Async Decorator Types
# =============================================================================

class AsyncTimeoutStrategy(Enum):
    """Async timeout strategy enumeration."""
    CANCEL = "cancel"
    RETURN_DEFAULT = "return_default"
    RAISE_ERROR = "raise_error"

class AsyncRetryStrategy(Enum):
    """Async retry strategy enumeration."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"

@dataclass
class AsyncDecoratorConfig:
    """Configuration for async decorators."""
    timeout: Optional[float] = None
    timeout_strategy: AsyncTimeoutStrategy = AsyncTimeoutStrategy.RAISE_ERROR
    max_retries: int = 3
    retry_strategy: AsyncRetryStrategy = AsyncRetryStrategy.EXPONENTIAL_BACKOFF
    retry_delay: float = 1.0
    retry_backoff_factor: float = 2.0
    max_retry_delay: float = 60.0
    default_value: Any = None
    log_errors: bool = True
    log_performance: bool = True
    cache_result: bool = False
    cache_ttl: int = 300

# =============================================================================
# Async Timeout Decorator
# =============================================================================

def async_timeout(
    timeout: Optional[float] = None,
    strategy: AsyncTimeoutStrategy = AsyncTimeoutStrategy.RAISE_ERROR,
    default_value: Any = None
):
    """Decorator to add timeout to async functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                if timeout:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                else:
                    return await func(*args, **kwargs)
            except asyncio.TimeoutError:
                if strategy == AsyncTimeoutStrategy.CANCEL:
                    # Cancel the task
                    raise
                elif strategy == AsyncTimeoutStrategy.RETURN_DEFAULT:
                    logger.warning(f"Function {func.__name__} timed out, returning default value")
                    return default_value
                elif strategy == AsyncTimeoutStrategy.RAISE_ERROR:
                    logger.error(f"Function {func.__name__} timed out after {timeout} seconds")
                    raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")
        
        return wrapper
    return decorator

# =============================================================================
# Async Retry Decorator
# =============================================================================

def async_retry(
    max_retries: int = 3,
    strategy: AsyncRetryStrategy = AsyncRetryStrategy.EXPONENTIAL_BACKOFF,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    max_retry_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """Decorator to add retry logic to async functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    # Calculate delay
                    if strategy == AsyncRetryStrategy.EXPONENTIAL_BACKOFF:
                        delay = min(retry_delay * (retry_backoff_factor ** attempt), max_retry_delay)
                    elif strategy == AsyncRetryStrategy.LINEAR_BACKOFF:
                        delay = min(retry_delay * (attempt + 1), max_retry_delay)
                    elif strategy == AsyncRetryStrategy.FIXED_DELAY:
                        delay = retry_delay
                    else:
                        delay = retry_delay
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    
                    await asyncio.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator

# =============================================================================
# Async Performance Monitor Decorator
# =============================================================================

def async_performance_monitor(
    operation_name: Optional[str] = None,
    log_slow_operations: bool = True,
    slow_operation_threshold_ms: float = 1000.0
):
    """Decorator to monitor async function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            operation = operation_name or func.__name__
            
            try:
                result = await func(*args, **kwargs)
                
                duration_ms = (time.time() - start_time) * 1000
                
                if duration_ms > slow_operation_threshold_ms and log_slow_operations:
                    logger.warning(
                        f"Slow operation detected: {operation} took {duration_ms:.2f}ms",
                        operation=operation,
                        duration_ms=duration_ms,
                        args=str(args),
                        kwargs=str(kwargs)
                    )
                else:
                    logger.debug(
                        f"Operation completed: {operation} took {duration_ms:.2f}ms",
                        operation=operation,
                        duration_ms=duration_ms
                    )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"Operation failed: {operation} failed after {duration_ms:.2f}ms: {e}",
                    operation=operation,
                    duration_ms=duration_ms,
                    error=str(e)
                )
                raise
        
        return wrapper
    return decorator

# =============================================================================
# Async Cache Decorator
# =============================================================================

def async_cache(
    ttl: int = 300,
    key_generator: Optional[Callable] = None,
    cache_manager: Optional[Any] = None
):
    """Decorator to cache async function results."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                # Default key generation
                key_data = {
                    "func": func.__name__,
                    "args": str(args),
                    "kwargs": str(sorted(kwargs.items()))
                }
                cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            
            # Try to get from cache
            if cache_manager:
                try:
                    cached_result = await cache_manager.get(cache_key)
                    if cached_result is not None:
                        logger.debug(f"Cache hit for {func.__name__}")
                        return cached_result
                except Exception as e:
                    logger.warning(f"Cache get error: {e}")
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if cache_manager:
                try:
                    await cache_manager.set(cache_key, result, ttl)
                    logger.debug(f"Cached result for {func.__name__}")
                except Exception as e:
                    logger.warning(f"Cache set error: {e}")
            
            return result
        
        return wrapper
    return decorator

# =============================================================================
# Async Rate Limiter Decorator
# =============================================================================

class AsyncRateLimiter:
    """Async rate limiter implementation."""
    
    def __init__(self, max_calls: int, time_window: float):
        
    """__init__ function."""
self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[float] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire rate limit permit."""
        async with self._lock:
            now = time.time()
            
            # Remove old calls
            self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            if len(self.calls) >= self.max_calls:
                return False
            
            self.calls.append(now)
            return True

def async_rate_limit(
    max_calls: int,
    time_window: float,
    rate_limiters: Optional[Dict[str, AsyncRateLimiter]] = None
):
    """Decorator to add rate limiting to async functions."""
    if rate_limiters is None:
        rate_limiters = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Get or create rate limiter for this function
            func_name = func.__name__
            if func_name not in rate_limiters:
                rate_limiters[func_name] = AsyncRateLimiter(max_calls, time_window)
            
            rate_limiter = rate_limiters[func_name]
            
            # Try to acquire permit
            if not await rate_limiter.acquire():
                logger.warning(f"Rate limit exceeded for {func_name}")
                raise Exception(f"Rate limit exceeded for {func_name}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# =============================================================================
# Async Circuit Breaker Decorator
# =============================================================================

class AsyncCircuitBreaker:
    """Async circuit breaker implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        
    """__init__ function."""
self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker logic."""
        async with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                
                return result
                
            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                
                raise

def async_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception,
    circuit_breakers: Optional[Dict[str, AsyncCircuitBreaker]] = None
):
    """Decorator to add circuit breaker to async functions."""
    if circuit_breakers is None:
        circuit_breakers = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Get or create circuit breaker for this function
            func_name = func.__name__
            if func_name not in circuit_breakers:
                circuit_breakers[func_name] = AsyncCircuitBreaker(
                    failure_threshold, recovery_timeout, expected_exception
                )
            
            circuit_breaker = circuit_breakers[func_name]
            return await circuit_breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator

# =============================================================================
# Async Batch Processing Decorator
# =============================================================================

def async_batch_processor(
    batch_size: int = 100,
    max_concurrent: Optional[int] = None,
    processor_func: Optional[Callable] = None
):
    """Decorator to process items in batches asynchronously."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(items: List[Any], *args, **kwargs):
            
    """wrapper function."""
if not items:
                return []
            
            # Use provided processor or the decorated function
            processor = processor_func or func
            
            # Process items in batches
            results = []
            semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None
            
            async def process_batch(batch) -> Any:
                if semaphore:
                    async with semaphore:
                        return await processor(batch, *args, **kwargs)
                else:
                    return await processor(batch, *args, **kwargs)
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_result = await process_batch(batch)
                results.append(batch_result)
            
            return results
        
        return wrapper
    return decorator

# =============================================================================
# Async Concurrent Processing Decorator
# =============================================================================

def async_concurrent_processor(
    max_concurrent: int = 50,
    return_exceptions: bool = False
):
    """Decorator to process items concurrently."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(items: List[Any], *args, **kwargs):
            
    """wrapper function."""
if not items:
                return []
            
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_item(item) -> Any:
                async with semaphore:
                    return await func(item, *args, **kwargs)
            
            tasks = [process_item(item) for item in items]
            return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        
        return wrapper
    return decorator

# =============================================================================
# Async Background Task Decorator
# =============================================================================

def async_background_task(
    task_name: Optional[str] = None,
    fire_and_forget: bool = True
):
    """Decorator to run function as background task."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            task = asyncio.create_task(func(*args, **kwargs), name=task_name or func.__name__)
            
            if not fire_and_forget:
                return await task
            
            # Add error handling for fire-and-forget tasks
            def handle_task_exception(task) -> Any:
                try:
                    task.result()
                except Exception as e:
                    logger.error(f"Background task {func.__name__} failed: {e}")
            
            task.add_done_callback(handle_task_exception)
            return task
        
        return wrapper
    return decorator

# =============================================================================
# Async Resource Management Decorator
# =============================================================================

def async_resource_manager(
    acquire_func: Callable,
    release_func: Callable,
    timeout: Optional[float] = None
):
    """Decorator to manage async resources."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            resource = None
            try:
                # Acquire resource
                if timeout:
                    resource = await asyncio.wait_for(acquire_func(), timeout=timeout)
                else:
                    resource = await acquire_func()
                
                # Execute function with resource
                return await func(resource, *args, **kwargs)
                
            finally:
                # Release resource
                if resource:
                    try:
                        await release_func(resource)
                    except Exception as e:
                        logger.error(f"Error releasing resource: {e}")
        
        return wrapper
    return decorator

# =============================================================================
# Async Validation Decorator
# =============================================================================

def async_validate_input(
    validator_func: Callable,
    error_message: str = "Validation failed"
):
    """Decorator to validate async function inputs."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Validate inputs
            if asyncio.iscoroutinefunction(validator_func):
                is_valid = await validator_func(*args, **kwargs)
            else:
                is_valid = validator_func(*args, **kwargs)
            
            if not is_valid:
                raise ValueError(error_message)
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# =============================================================================
# Async Logging Decorator
# =============================================================================

def async_logging(
    log_input: bool = False,
    log_output: bool = False,
    log_errors: bool = True,
    log_performance: bool = True
):
    """Decorator to add comprehensive logging to async functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            func_name = func.__name__
            
            # Log input
            if log_input:
                logger.debug(
                    f"Calling {func_name}",
                    function=func_name,
                    args=str(args),
                    kwargs=str(kwargs)
                )
            
            try:
                result = await func(*args, **kwargs)
                
                # Log performance
                if log_performance:
                    duration_ms = (time.time() - start_time) * 1000
                    logger.debug(
                        f"{func_name} completed in {duration_ms:.2f}ms",
                        function=func_name,
                        duration_ms=duration_ms
                    )
                
                # Log output
                if log_output:
                    logger.debug(
                        f"{func_name} returned result",
                        function=func_name,
                        result=str(result)
                    )
                
                return result
                
            except Exception as e:
                # Log errors
                if log_errors:
                    duration_ms = (time.time() - start_time) * 1000
                    logger.error(
                        f"{func_name} failed after {duration_ms:.2f}ms: {e}",
                        function=func_name,
                        duration_ms=duration_ms,
                        error=str(e),
                        args=str(args),
                        kwargs=str(kwargs)
                    )
                raise
        
        return wrapper
    return decorator

# =============================================================================
# Async Metrics Decorator
# =============================================================================

def async_metrics(
    metrics_collector: Optional[Any] = None,
    operation_name: Optional[str] = None
):
    """Decorator to collect metrics for async functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            operation = operation_name or func.__name__
            
            try:
                result = await func(*args, **kwargs)
                
                # Record success metrics
                if metrics_collector:
                    duration_ms = (time.time() - start_time) * 1000
                    metrics_collector.record_success(operation, duration_ms)
                
                return result
                
            except Exception as e:
                # Record error metrics
                if metrics_collector:
                    duration_ms = (time.time() - start_time) * 1000
                    metrics_collector.record_error(operation, duration_ms, str(e))
                raise
        
        return wrapper
    return decorator

# =============================================================================
# Combined Async Decorator
# =============================================================================

def async_optimized(
    config: AsyncDecoratorConfig
):
    """Combined decorator with multiple optimizations."""
    def decorator(func: Callable) -> Callable:
        # Apply all decorators based on config
        if config.timeout:
            func = async_timeout(config.timeout, config.timeout_strategy, config.default_value)(func)
        
        if config.max_retries > 0:
            func = async_retry(
                config.max_retries,
                config.retry_strategy,
                config.retry_delay,
                config.retry_backoff_factor,
                config.max_retry_delay
            )(func)
        
        if config.log_performance:
            func = async_performance_monitor()(func)
        
        if config.cache_result:
            func = async_cache(config.cache_ttl)(func)
        
        if config.log_errors:
            func = async_logging(log_errors=True)(func)
        
        return func
    
    return decorator

# =============================================================================
# Utility Functions
# =============================================================================

def is_async_function(func: Callable) -> bool:
    """Check if function is async."""
    return asyncio.iscoroutinefunction(func)

def ensure_async(func: Callable) -> Callable:
    """Ensure function is async, wrap if necessary."""
    if is_async_function(func):
        return func
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        return func(*args, **kwargs)
    
    return async_wrapper

def run_async_in_thread(func: Callable, *args, **kwargs):
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    """Run async function in thread pool."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, func, *args, **kwargs)

async def run_sync_in_async(func: Callable, *args, **kwargs):
    """Run sync function in async context."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args, **kwargs)

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "AsyncTimeoutStrategy",
    "AsyncRetryStrategy",
    "AsyncDecoratorConfig",
    "async_timeout",
    "async_retry",
    "async_performance_monitor",
    "async_cache",
    "AsyncRateLimiter",
    "async_rate_limit",
    "AsyncCircuitBreaker",
    "async_circuit_breaker",
    "async_batch_processor",
    "async_concurrent_processor",
    "async_background_task",
    "async_resource_manager",
    "async_validate_input",
    "async_logging",
    "async_metrics",
    "async_optimized",
    "is_async_function",
    "ensure_async",
    "run_async_in_thread",
    "run_sync_in_async",
] 