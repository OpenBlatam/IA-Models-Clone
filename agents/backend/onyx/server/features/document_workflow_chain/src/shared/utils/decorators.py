"""
Decorators
==========

Advanced decorators for the application.
"""

from __future__ import annotations
import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from datetime import datetime, timedelta

from .helpers import DateTimeHelpers, ErrorHelpers


logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class RateLimiter:
    """Rate limiter decorator"""
    
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[datetime] = []
    
    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            now = DateTimeHelpers.now_utc()
            
            # Remove old calls outside the time window
            self.calls = [
                call_time for call_time in self.calls
                if now - call_time < timedelta(seconds=self.time_window)
            ]
            
            # Check if we've exceeded the rate limit
            if len(self.calls) >= self.max_calls:
                raise Exception(f"Rate limit exceeded: {self.max_calls} calls per {self.time_window} seconds")
            
            # Add current call
            self.calls.append(now)
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            now = DateTimeHelpers.now_utc()
            
            # Remove old calls outside the time window
            self.calls = [
                call_time for call_time in self.calls
                if now - call_time < timedelta(seconds=self.time_window)
            ]
            
            # Check if we've exceeded the rate limit
            if len(self.calls) >= self.max_calls:
                raise Exception(f"Rate limit exceeded: {self.max_calls} calls per {self.time_window} seconds")
            
            # Add current call
            self.calls.append(now)
            
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


class CircuitBreaker:
    """Circuit breaker decorator"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.last_failure_time is None:
            return True
        
        return (DateTimeHelpers.now_utc() - self.last_failure_time).total_seconds() >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = DateTimeHelpers.now_utc()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class Timeout:
    """Timeout decorator"""
    
    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
    
    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout_seconds)
            except asyncio.TimeoutError:
                raise Exception(f"Function {func.__name__} timed out after {self.timeout_seconds} seconds")
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we can't easily implement timeout without threading
            # This is a placeholder implementation
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


class Cache:
    """Simple cache decorator"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_key = self._generate_cache_key(func.__name__, args, kwargs)
            
            # Check cache
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if self._is_cache_valid(cached_data):
                    return cached_data["value"]
                else:
                    del self.cache[cache_key]
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            self.cache[cache_key] = {
                "value": result,
                "timestamp": DateTimeHelpers.now_utc()
            }
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_key = self._generate_cache_key(func.__name__, args, kwargs)
            
            # Check cache
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if self._is_cache_valid(cached_data):
                    return cached_data["value"]
                else:
                    del self.cache[cache_key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            self.cache[cache_key] = {
                "value": result,
                "timestamp": DateTimeHelpers.now_utc()
            }
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key"""
        key_parts = [func_name]
        
        # Add args
        for arg in args:
            key_parts.append(str(arg))
        
        # Add kwargs
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}:{value}")
        
        return ":".join(key_parts)
    
    def _is_cache_valid(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cache entry is valid"""
        timestamp = cached_data["timestamp"]
        return (DateTimeHelpers.now_utc() - timestamp).total_seconds() < self.ttl_seconds


class LogExecution:
    """Log execution decorator"""
    
    def __init__(self, log_level: int = logging.INFO, include_args: bool = False):
        self.log_level = log_level
        self.include_args = include_args
    
    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Log function start
            log_data = {
                "function": func.__name__,
                "action": "started",
                "timestamp": DateTimeHelpers.now_utc().isoformat()
            }
            
            if self.include_args:
                log_data["args"] = str(args)
                log_data["kwargs"] = str(kwargs)
            
            logger.log(self.log_level, f"Function execution started: {log_data}")
            
            try:
                result = await func(*args, **kwargs)
                
                # Log successful completion
                execution_time = time.time() - start_time
                log_data.update({
                    "action": "completed",
                    "execution_time": execution_time,
                    "status": "success"
                })
                
                logger.log(self.log_level, f"Function execution completed: {log_data}")
                return result
                
            except Exception as e:
                # Log error
                execution_time = time.time() - start_time
                log_data.update({
                    "action": "failed",
                    "execution_time": execution_time,
                    "status": "error",
                    "error": str(e)
                })
                
                logger.log(self.log_level, f"Function execution failed: {log_data}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Log function start
            log_data = {
                "function": func.__name__,
                "action": "started",
                "timestamp": DateTimeHelpers.now_utc().isoformat()
            }
            
            if self.include_args:
                log_data["args"] = str(args)
                log_data["kwargs"] = str(kwargs)
            
            logger.log(self.log_level, f"Function execution started: {log_data}")
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                execution_time = time.time() - start_time
                log_data.update({
                    "action": "completed",
                    "execution_time": execution_time,
                    "status": "success"
                })
                
                logger.log(self.log_level, f"Function execution completed: {log_data}")
                return result
                
            except Exception as e:
                # Log error
                execution_time = time.time() - start_time
                log_data.update({
                    "action": "failed",
                    "execution_time": execution_time,
                    "status": "error",
                    "error": str(e)
                })
                
                logger.log(self.log_level, f"Function execution failed: {log_data}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


class ValidateInput:
    """Input validation decorator"""
    
    def __init__(self, validator_func: Callable):
        self.validator_func = validator_func
    
    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Validate inputs
            validation_result = self.validator_func(*args, **kwargs)
            if not validation_result.is_valid:
                raise ValueError(f"Validation failed: {validation_result.errors}")
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Validate inputs
            validation_result = self.validator_func(*args, **kwargs)
            if not validation_result.is_valid:
                raise ValueError(f"Validation failed: {validation_result.errors}")
            
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


class Retry:
    """Retry decorator with exponential backoff"""
    
    def __init__(self, max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
    
    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = self.delay
            
            for attempt in range(self.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < self.max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        await asyncio.sleep(current_delay)
                        current_delay *= self.backoff
                    else:
                        logger.error(f"All {self.max_attempts} attempts failed. Last error: {e}")
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = self.delay
            
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < self.max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= self.backoff
                    else:
                        logger.error(f"All {self.max_attempts} attempts failed. Last error: {e}")
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


class Deprecated:
    """Deprecated function decorator"""
    
    def __init__(self, message: str = "This function is deprecated"):
        self.message = message
    
    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.warning(f"DEPRECATED: {func.__name__} - {self.message}")
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger.warning(f"DEPRECATED: {func.__name__} - {self.message}")
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


class Singleton:
    """Singleton decorator"""
    
    def __init__(self, cls):
        self.cls = cls
        self.instance = None
    
    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = self.cls(*args, **kwargs)
        return self.instance


class Memoize:
    """Memoization decorator"""
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
    
    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = str(args) + str(sorted(kwargs.items()))
            
            # Check cache
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Manage cache size
            if len(self.cache) >= self.max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = result
            return result
        
        return wrapper


# Convenience functions
def rate_limit(max_calls: int, time_window: int):
    """Rate limit decorator"""
    return RateLimiter(max_calls, time_window)


def circuit_breaker(failure_threshold: int = 5, recovery_timeout: int = 60):
    """Circuit breaker decorator"""
    return CircuitBreaker(failure_threshold, recovery_timeout)


def timeout(seconds: int):
    """Timeout decorator"""
    return Timeout(seconds)


def cache(ttl_seconds: int = 300):
    """Cache decorator"""
    return Cache(ttl_seconds)


def log_execution(log_level: int = logging.INFO, include_args: bool = False):
    """Log execution decorator"""
    return LogExecution(log_level, include_args)


def validate_input(validator_func: Callable):
    """Input validation decorator"""
    return ValidateInput(validator_func)


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator"""
    return Retry(max_attempts, delay, backoff)


def deprecated(message: str = "This function is deprecated"):
    """Deprecated decorator"""
    return Deprecated(message)


def singleton(cls):
    """Singleton decorator"""
    return Singleton(cls)


def memoize(max_size: int = 128):
    """Memoization decorator"""
    return Memoize(max_size)




