"""
PDF Variantes API - Robustness Features
Enhanced error handling, retries, circuit breakers, and resilience patterns
"""

import asyncio
import time
import functools
from typing import Any, Callable, Optional, Dict, List
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            # Success - reset failure count and close circuit if half-open
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED - service recovered")
            else:
                self.failure_count = 0
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(f"Circuit breaker OPEN - {self.failure_count} failures")
            
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs):
        """Execute async function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED - service recovered")
            else:
                self.failure_count = 0
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(f"Circuit breaker OPEN - {self.failure_count} failures")
            
            raise


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Retry decorator with exponential backoff"""
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                delay = initial_delay
                last_exception = None
                
                for attempt in range(max_retries):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            wait_time = min(delay, max_delay)
                            logger.warning(
                                f"Retry attempt {attempt + 1}/{max_retries} for {func.__name__} "
                                f"after {wait_time:.2f}s: {str(e)}"
                            )
                            await asyncio.sleep(wait_time)
                            delay *= exponential_base
                        else:
                            logger.error(f"All {max_retries} retry attempts failed for {func.__name__}")
                
                raise last_exception
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                delay = initial_delay
                last_exception = None
                
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            wait_time = min(delay, max_delay)
                            logger.warning(
                                f"Retry attempt {attempt + 1}/{max_retries} for {func.__name__} "
                                f"after {wait_time:.2f}s: {str(e)}"
                            )
                            time.sleep(wait_time)
                            delay *= exponential_base
                        else:
                            logger.error(f"All {max_retries} retry attempts failed for {func.__name__}")
                
                raise last_exception
            
            return sync_wrapper
    
    return decorator


def timeout(
    seconds: float,
    timeout_exception: Exception = TimeoutError("Operation timed out")
):
    """Timeout decorator for async functions"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"Timeout after {seconds}s for {func.__name__}")
                raise timeout_exception
        
        return wrapper
    return decorator


class Bulkhead:
    """Bulkhead pattern - limit concurrent executions"""
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.current_count = 0
    
    async def execute(self, func: Callable, *args, **kwargs):
        """Execute with concurrency limit"""
        async with self.semaphore:
            self.current_count += 1
            try:
                return await func(*args, **kwargs)
            finally:
                self.current_count -= 1


class IdempotencyKey:
    """Idempotency key management for safe retries"""
    
    def __init__(self):
        self.processed_keys: Dict[str, tuple[Any, datetime]] = {}
        self.ttl_seconds = 3600  # 1 hour
    
    async def check_and_store(
        self,
        key: str,
        executor: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Check idempotency key and execute if new, return cached result if exists"""
        # Clean expired keys
        now = datetime.utcnow()
        self.processed_keys = {
            k: v for k, (result, timestamp) in self.processed_keys.items()
            if (now - timestamp).total_seconds() < self.ttl_seconds
        }
        
        # Check if key exists
        if key in self.processed_keys:
            result, _ = self.processed_keys[key]
            logger.info(f"Idempotency key {key} found - returning cached result")
            return result
        
        # Execute and store
        result = await executor(*args, **kwargs)
        self.processed_keys[key] = (result, now)
        return result


def graceful_degradation(
    fallback_value: Any = None,
    fallback_func: Optional[Callable] = None
):
    """Graceful degradation - return fallback on failure"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Graceful degradation for {func.__name__}: {str(e)}")
                
                if fallback_func:
                    try:
                        return await fallback_func(*args, **kwargs)
                    except:
                        return fallback_value
                
                return fallback_value
        
        return wrapper
    return decorator


class HealthChecker:
    """Enhanced health checking with dependency verification"""
    
    def __init__(self):
        self.checks: List[Callable] = []
        self.status: Dict[str, bool] = {}
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check"""
        self.checks.append((name, check_func))
    
    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        all_healthy = True
        for name, check_func in self.checks:
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                results["checks"][name] = {
                    "status": "healthy" if result else "unhealthy",
                    "result": result
                }
                self.status[name] = result
                
                if not result:
                    all_healthy = False
            except Exception as e:
                results["checks"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                self.status[name] = False
                all_healthy = False
        
        results["status"] = "healthy" if all_healthy else "degraded"
        return results


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity  # max tokens
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens"""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


# Global circuit breakers for services
service_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(service_name: str) -> CircuitBreaker:
    """Get or create circuit breaker for a service"""
    if service_name not in service_circuit_breakers:
        service_circuit_breakers[service_name] = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
    return service_circuit_breakers[service_name]






