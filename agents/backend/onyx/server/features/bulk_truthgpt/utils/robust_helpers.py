"""
Robust Helper Functions
=======================

Utility functions for robust error handling, validation, and system resilience.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, TypeVar, List
from functools import wraps
from datetime import datetime, timedelta
import hashlib
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')

class RobustRetry:
    """Robust retry mechanism with exponential backoff and jitter."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except self.retryable_exceptions as e:
                last_exception = e
                
                if attempt == self.max_attempts:
                    logger.error(f"All {self.max_attempts} attempts failed. Last error: {e}")
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.base_delay * (self.exponential_base ** (attempt - 1)),
                    self.max_delay
                )
                
                # Add jitter
                if self.jitter:
                    import random
                    delay = delay * (0.5 + random.random())
                
                logger.warning(f"Attempt {attempt}/{self.max_attempts} failed: {e}. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
        
        raise last_exception

def robust_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
):
    """Decorator for robust retry logic."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry = RobustRetry(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay
            )
            return await retry.execute(func, *args, **kwargs)
        return wrapper
    return decorator

class CircuitBreaker:
    """Improved circuit breaker with half-open state support."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        name: str = "circuit_breaker"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.success_count = 0
        self.half_open_threshold = 3  # Successes needed to close circuit
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker."""
        # Check if circuit should be reset
        if self.state == "open":
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = "half-open"
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} moved to HALF-OPEN")
        
        # Execute function
        try:
            if asyncio.iscoroutinefunction(func):
                result = asyncio.run(func(*args, **kwargs))
            else:
                result = func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == "half-open":
                self.success_count += 1
                if self.success_count >= self.half_open_threshold:
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info(f"Circuit breaker {self.name} CLOSED after {self.success_count} successes")
            else:
                self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker {self.name} OPENED after {self.failure_count} failures")
            
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs):
        """Async version of call."""
        # Check if circuit should be reset
        if self.state == "open":
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = "half-open"
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} moved to HALF-OPEN")
            
            # Still open - raise exception
            raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        # Execute function
        try:
            result = await func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == "half-open":
                self.success_count += 1
                if self.success_count >= self.half_open_threshold:
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info(f"Circuit breaker {self.name} CLOSED after {self.success_count} successes")
            else:
                self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker {self.name} OPENED after {self.failure_count} failures")
            
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count if self.state == "half-open" else 0,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "threshold": self.failure_threshold
        }

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, capacity: float):
        """
        Args:
            rate: Tokens per second
            capacity: Maximum tokens
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: float = 1.0) -> bool:
        """Acquire tokens from bucket."""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait(self, tokens: float = 1.0):
        """Wait until tokens are available."""
        while not await self.acquire(tokens):
            wait_time = (tokens - self.tokens) / self.rate
            await asyncio.sleep(min(wait_time, 1.0))

def validate_input(
    data: Dict[str, Any],
    required_fields: List[str],
    field_validators: Optional[Dict[str, Callable]] = None
) -> tuple[bool, Optional[str]]:
    """
    Validate input data.
    
    Returns:
        (is_valid, error_message)
    """
    # Check required fields
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
        
        if data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
            return False, f"Field {field} cannot be empty"
    
    # Run custom validators
    if field_validators:
        for field, validator in field_validators.items():
            if field in data:
                try:
                    if not validator(data[field]):
                        return False, f"Validation failed for field: {field}"
                except Exception as e:
                    return False, f"Validation error for field {field}: {str(e)}"
    
    return True, None

def safe_json_dumps(data: Any, default: Any = None) -> str:
    """Safely serialize data to JSON."""
    try:
        return json.dumps(data, default=default or str, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to serialize data to JSON: {e}")
        return json.dumps({"error": "Serialization failed"})

def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely deserialize JSON data."""
    try:
        return json.loads(data)
    except Exception as e:
        logger.error(f"Failed to deserialize JSON: {e}")
        return default

def generate_id(prefix: str = "id") -> str:
    """Generate unique ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    return f"{prefix}_{timestamp}_{random_part}"

def measure_time(func: Callable) -> Callable:
    """Decorator to measure execution time."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

class HealthChecker:
    """Health check manager."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.status: Dict[str, bool] = {}
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check."""
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                is_healthy = result if isinstance(result, bool) else result.get("healthy", False)
                self.status[name] = is_healthy
                results[name] = {
                    "healthy": is_healthy,
                    "details": result if not isinstance(result, bool) else {}
                }
                
                if not is_healthy:
                    overall_healthy = False
            except Exception as e:
                self.status[name] = False
                results[name] = {
                    "healthy": False,
                    "error": str(e)
                }
                overall_healthy = False
        
        return {
            "overall": "healthy" if overall_healthy else "unhealthy",
            "checks": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_status(self, name: str) -> bool:
        """Get status of a specific check."""
        return self.status.get(name, False)
































