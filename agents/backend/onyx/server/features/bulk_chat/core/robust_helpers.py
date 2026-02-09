"""
Robust Helpers for Bulk Chat
=============================

Advanced robustness utilities for bulk operations with circuit breakers, retry logic, and rate limiting.
"""

import asyncio
import logging
import time
import json
import uuid
from functools import wraps
from typing import Callable, Any, Tuple, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RobustRetry:
    """Decorator for robust retry logic with exponential backoff and jitter."""
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0, retryable_exceptions: Tuple[type, ...] = (Exception,)):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retryable_exceptions = retryable_exceptions

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            for attempt in range(self.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except self.retryable_exceptions as e:
                    if attempt == self.max_attempts - 1:
                        logger.error(f"Function {func.__name__} failed after {self.max_attempts} attempts. Last error: {e}")
                        raise
                    
                    delay = min(self.max_delay, self.base_delay * (2 ** attempt))
                    jitter = delay * 0.1 * (2 * (time.monotonic() % 1) - 1)  # +/- 10% jitter
                    final_delay = max(0, delay + jitter)
                    
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{self.max_attempts}). Retrying in {final_delay:.2f}s. Error: {e}")
                    await asyncio.sleep(final_delay)
        return wrapper

class CircuitBreaker:
    """Implements the Circuit Breaker pattern for resilience."""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, name: str = "default"):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = "closed"
        self.failures = 0
        self.last_failure_time = None
        self.lock = asyncio.Lock()
        logger.info(f"Circuit Breaker '{self.name}' initialized. Threshold: {failure_threshold}, Timeout: {recovery_timeout}s")

    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        async with self.lock:
            if self.state == "open":
                if time.monotonic() > self.last_failure_time + self.recovery_timeout:
                    self.state = "half-open"
                    logger.warning(f"Circuit Breaker '{self.name}' moved to HALF-OPEN state.")
                else:
                    logger.warning(f"Circuit Breaker '{self.name}' is OPEN. Skipping call to {func.__name__}.")
                    raise CircuitBreakerOpenException(f"Circuit Breaker '{self.name}' is OPEN.")
            
            if self.state == "half-open":
                try:
                    result = await func(*args, **kwargs)
                    self.state = "closed"
                    self.failures = 0
                    logger.info(f"Circuit Breaker '{self.name}' moved to CLOSED state (success in half-open).")
                    return result
                except Exception as e:
                    self.state = "open"
                    self.last_failure_time = time.monotonic()
                    logger.error(f"Circuit Breaker '{self.name}' moved to OPEN state (failure in half-open). Error: {e}")
                    raise
            
            # State is closed
            try:
                result = await func(*args, **kwargs)
                self.failures = 0  # Reset failures on success
                return result
            except Exception as e:
                self.failures += 1
                logger.warning(f"Circuit Breaker '{self.name}' recorded failure {self.failures}/{self.failure_threshold}. Error: {e}")
                if self.failures >= self.failure_threshold:
                    self.state = "open"
                    self.last_failure_time = time.monotonic()
                    logger.error(f"Circuit Breaker '{self.name}' moved to OPEN state after {self.failures} failures.")
                raise

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass

class RateLimiter:
    """Implements a Token Bucket algorithm for rate limiting."""
    def __init__(self, rate: float, capacity: float):
        self.rate = rate  # tokens per second
        self.capacity = capacity  # max tokens in bucket
        self.tokens = capacity
        self.last_refill_time = time.monotonic()
        self.lock = asyncio.Lock()
        logger.info(f"Rate Limiter initialized. Rate: {rate} req/s, Capacity: {capacity} burst.")

    async def acquire(self, tokens_needed: float = 1.0) -> bool:
        async with self.lock:
            self._refill()
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            return False

    async def wait(self, tokens_needed: float = 1.0):
        """Wait until tokens are available."""
        while not await self.acquire(tokens_needed):
            await asyncio.sleep(0.1)  # Small sleep to prevent busy-waiting

    def _refill(self):
        now = time.monotonic()
        time_passed = now - self.last_refill_time
        self.last_refill_time = now
        self.tokens = min(self.capacity, self.tokens + time_passed * self.rate)

def validate_input(data: Dict[str, Any], required_fields: List[str], field_validators: Optional[Dict[str, Callable]] = None) -> Tuple[bool, str]:
    """Validates input data against required fields and custom validators."""
    for field in required_fields:
        if field not in data or data[field] is None:
            return False, f"Missing required field: {field}"
    
    if field_validators:
        for field, validator_func in field_validators.items():
            if field in data and not validator_func(data[field]):
                return False, f"Validation failed for field: {field}"
    
    return True, ""

def safe_json_dumps(data: Any, **kwargs) -> str:
    """Safely dumps data to JSON, handling non-serializable types."""
    def default_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    
    return json.dumps(data, default=default_serializer, **kwargs)

def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely loads JSON data, returning a default value on error."""
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON: {e}. Data: {data[:200]}...")
        return default

def generate_id(prefix: str = "id") -> str:
    """Generates a unique ID with a given prefix."""
    return f"{prefix}_{uuid.uuid4()}"

# Global instances
circuit_breaker = CircuitBreaker(name="bulk_operations", failure_threshold=5, recovery_timeout=60.0)
rate_limiter = RateLimiter(rate=10.0, capacity=50.0)
















