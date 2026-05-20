from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import weakref
from typing import Dict, Any, Optional, Callable, TypeVar, Union, List
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging
from functools import wraps
import hashlib
import json
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, List, Dict, Optional
"""
Blocking Operations Limiter for Instagram Captions API v14.0

Comprehensive system for limiting blocking operations in routes:
- Rate limiting per user/IP
- Concurrency control
- Timeout management
- Circuit breaker pattern
- Resource usage tracking
- Adaptive throttling
"""


logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class OperationType(Enum):
    """Types of operations that can be limited"""
    CAPTION_GENERATION = "caption_generation"
    BATCH_PROCESSING = "batch_processing"
    AI_MODEL_LOADING = "ai_model_loading"
    DATABASE_QUERY = "database_query"
    EXTERNAL_API_CALL = "external_api_call"
    FILE_OPERATION = "file_operation"
    CACHE_OPERATION = "cache_operation"
    HEAVY_COMPUTATION = "heavy_computation"


class LimiterState(Enum):
    """States of the rate limiter"""
    ALLOW = "allow"
    THROTTLE = "throttle"
    BLOCK = "block"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_HALF_OPEN = "circuit_half_open"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10
    window_size: int = 60  # seconds
    cooldown_period: int = 300  # seconds


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrency control"""
    max_concurrent_requests: int = 50
    max_concurrent_per_user: int = 5
    queue_size: int = 100
    timeout_seconds: int = 30


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    monitor_interval: int = 10


@dataclass
class OperationMetrics:
    """Metrics for operation tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    blocked_requests: int = 0
    throttled_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: float = 0.0
    error_rate: float = 0.0


class TokenBucket:
    """Token bucket algorithm for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        
    """__init__ function."""
self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from the bucket"""
        async with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self) -> Any:
        """Refill tokens based on time elapsed"""
        now = time.time()
        time_passed = now - self.last_refill
        tokens_to_add = time_passed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now


class SlidingWindowCounter:
    """Sliding window counter for rate limiting"""
    
    def __init__(self, window_size: int = 60):
        
    """__init__ function."""
self.window_size = window_size
        self.requests = deque()
        self._lock = asyncio.Lock()
    
    async async def add_request(self, timestamp: float = None) -> int:
        """Add a request and return current count"""
        if timestamp is None:
            timestamp = time.time()
        
        async with self._lock:
            # Remove old requests outside the window
            cutoff = timestamp - self.window_size
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            # Add new request
            self.requests.append(timestamp)
            return len(self.requests)
    
    async def get_count(self, timestamp: float = None) -> int:
        """Get current request count"""
        if timestamp is None:
            timestamp = time.time()
        
        async with self._lock:
            cutoff = timestamp - self.window_size
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            return len(self.requests)


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, config: CircuitBreakerConfig):
        
    """__init__ function."""
self.config = config
        self.state = LimiterState.ALLOW
        self.failure_count = 0
        self.last_failure_time = 0
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == LimiterState.CIRCUIT_OPEN:
            if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                self.state = LimiterState.CIRCUIT_HALF_OPEN
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self.config.expected_exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self) -> Any:
        """Handle successful operation"""
        async with self._lock:
            if self.state == LimiterState.CIRCUIT_HALF_OPEN:
                self.state = LimiterState.ALLOW
            self.failure_count = 0
    
    async def _on_failure(self) -> Any:
        """Handle failed operation"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = LimiterState.CIRCUIT_OPEN


class ConcurrencyLimiter:
    """Concurrency control for operations"""
    
    def __init__(self, config: ConcurrencyConfig):
        
    """__init__ function."""
self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.user_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.user_locks: Dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
    
    async def get_user_semaphore(self, user_id: str) -> asyncio.Semaphore:
        """Get or create semaphore for user"""
        async with self._lock:
            if user_id not in self.user_semaphores:
                self.user_semaphores[user_id] = asyncio.Semaphore(self.config.max_concurrent_per_user)
                self.user_locks[user_id] = asyncio.Lock()
            return self.user_semaphores[user_id]
    
    @asynccontextmanager
    async def limit_concurrency(self, user_id: str = None):
        """Context manager for concurrency limiting"""
        # Global semaphore
        async with self.semaphore:
            # User-specific semaphore
            if user_id:
                user_sem = await self.get_user_semaphore(user_id)
                async with user_sem:
                    yield
            else:
                yield
    
    async def run_in_thread_pool(self, func: Callable, *args, **kwargs) -> Any:
        """Run blocking function in thread pool"""
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(self.thread_pool, func, *args, **kwargs),
            timeout=self.config.timeout_seconds
        )


class BlockingOperationsLimiter:
    """Main limiter for blocking operations"""
    
    def __init__(self) -> Any:
        self.rate_limiters: Dict[str, TokenBucket] = {}
        self.window_counters: Dict[str, SlidingWindowCounter] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.concurrency_limiters: Dict[str, ConcurrencyLimiter] = {}
        self.metrics: Dict[str, OperationMetrics] = defaultdict(OperationMetrics)
        self._lock = asyncio.Lock()
        
        # Default configurations
        self.default_rate_config = RateLimitConfig()
        self.default_concurrency_config = ConcurrencyConfig()
        self.default_circuit_config = CircuitBreakerConfig()
        
        # Operation type configurations
        self.operation_configs = {
            OperationType.CAPTION_GENERATION: {
                "rate_limit": RateLimitConfig(requests_per_minute=30, requests_per_hour=500),
                "concurrency": ConcurrencyConfig(max_concurrent_requests=20, max_concurrent_per_user=3),
                "circuit_breaker": CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30)
            },
            OperationType.BATCH_PROCESSING: {
                "rate_limit": RateLimitConfig(requests_per_minute=10, requests_per_hour=100),
                "concurrency": ConcurrencyConfig(max_concurrent_requests=10, max_concurrent_per_user=2),
                "circuit_breaker": CircuitBreakerConfig(failure_threshold=2, recovery_timeout=60)
            },
            OperationType.AI_MODEL_LOADING: {
                "rate_limit": RateLimitConfig(requests_per_minute=5, requests_per_hour=50),
                "concurrency": ConcurrencyConfig(max_concurrent_requests=5, max_concurrent_per_user=1),
                "circuit_breaker": CircuitBreakerConfig(failure_threshold=2, recovery_timeout=120)
            },
            OperationType.DATABASE_QUERY: {
                "rate_limit": RateLimitConfig(requests_per_minute=100, requests_per_hour=1000),
                "concurrency": ConcurrencyConfig(max_concurrent_requests=50, max_concurrent_per_user=10),
                "circuit_breaker": CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30)
            },
            OperationType.EXTERNAL_API_CALL: {
                "rate_limit": RateLimitConfig(requests_per_minute=60, requests_per_hour=500),
                "concurrency": ConcurrencyConfig(max_concurrent_requests=30, max_concurrent_per_user=5),
                "circuit_breaker": CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60)
            }
        }
    
    def _get_limiter_key(self, operation_type: OperationType, identifier: str) -> str:
        """Generate unique key for limiter"""
        return f"{operation_type.value}:{identifier}"
    
    async def get_rate_limiter(self, operation_type: OperationType, identifier: str) -> TokenBucket:
        """Get or create rate limiter"""
        key = self._get_limiter_key(operation_type, identifier)
        
        async with self._lock:
            if key not in self.rate_limiters:
                config = self.operation_configs.get(operation_type, {}).get("rate_limit", self.default_rate_config)
                refill_rate = config.requests_per_minute / 60.0
                self.rate_limiters[key] = TokenBucket(config.burst_limit, refill_rate)
            
            return self.rate_limiters[key]
    
    async def get_window_counter(self, operation_type: OperationType, identifier: str) -> SlidingWindowCounter:
        """Get or create window counter"""
        key = self._get_limiter_key(operation_type, identifier)
        
        async with self._lock:
            if key not in self.window_counters:
                config = self.operation_configs.get(operation_type, {}).get("rate_limit", self.default_rate_config)
                self.window_counters[key] = SlidingWindowCounter(config.window_size)
            
            return self.window_counters[key]
    
    async def get_circuit_breaker(self, operation_type: OperationType, identifier: str) -> CircuitBreaker:
        """Get or create circuit breaker"""
        key = self._get_limiter_key(operation_type, identifier)
        
        async with self._lock:
            if key not in self.circuit_breakers:
                config = self.operation_configs.get(operation_type, {}).get("circuit_breaker", self.default_circuit_config)
                self.circuit_breakers[key] = CircuitBreaker(config)
            
            return self.circuit_breakers[key]
    
    async def get_concurrency_limiter(self, operation_type: OperationType) -> ConcurrencyLimiter:
        """Get or create concurrency limiter"""
        key = operation_type.value
        
        async with self._lock:
            if key not in self.concurrency_limiters:
                config = self.operation_configs.get(operation_type, {}).get("concurrency", self.default_concurrency_config)
                self.concurrency_limiters[key] = ConcurrencyLimiter(config)
            
            return self.concurrency_limiters[key]
    
    async def check_rate_limit(self, operation_type: OperationType, identifier: str) -> bool:
        """Check if rate limit allows the operation"""
        # Check token bucket
        rate_limiter = await self.get_rate_limiter(operation_type, identifier)
        if not await rate_limiter.consume():
            await self._record_metric(operation_type, "throttled_requests")
            return False
        
        # Check sliding window
        window_counter = await self.get_window_counter(operation_type, identifier)
        config = self.operation_configs.get(operation_type, {}).get("rate_limit", self.default_rate_config)
        current_count = await window_counter.add_request()
        
        if current_count > config.requests_per_minute:
            await self._record_metric(operation_type, "throttled_requests")
            return False
        
        return True
    
    async def execute_with_limits(
        self,
        operation_type: OperationType,
        func: Callable,
        identifier: str = "default",
        user_id: str = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with all limits applied"""
        start_time = time.time()
        
        try:
            # Check rate limit
            if not await self.check_rate_limit(operation_type, identifier):
                raise Exception(f"Rate limit exceeded for {operation_type.value}")
            
            # Get concurrency limiter
            concurrency_limiter = await self.get_concurrency_limiter(operation_type)
            
            # Get circuit breaker
            circuit_breaker = await self.get_circuit_breaker(operation_type, identifier)
            
            # Execute with concurrency and circuit breaker
            async with concurrency_limiter.limit_concurrency(user_id):
                result = await circuit_breaker.call(func, *args, **kwargs)
            
            # Record success
            processing_time = time.time() - start_time
            await self._record_success(operation_type, processing_time)
            
            return result
            
        except Exception as e:
            # Record failure
            processing_time = time.time() - start_time
            await self._record_failure(operation_type, processing_time, str(e))
            raise
    
    async def execute_blocking_in_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        self,
        operation_type: OperationType,
        func: Callable,
        identifier: str = "default",
        user_id: str = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute blocking function in thread pool with limits"""
        async def async_wrapper():
            
    """async_wrapper function."""
concurrency_limiter = await self.get_concurrency_limiter(operation_type)
            return await concurrency_limiter.run_in_thread_pool(func, *args, **kwargs)
        
        return await self.execute_with_limits(
            operation_type, async_wrapper, identifier, user_id
        )
    
    async def _record_metric(self, operation_type: OperationType, metric: str):
        """Record operation metric"""
        key = operation_type.value
        self.metrics[key].total_requests += 1
        
        if metric == "successful_requests":
            self.metrics[key].successful_requests += 1
        elif metric == "failed_requests":
            self.metrics[key].failed_requests += 1
        elif metric == "blocked_requests":
            self.metrics[key].blocked_requests += 1
        elif metric == "throttled_requests":
            self.metrics[key].throttled_requests += 1
    
    async def _record_success(self, operation_type: OperationType, processing_time: float):
        """Record successful operation"""
        key = operation_type.value
        metrics = self.metrics[key]
        
        metrics.successful_requests += 1
        metrics.total_requests += 1
        metrics.last_request_time = time.time()
        
        # Update average response time
        if metrics.successful_requests == 1:
            metrics.average_response_time = processing_time
        else:
            metrics.average_response_time = (
                (metrics.average_response_time * (metrics.successful_requests - 1) + processing_time) /
                metrics.successful_requests
            )
        
        # Update error rate
        metrics.error_rate = metrics.failed_requests / max(1, metrics.total_requests)
    
    async def _record_failure(self, operation_type: OperationType, processing_time: float, error: str):
        """Record failed operation"""
        key = operation_type.value
        metrics = self.metrics[key]
        
        metrics.failed_requests += 1
        metrics.total_requests += 1
        metrics.last_request_time = time.time()
        metrics.error_rate = metrics.failed_requests / max(1, metrics.total_requests)
        
        logger.warning(f"Operation {operation_type.value} failed: {error}")
    
    async def get_metrics(self, operation_type: Optional[OperationType] = None) -> Dict[str, Any]:
        """Get operation metrics"""
        if operation_type:
            key = operation_type.value
            return {
                key: self.metrics[key].__dict__
            }
        else:
            return {
                key: metrics.__dict__ for key, metrics in self.metrics.items()
            }
    
    async def reset_metrics(self, operation_type: Optional[OperationType] = None):
        """Reset operation metrics"""
        if operation_type:
            key = operation_type.value
            self.metrics[key] = OperationMetrics()
        else:
            self.metrics.clear()


# Global limiter instance
blocking_limiter = BlockingOperationsLimiter()


# Decorators for easy usage
def limit_blocking_operations(
    operation_type: OperationType,
    identifier: str = "default",
    user_id_param: str = "user_id"
):
    """Decorator to limit blocking operations"""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Extract user_id from kwargs or request
            user_id = kwargs.get(user_id_param, "anonymous")
            
            return await blocking_limiter.execute_with_limits(
                operation_type, func, identifier, user_id, *args, **kwargs
            )
        return wrapper
    return decorator


def limit_blocking_thread_operations(
    operation_type: OperationType,
    identifier: str = "default",
    user_id_param: str = "user_id"
):
    """Decorator to limit blocking operations in thread pool"""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Extract user_id from kwargs or request
            user_id = kwargs.get(user_id_param, "anonymous")
            
            return await blocking_limiter.execute_blocking_in_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                operation_type, func, identifier, user_id, *args, **kwargs
            )
        return wrapper
    return decorator


# Context managers for manual control
@asynccontextmanager
async def rate_limit_context(operation_type: OperationType, identifier: str = "default"):
    """Context manager for rate limiting"""
    if not await blocking_limiter.check_rate_limit(operation_type, identifier):
        raise Exception(f"Rate limit exceeded for {operation_type.value}")
    try:
        yield
    finally:
        pass


@asynccontextmanager
async def concurrency_limit_context(operation_type: OperationType, user_id: str = None):
    """Context manager for concurrency limiting"""
    concurrency_limiter = await blocking_limiter.get_concurrency_limiter(operation_type)
    async with concurrency_limiter.limit_concurrency(user_id):
        yield


@asynccontextmanager
async def circuit_breaker_context(operation_type: OperationType, identifier: str = "default"):
    """Context manager for circuit breaker"""
    circuit_breaker = await blocking_limiter.get_circuit_breaker(operation_type, identifier)
    # Note: Circuit breaker is applied at function level, not context level
    yield 