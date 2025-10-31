"""Advanced performance optimizations with intelligent caching and resource management."""

from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
from functools import wraps, lru_cache, partial
import asyncio
import time
import weakref
from collections import defaultdict, deque
import logging
import hashlib
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class IntelligentCache:
    """Intelligent cache with adaptive strategies."""
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl: float = 300.0,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    ):
        self._cache = {}
        self._timestamps = {}
        self._access_counts = defaultdict(int)
        self._access_times = deque()
        self._max_size = max_size
        self._ttl = ttl
        self._strategy = strategy
        self._metrics = CacheMetrics()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value with intelligent strategy."""
        if key not in self._cache:
            self._metrics.misses += 1
            return None
        
        # Check TTL
        if time.time() - self._timestamps[key] > self._ttl:
            await self._evict(key)
            self._metrics.misses += 1
            return None
        
        # Update access patterns
        self._access_counts[key] += 1
        self._access_times.append((key, time.time()))
        self._metrics.hits += 1
        
        return self._cache[key]
    
    async def set(self, key: str, value: Any) -> None:
        """Set value with intelligent eviction."""
        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size:
                await self._evict_intelligent()
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._access_counts[key] = 1
            self._access_times.append((key, time.time()))
            self._metrics.size = len(self._cache)
    
    async def _evict_intelligent(self) -> None:
        """Intelligent eviction based on strategy."""
        if not self._cache:
            return
        
        if self._strategy == CacheStrategy.LRU:
            await self._evict_lru()
        elif self._strategy == CacheStrategy.LFU:
            await self._evict_lfu()
        elif self._strategy == CacheStrategy.ADAPTIVE:
            await self._evict_adaptive()
    
    async def _evict_lru(self) -> None:
        """Evict least recently used."""
        lru_key = min(self._cache.keys(), key=lambda k: self._access_times.count(k))
        await self._evict(lru_key)
    
    async def _evict_lfu(self) -> None:
        """Evict least frequently used."""
        lfu_key = min(self._cache.keys(), key=lambda k: self._access_counts[k])
        await self._evict(lfu_key)
    
    async def _evict_adaptive(self) -> None:
        """Adaptive eviction based on access patterns."""
        # Combine recency and frequency
        scores = {}
        current_time = time.time()
        
        for key in self._cache.keys():
            recency = current_time - self._timestamps[key]
            frequency = self._access_counts[key]
            scores[key] = frequency / (recency + 1)  # Avoid division by zero
        
        evict_key = min(scores.keys(), key=lambda k: scores[k])
        await self._evict(evict_key)
    
    async def _evict(self, key: str) -> None:
        """Evict specific key."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._access_counts.pop(key, None)
        self._metrics.evictions += 1
        self._metrics.size = len(self._cache)
    
    def get_metrics(self) -> CacheMetrics:
        """Get cache metrics."""
        return self._metrics


class ResourcePool:
    """Intelligent resource pool with load balancing."""
    
    def __init__(self, factory: Callable[[], Any], max_size: int = 100):
        self._factory = factory
        self._pool = deque()
        self._max_size = max_size
        self._active_count = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> Any:
        """Acquire resource from pool."""
        async with self._lock:
            if self._pool:
                resource = self._pool.popleft()
            else:
                resource = self._factory()
            
            self._active_count += 1
            return resource
    
    async def release(self, resource: Any) -> None:
        """Release resource back to pool."""
        async with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(resource)
            
            self._active_count -= 1
    
    @property
    def pool_size(self) -> int:
        """Get current pool size."""
        return len(self._pool)
    
    @property
    def active_count(self) -> int:
        """Get active resource count."""
        return self._active_count


class CircuitBreaker:
    """Advanced circuit breaker with adaptive thresholds."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold
        self._state = "closed"  # closed, open, half_open
        self._failures = 0
        self._successes = 0
        self._last_failure = None
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable[[], Awaitable[Any]]) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self._state == "open":
                if time.time() - self._last_failure > self._recovery_timeout:
                    self._state = "half_open"
                    self._successes = 0
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = await func()
                await self._on_success()
                return result
            except Exception as e:
                await self._on_failure()
                raise e
    
    async def _on_success(self) -> None:
        """Handle successful call."""
        if self._state == "half_open":
            self._successes += 1
            if self._successes >= self._success_threshold:
                self._state = "closed"
                self._failures = 0
        else:
            self._failures = max(0, self._failures - 1)
    
    async def _on_failure(self) -> None:
        """Handle failed call."""
        self._failures += 1
        self._last_failure = time.time()
        
        if self._failures >= self._failure_threshold:
            self._state = "open"
    
    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state


class AdaptiveRateLimiter:
    """Adaptive rate limiter with dynamic thresholds."""
    
    def __init__(
        self,
        base_rate: int = 100,
        max_rate: int = 1000,
        window_size: float = 60.0
    ):
        self._base_rate = base_rate
        self._max_rate = max_rate
        self._window_size = window_size
        self._requests = defaultdict(list)
        self._adaptive_rate = base_rate
        self._last_adjustment = time.time()
    
    async def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        current_time = time.time()
        
        # Clean old requests
        self._requests[client_id] = [
            req_time for req_time in self._requests[client_id]
            if current_time - req_time < self._window_size
        ]
        
        # Adjust rate based on system load
        await self._adjust_rate()
        
        # Check rate limit
        if len(self._requests[client_id]) >= self._adaptive_rate:
            return False
        
        # Add current request
        self._requests[client_id].append(current_time)
        return True
    
    async def _adjust_rate(self) -> None:
        """Adjust rate based on system load."""
        current_time = time.time()
        
        if current_time - self._last_adjustment > 10.0:  # Adjust every 10 seconds
            # Calculate system load (simplified)
            total_requests = sum(len(requests) for requests in self._requests.values())
            load_factor = total_requests / (self._base_rate * len(self._requests))
            
            if load_factor > 1.5:
                self._adaptive_rate = max(self._base_rate, self._adaptive_rate - 10)
            elif load_factor < 0.5:
                self._adaptive_rate = min(self._max_rate, self._adaptive_rate + 10)
            
            self._last_adjustment = current_time


class PerformanceMonitor:
    """Advanced performance monitoring with metrics aggregation."""
    
    def __init__(self):
        self._metrics = defaultdict(list)
        self._counters = defaultdict(int)
        self._gauges = defaultdict(float)
        self._start_time = time.time()
    
    def record_latency(self, operation: str, latency: float) -> None:
        """Record operation latency."""
        self._metrics[f"{operation}_latency"].append(latency)
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment counter."""
        self._counters[name] += value
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set gauge value."""
        self._gauges[name] = value
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        uptime = time.time() - self._start_time
        
        summary = {
            "uptime": uptime,
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "latency_metrics": {}
        }
        
        # Calculate latency statistics
        for metric_name, values in self._metrics.items():
            if values:
                summary["latency_metrics"][metric_name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "p95": self._percentile(values, 0.95),
                    "p99": self._percentile(values, 0.99)
                }
        
        return summary
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile."""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]


def intelligent_cache(maxsize: int = 1000, ttl: float = 300.0):
    """Intelligent caching decorator."""
    cache = IntelligentCache(maxsize, ttl)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create intelligent cache key
            key_data = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try cache first
            cached_result = await cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(key, result)
            return result
        
        return wrapper
    return decorator


def adaptive_retry(
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
):
    """Adaptive retry with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        await asyncio.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator


def performance_monitor(operation_name: str):
    """Performance monitoring decorator."""
    monitor = PerformanceMonitor()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                monitor.increment_counter(f"{operation_name}_success")
                return result
            except Exception as e:
                monitor.increment_counter(f"{operation_name}_error")
                raise e
            finally:
                duration = time.time() - start_time
                monitor.record_latency(operation_name, duration)
        
        return wrapper
    return decorator


def circuit_breaker_protection(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
):
    """Circuit breaker protection decorator."""
    breaker = CircuitBreaker(failure_threshold, recovery_timeout)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(lambda: func(*args, **kwargs))
        return wrapper
    return decorator


async def intelligent_batch_process(
    items: List[Any],
    processor: Callable[[Any], Awaitable[Any]],
    max_concurrent: int = 100,
    chunk_size: int = 50,
    adaptive: bool = True
) -> List[Any]:
    """Intelligent batch processing with adaptive concurrency."""
    results = []
    
    if adaptive:
        # Start with conservative concurrency and adapt
        current_concurrency = min(max_concurrent // 2, 10)
    else:
        current_concurrency = max_concurrent
    
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i + chunk_size]
        
        # Use semaphore for concurrency control
        semaphore = asyncio.Semaphore(current_concurrency)
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await processor(item)
        
        # Process chunk concurrently
        chunk_results = await asyncio.gather(
            *[process_with_semaphore(item) for item in chunk],
            return_exceptions=True
        )
        
        results.extend(chunk_results)
        
        # Adapt concurrency based on results
        if adaptive:
            success_rate = sum(1 for r in chunk_results if not isinstance(r, Exception)) / len(chunk_results)
            if success_rate > 0.9:
                current_concurrency = min(max_concurrent, current_concurrency + 5)
            elif success_rate < 0.7:
                current_concurrency = max(5, current_concurrency - 5)
    
    return results


def create_resource_pool(factory: Callable[[], Any], max_size: int = 100) -> ResourcePool:
    """Create resource pool."""
    return ResourcePool(factory, max_size)


def create_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
) -> CircuitBreaker:
    """Create circuit breaker."""
    return CircuitBreaker(failure_threshold, recovery_timeout)


def create_rate_limiter(
    base_rate: int = 100,
    max_rate: int = 1000
) -> AdaptiveRateLimiter:
    """Create adaptive rate limiter."""
    return AdaptiveRateLimiter(base_rate, max_rate)


def create_performance_monitor() -> PerformanceMonitor:
    """Create performance monitor."""
    return PerformanceMonitor()
