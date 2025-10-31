"""Ultra-efficient async operations with minimal overhead."""

from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
from functools import wraps, lru_cache
import asyncio
import time
import weakref
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# --- Ultra-Fast Caching ---
class UltraFastCache:
    """Ultra-fast in-memory cache with minimal overhead."""
    
    def __init__(self, max_size: int = 10000, ttl: float = 300.0):
        self._cache = {}
        self._timestamps = {}
        self._access_counts = defaultdict(int)
        self._max_size = max_size
        self._ttl = ttl
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value with O(1) lookup."""
        if key not in self._cache:
            return None
        
        # Check TTL
        if time.time() - self._timestamps[key] > self._ttl:
            await self._evict(key)
            return None
        
        self._access_counts[key] += 1
        return self._cache[key]
    
    async def set(self, key: str, value: Any) -> None:
        """Set value with automatic eviction."""
        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size:
                await self._evict_lru()
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._access_counts[key] = 1
    
    async def _evict(self, key: str) -> None:
        """Evict specific key."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._access_counts.pop(key, None)
    
    async def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._cache:
            return
        
        # Find LRU key
        lru_key = min(self._cache.keys(), key=lambda k: self._access_counts[k])
        await self._evict(lru_key)


# --- Ultra-Fast Decorators ---
def ultra_fast_cache(maxsize: int = 1000, ttl: float = 60.0):
    """Ultra-fast async caching decorator."""
    cache = UltraFastCache(maxsize, ttl)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
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


def ultra_fast_retry(max_retries: int = 3, base_delay: float = 0.1):
    """Ultra-fast retry with minimal delay."""
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
                        await asyncio.sleep(base_delay * (2 ** attempt))
            
            raise last_exception
        return wrapper
    return decorator


def ultra_fast_timeout(timeout: float = 5.0):
    """Ultra-fast timeout decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        return wrapper
    return decorator


# --- Ultra-Fast Batch Processing ---
async def ultra_fast_batch_process(
    items: List[Any],
    processor: Callable[[Any], Awaitable[Any]],
    max_concurrent: int = 100,
    chunk_size: int = 50
) -> List[Any]:
    """Ultra-fast batch processing with optimal concurrency."""
    results = []
    
    # Process in chunks for memory efficiency
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i + chunk_size]
        
        # Use semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await processor(item)
        
        # Process chunk concurrently
        chunk_results = await asyncio.gather(
            *[process_with_semaphore(item) for item in chunk],
            return_exceptions=True
        )
        
        results.extend(chunk_results)
    
    return results


# --- Ultra-Fast Parallel Operations ---
async def ultra_fast_parallel(
    operations: List[Callable[[], Awaitable[Any]]],
    max_concurrent: int = 50
) -> List[Any]:
    """Ultra-fast parallel execution with optimal concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_with_semaphore(op):
        async with semaphore:
            return await op()
    
    return await asyncio.gather(
        *[execute_with_semaphore(op) for op in operations],
        return_exceptions=True
    )


# --- Ultra-Fast Data Processing ---
def ultra_fast_map(
    data: List[Any],
    mapper: Callable[[Any], Any],
    chunk_size: int = 1000
) -> List[Any]:
    """Ultra-fast mapping with chunking."""
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunk_results = [mapper(item) for item in chunk]
        results.extend(chunk_results)
    
    return results


def ultra_fast_filter(
    data: List[Any],
    predicate: Callable[[Any], bool],
    chunk_size: int = 1000
) -> List[Any]:
    """Ultra-fast filtering with chunking."""
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunk_results = [item for item in chunk if predicate(item)]
        results.extend(chunk_results)
    
    return results


# --- Ultra-Fast Memory Management ---
class UltraFastPool:
    """Ultra-fast object pool for memory efficiency."""
    
    def __init__(self, factory: Callable[[], Any], max_size: int = 100):
        self._factory = factory
        self._pool = []
        self._max_size = max_size
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> Any:
        """Acquire object from pool."""
        async with self._lock:
            if self._pool:
                return self._pool.pop()
            return self._factory()
    
    async def release(self, obj: Any) -> None:
        """Release object back to pool."""
        async with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(obj)


# --- Ultra-Fast Metrics ---
class UltraFastMetrics:
    """Ultra-fast metrics collection with minimal overhead."""
    
    def __init__(self):
        self._counters = defaultdict(int)
        self._timers = defaultdict(list)
        self._gauges = defaultdict(float)
        self._lock = asyncio.Lock()
    
    def increment(self, name: str, value: int = 1) -> None:
        """Increment counter (lock-free for single-threaded)."""
        self._counters[name] += value
    
    def record_time(self, name: str, duration: float) -> None:
        """Record timing (lock-free for single-threaded)."""
        self._timers[name].append(duration)
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set gauge value (lock-free for single-threaded)."""
        self._gauges[name] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "counters": dict(self._counters),
            "timers": {
                name: {
                    "count": len(times),
                    "avg": sum(times) / len(times) if times else 0,
                    "min": min(times) if times else 0,
                    "max": max(times) if times else 0
                }
                for name, times in self._timers.items()
            },
            "gauges": dict(self._gauges)
        }


# --- Ultra-Fast Validation ---
def ultra_fast_validate(
    data: Any,
    validators: List[Callable[[Any], bool]]
) -> bool:
    """Ultra-fast validation with early exit."""
    for validator in validators:
        if not validator(data):
            return False
    return True


def ultra_fast_sanitize(text: str) -> str:
    """Ultra-fast text sanitization."""
    if not text:
        return ""
    
    # Use built-in methods for speed
    return text.strip()[:1000]  # Limit length


# --- Ultra-Fast Serialization ---
def ultra_fast_serialize(data: Any) -> str:
    """Ultra-fast serialization."""
    if isinstance(data, str):
        return data
    if isinstance(data, (int, float, bool)):
        return str(data)
    if isinstance(data, dict):
        return str(data)
    return str(data)


def ultra_fast_deserialize(data: str, target_type: type) -> Any:
    """Ultra-fast deserialization."""
    if target_type == str:
        return data
    if target_type == int:
        return int(data)
    if target_type == float:
        return float(data)
    if target_type == bool:
        return data.lower() == 'true'
    return data


# --- Ultra-Fast File Operations ---
async def ultra_fast_read_file(file_path: str) -> bytes:
    """Ultra-fast file reading."""
    with open(file_path, 'rb') as f:
        return f.read()


async def ultra_fast_write_file(file_path: str, data: bytes) -> None:
    """Ultra-fast file writing."""
    with open(file_path, 'wb') as f:
        f.write(data)


# --- Ultra-Fast Network Operations ---
async def ultra_fast_request(
    url: str,
    method: str = "GET",
    timeout: float = 5.0
) -> Dict[str, Any]:
    """Ultra-fast HTTP request."""
    import aiohttp
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.request(method, url) as response:
            return {
                "status": response.status,
                "data": await response.text(),
                "headers": dict(response.headers)
            }


# --- Ultra-Fast Database Operations ---
class UltraFastDB:
    """Ultra-fast database operations."""
    
    def __init__(self, connection_string: str):
        self._connection_string = connection_string
        self._pool = None
    
    async def connect(self) -> None:
        """Connect to database."""
        # Implementation depends on database type
        pass
    
    async def execute(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute query with minimal overhead."""
        # Implementation depends on database type
        return []
    
    async def close(self) -> None:
        """Close database connection."""
        pass


# --- Ultra-Fast Configuration ---
@lru_cache(maxsize=1)
def ultra_fast_get_config() -> Dict[str, Any]:
    """Ultra-fast configuration loading with caching."""
    return {
        "max_concurrent": 100,
        "cache_ttl": 60.0,
        "timeout": 5.0,
        "chunk_size": 1000
    }


# --- Ultra-Fast Error Handling ---
def ultra_fast_error_handler(func: Callable) -> Callable:
    """Ultra-fast error handling decorator."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper


# --- Ultra-Fast Health Check ---
async def ultra_fast_health_check() -> Dict[str, Any]:
    """Ultra-fast health check."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - time.time()  # Placeholder
    }
