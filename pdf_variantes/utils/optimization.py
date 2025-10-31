"""
PDF Variantes API - Optimization Utilities
Performance optimization utilities and helpers
"""

import functools
import time
import asyncio
from typing import Any, Callable, Optional, Dict, List
from collections import OrderedDict
from functools import lru_cache
import hashlib
import json

logger = None


def cached_async(cache_size: int = 128, ttl: Optional[float] = None):
    """
    Async cache decorator with TTL support
    
    Args:
        cache_size: Maximum cache size
        ttl: Time to live in seconds (None for no expiration)
    """
    cache: OrderedDict = OrderedDict()
    
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = _make_cache_key(func.__name__, args, kwargs)
            now = time.time()
            
            # Check cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if ttl is None or (now - timestamp) < ttl:
                    # Move to end (LRU)
                    cache.move_to_end(cache_key)
                    return result
                else:
                    # Expired
                    cache.pop(cache_key)
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache[cache_key] = (result, now)
            
            # Enforce size limit (LRU eviction)
            if len(cache) > cache_size:
                cache.popitem(last=False)
            
            return result
        
        return wrapper
    return decorator


def _make_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Create cache key from function name and arguments"""
    key_data = (func_name, args, tuple(sorted(kwargs.items())))
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def batch_process(items: List[Any], batch_size: int = 10) -> List[List[Any]]:
    """Split items into batches for processing"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


async def async_batch_process(
    items: List[Any],
    processor: Callable,
    batch_size: int = 10,
    max_concurrent: int = 5
) -> List[Any]:
    """
    Process items in batches asynchronously
    
    Args:
        items: List of items to process
        processor: Async function to process each batch
        batch_size: Number of items per batch
        max_concurrent: Maximum concurrent batches
    """
    batches = batch_process(items, batch_size)
    results = []
    
    # Process batches with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(batch):
        async with semaphore:
            return await processor(batch)
    
    tasks = [process_batch(batch) for batch in batches]
    batch_results = await asyncio.gather(*tasks)
    
    # Flatten results
    for batch_result in batch_results:
        if isinstance(batch_result, list):
            results.extend(batch_result)
        else:
            results.append(batch_result)
    
    return results


def measure_time(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            duration = time.perf_counter() - start
            if duration > 0.1:  # Only log slow functions
                if logger:
                    logger.debug(f"{func.__name__} took {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.perf_counter() - start
            if logger:
                logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            if duration > 0.1:  # Only log slow functions
                if logger:
                    logger.debug(f"{func.__name__} took {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.perf_counter() - start
            if logger:
                logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class ConnectionPool:
    """Simple connection pool for async resources"""
    
    def __init__(self, factory: Callable, max_size: int = 10):
        self.factory = factory
        self.max_size = max_size
        self.pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.connections = set()
    
    async def acquire(self) -> Any:
        """Acquire connection from pool"""
        try:
            # Try to get from pool (non-blocking)
            return self.pool.get_nowait()
        except asyncio.QueueEmpty:
            # Create new connection if pool is empty and under max_size
            if len(self.connections) < self.max_size:
                conn = await self.factory()
                self.connections.add(conn)
                return conn
            # Wait for available connection
            return await self.pool.get()
    
    async def release(self, conn: Any):
        """Release connection back to pool"""
        await self.pool.put(conn)
    
    async def close_all(self):
        """Close all connections"""
        while not self.pool.empty():
            conn = await self.pool.get()
            if hasattr(conn, 'close'):
                await conn.close()
            self.connections.discard(conn)
        
        # Close remaining connections
        for conn in list(self.connections):
            if hasattr(conn, 'close'):
                await conn.close()
            self.connections.discard(conn)


@lru_cache(maxsize=256)
def parse_json_cached(json_str: str) -> Dict:
    """Cached JSON parsing for repeated queries"""
    return json.loads(json_str)


def optimize_query(query: str) -> str:
    """Optimize database query (placeholder for query optimization)"""
    # Remove extra whitespace
    query = " ".join(query.split())
    return query


class LazyLoader:
    """Lazy loading helper for expensive resources"""
    
    def __init__(self, loader: Callable):
        self.loader = loader
        self._value = None
        self._loaded = False
    
    async def get(self):
        """Get value, loading if necessary"""
        if not self._loaded:
            self._value = await self.loader()
            self._loaded = True
        return self._value
    
    def reset(self):
        """Reset loader to force reload"""
        self._loaded = False
        self._value = None


def chunk_iterable(iterable: Any, chunk_size: int = 1000):
    """Chunk iterable into smaller pieces for memory efficiency"""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk






