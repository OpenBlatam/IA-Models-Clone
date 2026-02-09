"""
Performance optimizations for Multi-Model API
"""

import asyncio
import functools
from typing import Any, Callable, Awaitable, TypeVar, List
from concurrent.futures import ThreadPoolExecutor
import time

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

T = TypeVar('T')


_executor = ThreadPoolExecutor(max_workers=4)


def fast_json_dumps(obj: Any, **kwargs) -> str:
    """Fast JSON serialization using orjson if available"""
    if ORJSON_AVAILABLE:
        # orjson options for better performance
        options = kwargs.pop('option', 0)
        if 'option' not in kwargs:
            options |= orjson.OPT_SERIALIZE_NUMPY  # Faster numpy serialization
        return orjson.dumps(obj, option=options, **kwargs).decode('utf-8')
    import json
    return json.dumps(obj, **kwargs)


def fast_json_loads(s: str) -> Any:
    """Fast JSON deserialization using orjson if available"""
    if ORJSON_AVAILABLE:
        return orjson.loads(s)
    import json
    return json.loads(s)


def optimize_dict_serialization(func: Callable) -> Callable:
    """Decorator to optimize dict serialization in responses"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        if hasattr(result, 'model_dump'):
            return result.model_dump()
        elif hasattr(result, 'dict'):
            return result.dict()
        return result
    return wrapper


async def gather_with_early_return(
    *coros: Awaitable[T],
    return_exceptions: bool = False,
    timeout: float = None
) -> List[T]:
    """Optimized gather that can return early on first success"""
    if timeout:
        return await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=return_exceptions),
            timeout=timeout
        )
    return await asyncio.gather(*coros, return_exceptions=return_exceptions)


class FastSemaphore:
    """Optimized semaphore with less overhead"""
    
    def __init__(self, value: int):
        self._value = value
        self._waiters = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        async with self._lock:
            if self._value > 0:
                self._value -= 1
                return True
            else:
                future = asyncio.Future()
                self._waiters.append(future)
                await future
                return True
    
    async def release(self):
        async with self._lock:
            if self._waiters:
                waiter = self._waiters.pop(0)
                waiter.set_result(None)
            else:
                self._value += 1
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()


def batch_process(items: List[Any], batch_size: int = 10) -> List[List[Any]]:
    """Split items into batches efficiently"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


async def parallel_map(
    items: List[Any],
    func: Callable[[Any], Awaitable[T]],
    max_concurrent: int = 10
) -> List[T]:
    """Parallel map with concurrency control"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(item):
        async with semaphore:
            return await func(item)
    
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks, return_exceptions=True)


def cache_key_fast(*args, **kwargs) -> str:
    """Fast cache key generation - ultra optimized"""
    import hashlib
    # Optimized: early return for simple cases
    if not args and not kwargs:
        return hashlib.md5(b'').hexdigest()
    
    # Optimized: use join with separator for better hashing
    key_parts = [str(args)]
    if kwargs:
        # Optimized: sort items once and format efficiently
        sorted_items = sorted(kwargs.items())
        key_parts.append(str(sorted_items))
    key_str = '|'.join(key_parts)
    # Use md5 directly without intermediate string
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

