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
import logging
import functools
import hashlib
from typing import Any, Optional, Dict, List, Callable, Awaitable, TypeVar, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import weakref
import orjson
from pydantic import BaseModel
        import httpx
from typing import Any, List, Dict, Optional
"""
⚡ Async Performance Utilities
=============================

Utility functions and decorators for async performance optimization:
- I/O-bound task helpers
- Caching decorators
- Lazy loading utilities
- Performance monitoring
- Resource management
"""



logger = logging.getLogger(__name__)

T = TypeVar('T')
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AsyncTaskConfig:
    """Configuration for async task execution"""
    timeout: float = 30.0
    retries: int = 3
    retry_delay: float = 1.0
    priority: TaskPriority = TaskPriority.NORMAL
    cache_ttl: int = 3600
    max_concurrent: int = 10


class AsyncIOTaskManager:
    """
    Manager for I/O-bound async tasks with intelligent scheduling,
    retry logic, and performance monitoring.
    """
    
    def __init__(self, max_concurrent: int = 10):
        
    """__init__ function."""
self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.task_times = []
        
    async def execute(self, func: Callable, *args, 
                     timeout: float = 30.0,
                     retries: int = 3,
                     retry_delay: float = 1.0,
                     **kwargs) -> Any:
        """Execute I/O-bound task with retry logic"""
        start_time = time.time()
        self.active_tasks += 1
        
        try:
            async with self.semaphore:
                for attempt in range(retries + 1):
                    try:
                        if asyncio.iscoroutinefunction(func):
                            result = await asyncio.wait_for(
                                func(*args, **kwargs), timeout=timeout
                            )
                        else:
                            # Run sync function in thread pool
                            loop = asyncio.get_event_loop()
                            result = await asyncio.wait_for(
                                loop.run_in_executor(None, func, *args, **kwargs),
                                timeout=timeout
                            )
                        
                        execution_time = time.time() - start_time
                        self.task_times.append(execution_time)
                        self.completed_tasks += 1
                        
                        return result
                        
                    except asyncio.TimeoutError:
                        if attempt == retries:
                            raise
                        logger.warning(f"Task timeout, retrying... (attempt {attempt + 1})")
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        
                    except Exception as e:
                        if attempt == retries:
                            raise
                        logger.warning(f"Task failed, retrying... (attempt {attempt + 1}): {e}")
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        
        except Exception as e:
            self.failed_tasks += 1
            logger.error(f"Task execution failed: {e}")
            raise
        finally:
            self.active_tasks -= 1
    
    async def execute_batch(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """Execute batch of I/O-bound tasks"""
        async def execute_single(task_info) -> Any:
            func = task_info['func']
            args = task_info.get('args', ())
            kwargs = task_info.get('kwargs', {})
            timeout = task_info.get('timeout', 30.0)
            retries = task_info.get('retries', 3)
            retry_delay = task_info.get('retry_delay', 1.0)
            
            return await self.execute(
                func, *args, timeout=timeout,
                retries=retries, retry_delay=retry_delay, **kwargs
            )
        
        # Execute all tasks concurrently
        task_coroutines = [execute_single(task) for task in tasks]
        return await asyncio.gather(*task_coroutines, return_exceptions=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics"""
        avg_time = sum(self.task_times) / len(self.task_times) if self.task_times else 0
        return {
            "active_tasks": self.active_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.completed_tasks / (self.completed_tasks + self.failed_tasks) if (self.completed_tasks + self.failed_tasks) > 0 else 0,
            "average_execution_time": avg_time,
            "max_concurrent": self.max_concurrent
        }


class AsyncCache:
    """
    Simple async cache with TTL and automatic cleanup.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        
    """__init__ function."""
self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = {}
        self._access_times = {}
        self._lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key not in self._cache:
                return None
            
            # Check TTL
            if time.time() > self._access_times[key]:
                del self._cache[key]
                del self._access_times[key]
                return None
            
            return self._cache[key]
    
    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set value in cache"""
        async with self._lock:
            # Evict if cache is full
            if len(self._cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
                del self._cache[oldest_key]
                del self._access_times[oldest_key]
            
            ttl = ttl or self.default_ttl
            self._cache[key] = value
            self._access_times[key] = time.time() + ttl
    
    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        async with self._lock:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "default_ttl": self.default_ttl
        }


def async_cache(ttl: int = 3600, key_generator: Callable = None):
    """
    Decorator for async function caching.
    
    Args:
        ttl: Time to live in seconds
        key_generator: Function to generate cache key from function arguments
    """
    def decorator(func: Callable) -> Callable:
        cache = AsyncCache(default_ttl=ttl)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                # Default key generation
                key_data = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': sorted(kwargs.items())
                }
                cache_key = hashlib.md5(orjson.dumps(key_data)).hexdigest()
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl)
            
            return result
        
        # Add cache stats method
        wrapper.get_cache_stats = cache.get_stats
        wrapper.clear_cache = cache.clear
        
        return wrapper
    
    return decorator


def async_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for async function retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Multiplier for delay on each retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        raise
                    
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    
    return decorator


def async_timeout(timeout: float):
    """
    Decorator for async function timeout.
    
    Args:
        timeout: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        
        return wrapper
    
    return decorator


class LazyAsyncLoader:
    """
    Lazy loader for async resources with dependency tracking.
    """
    
    def __init__(self) -> Any:
        self._loaded_resources = {}
        self._loading_futures = {}
        self._dependencies = {}
        
    async def load(self, resource_name: str, loader_func: Callable, 
                  dependencies: List[str] = None) -> Any:
        """Load resource lazily"""
        # Return if already loaded
        if resource_name in self._loaded_resources:
            return self._loaded_resources[resource_name]
        
        # Wait if already loading
        if resource_name in self._loading_futures:
            return await self._loading_futures[resource_name]
        
        # Load dependencies first
        if dependencies:
            for dep in dependencies:
                if dep in self._dependencies:
                    await self.load(dep, self._dependencies[dep])
        
        # Start loading
        self._loading_futures[resource_name] = asyncio.create_task(
            self._load_resource(resource_name, loader_func)
        )
        
        try:
            result = await self._loading_futures[resource_name]
            return result
        finally:
            self._loading_futures.pop(resource_name, None)
    
    async def _load_resource(self, resource_name: str, loader_func: Callable) -> Any:
        """Actually load the resource"""
        try:
            if asyncio.iscoroutinefunction(loader_func):
                resource = await loader_func()
            else:
                # Run sync loader in thread pool
                loop = asyncio.get_event_loop()
                resource = await loop.run_in_executor(None, loader_func)
            
            self._loaded_resources[resource_name] = resource
            logger.info(f"Lazy loaded resource: {resource_name}")
            
            return resource
            
        except Exception as e:
            logger.error(f"Failed to lazy load resource {resource_name}: {e}")
            raise
    
    def register_dependency(self, resource_name: str, loader_func: Callable):
        """Register a resource dependency"""
        self._dependencies[resource_name] = loader_func
    
    def unload(self, resource_name: str):
        """Unload resource"""
        self._loaded_resources.pop(resource_name, None)
        logger.info(f"Unloaded resource: {resource_name}")


@asynccontextmanager
async def async_resource_manager(resource_name: str, loader_func: Callable, 
                               cleanup_func: Callable = None):
    """
    Context manager for async resource management.
    
    Args:
        resource_name: Name of the resource
        loader_func: Function to load the resource
        cleanup_func: Function to cleanup the resource
    """
    resource = None
    try:
        if asyncio.iscoroutinefunction(loader_func):
            resource = await loader_func()
        else:
            loop = asyncio.get_event_loop()
            resource = await loop.run_in_executor(None, loader_func)
        
        yield resource
        
    finally:
        if resource and cleanup_func:
            try:
                if asyncio.iscoroutinefunction(cleanup_func):
                    await cleanup_func(resource)
                else:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, cleanup_func, resource)
            except Exception as e:
                logger.error(f"Error cleaning up resource {resource_name}: {e}")


class AsyncPerformanceMonitor:
    """
    Monitor for async performance metrics.
    """
    
    def __init__(self) -> Any:
        self.metrics = {}
        self.start_time = time.time()
        
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.metrics[name] = {
            'start_time': time.time(),
            'end_time': None,
            'duration': None
        }
    
    def end_timer(self, name: str):
        """End timing an operation"""
        if name in self.metrics:
            self.metrics[name]['end_time'] = time.time()
            self.metrics[name]['duration'] = (
                self.metrics[name]['end_time'] - self.metrics[name]['start_time']
            )
    
    @asynccontextmanager
    async def timer(self, name: str):
        """Context manager for timing async operations"""
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'uptime': time.time() - self.start_time,
            'timers': self.metrics
        }


def monitor_async_performance(monitor: AsyncPerformanceMonitor):
    """
    Decorator to monitor async function performance.
    
    Args:
        monitor: Performance monitor instance
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            async with monitor.timer(func.__name__):
                return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Utility functions for common async patterns
async def async_map(func: Callable, items: List[Any], 
                   max_concurrent: int = 10) -> List[Any]:
    """
    Apply function to items with controlled concurrency.
    
    Args:
        func: Function to apply
        items: List of items to process
        max_concurrent: Maximum concurrent executions
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(item) -> Any:
        async with semaphore:
            if asyncio.iscoroutinefunction(func):
                return await func(item)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, item)
    
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def async_filter(func: Callable, items: List[Any],
                      max_concurrent: int = 10) -> List[Any]:
    """
    Filter items with controlled concurrency.
    
    Args:
        func: Filter function
        items: List of items to filter
        max_concurrent: Maximum concurrent executions
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def filter_item(item) -> Any:
        async with semaphore:
            if asyncio.iscoroutinefunction(func):
                should_include = await func(item)
            else:
                loop = asyncio.get_event_loop()
                should_include = await loop.run_in_executor(None, func, item)
            
            return item if should_include else None
    
    tasks = [filter_item(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None results and exceptions
    return [result for result in results if result is not None and not isinstance(result, Exception)]


async def async_reduce(func: Callable, items: List[Any], 
                      initial: Any = None) -> Any:
    """
    Reduce items with async function.
    
    Args:
        func: Reduce function
        items: List of items to reduce
        initial: Initial value
    """
    if not items:
        return initial
    
    result = initial if initial is not None else items[0]
    start_idx = 1 if initial is not None else 1
    
    for item in items[start_idx:]:
        if asyncio.iscoroutinefunction(func):
            result = await func(result, item)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, func, result, item)
    
    return result


# Example usage
async def example_usage():
    """Example of how to use the async performance utilities"""
    
    # Create components
    io_manager = AsyncIOTaskManager(max_concurrent=5)
    cache = AsyncCache(max_size=100)
    lazy_loader = LazyAsyncLoader()
    monitor = AsyncPerformanceMonitor()
    
    # Example async function with caching and retry
    @async_cache(ttl=300)
    @async_retry(max_retries=3)
    @monitor_async_performance(monitor)
    async async def fetch_data(url: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            return response.text
    
    # Execute with IO manager
    result = await io_manager.execute(fetch_data, "https://api.example.com/data")
    
    # Get statistics
    print("IO Manager Stats:", io_manager.get_stats())
    print("Cache Stats:", cache.get_stats())
    print("Performance Metrics:", monitor.get_metrics())


match __name__:
    case "__main__":
    asyncio.run(example_usage()) 