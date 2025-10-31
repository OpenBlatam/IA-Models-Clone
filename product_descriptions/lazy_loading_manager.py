from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Dict, Generic, Iterator, List, Optional, Set, TypeVar, Union
from typing_extensions import Protocol
import aiofiles
from pydantic import BaseModel, Field
import logging
from typing import Any, List, Dict, Optional
"""
Advanced Lazy Loading Manager for Product Descriptions API

This module provides comprehensive lazy loading capabilities including:
- Lazy loading for large datasets
- Streaming API responses
- Pagination and cursor-based loading
- Memory-efficient data processing
- Background loading and prefetching
- Lazy evaluation and generators
- Resource management and cleanup
"""

# Standard library imports (alphabetical)

# Third-party imports (alphabetical)

# Configure logging

logger = logging.getLogger(__name__)


class LoadingStrategy(Enum):
    """Lazy loading strategies for different use cases."""
    
    ON_DEMAND = "on_demand"           # Load only when accessed
    PAGINATED = "paginated"           # Load in pages
    STREAMING = "streaming"           # Stream data as it's available
    BACKGROUND = "background"         # Load in background
    PREFETCH = "prefetch"             # Prefetch next items
    CURSOR_BASED = "cursor_based"     # Cursor-based pagination
    WINDOWED = "windowed"             # Sliding window loading


class DataSourceType(Enum):
    """Data source types for lazy loading."""
    
    DATABASE = "database"
    API = "api"
    FILE = "file"
    CACHE = "cache"
    MEMORY = "memory"
    STREAM = "stream"


@dataclass
class LazyLoadingConfig:
    """
    Configuration for lazy loading with sensible defaults.
    
    This configuration controls the behavior of lazy loading operations
    including caching, memory management, and performance optimizations.
    """
    
    strategy: LoadingStrategy = LoadingStrategy.ON_DEMAND
    batch_size: int = 100
    max_memory: int = 1024 * 1024 * 100  # 100MB
    prefetch_size: int = 50
    window_size: int = 200
    cache_ttl: int = 300
    enable_monitoring: bool = True
    enable_cleanup: bool = True
    cleanup_interval: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class LoadingStats:
    """
    Statistics for lazy loading operations.
    
    Tracks performance metrics, cache behavior, and error rates
    for monitoring and optimization purposes.
    """
    
    total_items: int = 0
    loaded_items: int = 0
    cached_items: int = 0
    memory_usage: int = 0
    load_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def load_percentage(self) -> float:
        """Get percentage of items loaded."""
        return (self.loaded_items / max(1, self.total_items)) * 100
    
    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        total_requests = self.cache_hits + self.cache_misses
        return (self.cache_hits / max(1, total_requests)) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since start in seconds."""
        return time.time() - self.start_time


# Type variables for generic lazy loading
T = TypeVar('T')
K = TypeVar('K')


class DataSource(Protocol[T]):
    """
    Protocol for data sources in lazy loading system.
    
    Defines the interface that data sources must implement
    to work with the lazy loading system.
    """
    
    async def get_item(self, key: Any) -> Optional[T]:
        """Get single item by key."""
        ...
    
    async def get_batch(self, keys: List[Any]) -> List[T]:
        """Get batch of items by keys."""
        ...
    
    async def get_all(self) -> List[T]:
        """Get all items from the data source."""
        ...
    
    async def count(self) -> int:
        """Get total count of items."""
        ...


class LazyItem(Generic[T]):
    """
    Represents a lazy-loaded item with loading state management.
    
    This class handles the lazy loading of individual items with
    proper state tracking, error handling, and access timing.
    """
    
    def __init__(self, key: Any, loader: Callable[[], Any], config: LazyLoadingConfig):
        """
        Initialize lazy item.
        
        Args:
            key: Unique identifier for the item
            loader: Function to load the item value
            config: Configuration for loading behavior
        """
        self.key = key
        self._loader = loader
        self._config = config
        self._value: Optional[T] = None
        self._loaded = False
        self._loading = False
        self._error: Optional[Exception] = None
        self._last_accessed = time.time()
    
    async def get_value(self) -> T:
        """
        Get the item value, loading if necessary.
        
        Returns:
            The loaded item value.
            
        Raises:
            Exception: If loading fails or item has error state.
        """
        if self._error:
            raise self._error
        
        if not self._loaded and not self._loading:
            await self._load()
        
        if self._loading:
            # Wait for loading to complete
            while self._loading:
                await asyncio.sleep(0.01)
        
        if self._error:
            raise self._error
        
        self._last_accessed = time.time()
        return self._value
    
    async def _load(self) -> None:
        """
        Load the item value from the data source.
        
        Handles both sync and async loaders with proper error handling.
        """
        self._loading = True
        try:
            if asyncio.iscoroutinefunction(self._loader):
                self._value = await self._loader()
            else:
                self._value = self._loader()
            self._loaded = True
        except Exception as e:
            self._error = e
            logger.error(f"Failed to load item {self.key}: {e}")
        finally:
            self._loading = False
    
    @property
    def is_loaded(self) -> bool:
        """Check if item is loaded."""
        return self._loaded
    
    @property
    def is_loading(self) -> bool:
        """Check if item is currently loading."""
        return self._loading
    
    @property
    def has_error(self) -> bool:
        """Check if item has error state."""
        return self._error is not None


class LazyCache:
    """
    LRU cache for lazy-loaded items with TTL support.
    
    Implements a least-recently-used cache with time-to-live
    functionality for efficient memory management.
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """
        Initialize lazy cache.
        
        Args:
            max_size: Maximum number of items in cache
            ttl: Time-to-live for cached items in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[Any, Tuple[LazyItem, float]] = {}
        self._access_order = deque()
    
    def get(self, key: Any) -> Optional[LazyItem]:
        """
        Get item from cache with TTL check.
        
        Args:
            key: Cache key to lookup
            
        Returns:
            Cached item if valid, None otherwise
        """
        if key in self._cache:
            item, timestamp = self._cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl:
                self._remove(key)
                return None
            
            # Update access order for LRU
            self._access_order.remove(key)
            self._access_order.append(key)
            
            return item
        return None
    
    def set(self, key: Any, item: LazyItem) -> None:
        """
        Set item in cache with LRU eviction.
        
        Args:
            key: Cache key
            item: Lazy item to cache
        """
        if key in self._cache:
            self._access_order.remove(key)
        
        # Evict if cache is full
        if len(self._cache) >= self.max_size:
            oldest_key = self._access_order.popleft()
            self._remove(oldest_key)
        
        self._cache[key] = (item, time.time())
        self._access_order.append(key)
    
    def _remove(self, key: Any) -> None:
        """
        Remove item from cache safely.
        
        Args:
            key: Cache key to remove
        """
        self._cache.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)
    
    def clear(self) -> None:
        """Clear all items from cache."""
        self._cache.clear()
        self._access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "oldest_access": min(self._access_order) if self._access_order else None
        }


class LazyLoader(Generic[T]):
    """Base lazy loader class"""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
        self.cache = LazyCache(config.batch_size, config.cache_ttl)
        self.stats = LoadingStats()
        self._cleanup_task: Optional[asyncio.Task] = None
        
        if config.enable_cleanup:
            self._start_cleanup_task()
    
    def _start_cleanup_task(self) -> Any:
        """Start background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self) -> Any:
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _cleanup(self) -> Any:
        """Cleanup expired items"""
        # This will be implemented by subclasses
        pass
    
    async def close(self) -> Any:
        """Close the lazy loader"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


class OnDemandLoader(LazyLoader[T]):
    """On-demand lazy loader"""
    
    def __init__(self, data_source: DataSource[T], config: LazyLoadingConfig):
        
    """__init__ function."""
super().__init__(config)
        self.data_source = data_source
    
    async def get_item(self, key: Any) -> T:
        """Get item on demand"""
        # Check cache first
        cached_item = self.cache.get(key)
        if cached_item:
            self.stats.cache_hits += 1
            return await cached_item.get_value()
        
        self.stats.cache_misses += 1
        
        # Create lazy item
        async def loader():
            
    """loader function."""
return await self.data_source.get_item(key)
        
        lazy_item = LazyItem(key, loader, self.config)
        self.cache.set(key, lazy_item)
        
        # Load the item
        start_time = time.time()
        try:
            value = await lazy_item.get_value()
            self.stats.loaded_items += 1
            self.stats.load_time += time.time() - start_time
            return value
        except Exception as e:
            self.stats.errors += 1
            raise


class PaginatedLoader(LazyLoader[T]):
    """Paginated lazy loader"""
    
    def __init__(self, data_source: DataSource[T], config: LazyLoadingConfig):
        
    """__init__ function."""
super().__init__(config)
        self.data_source = data_source
        self._pages: Dict[int, List[T]] = {}
        self._total_count: Optional[int] = None
    
    async def get_page(self, page: int, page_size: Optional[int] = None) -> List[T]:
        """Get specific page"""
        page_size = page_size or self.config.batch_size
        
        if page in self._pages:
            self.stats.cache_hits += 1
            return self._pages[page]
        
        self.stats.cache_misses += 1
        
        # Load page from data source
        start_time = time.time()
        try:
            # This is a simplified implementation
            # In practice, you'd implement proper pagination logic
            all_items = await self.data_source.get_all()
            start_idx = page * page_size
            end_idx = start_idx + page_size
            page_items = all_items[start_idx:end_idx]
            
            self._pages[page] = page_items
            self.stats.loaded_items += len(page_items)
            self.stats.load_time += time.time() - start_time
            
            return page_items
        except Exception as e:
            self.stats.errors += 1
            raise
    
    async def get_total_count(self) -> int:
        """Get total count of items"""
        if self._total_count is None:
            self._total_count = await self.data_source.count()
        return self._total_count


class StreamingLoader(LazyLoader[T]):
    """Streaming lazy loader"""
    
    def __init__(self, data_source: DataSource[T], config: LazyLoadingConfig):
        
    """__init__ function."""
super().__init__(config)
        self.data_source = data_source
        self._stream_buffer: deque = deque(maxlen=config.window_size)
        self._streaming = False
    
    async def start_streaming(self) -> Any:
        """Start streaming data"""
        if self._streaming:
            return
        
        self._streaming = True
        asyncio.create_task(self._stream_data())
    
    async def _stream_data(self) -> Any:
        """Stream data from source"""
        try:
            # This is a simplified implementation
            # In practice, you'd implement proper streaming logic
            all_items = await self.data_source.get_all()
            
            for item in all_items:
                if not self._streaming:
                    break
                
                self._stream_buffer.append(item)
                self.stats.loaded_items += 1
                
                # Small delay to prevent blocking
                await asyncio.sleep(0.001)
                
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Streaming error: {e}")
        finally:
            self._streaming = False
    
    async def get_next_item(self) -> Optional[T]:
        """Get next item from stream"""
        if not self._streaming and not self._stream_buffer:
            await self.start_streaming()
        
        if self._stream_buffer:
            return self._stream_buffer.popleft()
        
        return None
    
    async def stop_streaming(self) -> Any:
        """Stop streaming"""
        self._streaming = False


class BackgroundLoader(LazyLoader[T]):
    """Background lazy loader with prefetching"""
    
    def __init__(self, data_source: DataSource[T], config: LazyLoadingConfig):
        
    """__init__ function."""
super().__init__(config)
        self.data_source = data_source
        self._prefetch_queue: asyncio.Queue = asyncio.Queue(maxsize=config.prefetch_size)
        self._background_task: Optional[asyncio.Task] = None
        self._keys_to_load: Set[Any] = set()
    
    async def start_background_loading(self, keys: List[Any]):
        """Start background loading of keys"""
        self._keys_to_load.update(keys)
        
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._background_load())
    
    async def _background_load(self) -> Any:
        """Background loading task"""
        while self._keys_to_load:
            try:
                # Load batch of keys
                batch_keys = list(self._keys_to_load)[:self.config.batch_size]
                self._keys_to_load.difference_update(batch_keys)
                
                # Load items
                items = await self.data_source.get_batch(batch_keys)
                
                # Add to prefetch queue
                for key, item in zip(batch_keys, items):
                    await self._prefetch_queue.put((key, item))
                
                self.stats.loaded_items += len(items)
                
                # Small delay to prevent blocking
                await asyncio.sleep(0.01)
                
            except Exception as e:
                self.stats.errors += 1
                logger.error(f"Background loading error: {e}")
                await asyncio.sleep(self.config.retry_delay)
    
    async def get_item(self, key: Any) -> T:
        """Get item (may be prefetched)"""
        # Check cache first
        cached_item = self.cache.get(key)
        if cached_item:
            self.stats.cache_hits += 1
            return await cached_item.get_value()
        
        self.stats.cache_misses += 1
        
        # Check if item is in prefetch queue
        # This is a simplified implementation
        # In practice, you'd implement proper prefetch checking
        
        # Load item directly
        async def loader():
            
    """loader function."""
return await self.data_source.get_item(key)
        
        lazy_item = LazyItem(key, loader, self.config)
        self.cache.set(key, lazy_item)
        
        return await lazy_item.get_value()


class CursorBasedLoader(LazyLoader[T]):
    """Cursor-based lazy loader"""
    
    def __init__(self, data_source: DataSource[T], config: LazyLoadingConfig):
        
    """__init__ function."""
super().__init__(config)
        self.data_source = data_source
        self._cursors: Dict[str, Any] = {}
    
    async def get_page_with_cursor(self, cursor: Optional[Any] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get page using cursor-based pagination"""
        limit = limit or self.config.batch_size
        
        # This is a simplified implementation
        # In practice, you'd implement proper cursor-based pagination
        all_items = await self.data_source.get_all()
        
        if cursor is None:
            start_idx = 0
        else:
            # Find cursor position (simplified)
            start_idx = cursor if isinstance(cursor, int) else 0
        
        end_idx = start_idx + limit
        page_items = all_items[start_idx:end_idx]
        
        # Generate next cursor
        next_cursor = end_idx if end_idx < len(all_items) else None
        
        return {
            "items": page_items,
            "next_cursor": next_cursor,
            "has_more": next_cursor is not None
        }


class WindowedLoader(LazyLoader[T]):
    """Sliding window lazy loader"""
    
    def __init__(self, data_source: DataSource[T], config: LazyLoadingConfig):
        
    """__init__ function."""
super().__init__(config)
        self.data_source = data_source
        self._window_start = 0
        self._window_end = config.window_size
        self._window_items: List[T] = []
        self._total_items: Optional[List[T]] = None
    
    async def get_window(self, start: Optional[int] = None, size: Optional[int] = None) -> List[T]:
        """Get items in current window"""
        size = size or self.config.window_size
        
        if start is not None:
            self._window_start = start
            self._window_end = start + size
        
        # Load total items if not loaded
        if self._total_items is None:
            self._total_items = await self.data_source.get_all()
        
        # Get items in window
        window_items = self._total_items[self._window_start:self._window_end]
        self._window_items = window_items
        
        return window_items
    
    async def slide_window(self, direction: str = "forward", size: Optional[int] = None):
        """Slide the window forward or backward"""
        size = size or self.config.batch_size
        
        if direction == "forward":
            self._window_start += size
            self._window_end += size
        else:
            self._window_start = max(0, self._window_start - size)
            self._window_end = max(size, self._window_end - size)
        
        return await self.get_window()


# Lazy loading decorators and utilities
def lazy_load(strategy: LoadingStrategy = LoadingStrategy.ON_DEMAND, **config_kwargs):
    """Decorator for lazy loading functions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            config = LazyLoadingConfig(strategy=strategy, **config_kwargs)
            
            # Create appropriate loader based on strategy
            if strategy == LoadingStrategy.ON_DEMAND:
                loader = OnDemandLoader(None, config)  # Data source would be injected
            elif strategy == LoadingStrategy.PAGINATED:
                loader = PaginatedLoader(None, config)
            elif strategy == LoadingStrategy.STREAMING:
                loader = StreamingLoader(None, config)
            elif strategy == LoadingStrategy.BACKGROUND:
                loader = BackgroundLoader(None, config)
            elif strategy == LoadingStrategy.CURSOR_BASED:
                loader = CursorBasedLoader(None, config)
            elif strategy == LoadingStrategy.WINDOWED:
                loader = WindowedLoader(None, config)
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            # Execute function with loader
            return await func(*args, loader=loader, **kwargs)
        
        return wrapper
    return decorator


def lazy_generator(func: Callable) -> Callable:
    """Decorator for creating lazy generators"""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        async for item in func(*args, **kwargs):
            yield item
    return wrapper


@asynccontextmanager
async def lazy_loading_context(config: LazyLoadingConfig):
    """Context manager for lazy loading"""
    loader = None
    try:
        # Create loader based on config
        if config.strategy == LoadingStrategy.ON_DEMAND:
            loader = OnDemandLoader(None, config)
        elif config.strategy == LoadingStrategy.PAGINATED:
            loader = PaginatedLoader(None, config)
        elif config.strategy == LoadingStrategy.STREAMING:
            loader = StreamingLoader(None, config)
        elif config.strategy == LoadingStrategy.BACKGROUND:
            loader = BackgroundLoader(None, config)
        elif config.strategy == LoadingStrategy.CURSOR_BASED:
            loader = CursorBasedLoader(None, config)
        elif config.strategy == LoadingStrategy.WINDOWED:
            loader = WindowedLoader(None, config)
        
        yield loader
    finally:
        if loader:
            await loader.close()


# Memory management utilities
class MemoryManager:
    """Memory management for lazy loading"""
    
    def __init__(self, max_memory: int):
        
    """__init__ function."""
self.max_memory = max_memory
        self.current_memory = 0
        self._items: Dict[Any, int] = {}
    
    def can_allocate(self, size: int) -> bool:
        """Check if we can allocate memory"""
        return self.current_memory + size <= self.max_memory
    
    def allocate(self, key: Any, size: int) -> bool:
        """Allocate memory for an item"""
        if self.can_allocate(size):
            self._items[key] = size
            self.current_memory += size
            return True
        return False
    
    def deallocate(self, key: Any):
        """Deallocate memory for an item"""
        if key in self._items:
            self.current_memory -= self._items[key]
            del self._items[key]
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return {
            "current_memory": self.current_memory,
            "max_memory": self.max_memory,
            "usage_percentage": (self.current_memory / self.max_memory) * 100,
            "item_count": len(self._items)
        }


# Performance monitoring
class LazyLoadingMonitor:
    """Monitor for lazy loading performance"""
    
    def __init__(self) -> Any:
        self.metrics: List[Dict[str, Any]] = []
        self.start_time = time.time()
    
    def record_metric(self, metric: Dict[str, Any]):
        """Record a performance metric"""
        metric["timestamp"] = time.time()
        self.metrics.append(metric)
        
        # Keep only last 1000 metrics
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        if not self.metrics:
            return {"error": "No metrics available"}
        
        # Calculate statistics
        load_times = [m.get("load_time", 0) for m in self.metrics]
        memory_usage = [m.get("memory_usage", 0) for m in self.metrics]
        
        return {
            "total_operations": len(self.metrics),
            "avg_load_time": sum(load_times) / len(load_times),
            "max_load_time": max(load_times),
            "min_load_time": min(load_times),
            "avg_memory_usage": sum(memory_usage) / len(memory_usage),
            "max_memory_usage": max(memory_usage),
            "elapsed_time": time.time() - self.start_time
        }


# Global lazy loading manager
class LazyLoadingManager:
    """Global manager for lazy loading operations"""
    
    def __init__(self) -> Any:
        self.loaders: Dict[str, LazyLoader] = {}
        self.monitor = LazyLoadingMonitor()
        self.memory_manager = MemoryManager(1024 * 1024 * 100)  # 100MB
    
    def register_loader(self, name: str, loader: LazyLoader):
        """Register a lazy loader"""
        self.loaders[name] = loader
    
    def get_loader(self, name: str) -> Optional[LazyLoader]:
        """Get a registered loader"""
        return self.loaders.get(name)
    
    async def close_all(self) -> Any:
        """Close all registered loaders"""
        for loader in self.loaders.values():
            await loader.close()
        self.loaders.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all loaders"""
        stats = {
            "loaders": len(self.loaders),
            "memory": self.memory_manager.get_memory_usage(),
            "performance": self.monitor.get_performance_report()
        }
        
        for name, loader in self.loaders.items():
            stats[name] = loader.stats.__dict__
        
        return stats


# Global instance
_lazy_loading_manager = LazyLoadingManager()


def get_lazy_loading_manager() -> LazyLoadingManager:
    """Get global lazy loading manager"""
    return _lazy_loading_manager


async def close_lazy_loading_manager():
    """Close global lazy loading manager"""
    await _lazy_loading_manager.close_all() 