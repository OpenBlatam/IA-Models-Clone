from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, TypeVar, Generic, Iterator, AsyncIterator
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
from functools import lru_cache, wraps
import inspect
import gc
import psutil
from pydantic import BaseModel, Field, validator, root_validator, ConfigDict
import aiofiles
import aiostream
from aiostream import stream
            import sys
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Lazy Loading System for HeyGen AI API
Efficient handling of large datasets and substantial API responses.
"""



logger = structlog.get_logger()

# =============================================================================
# Lazy Loading Types
# =============================================================================

class LoadingStrategy(Enum):
    """Lazy loading strategy enumeration."""
    STREAMING = "streaming"
    PAGINATION = "pagination"
    CURSOR_BASED = "cursor_based"
    WINDOW_BASED = "window_based"
    VIRTUAL_SCROLLING = "virtual_scrolling"
    INFINITE_SCROLL = "infinite_scroll"

class DataSourceType(Enum):
    """Data source type enumeration."""
    DATABASE = "database"
    API = "api"
    FILE = "file"
    CACHE = "cache"
    HYBRID = "hybrid"

class LoadingPriority(Enum):
    """Loading priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class LazyLoadingConfig:
    """Lazy loading configuration."""
    strategy: LoadingStrategy = LoadingStrategy.STREAMING
    source_type: DataSourceType = DataSourceType.DATABASE
    priority: LoadingPriority = LoadingPriority.NORMAL
    batch_size: int = 100
    max_concurrent_batches: int = 5
    buffer_size: int = 1000
    enable_caching: bool = True
    enable_compression: bool = False
    enable_prefetching: bool = True
    prefetch_distance: int = 2
    memory_limit_mb: int = 500
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    enable_backpressure: bool = True
    backpressure_threshold: float = 0.8
    enable_monitoring: bool = True
    enable_metrics: bool = True

@dataclass
class LoadingStats:
    """Loading statistics."""
    total_items: int = 0
    loaded_items: int = 0
    cached_items: int = 0
    streamed_items: int = 0
    paginated_items: int = 0
    batch_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    retries: int = 0
    total_loading_time_ms: float = 0.0
    average_batch_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    throughput_items_per_second: float = 0.0
    last_loaded_at: Optional[datetime] = None
    created_at: datetime = None
    
    def __post_init__(self) -> Any:
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
    
    def update_loading_stats(self, items_loaded: int, batch_time_ms: float, cache_hit: bool = False):
        """Update loading statistics."""
        self.loaded_items += items_loaded
        self.batch_operations += 1
        self.total_loading_time_ms += batch_time_ms
        self.average_batch_time_ms = self.total_loading_time_ms / self.batch_operations
        self.last_loaded_at = datetime.now(timezone.utc)
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Calculate throughput
        if self.total_loading_time_ms > 0:
            self.throughput_items_per_second = (self.loaded_items / self.total_loading_time_ms) * 1000

# =============================================================================
# Base Lazy Loading Classes
# =============================================================================

class LazyLoadingBase:
    """Base class for lazy loading implementations."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
        self.stats = LoadingStats()
        self._cache: Dict[str, Any] = {}
        self._loading_tasks: Dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(config.max_concurrent_batches)
        self._memory_monitor = MemoryMonitor(config.memory_limit_mb)
    
    async def load_data(self, key: str, loader_func: Callable, *args, **kwargs) -> Any:
        """Load data with lazy loading."""
        raise NotImplementedError
    
    async def stream_data(self, key: str, loader_func: Callable, *args, **kwargs) -> AsyncIterator[Any]:
        """Stream data with lazy loading."""
        raise NotImplementedError
    
    async def get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data."""
        if not self.config.enable_caching:
            return None
        
        return self._cache.get(key)
    
    async def set_cached_data(self, key: str, data: Any):
        """Set cached data."""
        if not self.config.enable_caching:
            return
        
        self._cache[key] = data
        self.stats.cached_items += 1
    
    async def clear_cache(self) -> Any:
        """Clear cache."""
        self._cache.clear()
        self.stats.cached_items = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics."""
        return {
            "total_items": self.stats.total_items,
            "loaded_items": self.stats.loaded_items,
            "cached_items": self.stats.cached_items,
            "batch_operations": self.stats.batch_operations,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "errors": self.stats.errors,
            "retries": self.stats.retries,
            "average_batch_time_ms": self.stats.average_batch_time_ms,
            "memory_usage_mb": self.stats.memory_usage_mb,
            "throughput_items_per_second": self.stats.throughput_items_per_second,
            "last_loaded_at": self.stats.last_loaded_at.isoformat() if self.stats.last_loaded_at else None,
            "config": {
                "strategy": self.config.strategy.value,
                "batch_size": self.config.batch_size,
                "max_concurrent_batches": self.config.max_concurrent_batches,
                "enable_caching": self.config.enable_caching
            }
        }

class MemoryMonitor:
    """Memory usage monitor for lazy loading."""
    
    def __init__(self, limit_mb: int):
        
    """__init__ function."""
self.limit_mb = limit_mb
        self.warning_threshold = limit_mb * 0.8
        self.critical_threshold = limit_mb * 0.95
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024
    
    def is_memory_high(self) -> bool:
        """Check if memory usage is high."""
        return self.get_memory_usage() > self.warning_threshold
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is critical."""
        return self.get_memory_usage() > self.critical_threshold
    
    async def optimize_memory(self) -> Any:
        """Optimize memory usage."""
        if self.is_memory_critical():
            logger.warning("Critical memory usage detected, forcing garbage collection")
            gc.collect()
            
            # Force memory cleanup
            if hasattr(sys, 'exc_clear'):
                sys.exc_clear()

# =============================================================================
# Streaming Lazy Loader
# =============================================================================

class StreamingLazyLoader(LazyLoadingBase):
    """Streaming lazy loader for large datasets."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
super().__init__(config)
        self._streams: Dict[str, AsyncIterator[Any]] = {}
        self._stream_buffers: Dict[str, List[Any]] = {}
    
    async def stream_data(self, key: str, loader_func: Callable, *args, **kwargs) -> AsyncIterator[Any]:
        """Stream data with lazy loading."""
        if key in self._streams:
            async for item in self._streams[key]:
                yield item
            return
        
        # Create new stream
        stream_iterator = self._create_stream(loader_func, *args, **kwargs)
        self._streams[key] = stream_iterator
        
        buffer = []
        async for item in stream_iterator:
            # Check memory usage
            if self._memory_monitor.is_memory_high():
                await self._memory_monitor.optimize_memory()
            
            # Add to buffer
            buffer.append(item)
            
            # Yield from buffer when it reaches batch size
            if len(buffer) >= self.config.batch_size:
                for buffered_item in buffer:
                    yield buffered_item
                    self.stats.streamed_items += 1
                buffer.clear()
            
            # Check backpressure
            if self.config.enable_backpressure and len(buffer) > self.config.batch_size * self.config.backpressure_threshold:
                await asyncio.sleep(0.1)  # Small delay to reduce backpressure
        
        # Yield remaining items in buffer
        for buffered_item in buffer:
            yield buffered_item
            self.stats.streamed_items += 1
    
    def _create_stream(self, loader_func: Callable, *args, **kwargs) -> AsyncIterator[Any]:
        """Create async stream from loader function."""
        async def stream_generator():
            
    """stream_generator function."""
try:
                if asyncio.iscoroutinefunction(loader_func):
                    result = await loader_func(*args, **kwargs)
                else:
                    result = loader_func(*args, **kwargs)
                
                if hasattr(result, '__aiter__'):
                    async for item in result:
                        yield item
                elif hasattr(result, '__iter__'):
                    for item in result:
                        yield item
                else:
                    yield result
                    
            except Exception as e:
                self.stats.errors += 1
                logger.error(f"Streaming error: {e}")
                raise
        
        return stream_generator()
    
    async def load_data(self, key: str, loader_func: Callable, *args, **kwargs) -> List[Any]:
        """Load all data from stream."""
        items = []
        async for item in self.stream_data(key, loader_func, *args, **kwargs):
            items.append(item)
        return items

# =============================================================================
# Pagination Lazy Loader
# =============================================================================

class PaginationLazyLoader(LazyLoadingBase):
    """Pagination-based lazy loader."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
super().__init__(config)
        self._pagination_states: Dict[str, Dict[str, Any]] = {}
    
    async def load_data(self, key: str, loader_func: Callable, *args, **kwargs) -> List[Any]:
        """Load data with pagination."""
        # Check cache first
        cached_data = await self.get_cached_data(key)
        if cached_data:
            self.stats.cache_hits += 1
            return cached_data
        
        # Load data with pagination
        all_items = []
        page = 1
        has_more = True
        
        while has_more:
            async with self._semaphore:
                start_time = time.time()
                
                try:
                    # Load page
                    page_data = await self._load_page(loader_func, page, *args, **kwargs)
                    
                    if page_data:
                        all_items.extend(page_data)
                        self.stats.paginated_items += len(page_data)
                        has_more = len(page_data) == self.config.batch_size
                        page += 1
                    else:
                        has_more = False
                    
                    # Update statistics
                    batch_time_ms = (time.time() - start_time) * 1000
                    self.stats.update_loading_stats(len(page_data or []), batch_time_ms)
                    
                    # Check memory usage
                    if self._memory_monitor.is_memory_high():
                        await self._memory_monitor.optimize_memory()
                    
                except Exception as e:
                    self.stats.errors += 1
                    logger.error(f"Pagination error on page {page}: {e}")
                    
                    # Retry logic
                    if self.stats.retries < self.config.retry_attempts:
                        self.stats.retries += 1
                        await asyncio.sleep(self.config.retry_delay_seconds)
                        continue
                    else:
                        break
        
        # Cache result
        await self.set_cached_data(key, all_items)
        
        return all_items
    
    async def _load_page(self, loader_func: Callable, page: int, *args, **kwargs) -> List[Any]:
        """Load a single page of data."""
        # Add pagination parameters
        page_kwargs = {
            **kwargs,
            'page': page,
            'per_page': self.config.batch_size,
            'offset': (page - 1) * self.config.batch_size
        }
        
        if asyncio.iscoroutinefunction(loader_func):
            return await loader_func(*args, **page_kwargs)
        else:
            return loader_func(*args, **page_kwargs)
    
    async def stream_data(self, key: str, loader_func: Callable, *args, **kwargs) -> AsyncIterator[Any]:
        """Stream data with pagination."""
        page = 1
        has_more = True
        
        while has_more:
            async with self._semaphore:
                try:
                    # Load page
                    page_data = await self._load_page(loader_func, page, *args, **kwargs)
                    
                    if page_data:
                        for item in page_data:
                            yield item
                            self.stats.streamed_items += 1
                        
                        has_more = len(page_data) == self.config.batch_size
                        page += 1
                    else:
                        has_more = False
                    
                    # Check memory usage
                    if self._memory_monitor.is_memory_high():
                        await self._memory_monitor.optimize_memory()
                    
                except Exception as e:
                    self.stats.errors += 1
                    logger.error(f"Pagination streaming error on page {page}: {e}")
                    break

# =============================================================================
# Cursor-Based Lazy Loader
# =============================================================================

class CursorBasedLazyLoader(LazyLoadingBase):
    """Cursor-based lazy loader for efficient pagination."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
super().__init__(config)
        self._cursors: Dict[str, str] = {}
    
    async def load_data(self, key: str, loader_func: Callable, *args, **kwargs) -> List[Any]:
        """Load data with cursor-based pagination."""
        # Check cache first
        cached_data = await self.get_cached_data(key)
        if cached_data:
            self.stats.cache_hits += 1
            return cached_data
        
        # Load data with cursor-based pagination
        all_items = []
        cursor = None
        has_more = True
        
        while has_more:
            async with self._semaphore:
                start_time = time.time()
                
                try:
                    # Load batch with cursor
                    batch_data = await self._load_batch(loader_func, cursor, *args, **kwargs)
                    
                    if batch_data and len(batch_data) > 0:
                        items, next_cursor = batch_data
                        all_items.extend(items)
                        self.stats.paginated_items += len(items)
                        
                        cursor = next_cursor
                        has_more = cursor is not None and len(items) == self.config.batch_size
                    else:
                        has_more = False
                    
                    # Update statistics
                    batch_time_ms = (time.time() - start_time) * 1000
                    self.stats.update_loading_stats(len(items or []), batch_time_ms)
                    
                    # Check memory usage
                    if self._memory_monitor.is_memory_high():
                        await self._memory_monitor.optimize_memory()
                    
                except Exception as e:
                    self.stats.errors += 1
                    logger.error(f"Cursor-based loading error: {e}")
                    
                    # Retry logic
                    if self.stats.retries < self.config.retry_attempts:
                        self.stats.retries += 1
                        await asyncio.sleep(self.config.retry_delay_seconds)
                        continue
                    else:
                        break
        
        # Cache result
        await self.set_cached_data(key, all_items)
        
        return all_items
    
    async def _load_batch(self, loader_func: Callable, cursor: Optional[str], *args, **kwargs) -> tuple[List[Any], Optional[str]]:
        """Load a batch of data with cursor."""
        # Add cursor parameters
        cursor_kwargs = {
            **kwargs,
            'cursor': cursor,
            'limit': self.config.batch_size
        }
        
        if asyncio.iscoroutinefunction(loader_func):
            return await loader_func(*args, **cursor_kwargs)
        else:
            return loader_func(*args, **cursor_kwargs)
    
    async def stream_data(self, key: str, loader_func: Callable, *args, **kwargs) -> AsyncIterator[Any]:
        """Stream data with cursor-based pagination."""
        cursor = None
        has_more = True
        
        while has_more:
            async with self._semaphore:
                try:
                    # Load batch with cursor
                    batch_data = await self._load_batch(loader_func, cursor, *args, **kwargs)
                    
                    if batch_data and len(batch_data) > 0:
                        items, next_cursor = batch_data
                        
                        for item in items:
                            yield item
                            self.stats.streamed_items += 1
                        
                        cursor = next_cursor
                        has_more = cursor is not None and len(items) == self.config.batch_size
                    else:
                        has_more = False
                    
                    # Check memory usage
                    if self._memory_monitor.is_memory_high():
                        await self._memory_monitor.optimize_memory()
                    
                except Exception as e:
                    self.stats.errors += 1
                    logger.error(f"Cursor-based streaming error: {e}")
                    break

# =============================================================================
# Window-Based Lazy Loader
# =============================================================================

class WindowBasedLazyLoader(LazyLoadingBase):
    """Window-based lazy loader for sliding window operations."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
super().__init__(config)
        self._windows: Dict[str, List[Any]] = {}
        self._window_positions: Dict[str, int] = {}
    
    async def load_data(self, key: str, loader_func: Callable, window_size: int = 100, *args, **kwargs) -> List[Any]:
        """Load data with window-based approach."""
        # Check cache first
        cached_data = await self.get_cached_data(key)
        if cached_data:
            self.stats.cache_hits += 1
            return cached_data
        
        # Load data with window-based approach
        all_items = []
        window_start = 0
        
        while True:
            async with self._semaphore:
                start_time = time.time()
                
                try:
                    # Load window
                    window_data = await self._load_window(loader_func, window_start, window_size, *args, **kwargs)
                    
                    if window_data and len(window_data) > 0:
                        all_items.extend(window_data)
                        self.stats.paginated_items += len(window_data)
                        
                        window_start += window_size
                        
                        # Stop if we got fewer items than window size
                        if len(window_data) < window_size:
                            break
                    else:
                        break
                    
                    # Update statistics
                    batch_time_ms = (time.time() - start_time) * 1000
                    self.stats.update_loading_stats(len(window_data), batch_time_ms)
                    
                    # Check memory usage
                    if self._memory_monitor.is_memory_high():
                        await self._memory_monitor.optimize_memory()
                    
                except Exception as e:
                    self.stats.errors += 1
                    logger.error(f"Window-based loading error: {e}")
                    
                    # Retry logic
                    if self.stats.retries < self.config.retry_attempts:
                        self.stats.retries += 1
                        await asyncio.sleep(self.config.retry_delay_seconds)
                        continue
                    else:
                        break
        
        # Cache result
        await self.set_cached_data(key, all_items)
        
        return all_items
    
    async def _load_window(self, loader_func: Callable, start: int, size: int, *args, **kwargs) -> List[Any]:
        """Load a window of data."""
        # Add window parameters
        window_kwargs = {
            **kwargs,
            'offset': start,
            'limit': size
        }
        
        if asyncio.iscoroutinefunction(loader_func):
            return await loader_func(*args, **window_kwargs)
        else:
            return loader_func(*args, **window_kwargs)
    
    async def stream_data(self, key: str, loader_func: Callable, window_size: int = 100, *args, **kwargs) -> AsyncIterator[Any]:
        """Stream data with window-based approach."""
        window_start = 0
        
        while True:
            async with self._semaphore:
                try:
                    # Load window
                    window_data = await self._load_window(loader_func, window_start, window_size, *args, **kwargs)
                    
                    if window_data and len(window_data) > 0:
                        for item in window_data:
                            yield item
                            self.stats.streamed_items += 1
                        
                        window_start += window_size
                        
                        # Stop if we got fewer items than window size
                        if len(window_data) < window_size:
                            break
                    else:
                        break
                    
                    # Check memory usage
                    if self._memory_monitor.is_memory_high():
                        await self._memory_monitor.optimize_memory()
                    
                except Exception as e:
                    self.stats.errors += 1
                    logger.error(f"Window-based streaming error: {e}")
                    break

# =============================================================================
# Virtual Scrolling Lazy Loader
# =============================================================================

class VirtualScrollingLazyLoader(LazyLoadingBase):
    """Virtual scrolling lazy loader for large datasets."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
super().__init__(config)
        self._virtual_windows: Dict[str, Dict[int, List[Any]]] = {}
        self._total_counts: Dict[str, int] = {}
    
    async def load_data(self, key: str, loader_func: Callable, start_index: int = 0, end_index: int = None, *args, **kwargs) -> List[Any]:
        """Load data for virtual scrolling."""
        # Check cache first
        cached_data = await self.get_cached_data(f"{key}_{start_index}_{end_index}")
        if cached_data:
            self.stats.cache_hits += 1
            return cached_data
        
        # Load data for virtual scrolling
        items = []
        
        async with self._semaphore:
            start_time = time.time()
            
            try:
                # Load virtual window
                window_data = await self._load_virtual_window(loader_func, start_index, end_index, *args, **kwargs)
                
                if window_data:
                    items = window_data
                    self.stats.paginated_items += len(items)
                
                # Update statistics
                batch_time_ms = (time.time() - start_time) * 1000
                self.stats.update_loading_stats(len(items), batch_time_ms)
                
                # Check memory usage
                if self._memory_monitor.is_memory_high():
                    await self._memory_monitor.optimize_memory()
                
            except Exception as e:
                self.stats.errors += 1
                logger.error(f"Virtual scrolling loading error: {e}")
        
        # Cache result
        await self.set_cached_data(f"{key}_{start_index}_{end_index}", items)
        
        return items
    
    async def _load_virtual_window(self, loader_func: Callable, start_index: int, end_index: Optional[int], *args, **kwargs) -> List[Any]:
        """Load a virtual window of data."""
        # Add virtual scrolling parameters
        virtual_kwargs = {
            **kwargs,
            'start_index': start_index,
            'end_index': end_index,
            'limit': end_index - start_index if end_index else self.config.batch_size
        }
        
        if asyncio.iscoroutinefunction(loader_func):
            return await loader_func(*args, **virtual_kwargs)
        else:
            return loader_func(*args, **virtual_kwargs)
    
    async def get_total_count(self, key: str, count_func: Callable, *args, **kwargs) -> int:
        """Get total count for virtual scrolling."""
        if key in self._total_counts:
            return self._total_counts[key]
        
        try:
            if asyncio.iscoroutinefunction(count_func):
                count = await count_func(*args, **kwargs)
            else:
                count = count_func(*args, **kwargs)
            
            self._total_counts[key] = count
            return count
            
        except Exception as e:
            logger.error(f"Error getting total count: {e}")
            return 0

# =============================================================================
# Lazy Loading Manager
# =============================================================================

class LazyLoadingManager:
    """Main lazy loading manager."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
        self.loaders: Dict[LoadingStrategy, LazyLoadingBase] = {}
        self._initialize_loaders()
    
    def _initialize_loaders(self) -> Any:
        """Initialize lazy loaders based on strategy."""
        if self.config.strategy == LoadingStrategy.STREAMING:
            self.loaders[LoadingStrategy.STREAMING] = StreamingLazyLoader(self.config)
        elif self.config.strategy == LoadingStrategy.PAGINATION:
            self.loaders[LoadingStrategy.PAGINATION] = PaginationLazyLoader(self.config)
        elif self.config.strategy == LoadingStrategy.CURSOR_BASED:
            self.loaders[LoadingStrategy.CURSOR_BASED] = CursorBasedLazyLoader(self.config)
        elif self.config.strategy == LoadingStrategy.WINDOW_BASED:
            self.loaders[LoadingStrategy.WINDOW_BASED] = WindowBasedLazyLoader(self.config)
        elif self.config.strategy == LoadingStrategy.VIRTUAL_SCROLLING:
            self.loaders[LoadingStrategy.VIRTUAL_SCROLLING] = VirtualScrollingLazyLoader(self.config)
    
    async def load_data(self, key: str, loader_func: Callable, *args, **kwargs) -> Any:
        """Load data using configured strategy."""
        loader = self.loaders.get(self.config.strategy)
        if not loader:
            raise ValueError(f"No loader configured for strategy: {self.config.strategy}")
        
        return await loader.load_data(key, loader_func, *args, **kwargs)
    
    async def stream_data(self, key: str, loader_func: Callable, *args, **kwargs) -> AsyncIterator[Any]:
        """Stream data using configured strategy."""
        loader = self.loaders.get(self.config.strategy)
        if not loader:
            raise ValueError(f"No loader configured for strategy: {self.config.strategy}")
        
        async for item in loader.stream_data(key, loader_func, *args, **kwargs):
            yield item
    
    async def get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data."""
        loader = self.loaders.get(self.config.strategy)
        if not loader:
            return None
        
        return await loader.get_cached_data(key)
    
    async def clear_cache(self) -> Any:
        """Clear all caches."""
        for loader in self.loaders.values():
            await loader.clear_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        all_stats = {}
        for strategy, loader in self.loaders.items():
            all_stats[strategy.value] = loader.get_stats()
        
        return {
            "config": {
                "strategy": self.config.strategy.value,
                "source_type": self.config.source_type.value,
                "batch_size": self.config.batch_size,
                "max_concurrent_batches": self.config.max_concurrent_batches,
                "enable_caching": self.config.enable_caching
            },
            "loaders": all_stats
        }

# =============================================================================
# Lazy Loading Decorators
# =============================================================================

def lazy_load(
    strategy: LoadingStrategy = LoadingStrategy.STREAMING,
    batch_size: int = 100,
    enable_caching: bool = True
):
    """Decorator for lazy loading functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Create lazy loading config
            config = LazyLoadingConfig(
                strategy=strategy,
                batch_size=batch_size,
                enable_caching=enable_caching
            )
            
            # Create lazy loading manager
            manager = LazyLoadingManager(config)
            
            # Generate cache key
            key_data = {
                "func": func.__name__,
                "args": str(args),
                "kwargs": str(sorted(kwargs.items())),
                "strategy": strategy.value
            }
            cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            
            # Load data with lazy loading
            return await manager.load_data(cache_key, func, *args, **kwargs)
        
        return wrapper
    return decorator

def lazy_stream(
    strategy: LoadingStrategy = LoadingStrategy.STREAMING,
    batch_size: int = 100,
    enable_caching: bool = True
):
    """Decorator for lazy streaming functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Create lazy loading config
            config = LazyLoadingConfig(
                strategy=strategy,
                batch_size=batch_size,
                enable_caching=enable_caching
            )
            
            # Create lazy loading manager
            manager = LazyLoadingManager(config)
            
            # Generate cache key
            key_data = {
                "func": func.__name__,
                "args": str(args),
                "kwargs": str(sorted(kwargs.items())),
                "strategy": strategy.value
            }
            cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            
            # Stream data with lazy loading
            async for item in manager.stream_data(cache_key, func, *args, **kwargs):
                yield item
        
        return wrapper
    return decorator

# =============================================================================
# FastAPI Integration
# =============================================================================

def get_lazy_loading_manager() -> LazyLoadingManager:
    """Dependency to get lazy loading manager."""
    config = LazyLoadingConfig(
        strategy=LoadingStrategy.STREAMING,
        batch_size=100,
        max_concurrent_batches=5,
        enable_caching=True,
        enable_prefetching=True
    )
    return LazyLoadingManager(config)

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "LoadingStrategy",
    "DataSourceType",
    "LoadingPriority",
    "LazyLoadingConfig",
    "LoadingStats",
    "LazyLoadingBase",
    "MemoryMonitor",
    "StreamingLazyLoader",
    "PaginationLazyLoader",
    "CursorBasedLazyLoader",
    "WindowBasedLazyLoader",
    "VirtualScrollingLazyLoader",
    "LazyLoadingManager",
    "lazy_load",
    "lazy_stream",
    "get_lazy_loading_manager",
] 